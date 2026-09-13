"""Edge-bucket padding: the prerequisite for CUDA-graph capture (coadvisor 2026-09-13).

At 79 atoms the model is launch-bound everywhere -- the base's forward+backward is 41.6 ms of
wall time of which 41.4 ms is CPU issue -- so the only lever that attacks the bottleneck is
removing launches, and that means capturing the work as CUDA graphs. Capture fixes both the
shapes and the addresses of every tensor in the captured region, while a neighbour list
changes length every step. This module makes the length constant.

THE PADDING IS EXACTLY ZERO, not approximately. A dummy edge is placed from node 0 to node 0
with a lattice shift far enough out that its length exceeds every cutoff in the model, and
every consumer of an edge multiplies by a factor that is identically zero there:

  * the base: `PolynomialCutoff.calculate_envelope` returns `envelope * (x < r_max)`, and
    e3nn's `FullyConnectedNet` (the radial weights) has no bias, so zero in is zero out and
    the message scattered from a dummy edge is exactly 0.0;
  * `H0`'s hoppings and the O3 descriptor: both carry H0's taper `(1 - (r/r_cut)^6)^2` with
    `r/r_cut` clamped at 1, which is exactly zero at and beyond `r_cut`;
  * `quadrupole_descriptor`: masked by `r < q_cut`.

The one consumer that is NOT automatically safe is O1's `pair_correction`, which deduplicates
the hopping MLP by unordered node pair: every dummy edge is the pair (0, 0), so it adds ONE
extra row to the unique set and cannot contaminate a real pair's input. Its correction is then
multiplied by the zero radial factor. `test_padding_leaves_every_output_unchanged` covers the
modulated head as well as O3's unmodulated one.

Bucketing rule (registered here): capacity is the count times `1 + HEADROOM`, rounded up to
`GRANULARITY`, and a bucket is only reduced when the count falls below `1 - HYSTERESIS` of it
-- so a relaxation that drifts by a few edges per step keeps one graph, and a trajectory that
genuinely changes size re-captures once rather than oscillating.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

GRANULARITY = 512
HEADROOM = 0.10
HYSTERESIS = 0.25


def edge_bucket(n_edges: int, current: Optional[int] = None) -> int:
    """The capacity to pad `n_edges` to, with hysteresis against a `current` bucket."""
    want = int(n_edges * (1.0 + HEADROOM))
    cap = max(GRANULARITY, -(-want // GRANULARITY) * GRANULARITY)
    if current is not None and current >= n_edges and cap <= current:
        # Only shrink once the count is well clear of the bucket; otherwise keep the graph.
        if n_edges > current * (1.0 - HYSTERESIS):
            return int(current)
    return int(cap)


def dummy_unit_shift(cell: torch.Tensor, r_cut: float) -> torch.Tensor:
    """A lattice translation `[3]` (integer, in `cell` units) whose image of node 0 is further
    than `r_cut` from node 0, for every cell shape. Uses the shortest lattice vector, so the
    same multiple is safe along all three axes."""
    lengths = cell.reshape(3, 3).norm(dim=-1)
    shortest = float(lengths.min())
    n = int(r_cut / max(shortest, 1e-9)) + 2
    return torch.tensor([n, n, n], dtype=torch.long, device=cell.device)


def pad_edges(edge_index: torch.Tensor, unit_shifts: torch.Tensor, capacity: int,
              dummy: torch.Tensor, keep: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad (or select-and-pad) an edge list to exactly `capacity` entries.

    `keep`, when given, marks the edges to retain -- the base's `r <= r_max` subset. The
    retained edges are moved to the front by a stable sort on the mask, which is a device-side
    operation: compacting with `nonzero` would copy a count to the host, and the point of
    capture is that the host does nothing. Returns `(edge_index, unit_shifts, live)`.
    """
    n = int(edge_index.shape[1])
    device = edge_index.device
    if keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=device)
    order = torch.argsort((~keep).to(torch.int8), stable=True)      # kept first, order preserved
    if capacity <= n:
        sel = order[:capacity]
    else:
        pad = order.new_zeros(capacity - n)
        sel = torch.cat([order, pad])
    live = torch.zeros(capacity, dtype=torch.bool, device=device)
    live[:min(n, capacity)] = keep[sel[:min(n, capacity)]]
    ei = torch.where(live.unsqueeze(0), edge_index[:, sel], torch.zeros_like(edge_index[:, sel]))
    us = torch.where(live.unsqueeze(-1), unit_shifts[sel].to(dummy.dtype),
                     dummy.reshape(1, 3).expand(capacity, 3))
    return ei, us, live


def pad_batch(data: Dict[str, torch.Tensor], capacity: int, r_cut: float
              ) -> Dict[str, torch.Tensor]:
    """A batch whose head edge list is exactly `capacity` long, the extras inert.

    Single-graph batches only: a dummy edge belongs to node 0, and in a multi-graph batch that
    is graph 0's node, so the other graphs' edge counts would no longer be theirs. Capture is
    a single-frame path (one MD/relaxation step), so this is the case that matters; a batched
    trainer does not capture.
    """
    if int(data["ptr"].numel() - 1) != 1:
        raise ValueError("edge-bucket padding is for single-graph batches (capture is a "
                         "one-frame path); this batch has "
                         f"{int(data['ptr'].numel() - 1)} graphs")
    cell = data["cell"].view(3, 3)
    dummy = dummy_unit_shift(cell, r_cut)
    ei, us, live = pad_edges(data["edge_index"], data["unit_shifts"], capacity, dummy)
    out = dict(data)
    out["edge_index"] = ei
    out["unit_shifts"] = us.to(data["unit_shifts"].dtype)
    out["shifts"] = us.to(data["positions"].dtype) @ cell.to(data["positions"].dtype)
    out["edge_live"] = live
    return out
