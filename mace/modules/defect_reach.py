"""In-loop assertion that the neighbour list the LOSS uses spans the carrier Hamiltonian.

Four times now a guard has measured a reimplementation rather than the thing it guarded, and
the last one voided a 20-seed screen: `graph_cutoff` was applied only to the validation
fallback path, so every training run built its graph at r_max and the two vacancy-sharing Pb
had no edge in 0 of 40 charged frames. A pre-flight on a path the loss never executes is
worse than no check at all, because it produces a passed check.

So this runs on a real batch drawn from the training DataLoader itself -- the same object the
optimiser iterates -- and aborts the run if the graph is short.

LABEL-FREE BY CONSTRUCTION. The plan phrases the criterion as "fraction of charged frames in
which the two hub Pb share an edge", which needs the vacancy assignment inside the training
loop. The equivalent structural fact does not: if the graph was built at `graph_cutoff` then
edges longer than r_max exist and the longest edge approaches the cutoff, whereas a graph
built at r_max cannot contain them at all. That distinguishes the two cases exactly -- a 5 A
graph has max edge <= 5.0 by construction -- without telling the training process anything
about where the defect is. The hub-pair fraction remains available as an evaluation-time
diagnostic, where the assignment is allowed.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

__all__ = ["reach_report", "assert_carrier_reach"]


def _edge_lengths(batch) -> Optional[torch.Tensor]:
    try:
        idx = batch["edge_index"] if isinstance(batch, dict) else batch.edge_index
        pos = batch["positions"] if isinstance(batch, dict) else batch.positions
    except (KeyError, AttributeError):
        return None
    if idx is None or idx.numel() == 0:
        return None
    vec = pos[idx[1]] - pos[idx[0]]
    shifts = batch["shifts"] if isinstance(batch, dict) else getattr(batch, "shifts", None)
    if shifts is not None and shifts.numel() == vec.numel():
        vec = vec + shifts
    return torch.linalg.norm(vec, dim=-1).detach()


def reach_report(batch, r_max: float, cutoff: float,
                 window=(5.2, 5.8)) -> Dict[str, float]:
    """Structural facts about the graph the loss is about to consume."""
    lengths = _edge_lengths(batch)
    if lengths is None or lengths.numel() == 0:
        return {}
    long_edges = (lengths > float(r_max)).float().mean().item()
    in_window = ((lengths >= window[0]) & (lengths < window[1])).sum().item()
    return {
        "max_edge": float(lengths.max()),
        "median_edge": float(lengths.median()),
        "frac_beyond_r_max": float(long_edges),
        "edges_in_hub_window": float(in_window),
        "n_edges": float(lengths.numel()),
        "configured_cutoff": float(cutoff),
        "r_max": float(r_max),
    }


def assert_carrier_reach(batch, r_max: float, cutoff: float, tolerance: float = 0.95,
                         strict: bool = True) -> Dict[str, float]:
    """Log the reach of the loss's own graph; abort if it was built at r_max.

    `cutoff` is what graph_cutoff() asked for. If the realised graph does not get within
    `tolerance` of it, the loader ignored it -- which is precisely the failure that voided
    R1, R2 and Test 1 -- and nothing downstream would be interpretable.
    """
    report = reach_report(batch, r_max, cutoff)
    if not report:
        logging.warning("Carrier reach check: batch carried no edges to measure")
        return report

    logging.info(
        "Carrier reach (from the training loader's own batch): "
        f"max edge {report['max_edge']:.2f} A, median {report['median_edge']:.2f} A, "
        f"{report['frac_beyond_r_max']:.1%} of edges beyond r_max={r_max:.1f}, "
        f"{int(report['edges_in_hub_window'])} edges in the 5.2-5.8 A window, "
        f"configured cutoff {cutoff:.1f} A")

    if cutoff <= float(r_max) + 1e-6:
        return report                                  # no long-range head; nothing to check

    if report["max_edge"] < tolerance * float(cutoff):
        msg = (f"CARRIER REACH FAILURE: the graph the loss consumes reaches only "
               f"{report['max_edge']:.2f} A but graph_cutoff asked for {cutoff:.1f} A. "
               f"The neighbour list was built at r_max={r_max:.1f}, so the two "
               f"vacancy-sharing Pb have no edge and the two-site state cannot be "
               f"represented. This is the fault that voided the previous screen; the run is "
               f"aborting rather than producing another uninterpretable result.")
        if strict:
            raise RuntimeError(msg)
        logging.error(msg)
    elif report["frac_beyond_r_max"] <= 0.0:
        msg = ("CARRIER REACH FAILURE: no edge in the batch is longer than r_max, so the "
               "carrier Hamiltonian has exactly the trunk's range.")
        if strict:
            raise RuntimeError(msg)
        logging.error(msg)
    return report
