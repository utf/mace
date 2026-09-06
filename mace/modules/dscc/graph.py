"""The head's neighbour graph at its own cutoff `r_cut` (plan section 2.1: `H0` is sparse
on the neighbour graph with cutoff `r_cut`; the vacancy-spanning Pb-Pb pair at 4.8-7.1 A
must be on it, so the base's 5 A list cannot serve). Edge vectors are rebuilt in torch from
positions, integer shifts and the cell, so they carry position and cell gradients; the
integer shifts are a property of the wrapped geometry and are recomputed per frame.

Convention (MACE's): `edge_index = [sender, receiver]`, `vector = r_receiver - r_sender +
shift @ cell`; the Slater-Koster block of an edge has its rows on the sender.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def neighbour_list(positions: np.ndarray, cell: np.ndarray, r_cut: float,
                   pbc=(True, True, True)) -> Tuple[np.ndarray, np.ndarray]:
    """`(edge_index [2, n_edges], unit_shifts [n_edges, 3])` of every ordered pair within
    `r_cut` (both directions present), from matscipy."""
    from matscipy.neighbours import neighbour_list as _nl

    i, j, S = _nl("ijS", positions=np.asarray(positions, dtype=np.float64),
                  cell=np.asarray(cell, dtype=np.float64), pbc=np.asarray(pbc, dtype=bool),
                  cutoff=float(r_cut))
    return np.stack([i, j]).astype(np.int64), S.astype(np.float64)


def edge_vectors(positions: torch.Tensor, cell: torch.Tensor, edge_index: torch.Tensor,
                 unit_shifts: torch.Tensor) -> torch.Tensor:
    """`r_receiver - r_sender + S @ cell`, `[n_edges, 3]`, differentiable in both."""
    sender, receiver = edge_index[0], edge_index[1]
    return positions[receiver] - positions[sender] + unit_shifts.to(positions.dtype) @ cell


def edges_within(edge_index: torch.Tensor, vectors: torch.Tensor, r_max: float
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The sub-graph of edges no longer than `r_max` (the base's own list)."""
    keep = vectors.detach().norm(dim=-1) <= r_max
    return edge_index[:, keep], vectors[keep]
