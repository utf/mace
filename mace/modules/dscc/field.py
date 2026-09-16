"""Plan P: the zero-field atomic dipoles from a frozen MACE-Field model (the "provider").

`p[a] = -dF/dE_node[a]` at `E_node = 0`, from `MACEField.forward(electric_field_node=...)`
(`mace/modules/extensions.py`), evaluated on the HEAD's graph restricted to the provider's
cutoff, with the head's species mapped onto the provider's element table. The dipoles come
back attached to the positions (`create_graph` inside the node-dipole backward), so a later
backward of any energy built from them -- `E_cP`, `E_PP` -- carries `dp/dR` without a second
pass through the provider: that is the plan's "one nested-autograd pass".
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn


class FieldProvider(nn.Module):
    def __init__(self, model: nn.Module, head: str = "mp-dielectric") -> None:
        super().__init__()
        if type(model).__name__ != "MACEField":
            raise TypeError(f"the field provider needs a MACEField model; got {type(model).__name__}")
        model = model.double().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        heads = list(getattr(model, "heads", ["Default"]))
        if head not in heads:
            raise ValueError(f"head {head!r} not in the provider's heads {heads}")
        self.head_name = head
        self.head_index = heads.index(head)
        self.r_max = float(model.r_max)
        zs = [int(z) for z in model.atomic_numbers]
        lookup = torch.full((120,), -1, dtype=torch.long)
        for i, z in enumerate(zs):
            lookup[z] = i
        self.register_buffer("z_lookup", lookup)
        self.n_elements = len(zs)

    def node_attrs(self, numbers: torch.Tensor) -> torch.Tensor:
        idx = self.z_lookup[numbers.to(torch.long)]
        if bool((idx < 0).any()):
            missing = sorted(set(numbers[idx < 0].tolist()))
            raise ValueError(f"elements {missing} are not in the provider's table")
        return torch.nn.functional.one_hot(idx, self.n_elements).to(torch.float64)

    def graph(self, positions: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor,
              ptr: torch.Tensor, edge_index: torch.Tensor, shifts: torch.Tensor,
              unit_shifts: torch.Tensor, numbers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The provider's batch: the given graph's edges no longer than `r_max`."""
        sender, receiver = edge_index[0], edge_index[1]
        vectors = positions[receiver] - positions[sender] + shifts
        keep = vectors.detach().norm(dim=-1) <= self.r_max
        num_graphs = int(ptr.numel() - 1)
        return {
            "positions": positions,
            "cell": cell.reshape(-1, 3),
            "node_attrs": self.node_attrs(numbers),
            "edge_index": edge_index[:, keep],
            "shifts": shifts[keep],
            "unit_shifts": unit_shifts[keep],
            "batch": batch,
            "ptr": ptr,
            "head": torch.full((num_graphs,), self.head_index, dtype=torch.long, device=positions.device),
        }

    def node_dipoles(self, positions: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor,
                     ptr: torch.Tensor, edge_index: torch.Tensor, shifts: torch.Tensor,
                     unit_shifts: torch.Tensor, numbers: torch.Tensor) -> torch.Tensor:
        """`p [N, 3]` (e A) at zero field, attached to `positions`."""
        data = self.graph(positions, cell, batch, ptr, edge_index, shifts, unit_shifts, numbers)
        e_node = torch.zeros(positions.shape[0], 3, dtype=torch.float64, device=positions.device,
                             requires_grad=True)
        out = self.model(data, training=True, compute_force=False, compute_stress=False,
                         electric_field_node=e_node, compute_node_dipoles=True)
        return out["node_dipoles"]

    def response(self, positions: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor,
                 ptr: torch.Tensor, edge_index: torch.Tensor, shifts: torch.Tensor,
                 unit_shifts: torch.Tensor, numbers: torch.Tensor,
                 electric_field: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        """The global-field observables (P in e/A^2, Z* in e, alpha per unit eps0) with
        energy and forces, for the diagnostics; `electric_field` defaults to zero."""
        data = self.graph(positions, cell, batch, ptr, edge_index, shifts, unit_shifts, numbers)
        num_graphs = int(ptr.numel() - 1)
        if electric_field is None:
            electric_field = torch.zeros(num_graphs, 3, dtype=torch.float64, device=positions.device)
        return self.model(data, training=True, compute_force=True, compute_polarization=True,
                          compute_becs=True, compute_polarizability=True,
                          electric_field=electric_field)


def load_field_provider(path: str, head: str = "mp-dielectric", device: str = "cpu") -> FieldProvider:
    model = torch.load(path, map_location=device, weights_only=False)
    return FieldProvider(model, head=head).to(device)
