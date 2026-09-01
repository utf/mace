"""Carrier-field response: the host's energetic response to the carrier's own potential.

R1 on the corrected graph found that no compact state in the current head reproduces the
force footprint (every clamp at axial_red ~0) while the delocalised state does (~0.8). E0
had already shown the footprint extends ~6 A with two thirds of the squared residual off the
hub. A short-ranged tight-binding head has no way to produce that reach from a compact state:
its forces come from dt/dR and deps/dR, both of which die with the hopping envelope.

Electrostatics does have the reach. The carrier carries charge; every ion in the cell sits in
its potential and responds. So:

    q_j^c  = a * alpha_j^c                       a = 1/sqrt(eps_inf), as in the LR branch
    V_i^c  = potential at atom i from {q_j^c}    periodic, Gaussian-smeared, SELF-TERM REMOVED
    u_i^c  = V_i^c * g(h_i, z_i)  -  0.5 * (V_i^c)^2 * softplus(p(h_i, z_i))
    E_resp = sum_c n_c sum_i u_i^c               vanishes at n = 0 through the prefactor

Three constraints on the readout, each load-bearing:

* **It vanishes identically at V = 0 and is low order in V.** A free u(h_i, V_i) MLP would be
  a second per-atom energy field with exactly the alpha/u gauge freedom the spectral head was
  built to remove. Linear plus quadratic is a learned effective charge (the site-energy shift
  of an ion in the carrier's field) plus a learned polarisability. Both start small.
* **The quadratic coefficient is passed through softplus**, so the polarisation term is always
  stabilising. A sign-free quadratic could lower the energy by inventing anti-polarisable
  ions, which is not a response, it is a fit.
* **V enters the energy readout only, never H.** alpha comes from H on carrier-blind features,
  V follows from alpha, E_resp follows from V. No self-consistent loop, forces by plain
  autograd.

That last point is deliberate and is NOT a weaker mechanism than feeding V back into H.
dL/dalpha flows through E_resp into eps and t, so training drives the Hamiltonian toward
whichever alpha makes the total fit the labels -- the ordinary route site selection has always
taken in this model. Feeding V into H would instead add a polaron prior (the carrier seeks its
own polarisation well at inference) whose sign in this system is unmeasured, and that is the
failure class of the banned carrier x host term: site selection by a built-in potential rather
than by data. Whether that prior would help is answered by measurement, not assumption --
compare E_resp under a hub clamp against a cage clamp after training.

The self-term is excluded. The ion's response to the carrier amplitude on its OWN orbitals is
already the on-site energy eps_i; including j = i double-counts it and adds a spurious force
of an ion moving in its own Gaussian.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["CarrierResponse"]


def _mlp(in_dim: int, hidden: int, scale: float) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    with torch.no_grad():
        net[-1].weight.mul_(scale)
        net[-1].bias.mul_(scale)
    return net


class CarrierResponse(nn.Module):
    """Linear + quadratic per-atom response to the carrier's electrostatic potential."""

    def __init__(self, feature_dim: int, num_elements: int, hidden: int = 32,
                 init_scale: float = 0.01, alpha_scale: float = 4.0) -> None:
        super().__init__()
        self.alpha_scale = alpha_scale
        self.species = nn.Embedding(num_elements, 8)
        self.g = _mlp(feature_dim + 8, hidden, init_scale)      # effective charge
        self.p = _mlp(feature_dim + 8, hidden, init_scale)      # polarisability (softplus'd)

    def potential(self, ewald, charges: torch.Tensor, positions: torch.Tensor,
                  cell: torch.Tensor, batch: torch.Tensor,
                  self_potential: Optional[torch.Tensor] = None) -> torch.Tensor:
        """V_i = dE/dq_i, with the self-term removed.

        Differentiating the Ewald energy with respect to the charges is exact and reuses the
        production kernel rather than re-deriving one -- a re-derivation would drift from
        whatever convention LES actually uses (smearing width, background term, units), and
        that class of drift has cost this project real time.

        E = 1/2 q^T A q, so dE/dq_i = (A q)_i = V_i, which INCLUDES the diagonal A_ii q_i.
        A_ii is the same constant for every atom in a given cell (a Gaussian's potential at
        its own centre does not depend on where it sits), so it is passed in and subtracted.
        """
        q = charges.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            energy = ewald.energy(q, positions, cell, batch).sum()
            v = torch.autograd.grad(energy, q, create_graph=self.training)[0]
        if self_potential is not None:
            v = v - self_potential[batch] * charges
        return v

    def forward(self, node_feats: torch.Tensor, node_species: torch.Tensor,
                alpha: torch.Tensor, counts: torch.Tensor, batch: torch.Tensor,
                positions: torch.Tensor, cell: torch.Tensor, ewald,
                self_potential: Optional[torch.Tensor] = None,
                num_graphs: int = 1) -> torch.Tensor:
        """E_resp per graph. `alpha` is [n_nodes, C]; `counts` is [n_graphs, C]."""
        from mace.tools.scatter import scatter_sum

        emb = self.species(node_species.long())
        feats = torch.cat([node_feats, emb], dim=-1)
        g = self.g(feats).squeeze(-1)                       # effective charge
        p = torch.nn.functional.softplus(self.p(feats)).squeeze(-1)   # >= 0 always

        total = torch.zeros(num_graphs, device=node_feats.device, dtype=node_feats.dtype)
        for c in range(alpha.shape[1]):
            n_c = counts[:, c]
            if not bool((n_c != 0).any()):
                continue                                    # channel carries no carrier
            q = self.alpha_scale * alpha[:, c]
            v = self.potential(ewald, q, positions, cell, batch, self_potential)
            u = v * g - 0.5 * v.pow(2) * p
            per_graph = scatter_sum(u, batch, dim=0, dim_size=num_graphs)
            total = total + n_c * per_graph
        return total


def self_potential_of(ewald, cell: torch.Tensor) -> torch.Tensor:
    """A_ii: the smeared potential of a unit charge at its own centre, per cell.

    Obtained from the kernel itself rather than from a formula, for the same reason the
    potential is: a single unit charge alone in the cell has energy 1/2 * A_ii, so A_ii is
    twice that. It depends on the cell (through the periodic images and the background) but
    not on where the atom sits.
    """
    cell = cell.view(-1, 3, 3)
    out = []
    for g in range(cell.shape[0]):
        q = torch.ones(1, device=cell.device, dtype=cell.dtype)
        r = torch.zeros(1, 3, device=cell.device, dtype=cell.dtype)
        b = torch.zeros(1, dtype=torch.long, device=cell.device)
        e = ewald.energy(q, r, cell[g: g + 1], b)
        out.append(2.0 * e.reshape(()))
    return torch.stack(out)
