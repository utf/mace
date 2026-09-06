"""Plan section 2.1: the runtime Hamiltonian `H0(R; theta) = H_SK + H_onsite_scalar +
H_onsite_dir`, real symmetric, `[4N, 4N]` (s, px, py, pz per atom), spin-independent.

`H_SK` and `H_onsite_scalar` are the retained `defect_counting.SlaterKosterH` (plan section
3, "keep"): Slater-Koster `ss_sigma, sp_sigma, pp_sigma, pp_pi` per species pair with the
Harrison initialisation at each pair's reference bond length, four learned positive decay
lengths (`L_b = L0 exp(ln 2 tanh u_b)`), the bounded log-form modulation from the frozen
first-block base scalars (`exp(beta tanh g)`), the smooth cutoff taper; species/shell
baselines plus bounded corrections `gamma tanh[h(x_i) - h(xbar_s)]` centred on the pristine
species feature means. No unbounded constant channel.

`H_onsite_dir = a_i (v_i . T_sp_i) + b_i (Q_i : T_pp_i)` is new: `v_i` is a per-species
linear combination of the base's `l = 1` first-block features (polar vectors, with
position derivatives through the recomputed block), entering the on-site s-p block
symmetrically; `Q_i = sum_j w(r_ij) (rhat rhat^T - I/3)` is the geometric traceless
quadrupole of the neighbourhood with a smooth cutoff `w` (zero at centrosymmetric sites),
entering the on-site p-p block. `a_i = a_max tanh(alpha[Z_i])`, `b_i = b_max tanh(beta[Z_i])`
are bounded species-level coefficients. With `directional=False` the block is identically
absent: the scalar-only control of Arm 1.

Gauge: `H0 -> H0 + a I` is not a parameter of this module; nothing here can add a constant
to every level except through the bounded, centred corrections.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Sequence

import torch
from torch import nn

from mace.modules.defect_counting import (ORBITALS_PER_ATOM, SlaterKosterH,
                                          harrison_initialise)

# Registered defaults (plan section 11: to register before use). `r_cut` is the old head's
# 10 A; `q_cut` reaches the first shell and the Cs neighbours (Pb-Cl 2.8, Cs-Cl 3.5-4.1 A).
R_CUT_DEFAULT = 10.0
Q_CUT_DEFAULT = 4.5
A_MAX_DEFAULT = 1.0       # eV per unit |v|
B_MAX_DEFAULT = 1.0       # eV per unit |Q|
ON_SITE_RANGE_DEFAULT = 3.0


def quadrupole_descriptor(edge_index: torch.Tensor, edge_vector: torch.Tensor, n_nodes: int,
                          q_cut: float) -> torch.Tensor:
    """`Q_i = sum_j w(r_ij) (rhat_ij rhat_ij^T - I/3)`, `[N, 3, 3]`, traceless symmetric,
    `w(r) = (1 - (r/q_cut)^6)^2` inside `q_cut` and zero outside. Vanishes at every
    centrosymmetric site by symmetry."""
    sender = edge_index[0]
    r = edge_vector.norm(dim=-1)
    inside = r < q_cut
    r_in, v_in = r[inside], edge_vector[inside]
    unit = v_in / r_in.unsqueeze(-1)
    w = (1.0 - (r_in / q_cut) ** 6) ** 2
    outer = unit.unsqueeze(-1) * unit.unsqueeze(-2) - torch.eye(3, dtype=r.dtype, device=r.device) / 3.0
    contrib = w.reshape(-1, 1, 1) * outer
    Q = torch.zeros(n_nodes, 3, 3, dtype=r.dtype, device=r.device)
    return Q.index_add(0, sender[inside], contrib)


class H0(nn.Module):
    """The runtime Hamiltonian for ONE graph (dense; the batched path stacks equal sizes)."""

    def __init__(self, atomic_numbers: Sequence[int], feature_dim: int = 128,
                 n_vectors: int = 128, r_cut: float = R_CUT_DEFAULT,
                 q_cut: float = Q_CUT_DEFAULT, directional: bool = True,
                 a_max: float = A_MAX_DEFAULT, b_max: float = B_MAX_DEFAULT,
                 on_site_range: float = ON_SITE_RANGE_DEFAULT, hidden: int = 64) -> None:
        super().__init__()
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        num_elements = len(self.atomic_numbers)
        self.r_cut, self.q_cut = float(r_cut), float(q_cut)
        self.directional = bool(directional)
        self.a_max, self.b_max = float(a_max), float(b_max)
        self.sk = SlaterKosterH(num_elements, feature_dim, atomic_numbers=self.atomic_numbers,
                                r_cut=self.r_cut, envelope="exp", hop_form="log",
                                decay_learned=True, centre_form="argument",
                                on_site_range=on_site_range, hidden=hidden)
        harrison_initialise(SimpleNamespace(h=self.sk), self.atomic_numbers)
        # Directional block: a per-species combination of the l = 1 channels and the two
        # bounded coefficients. Zero-initialised: the block starts absent and is learned.
        self.vector_mix = nn.Parameter(torch.zeros(num_elements, n_vectors))
        self.alpha = nn.Parameter(torch.zeros(num_elements))
        self.beta = nn.Parameter(torch.zeros(num_elements))
        # Species feature means over the pristine cell (plan 2.1), set once and stored.
        self.register_buffer("centre", torch.zeros(num_elements, feature_dim))
        self.register_buffer("centre_set", torch.tensor(False))
        # Test knobs (plan section 5 gates): `H0 -> H0 + a I` for the gauge gate, and a
        # per-site level shift `[N]` that binds a carrier on chosen sites for the tiling
        # ladder. Neither is a parameter; both are None/0 in production.
        self.gauge_shift = 0.0
        self.site_shift: Optional[torch.Tensor] = None

    # ------------------------------------------------------------- pristine centre

    @torch.no_grad()
    def set_centre(self, scalars: torch.Tensor, species: torch.Tensor) -> None:
        """Per-species mean of the first-block scalars over a pristine cell."""
        for s in range(self.centre.shape[0]):
            sel = species == s
            if bool(sel.any()):
                self.centre[s] = scalars[sel].to(self.centre.dtype).mean(dim=0)
        self.centre_set.fill_(True)

    # ------------------------------------------------------------- pieces

    def coefficients(self):
        """`(a[Z], b[Z])`, bounded."""
        return self.a_max * torch.tanh(self.alpha), self.b_max * torch.tanh(self.beta)

    def directional_block(self, vectors: torch.Tensor, species: torch.Tensor,
                          edge_index: torch.Tensor, edge_vector: torch.Tensor) -> torch.Tensor:
        """`H_onsite_dir` as a dense `[4N, 4N]` block-diagonal matrix."""
        n = int(species.shape[0])
        a, b = self.coefficients()
        v = torch.einsum("nc,nca->na", self.vector_mix[species].to(vectors.dtype), vectors)
        Q = quadrupole_descriptor(edge_index, edge_vector, n, self.q_cut)
        block = torch.zeros(n, ORBITALS_PER_ATOM, ORBITALS_PER_ATOM, dtype=vectors.dtype,
                            device=vectors.device)
        sp = a[species].unsqueeze(-1) * v                        # [N, 3]
        block[:, 0, 1:] = sp
        block[:, 1:, 0] = sp
        block[:, 1:, 1:] = b[species].reshape(-1, 1, 1) * Q
        return torch.block_diag(*block)

    def forward(self, scalars: torch.Tensor, vectors: Optional[torch.Tensor],
                species: torch.Tensor, edge_index: torch.Tensor, edge_vector: torch.Tensor
                ) -> torch.Tensor:
        """Dense `H0` `[4N, 4N]` in float64. `scalars` `[N, F]` and `vectors` `[N, n_vec, 3]`
        are the base's first-block features (frozen w.r.t. base parameters, attached to
        positions and cell); `edge_index`, `edge_vector` the head's graph at `r_cut`."""
        if not bool(self.centre_set):
            raise RuntimeError("H0 needs the pristine species feature means: call set_centre "
                               "on the pristine cell before the first forward")
        n = int(species.shape[0])
        scalars = scalars.to(torch.float64)
        edge_vector = edge_vector.to(torch.float64)
        H = self.sk(scalars, species, edge_index, edge_vector, madelung=None, n_nodes=n)
        levels = self.sk.on_site(scalars, species, None, centre=self.centre)
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
        H = H - torch.diag(torch.diagonal(H)) + torch.diag(diag)
        if self.directional:
            if vectors is None:
                raise ValueError("the directional block needs the base's l = 1 features")
            H = H + self.directional_block(vectors.to(torch.float64), species, edge_index,
                                           edge_vector)
        H = 0.5 * (H + H.transpose(0, 1))
        if self.site_shift is not None:
            H = H + torch.diag(self.site_shift.to(H.dtype).to(H.device).repeat_interleave(ORBITALS_PER_ATOM))
        if self.gauge_shift:
            H = H + float(self.gauge_shift) * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        return H
