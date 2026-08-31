"""Component H: carrier localisation as a bound state of a learned Hamiltonian.

Why replace Boltzmann attention
-------------------------------
The production head writes `alpha = softmax(l)` and `dE_SR = sum_c n_c sum_i alpha_i u_i`.
Localising then means winning an amplitude contest against cell size: full weight on a
two-atom shell needs `beta * delta_eps >~ ln N_bulk + margin`, about 5-6 here. At the physical
contrast `delta_eps ~ 0.18 eV` with `beta = 10 eV^-1`, per-bulk-Pb leakage is `e^-1.8 ~ 0.16`,
and fourteen bulk Pb outweigh the shell even with a PERFECT site-energy field. Models that did
localise won only through free logit gaps of ~20, an amplitude no physical site energy
supports. Carrier binding is physically a threshold -- a state either splits off the band or
it does not -- and softmax cannot express a threshold.

Deleting the alpha/u gauge freedom is not enough on its own: tying `alpha = softmax(-beta eps)`
gave 0 of 8 correct, with failures coherent but sublattice-level. The missing ingredient is
intra-species site discrimination, not gauge.

What this does
--------------
Per channel c, build a short-ranged one-particle Hamiltonian on the existing neighbour list:

    H_ii = eps_i(h_i, n)                        diagonal, from invariant node features
    H_ij = -t_ij(h_i, h_j, r_ij) * f_cut(r_ij)  off-diagonal, symmetric by construction

take its lowest `m` eigenpairs, and read the carrier energy straight off the spectrum:

    w_k      = softmax_k(-lambda_k / T_s)       thermal smearing, regularises crossings
    dE_SR^c  = n_c * sum_k w_k lambda_k
    alpha_i  = sum_k w_k psi_{k,i}^2            drop-in for every existing alpha diagnostic

Why this addresses the failure modes
------------------------------------
* Localisation becomes bound-state formation. Any eigenvalue split off from the band continuum
  of a short-ranged H has an exponentially localised eigenvector, at ANY N. The
  contrast-to-localisation map is a threshold rather than `e^(-beta delta_eps)` against N.
* "Right species, wrong site" stops being representable. An equivariant H gives
  bulk-identical sites identical matrix elements, so only a genuinely distinct environment can
  host a split-off level.
* The sublattice-uniform solution stops being fittable. A band state's eigenvalue has 1/N
  sensitivity to any single matrix element and cannot track shell-geometry-driven label
  fluctuations, while a bound state's eigenvalue is maximally sensitive to exactly those
  coordinates. Hellmann-Feynman forces are concentrated by |psi|^2 with no softmax-Jacobian
  loophole, so the existing force labels supervise localisation directly.

Hard guardrail: dE_SR IS the smeared eigenvalue. There is no auxiliary MLP on top of
(alpha, psi, lambda) -- such a readout would reopen exactly the two-field gauge this replaces.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import nn

__all__ = ["SpectralCarrierHead", "SpectralOutput"]


# Padded slots are given large positive on-site energies so their eigenvalues sit far above
# the physical spectrum and can never enter the lowest m.
#
# They must also be DISTINCT from each other. Giving every padded slot the same value makes
# that block exactly degenerate, and a batch mixing 79- and 159-atom cells then hands `eigh`
# an 80-fold repeated eigenvalue: LAPACK returns "the input matrix is ill-conditioned or has
# too many repeated eigenvalues" and the run dies. Spacing them by 1 eV keeps them mutually
# distinct and still far above anything physical.
#
# The base is 1e3 rather than 1e4 because the separation only has to be unambiguous, and a
# smaller dynamic range keeps the matrix better conditioned; physical eigenvalues here are of
# order 1 eV.
_PAD_ENERGY = 1.0e3
_PAD_SPACING = 1.0


class SpectralOutput(NamedTuple):
    delta_sr: torch.Tensor      # [n_graphs]
    alpha: torch.Tensor         # [n_nodes, C]   sum_k w_k psi^2
    site_energy: torch.Tensor   # [n_nodes, C]   the diagonal eps
    gap: torch.Tensor           # [n_graphs, C]  lambda_band - lambda_bound
    eigenvalues: torch.Tensor   # [n_graphs, C, m]
    weights: torch.Tensor       # [n_graphs, C, m]


def _mlp(sizes, activation=nn.SiLU, final_scale=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:-1]):
        layers += [nn.Linear(a, b), activation()]
    last = nn.Linear(sizes[-2], sizes[-1])
    if final_scale is not None:
        with torch.no_grad():
            last.weight.mul_(final_scale)
            last.bias.mul_(final_scale)
    layers.append(last)
    return nn.Sequential(*layers)


class SpectralCarrierHead(nn.Module):
    """Lowest eigenpairs of a learned per-channel carrier Hamiltonian."""

    def __init__(
        self,
        feature_dim: int,
        counter_dim: int,
        hidden: int = 64,
        num_channels: int = 4,
        num_states: int = 6,
        smearing: float = 0.020,      # eV
        radial_dim: int = 8,
        r_cut: float = 5.0,
        eps_init_scale: float = 0.01,
        hop_init_scale: float = 0.05,
        solver_dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_states = num_states
        self.smearing = smearing
        self.r_cut = r_cut
        self.radial_dim = radial_dim
        self.solver_dtype = solver_dtype

        # Diagonal: site energies, conditioned on the carrier counters so one head serves all
        # four channels and the electron/hole distinction lives entirely in this conditioning.
        #
        # Initialised near-flat (small final layer) so there is NO bound state at the start:
        # the data has to pull a level out of the band rather than find one already there.
        self.site = _mlp([feature_dim + counter_dim, hidden, hidden, num_channels],
                         final_scale=eps_init_scale)

        # Off-diagonal: hopping. Small but explicitly NONZERO at init -- a zero hopping makes
        # the graph disconnected, every eigenvector a delta function, and the gradient with
        # respect to t vanishes at the symmetric point, so it would never train. This is the
        # zero-init trap the plan asks to audit.
        self.hop = _mlp([2 * feature_dim + radial_dim, hidden, hidden, num_channels],
                        final_scale=hop_init_scale)
        with torch.no_grad():
            self.hop[-1].bias.fill_(0.5)      # a real band width at initialisation

    def _radial(self, r: torch.Tensor) -> torch.Tensor:
        """Smooth radial basis with a cutoff envelope, so forces stay continuous at r_cut."""
        n = torch.arange(1, self.radial_dim + 1, device=r.device, dtype=r.dtype)
        x = (r / self.r_cut).clamp(max=1.0).unsqueeze(-1)
        return torch.sin(n * torch.pi * x) / x.clamp_min(1e-6)

    def _envelope(self, r: torch.Tensor) -> torch.Tensor:
        """Polynomial cutoff going to zero with zero slope at r_cut."""
        x = (r / self.r_cut).clamp(max=1.0)
        return (1.0 - x) ** 3 * (1.0 + 3.0 * x + 6.0 * x * x)

    def hopping(self, feats_i, feats_j, r):
        """t_ij, symmetric in (i, j) by construction rather than by penalty.

        The MLP is fed only symmetric functions of the pair -- the sum and the absolute
        difference of the node features -- so t_ij == t_ji identically, for any weights and at
        every point of training. Symmetrising the OUTPUT instead would leave H non-symmetric
        during the backward pass and make `eigh` silently wrong about gradients.
        """
        # MACE's `get_edge_vectors_and_lengths` returns lengths as [n_edges, 1] while the unit
        # tests pass a flat [n_edges]. Normalise to 1-D so both callers work; a stray trailing
        # axis here silently becomes a third dimension in the radial basis.
        r = r.reshape(-1)
        sym = torch.cat([feats_i + feats_j, (feats_i - feats_j).abs(), self._radial(r)],
                        dim=-1)
        return self.hop(sym) * self._envelope(r).unsqueeze(-1)

    def forward(
        self,
        node_feats: torch.Tensor,      # [n_nodes, feature_dim]
        counter_emb: torch.Tensor,     # [n_graphs, counter_dim]
        counts: torch.Tensor,          # [n_graphs, C]
        batch: torch.Tensor,           # [n_nodes]
        num_graphs: int,
        edge_index: torch.Tensor,      # [2, n_edges]
        edge_length: torch.Tensor,     # [n_edges]
        site_bias: Optional[torch.Tensor] = None,   # [n_nodes, C]
    ) -> SpectralOutput:
        device = node_feats.device
        n_nodes = node_feats.shape[0]
        C = self.num_channels

        eps = self.site(torch.cat([node_feats, counter_emb[batch]], dim=-1))   # [n, C]
        if site_bias is not None:
            eps = eps + site_bias

        src, dst = edge_index[0], edge_index[1]
        t = self.hopping(node_feats[src], node_feats[dst], edge_length)        # [e, C]

        # Pack the per-graph Hamiltonians into a padded dense batch. Training cells here are
        # 79-398 atoms, so dense `eigh` is exact, millisecond-scale, and differentiates
        # cleanly -- and the smearing below regularises the eigenvalue crossings that make
        # eigendecomposition derivatives delicate.
        counts_per_graph = torch.bincount(batch, minlength=num_graphs)
        n_max = int(counts_per_graph.max())
        offsets = torch.cumsum(counts_per_graph, 0) - counts_per_graph
        local = torch.arange(n_nodes, device=device) - offsets[batch]          # index in cell

        # Flatten (graph, channel) into one batch dimension so scattering is a single
        # index_put_ with accumulation. Accumulation matters: with periodic images the same
        # (i, j) pair can appear on several edges with different shifts, and plain indexed
        # assignment would keep only the last one.
        ar_c = torch.arange(C, device=device)
        H = torch.zeros((num_graphs * C, n_max, n_max),
                        device=device, dtype=self.solver_dtype)

        node_bc = batch.repeat_interleave(C) * C + ar_c.repeat(n_nodes)
        local_rep = local.repeat_interleave(C)
        H.index_put_((node_bc, local_rep, local_rep),
                     eps.reshape(-1).to(self.solver_dtype), accumulate=True)

        # H_ij = -t_ij, written for each DIRECTED edge only. The neighbour list already
        # contains both (i, j) and (j, i), so writing the symmetric partner as well would
        # double every off-diagonal -- which it did, and the two-site closed form caught it.
        n_edges = src.shape[0]
        edge_bc = batch[src].repeat_interleave(C) * C + ar_c.repeat(n_edges)
        H.index_put_((edge_bc, local[src].repeat_interleave(C),
                      local[dst].repeat_interleave(C)),
                     (-t).reshape(-1).to(self.solver_dtype), accumulate=True)

        # Padded slots sit far above the physical spectrum. Added after the scatters so
        # nothing can accumulate on top of them.
        slot = torch.arange(n_max, device=device)
        pad = slot.unsqueeze(0) >= counts_per_graph.unsqueeze(1)
        pad_value = _PAD_ENERGY + _PAD_SPACING * slot.to(self.solver_dtype)
        pad_bc = (pad.repeat_interleave(C, dim=0).to(self.solver_dtype)
                  * pad_value.unsqueeze(0))
        H += torch.diag_embed(pad_bc)

        # t_ij is symmetric by construction, so H already is; this only removes floating-point
        # asymmetry, which `eigh` would otherwise resolve arbitrarily.
        H = 0.5 * (H + H.transpose(-1, -2))
        H = H.reshape(num_graphs, C, n_max, n_max)

        evals, evecs = torch.linalg.eigh(H)                    # ascending
        m = min(self.num_states, n_max)
        lam = evals[..., :m]                                   # [G, C, m]
        psi = evecs[..., :m]                                   # [G, C, N, m]

        # Thermal smearing over the lowest m. Concentrates on the bound state when one exists
        # and degrades gracefully to a band-edge ensemble when none does -- which is the
        # honest answer for a pristine cell, not a failure to be forced.
        w = torch.softmax(-lam / self.smearing, dim=-1)

        carrier_energy = (w * lam).sum(-1)                     # [G, C]
        delta_sr = (counts.to(carrier_energy.dtype) * carrier_energy).sum(-1)

        dens = (psi.pow(2) * w.unsqueeze(-2)).sum(-1)          # [G, C, N]
        alpha = dens[batch, :, local].to(node_feats.dtype)     # [n_nodes, C]

        # The spectral gap: distance from the occupied state to the rest of the band. Large
        # means a genuinely split-off bound level; ~0 means the carrier is a band state.
        #
        # Padded states must be excluded. When a cell has fewer atoms than `num_states` the
        # lowest-m slice necessarily contains padding, and reading the gap off it would report
        # ~_PAD_ENERGY instead of a physical splitting. They carry no weight in the energy
        # (softmax of -1e4/T_s underflows to zero) but the gap is a plain difference.
        physical = lam < 0.5 * _PAD_ENERGY
        lam_masked = torch.where(physical, lam, torch.full_like(lam, -float("inf")))
        highest = lam_masked.max(dim=-1).values
        gap = torch.where(torch.isfinite(highest), highest - lam[..., 0],
                          torch.zeros_like(lam[..., 0]))

        return SpectralOutput(
            delta_sr=delta_sr.to(node_feats.dtype),
            alpha=alpha,
            site_energy=eps,
            gap=gap.to(node_feats.dtype),
            eigenvalues=lam.to(node_feats.dtype),
            weights=w.to(node_feats.dtype),
        )
