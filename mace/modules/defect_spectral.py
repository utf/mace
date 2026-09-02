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

from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
from torch import nn

__all__ = ["SpectralCarrierHead", "SpectralOutput", "bandwidth_scale"]


def bandwidth_scale(epoch: int, s0: float, anneal_epochs: int) -> float:
    """Bandwidth anneal factor: s(e) = s0^(1 - e/E_a) for e <= E_a, then 1.

    All hoppings are multiplied by this, so the Hamiltonian starts with a band s0 times
    wider and narrows to its trained width by E_a. A level can then separate from the band
    gradually, instead of having to tunnel out of an already-converged delocalised solution.

    This lives here, as a pure function of the epoch, so it can be tested against its own
    definition. The schedule was previously implicit: `--defect_spectral_anneal_s0` was
    declared in the parser and passed by the launcher, but nothing ever assigned hop_scale,
    which therefore sat at its registered 1.0 for the whole run. The anneal arm was byte-
    identical to the plain arm, and no test caught it because every test set hop_scale by
    hand. s0 <= 0 disables the anneal.
    """
    # A zero-length anneal is a disabled anneal, not an instantaneous one: returning s0 at
    # epoch 0 would widen the band for exactly one epoch and call it a schedule.
    if s0 <= 0 or int(anneal_epochs) <= 0:
        return 1.0
    span = int(anneal_epochs)
    e = min(max(int(epoch), 0), span)
    return float(s0) ** (1.0 - e / span)


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
    eps_mean: torch.Tensor      # [n_graphs, C]  per-frame mean site energy (T5 gauge)
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
        num_elements: int = 1,
        use_decay: bool = False,      # H1
        t_min: float = 0.02,          # eV, hopping floor
        decay_r0: float = 3.0,        # A, reference distance
        decay_init: float = 1.0,      # A, initial decay length
        use_sigma: bool = False,      # H3: dangling-orbital sigma term
        sigma_r1: float = 3.6,        # A, first-shell radius for the dangling vector
        gauge_penalty: bool = False,  # T5: penalise mean(eps) instead of subtracting it
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_states = num_states
        self.smearing = smearing
        self.r_cut = r_cut
        self.radial_dim = radial_dim
        self.solver_dtype = solver_dtype
        self.use_decay = use_decay
        self.t_min = t_min
        self.decay_r0 = decay_r0
        self.counter_dim = int(counter_dim)   # recorded so a subclass can rebuild `site`
        self.gauge_penalty = gauge_penalty
        # Bandwidth anneal factor, set per epoch from outside. A buffer so it travels
        # with the model and is visible in checkpoints rather than being a hidden global.
        self.register_buffer("hop_scale", torch.ones(()))

        # Level position, held separately from the site energies (mandatory gauge anchor).
        #
        # eps has an exact uniform mode: adding a constant c to every site shifts every
        # eigenvalue by c and dE_SR by n_c * c. With a trainable base that shift is partly
        # absorbable, so eps wandered to 12-30 eV in E2 while carrying no extra information.
        # Removing the per-frame mean kills the mode exactly, and mu_c restores the one degree
        # of freedom that was doing real work -- where the level sits -- as a single scalar
        # rather than as an offset smeared across N site energies.
        self.mu = nn.Parameter(torch.zeros(num_channels))

        # s_c in dE_SR = sum_c s_c n_c (Lambda + mu_c). Ones by default: the head as originally
        # specified minimised for both carriers, which is right for an added electron and wrong
        # for a hole. A buffer rather than a constant so it travels with the checkpoint and a
        # saved model cannot silently acquire a different convention on reload.
        self.register_buffer("channel_sign", torch.ones(num_channels))

        # H1: per-species-pair decay length. The envelope says what is REACHABLE (10 A, so the
        # hub Pb pair is coupled at all); the decay says what is actually COUPLED. Without it,
        # a 10 A Hamiltonian is nearly a complete graph and every state is broad -- E2 at 10 A
        # sat at participation ~37. Floored at 0.3 A so it cannot collapse to a delta.
        if use_decay:
            self.decay_raw = nn.Parameter(
                torch.full((num_elements, num_elements),
                           float(np.log(np.expm1(max(decay_init - 0.3, 1e-3))))))

        # H3: dangling-orbital sigma term.
        #
        # R0 found the missing physics in the data: the residual force on the two
        # vacancy-sharing Pb lies ALONG their common axis (anisotropy 6.6-7.8 against ~0.9 for
        # the carrier-free null), is 15-19x the null, and scales with their separation. That
        # is a sigma bond between two dangling orbitals pointing at each other -- something a
        # purely scalar hopping cannot express, because it has no notion of direction.
        #
        # v_i = -sum_j f_cut(r_ij) rhat_ij points INTO whatever is missing from atom i's
        # coordination shell: ~0 in bulk by symmetry, large and axial for an atom facing a
        # vacancy. It is smooth, equivariant and thresholdless -- no defect label anywhere,
        # and identically silent for substitutionals, which is why this is a mechanism probe
        # rather than the general answer (H4 is).
        self.use_sigma = use_sigma
        if use_sigma:
            self.sigma_amp = _mlp([2 * feature_dim + radial_dim, hidden, num_channels],
                                  final_scale=0.1)
            self.sigma_site = _mlp([feature_dim, hidden, num_channels], final_scale=0.1)
            self.sigma_r1 = sigma_r1

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
        """Smooth truncation at r_cut: zero value and zero slope, as forces require.

        Two shapes, because the envelope's job changes once H1's decay exists.

        Without decay the envelope is the ONLY thing shaping the hopping with distance, so it
        must taper: the standard (1-x)^3 (1+3x+6x^2).

        With decay, the exponential does the physics and the envelope should only truncate.
        The tapering form then actively harms: at 10 A reach it gives the worst-case hub Pb
        pair (6.80 A) a weight of 0.19, below the 0.3 the pre-flight requires, and the check
        failed on 10 of 60 frames. (1 - x^6)^2 is ~0.95 at the median hub separation and 0.81
        at the worst case, while still vanishing smoothly at r_cut.
        """
        x = (r / self.r_cut).clamp(max=1.0)
        if self.use_decay:
            return (1.0 - x ** 6) ** 2
        return (1.0 - x) ** 3 * (1.0 + 3.0 * x + 6.0 * x * x)

    def dangling_vectors(self, edge_index, edge_vector, n_nodes):
        """v_i = -sum_j f_cut(r_ij; r1) * rhat_ij -- points into whatever is missing.

        Equivariant by construction (it is a sum of unit vectors, so it rotates with the
        frame), smooth, and thresholdless. In a complete coordination shell the terms cancel
        by symmetry and v_i ~ 0; an atom facing a vacancy gets a large vector pointing at it.

        No defect label is involved: this is a function of geometry alone, and it is
        identically silent for a substitutional, which is exactly why H3 is a mechanism probe
        rather than the general solution.
        """
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        rhat = edge_vector / r
        # First-shell weight: ~1 across the bond region, falling to zero with zero slope at
        # r1. The standard (1-x)^3(1+3x+6x^2) taper is useless here -- a real Pb-Cl bond at
        # 2.85 A against r1 = 3.6 A sits at x = 0.79, where it returns 0.066, so every ligand
        # contributes almost nothing and removing one leaves no measurable vector. What this
        # term needs is a plateau over the coordination shell and a sharp edge just below the
        # next shell (Pb-Cs at ~4 A), not a gradual decay.
        x = (r / self.sigma_r1).clamp(max=1.0)
        w = (1.0 - x ** 16) ** 2
        v = torch.zeros(n_nodes, 3, device=edge_vector.device, dtype=edge_vector.dtype)
        v.index_add_(0, src, -w * rhat)
        return v

    def decay_length(self, species_i, species_j):
        """Per-species-pair decay length, symmetric in (i, j) and floored at 0.3 A."""
        raw = 0.5 * (self.decay_raw + self.decay_raw.T)      # symmetric by construction
        return 0.3 + nn.functional.softplus(raw[species_i, species_j])

    def _site_energies(self, node_feats, counter_emb, batch, node_species,
                       edge_index, edge_length, edge_vector, n_nodes):
        """eps_raw, [n_nodes, C], before the gauge and before `site_bias`.

        Split out of `forward` so V3 can override the on-site form without a second copy of
        the eigensolve underneath it. Behaviour here is exactly what forward did inline.

        V2 (T-B): a counted on-site term, eps_i = e(z_i) + sum_j phi(z_i, z_j, r_ij), with no
        trunk features in eps. Installed by surgery on a built model and absent by default, so
        a head that never had it runs exactly the path it always did -- test_rigid_onsite.py
        asserts the bit-identity rather than trusting this comment.
        """
        rigid = getattr(self, "rigid_onsite", None)
        if rigid is not None:
            return rigid(node_species, edge_index, edge_length, n_nodes)        # [n, C]
        return self.site(torch.cat([node_feats, counter_emb[batch]], dim=-1))   # [n, C]

    def hopping(self, feats_i, feats_j, r, species_i=None, species_j=None):
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
        raw = self.hop(sym)

        if not self.use_decay:
            return self.hop_scale * raw * self._envelope(r).unsqueeze(-1)

        # H1: t = [t_min + softplus(B)] * exp(-(r - r0)/l) * envelope.
        #
        # The floor matters as much as the decay. Without it the head can drive t to zero and
        # recover the softmax limit -- independent site energies with no coupling -- which is
        # the regime the spectral head exists to leave. With it, every reachable pair stays
        # coupled by at least t_min and localisation must come from contrast, not from
        # disconnecting the graph.
        amp = self.t_min + nn.functional.softplus(raw)
        ell = self.decay_length(species_i, species_j).unsqueeze(-1)
        decay = torch.exp(-(r.unsqueeze(-1) - self.decay_r0) / ell)
        return self.hop_scale * amp * decay * self._envelope(r).unsqueeze(-1)

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
        node_species: Optional[torch.Tensor] = None,  # [n_nodes] element indices
        clamp_mask: Optional[torch.Tensor] = None,  # [n_nodes] bool, DIAGNOSTIC ONLY
        edge_vector: Optional[torch.Tensor] = None,  # [n_edges, 3], needed by H3
        madelung: Optional[torch.Tensor] = None,  # [n_nodes] Edit 1, see below
        occupations: Optional[torch.Tensor] = None,  # Edit 4 only; ignored here, see below
        internals: Optional[Dict[str, torch.Tensor]] = None,  # DIAGNOSTIC ONLY, see below
    ) -> SpectralOutput:
        """`internals`, when a dict is passed, is filled with H, psi, lam, w, eps and the

        `occupations` belongs to the counting head and is accepted-and-ignored here so that
        MACEDefect has ONE call site for both heads. A branch at the call site is how train
        and evaluate came to disagree about the forward pass; an ignored keyword is cheaper
        than that, and a spectral head has no fill to override -- its occupation is the
        smeared weight `w`, which is not a free input.
        """ + """
        node->(graph, slot) maps, STILL ATTACHED TO THE GRAPH.

        D1 needs to split the head's own axial force into on-site and hopping parts, which
        means contracting against the H the head actually assembled -- for H3 that includes
        the sigma term -- rather than against a formula re-derived in the analysis script.
        Re-deriving would silently drop whatever the head does that the formula forgets.

        Nothing is stored on the module: a tensor stashed as an attribute is not deepcopy-safe
        and would break the cuEq conversion. Default None means production forward is
        bit-identical; `test_internals_capture.py` asserts that.
        """
        device = node_feats.device
        n_nodes = node_feats.shape[0]
        C = self.num_channels

        eps_raw = self._site_energies(node_feats, counter_emb, batch, node_species,
                                      edge_index, edge_length, edge_vector, n_nodes)
        if site_bias is not None:
            eps_raw = eps_raw + site_bias
        # Kept for the A2 diagnostic: the learned on-site term alone, before Edit 1's
        # electrostatics and before the gauge.
        eps_learned = eps_raw

        # Edit 1. `madelung` is already `-phi_LR / eps_inf` -- the sign lives in
        # MadelungOnSite.on_site_shift and nothing here may re-apply it. It is added BEFORE
        # the gauge below, which in Stages 1 and 2 removes the per-frame mean: what acts
        # there is the Madelung CONTRAST, which is what the A1 rock-salt and perovskite
        # tables check. The absolute offset becomes load-bearing only when the gauge goes,
        # at the counting head, where the energy labels pin it.
        if madelung is not None:
            eps_raw = eps_raw + madelung.reshape(-1, 1)

        # Gauge control. eps has an exact uniform mode: adding a constant to every site shifts
        # every eigenvalue and hence dE_SR, which a trainable base can partly absorb -- that is
        # what took eps to 12-30 eV before any anchor existed.
        #
        # Subtracting the per-frame mean removes the mode exactly, but injects an O(1/N) term
        # into lambda whenever the well is localised: the mean is ~depth*k/N, measured as
        # -3*(1 - 1/N) in the toy and ~2.6 meV across the 640-5120 ladder, against a <= 1 meV
        # gate. So the mean is now REPORTED and penalised in the loss instead of subtracted,
        # which pins the gauge without making lambda depend on cell size.
        counts_per_graph = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
        mean_eps = (torch.zeros(num_graphs, C, device=device, dtype=eps_raw.dtype)
                    .index_add_(0, batch, eps_raw) / counts_per_graph.unsqueeze(-1))
        eps = eps_raw if self.gauge_penalty else eps_raw - mean_eps[batch]

        src, dst = edge_index[0], edge_index[1]
        if self.use_decay and node_species is None:
            raise ValueError("H1 decay needs node_species; the head was given none")
        t = self.hopping(node_feats[src], node_feats[dst], edge_length,
                         None if node_species is None else node_species[src],
                         None if node_species is None else node_species[dst])   # [e, C]

        # H3: sigma coupling between dangling orbitals that face each other.
        #
        # (v_i . rhat_ij)(v_j . rhat_ji) is large and positive only when BOTH atoms have a
        # hole in their coordination shell pointing along the bond -- the vacancy pair, and
        # essentially nothing else. It is ~0 on ligands (whose shells are complete) and ~0 in
        # bulk (where v cancels by symmetry), so it adds a channel the scalar hopping cannot
        # express without touching anything else.
        if self.use_sigma:
            if edge_vector is None:
                raise ValueError("H3 sigma term needs edge_vector; the head was given none")
            v = self.dangling_vectors(edge_index, edge_vector, n_nodes)
            rr = edge_vector.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            rhat = edge_vector / rr
            proj_i = (v[src] * rhat).sum(-1, keepdim=True)
            proj_j = -(v[dst] * rhat).sum(-1, keepdim=True)   # rhat_ji = -rhat_ij
            sym = torch.cat([node_feats[src] + node_feats[dst],
                             (node_feats[src] - node_feats[dst]).abs(),
                             self._radial(edge_length.reshape(-1))], dim=-1)
            amp = self.sigma_amp(sym)
            t = t + amp * proj_i * proj_j * self._envelope(edge_length.reshape(-1)).unsqueeze(-1)

            # And an on-site term: |v_i|^2 measures how incomplete atom i's shell is.
            eps = eps + self.sigma_site(node_feats) * (v * v).sum(-1, keepdim=True)

        # Pack the per-graph Hamiltonians into a padded dense batch. Training cells here are
        # 79-398 atoms, so dense `eigh` is exact, millisecond-scale, and differentiates
        # cleanly -- and the smearing below regularises the eigenvalue crossings that make
        # eigendecomposition derivatives delicate.
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

        # R1 clamped-state probe (DIAGNOSTIC ONLY -- production configs must refuse this).
        #
        # Restrict the eigenproblem to a subspace by pushing every atom OUTSIDE the mask to
        # the padded energy. Solving on the subspace, rather than projecting afterwards, keeps
        # the orbital composition optimal within the mask and leaves Hellmann-Feynman
        # gradients valid -- so "how well can this head fit the forces if the carrier is
        # forced onto these atoms?" is answered honestly for each candidate site set.
        # Energies for slots that must sit above the physical spectrum, spaced 1 eV apart so
        # they are mutually distinct. Shared by the clamp and the padding: giving them all the
        # SAME value makes that block exactly degenerate, and eigh refuses. The padding hit
        # this first with 80 identical slots; the clamp reintroduced it with 77.
        slot = torch.arange(n_max, device=device)
        pad_value = _PAD_ENERGY + _PAD_SPACING * slot.to(self.solver_dtype)

        if clamp_mask is not None:
            # A TRUE restriction: zero the couplings to excluded atoms and isolate them at the
            # padded energy, so H is exactly block-diagonal and the masked block's eigenvalues
            # are exactly those of H restricted to the mask.
            #
            # An energy penalty alone is not enough. Raising the excluded diagonals to 1e3
            # while leaving the off-diagonals intact leaves second-order mixing of order
            # t^2/dE ~ 3e-5 eV, and the state leaks off the mask by the same amount. R1
            # compares force fits BETWEEN masks, so a systematic leak of that size is exactly
            # the kind of bias that would make the comparison meaningless.
            #
            # H is [G*C, N, N] here; the reshape to [G, C, N, N] happens after the padding.
            keep = torch.zeros(num_graphs * C, n_max, device=device,
                               dtype=self.solver_dtype)
            keep.index_put_((node_bc, local_rep),
                            clamp_mask.to(self.solver_dtype).repeat_interleave(C),
                            accumulate=True)
            H = H * keep.unsqueeze(-1) * keep.unsqueeze(-2)
            H = H + torch.diag_embed((1.0 - keep) * pad_value.unsqueeze(0))

        # Padded slots sit far above the physical spectrum. Added after the scatters so
        # nothing can accumulate on top of them.
        pad = slot.unsqueeze(0) >= counts_per_graph.unsqueeze(1)
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

        if internals is not None:
            internals["H"] = H
            internals["psi"] = psi
            internals["lam"] = lam
            internals["eps"] = eps
            # Pre-gauge, and pre-Madelung. A2 clause 1 is a statement about the LEARNED
            # on-site term, which `eps` is not: the difference gauge subtracts a per-cell
            # mean, so eps differs between two cells by a constant even when every learned
            # value is bit-identical.
            internals["eps_raw"] = eps_learned
            internals["batch"] = batch
            internals["local"] = local

        # Thermal smearing over the lowest m. Concentrates on the bound state when one exists
        # and degrades gracefully to a band-edge ensemble when none does -- which is the
        # honest answer for a pristine cell, not a failure to be forced.
        w = torch.softmax(-lam / self.smearing, dim=-1)
        if internals is not None:
            internals["w"] = w

        # dE_SR = n_c * (lambda + mu_c). lambda now measures the level RELATIVE to the frame's
        # mean site energy, because the uniform mode was removed; mu_c carries its absolute
        # position as one learned scalar per channel.
        carrier_energy = (w * lam).sum(-1) + self.mu.to(lam.dtype).unsqueeze(0)   # [G, C]
        # Per-channel sign on the energy contribution. Lambda is the lowest eigenvalue of a
        # bonding-signed H, which is an ADDED ELECTRON's level: eps - t(d), falling as a pair
        # closes. A hole removed from a bonding state is the negative of that -- -eps + t(d),
        # RISING as the pair closes -- and a minimum eigenvalue cannot represent it, since a
        # minimum lies at or below the smallest diagonal while the hole's level lies ABOVE its
        # on-site energy by t. The sign therefore has to come from the counter, not from the
        # solver. Registered as a buffer of ones by default, so every existing head is
        # bit-identical and the 24 saved V3 checkpoints keep meaning what they meant.
        # getattr, not attribute access: heads pickled before this buffer existed restore a
        # __dict__ without it, and every saved checkpoint would otherwise fail to forward.
        # Absent means the original convention, which is what those checkpoints were trained
        # under -- so they keep meaning exactly what they meant.
        sign = getattr(self, "channel_sign", None)
        if sign is None:
            sign = torch.ones(C, device=carrier_energy.device, dtype=carrier_energy.dtype)
        delta_sr = (sign.to(carrier_energy.dtype).unsqueeze(0)
                    * counts.to(carrier_energy.dtype) * carrier_energy).sum(-1)

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
            eps_mean=mean_eps.to(node_feats.dtype),
            delta_sr=delta_sr.to(node_feats.dtype),
            alpha=alpha,
            site_energy=eps,
            gap=gap.to(node_feats.dtype),
            eigenvalues=lam.to(node_feats.dtype),
            weights=w.to(node_feats.dtype),
        )
