"""V3: a carrier Hamiltonian whose matrix elements are local by construction.

T-B on V2 showed the band-edge inequality is satisfiable without binding anything. The
mechanism was visible in the reference: lambda_1(pristine) floated over ~8 eV while
Delta_bind sat at ~m, so the constraint was met by moving the host origin rather than by
splitting a level off the continuum. V1 made that explicit, overshooting m by up to 24x.

V3 removes the freedom rather than penalising it. Every matrix element is a function of the
trunk's FIRST interaction block only, detached:

    D_i     = block-1 output for atom i, no gradient back into the trunk
    eps_i^c = MLP(D_i, E(z_i), c) + c(D_i) * |v_i|^2
    t_ij^c  = [t_min + softplus(B(D_i, D_j, E(z_i), E(z_j), c))] * g(r_ij) * f_env(r_ij; r_couple)
              + A(D_i, D_j) * (v_i . rhat_ij)(v_j . rhat_ji) * g * f_env
    v_i     = -sum_j f(r_ij / r_max) rhat_ij

One aggregation at r_max means an atom further than r_max from the vacancy has a descriptor
IDENTICAL to its pristine counterpart -- not approximately, by construction. So the far block
of H is pinned to the host, the continuum cannot slide, and Delta_bind measured against
pristine and against the defect cell's own far block must agree. Their agreement is a check
rather than an assumption (gate ��4), and the wholesale-lowering escape has no representation
at all.

TWO LENGTH CONSTANTS, both material-agnostic:

    r_max     the trunk's first-block cutoff -- descriptor reach, and the deficit envelope
    r_couple  the coupling reach (~10 A), so the vacancy-flanking pair at 5.3-6.8 A is coupled

Nothing else in this head has units of length. The parent's two material-specific constants
are both retired into r_max: `sigma_r1 = 3.6 A` (a CsPbCl3 first-shell radius, chosen against
the Pb-Cl bond and the Pb-Cs second shell) and `decay_r0 = 3.0 A` (a reference distance of the
same provenance). The plan's deficit formula already specifies r_max for the first; the second
is referenced at r_max so the exponential uses only sanctioned constants.

Re-referencing changes the initial magnitude, and the fix for that goes through the
AMPLITUDE, not the decay length. Lengthening l to 2.5 A restores the magnitude but gives
t(10 A)/t(2.85 A) ~ 6% across ~100 neighbours -- the long-ranged, nearly complete graph whose
lowest state is the in-phase superatom mode, which is what E2 measured at 10 A reach. Physical
hopping decays over ~0.7-1.0 A, so `decay_init` stays at 1.0 and the initial scale is set by
calibrating the amplitude bias instead (`t_ref_r`, `t_ref_value`).

The calibration target is the REALISED profile, which is what the physics constrains:

    t(first neighbour, ~2.85 A)   0.3-1.0 eV
    t(2 x bond, ~5.6 A)           5-20% of that
    t(r_couple, 10 A)             < 1% of that

`t_ref_r` is a calibration input and not a cutoff -- it is used once, at construction, and
never appears in the functional form. The harness passes the measured median nearest-neighbour
distance of the data, so no material-specific length enters the head. `hopping_profile` reports
the realised values at init and at the end of every run, the latter being the
effective-coupling-length diagnostic.

THE DETACH BOUNDARY, decided explicitly because the plan admits two readings. Section 2 says
the response channel takes a "trunk-feature effective charge and polarisability", and also
that "the trunk trains on the base loss only". In T-B the trunk is frozen so the two coincide,
which is exactly why the choice has to be recorded now rather than discovered at the
from-scratch step: V3 mode detaches the response channel's feature inputs TOO, so no head
term carries gradient into the trunk under any run configuration. `test_spectral_v3.py`
asserts every trunk parameter's grad is None after a head-loss backward.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from mace.modules.defect_spectral import SpectralCarrierHead, _mlp

__all__ = ["LocalSpectralHead"]


class LocalSpectralHead(SpectralCarrierHead):
    """H3's algebra, with a detached block-1 descriptor and no material-specific lengths."""

    def __init__(self, *, feature_dim: int, counter_dim: int, num_elements: int,
                 r_max: float, r_couple: float, elem_dim: int = 8,
                 hidden: int = 64, radial_dim: int = 8, decay_init: float = 1.0,
                 t_ref_r: Optional[float] = None, t_ref_value: float = 0.5,
                 single_manifold: bool = False, **kw) -> None:
        # The deficit envelope and the decay reference are both r_max; the coupling reach is
        # r_couple. use_decay and use_sigma are not optional in V3 -- the decay is what keeps
        # a 10 A Hamiltonian from being a complete graph, and the sigma term is the only
        # element that distinguishes a facing pair of incomplete shells.
        kw.pop("use_decay", None)
        kw.pop("use_sigma", None)
        kw.pop("sigma_r1", None)
        kw.pop("decay_r0", None)
        kw.pop("r_cut", None)
        super().__init__(feature_dim=feature_dim, counter_dim=counter_dim,
                         num_elements=num_elements, hidden=hidden, radial_dim=radial_dim,
                         r_cut=float(r_couple), use_decay=True, use_sigma=True,
                         sigma_r1=float(r_max), decay_r0=float(r_max),
                         decay_init=float(decay_init), **kw)
        self.r_max = float(r_max)
        self.r_couple = float(r_couple)
        self.elem = nn.Embedding(num_elements, elem_dim)

        # Rebuilt with the element embedding as an explicit input. The parent reaches species
        # only through the pair decay length, which cannot express an element-dependent on-site
        # level at all -- and an on-site level per element is the cheapest true statement in
        # any tight-binding model.
        # ONE electron Hamiltonian, sign from the counters.
        #
        # H_e is a single manifold: eps_i(D_i, E(z_i)) with NO counter conditioning, H_ij =
        # -t_ij with the bonding sign. Its lowest smeared eigenvalue Lambda is an added
        # electron's level. The hole's contribution is -(Lambda + mu_h), applied through
        # `channel_sign` -- because a hole removed from a bonding state has energy -eps + t(d),
        # which RISES as the pair closes, and no minimum eigenvalue can represent that: a
        # minimum lies at or below the smallest diagonal, while the hole's level lies above its
        # on-site energy by t.
        #
        # The four channels are kept at the interface by broadcasting one manifold across
        # them, so `channel_of`, the D1 scripts, the four-channel training log and every other
        # consumer keep working unchanged. Spin channels share H_e; mu_c stays per channel.
        self.single_manifold = bool(single_manifold)
        out_dim = 1 if self.single_manifold else self.num_channels
        site_in = (feature_dim + elem_dim if self.single_manifold
                   else feature_dim + counter_dim + elem_dim)
        self.site = _mlp([site_in, hidden, hidden, out_dim], final_scale=0.01)
        self.hop = _mlp([2 * feature_dim + 2 * elem_dim + radial_dim, hidden, hidden,
                         out_dim], final_scale=0.05)
        if self.single_manifold:
            s = torch.ones(self.num_channels)
            s[2:] = -1.0                     # (e_maj, e_min, h_maj, h_min) -> (+1, +1, -1, -1)
            self.channel_sign.copy_(s)

        # Connectivity is fixed through the AMPLITUDE, not the decay length.
        #
        # Lengthening the decay to compensate for the moved reference was the wrong lever: at
        # l = 2.5 A the profile gives t(10 A)/t(2.85 A) ~ 6% over ~100 neighbours, which is the
        # long-ranged, nearly complete graph that produced the superatom ground state in E2.
        # Physical hopping decays over ~0.7-1.0 A, so l stays there and the initial magnitude
        # is set by calibrating the amplitude instead.
        #
        # `t_ref_r` is a calibration input, NOT a cutoff: it appears once, at construction, to
        # place the initial scale, and never in the functional form. The harness passes the
        # MEASURED median nearest-neighbour distance of the data rather than a constant, so no
        # material-specific length is baked into the head.
        if t_ref_r is not None:
            with torch.no_grad():
                r = torch.tensor([float(t_ref_r)])
                factor = float(torch.exp(-(r - self.decay_r0) / float(decay_init))
                               * self._envelope(r))
                target_amp = float(t_ref_value) / max(factor, 1e-12)
                excess = max(target_amp - float(self.t_min), 1e-6)
                # softplus^-1, so amp = t_min + softplus(bias) lands on target_amp
                bias = float(torch.log(torch.expm1(torch.tensor(excess))))
                self.hop[-1].bias.fill_(bias)

    def hopping_profile(self, distances, species_i: int = 0, species_j: int = 0,
                        feature_dim: Optional[int] = None):
        """Realised |t| at the given distances, for zero features -- the diagnostic item 6 asks
        for. Logged at init and at the end of every run; the end-of-run values are the
        effective-coupling-length measurement."""
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        n = len(distances)
        d = feature_dim if feature_dim is not None else self.hop[0].in_features
        fdim = (d - 2 * self.elem.embedding_dim - self.radial_dim) // 2
        f = torch.zeros(n, fdim, device=dev, dtype=dt)
        r = torch.tensor([float(x) for x in distances], device=dev, dtype=dt)
        si = torch.full((n,), int(species_i), device=dev, dtype=torch.long)
        sj = torch.full((n,), int(species_j), device=dev, dtype=torch.long)
        with torch.no_grad():
            t = self.hopping(f, f, r, si, sj)
        return t.abs().max(dim=-1).values.detach().cpu().numpy()

    # -------------------------------------------------------------- descriptor boundary

    def forward(self, node_feats: torch.Tensor, *args, **kwargs):
        """Detach the descriptor once, here, so every element below is trunk-independent.

        Doing it at the entry point rather than per-term is deliberate: a term added later
        cannot accidentally reconnect the trunk, because there is no attached tensor in scope.
        """
        return super().forward(node_feats.detach(), *args, **kwargs)

    # -------------------------------------------------------------- matrix elements

    def _site_energies(self, node_feats, counter_emb, batch, node_species,
                       edge_index, edge_length, edge_vector, n_nodes):
        if node_species is None:
            raise ValueError("V3 on-site term needs node_species; the head was given none")
        e = self.elem(node_species)
        if getattr(self, "single_manifold", False):
            # Counter-free by construction, which also removes a trap: pristine frames can now
            # be scored natively for Delta_bind, so the hole-counter override and its
            # canonicalisation/m_s_ref_doubled bookkeeping drop out of that path entirely.
            return self.site(torch.cat([node_feats, e], dim=-1)).expand(-1, self.num_channels)
        return self.site(torch.cat([node_feats, counter_emb[batch], e], dim=-1))

    def hopping(self, feats_i, feats_j, r, species_i=None, species_j=None):
        """t_ij with element embeddings, symmetric in (i, j) by construction.

        Only symmetric functions of the pair are fed in -- sums and absolute differences of
        both the descriptors and the element embeddings -- so t_ij == t_ji identically at
        every point of training, for any weights. Symmetrising the output instead would leave
        H non-symmetric during the backward pass and make eigh's gradients silently wrong.
        """
        if species_i is None or species_j is None:
            raise ValueError("V3 hopping needs node_species; the head was given none")
        r = r.reshape(-1)
        e_i, e_j = self.elem(species_i), self.elem(species_j)
        sym = torch.cat([feats_i + feats_j, (feats_i - feats_j).abs(),
                         e_i + e_j, (e_i - e_j).abs(), self._radial(r)], dim=-1)
        raw = self.hop(sym)
        if getattr(self, "single_manifold", False):
            raw = raw.expand(-1, self.num_channels)
        amp = self.t_min + nn.functional.softplus(raw)
        ell = self.decay_length(species_i, species_j).unsqueeze(-1)
        decay = torch.exp(-(r.unsqueeze(-1) - self.decay_r0) / ell)
        return self.hop_scale * amp * decay * self._envelope(r).unsqueeze(-1)


def install_local_head(model, r_couple: Optional[float] = None, elem_dim: int = 8,
                       t_ref_r: Optional[float] = None, t_ref_value: float = 0.5,
                       single_manifold: bool = False) -> LocalSpectralHead:
    """Replace a built model's spectral head with the V3 form, in place.

    Asserts the descriptor really is block 1. `MACEDefect` selects it by slicing the first
    `spectral_feature_dim` columns of the concatenated per-block defect readouts, which is
    block 1 ONLY when there is more than one readout: with a single readout the assembly uses
    `feat_idx = -1` and every entry is the LAST block instead. That substitution is invisible
    at every later layer -- the tensor has the right width and the model trains -- so it is
    checked at construction, where it is cheap, as well as by the trunk-independence test.
    """
    head = getattr(model, "spectral", None)
    if head is None:
        raise ValueError("model has no spectral head to replace")
    if not bool(getattr(model, "spectral_first_shell", False)):
        raise ValueError(
            "V3 needs spectral_first_shell=True: without it the head is handed the "
            "concatenated features of every block, not the first block's, and the "
            "descriptor is no longer local at r_max.")
    n_readouts = len(getattr(model, "readouts", []))
    if n_readouts < 2:
        raise ValueError(
            f"V3 needs more than one readout (found {n_readouts}): MACEDefect builds its "
            "per-block defect features with feat_idx = -1 when there is a single readout, so "
            "the 'first block' slice would silently be the LAST block's features.")

    r_max = float(model.r_max)
    couple = float(r_couple if r_couple is not None
                   else getattr(model, "spectral_r_cut", 0.0) or r_max * 2.0)
    new = LocalSpectralHead(
        feature_dim=int(model.spectral_feature_dim),
        counter_dim=int(head.counter_dim),
        num_elements=int(model.atomic_numbers.numel()),
        r_max=r_max, r_couple=couple, elem_dim=elem_dim,
        t_ref_r=t_ref_r, t_ref_value=t_ref_value, single_manifold=single_manifold,
        num_channels=head.num_channels, num_states=head.num_states,
        smearing=head.smearing, radial_dim=head.radial_dim,
        t_min=head.t_min, gauge_penalty=bool(head.gauge_penalty),
    )
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    model.spectral = new.to(device=device, dtype=dtype)
    model.spectral_local = True          # read by the response channel's detach decision
    model.spectral_single_manifold = bool(single_manifold)
    return model.spectral
