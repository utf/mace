"""Edit 3: bounded matrix elements, and the floors come out.

V3's elements were unbounded learned quantities with floors underneath them:

    t_ij = [t_min + softplus(B(features))] * exp(-(r - r0) / ell[s_i, s_j]) * f_env
    eps_i = MLP(features)

Two failure classes live in that form. The **spectrum shift**, where the whole manifold slides
to satisfy a band-edge constraint without splitting anything off -- V1 reached Delta_bind =
+11.9 eV against a 0.5 eV target. And the **superatom**, where a long-ranged near-complete
graph's ground state is the in-phase mode over the whole cell, which E2 measured directly at a
10 A reach. Neither is a bug to be found; both are available minima of an unbounded form.

The bounded form has no representation for either:

    t_ij  = V0[s_i, s_j] * f(r) * (1 + 0.5 * tanh(g(D_i, D_j, e_i, e_j, r)))
    eps_i = eps0[s_i] - phi_LR_i / eps_inf + 1.0 eV * tanh(h(D_i, e_i))

The learned parts are now CORRECTIONS to a physical scale rather than the scale itself. The
hopping's feature dependence can only move it within a factor of two of its Harrison value,
and the on-site's within +-1 eV of a per-species constant. A manifold cannot slide by 12 eV
because nothing in the form can produce 12 eV.

WHAT IS REMOVED, and why each one has to go:

* **`t_min`.** It coupled every pair inside the envelope by at least 0.02 eV whatever the
  data said, which is a floor on delocalisation. The bounded form replaces it -- the
  amplitude is a real physical scale, so there is nothing left to protect against collapsing
  to zero.
* **The learned per-species decay length and its 0.3 A floor.** M3 found `|t'|` unmoved
  between two arms and sitting at the zero-feature profile slope, i.e. prior-dominated; a
  floored, learned decay is the mechanism that would produce exactly that. The decay is now
  ONE fixed global constant, so it is a stated prior rather than a parameter pretending to be
  fitted.

HARRISON SCALE. `V0` is initialised at Harrison's universal ss-sigma value,
`V_ss_sigma = -1.40 * hbar^2 / (m d^2)`, with `hbar^2/m = 7.62 eV A^2` and `d` the MEASURED
median nearest-neighbour distance of the data -- about -1.36 eV at 2.80 A. It stays learnable
(one symmetric matrix over species pairs, six numbers for three species), because a universal
parameter is a starting scale and not a measurement of this material. The boundedness that
matters is on the feature-dependent factor, which is where the escape routes lived.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from mace.modules.defect_spectral import _mlp
from mace.modules.defect_spectral_v3 import LocalSpectralHead

__all__ = ["BoundedLocalHead", "install_bounded_elements", "HBAR2_OVER_M", "ETA_SS_SIGMA"]

HBAR2_OVER_M = 7.62          # eV A^2
ETA_SS_SIGMA = -1.40         # Harrison's universal ss-sigma coefficient
ON_SITE_RANGE = 1.0          # eV, the half-range of the bounded on-site correction
HOP_RANGE = 0.5              # the bounded hopping correction spans (1 - 0.5, 1 + 0.5)


class BoundedLocalHead(LocalSpectralHead):
    """V3's descriptor and locality, with bounded elements over physical scales."""

    def __init__(self, *, decay_length_fixed: float = 1.0, d_ref: float = 2.8,
                 hidden: int = 64, **kw) -> None:
        # t_min is not merely set to zero -- the bounded form is its replacement, so a value
        # left in place would be a floor nobody remembered adding.
        kw["t_min"] = 0.0
        self_feature_dim = int(kw["feature_dim"])
        super().__init__(hidden=hidden, **kw)
        self.feature_dim = self_feature_dim
        self.decay_length_fixed = float(decay_length_fixed)
        self.d_ref = float(d_ref)

        n = self.elem.num_embeddings
        harrison = ETA_SS_SIGMA * HBAR2_OVER_M / (self.d_ref ** 2)
        self.v0_raw = nn.Parameter(torch.full((n, n), float(harrison)))
        self.eps0 = nn.Parameter(torch.zeros(n))

        # Rebuilt to emit ONE number per node/edge: the bounded correction, on explicit
        # widths rather than on the widths of the layers being replaced. The parent's on-site
        # input includes the counter embedding when single_manifold is off; the bounded
        # on-site is counter-free by construction, so inheriting that width would silently
        # feed it 32 columns of something else.
        fdim, edim = int(self.feature_dim), int(self.elem.embedding_dim)
        self.site = _mlp([fdim + edim, hidden, hidden, 1], final_scale=0.05)
        self.hop = _mlp([2 * fdim + 2 * edim + self.radial_dim, hidden, hidden, 1],
                        final_scale=0.05)
        # The learned decay is gone. Zeroing the buffer would leave a live parameter whose
        # gradient goes nowhere; deleting it makes the removal visible in the state dict.
        if hasattr(self, "decay_raw"):
            del self._parameters["decay_raw"]

    # ------------------------------------------------------------------ elements

    def v0(self, species_i, species_j):
        """Symmetric per-species-pair hopping scale."""
        m = 0.5 * (self.v0_raw + self.v0_raw.T)
        return m[species_i, species_j]

    def radial_shape(self, r: torch.Tensor) -> torch.Tensor:
        """f(r): one fixed global decay length, plus the truncating envelope.

        Harrison's own 1/d^2 scaling is the tempting choice here and is wrong at this reach:
        at 10 A it leaves t(10)/t(2.8) at 8% across ~100 neighbours, which is the
        nearly-complete graph whose ground state is the superatom mode. The exponential is a
        stated prior about how hopping decays (0.7-1.0 A in real solids), fixed rather than
        fitted, which is exactly what M3 says the previous form only pretended to be.
        """
        decay = torch.exp(-(r - self.d_ref) / self.decay_length_fixed)
        return decay * self._envelope(r)

    def hopping(self, feats_i, feats_j, r, species_i=None, species_j=None):
        if species_i is None or species_j is None:
            raise ValueError("bounded hopping needs node_species; the head was given none")
        r = r.reshape(-1)
        e_i, e_j = self.elem(species_i), self.elem(species_j)
        # Same symmetric construction as V3: t_ij == t_ji identically, for any weights, at
        # every point of training. Symmetrising the output instead leaves H non-symmetric
        # during the backward pass and eigh's gradients silently wrong.
        sym = torch.cat([feats_i + feats_j, (feats_i - feats_j).abs(),
                         e_i + e_j, (e_i - e_j).abs(), self._radial(r)], dim=-1)
        correction = 1.0 + HOP_RANGE * torch.tanh(self.hop(sym))       # [e, 1]
        scale = self.v0(species_i, species_j).unsqueeze(-1)            # [e, 1]
        t = scale * self.radial_shape(r).unsqueeze(-1) * correction
        return (self.hop_scale * t).expand(-1, self.num_channels)

    def _site_energies(self, node_feats, counter_emb, batch, node_species,
                       edge_index, edge_length, edge_vector, n_nodes):
        if node_species is None:
            raise ValueError("bounded on-site needs node_species; the head was given none")
        e = self.elem(node_species)
        base = self.eps0[node_species].unsqueeze(-1)
        correction = ON_SITE_RANGE * torch.tanh(
            self.site(torch.cat([node_feats, e], dim=-1)))
        return (base + correction).expand(-1, self.num_channels)

    # ------------------------------------------------------------------ diagnostics

    def hopping_profile(self, distances, species_i: int = 0, species_j: int = 0,
                        feature_dim: Optional[int] = None):
        """Realised |t| at zero features -- the prior alone, with the bounded factor at 1."""
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        r = torch.tensor([float(x) for x in distances], device=dev, dtype=dt)
        si = torch.full((len(r),), int(species_i), device=dev, dtype=torch.long)
        sj = torch.full((len(r),), int(species_j), device=dev, dtype=torch.long)
        d = feature_dim if feature_dim is not None else self.hop[0].in_features
        fdim = (d - 2 * self.elem.embedding_dim - self.radial_dim) // 2
        f = torch.zeros(len(r), fdim, device=dev, dtype=dt)
        with torch.no_grad():
            t = self.hopping(f, f, r, si, sj)
        return t.abs().max(dim=-1).values.detach().cpu().numpy()


def install_bounded_elements(model, d_ref: Optional[float] = None,
                             decay_length_fixed: float = 1.0) -> BoundedLocalHead:
    """Replace a V3 local head with the bounded form, in place.

    Requires the V3 head to be installed first: this is Edit 3 on top of Edit 1, and the
    stage harness composes them in that order. Building it from a non-local head would give
    bounded elements over a descriptor that is not local at r_max, which is a different
    architecture wearing this one's name.
    """
    head = getattr(model, "spectral", None)
    if not isinstance(head, LocalSpectralHead):
        raise ValueError(
            f"install_bounded_elements needs the V3 local head, found {type(head).__name__}. "
            "Stage 2 is Edit 3 ON TOP of Edit 1; install_local_head runs first.")
    feature_dim = int(getattr(model, "spectral_feature_dim", 0)) or int(
        head.hop[0].in_features - 2 * head.elem.embedding_dim - head.radial_dim) // 2
    new = BoundedLocalHead(
        feature_dim=feature_dim,
        counter_dim=int(head.counter_dim),
        num_elements=int(head.elem.num_embeddings),
        r_max=float(head.r_max), r_couple=float(head.r_couple),
        elem_dim=int(head.elem.embedding_dim),
        hidden=int(head.site[0].out_features),
        radial_dim=int(head.radial_dim),
        num_channels=head.num_channels, num_states=head.num_states,
        smearing=head.smearing, gauge_penalty=bool(head.gauge_penalty),
        single_manifold=bool(getattr(head, "single_manifold", False)),
        d_ref=float(d_ref) if d_ref is not None else 2.8,
        decay_length_fixed=float(decay_length_fixed),
    )
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    model.spectral = new.to(device=device, dtype=dtype)
    model.spectral_bounded = True
    return model.spectral
