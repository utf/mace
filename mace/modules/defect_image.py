"""Addendum sections 5.2 and 6.1: the image-active density and the boundary functional.

THE DENSITIES (section 5.2). With the exact counts `n_c,sigma` fixed by the state and the
per-channel normalised densities `rho^_c,sigma[P]` (integral 1) and localisation weights
`w_c,sigma`:

    rho_F[P]      = - sum n_e,sigma rho^_e,sigma + sum n_h,sigma rho^_h,sigma     int = q_F
    rho_F^img[P]  = - sum n_e,sigma w_e,sigma rho^_e,sigma + sum n_h,sigma w_h,sigma rho^_h,sigma
    rho_img[P]    = rho_S + rho_F^img[P]                                          int = q_img
    rho_Delta[P]  = rho_S + rho_F[P]                                              int = Q_formal
    q_img         = Q_core - sum n_e w_e + sum n_h w_h

`q_img = Q_formal` only when every active channel is compact (`w = 1`); the `(1 - w)` fraction
of an extended carrier keeps its orbital density in `rho_F` but is not treated as a compact
charge with an isolated self-image.

THE KERNELS (section 6.1). Before explicit polarisation every kernel is the bare Green
function screened by the registered `eps_inf`:

    K_inf = G_inf^0 / eps_inf,   K_PBC = G_PBC^0 / eps_inf,   K_img = K_PBC - K_inf

and the difference is an OPERATOR difference, never a pointwise one on a common grid:

    B_{K_img}[rho, eta] = B_{K_PBC}^{T^3}[rho, eta] - B_{K_inf}^{R^3}[U_h rho, U_h eta]

The periodic form acts on the torus representation (LES, with the jellium background and the
zero-mode convention the labels were computed under). The isolated form acts ONLY on the
canonical lift `U_h` of section 4.2 (`defect_lift`): a wrapped density handed to an isolated
kernel is prohibited, and the legacy frontier term's `isolated_energy(q, positions)` on
wrapped positions is exactly the thing this module does not do.

THE FUNCTIONAL. Two terms, each containing its interactions exactly once:

    Phi_SF^{inf,LR}[P] = B_{K_inf^LR}[U rho_S, U rho_F[P]]      boundary-common
    Phi_img^PBC[P]     = 1/2 B_{K_img}[rho_img[P], rho_img[P]]  Phi_img^inf = 0

`K_inf^LR = L_{r_split}[K_inf]` is the registered long-range filter: the complement of the
polynomial cutoff envelope that `defect_madelung.short_range_potential` already uses, so the
split is the one convention throughout. The static--frontier interaction is therefore the
long-range part under the isolated kernel (the same under both boundaries), and the periodic
image addition -- static--static, static--frontier and frontier--frontier images, once each --
comes only from `Phi_img`. No separate image potential or image energy may duplicate it
(section 6.1); the model's regime switch (`image_functional`) enforces that.

THE ISOLATED KERNEL IS A DIRECT SUM. LES's isolated evaluator is a real-space sum over pairs,
`erf(r / (sigma sqrt 2)) / r`, which is `0 / 0` for two DIFFERENT primitives at the same
point -- and the image-active density has exactly those: the frontier charge of a present
atom sits on the atom that carries its static charge, and on an unrattled reference geometry a
present atom sits on its pristine site. `isolated_bilinear` below is the same kernel with the
analytic `r -> 0` limit, summed directly (a few hundred primitives, so a few tens of thousands
of pairs). `test_image_functional.py` pins it against LES's evaluator where the two are both
defined, so the periodic and isolated halves of `K_img` share one smearing convention.

FACTOR OF TWO. `LatentEwald.energy` is the ENERGY, `E = 1/2 q^T A q`, so `1/2 B[rho, rho]`
is `E[rho]` and `Phi_img^PBC = (E_PBC[rho] - E_inf[U rho]) / eps_inf` -- with no further
half. The legacy `Phi_FF` writes `0.5 * (E_PBC - E_inf)`, which is a quarter of `B_img`, half
of what plan v8 section 2.8 defines. That path is left as it is (the Stage-1 reference and
the runs in flight depend on its numerics) and the discrepancy is recorded in the tracker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mace.modules.defect_density import GaussianDensity
from mace.modules.defect_lift import LiftRecord, lift_positions
from mace.modules.defect_madelung import COULOMB_CONSTANT

__all__ = ["IMAGE_REGIMES", "Channel", "ImageDensity", "image_active_density",
           "long_range_envelope", "isolated_pair_kernel", "isolated_bilinear",
           "isolated_self_energy", "converged_dl", "coulomb_constant_of", "lifted_centres",
           "phi_sf_lr",
           "phi_img_pbc", "boundary_functional"]

#: The model's `image_functional`: the legacy Stage-1.2 frontier-frontier patch, or the
#: unified functional of this module (addendum 6.1). A regime, not a term: the two may
#: never both be evaluated.
IMAGE_REGIMES = ("frontier_ff", "unified")
#: Below this separation (A) the pair kernel is taken from its even series in `r^2`. The
#: series is accurate to 1e-16 there and, unlike `erf(r) / r` evaluated at `r -> 0`, has a
#: finite derivative with respect to the centres.
_NEAR = 1e-3
#: The polynomial cutoff order of the registered filter (`short_range_potential`, p = 6).
_ENVELOPE_P = 6


@dataclass(frozen=True)
class Channel:
    """One active frontier channel: its normalised site density, count, sign and weight."""

    name: str                 # "e0", "h1", ...
    density: torch.Tensor     # [n_sites], integral 1, attached
    count: int                # n_c,sigma (exact)
    sign: float               # -1 for electrons, +1 for holes
    weight: torch.Tensor      # w_c,sigma in [0, 1], attached


@dataclass(frozen=True)
class ImageDensity:
    """`rho_S`, `rho_F`, `rho_F^img`, `rho_img` and `rho_Delta` of one graph, one width."""

    static: GaussianDensity
    frontier: GaussianDensity
    frontier_img: GaussianDensity
    q_core: int
    q_f: int
    q_img: torch.Tensor
    weights: Dict[str, torch.Tensor]

    @property
    def img(self) -> GaussianDensity:
        return self.static + self.frontier_img

    @property
    def delta(self) -> GaussianDensity:
        return self.static + self.frontier

    @property
    def q_formal(self) -> int:
        return int(self.q_core) + int(self.q_f)

    @property
    def compact(self) -> bool:
        """Every active channel at exactly `w = 1` -- the plateau on which
        `rho_img = rho_Delta` (section 5.2)."""
        return all(float(w) == 1.0 for w in self.weights.values())


def image_active_density(static: GaussianDensity, channels: Sequence[Channel],
                         positions: torch.Tensor, q_core: int) -> ImageDensity:
    """`rho_F`, `rho_F^img` and their sums with `rho_S` from the active channels.

    The channels are built separately and combined with their own signs and counts; nothing
    here reads the sign of `q_F`. Channels with zero count are simply absent."""
    n_sites = positions.shape[0]
    dtype, device = positions.dtype, positions.device
    site = torch.zeros(n_sites, dtype=dtype, device=device)
    site_img = torch.zeros(n_sites, dtype=dtype, device=device)
    weights: Dict[str, torch.Tensor] = {}
    q_f = 0
    q_img = float(q_core)
    for ch in channels:
        if int(ch.count) == 0:
            continue
        if ch.density.shape[0] != n_sites:
            raise ValueError(f"channel {ch.name} has {ch.density.shape[0]} site densities for "
                             f"{n_sites} sites")
        term = float(ch.sign) * float(ch.count) * ch.density
        site = site + term
        site_img = site_img + ch.weight * term
        weights[ch.name] = ch.weight
        q_f += int(round(ch.sign)) * int(ch.count)
        q_img = q_img + float(ch.sign) * float(ch.count) * ch.weight
    frontier = GaussianDensity(site, positions, static.sigma, static.cell)
    frontier_img = GaussianDensity(site_img, positions, static.sigma, static.cell)
    q_img_t = q_img if isinstance(q_img, torch.Tensor) else torch.tensor(q_img, dtype=dtype,
                                                                           device=device)
    return ImageDensity(static=static, frontier=frontier, frontier_img=frontier_img,
                        q_core=int(q_core), q_f=int(q_f), q_img=q_img_t, weights=weights)


# ------------------------------------------------------------------ the isolated kernel


def long_range_envelope(r: torch.Tensor, r_split: float, p: int = _ENVELOPE_P
                        ) -> torch.Tensor:
    """`L_{r_split}`: one minus the polynomial cutoff envelope of `short_range_potential`.

    Zero at `r = 0` with zero first and second derivatives, one beyond `r_split`, `C^2`
    throughout -- so the long-range kernel is as smooth as the full one and the split
    `K_inf = K_inf^SR + K_inf^LR` is exact by construction."""
    x = r / float(r_split)
    envelope = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x ** p + p * (p + 2.0) * x ** (p + 1)
                - (p * (p + 1.0) / 2.0) * x ** (p + 2)) * (x < 1.0)
    return 1.0 - envelope


def converged_dl(sigma: float, dl_config: float = 2.0) -> float:
    """The reciprocal-space resolution at which the periodic half is converged.

    LES sums `k` up to `2 pi / dl`. The isolated half of `K_img` is exact, so any remainder
    of the periodic sum lands in the image term undiminished: at LES's default `dl = 2` with
    `sigma = 1` that is 0.5 % of a lone charge's image energy and 2.4 % of a 64-site
    lattice's (`test_image_functional.py`). With `k_max sigma = 2 pi` the Gaussian weight
    at the cutoff is `exp(-2 pi^2) = 3e-9` and the term is converged to 1e-6. `dl` is a
    numerical floor (plan v8 section 7.3), never loosened past the configured value."""
    return min(float(dl_config), float(sigma))


def coulomb_constant_of(ewald) -> float:
    """The Coulomb constant the evaluator itself uses: LES folds `2 pi C` into its
    `norm_factor` (90.4756, i.e. C = 14.399640, not the 14.399645 of the tables). The two
    halves of `K_img` must share one constant to the last digit, or their difference carries
    a spurious `4e-7` of the full isolated energy."""
    return float(ewald.ewald.norm_factor) / (2.0 * math.pi)


def isolated_pair_kernel(r2: torch.Tensor, sigma: float,
                         coulomb: float = COULOMB_CONSTANT) -> torch.Tensor:
    """`C erf(r / (sigma sqrt 2)) / r` from `r^2`, with the analytic limit at `r -> 0`.

    `sigma` is LES's kernel width -- the width of the PAIR kernel `exp(-sigma^2 k^2 / 2)`,
    the convention `LatentEwald` registers -- so this is the same function of `r` that the
    periodic evaluator sums in reciprocal space, and `K_PBC - K_inf` is a pure image term.
    Written in `r^2` so that the derivative with respect to the centres is finite at zero
    separation (the kernel is even in `r`; `sqrt` is not differentiable there)."""
    a = float(sigma) * math.sqrt(2.0)
    near = r2 < _NEAR ** 2
    # The far branch is evaluated at a dummy separation where `near` holds, so no
    # infinite intermediate gradient is ever multiplied by the mask's zero.
    r = torch.sqrt(torch.where(near, torch.ones_like(r2), r2))
    far = torch.erf(r / a) / r
    # erf(x) / r = (2 / (a sqrt pi)) (1 - x^2 / 3 + x^4 / 10 - x^6 / 42), x = r / a
    x2 = r2 / (a * a)
    series = (2.0 / (a * math.sqrt(math.pi))) * (1.0 - x2 / 3.0 + x2 * x2 / 10.0
                                                 - x2 * x2 * x2 / 42.0)
    return float(coulomb) * torch.where(near, series, far)


def isolated_bilinear(charges_a: torch.Tensor, centres_a: torch.Tensor,
                      charges_b: torch.Tensor, centres_b: torch.Tensor, sigma: float,
                      r_split: Optional[float] = None,
                      coulomb: float = COULOMB_CONSTANT) -> torch.Tensor:
    """`B_{K_inf}^{R^3}[rho_a, rho_b]` (or its long-range filter) as a direct pair sum.

    Both centre sets must already be LIFTED; nothing here knows about a cell. Every pair
    `(i in a, j in b)` is summed, coincident centres included -- for two different densities
    the same-point term is an ordinary cross term, not a self-energy."""
    if charges_a.numel() == 0 or charges_b.numel() == 0:
        return torch.zeros((), dtype=charges_a.dtype, device=charges_a.device)
    d = centres_a[:, None, :] - centres_b[None, :, :]
    r2 = (d * d).sum(dim=-1)
    kernel = isolated_pair_kernel(r2, sigma, coulomb)
    if r_split is not None:
        kernel = kernel * long_range_envelope(torch.sqrt(r2.clamp_min(1e-30)),
                                             float(r_split))
    return charges_a @ kernel @ charges_b


def isolated_self_energy(charges: torch.Tensor, centres: torch.Tensor, sigma: float,
                         coulomb: float = COULOMB_CONSTANT) -> torch.Tensor:
    """`E_inf[rho] = 1/2 B_{K_inf}[rho, rho]` over distinct pairs: no self term, matching
    LES's `remove_self_interaction=True` convention for the periodic half."""
    n = charges.numel()
    if n < 2:
        return torch.zeros((), dtype=charges.dtype, device=charges.device)
    d = centres[:, None, :] - centres[None, :, :]
    r2 = (d * d).sum(dim=-1)
    kernel = isolated_pair_kernel(r2, sigma, coulomb)
    off = ~torch.eye(n, dtype=torch.bool, device=charges.device)
    return 0.5 * (charges @ (kernel * off) @ charges)


def lifted_centres(density: GaussianDensity, record: LiftRecord) -> torch.Tensor:
    """`U_h rho`'s centres: the density's primitives unwrapped under the record's branch,
    in the density's own dtype. The integer image assignment carries no gradient; the
    smooth dependence on the centres and the cell is retained."""
    cell = density.cell
    scaled = density.centres @ torch.linalg.inv(cell)
    return lift_positions(scaled, cell, record).to(density.centres.dtype)


# ------------------------------------------------------------------ the functional


def phi_sf_lr(static: GaussianDensity, frontier: GaussianDensity, record: LiftRecord,
              sigma: float, r_split: float, eps_inf: float,
              coulomb: float = COULOMB_CONSTANT) -> torch.Tensor:
    """`Phi_SF^{inf,LR}[P] = B_{K_inf^LR}[U rho_S, U rho_F[P]]`, one graph.

    Boundary-common: the same value enters the periodic and the isolated energy. `sigma` is
    the registered kernel width (the model's frontier evaluator's), `r_split` the filter's,
    `eps_inf` the registered screening."""
    if eps_inf <= 0.0:
        raise ValueError(f"eps_inf must be positive, got {eps_inf}")
    return isolated_bilinear(static.charges, lifted_centres(static, record),
                             frontier.charges, lifted_centres(frontier, record),
                             float(sigma), r_split=float(r_split),
                             coulomb=coulomb) / float(eps_inf)


def phi_img_pbc(rho: GaussianDensity, record: LiftRecord, ewald, eps_inf: float,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`Phi_img^PBC[P] = 1/2 B_{K_img}[rho_img, rho_img] = (E_PBC[rho] - E_inf[U rho]) / eps_inf`
    for one graph; `Phi_img^inf` is identically zero and is not a function.

    The periodic half is the model's own evaluator on the torus representation (jellium
    background included: `q_img` need not vanish); the isolated half is the direct sum on the
    lifted primitives. Both exclude the Gaussian self term, so their difference is the image
    interaction and nothing else."""
    if eps_inf <= 0.0:
        raise ValueError(f"eps_inf must be positive, got {eps_inf}")
    if abs(float(ewald.sigma) - float(rho.sigma)) > 1e-12:
        raise ValueError(f"the periodic evaluator's width {float(ewald.sigma)} differs from "
                         f"the density's {rho.sigma}: the two halves of K_img would not share "
                         "a smearing convention")
    n = rho.charges.numel()
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=rho.charges.device)
    periodic = ewald.energy(rho.charges, rho.centres, rho.cell.reshape(1, 3, 3),
                            batch).reshape(())
    isolated = isolated_self_energy(rho.charges, lifted_centres(rho, record),
                                    float(rho.sigma), coulomb_constant_of(ewald))
    return (periodic - isolated) / float(eps_inf)


def boundary_functional(density: ImageDensity, record: LiftRecord, ewald, *,
                        r_split: float, eps_inf: float, boundary: str
                        ) -> Dict[str, torch.Tensor]:
    """`Phi_B[P] = Phi_SF^{inf,LR}[P] + Phi_img^B[P]` and its two parts (section 6.2, before
    Stage 6, with the common frontier-frontier entry an exact zero under `m_F <= 1`).

    `boundary` is `"periodic"` or `"isolated"`. `phi_img_pbc` is reported under BOTH
    boundaries -- it is the two-boundary diagnostic of Stage 4 and the reference
    completion of section 6.3 -- while `phi_img` (what enters `phi`) is zero when the
    boundary is isolated."""
    if boundary not in ("periodic", "isolated"):
        raise ValueError(f"boundary must be 'periodic' or 'isolated', got {boundary!r}")
    sigma = float(ewald.sigma)
    sf = phi_sf_lr(density.static, density.frontier, record, sigma, r_split, eps_inf,
                   coulomb=coulomb_constant_of(ewald))
    img_pbc = phi_img_pbc(density.img, record, ewald, eps_inf)
    img = img_pbc if boundary == "periodic" else torch.zeros_like(img_pbc)
    return {"phi": sf + img, "phi_sf": sf, "phi_img": img, "phi_img_pbc": img_pbc,
            "q_img": density.q_img}
