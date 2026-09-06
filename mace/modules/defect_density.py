"""Density objects of plan v8 section 2.1: the static and frontier parts of the defect charge.

Every density here is a GAUSSIAN CHARGE LIST `(q_i, R_i, sigma)` plus, where a term needs
one, a uniform background of stated integral. Integrals are sums of the charges by
construction, and the L2 norm and overlaps are analytic under the minimum image (valid
while `sigma << L`), so "`rho_static^raw -> 0` as the displacement from the reference is
scaled to zero" is a real continuity test rather than a grid quadrature.

    rho_Z^present   = sum_i Z_i g(r - R_i)               present atoms, frame positions
    rho_Z^pristine  = sum_j Z0[s_j] g(r - R_j^0)         the pristine reference tiled to the
                                                        frame's supercell and PLACED in it
    rho_static^raw  = rho_Z^present - rho_Z^pristine     a density difference; no assignment
    q_raw           = int rho_static^raw
                    = sum_{i in present} Z_i - sum_{j in pristine} Z0[s_j]
    a_i             = sqrt(dZ_i^2 + eps_Z^2) - eps_Z + lambda_d d_i          (addendum 4.1)
    omega_i         = (a_i + eps_w/N^2) / sum_j (a_j + eps_w/N^2),  g_res = sum_i omega_i g(..)
    rho_static^def  = rho_static^raw + (Q_core - q_raw) g_res          int = Q_core

BOTH TERMS OF q_raw ARE WRITTEN OUT BECAUSE THE SHORTHAND IS FALSE HERE (addendum 4.1).
v8 wrote `q_raw = int rho_static^raw = sum_i Z_i`, which holds only if the pristine integral
vanishes -- that is, only if the tiled pristine baselines `Z0[s_j]` sum to zero. Measured on
this host they do not: `rho_Z^pristine.integral()` is about +0.65 e on the V_Cl frames, so
`sum_i Z_i` and `q_raw` differ by that amount. The two-term definition is therefore not
pedantry, and any code or test that reaches for the present-atom sum alone is wrong by a
number of order one electron. `test_defect_density.py` pins the difference of integrals.

PLACEMENT. The pristine reference is tiled (`defect_composition.tiling_map`) and put into
the frame's actual cell through scaled coordinates, then shifted by a rigid translation.
That translation is found ONCE per composition class by minimising `||rho_static^raw||^2`
on the class reference frame -- an alignment of DENSITIES, label-free and with no site
assignment in it -- and cached in the class record as a fractional shift (`placement`);
every frame of the class reuses it, and `||rho_static^raw||` is reported per frame so a
frame recorded from a different origin is visible rather than silently a dipole array.

dZ_i, THE LOCAL NET CHARGE. The plan's omega rule reads the per-site charge difference. At
the density level, with no site correspondence, `dZ_i` is the raw density READ at each
probe point in charge units, `dZ_i = rho_static^raw(R_i) / g(0) = sum_j q_j exp(-|R_i -
R_j|^2 / 2 r_res^2)`: an isolated Gaussian charge reads as itself, a thermally displaced
matched pair as `q (1 - exp(-d^2 / 2 r_res^2))` -- 12 % at d = 0.5 A, r_res = 1 A -- and a
vacancy's charge is read at its neighbours at 2 % (2.8 A) or less, so the residual weight
sits on the vacancy with a thermal halo. Section 7.6 is its sensitivity test.

FRONTIER (smooth; where the frontier charge resides, never how much):

    s_k^e = sigmoid((eps_k - VBM_al - delta) / Delta_s),  s_k^h = sigmoid((CBM_al - delta - eps_k) / Delta_s)
    rho_e,sigma = sum_k f_k s_k^e |psi_k|^2 / int(same),  rho_h,sigma = sum_k (1 - f_k) s_k^h |psi_k|^2 / int(same)
    rho_F(P)    = - sum_sigma n_e,sigma rho_e,sigma + sum_sigma n_h,sigma rho_h,sigma        int = q_F exactly
    w(P)        = sigmoid((p* - p) / Delta_p),  p = participation fraction of |rho_F|
    rho_F       = w rho_F,loc + (1 - w) rho_F,ext                   (ext: uniform, same integral)

`|psi_k|^2` on the orbital basis is the site-resolved diagonal of the PROJECTED DENSITY
MATRIX `U diag(f s) U^T`, one matrix per (spin, channel) -- never an eigenvector at a time,
which is the head's own degeneracy argument. The electron and hole objects are stored
separately; nothing here reconstructs a channel from the sign of q_F.

Stage 0 builds and tests these objects; no forward consumes them until Stage 4 (V_static^B)
and Stage 5 (V_F). `rho_ind` (Stage 6) is absent, so `Delta rho_def = rho_static^def + rho_F`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["GaussianDensity", "DEFAULT_FUNCTIONAL", "functional_config", "static_present",
           "pristine_placed", "align_pristine", "static_raw", "local_net_charge",
           "residual_shape", "residual_weights", "residual_support", "static_def", "edge_projectors", "channel_densities",
           "participation_fraction", "frontier_density", "defect_density"]

ORB = 4   # orbitals per site, defect_counting.ORBITALS_PER_ATOM

# The functional's own lengths and switches (plan sections 2.7 and 3). None entries resolve
# at construction: r_split to the first-block cutoff, r_orb[Z] to the covalent-radius table,
# delta to 2 x smearing, delta_s to the smearing width.
DEFAULT_FUNCTIONAL: Dict[str, Any] = {
    "r_split": None,      # A, range separation of V_static (Stage 4); None: first-block cutoff
    "r_orb": None,        # {Z: A}, damping of the full-range potential (Stage 2); None: r_cov
    "r_res": 1.0,         # A, width of g_res and of every density Gaussian here
    "delta": None,        # eV, edge margin (also the constructor's); None: 2 x smearing
    "delta_s": None,      # eV, edge projector width; None: the smearing width
    "p_star": 0.25,       # participation fraction at which w(P) switches
    "delta_p": 0.05,      # width of the switch
    "a_bounds": [-0.5, 0.5],   # bounds of a_i(h_i), Stage 2
    "b_bounds": [-0.5, 0.5],   # bounds of b_i(h_i), Stage 2
    "eps_p": 1.0e-8,      # SCF tolerance on ||P_{k+1} - P_k||, Stage 5
    "eps_h": 1.0e-6,      # SCF tolerance on ||[H, P]||, Stage 5
    "c_q_mode": "per_charge",  # C_Q: one constant per charge state, shared across sizes
}


# Addendum 4.1. eps_Z smooths |dZ| at zero; eps_omega is the size-vanishing uniform floor
# that replaces the old zero-denominator branch; SUPPORT_THRESHOLD is the departure signal
# below which a non-zero residual monopole is refused rather than delocalised.
EPS_Z = 1e-6
EPS_OMEGA = 1e-8
SUPPORT_THRESHOLD = 1e-3


def functional_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The functional config with `overrides` applied; unknown keys are refused."""
    config = dict(DEFAULT_FUNCTIONAL)
    for key, value in (overrides or {}).items():
        if key not in config:
            raise ValueError(f"unknown functional parameter {key!r}; expected one of "
                             f"{sorted(config)}")
        config[key] = value
    if config["r_orb"] is not None:
        config["r_orb"] = {int(z): float(r) for z, r in dict(config["r_orb"]).items()}
    for key in ("a_bounds", "b_bounds"):
        config[key] = [float(x) for x in config[key]]
    return config


# ------------------------------------------------------------------ the density object


def _minimum_image(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    frac = delta @ torch.linalg.inv(cell)
    frac = frac - torch.round(frac)
    return frac @ cell


@dataclass(frozen=True)
class GaussianDensity:
    """`sum_i q_i g(r - R_i; sigma) + background / V` on the periodic cell.

    `charges` `[n]`, `centres` `[n, 3]` (torch, differentiable), `sigma` a float,
    `cell` `[3, 3]`, `background` a float (the uniform part's integral).
    """
    charges: torch.Tensor
    centres: torch.Tensor
    sigma: float
    cell: torch.Tensor
    background: float = 0.0

    def integral(self) -> torch.Tensor:
        return self.charges.sum() + self.background

    @property
    def volume(self) -> torch.Tensor:
        return torch.abs(torch.linalg.det(self.cell))

    def __add__(self, other: "GaussianDensity") -> "GaussianDensity":
        if abs(self.sigma - other.sigma) > 1e-12:
            raise ValueError("densities of different widths cannot be added as one list")
        return GaussianDensity(torch.cat([self.charges, other.charges]),
                               torch.cat([self.centres, other.centres]), self.sigma,
                               self.cell, self.background + other.background)

    def scaled(self, factor) -> "GaussianDensity":
        return replace(self, charges=self.charges * factor, background=self.background * factor)

    def __neg__(self) -> "GaussianDensity":
        return self.scaled(-1.0)

    def __sub__(self, other: "GaussianDensity") -> "GaussianDensity":
        return self + (-other)

    def kernel(self, other: "GaussianDensity") -> torch.Tensor:
        """`K_ij = int g(r - R_i; s1) g(r - R_j; s2) dr` under the minimum image, `[n, m]`."""
        s2 = self.sigma ** 2 + other.sigma ** 2
        d = _minimum_image(self.centres[:, None, :] - other.centres[None, :, :], self.cell)
        return (2.0 * math.pi * s2) ** (-1.5) * torch.exp(-(d ** 2).sum(-1) / (2.0 * s2))

    def overlap(self, other: "GaussianDensity") -> torch.Tensor:
        """`int rho rho' dr`, including the uniform parts."""
        v = self.volume
        cross = self.charges @ self.kernel(other) @ other.charges
        return (cross + (self.charges.sum() * other.background
                         + other.charges.sum() * self.background
                         + self.background * other.background) / v)

    def norm2(self) -> torch.Tensor:
        return self.overlap(self)

    def norm(self) -> torch.Tensor:
        return torch.sqrt(self.norm2().clamp_min(0.0))

    def site_charges(self, positions: torch.Tensor) -> torch.Tensor:
        """The density read at `positions`, in charge units: `rho(R) / g(0)`, i.e.
        `sum_j q_j exp(-|R - R_j|^2 / 2 sigma^2)` under the minimum image (the uniform part
        excluded)."""
        d = _minimum_image(positions[:, None, :] - self.centres[None, :, :], self.cell)
        return torch.exp(-(d ** 2).sum(-1) / (2.0 * self.sigma ** 2)) @ self.charges

    def to_dict(self) -> Dict[str, Any]:
        return {"charges": self.charges.detach().cpu().tolist(),
                "centres": self.centres.detach().cpu().tolist(), "sigma": self.sigma,
                "cell": self.cell.detach().cpu().tolist(), "background": self.background}


# ------------------------------------------------------------------ the static part


def static_present(charges: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
                   r_res: float) -> GaussianDensity:
    """`rho_Z^present`: the static charges `Z_i` on the present atoms at the frame positions.
    `int = sum_i Z_i` by construction."""
    return GaussianDensity(charges, positions, float(r_res), cell)


def pristine_placed(z0: torch.Tensor, scaled_positions: torch.Tensor, cell: torch.Tensor,
                    shift: torch.Tensor, r_res: float) -> GaussianDensity:
    """`rho_Z^pristine`: the tiled pristine reference's species charges at its SCALED
    positions (already in the frame's axis order) placed in the frame's cell and shifted by
    the fractional `shift` `[3]`."""
    frac = scaled_positions + shift.reshape(1, 3)
    return GaussianDensity(z0, frac @ cell, float(r_res), cell)


def static_raw(present: GaussianDensity, pristine: GaussianDensity) -> GaussianDensity:
    """`rho_static^raw = rho_Z^present - rho_Z^pristine`."""
    return present - pristine


def align_pristine(present: GaussianDensity, z0: torch.Tensor, scaled_positions: torch.Tensor,
                   anchor_species_mask: torch.Tensor, present_anchor: int,
                   n_candidates: int = 8, newton_steps: int = 12
                   ) -> Tuple[torch.Tensor, float]:
    """The fractional shift `t*` minimising `||rho_Z^present - rho_Z^pristine(t)||^2`.

    Coarse candidates put the frame's `present_anchor` atom on every pristine site of the
    anchor species (`anchor_species_mask` over the pristine sites), ranked by the residual
    norm; the best few are refined by Newton steps on the three shift components (the
    objective is analytic in t). Returns `(t*, residual_norm)`. No site is ever assigned to
    another; only the total density difference is minimised.
    """
    cell = present.cell
    inv = torch.linalg.inv(cell)
    anchor_frac = present.centres[present_anchor] @ inv
    sites = torch.nonzero(anchor_species_mask).reshape(-1)

    def residual(t):
        pri = pristine_placed(z0, scaled_positions, cell, t, present.sigma)
        return static_raw(present, pri).norm2()

    scored = []
    for j in sites.tolist():
        t = (anchor_frac - scaled_positions[j]).detach()
        scored.append((float(residual(t)), t))
    scored.sort(key=lambda s: s[0])
    best_t, best_r = None, float("inf")
    for r0, t0 in scored[:n_candidates]:
        t = t0.clone()
        r = r0
        for _ in range(newton_steps):
            t_var = t.clone().requires_grad_(True)
            value = residual(t_var)
            grad = torch.autograd.grad(value, t_var, create_graph=True)[0]
            hess = torch.stack([torch.autograd.grad(grad[i], t_var, retain_graph=True)[0]
                                for i in range(3)])
            try:
                step = torch.linalg.solve(hess + 1e-9 * torch.eye(3, dtype=hess.dtype),
                                          grad)
            except RuntimeError:
                break
            t_new = (t_var - step).detach()
            r_new = float(residual(t_new))
            if r_new >= r - 1e-14:
                break
            t, r = t_new, r_new
        if r < best_r:
            best_t, best_r = t, r
    return best_t, math.sqrt(max(best_r, 0.0))


def local_net_charge(raw: GaussianDensity, centres: torch.Tensor) -> torch.Tensor:
    """`dZ_i`: the raw density smoothed with its own Gaussian, read at `centres`."""
    return raw.site_charges(centres)


def residual_shape(raw: GaussianDensity, present_positions: torch.Tensor,
                   pristine_positions: torch.Tensor, merge: float = 0.5,
                   eps_z: float = EPS_Z, eps_omega: float = EPS_OMEGA,
                   departure: Optional[torch.Tensor] = None,
                   lambda_d: float = 0.0) -> GaussianDensity:
    """`g_res = sum_i omega_i g(r - R_i; r_res)`, `omega_i = |dZ_i| / sum_j |dZ_j|`, over the
    present atoms and the pristine sites (a vacancy's weight lives at its pristine site).
    Integral 1 by construction; zero weights everywhere fall back to a uniform background.

    A pristine site within `merge x r_res` of a present atom is not a separate probe point:
    the local net charge there is read at the atom already, and a duplicate would count the
    thermal halo twice. A proximity merge of probe points, not an assignment -- nothing is
    matched to anything, and a site with no atom near it (a vacancy) keeps its probe.
    """
    d = _minimum_image(pristine_positions[:, None, :] - present_positions[None, :, :], raw.cell)
    near = (d.norm(dim=-1) < merge * raw.sigma).any(dim=1)
    centres = torch.cat([present_positions, pristine_positions[~near]])
    weights, _ = residual_weights(raw, centres, present_positions.shape[0],
                                  eps_z=eps_z, eps_omega=eps_omega,
                                  departure=departure, lambda_d=lambda_d)
    return GaussianDensity(weights, centres, raw.sigma, raw.cell)


def residual_weights(raw: GaussianDensity, centres: torch.Tensor, n_at: int,
                     eps_z: float = EPS_Z, eps_omega: float = EPS_OMEGA,
                     departure: Optional[torch.Tensor] = None, lambda_d: float = 0.0
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(omega, a)` -- the smooth residual weights and the departure signal they came from.

    Addendum section 4.1 replaces `omega_i = |dZ_i| / sum_j |dZ_j|` and its zero-denominator
    branch with

        a_i     = sqrt(dZ_i^2 + eps_Z^2) - eps_Z + lambda_d d_i
        omega_i = (a_i + eps_omega / N_at^2) / sum_j (a_j + eps_omega / N_at^2)

    Two things this fixes. `|dZ|` has a kink at zero, so `g_res` and every derivative through
    it were non-differentiable exactly where the learned deviations vanish -- which is the
    pristine limit, the case that has to be smooth. And the old `if total <= 1e-12` fallback
    was a discontinuous branch onto a *uniform background*: a cell that crossed that
    threshold jumped from a localised residual to a fully delocalised one.

    The `eps_omega / N_at^2` floor replaces the branch. It is always present, so there is no
    branch to cross; `sum_i omega_i = 1` holds by construction for any input; and because
    there are O(N_at) centres each carrying O(1/N_at^2), the total fallback weight is
    O(1/N_at) and vanishes as the cell grows rather than leaving a finite delocalised
    fraction. With every deviation zero it degrades to the uniform 1/N, which is the honest
    answer when nothing distinguishes any site.

    `departure` is the plan's optional `d_i >= 0`, a smooth bounded departure from the
    pristine species environment; `lambda_d = 0` by default leaves it out entirely.
    """
    if eps_z <= 0.0 or eps_omega <= 0.0:
        raise ValueError(f"eps_z and eps_omega must be positive, got {eps_z}, {eps_omega}")
    dz = local_net_charge(raw, centres)
    a = torch.sqrt(dz * dz + eps_z * eps_z) - eps_z
    if departure is not None and lambda_d:
        if departure.shape[0] != a.shape[0]:
            raise ValueError(
                f"departure signal has {departure.shape[0]} entries for {a.shape[0]} centres")
        a = a + float(lambda_d) * departure.clamp_min(0.0)
    floor = eps_omega / float(max(int(n_at), 1)) ** 2
    numerator = a + floor
    return numerator / numerator.sum(), a


def residual_support(raw: GaussianDensity, centres: torch.Tensor, n_at: int,
                     q_core: int, q_raw, threshold: float = SUPPORT_THRESHOLD,
                     eps_z: float = EPS_Z, eps_omega: float = EPS_OMEGA,
                     departure: Optional[torch.Tensor] = None, lambda_d: float = 0.0
                     ) -> Tuple[bool, float]:
    """`(supported, signal)` for the residual monopole (addendum section 4.1).

    `rho_static^def` places `Q_core - q_raw` on `g_res`. If that is non-zero while the local
    departure signal is below the registered threshold, `g_res` is essentially the uniform
    fallback -- so the model would be putting a physical monopole through a numerical
    tie-breaker, spread over the whole cell. That is reported as unsupported rather than
    evaluated: a delocalised monopole is not a small error, it is a different physical claim.
    """
    _, a = residual_weights(raw, centres, n_at, eps_z=eps_z, eps_omega=eps_omega,
                            departure=departure, lambda_d=lambda_d)
    signal = float(a.sum())
    residual_charge = abs(float(q_core) - float(q_raw))
    if residual_charge <= threshold:
        return True, signal          # nothing to place; the shape is irrelevant
    return signal > threshold, signal


def static_def(raw: GaussianDensity, g_res: GaussianDensity, q_core: int) -> GaussianDensity:
    """`rho_static^def = rho_static^raw + (Q_core - q_raw) g_res`; `int = Q_core`."""
    q_raw = raw.integral()
    return raw + g_res.scaled(float(q_core) - q_raw)


# ------------------------------------------------------------------ the frontier part


def edge_projectors(eps: torch.Tensor, vbm_al: float, cbm_al: float, delta: float,
                    delta_s: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """`s_k^e` and `s_k^h`: smooth membership of level k above the valence edge and below the
    conduction edge."""
    s_e = torch.sigmoid((eps - vbm_al - delta) / delta_s)
    s_h = torch.sigmoid((cbm_al - delta - eps) / delta_s)
    return s_e, s_h


def _site_weights(U: torch.Tensor, w: torch.Tensor, n_sites: int) -> torch.Tensor:
    """`diag(U diag(w) U^T)` summed per site: the site-resolved density of the projected
    density matrix, one matrix for the whole channel."""
    dens = torch.diagonal(U @ torch.diag(w) @ U.T)
    return dens.reshape(n_sites, ORB).sum(dim=-1)


def channel_densities(H: torch.Tensor, occupations: Sequence[torch.Tensor],
                      positions: torch.Tensor, cell: torch.Tensor, vbm_al: float,
                      cbm_al: float, delta: float, delta_s: float, r_res: float,
                      log_below: float = 0.5) -> Dict[str, Any]:
    """`rho_e,sigma` and `rho_h,sigma` for the spins whose occupation vectors are given, each
    a positive density normalised to 1 (or an EMPTY object when its total projector weight
    is zero -- it is unused when its count is zero). Returns them under
    `"electron"` / `"hole"` as lists over spin, plus the raw projector weights and the
    eigenvalues for the log."""
    eps, U = torch.linalg.eigh(H)
    n_sites = positions.shape[0]
    s_e, s_h = edge_projectors(eps, vbm_al, cbm_al, delta, delta_s)
    out: Dict[str, Any] = {"electron": [], "hole": [], "weight_e": [], "weight_h": [],
                           "eps": eps, "s_e": s_e, "s_h": s_h}
    for f in occupations:
        for key, w in (("electron", f * s_e), ("hole", (1.0 - f) * s_h)):
            site = _site_weights(U, w, n_sites)
            total = site.sum()
            out["weight_e" if key == "electron" else "weight_h"].append(total)
            if float(total) > 1e-12:
                out[key].append(GaussianDensity(site / total, positions, float(r_res), cell))
            else:
                out[key].append(None)
    # A frontier state whose projector weight is below `log_below` is logged, not acted on:
    # the class counts are integers and this is a per-frame diagnostic.
    for key in ("weight_e", "weight_h"):
        for s, total in enumerate(out[key]):
            if 0.0 < float(total) < log_below:
                logging.info("frontier %s channel, spin %d: projector weight %.3f < %.1f "
                             "(a level merging into a band)", key[-1], s, float(total),
                             log_below)
    return out


def participation_fraction(site_charges: torch.Tensor) -> torch.Tensor:
    """`p = (sum_i |rho_i|)^2 / (N sum_i rho_i^2)`: 1 for a uniform density, 1/N for one
    site."""
    a = site_charges.abs()
    n = a.numel()
    return a.sum() ** 2 / (n * (a ** 2).sum().clamp_min(1e-300))


def frontier_density(n_e: Sequence[int], n_h: Sequence[int], channels: Dict[str, Any],
                     positions: torch.Tensor, cell: torch.Tensor, r_res: float,
                     p_star: float, delta_p: float) -> Dict[str, Any]:
    """`rho_F = -sum n_e rho_e + sum n_h rho_h`, then the participation switch to the
    uniform background: `rho_F = w rho_F,loc + (1 - w) rho_F,ext`. `int rho_F = q_F` exactly
    at every w. Refuses a channel that is needed (count > 0) but empty."""
    n_sites = positions.shape[0]
    site = torch.zeros(n_sites, dtype=positions.dtype, device=positions.device)
    for s in range(len(n_e)):
        for count, key, sign in ((n_e[s], "electron", -1.0), (n_h[s], "hole", 1.0)):
            if int(count) == 0:
                continue
            rho = channels[key][s]
            if rho is None:
                raise ValueError(f"the {key} channel of spin {s} carries no projector weight "
                                 f"but its count is {int(count)}: no frontier density exists")
            site = site + sign * float(count) * rho.charges
    q_f = float(sum(n_h) - sum(n_e))
    loc = GaussianDensity(site, positions, float(r_res), cell)
    p = participation_fraction(site) if float(site.abs().sum()) > 0 else torch.zeros(())
    w = torch.sigmoid((p_star - p) / delta_p)
    ext = GaussianDensity(torch.zeros(0, dtype=site.dtype, device=site.device),
                          torch.zeros(0, 3, dtype=positions.dtype, device=positions.device),
                          float(r_res), cell, background=q_f)
    rho_f = loc.scaled(w) + ext.scaled(1.0 - w)
    return {"rho_F": rho_f, "rho_F_loc": loc, "rho_F_ext": ext, "w": w, "p": p, "q_F": q_f}


def defect_density(static: GaussianDensity, frontier: GaussianDensity) -> GaussianDensity:
    """`Delta rho_def = rho_static^def + rho_F` (+ rho_ind from Stage 6); `int = Q_formal`."""
    return static + frontier
