"""Plan section 2.4 / 2.5: the same-carrier kernel in its two registered regimes, the host
coupling matrix `Gamma_LR`, the centred pattern, and the kernel diagnostics.

Common structure (both regimes):

    Gamma_ij (i != j) = [K_LR_ij + lambda_dir K_SR_ij] / eps_inf
    Gamma_ii          =  K_LR_ii / eps_inf + U_eff[Z_i]
    E_PBC             =  K_SR + K_LR         (the Ewald matrix of unit Gaussians, width r_g)

`K_LR` is kept at full weight and carries the 1/L physics; the components are named
`K_SR` / `K_LR`, never "image" (plan section 10). With `U_eff -> 0` the diagonal carries no
short-range self term: `U_eff` replaces both the scaled Gaussian self-energy and the
hardness. All matrices are in eV per unit charge pair (`COULOMB` included).

Regime A (shell switch): `K_SR_ij = C erf(r_ij / (2 r_g)) / r_ij * w_dir(r_ij)` on the
minimum-image distance, `w_dir` a C2 switch 1 -> 0 on `[r_d1, r_d2]` in the first-second
shell gap, `K_SR_ii = C / (sqrt(pi) r_g)`, `K_LR = E_PBC - K_SR`.

Regime B (periodic range separation): `s(r) = C [erf(r/(2 r_g)) - erf(r/r_s)] / r`,
`K_SR_ij = sum_L s(|r_ij + L|)`, `K_SR_ii = s(0) + sum_{L != 0} s(|L|)`, and `K_LR` is the
Ewald matrix of the broad kernel `erf(r/r_s)/r` (two Gaussians of width `r_s / 2`) --
exactly `E_PBC - K_SR`, which the tests check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from scipy.special import erfcinv

from mace.modules.dscc.ewald import (COULOMB, EWALD_TOL, ewald_matrix, pair_gradient, self_term,
                                     short_range_lattice_sum, reciprocal_matrix, reciprocal_pair_gradient)

_SQRT_PI = math.sqrt(math.pi)
REGIMES = ("A", "B")


@dataclass(frozen=True)
class KernelConfig:
    """Registered kernel numbers (plan section 11). Defaults are the plan's; every value is
    serialised with the model."""
    regime: str = "B"           # v4.1 amendment: regime B primary; A a reduced ablation only
    r_g: float = 1.0            # Gaussian width of the site charges (A); to register
    eps_inf: float = 4.0        # host input (plan section 1)
    r_d1: float = 3.2           # regime A switch, A
    r_d2: float = 3.6
    r_s: float = 6.5            # regime B range-separation width, A
    tol: float = EWALD_TOL
    lr_route: str = "reciprocal"  # v5 W2 item 3: 'reciprocal' (direct k-space K_LR / Gamma_LR) or 'ewald' (the v4 route)

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unknown kernel regime {self.regime!r}; expected one of {REGIMES}")
        if self.regime == "A" and not 0.0 < self.r_d1 < self.r_d2:
            raise ValueError("regime A needs 0 < r_d1 < r_d2")
        if self.regime == "B" and self.r_s <= 2.0 * self.r_g:
            raise ValueError("regime B needs r_s > 2 r_g")
        if self.lr_route not in ("reciprocal", "ewald"):
            raise ValueError(f"unknown lr_route {self.lr_route!r}")


def switch_c2(r: torch.Tensor, r1: float, r2: float) -> torch.Tensor:
    """C2 switch, 1 below `r1`, 0 above `r2`: the quintic smoothstep in between."""
    t = ((r - r1) / (r2 - r1)).clamp(0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def minimum_image_distances(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """`r_ij` under the minimum-image convention, `[N, N]`, differentiable (the integer
    shift is a constant of the local geometry)."""
    d = positions.unsqueeze(1) - positions.unsqueeze(0)
    shift = torch.round(d.detach() @ torch.linalg.inv(cell.detach()))
    return (d - shift @ cell).norm(dim=-1)


def k_sr_regime_a(positions: torch.Tensor, cell: torch.Tensor, cfg: KernelConfig,
                  pair_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
    n = positions.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=positions.device)
    if pair_vectors is None:
        r = minimum_image_distances(positions, cell)
    else:
        shift = torch.round(pair_vectors.detach() @ torch.linalg.inv(cell.detach()))
        r = (pair_vectors - shift @ cell).norm(dim=-1)
    r_safe = torch.where(eye, torch.ones_like(r), r)
    off = COULOMB * torch.erf(r_safe / (2.0 * cfg.r_g)) / r_safe * switch_c2(r_safe, cfg.r_d1, cfg.r_d2)
    diag = self_term(cfg.r_g).to(dtype=positions.dtype, device=positions.device).expand(n)
    return torch.where(eye, torch.diag_embed(diag), off)


def regime_b_cutoff(cfg: KernelConfig) -> float:
    """Image range of the regime-B lattice sum: `s(r) ~ C erfc(r / r_s) / r <= tol`."""
    r_c = cfg.r_s * float(erfcinv(cfg.tol / COULOMB))
    for _ in range(4):
        r_c = cfg.r_s * float(erfcinv(cfg.tol * max(r_c, 1.0) / COULOMB))
    return r_c


def k_sr_regime_b(positions: torch.Tensor, cell: torch.Tensor, cfg: KernelConfig,
                  r_c: Optional[float] = None, pair_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
    two_rg, r_s = 2.0 * cfg.r_g, cfg.r_s

    def s_of(r: torch.Tensor) -> torch.Tensor:
        return COULOMB * (torch.erf(r / two_rg) - torch.erf(r / r_s)) / r

    s_zero = COULOMB * (1.0 / (_SQRT_PI * cfg.r_g) - 2.0 / (_SQRT_PI * r_s))
    return short_range_lattice_sum(positions, cell, s_of, s_zero,
                                   regime_b_cutoff(cfg) if r_c is None else r_c, pair_vectors=pair_vectors)


def lr_route(cfg: KernelConfig) -> str:
    """The long-range route of a config, 'reciprocal' for configs pickled before v5."""
    return getattr(cfg, "lr_route", "reciprocal")


def kernel_components(positions: torch.Tensor, cell: torch.Tensor, cfg: KernelConfig,
                      eta: Optional[float] = None,
                      pair_vectors: Optional[torch.Tensor] = None,
                      need_sr: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(K_SR, K_LR)` with `K_SR + K_LR = E_PBC`. With `pair_vectors` both are functions of
    the pair leaf `d0_ij = r_i - r_j` entry by entry (see `ewald.pair_gradient`). With
    `need_sr=False` (v5 W2: `lambda_dir` held at zero, so `K_SR` never enters `Gamma`) the
    short-range component is returned as zeros and its lattice sum is skipped."""
    if cfg.regime == "A":
        k_sr = k_sr_regime_a(positions, cell, cfg, pair_vectors=pair_vectors)
        k_lr = ewald_matrix(positions, cell, cfg.r_g, eta=eta, tol=cfg.tol, pair_vectors=pair_vectors) - k_sr
        return k_sr, k_lr
    # Regime B: K_LR is the periodic kernel of the broad Gaussian erf(r / r_s) / r (two
    # Gaussians of width r_s / 2, pair width r_s), evaluated in reciprocal space alone
    # (v5 W2 item 3; the v4 Ewald route kept under lr_route = 'ewald', identical to tol).
    if lr_route(cfg) == "ewald":
        k_lr = ewald_matrix(positions, cell, 0.5 * cfg.r_s, eta=eta, tol=cfg.tol, pair_vectors=pair_vectors)
    else:
        k_lr = reciprocal_matrix(positions, cell, cfg.r_s, tol=cfg.tol, pair_vectors=pair_vectors)
    if need_sr:
        k_sr = k_sr_regime_b(positions, cell, cfg, pair_vectors=pair_vectors)
    else:
        k_sr = torch.zeros_like(k_lr)
    return k_sr, k_lr


def kernel_pair_gradients(positions: torch.Tensor, cell: torch.Tensor, cfg: KernelConfig,
                          eta: Optional[float] = None, need_sr: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(D_SR, D_LR)`, `[N, N, 3]` each: `d K_ij / d(r_i - r_j)` of the two kernel
    components, detached constants of the geometry. Regime B on the reciprocal route: `D_LR`
    analytic through the structure factors, `D_SR` one first-order backward through the
    short-range lattice sum (skipped, zeros, with `need_sr=False`); otherwise two backwards
    through the lattice sums, no graph kept."""
    pos, cel = positions.detach(), cell.detach()
    if cfg.regime == "B" and lr_route(cfg) != "ewald":
        d_lr = reciprocal_pair_gradient(pos, cel, cfg.r_s, tol=cfg.tol)
        if need_sr:
            (d_sr,) = pair_gradient(lambda d0: (k_sr_regime_b(pos, cel, cfg, pair_vectors=d0),), pos)
        else:
            d_sr = torch.zeros_like(d_lr)
        return d_sr, d_lr
    if need_sr:
        return pair_gradient(lambda d0: kernel_components(pos, cel, cfg, eta=eta, pair_vectors=d0), pos)
    (d_lr,) = pair_gradient(lambda d0: (kernel_components(pos, cel, cfg, eta=eta, pair_vectors=d0, need_sr=False)[1],), pos)
    return torch.zeros_like(d_lr), d_lr


def gamma_pair_derivative(d_sr: torch.Tensor, d_lr: torch.Tensor, lambda_dir: torch.Tensor,
                          eps_inf: float) -> torch.Tensor:
    """`d Gamma_ij / d(r_i - r_j)` from the components' pair derivatives: the off-diagonal
    `(D_LR + lambda_dir D_SR) / eps_inf` (attached to `lambda_dir`), zero on the diagonal
    (`K_LR_ii` and `U_eff` do not move with the geometry). `[..., N, N, 3]`."""
    n = d_sr.shape[-2]
    eye = torch.eye(n, dtype=torch.bool, device=d_sr.device).unsqueeze(-1)
    off = (d_lr + lambda_dir * d_sr) / eps_inf
    return torch.where(eye, torch.zeros_like(off), off)


def gamma_matrix(k_sr: torch.Tensor, k_lr: torch.Tensor, lambda_dir: torch.Tensor,
                 u_eff_site: torch.Tensor, eps_inf: float) -> torch.Tensor:
    """`Gamma` from the components: off-diagonal `(K_LR + lambda_dir K_SR) / eps_inf`,
    diagonal `K_LR_ii / eps_inf + U_eff[Z_i]` (`u_eff_site` is `U_eff` gathered per site)."""
    n = k_sr.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=k_sr.device)
    off = (k_lr + lambda_dir * k_sr) / eps_inf
    diag = torch.diagonal(k_lr, dim1=-2, dim2=-1) / eps_inf + u_eff_site
    return torch.where(eye, torch.diag_embed(diag), off)


def phi_cc(gamma: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
    """`0.5 dq^T Gamma dq`."""
    return 0.5 * dq @ gamma @ dq


def gamma_lr(positions: torch.Tensor, cell: torch.Tensor, r_g: float, r_split: float,
             eps_inf: float, eta: Optional[float] = None, tol: float = EWALD_TOL,
             route: str = "reciprocal") -> torch.Tensor:
    """Plan section 2.5: the Ewald matrix between a Gaussian of width `r_g` and one of
    width `r_split` (`r_split > r_g`, so `H0` owns the sharp near field), over `eps_inf`."""
    if r_split <= r_g:
        raise ValueError(f"r_split ({r_split}) must exceed r_g ({r_g})")
    if route == "ewald":
        return ewald_matrix(positions, cell, r_g, r_split, eta=eta, tol=tol) / eps_inf
    # v5 W2 item 3: the combined pair width sqrt(2 (r_g^2 + r_split^2)), reciprocal space alone.
    return reciprocal_matrix(positions, cell, math.sqrt(2.0 * (r_g ** 2 + r_split ** 2)), tol=tol) / eps_inf


def gamma_lr_pair_gradient(positions: torch.Tensor, cell: torch.Tensor, r_g: float, r_split: float,
                           eps_inf: float, eta: Optional[float] = None, tol: float = EWALD_TOL,
                           route: str = "reciprocal") -> torch.Tensor:
    """Route B' under the pair force route: `d Gamma_LR_ij / d(r_i - r_j)`, the pair
    derivative of the `r_g`/`r_split` Ewald matrix over `eps_inf` -- a detached constant of
    the geometry (`[N, N, 3]`, one first-order backward, no graph kept)."""
    pos, cel = positions.detach(), cell.detach()
    if route == "ewald":
        (d,) = pair_gradient(lambda d0: (ewald_matrix(pos, cel, r_g, r_split, eta=eta, tol=tol, pair_vectors=d0),), pos)
        return d / eps_inf
    return reciprocal_pair_gradient(pos, cel, math.sqrt(2.0 * (r_g ** 2 + r_split ** 2)), tol=tol) / eps_inf


def centred_pattern(zstar_site: torch.Tensor) -> torch.Tensor:
    """`Zbar_i = Zstar[Z_i] - mean_j Zstar[Z_j]` per cell: mandatory centring (plan
    section 2.5 -- an uncentred pattern is a bug)."""
    return zstar_site - zstar_site.mean()


def project_sum_rule(zstar: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
    """`sum_s c_s Zstar_s = 0` over the pristine composition `c` (per species), by
    orthogonal projection: the two free scalars of the plan's parameterisation."""
    c = composition.to(zstar.dtype)
    return zstar - (zstar @ c) / (c @ c) * c


def host_potential(gamma_lr_matrix: torch.Tensor, zbar: torch.Tensor) -> torch.Tensor:
    """`W = Gamma_LR @ Zbar`, geometry-only, once per frame."""
    return gamma_lr_matrix @ zbar


# ---------------------------------------------------------------- diagnostics


def m_sw(dq: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, cfg: KernelConfig
         ) -> torch.Tensor:
    """Regime A switch diagnostic: `sum_{i<j, r_d1 < r_ij < r_d2} |dq_i dq_j|`."""
    r = minimum_image_distances(positions.detach(), cell.detach())
    inside = (r > cfg.r_d1) & (r < cfg.r_d2)
    inside = torch.triu(inside, diagonal=1)
    pair = (dq.unsqueeze(-1) * dq.unsqueeze(-2)).abs()
    return pair[inside].sum()


def f_sr(dq: torch.Tensor, k_sr: torch.Tensor, k_lr: torch.Tensor) -> torch.Tensor:
    """Regime B analogue: the short-range fraction of the intra-carrier interaction,
    `sum_{i<j} |dq_i dq_j| K_SR_ij / sum_{i<j} |dq_i dq_j| (K_SR + K_LR)_ij`."""
    pair = (dq.unsqueeze(-1) * dq.unsqueeze(-2)).abs()
    upper = torch.triu(torch.ones_like(pair, dtype=torch.bool), diagonal=1)
    num = (pair * k_sr)[upper].sum()
    den = (pair * (k_sr + k_lr))[upper].sum()
    return num / den


def f_sr_abs(dq: torch.Tensor, k_sr: torch.Tensor, k_lr: torch.Tensor) -> torch.Tensor:
    """C10 ruling (2026-09-08): the ABSOLUTE short-range share
    `sum_{i<j} |dq_i dq_j K_SR_ij| / sum_{i<j} |dq_i dq_j| (|K_SR_ij| + |K_LR_ij|)`, in [0, 1].
    The signed `f_sr` is unbounded in a small cell (`K_LR` is self-image dominated and
    negative on every pair, so its denominator can shrink below the numerator): recorded as
    a criterion defect; this share is reported instead and is NOT a gate."""
    pair = (dq.unsqueeze(-1) * dq.unsqueeze(-2)).abs()
    upper = torch.triu(torch.ones_like(pair, dtype=torch.bool), diagonal=1)
    num = (pair * k_sr.abs())[upper].sum()
    den = (pair * (k_sr.abs() + k_lr.abs()))[upper].sum()
    return num / den


def phi_cc_rs_sensitivity(positions: torch.Tensor, cell: torch.Tensor, cfg: KernelConfig,
                          dq: torch.Tensor, lambda_dir: torch.Tensor, u_eff_site: torch.Tensor,
                          delta: float = 0.5) -> Dict[str, float]:
    """Regime B diagnostic (v4.1 section 11): `Phi_cc` at `r_s`, `r_s - delta`, `r_s + delta`
    with the same `dq` -- how much of the intra-carrier energy the range split decides."""
    out = {}
    for name, r_s in (("minus", cfg.r_s - delta), ("centre", cfg.r_s), ("plus", cfg.r_s + delta)):
        c = KernelConfig(**{**cfg.__dict__, "r_s": r_s})
        k_sr, k_lr = kernel_components(positions.detach(), cell.detach(), c)
        g = gamma_matrix(k_sr, k_lr, lambda_dir.detach(), u_eff_site.detach(), cfg.eps_inf)
        out[name] = float(phi_cc(g, dq.detach()))
    out["delta"] = float(delta)
    out["sensitivity"] = (out["plus"] - out["minus"]) / (2.0 * delta)      # eV per A
    return out


def flanking_pb_fraction(positions: torch.Tensor, cell: torch.Tensor, numbers: Sequence[int],
                         cfg: KernelConfig, cation: int = 82, anion: int = 17,
                         shell_gap: float = 4.0) -> Dict[str, float]:
    """v4.1 section 2.4 (regime-A ablation report): the placement fraction restricted to the
    bonds of the flanking Pb -- identified label-free as the Pb whose sixth-nearest Cl lies
    beyond `shell_gap` (a first shell of five) -- against the other Pb."""
    z = torch.as_tensor([int(x) for x in numbers], device=positions.device)
    r = minimum_image_distances(positions.detach(), cell.detach())
    pb = torch.nonzero(z == cation).reshape(-1)
    cl = torch.nonzero(z == anion).reshape(-1)
    d = torch.sort(r[pb][:, cl], dim=1).values
    flank = d[:, 5] > shell_gap
    flank_bonds = d[flank][:, :5].reshape(-1)
    other_bonds = d[~flank][:, :6].reshape(-1)
    return {"n_flanking_pb": int(flank.sum()),
            "flanking_beyond_r_d1": float((flank_bonds > cfg.r_d1).to(torch.float64).mean()) if flank_bonds.numel() else float("nan"),
            "other_beyond_r_d1": float((other_bonds > cfg.r_d1).to(torch.float64).mean()) if other_bonds.numel() else float("nan"),
            "n_flanking_bonds": int(flank_bonds.numel()), "n_other_bonds": int(other_bonds.numel())}


def placement_fractions(positions: torch.Tensor, cell: torch.Tensor, numbers: Sequence[int],
                        cfg: KernelConfig, cation: int = 82, anion: int = 17, shell: int = 6
                        ) -> Dict[str, float]:
    """Regime A placement check over IDENTIFIED bonds (plan section 2.4, label-free): the
    fraction of first-shell Pb-Cl bonds (the `shell` nearest Cl of each Pb by distance
    rank) with `r > r_d1`, and of intra-octahedron Cl-Cl nearest neighbours (the nearest
    Cl of each Cl, both of them in the same Pb's first shell) with `r < r_d2`."""
    z = torch.as_tensor([int(x) for x in numbers], device=positions.device)
    r = minimum_image_distances(positions.detach(), cell.detach())
    pb = torch.nonzero(z == cation).reshape(-1)
    cl = torch.nonzero(z == anion).reshape(-1)
    if pb.numel() == 0 or cl.numel() < shell:
        raise ValueError("placement check needs Pb and at least a shell of Cl")
    d_pb_cl = r[pb][:, cl]                                   # [n_Pb, n_Cl]
    order = torch.argsort(d_pb_cl, dim=1)[:, :shell]          # first shell by rank
    first_shell = torch.gather(d_pb_cl, 1, order)             # [n_Pb, shell]
    frac_pb_cl = float((first_shell > cfg.r_d1).to(torch.float64).mean())
    # Intra-octahedron Cl-Cl: pairs of Cl in the same Pb's first shell, nearest-neighbour
    # pairs of that octahedron (the 12 edges: each Cl's 4 nearest of the other 5).
    members = cl[order]                                       # [n_Pb, shell] atom indices
    edges = []
    for row in members:
        sub = r[row][:, row]
        sub = sub + torch.eye(shell, dtype=sub.dtype, device=sub.device) * 1e9
        nearest = torch.sort(sub, dim=1).values[:, :4]        # 4 edge neighbours per Cl
        edges.append(nearest.reshape(-1))
    edges_t = torch.cat(edges)
    frac_cl_cl = float((edges_t < cfg.r_d2).to(torch.float64).mean())
    return {"pb_cl_first_shell_beyond_r_d1": frac_pb_cl,
            "cl_cl_octahedron_edge_below_r_d2": frac_cl_cl,
            "n_pb_cl_bonds": int(first_shell.numel()), "n_cl_cl_edges": int(edges_t.numel())}
