"""Plan P (2026-09-16): the polarisation coupling -- matrix-free periodic dipole kernels.

Conventions (registered in the tracker, "Plan P -- REGISTRATION"): every smoothed periodic
kernel is the pure reciprocal-space sum of `ewald.reciprocal_matrix`,

    Gamma_w(r) = C sum_L erf(|r + L| / w) / |r + L|  (neutralising background, k = 0 excluded)
               = C (4 pi / V) sum_{k != 0} exp(-k^2 w^2 / 4) / k^2 cos(k . r),

with `C = COULOMB` (eV A / e^2). The carrier--dipole kernel is the existing carrier--host one,
`w_cP = sqrt(2 (r_g^2 + r_split^2))` (`kernels.gamma_lr`); the dipole--dipole kernel takes both
widths `r_split`, `w_PP = 2 r_split`. Both are divided by ONE `eps_inf`.

  E_cP = - sum_a p_a . E_c(R_a),   E_c(R_a) = - grad_a sum_c dq_c Gamma_{w_cP}(R_a - R_c) / eps_inf
  E_PP = 1/2 sum_{a != b} p_a^T T(a, b) p_b / eps_inf,   T(a, b) = - grad grad Gamma_{w_PP}(R_a - R_b)
         (images of a = b included; the direct a = b, L = 0 term removed analytically:
          - p^T grad grad Gamma_w(0) p = 4 C |p|^2 / (3 sqrt(pi) w^3))

Everything is a structure-factor sum, O(N n_k), differentiable in `positions` (through the
phases) and in `p` / `dq` (through the factors); `cell` is detached in the k set, as
`reciprocal_plan` does. float64 only (plan section 1).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .ewald import COULOMB, EWALD_TOL, reciprocal_plan

_SQRT_PI = math.sqrt(math.pi)


def pair_width_cp(r_g: float, r_split: float) -> float:
    """The carrier--host pair width `gamma_lr` uses: sqrt(2 (r_g^2 + r_split^2))."""
    return math.sqrt(2.0 * (r_g ** 2 + r_split ** 2))


def pair_width_pp(r_split: float) -> float:
    """Two dipoles smoothed at `r_split`: sqrt(2 (r_split^2 + r_split^2)) = 2 r_split."""
    return 2.0 * r_split


def _check(positions: torch.Tensor, cell: torch.Tensor) -> None:
    if positions.dtype != torch.float64 or cell.dtype != torch.float64:
        raise TypeError("the polarisation kernels are evaluated in float64 (plan section 1)")
    if positions.dim() != 2 or positions.shape[-1] != 3 or cell.shape != (3, 3):
        raise ValueError(f"expected positions [N, 3] and cell [3, 3]; got {tuple(positions.shape)}, {tuple(cell.shape)}")


def _phases(positions: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    phase = positions @ k.transpose(0, 1)                    # [N, n_k]
    return torch.cos(phase), torch.sin(phase)


def carrier_field(positions: torch.Tensor, cell: torch.Tensor, dq: torch.Tensor,
                  r_g: float, r_split: float, eps_inf: float,
                  tol: float = EWALD_TOL) -> torch.Tensor:
    """`E_c(R_a) = - grad_a phi_c(R_a)`, `phi_c(r) = sum_c dq_c Gamma_{w_cP}(r - R_c) / eps_inf`,
    at every atom, `[N, 3]` in V/A (C dq / A^2 with C in eV A / e^2 is eV / (e A) = V / A)."""
    _check(positions, cell)
    plan = reciprocal_plan(cell, pair_width_cp(r_g, r_split), tol)
    k, weight, volume = plan.k, plan.weight, plan.volume                       # [n_k, 3], [n_k]
    c, s = _phases(positions, k)
    qc = dq @ c                                                                  # [n_k]
    qs = dq @ s
    # E_c(r) = (4 pi C / (V eps)) sum_k w_k k [sin(k.r) Qc - cos(k.r) Qs]
    coef = (s * (weight * qc) - c * (weight * qs))                               # [N, n_k]
    return (4.0 * math.pi * COULOMB / (volume * eps_inf)) * (coef @ k)           # [N, 3]


def carrier_dipole_energy(positions: torch.Tensor, cell: torch.Tensor, dq: torch.Tensor,
                          p: torch.Tensor, r_g: float, r_split: float, eps_inf: float,
                          tol: float = EWALD_TOL) -> torch.Tensor:
    """`E_cP = - sum_a p_a . E_c(R_a)` (eV), matrix-free:
    `-(4 pi C / (V eps)) sum_k w_k [P^s_k Q^c_k - P^c_k Q^s_k]` with
    `P^{c,s}_k = sum_a (k . p_a) {cos, sin}(k . R_a)` and `Q^{c,s}_k = sum_c dq_c {cos, sin}(k . R_c)`."""
    _check(positions, cell)
    plan = reciprocal_plan(cell, pair_width_cp(r_g, r_split), tol)
    k, weight, volume = plan.k, plan.weight, plan.volume
    c, s = _phases(positions, k)
    kp = p @ k.transpose(0, 1)                                                   # [N, n_k]
    pc, ps = (kp * c).sum(0), (kp * s).sum(0)
    qc, qs = dq @ c, dq @ s
    return -(4.0 * math.pi * COULOMB / (volume * eps_inf)) * (weight * (ps * qc - pc * qs)).sum()


def dipole_self_coefficient(w: float) -> float:
    """`- p^T grad grad Gamma_w(0) p / |p|^2 = 4 C / (3 sqrt(pi) w^3)`: the direct self term
    of a Gaussian-smoothed dipole, which the k-sum over all pairs contains and E_PP excludes."""
    return 4.0 * COULOMB / (3.0 * _SQRT_PI * w ** 3)


def dipole_dipole_energy(positions: torch.Tensor, cell: torch.Tensor, p: torch.Tensor,
                         r_split: float, eps_inf: float, tol: float = EWALD_TOL) -> torch.Tensor:
    """`E_PP = 1/(2 eps) [ (4 pi C / V) sum_k w_k |sum_a (k . p_a) e^{i k . R_a}|^2
    - 4 C / (3 sqrt(pi) w^3) sum_a |p_a|^2 ]` (eV), `w = 2 r_split`."""
    _check(positions, cell)
    w = pair_width_pp(r_split)
    plan = reciprocal_plan(cell, w, tol)
    k, weight, volume = plan.k, plan.weight, plan.volume
    c, s = _phases(positions, k)
    kp = p @ k.transpose(0, 1)                                                   # [N, n_k]
    pc, ps = (kp * c).sum(0), (kp * s).sum(0)
    recip = (4.0 * math.pi * COULOMB / volume) * (weight * (pc * pc + ps * ps)).sum()
    self_term = dipole_self_coefficient(w) * (p * p).sum()
    return 0.5 * (recip - self_term) / eps_inf


def dipole_field(positions: torch.Tensor, cell: torch.Tensor, p: torch.Tensor,
                 r_split: float, eps_inf: float, tol: float = EWALD_TOL) -> torch.Tensor:
    """`(T_LR p)_a / eps = dE_PP / dp_a` at fixed positions: the (negative) field the other
    dipoles and the images exert on `a`, `[N, 3]` in V/A. For the cotangent form of the
    forces and as a diagnostic; the energies above do not need it."""
    _check(positions, cell)
    w = pair_width_pp(r_split)
    plan = reciprocal_plan(cell, w, tol)
    k, weight, volume = plan.k, plan.weight, plan.volume
    c, s = _phases(positions, k)
    kp = p @ k.transpose(0, 1)
    pc, ps = (kp * c).sum(0), (kp * s).sum(0)
    coef = c * (weight * pc) + s * (weight * ps)                                 # [N, n_k]
    recip = (4.0 * math.pi * COULOMB / volume) * (coef @ k)                      # [N, 3]
    return (recip - dipole_self_coefficient(w) * p) / eps_inf


# ------------------------------------------------------------------ brute-force references

def _images_within(cell: torch.Tensor, radius: float) -> torch.Tensor:
    from .ewald import image_vectors
    return image_vectors(cell, radius)


def dipole_dipole_energy_real_space(positions: torch.Tensor, cell: torch.Tensor, p: torch.Tensor,
                                    r_split: float, eps_inf: float, radius: float) -> torch.Tensor:
    """Reference: the same energy as a real-space sum over images within `radius` of the
    smoothed kernel `C erf(r / w) / r`, `a = b, L = 0` skipped. Converges absolutely only for a
    net-zero dipole per cell (tests use one); O(N^2 n_L), tests only."""
    w = pair_width_pp(r_split)
    n = positions.shape[0]
    images = _images_within(cell, radius)                                       # [n_L, 3]
    d0 = positions.unsqueeze(1) - positions.unsqueeze(0)                         # [N, N, 3]
    total = torch.zeros((), dtype=torch.float64, device=positions.device)
    eye = torch.eye(n, dtype=torch.bool, device=positions.device)
    for L in images:
        d = d0 + L                                                               # [N, N, 3]
        r = d.norm(dim=-1)
        skip = eye & (L.abs().sum() < 1e-12)
        r = torch.where(skip, torch.ones_like(r), r)
        r = torch.where(r > radius, torch.full_like(r, float("nan")), r)
        # grad grad [erf(r/w)/r] = A(r) I + B(r) r r^T with
        #   f(r) = erf(r/w)/r; f' ; f'' ; grad grad f = (f'/r) I + (f'' - f'/r) rhat rhat^T
        x = r / w
        erf = torch.erf(x)
        g = (2.0 / _SQRT_PI) * torch.exp(-x * x) / w                             # d erf(r/w)/dr
        f1 = g / r - erf / r ** 2
        f2 = (-2.0 * x / w) * g / r - 2.0 * g / r ** 2 + 2.0 * erf / r ** 3
        a_r = f1 / r
        b_r = (f2 - f1 / r) / r ** 2
        pa_pb = torch.einsum("ai,bi->ab", p, p)
        pa_d = torch.einsum("ai,abi->ab", p, d)
        pb_d = torch.einsum("bi,abi->ab", p, d)
        pair = -(a_r * pa_pb + b_r * pa_d * pb_d)                                # - p_a^T grad grad Gamma p_b
        pair = torch.where(skip | torch.isnan(pair), torch.zeros_like(pair), pair)
        total = total + pair.sum()
    return 0.5 * COULOMB * total / eps_inf


# ------------------------------------------------------------------ Phase 2b: the range-separated coupling

def switch(r: torch.Tensor, r_c: float, delta: float) -> torch.Tensor:
    """`S(r)`: 0 for `r <= r_c - delta`, 1 for `r >= r_c`, the quintic (C^2) ramp
    `x^3 (10 - 15 x + 6 x^2)` between, `x = (r - r_c + delta) / delta`. `r_c <= 0` means S = 1
    everywhere (the plan's `S == 1` identity test)."""
    if r_c <= 0.0:
        return torch.ones_like(r)
    x = ((r - (r_c - delta)) / delta).clamp(0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def gamma_direct(r: torch.Tensor, w: float, eps_inf: float) -> torch.Tensor:
    """The direct (non-periodic) smoothed Coulomb kernel of a pair of Gaussians with combined
    width `w`: `C erf(r / w) / r / eps_inf` (eV / e^2); its `r -> 0` limit is `gamma_direct_self`."""
    return COULOMB * torch.erf(r / w) / r / eps_inf


def gamma_direct_self(w: float, eps_inf: float) -> float:
    """`lim_{r -> 0} C erf(r / w) / r / eps = 2 C / (sqrt(pi) w eps)`."""
    return 2.0 * COULOMB / (_SQRT_PI * w * eps_inf)


def direct_field(d: torch.Tensor, w: float, eps_inf: float) -> torch.Tensor:
    """The direct field at `R_a` of a unit Gaussian charge (combined pair width `w`) at `R_c`,
    `d = R_a - R_c` `[E, 3]`: `E = - grad_a [C erf(r / w) / r] / eps`, in V/A per unit charge."""
    r = d.norm(dim=-1).clamp_min(1e-12)
    x = r / w
    f = torch.erf(x) / r
    df = (2.0 / _SQRT_PI) * torch.exp(-x * x) / (w * r) - torch.erf(x) / (r * r)   # d f / d r
    return -(COULOMB / eps_inf) * (df / r).unsqueeze(-1) * d


def switched_terms(dq: torch.Tensor, p: torch.Tensor, edge_index: torch.Tensor, edge_vector: torch.Tensor,
                   r_g: float, r_split: float, eps_inf: float, r_c: float, delta: float,
                   sq0: Optional[torch.Tensor] = None, near_r_split: float = 2.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Phase 2b, one graph. Over the head's directed edges `c = edge_index[0] -> a = edge_index[1]`
    with `edge_vector = R_a - R_c` (image shifts included; every ordered pair once):

      e_far_corr = + sum_{a, c} dq_c (1 - S(r_ac)) p_a . E_direct(R_a - R_c)   [w = pair_width_cp(r_g, r_split)]
        so that `carrier_dipole_energy + e_far_corr = E_far` of the plan (the periodic coupling
        minus its near-field part; identical to E_cP beyond r_c, zero inside r_c - delta);
      e_near = sum_{a, c} dq_c (1 - S(r_ac)) (s q0)_a gamma_2.5(r_ac) + sum_a dq_a (s q0)_a gamma_2.5(0)
        (arm B_s; `sq0 = s q0`; the a = c self term added explicitly since the edge list has no
        self edges and `1 - S(0) = 1`), or 0 when `sq0` is None (arm B_0).
    Pairs beyond `r_c` contribute nothing (`1 - S = 0`) and may be absent from the edge list."""
    c_idx, a_idx = edge_index[0], edge_index[1]
    r = edge_vector.norm(dim=-1)
    one_minus_s = 1.0 - switch(r, r_c, delta)
    keep = one_minus_s > 0.0
    if keep.any():
        c_k, a_k, d_k, w_k = c_idx[keep], a_idx[keep], edge_vector[keep], one_minus_s[keep]
        e_dir = direct_field(d_k, pair_width_cp(r_g, r_split), eps_inf)                 # [E, 3]
        e_far_corr = (dq[c_k] * w_k * (p[a_k] * e_dir).sum(-1)).sum()
    else:
        e_far_corr = (dq * 0.0).sum()
    if sq0 is None:
        return e_far_corr, e_far_corr * 0.0
    w_near = pair_width_cp(r_g, near_r_split)
    self_weight = 1.0 - float(switch(torch.zeros(1, dtype=dq.dtype, device=dq.device), r_c, delta)[0])
    e_near = (dq * sq0).sum() * (gamma_direct_self(w_near, eps_inf) * self_weight)
    if keep.any():
        e_near = e_near + (dq[c_k] * w_k * sq0[a_k] * gamma_direct(r[keep], w_near, eps_inf)).sum()
    return e_far_corr, e_near
