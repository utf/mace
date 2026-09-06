"""Plan section 2.6: the D-SCC solve. This module holds the fill-level primitives shared by
the `Phi = 0` path (two fillings of `H0`, Arm 1) and the self-consistent loop (Phase 1).

`two_fillings(H, N_S, N_ref)`: one Hamiltonian, one `eigh`, four fills (two spins x two
states); returns the band-form head energy `sum_sigma [F_band(H, N_S_sigma) - F_band(H,
N_ref_sigma)]`, the attached density difference `dP = sum_sigma (P_S - P_ref)` and the
site charge difference `dq_i = -Tr(Pi_i dP)` (which sums to `Q` exactly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from mace.modules.dscc.fill import SIGMA_S, FillResult, fill

ORBITALS = 4


def site_charge_difference(dP: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """`dq_i = -sum_{mu in i} dP_mu mu` (charges in units of +e: an electron removed from
    site i makes `dq_i` positive)."""
    return -torch.diagonal(dP, dim1=-2, dim2=-1).reshape(*dP.shape[:-2], n_atoms, ORBITALS).sum(-1)


@dataclass
class TwoFillings:
    energy: torch.Tensor        # J (band form), differentiable through H
    dP: torch.Tensor            # sum_sigma (P_S - P_ref), attached
    dq: torch.Tensor            # [N], attached
    fills: Tuple[FillResult, FillResult, FillResult, FillResult]   # S_up, S_dn, ref_up, ref_dn


def two_fillings(H: torch.Tensor, n_s: Tuple[int, int], n_ref: Tuple[int, int],
                 sigma_s: float = SIGMA_S) -> TwoFillings:
    """Fill one `H` at the state's and the reference's per-spin counts."""
    with torch.no_grad():
        eps, U = torch.linalg.eigh(H)
    spectrum = (eps, U)
    s_up = fill(H, float(n_s[0]), sigma_s, spectrum)
    s_dn = fill(H, float(n_s[1]), sigma_s, spectrum)
    r_up = fill(H, float(n_ref[0]), sigma_s, spectrum)
    r_dn = fill(H, float(n_ref[1]), sigma_s, spectrum)
    energy = (s_up.F_band - r_up.F_band) + (s_dn.F_band - r_dn.F_band)
    dP = (s_up.P - r_up.P) + (s_dn.P - r_dn.P)
    n_atoms = H.shape[-1] // ORBITALS
    return TwoFillings(energy=energy, dP=dP, dq=site_charge_difference(dP, n_atoms),
                       fills=(s_up, s_dn, r_up, r_dn))


# ------------------------------------------------------------------ the D-SCC loop

@dataclass(frozen=True)
class ScfOptions:
    """Registered solver numbers (plan section 2.6; defaults to confirm before use)."""
    tol_q: float = 1e-8          # unmixed residual, max-norm, e
    tol_E: float = 1e-10         # change of J between iterations, eV
    tol_c: float = 1e-7          # commutator norm at the fixed point
    n_max: int = 100
    mixing: float = 0.3          # Anderson damping
    history: int = 6             # Anderson depth
    tol_root: float = 1e-6       # root rule: |dq_a - dq_b| max-norm
    rho_ceiling: float = 0.9     # contraction ratio ceiling
    continuation_steps: int = 4  # registered schedule: s = k / steps, k = 1..steps


class ScfError(RuntimeError):
    """A solve that did not converge, or violated a fixed-point bound."""


@dataclass
class ScfResult:
    energy: torch.Tensor            # J* (band form), attached
    energy_primary: torch.Tensor    # the primary functional at the solution, detached
    dP: torch.Tensor                # attached
    dq: torch.Tensor                # attached
    V: torch.Tensor                 # site potential at the solution, detached
    iterations: int
    converged: bool
    residual: float                 # unmixed residual max-norm at the solution
    delta_energy: float
    commutator: float
    rho: float                      # contraction ratio over the last unmixed residuals
    history: List[float]
    fills: Tuple[FillResult, FillResult, FillResult, FillResult]


def site_potential_matrix(V: torch.Tensor) -> torch.Tensor:
    """`sum_i V_i Pi_i` as a diagonal `[4N, 4N]` matrix."""
    return torch.diag(V.repeat_interleave(ORBITALS))


def primary_functional(H0: torch.Tensor, gamma: torch.Tensor, sol: TwoFillings,
                       W: Optional[torch.Tensor]) -> torch.Tensor:
    """`J[P_S, P_ref] = sum_sigma {Tr[(P_S - P_ref) H0] + R(P_S) - R(P_ref)} + Phi_cc + E_SF`
    at the given densities (detached)."""
    s_up, s_dn, r_up, r_dn = sol.fills
    dP = sol.dP.detach()
    dq = sol.dq.detach()
    value = (dP * H0.detach()).sum() + (s_up.entropy + s_dn.entropy - r_up.entropy - r_dn.entropy)
    value = value + 0.5 * dq @ gamma.detach() @ dq
    if W is not None:
        value = value + dq @ W.detach()
    return value


def _anderson(dq_hist: List[torch.Tensor], res_hist: List[torch.Tensor], mixing: float
              ) -> torch.Tensor:
    """Anderson acceleration: the combination of past iterates whose residuals cancel
    best, stepped by `mixing` along the combined residual."""
    if len(res_hist) == 1:
        return dq_hist[0] + mixing * res_hist[0]
    F = torch.stack(res_hist, dim=1)                           # [N, m]
    X = torch.stack(dq_hist, dim=1)
    dF = F[:, 1:] - F[:, :-1]
    dX = X[:, 1:] - X[:, :-1]
    # Least squares for the coefficients on the residual differences (Walker-Ni form).
    gamma_coef = torch.linalg.lstsq(dF, F[:, -1:]).solution.reshape(-1)
    x_new = X[:, -1] - dX @ gamma_coef
    f_new = F[:, -1] - dF @ gamma_coef
    return x_new + mixing * f_new


def solve_dscc(H0: torch.Tensor, gamma: torch.Tensor, n_s: Tuple[int, int],
               n_ref: Tuple[int, int], sigma_s: float = SIGMA_S,
               W: Optional[torch.Tensor] = None, dq0: Optional[torch.Tensor] = None,
               options: Optional[ScfOptions] = None, unroll: bool = False) -> ScfResult:
    """The stationary point of the D-SCC functional (plan section 2.6).

    Iterates `V = Gamma dq + W`, `H = H0 - sum_i V_i Pi_i`, four fills, `dq_new`, with the
    UNMIXED residual `dq_new - dq` deciding convergence and Anderson mixing proposing the
    next `dq`. With `unroll` the iterations stay on the autograd graph (the training
    gradient of plan section 6); otherwise they run under `no_grad` and one attached pass
    at the fixed point provides `J*`, `dP` and `dq`.
    """
    opt = options or ScfOptions()
    n_atoms = H0.shape[-1] // ORBITALS
    dq = (torch.zeros(n_atoms, dtype=H0.dtype, device=H0.device) if dq0 is None
          else dq0.to(H0.dtype))
    dq_hist: List[torch.Tensor] = []
    res_hist: List[torch.Tensor] = []
    history: List[float] = []
    energy_prev: Optional[float] = None
    converged = False
    iterations = 0
    delta_energy = float("inf")
    context = torch.enable_grad() if unroll else torch.no_grad()
    with context:
        for k in range(opt.n_max):
            iterations = k + 1
            V = gamma @ dq + (W if W is not None else 0.0)
            H = H0 - site_potential_matrix(V)
            sol = two_fillings(H, n_s, n_ref, sigma_s)
            res = sol.dq - dq
            r_norm = float(res.detach().abs().max())
            history.append(r_norm)
            energy_k = float(sol.energy.detach() - 0.5 * dq.detach() @ gamma.detach() @ dq.detach())
            delta_energy = abs(energy_k - energy_prev) if energy_prev is not None else float("inf")
            energy_prev = energy_k
            if r_norm < opt.tol_q and delta_energy < opt.tol_E:
                converged = True
                break
            dq_hist.append(dq)
            res_hist.append(res)
            if len(dq_hist) > opt.history:
                dq_hist.pop(0)
                res_hist.pop(0)
            dq = _anderson(dq_hist, res_hist, opt.mixing)
    # The attached pass at the fixed point (or the last iterate, flagged).
    dq_star = dq if unroll else dq.detach()
    V = gamma @ dq_star + (W if W is not None else 0.0)
    H = H0 - site_potential_matrix(V)
    sol = two_fillings(H, n_s, n_ref, sigma_s)
    residual = float((sol.dq.detach() - dq_star.detach()).abs().max())
    # Band form: J* = sum_sigma [F_band(H, N_S) - F_band(H, N_ref)] - 0.5 dq^T Gamma dq, with
    # W (linear in dq) already inside H through V -- the identity holds with it there.
    energy = sol.energy - 0.5 * dq_star @ gamma @ dq_star
    energy_primary = primary_functional(H0, gamma, sol, W)
    # Commutator of the self-consistent H with the state's majority density.
    P_s = sol.fills[0].P.detach()
    H_next = (H0 - site_potential_matrix(gamma @ sol.dq.detach() + (W if W is not None else 0.0))).detach()
    commutator = float((H_next @ P_s - P_s @ H_next).norm())
    tail = [h for h in history[-4:] if h > 0]
    rho = float(tail[-1] / tail[-2]) if len(tail) >= 2 else 0.0
    return ScfResult(energy=energy, energy_primary=energy_primary, dP=sol.dP, dq=sol.dq,
                     V=V.detach(), iterations=iterations, converged=converged,
                     residual=residual, delta_energy=delta_energy, commutator=commutator,
                     rho=rho, history=history, fills=sol.fills)


def root_rule(H0: torch.Tensor, gamma_full: torch.Tensor, gamma_zero: torch.Tensor,
              n_s: Tuple[int, int], n_ref: Tuple[int, int], sigma_s: float = SIGMA_S,
              W: Optional[torch.Tensor] = None, dq_previous: Optional[torch.Tensor] = None,
              options: Optional[ScfOptions] = None) -> Dict[str, Any]:
    """Plan section 2.6: the registered initialisations -- (i) `dq = 0`; (ii) continuation
    from zero coupling (`gamma_zero`: `lambda_dir = 0`, `U_eff = 0`, `W = 0`) to the trained
    values in `continuation_steps`; (iii) warm start from `dq_previous` -- must reach the
    same fixed point to `tol_root`. Returns the solutions and the largest pairwise
    distance; `passed` is the gate."""
    opt = options or ScfOptions()
    sols: Dict[str, ScfResult] = {}
    sols["zero"] = solve_dscc(H0, gamma_full, n_s, n_ref, sigma_s, W, None, opt)
    dq = None
    for k in range(1, opt.continuation_steps + 1):
        s = k / opt.continuation_steps
        gamma_s = gamma_zero + s * (gamma_full - gamma_zero)
        W_s = None if W is None else s * W
        step = solve_dscc(H0, gamma_s, n_s, n_ref, sigma_s, W_s, dq, opt)
        dq = step.dq.detach()
    sols["continuation"] = solve_dscc(H0, gamma_full, n_s, n_ref, sigma_s, W, dq, opt)
    if dq_previous is not None:
        sols["warm"] = solve_dscc(H0, gamma_full, n_s, n_ref, sigma_s, W, dq_previous, opt)
    names = list(sols)
    spread = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            spread = max(spread, float((sols[names[i]].dq.detach() - sols[names[j]].dq.detach()).abs().max()))
    return {"solutions": sols, "spread": spread,
            "passed": spread < opt.tol_root and all(s.converged for s in sols.values())}
