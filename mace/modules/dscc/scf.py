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

import math

from mace.modules.dscc.fill import SIGMA_S, FillResult, chemical_potential, fill

_SQRT_PI = math.sqrt(math.pi)

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
    method: str = "newton"       # "newton" (exact response Jacobian, damped) or "anderson"
    max_backtrack: int = 6       # Newton: halvings of the step when the residual grows


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
    newton_steps: int = 0
    anderson_steps: int = 0


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
    # Least squares for the coefficients on the residual differences (Walker-Ni form), by
    # the Tikhonov-regularised normal equations: the residual differences are nearly
    # collinear late in a solve, and a full-rank QR driver (CUDA's `gels`) returns garbage
    # there where a rank-revealing one does not -- the solve must not depend on the device.
    G = dF.transpose(0, 1) @ dF
    ridge = 1e-10 * float(torch.diagonal(G).max().clamp_min(1e-300))
    rhs = dF.transpose(0, 1) @ F[:, -1]
    gamma_coef = torch.linalg.solve(G + ridge * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), rhs)
    x_new = X[:, -1] - dX @ gamma_coef
    f_new = F[:, -1] - dF @ gamma_coef
    return x_new + mixing * f_new


def solve_dscc(H0: torch.Tensor, gamma: torch.Tensor, n_s: Tuple[int, int],
               n_ref: Tuple[int, int], sigma_s: float = SIGMA_S,
               W: Optional[torch.Tensor] = None, dq0: Optional[torch.Tensor] = None,
               options: Optional[ScfOptions] = None, unroll: bool = False,
               implicit: bool = False) -> ScfResult:
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
    newton_steps = anderson_steps = 0
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
            accepted = False
            if opt.method == "newton":
                jac = hole_response(H, n_s, n_ref, sigma_s, spectrum=(sol.fills[0].eps, sol.fills[0].U)) @ gamma
                step = _newton_step(res.detach(), jac.detach())
                # Damped: halve the step while the unmixed residual grows (the map is
                # strongly non-linear where levels are nearly degenerate on the smearing
                # scale); if no damping helps, the Anderson step on the history is taken
                # instead, so the solver is never worse than Anderson on that iteration.
                scale = 1.0
                for _ in range(opt.max_backtrack):
                    trial = dq + scale * step
                    V_t = gamma @ trial + (W if W is not None else 0.0)
                    r_t = two_fillings(H0 - site_potential_matrix(V_t), n_s, n_ref, sigma_s).dq - trial
                    if float(r_t.detach().abs().max()) < r_norm:
                        accepted = True
                        break
                    scale *= 0.5
                if accepted:
                    dq = dq + scale * step
                    newton_steps += 1
            if not accepted:
                dq = _anderson(dq_hist, res_hist, opt.mixing)
                anderson_steps += 1
    # The attached pass at the fixed point (or the last iterate, flagged).
    if implicit and not unroll:
        # dq* = g(dq*, theta): d dq*/d theta = (I - J)^{-1} dg/d theta (implicit-function
        # theorem, exact for a converged solve). One attached evaluation of g at the
        # detached fixed point supplies dg/d theta; the hook on its output applies
        # (I - J^T)^{-1} to whatever cotangent arrives from downstream. One extra fill
        # instead of the whole unrolled history.
        dq_leaf = dq.detach().clone().requires_grad_(True)
        V1 = gamma @ dq_leaf + (W if W is not None else 0.0)
        sol1 = two_fillings(H0 - site_potential_matrix(V1), n_s, n_ref, sigma_s)
        jac = hole_response((H0 - site_potential_matrix(V1)).detach(), n_s, n_ref, sigma_s,
                            spectrum=(sol1.fills[0].eps, sol1.fills[0].U)) @ gamma.detach()
        A_T = (torch.eye(n_atoms, dtype=H0.dtype, device=H0.device) - jac).transpose(0, 1)
        dq_star = sol1.dq
        dq_star.register_hook(lambda c: torch.linalg.solve(A_T, c))
    else:
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
                     rho=rho, history=history, fills=sol.fills,
                     newton_steps=newton_steps, anderson_steps=anderson_steps)


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


# ------------------------------------------------------------------ exact Jacobian, Newton

def hole_response(H: torch.Tensor, n_s: Tuple[int, int], n_ref: Tuple[int, int],
                  sigma_s: float = SIGMA_S, active_tol: float = 1e-12,
                  spectrum: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    """`M_ij = d dq_new_i / d V_j` at `H = H0 - diag(V)`: the site-resolved response of the
    charge DIFFERENCE `dq = -sum_sigma Tr(Pi (P_S - P_ref))` to the site potential, exact
    in the eigenbasis (Daleckii-Krein divided differences with the fixed-N correction, as
    the fill's backward). The difference of two fills of one `H` responds only through the
    levels whose occupation differs between them (plus the smearing tails): the sum runs
    over pairs with one index in that active set, so the cost is `O(N_act n_orb N^2)`.
    `dq_new(dq) = dq_new(V = Gamma dq + W)`, so the fixed-point Jacobian is `M Gamma`.
    """
    with torch.no_grad():
        eps, U = torch.linalg.eigh(H) if spectrum is None else spectrum
        n_orb = H.shape[-1]
        n_atoms = n_orb // ORBITALS
        M = torch.zeros(n_atoms, n_atoms, dtype=H.dtype, device=H.device)
        for n_state, n_reference in ((n_s[0], n_ref[0]), (n_s[1], n_ref[1])):
            if n_state == n_reference:
                continue
            terms = []
            for n_el, sign in ((n_state, 1.0), (n_reference, -1.0)):
                mu = chemical_potential(eps, float(n_el), sigma_s)
                x = (eps - mu) / sigma_s
                f = 0.5 * torch.erfc(x)
                fp = -torch.exp(-x * x) / (sigma_s * _SQRT_PI)           # df/d eps <= 0
                terms.append((f, fp, sign))
            f_s, f_r = terms[0][0], terms[1][0]
            active = torch.nonzero((f_s - f_r).abs() > active_tol).reshape(-1)
            if active.numel() == 0:
                continue
            # A^(i)_ab = sum_{mu in i} U_mu a U_mu b for a in the active set, all b: [N, n_act, n].
            U_act = U[:, active]                                              # [n, n_act]
            A = torch.einsum("ma,mb->mab", U_act, U).reshape(n_atoms, ORBITALS, active.numel(), n_orb).sum(1)
            for f, fp, sign in terms:
                # Divided differences L_ab for a active, all b; the limit f'(mid) at coincidence.
                d_eps = eps[active].unsqueeze(-1) - eps.unsqueeze(0)
                d_f = f[active].unsqueeze(-1) - f.unsqueeze(0)
                near = d_eps.abs() <= 1e-7
                mid = 0.5 * (eps[active].unsqueeze(-1) + eps.unsqueeze(0))
                x_mid = (mid - chemical_potential(eps, float(n_state if sign > 0 else n_reference), sigma_s)) / sigma_s
                L = torch.where(near, -torch.exp(-x_mid * x_mid) / (sigma_s * _SQRT_PI),
                                d_f / torch.where(near, torch.ones_like(d_eps), d_eps))
                # Pairs (a in act, b any) counted once, (b in act, a any) once, minus both-in-act.
                both = torch.zeros(active.numel(), n_orb, dtype=torch.bool, device=H.device)
                both[:, active] = True
                weight = torch.where(both, torch.ones_like(L), 2.0 * torch.ones_like(L))
                # dP/dV_j = -sum_ab L_ab A^(j)_ab (U_a U_b^T) -> Tr(Pi_i dP/dV_j) = -sum L A^i A^j;
                # dq_new_i = -sum_sigma sign Tr(Pi_i P) -> +sum L A^i A^j per fill, signed.
                contrib = torch.einsum("iab,ab,jab->ij", A, weight * L, A)
                # Fixed-N correction: mu moves with V. d mu / d V_j = -(sum_b f'_b A^j_bb) / (sum f'),
                # and the density responds by f'_a along the diagonal: subtract the rank-one term.
                diag_A = torch.einsum("ma,ma->ma", U, U).reshape(n_atoms, ORBITALS, n_orb).sum(1)  # [N, n]
                s_fp = fp.sum()
                if float(s_fp.abs()) > 1e-300:
                    g = diag_A @ fp                                            # [N]
                    contrib = contrib - torch.outer(g, g) / s_fp
                M = M + sign * contrib
        return M


def _newton_step(res: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
    """`delta` with `(I - J) delta = res` for the fixed point of `dq -> dq_new(dq)`."""
    n = res.shape[0]
    return torch.linalg.solve(torch.eye(n, dtype=res.dtype, device=res.device) - jac, res)
