"""Plan section 2.6: the D-SCC solve. This module holds the fill-level primitives shared by
the `Phi = 0` path (two fillings of `H0`, Arm 1) and the self-consistent loop (Phase 1).

`two_fillings(H, N_S, N_ref)`: one Hamiltonian, one `eigh`, four fills (two spins x two
states); returns the band-form head energy `sum_sigma [F_band(H, N_S_sigma) - F_band(H,
N_ref_sigma)]`, the attached density difference `dP = sum_sigma (P_S - P_ref)` and the
site charge difference `dq_i = -Tr(Pi_i dP)` (which sums to `Q` exactly).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import math

from mace.modules.dscc.fill import (EIGH_CPU_MAX_ORBITALS, SIGMA_S, FillResult, chemical_potential, eigh_for, fill,
                                    generalised_entropy, occupations)

_SQRT_PI = math.sqrt(math.pi)

ORBITALS = 4


def site_charge_difference(dP: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """`dq_i = -sum_{mu in i} dP_mu mu` (charges in units of +e: an electron removed from
    site i makes `dq_i` positive)."""
    return -torch.diagonal(dP, dim1=-2, dim2=-1).reshape(*dP.shape[:-2], n_atoms, ORBITALS).sum(-1)


FRONTIER_TOL = 1e-10     # v5 W2 item 2 (registered): a level is active when |f_S - f_ref| exceeds this


class TwoFillings:
    """The two fillings of one `H`: `energy` (band form, differentiable through `H` on the
    eager path), `dq` `[..., N]`, the four `fills` (S_up, S_dn, ref_up, ref_dn) and `dP =
    sum_sigma (P_S - P_ref)`, formed on demand from the active levels on the frontier path
    (v5 W2 item 2) and held eagerly (attached) on the training path."""

    def __init__(self, energy: torch.Tensor, dq: torch.Tensor, fills, dP: Optional[torch.Tensor] = None,
                 frontier: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> None:
        self.energy, self.dq, self.fills = energy, dq, fills
        self._dP, self._frontier = dP, frontier

    @property
    def dP(self) -> torch.Tensor:
        if self._dP is None:
            U0 = self.fills[0].U
            dP = torch.zeros(U0.shape, dtype=U0.dtype, device=U0.device)
            for U_act, dfa in (self._frontier or []):
                dP = dP + (U_act * dfa.unsqueeze(-2)) @ U_act.transpose(-1, -2)
            self._dP = dP
        return self._dP


def two_fillings(H: torch.Tensor, n_s: Tuple[int, int], n_ref: Tuple[int, int],
                 sigma_s: float = SIGMA_S, eigh_device: str = "auto") -> TwoFillings:
    """Fill one `H` at the state's and the reference's per-spin counts: one `eigh`, ONE
    chemical-potential solve for the distinct counts (v5 W2 item 5: it was eight), a spin
    channel whose counts agree in state and reference skipped exactly (its two fills would
    cancel term by term; v5 W2 item 2), and -- under `no_grad`, the inference and SCF-loop
    path -- no density matrices at all: `dq_i = -sum_a df_a |Pi_i a|^2` over the active
    levels `|df_a| > FRONTIER_TOL`, `energy = sum_a df_a eps_a + R_S - R_ref` over all
    levels, `dP` formed from the active vectors on demand. With gradients enabled (training,
    the attached pass) the eager fills with the divided-difference backward are used, so
    the training gradient is unchanged."""
    with torch.no_grad():
        eps, U = eigh_for(H, eigh_device)
    spectrum = (eps, U)
    batch_shape = eps.shape[:-1]
    n_orb = eps.shape[-1]
    n_atoms = n_orb // ORBITALS

    def as_count(x):
        c = x if torch.is_tensor(x) else torch.tensor(float(x), dtype=eps.dtype, device=eps.device)
        c = c.to(dtype=eps.dtype, device=eps.device)
        return c.expand(batch_shape) if c.dim() == 0 else c
    counts = [as_count(n_s[0]), as_count(n_s[1]), as_count(n_ref[0]), as_count(n_ref[1])]   # s_up, s_dn, r_up, r_dn
    same = (bool(torch.equal(counts[0], counts[2])), bool(torch.equal(counts[1], counts[3])))
    order = [0, 1] + ([] if same[0] else [2]) + ([] if same[1] else [3])
    with torch.no_grad():
        mus = chemical_potential(eps.unsqueeze(0).expand(len(order), *eps.shape),
                                 torch.stack([counts[k] for k in order]), sigma_s)
    mu = {k: mus[i] for i, k in enumerate(order)}
    mu.setdefault(2, mu[0]); mu.setdefault(3, mu[1])

    if torch.is_grad_enabled():
        # Eager path: attached densities with the divided-difference backward.
        res = {k: fill(H, counts[k], sigma_s, spectrum, mu=mu[k]) for k in order}
        res.setdefault(2, res[0]); res.setdefault(3, res[1])
        energy = torch.zeros_like(res[0].F_band); dP = torch.zeros_like(res[0].P)
        for sigma in (0, 1):
            if not same[sigma]:
                energy = energy + (res[sigma].F_band - res[sigma + 2].F_band)
                dP = dP + (res[sigma].P - res[sigma + 2].P)
        return TwoFillings(energy=energy, dq=site_charge_difference(dP, n_atoms),
                           fills=(res[0], res[1], res[2], res[3]), dP=dP)

    # Frontier path (no gradients): occupations and entropies only; the active levels carry dq.
    fills: List[Optional[FillResult]] = [None, None, None, None]
    energy = torch.zeros(batch_shape, dtype=eps.dtype, device=eps.device)
    dq = torch.zeros(*batch_shape, n_atoms, dtype=eps.dtype, device=eps.device)
    frontier: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for sigma in (0, 1):
        f_s = occupations(eps, mu[sigma], sigma_s); R_s = generalised_entropy(eps, mu[sigma], sigma_s)
        fills[sigma] = FillResult(P=None, mu=mu[sigma], eps=eps, U=U, f=f_s, F_band=(f_s * eps).sum(-1) + R_s, entropy=R_s)
        if same[sigma]:
            fills[sigma + 2] = fills[sigma]
            continue
        f_r = occupations(eps, mu[sigma + 2], sigma_s); R_r = generalised_entropy(eps, mu[sigma + 2], sigma_s)
        fills[sigma + 2] = FillResult(P=None, mu=mu[sigma + 2], eps=eps, U=U, f=f_r, F_band=(f_r * eps).sum(-1) + R_r, entropy=R_r)
        df = f_s - f_r
        energy = energy + (df * eps).sum(-1) + (R_s - R_r)
        idx = torch.nonzero(df.abs().reshape(-1, n_orb).amax(0) > FRONTIER_TOL).reshape(-1)
        U_act = U[..., :, idx]; dfa = df[..., idx]
        w = (U_act * U_act).reshape(*batch_shape, n_atoms, ORBITALS, idx.numel()).sum(-2)     # |Pi_i a|^2, [..., N, n_act]
        dq = dq - torch.einsum("...ia,...a->...i", w, dfa)
        frontier.append((U_act, dfa))
    return TwoFillings(energy=energy, dq=dq, fills=tuple(fills), dP=None, frontier=frontier)


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
    max_backtrack: int = 6       # Newton, damping "backtrack": halvings of the step when the residual grows
    # v4.5: Levenberg-Marquardt damping (no trial re-diagonalisations) and the tangent
    # predictor between continuation stages. Solver engineering: the fixed points, the
    # tolerances and the gradients are unchanged; "backtrack" / predictor=False is the
    # pre-v4.5 solver, kept for the gate.
    damping: str = "newton"      # "newton": damped Newton, the step judged by the NEXT iteration's fill (v4.5,
                                 # no trial re-diagonalisation); "lm": Levenberg-Marquardt; "backtrack": pre-v4.5
    # The tangent predictor between continuation stages is implemented and selectable but
    # OFF by default. Measured on real frames (gate, 2026-09-07; tracker C9): it saves one
    # iteration in fifteen on 79-atom frames (15 -> 14) and costs on the hard 159-atom class
    # (40 -> 42 iterations at lambda 0.05; 18 -> 23 iterations, 47 fills against 19, at
    # lambda 0.5), on first visits only (one epoch in sixty plus the 5 % check). It does not
    # pay for itself on this data; the fixed points are the same (<= 2.4e-9).
    predictor: bool = False      # continuation: first-order (tangent) predictor of the next stage's dq
    predictor_trust: float = 1.0 # the predicted change is capped at this multiple of the previous stage's change
    lm_mu0: float = 1e-6         # LM: initial damping, relative to diag(A^T A) (Newton's step to 1e-6)
    lm_mu_min: float = 1e-12     # LM: floor of the damping (quadratic convergence near the fixed point)
    lm_up: float = 100.0         # LM: damping factor on a rejected step (1e-6 -> 1 in three rejections)
    lm_down: float = 0.1         # LM: damping factor on an accepted step
    lm_max_reject: int = 4       # consecutive rejections before the Anderson step on the history
    # v5 W2 item 7 (registered 2026-09-09): under `no_grad` (inference, the SCF loop) the batched
    # solver first iterates in float32 to `pre_tol_q`, then continues in float64 from that
    # iterate to the registered tolerances -- the fixed point, judged in float64, is unchanged;
    # a float32 `eigh` costs a fifth of a float64 one on the A4000. Off under gradients.
    mixed_precision: bool = True
    pre_tol_q: float = 1e-5      # float32 stage: unmixed residual, max-norm
    pre_tol_E: float = 1e-5      # float32 stage: energy change, eV
    pre_tol_c: float = 1e-2      # float32 stage: commutator norm
    eigh_device: str = "auto"    # v5 W2 item 7: 'auto' (single float64 eigh <= 512 orbitals on the CPU), 'cpu', 'cuda'
    # v5 W2 closing item (registered): the inference tolerance on the unmixed residual,
    # separate from the gate tolerance `tol_q` = 1e-8; the model applies it when called with
    # training=False. The force-error bound |dF| <= ||Gamma||_2 tol_q,inf is recorded per frame.
    tol_q_inference: float = 1e-6


_TRACE = False   # debugging: print the per-graph solver's iterations


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
    n_fills: int = 0                # diagonalisations spent (iterations + rejected deferred steps)


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
    lm = opt.method == "newton" and opt.damping == "lm"
    deferred = opt.method == "newton" and opt.damping in ("lm", "newton")
    mu = float(opt.lm_mu0)
    alpha = 1.0         # damped Newton: the step fraction (halved on a rejection, back to 1 on an acceptance)
    rejects = 0
    pending = None      # deferred damping: the accepted iterate a step was taken from -- (dq, res, r_norm, r2, jac, energy)
    n_fills = 0
    context = torch.enable_grad() if unroll else torch.no_grad()
    with context:
        # `n_max` bounds the ACCEPTED iterates (as before v4.5, when every pass was one);
        # a deferred step's rejected fill costs a diagonalisation but not an iteration.
        for _pass in range(opt.n_max * (1 + max(opt.max_backtrack, opt.lm_max_reject))):
            if iterations >= opt.n_max:
                break
            n_fills += 1
            V = gamma @ dq + (W if W is not None else 0.0)
            H = H0 - site_potential_matrix(V)
            sol = two_fillings(H, n_s, n_ref, sigma_s)
            res = sol.dq - dq
            r_norm = float(res.detach().abs().max())
            # The norm that judges a deferred step: the max-norm for damped Newton (the
            # pre-v4.5 backtracking test, so the iterates are the same), the 2-norm for LM.
            r2 = float(res.detach().norm()) if lm else r_norm
            energy_k = float(sol.energy.detach() - 0.5 * dq.detach() @ gamma.detach() @ dq.detach())
            delta_energy = abs(energy_k - energy_prev) if energy_prev is not None else float("inf")
            if (r_norm < opt.tol_q and delta_energy < opt.tol_E
                    and _commutator_norm(H0, gamma, W, sol) < opt.tol_c):
                history.append(r_norm)
                iterations += 1
                converged = True
                break
            jac: Optional[torch.Tensor] = None
            if _TRACE:
                print(f"    scf it={iterations} fills={n_fills} r={r_norm:.3e} r2={r2:.3e} dE={delta_energy:.1e} mu={mu:.1e} alpha={alpha:.3f} rejects={rejects} "
                      f"{'REJECT' if (deferred and pending is not None and r2 >= pending[3]) else 'accept'}", flush=True)
            if deferred and pending is not None and r2 >= pending[3]:
                # Rejected (the residual's 2-norm grew): back to the accepted iterate and ITS
                # Jacobian, with more damping -- a halved step (damped Newton) or a larger
                # `mu` (LM). The fill just done is the only cost; no trial re-diagonalisation.
                dq, res, r_norm, r2, jac, energy_k = pending
                if lm:
                    mu *= opt.lm_up
                else:
                    alpha *= 0.5
                rejects += 1
            else:
                if deferred and pending is not None:
                    mu = max(mu * opt.lm_down, opt.lm_mu_min)
                    alpha = 1.0
                    rejects = 0
                iterations += 1
                energy_prev = energy_k
                history.append(r_norm)
                dq_hist.append(dq)
                res_hist.append(res)
                if len(dq_hist) > opt.history:
                    dq_hist.pop(0)
                    res_hist.pop(0)
            accepted = False
            max_reject = opt.lm_max_reject if lm else opt.max_backtrack
            if opt.method == "newton" and not (deferred and rejects >= max_reject):
                if jac is None:
                    jac = hole_response(H, n_s, n_ref, sigma_s, spectrum=(sol.fills[0].eps, sol.fills[0].U), mus=tuple(f.mu for f in sol.fills)) @ gamma
                if deferred:
                    # v4.5: the step is taken and judged by the next iteration's residual --
                    # the fill it needs anyway. Damped Newton halves the step on a rejection
                    # (the pre-v4.5 backtracking sequence, without re-diagonalising the
                    # accepted point); LM raises the damping instead.
                    step = _lm_step(res.detach(), jac.detach(), mu) if lm else alpha * _newton_step(res.detach(), jac.detach())
                    pending = (dq, res, r_norm, r2, jac, energy_k)
                    dq = dq + step
                    newton_steps += 1
                    accepted = True
                else:
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
                # Anderson on the accepted history (the fallback after `lm_max_reject`
                # consecutive LM rejections, or a failed backtracking search); not judged.
                dq = _anderson(dq_hist, res_hist, opt.mixing)
                anderson_steps += 1
                pending = None
                rejects = 0
                mu = float(opt.lm_mu0)
                alpha = 1.0
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
                            spectrum=(sol1.fills[0].eps, sol1.fills[0].U), mus=tuple(f.mu for f in sol1.fills)) @ gamma.detach()
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
    P_s = sol.fills[0].density().detach()
    H_next = (H0 - site_potential_matrix(gamma @ sol.dq.detach() + (W if W is not None else 0.0))).detach()
    commutator = float((H_next @ P_s - P_s @ H_next).norm())
    tail = [h for h in history[-4:] if h > 0]
    rho = float(tail[-1] / tail[-2]) if len(tail) >= 2 else 0.0
    return ScfResult(energy=energy, energy_primary=energy_primary, dP=sol.dP, dq=sol.dq,
                     V=V.detach(), iterations=iterations, converged=converged,
                     residual=residual, delta_energy=delta_energy, commutator=commutator,
                     rho=rho, history=history, fills=sol.fills,
                     newton_steps=newton_steps, anderson_steps=anderson_steps, n_fills=n_fills)


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
                  spectrum: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  mus: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
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
        for sigma, (n_state, n_reference) in enumerate(((n_s[0], n_ref[0]), (n_s[1], n_ref[1]))):
            if n_state == n_reference:
                continue
            terms = []
            mu_of = {}
            for n_el, sign in ((n_state, 1.0), (n_reference, -1.0)):
                # v5 W2 item 5: the fills' chemical potentials are reused when given.
                mu = (mus[sigma if sign > 0 else sigma + 2] if mus is not None else chemical_potential(eps, float(n_el), sigma_s))
                mu_of[sign] = mu
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
                x_mid = (mid - mu_of[sign]) / sigma_s
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


def _lm_step(res: torch.Tensor, jac: torch.Tensor, mu) -> torch.Tensor:
    """Levenberg-Marquardt step for `R(dq) = dq_new(dq) - dq`, `A = dR/ddq = -(I - J)`:
    `(A^T A + mu diag(A^T A)) delta = A^T res` (Marquardt scaling) -- Newton's step at
    `mu -> 0`, a short step along `A^T res` at large `mu`. Batched over leading dims."""
    n = res.shape[-1]
    A = torch.eye(n, dtype=res.dtype, device=res.device) - jac
    AtA = A.transpose(-1, -2) @ A
    D = torch.diagonal(AtA, dim1=-2, dim2=-1)
    mu_t = torch.as_tensor(mu, dtype=res.dtype, device=res.device)
    if mu_t.dim() > 0:
        mu_t = mu_t.unsqueeze(-1)
    rhs = (A.transpose(-1, -2) @ res.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(AtA + torch.diag_embed(mu_t * D), rhs.unsqueeze(-1)).squeeze(-1)


def _commutator_norm(H0: torch.Tensor, gamma: torch.Tensor, W: Optional[torch.Tensor],
                     sol: TwoFillings) -> float:
    """`|[H(dq_new), P_S]|` (Frobenius): the state's majority density against the
    Hamiltonian its own charges produce -- the plan's `tol_c` test."""
    P_s = sol.fills[0].density().detach()
    H_next = (H0 - site_potential_matrix(gamma @ sol.dq.detach() + (W if W is not None else 0.0))).detach()
    return float((H_next @ P_s - P_s @ H_next).norm())


def _commutator_norm_batched(H0: torch.Tensor, gamma: torch.Tensor, W: Optional[torch.Tensor],
                             sol: TwoFillings) -> torch.Tensor:
    P_s = sol.fills[0].density().detach()
    V = torch.einsum("bij,bj->bi", gamma, sol.dq.detach()) + (W if W is not None else 0.0)
    H_next = (H0 - torch.diag_embed(V.repeat_interleave(ORBITALS, dim=-1))).detach()
    return (H_next @ P_s - P_s @ H_next).flatten(1).norm(dim=-1)


def tangent(H0: torch.Tensor, gamma: torch.Tensor, W: Optional[torch.Tensor], dq: torch.Tensor,
            n_s: Tuple[int, int], n_ref: Tuple[int, int], sigma_s: float, s: float,
            spectrum: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """`d dq*/ds` at the fixed point of the continuation stage `s` (`V = s (Gamma dq + W)`),
    by the implicit function theorem on `R(dq, s) = dq_new(s (Gamma dq + W)) - dq = 0`:
    `d dq/ds = (I - s M Gamma)^{-1} M (Gamma dq + W)`, `M` the hole response at that
    stage's Hamiltonian (its spectrum is known: no eigendecomposition). The v4.5 tangent
    predictor starts the next stage at `dq + (s' - s) d dq/ds`."""
    with torch.no_grad():
        V_full = gamma @ dq + (W if W is not None else 0.0)
        H = H0 - site_potential_matrix(s * V_full)
        M = hole_response(H, n_s, n_ref, sigma_s, spectrum=spectrum)
        n = dq.shape[-1]
        A = torch.eye(n, dtype=dq.dtype, device=dq.device) - s * (M @ gamma)
        return torch.linalg.solve(A, M @ V_full)


def continuation_solve(H0: torch.Tensor, gamma: torch.Tensor, n_s: Tuple[int, int],
                       n_ref: Tuple[int, int], sigma_s: float = SIGMA_S,
                       W: Optional[torch.Tensor] = None, options: Optional[ScfOptions] = None,
                       unroll: bool = False, implicit: bool = False) -> ScfResult:
    """D11 / v4.2 (C5): the production per-frame solve with Phi on starts from the Phi = 0
    two-fillings solution and ramps `Gamma` and `W` together from zero in
    `continuation_steps` warm-started solves; the last (full coupling) is the solution and
    the only one carrying the training gradient. Its `history` and `iterations` are the
    totals over the ramp."""
    opt = options or ScfOptions()
    if int(opt.continuation_steps) <= 0:
        # No continuation registered: the zero start (root-rule initialisation i).
        return solve_dscc(H0, gamma, n_s, n_ref, sigma_s, W, None, opt, unroll=unroll, implicit=implicit)
    steps = int(opt.continuation_steps)
    sol0 = two_fillings(H0, n_s, n_ref, sigma_s)
    dq = sol0.dq.detach()
    spectrum = (sol0.fills[0].eps.detach(), sol0.fills[0].U.detach())
    s_prev = 0.0
    total_iterations, total_fills, history = 0, 0, []
    result: Optional[ScfResult] = None
    dq_before = None          # the previous stage's start, for the trust cap
    for k in range(1, steps + 1):
        frac = k / steps
        last = k == steps
        start = dq
        if opt.predictor and dq_before is not None:
            # v4.5 tangent predictor: first order in the stage length from the previous
            # stage's fixed point (its spectrum is in hand; no eigendecomposition). The
            # predicted change is capped at `predictor_trust` times the previous stage's
            # actual change: at a near-degenerate frontier the response, and with it the
            # tangent, is large, and an overshot start costs more than it saves.
            delta = (frac - s_prev) * tangent(H0.detach(), gamma.detach(), None if W is None else W.detach(),
                                              dq, n_s, n_ref, sigma_s, s_prev, spectrum)
            cap = opt.predictor_trust * float((dq - dq_before).abs().max())
            size = float(delta.abs().max())
            if size > cap:
                delta = delta * (cap / size)
            start = dq + delta
        dq_before = start
        result = solve_dscc(H0, frac * gamma, n_s, n_ref, sigma_s,
                            None if W is None else frac * W, start, opt,
                            unroll=unroll and last, implicit=implicit and last)
        dq = result.dq.detach()
        spectrum = (result.fills[0].eps.detach(), result.fills[0].U.detach())
        s_prev = frac
        total_iterations += result.iterations
        total_fills += result.n_fills
        history.extend(result.history)
    assert result is not None
    result.iterations = total_iterations
    result.n_fills = total_fills
    result.history = history
    return result



# ------------------------------------------------------------------ batched solve (equal sizes)

@dataclass
class BatchedScfResult:
    energy: torch.Tensor            # [B], attached (band form)
    energy_primary: torch.Tensor    # [B], detached
    dP: torch.Tensor                # [B, 4n, 4n], attached
    dq: torch.Tensor                # [B, N], attached
    V: torch.Tensor                 # [B, N], detached
    iterations: List[int]
    converged: List[bool]
    residual: List[float]
    commutator: List[float]
    rho: List[float]
    fills: Tuple[FillResult, FillResult, FillResult, FillResult]
    n_fills: List[int] = field(default_factory=list)   # diagonalisations spent per graph
    pre_fills: List[int] = field(default_factory=list)  # v5 W2 item 7: float32 pre-stage fills per graph


def _batched_fillings(H0: torch.Tensor, gamma: torch.Tensor, W: Optional[torch.Tensor],
                      dq: torch.Tensor, n_s, n_ref, sigma_s: float, eigh_device: str = "auto") -> TwoFillings:
    V = torch.einsum("bij,bj->bi", gamma, dq) + (W if W is not None else 0.0)
    H = H0 - torch.diag_embed(V.repeat_interleave(ORBITALS, dim=-1))
    return two_fillings(H, n_s, n_ref, sigma_s, eigh_device=eigh_device)


def _batched_fixed_point(H0: torch.Tensor, gamma: torch.Tensor, W: Optional[torch.Tensor], dq: torch.Tensor,
                         n_s, n_ref, sigma_s: float, implicit: bool, iterations: torch.Tensor,
                         converged: torch.Tensor, hist: List[List[float]],
                         n_fills: Optional[torch.Tensor] = None, eigh_device: str = "auto") -> BatchedScfResult:
    """The attached pass at the (per graph) fixed point `dq`, the implicit-function hook
    when asked, and the diagnostics -- shared by the LM and the backtracking loops."""
    B, n_atoms = dq.shape
    if implicit:
        dq_leaf = dq.detach().clone().requires_grad_(True)
        sol1 = _batched_fillings(H0, gamma, W, dq_leaf, n_s, n_ref, sigma_s, eigh_device=eigh_device)
        A_T = torch.zeros(B, n_atoms, n_atoms, dtype=H0.dtype, device=H0.device)
        for b in range(B):
            V_b = (gamma[b] @ dq_leaf[b] + (W[b] if W is not None else 0.0)).detach()
            H_b = (H0[b] - site_potential_matrix(V_b)).detach()
            jac = hole_response(H_b, (int(n_s[0][b]), int(n_s[1][b])), (int(n_ref[0][b]), int(n_ref[1][b])),
                                sigma_s, spectrum=(sol1.fills[0].eps[b], sol1.fills[0].U[b]), mus=tuple(f.mu[b] for f in sol1.fills)) @ gamma[b].detach()
            A_T[b] = (torch.eye(n_atoms, dtype=H0.dtype, device=H0.device) - jac).transpose(0, 1)
        dq_star = sol1.dq
        dq_star.register_hook(lambda c: torch.linalg.solve(A_T, c.unsqueeze(-1)).squeeze(-1))
    else:
        dq_star = dq.detach()
    V = torch.einsum("bij,bj->bi", gamma, dq_star) + (W if W is not None else 0.0)
    H = H0 - torch.diag_embed(V.repeat_interleave(ORBITALS, dim=-1))
    sol = two_fillings(H, n_s, n_ref, sigma_s, eigh_device=eigh_device)
    residual = (sol.dq.detach() - dq_star.detach()).abs().amax(dim=-1)
    energy = sol.energy - 0.5 * torch.einsum("bi,bij,bj->b", dq_star, gamma, dq_star)
    primary = []
    commutator = []
    dP_all = sol.dP; P_s_all = sol.fills[0].density().detach()
    for b in range(B):
        one = TwoFillings(energy=sol.energy[b], dP=dP_all[b], dq=sol.dq[b],
                          fills=tuple(FillResult(P=None, mu=f.mu[b], eps=f.eps[b], U=f.U[b], f=f.f[b],
                                                 F_band=f.F_band[b], entropy=f.entropy[b]) for f in sol.fills))
        primary.append(primary_functional(H0[b], gamma[b], one, None if W is None else W[b]))
        P_s = P_s_all[b]
        H_next = (H0[b] - site_potential_matrix(gamma[b] @ sol.dq[b].detach() + (W[b] if W is not None else 0.0))).detach()
        commutator.append(float((H_next @ P_s - P_s @ H_next).norm()))
    rho = []
    for b in range(B):
        # Each graph's history is its own accepted residuals up to ITS convergence.
        tail = [x for x in hist[b] if x > 0][-4:]
        rho.append(float(tail[-1] / tail[-2]) if len(tail) >= 2 else 0.0)
    return BatchedScfResult(energy=energy, energy_primary=torch.stack(primary), dP=sol.dP, dq=sol.dq,
                            V=V.detach(), iterations=iterations.tolist(), converged=converged.tolist(),
                            residual=residual.tolist(), commutator=commutator, rho=rho, fills=sol.fills,
                            n_fills=(iterations if n_fills is None else n_fills).tolist())


def solve_dscc_batched(H0: torch.Tensor, gamma: torch.Tensor, n_s: Tuple[torch.Tensor, torch.Tensor],
                       n_ref: Tuple[torch.Tensor, torch.Tensor], sigma_s: float = SIGMA_S,
                       W: Optional[torch.Tensor] = None, dq0: Optional[torch.Tensor] = None,
                       options: Optional[ScfOptions] = None, implicit: bool = False,
                       mixed: Optional[bool] = None) -> BatchedScfResult:
    """`solve_dscc` for a batch of EQUAL-SIZED graphs at once: `H0 [B, 4n, 4n]`, `gamma
    [B, N, N]`, per-graph counts. One batched `eigh` and fill per iteration; the damped
    Newton step (v4.5: judged by the next iteration's residual, no trial
    re-diagonalisation; LM when selected), its damping, the Anderson fallback on the graph's own
    history and the convergence test (`tol_q`, `tol_E`, `tol_c`) are per graph (masked) --
    the reference solver's algorithm, graph by graph. `implicit` attaches the
    implicit-function derivative at the fixed point (as `solve_dscc`)."""
    opt = options or ScfOptions()
    if opt.method != "newton":
        raise ValueError("the batched solver is the Newton solver")
    if opt.damping == "backtrack":
        return _solve_dscc_batched_backtrack(H0, gamma, n_s, n_ref, sigma_s, W, dq0, opt, implicit)
    pre_fills = None
    # The pre-stage runs whenever no implicit derivative is attached (training passes
    # `implicit=True`); the loop itself is always under `no_grad`, and the outer grad mode at
    # inference only serves the attached final pass.
    use_mixed = (getattr(opt, "mixed_precision", False) if mixed is None else bool(mixed))
    # No pre-stage from a warm start (`dq0`): the float64 solve needs two fills from a good
    # start anyway, so the float32 stage would only add diagonalisations.
    if use_mixed and H0.dtype == torch.float64 and not implicit and dq0 is None:
        import dataclasses
        pre = dataclasses.replace(opt, mixed_precision=False, tol_q=opt.pre_tol_q, tol_E=opt.pre_tol_E, tol_c=opt.pre_tol_c)
        with torch.no_grad():                      # a starting point only: no attached pass in float32
            r32 = solve_dscc_batched(H0.detach().float(), gamma.detach().float(), n_s, n_ref, sigma_s,
                                     None if W is None else W.detach().float(), None if dq0 is None else dq0.detach().float(),
                                     pre, implicit=False)
        dq0 = r32.dq.detach().double()
        pre_fills = list(r32.n_fills)
    B, dim = H0.shape[0], H0.shape[-1]
    n_atoms = dim // ORBITALS
    lm = opt.damping == "lm"
    max_reject = opt.lm_max_reject if lm else opt.max_backtrack
    dq = (torch.zeros(B, n_atoms, dtype=H0.dtype, device=H0.device) if dq0 is None else dq0.to(H0.dtype))
    converged = torch.zeros(B, dtype=torch.bool, device=H0.device)
    iterations = torch.zeros(B, dtype=torch.long, device=H0.device)
    energy_prev = torch.full((B,), float("nan"), dtype=H0.dtype, device=H0.device)
    hist: List[List[float]] = [[] for _ in range(B)]
    alpha = torch.ones(B, dtype=H0.dtype, device=H0.device)
    eye = torch.eye(n_atoms, dtype=H0.dtype, device=H0.device)
    n_fills = torch.zeros(B, dtype=torch.long, device=H0.device)
    dq_hist: List[List[torch.Tensor]] = [[] for _ in range(B)]
    res_hist: List[List[torch.Tensor]] = [[] for _ in range(B)]
    counts = [((int(n_s[0][b]), int(n_s[1][b])), (int(n_ref[0][b]), int(n_ref[1][b]))) for b in range(B)]
    mu = torch.full((B,), float(opt.lm_mu0), dtype=H0.dtype, device=H0.device)
    rejects = torch.zeros(B, dtype=torch.long, device=H0.device)
    jac = torch.zeros(B, n_atoms, n_atoms, dtype=H0.dtype, device=H0.device)
    pending = torch.zeros(B, dtype=torch.bool, device=H0.device)
    dq_prev, res_prev = dq.clone(), torch.zeros_like(dq)
    r_prev = torch.full((B,), float("inf"), dtype=H0.dtype, device=H0.device)
    r2_prev = torch.full((B,), float("inf"), dtype=H0.dtype, device=H0.device)
    with torch.no_grad():
        # `n_max` bounds each graph's ACCEPTED iterates (the pre-v4.5 meaning); rejected
        # deferred steps cost fills, not iterations. A graph at the cap is left unconverged.
        for _pass in range(opt.n_max * (1 + max_reject)):
            live = (~converged) & (iterations < opt.n_max)
            if not bool(live.any()):
                break
            n_fills = n_fills + live.long()
            sol = _batched_fillings(H0, gamma, W, dq, n_s, n_ref, sigma_s, eigh_device=getattr(opt, 'eigh_device', 'auto'))
            res = sol.dq - dq                                                     # [B, N]
            r_norm = res.abs().amax(dim=-1)
            r2 = res.norm(dim=-1) if lm else r_norm                               # judges the deferred steps
            energy_k = sol.energy - 0.5 * torch.einsum("bi,bij,bj->b", dq, gamma, dq)
            d_energy = (energy_k - energy_prev).abs()                             # nan on the first pass: not converged
            cand = live & (r_norm < opt.tol_q) & (d_energy < opt.tol_E)
            newly = cand & (_commutator_norm_batched(H0, gamma, W, sol) < opt.tol_c) if bool(cand.any()) else cand
            for b in torch.nonzero(newly).reshape(-1).tolist():
                hist[b].append(float(r_norm[b]))
            iterations = torch.where(newly, iterations + 1, iterations)
            converged = converged | newly
            active = live & ~newly
            if not bool(active.any()):
                continue
            # LM: judge the pending steps by this residual. Rejected graphs return to their
            # accepted iterate (and its Jacobian) with more damping; accepted ones relax it.
            worse = pending & active & (r2 >= r2_prev)
            if _TRACE:
                print(f"    scfb it={iterations.tolist()} fills={n_fills.tolist()} r={[f'{x:.2e}' for x in r_norm.tolist()]} mu={[f'{x:.0e}' for x in mu.tolist()]} "
                      f"rej={rejects.tolist()} worse={worse.tolist()} dE={[f'{x:.0e}' for x in d_energy.tolist()]}", flush=True)
            if bool(worse.any()):
                dq = torch.where(worse.unsqueeze(-1), dq_prev, dq)
                res = torch.where(worse.unsqueeze(-1), res_prev, res)
                r_norm = torch.where(worse, r_prev, r_norm)
                r2 = torch.where(worse, r2_prev, r2)
                mu = torch.where(worse, mu * opt.lm_up, mu)
                alpha = torch.where(worse, 0.5 * alpha, alpha)
                rejects = torch.where(worse, rejects + 1, rejects)
            accepted = pending & active & ~worse
            mu = torch.where(accepted, (mu * opt.lm_down).clamp_min(opt.lm_mu_min), mu)
            alpha = torch.where(accepted, torch.ones_like(alpha), alpha)
            rejects = torch.where(accepted, torch.zeros_like(rejects), rejects)
            fresh = active & ~worse                                               # accepted or first pass
            iterations = torch.where(fresh, iterations + 1, iterations)
            energy_prev = torch.where(fresh, energy_k, energy_prev)
            eps, U = sol.fills[0].eps, sol.fills[0].U
            for b in torch.nonzero(fresh).reshape(-1).tolist():
                hist[b].append(float(r_norm[b]))
                dq_hist[b].append(dq[b])
                res_hist[b].append(res[b])
                if len(dq_hist[b]) > opt.history:
                    dq_hist[b].pop(0)
                    res_hist[b].pop(0)
                V_b = gamma[b] @ dq[b] + (W[b] if W is not None else 0.0)
                H_b = H0[b] - site_potential_matrix(V_b)
                jac[b] = hole_response(H_b, counts[b][0], counts[b][1], sigma_s, spectrum=(eps[b], U[b]), mus=tuple(f.mu[b] for f in sol.fills)) @ gamma[b]
            if lm:
                step = _lm_step(res, jac, mu)
            else:
                step = alpha.unsqueeze(-1) * torch.linalg.solve(eye - jac, res.unsqueeze(-1)).squeeze(-1)
            new_dq = dq + step
            fallback = active & (rejects >= max_reject)
            for b in torch.nonzero(fallback).reshape(-1).tolist():
                new_dq[b] = _anderson(dq_hist[b], res_hist[b], opt.mixing)
            mu = torch.where(fallback, torch.full_like(mu, float(opt.lm_mu0)), mu)
            alpha = torch.where(fallback, torch.ones_like(alpha), alpha)
            rejects = torch.where(fallback, torch.zeros_like(rejects), rejects)
            dq_prev, res_prev, r_prev, r2_prev = dq, res, r_norm, r2
            pending = active & ~fallback                                          # Anderson steps are not judged
            dq = torch.where(active.unsqueeze(-1), new_dq, dq)
    result = _batched_fixed_point(H0, gamma, W, dq, n_s, n_ref, sigma_s, implicit, iterations, converged, hist, n_fills, eigh_device=getattr(opt, 'eigh_device', 'auto'))
    if pre_fills is not None:                      # diagnostics: float32 fills spent before the float64 stage
        result.pre_fills = pre_fills
        result.n_fills = [int(a) + int(b) for a, b in zip(result.n_fills, pre_fills)]
    return result


def _solve_dscc_batched_backtrack(H0: torch.Tensor, gamma: torch.Tensor, n_s: Tuple[torch.Tensor, torch.Tensor],
                       n_ref: Tuple[torch.Tensor, torch.Tensor], sigma_s: float = SIGMA_S,
                       W: Optional[torch.Tensor] = None, dq0: Optional[torch.Tensor] = None,
                       options: Optional[ScfOptions] = None, implicit: bool = False) -> BatchedScfResult:
    """The pre-v4.5 batched loop (`damping = "backtrack"`; kept for the gate): one batched
    `eigh` and fill per iteration; the damped Newton step, its backtracking, the Anderson fallback on the graph's own history and
    the convergence test are per graph (masked) -- the reference solver's algorithm, graph
    by graph (an earlier version fell back to a plain damped step, which stalled on a
    159-atom frame the reference solver converges in 40 iterations). `implicit` attaches
    the fixed point's parameter derivative per graph."""
    opt = options or ScfOptions()
    B, dim = H0.shape[0], H0.shape[-1]
    n_atoms = dim // ORBITALS
    dq = (torch.zeros(B, n_atoms, dtype=H0.dtype, device=H0.device) if dq0 is None else dq0.to(H0.dtype))
    converged = torch.zeros(B, dtype=torch.bool, device=H0.device)
    iterations = torch.zeros(B, dtype=torch.long, device=H0.device)
    energy_prev = torch.full((B,), float("nan"), dtype=H0.dtype, device=H0.device)
    history: List[torch.Tensor] = []
    dq_hist: List[List[torch.Tensor]] = [[] for _ in range(B)]
    res_hist: List[List[torch.Tensor]] = [[] for _ in range(B)]
    with torch.no_grad():
        for k in range(opt.n_max):
            sol = _batched_fillings(H0, gamma, W, dq, n_s, n_ref, sigma_s, eigh_device=getattr(opt, 'eigh_device', 'auto'))
            res = sol.dq - dq                                                     # [B, N]
            r_norm = res.abs().amax(dim=-1)
            history.append(r_norm.clone())
            energy_k = sol.energy - 0.5 * torch.einsum("bi,bij,bj->b", dq, gamma, dq)
            d_energy = (energy_k - energy_prev).abs()
            energy_prev = energy_k
            newly = (~converged) & (r_norm < opt.tol_q) & (d_energy < opt.tol_E)
            iterations = torch.where(~converged, torch.full_like(iterations, k + 1), iterations)
            converged = converged | newly
            if bool(converged.all()):
                break
            active = ~converged
            # Newton step per active graph (its history appended first, as in `solve_dscc`).
            step = torch.zeros_like(dq)
            for b in torch.nonzero(active).reshape(-1).tolist():
                dq_hist[b].append(dq[b])
                res_hist[b].append(res[b])
                if len(dq_hist[b]) > opt.history:
                    dq_hist[b].pop(0)
                    res_hist[b].pop(0)
                V_b = gamma[b] @ dq[b] + (W[b] if W is not None else 0.0)
                H_b = H0[b] - site_potential_matrix(V_b)
                jac = hole_response(H_b, (int(n_s[0][b]), int(n_s[1][b])), (int(n_ref[0][b]), int(n_ref[1][b])),
                                    sigma_s, spectrum=(sol.fills[0].eps[b], sol.fills[0].U[b]), mus=tuple(f.mu[b] for f in sol.fills)) @ gamma[b]
                step[b] = _newton_step(res[b], jac)
            scale = torch.ones(B, dtype=H0.dtype, device=H0.device)
            accepted = torch.zeros(B, dtype=torch.bool, device=H0.device)
            for _ in range(opt.max_backtrack):
                trial = dq + scale.unsqueeze(-1) * step
                r_t = (_batched_fillings(H0, gamma, W, trial, n_s, n_ref, sigma_s, eigh_device=getattr(opt, 'eigh_device', 'auto')).dq - trial).abs().amax(dim=-1)
                ok = active & ~accepted & (r_t < r_norm)
                accepted = accepted | ok
                scale = torch.where(active & ~accepted, scale * 0.5, scale)
                if bool((accepted | ~active).all()):
                    break
            newton_dq = dq + scale.unsqueeze(-1) * step
            fallback_dq = dq.clone()
            for b in torch.nonzero(active & ~accepted).reshape(-1).tolist():
                fallback_dq[b] = _anderson(dq_hist[b], res_hist[b], opt.mixing)
            dq = torch.where((active & accepted).unsqueeze(-1), newton_dq,
                             torch.where(active.unsqueeze(-1), fallback_dq, dq))
    hist_lists = [[float(x) for x in torch.stack(history, dim=0)[:int(iterations[b]), b].tolist()] for b in range(B)]
    return _batched_fixed_point(H0, gamma, W, dq, n_s, n_ref, sigma_s, implicit, iterations, converged, hist_lists)




def continuation_solve_batched(H0: torch.Tensor, gamma: torch.Tensor, n_s, n_ref, sigma_s: float = SIGMA_S,
                               W: Optional[torch.Tensor] = None, options: Optional[ScfOptions] = None,
                               implicit: bool = False, mixed: Optional[bool] = None) -> BatchedScfResult:
    """D11 for a batch: the continuation from Phi = 0 with the last solve carrying the
    gradient; iteration counts summed over the ramp. `mixed` overrides the options' float32
    pre-stage (the model passes `not training`, so the training path is unchanged)."""
    opt = options or ScfOptions()
    if int(opt.continuation_steps) <= 0:
        return solve_dscc_batched(H0, gamma, n_s, n_ref, sigma_s, W, None, opt, implicit=implicit, mixed=mixed)
    steps = int(opt.continuation_steps)
    B = H0.shape[0]
    mixed = (bool(getattr(opt, "mixed_precision", False)) if mixed is None else bool(mixed)) and H0.dtype == torch.float64 and not implicit
    if mixed:
        import dataclasses
        pre = dataclasses.replace(opt, mixed_precision=False, tol_q=opt.pre_tol_q, tol_E=opt.pre_tol_E, tol_c=opt.pre_tol_c)
        H0_32, gamma_32 = H0.detach().float(), gamma.detach().float()
        W_32 = None if W is None else W.detach().float()
    sol0 = two_fillings(H0, n_s, n_ref, sigma_s)
    dq = sol0.dq.detach()
    eps, U = sol0.fills[0].eps.detach(), sol0.fills[0].U.detach()
    s_prev = 0.0
    total = None
    fills_total = None
    result = None
    dq_before = None
    for k in range(1, steps + 1):
        frac = k / steps
        start = dq
        if opt.predictor and dq_before is not None:
            # v4.5 tangent predictor, per graph from the previous stage's spectrum, capped
            # at `predictor_trust` times the graph's previous change (see `continuation_solve`).
            start = dq.clone()
            for b in range(B):
                delta = (frac - s_prev) * tangent(
                    H0[b].detach(), gamma[b].detach(), None if W is None else W[b].detach(), dq[b],
                    (int(n_s[0][b]), int(n_s[1][b])), (int(n_ref[0][b]), int(n_ref[1][b])), sigma_s, s_prev, (eps[b], U[b]))
                cap = opt.predictor_trust * float((dq[b] - dq_before[b]).abs().max())
                size = float(delta.abs().max())
                if size > cap:
                    delta = delta * (cap / size)
                start[b] = dq[b] + delta
        dq_before = start
        if mixed and k < steps:
            # v5 W2 item 7: the ramp's intermediate stages only seed the next stage -- solved in
            # float32 to the pre-tolerances, under no_grad; the last stage runs the float64
            # solver (with its own float32 pre-stage) from the float32 seed.
            with torch.no_grad():
                result = solve_dscc_batched(H0_32, frac * gamma_32, n_s, n_ref, sigma_s, None if W is None else frac * W_32,
                                            start.float(), pre, implicit=False)
            dq = result.dq.detach().double()
        else:
            # The last stage of a mixed ramp starts from the float32 seed, which is a warm
            # start for the float64 solver (two or three fills); no further pre-stage.
            result = solve_dscc_batched(H0, frac * gamma, n_s, n_ref, sigma_s, None if W is None else frac * W,
                                        start, opt, implicit=implicit and k == steps, mixed=False if mixed else mixed)
            dq = result.dq.detach()
        eps, U = result.fills[0].eps.detach(), result.fills[0].U.detach()
        s_prev = frac
        total = result.iterations if total is None else [a + b for a, b in zip(total, result.iterations)]
        fills_total = result.n_fills if fills_total is None else [a + b for a, b in zip(fills_total, result.n_fills)]
    assert result is not None
    result.iterations = total
    result.n_fills = fills_total
    return result
