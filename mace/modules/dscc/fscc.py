"""Plan section 2.9 (Arm 4): the F-SCC comparators, offline only.

Same `H0`, same `fill`. Two INDEPENDENT self-consistent-charge solves with ABSOLUTE site
charges `Dq_X_i = n0[Z_i] - sum_sigma Tr(Pi_i P_X_sigma)`, `X in {S, ref}`, each
self-consistent in its own potential `V_X = Gamma_F Dq_X`;

    E_SCC(X) = sum_sigma [Tr(P_X H0) + R(P_X)] + 0.5 Dq_X^T Gamma_F Dq_X,   head = E_SCC(S) - E_SCC(ref).

Host coupling is intrinsic (absolute charges); no Route B. float64. Two comparators:
matched-kernel (`Gamma_F` = the selected D-SCC kernel regime and bounds, `lambda_dir`,
`U_eff` learned under the same protocol) and full-kernel (`Gamma_F = E_PBC / eps_inf +
diag(U_eff)`). Reported per comparator: the trace norm of `P_S - P_ref` beyond its first
`|Q|` singular values, and `N_eff` / `R_eff` of the carrier.

The solver is the D-SCC Newton on the exact response Jacobian of a SINGLE fill (the
occupied electrons of the charged state respond to the deficit and screen it: positive
`Gamma` delocalises here, section 2.10), with the same convergence rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from mace.modules.dscc.fill import SIGMA_S, FillResult, chemical_potential, fill
from mace.modules.dscc.scf import (ORBITALS, ScfOptions, _anderson, _newton_step, _SQRT_PI,
                                   site_potential_matrix)


def absolute_charges(P_total: torch.Tensor, n0: torch.Tensor) -> torch.Tensor:
    """`Dq_i = n0_i - Tr(Pi_i P)` for the spin-summed density `P`."""
    return n0 - torch.diagonal(P_total, dim1=-2, dim2=-1).reshape(*P_total.shape[:-2], n0.shape[-1], ORBITALS).sum(-1)


def single_fill_response(H: torch.Tensor, n_up: int, n_dn: int, sigma_s: float = SIGMA_S,
                         spectrum: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                         active_tol: float = 1e-12) -> torch.Tensor:
    """`M_ij = d Dq_i / dV_j` at `H = H0 - diag(V)` for the spin-summed absolute charge:
    the site-resolved static susceptibility (Daleckii-Krein, fixed-N correction), summed
    over the two spin fills. `Dq` DEcreases where the potential is lowered (electrons flow
    in), so `M` is negative semi-definite on the neutral direction."""
    with torch.no_grad():
        eps, U = torch.linalg.eigh(H) if spectrum is None else spectrum
        n_orb = H.shape[-1]
        n_atoms = n_orb // ORBITALS
        M = torch.zeros(n_atoms, n_atoms, dtype=H.dtype, device=H.device)
        diag_A = torch.einsum("ma,ma->ma", U, U).reshape(n_atoms, ORBITALS, n_orb).sum(1)   # [N, n]
        for n_el in (n_up, n_dn):
            mu = chemical_potential(eps, float(n_el), sigma_s)
            x = (eps - mu) / sigma_s
            f = 0.5 * torch.erfc(x)
            fp = -torch.exp(-x * x) / (sigma_s * _SQRT_PI)
            # Active levels: those with f' or a partner occupation difference that matters
            # -- every pair (a, b) with f_a != f_b contributes; keep pairs involving levels
            # within the smearing window OR one occupied and one empty (the bulk of the
            # response). Use all pairs where |f_a - f_b| > tol via the occupied/empty split.
            occ = torch.nonzero(f > active_tol).reshape(-1)
            emp = torch.nonzero(f < 1.0 - active_tol).reshape(-1)
            # Pairs (a in occ, b in emp) count twice (symmetric), pairs within the window
            # where both are fractional are included once each way through the same sets.
            A_occ = torch.einsum("ma,mb->mab", U[:, occ], U[:, emp]).reshape(n_atoms, ORBITALS, occ.numel(), emp.numel()).sum(1)
            d_eps = eps[occ].unsqueeze(-1) - eps[emp].unsqueeze(0)
            d_f = f[occ].unsqueeze(-1) - f[emp].unsqueeze(0)
            near = d_eps.abs() <= 1e-7
            mid = 0.5 * (eps[occ].unsqueeze(-1) + eps[emp].unsqueeze(0))
            L = torch.where(near, -torch.exp(-((mid - mu) / sigma_s) ** 2) / (sigma_s * _SQRT_PI),
                            d_f / torch.where(near, torch.ones_like(d_eps), d_eps))
            # Each unordered pair once: the (occ x emp) rectangle covers pairs with a in occ,
            # b in emp; pairs with both indices in occ∩emp (fractional) appear twice in the
            # rectangle (as (a,b) and (b,a)) -- weight them 1/2.
            both = torch.zeros(occ.numel(), emp.numel(), dtype=H.dtype, device=H.device)
            frac_occ = (f[occ] < 1.0 - active_tol); frac_emp = (f[emp] > active_tol)
            both = torch.where(frac_occ.unsqueeze(-1) & frac_emp.unsqueeze(0), torch.full_like(both, 0.5), torch.ones_like(both))
            # dP/dV_j = -sum_pairs L A^j (U_a U_b^T + U_b U_a^T) -> Tr(Pi_i dP/dV_j) = -2 sum L A^i A^j (unordered)
            # Dq = n0 - Tr(Pi P)  ->  dDq_i/dV_j = +2 sum_pairs L_ab A^i_ab A^j_ab (weighted)
            contrib = 2.0 * torch.einsum("iab,ab,jab->ij", A_occ, both * L, A_occ)
            s_fp = fp.sum()
            if float(s_fp.abs()) > 1e-300:
                g = diag_A @ fp
                # fixed-N: dP gets -f'_a (sum_b f'_b Ghat_bb)/sum f' on the diagonal; on Dq that is
                # +(g_i g_j)/sum f' ... with the sign of the leading term.
                contrib = contrib - torch.outer(g, g) / s_fp
            M = M + contrib
        return M


@dataclass
class FsccResult:
    energy: torch.Tensor        # E_SCC(X), attached (envelope: P detached in the band form)
    P: torch.Tensor             # spin-summed density at the fixed point, attached
    dq: torch.Tensor            # absolute charges Dq_X at the fixed point, attached
    V: torch.Tensor
    iterations: int
    converged: bool
    residual: float
    fills: Tuple[FillResult, FillResult]


def solve_fscc(H0: torch.Tensor, gamma: torch.Tensor, n0: torch.Tensor, n_up: int, n_dn: int,
               sigma_s: float = SIGMA_S, dq0: Optional[torch.Tensor] = None,
               options: Optional[ScfOptions] = None) -> FsccResult:
    """One full SCC solve at fixed electron counts: `Dq -> Dq(H0 - diag(Gamma Dq))`."""
    opt = options or ScfOptions()
    n_atoms = n0.shape[0]
    dq = torch.zeros(n_atoms, dtype=H0.dtype, device=H0.device) if dq0 is None else dq0.to(H0.dtype)
    dq_hist: List[torch.Tensor] = []; res_hist: List[torch.Tensor] = []
    converged = False; iterations = 0
    with torch.no_grad():
        for k in range(opt.n_max):
            iterations = k + 1
            H = H0 - site_potential_matrix(gamma @ dq)
            eps, U = torch.linalg.eigh(H)
            P = fill(H, float(n_up), sigma_s, (eps, U)).P + fill(H, float(n_dn), sigma_s, (eps, U)).P
            dq_new = absolute_charges(P, n0)
            res = dq_new - dq
            r_norm = float(res.abs().max())
            if r_norm < opt.tol_q:
                converged = True
                break
            dq_hist.append(dq); res_hist.append(res)
            if len(dq_hist) > opt.history:
                dq_hist.pop(0); res_hist.pop(0)
            accepted = False
            if opt.method == "newton":
                jac = single_fill_response(H, n_up, n_dn, sigma_s, (eps, U)) @ gamma
                step = _newton_step(res, jac)
                scale = 1.0
                for _ in range(opt.max_backtrack):
                    trial = dq + scale * step
                    Ht = H0 - site_potential_matrix(gamma @ trial)
                    et, Ut = torch.linalg.eigh(Ht)
                    Pt = fill(Ht, float(n_up), sigma_s, (et, Ut)).P + fill(Ht, float(n_dn), sigma_s, (et, Ut)).P
                    if float((absolute_charges(Pt, n0) - trial).abs().max()) < r_norm:
                        accepted = True; break
                    scale *= 0.5
                if accepted:
                    dq = dq + scale * step
            if not accepted:
                dq = _anderson(dq_hist, res_hist, opt.mixing)
    # Attached pass at the fixed point.
    dq_star = dq.detach()
    V = gamma @ dq_star
    H = H0 - site_potential_matrix(V)
    with torch.no_grad():
        spectrum = torch.linalg.eigh(H)
    f_up = fill(H, float(n_up), sigma_s, spectrum); f_dn = fill(H, float(n_dn), sigma_s, spectrum)
    P = f_up.P + f_dn.P
    dq_out = absolute_charges(P, n0)
    residual = float((dq_out.detach() - dq_star).abs().max())
    # E_SCC = sum_sigma [Tr(P H0) + R] + 0.5 Dq^T Gamma Dq  (band form: F_band(H) + 0.5 Dq^T Gamma Dq
    # since Tr(P H) = Tr(P H0) - Dq_shift ... written directly with P detached in the linear term).
    energy = (P.detach() * H0).sum() + f_up.entropy + f_dn.entropy + 0.5 * dq_star @ gamma @ dq_star
    return FsccResult(energy=energy, P=P, dq=dq_out, V=V.detach(), iterations=iterations,
                      converged=converged, residual=residual, fills=(f_up, f_dn))


def fscc_head(H0: torch.Tensor, gamma: torch.Tensor, n0: torch.Tensor, n_s: Tuple[int, int],
              n_ref: Tuple[int, int], sigma_s: float = SIGMA_S, options: Optional[ScfOptions] = None):
    """`head = E_SCC(S) - E_SCC(ref)` from two independent solves; returns both results."""
    ref = solve_fscc(H0, gamma, n0, n_ref[0], n_ref[1], sigma_s, None, options)
    state = solve_fscc(H0, gamma, n0, n_s[0], n_s[1], sigma_s, ref.dq.detach(), options)
    return state.energy - ref.energy, state, ref


def excess_trace_norm(P_s: torch.Tensor, P_ref: torch.Tensor, q: int) -> float:
    """Diagnostic: the trace norm of `P_S - P_ref` beyond its first `|Q|` singular values."""
    sv = torch.linalg.svdvals((P_s - P_ref).detach())
    return float(sv[abs(int(q)):].sum())
