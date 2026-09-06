"""Addendum section 6.2, Stage 5: the stationary auxiliary functional.

    A_B[P; S] = sum_sigma [ Tr(P_sigma H_fix,sigma) + R_sm[P_sigma] ] + Phi_B[P]

subject to the exact per-spin trace constraints of `S`, with `Phi_B` the section-6.2 boundary
functional (`defect_image.boundary_functional` of the image-active density built from `P`)
and `R_sm` the occupation regulariser of the configured smearing -- defined so that
minimising `Tr(P H) + R_sm[P]` at fixed `Tr P = N` gives the implemented `F_band(H, N)`.

The stationary equations are

    H_B,sigma[P] = H_fix,sigma + V_B,sigma[P],   V_B,sigma = dPhi_B / dP_sigma,
    P_sigma = count_fill_sigma(H_B,sigma[P], N_S,sigma),

solved on the UNMIXED fixed-point residual `r_P = P - count_fill(H_B[P], N)` (Anderson mixing
is only an accelerator), with a deterministic multistart, a competing-solution gap guard and
the constrained energy-Hessian guards (`lambda_min`, `kappa`) in the registered tangent
metric. The converged value `A_B^*(S)` has the equivalent band form

    A_B^* = sum_sigma [ F_band(H_B,sigma, N_sigma) - Tr(P_sigma V_B,sigma) ] + Phi_B[P],

which is the definition of the variational energy and its double-count correction; the two
forms agreeing to the floor is a section-11.1 gate.

Forces (section 6.5) are the envelope theorem: `A_B` evaluated once at the DETACHED stationary
`P` with the geometry attached; nothing here differentiates the SCF iteration.

This module is built bottom-up and each layer is tested before the next is used:
`occupation_regulariser` (the defining identity with `free_energy`), the boundary potential
`V_B` (matrix finite differences of `Phi_B`), the fixed point, the guards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mace.modules import defect_counting as _cnt
from mace.modules.defect_counting import _SQRT_PI

__all__ = ["occupation_regulariser", "regulariser_slope", "count_fill_density",
           "ScfOptions", "ScfResult", "ScfError", "primary_functional", "scf_map",
           "stationary_solution", "TangentSpace", "HessianGuard", "hessian_matvec",
           "hessian_guards"]

#: Occupations are clamped away from the exact saturation values before the smearing's
#: inverse is taken: `erfcinv(0)` and `log(0)` are infinite, while a saturated level's
#: contribution to the regulariser is an exact zero for both families (the Gaussian term
#: `exp(-x^2)` vanishes as `x -> +-inf`, the Shannon term as `f -> 0, 1`), so the clamp
#: changes no value.
OCCUPATION_FLOOR = 1e-15


def _clamped_eigenvalues(P: torch.Tensor) -> torch.Tensor:
    f = torch.linalg.eigvalsh(P.double())
    return f.clamp(OCCUPATION_FLOOR, 1.0 - OCCUPATION_FLOOR)


def _per_level(f: torch.Tensor, family: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(S(f), dS/df)` per level for the smearing family, as functions of the OCCUPATION.

    Gaussian (Methfessel-Paxton order 0): `f = erfc(x)/2`, so `x = erfcinv(2f)` and
    `S = exp(-x^2) / (2 sqrt(pi))`; `dS/df = dS/dx dx/df = (-x exp(-x^2)/sqrt(pi))
    (-sqrt(pi) exp(x^2)) = x`. Fermi-Dirac: the Shannon entropy, `dS/df = ln((1-f)/f)`.
    """
    if family == "gaussian":
        x = torch.special.erfinv(1.0 - 2.0 * f)          # erfcinv(2f)
        return torch.exp(-x * x) / (2.0 * _SQRT_PI), x
    if family == "fermi":
        return -(f * f.log() + (1.0 - f) * (1.0 - f).log()), ((1.0 - f) / f).log()
    raise ValueError(f"unknown smearing family {family!r}; expected gaussian or fermi")


def occupation_regulariser(P: torch.Tensor, t_el: float, family: Optional[str] = None
                           ) -> torch.Tensor:
    """`R_sm[P] = -T sum_k S(f_k)` over the eigenvalues `f_k` of `P` (section 6.2).

    A function of the eigenvalues alone, so its derivative in `P` is `U diag(dR/df) U^T` and
    involves no eigenvector response; `torch.linalg.eigvalsh`'s backward is exactly that.
    Minimising `Tr(P H) + R_sm[P]` at fixed `Tr P = N` gives `P = f((H - mu)/T)` with the
    family's occupation function and hence `F_band(H, N)` -- the defining identity, which
    the tests check against `defect_counting.free_energy` for both families.
    """
    family = _cnt._FAMILY if family is None else family   # the LIVE family
    entropy, _ = _per_level(_clamped_eigenvalues(P), family)
    return (-float(t_el) * entropy.sum()).to(P.dtype)


def regulariser_slope(P: torch.Tensor, t_el: float, family: Optional[str] = None
                      ) -> torch.Tensor:
    """`dR_sm/dP = -T U diag(dS/df) U^T`: `T x_k` under Gaussian smearing, `T ln(f/(1-f))`
    under Fermi-Dirac. At a count-fill density this is `-(H - mu)` on the levels, which is
    how `Tr(P H) + R_sm` is stationary there."""
    family = _cnt._FAMILY if family is None else family
    f, U = torch.linalg.eigh(P.double())
    f = f.clamp(OCCUPATION_FLOOR, 1.0 - OCCUPATION_FLOOR)
    _, slope = _per_level(f, family)
    return (-float(t_el) * (U * slope.unsqueeze(-2)) @ U.transpose(-1, -2)).to(P.dtype)


def count_fill_density(H: torch.Tensor, n_electrons: float, t_el: float
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(P, lam, U, mu)` of `count_fill(H, N)`: the density matrix of the configured smearing
    at the `mu` holding exactly `N` electrons, with the spectrum it was built from. Detached
    (the SCF's map; the energy at the fixed point is attached elsewhere)."""
    from mace.modules.defect_counting import fermi_fill, find_mu

    with torch.no_grad():
        lam, U = torch.linalg.eigh(H.detach().double())
        mu = find_mu(lam, float(n_electrons), float(t_el), _cnt._FAMILY)
        f = fermi_fill(lam, float(n_electrons), float(t_el), mu=mu)
        P = (U * f.unsqueeze(-2)) @ U.transpose(-1, -2)
    return P, lam, U, mu


# ------------------------------------------------------------------ layer 3: the fixed point


@dataclass
class ScfOptions:
    """The registered solver regime (section 6.2 / 6.3: part of every stationary-value cache
    key). Anderson mixing is only the accelerator; convergence is judged on the UNMIXED
    fixed-point residual `r_P = P - count_fill(H_B[P], N)`."""

    tol_residual: float = 1e-8      # max |r_P| (Frobenius), per spin
    tol_energy: float = 1e-10       # |A_B(P_k) - A_B(P_{k-1})| at acceptance, eV
    max_iter: int = 200
    mixing: float = 0.3             # Anderson damping beta
    history: int = 6                # Anderson depth (0: plain damped iteration)

    def to_dict(self) -> Dict[str, Any]:
        return {"tol_residual": self.tol_residual, "tol_energy": self.tol_energy,
                "max_iter": self.max_iter, "mixing": self.mixing, "history": self.history,
                "mixer": "anderson/v1"}


class ScfError(RuntimeError):
    """The stationary solve did not meet its registered bounds: an unsupported state."""


@dataclass
class ScfResult:
    P: List[torch.Tensor]                 # the stationary densities, detached, per spin
    V: List[torch.Tensor]                 # V_B,sigma at P, detached
    H_B: List[torch.Tensor]               # H_fix + V_B,sigma, detached
    mu: List[torch.Tensor]
    energy: torch.Tensor                  # A_B^* in the primary form, ATTACHED (envelope)
    energy_band: torch.Tensor             # the band form at the same point (detached)
    phi: Dict[str, torch.Tensor]          # Phi_B and its parts at P^*
    rho: Any
    diagnostics: Dict[str, Any]
    residual: float                       # max_sigma |P^* - F(P^*)|_F, the unmixed residual
    commutator: float                     # max_sigma |[H_B, P^*]|_F
    delta_energy: float                   # |A_B(P^*) - A_B(previous iterate)|
    iterations: int
    converged: bool
    history: List[float] = field(default_factory=list)


def _frobenius(x: torch.Tensor) -> float:
    return float(x.detach().norm())


def primary_functional(ctx, P: Sequence[torch.Tensor], n_e: Sequence[int], n_h: Sequence[int],
                       t_el: float, H_fix: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any, Dict[str, Any]]:
    """`A_B[P] = sum_sigma [Tr(P_sigma H_fix) + R_sm[P_sigma]] + Phi_B[P]` for one graph, with
    `P` as given (attached or not) and `H_fix` attached (the context's entry by default)."""
    from mace.modules.defect_boundary import phi_b

    H = ctx.entry.H if H_fix is None else H_fix
    band = sum((p * H).sum() + occupation_regulariser(p, t_el) for p in P)
    phi, rho, diagnostics = phi_b(ctx, P, n_e, n_h)
    return band + phi["phi"], phi, rho, diagnostics


def scf_map(ctx, P: Sequence[torch.Tensor], n_e: Sequence[int], n_h: Sequence[int],
            N: Sequence[float], t_el: float, H_fix: torch.Tensor):
    """One application of the stationary map: `V_B[P]`, `H_B = H_fix + V_B`, and
    `F(P) = count_fill(H_B, N)` per spin. Everything detached (the map is a solver)."""
    from mace.modules.defect_boundary import boundary_potential

    V, phi, rho, diagnostics = boundary_potential(ctx, P, n_e, n_h)
    out, H_B, mus = [], [], []
    for spin in range(2):
        H = H_fix.detach() + V[spin].detach()
        P_new, lam, U, mu = count_fill_density(H, float(N[spin]), t_el)
        out.append(P_new.to(H_fix.dtype))
        H_B.append(H)
        mus.append(mu)
    return out, V, H_B, mus, phi, rho, diagnostics


def _anderson_step(P_hist: List[List[torch.Tensor]], F_hist: List[List[torch.Tensor]],
                   beta: float) -> List[torch.Tensor]:
    """Anderson mixing over the stored iterates (both spins stacked as one vector): the
    affine combination of `P_i + beta (F_i - P_i)` whose residual combination is smallest in
    the least-squares sense. Affine, so the traces are preserved exactly."""
    m = len(P_hist)
    flat = lambda mats: torch.cat([x.reshape(-1) for x in mats])
    R = torch.stack([flat(F) - flat(P) for P, F in zip(P_hist, F_hist)])      # [m, d]
    if m == 1:
        c = torch.ones(1, dtype=R.dtype, device=R.device)
    else:
        # min_c |sum_i c_i r_i|^2 with sum c_i = 1: solve on the differences.
        D = (R[:-1] - R[-1:]).T                                                # [d, m-1]
        rhs = -R[-1]
        sol = torch.linalg.lstsq(D, rhs.unsqueeze(-1)).solution.reshape(-1)
        c = torch.cat([sol, (1.0 - sol.sum()).reshape(1)])
    mixed = []
    for spin in range(len(P_hist[0])):
        acc = sum(c[i] * (P_hist[i][spin] + beta * (F_hist[i][spin] - P_hist[i][spin]))
                  for i in range(m))
        mixed.append(0.5 * (acc + acc.transpose(-1, -2)))
    return mixed


def stationary_solution(ctx, n_e: Sequence[int], n_h: Sequence[int], N: Sequence[float],
                        t_el: float, P0: Optional[Sequence[torch.Tensor]] = None,
                        options: Optional[ScfOptions] = None) -> ScfResult:
    """The stationary solution of `A_B` at fixed per-spin traces `N` (section 6.2).

    Iterates `P <- F(P) = count_fill(H_fix + V_B[P], N)` with Anderson mixing from `P0`
    (`count_fill(H_fix, N)` by default), accepting on the UNMIXED residual `|P - F(P)|`,
    `|[H_B, P]|` and the change of `A_B`. The returned `P^*` is a genuine count-fill density
    (`F` of the last iterate: exact traces); its residual is reported, not assumed. The
    energy is the primary form at the DETACHED `P^*` with the geometry attached -- the
    envelope-theorem value whose autograd is the force (section 6.5).
    """
    opts = options or ScfOptions()
    H_fix = ctx.entry.H
    if P0 is None:
        P = [count_fill_density(H_fix.detach(), float(N[s]), t_el)[0].to(H_fix.dtype)
             for s in range(2)]
    else:
        P = [p.detach().clone() for p in P0]
    P_hist: List[List[torch.Tensor]] = []
    F_hist: List[List[torch.Tensor]] = []
    history: List[float] = []
    energy_prev: Optional[float] = None
    converged = False
    delta_energy = float("inf")
    iterations = 0
    with torch.no_grad():
        for iterations in range(1, opts.max_iter + 1):
            F, V, H_B, mus, phi, rho, diag = scf_map(ctx, P, n_e, n_h, N, t_el, H_fix)
            residual = max(_frobenius(F[s] - P[s]) for s in range(2))
            history.append(residual)
            energy_now = float(primary_functional(ctx, P, n_e, n_h, t_el)[0])
            delta_energy = abs(energy_now - energy_prev) if energy_prev is not None else float("inf")
            energy_prev = energy_now
            if residual < opts.tol_residual and delta_energy < opts.tol_energy:
                converged = True
                break
            P_hist.append([p.clone() for p in P])
            F_hist.append([f.clone() for f in F])
            if len(P_hist) > max(opts.history, 1):
                P_hist.pop(0)
                F_hist.pop(0)
            P = _anderson_step(P_hist, F_hist, opts.mixing) if opts.history > 0 else \
                [p + opts.mixing * (f - p) for p, f in zip(P, F)]
    # The accepted point is F of the last iterate: exact traces, and its own residual and
    # commutator are re-measured there rather than inherited.
    P_star = [f.detach() for f in F]
    with torch.no_grad():
        F2, V2, H_B2, mus2, phi2, rho2, diag2 = scf_map(ctx, P_star, n_e, n_h, N, t_el, H_fix)
        residual_star = max(_frobenius(F2[s] - P_star[s]) for s in range(2))
        commutator = max(_frobenius(H_B2[s] @ P_star[s] - P_star[s] @ H_B2[s]) for s in range(2))
    if not converged or residual_star > 10.0 * opts.tol_residual:
        raise ScfError(
            f"stationary solve did not converge in {iterations} iterations: unmixed residual "
            f"{residual_star:.3e} (bound {opts.tol_residual:.1e}), commutator {commutator:.3e}, "
            f"energy change {delta_energy:.3e}; the state is unsupported (section 6.2)")
    # The envelope-theorem energy: A_B at the DETACHED P^*, geometry attached through H_fix
    # and Phi_B (registration, lift primitives, windows, static and channel densities).
    energy, phi_star, rho_star, diag_star = primary_functional(ctx, P_star, n_e, n_h, t_el)
    with torch.no_grad():
        from mace.modules.defect_counting import free_energy

        band_form = sum(free_energy(torch.linalg.eigvalsh(H_B2[s].double()), float(N[s]), t_el)
                        - (P_star[s].double() * V2[s].double()).sum() for s in range(2))
        energy_band = (band_form + phi_star["phi"].detach().double()).to(energy.dtype)
    return ScfResult(P=P_star, V=[v.detach() for v in V2], H_B=[h.detach() for h in H_B2],
                     mu=mus2, energy=energy, energy_band=energy_band.detach(),
                     phi={k: v.detach() for k, v in phi_star.items()}, rho=rho_star,
                     diagnostics=diag_star, residual=residual_star, commutator=commutator,
                     delta_energy=delta_energy, iterations=iterations, converged=converged,
                     history=history)


# ------------------------------------------------------------------ layer 4: the Hessian guards


@dataclass
class HessianGuard:
    """Section 6.2: `lambda_min(grad^2_{T_P} A_B) >= lambda_guard > 0` and `kappa <= kappa_max`
    in the registered tangent metric (Frobenius on `dP`), at an accepted solution."""

    lambda_min: float
    lambda_max: float
    kappa: float
    n_directions: int
    lanczos_steps: int
    fd_step: float

    def passes(self, lambda_guard: float, kappa_max: float) -> bool:
        return self.lambda_min >= lambda_guard and self.kappa <= kappa_max

    def to_dict(self) -> Dict[str, Any]:
        return {"lambda_min": self.lambda_min, "lambda_max": self.lambda_max,
                "kappa": self.kappa, "n_directions": self.n_directions,
                "lanczos_steps": self.lanczos_steps, "fd_step": self.fd_step}


#: A pair of levels spans a tangent direction only when its occupations differ by more than
#: this: a rotation between two levels of equal occupation does not move `P` (a null
#: direction of the manifold of densities with the spectrum of `P^*`), and in the linear
#: parametrisation it leaves the physical manifold (an eigenvalue of `P` goes negative),
#: where the regulariser is infinite. The occupation active set is registered with this
#: threshold (addendum 6.2: "the occupation active set and this metric are checkpointed").
PAIR_ETA = 1e-4


class TangentSpace:
    """The admissible trace-preserving Hermitian tangent space `T_P` at a stationary `P`
    of one spin, in the eigenbasis of `H_B`: the rotations between levels of different
    occupation (the particle-hole block of a gapped state, and every pair of a fractional
    level), with the band Hessian `(eps_a - eps_i) / (f_i - f_a)` per unit Frobenius norm
    of `dP` -- read from the SPECTRUM, never by inverting an occupation (the inverse of a
    saturated `f` is infinite, the divided difference is not). Exact symmetry directions
    are not quotiented here (a registered symmetry rule is a follow-up); the metric is
    Frobenius on `dP`.
    """

    def __init__(self, H_B: torch.Tensor, P: torch.Tensor, t_el: float,
                 eta: float = PAIR_ETA):
        lam, U = torch.linalg.eigh(H_B.double())
        f = torch.diagonal(U.T @ P.double() @ U)
        self.lam, self.U, self.f = lam, U, f
        n = lam.numel()
        # Every pair (i < a, ordered by H_B's spectrum) whose occupations differ: the
        # occupied-empty block of a gapped state, plus the pairs of any fractional level.
        pairs = [(i, a) for i in range(n) for a in range(i + 1, n)
                 if float(f[i] - f[a]) > eta]
        self.pairs = pairs
        self.band = torch.tensor(
            [float((lam[a] - lam[i]) / (f[i] - f[a])) for i, a in pairs], dtype=torch.float64)

    @property
    def dim(self) -> int:
        return len(self.pairs)

    def direction(self, coefficients: torch.Tensor) -> torch.Tensor:
        """`dP = sum_p c_p D_p`, `D_p = (|i><a| + |a><i|) / sqrt(2)` (unit Frobenius)."""
        n = self.lam.numel()
        M = torch.zeros(n, n, dtype=torch.float64)
        for c, (i, a) in zip(coefficients.tolist(), self.pairs):
            M[i, a] += c / math.sqrt(2.0)
            M[a, i] += c / math.sqrt(2.0)
        return self.U @ M @ self.U.T

    def coefficients(self, dP: torch.Tensor) -> torch.Tensor:
        """The projection of a symmetric matrix onto the tangent directions."""
        M = self.U.T @ dP.double() @ self.U
        return torch.tensor([float(M[i, a] * math.sqrt(2.0)) for i, a in self.pairs],
                            dtype=torch.float64)


def hessian_matvec(ctx, res: "ScfResult", spaces: Sequence[TangentSpace], n_e, n_h,
                   v: torch.Tensor, fd_step: float = 1e-4) -> torch.Tensor:
    """`grad^2 A_B v` on the joint tangent space of both spins: the band part is diagonal
    (from the spectrum), the boundary part is the directional derivative of `V_B` along the
    direction, by a central finite difference of `V_B` -- not a double backward through the
    Daleckii-Krein Functions, whose second-order term is dropped (D19)."""
    from mace.modules.defect_boundary import boundary_potential

    dims = [s.dim for s in spaces]
    parts = torch.split(v.double(), dims)
    dP = [s.direction(c) if s.dim else torch.zeros_like(p) for s, c, p in zip(spaces, parts, res.P)]
    plus = [p.double() + fd_step * d for p, d in zip(res.P, dP)]
    minus = [p.double() - fd_step * d for p, d in zip(res.P, dP)]
    V_plus = boundary_potential(ctx, [x.to(res.P[0].dtype) for x in plus], n_e, n_h)[0]
    V_minus = boundary_potential(ctx, [x.to(res.P[0].dtype) for x in minus], n_e, n_h)[0]
    out = []
    for s, c, vp, vm in zip(spaces, parts, V_plus, V_minus):
        if s.dim == 0:
            continue
        dV = (vp.double() - vm.double()) / (2.0 * fd_step)
        out.append(s.band * c + s.coefficients(dV))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


def hessian_guards(ctx, res: "ScfResult", n_e, n_h, t_el: float, steps: int = 30,
                   fd_step: float = 1e-4, seed: int = 0) -> HessianGuard:
    """`lambda_min`, `lambda_max` and `kappa` of the tangent Hessian by Lanczos on
    `hessian_matvec` (full reorthogonalisation; `steps` iterations or the dimension)."""
    spaces = [TangentSpace(h, p, t_el) for h, p in zip(res.H_B, res.P)]
    dim = sum(s.dim for s in spaces)
    if dim == 0:
        return HessianGuard(float("inf"), float("inf"), 1.0, 0, 0, fd_step)
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g, dtype=torch.float64)
    q = q / q.norm()
    Q = [q]
    alphas, betas = [], []
    k = min(int(steps), dim)
    for j in range(k):
        w = hessian_matvec(ctx, res, spaces, n_e, n_h, Q[-1], fd_step)
        alpha = float(w @ Q[-1])
        w = w - alpha * Q[-1] - (betas[-1] * Q[-2] if betas else 0.0)
        for prev in Q:                                    # full reorthogonalisation
            w = w - (w @ prev) * prev
        alphas.append(alpha)
        beta = float(w.norm())
        if j == k - 1 or beta < 1e-12:
            break
        betas.append(beta)
        Q.append(w / beta)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        b = torch.tensor(betas[: len(alphas) - 1], dtype=torch.float64)
        T = T + torch.diag(b, 1) + torch.diag(b, -1)
    ritz = torch.linalg.eigvalsh(T)
    lo, hi = float(ritz.min()), float(ritz.max())
    kappa = hi / lo if lo > 0 else float("inf")
    return HessianGuard(lo, hi, kappa, dim, len(alphas), fd_step)
