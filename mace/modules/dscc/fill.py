"""Plan section 1: the smeared fill `P = fill(H, N)` and the band free energy `F_band`.

Gaussian smearing at `sigma_s = 0.05 eV` (registered): `f(x) = erfc(x)/2`,
`x = (eps - mu)/sigma_s`, generalised entropy `R = -sigma_s sum_a exp(-x_a^2)/(2 sqrt(pi))`,
`F_band(H, N) = min_P [Tr(P H) + R(P)]` at `Tr P = N`. Required identity:
`dF_band/dH_ab = P_ba` -- `F_band` is stationary in the occupations at fixed `N`, so its
`H`-derivative is the envelope value `P` (tested by finite differences).

The differentiable density matrix reuses `defect_counting._FermiDensityMatrix`: the
Daleckii-Krein divided-difference backward with the fixed-`N` chemical-potential correction
(plan section 2.7: "never differentiate through raw eigenvectors"). Its smearing family is the
process-wide setting of that module; this head is Gaussian by construction and refuses to
run under any other setting rather than silently computing a different functional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from mace.modules.dscc import legacy as _cnt

# Registered 2026-09-06 (plan section 11).
SIGMA_S = 0.05
DEGENERACY_TOL = 1e-7
_SQRT_PI = math.sqrt(math.pi)


EIGH_CPU_MAX_ORBITALS = 512   # v5 W2 item 7 (registered): a single float64 eigh up to this size runs on the CPU


def eigh_for(H: torch.Tensor, device: str = "auto") -> "tuple[torch.Tensor, torch.Tensor]":
    """`torch.linalg.eigh` on the backend the size warrants (v5 W2 item 7, measured on the
    A4000 + Xeon Gold 6248R): a single float64 316-orbital `eigh` costs 11.8 ms in cuSOLVER and
    5.9 ms through MKL including the round trip, while batched matrices (3-4x cheaper per
    matrix on the GPU) and float32 (2.5 ms on the GPU against 4.0 on the CPU) stay where they
    are. `device`: 'auto' (the rule), 'cpu', 'cuda' (never move)."""
    single = H.dim() == 2 or int(torch.tensor(H.shape[:-2]).prod()) == 1
    use_cpu = (device == "cpu") or (device == "auto" and H.is_cuda and H.dtype == torch.float64
                                     and single and H.shape[-1] <= EIGH_CPU_MAX_ORBITALS)
    if use_cpu and H.is_cuda:
        eps, U = torch.linalg.eigh(H.detach().cpu())
        return eps.to(H.device), U.to(H.device)
    return torch.linalg.eigh(H)


def _require_gaussian() -> None:
    if _cnt._FAMILY != "gaussian":                     # pylint: disable=protected-access
        raise RuntimeError(
            f"the D-SCC fill is Gaussian (plan section 1); the process-wide smearing family "
            f"is {_cnt._FAMILY!r}")                    # pylint: disable=protected-access


def occupations(eps: torch.Tensor, mu: torch.Tensor, sigma_s: float) -> torch.Tensor:
    """`f = erfc(x)/2`, `x = (eps - mu)/sigma_s`; `mu` broadcast over the last axis."""
    return 0.5 * torch.erfc((eps - mu.unsqueeze(-1)) / sigma_s)


def generalised_entropy(eps: torch.Tensor, mu: torch.Tensor, sigma_s: float) -> torch.Tensor:
    """`R = -sigma_s sum_a exp(-x_a^2) / (2 sqrt(pi))` per spectrum (negative)."""
    x = (eps - mu.unsqueeze(-1)) / sigma_s
    return -sigma_s * torch.exp(-x * x).sum(dim=-1) / (2.0 * _SQRT_PI)


def chemical_potential(eps: torch.Tensor, n_electrons, sigma_s: float,
                       tol: float = 1e-12, polish: int = 3) -> torch.Tensor:
    """The `mu` with `sum_a f_a = N`, by the batched bisection (no host synchronisation in
    the loop) followed by Newton steps that take the count to the rounding floor (the
    `sum dq = Q` gate of plan section 5 is 1e-12, below what a 1e-12 eV bracket gives at
    `sigma_s = 0.05 eV`); detached, which is exact at fixed `N`."""
    eps = eps.detach()
    n = torch.as_tensor(n_electrons, dtype=eps.dtype, device=eps.device)
    if n.dim() == 0:
        n = n.expand(eps.shape[:-1])
    if eps.dtype != torch.float64:                      # v5 W2 item 7: float32 pre-iterations
        tol = max(tol, 1e-5)
    count_tol = 1e-13 if eps.dtype == torch.float64 else 1e-5
    # Safeguarded Newton from the mid-gap of the integer count (the SCF profile put the
    # ~50-step bisection at a quarter of the forward): the count is monotone in mu, so a
    # Newton step is accepted when it stays inside the shrinking bracket [lo, hi] and a
    # bisection step is taken otherwise. Converges in a handful of steps for a gapped
    # spectrum and never worse than bisection.
    sorted_eps = torch.sort(eps, dim=-1).values
    n_int = n.round().long().clamp(1, eps.shape[-1] - 1)
    e_below = torch.gather(sorted_eps, -1, (n_int - 1).unsqueeze(-1)).squeeze(-1)
    e_above = torch.gather(sorted_eps, -1, n_int.unsqueeze(-1)).squeeze(-1)
    mu = 0.5 * (e_below + e_above)
    lo = eps.amin(dim=-1) - 50.0 * sigma_s - 1.0
    hi = eps.amax(dim=-1) + 50.0 * sigma_s + 1.0
    for _ in range(60):
        x = (eps - mu.unsqueeze(-1)) / sigma_s
        count = (0.5 * torch.erfc(x)).sum(dim=-1) - n
        slope = (torch.exp(-x * x) / (sigma_s * _SQRT_PI)).sum(dim=-1)    # d(sum f)/d mu > 0
        hi = torch.where(count > 0, mu, hi)                                # too many electrons: mu too high
        lo = torch.where(count > 0, lo, mu)
        newton = mu - count / slope.clamp_min(1e-300)
        inside = (newton > lo) & (newton < hi) & (slope > 1e-300)
        mu_new = torch.where(inside, newton, 0.5 * (lo + hi))
        done = (count.abs() < count_tol) | ((hi - lo) < tol)
        mu = torch.where(done, mu, mu_new)
        if bool(done.all()):
            break
    for _ in range(polish):
        x = (eps - mu.unsqueeze(-1)) / sigma_s
        count = (0.5 * torch.erfc(x)).sum(dim=-1) - n
        slope = (torch.exp(-x * x) / (sigma_s * _SQRT_PI)).sum(dim=-1)
        mu = torch.where(slope > 1e-300, mu - count / slope, mu)
    return mu


class _DensityMatrix(torch.autograd.Function):
    """`P = fill(H, N)` for a batch of Hamiltonians `[..., n, n]` with per-frame `N`, with
    the Daleckii-Krein divided-difference backward and the fixed-`N` chemical-potential
    correction of `defect_counting._dk_backward` (bounded at degeneracy; no eigenvector
    derivative). `defect_counting._FermiDensityMatrix` is the single-frame original; this
    one takes a tensor `N` so equal-size frames fill in one call."""

    @staticmethod
    def forward(ctx, H, n_electrons, sigma_s, degeneracy_tol, eps, U, mu=None):
        n = torch.as_tensor(n_electrons, dtype=H.dtype, device=H.device)
        if mu is None:
            mu = chemical_potential(eps, n, sigma_s)
        f = occupations(eps, mu, sigma_s)
        P = (U * f.unsqueeze(-2)) @ U.transpose(-1, -2)
        ctx.save_for_backward(eps, U, f, mu)
        ctx.sigma_s = float(sigma_s)
        ctx.tol = float(degeneracy_tol)
        return P

    @staticmethod
    def backward(ctx, grad_P):
        eps, U, f, mu = ctx.saved_tensors
        dH = _cnt._dk_backward(eps, U, f, ctx.sigma_s, ctx.tol, grad_P, mu)   # pylint: disable=protected-access
        return dH, None, None, None, None, None, None


@dataclass
class FillResult:
    """One fill of one (batch of) Hamiltonian(s). `P` is `None` for a frontier-only fill
    (v5 W2 item 2, inference: the density is not formed); `density()` forms it on demand."""
    P: Optional[torch.Tensor]  # [..., n, n], differentiable in H (divided-difference backward), or None
    mu: torch.Tensor         # [...], detached
    eps: torch.Tensor        # [..., n], detached spectrum
    U: torch.Tensor          # [..., n, n], detached eigenvectors
    f: torch.Tensor          # [..., n], detached occupations
    F_band: torch.Tensor     # [...], differentiable in H with dF/dH = P exactly (eager fills)
    entropy: torch.Tensor    # [...], R, detached

    def density(self) -> torch.Tensor:
        """`P = U f U^T` (detached when formed on demand)."""
        if self.P is None:
            self.P = (self.U * self.f.unsqueeze(-2)) @ self.U.transpose(-1, -2)
        return self.P


def fill(H: torch.Tensor, n_electrons: Union[float, torch.Tensor],
         sigma_s: float = SIGMA_S, spectrum: Optional[Sequence[torch.Tensor]] = None,
         mu: Optional[torch.Tensor] = None, eigh_device: str = "auto") -> FillResult:
    """`P = fill(H, N)` and `F_band(H, N)` for a float64 symmetric `H` `[..., n, n]`.

    `spectrum = (eps, U)` of this `H` may be supplied to share one `eigh` between several
    fills (the D-SCC loop fills the same `H` at `N_S` and `N_ref` per spin).
    """
    _require_gaussian()
    if H.dtype != torch.float64:
        raise TypeError(f"head arithmetic is float64 (plan section 1); got {H.dtype}")
    if spectrum is None:
        with torch.no_grad():
            eps, U = eigh_for(H, eigh_device)
    else:
        eps, U = (t.detach() for t in spectrum)
    n = torch.as_tensor(n_electrons, dtype=H.dtype, device=H.device)
    if mu is None:                       # v5 W2: the caller may share one solve between fills
        mu = chemical_potential(eps, n, sigma_s)
    f = occupations(eps, mu, sigma_s)
    P = _DensityMatrix.apply(H, n, sigma_s, DEGENERACY_TOL, eps, U, mu)
    R = generalised_entropy(eps, mu, sigma_s)
    # The envelope value: `F_band` is stationary in the occupations at fixed N, so its only
    # H-derivative is Tr(P dH). Written with the detached density so that autograd returns
    # exactly P, and the value is the minimum of Tr(PH) + R at this H.
    F_band = (P.detach() * H).sum(dim=(-2, -1)) + R
    return FillResult(P=P, mu=mu, eps=eps, U=U, f=f, F_band=F_band, entropy=R)


def band_free_energy_value(eps: torch.Tensor, n_electrons, sigma_s: float = SIGMA_S
                           ) -> torch.Tensor:
    """`F_band = sum_a f_a eps_a + R` from a spectrum alone (no gradient); the check value."""
    mu = chemical_potential(eps, n_electrons, sigma_s)
    f = occupations(eps, mu, sigma_s)
    return (f * eps).sum(dim=-1) + generalised_entropy(eps, mu, sigma_s)
