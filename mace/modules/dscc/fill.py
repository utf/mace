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

from mace.modules import defect_counting as _cnt

# Registered 2026-09-06 (plan section 11).
SIGMA_S = 0.05
DEGENERACY_TOL = 1e-7
_SQRT_PI = math.sqrt(math.pi)


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
                       tol: float = 1e-12) -> torch.Tensor:
    """The `mu` with `sum_a f_a = N`, by the batched bisection (no host synchronisation in
    the loop); detached, which is exact at fixed `N`."""
    return _cnt.find_mu(eps.detach(), n_electrons, sigma_s, "gaussian", tol=tol)


class _DensityMatrix(torch.autograd.Function):
    """`P = fill(H, N)` for a batch of Hamiltonians `[..., n, n]` with per-frame `N`, with
    the Daleckii-Krein divided-difference backward and the fixed-`N` chemical-potential
    correction of `defect_counting._dk_backward` (bounded at degeneracy; no eigenvector
    derivative). `defect_counting._FermiDensityMatrix` is the single-frame original; this
    one takes a tensor `N` so equal-size frames fill in one call."""

    @staticmethod
    def forward(ctx, H, n_electrons, sigma_s, degeneracy_tol, eps, U):
        n = torch.as_tensor(n_electrons, dtype=H.dtype, device=H.device)
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
        return dH, None, None, None, None, None


@dataclass
class FillResult:
    """One fill of one (batch of) Hamiltonian(s)."""
    P: torch.Tensor          # [..., n, n], differentiable in H (divided-difference backward)
    mu: torch.Tensor         # [...], detached
    eps: torch.Tensor        # [..., n], detached spectrum
    U: torch.Tensor          # [..., n, n], detached eigenvectors
    f: torch.Tensor          # [..., n], detached occupations
    F_band: torch.Tensor     # [...], differentiable in H with dF/dH = P exactly
    entropy: torch.Tensor    # [...], R, detached


def fill(H: torch.Tensor, n_electrons: Union[float, torch.Tensor],
         sigma_s: float = SIGMA_S, spectrum: Optional[Sequence[torch.Tensor]] = None
         ) -> FillResult:
    """`P = fill(H, N)` and `F_band(H, N)` for a float64 symmetric `H` `[..., n, n]`.

    `spectrum = (eps, U)` of this `H` may be supplied to share one `eigh` between several
    fills (the D-SCC loop fills the same `H` at `N_S` and `N_ref` per spin).
    """
    _require_gaussian()
    if H.dtype != torch.float64:
        raise TypeError(f"head arithmetic is float64 (plan section 1); got {H.dtype}")
    if spectrum is None:
        with torch.no_grad():
            eps, U = torch.linalg.eigh(H)
    else:
        eps, U = (t.detach() for t in spectrum)
    n = torch.as_tensor(n_electrons, dtype=H.dtype, device=H.device)
    mu = chemical_potential(eps, n, sigma_s)
    f = occupations(eps, mu, sigma_s)
    P = _DensityMatrix.apply(H, n, sigma_s, DEGENERACY_TOL, eps, U)
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
