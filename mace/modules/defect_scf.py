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
from typing import Optional, Tuple

import torch

from mace.modules import defect_counting as _cnt
from mace.modules.defect_counting import _SQRT_PI

__all__ = ["occupation_regulariser", "regulariser_slope", "count_fill_density"]

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
