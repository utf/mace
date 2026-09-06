"""Addendum section 6.2 / 11.1, Stage 5: the stationary auxiliary functional, layer by layer.

* `R_sm`: minimising `Tr(P H) + R_sm[P]` at fixed `Tr P = N` gives the implemented
  `F_band(H, N)` -- the defining identity, for both smearing families, on random spectra
  including gapped ones where every occupation saturates;
* `dF_band / dH_ab = P_ba` for the production smearing;
* the regulariser's own derivative is `-(H - mu)` on the levels at a count-fill density.
"""

from __future__ import annotations

import pytest
import torch

from mace.modules import defect_counting as cnt
from mace.modules import defect_scf as scf

torch.set_default_dtype(torch.float64)


def _random_h(n: int, seed: int, gap_at: int = 0, gap: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    H = 0.5 * (A + A.T)
    if gap > 0.0:
        lam, U = torch.linalg.eigh(H)
        lam = lam.clone()
        lam[gap_at:] += gap
        H = (U * lam) @ U.T
    return H


@pytest.mark.parametrize("family", ["gaussian", "fermi"])
@pytest.mark.parametrize("n, N, gap", [(12, 5.0, 0.0), (12, 5.0, 6.0), (30, 11.0, 0.0)])
def test_the_regulariser_reproduces_the_band_free_energy(family, n, N, gap):
    """`Tr(P H) + R_sm[P] = F_band(H, N)` at `P = count_fill(H, N)`; with `gap` the frontier
    sits in a gap wider than the smearing, every occupation is exactly 0 or 1 and the
    regulariser is an exact zero (no `erfcinv(0)` blows up)."""
    previous = cnt.use_smearing(family, 0.05)
    try:
        H = _random_h(n, 3, gap_at=int(N), gap=gap)
        t_el = 0.05
        P, lam, U, mu = scf.count_fill_density(H, N, t_el)
        assert float(P.trace()) == pytest.approx(N, abs=1e-9)
        value = float((P * H).sum() + scf.occupation_regulariser(P, t_el))
        reference = float(cnt.free_energy(lam, N, t_el))
        assert value == pytest.approx(reference, abs=1e-9)
        if gap > 0.0:
            assert float(scf.occupation_regulariser(P, t_el)) == pytest.approx(0.0, abs=1e-12)
    finally:
        cnt.use_smearing(*previous)


@pytest.mark.parametrize("family", ["gaussian", "fermi"])
def test_the_count_fill_density_minimises_the_primary_form(family):
    """The variational statement behind the identity: any other density of the same trace
    -- here the count-fill of a perturbed Hamiltonian -- gives a larger value."""
    previous = cnt.use_smearing(family, 0.05)
    try:
        H = _random_h(16, 5)
        N, t_el = 7.0, 0.05
        P, lam, *_ = scf.count_fill_density(H, N, t_el)
        best = float((P * H).sum() + scf.occupation_regulariser(P, t_el))
        for seed in range(4):
            Q, *_ = scf.count_fill_density(H + 0.3 * _random_h(16, 10 + seed), N, t_el)
            assert float(Q.trace()) == pytest.approx(N, abs=1e-9)
            other = float((Q * H).sum() + scf.occupation_regulariser(Q, t_el))
            assert other > best + 1e-6
    finally:
        cnt.use_smearing(*previous)


def test_the_band_free_energy_derivative_is_the_density_matrix():
    """Section 11.1: `dF_band / dH_ab = P_ba` for the production smearing."""
    H = _random_h(14, 8).requires_grad_(True)
    N, t_el = 6.0, 0.05
    lam, U = torch.linalg.eigh(H)
    F = cnt.free_energy(lam, N, t_el)
    (dF,) = torch.autograd.grad(F, H)
    P, *_ = scf.count_fill_density(H.detach(), N, t_el)
    # eigh's backward returns the symmetrised gradient; P is symmetric.
    assert torch.allclose(dF, P.T, atol=1e-8)


def test_the_regulariser_slope_is_minus_h_plus_mu_on_the_levels():
    """`dR_sm/dP + H = mu I` in the eigenbasis of a count-fill density: the stationarity of
    the primary form at fixed trace, level by level, wherever the occupation is not
    saturated."""
    # A spectrum whose spacing is comparable to the smearing, so that several levels are
    # fractionally occupied (a spread of several eV saturates every Gaussian occupation).
    H = 0.03 * _random_h(12, 11)
    N, t_el = 5.0, 0.05
    P, lam, U, mu = scf.count_fill_density(H, N, t_el)
    slope = scf.regulariser_slope(P, t_el)
    total = U.T @ (slope + H) @ U
    f = U.T @ P @ U
    active = (torch.diagonal(f) > 1e-8) & (torch.diagonal(f) < 1.0 - 1e-8)
    assert int(active.sum()) >= 2
    assert torch.allclose(torch.diagonal(total)[active], mu.expand_as(lam)[active], atol=1e-7)
    # And it is an autograd-consistent derivative of the regulariser itself.
    Q = P.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(scf.occupation_regulariser(Q, t_el), Q)
    assert torch.allclose(0.5 * (auto + auto.T), slope, atol=1e-7)
