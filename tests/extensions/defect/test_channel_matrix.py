"""Plan v8 sections 2.1 and 2.5: the channel projectors as matrix functions of H.

`Q^e = U diag(f s^e) U^T` and `Q^h = U diag((1 - f) s^h) U^T` are the objects the frontier
density is read from. Their derivative in `H` is what carries the response-density force
of every term that depends on P through them (section 2.5's `Tr(P~ dH/dR)`), and it is
taken by the Daleckii-Krein route -- never through individual eigenvectors. The test is the
matrix-level one `test_band_functional` makes for `dF/dH = P`: a scalar contraction
`E = Tr(G Q)` of the channel matrix, its autograd in H against central differences, on a
spectrum with a gap, at fills with `mu` in the gap and at a partially occupied frontier, and
on a spectrum with an exactly degenerate pair.
"""

from __future__ import annotations

import pytest
import torch

from mace.modules import defect_counting as dc
from mace.modules import defect_density as dd

torch.set_default_dtype(torch.float64)


def synthetic_h(seed: int, n: int = 12, degenerate: bool = False) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=torch.float64)
    m = 0.5 * (m + m.T)
    lam, U = torch.linalg.eigh(m)
    lam = lam + torch.where(torch.arange(n) >= n // 2, 3.0, 0.0)
    if degenerate:
        lam[n // 2 + 1] = lam[n // 2]          # a degenerate pair at the conduction edge
    return (U * lam) @ U.T


def channel_value(H, n, t_el, hole, edges):
    """`Tr(G Q)` for a fixed symmetric `G`, from a fresh eigendecomposition of `H` (the
    reference the derivative is checked against)."""
    lam, U = torch.linalg.eigh(H)
    mu = dc.find_mu(lam, float(n), t_el, dc._FAMILY, tol=1e-14)
    f = dc.fermi_fill(lam, float(n), t_el, mu=mu)
    s_e, s_h = dd.edge_projectors(lam, *edges)
    g = (1.0 - f) * s_h if hole else f * s_e
    return (U * g) @ U.T


def fd_matrix(fn, H, h=1e-5):
    n = H.shape[0]
    G = torch.zeros_like(H)
    for a in range(n):
        for b in range(a, n):
            E = torch.zeros_like(H)
            E[a, b] = 1.0
            E[b, a] = 1.0
            G[a, b] = G[b, a] = (fn(H + h * E) - fn(H - h * E)) / (2 * h)
    return G


def _edges(H):
    lam = torch.linalg.eigvalsh(H)
    n = lam.numel()
    vbm, cbm = float(lam[n // 2 - 1]), float(lam[n // 2])
    delta, delta_s = 0.2, 0.1
    return (vbm, cbm, delta, delta_s)


def _apply(H, n, t_el, hole, edges, G):
    lam, U = torch.linalg.eigh(H.detach())
    mu = dc.find_mu(lam, float(n), t_el, dc._FAMILY, tol=1e-14)
    f = dc.fermi_fill(lam, float(n), t_el, mu=mu)
    s_e, s_h = dd.edge_projectors(lam, edges[0], edges[1], edges[2], edges[3])
    s = s_h if hole else s_e
    slope = s * (1.0 - s) / edges[3] * (-1.0 if hole else 1.0)
    Q = dc.channel_matrix(H, lam, U, f, mu, s, slope, hole, t_el)
    return (G * Q).sum()


@pytest.mark.parametrize("family", ["gaussian", "fermi"])
@pytest.mark.parametrize("hole", [False, True], ids=["electron", "hole"])
@pytest.mark.parametrize("n", [6, 7, 5], ids=["gap", "electron_fill", "hole_fill"])
def test_the_channel_matrix_derivative_in_H_matches_finite_differences(family, hole, n):
    t_el = 0.05
    previous = dc.use_smearing(family, t_el)
    try:
        H = synthetic_h(3)
        edges = _edges(H)
        g = torch.Generator().manual_seed(11)
        G = torch.randn(12, 12, generator=g)
        G = 0.5 * (G + G.T)
        Hv = H.clone().requires_grad_(True)
        E = _apply(Hv, n, t_el, hole, edges, G)
        (grad,) = torch.autograd.grad(E, Hv)
        # off-diagonal FD perturbs H_ab and H_ba together
        numeric = fd_matrix(lambda M: (G * channel_value(M, n, t_el, hole, edges)).sum(), H)
        analytic = grad + grad.T - torch.diag(torch.diagonal(grad))
        err = float((analytic - numeric).abs().max())
        scale = float(numeric.abs().max())
        assert err < 1e-6 * max(scale, 1.0), (family, hole, n, err, scale)
    finally:
        dc.use_smearing(*previous)


def test_the_degenerate_pair_is_finite_and_matches_finite_differences():
    t_el = 0.05
    previous = dc.use_smearing("gaussian", t_el)
    try:
        H = synthetic_h(5, degenerate=True)
        edges = _edges(H)
        G = torch.eye(12) + 0.1 * torch.ones(12, 12)
        Hv = H.clone().requires_grad_(True)
        E = _apply(Hv, 7, t_el, False, edges, G)
        (grad,) = torch.autograd.grad(E, Hv)
        assert torch.isfinite(grad).all()
        numeric = fd_matrix(lambda M: (G * channel_value(M, 7, t_el, False, edges)).sum(), H)
        analytic = grad + grad.T - torch.diag(torch.diagonal(grad))
        assert float((analytic - numeric).abs().max()) < 1e-6 * max(
            float(numeric.abs().max()), 1.0)
    finally:
        dc.use_smearing(*previous)


def test_it_reduces_to_the_density_matrix_at_unit_projector():
    """`s = 1` everywhere makes `Q^e = P`, whose derivative is the fill's own DK map --
    the one `_FermiDensitySum` already carries."""
    t_el = 0.05
    previous = dc.use_smearing("gaussian", t_el)
    try:
        H = synthetic_h(9)
        lam, U = torch.linalg.eigh(H)
        mu = dc.find_mu(lam, 7.0, t_el, dc._FAMILY)
        f = dc.fermi_fill(lam, 7.0, t_el, mu=mu)
        ones = torch.ones_like(lam)
        G = torch.randn(12, 12, generator=torch.Generator().manual_seed(2))
        Ha = H.clone().requires_grad_(True)
        Qa = dc.channel_matrix(Ha, lam, U, f, mu, ones, torch.zeros_like(lam), False, t_el)
        (ga,) = torch.autograd.grad((G * Qa).sum(), Ha)
        Hb = H.clone().requires_grad_(True)
        Pb = dc.fermi_density_difference(Hb, (7.0,), (1.0,), t_el, spectrum=(lam, U),
                                         occupations=f.unsqueeze(0), mus=(mu,))
        (gb,) = torch.autograd.grad((G * Pb).sum(), Hb)
        assert torch.allclose(ga, gb, atol=1e-12)
    finally:
        dc.use_smearing(*previous)
