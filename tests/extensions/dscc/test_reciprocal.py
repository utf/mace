"""v5 W2 item 3: the direct reciprocal Gaussian kernel equals the Ewald route, its analytic pair
gradient equals the autograd pair gradient, its cell derivative is right, and the reciprocal
cutoff converges."""
import math
import numpy as np
import pytest
import torch

from mace.modules.dscc.ewald import COULOMB, ewald_matrix, pair_gradient, reciprocal_matrix, reciprocal_pair_gradient
from mace.modules.dscc.kernels import KernelConfig, kernel_components, kernel_pair_gradients, gamma_lr, gamma_lr_pair_gradient, k_lr_constant

torch.set_default_dtype(torch.float64)


def _cell_and_positions(n=7, seed=0):
    rng = np.random.default_rng(seed)
    cell = torch.tensor([[9.0, 0.3, 0.0], [-0.4, 8.2, 0.5], [0.2, 0.1, 10.5]], dtype=torch.float64)
    pos = torch.tensor(rng.random((n, 3)), dtype=torch.float64) @ cell
    return pos, cell


@pytest.mark.parametrize("width", [3.0, 6.5])
def test_reciprocal_matrix_equals_the_ewald_route_including_the_diagonal(width):
    pos, cell = _cell_and_positions()
    e = ewald_matrix(pos, cell, 0.5 * width)                  # pair width sqrt(2 (2 (w/2)^2)) = w; density background: no constant
    r = reciprocal_matrix(pos, cell, width)
    assert float((e - r).abs().max()) < 1e-12
    assert float((torch.diagonal(e) - torch.diagonal(r)).abs().max()) < 1e-12


def test_reciprocal_matrix_matches_mixed_width_ewald_for_gamma_lr():
    pos, cell = _cell_and_positions(seed=1)
    r_g, r_split = 1.0, 2.5
    e = ewald_matrix(pos, cell, r_g, r_split)
    r = reciprocal_matrix(pos, cell, math.sqrt(2.0 * (r_g ** 2 + r_split ** 2)))
    assert float((e - r).abs().max()) < 1e-12
    # the v4 point convention differs by the uniform pi w^2 C / V
    e_pt = ewald_matrix(pos, cell, r_g, r_split, background="point")
    w2 = 2.0 * (r_g ** 2 + r_split ** 2); vol = float(torch.det(cell).abs())
    assert float(((e - e_pt) - math.pi * w2 * COULOMB / vol).abs().max()) < 1e-12
    cfg_e, cfg_r = KernelConfig(regime="B", lr_route="ewald"), KernelConfig(regime="B")
    g_e = gamma_lr(pos, cell, r_g, r_split, 4.0, route="ewald"); g_r = gamma_lr(pos, cell, r_g, r_split, 4.0)
    assert float((g_e - g_r).abs().max()) < 1e-12
    for cfg in (cfg_e, cfg_r):
        k_sr, k_lr = kernel_components(pos, cell, cfg)
        assert k_sr.shape == k_lr.shape == (7, 7)
    k_e = kernel_components(pos, cell, cfg_e)[1]; k_r = kernel_components(pos, cell, cfg_r)[1]
    assert float((k_e - k_r).abs().max()) < 1e-12


def test_analytic_pair_gradient_equals_autograd():
    pos, cell = _cell_and_positions(seed=2)
    w = 4.0
    d_an = reciprocal_pair_gradient(pos, cell, w)
    (d_ag,) = pair_gradient(lambda d0: (reciprocal_matrix(pos, cell, w, pair_vectors=d0),), pos)
    assert float((d_an - d_ag).abs().max()) < 1e-10
    (d_ew,) = pair_gradient(lambda d0: (ewald_matrix(pos, cell, 0.5 * w, pair_vectors=d0),), pos)
    assert float((d_an - d_ew).abs().max()) < 1e-10
    cfg = KernelConfig(regime="B")
    d_sr, d_lr = kernel_pair_gradients(pos, cell, cfg)
    d_sr_e, d_lr_e = kernel_pair_gradients(pos, cell, KernelConfig(regime="B", lr_route="ewald"))
    assert float((d_lr - d_lr_e).abs().max()) < 1e-10 and float((d_sr - d_sr_e).abs().max()) < 1e-10
    g = gamma_lr_pair_gradient(pos, cell, 1.0, 2.5, 4.0); g_e = gamma_lr_pair_gradient(pos, cell, 1.0, 2.5, 4.0, route="ewald")
    assert float((g - g_e).abs().max()) < 1e-10


def test_need_sr_false_returns_zeros_and_the_same_lr():
    pos, cell = _cell_and_positions(seed=3)
    cfg = KernelConfig(regime="B")
    k_sr, k_lr = kernel_components(pos, cell, cfg); z_sr, z_lr = kernel_components(pos, cell, cfg, need_sr=False)
    assert float(z_sr.abs().max()) == 0.0 and float((k_lr - z_lr).abs().max()) == 0.0
    d_sr, d_lr = kernel_pair_gradients(pos, cell, cfg, need_sr=False)
    assert float(d_sr.abs().max()) == 0.0 and d_lr.shape == (7, 7, 3)


def test_cell_derivative_of_the_reciprocal_matrix_by_finite_differences():
    pos, cell = _cell_and_positions(seed=4)
    q = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.0], dtype=torch.float64)
    def energy(c):
        return 0.5 * q @ reciprocal_matrix(pos, c, 5.0) @ q
    c = cell.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(energy(c), c)
    h = 1e-5
    for a, b in ((0, 0), (1, 2), (2, 2)):
        cp, cm = cell.clone(), cell.clone(); cp[a, b] += h; cm[a, b] -= h
        fd = (float(energy(cp)) - float(energy(cm))) / (2 * h)
        assert abs(fd - float(g[a, b])) < 1e-6 * max(1.0, abs(fd))


def test_reciprocal_cutoff_convergence():
    """Registered (v5 W2 item 3): tightening the cutoff beyond the production tolerance moves a
    test charge's energy by < 1e-10 eV and its pair forces by < 1e-8 eV/A."""
    pos, cell = _cell_and_positions(seed=5)
    q = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.0], dtype=torch.float64)
    e = {tol: float(0.5 * q @ reciprocal_matrix(pos, cell, 6.5, tol=tol) @ q) for tol in (1e-10, 1e-16, 1e-20)}
    d = {tol: reciprocal_pair_gradient(pos, cell, 6.5, tol=tol) for tol in (1e-10, 1e-16, 1e-20)}
    f = {tol: torch.einsum("i,ijc,j->ic", q, d[tol], q) for tol in d}
    assert abs(e[1e-16] - e[1e-20]) < 1e-10
    assert float((f[1e-16] - f[1e-20]).abs().max()) < 1e-8
    # information: the loose cutoff's error against the converged value
    assert abs(e[1e-10] - e[1e-20]) < 1e-6 and float((f[1e-10] - f[1e-20]).abs().max()) < 1e-5


def test_c13_conventions_differ_by_the_uniform_second_moment_term():
    """C13: K_LR(density) - K_LR(point) = 4 pi r_g^2 C / V on every entry; Gamma_LR(density) -
    Gamma_LR(point) = pi w^2 C / (eps V); the K_LR constant is -pi (r_s^2 - 4 r_g^2)."""
    pos, cell = _cell_and_positions(seed=6); vol = float(torch.det(cell).abs())
    dens, point = KernelConfig(regime="B"), KernelConfig(regime="B", background="point")
    assert k_lr_constant(dens) == pytest.approx(-math.pi * (dens.r_s ** 2 - 4 * dens.r_g ** 2)) and k_lr_constant(point) == pytest.approx(-math.pi * point.r_s ** 2)
    for route in ("reciprocal", "ewald"):
        d = KernelConfig(regime="B", lr_route=route); q = KernelConfig(regime="B", lr_route=route, background="point")
        k_sr_d, k_lr_d = kernel_components(pos, cell, d); k_sr_p, k_lr_p = kernel_components(pos, cell, q)
        assert float((k_sr_d - k_sr_p).abs().max()) == 0.0
        assert float(((k_lr_d - k_lr_p) - 4 * math.pi * d.r_g ** 2 * COULOMB / vol).abs().max()) < 1e-12
    w2 = 2.0 * (1.0 ** 2 + 2.5 ** 2)
    g_d = gamma_lr(pos, cell, 1.0, 2.5, 4.0); g_p = gamma_lr(pos, cell, 1.0, 2.5, 4.0, background="point")
    assert float(((g_d - g_p) - math.pi * w2 * COULOMB / (4.0 * vol)).abs().max()) < 1e-12
    # a net-neutral pattern feels no difference in W = Gamma_LR q0
    q0 = torch.tensor([0.3, -0.2, 0.1, -0.4, 0.5, -0.1, -0.2], dtype=torch.float64)
    assert float((g_d @ q0 - g_p @ q0).abs().max()) < 1e-12
    # the density-convention E_PBC of width r_g has no constant: it equals the G != 0 sum alone
    e = ewald_matrix(pos, cell, 1.0); r = reciprocal_matrix(pos, cell, 2.0, constant=0.0)
    assert float((e - r).abs().max()) < 1e-12
