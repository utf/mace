"""Plan P kernels: E_PP against a real-space image sum, E_cP against the dense kernel, forces
against finite differences, translation invariance."""
import math
import numpy as np
import pytest
import torch

from mace.modules.dscc import polarisation as pol
from mace.modules.dscc.kernels import gamma_lr
from mace.modules.dscc.ewald import COULOMB

torch.set_default_dtype(torch.float64)


def _cell(n=6, seed=0, a=7.0):
    g = torch.Generator().manual_seed(seed)
    cell = torch.diag(torch.tensor([a, a * 1.1, a * 0.9])) + 0.05 * torch.rand(3, 3, generator=g)
    frac = torch.rand(n, 3, generator=g)
    return frac @ cell, cell


def test_dipole_dipole_matches_dense_hessian_and_real_space_sum():
    from mace.modules.dscc.ewald import reciprocal_matrix
    pos, cell = _cell(n=6)
    p = torch.randn(6, 3, generator=torch.Generator().manual_seed(1))
    p = p - p.mean(0, keepdim=True)                       # net-zero dipole: the real-space sum converges
    r_split = 2.0
    w = pol.pair_width_pp(r_split)
    e_k = pol.dipole_dipole_energy(pos, cell, p, r_split, eps_inf=1.0, tol=1e-12)
    # exact reference: the Hessian of the dense smoothed kernel entry by entry (pair leaf),
    # the direct self term removed analytically
    n = 6
    d0 = (pos.unsqueeze(1) - pos.unsqueeze(0)).clone().requires_grad_(True)
    G = reciprocal_matrix(pos, cell, w, tol=1e-12, pair_vectors=d0)
    e_dense = 0.0
    for a in range(n):
        for b in range(n):
            (grad,) = torch.autograd.grad(G[a, b], d0, create_graph=True)
            H = torch.stack([torch.autograd.grad(grad[a, b, i], d0, retain_graph=True)[0][a, b] for i in range(3)])
            term = -p[a] @ H @ p[b]
            if a == b:
                term = term - pol.dipole_self_coefficient(w) * (p[a] @ p[a])
            e_dense += 0.5 * float(term)
    assert abs(float(e_k) - e_dense) < 1e-9 * max(1.0, abs(e_dense)), (float(e_k), e_dense)
    # the real-space image sum of the same smoothed kernel (a = b, L = 0 skipped) converges to
    # it as the radius grows (quadrupolar tails: ~radius^-2)
    e_r = pol.dipole_dipole_energy_real_space(pos, cell, p, r_split, eps_inf=1.0, radius=90.0)
    assert abs(float(e_k - e_r)) < 1e-4 * max(1.0, abs(float(e_r))), (float(e_k), float(e_r))


def test_dipole_self_coefficient_is_the_r_to_zero_limit():
    w = 3.0
    r = torch.tensor([1e-3], dtype=torch.float64, requires_grad=True)
    f = torch.erf(r / w) / r
    (f1,) = torch.autograd.grad(f, r, create_graph=True)
    (f2,) = torch.autograd.grad(f1, r)
    # grad grad f at small r ~ f''(0) I  (isotropic); f''(0) = -4/(3 sqrt(pi) w^3)
    assert abs(float(-f2) * COULOMB - pol.dipole_self_coefficient(w)) < 1e-6 * pol.dipole_self_coefficient(w)


def test_carrier_dipole_energy_matches_dense_kernel():
    pos, cell = _cell(n=5, seed=3)
    dq = torch.tensor([0.4, -0.3, 0.2, -0.1, -0.2])
    p = 0.3 * torch.randn(5, 3, generator=torch.Generator().manual_seed(2))
    r_g, r_split, eps = 0.7071, 3.0, 4.0
    e = pol.carrier_dipole_energy(pos, cell, dq, p, r_g, r_split, eps, tol=1e-12)
    # dense: phi at probe points = Gamma(probes+charges) @ [0; dq]; E_c = -grad phi by central FD
    h = 1e-4
    total = torch.zeros(())
    field = pol.carrier_field(pos, cell, dq, r_g, r_split, eps, tol=1e-12)
    for a in range(5):
        for c in range(3):
            def phi(shift):
                probe = pos[a].clone(); probe[c] += shift
                allpos = torch.cat([probe.unsqueeze(0), pos])
                q = torch.cat([torch.zeros(1), dq])
                G = gamma_lr(allpos, cell, r_g, r_split, eps, tol=1e-12)
                return float(G[0] @ q)
            ec = -(phi(h) - phi(-h)) / (2 * h)
            assert abs(ec - float(field[a, c])) < 1e-6 * max(1.0, abs(ec)), (a, c, ec, float(field[a, c]))
            total = total - p[a, c] * ec
    assert abs(float(e) - float(total)) < 1e-6 * max(1.0, abs(float(total)))


@pytest.mark.parametrize("which", ["pp", "cp"])
def test_forces_match_finite_differences(which):
    pos, cell = _cell(n=5, seed=4)
    p = 0.3 * torch.randn(5, 3, generator=torch.Generator().manual_seed(5))
    dq = torch.tensor([0.4, -0.3, 0.2, -0.1, -0.2])

    def energy(x):
        if which == "pp":
            return pol.dipole_dipole_energy(x, cell, p, 2.5, 4.0, tol=1e-12)
        return pol.carrier_dipole_energy(x, cell, dq, p, 0.7071, 2.5, 4.0, tol=1e-12)

    x = pos.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(energy(x), x)
    h = 1e-5
    for a, c in [(0, 0), (2, 1), (4, 2)]:
        xp = pos.clone(); xp[a, c] += h
        xm = pos.clone(); xm[a, c] -= h
        fd = (float(energy(xp)) - float(energy(xm))) / (2 * h)
        assert abs(fd - float(g[a, c])) < 1e-7 * max(1.0, abs(fd)), (a, c, fd, float(g[a, c]))


def test_translation_invariance_and_dipole_field_is_the_gradient():
    pos, cell = _cell(n=5, seed=6)
    p = 0.3 * torch.randn(5, 3, generator=torch.Generator().manual_seed(7))
    e0 = pol.dipole_dipole_energy(pos, cell, p, 2.5, 4.0)
    e1 = pol.dipole_dipole_energy(pos + torch.tensor([1.3, -0.7, 2.1]), cell, p, 2.5, 4.0)
    assert abs(float(e0 - e1)) < 1e-10
    pp = p.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(pol.dipole_dipole_energy(pos, cell, pp, 2.5, 4.0), pp)
    assert torch.allclose(g, pol.dipole_field(pos, cell, p, 2.5, 4.0), atol=1e-10)
