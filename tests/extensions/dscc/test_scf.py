"""Plan section 5 gates on the solver: convergence, `sum dq = Q` to 1e-12, primary vs band
form to 1e-9 eV, unmixed residual and commutator at the fixed point, history
independence, the root rule, and the unrolled gradient against finite differences."""
import math

import pytest
import torch

from mace.modules.dscc import kernels as kn
from mace.modules.dscc import scf

torch.set_default_dtype(torch.float64)
N_ATOMS = 8


def _toy(seed=0, gap=3.0, coupling=1.0):
    """A random sp Hamiltonian with a gap and a regime-A Gamma on a random geometry."""
    g = torch.Generator().manual_seed(seed)
    n = 4 * N_ATOMS
    A = torch.randn(n, n, generator=g)
    H0 = 0.5 * (A + A.T)
    eps, U = torch.linalg.eigh(H0)
    # Reference fill 17 + 16 = 33 electrons of 32 levels per spin? no: N_ref = 26 (Cs 1 x2,
    # Pb 4 x2, Cl 7 x2 ... ) -- take a species-free toy: 20 electrons, gap after level 10.
    eps = torch.sort(eps).values
    eps[10:] += gap
    H0 = U @ torch.diag(eps) @ U.T
    cell = torch.eye(3) * 8.0
    positions = torch.rand(N_ATOMS, 3, generator=g) * 8.0
    k_sr, k_lr = kn.kernel_components(positions, cell, kn.KernelConfig(regime="A", r_g=1.0))
    u = torch.full((N_ATOMS,), 1.0)
    gamma = coupling * kn.gamma_matrix(k_sr, k_lr, torch.tensor(0.5), u, 4.0)
    gamma_zero = coupling * kn.gamma_matrix(k_sr, k_lr, torch.tensor(0.0), torch.zeros(N_ATOMS), 4.0)
    return H0, gamma, gamma_zero, positions, cell


N_REF = (10, 10)
N_S = (9, 10)          # one hole in the majority channel, Q = +1


class TestSolve:
    def test_fixed_point_identities(self):
        H0, gamma, _, _, _ = _toy()
        opt = scf.ScfOptions(tol_q=1e-11, tol_E=1e-13)
        res = scf.solve_dscc(H0, gamma, N_S, N_REF, options=opt)
        assert res.converged and res.iterations < opt.n_max
        assert float(res.dq.sum()) == pytest.approx(1.0, abs=1e-12)
        assert abs(float(res.energy) - float(res.energy_primary)) < 1e-9
        assert res.residual < 1e-10 and res.commutator < 1e-7 and res.rho < 0.9
        # A second solve from the answer converges at once (two iterations: the energy
        # criterion needs a change to measure) to the same energy.
        again = scf.solve_dscc(H0, gamma, N_S, N_REF, dq0=res.dq.detach(), options=opt)
        assert again.iterations <= 2 and abs(float(again.energy - res.energy)) < 1e-12

    def test_energy_is_independent_of_the_mixing_history(self):
        H0, gamma, _, _, _ = _toy(seed=1)
        e = []
        for mixing, history in ((0.3, 6), (0.1, 3), (0.5, 8)):
            res = scf.solve_dscc(H0, gamma, N_S, N_REF,
                                 options=scf.ScfOptions(tol_q=1e-11, tol_E=1e-13, mixing=mixing, history=history))
            assert res.converged
            e.append(float(res.energy))
        assert max(e) - min(e) < 1e-10

    def test_zero_coupling_is_two_fillings(self):
        H0, gamma, _, _, _ = _toy(coupling=0.0)
        res = scf.solve_dscc(H0, gamma, N_S, N_REF)
        two = scf.two_fillings(H0, N_S, N_REF)
        # Three iterations: the damped first step, then Anderson lands exactly.
        assert abs(float(res.energy - two.energy)) < 1e-12 and res.iterations <= 3

    def test_root_rule(self):
        H0, gamma, gamma_zero, _, _ = _toy(seed=2)
        opt = scf.ScfOptions(tol_q=1e-11, tol_E=1e-13)
        warm = scf.solve_dscc(H0, gamma, N_S, N_REF, options=opt).dq.detach() + 0.05
        out = scf.root_rule(H0, gamma, gamma_zero, N_S, N_REF, dq_previous=warm, options=opt)
        assert out["passed"] and out["spread"] < 1e-8
        assert set(out["solutions"]) == {"zero", "continuation", "warm"}

    def test_unrolled_gradient_matches_finite_differences(self):
        """`dJ*/dtheta` through the unrolled loop against central differences on a
        one-parameter family `H0 + theta B`."""
        H0, gamma, _, _, _ = _toy(seed=3)
        g = torch.Generator().manual_seed(4)
        B = torch.randn(H0.shape[0], H0.shape[0], generator=g)
        B = 0.05 * (B + B.T)
        opt = scf.ScfOptions(tol_q=1e-12, tol_E=1e-14)
        theta = torch.zeros((), requires_grad=True)
        res = scf.solve_dscc(H0 + theta * B, gamma, N_S, N_REF, options=opt, unroll=True)
        (grad,) = torch.autograd.grad(res.energy, theta)
        h = 1e-4
        e_plus = float(scf.solve_dscc(H0 + h * B, gamma, N_S, N_REF, options=opt).energy)
        e_minus = float(scf.solve_dscc(H0 - h * B, gamma, N_S, N_REF, options=opt).energy)
        assert float(grad) == pytest.approx((e_plus - e_minus) / (2 * h), abs=1e-6, rel=1e-6)
        # The envelope value (no unrolling) gives the same number: J* is stationary.
        res2 = scf.solve_dscc(H0 + theta * B, gamma, N_S, N_REF, options=opt, unroll=False)
        (grad2,) = torch.autograd.grad(res2.energy, theta)
        assert float(grad2) == pytest.approx(float(grad), abs=1e-6)

    def test_cap_is_flagged_not_accepted(self):
        H0, gamma, _, _, _ = _toy(seed=5)
        res = scf.solve_dscc(H0, gamma, N_S, N_REF, options=scf.ScfOptions(n_max=2, tol_q=1e-14))
        assert not res.converged and res.iterations == 2


class TestNewton:
    def test_hole_response_matches_finite_differences(self):
        """`M = d dq_new / dV` against central differences on the toy."""
        H0, gamma, _, _, _ = _toy(seed=6)
        n = N_ATOMS
        V0 = 0.05 * torch.randn(n, generator=torch.Generator().manual_seed(7))
        H = H0 - scf.site_potential_matrix(V0)
        M = scf.hole_response(H, N_S, N_REF)
        h = 1e-5
        for j in (0, 3, 5):
            e = torch.zeros(n); e[j] = h
            plus = scf.two_fillings(H0 - scf.site_potential_matrix(V0 + e), N_S, N_REF).dq
            minus = scf.two_fillings(H0 - scf.site_potential_matrix(V0 - e), N_S, N_REF).dq
            fd = (plus - minus).detach() / (2 * h)
            assert torch.allclose(M[:, j], fd, atol=1e-6, rtol=1e-5), (j, M[:, j], fd)
        assert torch.allclose(M, M.T, atol=1e-8)          # a static response is symmetric

    def test_newton_converges_fast_and_agrees_with_anderson(self):
        H0, gamma, _, _, _ = _toy(seed=8)
        newton = scf.solve_dscc(H0, gamma, N_S, N_REF, options=scf.ScfOptions(method="newton", tol_q=1e-11, tol_E=1e-13))
        anderson = scf.solve_dscc(H0, gamma, N_S, N_REF, options=scf.ScfOptions(method="anderson", tol_q=1e-11, tol_E=1e-13))
        assert newton.converged and anderson.converged
        assert newton.iterations <= 8 and newton.iterations < anderson.iterations
        assert float((newton.dq - anderson.dq).abs().max()) < 1e-9
        assert abs(float(newton.energy - anderson.energy)) < 1e-10
