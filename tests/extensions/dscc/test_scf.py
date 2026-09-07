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


class TestImplicitDifferentiation:
    def test_response_gradient_matches_unrolled_and_finite_differences(self):
        """A NON-stationary observable of the fixed point, `Tr(dP* B)`, differentiated
        w.r.t. a Hamiltonian parameter: implicit (Newton) == unrolled (Anderson) == FD."""
        H0, gamma, _, _, _ = _toy(seed=9)
        g = torch.Generator().manual_seed(10)
        B = torch.randn(H0.shape[0], H0.shape[0], generator=g); B = 0.5 * (B + B.T)
        Bd = torch.randn(H0.shape[0], H0.shape[0], generator=g); Bd = 0.05 * (Bd + Bd.T)
        tight = dict(tol_q=1e-12, tol_E=1e-14)

        def observable(theta, method, train):
            res = scf.solve_dscc(H0 + theta * Bd, gamma, N_S, N_REF,
                                 options=scf.ScfOptions(method=method, **tight),
                                 unroll=train and method == "anderson", implicit=train and method == "newton")
            return (res.dP * B).sum()

        theta = torch.zeros((), requires_grad=True)
        (g_impl,) = torch.autograd.grad(observable(theta, "newton", True), theta)
        (g_unrolled,) = torch.autograd.grad(observable(theta, "anderson", True), theta)
        (g_envelope,) = torch.autograd.grad(observable(theta, "newton", False), theta)
        h = 1e-4
        fd = (float(observable(torch.tensor(h), "newton", False)) - float(observable(torch.tensor(-h), "newton", False))) / (2 * h)
        assert float(g_impl) == pytest.approx(fd, abs=1e-5, rel=1e-5)
        assert float(g_unrolled) == pytest.approx(fd, abs=1e-5, rel=1e-5)
        # Without the fixed-point derivative the response is missing: a different number.
        assert abs(float(g_envelope) - fd) > 1e-3 * max(1.0, abs(fd))


class TestV45Solver:
    """v4.5: Levenberg-Marquardt damping, the tangent predictor and warm starts reach the
    fixed points of the backtracking solver (physics, tolerances unchanged); the batched
    solver is the per-graph one, graph by graph; `tol_c` holds at convergence."""

    @staticmethod
    def _opts(**kw):
        return scf.ScfOptions(tol_q=1e-11, tol_E=1e-13, **kw)

    @pytest.mark.parametrize("seed", [0, 8, 11])
    def test_lm_and_backtracking_reach_the_same_fixed_point(self, seed):
        H0, gamma, _, _, _ = _toy(seed=seed)
        lm = scf.solve_dscc(H0, gamma, N_S, N_REF, options=self._opts(damping="lm"))
        bt = scf.solve_dscc(H0, gamma, N_S, N_REF, options=self._opts(damping="backtrack"))
        assert lm.converged and bt.converged
        assert float((lm.dq - bt.dq).abs().max()) < 1e-9
        assert abs(float(lm.energy - bt.energy)) < 1e-10
        assert lm.commutator < scf.ScfOptions().tol_c
        assert lm.iterations <= bt.iterations + 1

    def test_predictor_reaches_the_same_fixed_point_in_no_more_iterations(self):
        H0, gamma, _, _, _ = _toy(seed=8)
        with_p = scf.continuation_solve(H0, gamma, N_S, N_REF, options=self._opts(predictor=True))
        without = scf.continuation_solve(H0, gamma, N_S, N_REF, options=self._opts(predictor=False))
        assert with_p.converged and without.converged
        assert float((with_p.dq - without.dq).abs().max()) < 1e-9
        assert with_p.iterations <= without.iterations

    def test_warm_start_reaches_the_continuation_fixed_point(self):
        H0, gamma, _, _, _ = _toy(seed=8)
        cont = scf.continuation_solve(H0, gamma, N_S, N_REF, options=self._opts())
        stale = cont.dq.detach() + 0.05 * torch.randn(N_ATOMS, generator=torch.Generator().manual_seed(3))
        warm = scf.solve_dscc(H0, gamma, N_S, N_REF, dq0=stale, options=self._opts())
        assert warm.converged and float((warm.dq - cont.dq).abs().max()) < scf.ScfOptions().tol_root
        assert warm.iterations < cont.iterations

    def test_batched_lm_solver_matches_per_graph(self):
        toys = [_toy(seed=s) for s in (0, 8)]
        H0 = torch.stack([t[0] for t in toys])
        gamma = torch.stack([t[1] for t in toys])
        n_s = (torch.tensor([N_S[0]] * 2), torch.tensor([N_S[1]] * 2))
        n_ref = (torch.tensor([N_REF[0]] * 2), torch.tensor([N_REF[1]] * 2))
        opts = self._opts()
        b = scf.solve_dscc_batched(H0, gamma, n_s, n_ref, options=opts)
        c = scf.continuation_solve_batched(H0, gamma, n_s, n_ref, options=opts)
        for k, (H, G, *_) in enumerate(toys):
            p = scf.solve_dscc(H, G, N_S, N_REF, options=opts)
            assert b.converged[k] and float((b.dq[k] - p.dq).abs().max()) < 1e-10
            assert b.iterations[k] == p.iterations
            assert b.rho[k] == pytest.approx(p.rho, rel=0.05, abs=1e-3) and b.rho[k] < 1.0
            assert b.commutator[k] < opts.tol_c
            pc = scf.continuation_solve(H, G, N_S, N_REF, options=opts)
            assert c.converged[k] and float((c.dq[k] - pc.dq).abs().max()) < 1e-10
            assert c.iterations[k] == pc.iterations
