"""Addendum section 5.1: compact-support windows, the continued-valence projector, the
positive excess and the exact-plateau localisation switch (`defect_windows`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mace.modules import defect_windows as dw
from mace.modules.defect_carriers import UnsupportedStateError
from mace.modules.defect_density import DEFAULT_FUNCTIONAL, functional_config

torch.set_default_dtype(torch.float64)


def cfg(**over) -> dw.WindowConfig:
    f = functional_config({"delta": 0.1, "delta_s": 0.05, **over})
    return dw.WindowConfig.from_functional(f)


def _second_derivative_jump(fn, x, h=1e-3):
    """|f''(x + 3h) - f''(x - 3h)| by central differences either side of a knot."""
    def d2(x0):
        return (fn(x0 + h) - 2 * fn(x0) + fn(x0 - h)) / (h * h)
    return abs(d2(x + 3 * h) - d2(x - 3 * h))


def assert_c2(fn, knot):
    """C^2 at a knot: the second derivative is continuous, so the jump measured a distance
    3h either side must vanish linearly with h (a C^1-only function leaves it finite)."""
    coarse = _second_derivative_jump(fn, knot, h=1e-3)
    fine = _second_derivative_jump(fn, knot, h=1e-4)
    assert fine < 0.2 * coarse + 1e-9, (knot, coarse, fine)


# ------------------------------------------------------------------ scalar shapes


class TestShapes:
    def test_s5_is_a_smooth_step(self):
        t = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        assert dw.s5(t).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]
        assert dw.s5_prime(t).tolist()[:2] == [0.0, 0.0] and dw.s5_prime(t).tolist()[3:] == [0.0, 0.0]
        # C^2 at both knots: the first and second derivatives vanish there.
        for knot in (0.0, 1.0):
            assert_c2(lambda v: float(dw.s5(torch.tensor(v))), knot)

    def test_the_plateau_window_has_exact_plateaus_and_is_c2(self):
        x = torch.linspace(-2.0, 4.0, 601)
        b, slope = dw.plateau_window(x, 0.0, 0.5, 2.0, 2.5)
        assert bool((b[x <= 0.0] == 0.0).all()) and bool((b[x >= 2.5] == 0.0).all())
        assert bool((b[(x >= 0.5) & (x <= 2.0)] == 1.0).all())
        assert bool((slope[(x >= 0.5) & (x <= 2.0)] == 0.0).all())
        assert bool((b >= 0.0).all()) and bool((b <= 1.0).all())
        for knot in (0.0, 0.5, 2.0, 2.5):
            assert_c2(lambda v: float(dw.plateau_window(torch.tensor(v), 0.0, 0.5, 2.0, 2.5)[0]),
                      knot)
        # The reported slope is the derivative of the value.
        xg = torch.tensor([0.2, 2.3], requires_grad=True)
        bg, sg = dw.plateau_window(xg, 0.0, 0.5, 2.0, 2.5)
        (auto,) = torch.autograd.grad(bg.sum(), xg)
        assert torch.allclose(auto, sg, atol=1e-12)
        with pytest.raises(ValueError, match="x0 < x1"):
            dw.plateau_window(x, 1.0, 0.5, 2.0, 2.5)

    def test_r_plus_is_zero_then_the_identity_and_c2(self):
        x = torch.linspace(-1.0, 1.0, 2001)
        r, slope = dw.r_plus(x, 0.05, 0.2)
        assert bool((r[x <= 0.05] == 0.0).all())
        assert bool((r[x >= 0.2] == x[x >= 0.2]).all())
        assert bool((r >= 0.0).all()) and bool((r <= x.clamp_min(0.0) + 1e-15).all())
        for knot in (0.05, 0.2):
            assert_c2(lambda v: float(dw.r_plus(torch.tensor(v), 0.05, 0.2)[0]), knot)
        xg = torch.tensor([0.1, 0.15, 0.5], requires_grad=True)
        rg, sg = dw.r_plus(xg, 0.05, 0.2)
        (auto,) = torch.autograd.grad(rg.sum(), xg)
        assert torch.allclose(auto, sg, atol=1e-12)
        with pytest.raises(ValueError):
            dw.r_plus(x, 0.3, 0.2)

    def test_the_switch_has_exact_plateaus(self):
        x = torch.tensor([0.0, 4.0, 10.0, 16.0, 40.0])
        w = dw.exact_switch(x, 4.0, 16.0)
        assert w.tolist()[:2] == [1.0, 1.0] and w.tolist()[3:] == [0.0, 0.0]
        assert 0.0 < float(w[2]) < 1.0


# ------------------------------------------------------------------ the windows


class TestWindows:
    def test_electron_and_hole_windows_are_fixed_to_the_edges(self):
        c = cfg()
        vbm, cbm = -1.0, 1.4
        eps = torch.linspace(-8.0, 8.0, 3201)
        b_e, _ = dw.electron_window(eps, vbm, cbm, c)
        b_h, _ = dw.hole_window(eps, vbm, cbm, c)
        lo_e, hi_e = vbm + c.delta, cbm + c.extent
        assert bool((b_e[eps <= lo_e - c.delta_s] == 0.0).all())
        assert bool((b_e[(eps >= lo_e + c.delta_s) & (eps <= hi_e - c.delta_s)] == 1.0).all())
        assert bool((b_e[eps >= hi_e + c.delta_s] == 0.0).all())
        hi_h, lo_h = cbm - c.delta, vbm - c.extent
        assert bool((b_h[eps >= hi_h + c.delta_s] == 0.0).all())
        assert bool((b_h[(eps <= hi_h - c.delta_s) & (eps >= lo_h + c.delta_s)] == 1.0).all())
        assert bool((b_h[eps <= lo_h - c.delta_s] == 0.0).all())
        # Deep valence and far conduction states are EXACTLY excluded: no O(N_orb) tail.
        assert float(b_e[eps < -3.0].abs().sum()) == 0.0
        assert float(b_h[eps > 3.0].abs().sum()) == 0.0

    def test_the_config_is_validated_and_falls_back_to_the_registered_defaults(self):
        c = dw.WindowConfig.from_functional({"delta": 0.1, "delta_s": 0.05})
        assert c.extent == DEFAULT_FUNCTIONAL["window_extent"]
        assert c.n_ext == DEFAULT_FUNCTIONAL["n_ext"]
        with pytest.raises(ValueError, match="N_loc < N_ext"):
            cfg(n_loc=20.0, n_ext=16.0)
        with pytest.raises(ValueError, match="alpha_min"):
            cfg(alpha_bounds=[1.2, 1.5])


# ------------------------------------------------------------------ matrix functions


def _gapped_hamiltonian(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    levels = torch.cat([torch.linspace(-4.0, -1.0, n // 2), torch.linspace(1.5, 4.0, n // 2)])
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return Q @ torch.diag(levels) @ Q.T, levels


class TestSpectralFunctions:
    def test_a_spectral_function_is_the_matrix_function(self):
        H, _ = _gapped_hamiltonian()
        lam, U = torch.linalg.eigh(H)
        g = torch.tanh(lam)
        F = dw.spectral_function(H, g, 1.0 - g * g)
        assert torch.allclose(F, U @ torch.diag(g) @ U.T, atol=1e-12)

    def test_the_derivative_is_the_daleckii_krein_map(self):
        """Autograd through `spectral_function` against finite differences of
        `Tr(g(H) Q)` in a random symmetric direction."""
        H0, _ = _gapped_hamiltonian(n=10, seed=1)
        Q = torch.randn(10, 10, generator=torch.Generator().manual_seed(2))
        Q = 0.5 * (Q + Q.T)

        def value(H):
            lam, U = torch.linalg.eigh(H.detach())
            g, slope = dw.plateau_window(lam, -2.0, -1.5, 2.0, 3.0)
            return (dw.spectral_function(H, g, slope, spectrum=(lam, U)) * Q).sum()

        H = H0.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(value(H), H)
        direction = torch.randn(10, 10, generator=torch.Generator().manual_seed(3))
        direction = 0.5 * (direction + direction.T)
        h = 1e-5
        fd = (float(value(H0 + h * direction)) - float(value(H0 - h * direction))) / (2 * h)
        assert float((grad * direction).sum()) == pytest.approx(fd, abs=1e-7, rel=1e-6)

    def test_degeneracy_is_finite(self):
        H = torch.diag(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 3.0]))
        H = H.clone().requires_grad_(True)
        lam, U = torch.linalg.eigh(H.detach())
        g, slope = dw.plateau_window(lam, -0.5, 0.5, 2.0, 2.5)
        (grad,) = torch.autograd.grad(dw.spectral_function(H, g, slope, spectrum=(lam, U)).sum(),
                                      H)
        assert torch.isfinite(grad).all()

    def test_the_valence_projector_is_a_rank_m_projector_with_a_gated_gap(self):
        H, levels = _gapped_hamiltonian()
        lam, U = torch.linalg.eigh(H)
        P = dw.valence_projector(H, lam, U, 8, gap_floor=0.2)
        assert torch.allclose(P @ P, P, atol=1e-12)
        assert float(torch.trace(P)) == pytest.approx(8.0)
        with pytest.raises(UnsupportedStateError, match="separating gap"):
            dw.valence_projector(H, lam, U, 5, gap_floor=0.5)      # inside the valence band
        with pytest.raises(UnsupportedStateError, match="outside"):
            dw.valence_projector(H, lam, U, 16, gap_floor=0.2)


# ------------------------------------------------------------------ the excess


class TestPositiveExcess:
    def test_one_extra_electron_is_one_electron_exactly_and_the_reference_is_zero(self):
        """Fill M+1 levels of a gapped H: Delta P is the (M+1)-th eigenprojector, r_+ of it
        is itself (eigenvalue 1 >= eta_1), and inside the electron window `Tr D_e = 1`
        exactly. The reference fill (M levels) gives `Delta P = 0` and `Tr D_e = 0` exactly;
        a smearing tail below eta_0 is annihilated by the exact-zero plateau."""
        c = cfg()
        H, levels = _gapped_hamiltonian()
        lam, U = torch.linalg.eigh(H)
        vbm, cbm = float(lam[7]), float(lam[8])
        b_e, s_e = dw.electron_window(lam, vbm, cbm, c)
        B_e = dw.spectral_function(H, b_e, s_e, spectrum=(lam, U))
        P_V = dw.valence_projector(H, lam, U, 8, c.gap_floor)
        f_charged = torch.zeros(16)
        f_charged[:9] = 1.0
        P = U @ torch.diag(f_charged) @ U.T
        D = B_e @ dw.positive_excess(P - P_V, c.eta) @ B_e
        assert float(torch.trace(D)) == pytest.approx(1.0, abs=1e-12)
        rho, trace = dw.channel_density(D, 4, 1, c)
        assert float(rho.sum()) == pytest.approx(1.0) and bool((rho >= 0).all())
        # Reference: no excess.
        f_ref = torch.zeros(16)
        f_ref[:8] = 1.0
        D0 = B_e @ dw.positive_excess(U @ torch.diag(f_ref) @ U.T - P_V, c.eta) @ B_e
        assert float(torch.trace(D0)) == 0.0
        # A tail below eta_0 on every conduction level: still exactly zero.
        f_tail = f_ref.clone()
        f_tail[8:] = 0.02
        D_tail = B_e @ dw.positive_excess(U @ torch.diag(f_tail) @ U.T - P_V, c.eta) @ B_e
        assert float(torch.trace(D_tail)) == 0.0
        # A hole: the mirror statement.
        f_hole = torch.zeros(16)
        f_hole[:7] = 1.0
        b_h, s_h = dw.hole_window(lam, vbm, cbm, c)
        B_h = dw.spectral_function(H, b_h, s_h, spectrum=(lam, U))
        D_h = B_h @ dw.positive_excess(U @ torch.diag(f_hole) @ U.T - P_V, c.eta, sign=-1.0) @ B_h
        assert float(torch.trace(D_h)) == pytest.approx(1.0, abs=1e-12)

    def test_the_trace_bounds_refuse_an_unaccounted_carrier(self):
        c = cfg()
        D = torch.diag(torch.tensor([0.3, 0.0, 0.0, 0.0]))
        with pytest.raises(UnsupportedStateError, match="channel trace"):
            dw.channel_density(D, 1, 1, c)

    def test_the_excess_is_differentiable_through_delta_p(self):
        H0, _ = _gapped_hamiltonian(n=8, seed=4)
        Q = torch.randn(8, 8, generator=torch.Generator().manual_seed(5))
        Q = 0.5 * (Q + Q.T)
        lam0, U0 = torch.linalg.eigh(H0)
        f = torch.zeros(8)
        f[:5] = 1.0

        def value(dp):
            return (dw.positive_excess(dp, (0.05, 0.2)) * Q).sum()

        P_V = U0 @ torch.diag((torch.arange(8) < 4).double()) @ U0.T
        dp0 = U0 @ torch.diag(f) @ U0.T - P_V + 0.05 * Q / Q.norm()
        dp = dp0.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(value(dp), dp)
        direction = torch.randn(8, 8, generator=torch.Generator().manual_seed(6))
        direction = 0.5 * (direction + direction.T)
        h = 1e-5
        fd = (float(value(dp0 + h * direction)) - float(value(dp0 - h * direction))) / (2 * h)
        assert float((grad * direction).sum()) == pytest.approx(fd, abs=1e-7, rel=1e-6)

    def test_the_background_gates(self):
        """A carrier-free channel must be empty inside its window; an active channel's
        excess must be captured by its window."""
        c = cfg()
        H, _ = _gapped_hamiltonian()
        lam, U = torch.linalg.eigh(H)
        vbm, cbm = float(lam[7]), float(lam[8])
        b_e, s_e = dw.electron_window(lam, vbm, cbm, c)
        B_e = dw.spectral_function(H, b_e, s_e, spectrum=(lam, U))
        P_V = dw.valence_projector(H, lam, U, 8, c.gap_floor)
        f = torch.zeros(16)
        f[:9] = 1.0
        R_e, R_h = dw.positive_excess_pair(U @ torch.diag(f) @ U.T - P_V, c.eta)
        D_e = B_e @ R_e @ B_e
        assert dw.background_gates(R_e, D_e, 1, c.leakage_tol) == pytest.approx(0.0, abs=1e-12)
        # The hole channel of this electron state is carrier-free and exactly empty.
        b_h, s_h = dw.hole_window(lam, vbm, cbm, c)
        B_h = dw.spectral_function(H, b_h, s_h, spectrum=(lam, U))
        assert dw.background_gates(R_h, B_h @ R_h @ B_h, 0, c.leakage_tol) == 0.0
        # Put the electron above the window's compact support (a window reaching only 1 eV
        # past the CBM; the top level is 2.5 eV above it): not captured -> refused.
        narrow = cfg(window_extent=1.0)
        b_n, s_n = dw.electron_window(lam, vbm, cbm, narrow)
        B_n = dw.spectral_function(H, b_n, s_n, spectrum=(lam, U))
        far = torch.zeros(16)
        far[:8] = 1.0
        far[15] = 1.0
        R_far, _ = dw.positive_excess_pair(U @ torch.diag(far) @ U.T - P_V, c.eta)
        assert float(torch.trace(B_n @ R_far @ B_n)) < 1e-12      # outside (round-off only)
        with pytest.raises(UnsupportedStateError, match="outside the compact window"):
            dw.background_gates(R_far, B_n @ R_far @ B_n, 1, c.leakage_tol)
        # And a bulk tail above eta_0 inside the window on a carrier-free channel: refused.
        tail = torch.zeros(16)
        tail[:8] = 1.0
        tail[8:12] = 0.3
        _, R_h_tail = dw.positive_excess_pair(U @ torch.diag(tail) @ U.T - P_V, c.eta)
        R_e_tail, _ = dw.positive_excess_pair(U @ torch.diag(tail) @ U.T - P_V, c.eta)
        with pytest.raises(UnsupportedStateError, match="carrier-free background"):
            dw.background_gates(R_e_tail, B_e @ R_e_tail @ B_e, 0, c.leakage_tol)


# ------------------------------------------------------------------ localisation


class TestLocalisation:
    CELL = torch.eye(3) * 20.0

    def test_a_single_site_is_on_the_compact_plateau(self):
        c = cfg()
        pos = torch.tensor([[1.0, 1.0, 1.0], [8.0, 1.0, 1.0], [1.0, 9.0, 1.0], [15.0, 15.0, 15.0]])
        rho = torch.tensor([1.0, 0.0, 0.0, 0.0])
        w, n_eff, r_eff = dw.localisation(rho, pos, self.CELL, c)
        assert float(n_eff) == pytest.approx(1.0) and float(r_eff) == pytest.approx(0.0)
        assert float(w) == 1.0

    def test_a_compact_cluster_is_on_the_plateau_and_a_uniform_density_is_extended(self):
        c = cfg()
        pos = torch.tensor([[1.0, 1.0, 1.0], [3.8, 1.0, 1.0], [1.0, 3.8, 1.0]])
        rho = torch.tensor([0.5, 0.25, 0.25])
        w, n_eff, r_eff = dw.localisation(rho, pos, self.CELL, c)
        assert 1.0 < float(n_eff) < c.n_loc and float(r_eff) < c.r_loc
        assert float(w) == 1.0
        n = 40
        pos = torch.rand(n, 3, generator=torch.Generator().manual_seed(7)) * 20.0
        w, n_eff, r_eff = dw.localisation(torch.full((n,), 1.0 / n), pos, self.CELL, c)
        assert float(n_eff) == pytest.approx(float(n))
        assert float(w) == 0.0                          # exactly, both switches past x2

    def test_the_spread_uses_the_periodic_metric(self):
        """Two sites 1 A apart across a face: the wrapped separation, not 19 A."""
        c = cfg()
        pos = torch.tensor([[0.5, 5.0, 5.0], [19.5, 5.0, 5.0]])
        w, n_eff, r_eff = dw.localisation(torch.tensor([0.5, 0.5]), pos, self.CELL, c)
        assert float(r_eff) == pytest.approx(0.5, abs=1e-9)      # sqrt(1/2 * 2 * 1/4 * 1)
        assert float(w) == 1.0

    def test_the_switch_is_differentiable_in_the_density(self):
        c = cfg(n_loc=1.0, n_ext=3.0)
        pos = torch.tensor([[1.0, 1.0, 1.0], [3.8, 1.0, 1.0], [1.0, 3.8, 1.0]])
        rho = torch.tensor([0.6, 0.3, 0.1], requires_grad=True)
        w, _, _ = dw.localisation(rho, pos, self.CELL, c)
        assert 0.0 < float(w) < 1.0
        (grad,) = torch.autograd.grad(w, rho)
        assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0.0
