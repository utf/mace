"""Plan section 2.4 / 2.5: kernel regimes A and B, `Gamma`, `Gamma_LR`, centring, the
placement check and the switch diagnostics."""
import math

import numpy as np
import pytest
import torch

from mace.modules.dscc import ewald as ew
from mace.modules.dscc import kernels as kn

torch.set_default_dtype(torch.float64)


def _frame(n: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    cell = torch.tensor([[9.5, 0.0, 0.0], [0.4, 9.0, 0.0], [-0.3, 0.6, 10.5]])
    return torch.rand(n, 3, generator=g) @ cell, cell


def _perovskite(reps=(2, 2, 2), a=5.6):
    from ase import Atoms
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
                                              [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * a, pbc=True).repeat(reps)
    return (torch.tensor(atoms.get_positions()), torch.tensor(np.array(atoms.get_cell())),
            [int(z) for z in atoms.get_atomic_numbers()])


class TestSwitch:
    def test_c2_switch_values_and_smoothness(self):
        r = torch.linspace(2.5, 4.5, 2001, requires_grad=True)
        w = kn.switch_c2(r, 3.2, 3.6)
        assert float(w[0]) == 1.0 and float(w[-1]) == 0.0
        assert float(w[900]) == pytest.approx(0.5)               # the midpoint, r = 3.4
        (d1,) = torch.autograd.grad(w.sum(), r, create_graph=True)
        (d2,) = torch.autograd.grad(d1.sum(), r)
        # Continuous first and second derivatives: no jumps at either end.
        h = float(r[1] - r[0])
        assert float((d1[1:] - d1[:-1]).abs().max()) < 10 * h * 2.0 / 0.4 ** 2
        assert float((d2[1:] - d2[:-1]).abs().max()) < 10 * h * 60.0 / 0.4 ** 3


class TestRegimeA:
    def test_components_add_to_the_ewald_matrix_and_the_switch_kills_the_short_range(self):
        positions, cell = _frame()
        cfg = kn.KernelConfig(regime="A", r_g=1.0)
        k_sr, k_lr = kn.kernel_components(positions, cell, cfg)
        E = ew.ewald_matrix(positions, cell, cfg.r_g)
        assert float((k_sr + k_lr - E).abs().max()) < 1e-12
        r = kn.minimum_image_distances(positions, cell)
        far = (r > cfg.r_d2) & ~torch.eye(len(positions), dtype=torch.bool)
        assert float(k_sr[far].abs().max()) == 0.0
        near = (r < cfg.r_d1) & ~torch.eye(len(positions), dtype=torch.bool)
        if near.any():
            expected = ew.COULOMB * torch.erf(r[near] / (2 * cfg.r_g)) / r[near]
            assert torch.allclose(k_sr[near], expected, atol=1e-12)
        assert torch.allclose(torch.diagonal(k_sr), ew.self_term(cfg.r_g).expand(len(positions)))

    def test_short_range_kernel_is_rewrapping_invariant(self):
        positions, cell = _frame(seed=2)
        cfg = kn.KernelConfig(regime="A")
        k = kn.k_sr_regime_a(positions, cell, cfg)
        wrapped = positions.clone(); wrapped[1] += cell[2] - cell[0]
        assert float((k - kn.k_sr_regime_a(wrapped, cell, cfg)).abs().max()) < 1e-12


class TestRegimeB:
    def test_components_add_to_the_ewald_matrix_and_converge_in_range(self):
        positions, cell = _frame(seed=3)
        cfg = kn.KernelConfig(regime="B", r_g=1.0, r_s=5.0)
        k_sr, k_lr = kn.kernel_components(positions, cell, cfg)
        E = ew.ewald_matrix(positions, cell, cfg.r_g)
        assert float((k_sr + k_lr - E).abs().max()) < 1e-10
        k_sr_far = kn.k_sr_regime_b(positions, cell, cfg, r_c=kn.regime_b_cutoff(cfg) + 10.0)
        assert float((k_sr - k_sr_far).abs().max()) < 1e-12
        wrapped = positions.clone(); wrapped[4] -= cell[1]
        assert float((k_sr - kn.k_sr_regime_b(wrapped, cell, cfg)).abs().max()) < 1e-12

    def test_k_lr_diagonal_tends_to_the_madelung_term(self):
        """The size-dependent part of `K_LR_ii` on a cubic ladder is `-alpha_M C / L`."""
        cfg = kn.KernelConfig(regime="B", r_g=1.0, r_s=5.0)
        # The broad Gaussian (width r_s/2 = 2.5 A) overlaps its images by 6 erfc(L/5) C/L:
        # 3e-5 at L = 16, below 1e-16 from L = 32.
        for L in (32.0, 64.0):
            _, k_lr = kn.kernel_components(torch.zeros(1, 3), torch.eye(3) * L, cfg)
            broad_self = 2 * ew.COULOMB / (math.sqrt(math.pi) * cfg.r_s)
            assert float(k_lr[0, 0]) - broad_self == pytest.approx(
                -ew.madelung_constant_cubic() * ew.COULOMB / L, abs=1e-8)


class TestGamma:
    def test_structure_and_symmetry(self):
        positions, cell = _frame(seed=4)
        cfg = kn.KernelConfig(regime="A")
        k_sr, k_lr = kn.kernel_components(positions, cell, cfg)
        u = torch.tensor([0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5])
        g = kn.gamma_matrix(k_sr, k_lr, torch.tensor(0.3), u, cfg.eps_inf)
        assert torch.allclose(g, g.T, atol=1e-13)
        assert torch.allclose(torch.diagonal(g), torch.diagonal(k_lr) / cfg.eps_inf + u)
        off = ~torch.eye(10, dtype=torch.bool)
        assert torch.allclose(g[off], ((k_lr + 0.3 * k_sr) / cfg.eps_inf)[off])
        # U_eff -> 0: no short-range self term on the diagonal.
        g0 = kn.gamma_matrix(k_sr, k_lr, torch.tensor(0.3), torch.zeros(10), cfg.eps_inf)
        assert float((torch.diagonal(g0) - torch.diagonal(k_lr) / cfg.eps_inf).abs().max()) == 0.0
        dq = torch.randn(10, generator=torch.Generator().manual_seed(5))
        assert float(kn.phi_cc(g, dq)) == pytest.approx(float(0.5 * dq @ g @ dq))


class TestHostCoupling:
    def test_gamma_lr_is_eta_independent_and_the_pattern_is_centred(self):
        positions, cell = _frame(seed=6)
        a = kn.gamma_lr(positions, cell, 1.0, 2.0, 4.0, eta=2.2)
        b = kn.gamma_lr(positions, cell, 1.0, 2.0, 4.0, eta=3.0)
        assert float((a - b).abs().max()) < 1e-10
        with pytest.raises(ValueError):
            kn.gamma_lr(positions, cell, 2.0, 1.0, 4.0)
        zstar = kn.project_sum_rule(torch.tensor([0.7, 1.9, -0.4]), torch.tensor([1.0, 1.0, 3.0]))
        assert float(zstar @ torch.tensor([1.0, 1.0, 3.0])) == pytest.approx(0.0, abs=1e-14)
        species = torch.tensor([0, 1, 2, 2, 2, 0, 1, 2, 2, 2])
        zbar = kn.centred_pattern(zstar[species])
        assert float(zbar.sum()) == pytest.approx(0.0, abs=1e-14)
        # A vacancy cell (one Cl fewer) is still centred, by construction.
        zbar_v = kn.centred_pattern(zstar[species[:-1]])
        assert float(zbar_v.sum()) == pytest.approx(0.0, abs=1e-14)
        w = kn.host_potential(a, zbar)
        assert w.shape == (10,)


class TestDiagnostics:
    def test_placement_check_on_the_cubic_toy(self):
        positions, cell, numbers = _perovskite()
        cfg = kn.KernelConfig(regime="A", r_d1=3.2, r_d2=3.6)
        out = kn.placement_fractions(positions, cell, numbers, cfg)
        assert out["pb_cl_first_shell_beyond_r_d1"] == 0.0        # Pb-Cl = 2.8 A
        assert out["cl_cl_octahedron_edge_below_r_d2"] == 0.0     # Cl-Cl edge = 3.96 A
        assert out["n_pb_cl_bonds"] == 8 * 6 and out["n_cl_cl_edges"] == 8 * 6 * 4
        # A window placed on the bond itself fails the check.
        bad = kn.placement_fractions(positions, cell, numbers, kn.KernelConfig(r_d1=2.5, r_d2=4.2))
        assert bad["pb_cl_first_shell_beyond_r_d1"] == 1.0 and bad["cl_cl_octahedron_edge_below_r_d2"] == 1.0

    def test_m_sw_and_f_sr(self):
        positions, cell = _frame(seed=8)
        cfg = kn.KernelConfig(regime="A", r_d1=3.0, r_d2=4.0)
        dq = torch.randn(10, generator=torch.Generator().manual_seed(9))
        r = kn.minimum_image_distances(positions, cell)
        manual = sum(abs(float(dq[i] * dq[j])) for i in range(10) for j in range(i + 1, 10)
                     if 3.0 < float(r[i, j]) < 4.0)
        assert float(kn.m_sw(dq, positions, cell, cfg)) == pytest.approx(manual)
        cfg_b = kn.KernelConfig(regime="B", r_g=1.0, r_s=5.0)
        k_sr, k_lr = kn.kernel_components(positions, cell, cfg_b)
        f = float(kn.f_sr(dq, k_sr, k_lr))
        assert math.isfinite(f)
