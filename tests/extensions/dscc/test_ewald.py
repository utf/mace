"""Plan section 4 gates for `E_PBC`: independence of the Ewald splitting parameter to
1e-10 eV (energy, forces, stress); the LES oracle; the tiling-ladder self term; the
Gaussian-width convention; invariances."""
import math

import pytest
import torch

from mace.modules.dscc import ewald as ew

torch.set_default_dtype(torch.float64)


def _frame(n: int = 12, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    cell = torch.tensor([[9.0, 0.0, 0.0], [0.8, 8.5, 0.0], [0.3, -0.5, 10.0]])
    frac = torch.rand(n, 3, generator=g)
    q = torch.randn(n, generator=g)
    return frac @ cell, cell, q


def _energy_forces_stress(positions, cell, q, width, eta):
    pos = positions.clone().requires_grad_(True)
    strain = torch.zeros(3, 3, requires_grad=True)
    deformed = torch.eye(3) + strain
    E = ew.energy(ew.ewald_matrix(pos @ deformed, cell @ deformed, width, eta=eta), q)
    forces, stress = torch.autograd.grad(E, (pos, strain))
    return float(E), -forces, stress / float(torch.det(cell).abs())


class TestSplittingIndependence:
    @pytest.mark.parametrize("net", [False, True])
    def test_energy_forces_stress_do_not_depend_on_eta(self, net):
        positions, cell, q = _frame()
        if not net:
            q = q - q.mean()
        ref = _energy_forces_stress(positions, cell, q, 1.0, eta=1.3)
        for eta in (1.9, 2.8):
            E, F, S = _energy_forces_stress(positions, cell, q, 1.0, eta=eta)
            assert E == pytest.approx(ref[0], abs=1e-10)
            assert float((F - ref[1]).abs().max()) < 1e-10
            assert float((S - ref[2]).abs().max()) < 1e-10

    def test_cross_width_matrix_is_symmetric_and_eta_independent(self):
        positions, cell, _ = _frame(seed=3)
        a = ew.ewald_matrix(positions, cell, 0.8, 1.6, eta=1.7)
        b = ew.ewald_matrix(positions, cell, 0.8, 1.6, eta=2.5)
        c = ew.ewald_matrix(positions, cell, 1.6, 0.8, eta=2.1)
        assert float((a - b).abs().max()) < 1e-10
        assert float((a - c).abs().max()) < 1e-10
        assert float((a - a.T).abs().max()) < 1e-12


class TestConvention:
    def test_les_oracle(self):
        """`0.5 q^T (E_PBC - self) q` equals the LES Ewald energy with its background,
        for neutral and net-charged Gaussian charges; LES `sigma = sqrt(2) r_g`."""
        from mace.modules.latent_ewald import LatentEwald
        positions, cell, q = _frame(seed=7)
        r_g = 0.9
        E = ew.ewald_matrix(positions, cell, r_g)
        les = LatentEwald({"sigma": math.sqrt(2.0) * r_g, "dl": 0.3})
        for charges in (q - q.mean(), q):
            mine = float(ew.energy(E, charges) - 0.5 * (charges ** 2).sum() * ew.self_term(r_g))
            oracle = float(les.energy(charges, positions, cell.reshape(1, 3, 3),
                                      torch.zeros(len(charges), dtype=torch.long)))
            assert mine == pytest.approx(oracle, rel=1e-6, abs=1e-6)

    def test_tiling_ladder_self_term_is_the_madelung_one(self):
        """`E_PBC_ii - C/(sqrt(pi) r_g) = -alpha_M C / L` for one Gaussian in a cubic cell
        (the image-overlap correction `6 erfc(L/(2 r_g)) C/L` is 1.7e-7 at L = 8 and
        below 1e-16 from L = 12)."""
        for L in (12.0, 16.0, 32.0):
            E = ew.ewald_matrix(torch.zeros(1, 3), torch.eye(3) * L, 1.0)
            k_lr = float(E[0, 0] - ew.self_term(1.0))
            assert k_lr == pytest.approx(-ew.madelung_constant_cubic() * ew.COULOMB / L, abs=1e-8)

    def test_pair_kernel_is_erf_over_r_at_the_pair_width(self):
        """Two charges in cubic boxes, the O(1/L) image term removed by extrapolation:
        `C erf(r / w) / r`, `w = sqrt(2 (s_i^2 + s_j^2))`."""
        import numpy as np
        for s_i, s_j, r in ((1.0, 1.0, 2.0), (0.7, 1.5, 3.0)):
            sizes, vals = (30.0, 60.0, 120.0), []
            for L in sizes:
                pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
                E = ew.ewald_matrix(pos, torch.eye(3) * L, torch.tensor([s_i, s_j]))
                vals.append(float(E[0, 1]))
            # E(L) = E_inf + a/L + b/L^3: three sizes fix the three unknowns; O(1/L^5) remains.
            A = np.array([[1.0, 1.0 / L, 1.0 / L ** 3] for L in sizes])
            cross = float(np.linalg.solve(A, np.array(vals))[0])
            w = math.sqrt(2 * (s_i ** 2 + s_j ** 2))
            assert cross == pytest.approx(ew.COULOMB * math.erf(r / w) / r, abs=5e-5)
            # The candidate conventions differ by tenths of an eV or more.
            assert abs(cross - ew.COULOMB * math.erf(r / (w / math.sqrt(2))) / r) > 0.05


class TestInvariances:
    def test_translation_rewrapping_permutation(self):
        positions, cell, q = _frame(seed=11)
        E = ew.ewald_matrix(positions, cell, 1.0)
        shifted = ew.ewald_matrix(positions + torch.tensor([1.3, -2.2, 0.7]), cell, 1.0)
        assert float((E - shifted).abs().max()) < 1e-11
        wrapped = positions.clone()
        wrapped[3] += cell[0] - cell[2]
        assert float((E - ew.ewald_matrix(wrapped, cell, 1.0)).abs().max()) < 1e-11
        perm = torch.randperm(positions.shape[0], generator=torch.Generator().manual_seed(1))
        Ep = ew.ewald_matrix(positions[perm], cell, 1.0)
        assert float((Ep - E[perm][:, perm]).abs().max()) < 1e-11

    def test_rotation(self):
        positions, cell, q = _frame(seed=12)
        R = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(2)))[0]
        E = ew.ewald_matrix(positions, cell, 1.0)
        Er = ew.ewald_matrix(positions @ R.T, cell @ R.T, 1.0)
        assert float((E - Er).abs().max()) < 1e-10


class TestRegimeBLatticeSum:
    def test_short_range_sum_equals_the_difference_of_two_ewald_matrices(self):
        """`sum_L [erf(r/(2 r_g)) - erf(r/r_s)]/r` over images is `E_PBC(r_g) - E_PBC(r_s/2)`
        (the broad kernel `erf(r/r_s)/r` is the pair kernel of two Gaussians of width
        `r_s/2`), self term included: `1/(sqrt(pi) r_g) - 2/(sqrt(pi) r_s)`."""
        positions, cell, _ = _frame(seed=5)
        r_g, r_s = 0.8, 3.0

        def s_of(r):
            return ew.COULOMB * (torch.erf(r / (2 * r_g)) - torch.erf(r / r_s)) / r

        self_value = ew.COULOMB * (1 / (math.sqrt(math.pi) * r_g) - 2 / (math.sqrt(math.pi) * r_s))
        K = ew.short_range_lattice_sum(positions, cell, s_of, self_value, r_c=30.0)
        diff = ew.ewald_matrix(positions, cell, r_g) - ew.ewald_matrix(positions, cell, r_s / 2)
        assert float((K - diff).abs().max()) < 1e-10
        # Converged in the image range and rewrapping-invariant.
        K2 = ew.short_range_lattice_sum(positions, cell, s_of, self_value, r_c=40.0)
        assert float((K - K2).abs().max()) < 1e-12
        wrapped = positions.clone(); wrapped[2] -= cell[1]
        K3 = ew.short_range_lattice_sum(wrapped, cell, s_of, self_value, r_c=30.0)
        assert float((K - K3).abs().max()) < 1e-12
