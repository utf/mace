"""Edit 4's toy tables, written BEFORE the counting head is ever trained.

The standing rule in this project is that a sign or a convention ships with a case whose
answer is known independently of the code that computes it. Edit 4 introduces three new ones
at once -- the Slater-Koster angular table, the electron-counting map, and the free-energy
difference -- and each is the kind of thing that produces a plausible number when wrong.

1. An **s-only dimer**: the two-level problem, where the answer is `eps +- |t|` on paper.
2. An **sp dimer with a node check**: the bonding combination must have no node between the
   atoms and the antibonding one must have exactly one, which is what fixes the relative sign
   of `V_sp_sigma` in the table.
3. A **V_Cl toy**: removing an electron from the top of a bonding manifold must push the two
   flanking atoms APART, and more strongly as they approach -- the hole force, which is the
   quantity the whole programme is about and the one the previous head could not represent.

Plus the counting arithmetic, which the plan states as a checkable table, and the exact
vanishing at n = 0.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_counting import (ORBITALS_PER_ATOM, VALENCE, density_matrix,
                                          fermi_fill, free_energy, head_energy,
                                          neutral_electrons, site_charges, sk_block,
                                          spin_targets)

torch.set_default_dtype(torch.float64)


def dimer_h(t_ss=-1.0, eps_a=0.0, eps_b=0.0):
    """Two s orbitals, one bond. H = [[eps_a, t], [t, eps_b]]."""
    return torch.tensor([[eps_a, t_ss], [t_ss, eps_b]])


# --------------------------------------------------------------------------- table 1


class TestSOnlyDimer:
    def test_levels_are_eps_plus_minus_t(self):
        t = -1.3
        lam = torch.linalg.eigvalsh(dimer_h(t_ss=t))
        assert float(lam[0]) == pytest.approx(-abs(t))
        assert float(lam[1]) == pytest.approx(+abs(t))

    def test_the_ground_state_is_nodeless(self):
        """Perron-Frobenius on the bonding-signed matrix: the lowest state has no sign
        change. If it did, the 'bonding' label would be wrong and every force built on it
        would have the wrong sign."""
        _, vec = torch.linalg.eigh(dimer_h(t_ss=-1.0))
        ground = vec[:, 0]
        assert float(ground[0] * ground[1]) > 0

    def test_sk_block_reduces_to_the_two_level_problem(self):
        """The 4x4 SK block's s-s entry must be exactly V_ss_sigma, independent of
        direction -- an s orbital has no angular dependence."""
        for direction in ([1.0, 0, 0], [0, 1.0, 0], [0.577, 0.577, 0.577]):
            d = torch.tensor([direction])
            d = d / d.norm()
            v = torch.tensor([[-1.3, 0.0, 0.0, 0.0]])
            block = sk_block(d, v)
            assert float(block[0, 0, 0]) == pytest.approx(-1.3)


# --------------------------------------------------------------------------- table 2


class TestSPDimerNodeCheck:
    @staticmethod
    def build(v_sp=1.84, eps_s=-2.0, eps_p=0.0, axis=(1.0, 0.0, 0.0)):
        """Two atoms along `axis`, s and p only, no ss or pp coupling -- so the only thing
        under test is the sp-sigma entry and its sign."""
        d = torch.tensor([list(axis)])
        d = d / d.norm()
        v = torch.tensor([[0.0, float(v_sp), 0.0, 0.0]])
        block = sk_block(d, v)[0]                       # [4, 4], rows on i, cols on j
        H = torch.zeros(8, 8)
        H[:4, 4:] = block
        H[4:, :4] = block.transpose(0, 1)
        on = torch.tensor([eps_s, eps_p, eps_p, eps_p])
        H[:4, :4] += torch.diag(on)
        H[4:, 4:] += torch.diag(on)
        return H

    def test_the_reversed_edge_is_the_transpose(self):
        """Hermiticity by construction. E_{s,x} = +l V and E_{x,s} = -l V, so evaluating the
        block on -d must give the transpose -- otherwise H is only symmetric after a
        symmetrisation, and eigh's gradients are silently wrong during the backward pass."""
        d = torch.tensor([[0.3, -0.5, 0.81]])
        d = d / d.norm()
        v = torch.tensor([[-1.3, 1.84, 3.24, -0.81]])
        forward = sk_block(d, v)[0]
        reverse = sk_block(-d, v)[0]
        assert torch.allclose(forward, reverse.transpose(0, 1), atol=1e-12)

    @staticmethod
    def sign_changes_between_the_nuclei(coeffs, separation=2.8, n_grid=400):
        """Count nodes of the wavefunction ON the bond axis, between the two nuclei.

        Counting sign changes of the s COEFFICIENTS is the obvious test and it is wrong: with
        sp coupling only, the Hamiltonian decouples into two 2x2 blocks and the lowest state
        has exactly zero s amplitude on one atom, so the product is 0 and says nothing. A p
        orbital also carries its own sign structure, which a coefficient comparison cannot
        see. So evaluate the actual function: s is even about its centre, p_x is odd.
        """
        x = np.linspace(0.0, separation, n_grid)
        c = np.asarray([float(v) for v in coeffs])

        def orbital(u, kind):
            radial = np.exp(-np.abs(u))
            return radial if kind == "s" else u * radial

        psi = (c[0] * orbital(x, "s") + c[1] * orbital(x, "p")
               + c[4] * orbital(x - separation, "s") + c[5] * orbital(x - separation, "p"))
        interior = psi[5:-5]
        return psi, int(np.sum(np.sign(interior[1:]) != np.sign(interior[:-1])))

    def test_the_bonding_state_piles_density_between_the_nuclei(self):
        """What fixes the RELATIVE sign of V_sp_sigma. Reverse it and the two states swap:
        every bonding/antibonding statement downstream inverts while the eigenvalues stay
        identical, so no spectrum check would notice.

        Read off the {s_i, p_x_j} BLOCK, not the full 8x8. With sp coupling alone the
        Hamiltonian decouples into two such blocks related by swapping the atoms, so they are
        exactly degenerate and `eigh` returns an arbitrary member of each degenerate pair --
        comparing those is comparing two arbitrary rotations, which is the hazard this whole
        head is built to avoid. The block is where the sign convention actually lives.
        """
        # eps_s == eps_p on purpose. With the levels split, the lower state is simply
        # s-dominated and its density piles on that ATOM -- a polarity effect that swamps the
        # bonding one and has nothing to do with the sign under test. Degenerate levels give
        # |c_s| == |c_p|, so the relative sign is the only thing left.
        H = self.build(eps_s=0.0, eps_p=0.0)
        # rows/cols 0 = s on atom i, 5 = p_x on atom j.
        block = torch.tensor([[H[0, 0], H[0, 5]], [H[5, 0], H[5, 5]]])
        assert float(block[0, 1].abs()) > 0.1, "the sp coupling is not in this block"
        _, vec = torch.linalg.eigh(block)

        def midbond_density(c_s, c_p, separation=2.8, n=1200):
            # Normalise over ALL space, not over the bond segment: restricting the
            # denominator makes a state that piles charge on one nucleus look delocalised.
            x = np.linspace(-4.0, separation + 4.0, n)
            psi = (c_s * np.exp(-np.abs(x))
                   + c_p * (x - separation) * np.exp(-np.abs(x - separation)))
            mid = (x > separation / 3.0) & (x < 2.0 * separation / 3.0)
            return float(np.sum(psi[mid] ** 2) / np.sum(psi ** 2))

        low = midbond_density(float(vec[0, 0]), float(vec[1, 0]))
        high = midbond_density(float(vec[0, 1]), float(vec[1, 1]))
        assert low > high, (
            f"the lower state has LESS density between the nuclei ({low:.3f}) than the "
            f"upper ({high:.3f}) -- the sp-sigma sign is inverted")

    def test_the_two_sp_blocks_are_degenerate(self):
        """Recorded rather than worked around: with sp coupling alone the extremal states of
        the full 8x8 are degenerate, which is why the test above reads the block."""
        H = self.build()
        lam = torch.linalg.eigvalsh(H)
        assert float(lam[0]) == pytest.approx(float(lam[1]), abs=1e-12)
        assert float(lam[-1]) == pytest.approx(float(lam[-2]), abs=1e-12)

    def test_the_p_orbital_along_the_bond_is_the_one_that_couples(self):
        """Along x, only p_x mixes with s. p_y and p_z must be untouched -- if they move,
        the angular factors are wrong and every anisotropy downstream is fictitious."""
        H = self.build(axis=(1.0, 0.0, 0.0))
        assert float(H[0, 6].abs()) < 1e-12          # s(i) - p_y(j)
        assert float(H[0, 7].abs()) < 1e-12          # s(i) - p_z(j)
        assert float(H[0, 5].abs()) > 0.1            # s(i) - p_x(j)


# --------------------------------------------------------------------------- table 3


class TestVacancyHoleForce:
    """The hole's OWN contribution to the hub-hub force is outward and steepens as d closes.

    Stated carefully, because the loose version is false and writing this test is what showed
    it. Perron-Frobenius already established that the bonding-signed H has a nodeless ground
    state, hence `beta_ab >= 0`, hence the hub-hub force is inward FOR ANY OCCUPANCY. So "the
    hole pushes the pair apart" cannot mean the total force reverses -- it does not.

    What it does mean, and what the counting head has to reproduce, is a statement about the
    DIFFERENCE from the filled reference: removing an electron from the bonding pair halves
    the bond order, so `E(hole) - E(filled)` falls as the pair separates. That difference is
    exactly `E_head`, and it is the quantity the labels supervise.
    """

    @staticmethod
    def energy_at(d, n_hole=0):
        """Two flanking orbitals across the vacancy, coupling decaying with separation.

        ONE SPIN CHANNEL, so the two-site model holds two levels and the neutral vacancy puts
        ONE electron in the bonding one. Filling both levels -- the obvious reading of "two
        electrons" -- gives `F = eps_bonding + eps_antibonding = 0` identically, a flat energy
        with no force at all, which is how this test first failed.
        """
        t = -1.0 * float(np.exp(-(d - 2.8) / 1.0))
        H = torch.tensor([[0.0, t], [t, 0.0]])
        lam = torch.linalg.eigvalsh(H)
        return free_energy(lam, 1.0 - n_hole, t_el=0.025)

    def slope(self, d, n_hole, h=1e-4):
        return float((self.energy_at(d + h, n_hole) - self.energy_at(d - h, n_hole))
                     / (2 * h))

    def test_the_total_force_stays_inward_at_every_occupancy(self):
        """The Perron-Frobenius statement, checked rather than cited. The empty channel is
        flat, not repulsive -- it contributes no force at all, which is the boundary case."""
        for d in (4.5, 5.6, 6.5):
            assert self.slope(d, 0) > 0.0, f"the filled pair is not bound at d={d}"
            assert self.slope(d, 1) >= 0.0, f"the total force reversed at d={d}"

    def test_the_hole_contribution_is_outward(self):
        """`d/dd [E(hole) - E(filled)] < 0`: the hole term alone favours separation, which is
        what "the hole pushes them apart" means once the total is held fixed."""
        for d in (4.5, 5.6, 6.5):
            assert self.slope(d, 1) - self.slope(d, 0) < 0.0, (
                f"the hole's own contribution is inward at d={d}")

    def test_the_hole_contribution_steepens_as_the_pair_closes(self):
        """|d/dd [E(hole) - E(filled)]| must grow at short d -- the coupling is exponential,
        so the force it produces is too. A FLAT force is the prior-dominated signature M3
        measured, and it is what this head has to avoid reproducing."""
        magnitudes = [abs(self.slope(d, 1) - self.slope(d, 0)) for d in (4.5, 5.6, 6.5)]
        assert magnitudes[0] > magnitudes[1] > magnitudes[2]


# --------------------------------------------------------------------------- counting


class TestElectronCounting:
    def test_the_plan_s_arithmetic(self):
        """The three cases the plan states, checked rather than trusted."""
        pristine = [55] * 16 + [82] * 16 + [17] * 48
        assert neutral_electrons(pristine) == 416
        assert spin_targets(416, (0, 0, 0, 0)) == (208.0, 208.0)

        neutral_vacancy = [55] * 16 + [82] * 16 + [17] * 47
        assert neutral_electrons(neutral_vacancy) == 409
        # The doublet reference is automatic: nothing was asked of the caller.
        assert spin_targets(409, (0, 0, 0, 0)) == (205.0, 204.0)

        # V_Cl+: one electron short of the neutral vacancy, taken from the majority channel.
        assert spin_targets(409, (0, 0, 1, 0)) == (204.0, 204.0)

    def test_orbital_count_matches_the_plan(self):
        assert 80 * ORBITALS_PER_ATOM == 320
        assert VALENCE == {55: 1, 82: 4, 17: 7}

    def test_an_electron_and_a_hole_move_opposite_ways(self):
        assert spin_targets(100, (1, 0, 0, 0))[0] == 51.0
        assert spin_targets(100, (0, 0, 1, 0))[0] == 49.0
        assert spin_targets(100, (0, 1, 0, 0))[1] == 51.0
        assert spin_targets(100, (0, 0, 0, 1))[1] == 49.0


class TestFreeEnergy:
    def test_the_fill_hits_the_requested_electron_count(self):
        eps = torch.linspace(-5.0, 5.0, 40)
        for n in (1.0, 7.5, 20.0, 39.0):
            assert float(fermi_fill(eps, n).sum()) == pytest.approx(n, abs=1e-6)

    def test_the_correction_vanishes_exactly_at_zero_counters(self):
        """Not approximately: the reference is the same spectrum at the neutral fill, so the
        two terms are identical expressions and cancel bit for bit."""
        torch.manual_seed(0)
        eps_a = torch.randn(30).sort().values
        eps_b = torch.randn(30).sort().values
        e = head_energy(eps_a, eps_b, 30, (0, 0, 0, 0))
        assert float(e) == 0.0

    def test_removing_an_electron_from_a_filled_manifold_raises_the_energy(self):
        eps = torch.linspace(-5.0, -1.0, 20)
        filled = free_energy(eps, 20.0)
        with_hole = free_energy(eps, 19.0)
        assert float(with_hole) > float(filled)

    def test_gradients_flow_to_the_eigenvalues_and_not_through_mu(self):
        """Hellmann-Feynman: dF/deps_k = f_k exactly. If mu were differentiated through, this
        would pick up an extra term and the identity would fail."""
        eps = torch.linspace(-3.0, 3.0, 16).requires_grad_(True)
        f = fermi_fill(eps, 8.0).detach()
        free_energy(eps, 8.0).backward()
        assert torch.allclose(eps.grad, f, atol=1e-8)


class TestDensityMatrix:
    def test_the_monopole_is_exactly_the_carrier_count(self):
        """sum_i q_i = -Delta n, with the sign carried by P natively -- there is no sgn(q_c)
        anywhere for a future edit to get wrong."""
        torch.manual_seed(1)
        n_nodes = 5
        dim = n_nodes * ORBITALS_PER_ATOM
        H = torch.randn(dim, dim)
        H = 0.5 * (H + H.T)
        lam, psi = torch.linalg.eigh(H)
        p_ref = density_matrix(psi, fermi_fill(lam, 10.0))
        p_now = density_matrix(psi, fermi_fill(lam, 11.0))
        q = site_charges(p_now, p_ref, n_nodes)
        assert float(q.sum()) == pytest.approx(-1.0, abs=1e-6)

    def test_it_is_invariant_under_a_rotation_inside_a_degenerate_subspace(self):
        """The reason nothing here backpropagates through individual eigenvectors: they are
        only defined up to this rotation, and a defect level near the continuum is exactly
        the degenerate case."""
        torch.manual_seed(2)
        dim = 8
        psi, _ = torch.linalg.qr(torch.randn(dim, dim))
        f = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        p = density_matrix(psi, f)
        theta = 0.7
        rot = torch.eye(dim)
        rot[0, 0] = rot[1, 1] = np.cos(theta)
        rot[0, 1], rot[1, 0] = -np.sin(theta), np.sin(theta)
        p_rot = density_matrix(psi @ rot, f)
        assert torch.allclose(p, p_rot, atol=1e-12)
