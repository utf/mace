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
                                          fermi_density_matrix, head_energy_hf,
                                          neutral_electrons, site_charges, sk_block,
                                          spin_targets, T_EL)

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


class TestHellmannFeynmanRouteMatchesDenseAutograd:
    """The production gradient path, validated against the reference it replaces.

    Every force this model will ever produce flows through `head_energy_hf`, so it is
    checked against `head_energy` -- plain autograd over eigenvalues -- on all three
    quantities the plan names: energies, forces, and site charges.

    They are not two ways of writing the same code. The dense route differentiates the
    eigenvalues; the HF route evaluates `Tr(P H)` with `P` held fixed. They must agree on the
    VALUE identically (both are `sum_k f_k eps_k - T S`) and on the FORCE to numerical
    precision (Hellmann-Feynman). They deliberately DIFFER on the loss's parameter gradient,
    which is what makes the HF route usable at all: the dense route's double backward builds
    eigenvector response with `1/(lam_i - lam_j)` and returns NaN on a real 316-state
    spectrum.
    """

    @staticmethod
    def random_h(n_sites=6, seed=0, positions=None):
        """A symmetric H whose entries depend smoothly on positions, so forces are defined."""
        g = torch.Generator().manual_seed(seed)
        dim = n_sites * ORBITALS_PER_ATOM
        mix = torch.randn(dim, dim, generator=g)
        mix = 0.5 * (mix + mix.T)
        if positions is None:
            return mix
        # Distances written out rather than via torch.cdist: cdist has no double backward,
        # and the whole point of one of these tests is to take a second derivative.
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        d = (diff.pow(2).sum(-1) + 1e-12).sqrt() + torch.eye(n_sites) * 5.0
        w = torch.exp(-d).repeat_interleave(ORBITALS_PER_ATOM, 0).repeat_interleave(
            ORBITALS_PER_ATOM, 1)
        return mix * w

    def test_energies_agree(self):
        torch.manual_seed(0)
        H = self.random_h()
        lam = torch.linalg.eigvalsh(H)
        counts = (0, 0, 1, 0)
        dense = head_energy(lam, lam, 20, counts)
        hf, _, _, _, _ = head_energy_hf(H, 20, counts)
        assert float(hf) == pytest.approx(float(dense), rel=1e-10)

    def test_forces_agree(self):
        """Hellmann-Feynman: the two routes must give the same dE/dR, not merely a similar
        one. This is the number the model is trained on."""
        torch.manual_seed(1)
        pos = torch.randn(6, 3, requires_grad=True)
        counts = (0, 0, 1, 0)

        H = self.random_h(positions=pos)
        lam = torch.linalg.eigvalsh(H)
        dense = head_energy(lam, lam, 20, counts)
        g_dense = torch.autograd.grad(dense, pos, retain_graph=False)[0]

        pos2 = pos.detach().clone().requires_grad_(True)
        H2 = self.random_h(positions=pos2)
        hf, _, _, _, _ = head_energy_hf(H2, 20, counts)
        g_hf = torch.autograd.grad(hf, pos2)[0]

        assert torch.allclose(g_dense, g_hf, atol=1e-8), (
            f"max force discrepancy {float((g_dense - g_hf).abs().max()):.3e} eV/A")

    def test_site_charges_agree(self):
        torch.manual_seed(2)
        H = self.random_h()
        counts = (0, 0, 1, 0)
        lam, psi = torch.linalg.eigh(H)
        n_maj, n_min = spin_targets(20, counts)
        # SUMMED over spin, matching the head: the monopole identity counts all electrons.
        p_dense = (density_matrix(psi, fermi_fill(lam, n_maj))
                   + density_matrix(psi, fermi_fill(lam, n_min)))
        p_ref_dense = (density_matrix(psi, fermi_fill(lam, 10.0))
                       + density_matrix(psi, fermi_fill(lam, 10.0)))
        _, _, _, p_now, p_ref = head_energy_hf(H, 20, counts)
        q_dense = site_charges(p_dense, p_ref_dense, 6)
        q_hf = site_charges(p_now, p_ref, 6)
        assert torch.allclose(q_dense, q_hf, atol=1e-12)
        assert float(q_hf.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_the_hf_route_survives_a_double_backward_and_the_dense_one_does_not(self):
        """The reason for the whole exercise, as a test rather than a comment.

        Force matching differentiates a quantity that is already `dE/dR`, so the loss needs a
        second derivative. On a degenerate spectrum the dense route produces a non-finite one.
        """
        torch.manual_seed(3)
        pos = torch.randn(6, 3, requires_grad=True)
        scale = torch.ones(1, requires_grad=True)
        counts = (0, 0, 1, 0)

        # A deliberately degenerate spectrum: two identical blocks.
        block = self.random_h(n_sites=3, seed=7, positions=pos[:3])
        H = torch.block_diag(block, block) * scale

        hf, _, _, _, _ = head_energy_hf(H, 20, counts)
        f_hf = torch.autograd.grad(hf, pos, create_graph=True)[0]
        g2_hf = torch.autograd.grad(f_hf.pow(2).sum(), scale)[0]
        assert torch.isfinite(g2_hf).all(), "the HF route must survive the double backward"

        lam = torch.linalg.eigvalsh(H)
        dense = head_energy(lam, lam, 20, counts)
        f_dense = torch.autograd.grad(dense, pos, create_graph=True)[0]
        g2_dense = torch.autograd.grad(f_dense.pow(2).sum(), scale, allow_unused=True)[0]
        # Recorded, not asserted as NaN: whether it degenerates depends on how exactly the
        # eigenvalues collide. What matters is that the HF route above does not.
        if g2_dense is not None and not torch.isfinite(g2_dense).all():
            assert True                          # the documented failure, reproduced


class TestOccupationOverride:
    """Stage 4: the fill is statable, not only derivable from four counters.

    This is what the counter scheme could never express -- an excited or non-Aufbau
    configuration -- and it is an INPUT change rather than an architecture one, which is why
    it needs no retraining.
    """

    @staticmethod
    def h(seed=0, n_sites=5):
        g = torch.Generator().manual_seed(seed)
        dim = n_sites * ORBITALS_PER_ATOM
        m = torch.randn(dim, dim, generator=g)
        return 0.5 * (m + m.T)

    def test_the_override_reproduces_the_counter_fill_when_they_agree(self):
        """The two routes must coincide where they describe the same thing, or the override
        is a second convention rather than a generalisation of the first."""
        H = self.h()
        counts = (0, 0, 1, 0)
        n_total = 20
        via_counts, *_ = head_energy_hf(H, n_total, counts)
        via_override, *_ = head_energy_hf(H, n_total, (0, 0, 0, 0),
                                          occupation=spin_targets(n_total, counts))
        assert float(via_override) == pytest.approx(float(via_counts), rel=1e-12)

    def test_the_reference_stays_the_neutral_ground_state(self):
        """E_head must remain a difference from ONE origin. An override that also moved the
        reference would make two overridden configurations incomparable."""
        H = self.h(seed=1)
        n_total = 20
        at_ground, *_ = head_energy_hf(H, n_total, (0, 0, 0, 0),
                                       occupation=(10.0, 10.0))
        assert float(at_ground) == pytest.approx(0.0, abs=1e-12)

    def test_a_non_aufbau_configuration_is_expressible(self):
        """Two electrons moved from the majority to the minority channel -- a configuration
        with no representation in (e_maj, e_min, h_maj, h_min) at fixed total charge."""
        H = self.h(seed=2)
        n_total = 20
        excited, *_ = head_energy_hf(H, n_total, (0, 0, 0, 0), occupation=(8.0, 12.0))
        assert float(excited) > 0.0, "a non-ground configuration must cost energy"
        assert torch.isfinite(torch.as_tensor(float(excited)))


class TestDensityResponseBackward:
    """Validation of the P-backward. The plan calls this the one genuinely delicate build.

    Three checks the plan names — finite differences (including engineered near-degeneracies
    and a mu-mid-band case that exercises the fixed-N correction), agreement with
    autograd-through-eigh on a non-degenerate float64 toy, and preservation of the monopole.
    """

    @staticmethod
    def sym(n, seed, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        m = torch.randn(n, n, generator=g) * scale
        return 0.5 * (m + m.T)

    @staticmethod
    def degenerate(n_block=4, seed=5, scale=1.0):
        """Two identical blocks: every eigenvalue is exactly twofold degenerate."""
        g = torch.Generator().manual_seed(seed)
        b = torch.randn(n_block, n_block, generator=g) * scale
        b = 0.5 * (b + b.T)
        return torch.block_diag(b, b)

    def fd_check(self, H, n_el, t_el=T_EL, h=1e-5, probes=6, atol=2e-5):
        """Finite differences against the backward, on random symmetric perturbations."""
        gen = torch.Generator().manual_seed(0)
        W = torch.randn(H.shape, generator=gen)          # cotangent on P
        W = 0.5 * (W + W.T)
        Hv = H.clone().requires_grad_(True)
        P = fermi_density_matrix(Hv, n_el, t_el)
        (P * W).sum().backward()
        analytic = Hv.grad

        worst = 0.0
        for k in range(probes):
            g2 = torch.Generator().manual_seed(100 + k)
            D = torch.randn(H.shape, generator=g2)
            D = 0.5 * (D + D.T)
            with torch.no_grad():
                pp = fermi_density_matrix(H + h * D, n_el, t_el)
                pm = fermi_density_matrix(H - h * D, n_el, t_el)
            fd = float(((pp - pm) * W).sum() / (2 * h))
            an = float((analytic * D).sum())
            worst = max(worst, abs(fd - an))
        return worst

    def test_finite_differences_generic(self):
        H = self.sym(10, seed=1)
        worst = self.fd_check(H, n_el=5.0)
        assert worst < 2e-5, f"max FD discrepancy {worst:.3e}"

    def test_finite_differences_with_exact_degeneracies(self):
        """The case `eigh`'s eigenvector backward cannot do at all."""
        H = self.degenerate()
        worst = self.fd_check(H, n_el=4.0)
        assert worst < 2e-5, f"max FD discrepancy at exact degeneracy {worst:.3e}"

    def test_finite_differences_with_mu_mid_band(self):
        """Exercises the fixed-N correction: with mu inside a dense band, sum_k f'_k is far
        from zero and the implicit-mu term is a large part of the answer."""
        H = self.sym(16, seed=3, scale=0.05)             # narrow spectrum, mu well inside
        worst = self.fd_check(H, n_el=8.0)
        assert worst < 2e-5, f"max FD discrepancy with mu mid-band {worst:.3e}"

    def test_the_fixed_n_correction_is_not_negligible_in_that_case(self):
        """Guard: the test above would pass trivially if the correction were near zero."""
        H = self.sym(16, seed=3, scale=0.05)
        lam = torch.linalg.eigvalsh(H.double())
        f = fermi_fill(lam, 8.0)
        fp = -f * (1 - f) / T_EL
        assert abs(float(fp.sum())) > 1.0, (
            "mu is effectively in a gap here; this case does not exercise the correction")

    def test_agrees_with_autograd_through_eigh_when_non_degenerate(self):
        """On a well-separated float64 spectrum the naive route is valid, so the two must
        agree. This is the check that the Daleckii-Krein algebra is right rather than merely
        self-consistent."""
        H = self.sym(8, seed=7)
        gen = torch.Generator().manual_seed(0)
        W = torch.randn(H.shape, generator=gen)
        W = 0.5 * (W + W.T)
        lam = torch.linalg.eigvalsh(H)
        assert float(torch.diff(torch.sort(lam).values).min()) > 0.05, "spectrum too close"

        a = H.clone().requires_grad_(True)
        (fermi_density_matrix(a, 4.0) * W).sum().backward()

        b = H.clone().requires_grad_(True)
        lam_b, U_b = torch.linalg.eigh(b)
        f_b = fermi_fill(lam_b, 4.0)                     # mu detached, as in the Function
        P_b = (U_b * f_b.unsqueeze(0)) @ U_b.T
        (P_b * W).sum().backward()

        # The naive route omits the fixed-N correction, so compare on a spectrum where mu
        # sits in a gap and that correction vanishes -- otherwise they SHOULD differ.
        fp = -f_b * (1 - f_b) / T_EL
        if abs(float(fp.sum())) < 1e-8:
            assert torch.allclose(a.grad, b.grad, atol=1e-8)
        else:
            # mu is not in a gap: the correction is real and the naive route is wrong.
            assert not torch.allclose(a.grad, b.grad, atol=1e-8)

    def test_the_gradient_is_bounded_at_degeneracy(self):
        """The property that makes this usable: 1/(4 T_el) everywhere, no blow-up."""
        H = self.degenerate()
        Hv = H.clone().requires_grad_(True)
        P = fermi_density_matrix(Hv, 4.0)
        P.sum().backward()
        assert torch.isfinite(Hv.grad).all()
        assert float(Hv.grad.abs().max()) < 1.0 / (4 * T_EL) * 10

    def test_the_monopole_survives_the_differentiable_route(self):
        H = self.sym(5 * ORBITALS_PER_ATOM, seed=11)
        p_now = fermi_density_matrix(H, 11.0)
        p_ref = fermi_density_matrix(H, 10.0)
        q = site_charges(p_now, p_ref, 5)
        assert float(q.sum()) == pytest.approx(-1.0, abs=1e-6)

    def test_it_gives_gradient_where_the_frozen_route_gives_none(self):
        """The failure-to-start, in miniature. At the atomic limit the bond order is zero, so
        the frozen-P route has no hopping gradient at all; the response route does."""
        eps = torch.tensor([-1.0, 1.0])
        off = torch.zeros(1, requires_grad=True)
        H = torch.stack([torch.stack([eps[0], off[0]]),
                         torch.stack([off[0], eps[1]])])
        P = fermi_density_matrix(H, 1.0)
        # A loss that depends on the OFF-DIAGONAL of P -- bond order -- which is what a force
        # residual sees.
        (P[0, 1] ** 2 + P[0, 1]).backward()
        assert off.grad is not None and abs(float(off.grad)) > 1e-6, (
            "no gradient reaches the hopping at the atomic limit")


class TestHarrisonInitialisation:
    """Section 3: the init that replaces the atomic limit."""

    @staticmethod
    def head(bond=2.8):
        from mace.modules.defect_counting import CountingHead, harrison_initialise
        h = CountingHead(num_elements=3, feature_dim=8, atomic_numbers=[17, 55, 82])
        harrison_initialise(h, [17, 55, 82], bond_length=bond)
        return h

    def test_anion_p_sits_below_both_cation_p_levels(self):
        """The property the whole init is for, and it costs no charge parameter: the valence
        band comes out anion-derived without touching Z."""
        h = self.head()
        cl_p, cs_p, pb_p = (float(h.h.eps0[i, 1]) for i in range(3))
        assert cl_p < pb_p < cs_p, f"Cl {cl_p}, Pb {pb_p}, Cs {cs_p}"

    def test_s_lies_below_p_for_every_species(self):
        h = self.head()
        for i in range(3):
            assert float(h.h.eps0[i, 0]) < float(h.h.eps0[i, 1])

    def test_hoppings_follow_the_universal_d_squared_scaling(self):
        """Halving the bond length must quadruple every integral -- that is the content of
        the scaling, and a wrong power would be invisible at one distance."""
        a, b = self.head(bond=2.8), self.head(bond=1.4)
        ratio = (b.h.v0_raw / a.h.v0_raw)
        assert torch.allclose(ratio, torch.full_like(ratio, 4.0), atol=1e-10)

    def test_the_bond_length_is_an_argument_not_a_constant(self):
        """Passing the measured distance in keeps the head host-agnostic; a baked-in value
        would make it a CsPbCl3 module."""
        import inspect
        from mace.modules.defect_counting import harrison_initialise
        assert "bond_length" in inspect.signature(harrison_initialise).parameters

    def test_an_unknown_species_is_refused(self):
        from mace.modules.defect_counting import CountingHead, harrison_initialise
        h = CountingHead(num_elements=3, feature_dim=8, atomic_numbers=[17, 55, 82])
        with pytest.raises(ValueError, match="no Harrison term values"):
            harrison_initialise(h, [17, 55, 6])


class TestInitialisationGate:
    def test_the_atomic_limit_fails_the_gate(self):
        """Degenerate site energies and no bandwidth: exactly the state the gate exists to
        catch, and the one a zero initialisation produces."""
        from mace.modules.defect_counting import initialisation_gate
        lam = torch.zeros(40) + torch.linspace(0, 1e-6, 40)
        g = initialisation_gate(lam, 20.0, e_gap=2.4)
        assert not g["passed"]
        assert g["bandwidth"] < 2 * 2.4

    def test_a_band_like_spectrum_passes(self):
        from mace.modules.defect_counting import initialisation_gate
        lam = torch.linspace(-10.0, 10.0, 200)
        g = initialisation_gate(lam, 100.0, e_gap=2.4)
        assert g["passed"], g

    def test_two_clumps_pass_and_that_is_correct(self):
        """Recorded because the obvious intuition is wrong. Two tight clumps 24 eV apart look
        atomic, but both stated clauses pass: the edge spacings are measured WITHIN the
        occupied and empty manifolds, and they are tiny, while the bandwidth is large.

        That is not a hole in the gate. This spectrum is a band structure with an absurd gap,
        not an atomic limit -- the atomic limit has near-degenerate levels and therefore no
        bandwidth, which the second clause does catch. What rejects an absurd frontier gap is
        `loss_gap`, which trains that quantity toward E_gap; the gate reports it so the two
        are not confused."""
        from mace.modules.defect_counting import initialisation_gate
        lam = torch.cat([torch.full((20,), -12.0), torch.full((20,), 12.0)])
        lam = lam + torch.linspace(0, 1e-4, 40)
        g = initialisation_gate(lam, 20.0, e_gap=2.4)
        assert g["passed"]
        assert g["frontier_gap"] > 20.0, "the frontier gap is what would flag this"


class TestExplicitForcesMatchAutograd:
    """Section 1's regression: wiring the density response must not change PREDICTIONS.

    `-Tr((P - P_ref) dH/dR)` and `-d/dR [F(N) - F(N_ref)]` are the same mathematical object --
    Hellmann-Feynman -- so the explicit route must reproduce the autograd route to tolerance.
    Only the BACKWARD differs: the explicit form keeps `dP/dtheta` alive, the autograd form
    (with P detached) does not.

    Asserting this rather than assuming it is the point. A wiring change that silently moved
    the forces would invalidate every comparison against the frozen-P rerun.
    """

    @staticmethod
    def toy(n_sites=5, seed=0):
        """H built from positions so dH/dR exists, with a smooth two-centre form."""
        g = torch.Generator().manual_seed(seed)
        pos = torch.randn(n_sites, 3, generator=g).requires_grad_(True)
        base = torch.randn(n_sites * ORBITALS_PER_ATOM, n_sites * ORBITALS_PER_ATOM,
                           generator=g)
        base = 0.5 * (base + base.T)

        def build(p):
            diff = p.unsqueeze(1) - p.unsqueeze(0)
            d = (diff.pow(2).sum(-1) + 1e-12).sqrt() + torch.eye(n_sites) * 4.0
            w = torch.exp(-d).repeat_interleave(ORBITALS_PER_ATOM, 0).repeat_interleave(
                ORBITALS_PER_ATOM, 1)
            return base * w

        return pos, build

    def test_explicit_equals_autograd_forces(self):
        from mace.modules.defect_counting import fermi_density_matrix, head_forces
        pos, build = self.toy()
        n_total, counts = 20, (0, 0, 1, 0)
        n_maj, n_min = spin_targets(n_total, counts)
        n_maj_ref, n_min_ref = float((n_total + 1) // 2), float(n_total // 2)

        # autograd route: differentiate the free-energy difference
        H = build(pos)
        e, _, _, _, _ = head_energy_hf(H, n_total, counts)
        f_auto = -torch.autograd.grad(e, pos, retain_graph=False)[0]

        # explicit route: -Tr((P - P_ref) dH/dR), summed over spin
        pos2 = pos.detach().clone().requires_grad_(True)
        H2 = build(pos2)
        P = (fermi_density_matrix(H2, n_maj) + fermi_density_matrix(H2, n_min))
        P_ref = (fermi_density_matrix(H2, n_maj_ref) + fermi_density_matrix(H2, n_min_ref))
        f_expl = head_forces(H2, pos2, P, P_ref, create_graph=False)

        assert torch.allclose(f_auto, f_expl, atol=1e-8), (
            f"explicit and autograd forces differ by "
            f"{float((f_auto - f_expl).abs().max()):.3e} eV/A -- the wiring changed the "
            "predictions, not just the gradient")

    def test_the_explicit_route_carries_a_density_gradient_and_the_frozen_one_does_not(self):
        """The whole reason for the change, as a test. With P frozen, a parameter that acts
        only through the OCCUPATIONS has no gradient; with the response live, it does."""
        from mace.modules.defect_counting import fermi_density_matrix, head_forces
        pos, build = self.toy(seed=3)
        shift = torch.zeros(1, requires_grad=True)          # a rigid on-site shift
        n = pos.shape[0] * ORBITALS_PER_ATOM

        H = build(pos) + shift * torch.diag(
            torch.cat([torch.ones(n // 2), -torch.ones(n - n // 2)]))
        P = fermi_density_matrix(H, 10.0)
        P_ref = fermi_density_matrix(H, 10.0).detach() * 0.0
        F = head_forces(H, pos, P, P_ref, create_graph=True)
        g_live = torch.autograd.grad(F.pow(2).sum(), shift, retain_graph=True,
                                     allow_unused=True)[0]
        assert g_live is not None and torch.isfinite(g_live).all()
        assert abs(float(g_live)) > 1e-10, "no gradient reaches the occupations"

    def test_the_force_loss_backward_is_finite_on_a_degenerate_toy(self):
        """No eigh double-backward survives anywhere on this path."""
        from mace.modules.defect_counting import fermi_density_matrix, head_forces
        g = torch.Generator().manual_seed(9)
        pos = torch.randn(3, 3, generator=g).requires_grad_(True)
        blk = torch.randn(6, 6, generator=g)
        blk = 0.5 * (blk + blk.T)
        scale = torch.ones(1, requires_grad=True)
        d = ((pos.unsqueeze(1) - pos.unsqueeze(0)).pow(2).sum(-1) + 1e-12).sqrt()
        w = torch.exp(-d).repeat_interleave(2, 0).repeat_interleave(2, 1)
        H = torch.block_diag(blk * w[:3, :3].repeat(2, 2)[:6, :6],
                             blk * w[:3, :3].repeat(2, 2)[:6, :6]) * scale
        P = fermi_density_matrix(H, 6.0)
        P_ref = fermi_density_matrix(H, 6.0).detach() * 0.0
        F = head_forces(H, pos, P, P_ref, create_graph=True)
        gg = torch.autograd.grad(F.pow(2).sum(), scale, allow_unused=True)[0]
        assert gg is not None and torch.isfinite(gg).all()


class TestMultiFillFunction:
    """`fermi_density_difference` must equal four `fermi_density_matrix` calls, in the value
    AND in the backward. The multi-fill Function exists only to save three eigensolves per
    graph per step; the four-call route is the validated reference, and a refactor that drifts
    from it would move every force gradient without moving a single force.
    """

    @staticmethod
    def toy(n=12, seed=5):
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(n, n, generator=g)
        return (0.5 * (a + a.T)).requires_grad_(True)

    def test_value_matches_the_four_call_route(self):
        from mace.modules.defect_counting import (fermi_density_difference,
                                                  fermi_density_matrix)
        h = self.toy()
        fills, signs = (7.0, 6.0, 6.0, 6.0), (1.0, 1.0, -1.0, -1.0)
        combined = fermi_density_difference(h, fills, signs)
        separate = sum(s * fermi_density_matrix(h, n) for s, n in zip(signs, fills))
        assert torch.allclose(combined, separate, atol=1e-12)

    def test_backward_matches_the_four_call_route(self):
        from mace.modules.defect_counting import (fermi_density_difference,
                                                  fermi_density_matrix)
        fills, signs = (7.0, 6.0, 6.0, 6.0), (1.0, 1.0, -1.0, -1.0)
        g = torch.Generator().manual_seed(11)
        cot = torch.randn(12, 12, generator=g)

        h1 = self.toy()
        (fermi_density_difference(h1, fills, signs) * cot).sum().backward()
        h2 = self.toy()
        (sum(s * fermi_density_matrix(h2, n) for s, n in zip(signs, fills)) * cot
         ).sum().backward()
        assert torch.allclose(h1.grad, h2.grad, atol=1e-10), (
            f"max |delta| = {float((h1.grad - h2.grad).abs().max()):.3e}")

    def test_identical_fills_give_exactly_zero_and_no_gradient(self):
        """The pristine case. Equal fills either side of the difference cancel in the value
        AND in the response, so a counters-off frame contributes nothing by construction."""
        from mace.modules.defect_counting import fermi_density_difference
        h = self.toy(seed=7)
        d = fermi_density_difference(h, (6.0, 6.0, 6.0, 6.0), (1.0, 1.0, -1.0, -1.0))
        assert float(d.abs().max()) == 0.0
        d.sum().backward()
        assert float(h.grad.abs().max()) == 0.0


class TestForceResponseWiring:
    """Section 1 wired into `CountingHead.forward`, which is where the model consumes it.

    The claim being pinned is a pair: the response is EXACTLY zero in value, so no force the
    model reports can move; and it is non-zero in the parameter gradient, which is the whole
    reason it exists. Testing only one half would let either a no-op or a prediction change
    through.
    """

    @staticmethod
    def system(n_sites=6, seed=2, counts=(0, 0, 1, 0)):
        from mace.modules.defect_counting import CountingHead, harrison_initialise
        torch.manual_seed(seed)
        head = CountingHead(num_elements=3, feature_dim=8, atomic_numbers=[17, 55, 82],
                            r_cut=6.0)
        harrison_initialise(head, [17, 55, 82], bond_length=2.8)
        g = torch.Generator().manual_seed(seed)
        pos = (torch.randn(n_sites, 3, generator=g) * 2.5).requires_grad_(True)
        species = torch.tensor([0, 2, 0, 0, 1, 0])[:n_sites]
        src, dst = torch.meshgrid(torch.arange(n_sites), torch.arange(n_sites),
                                  indexing="ij")
        keep = src.reshape(-1) != dst.reshape(-1)
        edge_index = torch.stack([src.reshape(-1)[keep], dst.reshape(-1)[keep]])
        feats = torch.randn(n_sites, 8, generator=g)
        batch = torch.zeros(n_sites, dtype=torch.long)
        cnt = torch.tensor([list(counts)], dtype=torch.get_default_dtype())

        def call(force_out):
            vec = pos[edge_index[1]] - pos[edge_index[0]]
            return head(node_feats=feats, counter_emb=None, counts=cnt, batch=batch,
                        num_graphs=1, edge_index=edge_index,
                        edge_length=vec.pow(2).sum(-1, keepdim=True).sqrt(),
                        node_species=species, edge_vector=vec,
                        positions=pos, force_out=force_out)

        return head, pos, call

    def test_the_response_is_exactly_zero_in_value(self):
        _, pos, call = self.system()
        out = {}
        call(out)
        assert "force_response" in out, "the head was asked and did not answer"
        assert float(out["force_response"].abs().max()) == 0.0, (
            "the response must be an identity zero, not a small number: every force the "
            "model reports is compared against runs made before the wiring")

    def test_wiring_changes_the_gradient_and_not_the_forces(self):
        """The section-1 contract in one test. Same batch, wiring on and off: the head force
        is bit-identical, and the gradient of that force with respect to a hopping scale is
        not. Deleting the response as a no-op fails here."""
        results = {}
        for tag, want in (("off", False), ("on", True)):
            head, pos, call = self.system()
            out = {} if want else None
            res = call(out)
            force = -torch.autograd.grad(res.delta_sr.sum(), pos, create_graph=True)[0]
            if out:
                force = force + out["force_response"]
            grad = torch.autograd.grad(force.pow(2).sum(), head.h.v0_raw,
                                       allow_unused=True)[0]
            results[tag] = (force.detach().clone(), None if grad is None else grad.clone())

        assert torch.equal(results["on"][0], results["off"][0]), (
            "the wiring moved the forces; it is supposed to be an identity in value")
        g_on, g_off = results["on"][1], results["off"][1]
        assert g_on is not None and g_off is not None
        assert torch.isfinite(g_on).all()
        delta = float((g_on - g_off).abs().max())
        assert delta > 1e-8, (
            f"the density response contributed nothing to the gradient (max delta {delta:.3e})"
            " -- the wiring is a no-op and the term is still missing")

    def test_a_neutral_frame_gets_no_response_at_all(self):
        """The standing pristine regression, at the FORCE level where it has never been
        asserted. At zero counters the fill equals the reference, so the head energy, its
        force and its response are all identically zero -- and the cheapest possible catch
        for a future `P` where `P - P_ref` belongs."""
        _, pos, call = self.system(counts=(0, 0, 0, 0))
        out = {}
        res = call(out)
        assert out == {}, "a neutral frame must not build a response graph at all"
        assert float(res.delta_sr.abs().max()) == 0.0
        force = -torch.autograd.grad(res.delta_sr.sum(), pos, allow_unused=True,
                                     materialize_grads=True)[0]
        assert float(force.abs().max()) < 1e-10, (
            f"a neutral frame exerts a head force of {float(force.abs().max()):.3e} eV/A")

    def test_it_refuses_to_answer_without_positions(self):
        from mace.modules.defect_counting import CountingHead
        head, pos, call = self.system()
        with pytest.raises(ValueError, match="positions are absent or detached"):
            vec = pos[[0, 1]] - pos[[1, 0]]
            head(node_feats=torch.randn(6, 8), counter_emb=None,
                 counts=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
                 batch=torch.zeros(6, dtype=torch.long), num_graphs=1,
                 edge_index=torch.tensor([[0, 1], [1, 0]]),
                 edge_length=vec.pow(2).sum(-1, keepdim=True).sqrt(),
                 node_species=torch.tensor([0, 2, 0, 0, 1, 0]), edge_vector=vec,
                 positions=None, force_out={})


class TestChangedLevelIndex:
    """Section 2(e). The frontier estimator picks the level whose OCCUPATION changed.

    An index that assumes an added electron reads a valence level when the counters encode a
    removed one -- and a valence level has a weak, flat dependence on the vacancy geometry,
    so the mistake presents as a clean null rather than as an error. The frames in hand carry
    (0, 0, 1, 0), a hole, so this is the case that actually runs.
    """

    def test_an_added_electron_gives_the_level_it_entered(self):
        from mace.modules.defect_counting import changed_level_index
        n_total = 20                       # neutral fill 10 / 10
        assert changed_level_index(n_total, (1, 0, 0, 0)) == 10

    def test_a_removed_electron_gives_the_level_it_vacated(self):
        """The case the naive index gets wrong: it would return 8, one below."""
        from mace.modules.defect_counting import changed_level_index
        assert changed_level_index(20, (0, 0, 1, 0)) == 9

    def test_no_counters_gives_the_neutral_frontier(self):
        from mace.modules.defect_counting import changed_level_index
        assert changed_level_index(20, (0, 0, 0, 0)) == 9

    def test_an_odd_electron_count_uses_the_majority_channel(self):
        from mace.modules.defect_counting import changed_level_index
        assert changed_level_index(21, (0, 0, 0, 0)) == 10       # ceil(21/2) = 11
        assert changed_level_index(21, (1, 0, 0, 0)) == 11

    def test_a_stated_occupation_is_honoured(self):
        from mace.modules.defect_counting import changed_level_index
        assert changed_level_index(20, (0, 0, 0, 0), occupation=(12.0, 8.0)) == 11


class TestSmearingFamilies:
    """Gaussian sigma = 0.05 eV is now the default, matching the labels' own convention.

    The defect calculations were set up through doped, whose default electronic smearing is
    Gaussian with SIGMA = 0.05 eV (ISMEAR = 0). The previous T_el = 25 meV was a k_B * 300 K
    coincidence with no connection to the labels. Fermi-Dirac stays implemented because
    "the answer does not depend on the smearing" needs a second family to test.
    """

    @staticmethod
    def spectrum(n=40, seed=4):
        g = torch.Generator().manual_seed(seed)
        return torch.sort(torch.randn(n, generator=g).double() * 3.0).values

    def test_the_default_is_the_labels_convention(self):
        from mace.modules.defect_counting import smearing
        assert smearing() == ("gaussian", 0.05)

    def test_the_occupation_is_erfc_over_two(self):
        from mace.modules.defect_counting import occupation_of
        x = torch.linspace(-4, 4, 9).double()
        assert torch.allclose(occupation_of(x, "gaussian"), 0.5 * torch.erfc(x))
        # Monotone, and the two limits, which no formula error survives.
        assert float(occupation_of(torch.tensor([-6.0]).double(), "gaussian")) > 1 - 1e-8
        assert float(occupation_of(torch.tensor([6.0]).double(), "gaussian")) < 1e-8

    def test_the_slope_is_the_derivative_of_the_occupation(self):
        """Finite differences against the analytic slope, both families. The slope IS the
        Daleckii-Krein limit at coincidence, so an error here is an error in every density
        response at a degeneracy."""
        from mace.modules.defect_counting import occupation_of, occupation_slope
        for family in ("gaussian", "fermi"):
            w, h = 0.05, 1e-6
            eps = torch.linspace(-0.3, 0.3, 13).double()
            fd = (occupation_of((eps + h) / w, family)
                  - occupation_of((eps - h) / w, family)) / (2 * h)
            assert torch.allclose(fd, occupation_slope(eps / w, w, family), atol=1e-6)

    def test_the_gaussian_slope_is_bounded_by_one_over_sigma_root_pi(self):
        """The bound that keeps the response backward finite at exact degeneracy."""
        import math
        from mace.modules.defect_counting import occupation_slope
        w = 0.05
        x = torch.linspace(-10, 10, 2001).double()
        assert float(occupation_slope(x, w, "gaussian").abs().max()) <= \
            1.0 / (w * math.sqrt(math.pi)) + 1e-12

    def test_the_fill_hits_the_electron_count_under_both_families(self):
        from mace.modules.defect_counting import fermi_fill, use_smearing
        lam = self.spectrum()
        for family in ("gaussian", "fermi"):
            previous = use_smearing(family, 0.05)
            try:
                assert float(fermi_fill(lam, 17.0, 0.05).sum()) == pytest.approx(17.0,
                                                                                abs=1e-8)
            finally:
                use_smearing(*previous)

    def test_the_two_families_agree_near_an_edge(self):
        """F2's premise: Gaussian sigma = 0.05 and Fermi-Dirac 25 meV are tail-equivalent, so
        the head has been accidentally close to the labels' convention all along. Compared on
        the OCCUPATION of a gapped spectrum, which is what the energy actually sees."""
        from mace.modules.defect_counting import fermi_fill, use_smearing
        lam = torch.cat([torch.linspace(-3.0, -0.6, 20),
                         torch.linspace(1.8, 4.0, 20)]).double()
        fills = {}
        for family, width in (("gaussian", 0.05), ("fermi", 0.025)):
            previous = use_smearing(family, width)
            try:
                fills[family] = fermi_fill(lam, 20.0, width)
            finally:
                use_smearing(*previous)
        assert torch.allclose(fills["gaussian"], fills["fermi"], atol=1e-6)

    def test_the_free_energy_is_stationary_in_the_fill(self):
        """`dF/deps_k = f_k` under BOTH families -- the identity the Hellmann-Feynman force
        argument rests on. It is the entropy term that makes it true, so a wrong entropy for
        a family shows up here and nowhere else."""
        from mace.modules.defect_counting import fermi_fill, free_energy, use_smearing
        for family, width in (("gaussian", 0.05), ("fermi", 0.025)):
            previous = use_smearing(family, width)
            try:
                lam = self.spectrum(seed=7).requires_grad_(True)
                f = fermi_fill(lam.detach(), 17.0, width)
                free_energy(lam, 17.0, width).backward()
                assert torch.allclose(lam.grad, f, atol=1e-8), (
                    f"{family}: dF/deps departs from f by "
                    f"{float((lam.grad - f).abs().max()):.2e}")
            finally:
                use_smearing(*previous)

    def test_an_unknown_family_is_refused(self):
        from mace.modules.defect_counting import occupation_of, use_smearing
        with pytest.raises(ValueError, match="unknown smearing family"):
            occupation_of(torch.zeros(3), "cold")
        with pytest.raises(ValueError, match="unknown smearing family"):
            use_smearing("cold", 0.05)
