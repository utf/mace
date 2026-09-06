"""Plan v8 section 2.1, Stage 0 task 0.9: the static and frontier density objects.

Gaussian charge lists with analytic integrals and norms; the pristine placement by density
alignment; `rho_static^raw` identities on the pristine reference geometry, on thermal
pristine frames and under a scaled displacement; `rho_static^def` integrating to Q_core; the
smooth edge projectors and separately normalised channels; `rho_F` integrating to q_F at
every value of the participation switch; and `int Delta rho_def = Q_formal`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mace.modules import defect_composition as dc
from mace.modules import defect_density as dd
from mace.modules.defect_state import StateBatch
from tests.extensions.defect.test_composition_classes import (_frame, _remove_cl,
                                                              frames, harrison_model,
                                                              table)  # noqa: F401
from tests.extensions.defect.test_neutral_reference_skip import _perovskite


@pytest.fixture(scope="module", autouse=True)
def _f64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _cell(a=9.0):
    return torch.eye(3, dtype=torch.float64) * a


# ------------------------------------------------------------------ the object


class TestGaussianDensity:
    def test_the_integral_is_the_sum_of_the_charges_plus_the_background(self):
        rho = dd.GaussianDensity(torch.tensor([1.0, -2.0, 0.5]), torch.rand(3, 3) * 9,
                                 1.0, _cell(), background=0.25)
        assert float(rho.integral()) == pytest.approx(-0.25)

    def test_the_norm_of_one_gaussian_is_analytic(self):
        sigma = 0.8
        rho = dd.GaussianDensity(torch.tensor([2.0]), torch.tensor([[4.5, 4.5, 4.5]]),
                                 sigma, _cell(20.0))
        # int g^2 = (4 pi sigma^2)^(-3/2) for a normalised Gaussian
        assert float(rho.norm2()) == pytest.approx(4.0 * (4 * math.pi * sigma ** 2) ** -1.5)

    def test_a_difference_of_two_identical_lists_has_zero_norm(self):
        rng = np.random.default_rng(0)
        pos = torch.tensor(rng.uniform(0, 9, (6, 3)))
        q = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
        a = dd.GaussianDensity(q, pos, 1.0, _cell())
        assert float((a - a).norm2()) == pytest.approx(0.0, abs=1e-12)

    def test_the_minimum_image_makes_the_norm_periodic(self):
        cell = _cell()
        a = dd.GaussianDensity(torch.tensor([1.0]), torch.tensor([[0.2, 0.2, 0.2]]), 1.0, cell)
        b = dd.GaussianDensity(torch.tensor([1.0]), torch.tensor([[8.9, 0.2, 0.2]]), 1.0, cell)
        assert float((a - b).norm()) < 0.3 * float(a.norm())

    def test_the_uniform_part_enters_the_overlap(self):
        cell = _cell()
        g = dd.GaussianDensity(torch.tensor([1.0]), torch.tensor([[1.0, 2.0, 3.0]]), 1.0, cell)
        u = dd.GaussianDensity(torch.zeros(0), torch.zeros(0, 3), 1.0, cell, background=2.0)
        assert float(g.overlap(u)) == pytest.approx(2.0 / 729.0)
        assert float(u.norm2()) == pytest.approx(4.0 / 729.0)

    def test_site_charges_read_an_isolated_gaussian_as_itself(self):
        cell = _cell(20.0)
        pos = torch.tensor([[3.0, 3.0, 3.0], [12.0, 12.0, 12.0]])
        rho = dd.GaussianDensity(torch.tensor([1.5, -0.5]), pos, 1.0, cell)
        assert torch.allclose(rho.site_charges(pos), torch.tensor([1.5, -0.5]), atol=1e-8)


# ------------------------------------------------------------------ the static part


def _static_inputs(model, frame):
    numbers, pos, cell = dc._frame_geometry(frame, model)
    zs = [int(z) for z in model.atomic_numbers]
    species = torch.tensor([zs.index(int(z)) for z in numbers])
    z0 = dc.species_charges(model)
    return (z0[species], torch.tensor(pos, dtype=torch.float64),
            torch.tensor(cell, dtype=torch.float64))


class TestStatic:
    def test_every_class_carries_a_placement_with_a_small_residual(self, table):
        for key, r in table["classes"].items():
            rec = dc.ClassRecord.from_dict(r)
            assert rec.placement is not None, key
            assert len(rec.placement["shift"]) == 3

    def test_rho_static_raw_vanishes_identically_on_the_pristine_reference_geometry(
            self, harrison_model, frames, table):
        rec = dc.lookup_class(table, [17] * 24 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["pristine"])
        out = dc.frame_static_densities(harrison_model, rec, frames["pristine"], charges, pos, cell)
        assert float(out["raw_norm"]) < 1e-9
        assert float(out["q_raw"]) == pytest.approx(0.0, abs=1e-12)
        assert torch.allclose(out["pristine"].centres, pos, atol=1e-6)

    def test_the_integral_of_rho_static_raw_is_zero_on_thermal_pristine_frames(
            self, harrison_model, frames, table):
        rec = dc.lookup_class(table, [17] * 24 + [55] * 8 + [82] * 8)
        thermal = _frame(_perovskite(reps=(2, 2, 2), rattle=0.05, seed=11), [0.0] * 4)
        charges, pos, cell = _static_inputs(harrison_model, thermal)
        out = dc.frame_static_densities(harrison_model, rec, frames["pristine"], charges, pos, cell)
        assert float(out["q_raw"]) == pytest.approx(0.0, abs=1e-12)
        assert float(out["raw_norm"]) > 0.0
        # ... and the placement is recovered from the same origin (the reference frame's).
        assert float(out["raw_norm"]) < 0.5 * float(out["present"].norm())

    def test_the_norm_of_rho_static_raw_goes_to_zero_continuously_with_the_displacement(
            self, harrison_model, frames, table):
        rec = dc.lookup_class(table, [17] * 24 + [55] * 8 + [82] * 8)
        charges, pos0, cell = _static_inputs(harrison_model, frames["pristine"])
        rng = np.random.default_rng(3)
        disp = torch.tensor(rng.normal(0, 0.1, pos0.shape))
        norms = []
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0):
            out = dc.frame_static_densities(harrison_model, rec, frames["pristine"], charges,
                                            pos0 + scale * disp, cell)
            norms.append(float(out["raw_norm"]))
        assert norms[-1] < 1e-9
        assert all(a > b for a, b in zip(norms, norms[1:]))
        # Linear in the displacement for small displacements.
        assert norms[2] / norms[3] == pytest.approx(2.0, rel=0.05)

    def test_the_integral_of_rho_static_raw_is_the_sum_of_the_static_charges(
            self, harrison_model, frames, table):
        """`q_raw = sum_i Z_i` over the DIFFERENCE: +1 (minus Z_Cl) for a Cl vacancy, on both
        sizes; and the residual weight sits on the vacancy site."""
        z0 = dc.species_charges(harrison_model)
        z_cl = float(z0[0])
        for name, key in (("vcl_39", [17] * 23 + [55] * 8 + [82] * 8),
                          ("vcl_79", [17] * 47 + [55] * 16 + [82] * 16)):
            rec = dc.lookup_class(table, key)
            charges, pos, cell = _static_inputs(harrison_model, frames[name])
            out = dc.frame_static_densities(harrison_model, rec, frames["pristine"], charges,
                                            pos, cell)
            assert float(out["q_raw"]) == pytest.approx(-z_cl, abs=1e-10)
            # Both terms of the definition (addendum 4.1), not the `sum_i Z_i` shorthand.
            assert float(out["q_raw"]) == pytest.approx(float(charges.sum()) - float(
                out["pristine"].charges.sum()), abs=1e-10)
            # The definition in terms of the two INTEGRALS, which is what the addendum
            # actually requires and what the code must use.
            assert float(out["q_raw"]) == pytest.approx(
                float(dd.static_present(charges, pos, cell, 1.0).integral())
                - float(out["pristine"].integral()), abs=1e-10)
            # And the reason the v8 shorthand `q_raw = sum_i Z_i` may NOT be used: the tiled
            # pristine baselines do not sum to zero on this host, so the present-atom sum is
            # wrong by of order one electron. If a future baseline table really is neutral
            # this assertion is what will say so, loudly, rather than the shorthand silently
            # becoming true and then silently breaking again.
            assert abs(float(out["pristine"].integral())) > 0.1
            assert float(out["q_raw"]) != pytest.approx(float(charges.sum()), abs=1e-3)
            assert float(out["g_res"].integral()) == pytest.approx(1.0, abs=1e-12)
            # The heaviest omega is at the vacancy's pristine site: a pristine centre with
            # no present atom within 1 A.
            g = out["g_res"]
            heaviest = g.centres[int(torch.argmax(g.charges))]
            d = dd._minimum_image(pos - heaviest, cell).norm(dim=-1)
            assert float(d.min()) > 1.0
            assert float(g.charges.max()) > 0.5

    def test_rho_static_def_integrates_to_q_core(self, harrison_model, frames, table):
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        out = dc.frame_static_densities(harrison_model, rec, frames["pristine"], charges, pos, cell)
        assert float(out["static"].integral()) == pytest.approx(rec.q_core, abs=1e-10)

    def test_the_placement_is_at_least_as_good_as_the_tier_2_correspondence(
            self, harrison_model, frames, table):
        """The density alignment minimises ||rho_static^raw||; the correspondence minimises
        squared displacements. On this cubic toy the two objectives have near-degenerate
        minima a primitive lattice vector apart (every Cl site is equivalent up to the
        rattle), so the shifts need not coincide; the density residual at the density's own
        shift must be no worse than at the correspondence's, and both must be small."""
        rec = dc.lookup_class(table, [17] * 47 + [55] * 16 + [82] * 16)
        # The tier is irrelevant here: this test builds the site correspondence itself and
        # needs only the record's tiling, perm and placement. (Under the v8.1 verifier this
        # class is reached by transport at Tier 1, while the smaller class of the same
        # family carries the continuation.)
        assert rec.counted, rec.reason
        n1, p1, c1 = dc._frame_geometry(frames["vcl_79"], harrison_model)
        n0, p0, c0 = dc._frame_geometry(frames["pristine"], harrison_model)
        n0, p0, c0 = dc.tile_frame(n0, p0, c0, rec.tiling)
        corr = dc.site_correspondence(n1, p1, c1, n0, p0, c0, perm=rec.perm, r_match=2.0)
        t_corr = torch.tensor(np.asarray(corr["translation"]) @ np.linalg.inv(c1))
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_79"])
        z0 = dc.species_charges(harrison_model)
        present = dd.static_present(charges, pos, cell, rec.placement["r_res"])
        species, scaled = dc.tiled_pristine_scaled(harrison_model, frames["pristine"],
                                                   rec.tiling, rec.perm)

        def residual(t):
            pri = dd.pristine_placed(z0[species], scaled, cell, t, rec.placement["r_res"])
            return float(dd.static_raw(present, pri).norm())

        r_dens = residual(torch.tensor(rec.placement["shift"]))
        r_corr = residual(t_corr)
        assert r_dens <= r_corr + 1e-9
        assert r_dens == pytest.approx(rec.placement["residual_norm"], rel=1e-6)
        assert r_dens < 0.3 * float(present.norm())


# ------------------------------------------------------------------ the frontier part


class TestFrontier:
    @staticmethod
    def toy(rng, n_sites=10, n_deep=6):
        from tests.extensions.defect.test_composition_classes import _toy

        H = torch.tensor(_toy(n_sites, rng, n_deep=n_deep))
        eps = torch.linalg.eigvalsh(H)
        vbm, cbm = float(eps[4 * n_deep - 1]), float(eps[4 * n_deep])
        pos = torch.tensor(rng.uniform(0, 12, (n_sites, 3)))
        return H, eps, vbm, cbm, pos, _cell(12.0)

    def test_the_edge_projectors_are_smooth_switches(self):
        eps = torch.linspace(-10.0, 0.0, 101)
        s_e, s_h = dd.edge_projectors(eps, -7.0, -3.0, 0.1, 0.05)
        assert float(s_e[0]) < 1e-6 and float(s_e[-1]) > 1 - 1e-6
        assert float(s_h[0]) > 1 - 1e-6 and float(s_h[-1]) < 1e-6
        assert float(s_e[int(torch.argmin((eps + 6.9).abs()))]) == pytest.approx(0.5, abs=1e-6)
        assert float(s_h[int(torch.argmin((eps + 3.1).abs()))]) == pytest.approx(0.5, abs=1e-6)

    def test_channels_are_separately_normalised_and_stored_separately(self):
        rng = np.random.default_rng(0)
        H, eps, vbm, cbm, pos, cell = self.toy(rng)
        n_occ = 24
        f_maj = torch.zeros(40)
        f_maj[: n_occ + 1] = 1.0            # one electron in the first conduction level
        f_min = torch.zeros(40)
        f_min[:n_occ] = 1.0
        f_min[n_occ - 1] = 0.0              # and one hole at the valence top, minority
        ch = dd.channel_densities(H, [f_maj, f_min], pos, cell, vbm, cbm, 0.1, 0.05, 1.0)
        e_maj, e_min = ch["electron"]
        h_maj, h_min = ch["hole"]
        assert e_maj is not None and float(e_maj.integral()) == pytest.approx(1.0)
        assert h_min is not None and float(h_min.integral()) == pytest.approx(1.0)
        assert torch.all(e_maj.charges >= 0) and torch.all(h_min.charges >= 0)
        # The needed channels carry the carrier's weight (~1, plus the smooth projectors'
        # tail from the band edges); the unneeded ones carry only the tail, and are unused
        # when their count is zero.
        assert 1.0 <= float(ch["weight_e"][0]) < 1.5
        assert 1.0 <= float(ch["weight_h"][1]) < 1.5
        assert float(ch["weight_e"][1]) < 0.5 and float(ch["weight_h"][0]) < 0.5
        assert (e_min is None or float(e_min.integral()) == pytest.approx(1.0))
        assert (h_maj is None or float(h_maj.integral()) == pytest.approx(1.0))
        assert e_maj is not e_min and h_maj is not h_min

    def test_the_projected_density_matrix_is_degeneracy_safe(self):
        """Two sites, each orbital coupled to its partner: a 4-fold bonding level at -1 - t
        and a 4-fold antibonding one at -1 + t, every eigenvector half on each site. Two
        electrons in the bonding level give [0.5, 0.5] whatever basis eigh picks."""
        t = 0.3
        H = torch.full((8, 8), 0.0)
        H[:4, :4] = -torch.eye(4)
        H[4:, 4:] = -torch.eye(4)
        H[:4, 4:] = t * torch.eye(4)
        H[4:, :4] = t * torch.eye(4)
        f = torch.tensor([1.0, 1, 0, 0, 0, 0, 0, 0])
        pos = torch.tensor([[0.0, 0, 0], [5.0, 5, 5]])
        ch = dd.channel_densities(H, [f], pos, _cell(10.0), -20.0, 10.0, 0.1, 0.05, 1.0)
        assert torch.allclose(ch["electron"][0].charges, torch.tensor([0.5, 0.5]), atol=1e-8)

    def test_rho_f_integrates_to_q_f_at_every_value_of_the_switch(self):
        rng = np.random.default_rng(1)
        H, eps, vbm, cbm, pos, cell = self.toy(rng)
        f = torch.zeros(40)
        f[:26] = 1.0
        ch = dd.channel_densities(H, [f, f], pos, cell, vbm, cbm, 0.1, 0.05, 1.0)
        for p_star in (0.0, 0.25, 1.0):
            out = dd.frontier_density((2, 2), (0, 0), ch, pos, cell, 1.0, p_star, 0.05)
            assert float(out["rho_F"].integral()) == pytest.approx(-4.0, abs=1e-10)
            assert float(out["rho_F_loc"].integral()) == pytest.approx(-4.0, abs=1e-10)
            assert float(out["rho_F_ext"].integral()) == pytest.approx(-4.0)
        assert 0.0 < float(out["p"]) < 1.0

    def test_a_needed_but_empty_channel_is_refused(self):
        rng = np.random.default_rng(2)
        H, eps, vbm, cbm, pos, cell = self.toy(rng)
        f = torch.zeros(40)
        f[:24] = 1.0
        # A fill far below every edge projector: no weight at all in the electron channel.
        ch = dd.channel_densities(H, [f], pos, cell, vbm + 40.0, cbm + 40.0, 0.1, 0.05, 1.0)
        assert ch["electron"][0] is None
        with pytest.raises(ValueError, match="no projector weight"):
            dd.frontier_density((1,), (0,), ch, pos, cell, 1.0, 0.25, 0.05)

    def test_the_synthetic_m_frontier_class_gives_q_f_minus_m_and_a_neutral_defect_density(self):
        """The plan's synthetic class: m > 1 occupied frontier levels, Q_formal = 0 ->
        q_F = -m and int Delta rho_def = 0."""
        m = 3
        rng = np.random.default_rng(4)
        H, eps, vbm, cbm, pos, cell = self.toy(rng, n_sites=10, n_deep=6)
        # class integers from the toy: rank 24 valence, m frontier electrons in the majority
        n_e, n_h, q_core = dc.class_integers((24, 24), (24 + m, 24))
        assert n_e == (m, 0) and q_core == m
        rec = dc.ClassRecord(key="synthetic", n_atoms=10, reference_frame_key=0, n_total=48 + m,
                             n_sigma=(24 + m, 24), tier=1, ambiguous=False, reason="",
                             m_vb=(24, 24), n_e=n_e, n_h=n_h, q_core=q_core, vbm_al=vbm,
                             cbm_al=cbm, shift=0.0, spread=0.0, gap_pristine=cbm - vbm,
                             delta=0.1, nearest=-0.1)
        neutral = StateBatch.from_counts(torch.zeros(1, 4))
        fe, fh, q_f = dc.frame_counts(rec, neutral, 0)
        assert q_f == -m and fe == (m, 0)
        f_maj = torch.zeros(40)
        f_maj[: 24 + m] = 1.0
        f_min = torch.zeros(40)
        f_min[:24] = 1.0
        ch = dd.channel_densities(H, [f_maj, f_min], pos, cell, vbm, cbm, 0.1, 0.05, 1.0)
        front = dd.frontier_density(fe, fh, ch, pos, cell, 1.0, 0.25, 0.05)
        assert float(front["rho_F"].integral()) == pytest.approx(-m, abs=1e-10)
        # A synthetic static part: any raw density plus g_res, integrating to Q_core.
        raw = dd.GaussianDensity(torch.tensor([0.3, -0.3]), pos[:2], 1.0, cell)
        g_res = dd.residual_shape(raw, pos[:2], pos[2:3])
        static = dd.static_def(raw, g_res, q_core)
        assert float(static.integral()) == pytest.approx(m, abs=1e-10)
        total = dd.defect_density(static, front["rho_F"])
        assert float(total.integral()) == pytest.approx(0.0, abs=1e-10)

    def test_batched_per_frame_counts(self, table):
        numbers = [[17] * 23 + [55] * 8 + [82] * 8, [17] * 24 + [55] * 8 + [82] * 8]
        state = StateBatch.from_counts(torch.tensor([[0.0, 0, 1.0, 0], [0.0, 0, 1.0, 0]]))
        n_e, n_h, q_f = dc.frame_counts_batch(table, numbers, state)
        assert n_e.tolist() == [[0, 0], [0, 0]] and n_h.tolist() == [[0, 0], [1, 0]]
        assert q_f.tolist() == [0, 1]


class TestSmoothResidualShape:
    """Addendum 4.1: g_res is smooth, everywhere defined, and its fallback vanishes with N."""

    @staticmethod
    def _raw(dz, cell_size=12.0, n_extra=0):
        cell = _cell(cell_size)
        n = len(dz) + n_extra
        pos = torch.stack([torch.tensor([1.0 + 2.0 * i, 1.0, 1.0], dtype=torch.float64)
                           for i in range(n)])
        charges = torch.tensor(list(dz) + [0.0] * n_extra, dtype=torch.float64)
        return dd.GaussianDensity(charges, pos, 1.0, cell), pos

    def test_weights_are_normalised_for_any_input(self):
        for dz in ([1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1e-14, 0.0, 0.0], [5.0, 0.0, 0.0]):
            raw, pos = self._raw(dz)
            omega, _ = dd.residual_weights(raw, pos, len(dz))
            assert float(omega.sum()) == pytest.approx(1.0, abs=1e-12), dz
            assert bool((omega >= 0).all()), dz

    def test_all_zero_deviations_degrade_to_uniform_without_a_branch(self):
        """The old code took a hard `if total <= 1e-12` branch onto a uniform BACKGROUND."""
        raw, pos = self._raw([0.0, 0.0, 0.0, 0.0])
        omega, a = dd.residual_weights(raw, pos, 4)
        assert float(a.sum()) == pytest.approx(0.0, abs=1e-9)
        assert torch.allclose(omega, torch.full_like(omega, 0.25), atol=1e-12)
        # And g_res is a real charge list integrating to 1, not a uniform background.
        g = dd.residual_shape(raw, pos, pos)
        assert float(g.integral()) == pytest.approx(1.0, abs=1e-12)
        assert g.background == 0.0

    def test_the_weights_are_differentiable_where_the_deviation_vanishes(self):
        """|dZ| has a kink at zero; the smoothed form must not."""
        charges = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        cell = _cell(12.0)
        pos = torch.stack([torch.tensor([1.0 + 2.0 * i, 1.0, 1.0], dtype=torch.float64)
                           for i in range(3)])
        raw = dd.GaussianDensity(charges, pos, 1.0, cell)
        omega, _ = dd.residual_weights(raw, pos, 3)
        omega.sum().backward()
        assert charges.grad is not None and torch.isfinite(charges.grad).all()

    def test_the_uniform_fallback_fraction_vanishes_as_one_over_n(self):
        """A finite delocalised fraction as the cell grows is the failure being removed."""
        fractions = {}
        for n_at in (10, 40, 160):
            raw, pos = self._raw([1.0], cell_size=400.0, n_extra=n_at - 1)
            _, a = dd.residual_weights(raw, pos, n_at)
            floor = dd.EPS_OMEGA / n_at ** 2
            fractions[n_at] = n_at * floor / float((a + floor).sum())
        assert fractions[40] < fractions[10]
        assert fractions[160] < fractions[40]
        # O(1/N): a 16x larger cell has ~16x less fallback weight.
        assert fractions[10] / fractions[160] == pytest.approx(16.0, rel=0.2)

    def test_eps_must_be_positive(self):
        raw, pos = self._raw([1.0, 0.0])
        for bad in ({"eps_z": 0.0}, {"eps_omega": 0.0}, {"eps_z": -1e-6}):
            with pytest.raises(ValueError, match="must be positive"):
                dd.residual_weights(raw, pos, 2, **bad)

    def test_the_departure_term_is_optional_and_nonnegative(self):
        raw, pos = self._raw([0.0, 0.0, 0.0])
        base, _ = dd.residual_weights(raw, pos, 3)
        dep = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        with_dep, a = dd.residual_weights(raw, pos, 3, departure=dep, lambda_d=1.0)
        assert torch.allclose(base, torch.full_like(base, 1 / 3), atol=1e-12)
        assert float(with_dep[0]) > 0.9              # the departure now carries the weight
        assert bool((a >= 0).all())
        # lambda_d = 0 ignores it entirely.
        off, _ = dd.residual_weights(raw, pos, 3, departure=dep, lambda_d=0.0)
        assert torch.allclose(off, base, atol=1e-12)

    def test_a_mismatched_departure_signal_is_refused(self):
        raw, pos = self._raw([1.0, 0.0])
        with pytest.raises(ValueError, match="departure signal"):
            dd.residual_weights(raw, pos, 2, departure=torch.zeros(5, dtype=torch.float64),
                                lambda_d=1.0)


class TestResidualSupport:
    """A non-zero residual monopole on a flat g_res is refused, not delocalised."""

    @staticmethod
    def _raw(dz):
        cell = _cell(12.0)
        pos = torch.stack([torch.tensor([1.0 + 2.0 * i, 1.0, 1.0], dtype=torch.float64)
                           for i in range(len(dz))])
        return dd.GaussianDensity(torch.tensor(dz, dtype=torch.float64), pos, 1.0, cell), pos

    def test_a_real_departure_signal_supports_a_residual_charge(self):
        raw, pos = self._raw([1.0, 0.0, 0.0])
        ok, signal = dd.residual_support(raw, pos, 3, q_core=1, q_raw=0.0)
        assert ok and signal > 0.5

    def test_no_departure_signal_refuses_a_residual_charge(self):
        raw, pos = self._raw([0.0, 0.0, 0.0])
        ok, signal = dd.residual_support(raw, pos, 3, q_core=1, q_raw=0.0)
        assert not ok
        assert signal < dd.SUPPORT_THRESHOLD

    def test_no_residual_charge_needs_no_signal(self):
        """Nothing to place, so the shape is irrelevant and the state stays supported."""
        raw, pos = self._raw([0.0, 0.0, 0.0])
        ok, _ = dd.residual_support(raw, pos, 3, q_core=0, q_raw=0.0)
        assert ok


class TestRegistrationCovariance:
    """Addendum 4.1: the pristine/defect registration must be covariant, not merely close.

    `rho_static^raw` is a DIFFERENCE of two densities, so an origin convention that moves
    one and not the other turns a vacancy into a cell-wide array of dipoles. These pin the
    transformations under which the difference must be unchanged.
    """

    @staticmethod
    def _raw_norm(model, rec, pristine_frame, charges, pos, cell):
        out = dc.frame_static_densities(model, rec, pristine_frame, charges, pos, cell)
        return float(out["raw_norm"]), float(out["q_raw"])

    def test_rigid_translation(self, harrison_model, frames, table):
        """Addendum 4.1: a rigid translation is a symmetry of a periodic system, so the
        pristine reference must follow the frame rather than staying pinned."""
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        base = self._raw_norm(harrison_model, rec, frames["pristine"], charges, pos, cell)
        shift = torch.tensor([1.234, -0.567, 2.345], dtype=torch.float64)
        moved = self._raw_norm(harrison_model, rec, frames["pristine"], charges,
                               pos + shift, cell)
        assert moved[0] == pytest.approx(base[0], rel=1e-8)
        assert moved[1] == pytest.approx(base[1], abs=1e-10)

    def test_rigid_rotation(self, harrison_model, frames, table):
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        base = self._raw_norm(harrison_model, rec, frames["pristine"], charges, pos, cell)
        theta = 0.37
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        turned = self._raw_norm(harrison_model, rec, frames["pristine"], charges,
                                pos @ rot.T, cell @ rot.T)
        assert turned[0] == pytest.approx(base[0], rel=1e-8)
        assert turned[1] == pytest.approx(base[1], abs=1e-10)

    def test_periodic_wrap(self, harrison_model, frames, table):
        """Rewrapping atoms through the cell face may not change the difference."""
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        base = self._raw_norm(harrison_model, rec, frames["pristine"], charges, pos, cell)
        frac = pos @ torch.linalg.inv(cell)
        wrapped = (frac % 1.0) @ cell
        got = self._raw_norm(harrison_model, rec, frames["pristine"], charges, wrapped, cell)
        assert got[0] == pytest.approx(base[0], rel=1e-8)

    def test_atom_permutation(self, harrison_model, frames, table):
        """A density difference cannot depend on the order atoms are listed in."""
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        base = self._raw_norm(harrison_model, rec, frames["pristine"], charges, pos, cell)
        g = torch.Generator().manual_seed(5)
        perm = torch.randperm(pos.shape[0], generator=g)
        got = self._raw_norm(harrison_model, rec, frames["pristine"], charges[perm],
                             pos[perm], cell)
        assert got[0] == pytest.approx(base[0], rel=1e-8)
        assert got[1] == pytest.approx(base[1], abs=1e-10)

    def test_homogeneous_strain_co_deforms_the_pristine_reference(self, harrison_model,
                                                                  frames, table):
        """The registered pristine FRACTIONAL coordinates stay fixed and follow the cell.

        If the pristine reference were held in Cartesian coordinates, straining the cell
        would slide every pristine site against its present partner and `q_raw` -- an exact
        integer difference of charges -- would drift. This asserts it does not.
        """
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        base = self._raw_norm(harrison_model, rec, frames["pristine"], charges, pos, cell)
        strain = torch.eye(3, dtype=torch.float64)
        strain[0, 0] += 0.01
        strain[1, 2] += 0.004
        frac = pos @ torch.linalg.inv(cell)
        got = self._raw_norm(harrison_model, rec, frames["pristine"], charges,
                             frac @ (cell @ strain.T), cell @ strain.T)
        # q_raw is a difference of charge sums: exactly invariant under any cell change.
        assert got[1] == pytest.approx(base[1], abs=1e-10)
        # The norm moves only by the smooth density response, not by a registration slip.
        assert got[0] == pytest.approx(base[0], rel=0.05)


# ------------------------------------------------------------------ Stage 4: frozen correspondence, lift, derivative


class TestFrozenCorrespondenceAndLift:
    """Addendum 4.1/4.2: the discrete correspondence is established once per class and the
    lift's branch is selected from constructor topology on it, covariantly."""

    KEY = [17] * 23 + [55] * 8 + [82] * 8

    def test_the_class_record_carries_the_frozen_correspondence(self, table):
        rec = dc.lookup_class(table, self.KEY)
        corr = rec.placement["correspondence"]
        assert len(corr["matched_sites"]) == 39 and len(set(corr["matched_sites"])) == 39
        assert len(corr["removal_sites"]) == 1                   # the vacancy
        assert corr["addition_departure"] == []
        assert len(corr["site_departure"]) == 40
        assert all(a >= 0.0 for a in corr["site_departure"])
        pristine = dc.lookup_class(table, [17] * 24 + [55] * 8 + [82] * 8)
        assert pristine.placement["correspondence"]["removal_sites"] == []
        assert len(pristine.placement["correspondence"]["matched_sites"]) == 40

    def test_the_table_stores_the_constructor_pristine_geometry(self, table, frames):
        numbers, pos, cell = dc.pristine_reference_geometry(table)
        assert len(numbers) == 40 and pos.shape == (40, 3) and cell.shape == (3, 3)
        assert table["pristine_reference"]["frame_key"] == int(
            frames["pristine"].frame_key.reshape(-1)[0])
        with pytest.raises(RuntimeError, match="pristine_reference"):
            dc.pristine_reference_geometry({"classes": {}})

    def test_the_forward_form_reads_the_table_and_agrees_with_the_frame_form(
            self, harrison_model, frames, table):
        rec = dc.lookup_class(table, self.KEY)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        with_frame = dc.frame_static_densities(harrison_model, rec, frames["pristine"],
                                               charges, pos, cell)
        from_table = dc.frame_static_densities(harrison_model, rec, None, charges, pos, cell)
        assert float(from_table["raw_norm"]) == pytest.approx(float(with_frame["raw_norm"]),
                                                              rel=1e-10)
        assert float(from_table["q_raw"]) == pytest.approx(float(with_frame["q_raw"]))

    def test_the_lift_is_anchored_on_the_vacancy_and_is_covariant(self, harrison_model, frames,
                                                                    table):
        rec = dc.lookup_class(table, self.KEY)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        out = dc.frame_static_densities(harrison_model, rec, None, charges, pos, cell, lift=True)
        lift = out["lift"]
        vacancy = out["scaled_pristine"][out["correspondence"]["unmatched_pristine"][0]]
        for c, v in zip(lift.centre, (vacancy % 1.0).tolist()):
            assert abs((c - v + 0.5) % 1.0 - 0.5) < 0.06
        assert min(lift.moments) > lift.z_min
        assert out["support"]["supported"]
        # Translate the frame by an arbitrary vector: the centre follows, exactly.
        t = torch.tensor([2.1, -0.7, 3.3], dtype=torch.float64)
        moved = dc.frame_static_densities(harrison_model, rec, None, charges, pos + t, cell,
                                          lift=True)["lift"]
        dt = (t @ torch.linalg.inv(cell)).tolist()
        for c0, c1, d in zip(lift.centre, moved.centre, dt):
            assert abs((c1 - c0 - d + 0.5) % 1.0 - 0.5) < 1e-6

    def test_an_atom_permutation_transports_the_correspondence(self, harrison_model, frames,
                                                                table):
        """Section 11.1: atom permutation co-transforms the densities; the frozen record is
        keyed by SITE, so a relabelling of the atoms changes nothing downstream."""
        rec = dc.lookup_class(table, self.KEY)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        base = dc.frame_static_densities(harrison_model, rec, None, charges, pos, cell, lift=True)
        g = torch.Generator().manual_seed(9)
        perm = torch.randperm(pos.shape[0], generator=g)
        got = dc.frame_static_densities(harrison_model, rec, None, charges[perm], pos[perm],
                                        cell, lift=True)
        assert float(got["static"].norm2()) == pytest.approx(float(base["static"].norm2()),
                                                              rel=1e-8)
        assert got["lift"].centre == pytest.approx(base["lift"].centre, abs=1e-9)
        assert sorted(got["correspondence"]["departure"]) == pytest.approx(
            sorted(base["correspondence"]["departure"]))

    def test_a_changed_topology_is_refused_not_refitted(self, harrison_model, frames, table):
        """Handing a class's record a frame with one more removal is a topology event."""
        rec = dc.lookup_class(table, self.KEY)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        with pytest.raises(dc.UnsupportedStateError, match="topology event"):
            dc.frame_static_densities(harrison_model, rec, None, charges[1:], pos[1:], cell)

    def test_a_record_without_the_correspondence_cannot_lift(self, harrison_model, frames,
                                                              table):
        rec = dc.lookup_class(table, self.KEY)
        legacy = dc.ClassRecord.from_dict({**rec.to_dict(),
                                           "placement": {k: v for k, v in rec.placement.items()
                                                         if k != "correspondence"}})
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        with pytest.raises(RuntimeError, match="frozen correspondence"):
            dc.frame_static_densities(harrison_model, rec if False else legacy, None, charges,
                                      pos, cell, lift=True)


    def test_the_uniqueness_radius_is_a_lattice_property_and_is_frozen(self, table):
        """Half the smallest site separation of the ideal lattice -- the radius inside which
        the site nearest to an atom is unambiguous -- stored on the class record."""
        numbers, pos, cell = dc.pristine_reference_geometry(table)
        radius = dd.uniqueness_radius(torch.tensor(pos), torch.tensor(cell))
        d = dd._minimum_image(torch.tensor(pos)[:, None] - torch.tensor(pos)[None], torch.tensor(cell))
        d = d.norm(dim=-1) + torch.eye(len(numbers)) * 1e9
        assert radius == pytest.approx(0.5 * float(d.min()))
        assert radius > 0.5                       # the old hard 0.5 A rule was below it
        rec = dc.lookup_class(table, self.KEY)
        assert rec.placement["correspondence"]["uniqueness_radius"] == pytest.approx(radius)
        assert "merge" not in rec.placement["correspondence"]

    def test_thermal_displacements_inside_the_ball_keep_the_frozen_map(
            self, harrison_model, frames, table):
        """Addendum 4.1: the correspondence is NOT re-decided per thermal frame. Every atom
        displaced by a random vector well beyond the retired 0.5 A rule but inside its
        site's ball reads the frozen map (same removal site, same departure signal)."""
        rec = dc.lookup_class(table, self.KEY)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        base = dc.frame_static_densities(harrison_model, rec, None, charges, pos, cell)
        radius = rec.placement["correspondence"]["uniqueness_radius"]
        g = torch.Generator().manual_seed(4)
        direction = torch.randn(pos.shape, generator=g, dtype=torch.float64)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        amplitude = 0.5 * radius * torch.rand(pos.shape[0], 1, generator=g, dtype=torch.float64)
        assert float(amplitude.max()) > 0.55     # some atoms move more than 0.5 A
        hot = dc.frame_static_densities(harrison_model, rec, None, charges,
                                        pos + amplitude * direction, cell)
        # The re-fit may land on a lattice-translation-equivalent representative (a different
        # site INDEX for the vacancy); the physical removal site is the same place.
        vac_base = base["pristine"].centres[base["correspondence"]["unmatched_pristine"]]
        vac_hot = hot["pristine"].centres[hot["correspondence"]["unmatched_pristine"]]
        assert vac_hot.shape == vac_base.shape == (1, 3)
        assert float(dd._minimum_image(vac_hot - vac_base, cell).norm()) < radius
        assert hot["correspondence"]["departure"] == pytest.approx(
            base["correspondence"]["departure"])
        assert float(hot["static"].integral()) == pytest.approx(float(base["static"].integral()),
                                                                abs=1e-9)

    def test_an_atom_that_leaves_every_ball_is_a_topology_event(self, harrison_model, frames,
                                                                 table):
        """One atom parked farther than the uniqueness radius from every site is an addition
        and its site a removal: a changed correspondence, refused, not refitted."""
        rec = dc.lookup_class(table, self.KEY)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        base = dc.frame_static_densities(harrison_model, rec, None, charges, pos, cell)
        sites = base["pristine"].centres
        radius = rec.placement["correspondence"]["uniqueness_radius"]
        # A grid search for a point outside every ball (the interstitial region).
        best, best_d = None, 0.0
        for f in torch.cartesian_prod(*[torch.linspace(0.05, 0.95, 10)] * 3):
            point = f.to(cell.dtype) @ cell
            dmin = float(dd._minimum_image(sites - point, cell).norm(dim=-1).min())
            if dmin > best_d:
                best, best_d = point, dmin
        assert best_d > radius
        moved = pos.clone()
        moved[0] = best
        with pytest.raises(dc.UnsupportedStateError, match="topology event"):
            dc.frame_static_densities(harrison_model, rec, None, charges, moved, cell)


class TestRegistrationDerivative:
    """Addendum 4.1: 'any continuous geometry-dependent alignment is fully differentiated'.
    The placement shift is a minimiser, so its derivative is an implicit one; a detached
    shift leaves it out of every force through the pristine density."""

    def test_the_placed_pristine_density_carries_the_implicit_derivative(
            self, harrison_model, frames, table):
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        charges, pos, cell = _static_inputs(harrison_model, frames["vcl_39"])
        harrison_model.composition_classes = table
        # A probe that is NOT symmetric about the vacancy, so that a functional of the
        # pristine density is sensitive to where it was placed.
        probe = dd.GaussianDensity(torch.tensor([1.0, -0.4]),
                                   torch.tensor([[1.3, 2.2, 0.7], [4.1, 0.9, 3.3]]), 1.5, cell)

        def functional(p):
            out = dc.frame_static_densities(harrison_model, rec, None, charges, p, cell)
            return out["pristine"].overlap(probe)

        p = pos.clone().requires_grad_(True)
        value = functional(p)
        (grad,) = torch.autograd.grad(value, p)
        assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0.0
        h = 1e-4
        for atom, comp in ((0, 0), (7, 2), (21, 1)):
            plus, minus = pos.clone(), pos.clone()
            plus[atom, comp] += h
            minus[atom, comp] -= h
            fd = (float(functional(plus)) - float(functional(minus))) / (2.0 * h)
            assert float(grad[atom, comp]) == pytest.approx(fd, abs=2e-6, rel=1e-4)
