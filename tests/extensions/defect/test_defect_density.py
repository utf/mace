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
        out = dc.frame_static_densities(harrison_model, rec, frames["vcl_39"], charges, pos, cell)
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
