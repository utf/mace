"""Addendum section 8, Stage 4: the unified image regime wired into the forward.

On the Harrison-initialised counting-head toy (a gapped CsPbCl3 2x2x2 cell and its Cl
vacancy) with `image_functional="unified"`:

* the regime replaces the frontier patch: the registry reads base / band / static_frontier /
  image, the terms sum to the assembled energy, and the head correction is an exact zero at
  `S = S_ref`;
* `H_fix = H_local`: the unified regime refuses a Madelung shift inside H (double count), at
  construction and on load; the default regime is untouched;
* the two-boundary diagnostic: `Phi_SF` is boundary-common, `Phi_img` is zero under the
  isolated boundary while `phi_img_pbc` is reported under both;
* the gates fire BEFORE any energy: `m_F > 1` in the requested state is refused;
* section 7.2 / 11.1: the force and the stress of both terms pass the finite-difference
  harness under both boundaries -- the registration response, the lift's smooth geometry
  dependence and the density response `dP^(0)/dR` are all in the analytic derivative;
* addendum 4.1's thermal-background clause: a thermal PRISTINE frame's static density is a
  cell-wide array of displacement dipoles, reported as mass in the cut buffers and refused
  by the lift, never silently treated as compact.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mace.modules import defect_composition as dc
from mace.modules import defect_fd as fd
from mace.modules import defect_terms as dt
from mace.modules.defect_cache import attach_frame_keys
from mace.modules.defect_carriers import UnsupportedStateError
from mace.modules.defect_context import ForwardContext
from mace.modules.defect_lift import clearance_report
from mace.modules.defect_protocol import apply_harrison
from tests.extensions.defect.test_composition_classes import _remove_cl
from tests.extensions.defect.test_frontier_term import HOLE, NEUTRAL, PRISTINE, VACANCY, _frame
from tests.extensions.defect.test_neutral_reference_skip import Z_TABLE, _batch, _model, _perovskite

torch.set_default_dtype(torch.float64)

#: One minority electron on the neutral vacancy: d = (1, 1), m_F = 2.
TWO_CARRIERS = [0.0, 1.0, 0.0, 0.0]


def _unified_model(**overrides):
    kw = dict(image_functional="unified", madelung_range="off")
    kw.update(overrides)
    model = _model(**kw)
    apply_harrison(model, model.atomic_numbers)
    return model


@pytest.fixture(scope="module")
def model():
    m = _unified_model()
    frames = [_frame(PRISTINE, NEUTRAL), _frame(VACANCY, NEUTRAL)]
    attach_frame_keys(frames, z_table=Z_TABLE)
    dc.ensure_class_table(m, frames, log=False)
    return m


def _forward(model, frames, counts, **kw):
    batch = _batch(frames, counts)
    return model(batch.to_dict(), **kw), batch


# ------------------------------------------------------------------ the regime


class TestRegime:
    def test_the_unified_regime_replaces_the_frontier_patch(self, model):
        assert model.boundary_active and not model.frontier_active
        assert [t.name for t in model.terms()] == ["base", "band", "static_frontier", "image"]
        by_name = {t.name: t for t in model.terms()}
        assert not by_name["static_frontier"].gauge_dependent      # boundary-common
        assert by_name["image"].gauge_dependent
        assert all(t.potential == "absent" for t in model.terms()
                   if t.name in ("static_frontier", "image"))      # forward-only
        assert all(t.response == "divided_difference" for t in model.terms()
                   if t.name in ("static_frontier", "image"))

    def test_the_default_regime_is_untouched(self):
        legacy = _model()
        assert legacy.image_functional == "frontier_ff"
        assert legacy.frontier_active and not legacy.boundary_active
        assert [t.name for t in legacy.terms()] == ["base", "band", "frontier"]

    def test_h_fix_is_h_local_the_madelung_shift_is_refused(self):
        with pytest.raises(ValueError, match="requires madelung_range='off'"):
            _model(image_functional="unified", madelung_range="full")
        with pytest.raises(ValueError, match="requires madelung_range='off'"):
            _model(image_functional="unified", madelung_range="long_range")
        with pytest.raises(ValueError, match="image_functional"):
            _model(image_functional="both")

    def test_the_periodic_evaluator_is_converged_and_the_legacy_one_is_not_touched(self, model):
        from mace.modules.defect_image import converged_dl

        r_res = float(model.functional["r_res"])
        assert float(model.image_ewald.sigma) == r_res
        assert float(model.image_ewald.ewald.dl) == converged_dl(r_res, 2.0)
        assert float(model.frontier_ewald.ewald.dl) == 2.0

    def test_the_regime_survives_a_config_round_trip(self, model):
        from mace.tools.scripts_utils import extract_config_mace_model

        config = extract_config_mace_model(model)
        assert config["image_functional"] == "unified"
        assert config["madelung_range"] == "off"


# ------------------------------------------------------------------ identities


class TestIdentities:
    def test_the_head_correction_is_an_exact_zero_at_the_reference_state(self, model):
        out, _ = _forward(model, [PRISTINE, VACANCY], [NEUTRAL, NEUTRAL], training=False,
                          compute_force=False)
        assert torch.equal(out["correction_energy"], out["delta_sr_energy"])
        assert float(out["phi_sf_energy"].abs().max()) == 0.0
        assert float(out["phi_img_energy"].abs().max()) == 0.0

    def test_the_registered_terms_sum_to_the_assembled_energy(self, model):
        out, _ = _forward(model, [VACANCY, PRISTINE], [HOLE, NEUTRAL], training=False,
                          compute_force=False)
        energies = dt.term_energies(model, out)
        total = sum(energies[t.name] for t in model.terms())
        assert torch.allclose(total, energies["assembled"], atol=1e-10)
        assert float(energies["static_frontier"][0]) != 0.0
        assert float(energies["image"][0]) != 0.0
        assert float(energies["static_frontier"][1]) == 0.0 == float(energies["image"][1])

    def test_the_two_boundary_diagnostic(self, model):
        """Phi_SF is the same under both boundaries; Phi_img enters only under the periodic
        one, and its periodic value is reported under both (the reference completion)."""
        per, _ = _forward(model, [VACANCY], [HOLE], training=False, compute_force=False)
        model.gauge = "isolated"
        try:
            iso, _ = _forward(model, [VACANCY], [HOLE], training=False, compute_force=False)
        finally:
            model.gauge = "periodic"
        assert float(iso["phi_sf_energy"][0]) == pytest.approx(float(per["phi_sf_energy"][0]),
                                                               rel=1e-12)
        assert float(iso["phi_img_energy"][0]) == 0.0
        assert float(per["phi_img_energy"][0]) != 0.0
        assert float(iso["phi_img_pbc"][0]) == pytest.approx(float(per["phi_img_pbc"][0]),
                                                             rel=1e-12)
        assert float(per["phi_img_pbc_ref"][0]) != float(per["phi_img_pbc"][0])
        # The hole state has no carrier (d = 0): rho_img = rho_S, q_img = Q_core = 1. The
        # reference carries the vacancy electron with its own weight w (section 5.2):
        # q_img = Q_core - w, which is Q_formal = 0 only on the compact plateau w = 1.
        assert float(per["boundary_q_img"][0]) == pytest.approx(1.0)
        w = float(per["boundary_w_min"][0])
        assert 0.0 < w <= 1.0
        assert float(per["boundary_q_img_ref"][0]) == pytest.approx(1.0 - w, abs=1e-9)
        assert per["boundary_lift_fingerprint"][0] is not None
        assert float(per["boundary_clearance_mass"][0]) >= 0.0

    def test_a_wrapped_frame_gives_the_same_energy(self, model):
        """The static density and the lift are covariant under periodic rewrapping."""
        base, _ = _forward(model, [VACANCY], [HOLE], training=False, compute_force=False)
        shifted = VACANCY.copy()
        shifted.positions += np.array([3.1, -2.7, 4.4])
        shifted.wrap()
        moved, _ = _forward(model, [shifted], [HOLE], training=False, compute_force=False)
        assert float(moved["phi_sf_energy"][0]) == pytest.approx(float(base["phi_sf_energy"][0]),
                                                                 rel=1e-8)
        assert float(moved["phi_img_energy"][0]) == pytest.approx(
            float(base["phi_img_energy"][0]), rel=1e-8)


# ------------------------------------------------------------------ the gates


class TestGates:
    def test_two_carriers_are_refused_before_any_energy(self, model):
        with pytest.raises(UnsupportedStateError, match="m_F = 2"):
            _forward(model, [VACANCY], [TWO_CARRIERS], training=False, compute_force=False)

    def test_a_table_without_the_pristine_reference_is_refused(self):
        m = _unified_model()
        frames = [_frame(PRISTINE, NEUTRAL), _frame(VACANCY, NEUTRAL)]
        attach_frame_keys(frames, z_table=Z_TABLE)
        table = dc.ensure_class_table(m, frames, log=False)
        m.composition_classes = {k: v for k, v in table.items() if k != "pristine_reference"}
        with pytest.raises(RuntimeError, match="pristine_reference"):
            _forward(m, [VACANCY], [HOLE], training=False, compute_force=False)

    def test_a_thermal_pristine_background_is_reported_not_treated_as_compact(self, model):
        """Addendum 4.1: displacement dipoles throughout a thermal pristine cell need not
        define a compact density. The static density is built and its mass in the cut
        buffers is reported; the lift refuses to pick a branch for it."""
        rec = dc.lookup_class(model.composition_classes, [17] * 24 + [55] * 8 + [82] * 8)
        thermal = _perovskite(reps=(2, 2, 2), rattle=0.08, seed=23)
        numbers = thermal.get_atomic_numbers()
        zs = [int(z) for z in model.atomic_numbers]
        species = torch.tensor([zs.index(int(z)) for z in numbers])
        z0 = dc.species_charges(model)
        pos = torch.tensor(thermal.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(thermal.get_cell()), dtype=torch.float64)
        out = dc.frame_static_densities(model, rec, None, z0[species], pos, cell)
        assert float(out["raw_norm"]) > 0.0
        assert float(out["q_raw"]) == pytest.approx(0.0, abs=1e-10)
        # The pristine class's topology is zero everywhere and its frozen departure signal
        # is exactly zero (the class reference IS the pristine reference): the envelope is
        # identically zero while the thermal density is not -- an UNSUPPORTED state, since
        # every term of the regime calls G_inf on the lift (addendum 4.2; D17).
        with pytest.raises(UnsupportedStateError, match="no defined branch"):
            dc.frame_static_densities(model, rec, None, z0[species], pos, cell, lift=True)
        # Against ANY branch the thermal background fills the buffers: nothing compact here.
        from mace.modules.defect_lift import build_lift

        branch = build_lift(-torch.ones(1), torch.tensor([[0.5, 0.5, 0.5]]))
        report = clearance_report(out["raw"].centres @ torch.linalg.inv(cell),
                                  out["raw"].charges, cell, branch)
        assert report["buffer_fraction"] > 0.3


# ------------------------------------------------------------------ section 7.2 / 11.1


@pytest.fixture(scope="module")
def fd_setup(model):
    ctx = ForwardContext.production(model)
    batch = _batch([VACANCY], [HOLE], cutoff=ctx.cutoff)
    return ctx.forward_dict(batch)


class TestFiniteDifferences:
    COMPONENTS = [(0, 0), (5, 2), (17, 1)]
    TERMS = ["static_frontier", "image", "assembled"]

    @pytest.mark.parametrize("gauge", ["periodic", "isolated"])
    def test_the_forces_are_the_derivatives_of_the_terms(self, model, fd_setup, gauge):
        model.gauge = gauge
        try:
            reports = {r.term: r for r in fd.force_check(
                model, fd_setup, components=self.COMPONENTS, tol=1e-5, terms=self.TERMS)}
        finally:
            model.gauge = "periodic"
        for term in self.TERMS:
            assert reports[term].status == "pass", (term, reports[term].fit)
        assert reports["assembled_model"].status == "pass"

    @pytest.mark.parametrize("gauge", ["periodic", "isolated"])
    def test_the_stresses_are_the_strain_derivatives(self, model, fd_setup, gauge):
        """Homogeneous strain, with the registered pristine mapping co-deforming (the
        affine-reference strain test of section 8, Stage 4)."""
        model.gauge = gauge
        try:
            reports = {r.term: r for r in fd.strain_check(model, fd_setup, tol=1e-6,
                                                          terms=self.TERMS)}
        finally:
            model.gauge = "periodic"
        for term in self.TERMS:
            assert reports[term].status == "pass", (term, reports[term].fit)
