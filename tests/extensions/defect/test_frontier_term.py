"""Plan v8 Stage 1.2: the frontier-frontier image term `Phi_FF` as wired into the forward.

What is pinned, on a Harrison-initialised counting-head toy (a gapped CsPbCl3 2x2x2 cell
and its Cl vacancy) with a class table built by the same helper every driver uses:

* the identities of sections 2.2 and 7.1 -- the head correction is an EXACT zero at
  `S = S_ref` (no graph, bit-identical energy), `int rho_F = q_F` exactly, the registered
  terms sum to the assembled energy, the isolated gauge has no image term;
* the registry's new column: the frontier term's response is in the forces by the
  divided-difference route while its potential is absent from `H` until Stage 5;
* section 7.2 on the toy: the frontier term's force and stress pass the finite-difference
  harness in both gauges, and the density response is a VISIBLE part of that force -- with
  the channel objects held fixed the harness reports a missing derivative;
* the batched and per-graph solver paths hand back the same term;
* the class-table dependence: a frame off its reference state without a table is refused
  with the remedy named, a reference-state batch runs without one;
* section 7.1's continuity: along an on-site sweep that drives the frontier level through
  the projector windows and into a band, and along a geometric path between two thermal
  frames, `E`, `F`, `rho_F` and `w` are continuous (the consecutive change halves with the
  step) while every integer is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mace import data
from mace.modules import defect_composition as dc
from mace.modules import defect_counting as dcount
from mace.modules import defect_fd as fd
from mace.modules import defect_frontier as df
from mace.modules import defect_terms as dt
from mace.modules.defect_cache import attach_frame_keys
from mace.modules.defect_context import ForwardContext
from mace.modules.defect_protocol import apply_harrison
from mace.modules.defect_state import StateBatch
from tests.extensions.defect.test_composition_classes import _remove_cl
from tests.extensions.defect.test_neutral_reference_skip import (Z_TABLE, _batch, _model,
                                                                  _perovskite)

torch.set_default_dtype(torch.float64)

PRISTINE = _perovskite(reps=(2, 2, 2), rattle=0.02, seed=1)
VACANCY = _remove_cl(_perovskite(reps=(2, 2, 2), rattle=0.02, seed=2), 0)
HOLE = [0.0, 0.0, 1.0, 0.0]
NEUTRAL = [0.0, 0.0, 0.0, 0.0]


def _frame(atoms, counts):
    config = data.Configuration(
        atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
        cell=np.array(atoms.get_cell()), pbc=(True, True, True),
        properties={"carrier_counts": counts}, property_weights={})
    return data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=6.0)


def _harrison_model(**overrides):
    model = _model(**overrides)
    apply_harrison(model, model.atomic_numbers)
    return model


@pytest.fixture(scope="module")
def model():
    m = _harrison_model()
    frames = [_frame(PRISTINE, NEUTRAL), _frame(VACANCY, NEUTRAL)]
    attach_frame_keys(frames, z_table=Z_TABLE)
    dc.ensure_class_table(m, frames, log=False)
    return m


def _forward(model, frames, counts, **kw):
    batch = _batch(frames, counts)
    d = batch.to_dict()
    return model(d, **kw), batch


# ------------------------------------------------------------------ identities


class TestIdentities:
    def test_the_class_table_counts_the_vacancy_electron(self, model):
        rec = dc.lookup_class(model.composition_classes, VACANCY.get_atomic_numbers())
        assert rec.counted and rec.n_e == (1, 0) and rec.q_core == 1

    def test_the_head_correction_is_an_exact_zero_at_the_reference_state(self, model):
        out, _ = _forward(model, [PRISTINE, VACANCY], [NEUTRAL, NEUTRAL],
                          training=False, compute_force=False)
        assert torch.equal(out["frontier_energy"], torch.zeros(2))
        assert not out["frontier_energy"].requires_grad
        assert torch.equal(out["correction_energy"], out["delta_sr_energy"])
        assert torch.equal(out["frontier_w"], torch.full((2,), df.NOT_COMPUTED))

    def test_the_registered_terms_sum_to_the_assembled_energy(self, model):
        out, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE],
                          training=False, compute_force=False)
        e = dt.term_energies(model, out)
        assert [t.name for t in model.terms()] == ["base", "band", "frontier"]
        assert torch.allclose(e["base"] + e["band"] + e["frontier"], e["assembled"],
                              atol=1e-12)
        assert float(out["frontier_energy"].abs().min()) > 0.0

    def test_the_vacancy_at_its_own_charge_state_subtracts_its_reference_cloud(self, model):
        """`V_Cl^+`: `n_e = 0`, so `Phi_FF(Q) = 0` and the term is `-Phi_FF(0)` -- the image
        energy of the reference electron, which is negative (an image attraction), so the
        term is positive; and `q_F = 0` is reported."""
        out, _ = _forward(model, [VACANCY], [HOLE], training=False, compute_force=False)
        assert float(out["frontier_q_F"][0]) == 0.0
        assert float(out["frontier_energy"][0]) > 0.0
        assert 0.0 < float(out["frontier_w_ref"][0]) <= 1.0

    def test_the_frontier_density_integrates_to_q_F_exactly(self, model):
        head = model.spectral
        batch = _batch([PRISTINE], [HOLE])
        grabbed = []
        original = head.forward

        def wrapped(*a, **k):
            out = original(*a, **k)
            grabbed.append(out.frontier)
            return out

        head.forward = wrapped
        try:
            with torch.no_grad():
                model(batch.to_dict(), training=False, compute_force=False)
        finally:
            head.forward = original
        entry = df.entries_from_head(grabbed[0])[0]
        rec = dc.lookup_class(model.composition_classes, PRISTINE.get_atomic_numbers())
        state = StateBatch.from_counts(torch.tensor([HOLE]))
        n_e, n_h, q_f = dc.frame_counts(rec, state, 0)
        f = model.functional
        res = df.frontier_site_charges(entry, df.FILL_AT_STATE, n_e, n_h,
                                       (rec.vbm_al, rec.cbm_al, f["delta"], f["delta_s"]),
                                       float(head.t_el), len(PRISTINE), f["p_star"],
                                       f["delta_p"])
        assert float(res["site"].sum()) == pytest.approx(q_f, abs=1e-10)
        assert res["q_F"] == q_f == 1
        assert set(res["weights"]) == {"h0"}, "a hole in the majority channel, nothing else"

    def test_the_isolated_gauge_has_no_image_term(self, model):
        model.gauge = "isolated"
        try:
            out, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE],
                              training=False, compute_force=False)
        finally:
            model.gauge = "periodic"
        assert torch.equal(out["frontier_energy"], torch.zeros(2))
        dilute, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE],
                             training=False, compute_force=False, dilute=True)
        assert torch.equal(dilute["frontier_energy"], torch.zeros(2))
        periodic, _ = _forward(model, [PRISTINE, VACANCY], [HOLE, HOLE],
                               training=False, compute_force=False)
        assert float((periodic["energy"] - dilute["energy"]).abs().min()) > 0.0

    def test_the_uniform_part_carries_no_image_energy(self, model):
        """`B_img[rho_ext, .] = 0`: with the participation switch forced to the extended
        side (`p* -> -inf`, so `w -> 0`) the term vanishes."""
        p_star = model.functional["p_star"]
        model.functional["p_star"] = -50.0
        try:
            out, _ = _forward(model, [PRISTINE], [HOLE], training=False, compute_force=False)
        finally:
            model.functional["p_star"] = p_star
        assert float(out["frontier_w"][0]) < 1e-12
        assert abs(float(out["frontier_energy"][0])) < 1e-20

    def test_the_batched_and_per_graph_solvers_hand_back_the_same_term(self, model):
        together, _ = _forward(model, [PRISTINE, _perovskite(seed=3)], [HOLE, HOLE],
                               training=False, compute_force=False)
        alone = [_forward(model, [fr], [HOLE], training=False, compute_force=False)[0]
                 for fr in (PRISTINE, _perovskite(seed=3))]
        for g in range(2):
            assert float(together["frontier_energy"][g]) == pytest.approx(
                float(alone[g]["frontier_energy"][0]), abs=1e-9)
            assert float(together["frontier_w"][g]) == pytest.approx(
                float(alone[g]["frontier_w"][0]), abs=1e-9)


# ------------------------------------------------------------------ the registry


class TestRegistry:
    def test_the_response_column_separates_the_force_from_the_potential(self, model):
        by_name = {t.name: t for t in model.terms()}
        assert by_name["frontier"].depends_on_P
        assert by_name["frontier"].response == "divided_difference"
        assert by_name["frontier"].potential == "absent", "V_FF enters H at Stage 5"
        assert by_name["band"].response == "band" and by_name["band"].potential == "band"
        assert by_name["base"].response == "none" and by_name["base"].potential == "none"
        assert by_name["frontier"].gauge_dependent and by_name["frontier"].kernel == "periodic"

    def test_no_term_of_the_functional_has_an_absent_response(self, model):
        assert all(t.response != "absent" for t in model.terms())

    def test_a_model_without_the_periodic_evaluator_has_no_frontier_term(self):
        m = _harrison_model(use_long_range=False)
        assert [t.name for t in m.terms()] == ["base", "band"]
        assert not m.frontier_active


# ------------------------------------------------------------------ the class table


class TestClassTableDependence:
    def test_a_frame_off_its_reference_state_needs_the_table(self):
        m = _harrison_model()
        assert m.composition_classes is None
        with pytest.raises(RuntimeError, match="ensure_class_table"):
            _forward(m, [PRISTINE], [HOLE], training=False, compute_force=False)

    def test_a_reference_state_batch_runs_without_one(self):
        m = _harrison_model()
        out, _ = _forward(m, [PRISTINE, VACANCY], [NEUTRAL, NEUTRAL], training=False,
                          compute_force=False)
        assert torch.equal(out["frontier_energy"], torch.zeros(2))

    def test_ensure_class_table_is_idempotent_and_records_the_reference_frames(self, model):
        before = model.composition_classes
        again = dc.ensure_class_table(model, [], log=False)
        assert again is before
        for rec in before["classes"].values():
            assert rec["reference_frame_key"] != -1


# ------------------------------------------------------------------ section 7.2 on the toy


@pytest.fixture(scope="module")
def fd_setup(model):
    ctx = ForwardContext.production(model)
    batch = _batch([VACANCY], [HOLE], cutoff=ctx.cutoff)
    return ctx.forward_dict(batch)


class TestFiniteDifferences:
    COMPONENTS = [(0, 0), (5, 2), (17, 1)]

    @pytest.mark.parametrize("gauge", ["periodic", "isolated"])
    def test_the_frontier_force_is_the_derivative_of_the_frontier_energy(self, model,
                                                                         fd_setup, gauge):
        model.gauge = gauge
        try:
            reports = {r.term: r for r in fd.force_check(
                model, fd_setup, components=self.COMPONENTS, tol=1e-5,
                terms=["frontier", "assembled"])}
        finally:
            model.gauge = "periodic"
        assert reports["frontier"].status == "pass", reports["frontier"].fit
        assert reports["assembled"].status == "pass", reports["assembled"].fit
        assert reports["assembled_model"].status == "pass"

    def test_the_frontier_stress_is_the_strain_derivative(self, model, fd_setup):
        reports = {r.term: r for r in fd.strain_check(model, fd_setup, tol=1e-6,
                                                      terms=["frontier", "assembled"])}
        assert reports["frontier"].status == "pass", reports["frontier"].fit
        assert reports["assembled"].status == "pass", reports["assembled"].fit

    def test_the_density_response_is_a_visible_part_of_the_force(self, model, fd_setup,
                                                                 monkeypatch):
        """Hold the channel objects fixed (the frozen-P force, `dPhi/dR` at fixed P) on the
        analytic side only: the harness must report the missing `Tr(P~ dH/dR)`."""
        original = dcount.channel_matrix

        def frozen(H, *args, **kwargs):
            return original(H.detach(), *args, **kwargs)

        monkeypatch.setattr(df, "channel_matrix", frozen)
        reports = {r.term: r for r in fd.force_check(
            model, fd_setup, components=self.COMPONENTS, tol=1e-5, terms=["frontier"])}
        assert reports["frontier"].status == "missing_derivative", reports["frontier"].fit
        assert reports["frontier"].fit["floor"] > 1e-4


# ------------------------------------------------------------------ section 7.1 continuity


def _sweep(model, atoms, counts, shifts, site):
    """E, F, rho_F (as w * site charges) and w along an on-site sweep of one site."""
    head = model.spectral
    original = head.h.on_site
    out = []
    try:
        for shift in shifts:
            def shifted(feats, species, madelung, centre=None, _s=shift):
                levels = original(feats, species, madelung, centre=centre)
                bump = torch.zeros_like(levels)
                bump[site] = _s
                return levels + bump
            head.h.on_site = shifted
            o, _ = _forward(model, [atoms], [counts], training=False, compute_force=True)
            out.append(dict(E=float(o["energy"][0]), F=o["forces"].detach().clone(),
                            w=float(o["frontier_w"][0]), q=float(o["frontier_q_F"][0]),
                            phi=float(o["frontier_energy"][0]),
                            weight=float(o["frontier_min_weight"][0])))
    finally:
        head.h.on_site = original
    return out


def _max_consecutive(values):
    return max(abs(b - a) for a, b in zip(values[:-1], values[1:]))


class TestContinuity:
    def test_an_on_site_sweep_through_the_windows_into_the_band_is_continuous(self, model):
        """The vacancy's frontier level is driven by the on-site energy of the site that
        carries most of its density, from the gap through both projector windows into the
        band. The integers never move; E, F, w are continuous: the largest consecutive
        change at the fine step is at most ~half the one at the coarse step (a jump would
        survive the halving)."""
        rec = dc.lookup_class(model.composition_classes, VACANCY.get_atomic_numbers())
        o, batch = _forward(model, [VACANCY], [NEUTRAL], training=False, compute_force=False)
        # the site: the Pb with the most vacancy-state weight, read from the head's alpha
        # is unavailable at S_ref, so take the heaviest species nearest the removed Cl
        pos = VACANCY.get_positions()
        removed = _perovskite(reps=(2, 2, 2), rattle=0.02, seed=2).get_positions()[0]
        pb = [i for i, z in enumerate(VACANCY.get_atomic_numbers()) if z == 82]
        site = min(pb, key=lambda i: np.linalg.norm(pos[i] - removed))
        # A step of 0.25 eV against a 0.05 eV projector window, then the step quartered:
        # a smooth curve's largest consecutive change falls by ~4x, a kink's too, a sharp
        # (window-scale) feature's by less, a JUMP's not at all. The bound is the jump.
        coarse = np.linspace(-3.0, 3.0, 25)
        fine = np.linspace(-3.0, 3.0, 97)
        sc = _sweep(model, VACANCY, HOLE, coarse, site)
        sf = _sweep(model, VACANCY, HOLE, fine, site)
        # the sweep does cross the windows: the projector weight moves
        assert max(x["weight"] for x in sf) - min(x["weight"] for x in sf) > 0.2
        for key in ("E", "phi", "w"):
            jump_c = _max_consecutive([x[key] for x in sc])
            jump_f = _max_consecutive([x[key] for x in sf])
            assert jump_f <= 0.5 * jump_c + 1e-9, (key, jump_c, jump_f)
        jump_c = max(float((b["F"] - a["F"]).abs().max()) for a, b in zip(sc[:-1], sc[1:]))
        jump_f = max(float((b["F"] - a["F"]).abs().max()) for a, b in zip(sf[:-1], sf[1:]))
        assert jump_f <= 0.5 * jump_c + 1e-9, ("F", jump_c, jump_f)
        # every integer is unchanged along the sweep
        assert {x["q"] for x in sf} == {0.0}
        state = StateBatch.from_counts(torch.tensor([HOLE]))
        assert dc.frame_counts(rec, state, 0) == ((0, 0), (0, 0), 0)
        assert rec.q_core == 1

    def test_an_integer_changes_only_when_the_counter_does(self, model):
        rec = dc.lookup_class(model.composition_classes, VACANCY.get_atomic_numbers())
        neutral = StateBatch.from_counts(torch.tensor([NEUTRAL]))
        hole = StateBatch.from_counts(torch.tensor([HOLE]))
        assert dc.frame_counts(rec, neutral, 0) == ((1, 0), (0, 0), -1)
        assert dc.frame_counts(rec, hole, 0) == ((0, 0), (0, 0), 0)

    def test_a_geometric_path_between_two_thermal_frames_is_continuous(self, model):
        a = _perovskite(reps=(2, 2, 2), rattle=0.03, seed=11)
        b = _perovskite(reps=(2, 2, 2), rattle=0.03, seed=12)

        def along(n):
            out = []
            for t in np.linspace(0.0, 1.0, n):
                x = a.copy()
                x.positions = (1 - t) * a.positions + t * b.positions
                o, _ = _forward(model, [x], [HOLE], training=False, compute_force=True)
                out.append(dict(E=float(o["energy"][0]), phi=float(o["frontier_energy"][0]),
                                w=float(o["frontier_w"][0]), F=o["forces"].detach()))
            return out

        coarse, fine = along(9), along(33)
        for key in ("E", "phi", "w"):
            jump_c = _max_consecutive([x[key] for x in coarse])
            jump_f = _max_consecutive([x[key] for x in fine])
            assert jump_f <= 0.5 * jump_c + 1e-9, (key, jump_c, jump_f)
        jump_c = max(float((b["F"] - a["F"]).abs().max())
                     for a, b in zip(coarse[:-1], coarse[1:]))
        jump_f = max(float((b["F"] - a["F"]).abs().max()) for a, b in zip(fine[:-1], fine[1:]))
        assert jump_f <= 0.5 * jump_c + 1e-9, ("F", jump_c, jump_f)
