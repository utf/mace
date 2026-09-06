"""Plan v8 section 7.2: the finite-difference harness measures what it claims to.

Three things are pinned. The classifier reads the `A h^2 + epsilon/h` pattern the way the
plan describes it -- a clean minimum is a pass, a flat curve far above the fitted noise is a
missing derivative. The trunk's force and stress, which are exact autograd of the energy,
pass on a real (small) model. And the harness is not vacuous: a derivative the v6 forward
is KNOWN to omit is reported as missing, not as a pass. That last assertion is the record
of the v6 status this stage exists to write down, and Stage 1 -- where per-term FD becomes
pass-required -- is where it is expected to flip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mace.modules import defect_fd as fd
from mace.modules.defect_context import ForwardContext
from tests.extensions.defect.test_neutral_reference_skip import _batch, _perovskite
from tests.unit.test_base_cache_precision import _model as _model_no_lr

torch.set_default_dtype(torch.float64)


class TestClassifier:
    def test_a_clean_minimum_below_tolerance_passes(self):
        h = fd.H_VALUES
        err = [2.0 * x ** 2 + 1e-9 / x for x in h]          # A = 2, eps = 1e-9
        status, info = fd.classify(h, err, tol=1e-4)
        # d/dh (A h^2 + eps/h) = 0 at h = (eps / 2A)^(1/3) = 6.3e-4; nearest grid point 1e-3.
        assert status == "pass" and info["h_at_min"] == 1e-3

    def test_a_flat_curve_far_above_the_noise_is_a_missing_derivative(self):
        h = fd.H_VALUES
        err = [0.05 + 2.0 * x ** 2 + 1e-9 / x for x in h]   # a fixed 0.05 eV/A offset
        status, info = fd.classify(h, err, tol=1e-4)
        assert status == "missing_derivative", info

    def test_noise_that_dominates_is_a_failure_not_a_floor(self):
        h = fd.H_VALUES
        err = [1e-5 / x for x in h]                           # pure epsilon / h
        status, _ = fd.classify(h, err, tol=1e-4)
        assert status == "fail"

    def test_the_fit_recovers_its_own_coefficients(self):
        h = fd.H_VALUES
        A, eps, c = fd.fit_floor(h, [0.01 + 3.0 * x ** 2 + 2e-8 / x for x in h])
        assert A == pytest.approx(3.0, rel=1e-6) and eps == pytest.approx(2e-8, rel=1e-6)
        assert c == pytest.approx(0.01, rel=1e-6)


@pytest.fixture(scope="module")
def setup():
    model = _model_no_lr()
    ctx = ForwardContext.production(model)
    batch = _batch([_perovskite(seed=1, reps=(2, 2, 2))], [[0.0, 0.0, 1.0, 0.0]],
                   cutoff=ctx.cutoff)
    data = ctx.forward_dict(batch)
    return model, data


class TestForces:
    def test_the_trunk_force_is_the_derivative_of_the_trunk_energy(self, setup):
        model, data = setup
        reports = {r.term: r for r in fd.force_check(
            model, data, components=[(0, 0), (7, 2), (21, 1)], tol=1e-5,
            terms=["base"])}
        assert reports["base"].status == "pass", reports["base"].fit

    def test_the_band_force_carries_the_feature_path(self):
        """STAGE 1 (plan v8 section 4): the counting head no longer detaches the trunk
        features at entry, so `d(delta_sr)/dR` carries `dH/dh . dh/dR`, and so does the
        Madelung per-site charge channel. At Stage 0 this test asserted the OMISSION (an
        h-independent floor once the on-site channel's last layer is scaled up so the
        omitted piece is visible on the small fixture); the same amplification now has to
        pass, and with the features held fixed on the analytic side only, the harness has
        to report the omission.

        THE FIXTURE IS CHOSEN, NOT DRAWN. How large the feature path is on this toy depends
        on the random readout and on-site draws (a parameter added to or removed from the
        head shifts every later draw), and on one draw the omitted piece sat below the
        harness's tolerance and the test read "pass" for the wrong reason. So the readout
        and on-site MLPs are re-drawn over a few seeds until the omitted piece is visible
        (> 1e-3 eV/A on some component), and the harness is asked about THAT component.
        """
        model, data, component, visible = None, None, None, 0.0
        for seed in range(12):
            model = _model_no_lr()
            torch.manual_seed(100 + seed)
            for module in list(model.defect_feature_readouts[0].modules()) + list(
                    model.spectral.h.site.modules()):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            with torch.no_grad():
                last = [m for m in model.spectral.h.site.modules()
                        if isinstance(m, torch.nn.Linear)][-1]
                last.weight.mul_(100.0)
            ctx = ForwardContext.production(model)
            batch = _batch([_perovskite(seed=1, reps=(2, 2, 2))], [[0.0, 0.0, 1.0, 0.0]],
                           cutoff=ctx.cutoff)
            data = ctx.forward_dict(batch)
            attached, _ = fd._analytic_forces(model, data)
            readouts = model.defect_feature_readouts
            originals = [r.forward for r in readouts]
            try:
                for r, f in zip(readouts, originals):
                    r.forward = (lambda f: (lambda x: f(x).detach()))(f)
                detached, _ = fd._analytic_forces(model, data)
            finally:
                for r, f in zip(readouts, originals):
                    r.forward = f
            diff = (attached["band"] - detached["band"]).abs()
            visible = float(diff.max())
            if visible > 1e-3:
                flat = int(diff.argmax())
                component = (flat // 3, flat % 3)
                break
        assert component is not None, f"no draw with a visible feature path (max {visible:.2e})"
        reports = {r.term: r for r in fd.force_check(
            model, data, components=[component], tol=1e-5, terms=["band"])}
        assert reports["band"].status == "pass", reports["band"].fit
        # ... and with the features held fixed on the analytic side only, the omission is
        # what the harness reports: an h-independent floor, not a fit.
        readouts = model.defect_feature_readouts
        originals = [r.forward for r in readouts]
        try:
            for r, f in zip(readouts, originals):
                r.forward = (lambda f: (lambda x: f(x).detach()))(f)
            omitted = {r.term: r for r in fd.force_check(
                model, data, components=[component], tol=1e-5, terms=["band"])}
        finally:
            for r, f in zip(readouts, originals):
                r.forward = f
        assert omitted["band"].status == "missing_derivative", omitted["band"].fit
        assert omitted["band"].fit["floor"] > 1e-3

    def test_the_model_force_equals_the_autograd_of_its_reported_energy(self, setup):
        """`assembled_model` compares the model's `forces` output to the numerical
        derivative of the reported total; whatever the per-term verdicts, the two must
        carry the same status as the autograd assembled term."""
        model, data = setup
        reports = {r.term: r for r in fd.force_check(
            model, data, components=[(3, 1)], tol=1e-5, terms=["assembled"])}
        assert reports["assembled"].status == reports["assembled_model"].status
        for h in fd.H_VALUES:
            a = reports["assembled"].points[0].numerical[h]
            b = reports["assembled_model"].points[0].numerical[h]
            assert a == b


class TestStrain:
    def test_the_trunk_stress_is_the_strain_derivative_of_the_trunk_energy(self, setup):
        model, data = setup
        reports = {r.term: r for r in fd.strain_check(model, data, tol=1e-6,
                                                      terms=["base"])}
        assert reports["base"].status == "pass", reports["base"].fit

    def test_the_band_stress_carries_the_madelung_cell_and_position_dependence(self, setup):
        """STAGE 1: the forward hands the head the DISPLACED cell and the DISPLACED
        positions, so the Madelung shift's strain derivative is in the analytic stress. At
        Stage 0 this test asserted the omission (`data["cell"]` undisplaced; `ctx.positions`
        the undisplaced leaf)."""
        model, data = setup
        reports = {r.term: r for r in fd.strain_check(model, data, tol=1e-6,
                                                      terms=["band"])}
        assert reports["band"].status == "pass", reports["band"].fit


def test_frame_selection_by_the_models_own_gap():
    gaps = [0.9, 0.05, 0.4, 0.6, 0.02, 0.5]
    sel = fd.select_frames(gaps, n_ordinary=1, n_crossing=2)
    assert sel["near_crossing"] == [4, 1]
    assert sel["ordinary"] == [5]                      # the upper median of six gaps
    assert sel["in_projector_window"] == []
