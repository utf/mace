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

    def test_the_v6_band_force_omits_the_feature_path_and_the_harness_says_so(self):
        """The counting head detaches the trunk features at entry, so `d(delta_sr)/dR`
        lacks `dH/dh . dh/dR`; the numerical derivative has it. Section 7.2 calls this an
        h-independent floor and section 2.5 forbids it ("no detached quantity anywhere").

        On the small fixture the on-site channel's feature sensitivity is ~1e-6 eV/A and
        the omission sits below tolerance -- which is a statement about the fixture, not
        about the derivative. The channel's last layer is scaled up here so the omitted
        piece is 5e-3 eV/A, and the verdict is then a floor with A = eps = 0: flat across a
        factor of a hundred in h. Stage 1 removes the detach; this expectation flips there.
        """
        model = _model_no_lr()
        with torch.no_grad():
            last = [m for m in model.spectral.h.site.modules()
                    if isinstance(m, torch.nn.Linear)][-1]
            last.weight.mul_(100.0)
        ctx = ForwardContext.production(model)
        batch = _batch([_perovskite(seed=1, reps=(2, 2, 2))], [[0.0, 0.0, 1.0, 0.0]],
                       cutoff=ctx.cutoff)
        data = ctx.forward_dict(batch)
        reports = {r.term: r for r in fd.force_check(
            model, data, components=[(0, 0), (7, 2), (21, 1)], tol=1e-5,
            terms=["band"])}
        assert reports["band"].status == "missing_derivative", reports["band"].fit
        assert reports["band"].fit["floor"] > 1e-3

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

    def test_the_v6_band_stress_misses_the_cell_dependence_of_the_madelung_shift(self, setup):
        """The forward hands the head `data["cell"]`, which the displacement machinery
        never displaces, so the Madelung shift's cell derivative is absent from the
        analytic stress while a strain of the geometry has it. Recorded here; Stage 1."""
        model, data = setup
        reports = {r.term: r for r in fd.strain_check(model, data, tol=1e-6,
                                                      terms=["band"])}
        assert reports["band"].status != "pass", reports["band"].fit


def test_frame_selection_by_the_models_own_gap():
    gaps = [0.9, 0.05, 0.4, 0.6, 0.02, 0.5]
    sel = fd.select_frames(gaps, n_ordinary=1, n_crossing=2)
    assert sel["near_crossing"] == [4, 1]
    assert sel["ordinary"] == [5]                      # the upper median of six gaps
    assert sel["in_projector_window"] == []
