"""Plan v8.1 section 6.3 / Stage 0 list: every result cache key carries the exact geometry
and cell, the canonical state, the checkpoint/parameter hash, the constructor record, the
boundary and potential-zero convention, the occupation/smearing implementation, the
functional settings, the solver regime and the spectral-gauge record."""
import numpy as np
import torch

from mace.modules.defect_cache import model_fingerprint, result_key
from tests.extensions.defect.test_gauge_wiring import _gauged_model
from tests.extensions.defect.test_frontier_term import PRISTINE, VACANCY

torch.set_default_dtype(torch.float64)


def _geom(atoms):
    return atoms.get_atomic_numbers(), atoms.get_positions(), np.array(atoms.get_cell())


def test_every_field_changes_the_key():
    model, _, _ = _gauged_model()
    fp = model_fingerprint(model)
    assert set(fp) == {"parameters", "base", "gauge", "constructor", "boundary", "occupation",
                       "functional", "solver"}
    assert fp["gauge"] != "ungauged"
    k0 = result_key(model, *_geom(VACANCY), "state-a")
    assert k0 == result_key(model, *_geom(VACANCY), "state-a")
    assert k0 != result_key(model, *_geom(VACANCY), "state-b")          # state
    assert k0 != result_key(model, *_geom(PRISTINE), "state-a")         # geometry
    moved = VACANCY.copy()
    moved.positions[0, 0] += 1e-3
    assert k0 != result_key(model, *_geom(moved), "state-a")           # exact geometry
    strained = VACANCY.copy()
    strained.set_cell(strained.get_cell() * 1.001, scale_atoms=True)
    assert k0 != result_key(model, *_geom(strained), "state-a")        # cell
    # the gauge: a common-mode shift of the head changes mu_g and therefore the key
    with torch.no_grad():
        model.spectral.h.eps0.add_(0.5)
    model._gauge_shift(model.pristine_centre(torch.float64))
    k1 = result_key(model, *_geom(VACANCY), "state-a")
    assert k1 != k0
    fp1 = model_fingerprint(model)
    assert fp1["gauge"] != fp["gauge"] and fp1["parameters"] != fp["parameters"]
    # boundary convention and functional settings
    model.gauge = "isolated"
    assert result_key(model, *_geom(VACANCY), "state-a") != k1
    model.gauge = "periodic"
    model.functional["r_res"] = float(model.functional["r_res"]) * 1.5
    assert result_key(model, *_geom(VACANCY), "state-a") != k1


def test_the_image_regime_its_evaluator_and_the_lift_enter_the_key():
    """Stage 4 (addendum 3.5, 4.2, 6.3): the regime, the converged periodic evaluator and
    the lift parameters are in the model fingerprint; a frame's lift record fingerprint
    goes into the result key through `lift_extra`."""
    from mace.modules.defect_cache import lift_extra
    from mace.modules.defect_protocol import apply_harrison
    from tests.extensions.defect.test_neutral_reference_skip import _model

    legacy = _model()
    unified = _model(image_functional="unified", madelung_range="off")
    for m in (legacy, unified):
        apply_harrison(m, m.atomic_numbers)
    assert model_fingerprint(legacy)["boundary"] != model_fingerprint(unified)["boundary"]
    # madelung_range differs too, so isolate the regime: same range, different regime.
    legacy_off = _model(madelung_range="off")
    apply_harrison(legacy_off, legacy_off.atomic_numbers)
    assert model_fingerprint(legacy_off)["boundary"] != model_fingerprint(unified)["boundary"]
    k = result_key(unified, *_geom(VACANCY), "state-a")
    assert result_key(unified, *_geom(VACANCY), "state-a", extra=lift_extra(["deadbeef"])) != k
    assert (result_key(unified, *_geom(VACANCY), "state-a", extra=lift_extra(["deadbeef"]))
            != result_key(unified, *_geom(VACANCY), "state-a", extra=lift_extra(["cafebabe"])))
