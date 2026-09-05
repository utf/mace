"""Decision 21: the class table's alignment follows the head; the integers do not.

The trainer refreshes the table every epoch from a fresh construction: the aligned edges and
the placement are taken from the head as it is now, a class uncounted before adopts a fresh
count, a counted class keeps its integers (a disagreeing fresh count is a logged failed
invariance). And an uncounted class is a transient of the untrained head: in a training
forward its graphs contribute an exact zero with a warning; in evaluation the class is
refused.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from mace import data
from mace.modules import defect_composition as dc
from mace.modules.defect_cache import attach_frame_keys
from mace.modules.defect_protocol import apply_harrison
from tests.extensions.defect.test_neutral_reference_skip import (Z_TABLE, _batch, _model,
                                                                  _perovskite)

torch.set_default_dtype(torch.float64)
PRISTINE = _perovskite(reps=(2, 2, 2), rattle=0.02, seed=1)
HOLE = [0.0, 0.0, 1.0, 0.0]


def _frame(atoms):
    config = data.Configuration(
        atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
        cell=np.array(atoms.get_cell()), pbc=(True, True, True),
        properties={"carrier_counts": [0.0] * 4}, property_weights={})
    return data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=6.0)


def _frames():
    fr = [_frame(PRISTINE)]
    attach_frame_keys(fr, z_table=Z_TABLE)
    return fr


class TestRefresh:
    def test_a_rigid_head_shift_moves_the_edges_and_not_the_integers(self):
        model = _model()
        apply_harrison(model, model.atomic_numbers)
        frames = _frames()
        dc.ensure_class_table(model, frames, log=False)
        before = dc.lookup_class(model.composition_classes, PRISTINE.get_atomic_numbers())
        with torch.no_grad():
            model.spectral.c_shift.add_(0.7)          # every level up by 0.7 eV
        summary = dc.refresh_class_table(model, frames, log=False)
        after = dc.lookup_class(model.composition_classes, PRISTINE.get_atomic_numbers())
        assert after.vbm_al == pytest.approx(before.vbm_al + 0.7, abs=1e-6)
        assert after.cbm_al == pytest.approx(before.cbm_al + 0.7, abs=1e-6)
        assert (after.n_e, after.n_h, after.q_core, after.m_vb) == (
            before.n_e, before.n_h, before.q_core, before.m_vb)
        assert summary["refreshed"] == [before.key] and not summary["disagree"]

    def test_an_uncounted_class_is_adopted_once_the_head_counts_it(self):
        """The random-initialised head has no gap (the pristine class is uncounted); after
        the Harrison initialisation the refresh counts it."""
        model = _model()
        frames = _frames()
        dc.ensure_class_table(model, frames, log=False)
        rec = dc.lookup_class(model.composition_classes, PRISTINE.get_atomic_numbers())
        assert not rec.counted
        apply_harrison(model, model.atomic_numbers)
        summary = dc.refresh_class_table(model, frames, log=False)
        rec = dc.lookup_class(model.composition_classes, PRISTINE.get_atomic_numbers())
        assert rec.counted and rec.q_core == 0
        assert summary["adopted"] == [rec.key]


class TestTrainingTransient:
    def test_training_gives_zero_with_a_warning_and_evaluation_refuses(self, caplog):
        model = _model()                      # gapless: the pristine class is uncounted
        dc.ensure_class_table(model, _frames(), log=False)
        assert not dc.lookup_class(model.composition_classes,
                                   PRISTINE.get_atomic_numbers()).counted
        batch = _batch([PRISTINE], [HOLE])
        d = batch.to_dict()
        d["positions"].requires_grad_(True)
        with caplog.at_level(logging.WARNING):
            out = model(d, training=True, compute_force=True)
        assert torch.equal(out["frontier_energy"], torch.zeros(1))
        assert any("uncounted" in r.message for r in caplog.records)
        with pytest.raises(ValueError, match="no core/frontier decomposition"):
            model(batch.to_dict(), training=False, compute_force=False)
        # inside a training session the trainer sets the policy, and evaluation forwards
        # (the base-cache build, validation) give the same zero
        model.uncounted_class_policy = "zero"
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
        assert torch.equal(out["frontier_energy"], torch.zeros(1))
        assert model.__class__.__name__ == "MACEDefect"
        # the policy is not a config key and a fresh construction refuses
        from mace.tools.scripts_utils import extract_config_mace_model

        assert "uncounted_class_policy" not in extract_config_mace_model(model)
