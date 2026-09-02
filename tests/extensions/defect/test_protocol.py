"""The Stage-3 protocol, now in the package so the trainer and the harness share it.

Section 3 established that the config path builds a bit-identical MODEL. A model is not a
run: the Harrison initialisation, the c-shift, the warmup, `loss_gap` and the init gate are
what make a Stage-3 run work, and they lived in a harness script the production trainer
cannot see. These tests pin the pieces that have a right answer.
"""

from __future__ import annotations

import pytest
import torch

from mace.modules import defect_protocol as protocol


class TestCShift:
    """`E_head` moves by `c * Delta_n` and by nothing else, so one scalar solves the median
    mismatch exactly."""

    @staticmethod
    def batch(resid, dn):
        out = {"base_energy": torch.zeros(len(resid)),
               "delta_sr_energy": torch.zeros(len(resid))}
        counts = torch.zeros(len(dn), 4)
        for i, n in enumerate(dn):
            counts[i, 0 if n > 0 else 2] = abs(n)
        return out, torch.tensor(resid, dtype=torch.get_default_dtype()), counts

    def test_it_recovers_a_known_offset(self):
        out, target, counts = self.batch([2.0, 4.0, -2.0], [1, 2, -1])
        assert protocol.calibrate_c_shift(out, target, counts) == pytest.approx(2.0)

    def test_it_takes_the_median_not_the_mean(self):
        """One badly extrapolated frame must not set the offset for the whole run."""
        out, target, counts = self.batch([2.0, 2.0, 2.0, 200.0], [1, 1, 1, 1])
        assert protocol.calibrate_c_shift(out, target, counts) == pytest.approx(2.0)

    def test_a_neutral_batch_returns_none_rather_than_zero(self):
        """Delta_n = 0 makes the ratio undefined. Returning 0.0 would look like a
        calibration that had happened."""
        out, target, counts = self.batch([1.0, 2.0], [0, 0])
        assert protocol.calibrate_c_shift(out, target, counts) is None

    def test_no_energy_label_returns_none(self):
        out, _, counts = self.batch([1.0], [1])
        assert protocol.calibrate_c_shift(out, None, counts) is None


class TestWarmup:
    def test_it_is_linear_and_reaches_one(self):
        assert protocol.warmup_factor(0, 5) == pytest.approx(0.2)
        assert protocol.warmup_factor(4, 5) == pytest.approx(1.0)
        assert protocol.warmup_factor(50, 5) == pytest.approx(1.0)

    def test_zero_warmup_is_a_no_op(self):
        assert protocol.warmup_factor(0, 0) == 1.0


class TestTrainableMask:
    def test_the_head_and_the_charges_train(self):
        assert protocol.trainable_mask("spectral.h.eps0")
        assert protocol.trainable_mask("madelung.z")

    def test_freeze_z_pins_only_the_charges(self):
        assert not protocol.trainable_mask("madelung.z", freeze_z=True)
        assert protocol.trainable_mask("spectral.h.eps0", freeze_z=True)

    def test_the_trunk_does_not(self):
        assert not protocol.trainable_mask("interactions.0.linear.weight")


class TestProtocolSummary:
    def test_it_records_the_live_smearing(self):
        """A run whose artefact does not record its own protocol cannot be compared to
        anything later.

        `c_shift_calibrated` and `harrison_init` are now OUTCOMES, so a summary written
        without either being reported comes back false. That is the point: this call passes
        neither, so neither happened as far as the artefact is concerned.
        """
        s = protocol.protocol_summary(stage=3, e_gap=2.4, w_gap=1.0, warmup=5,
                                      clip=1.0, freeze_z=False)
        assert s["smearing_family"] == "gaussian"
        assert s["smearing_width"] == pytest.approx(0.05)
        assert s["harrison_init"] and not s["c_shift_calibrated"]
        done = protocol.protocol_summary(stage=3, e_gap=2.4, w_gap=1.0, warmup=5,
                                         clip=1.0, freeze_z=False, c_shift=-0.5,
                                         harrison_applied=True)
        assert done["harrison_init"] and done["c_shift_calibrated"]

    def test_stage_two_reports_no_harrison(self):
        s = protocol.protocol_summary(stage=2, e_gap=2.4, w_gap=1.0, warmup=0,
                                      clip=10.0, freeze_z=True)
        assert not s["harrison_init"] and not s["c_shift_calibrated"]
        assert s["freeze_z"]


class TestPostStep:
    def test_the_projection_runs_and_a_model_without_madelung_is_fine(self):
        from mace.modules.defect_madelung import MadelungOnSite

        class Fake:
            pass

        empty = Fake()
        empty.madelung = None
        protocol.post_step(empty)                      # must not raise

        m = Fake()
        m.madelung = MadelungOnSite(num_elements=3, composition=[3.0, 1.0, 1.0],
                                    z_init=[-1.0, 1.0, 2.0])
        with torch.no_grad():
            m.madelung.z += 1.0                        # push off the hyperplane
        protocol.post_step(m)
        n = m.madelung.composition
        assert float(torch.dot(n, m.madelung.z.detach())) == pytest.approx(0.0, abs=1e-6)
