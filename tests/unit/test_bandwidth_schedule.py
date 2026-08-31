"""The bandwidth anneal must actually be driven, not merely be drivable.

`test_bandwidth_anneal.py` covers what the head does when hop_scale is set. It sets the buffer
by hand in every case, so it passed while NOTHING in production ever wrote hop_scale:
`--defect_spectral_anneal_s0` was declared in the parser, passed by the launcher and described
in the run script's header, but the value reached no code. hop_scale stayed at its registered
1.0 and the anneal arm of R2 was byte-identical to the plain arm.

That is the same class of defect as the flag-plumbing gaps, with one extra hiding place: the
setting is consumed by the TRAINING LOOP rather than by the model constructor, so a
constructor round-trip test cannot see it either. These tests pin both halves -- the schedule's
values, and the fact that the training loop assigns them.
"""

import re
from pathlib import Path

import pytest

from mace.modules.defect_spectral import bandwidth_scale

REPO = Path(__file__).resolve().parents[2]
RUN_TRAIN = REPO / "mace" / "cli" / "run_train.py"


def test_starts_at_s0_and_ends_at_one():
    assert bandwidth_scale(0, 4.0, 20) == pytest.approx(4.0)
    assert bandwidth_scale(20, 4.0, 20) == pytest.approx(1.0)


def test_midpoint_is_the_geometric_mean():
    """s(E_a/2) = s0^(1/2); the schedule is geometric in the epoch, not linear."""
    assert bandwidth_scale(10, 4.0, 20) == pytest.approx(2.0)


def test_monotone_decreasing_over_the_anneal():
    vals = [bandwidth_scale(e, 4.0, 20) for e in range(21)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_clamped_after_the_anneal_completes():
    """Past E_a the exponent would go negative and shrink the band below its trained width."""
    for e in (21, 40, 500):
        assert bandwidth_scale(e, 4.0, 20) == pytest.approx(1.0)


def test_disabled_returns_unity():
    for s0 in (0.0, -1.0):
        assert bandwidth_scale(0, s0, 20) == pytest.approx(1.0)
        assert bandwidth_scale(7, s0, 20) == pytest.approx(1.0)


def test_zero_anneal_epochs_does_not_divide_by_zero():
    assert bandwidth_scale(0, 4.0, 0) == pytest.approx(1.0)


def test_negative_epoch_is_treated_as_the_start():
    assert bandwidth_scale(-3, 4.0, 20) == pytest.approx(4.0)


def test_the_training_loop_actually_writes_hop_scale():
    """The half that was missing: a schedule nothing calls is not a schedule.

    Asserted against the source because the write happens inside a closure over the trainer's
    epoch loop, which cannot be constructed here without a full training run.
    """
    text = RUN_TRAIN.read_text()
    assert "defect_spectral_anneal_s0" in text, (
        "run_train never reads args.defect_spectral_anneal_s0; the flag is inert")
    assert "bandwidth_scale" in text, "run_train does not use the schedule"
    assert re.search(r"hop_scale\.fill_", text), (
        "run_train never assigns hop_scale, so the anneal cannot take effect")


def test_the_anneal_is_applied_before_the_gradient_steps():
    """It must be in epoch_hook, not post_eval_hook.

    post_eval_hook fires after validation, i.e. after the epoch's optimizer steps have already
    run at the previous bandwidth, and only on eval epochs -- so on any eval_interval > 1 the
    schedule would skip values entirely.
    """
    text = RUN_TRAIN.read_text()
    hook_start = text.index("def defect_seed_hook")
    hook_end = text.index("def _release_hook")
    assert "hop_scale.fill_" in text[hook_start:hook_end], (
        "hop_scale is not assigned inside defect_seed_hook (the pre-gradient epoch_hook)")
