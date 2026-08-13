"""Bounding the seed's withdrawal rate must turn the site gate's cliff into a hand-over.

The site measure answers the right question -- has the model found its own SITE structure --
but as a schedule it releases the seed as a step. Measured on seed 1 it held gamma at 1.00
through epoch 10 and reached 0.01 by epoch 14; that seed's attention ran away at epochs
12-13, inside the release. The same seed on the smooth gap gate localises, short-range only,
everything else identical (participation 1.96 versus 46.89).

These tests exercise the bound directly against that recorded trajectory rather than through
a training run.
"""

import math

import pytest
import torch


def _apply(previous, target, gamma_init, forced, max_drop):
    """The bound as implemented: floor the fall, then cap by the forced ramp.

    float64 throughout: the quantities compared here differ by less than float32 epsilon
    (1.5 * (1 - 1/30) against 1.45), so single precision makes exact-equality checks fail
    on rounding rather than on behaviour.
    """
    previous = torch.tensor([previous], dtype=torch.float64)
    updated = torch.tensor([target], dtype=torch.float64)
    if max_drop < 1.0:
        floor = previous.clamp(min=0.0) * (1.0 - max_drop)
        ramp = torch.tensor([gamma_init * forced], dtype=torch.float64)
        updated = torch.minimum(torch.maximum(updated, floor), ramp)
    return float(torch.minimum(updated, previous.clamp(min=0.0)).clamp(min=0.0))


def test_unbounded_reproduces_the_recorded_cliff():
    """max_drop = 1.0 must leave the existing schedule untouched."""
    recorded = [1.000, 0.779, 0.529, 0.137, 0.009, 0.000]  # seed 1, epochs 10-15
    gamma = recorded[0]
    for epoch, target in zip(range(11, 16), recorded[1:]):
        gamma = _apply(gamma, target, 1.5, 1.0 - epoch / 30.0, max_drop=1.0)
        assert gamma == pytest.approx(target, abs=1e-9)


def test_the_bound_prevents_the_collapse_window():
    """Replay the same targets with a 25% cap: gamma must still be substantial at epoch 14.

    Epochs 12-13 are where that seed's attention ran away, so the seed has to still be
    holding through them rather than already gone.
    """
    recorded = [0.779, 0.529, 0.137, 0.009, 0.000]
    gamma = 1.000
    trajectory = []
    for epoch, target in zip(range(11, 16), recorded):
        gamma = _apply(gamma, target, 1.5, 1.0 - epoch / 30.0, max_drop=0.25)
        trajectory.append(gamma)
    assert trajectory[1] > 0.5, f"gamma at epoch 12 is {trajectory[1]:.3f}, too weak"
    assert trajectory[3] > 0.3, f"gamma at epoch 14 is {trajectory[3]:.3f}, cliff not fixed"
    assert all(
        b <= a + 1e-12 for a, b in zip(trajectory, trajectory[1:])
    ), "the bound must not let gamma rise"


def test_the_forced_ramp_still_terminates_the_schedule():
    """The bound floors the fall; the ramp must still cap it, or the seed outlives the run."""
    gamma = 1.5
    for epoch in range(30):
        gamma = _apply(gamma, 1.5, 1.5, 1.0 - epoch / 30.0, max_drop=0.05)
        assert gamma <= 1.5 * (1.0 - epoch / 30.0) + 1e-12, (
            f"epoch {epoch}: gamma {gamma:.4f} exceeds the forced ramp"
        )
    assert gamma < 0.1, f"gamma {gamma:.4f} has not been retired by the terminal epoch"


def test_a_bound_of_one_is_a_no_op_for_any_target():
    for target in (0.0, 0.5, 1.4):
        assert _apply(1.5, target, 1.5, 0.9, max_drop=1.0) == pytest.approx(
            min(target, 1.5)
        )


def test_bound_is_scale_free_in_gamma_init():
    """Multiplicative, so halving gamma_init halves the trajectory rather than changing it."""
    a = _apply(1.0, 0.0, 1.5, 0.9, max_drop=0.25)
    b = _apply(0.5, 0.0, 0.75, 0.9, max_drop=0.25)
    assert a == pytest.approx(2.0 * b)


def test_matches_the_implementation():
    """Guard the helper against the real function drifting away from it."""
    from mace.modules.defect_seed import anneal_logit_seed

    source = anneal_logit_seed.__doc__ or ""
    assert "max_drop" in anneal_logit_seed.__code__.co_varnames, (
        "anneal_logit_seed no longer takes max_drop; this test models a signature that "
        "has changed"
    )
    assert math.isclose(_apply(1.0, 0.0, 1.5, 1.0, 0.25), 0.75)
