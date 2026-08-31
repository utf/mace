"""The two-timescale base must release on schedule and roll back when it costs localisation.

Both halves matter. Never releasing reproduces E1's hard freeze, which cost 20-33 meV/A on
forces. Releasing without a guard risks the base quietly re-absorbing the site signal after
the screen's decisive epochs -- the M1 laundering the whole cycle exists to avoid -- with
nothing in the logs to show it happened.
"""

import numpy as np
import torch

from mace.modules.defect_release import BaseRelease


class Toy(torch.nn.Module):
    """Two named parameter groups standing in for base and correction."""

    def __init__(self):
        super().__init__()
        self.readouts = torch.nn.Linear(3, 3)          # base-side name
        self.carrier_head = torch.nn.Linear(3, 3)      # correction-side name


def make(tol=0.05, release_epoch=30, factor=0.01):
    model = Toy()
    base = [p for n, p in model.named_parameters() if n.startswith("readouts")]
    corr = [p for n, p in model.named_parameters() if n.startswith("carrier_")]
    opt = torch.optim.AdamW([
        {"name": "readouts", "params": base, "lr": 0.0},
        {"name": "carrier", "params": corr, "lr": 0.01},
    ])
    return model, opt, BaseRelease(model, opt, release_epoch, factor, tol)


def base_lr(opt):
    return [g["lr"] for g in opt.param_groups if g.get("name") == "readouts"][0]


def test_base_stays_frozen_before_the_release_epoch():
    model, opt, rel = make()
    for e in range(0, 30):
        rel.on_epoch_start(e, base_lr=0.01, hub_mass=0.8)
    assert not rel.released
    assert base_lr(opt) == 0.0


def test_release_sets_the_slow_learning_rate():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, hub_mass=0.8)
    assert rel.released
    assert np.isclose(base_lr(opt), 0.01 * 0.01)


def test_no_revert_when_localisation_holds():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, hub_mass=0.80)
    rel.on_epoch_end(31, hub_mass=0.79)      # within tolerance
    assert not rel.reverted
    assert base_lr(opt) > 0.0


def test_revert_restores_the_release_epoch_weights_and_refreezes():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, hub_mass=0.80)
    saved = model.readouts.weight.detach().clone()

    with torch.no_grad():                     # the base drifts after release
        model.readouts.weight.add_(1.0)
    rel.on_epoch_end(35, hub_mass=0.60)       # hub mass collapsed

    assert rel.reverted
    assert torch.allclose(model.readouts.weight, saved), "base was not rolled back"
    assert base_lr(opt) == 0.0, "base was not refrozen after the revert"


def test_revert_leaves_the_correction_alone():
    """Only the base is under the guard; the head must keep whatever it learned."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, hub_mass=0.80)
    with torch.no_grad():
        model.carrier_head.weight.add_(2.0)
    moved = model.carrier_head.weight.detach().clone()
    rel.on_epoch_end(35, hub_mass=0.10)
    assert torch.allclose(model.carrier_head.weight, moved)


def test_revert_happens_at_most_once():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, hub_mass=0.80)
    rel.on_epoch_end(31, hub_mass=0.10)
    first = model.readouts.weight.detach().clone()
    with torch.no_grad():
        model.readouts.weight.add_(5.0)
    rel.on_epoch_end(32, hub_mass=0.05)       # already reverted; must not fire again
    assert not torch.allclose(model.readouts.weight, first)


def test_missing_hub_mass_is_not_treated_as_collapse():
    """A frame batch with no measurable hub mass must not trigger a spurious rollback."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, hub_mass=0.80)
    rel.on_epoch_end(31, hub_mass=None)
    assert not rel.reverted
