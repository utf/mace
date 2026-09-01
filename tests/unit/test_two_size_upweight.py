"""The two-size upweight must hit its target share and stay label-free.

The 17 charged 159-atom frames carry the only measurement that distinguishes a bound carrier
from a band state (Test 2: R_DFT = 0.95). At natural weighting they are ~1.6% of the charged
force loss. The share must be computed in weight x atoms, not frame count: a 159-atom frame
already contributes twice the terms, and counting frames would overshoot 2x.
"""

import torch

from mace.data.two_size import apply_two_size_upweight


class FakeData:
    def __init__(self, n, charged, weight=1.0):
        self.positions = torch.zeros(n, 3)
        self.carrier_counts = torch.tensor([0.0, 0.0, 1.0, 0.0]) if charged \
            else torch.zeros(4)
        self.forces_weight = torch.tensor(weight)


def make(n_small=1030, n_large=17):   # the real ratio: 1030 vs 17
    return ([FakeData(79, True) for _ in range(n_small)]
            + [FakeData(159, True) for _ in range(n_large)]
            + [FakeData(79, False) for _ in range(50)])


def test_realised_share_hits_the_target():
    """At the real 1030:17 ratio the large frames are ~3% of the charged force loss, so
    reaching 25% needs a factor near 10. (With only 100 small frames they would already be
    at 25% and the correct factor is BELOW 1 -- the target is a share, not an increase.)"""
    ds = make()
    factor, share = apply_two_size_upweight(ds, target_share=0.25)
    assert abs(share - 0.25) < 1e-6, f"share {share}"
    assert 5.0 < factor < 20.0, f"factor {factor}"


def test_share_is_weighted_by_atom_count_not_frame_count():
    """A 159-atom frame contributes twice the force terms; frame-counting would overshoot."""
    ds = make(n_small=100, n_large=100)   # equal frames, unequal atoms
    _, share = apply_two_size_upweight(ds, target_share=0.5)
    assert abs(share - 0.5) < 1e-6


def test_neutral_frames_are_untouched():
    ds = make()
    before = [float(d.forces_weight) for d in ds if float(d.carrier_counts.abs().sum()) == 0]
    apply_two_size_upweight(ds, target_share=0.25)
    after = [float(d.forces_weight) for d in ds if float(d.carrier_counts.abs().sum()) == 0]
    assert before == after


def test_small_charged_frames_are_untouched():
    ds = make()
    apply_two_size_upweight(ds, target_share=0.25)
    for d in ds:
        if d.positions.shape[0] == 79 and float(d.carrier_counts.abs().sum()) > 0:
            assert float(d.forces_weight) == 1.0


def test_no_large_frames_is_a_warning_not_a_crash():
    ds = [FakeData(79, True) for _ in range(10)]
    factor, share = apply_two_size_upweight(ds, target_share=0.25)
    assert factor == 1.0 and share == 0.0


def test_idempotent_target_when_applied_to_an_already_weighted_set():
    """Re-applying must re-solve to the same share, not compound the factor."""
    ds = make()
    apply_two_size_upweight(ds, target_share=0.25)
    _, share = apply_two_size_upweight(ds, target_share=0.25)
    assert abs(share - 0.25) < 1e-6
