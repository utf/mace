"""Section 1.1: the size-grouped batch sampler.

The properties that matter, and each one is a way the grouping could be wrong while looking
right: every batch has one atom count (or the batched solver never fires), every frame is
delivered exactly once per epoch (or the 34 large frames quietly shrink), the batches are
not delivered group by group (or every large-cell step lands at the same point in the
epoch), and the epoch length matches `__len__` (or the trainer's progress and its learning
rate schedule disagree with what it actually ran).
"""

from collections import Counter

import torch

from mace.data.size_sampler import SizeGroupedBatchSampler, frame_sizes


class _Item:
    def __init__(self, n):
        self.positions = torch.zeros(n, 3)


SIZES = [79] * 23 + [80] * 7 + [159] * 5 + [160] * 1


def _sampler(**kw):
    kw.setdefault("batch_size", 4)
    kw.setdefault("generator", torch.Generator().manual_seed(0))
    return SizeGroupedBatchSampler(SIZES, **kw)


def test_frame_sizes_reads_positions_when_there_is_no_num_nodes():
    assert frame_sizes([_Item(3), _Item(11)]) == [3, 11]


def test_every_batch_holds_one_atom_count():
    for batch in _sampler():
        assert len({SIZES[i] for i in batch}) == 1, batch


def test_every_frame_appears_exactly_once_per_epoch():
    seen = Counter(i for batch in _sampler() for i in batch)
    assert set(seen) == set(range(len(SIZES)))
    assert set(seen.values()) == {1}


def test_the_short_tail_batch_is_kept():
    # 5 frames of 159 at batch 4 is 4 + 1; dropping the tail would lose one large frame an
    # epoch out of a population of five.
    batches = [b for b in _sampler() if SIZES[b[0]] == 159]
    assert sorted(len(b) for b in batches) == [1, 4]
    assert sum(len(b) for b in batches) == 5


def test_drop_last_drops_per_group_when_asked():
    batches = list(_sampler(drop_last=True))
    assert all(len(b) == 4 for b in batches)
    # 23//4 + 7//4 + 5//4 + 1//4 = 5 + 1 + 1 + 0
    assert len(batches) == 7


def test_len_matches_the_epoch():
    s = _sampler()
    assert len(list(s)) == len(s)


def test_batches_are_shuffled_across_groups():
    # Group-then-sequential would put all six 79-atom batches first. The test is that the
    # sizes are not sorted -- a run of one size at a fixed point in the epoch is a change to
    # the optimiser's trajectory, not a speed fix.
    order = [SIZES[b[0]] for b in _sampler()]
    assert order != sorted(order)


def test_shuffle_off_is_deterministic_and_ordered():
    a = list(SizeGroupedBatchSampler(SIZES, batch_size=4, shuffle=False))
    b = list(SizeGroupedBatchSampler(SIZES, batch_size=4, shuffle=False))
    assert a == b
    assert [SIZES[x[0]] for x in a] == sorted(SIZES[x[0]] for x in a)


def test_two_epochs_differ_but_still_cover_everything():
    s = _sampler()
    first = [tuple(b) for b in s]
    second = [tuple(b) for b in s]
    assert first != second
    for epoch in (first, second):
        assert sorted(i for b in epoch for i in b) == list(range(len(SIZES)))


def test_describe_names_every_group():
    text = _sampler().describe()
    for n in (79, 80, 159, 160):
        assert f"{n} atoms" in text
