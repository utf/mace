"""Stage B must normalise its messages exactly as Stage A did.

WHY THIS TEST EXISTS. `avg_num_neighbors` is a plain float on each interaction block, dividing
every message. It is not a parameter and not a buffer, so it does not appear in `state_dict` --
which means `load_stage_a_base`'s name-and-shape check cannot see it and its copy loop cannot
carry it. A Stage-B model built with a different value loads Stage A's weights into a trunk
that normalises them differently, the loader reports success, and the base branch silently
stops reproducing Stage A.

The mismatch is real and large. Stage A trained at `r_max = 5.0` with no carrier head and got
14.08. A Stage-B run with the spectral head builds its graph at the CARRIER cutoff of 10 A and
computes 112.5 on the same data: eight times the divisor on every message.

How it was found is worth recording, because nothing else caught it. The c-shift calibration is
the median of `(E_label - E_base - E_head)/Delta_n`, so it is a direct read on `E_base`; the
harness and the trainer disagreed about it by a factor of four on identical data. Frame
selection, the forward path, the referencing, the head initialisation and the atomic energies
were each measured and eliminated, and the trunk normalisation was what was left.
"""

from __future__ import annotations

import logging

import pytest
import torch

from mace.modules.defect_stage import load_stage_a_base


class _Block(torch.nn.Module):
    def __init__(self, ann: float):
        super().__init__()
        self.avg_num_neighbors = float(ann)
        self.linear = torch.nn.Linear(2, 2, bias=False)


class _Model(torch.nn.Module):
    def __init__(self, ann: float, n_blocks: int = 2):
        super().__init__()
        self.interactions = torch.nn.ModuleList(_Block(ann) for _ in range(n_blocks))
        self.register_buffer("atomic_energies", torch.zeros(3))


def _save(tmp_path, model, name):
    path = tmp_path / name
    torch.save(model, path)
    return path


def test_avg_num_neighbors_is_invisible_to_state_dict():
    """The premise. If this ever becomes a buffer, the copy loop covers it and the special
    case below is redundant -- but it must not be removed on the assumption that it is."""
    m = _Model(14.08)
    assert not any("avg_num_neighbors" in k for k in m.state_dict())


def test_stage_b_inherits_stage_a_normalisation(tmp_path, caplog):
    source = _Model(14.08)
    target = _Model(112.5)
    path = _save(tmp_path, source, "stage_a.model")

    with caplog.at_level(logging.WARNING):
        load_stage_a_base(target, path, device="cpu")

    assert [b.avg_num_neighbors for b in target.interactions] == [14.08, 14.08], (
        "the Stage-B trunk still divides by its own value, so it does not reproduce the "
        "base whose weights it just loaded")
    assert any("avg_num_neighbors" in r.getMessage() for r in caplog.records), (
        "the reset must be logged: a silent correction is as hard to audit as a silent bug")


def test_a_matching_model_is_left_alone_and_says_nothing(tmp_path, caplog):
    source = _Model(14.08)
    target = _Model(14.08)
    path = _save(tmp_path, source, "stage_a.model")
    with caplog.at_level(logging.WARNING):
        load_stage_a_base(target, path, device="cpu")
    assert [b.avg_num_neighbors for b in target.interactions] == [14.08, 14.08]
    assert not any("avg_num_neighbors" in r.getMessage() for r in caplog.records)


def test_a_different_number_of_blocks_is_refused(tmp_path):
    source = _Model(14.08, n_blocks=2)
    target = _Model(112.5, n_blocks=3)
    path = _save(tmp_path, source, "stage_a.model")
    with pytest.raises(RuntimeError, match="avg_num_neighbors cannot be carried"):
        load_stage_a_base(target, path, device="cpu", strict=False)
