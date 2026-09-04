"""Standing rules 1 and 2 of the speed cycle's spec.

  1. No per-host constant may live in a default. The c table's size class was
     `n_atoms >= 100`, a CsPbCl3 number; it is now `round(N / N_pristine)`, and
     `N_pristine` comes from the pristine geometry, which the rule lists as an input.
  2. A charged frame's ENERGY enters the loss only for a size class that has a neutral
     null. Forces enter for every charged frame. `w_E` is gone.
"""

import json

import pytest
import torch

from mace.data.two_size import gate_charged_energies_by_null, nulled_sizes_from
from mace.modules.defect_counting import C_SHIFT_SIZE_THRESHOLD, c_shift_classes


class _Frame:
    """The parts of an `AtomicData` the weighting code touches."""

    def __init__(self, n_atoms, counts, energy_weight=1.0, weight=1.0):
        self.positions = torch.zeros(n_atoms, 3)
        self.carrier_counts = torch.tensor([counts], dtype=torch.float64)
        self.energy_weight = torch.tensor(energy_weight, dtype=torch.float64)
        self.forces_weight = torch.tensor(1.0, dtype=torch.float64)
        self.base_energy_weight = torch.tensor(1.0, dtype=torch.float64)
        self.base_forces_weight = torch.tensor(1.0, dtype=torch.float64)
        self.weight = torch.tensor(weight, dtype=torch.float64)


HOLE = [0.0, 0.0, 1.0, 0.0]
NEUTRAL = [0.0, 0.0, 0.0, 0.0]


class TestSizeClass:
    def test_the_pristine_multiple_reproduces_the_old_threshold_on_this_dataset(self):
        sizes = torch.tensor([79, 80, 159, 160])
        counts = torch.zeros(4, 4)
        _, legacy = c_shift_classes(counts, sizes)
        _, ruled = c_shift_classes(counts, sizes, pristine_atoms=80)
        assert legacy.tolist() == [0, 0, 1, 1]
        assert ruled.tolist() == [0, 0, 1, 1]

    def test_a_host_with_different_cells_needs_no_edit(self):
        """The point of the rule. A 128-atom pristine cell puts 127 and 128 in class 0 and
        255/256 in class 1; the 100-atom threshold would have put every one of them in
        class 1 and collapsed the table."""
        sizes = torch.tensor([127, 128, 255, 256])
        counts = torch.zeros(4, 4)
        _, legacy = c_shift_classes(counts, sizes)
        _, ruled = c_shift_classes(counts, sizes, pristine_atoms=128)
        assert legacy.tolist() == [1, 1, 1, 1]
        assert ruled.tolist() == [0, 0, 1, 1]

    def test_a_larger_multiple_clamps_to_the_last_trained_class(self):
        sizes = torch.tensor([80, 240, 640])
        counts = torch.zeros(3, 4)
        _, ruled = c_shift_classes(counts, sizes, pristine_atoms=80)
        assert ruled.tolist() == [0, 1, 1]

    def test_an_unrecorded_pristine_count_falls_back_to_the_legacy_threshold(self):
        """A model pickled before the buffer existed must read its size classes the way it
        was trained, or every constant it learned moves column."""
        sizes = torch.tensor([79, 159])
        counts = torch.zeros(2, 4)
        for unset in (None, torch.zeros((), dtype=torch.long)):
            _, cls = c_shift_classes(counts, sizes, pristine_atoms=unset)
            assert cls.tolist() == [0, 1]
        assert C_SHIFT_SIZE_THRESHOLD == 100

    def test_the_charge_class_is_unchanged(self):
        counts = torch.tensor([[0.0, 0, 1, 0], [0, 0, 0, 0], [1.0, 0, 0, 0]])
        charge, _ = c_shift_classes(counts, torch.tensor([79, 80, 79]), pristine_atoms=80)
        assert charge.tolist() == [0, 1, 2]


class TestNullFile:
    def test_a_bracketed_null_admits_its_size_and_a_resolved_one_does_not(self):
        payload = {"nulls": {
            "159": {"slope": 0.0243, "ci": [-0.0697, 0.1184]},
            "79": {"slope": 0.1147, "ci": [0.0947, 0.1348]}}}
        assert nulled_sizes_from(payload) == [159]

    def test_an_entry_without_an_interval_is_refused(self):
        """A point estimate of zero with unknown width is not a null."""
        assert nulled_sizes_from({"nulls": {"159": {"slope": 0.0}}}) == []

    def test_the_file_may_be_the_bare_mapping(self):
        assert nulled_sizes_from({"159": {"ci": [-0.1, 0.1]}}) == [159]

    def test_non_numeric_keys_are_ignored_not_crashed_on(self):
        payload = {"nulls": {"source": {"ci": [-1, 1]}, "159": {"ci": [-0.1, 0.1]}}}
        assert nulled_sizes_from(payload) == [159]

    def test_it_reads_a_real_json_document(self, tmp_path):
        path = tmp_path / "nulls.json"
        path.write_text(json.dumps({"nulls": {"159": {"ci": [-0.07, 0.12]},
                                              "79": {"ci": [0.09, 0.13]}}}))
        assert nulled_sizes_from(json.loads(path.read_text())) == [159]


class TestGate:
    @staticmethod
    def _set():
        return [_Frame(79, HOLE), _Frame(159, HOLE), _Frame(80, NEUTRAL),
                _Frame(159, NEUTRAL), _Frame(79, HOLE)]

    def test_charged_energies_of_un_nulled_sizes_are_zeroed(self):
        data = self._set()
        kept, dropped, per_size = gate_charged_energies_by_null(data, [159])
        assert (kept, dropped) == (1, 2)
        assert per_size == {79: {"kept": 0, "dropped": 2},
                            159: {"kept": 1, "dropped": 0}}
        assert float(data[0].energy_weight) == 0.0
        assert float(data[4].energy_weight) == 0.0
        assert float(data[1].energy_weight) == 1.0

    def test_neutral_frames_are_untouched(self):
        data = self._set()
        gate_charged_energies_by_null(data, [159])
        for i in (2, 3):
            assert float(data[i].energy_weight) == 1.0
            assert float(data[i].base_energy_weight) == 1.0

    def test_forces_are_untouched_for_every_charged_frame(self):
        """The rule is about energies. A charged 79-atom frame's forces still train the
        head -- that is where the head learns the carrier at all."""
        data = self._set()
        gate_charged_energies_by_null(data, [159])
        for d in data:
            assert float(d.forces_weight) == 1.0
            assert float(d.base_forces_weight) == 1.0

    def test_an_empty_allowance_drops_every_charged_energy(self):
        data = self._set()
        kept, dropped, _ = gate_charged_energies_by_null(data, [])
        assert (kept, dropped) == (0, 3)

    def test_the_w_E_path_is_gone_not_merely_disabled(self):
        """Standing rule 2 says delete the code path. A flag left at 1.0 is a path that can
        be switched back on by a stale launcher."""
        import mace.data.two_size as two_size
        from mace.tools.arg_parser import build_default_arg_parser

        assert not hasattr(two_size, "apply_energy_weights_from_json")
        flags = {a.option_strings[0] for a in build_default_arg_parser()._actions
                 if a.option_strings}
        assert "--defect_energy_weights_json" not in flags
        assert "--defect_null_reference" in flags


def test_the_trainer_refuses_a_charged_energy_share_without_a_null_file():
    """A configuration that asks for charged energies without saying which sizes have a
    null is refused rather than silently admitting all of them."""
    import inspect

    from mace.cli import run_train

    src = inspect.getsource(run_train.run)
    assert "--defect_charged_energy_share puts charged energies in the loss" in src


@pytest.mark.parametrize("size,expected", [(79, 0), (80, 0), (159, 1), (160, 1)])
def test_the_two_rules_agree_about_which_class_is_nulled(size, expected):
    """The formation-energy reference is 'c of the largest size class that has a neutral
    null'. With the null at 159 and the pristine cell at 80, that is class 1 -- the two
    rules have to name the same column or the reference is read off the wrong constant."""
    _, cls = c_shift_classes(torch.zeros(1, 4), torch.tensor([size]), pristine_atoms=80)
    assert int(cls[0]) == expected
    largest_nulled = max(c_shift_classes(
        torch.zeros(1, 4), torch.tensor([n]), pristine_atoms=80)[1].item()
        for n in nulled_sizes_from({"nulls": {"159": {"ci": [-0.07, 0.12]},
                                              "79": {"ci": [0.09, 0.13]}}}))
    assert largest_nulled == 1
