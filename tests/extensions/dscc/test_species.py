"""Plan section 1 / decision D2: species table, reference count, spin split, state."""
import pytest

from mace.modules.dscc import species as sp


def test_neutral_count_and_split_on_the_training_compositions():
    pristine = [17] * 48 + [55] * 16 + [82] * 16
    vacancy = [17] * 47 + [55] * 16 + [82] * 16
    assert sp.neutral_count(pristine) == 416 and sp.reference_split(416) == (208, 208)
    assert sp.neutral_count(vacancy) == 409 and sp.reference_split(409) == (205, 204)
    assert sp.neutral_count(vacancy * 2 + [17]) == 825


def test_state_from_the_dataset_counters():
    ref = sp.state_from_carrier_counts([0, 0, 0, 0])
    assert ref.is_reference and ref == sp.S_REF and ref.Q == 0
    plus = sp.state_from_carrier_counts([0, 0, 1, 0])          # one hole, majority channel
    assert (plus.Q, plus.dN_up, plus.dN_dn) == (1, -1, 0) and not plus.is_reference
    assert plus.counts(409) == (204, 204)
    minus = sp.state_from_carrier_counts([1, 0, 0, 0])
    assert (minus.Q, minus.dN_up, minus.dN_dn) == (-1, 1, 0)
    assert plus.as_dict()["occupation_policy"] == "count_fill"


def test_inconsistent_or_unknown_states_are_refused():
    with pytest.raises(ValueError):
        sp.State(Q=0, dN_up=1, dN_dn=0)
    with pytest.raises(ValueError):
        sp.State(Q=0, dN_up=0, dN_dn=0, occupation_policy="lowest_root")
    with pytest.raises(KeyError):
        sp.neutral_count([17, 8])
