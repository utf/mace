"""Addendum section 8, Stages 2 and 3: the routing is exhaustive and the contract is inherited."""

from __future__ import annotations

import itertools

import pytest

from mace.modules import defect_routing as dr
from mace.modules.defect_routing import Criteria, Outcome, Stage2Router, route


class TestRoutingIsExhaustive:
    def test_every_verdict_vector_lands_in_exactly_one_outcome(self):
        seen = {}
        for c1, c2, c3, c4 in itertools.product([True, False], repeat=4):
            out = route(Criteria(c1, c2, c3, c4))
            assert out in (Outcome.A, Outcome.B, Outcome.C, Outcome.D)
            seen[(c1, c2, c3, c4)] = out
        assert len(seen) == 16
        assert set(seen.values()) == set(Outcome)
        assert {c.as_tuple() for c in dr.all_verdicts()} == set(seen)

    def test_the_table_as_written(self):
        assert route(Criteria(True, True, True, True)) is Outcome.A
        # D: only the hopping-stop criterion fails -- the case v8 had no row for.
        assert route(Criteria(False, True, True, True)) is Outcome.D
        # B: 4 fails while 2 and 3 hold, whatever 1 does.
        assert route(Criteria(True, True, True, False)) is Outcome.B
        assert route(Criteria(False, True, True, False)) is Outcome.B
        # C: 2 or 3 fails, whatever the rest do.
        for c1, c4 in itertools.product([True, False], repeat=2):
            assert route(Criteria(c1, False, True, c4)) is Outcome.C
            assert route(Criteria(c1, True, False, c4)) is Outcome.C
            assert route(Criteria(c1, False, False, c4)) is Outcome.C

    def test_outcome_counts(self):
        counts = {o: 0 for o in Outcome}
        for c in dr.all_verdicts():
            counts[route(c)] += 1
        assert counts == {Outcome.A: 1, Outcome.D: 1, Outcome.B: 2, Outcome.C: 12}

    def test_every_outcome_names_its_action(self):
        for o in Outcome:
            assert dr.ACTIONS[o]
        assert "Stage 4" in dr.ACTIONS[Outcome.A]
        assert "Stage-3" in dr.ACTIONS[Outcome.D]
        assert "identifiability" in dr.ACTIONS[Outcome.B]
        assert "bound release" in dr.ACTIONS[Outcome.C]


class TestCriteriaInput:
    def test_accepts_numbered_named_or_c_keys(self):
        a = Criteria.from_mapping({"1": True, "2": True, "3": False, "4": True})
        b = Criteria.from_mapping({"hopping_stop": True, "delta_b_active": True,
                                   "splitting_sign": False, "participation_spread": True})
        c = Criteria.from_mapping({"c1": True, "c2": True, "c3": False, "c4": True})
        assert a == b == c and route(a) is Outcome.C

    def test_a_missing_or_non_boolean_verdict_is_refused(self):
        with pytest.raises(KeyError, match="criterion 4"):
            Criteria.from_mapping({"1": True, "2": True, "3": True})
        with pytest.raises(TypeError, match="bool verdict"):
            Criteria.from_mapping({"1": True, "2": True, "3": True, "4": 0.7})


class TestTheSingleRepeat:
    def test_c_may_be_repeated_once_then_stops(self):
        router = Stage2Router()
        first = router.decide(Criteria(True, False, True, True))
        assert first["outcome"] == "C" and not first["stop"]
        assert "ONCE" in first["action"]
        second = router.decide(Criteria(True, False, True, True))
        assert second["outcome"] == "C" and second["stop"]
        assert router.releases_used == 2

    def test_a_pass_after_the_release_proceeds(self):
        router = Stage2Router()
        router.decide(Criteria(True, False, True, True))
        after = router.decide(Criteria(True, True, True, True))
        assert after["outcome"] == "A" and not after["stop"]

    def test_b_stops(self):
        assert Stage2Router().decide(Criteria(True, True, True, False))["stop"]


class TestInheritedContract:
    GOOD = dict(spectral_gauge=True, energy_shape_weight=0.5, energy_scale_eV=1.0,
                total_energy_weight=0.0, c_shift_per_class=False, null_reference="",
                madelung_range="full")

    def test_the_stage_1_regime_passes(self):
        checked = dr.assert_inherited_contract(self.GOOD, stage=2,
                                               reference_madelung_range="full")
        assert checked["madelung_range"] == "full"
        dr.assert_inherited_contract(self.GOOD, stage=3)

    @pytest.mark.parametrize("override,reason", [
        (dict(spectral_gauge=False), "gauge-fixed"),
        (dict(energy_shape_weight=0.0), "energy_shape_weight is 0"),
        (dict(energy_scale_eV=1.0 / 79.0), "total-cell eV"),
        (dict(total_energy_weight=1.0), "per-atom total-energy"),
        (dict(c_shift_per_class=True), "per-size constant"),
        (dict(null_reference="null.json"), "neutral-null"),
    ])
    def test_each_departure_is_refused_by_name(self, override, reason):
        with pytest.raises(ValueError, match=reason):
            dr.assert_inherited_contract({**self.GOOD, **override}, stage=2)

    def test_a_new_scalar_range_separation_is_refused_before_stage_4(self):
        with pytest.raises(ValueError, match="no new scalar range separation"):
            dr.assert_inherited_contract({**self.GOOD, "madelung_range": "long_range"},
                                         stage=2, reference_madelung_range="full")

    def test_only_stages_2_and_3_carry_it(self):
        with pytest.raises(ValueError, match="Stage 2's and Stage 3's"):
            dr.assert_inherited_contract(self.GOOD, stage=4)

    def test_all_departures_are_reported_together(self):
        bad = {**self.GOOD, "c_shift_per_class": True, "null_reference": "x.json"}
        with pytest.raises(ValueError) as err:
            dr.assert_inherited_contract(bad, stage=2)
        assert "per-size constant" in str(err.value) and "neutral-null" in str(err.value)
