"""Carrier-count algebra and the m_F guard (addendum sections 3.3-3.4, gates in 11.1)."""

import pytest

from mace.modules.defect_carriers import (
    MAX_FRONTIER_MULTIPLICITY,
    UnsupportedStateError,
    assert_charge_consistency,
    assert_supported_multiplicity,
    carrier_multiplicity,
    class_counts,
    core_charge,
    counts_from_excess,
    require_neutral_reference,
    signed_excess,
)


class TestSignedExcess:
    def test_electron_excess(self):
        assert signed_excess((10, 10), (11, 10)) == (1, 0)
        n_e, n_h, q_f = counts_from_excess((1, 0))
        assert n_e == (1, 0) and n_h == (0, 0) and q_f == -1

    def test_hole_excess(self):
        assert signed_excess((10, 10), (9, 10)) == (-1, 0)
        n_e, n_h, q_f = counts_from_excess((-1, 0))
        assert n_e == (0, 0) and n_h == (1, 0) and q_f == 1

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="valence ranks"):
            signed_excess((10, 10), (10,))


class TestCrossing:
    """Section 11.1: counts stay valid across an electron/hole crossing."""

    @pytest.mark.parametrize("n_maj", [8, 9, 10, 11, 12])
    def test_no_negative_counts_across_the_rank(self, n_maj):
        d = signed_excess((10, 10), (n_maj, 10))
        n_e, n_h, q_f = counts_from_excess(d)
        assert all(x >= 0 for x in n_e), n_e
        assert all(x >= 0 for x in n_h), n_h
        assert q_f == -(sum(d))

    def test_a_sweep_through_the_rank_is_continuous_in_q_f(self):
        q = [counts_from_excess(signed_excess((10, 10), (n, 10)))[2] for n in range(8, 13)]
        assert q == [2, 1, 0, -1, -2]

    def test_electron_and_hole_never_coexist_in_one_channel(self):
        for n in range(6, 15):
            n_e, n_h, _ = counts_from_excess(signed_excess((10, 10), (n, 10)))
            assert all(e == 0 or h == 0 for e, h in zip(n_e, n_h))


class TestMultiplicity:
    def test_counts_every_carrier_present(self):
        # One electron in one channel, one hole in the other: q_F = 0 but m_F = 2.
        n_e, n_h, q_f = counts_from_excess((1, -1))
        assert q_f == 0
        assert carrier_multiplicity(n_e, n_h) == 2

    def test_guard_uses_absolute_counts_not_the_charge_difference(self):
        """A state can differ from the reference by zero net charge and still be refused."""
        n_e, n_h, q_f = counts_from_excess((1, -1))
        assert q_f == 0
        with pytest.raises(UnsupportedStateError, match="m_F = 2"):
            assert_supported_multiplicity(carrier_multiplicity(n_e, n_h))

    def test_single_carrier_passes(self):
        n_e, n_h, _ = counts_from_excess((1, 0))
        assert_supported_multiplicity(carrier_multiplicity(n_e, n_h))

    def test_carrier_free_state_passes(self):
        n_e, n_h, _ = counts_from_excess((0, 0))
        assert carrier_multiplicity(n_e, n_h) == 0
        assert_supported_multiplicity(0)

    def test_limit_is_one_until_the_common_functional_exists(self):
        assert MAX_FRONTIER_MULTIPLICITY == 1

    def test_a_raised_limit_is_explicit(self):
        assert_supported_multiplicity(2, limit=2)


class TestCoreCharge:
    def test_benchmark_v_cl_neutral(self):
        """V_Cl0: Q_core = +1, one majority electron, q_F = -1."""
        counts = class_counts((10, 10), (11, 10), q_formal_ref=0)
        assert counts["n_e"] == (1, 0)
        assert counts["q_f"] == -1
        assert counts["q_core"] == 1
        assert counts["m_f"] == 1

    def test_charge_consistency_holds_and_is_asserted(self):
        counts = class_counts((10, 10), (11, 10))
        assert_charge_consistency(q_formal=counts["q_core"] + counts["q_f"],
                                  q_core=counts["q_core"], q_f=counts["q_f"])

    def test_inconsistent_charge_is_caught(self):
        with pytest.raises(AssertionError, match="disagree"):
            assert_charge_consistency(q_formal=0, q_core=1, q_f=0)

    def test_nonzero_formal_reference_is_algebraically_defined(self):
        """Retained as a synthetic identity test only."""
        assert core_charge(q_formal_ref=2, q_f_ref=-1) == 3

    def test_but_is_refused_on_the_production_path(self):
        require_neutral_reference(0)
        with pytest.raises(UnsupportedStateError, match="calibrated to a neutral reference"):
            require_neutral_reference(2)


class TestSyntheticMultiCarrierClass:
    """Section 3 of v8: a toy with m explicitly occupied frontier levels."""

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_n_e_equals_m_and_q_core_equals_m(self, m):
        counts = class_counts((10, 10), (10 + m, 10), q_formal_ref=0)
        assert counts["n_e"] == (m, 0)
        assert counts["q_core"] == m
        assert counts["q_f"] == -m
        assert counts["m_f"] == m

    @pytest.mark.parametrize("m", [2, 3])
    def test_multi_carrier_toys_are_refused_by_the_guard(self, m):
        counts = class_counts((10, 10), (10 + m, 10))
        with pytest.raises(UnsupportedStateError):
            assert_supported_multiplicity(counts["m_f"])
