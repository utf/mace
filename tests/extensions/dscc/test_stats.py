"""v5 W0.3: the paired TOST reading."""
import numpy as np
import pytest

from mace.modules.dscc.stats import tost


def test_equivalent_when_both_bounds_inside_tau():
    a = [19.0, 19.1, 18.9, 18.0, 19.1, 18.4]
    b = [18.6, 19.1, 18.8, 19.1, 18.6, 18.3]          # differences of a few tenths, sd ~ 0.5
    r = tost(a, b, tau=1.7)
    assert r.n == 6 and r.equivalent and not r.superior and not r.inferior and r.reading == "equivalent"
    assert r.t_crit == pytest.approx(2.015, abs=1e-3)
    assert -1.7 < r.lower < r.upper < 1.7


def test_superior_when_lower_bound_above_zero():
    a = [23.0, 24.0, 22.5, 23.5, 24.1, 23.2]          # A worse by ~4.5 on every pair
    b = [19.0, 19.1, 18.9, 18.0, 19.1, 18.4]
    r = tost(a, b, tau=1.7)
    assert r.superior and not r.equivalent and r.reading == "superior" and r.lower > 0


def test_inconclusive_with_a_wide_spread():
    a = [25.0, 18.0, 22.0, 30.0, 17.0, 26.0]
    b = [19.0, 21.0, 18.0, 24.0, 20.0, 18.0]
    r = tost(a, b, tau=1.7)
    assert not r.equivalent and not r.superior and not r.inferior and r.reading == "inconclusive"


def test_equivalent_and_superior_together_is_reported():
    a = np.array([19.0, 19.1, 18.9, 18.0, 19.1, 18.4]) + 0.5   # a real 0.5 shift, sd 0
    b = [19.0, 19.1, 18.9, 18.0, 19.1, 18.4]
    r = tost(a, b + np.zeros(6) + 1e-9 * np.arange(6), tau=1.7)
    assert r.equivalent and r.superior and r.reading.startswith("equivalent (superior")


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        tost([1.0], [1.0], tau=1.0)
