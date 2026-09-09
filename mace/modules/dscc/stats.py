"""v5 W0.3: paired two one-sided tests (TOST) for arm comparisons.

Pairs are seeds (the same seed = the same fold, the same held-out set and the same `H0`
initialisation). `d_k = a_k - b_k` per pair; a positive mean favours B (B's error is smaller).
Paired t with n - 1 degrees of freedom, one-sided alpha (default 0.05). Readings:
  superior    -- the lower one-sided bound of mean d is above 0 (B beats A);
  inferior    -- the upper one-sided bound is below 0 (A beats B);
  equivalent  -- both one-sided bounds lie inside (-tau, +tau);
  inconclusive -- none of the above (the simpler arm is chosen provisionally by the caller).
Superiority and equivalence can hold together (a real but sub-tau difference): both flags are
returned; the selection rule uses `equivalent`, the report shows `superior`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np
from scipy import stats as sps


@dataclass
class TostResult:
    n: int
    mean_d: float
    se: float
    t_crit: float
    lower: float
    upper: float
    tau: float
    superior: bool
    inferior: bool
    equivalent: bool
    reading: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def tost(a: Sequence[float], b: Sequence[float], tau: float, alpha: float = 0.05) -> TostResult:
    """Paired TOST of B against A at margin `tau` (same units as the inputs)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("tost needs two equal-length 1-d sequences of at least two pairs")
    d = a - b; n = int(d.size)
    mean_d = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n))
    t_crit = float(sps.t.ppf(1.0 - alpha, n - 1))
    lower, upper = mean_d - t_crit * se, mean_d + t_crit * se
    superior = bool(lower > 0.0); inferior = bool(upper < 0.0)
    equivalent = bool(lower > -tau and upper < tau)
    reading = "equivalent" if equivalent else ("superior" if superior else ("inferior" if inferior else "inconclusive"))
    if equivalent and superior:
        reading = "equivalent (superior within tau)"
    if equivalent and inferior:
        reading = "equivalent (inferior within tau)"
    return TostResult(n, mean_d, se, t_crit, float(lower), float(upper), float(tau), superior, inferior, equivalent, reading)
