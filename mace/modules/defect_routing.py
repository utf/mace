"""Addendum section 8, Stages 2 and 3: the exhaustive routing rule and the inherited contract.

THE ROUTING (addendum, "Stages 2 and 3"). The four Stage-2 readouts keep their plan-v8
section-8 numerical definitions; what changes is that every combination of their four
verdicts now lands in exactly one outcome:

    A  1, 2, 3 and 4 pass                       skip Stage 3, continue to Stage 4
    D  2, 3 and 4 pass, 1 fails                 activate the conditional Stage-3 edge residual
    B  2 and 3 pass, 4 fails (whatever 1 does)  stop architecture growth; investigate
                                                objective identifiability
    C  2 or 3 fails                             repeat the covariance/sign tests and the
                                                single registered bound release; stop if
                                                the repeat fails

v8 had no row for "only the hopping-stop criterion fails" (criterion 1), which is the case
Stage 3 was written for; outcome D is that row. `route` is total on the sixteen verdict
vectors and `test_stage_routing.py` enumerates them.

THE INHERITED CONTRACT (same section). Both stages inherit the gauge-fixed Hamiltonian and
the total-eV, analytically centred Stage-1--4 energy-shape objective, and "no stage may
restore a trainable per-size constant or the old neutral-null admission rule".
`assert_inherited_contract` is that sentence as a function of a run's resolved settings, and
the trainer calls it whenever the v8.1 objective is on -- so a recipe that re-enables the
per-class c table or the null file cannot start.

THE CRITERIA (plan v8 section 8, for reference; the scorers own the numbers):

    1  the pp stop fraction on hub bonds falls to <= 1/6 of seeds
    2  delta b on the flanking Pb is active (|delta b| > 3x its bulk-Pb spread) and its
       ablation removes >= 30 % of the head's d lambda / d d
    3  the flanking-Pb on-site block shows the p_sigma / p_pi splitting with the frozen sign
    4  the seed spread of the participation ratio at least halves against the Stage-B
       cohort, at unchanged or better force fit, with no systematic bound saturation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = ["Criteria", "Outcome", "ACTIONS", "route", "Stage2Router",
           "assert_inherited_contract", "all_verdicts"]


@dataclass(frozen=True)
class Criteria:
    """The four Stage-2 verdicts, in plan-v8 section-8 order."""

    hopping_stop: bool          # 1: the pp stop fraction on hub bonds falls to <= 1/6 seeds
    delta_b_active: bool        # 2: delta b active on the flanking Pb, ablation >= 30 %
    splitting_sign: bool        # 3: p_sigma / p_pi splitting with the frozen sign
    participation_spread: bool  # 4: seed spread of the participation ratio halves, at parity

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "Criteria":
        """From `{"1": bool, ..., "4": bool}` or the field names; every key required."""
        keys = ("hopping_stop", "delta_b_active", "splitting_sign", "participation_spread")
        values = []
        for i, name in enumerate(keys, start=1):
            if name in m:
                v = m[name]
            elif str(i) in m:
                v = m[str(i)]
            elif f"c{i}" in m:
                v = m[f"c{i}"]
            else:
                raise KeyError(f"criterion {i} ({name}) is missing: every one of the four "
                               "verdicts is required, the rule is exhaustive")
            if not isinstance(v, bool):
                raise TypeError(f"criterion {i} ({name}) must be a bool verdict, got {v!r}")
            values.append(v)
        return cls(*values)

    def as_tuple(self) -> Tuple[bool, bool, bool, bool]:
        return (self.hopping_stop, self.delta_b_active, self.splitting_sign,
                self.participation_spread)


class Outcome(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


ACTIONS: Dict[Outcome, str] = {
    Outcome.A: "skip Stage 3 and continue to Stage 4",
    Outcome.D: "activate the conditional Stage-3 edge residual",
    Outcome.B: "stop architecture growth and investigate objective identifiability",
    Outcome.C: ("repeat the covariance/sign tests and the single registered bound release; "
                "stop if the repeat fails"),
}


def route(criteria: Criteria) -> Outcome:
    """The exhaustive rule. Read top to bottom: the most specific failure wins."""
    if not (criteria.delta_b_active and criteria.splitting_sign):
        return Outcome.C
    if not criteria.participation_spread:
        return Outcome.B
    if not criteria.hopping_stop:
        return Outcome.D
    return Outcome.A


def all_verdicts() -> List[Criteria]:
    """The sixteen verdict vectors, for the exhaustiveness test and the driver's table."""
    out = []
    for bits in range(16):
        out.append(Criteria(*(bool((bits >> (3 - k)) & 1) for k in range(4))))
    return out


class Stage2Router:
    """Outcome C's one permitted repeat, kept as state so it cannot be repeated twice.

    Plan v8: "re-run the toys, loosen a_max/b_max by one step, repeat once; if still C,
    report and stop". The addendum keeps that: ONE registered bound release, then stop.
    """

    def __init__(self) -> None:
        self.history: List[Tuple[Criteria, Outcome]] = []

    @property
    def releases_used(self) -> int:
        return sum(1 for _, o in self.history if o is Outcome.C)

    def decide(self, criteria: Criteria) -> Dict[str, Any]:
        outcome = route(criteria)
        self.history.append((criteria, outcome))
        record = {"criteria": criteria.as_tuple(), "outcome": outcome.value,
                  "action": ACTIONS[outcome], "stop": False}
        if outcome is Outcome.C:
            if self.releases_used >= 2:
                record["action"] = ("outcome C again after the single registered bound "
                                    "release: report and STOP")
                record["stop"] = True
            else:
                record["action"] = ("release the registered bound by one step and repeat the "
                                    "covariance/sign tests ONCE")
        elif outcome is Outcome.B:
            record["stop"] = True
        return record


# ------------------------------------------------------------------ the inherited contract


def assert_inherited_contract(settings: Mapping[str, Any], stage: Optional[int] = 2,
                              reference_madelung_range: Optional[str] = None) -> Dict[str, Any]:
    """Refuse a Stage-2/3 run that does not inherit the Stage-1 regime.

    `settings` are the run's resolved values (the trainer's args, or a recipe's environment
    rendered to the same keys):

        spectral_gauge          True     the gauge-fixed Hamiltonian (addendum 3.1)
        energy_shape_weight     > 0      the within-stratum energy-shape objective
        energy_scale_eV         1.0      total-cell eV, never per atom
        total_energy_weight     0        no per-atom total-energy term alongside it
        c_shift_per_class       False    no trainable per-(charge, size) constant
        null_reference          ""       the neutral-null admission rule stays retired
        madelung_range          = the Stage-1 selected arm, when one is given: "no new
                                 scalar range separation until Stage 4" (plan v8 section 5)

    `stage` None means "a run under the v8.1 objective whose stage is not declared" (the
    trainer's check): the same conditions, reported without a stage label.

    Returns the checked values so a manifest can record them."""
    if stage is not None and int(stage) not in (2, 3):
        raise ValueError(f"the inherited contract is Stage 2's and Stage 3's, not stage {stage}")
    problems: List[str] = []
    get = settings.get
    if not bool(get("spectral_gauge", False)):
        problems.append("spectral_gauge is off: Stages 2 and 3 inherit the gauge-fixed "
                        "Hamiltonian (addendum 3.1)")
    if float(get("energy_shape_weight", 0.0) or 0.0) <= 0.0:
        problems.append("energy_shape_weight is 0: the total-eV within-stratum shape "
                        "objective is inherited, not optional")
    if abs(float(get("energy_scale_eV", 1.0)) - 1.0) > 1e-12:
        problems.append(f"energy_scale_eV = {get('energy_scale_eV')}: residuals are "
                        "total-cell eV")
    if float(get("total_energy_weight", 0.0) or 0.0) != 0.0:
        problems.append("total_energy_weight is nonzero: the per-atom total-energy term is "
                        "not part of the inherited objective")
    if bool(get("c_shift_per_class", False)):
        problems.append("c_shift_per_class is on: no stage may restore a trainable per-size "
                        "constant (addendum section 8)")
    if str(get("null_reference", "") or ""):
        problems.append("null_reference is set: the same-size-neutral-null admission rule "
                        "is retired (addendum section 8)")
    if reference_madelung_range is not None:
        got = str(get("madelung_range", ""))
        if got != str(reference_madelung_range):
            problems.append(f"madelung_range = {got!r} but the Stage-1 reference selected "
                            f"{reference_madelung_range!r}: no new scalar range separation "
                            "before Stage 4 (plan v8 section 5)")
    if problems:
        who = (f"Stage {stage} does not inherit the Stage-1 regime" if stage is not None
               else "this run violates the v8.1 objective's contract (addendum section 8)")
        raise ValueError(f"{who}:\n  - " + "\n  - ".join(problems))
    return {k: get(k) for k in ("spectral_gauge", "energy_shape_weight", "energy_scale_eV",
                                 "total_energy_weight", "c_shift_per_class",
                                 "null_reference", "madelung_range")}
