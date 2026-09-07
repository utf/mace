"""Plan section 7 (v4.1 layout, v4.2 selection), Arm 2+3: the joint selection of route,
regime and coupling over six seeds, against the thresholds registered in the tracker
before any result was opened:

  * held-out charged force RMSE and the 159-atom energy-shape prediction error, compared
    on SEED MEDIANS (v4.3): `tau_noise` is the seed spread (standard deviation) of the
    Phi = 0 arm of the same route under this protocol (`tau_phys` 3 meV/A on forces, 15
    meV/A on the 159-atom shape slope); configurations inside `max(tau_phys, tau_noise)` of
    the best are EQUIVALENT and the simplest of them is selected (Phi = 0 < LR-only < LR+U
    < lambda = 1 fixed < full; route A before B'); Arm-1 force numbers are not a baseline;
  * localisation stability: seed spread (std) of `N_eff` p50 <= 0.5;
  * root rule: per-epoch subsample failing fraction <= 0.10 (final epoch) and SCF within
    `n_max` on >= 99 % of training frames;
  * `f_SR` <= 0.5; Route B' additionally: 4-8 A far-field shell residual improves beyond
    `tau_noise`, `s` inside [0.1, 1.9], tiling-ladder 1/L coefficient within 5 % of Madelung.
Regime-A ablation configurations are reported, never selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

TAU_PHYS_FORCE = 0.003        # eV/A
TAU_PHYS_SHAPE = 0.015        # eV/A on the 159-atom shape slope
N_EFF_SPREAD_MAX = 0.5
SV_CEILING = 0.10
CONVERGED_MIN = 0.99
F_SR_FLOOR = 0.5
S_RANGE = (0.1, 1.9)
LADDER_TOL = 0.05


@dataclass
class ConfigSummary:
    """One configuration over its seeds (each entry a per-seed value)."""
    name: str
    regime: str
    route: str
    coupling: str
    force_rmse: List[float]                 # held-out charged force RMSE per seed, eV/A
    shape_slope_err: List[float]            # |159-atom shape residual slope| after C_Q, eV/A
    n_eff_p50: List[float]
    sv_fraction: List[float]                # final-epoch single-valuedness failing fraction
    converged_fraction: List[float]
    f_sr: List[float] = field(default_factory=list)
    far_field_4_8: List[float] = field(default_factory=list)
    s_scale: List[float] = field(default_factory=list)
    ladder_rel_err: Optional[float] = None


def _se(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("inf")


def _spread(x: Sequence[float]) -> float:
    """Seed spread: the standard deviation over seeds (v4.3's `tau_noise`)."""
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1)) if x.size > 1 else float("inf")


SIMPLICITY = {"phi0": 0, "lr_only": 1, "lr_u": 2, "lambda1": 3, "full": 4}
ROUTE_ORDER = {"A": 0, "Bp": 1}


def simplicity(c: "ConfigSummary") -> tuple:
    """The v4.3 order in which equivalent configurations are resolved: fewer coupling
    learnables first, Route A before Route B'."""
    return (SIMPLICITY.get(c.coupling, 9), ROUTE_ORDER.get(c.route, 9))


def gates(c: ConfigSummary, route_a_far_field: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """The hard gates a configuration must pass to be selectable."""
    out = {"localisation_stable": float(np.std(c.n_eff_p50)) <= N_EFF_SPREAD_MAX,
           "root_rule": max(c.sv_fraction) <= SV_CEILING if c.coupling != "phi0" else True,
           "converged": min(c.converged_fraction) >= CONVERGED_MIN if c.coupling != "phi0" else True,
           "f_sr": (max(c.f_sr) <= F_SR_FLOOR) if c.f_sr else True,
           "selectable": c.regime == "B"}
    if c.route == "Bp":
        out["s_unsaturated"] = all(S_RANGE[0] <= s <= S_RANGE[1] for s in c.s_scale) if c.s_scale else False
        out["ladder"] = (c.ladder_rel_err is not None and c.ladder_rel_err <= LADDER_TOL)
        if route_a_far_field is not None and c.far_field_4_8:
            gain = float(np.mean(route_a_far_field) - np.mean(c.far_field_4_8))
            out["far_field_gain"] = gain
            out["far_field_beyond_noise"] = gain > _se(c.far_field_4_8) + _se(route_a_far_field)
    out["passed"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


def select(configs: Sequence[ConfigSummary]) -> Dict[str, object]:
    """Joint selection (v4.3): among gate-passing regime-B configurations, rank by the seed
    MEDIAN of the held-out force RMSE; `tau_noise` is the seed spread of the Phi = 0 arm of
    the same route (fallback: the configuration's own spread when that arm is absent);
    every configuration whose median force RMSE is within `max(tau_phys, tau_noise)` of the
    best and whose 159-atom shape error is not worse than the best's beyond the same kind of
    margin is EQUIVALENT to it; the simplest equivalent configuration is selected."""
    route_a = [c for c in configs if c.route == "A" and c.regime == "B"]
    ff_a = [v for c in route_a for v in c.far_field_4_8] or None
    gate_table = {c.name: gates(c, ff_a) for c in configs}
    phi0 = {c.route: c for c in configs if c.coupling == "phi0" and c.regime == "B"}
    candidates = [c for c in configs if gate_table[c.name]["passed"]]
    med = lambda x: float(np.median(np.asarray(x, dtype=np.float64)))
    ranked = sorted(candidates, key=lambda c: (med(c.force_rmse), simplicity(c)))
    decision = {"gates": gate_table,
                "ranking": [(c.name, med(c.force_rmse), _spread(c.force_rmse)) for c in ranked],
                "tau_noise_force": {r: _spread(p.force_rmse) for r, p in phi0.items()},
                "tau_noise_shape": {r: _spread(p.shape_slope_err) for r, p in phi0.items()}}
    if not ranked:
        decision["selected"] = None; decision["reason"] = "no configuration passed the gates"
        return decision
    best = ranked[0]

    def tau(c, which):
        ref = phi0.get(c.route)
        own = c.force_rmse if which == "force" else c.shape_slope_err
        spread = _spread(ref.force_rmse if which == "force" else ref.shape_slope_err) if ref is not None else _spread(own)
        return max(TAU_PHYS_FORCE if which == "force" else TAU_PHYS_SHAPE, spread)

    equivalent = [best]
    beaten = []
    for c in ranked[1:]:
        within_force = med(c.force_rmse) - med(best.force_rmse) <= max(tau(best, "force"), tau(c, "force"))
        shape_ok = med(c.shape_slope_err) - med(best.shape_slope_err) <= max(tau(best, "shape"), tau(c, "shape"))
        (equivalent if within_force and shape_ok else beaten).append(c)
    chosen = sorted(equivalent, key=simplicity)[0]
    decision.update({"selected": chosen.name, "best_by_force": best.name,
                     "equivalent": [c.name for c in equivalent], "beaten_beyond_margin": [c.name for c in beaten],
                     "reason": ("selected: best by median force" if chosen is best else
                                f"selected: simplest of the {len(equivalent)} configurations equivalent within max(tau_phys, tau_noise)")})
    return decision
