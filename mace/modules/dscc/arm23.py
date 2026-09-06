"""Plan section 7 (v4.1 layout, v4.2 selection), Arm 2+3: the joint selection of route,
regime and coupling over six seeds, against the thresholds registered in the tracker
before any result was opened:

  * held-out charged force RMSE and the 159-atom energy-shape prediction error must beat
    the best competitor by more than `max(tau_phys, tau_noise)` (`tau_phys` 3 meV/A on
    forces, 15 meV/A on the 159-atom shape slope; `tau_noise` the six-seed standard error);
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
    """Joint selection: among gate-passing regime-B configurations, the one whose force
    RMSE beats every other by more than `max(tau_phys, tau_noise)` and whose 159-atom shape
    error is not worse beyond the same margin; ties are reported as such."""
    route_a = [c for c in configs if c.route == "A" and c.regime == "B"]
    ff_a = [v for c in route_a for v in c.far_field_4_8] or None
    gate_table = {c.name: gates(c, ff_a) for c in configs}
    candidates = [c for c in configs if gate_table[c.name]["passed"]]
    ranked = sorted(candidates, key=lambda c: float(np.mean(c.force_rmse)))
    decision = {"gates": gate_table, "ranking": [(c.name, float(np.mean(c.force_rmse)), _se(c.force_rmse)) for c in ranked]}
    if not ranked:
        decision["selected"] = None; decision["reason"] = "no configuration passed the gates"
        return decision
    best = ranked[0]
    margin_f = max(TAU_PHYS_FORCE, _se(best.force_rmse))
    beaten = [c.name for c in ranked[1:] if float(np.mean(c.force_rmse)) - float(np.mean(best.force_rmse)) > max(margin_f, _se(c.force_rmse))]
    ties = [c.name for c in ranked[1:] if c.name not in beaten]
    shape_ok = all(float(np.mean(best.shape_slope_err)) - float(np.mean(c.shape_slope_err)) <= max(TAU_PHYS_SHAPE, _se(c.shape_slope_err))
                   for c in ranked[1:])
    decision.update({"selected": best.name if not ties and shape_ok else None,
                     "best_by_force": best.name, "beaten_beyond_margin": beaten, "ties": ties,
                     "shape_not_worse": shape_ok,
                     "reason": "selected" if (not ties and shape_ok) else
                               ("tie within max(tau_phys, tau_noise): " + ", ".join(ties) if ties else "159-atom shape error worse beyond margin")})
    return decision
