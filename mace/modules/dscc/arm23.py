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
  * root rule (C10 ruling, 2026-09-08): the v4.5 check's failing fraction <= 0.10 on the
    final model and on each of the last ten epochs (the early-epoch failures are the
    initialised-map transient, recorded per run; ceiling from epoch 1) and SCF within
    `n_max` on >= 99 % of training frames;
  * `f_SR` <= 0.5 was registered and is a criterion defect (the signed fraction is unbounded
    in a small cell); it is no longer a gate -- the absolute share `f_sr_abs` is reported;
    Route B' additionally (C10 addendum, 2026-09-08): `s` inside [0.1, 1.9]; its GAIN over the
    same-coupling Route A arm read on the full per-atom RMSE and on the 0-2, 2-4 and 4-8 A
    shells, each beyond the Phi = 0 seed spread of that quantity, with the 4-8 A pooled reading
    reported but not decisive (the decisive readings: the full RMSE and the 0-2 / 2-4 shells);
    the tiling-ladder `1/L` gate is re-read as a MODEL PROPERTY, reported, not a gate -- it
    cannot pass on cells with L_min < ~3 R_c (compensation cloud 8-11 A), so B' is judged on
    forces and no cross-size energy claim includes E_SF until a ladder with L >> R_c exists.
Regime-A ablation configurations are reported, never selected.

`select` is the v4.3 rule as registered before the campaign (Phi = 0 a candidate, simplest
equivalent wins). `select_v44` is the v4.4 amendment's rule (`DSCC_PLAN_V4_4_AMENDMENT.md`, received
2026-09-08 after the campaign: `K_LR` required physics with a fixed coefficient, not subject
to the equivalence rule; the minimal production candidate is LR-only; the Phi = 0 arms are
force REFERENCES, not candidates), applied post hoc as the C10 ruling anticipated.
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
    sv_fraction: List[float]                # worst-epoch (>= 1) failing fraction of the single-valuedness check (v4.5)
    converged_fraction: List[float]
    f_sr: List[float] = field(default_factory=list)              # signed fraction: reported, defective, not a gate
    sv_fraction_last10: List[float] = field(default_factory=list)   # max over the final model and the last ten epochs (C10)
    sv_transient_end: List[int] = field(default_factory=list)       # last over-ceiling epoch per seed (0: none)
    f_sr_abs: List[float] = field(default_factory=list)          # absolute share, reported only
    far_field_4_8: List[float] = field(default_factory=list)      # the 4-6 and 6-8 A shells pooled, per seed
    s_scale: List[float] = field(default_factory=list)
    ladder_rel_err: Optional[float] = None                        # reported (model property), not a gate (C10 addendum)
    shells: Dict[str, List[float]] = field(default_factory=dict)  # per-seed shell RMS by "a-b" key (held-out set)


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


def gates(c: ConfigSummary, route_a: Optional["ConfigSummary"] = None,
          phi0: Optional["ConfigSummary"] = None) -> Dict[str, object]:
    """The hard gates a configuration must pass to be selectable. For a Route B' configuration
    `route_a` is the SAME-coupling Route A configuration and `phi0` the Route A Phi = 0 arm
    (its seed spread of each quantity is that quantity's `tau_noise`, v4.3)."""
    sv_read = c.sv_fraction_last10 if c.sv_fraction_last10 else c.sv_fraction
    out = {"localisation_stable": float(np.std(c.n_eff_p50)) <= N_EFF_SPREAD_MAX,
           "root_rule": max(sv_read) <= SV_CEILING if c.coupling != "phi0" else True,
           "root_rule_worst_epoch": max(c.sv_fraction) <= SV_CEILING if c.coupling != "phi0" else True,   # information: the pre-C10 reading
           "converged": min(c.converged_fraction) >= CONVERGED_MIN if c.coupling != "phi0" else True,
           "selectable": c.regime == "B"}
    if c.f_sr:                                                   # information only (C10: the signed gate is defective)
        out["f_sr_signed_median"] = float(np.median(c.f_sr))
    if c.f_sr_abs:
        out["f_sr_abs_median"] = float(np.median(c.f_sr_abs))
    if c.route == "Bp":
        out["s_unsaturated"] = all(S_RANGE[0] <= s <= S_RANGE[1] for s in c.s_scale) if c.s_scale else False
        # C10 addendum: the ladder is a model property, reported; not a gate.
        out["ladder_rel_err"] = c.ladder_rel_err
        out["ladder_within_tol"] = (c.ladder_rel_err is not None and c.ladder_rel_err <= LADDER_TOL) if c.ladder_rel_err is not None else None
        if route_a is not None:
            med = lambda x: float(np.median(np.asarray(x, dtype=np.float64)))
            readings = {"force_rmse": (route_a.force_rmse, c.force_rmse, phi0.force_rmse if phi0 else route_a.force_rmse)}
            for key in ("0-2", "2-4", "4-6", "6-8"):
                if key in route_a.shells and key in c.shells:
                    readings[key] = (route_a.shells[key], c.shells[key], (phi0.shells.get(key) if phi0 else None) or route_a.shells[key])
            if route_a.far_field_4_8 and c.far_field_4_8:
                readings["4-8"] = (route_a.far_field_4_8, c.far_field_4_8, (phi0.far_field_4_8 if phi0 else None) or route_a.far_field_4_8)
            gains = {}
            for key, (a, b, ref) in readings.items():
                gain = med(a) - med(b); noise = _spread(ref)
                gains[key] = {"gain": gain, "tau_noise": noise, "beyond_noise": gain > noise}
            out["bprime_gains"] = gains
            decisive = [k for k in ("force_rmse", "0-2", "2-4") if k in gains]
            out["bprime_gain_beyond_noise"] = any(gains[k]["beyond_noise"] for k in decisive) if decisive else False
    info_keys = {"root_rule_worst_epoch", "ladder_within_tol"}          # reported, not gated
    out["passed"] = all(v for k, v in out.items() if isinstance(v, bool) and k not in info_keys)
    return out


def _gate_table(configs: Sequence[ConfigSummary]) -> Dict[str, Dict[str, object]]:
    """Gates per configuration; a Route B' configuration is read against its same-coupling
    Route A counterpart's far field, with the Route A Phi = 0 seed spread as the noise."""
    route_a = {c.coupling: c for c in configs if c.route == "A" and c.regime == "B"}
    phi0 = route_a.get("phi0")
    table = {}
    for c in configs:
        counterpart = route_a.get(c.coupling) if c.route == "Bp" else None
        table[c.name] = gates(c, counterpart, phi0)
    return table


def select(configs: Sequence[ConfigSummary]) -> Dict[str, object]:
    """Joint selection (v4.3): among gate-passing regime-B configurations, rank by the seed
    MEDIAN of the held-out force RMSE; `tau_noise` is the seed spread of the Phi = 0 arm of
    the same route (fallback: the configuration's own spread when that arm is absent);
    every configuration whose median force RMSE is within `max(tau_phys, tau_noise)` of the
    best and whose 159-atom shape error is not worse than the best's beyond the same kind of
    margin is EQUIVALENT to it; the simplest equivalent configuration is selected."""
    gate_table = _gate_table(configs)
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


def select_v44(configs: Sequence[ConfigSummary]) -> Dict[str, object]:
    """The v4.4 rule (amendment received 2026-09-08 after the campaign -- post hoc):
    `K_LR` is required physics with no learnable coefficient and is not subject to the
    equivalence rule, so the Phi = 0 arms are force REFERENCES and the candidates are the
    gate-passing regime-B configurations that carry `K_LR` (LR-only, LR+U, lambda = 1, full).
    Among the candidates: rank by the seed median of the held-out force RMSE; every candidate
    within `max(tau_phys, tau_noise)` of the best candidate on forces and not worse beyond
    the same kind of margin on the 159-atom shape error is equivalent to it; the simplest
    equivalent candidate (LR-only < LR+U < lambda = 1 < full) is selected. The selected
    candidate's force cost against the Phi = 0 reference of its route is reported (inside
    the margin: "costs nothing in forces")."""
    gate_table = _gate_table(configs)
    phi0 = {c.route: c for c in configs if c.coupling == "phi0" and c.regime == "B"}
    med = lambda x: float(np.median(np.asarray(x, dtype=np.float64)))
    candidates = [c for c in configs if gate_table[c.name]["passed"] and c.coupling != "phi0"]
    ranked = sorted(candidates, key=lambda c: (med(c.force_rmse), simplicity(c)))
    decision = {"rule": "v4.4 amendment (received 2026-09-08 after the campaign; applied post hoc)",
                "gates": gate_table,
                "ranking": [(c.name, med(c.force_rmse), _spread(c.force_rmse)) for c in ranked],
                "references": {r: (p.name, med(p.force_rmse), _spread(p.force_rmse)) for r, p in phi0.items()},
                "tau_noise_force": {r: _spread(p.force_rmse) for r, p in phi0.items()},
                "tau_noise_shape": {r: _spread(p.shape_slope_err) for r, p in phi0.items()}}
    if not ranked:
        decision["selected"] = None; decision["reason"] = "no K_LR-carrying configuration passed the gates"
        return decision
    best = ranked[0]

    def tau(c, which):
        ref = phi0.get(c.route)
        own = c.force_rmse if which == "force" else c.shape_slope_err
        spread = _spread(ref.force_rmse if which == "force" else ref.shape_slope_err) if ref is not None else _spread(own)
        return max(TAU_PHYS_FORCE if which == "force" else TAU_PHYS_SHAPE, spread)

    equivalent, beaten = [best], []
    for c in ranked[1:]:
        within_force = med(c.force_rmse) - med(best.force_rmse) <= max(tau(best, "force"), tau(c, "force"))
        shape_ok = med(c.shape_slope_err) - med(best.shape_slope_err) <= max(tau(best, "shape"), tau(c, "shape"))
        (equivalent if within_force and shape_ok else beaten).append(c)
    chosen = sorted(equivalent, key=simplicity)[0]
    ref = phi0.get(chosen.route) or phi0.get("A")        # v4.4: the Route A Phi = 0 arms are the force references (a B' Phi = 0 arm is not run)
    cost = (med(chosen.force_rmse) - med(ref.force_rmse)) if ref is not None else None
    decision.update({"selected": chosen.name, "best_candidate_by_force": best.name,
                     "equivalent": [c.name for c in equivalent], "beaten_beyond_margin": [c.name for c in beaten],
                     "force_cost_vs_phi0": cost,
                     "costs_nothing_in_forces": (cost is not None and cost <= tau(chosen, "force")),
                     "reason": ("selected: best candidate by median force" if chosen is best else
                                f"selected: simplest of the {len(equivalent)} candidates equivalent within max(tau_phys, tau_noise)")})
    return decision
