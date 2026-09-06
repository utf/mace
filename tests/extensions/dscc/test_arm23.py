"""Arm 2+3 selection logic against the registered thresholds."""
from mace.modules.dscc import arm23


def _cfg(name, route="A", coupling="full", force=0.040, shape=0.010, n_eff=2.0, sv=0.02, conv=1.0, f_sr=0.3, ff=0.035, s=1.0, ladder=0.02, regime="B"):
    return arm23.ConfigSummary(name=name, regime=regime, route=route, coupling=coupling,
                               force_rmse=[force + 0.0005 * k for k in range(6)], shape_slope_err=[shape] * 6,
                               n_eff_p50=[n_eff + 0.01 * k for k in range(6)], sv_fraction=[sv] * 6,
                               converged_fraction=[conv] * 6, f_sr=[f_sr] * 6, far_field_4_8=[ff] * 6,
                               s_scale=[s] * 6, ladder_rel_err=ladder)


def test_selection_picks_the_clear_winner_and_reports_ties():
    a_full = _cfg("B_A_full", force=0.040)
    a_lr = _cfg("B_A_lr_only", coupling="lr_only", force=0.046)
    bp_full = _cfg("B_Bp_full", route="Bp", force=0.041, ff=0.030)
    out = arm23.select([a_full, a_lr, bp_full])
    assert out["gates"]["B_Bp_full"]["passed"] and out["gates"]["B_Bp_full"]["far_field_beyond_noise"]
    assert out["selected"] is None and "B_Bp_full" in out["ties"]         # 1 meV/A < tau_phys
    bp_far = _cfg("B_Bp_full", route="Bp", force=0.030, ff=0.030)
    out = arm23.select([a_full, a_lr, bp_far])
    assert out["selected"] == "B_Bp_full" and out["beaten_beyond_margin"] == ["B_A_full", "B_A_lr_only"]


def test_gates_exclude_unstable_or_ablation_configurations():
    bad_sv = _cfg("B_A_full_sv", sv=0.2)
    assert not arm23.gates(bad_sv)["root_rule"]
    ablation = _cfg("A_A_full", regime="A")
    assert not arm23.gates(ablation)["selectable"]
    spread = _cfg("B_A_full_spread"); spread.n_eff_p50 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert not arm23.gates(spread)["localisation_stable"]
    saturated = _cfg("B_Bp_sat", route="Bp", s=1.95)
    assert not arm23.gates(saturated)["s_unsaturated"]
    out = arm23.select([bad_sv, ablation])
    assert out["selected"] is None and "no configuration" in out["reason"]
