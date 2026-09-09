"""Arm 2+3 selection logic against the registered thresholds."""
import pytest

from mace.modules.dscc import arm23


def _cfg(name, route="A", coupling="full", force=0.040, shape=0.010, n_eff=2.0, sv=0.02, conv=1.0, f_sr=0.3, ff=0.035, s=1.0, ladder=0.02, regime="B", near=None):
    near = near if near is not None else 2.0 * force
    return arm23.ConfigSummary(name=name, regime=regime, route=route, coupling=coupling,
                               force_rmse=[force + 0.0005 * k for k in range(6)], shape_slope_err=[shape] * 6,
                               n_eff_p50=[n_eff + 0.01 * k for k in range(6)], sv_fraction=[sv] * 6,
                               converged_fraction=[conv] * 6, f_sr=[f_sr] * 6, far_field_4_8=[ff] * 6,
                               s_scale=[s] * 6, ladder_rel_err=ladder,
                               shells={"0-2": [near + 0.001 * k for k in range(6)], "2-4": [0.8 * near] * 6, "4-6": [ff] * 6, "6-8": [ff] * 6})


def test_selection_picks_the_clear_winner_and_resolves_equivalents_to_the_simplest():
    a_full = _cfg("B_A_full", force=0.040)
    a_lr = _cfg("B_A_lr_only", coupling="lr_only", force=0.046)
    bp_full = _cfg("B_Bp_full", route="Bp", force=0.041, ff=0.030, near=0.070)   # near-field gain: the decisive reading
    out = arm23.select([a_full, a_lr, bp_full])
    g = out["gates"]["B_Bp_full"]
    assert g["passed"] and g["bprime_gains"]["4-8"]["beyond_noise"] and "ladder" not in g and g["ladder_rel_err"] == 0.02
    # 1 meV/A < tau_phys: B' full is equivalent to A full; route A is the simpler and is chosen (v4.3)
    assert out["selected"] == "B_A_full" and "B_Bp_full" in out["equivalent"] and out["beaten_beyond_margin"] == ["B_A_lr_only"]
    bp_far = _cfg("B_Bp_full", route="Bp", force=0.030, ff=0.030)
    out = arm23.select([a_full, a_lr, bp_far])
    assert out["selected"] == "B_Bp_full" and out["beaten_beyond_margin"] == ["B_A_full", "B_A_lr_only"]


def test_selection_uses_seed_medians_and_the_phi0_seed_spread_as_tau_noise():
    """v4.3: `tau_noise` is the Phi = 0 arm's seed spread; comparisons on medians; among
    equivalent configurations the simplest (here Phi = 0 itself) is selected."""
    phi0 = _cfg("B_A_phi0", coupling="phi0", force=0.044)
    phi0.force_rmse = [0.040, 0.050, 0.045, 0.038, 0.052, 0.041]       # median 0.043, spread ~5.7 meV/A
    full = _cfg("B_A_full", force=0.040)                               # best by median (0.04125)
    lr_u = _cfg("B_A_lr_u", coupling="lr_u", force=0.044)
    far = _cfg("B_A_lambda1", coupling="lambda1", force=0.060)         # beaten beyond the margin
    out = arm23.select([phi0, full, lr_u, far])
    assert out["tau_noise_force"]["A"] > 0.005
    assert out["best_by_force"] == "B_A_full"
    assert set(out["equivalent"]) == {"B_A_full", "B_A_lr_u", "B_A_phi0"} and out["beaten_beyond_margin"] == ["B_A_lambda1"]
    assert out["selected"] == "B_A_phi0" and "simplest" in out["reason"]
    # a worse 159-atom shape error beyond its margin breaks the equivalence
    lr_u.shape_slope_err = [0.040] * 6
    out = arm23.select([phi0, full, lr_u, far])
    assert "B_A_lr_u" in out["beaten_beyond_margin"] and out["selected"] == "B_A_phi0"


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


def test_root_rule_reads_the_final_model_and_last_ten_epochs_and_f_sr_is_not_a_gate():
    """C10 (2026-09-08): a run whose early epochs exceeded the ceiling passes when its final
    model and last ten epochs are under it (the transient is recorded); the signed f_SR is
    reported, not gated."""
    c = _cfg("B_A_full", sv=0.25, f_sr=1.7)
    c.sv_fraction_last10 = [0.0] * 6; c.sv_transient_end = [14] * 6; c.f_sr_abs = [0.3] * 6
    g = arm23.gates(c)
    assert g["root_rule"] and not g["root_rule_worst_epoch"] and g["passed"] and "f_sr" not in g
    assert g["f_sr_signed_median"] == 1.7 and g["f_sr_abs_median"] == 0.3
    c.sv_fraction_last10 = [0.0] * 5 + [0.15]
    assert not arm23.gates(c)["root_rule"]


def test_select_v44_treats_phi0_as_the_reference_and_picks_the_minimal_k_lr_candidate():
    """The 2026-09-08 rule (post hoc): K_LR required, Phi = 0 a force reference; the simplest
    candidate equivalent to the best candidate wins, and its cost against Phi = 0 is reported."""
    phi0 = _cfg("B_A_phi0", coupling="phi0")
    phi0.force_rmse = [0.0368, 0.0412, 0.0400, 0.0391, 0.0416, 0.0398]       # median 0.0399, spread 1.7 meV/A
    lr = _cfg("B_A_lr_only", coupling="lr_only", force=0.0405)               # median 0.04175
    lr_u = _cfg("B_A_lr_u", coupling="lr_u", force=0.0400)                   # median 0.04125 (best candidate)
    full = _cfg("B_A_full", force=0.0407)
    far = _cfg("B_A_lambda1", coupling="lambda1", force=0.0480)
    out = arm23.select_v44([phi0, lr, lr_u, full, far])
    assert out["best_candidate_by_force"] == "B_A_lr_u"
    assert out["selected"] == "B_A_lr_only" and set(out["equivalent"]) == {"B_A_lr_u", "B_A_lr_only", "B_A_full"}
    assert out["beaten_beyond_margin"] == ["B_A_lambda1"] and all(n != "B_A_phi0" for n, *_ in out["ranking"])
    assert out["costs_nothing_in_forces"] and abs(out["force_cost_vs_phi0"] - (0.04175 - 0.0399)) < 1e-9
    assert "post hoc" in out["rule"]
    assert arm23.select([phi0, lr, lr_u, full, far])["selected"] == "B_A_phi0"     # v4.3 on the same records


def test_bprime_gates_after_the_c10_addendum():
    """The ladder is reported, not gated; the B' gain is read on the full RMSE and the 0-2 / 2-4
    shells against the same-coupling Route A arm beyond the Phi = 0 seed spread; the 4-8 A
    pooled gain is reported but not decisive."""
    phi0 = _cfg("B_A_phi0", coupling="phi0", force=0.040)
    phi0.force_rmse = [0.0368, 0.0412, 0.0400, 0.0391, 0.0416, 0.0398]
    a_lr = _cfg("B_A_lr_only", coupling="lr_only", force=0.0417, ff=0.035)
    bp_far_only = _cfg("B_Bp_lr_only", route="Bp", coupling="lr_only", force=0.0417, ff=0.020, ladder=0.55)   # far field only
    out = arm23.select([phi0, a_lr, bp_far_only])
    g = out["gates"]["B_Bp_lr_only"]
    assert g["ladder_rel_err"] == 0.55 and g["ladder_within_tol"] is False and g["s_unsaturated"]
    assert g["bprime_gains"]["4-8"]["beyond_noise"] and not g["bprime_gain_beyond_noise"] and not g["passed"]
    bp_near = _cfg("B_Bp_lr_only", route="Bp", coupling="lr_only", force=0.0417, ff=0.035, ladder=0.55, near=0.060)  # near-field gain
    out = arm23.select([phi0, a_lr, bp_near])
    g = out["gates"]["B_Bp_lr_only"]
    assert g["bprime_gains"]["0-2"]["beyond_noise"] and g["bprime_gain_beyond_noise"] and g["passed"]
    assert out["selected"] == "B_A_phi0"                                    # equivalent on forces: the simpler route wins
    assert arm23.select_v44([phi0, a_lr, bp_near])["selected"] == "B_A_lr_only"
    # a B' winner is costed against the Route A Phi = 0 reference (there is no B' Phi = 0 arm)
    bp_win = _cfg("B_Bp_lr_only", route="Bp", coupling="lr_only", force=0.0330, ff=0.030, ladder=0.55, near=0.060)
    out = arm23.select_v44([phi0, a_lr, bp_win])
    assert out["selected"] == "B_Bp_lr_only" and out["beaten_beyond_margin"] == ["B_A_lr_only"]
    assert abs(out["force_cost_vs_phi0"] - (0.03425 - 0.0399)) < 1e-9 and out["costs_nothing_in_forces"]


def test_v5_far_field_gate_reads_all_of_rmse_2_4_and_4_8():
    """W0.2 (v5, registered 2026-09-09): the decisive keys become ("force_rmse", "2-4", "4-8"),
    the 0-2 A shell drops to information, and "each beyond the Phi = 0 seed spread" is read as
    ALL of them. The v4 reading (any of force_rmse / 0-2 / 2-4) stays reproducible."""
    phi0 = _cfg("B_A_phi0", coupling="phi0", force=0.040)
    phi0.force_rmse = [0.0368, 0.0412, 0.0400, 0.0391, 0.0416, 0.0398]
    a_lr = _cfg("B_A_lr_only", coupling="lr_only", force=0.0417, ff=0.035)
    # gains on the full RMSE and 0-2 only: passes v4 (any), fails v5 (2-4 and 4-8 flat)
    bp = _cfg("B_Bp_lr_only", route="Bp", coupling="lr_only", force=0.0417, ff=0.035, ladder=0.55, near=0.060)
    g4 = arm23.gates(bp, a_lr, phi0, spec="v4")
    g5 = arm23.gates(bp, a_lr, phi0, spec="v5")
    assert g4["bprime_decisive_keys"] == ["force_rmse", "0-2", "2-4"]
    assert g5["bprime_decisive_keys"] == ["force_rmse", "2-4", "4-8"]
    assert g4["bprime_gain_beyond_noise"] and not g5["bprime_gain_beyond_noise"]
    assert g5["bprime_gain_beyond_noise_any"] is True         # both quantifiers are reported
    # the pooled 4-8 shell is taken from `shells` when the trainer wrote it (W0.2), not far_field_4_8
    a2 = _cfg("B_A_lr_only", coupling="lr_only", force=0.0417, ff=0.035)
    b2 = _cfg("B_Bp_lr_only", route="Bp", coupling="lr_only", force=0.0300, ff=0.035, ladder=0.55, near=0.060)
    for c, val in ((a2, 0.050), (b2, 0.020)):
        c.shells["4-8"] = [val] * 6
        c.shells["2-4"] = [0.080 if c is a2 else 0.050] * 6
    phi0.shells["4-8"] = [0.050, 0.0505, 0.0495, 0.0502, 0.0498, 0.0501]
    phi0.shells["2-4"] = [0.080, 0.0805, 0.0795, 0.0802, 0.0798, 0.0801]
    g5 = arm23.gates(b2, a2, phi0, spec="v5")
    assert g5["bprime_gains"]["4-8"]["gain"] == pytest.approx(0.030)
    assert g5["bprime_gain_beyond_noise"] and g5["passed"]
    with pytest.raises(ValueError):
        arm23.gates(b2, a2, phi0, spec="v6")
