"""Plan section 7, Arm 1: the decision diagnostics run on the toy and the routing logic."""
import numpy as np
import pytest
import torch

from mace.modules.dscc import arm1
from tests.extensions.dscc.test_model import VAC0, _batch, _coupled

torch.set_default_dtype(torch.float64)


def test_frame_diagnostics_and_summary_on_the_toy():
    m = _coupled(regime="B", route_b=False)
    records = arm1.run_diagnostics(m, [_batch([VAC0, VAC0])])
    assert len(records) == 2
    r = records[0]
    assert r.d == pytest.approx(5.6, abs=0.3) and np.isfinite(r.homo) and r.n_eff > 0   # rattled toy
    assert np.isfinite(r.pp_modulation) and 0.3 < r.pp_modulation < 3.0      # exp(+-ln 3) bounds
    assert r.flank_tensor >= 0 and 0.0 <= r.coefficient_saturation <= 1.0
    summary = arm1.summarise(records + records)
    assert summary["n_frames"] == 4 and np.isfinite(summary["n_eff_p50"])


def test_decision_routing():
    full = [{"pp_stop": False, "flank_tensor_over_bulk_spread": 5.0, "level_vs_bond_slope": -1.0,
             "cl_splitting_mean": -0.2, "n_eff_p50": 2.0 + 0.01 * k, "coefficient_saturation": 0.0} for k in range(6)]
    control = [{"pp_stop": False, "flank_tensor_over_bulk_spread": 0.0, "level_vs_bond_slope": -0.5,
                "cl_splitting_mean": 0.0, "n_eff_p50": 2.0 + 0.1 * k, "coefficient_saturation": 0.0} for k in range(6)]
    out = arm1.decide(full, control, [0.05] * 6, [0.06] * 6)
    assert out["route"] == "A" and out["i"] and out["ii"] and out["iii"] and out["iv"]
    full[0]["pp_stop"] = full[1]["pp_stop"] = True          # two seeds stop -> (i) fails
    out = arm1.decide(full, control, [0.05] * 6, [0.06] * 6)
    assert not out["i"] and out["route"] == "D"
    for s in full:
        s["cl_splitting_mean"] = +0.2                        # wrong sign -> (iii) fails too
    out = arm1.decide(full, control, [0.05] * 6, [0.06] * 6)
    assert out["route"] == "C"                               # a sign failure always routes to C


def test_decide_routing_table_exhaustive():
    """Every one of the 16 verdict combinations lands where the Stage-2 table puts it."""
    import itertools
    import numpy as np
    from mace.modules.dscc import arm1

    def summaries(stop, ratio_ok, slope_full, sign, n_eff, sat):
        return [{"pp_stop": stop, "flank_tensor_over_bulk_spread": 10.0 if ratio_ok else 1.0,
                 "level_vs_bond_slope": slope_full, "cl_splitting_mean": sign * 0.1,
                 "n_eff_p50": n, "coefficient_saturation": sat} for n in n_eff]

    for i, ii, iii, iv in itertools.product([True, False], repeat=4):
        full = summaries(stop=not i, ratio_ok=ii, slope_full=1.0, sign=-1.0 if iii else 1.0,
                         n_eff=[2.0, 2.1, 2.0, 2.1, 2.0, 2.1] if iv else [1.0, 3.0, 1.0, 3.0, 1.0, 3.0], sat=0.0)
        control = summaries(stop=False, ratio_ok=False, slope_full=0.5, sign=-1.0,
                            n_eff=[1.0, 3.0, 1.0, 3.0, 1.0, 3.0], sat=0.0)
        out = arm1.decide(full, control, [0.05] * 6, [0.06] * 6)
        assert (out["i"], out["ii"], out["iii"], out["iv"]) == (i, ii, iii, iv), (i, ii, iii, iv, out)
        expected = "C" if not (ii and iii) else ("B" if not iv else ("D" if not i else "A"))
        assert out["route"] == expected, (i, ii, iii, iv, out["route"], expected)
