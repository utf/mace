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
    assert out["route"] == "B"
