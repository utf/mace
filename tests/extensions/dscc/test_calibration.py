"""v4.2 (C7): post-hoc C_Q on interpolation-only frames, with the support flag."""
import numpy as np
import pytest

from mace.modules.dscc import calibration as cal
from mace.modules.dscc.admission import AdmissionConfig


def test_profile_c_q_uses_only_interpolated_frames_and_flags_weak_support():
    rng = np.random.default_rng(0)
    d_oof = np.linspace(5.0, 6.6, 12)
    r_oof = 0.02 * (d_oof - 5.8) + rng.normal(0, 0.12, d_oof.size)     # noisy null, 12 frames
    d = np.array([4.9, 5.2, 5.8, 6.3, 6.9])
    resid = 0.37 + np.array([0.0, 0.01, -0.01, 0.005, 0.0])
    out = cal.profile_c_q(1, d, resid, d_oof, r_oof)
    assert out["n_used"] == 3 and out["window"] == (5.0, 6.6)
    assert out["c_q"] == pytest.approx(0.37 + (0.01 - 0.01 + 0.005) / 3)
    assert out["flag"] == "base support at 159 unverified"           # SE from 12 noisy frames
    clean = cal.profile_c_q(1, d, resid, d_oof, 0.001 * rng.normal(size=12),
                            adm=AdmissionConfig(s_tol=0.05, z=2.0))
    assert clean["flag"] == ""
    with pytest.raises(ValueError):
        cal.profile_c_q(1, d, resid, [], [])
