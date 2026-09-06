"""v4.2 (C7): `C_Q` outside the loss.

Training is on forces at every size (no energy term). `C_Q` -- one constant per non-zero
formal charge -- is profiled post hoc on the 159-atom charged frames whose collective
coordinate `d` lies inside the out-of-fold neutral `d`-range (interpolation only, with a
registered margin): a retrospective refinement of the coverage rule, labelled so, with the
bin-level admission result recorded alongside. Reported with its standard error; the
`s0(159) +- SE` slope test decides the flag "base support at 159 unverified".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from mace.modules.dscc.admission import AdmissionConfig, slope_with_se


@dataclass(frozen=True)
class CalibrationConfig:
    margin: float = 0.0          # A inside the out-of-fold neutral d-range (registered)
    size_class: int = 2          # 159-atom cells
    label: str = "retrospective: C_Q profiled post hoc on interpolation-only 159-atom frames (v4.2 C7)"


def profile_c_q(charge: int, d: Sequence[float], residual: Sequence[float],
                neutral_d_oof: Sequence[float], neutral_resid_oof: Sequence[float],
                cfg: CalibrationConfig = CalibrationConfig(), adm: AdmissionConfig = AdmissionConfig()
                ) -> Dict[str, object]:
    """`C_Q` = mean of `E_label - E_base - J*` over the charged frames of `charge` with
    `d` inside `[min(d_oof) + margin, max(d_oof) - margin]`, with its SE; `s0 +- SE` of
    the out-of-fold null and the support flag."""
    d = np.asarray(d, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    d0 = np.asarray(neutral_d_oof, dtype=np.float64)
    r0 = np.asarray(neutral_resid_oof, dtype=np.float64)
    if d0.size == 0:
        raise ValueError("no out-of-fold neutral frames at this size: C_Q cannot be profiled")
    lo, hi = float(d0.min()) + cfg.margin, float(d0.max()) - cfg.margin
    inside = (d >= lo) & (d <= hi)
    n = int(inside.sum())
    c_q = float(r[inside].mean()) if n else float("nan")
    se = float(r[inside].std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    s0, s0_se, _ = slope_with_se(d0, r0)
    support_unverified = not np.isfinite(s0_se) or adm.z * s0_se > adm.s_tol
    return {"charge": int(charge), "config": asdict(cfg), "window": (lo, hi), "n_used": n,
            "n_total": int(d.size), "c_q": c_q, "c_q_se": se,
            "s0": s0, "s0_se": s0_se, "n_neutral_oof": int(d0.size),
            "flag": "base support at 159 unverified" if support_unverified else "",
            "shape_residual_slope": slope_with_se(d[inside], r[inside] - c_q)[0] if n > 2 else float("nan")}
