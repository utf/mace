#!/usr/bin/env python3
"""Summarise a Stage-3 queue's json rows. Same table the last rerun printed, as a file.

Inlined in `queue_stage3_rerun.sh` before; a file so the wired rerun and the frozen-P rerun
are read by the SAME code and a difference between them is a difference in the models.
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np


def summarise(prefix: str, label: str, root: str = "/home/alex/runs") -> None:
    rows = []
    for f in sorted(glob.glob(f"{root}/{prefix}_*.json")):
        try:
            rows += [r for r in json.load(open(f)) if "error" not in r]
        except Exception:
            pass
    if not rows:
        print(f"{label}: no rows")
        return
    conv = [r for r in rows if r.get("force_final", 9) < 0.05]
    trips = [r["seed"] for r in rows if r.get("init_passed") is False]
    m = lambda k, g: np.mean([r[k] for r in g if k in r])      # noqa: E731
    sd = lambda k, g: np.std([r[k] for r in g if k in r])      # noqa: E731
    print(f"\n{label}")
    print(f"  TRAINED {len(conv)}/{len(rows)}   init-gate trips: {trips or 'none'}")
    if conv:
        print(f"  axial_red {m('axial_red', conv):+.3f}+-{sd('axial_red', conv):.3f}"
              f"  rmse_all {m('rmse_all', conv):.1f}+-{sd('rmse_all', conv):.1f}"
              f"  N_eff {m('neff', conv):6.2f}"
              f"  bandwidth {m('init_bandwidth', conv):.1f} eV")
        if "pristine_frontier_gap" in conv[0]:
            print(f"  pristine frontier gap {m('pristine_frontier_gap', conv):.3f} eV "
                  f"(target 2.40, gate |delta| <= 0.10)")
    for r in sorted(rows, key=lambda x: x["seed"]):
        stuck = r.get("force_final", 9) >= 0.05
        print(f"    seed {r['seed']}  force {r.get('force_final', float('nan')):.5f}"
              f"  axial_red {r['axial_red']:+.3f}  rmse {r['rmse_all']:6.1f}"
              f"  N_eff {r['neff']:6.2f}"
              f"  gap {r.get('pristine_frontier_gap', float('nan')):.3f}"
              + ("   STUCK" if stuck else "")
              + ("   INIT-TRIP" if r.get("init_passed") is False else ""))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "s3w"
    summarise(tag, "STAGE 3, density response WIRED (counting, s+p)")
