#!/usr/bin/env python3
"""Per-seed table for a stage queue, tolerant of a row that is not a dict.

`summarise_stage3.py` assumed every json row was a dict and crashed on one that was a bare
string -- which is what a harness writes when a cell fails before it has metrics. Crashing
there loses the five seeds that DID finish, so this reports them and names the bad row.
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "s7"
rows, bad = [], []
for f in sorted(glob.glob(f"/home/alex/runs/{tag}_[abc].json")):
    try:
        payload = json.load(open(f))
    except Exception as exc:
        bad.append(f"{f}: unreadable ({exc})")
        continue
    for r in payload:
        if isinstance(r, dict) and "error" not in r:
            rows.append(r)
        else:
            bad.append(f"{f}: {str(r)[:160]}")

for b in bad:
    print(f"  SKIPPED  {b}")
for r in sorted(rows, key=lambda x: x["seed"]):
    print("  seed {seed}  force {f:.5f}  axial_red {a:+.3f}  rmse {m:6.1f}  "
          "N_eff {n:6.2f}  gap {g:.3f}".format(
              seed=r["seed"], f=r.get("force_final", float("nan")),
              a=r["axial_red"], m=r["rmse_all"], n=r["neff"],
              g=r.get("pristine_frontier_gap", float("nan"))))
if rows:
    a = np.array([r["axial_red"] for r in rows])
    m = np.array([r["rmse_all"] for r in rows])
    n = np.array([r["neff"] for r in rows])
    g = np.array([r.get("pristine_frontier_gap", np.nan) for r in rows], dtype=float)
    print(f"\n  n = {len(rows)}   axial_red {a.mean():+.3f} +- {a.std():.3f}   "
          f"rmse {m.mean():.1f} +- {m.std():.1f}   N_eff {n.mean():.2f} +- {n.std():.2f}   "
          f"gap {np.nanmean(g):.3f} +- {np.nanstd(g):.3f}")
