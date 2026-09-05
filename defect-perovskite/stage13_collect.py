#!/usr/bin/env python3
"""Plan v8 Stage 1.3: one table per arm from the scorers' JSON (`~/runs/<tag>_*.json`).

Reports, per arm and with the seed spread, what section 4 asks for: F4 (the head's
d(delta_sr)/dd on the charged 159-atom frames, against the band [-0.142, -0.063]), depth
from the CBM (b6, aligned), participation (N_eff and the charged/pristine ratio, b13),
R_bound (s3_dilution: the bound-state dilution ratio, R <= 1.3 with the bound fraction), the
79-atom force loss (stage13_forces: force RMSE on the charged 79-atom validation frames) and
the learned static charges Z. Missing files are reported as missing, never skipped.

    python defect-perovskite/stage13_collect.py --tags s13a s13b s13c --runs ~/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

F4_BAND = (-0.142, -0.063)


def load(runs: Path, tag: str, name: str):
    p = runs / f"{tag}_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def spread(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    return "—" if v.size == 0 else f"{v.mean():+.4f} ± {v.std():.4f} (n={v.size})"


def arm_row(runs: Path, tag: str):
    row = {"tag": tag}
    adopt = load(runs, tag, "adopt")
    if adopt:
        f4 = [r["f4"]["slope"] for r in adopt]
        row["F4"] = spread(f4)
        row["F4_in_band"] = f"{sum(F4_BAND[0] <= x <= F4_BAND[1] for x in f4)}/{len(f4)}"
        row["gap"] = spread([r.get("depth", {}).get("pristine_gap") if isinstance(r.get("depth"), dict)
                             else None for r in adopt])
        row["neutral79_force"] = spread([r["neutral_79_window"]["force"] for r in adopt])
    depth = load(runs, tag, "depth")
    if depth:
        arm = depth["arms"].get(tag, [])
        row["depth_from_cbm"] = spread([r["from_cbm"] for r in arm])
        row["depth_shift"] = spread([r["shift"] for r in arm])
    part = load(runs, tag, "participation")
    if part:
        row["N_eff_159"] = spread([r["neff"] for r in part["on"]])
        row["pristine_ratio"] = spread([r["pristine_ratio"] for r in part["on"]])
    dil = load(runs, tag, "dilution")
    if dil:
        row["R_bound"] = spread([r["all"]["median"] for r in dil])
        row["R_bound_pass"] = f"{sum(bool(r['passed']) for r in dil)}/{len(dil)}"
        row["bound_fraction"] = spread([r["bound_fraction"] for r in dil])
    forces = load(runs, tag, "forces")
    if forces:
        rows = list(forces.values())
        row["madelung_range"] = sorted({r["madelung_range"] for r in rows})
        row["F_rmse_79_meV_A"] = spread([r["79"]["force_rmse_meV_A"] for r in rows])
        row["F_rmse_159_meV_A"] = spread([r["159"]["force_rmse_meV_A"] for r in rows])
        row["N_eff_79"] = spread([r["79"]["n_eff_mean"] for r in rows])
        row["Phi_79"] = spread([r["79"]["phi_ff_mean"] for r in rows])
        row["Phi_159"] = spread([r["159"]["phi_ff_mean"] for r in rows])
        zs = np.array([r["z"] for r in rows if r["z"] is not None])
        if zs.size:
            row["Z"] = f"{np.round(zs.mean(0), 3).tolist()} ± {np.round(zs.std(0), 3).tolist()}"
    extras = load(runs, tag, "extras")
    if extras:
        row["c_79"] = spread([r["c_79"] for r in extras["rows"]])
        row["c_159"] = spread([r["c_159"] for r in extras["rows"]])
        row["stops"] = extras.get("gate7")
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--runs", type=Path, default=Path.home() / "runs")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    rows = [arm_row(args.runs, t) for t in args.tags]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k != "tag":
                keys.append(k)
    lines = ["| quantity | " + " | ".join(r["tag"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(r.get(k, "missing")) for r in rows) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
