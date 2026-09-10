"""W3 read: paired TOST over seeds, within the new-base arms and against the re-evaluated
old-base arms of the same name. Forces only (C13: old-base energies are a different quantity).

All numbers are PER COMPONENT in meV/A. The pair is the seed (W0.3: the same seed is the same
fold and the same held-out set). `tau` is the registered `tau_phys` = 1.7 meV/A per component.
The new-base runs are read from `held_final_avg.json` where they have one (W0.4) and from
`held_final.json` otherwise; the old-base arms have only a last-epoch reading, so the CROSS-BASE
comparison uses the last-epoch reading on both sides and says so.

  python defect-perovskite/w3_report.py --runs ~/runs/dscc/dscc_w3_* --old ~/runs/dscc/w3_oldbase_reeval.json --out ~/runs/dscc/w3_report.json
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.modules.dscc.stats import tost

TAU = 1.7                     # meV/A per component (W0.3)
KEYS = ["force_rmse", "2-4", "4-8", "8-10", "10-12", "12+"]
NEAR = ["2-4:pb_flank", "2-4:cl_first", "2-4:other"]
OLD_ARM = {"phi0": "B_A_phi0", "lr_only": "B_A_lr_only", "bp_lr_only": "B_Bp_lr_only"}


def per_component(v):
    return None if v is None else 1000.0 * v / 3 ** 0.5


def read(report):
    """The per-component readings of one held_final-style dict."""
    out = {"force_rmse": per_component(report["force_rmse"])}
    sh = report.get("shell_rmse") or {}
    for k in KEYS[1:]:
        key = "12-99" if k == "12+" else k
        out[k] = per_component(sh.get(k) if sh.get(k) is not None else sh.get(key))
    for k in NEAR:
        out[k] = per_component((report.get("near_rmse") or {}).get(k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    new_avg, new_last = {}, {}
    for run in args.runs:
        run = Path(run)
        # `w3a1` is the A1 form (v5 amendment A1), `w3` the superseded centred one; both
        # parse so a pre-A1 archive can still be read, and `form` keeps them apart.
        m = re.match(r"dscc_(?P<form>w3a1|w3)_(?P<arm>phi0|lr_only|bp_lr_only)_s(?P<seed>\d+)$", run.name)
        if m is None or not (run / "held_final.json").exists():
            continue
        arm, seed = m["arm"], int(m["seed"])
        new_last.setdefault(arm, {})[seed] = read(json.load(open(run / "held_final.json")))
        avg = run / "held_final_avg.json"
        new_avg.setdefault(arm, {})[seed] = read(json.load(open(avg))) if avg.exists() else new_last[arm][seed]

    old_raw = json.load(open(args.old))
    old = {}
    for name, rep in old_raw.items():
        for arm, tag in OLD_ARM.items():
            if name.startswith(f"dscc_arm23_{tag}_s"):
                old.setdefault(arm, {})[int(name.rsplit("_s", 1)[1])] = read(rep)

    report = {"tau_per_component_meV_per_A": TAU, "medians": {}, "within_w3": {}, "cross_base": {}}
    for label, table in (("new_avg", new_avg), ("new_last", new_last), ("old", old)):
        report["medians"][label] = {
            arm: {k: (float(np.median([v[k] for v in seeds.values() if v.get(k) is not None]))
                      if any(v.get(k) is not None for v in seeds.values()) else None)
                  for k in KEYS + NEAR}
            for arm, seeds in table.items()}

    def paired(a_table, b_table, arm_a, arm_b):
        seeds = sorted(set(a_table.get(arm_a, {})) & set(b_table.get(arm_b, {})))
        out = {"seeds": seeds}
        for k in KEYS + NEAR:
            a = [a_table[arm_a][s][k] for s in seeds if a_table[arm_a][s].get(k) is not None
                 and b_table[arm_b][s].get(k) is not None]
            b = [b_table[arm_b][s][k] for s in seeds if a_table[arm_a][s].get(k) is not None
                 and b_table[arm_b][s].get(k) is not None]
            out[k] = tost(a, b, TAU).as_dict() if len(a) >= 2 else None
        return out

    for a, b in (("phi0", "lr_only"), ("phi0", "bp_lr_only"), ("lr_only", "bp_lr_only")):
        if a in new_avg and b in new_avg:
            report["within_w3"][f"{a}_vs_{b}"] = paired(new_avg, new_avg, a, b)
    for arm in OLD_ARM:                       # old (A) against new (B): positive mean favours new
        if arm in old and arm in new_last:
            report["cross_base"][arm] = paired(old, new_last, arm, arm)
    report["cross_base_note"] = ("last-epoch reading on both sides (no pre-v5 run has an epoch "
                                 "average); forces only (C13 changed the energy background)")
    json.dump(report, open(args.out, "w"), indent=1, default=str)
    for arm, seeds in sorted(report["medians"].get("old", {}).items()):
        print(f"old  {arm:11s} force {seeds['force_rmse']:.2f}  2-4 {seeds['2-4']:.2f}  4-8 {seeds['4-8']:.2f}")
    for arm, seeds in sorted(report["medians"].get("new_avg", {}).items()):
        print(f"new  {arm:11s} force {seeds['force_rmse']:.2f}  2-4 {seeds['2-4']:.2f}  4-8 {seeds['4-8']:.2f}")
    for name, res in report["cross_base"].items():
        r = res.get("force_rmse")
        if r:
            print(f"cross-base {name}: mean d {r['mean_d']:+.2f} meV/A ({r['reading']}, n={r['n']})")
    print("saved", args.out)


if __name__ == "__main__":
    main()
