"""Adoption verdicts of the LES head line from `lx_read.py` JSON (the `avg` block is primary).
  --stage L1: completeness only (L1 is the baseline; its gates are recorded, none blocks L2).
  --stage L2/L3: adopt iff G1 not inferior (paired TOST lower bound > -tau) AND the named
  far-field 8-12 A metric is superior (registered 2026-09-15). Writes {stage}_GATES.json."""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True); ap.add_argument("--read", required=True)
ap.add_argument("--c5", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
r = json.load(open(a.read)); res = {"stage": a.stage, "read": a.read}
# the epoch-averaged block is primary only when all six seeds pair on it (the W6-on-v2 reference has
# `held_final_avg` for two seeds only); otherwise the W6-era `held_final` block carries the reading
blk = r.get("avg") if r.get("avg", {}).get("n_paired", 0) == 6 else r.get("final")
res["estimator"] = "avg" if blk is r.get("avg") else "final"
res["n_paired"] = blk.get("n_paired", 0) if blk else 0
lab_b = [k for k in r["arms"]][1]
per = blk["per_seed"][lab_b] if blk else {}
res["complete"] = res["n_paired"] == 6
res["sv_failed_seeds"] = [s for s, v in per.items() if v and v.get("sv_failed")]
if a.c5 and os.path.exists(a.c5):
    c5 = json.load(open(a.c5))
    res["C5"] = {k: {"pass_fraction": v["precondition"]["pass_fraction"], "passed": v["precondition"]["passed"],
                     "separation_p50_meV": 1000 * v["precondition"]["separation_p50"], "n_eff_p50": v["precondition"]["n_eff_p50"]}
                 for k, v in c5.items()}
    res["C5_passed_n"] = sum(1 for v in res["C5"].values() if v["passed"])
if blk:
    res["G1"], res["G2a"], res["G2b"], res["named_metric"] = blk["G1"], blk["G2a"], blk["G2b"], blk["named_metric_far_8_12"]
    res["median"] = blk.get("median")
if a.stage == "L1":
    res["verdict"] = "L1 READ; L2 LAUNCHES" if res["complete"] else "INCOMPLETE"
else:
    ok = bool(blk) and res["complete"] and res["G1"]["not_inferior"] and res["named_metric"]["superior"]
    res["verdict"] = f"{a.stage} ADOPTED" if ok else f"{a.stage} NOT ADOPTED; previous stands"
json.dump(res, open(a.out, "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("stage", "estimator", "n_paired", "complete", "verdict")}))
print("VERDICT:", res["verdict"])
