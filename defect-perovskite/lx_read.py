"""Arm-vs-arm reader for the LES head line (the registered W6 reader `w6_gates.py`, generalised).

    python lx_read.py --a dscc_w6_s%d --b dscc_l1_s%d --label_a "W6 on v2" --label_b L1 --out L1_vs_W6.json

Per seed (0 1 2 3 4 6, folds by the W0.3 map) and per arm, from `held_final.json` (the W6-era
estimator, "final") and `held_final_avg.json` (epoch-averaged, the line's primary): force RMSE
(per component, meV/A), flanking-Pb, energy 79 after ONE C_Q, energy 159 predicted, between-size,
the per-component shell table, the far-field 8-12 A shell (the L2 adoption metric, 8-10 and 10-12
combined by count), saturation and the hop knee fraction. Paired readings (B against A):
G1 TOST at tau (forces); G2a energy-79 non-inferiority at 0.10 meV/atom; G2b |between| <= 10 meV
(cohort median); the named metric: superior if the one-sided 90 % CI upper bound of (B - A) is
below 0 and B is lower on >= 5 of 6 seeds."""
import argparse, json, math, os, statistics, sys
import numpy as np
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.modules.dscc.stats import tost
R3 = math.sqrt(3.0)
SEEDS = (0, 1, 2, 3, 4, 6)
B = "/home/alex/runs/dscc/"


def load(pat, s, which):
    d = B + (pat % s); f = f"{d}/held_final{'_avg' if which == 'avg' else ''}.json"
    if not os.path.exists(f):
        return None
    f = json.load(open(f)); e = f["energies"]
    out = {"force": 1000 * f["force_rmse"] / R3, "pb": 1000 * f["near_rmse"]["2-4:pb_flank"] / R3}
    r79 = np.array([x["resid_uncal"] for x in e if x["n"] == 79]); r159 = np.array([x["resid_uncal"] for x in e if x["n"] == 159])
    cq = r79.mean(); out["cq"] = float(cq)
    out["e79"] = 1000 * float(np.sqrt(((r79 - cq) ** 2).mean())) / 79
    if r159.size:
        out["e159_pred"] = 1000 * float(np.sqrt(((r159 - cq) ** 2).mean())) / 159
        out["between"] = 1000 * float(r159.mean() - cq)
    sh, cnt = f["shell_rmse"], f["shell_counts"]
    out["shells"] = {k: 1000 * v / R3 for k, v in sh.items() if v is not None}
    n1, n2 = cnt.get("8-10", 0) or 0, cnt.get("10-12", 0) or 0
    if n1 + n2 > 0 and sh.get("8-10") is not None and sh.get("10-12") is not None:
        out["far"] = 1000 * math.sqrt((sh["8-10"] ** 2 * n1 + sh["10-12"] ** 2 * n2) / (n1 + n2)) / R3
    sat = f.get("saturation", {})
    out["sat_hop"] = sat.get("hop"); out["sat_on_site"] = sat.get("on_site"); out["on_site_p95"] = sat.get("on_site_shift_eV_p95")
    out["sv_failed"] = bool(f.get("arm_failed_single_valuedness", False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="A"); ap.add_argument("--label_b", default="B")
    ap.add_argument("--tau", type=float, default=1.7); ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = {"arms": {args.label_a: args.a, args.label_b: args.b}, "tau": args.tau}
    for which in ("final", "avg"):
        da = {s: load(args.a, s, which) for s in SEEDS}; db = {s: load(args.b, s, which) for s in SEEDS}
        block = {"per_seed": {args.label_a: da, args.label_b: db}}
        print(f"\n### {which}: per-seed force (meV/A per component) | far 8-12 | e79 | between")
        for s in SEEDS:
            row = []
            for lab, d in ((args.label_a, da), (args.label_b, db)):
                v = d[s]
                row.append("%s %6.2f | %6.2f | %.3f | %+5.1f" % (lab, v["force"], v.get("far", float("nan")), v["e79"], v.get("between", float("nan"))) if v else f"{lab} -")
            print(f"s{s}: " + "   ".join(row))
        pairs = [(da[s], db[s], s) for s in SEEDS if da[s] and db[s]]
        block["n_paired"] = len(pairs)
        if len(pairs) >= 2:
            med = lambda d, k: statistics.median([v[k] for v in d.values() if v and k in v])  # noqa: E731
            block["median"] = {lab: {k: med(d, k) for k in ("force", "pb", "e79", "e159_pred", "between", "far")}
                               for lab, d in ((args.label_a, da), (args.label_b, db))}
            g1 = tost([a["force"] for a, _, _ in pairs], [b["force"] for _, b, _ in pairs], tau=args.tau)
            g2a = tost([a["e79"] for a, _, _ in pairs], [b["e79"] for _, b, _ in pairs], tau=0.10)
            far = tost([a["far"] for a, _, _ in pairs], [b["far"] for _, b, _ in pairs], tau=args.tau)
            wins = sum(1 for a, b, _ in pairs if b["far"] < a["far"])
            between_b = statistics.median([b["between"] for _, b, _ in pairs if "between" in b])
            block["G1"] = {"mean_d": g1.mean_d, "ci90": [g1.lower, g1.upper], "reading": g1.reading, "equivalent": g1.equivalent,
                           "not_inferior": bool(g1.lower > -args.tau)}
            block["G2a"] = {"mean_d": g2a.mean_d, "lower": g2a.lower, "pass": bool(g2a.lower > -0.10)}
            block["G2b"] = {"between_median_meV": between_b, "pass": bool(abs(between_b) <= 10.0)}
            # tost's d = A - B: B lower on the far-field shell <=> d > 0, so "B superior" is the
            # one-sided 90 % lower bound of d above 0, plus B lower on >= 5 of 6 seeds.
            block["named_metric_far_8_12"] = {"mean_d": far.mean_d, "ci90": [far.lower, far.upper],
                                              "b_lower_on_n_of_6": wins,
                                              "superior": bool(far.lower > 0.0 and wins >= 5)}
            print(f"{which}: G1 d={g1.mean_d:+.2f} [{g1.lower:+.2f},{g1.upper:+.2f}] {g1.reading} (d = A - B, + means B better); "
                  f"G2a lower {g2a.lower:+.4f} {'PASS' if block['G2a']['pass'] else 'FAIL'}; G2b {between_b:+.1f} meV "
                  f"{'PASS' if block['G2b']['pass'] else 'FAIL'}; far 8-12: d={far.mean_d:+.2f} [{far.lower:+.2f},{far.upper:+.2f}], "
                  f"B lower on {wins}/6 -> {'SUPERIOR' if block['named_metric_far_8_12']['superior'] else 'not superior'}")
            print("shells (median per component, meV/A):")
            for lab, d in ((args.label_a, da), (args.label_b, db)):
                keys = list(next(v for v in d.values() if v)["shells"].keys())
                print("  %-10s " % lab + "  ".join("%s:%.1f" % (k, statistics.median([v["shells"][k] for v in d.values() if v and k in v["shells"]])) for k in keys))
            print("saturation hop / on-site p95 (median): " + "; ".join(
                "%s %.1e / %.2f eV" % (lab, statistics.median([v["sat_hop"] or 0 for v in d.values() if v]),
                                       statistics.median([v["on_site_p95"] or 0 for v in d.values() if v]))
                for lab, d in ((args.label_a, da), (args.label_b, db))))
        res[which] = block
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)
        print("saved", args.out)


if __name__ == "__main__":
    main()
