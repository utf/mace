"""Plan P Phase 2 read (registered 2026-09-17 04:50): paired TOST of arm B (E_cP + E_PP,
polarisation coupling) against arm A (W6 Mulliken control) on the held-out charged-frame forces,
total and the 4-12 A shell, per component, tau = 1.7 meV/A; B(M) against B for the record.
Gate 2 per run: C_Q profiled at 79 atoms transfers to 159 within 10 meV, 79-atom energy after C_Q
<= 0.6 meV/atom. Reads held_final_avg.json where present, held_final.json otherwise (the pair is
the seed = the fold). Also reports s/epoch per arm from train.log. Writes JSON and P2_REPORT.md.

  python defect-perovskite/p2_report.py --runs ~/runs/dscc/dscc_p2*_s* --out ~/runs/dscc/p2/P2_GATES.json --md defect-perovskite/P2_REPORT.md
"""
import argparse, json, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.modules.dscc.stats import tost

TAU = 1.7
SHELLS = ["2-4", "4-8", "8-10", "10-12", "12-99"]


def pc(v):
    return None if v is None else 1000.0 * v / 3 ** 0.5


def read(rep):
    sh, cnt = rep.get("shell_rmse") or {}, rep.get("shell_counts") or {}
    out = {"force_rmse": pc(rep["force_rmse"])}
    for k in SHELLS:
        out[k] = pc(sh.get(k))
    # 4-12 A shell: count-weighted rms over 4-8, 8-10, 10-12
    num = sum((sh[k] ** 2) * cnt[k] for k in ("4-8", "8-10", "10-12") if sh.get(k) is not None and cnt.get(k))
    den = sum(cnt[k] for k in ("4-8", "8-10", "10-12") if sh.get(k) is not None and cnt.get(k))
    out["4-12"] = pc(float(np.sqrt(num / den))) if den else None
    e = rep.get("energy_after_cq") or {}
    e79, e159 = e.get("79") or e.get(79) or {}, e.get("159") or e.get(159) or {}
    out["cq_79_eV"] = e79.get("c_q"); out["cq_159_eV"] = e159.get("c_q")
    out["e79_meV_atom"] = e79.get("meV_per_atom"); out["e159_meV_atom"] = e159.get("meV_per_atom")
    out["cq_transfer_meV"] = None if (out["cq_79_eV"] is None or out["cq_159_eV"] is None) else 1000.0 * abs(out["cq_159_eV"] - out["cq_79_eV"])
    out["gate2"] = (out["cq_transfer_meV"] is not None and out["cq_transfer_meV"] <= 10.0 and out["e79_meV_atom"] is not None and out["e79_meV_atom"] <= 0.6)
    out["sv_failed"] = bool(rep.get("arm_failed_single_valuedness", False))
    return out


def s_per_epoch(run):
    log = run / "train.log"
    if not log.exists():
        return None
    ts = [datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") for m in re.finditer(r"^(\S+ \S+),\d+ epoch \d+ ", log.read_text(), re.M)]
    return None if len(ts) < 3 else float(np.median(np.diff([t.timestamp() for t in ts])))


ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--md", default=None)
args = ap.parse_args()
arms = {}; cost = {}
for run in map(Path, args.runs):
    m = re.match(r"dscc_p2(?P<arm>a|b|bm|bs|b0|bsm|b0m)_s(?P<seed>\d+)$", run.name)
    if m is None or not (run / "held_final.json").exists():
        continue
    arm, seed = m["arm"], int(m["seed"])
    f = run / "held_final_avg.json"
    arms.setdefault(arm, {})[seed] = read(json.load(open(f if f.exists() else run / "held_final.json")))
    arms[arm][seed]["estimator"] = "avg" if f.exists() else "final"
    c = s_per_epoch(run)
    if c: cost.setdefault(arm, []).append(c)
KEYS = ["force_rmse", "4-12"] + SHELLS
rep = {"tau": TAU, "n_runs": {a: len(s) for a, s in arms.items()}, "per_seed": arms, "medians": {}, "paired": {}, "gate2": {}, "s_per_epoch_median": {a: float(np.median(v)) for a, v in cost.items()}}
for a, seeds in arms.items():
    rep["medians"][a] = {k: float(np.median([v[k] for v in seeds.values() if v.get(k) is not None])) if any(v.get(k) is not None for v in seeds.values()) else None for k in KEYS + ["e79_meV_atom", "cq_transfer_meV"]}
    rep["gate2"][a] = {"pass_seeds": [s for s, v in seeds.items() if v["gate2"]], "fail_seeds": [s for s, v in seeds.items() if not v["gate2"]]}


def paired(x, y):
    seeds = sorted(set(arms.get(x, {})) & set(arms.get(y, {})))
    out = {"seeds": seeds}
    for k in KEYS:
        a = [arms[x][s][k] for s in seeds if arms[x][s].get(k) is not None and arms[y][s].get(k) is not None]
        b = [arms[y][s][k] for s in seeds if arms[x][s].get(k) is not None and arms[y][s].get(k) is not None]
        out[k] = tost(a, b, TAU).as_dict() if len(a) >= 2 else None   # positive mean_d favours y
    return out


for x, y in (("a", "b"), ("b", "bm"), ("a", "bs"), ("a", "b0"), ("bs", "b0"), ("bs", "bsm"), ("b0", "b0m")):
    if x in arms and y in arms:
        rep["paired"][f"{x}_vs_{y}"] = paired(x, y)


def adopted_over_a(arm):
    """Plan P §5 (2b): adopted over A on superiority or equivalence in total OR 4-12 A."""
    g = rep["paired"].get(f"a_vs_{arm}", {})
    r_tot, r_sh = g.get("force_rmse"), g.get("4-12")
    if not (r_tot and r_sh):
        return None
    ok = any(r["reading"] in ("superior", "equivalent") for r in (r_tot, r_sh))
    inferior = all(r["reading"] == "inferior" for r in (r_tot, r_sh))
    return {"total": r_tot["reading"], "4-12": r_sh["reading"], "adopted": bool(ok), "inferior_both": bool(inferior), "n_pairs": r_tot["n"]}


rep["gate1"] = {arm: adopted_over_a(arm) for arm in ("b", "bs", "b0") if arm in arms}
decision = []
if "bs" in arms or "b0" in arms:
    gs, g0 = rep["gate1"].get("bs"), rep["gate1"].get("b0")
    if gs and g0:
        if gs["inferior_both"] and g0["inferior_both"]:
            decision.append("both B_s and B_0 inferior to A: registered fallback r_c = 10 A (ramp 8-10), B_s only, six seeds, once")
        else:
            pair = rep["paired"].get("bs_vs_b0", {})
            within = pair and all(pair.get(k) and pair[k]["reading"] == "equivalent" for k in ("force_rmse", "4-12"))
            chosen = "B_0" if (g0["adopted"] and within) else ("B_s" if gs["adopted"] else ("B_0" if g0["adopted"] else None))
            decision.append(f"B_s adopted over A: {gs['adopted']} ({gs['total']}, {gs['4-12']}); B_0 adopted over A: {g0['adopted']} ({g0['total']}, {g0['4-12']}); B_0 within tau of B_s on both: {bool(within)} -> chosen arm {chosen}")
            rep["chosen_arm"] = chosen
    else:
        decision.append("incomplete: both B_s and B_0 need six paired seeds")
    rep["verdict"] = "; ".join(decision)
else:
    g = rep["gate1"].get("b")
    rep["verdict"] = ("B ADOPTED" if (g and g["adopted"]) else ("B NOT ADOPTED" if g else "INCOMPLETE"))
json.dump(rep, open(args.out, "w"), indent=1, default=str)
lines = [f"# Plan P Phase 2 report ({datetime.now():%Y-%m-%d %H:%M})", "", f"Runs read: {rep['n_runs']}; estimator per run in the JSON. Base: base_v7_lr3e4 (user override of the Phase 1 gates). Units: forces per component meV/Å, energies meV/atom; τ = {TAU}.", "",
         "## Medians over seeds", "", "| arm | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E | C_Q transfer meV |", "|---|---|---|---|---|---|---|---|---|---|"]
NAMES = {"a": "A (Mulliken)", "b": "B (E_cP + E_PP)", "bm": "B(M)", "bs": "B_s (near Mulliken + far dipoles)", "b0": "B_0 (H0 near + far dipoles)", "bsm": "B_s(M)", "b0m": "B_0(M)"}
for a in ("a", "b", "bm", "bs", "b0", "bsm", "b0m"):
    if a in rep["medians"]:
        md = rep["medians"][a]; f = lambda k: "—" if md.get(k) is None else f"{md[k]:.2f}"
        lines.append(f"| {NAMES[a]} | {f('force_rmse')} | {f('4-12')} | {f('2-4')} | {f('4-8')} | {f('8-10')} | {f('10-12')} | {f('12-99')} | {f('e79_meV_atom')} | {f('cq_transfer_meV')} |")
lines += ["", "## Paired TOST (positive mean favours the second arm)", "", "| pair | metric | n | mean d | 90 % CI | reading |", "|---|---|---|---|---|---|"]
for pair, res in rep["paired"].items():
    for k in KEYS:
        r = res.get(k)
        if r: lines.append(f"| {pair} | {k} | {r['n']} | {r['mean_d']:+.2f} | [{r['lower']:+.2f}, {r['upper']:+.2f}] | {r['reading']} |")
lines += ["", f"## Gates", "", f"Gate 1 (adoption over A, total and 4–12 Å): {rep.get('gate1')}", "", f"Gate 2 per arm (C_Q 79→159 within 10 meV and 79-atom E ≤ 0.6): {rep['gate2']}", "", f"Cost, median s/epoch: {rep['s_per_epoch_median']}", "", f"**Verdict: {rep['verdict']}**"]
if args.md:
    open(args.md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[6:]))
