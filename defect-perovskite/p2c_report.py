"""Plan P Phase 2c Step 1 read (registered 2026-10-11): arm B_w against B_s on the same seeds (0, 2, 6), paired mean
difference of the held-out force RMSE (per-component meV/A), total and 4-12 A, n = 3 point estimate (no TOST at this n);
kept if within +tau or better on either, failed if worse by more than tau on both. Also the per-epoch level readouts
(held 79-atom +1 frames; the embedded structure), s_near per seed, energies and cost. Writes JSON and a markdown block.

  python defect-perovskite/p2c_report.py --bw ~/runs/dscc/dscc_p2cwbw_s0 ~/runs/dscc/dscc_p2cwbw_s2 ~/runs/dscc/dscc_p2cwbw_s6 \
      --bs ~/runs/dscc/dscc_p2v8wbs_s0 ~/runs/dscc/dscc_p2v8wbs_s2 ~/runs/dscc/dscc_p2v8wbs_s6 --out ~/runs/dscc/p2c/P2C_GATES.json --md ~/runs/dscc/p2c/step1.md
"""
import argparse, importlib.util, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TAU = 1.7
SHELLS = ["2-4", "4-8", "8-10", "10-12", "12+"]


def pc(v):
    return None if v is None else 1000.0 * v / 3 ** 0.5


def read(rep):
    sh, cnt = rep.get("shell_rmse") or {}, rep.get("shell_counts") or {}
    out = {"force_rmse": pc(rep["force_rmse"])}
    for k in SHELLS:
        out[k] = pc(sh.get(k if k != "12+" else "12-99"))
    num = sum((sh[k] ** 2) * cnt[k] for k in ("4-8", "8-10", "10-12") if sh.get(k) is not None and cnt.get(k))
    den = sum(cnt[k] for k in ("4-8", "8-10", "10-12") if sh.get(k) is not None and cnt.get(k))
    out["4-12"] = pc(float(np.sqrt(num / den))) if den else None
    e = rep.get("energy_after_cq") or {}
    e79, e159 = e.get("79") or {}, e.get("159") or {}
    out["e79_meV_atom"] = e79.get("meV_per_atom"); out["e159_meV_atom"] = e159.get("meV_per_atom")
    out["cq_transfer_meV"] = None if (e79.get("c_q") is None or e159.get("c_q") is None) else 1000.0 * abs(e159["c_q"] - e79["c_q"])
    out["gate2"] = bool(out["cq_transfer_meV"] is not None and out["cq_transfer_meV"] <= 10.0 and out["e79_meV_atom"] is not None and out["e79_meV_atom"] <= 0.6)
    out["level_readout"] = rep.get("level_readout")
    return out


def load(run):
    run = Path(run).expanduser()
    f = run / "held_final_avg.json"
    rec = read(json.load(open(f if f.exists() else run / "held_final.json")))
    rec["estimator"] = "avg" if f.exists() else "final"
    rec["name"] = run.name
    hist = json.load(open(run / "history.json")) if (run / "history.json").exists() else []
    rec["history"] = [{"epoch": h["epoch"], "held": (h.get("held") or {}).get("force_rmse"), "dEp95": h.get("on_site_shift_eV_p95"), "sat": h.get("sat"),
                       "s_near": (h.get("level_readout") or {}).get("s_near"), "level": h.get("level_readout"), "time": h.get("time")} for h in hist]
    rec["seed"] = int(re.search(r"_s(\d+)$", run.name).group(1))
    return rec


ap = argparse.ArgumentParser()
ap.add_argument("--bw", nargs="+", required=True); ap.add_argument("--bs", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--md", default=None)
args = ap.parse_args()
bw = {r["seed"]: r for r in map(load, args.bw)}; bs = {r["seed"]: r for r in map(load, args.bs)}
seeds = sorted(set(bw) & set(bs))
rep = {"tau": TAU, "seeds": seeds, "per_seed": {"bw": {s: {k: v for k, v in bw[s].items() if k != "history"} for s in bw},
                                                "bs": {s: {k: v for k, v in bs[s].items() if k != "history"} for s in bs}}, "paired": {}}
for key in ("force_rmse", "4-12"):
    d = [bw[s][key] - bs[s][key] for s in seeds if bw[s][key] is not None and bs[s][key] is not None]
    rep["paired"][key] = {"n": len(d), "mean_bw_minus_bs": float(np.mean(d)) if d else None, "per_seed": d}
m_tot, m_far = rep["paired"]["force_rmse"]["mean_bw_minus_bs"], rep["paired"]["4-12"]["mean_bw_minus_bs"]
within = [m is not None and m <= TAU for m in (m_tot, m_far)]
worse = [m is not None and m > TAU for m in (m_tot, m_far)]
rep["force_gate"] = {"kept": any(within), "failed": all(worse), "reading": ("B_w kept (within +tau or better on " + ", ".join(k for k, w in zip(("total", "4-12 A"), within) if w) + ")") if any(within)
                     else "Step 1 force gate FAILED (worse than B_s by more than tau on both)"}
rep["medians"] = {arm: {k: float(np.median([r[k] for r in runs.values() if r.get(k) is not None])) for k in ["force_rmse", "4-12"] + SHELLS + ["e79_meV_atom", "cq_transfer_meV"] if any(r.get(k) is not None for r in runs.values())}
                  for arm, runs in (("bw", bw), ("bs", bs))}
rep["s_near_final"] = {s: (bw[s]["level_readout"] or {}).get("s_near") for s in bw}
rep["cost_s_per_epoch"] = {arm: {s: float(np.median([h["time"] for h in r["history"] if h.get("time")])) for s, r in runs.items() if r["history"]} for arm, runs in (("bw", bw), ("bs", bs))}
json.dump(rep, open(Path(args.out).expanduser(), "w"), indent=1, default=str)

f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
L = ["### Step 1: arm B_w against B_s (same seeds; held-out forces, per-component meV/Å; `model_avg.pt`)", "",
     "| arm | seed | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E (meV/atom) | C_Q transfer (meV) | energy gate | s (B_s) / s_near (B_w) final |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for arm, runs, name in (("bs", bs, "B_s"), ("bw", bw, "B_w")):
    for s in sorted(runs):
        r = runs[s]
        sval = rep["s_near_final"].get(s) if arm == "bw" else None
        L.append(f"| {name} | {s} | {f(r['force_rmse'])} | {f(r['4-12'])} | " + " | ".join(f(r[k], 1) for k in SHELLS) + f" | {f(r['e79_meV_atom'])} | {f(r['cq_transfer_meV'], 1)} | {'pass' if r['gate2'] else 'FAIL'} | {f(sval, 3)} |")
L += ["", f"Paired mean difference B_w − B_s (n = {len(seeds)}, point estimate): total {f(m_tot)} (per seed {[round(x, 2) for x in rep['paired']['force_rmse']['per_seed']]}), "
      f"4–12 Å {f(m_far)} (per seed {[round(x, 2) for x in rep['paired']['4-12']['per_seed']]}); τ = {TAU}. **{rep['force_gate']['reading']}.**", ""]
L += ["### Level readouts (B_w; separation eV / N_eff / pass on the held fold's 79-atom +1 frames; separation / N_eff / Σdq(< 5 Å) on the 639-atom label-tiled structure (b))", "",
      "| seed | epoch | held F | s_near | dEp95 | +1 frames sep p50 / N_eff p50 / pass | embed sep / N_eff / Σdq(<5 Å) |", "|---|---|---|---|---|---|---|"]
for s in sorted(bw):
    H = bw[s]["history"]
    picks = [h for h in H if h["epoch"] in (0, 5, 10, 20, 30, 40, 50, 59) or h is H[-1]]
    for h in picks:
        lv = h.get("level") or {}; p1 = lv.get("plus1_held79") or {}; em = lv.get("embed") or {}
        L.append(f"| {s} | {h['epoch']} | {f(pc(h['held']) if h['held'] is not None else None)} | {f(h.get('s_near'), 3)} | {f(h.get('dEp95'), 3)} | "
                 f"{f(p1.get('separation_p50'), 3)} / {f(p1.get('n_eff_p50'), 1)} / {f(p1.get('pass_fraction'), 2)} | {f(em.get('separation'), 3)} / {f(em.get('n_eff'), 1)} / {f((em.get('dq_within') or {}).get('5'), 2)} |")
    fin = bw[s]["level_readout"] or {}; p1 = fin.get("plus1_held79") or {}; em = fin.get("embed") or {}
    L.append(f"| {s} | avg | {f(bw[s]['force_rmse'])} | {f(fin.get('s_near'), 3)} | — | {f(p1.get('separation_p50'), 3)} / {f(p1.get('n_eff_p50'), 1)} / {f(p1.get('pass_fraction'), 2)} | {f(em.get('separation'), 3)} / {f(em.get('n_eff'), 1)} / {f((em.get('dq_within') or {}).get('5'), 2)} |")
L += ["", f"Cost, median s/epoch: B_w {rep['cost_s_per_epoch']['bw']}, B_s {rep['cost_s_per_epoch']['bs']}."]
md = "\n".join(L) + "\n"
if args.md:
    Path(args.md).expanduser().write_text(md)
print(md)
