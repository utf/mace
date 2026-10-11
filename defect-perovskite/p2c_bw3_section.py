"""Assemble P2C_REPORT.md section 1.5 (third B_w run, bw3) from the finisher's gate files, the histories, the step
records and the large-N read. Prints markdown.  Usage:
  python p2c_bw3_section.py --g2 ~/runs/dscc/p2c/P2C2_GATES.json --g3 ~/runs/dscc/p2c/P2C3_GATES.json \
     --hist3 ~/runs/dscc/dscc_p2cwbw3_s{0,2,6}/history.json --steps3 ~/runs/dscc/p2c/bw3_steps/s{0,2,6}.jsonl --largen ~/runs/dscc/p2c/bw3_sN/step0.json
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--g2", required=True); ap.add_argument("--g3", required=True)
ap.add_argument("--hist3", nargs=3, required=True); ap.add_argument("--steps3", nargs="*", default=[]); ap.add_argument("--largen", default=None)
a = ap.parse_args()
g2, g3 = json.load(open(Path(a.g2).expanduser())), json.load(open(Path(a.g3).expanduser()))
seeds = ["0", "2", "6"]
def row(arm, s, b):
    gate = "pass" if b["gate2"] else "FAIL"
    return f"| {arm} | {s} | {b['force_rmse']:.2f} | {b['4-12']:.2f} | {b['2-4']:.1f} | {b['4-8']:.1f} | {b['8-10']:.1f} | {b['10-12']:.1f} | {b['12+']:.1f} | {b['e79_meV_atom']:.2f} | {b['cq_transfer_meV']:.1f} | {gate} | {b['s_final']:.3f} |"
print("#### Forces and energies: B_s, rerun (bw2) and third run (bw3), same seeds (held-out, per-component meV/Å; `model_avg.pt`)\n")
print("| arm | seed | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E (meV/atom) | C_Q transfer (meV) | energy gate | s / s_near final |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in seeds: print(row("B_s", s, g2["per_seed"]["bs"][s]))
for s in seeds:
    b = dict(g2["per_seed"]["bw"][s]); b["s_final"] = g2["s_near_final"][s]; print(row("B_w rerun", s, b))
for s in seeds:
    b = dict(g3["per_seed"]["bw"][s]); b["s_final"] = g3["s_near_final"][s]; print(row("B_w third", s, b))
p3, p2 = g3["paired"], g2["paired"]
print(f"\nPaired mean difference against B_s (n = 3, point estimate; τ = 1.7): third run total {p3['force_rmse']['mean_bw_minus_bs']:+.2f} (per seed {[round(x, 2) for x in p3['force_rmse']['per_seed']]}), "
      f"4–12 Å {p3['4-12']['mean_bw_minus_bs']:+.2f} (per seed {[round(x, 2) for x in p3['4-12']['per_seed']]}); rerun was {p2['force_rmse']['mean_bw_minus_bs']:+.2f} / {p2['4-12']['mean_bw_minus_bs']:+.2f}. "
      f"**{g3['force_gate']['reading']}**\n")
# level readouts
print("#### Localisation (model_avg): separation / N_eff / pass on the held fold's 79-atom +1 frames; separation / N_eff / Σdq(< 5 Å) on the 639-atom label-tiled structure\n")
print("| arm | seed | +1 frames sep p50 / N_eff p50 / pass | 639 atoms sep / N_eff / Σdq(<5 Å) |\n|---|---|---|---|")
for arm, g in (("rerun", g2), ("third", g3)):
    for s in seeds:
        lr = g["per_seed"]["bw"][s]["level_readout"]; p, e = lr["plus1_held79"], lr["embed"]
        print(f"| B_w {arm} | {s} | {p['separation_p50']:.3f} / {p['n_eff_p50']:.1f} / {p['pass_fraction']:.2f} | {e['separation']:.3f} / {e['n_eff']:.1f} / {e['dq_within']['5']:.2f} |")
# spikes and gap resting value from histories
print("\n#### Stability: epoch losses, the gap's resting value and the period-2 statistic\n")
print("| seed | max epoch loss / median (epochs ≥ 2) | epochs above 10× | gap at epoch end: min / max / final (eV) | alternation fraction, epochs ≥ 3: median / max | lag-1 autocorrelation: median | steps with l_gap > 1e-3, epochs ≥ 3 |\n|---|---|---|---|---|---|---|")
stats = {}
for p in a.steps3:
    rows = [json.loads(l) for l in open(Path(p).expanduser())]; by = defaultdict(list)
    for r in rows: by[r["epoch"]].append(r)
    alts, acs, nbig = [], [], 0
    for e in sorted(by):
        if e < 3: continue
        x = np.array([r["gap"] - 2.40 for r in by[e]]); sg = np.sign(x)
        alts.append(float(np.mean(sg[1:] * sg[:-1] < 0))); acs.append(float(np.corrcoef(x[1:], x[:-1])[0, 1]) if x.std() > 0 else 0.0)
        nbig += int(sum(r["l_gap"] > 1e-3 for r in by[e]))
    stats[Path(p).stem.lstrip("s")] = (np.median(alts), max(alts), np.median(acs), nbig) if alts else None
for s, hp in zip(seeds, a.hist3):
    h = json.load(open(Path(hp).expanduser())); L = np.array([r["loss"] for r in h[2:]]); gaps = np.array([r["gap"] for r in h])
    st = stats.get(s)
    print(f"| {s} | {L.max() / np.median(L):.1f}× | {(L > 10 * np.median(L)).sum()} | {gaps.min():.3f} / {gaps.max():.3f} / {gaps[-1]:.3f} | " + (f"{st[0]:.2f} / {st[1]:.2f} | {st[2]:+.2f} | {st[3]} |" if st else "— | — | — |"))
print("\nReplay of the rerun's seed 6 for comparison: alternation 0.80–0.92, lag-1 autocorrelation −0.9 to −0.98 in every epoch from 3 to 29; epoch 30 loss 190× (original) / 440× (replay) the median.")
if a.largen:
    d = json.load(open(Path(a.largen).expanduser())); print(f"\n#### Large-N on the seed-median checkpoint ({d['tag']}, label tiling, core re-relaxed by that checkpoint)\n")
    print("| structure (639 atoms) | separation (eV) | N_eff | Σdq(< 3 / 5 / 8 / 12 Å) | Pb–Pb (Å) |\n|---|---|---|---|---|")
    for v in d["0.2"]:
        dq = {str(int(float(k))): x for k, x in v["within"].items()} if isinstance(v["within"], dict) else dict(zip(("3", "5", "8", "12", "16"), v["within"]))
        print(f"| {v['label']} | {v['separation_eV']:.3f} | {v['n_eff_hole']:.1f} | {dq['3']:.2f} / {dq['5']:.2f} / {dq['8']:.2f} / {dq['12']:.2f} | {v['pb_pb_across_vacancy_A']:.2f} |")
    r3 = d["0.3"]; print(f"\n79-atom +1 frames (0.3): {r3['n_frames']} frames; corr(sep, Pb–Pb) {r3['pearson_sep_vs_pbpb']:+.2f}, corr(N_eff, Pb–Pb) {r3['pearson_neff_vs_pbpb']:+.2f}.")
