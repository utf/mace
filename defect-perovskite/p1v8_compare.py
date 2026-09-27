"""Base v8 (A2B provider) against the old base (base_v7_lr3e4, MH-0 provider): the three Phase 1 gates,
the E_PP mismatch, the run's own final validation and replay errors, side by side (markdown).

  python defect-perovskite/p1v8_compare.py --runs 'base v8 (A2B, lr 3e-4)=~/runs/dscc/p1v8/base_v8' 'base v8 lr 1e-4=~/runs/dscc/p1v8/base_v8_lr1e4' 'base v7 lr 3e-4 (MH-0)=~/runs/dscc/p1v2/base_v7_lr3e4' --out ~/runs/dscc/p1v8/COMPARE.md
"""
import argparse, json, re
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir pairs (P1_GATES.json, P2_EPP.json, *_run-1.log inside); the first is the reference base for the gate-3 shells")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(d):
    d = Path(d); g = json.load(open(d / "P1_GATES.json")); e = json.load(open(d / "P2_EPP.json")) if (d / "P2_EPP.json").exists() else {}
    log = next(iter(sorted(d.glob("*_run-1.log"))), None); fin = {}
    if log:
        txt = log.read_text()
        for head in ("Default", "pt_head"):
            m = re.findall(rf"head: {head}, .*?RMSE_E_per_atom=\s*([\d.]+) meV, RMSE_F=\s*([\d.]+) meV / A", txt)
            if m: fin[head] = (float(m[-1][0]), float(m[-1][1]))
    return g, e, fin


cols = [(r.split("=", 1)[0], load(r.split("=", 1)[1])) for r in args.runs]
BASE_V2 = {"F": 10.23, "E": 1.16, "replay": (4.05, 17.11), "shells": {"total": 33.2, "2-4": 74.9, "4-8": 30.4, "8-10": 15.3, "10-12": 8.9, "12+": 8.0}}
rows = []
def row(name, f, v2=""):
    rows.append((name, [f(g, e, fin) for _, (g, e, fin) in cols], v2))
row("gate 1: neutral validation F (meV/Å, ≤ 11.2)", lambda g, e, fin: f"{g['gate1']['rmse_F_meV_A']:.2f}", f"{BASE_V2['F']:.2f}")
row("gate 1: neutral validation E (meV/atom, ≤ 1.5)", lambda g, e, fin: f"{g['gate1']['rmse_E_meV_atom']:.2f}", f"{BASE_V2['E']:.2f}")
row("gate 2: ε∞ / provider reference", lambda g, e, fin: f"{g['gate2']['eps_inf_poly']:.3f} / {g['gate2']['eps_inf_ref']:.3f}", "no response")
for sp in ("Cl", "Cs", "Pb"):
    row(f"gate 2: |Z*| {sp} / reference (rel)", lambda g, e, fin, sp=sp: f"{g['gate2']['species'][sp]['frobenius']:.3f} / {g['gate2']['species'][sp]['ref']:.3f} ({100*g['gate2']['species'][sp]['rel']:.1f} %)", "no response")
for sh in ("total", "2-4", "4-8", "8-10", "10-12", "12+"):
    row(f"gate 3: charged frames {sh} Å (meV/Å; +3 over base v2 allowed)", lambda g, e, fin, sh=sh: f"{g['gate3']['shells'][sh]['rmse_meV_A']:.1f} ({g['gate3']['shells'][sh]['excess']:+.1f})", f"{BASE_V2['shells'][sh]:.1f} (reference)")
row("gates 1 / 2 / 3", lambda g, e, fin: " / ".join("pass" if g[k]["pass"] else "FAIL" for k in ("gate1", "gate2", "gate3")), "1 pass / — / reference")
row("E_PP mismatch on neutral validation (meV/Å, τ 1.7)", lambda g, e, fin: f"{e['rms_F_PP_meV_per_A']:.2f} {e['verdict']}" if e else "—", "—")
row("run's final validation, target head E / F", lambda g, e, fin: f"{fin['Default'][0]:.2f} / {fin['Default'][1]:.1f}" if "Default" in fin else "—", f"{BASE_V2['E']:.2f} / {BASE_V2['F']:.2f}")
row("run's final replay head E / F (drift from its provider)", lambda g, e, fin: f"{fin['pt_head'][0]:.2f} / {fin['pt_head'][1]:.1f}" if "pt_head" in fin else "—", f"{BASE_V2['replay'][0]:.2f} / {BASE_V2['replay'][1]:.1f}")
lines = ["| quantity | " + " | ".join(n for n, _ in cols) + " | base v2 (MH-1, no response) |", "|---|" + "---|" * (len(cols) + 1)]
for name, vals, v2 in rows: lines.append(f"| {name} | " + " | ".join(vals) + f" | {v2} |")
Path(args.out).write_text("\n".join(lines) + "\n"); print("\n".join(lines))
