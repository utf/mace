"""Base v8 (A2B provider) against the old base (base_v7_lr3e4, MH-0 provider): the three Phase 1 gates,
the E_PP mismatch, the run's own final validation and replay errors, side by side (markdown).

  python defect-perovskite/p1v8_compare.py --new ~/runs/dscc/p1v8/base_v8 --old ~/runs/dscc/p1v2/base_v7_lr3e4 --out ~/runs/dscc/p1v8/COMPARE.md
"""
import argparse, json, re
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True); ap.add_argument("--old", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(d):
    d = Path(d); g = json.load(open(d / "P1_GATES.json")); e = json.load(open(d / "P2_EPP.json")) if (d / "P2_EPP.json").exists() else {}
    log = next(iter(sorted(d.glob("*_run-1.log"))), None); fin = {}
    if log:
        txt = log.read_text()
        for head in ("Default", "pt_head"):
            m = re.findall(rf"head: {head}, .*?RMSE_E_per_atom=([\d.]+) meV, RMSE_F=([\d.]+) meV / A", txt)
            if m: fin[head] = (float(m[-1][0]), float(m[-1][1]))
    return g, e, fin


gn, en, fn = load(args.new); go, eo, fo = load(args.old)
rows = [("gate 1: neutral validation F (meV/Å, ≤ 11.2)", gn["gate1"]["rmse_F_meV_A"], go["gate1"]["rmse_F_meV_A"]),
        ("gate 1: neutral validation E (meV/atom, ≤ 1.5)", gn["gate1"]["rmse_E_meV_atom"], go["gate1"]["rmse_E_meV_atom"]),
        ("gate 2: ε∞ (relaxed cell) / its provider reference", f"{gn['gate2']['eps_inf_poly']:.3f} / {gn['gate2']['eps_inf_ref']:.3f}", f"{go['gate2']['eps_inf_poly']:.3f} / {go['gate2']['eps_inf_ref']:.3f}")]
for s in ("Cl", "Cs", "Pb"):
    rows.append((f"gate 2: |Z*| {s} / reference (rel)", f"{gn['gate2']['species'][s]['frobenius']:.3f} / {gn['gate2']['species'][s]['ref']:.3f} ({100*gn['gate2']['species'][s]['rel']:.1f} %)", f"{go['gate2']['species'][s]['frobenius']:.3f} / {go['gate2']['species'][s]['ref']:.3f} ({100*go['gate2']['species'][s]['rel']:.1f} %)"))
for sh in ("total", "2-4", "4-8", "8-10", "10-12", "12+"):
    a, b = gn["gate3"]["shells"][sh], go["gate3"]["shells"][sh]
    rows.append((f"gate 3: charged frames {sh} Å (meV/Å; base v2 {a['base_v2']}, +3 allowed)", f"{a['rmse_meV_A']:.1f} ({a['excess']:+.1f})", f"{b['rmse_meV_A']:.1f} ({b['excess']:+.1f})"))
rows.append(("gates 1 / 2 / 3", " / ".join("pass" if gn[g]["pass"] else "FAIL" for g in ("gate1", "gate2", "gate3")), " / ".join("pass" if go[g]["pass"] else "FAIL" for g in ("gate1", "gate2", "gate3"))))
if en and eo: rows.append(("E_PP mismatch on neutral validation (meV/Å, τ 1.7)", f"{en['rms_F_PP_meV_per_A']:.2f} {en['verdict']}", f"{eo['rms_F_PP_meV_per_A']:.2f} {eo['verdict']}"))
for head, name in (("Default", "run's final validation, target head E / F"), ("pt_head", "run's final replay head E / F (drift from the provider)")):
    if head in fn and head in fo: rows.append((name, f"{fn[head][0]:.2f} / {fn[head][1]:.1f}", f"{fo[head][0]:.2f} / {fo[head][1]:.1f}"))
lines = ["| quantity | base v8 (A2B provider) | base v7 lr 3e-4 (MH-0 provider) |", "|---|---|---|"]
for r in rows: lines.append(f"| {r[0]} | {r[1] if isinstance(r[1], str) else f'{r[1]:.2f}'} | {r[2] if isinstance(r[2], str) else f'{r[2]:.2f}'} |")
Path(args.out).write_text("\n".join(lines) + "\n"); print("\n".join(lines))
