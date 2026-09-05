"""Stage 1.4 cohort table: the tiling ladder over several seeds' ladder records.

Reads the JSON records `stage14_ladder.py` writes and prints one markdown table with the
per-seed values and the mean ± sd of: Phi_FF at each rung, the a/L fit, the band term at
each rung and its 1/L fit, the near-vacancy force convergence per rung (charged, periodic and
isolated; neutral), the charged and neutral defect virial trace per rung, and the thermal
image contribution per rung.

Usage: python stage14_cohort.py golden/stage14_ladder_s13a_s*.json [--out table.md]
"""
import argparse
import json
import sys

import numpy as np


def ms(vals, fmt="{:+.3f}"):
    a = np.asarray(vals, dtype=float)
    if a.size == 1:
        return fmt.format(a[0])
    return (fmt + " ± " + fmt.replace("+", "")).format(a.mean(), a.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    recs = [json.load(open(p)) for p in args.records]
    rungs = [k for k in recs[0]["sizes"] if k != "fits"]
    L = [recs[0]["sizes"][k]["L"] for k in rungs]
    rows = []

    def row(name, per_seed, fmt="{:+.3f}"):
        rows.append((name, [fmt.format(v) for v in per_seed], ms(per_seed, fmt)))

    for k, l in zip(rungs, L):
        row(f"Φ_FF, charged, {k}× (L {l:.1f} Å) [eV]",
            [r["sizes"][k]["charged"]["periodic"]["frontier"] for r in recs])
    fits = [r["sizes"]["fits"]["charged"]["e_pbc_minus_e_inf"] for r in recs]
    row("Φ_FF fit a (eV·Å)", [f["inv_L_coefficient"] for f in fits])
    row("Φ_FF fit b (eV)", [f["inv_L_offset"] for f in fits])
    row("Φ_FF log-log exponent", [f["exponent"] for f in fits])
    for k in rungs:
        row(f"band term, charged, {k}× [eV]",
            [r["sizes"][k]["charged"]["periodic"]["band"] for r in recs])
    bf = [r["sizes"]["fits"]["charged"]["band"] for r in recs]
    row("band fit: L→∞ offset (eV)", [f["inv_L_offset"] for f in bf])
    row("band fit: 1/L coefficient (eV·Å)", [f["inv_L_coefficient"] for f in bf])
    for k in rungs[:-1]:
        row(f"max |ΔF| within radius vs {rungs[-1]}×, charged periodic, {k}× [eV/Å]",
            [r["sizes"][k]["charged"]["force_convergence"]["max_dF_periodic_vs_largest"]
             for r in recs], "{:.3f}")
        row(f"same, charged isolated, {k}× [eV/Å]",
            [r["sizes"][k]["charged"]["force_convergence"]["max_dF_isolated_vs_largest"]
             for r in recs], "{:.3f}")
        row(f"same, neutral, {k}× [eV/Å]",
            [r["sizes"][k]["neutral"]["force_convergence"]["max_dF_periodic_vs_largest"]
             for r in recs], "{:.2e}")
    for k in rungs:
        row(f"defect virial trace, charged periodic, {k}× [eV]",
            [r["sizes"][k]["charged"]["defect_virial_trace"] for r in recs])
    for k in rungs:
        row(f"defect virial trace, neutral, {k}× [eV]",
            [r["sizes"][k]["neutral"]["defect_virial_trace"] for r in recs])
    for k in rungs:
        if k in recs[0].get("thermal", {}):
            row(f"thermal image contribution, {k}× [eV]",
                [r["thermal"][k]["thermal_image_contribution"] for r in recs])
    row("N_eff, charged 1×", [r["sizes"][rungs[0]]["charged"]["periodic"]["n_eff"]
                              for r in recs], "{:.1f}")
    row("w_ref, charged 1×", [r["sizes"][rungs[0]]["charged"]["periodic"]["w_ref"]
                              for r in recs], "{:.3f}")

    names = [r["model"].split("/")[-1].replace(".model", "") for r in recs]
    lines = ["| quantity | " + " | ".join(names) + " | mean ± sd |",
             "|---|" + "---|" * len(names) + "---|"]
    for name, per, m in rows:
        lines.append(f"| {name} | " + " | ".join(per) + f" | {m} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
