#!/usr/bin/env python3
"""Plan v8 Stage 1.5: select the Stage-1 reference arm by the section 7.7 gates.

For each retrained arm tag (`s13ra` full, `s13rb` long_range, `s13rc` off) the decided
gates of section 7.7 are read from the scorers' JSON exactly as `c12_gate_table` decides
them -- F4 within the band, the pristine gap, dilution R inside its interval, F10 on the
centred channel, the A' regression criteria -- and the report quantities (depth,
participation, 79-atom force loss, learned Z) are tabulated with their seed spreads
(`stage13_collect`). The reference is the arm that passes the most decided gates; a tie goes
to the arm with the smaller change from the Stage B recipe, in the order full, long_range,
off. Writes the markdown table and the selection.

    python defect-perovskite/stage15_select.py --tags s13ra s13rb s13rc --runs ~/runs \\
        --out defect-perovskite/golden/stage15_selection.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c12_gate_table import F4_BAND, GAP, PASS_RATE, R_BAND, load  # noqa: E402
from stage13_collect import arm_row  # noqa: E402

ORDER = ["s13ra", "s13rb", "s13rc"]


def decided_gates(runs: Path, tag: str):
    """The section 7.7 gates c12 decides, as {name: (verdict, text)}."""
    out = {}
    adopt = load(runs, tag, "adopt")
    if adopt:
        f4 = [r["f4"]["slope"] for r in adopt]
        n_in = sum(F4_BAND[0] <= x <= F4_BAND[1] for x in f4)
        out["F4"] = (n_in >= PASS_RATE * len(f4), f"{np.mean(f4):+.4f} ± {np.std(f4):.4f}, "
                     f"{n_in}/{len(f4)} in band")
        gaps = [r["depth"]["pristine_gap"] for r in adopt
                if isinstance(r.get("depth"), dict) and "pristine_gap" in r["depth"]]
        if gaps:
            n_in = sum(abs(g - GAP[0]) <= GAP[1] for g in gaps)
            out["gap"] = (n_in >= PASS_RATE * len(gaps),
                          f"{np.mean(gaps):+.4f} ± {np.std(gaps):.4f}, {n_in}/{len(gaps)}")
        crits = {}
        for r in adopt:
            for k, v in (r.get("verdict") or {}).items():
                if isinstance(v, bool):
                    crits.setdefault(k, []).append(v)
        if crits:
            held = {k: sum(v) for k, v in crits.items()}
            out["regression"] = (all(sum(v) >= PASS_RATE * len(v) for v in crits.values()),
                                 "; ".join(f"{k} {n}/{len(crits[k])}" for k, n in held.items()))
    dil = load(runs, tag, "dilution")
    if dil:
        rs = [r["all"]["median"] for r in dil]
        n_in = sum(R_BAND[0] <= x <= R_BAND[1] for x in rs)
        out["dilution"] = (n_in >= PASS_RATE * len(rs),
                           f"R {np.mean(rs):+.4f} ± {np.std(rs):.4f}, {n_in}/{len(rs)} inside")
    f10 = load(runs, tag, "f10") or load(runs, tag, "centred_f10")
    if f10:
        rows = f10["rows"]
        n_pass = int(f10.get("n_pass", -1))
        lig = [r["shells"]["ligand_Cl"]["mean"] for r in rows if "ligand_Cl" in r["shells"]]
        blk = [r["shells"]["bulk_Cl"]["mean"] for r in rows if "bulk_Cl" in r["shells"]]
        diffs = [a - b for a, b in zip(lig, blk)]
        out["F10"] = (n_pass >= PASS_RATE * len(rows),
                      f"{n_pass}/{len(rows)} seeds; ligand-Cl − bulk-Cl "
                      f"{np.mean(diffs):+.4f} ± {np.std(diffs):.4f} eV" if diffs else
                      f"{n_pass}/{len(rows)} seeds")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tags", nargs="+", default=ORDER)
    p.add_argument("--runs", type=Path, default=Path.home() / "runs")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    lines = ["## Section 7.7 gates per arm", "",
             "| gate | " + " | ".join(args.tags) + " |", "|---|" + "---|" * len(args.tags)]
    gates = {t: decided_gates(args.runs, t) for t in args.tags}
    names = []
    for t in args.tags:
        for g in gates[t]:
            if g not in names:
                names.append(g)
    score = {t: 0 for t in args.tags}
    for g in names:
        cells = []
        for t in args.tags:
            v = gates[t].get(g)
            if v is None:
                cells.append("missing")
            else:
                cells.append(("**PASS** " if v[0] else "FAIL ") + v[1])
                score[t] += int(v[0])
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Report quantities (seed spread)", ""]
    rows = [arm_row(args.runs, t) for t in args.tags]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k != "tag":
                keys.append(k)
    lines += ["| quantity | " + " | ".join(args.tags) + " |", "|---|" + "---|" * len(args.tags)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(r.get(k, "missing")) for r in rows) + " |")
    best = max(score.values()) if score else 0
    winners = [t for t in args.tags if score[t] == best]
    selected = next((t for t in ORDER if t in winners), winners[0] if winners else None)
    lines += ["", f"**Selected Stage-1 reference: `{selected}`** (decided gates passed: "
              + ", ".join(f"{t} {score[t]}/{len(names)}" for t in args.tags)
              + "; ties go to the smaller change from the Stage B recipe)."]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        json.dump({"scores": score, "selected": selected, "gates": {
            t: {g: [bool(v[0]), v[1]] for g, v in gates[t].items()} for t in args.tags}},
                  open(args.out.with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
