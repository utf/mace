#!/usr/bin/env python3
"""Assemble the ten-gate table for one arm from the scorers' JSON, as markdown.

WHY THIS EXISTS. The gate table has been retyped from logs every cycle, and this cycle a
per-seed list retyped that way turned out not to match its JSON. A gate is a
pre-registered condition applied to a number; both halves should come out of the file that
holds the number. Everything here reads `~/runs/<tag>_*.json` and applies the conditions as
they are written in `CYCLE_SPEED_ARMS_SPEC.md` §6.

WHAT IS DECIDED HERE AND WHAT IS NOT. Gates 2, 3, 5, 6, 7, 9 and 10 are arithmetic and are
decided. Gates 1, 4 and 8 rest on a comparison to an earlier cohort or on a scorer's own
verdict; those are printed with the numbers a reader needs and marked `(read)` rather than
being given a fabricated boolean.

Missing files are reported as missing rather than skipped: a gate that is absent from a
table because its scorer never ran is exactly the failure this script is written against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

# Gate 2's pre-registered band for F4, and gate 5's gap window.
F4_BAND = (-0.142, -0.063)
GAP = (2.4, 0.1)
R_BAND = (0.66, 1.34)
SLOPE_REF = -0.17          # gate 9: within 2x of this, and never near +0.37
SLOPE_ARTEFACT = +0.37
STOP_MAX_SEEDS = 1         # gate 7 fires above this many seeds at a stop


def load(runs: Path, tag: str, name: str):
    p = runs / f"{tag}_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(v: Optional[float], places: int = 4, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{places}f}" if sign else f"{v:.{places}f}"


def spread(vals) -> str:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return "—"
    return f"{v.mean():+.4f} ± {v.std():.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="arma, armb, armc, stageb …")
    ap.add_argument("--runs", type=Path, default=Path.home() / "runs")
    ap.add_argument("--out", type=Path, default=None, help="write the markdown here too")
    args = ap.parse_args()
    R, tag = args.runs, args.tag

    adopt = load(R, tag, "adopt")
    extras = load(R, tag, "extras")
    f10 = load(R, tag, "f10") or load(R, tag, "centred_f10")
    dil = load(R, tag, "dilution")
    part = load(R, tag, "participation")
    b2 = load(R, tag, "b2")
    tiling = load(R, tag, "tiling")
    site = load(R, tag, "site_dev")
    depth = load(R, tag, "depth")

    lines: list[str] = []
    add = lines.append
    add(f"### Gates — {tag}")
    add("")
    add("| gate | condition | measured | verdict |")
    add("|---|---|---|---|")

    def missing(n: int, what: str, cond: str) -> None:
        add(f"| {n} | {cond} | **{what} did not run** | — |")

    # ---- 1 regression -------------------------------------------------------------
    if adopt is None:
        missing(1, f"{tag}_adopt.json", "no criterion that held in Stage B now fails")
    else:
        keys = [k for k in adopt[0]["verdict"] if k != "adopted"]
        counts = {k: sum(bool(r["verdict"][k]) for r in adopt) for k in keys}
        add(f"| 1 regression | no criterion that held in Stage B now fails | "
            + "; ".join(f"{k} {v}/{len(adopt)}" for k, v in counts.items())
            + " | (read) |")

    # ---- 2 F4 ---------------------------------------------------------------------
    if adopt is None:
        missing(2, f"{tag}_adopt.json", f"F4 in [{F4_BAND[0]}, {F4_BAND[1]}], ≥ 4/6")
    else:
        f4 = [r["f4"]["slope"] for r in adopt]
        n_in = sum(F4_BAND[0] <= s <= F4_BAND[1] for s in f4)
        add(f"| 2 F4 | slope in [{F4_BAND[0]:+.3f}, {F4_BAND[1]:+.3f}] in ≥ 4/6 | "
            f"{spread(f4)}; {n_in}/{len(f4)} in band "
            f"({', '.join(fmt(s, 4) for s in f4)}) | "
            f"**{'PASS' if n_in >= 4 else 'FAIL'}** |")

    # ---- 3 c consistency (report) --------------------------------------------------
    if extras is None:
        missing(3, f"{tag}_extras.json", "report c(79), c(159), Δc, predicted Δc")
    else:
        rows = extras["rows"]
        c79 = [r["c_79"] for r in rows]
        c159 = [r["c_159"] for r in rows]
        dc = [r["dc"] for r in rows]
        pdc = [r["predicted_dc"] for r in rows]
        add(f"| 3 c consistency | report only | c(79) {spread(c79)}, c(159) {spread(c159)}, "
            f"**Δc {spread(dc)}**, predicted {spread(pdc)} | (report) |")

    # ---- 4 F10 ---------------------------------------------------------------------
    if f10 is None:
        missing(4, f"{tag}_f10.json", "ligand-Cl − bulk-Cl > 50 meV and > 2σ, ≥ 4/6")
    else:
        n_pass = int(f10.get("n_pass", -1))
        rows = f10["rows"]
        lig = [r["shells"]["ligand_Cl"]["mean"] for r in rows if "ligand_Cl" in r["shells"]]
        blk = [r["shells"]["bulk_Cl"]["mean"] for r in rows if "bulk_Cl" in r["shells"]]
        gap_mev = [1000.0 * (a - b) for a, b in zip(lig, blk)]
        add(f"| 4 F10 | ligand-Cl − bulk-Cl > 50 meV and > 2σ, ≥ 4/6 | "
            f"{n_pass}/{len(rows)} seeds; difference {spread([g/1000 for g in gap_mev])} eV | "
            f"**{'PASS' if n_pass >= 4 else 'FAIL'}** |")

    # ---- 5 gap ---------------------------------------------------------------------
    src = depth["arms"].get(tag) if isinstance(depth, dict) and "arms" in depth else None
    gaps = ([r["pristine_gap"] for r in src] if src else
            ([r["depth"]["pristine_gap"] for r in adopt] if adopt else None))
    if gaps is None:
        missing(5, f"{tag}_adopt.json / {tag}_depth.json",
                f"pristine gap {GAP[0]} ± {GAP[1]} eV")
    else:
        n_in = sum(abs(g - GAP[0]) <= GAP[1] for g in gaps)
        add(f"| 5 gap | pristine gap {GAP[0]} ± {GAP[1]} eV | "
            f"{spread(gaps)}; {n_in}/{len(gaps)} in window | "
            f"**{'PASS' if n_in == len(gaps) else 'PARTIAL'}** "
            f"(pinned continuum and bandwidth: see the init-gate line) |")

    # ---- 6 dilution ----------------------------------------------------------------
    if dil is None:
        missing(6, f"{tag}_dilution.json",
                f"R in [{R_BAND[0]}, {R_BAND[1]}]; bound fraction; depth")
    else:
        rr = [r["all"]["median"] for r in dil]
        bf = [r["bound_fraction"] for r in dil]
        dm = [r["depth_med"] for r in dil]
        dl = [r["delta_L"] for r in dil]
        n_in = sum(R_BAND[0] <= x <= R_BAND[1] for x in rr)
        add(f"| 6 dilution | R inside [{R_BAND[0]}, {R_BAND[1]}] | "
            f"R {spread(rr)}, {n_in}/{len(rr)} inside; bound fraction {spread(bf)}; "
            f"depth {spread(dm)} eV; δ_L {spread(dl)} eV | "
            f"**{'PASS' if n_in >= 4 else 'FAIL'}** |")

    # ---- 7 stops -------------------------------------------------------------------
    if extras is None:
        missing(7, f"{tag}_extras.json",
                f"no integral type at its stop in more than {STOP_MAX_SEEDS}/6")
    else:
        g7 = extras["gate7"]
        n = len(extras["rows"])
        fired = [k for k, v in g7.items() if v > STOP_MAX_SEEDS]
        lb = {k: [r["decay_lengths"][k] for r in extras["rows"]]
              for k in extras["rows"][0]["decay_lengths"]}
        add(f"| 7 stops / L_b | no type at its stop in more than {STOP_MAX_SEEDS}/{n} | "
            + "; ".join(f"{k} {v}/{n}" for k, v in g7.items())
            + "; L_b " + ", ".join(f"{k} {np.mean(v):.3f}±{np.std(v):.3f}"
                                   for k, v in lb.items())
            + f" | **{'FAIL — arm C fires' if fired else 'PASS'}**"
            + (f" ({', '.join(fired)})" if fired else "") + " |")

    # ---- 8 participation ------------------------------------------------------------
    if part is None:
        missing(8, f"{tag}_participation.json", "participation ratio")
    else:
        on = [r["pristine_ratio"] for r in part["on"]]
        off = [r["pristine_ratio"] for r in part.get("off", [])]
        base = [r["pristine_ratio"] for r in part.get("baseline", [])]
        add(f"| 8 participation | report; spread against the comparison cohorts | "
            f"this arm {spread(on)}; head-only {spread(off)}; E_LR-off {spread(base)} | "
            "(read) |")

    # ---- 9 head slope ---------------------------------------------------------------
    if b2 is None:
        missing(9, f"{tag}_b2.json",
                f"79-atom d(δ_sr)/dd negative, within 2× of {SLOPE_REF}, not near "
                f"{SLOPE_ARTEFACT:+.2f}")
    else:
        s = [r["delta_sr_79_matched"]["slope"] for r in b2]
        neg = all(x < 0 for x in s)
        band = all(abs(x) <= 2 * abs(SLOPE_REF) and abs(x) >= abs(SLOPE_REF) / 2
                   for x in s)
        add(f"| 9 head slope | 79-atom d(δ_sr)/dd < 0, within 2× of {SLOPE_REF:+.2f}, "
            f"never near {SLOPE_ARTEFACT:+.2f} | {spread(s)} "
            f"({', '.join(fmt(x, 4) for x in s)}) | "
            f"**{'PASS' if neg and band else 'FAIL'}** |")

    # ---- 10 tiling (arm B) ----------------------------------------------------------
    if tiling is None:
        add("| 10 tiling (arm B) | ideal and thermal drift ≤ 0.3·D0; log the s distribution "
            f"| **{tag}_tiling.json did not run** — `c3_tiling_drift.py --thermal 3` is "
            "launched by hand, the chain parsed the older gates() body | — |")
    else:
        add(f"| 10 tiling (arm B) | ideal and thermal drift ≤ 0.3·D0 | "
            f"{json.dumps(tiling.get('summary', 'see the JSON'))[:220]} | (read) |")

    add("")
    if site is not None and site.get("rows"):
        dev = site["rows"]
        add("**§2.2 per-site charge deviation** (report): max "
            + spread([r["max_abs"] for r in dev]) + " e, rms "
            + spread([r["rms"] for r in dev]) + " e, sites at the ζ stop "
            + spread([r["at_stop"] for r in dev]) + ", worst |per-graph sum| "
            + f"{max(r['graph_sum_max'] for r in dev):.1e} e.")
    else:
        add("**§2.2 per-site charge deviation**: `c11_site_deviation.py` has not been run "
            f"for {tag} (`{tag}_site_dev.json`).")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
