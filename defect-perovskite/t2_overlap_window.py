"""T2 -- is the R0 axial signal real, or base extrapolation along a learned bond?

R0 found the residual on the two vacancy-sharing Pb to be axial, 15-19x the neutral null, and
linear in d(Pb-Pb). The open caveat: charged frames sample longer Pb-Pb separations than
neutral ones, so at large d the neutral-only base is extrapolating -- and extrapolation error
along a learned bond is itself axial and d-dependent. Seed agreement rules out noise but not a
shared systematic.

The test is to repeat the regression only where BOTH ensembles have support, so the base is
interpolating for every point used. If the slope and the excess survive there, the signal is
carrier physics; if they collapse, part of R0 was extrapolation.

PASS: slope within ~50% of the full-range 0.549 eV/A^2, and excess over null still >= 5x.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import evaluate  # noqa: E402
from r0_pair_force import axis_and_residuals  # noqa: E402

FULL_SLOPE = 0.549      # eV/A^2, from the full-range R0 fit


def collect(model, frames, z_table, cutoff, device):
    base, _ = evaluate(model, frames, z_table, cutoff, device)
    rows = [axis_and_residuals(a, base[k]) for k, a in enumerate(frames)]
    return [r for r in rows if r is not None]


def summarise(tag, rows, lo, hi):
    d = np.array([r["d_pbpb"] for r in rows])
    ax = np.array([r["axial"] for r in rows])
    inside = (d >= lo) & (d <= hi)
    if inside.sum() < 8:
        return None
    dd, aa = d[inside], ax[inside]
    slope, intercept = np.polyfit(dd, aa, 1)
    r = float(np.corrcoef(dd, aa)[0, 1])
    print(f"  {tag:<22} n={int(inside.sum()):4d}  slope {slope:+.4f}  r {r:+.3f}  "
          f"|axial| median {np.median(np.abs(aa)):.4f}")
    return dict(n=int(inside.sum()), slope=float(slope), r=r,
                axial_abs_median=float(np.median(np.abs(aa))))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_e0")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    sets = {}
    for label, fname in (("charged", "eval_qp1.xyz"), ("null", "eval_q0_null.xyz")):
        frames = read(args.data / fname, ":")[: args.limit]
        sets[label] = collect(model, frames, z_table, args.cutoff, args.device)

    d_ch = np.array([r["d_pbpb"] for r in sets["charged"]])
    d_nu = np.array([r["d_pbpb"] for r in sets["null"]])
    print(f"d(Pb-Pb) support:  charged [{d_ch.min():.2f}, {d_ch.max():.2f}] "
          f"median {np.median(d_ch):.2f}")
    print(f"                   null    [{d_nu.min():.2f}, {d_nu.max():.2f}] "
          f"median {np.median(d_nu):.2f}")

    # Where both ensembles have support: from the 10th percentile of the charged
    # distribution to the 90th of the neutral one.
    lo = float(np.percentile(d_ch, 10))
    hi = float(np.percentile(d_nu, 90))
    print(f"\noverlap window: [{lo:.2f}, {hi:.2f}] A  "
          f"(charged p10 to null p90)\n")

    out = {}
    print("full range:")
    for label in ("charged", "null"):
        out[f"{label}_full"] = summarise(label, sets[label], -np.inf, np.inf)
    print("\noverlap window only:")
    for label in ("charged", "null"):
        out[f"{label}_window"] = summarise(label, sets[label], lo, hi)

    cw, nw = out.get("charged_window"), out.get("null_window")
    print("\n=== T2 verdict ===")
    if not cw or not nw:
        print("  too few frames inside the window to judge")
        return
    excess = cw["axial_abs_median"] / max(nw["axial_abs_median"], 1e-12)
    ratio = cw["slope"] / FULL_SLOPE
    print(f"  slope in window {cw['slope']:+.4f} vs full-range {FULL_SLOPE:+.4f} "
          f"-> {ratio:.2f}x")
    print(f"  excess over null in window: x{excess:.2f}")
    ok = (0.5 <= ratio <= 1.5) and excess >= 5.0
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    print("  FAIL means part of the R0 signal is base extrapolation: weight the T1")
    print("  nbhd-loss results over the full-loss ones and stop treating the slope as a")
    print("  target.")

    out.update(window=[lo, hi], slope_ratio=float(ratio), excess=float(excess),
               verdict="PASS" if ok else "FAIL")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
