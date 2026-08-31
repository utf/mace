"""T2 Option B -- a one-sided screen while the cross-fit bases train.

Matched-d regression using neutral frames the production base was TRAINED on. Their residual
is fitting error, not generalisation error, so the null is biased LOW and the excess over it
is inflated by an unknown amount.

That makes this test one-sided, and only in one direction:

  * a large excess proves nothing -- it is what a biased-low null produces anyway;
  * an excess that COLLAPSES toward 1 at matched d is real trouble for R0, because it
    survives a null that was stacked in R0's favour.

Run to find out today whether R0 is in danger, not to confirm it. The cross-fit null is what
can confirm it.
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


def counters(atoms):
    c = atoms.info.get("carrier_counts")
    if isinstance(c, str):
        c = [int(x) for x in c.split()]
    return np.asarray(c, dtype=int)


def collect(model, frames, z_table, cutoff, device, batch=4):
    base, _ = evaluate(model, frames, z_table, cutoff, device, batch_size=batch)
    rows = [axis_and_residuals(a, base[k]) for k, a in enumerate(frames)]
    return [r for r in rows if r is not None]


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path,
                    default=Path.home() / "runs/e0_base_s1/e0_base_s1.model")
    ap.add_argument("--source", type=Path, default=here / "dataset_pbe")
    ap.add_argument("--limit-null", type=int, default=400)
    ap.add_argument("--limit-charged", type=int, default=400)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    frames = read(args.source / "train.xyz", ":")
    neutral = [a for a in frames if not counters(a).any() and len(a) != 80][: args.limit_null]
    charged = [a for a in frames if counters(a).any()][: args.limit_charged]

    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    nulls = collect(model, neutral, z_table, args.cutoff, args.device)
    chgd = collect(model, charged, z_table, args.cutoff, args.device)

    dn = np.array([r["d_pbpb"] for r in nulls]); an = np.array([r["axial"] for r in nulls])
    dc = np.array([r["d_pbpb"] for r in chgd]); ac = np.array([r["axial"] for r in chgd])

    print("NOTE: null frames were in the base's TRAINING set -- fitting error, not")
    print("generalisation. The null is biased low and the excess is inflated.\n")
    print(f"null    n={len(dn):4d}  d [{dn.min():.2f}, {dn.max():.2f}] median {np.median(dn):.2f}")
    print(f"charged n={len(dc):4d}  d [{dc.min():.2f}, {dc.max():.2f}] median {np.median(dc):.2f}\n")

    edges = np.arange(4.8, 6.8001, 0.2)
    print(f"{'d window':>12} {'n_null':>7} {'n_chg':>6} {'null ax':>9} {'chg ax':>9} {'excess':>8}")
    print("-" * 56)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mn = (dn >= lo) & (dn < hi)
        mc = (dc >= lo) & (dc < hi)
        if mn.sum() < 5 or mc.sum() < 5:
            continue
        n_ax = float(np.median(np.abs(an[mn])))
        c_ax = float(np.median(np.abs(ac[mc])))
        ex = c_ax / max(n_ax, 1e-12)
        rows.append(dict(lo=float(lo), hi=float(hi), n_null=int(mn.sum()),
                         n_chg=int(mc.sum()), null_axial=n_ax, chg_axial=c_ax, excess=ex))
        print(f"{lo:5.1f}-{hi:4.1f} {mn.sum():7d} {mc.sum():6d} {n_ax:9.4f} "
              f"{c_ax:9.4f} {ex:8.2f}")

    if not rows:
        print("\nno bin has both ensembles populated")
        return

    ex = np.array([r["excess"] for r in rows])
    print(f"\nmatched-d excess: min {ex.min():.2f}  median {np.median(ex):.2f}  "
          f"max {ex.max():.2f}")
    verdict = ("R0 IN TROUBLE" if np.median(ex) < 2.0 else "not refuted by this screen")
    print(f"verdict: {verdict}")
    print("  A collapse toward 1 here would be decisive against R0, since the null is")
    print("  biased in R0's favour. A large value proves nothing; wait for the cross-fit.")

    if args.out:
        args.out.write_text(json.dumps(dict(bins=rows, verdict=verdict), indent=2))
        print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
