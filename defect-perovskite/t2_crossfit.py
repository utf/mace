"""T2: the honest null control, using cross-fit bases instead of the 60-frame hold-out.

The original null could not answer the question. Its 60 neutral frames have median d(Pb-Pb)
4.81 A and top out at 5.77 A, while the charged frames sit at median 5.49 A and the neutral
training set reaches 6.54 A. Comparing charged against that null therefore compared two
different regions of the d(Pb-Pb) axis, and any "excess" could have been the base extrapolating
into a geometry the null never sampled. E0's excess rested on the same 60 frames.

Cross-fitting fixes the range without starving any base of the long-d tail: the 1191 neutral
defective frames are split into 4 folds, one diagnostic base is trained per fold on the other
three, and every frame is scored by the base that never saw it.

The charged frames are PARTITIONED across the folds the same way (frame i -> base i mod 4)
rather than all 1047 being pushed through all four bases. Each frame is then counted exactly
once, so the two arms carry equal weight per base, and the pooled rows are not four correlated
copies of the same charged set. Charged frames are out-of-sample for every fold base anyway --
no base trained on any charged frame -- so no fold assignment is more valid than another.

Reported per 0.2 A bin of d(Pb-Pb) over 4.84-6.54 A, which is where the two populations
actually overlap. Comparing pooled medians alone would let the d-distribution difference
masquerade as a carrier signal, which is the exact failure the 60-frame null had.

The vacancy assignment is used to define the axis and to bin. It never touches the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r0_pair_force import collect  # noqa: E402


def load(path, device):
    return torch.load(path, map_location=device, weights_only=False).to(device).eval()


def binned(rows, edges, key="axial", absolute=True):
    d = np.array([r["d_pbpb"] for r in rows])
    v = np.array([r[key] for r in rows])
    if absolute:
        v = np.abs(v)
    which = np.digitize(d, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        m = which == b
        out.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(m.sum()),
                        median=float(np.median(v[m])) if m.any() else float("nan"),
                        mean=float(v[m].mean()) if m.any() else float("nan")))
    return out


def ratio_stats(rows, key_a="hub_mag", key_b="bulk_pb_mag"):
    """E0's shell/bulk statistic, recomputed so it can be read against this null."""
    a = np.array([r[key_a] for r in rows])
    b = np.array([r[key_b] for r in rows])
    good = np.isfinite(a) & np.isfinite(b) & (b > 0)
    r = a[good] / b[good]
    return dict(median=float(np.median(r)), mean=float(r.mean()), n=int(good.sum()))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=here / "dataset_cf")
    ap.add_argument("--runs", type=Path, default=Path.home() / "runs")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap frames per fold per arm (debugging only)")
    ap.add_argument("--out", type=Path, default=here / "t2_crossfit.json")
    # CPU by default: the GPU is running the local R2 cell, and this analysis must not
    # inflate the footprint that cell measures to size its own concurrency.
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    _assert_repo()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    charged_all = read(args.data / "eval_qp1.xyz", ":")

    null_rows, charged_rows = [], []
    for k in range(args.folds):
        path = args.runs / f"cf_base_f{k}" / f"cf_base_f{k}.model"
        if not path.exists():
            raise SystemExit(f"missing fold base: {path}")
        model = load(path, args.device)

        held = read(args.data / f"fold{k}" / "null_oof.xyz", ":")
        mine = [a for i, a in enumerate(charged_all) if i % args.folds == k]
        if args.limit:
            held, mine = held[: args.limit], mine[: args.limit]

        nr = collect(model, held, z_table, args.cutoff, args.device)
        cr = collect(model, mine, z_table, args.cutoff, args.device)
        print(f"fold {k}: null {len(nr)}/{len(held)} usable, "
              f"charged {len(cr)}/{len(mine)} usable", flush=True)
        null_rows += nr
        charged_rows += cr

    edges = np.round(np.arange(4.84, 6.54 + 1e-9, 0.2), 3)
    nb = binned(null_rows, edges)
    cb = binned(charged_rows, edges)

    print(f"\n=== T2 cross-fit null: |axial| residual by d(Pb-Pb) ===")
    print(f"  null {len(null_rows)} frames (out-of-fold neutral), "
          f"charged {len(charged_rows)} frames")
    print(f"  {'bin (A)':>12}  {'n_null':>7} {'n_chg':>6}  {'null':>8} {'charged':>8}  excess")
    for a, b in zip(nb, cb):
        if not a["n"] or not b["n"]:
            print(f"  {a['lo']:5.2f}-{a['hi']:5.2f}  {a['n']:7d} {b['n']:6d}  "
                  f"{'--':>8} {'--':>8}   --")
            continue
        ex = b["median"] / max(a["median"], 1e-12)
        print(f"  {a['lo']:5.2f}-{a['hi']:5.2f}  {a['n']:7d} {b['n']:6d}  "
              f"{a['median']:8.4f} {b['median']:8.4f}  x{ex:.2f}")

    nd = np.array([r["d_pbpb"] for r in null_rows])
    cd = np.array([r["d_pbpb"] for r in charged_rows])
    print(f"\n  d(Pb-Pb) coverage: null median {np.median(nd):.2f} A "
          f"[{nd.min():.2f}, {nd.max():.2f}], charged median {np.median(cd):.2f} A "
          f"[{cd.min():.2f}, {cd.max():.2f}]")

    na = np.abs([r["axial"] for r in null_rows])
    ca = np.abs([r["axial"] for r in charged_rows])
    pooled = float(np.median(ca)) / max(float(np.median(na)), 1e-12)
    print(f"  pooled |axial| median: null {np.median(na):.4f}  "
          f"charged {np.median(ca):.4f}  excess x{pooled:.2f}")

    ns = np.array([r["axial"] for r in null_rows])
    cs = np.array([r["axial"] for r in charged_rows])
    print(f"  signed axial mean: null {ns.mean():+.4f}  charged {cs.mean():+.4f} eV/A "
          f"(+ = the two Pb pushed apart)")

    nh, ch = ratio_stats(null_rows), ratio_stats(charged_rows)
    print(f"\n=== E0 recheck: hub/bulk-Pb |dF| against the cross-fit null ===")
    print(f"  null median {nh['median']:.2f}  charged median {ch['median']:.2f}  "
          f"excess x{ch['median'] / max(nh['median'], 1e-12):.2f}")
    print("  (E0's own null gave charged 3.93 / null 2.29 = x1.72 on seed 1 and "
          "4.01 / 2.06 = x1.95 on seed 2, from 60 frames that did not cover the charged\n"
          "   d-range. The 1.9 figure quoted elsewhere is R0's hub/CAGE ratio -- a different\n"
          "   statistic, and not what this line compares against.)")

    payload = dict(
        bins=dict(edges=edges.tolist(), null=nb, charged=cb),
        pooled=dict(null_abs_median=float(np.median(na)),
                    charged_abs_median=float(np.median(ca)), excess=pooled,
                    null_signed_mean=float(ns.mean()),
                    charged_signed_mean=float(cs.mean())),
        hub_bulk=dict(null=nh, charged=ch),
        n=dict(null=len(null_rows), charged=len(charged_rows), folds=args.folds),
    )
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
