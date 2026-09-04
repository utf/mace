#!/usr/bin/env python3
"""The head's own axial FORCE slope per size, against the label force slope per size.

WHY THE FORCE CHANNEL AND NOT THE ENERGY ONE. `delta_sr` is what F4 reads, and b2 measures its
d-slope at both sizes -- but the Stage-3 loss is forces plus `loss_gap` and contains no energy
term at all, so `d(delta_sr)/dd` has never been a fitted quantity. It is a by-product. The
axial hub force IS fitted, on every charged frame, at both sizes. So this is the one place
where "does the head reproduce the labels' d-trend?" is a fair question rather than a question
about a readout nobody optimised.

Compared against `b1_label_slope_by_size.py`'s axial arm, which is the same statistic computed
on `F_DFT - F_base` with cross-fit bases. Here it is `forces - base_forces`: the head's own
carrier force, through the model's own forward, so the two differ in what produced the
correction and in nothing else. The long-range branch is off in Stage 3, so `forces -
base_forces` is the counting head alone.

`model_axial` is imported from `s3_dilution` rather than rewritten. A second implementation of
the hub-axis projection would make any disagreement between this and the dilution gate a
question about two conventions instead of about two models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_size_slopes import stratified  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from s3_dehead_trend import fit_with_ci, hub_separation  # noqa: E402
from s3_dilution import model_axial  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

SIZES = (79, 159)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-small", type=int, default=120)
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    rng = np.random.default_rng(args.seed)
    charged = select(load_frames(args.data), charged=True)
    pool = {}
    for n in SIZES:
        fr = [a for a in charged if len(a) == n]
        d = np.array([hub_separation(a) for a in fr])
        ok = np.isfinite(d)
        fr = [a for a, o in zip(fr, ok) if o]
        d = d[ok]
        idx = stratified(fr, d, args.n_small if n != 159 else len(fr), rng)
        pool[n] = [fr[i] for i in idx]
        print(f"  {n} atoms: {len(idx)} of {len(fr)} frames, "
              f"d {d[idx].min():.2f}-{d[idx].max():.2f} A", flush=True)
    window = (max(min(hub_separation(a) for a in v) for v in pool.values()),
              min(max(hub_separation(a) for a in v) for v in pool.values()))
    print(f"  matched window {window[0]:.2f}-{window[1]:.2f} A", flush=True)

    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        row = {"model": Path(mp).name}
        for n in SIZES:
            got = model_axial(model, pool[n], z, cutoff, args.device, ctx, batch=2)
            d = np.array([r["d"] for r in got])
            v = np.array([r["axial"] for r in got])
            good = np.isfinite(d) & np.isfinite(v)
            s, lo, hi, c = fit_with_ci(d[good], v[good])
            row[f"{n}_full"] = dict(n=int(good.sum()), slope=s, ci=[lo, hi], corr=c)
            g2 = good & (d >= window[0]) & (d <= window[1])
            if g2.sum() >= 5:
                s2, lo2, hi2, c2 = fit_with_ci(d[g2], v[g2])
                row[f"{n}_matched"] = dict(n=int(g2.sum()), slope=s2, ci=[lo2, hi2],
                                           corr=c2)
        rows.append(row)
        print(f"  {row['model']:20s} model axial slope  79 "
              f"{row['79_matched']['slope']:+.4f}  159 "
              f"{row['159_matched']['slope']:+.4f} (eV/A) per A", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        print("\n=== pooled over seeds (matched window) ===")
        for n in SIZES:
            v = np.array([r[f"{n}_matched"]["slope"] for r in rows
                          if f"{n}_matched" in r])
            print(f"  model axial slope {n:4d} atoms  {v.mean():+.4f} +- {v.std():.4f}")
        print("\n  Against b1's LABEL axial slopes: 79 atoms +0.4202 over the full range but "
              "-0.047\n  [-0.308, +0.214] inside the neutral-dense window, and 159 atoms "
              "-0.1901 with corr\n  -0.982 against the production base. The 79-atom label "
              "trend is base extrapolation;\n  the question here is whether the head "
              "reproduces it or the 159-atom one.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
