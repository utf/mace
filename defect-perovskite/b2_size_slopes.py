#!/usr/bin/env python3
"""The model's own d-slopes at BOTH cell sizes: `delta_sr` and the frontier level.

WHAT IT DISCRIMINATES. F4 is scored on the 159-atom subset alone, so a shortfall there has
two readings that the gate cannot tell apart:

  * the labels do not contain the trend at the size the head mostly trains on (F6's
    question, measured in `b1_label_slope_by_size.py`), so the head is fitting a mixture in
    which the large-cell trend is 1.6% of the charged frames; or
  * the head cannot express the trend at EITHER size, in which case the label mixture is
    irrelevant and the deficiency is architectural.

Only a per-size measurement of the MODEL separates them, and the model's 79-atom slope has
never been looked at -- s3_dehead_trend refuses the small size on purpose, because pooling
re-imports M1b's base-extrapolation contamination into a LABEL comparison. That refusal does
not apply here: `delta_sr` is the head's own output, with no base in it, so its d-dependence
at 79 atoms is a clean statement about the head.

BOTH OBSERVABLES, because they answer different halves. `delta_sr` is F4's own quantity.
`lambda_frontier` is the level the two-centre argument speaks about, and it is what an
envelope or coupling fix would move first -- an energy slope can be flat while the level
slope is not, if the occupation difference cancels it.

The 79-atom arm is SUBSAMPLED on a stratified d-grid rather than truncated: taking the first
N frames takes them in file order, which is trajectory order, so it would sample a few
snapshots' worth of d rather than the range. The subsample size is reported.
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
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import capture, load_frames, select  # noqa: E402

SIZES = (79, 159)


def stratified(frames, d, n_keep, rng):
    """Up to `n_keep` frames spread evenly over the d range, not the first `n_keep`."""
    order = np.argsort(d)
    if len(order) <= n_keep:
        return list(order)
    edges = np.linspace(0, len(order), n_keep + 1).astype(int)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            picked.append(int(order[rng.integers(lo, hi)]))
    return picked


def measure(model, frames, z_table, cutoff, device, ctx):
    """`(delta_sr, lambda_frontier)` per frame, one forward each."""
    from mace.modules.defect_counting import VALENCE, changed_level_index

    sr, lam_f = [], []
    for b, fr in make_batches(frames, z_table, cutoff, 1, device):
        internals, out = capture(model, b, ctx=ctx, frames=fr)
        sr.append(float(out["delta_sr_energy"].reshape(-1)[0]))
        lam = internals["lam"][0, 0]
        lam = lam[lam < 500.0]
        n_total = sum(VALENCE[int(z)] for z in fr[0].get_atomic_numbers())
        k = changed_level_index(n_total, b.carrier_counts.reshape(-1).tolist())
        lam_f.append(float(lam[k]) if 0 <= k < lam.numel() else float("nan"))
    return np.array(sr), np.array(lam_f)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[here / "dataset_pbe" / "train.xyz"])
    ap.add_argument("--n-small", type=int, default=120)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    rng = np.random.default_rng(args.seed)
    frames = []
    for p in args.data:
        frames += select(load_frames(p), charged=True)
    pool = {}
    for n in SIZES:
        fr = [a for a in frames if len(a) == n]
        d = np.array([hub_separation(a) for a in fr])
        ok = np.isfinite(d)
        fr = [a for a, o in zip(fr, ok) if o]
        d = d[ok]
        idx = stratified(fr, d, args.n_small if n != 159 else len(fr), rng)
        pool[n] = ([fr[i] for i in idx], d[idx])
        print(f"  {n} atoms: {len(idx)} of {len(fr)} frames, "
              f"d {d[idx].min():.2f}-{d[idx].max():.2f} A", flush=True)

    window = (max(v[1].min() for v in pool.values()),
              min(v[1].max() for v in pool.values()))
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
        row = {"model": Path(mp).name,
               "gamma": float(model.spectral.h.on_site_range),
               "t_el": float(model.spectral.t_el),
               "smearing_family": getattr(model.spectral, "smearing_family",
                                          "fermi (attribute absent)")}
        for n in SIZES:
            fr, d = pool[n]
            sr, lam = measure(model, fr, z, cutoff, args.device, ctx)
            inw = (d >= window[0]) & (d <= window[1])
            for key, v in (("delta_sr", sr), ("lambda", lam)):
                good = np.isfinite(v)
                s, lo, hi, c = fit_with_ci(d[good], v[good])
                row[f"{key}_{n}_full"] = dict(n=int(good.sum()), slope=s, ci=[lo, hi],
                                              corr=c)
                g2 = good & inw
                if g2.sum() >= 5:
                    s2, lo2, hi2, c2 = fit_with_ci(d[g2], v[g2])
                    row[f"{key}_{n}_matched"] = dict(n=int(g2.sum()), slope=s2,
                                                     ci=[lo2, hi2], corr=c2)
        rows.append(row)
        print(f"  {row['model']:20s} "
              f"d(delta_sr)/dd  79 {row['delta_sr_79_matched']['slope']:+.4f}  "
              f"159 {row['delta_sr_159_matched']['slope']:+.4f}   |   "
              f"dlambda/dd  79 {row['lambda_79_matched']['slope']:+.4f}  "
              f"159 {row['lambda_159_matched']['slope']:+.4f} eV/A", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        print("\n=== pooled over seeds (matched window) ===")
        for key, label in (("delta_sr", "d(delta_sr)/dd"), ("lambda", "dlambda/dd")):
            for n in SIZES:
                v = np.array([r[f"{key}_{n}_matched"]["slope"] for r in rows
                              if f"{key}_{n}_matched" in r])
                if v.size:
                    print(f"  {label:16s} {n:4d} atoms  {v.mean():+.4f} +- {v.std():.4f} "
                          f"eV/A over {v.size} seeds")
        print("\n  Read against b1's LABEL slopes at the same two sizes. If the head's "
              "79-atom\n  slope tracks the 79-atom label slope while both fall short at "
              "159, the head is\n  fitting the mixture it was given; if the head is short "
              "at both sizes, it is not.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
