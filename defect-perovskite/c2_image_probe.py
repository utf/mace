#!/usr/bin/env python3
"""Section 2.3 forward-only probe on the s7 cohort (F15).

The image-compensation term switched on and off on the same trained models, both cell
sizes, four readings per model and size:

  * depth of the frontier level below the pristine conduction manifold (b13's alignment);
  * the participation ratio charged / pristine under the same counters (b13's statistic);
  * R_bound, the model's own axial hub-force dilution 79 -> 159 on bound frames
    (s3_dilution's statistic, common-delta_L convention);
  * the 79-atom force loss on a fixed frame subset (b9's statistic at scale 1).

F15 forecasts: depth increases more at 79 than at 159 atoms, the ratio falls, R moves to
0.9-1.0, and the 79-atom force loss does not rise. Forward-only: nothing trains, and the
term is toggled by the model attribute the config would set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_participation import measure  # noqa: E402
from b9_hub_ceiling import force_loss  # noqa: E402
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, graph_cutoff_for  # noqa: E402
from s3_dilution import depths, hub_axis, level_spacing, model_axial  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402
from test2_size import matched_ratios  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402

try:
    from b13_participation import with_hole_counter
except ImportError:  # pragma: no cover
    with_hole_counter = lambda x: x  # noqa: E731


def r_bound(model, small, large, pristine, z, cutoff, device, ctx):
    """s3_dilution's bound-conditioned R, in one call."""
    delta_l = level_spacing(model, pristine, z, cutoff, device, ctx)
    dep = depths(model, large, z, cutoff, device, ctx)
    bound = dep / delta_l > 2.0
    rows_s = model_axial(model, small, z, cutoff, device, ctx)
    rows_l = model_axial(model, large, z, cutoff, device, ctx)
    keep = [r for r, b in zip(rows_l, bound) if b]
    if len(keep) < 3:
        return float("nan"), float(bound.mean()), delta_l
    bnd = matched_ratios(keep, rows_s, 0.10, np.random.default_rng(0))
    return float(bnd["median"]), float(bound.mean()), float(delta_l)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-small", type=int, default=24)
    ap.add_argument("--n-pristine", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable([17, 55, 82])
    frames = load_frames(args.data)
    charged = select(frames, charged=True)
    large = [a for a in charged if len(a) == 159]
    rng = np.random.default_rng(0)
    # Small frames inside the +-0.1 A d-windows of the large ones, as s3_dilution selects
    # them, so R_bound has matches; a random subset of those bounds the cost.
    big_d = np.array([hub_axis(a)[3] for a in large])
    small_all = [a for a in charged if len(a) == 79
                 and np.min(np.abs(big_d - hub_axis(a)[3])) <= 0.10]
    small = [small_all[i] for i in sorted(rng.choice(len(small_all),
                                                     min(args.n_small, len(small_all)),
                                                     replace=False))]
    small_force = small
    all_pristine = select_pristine(frames, 64)
    biggest = max(len(a) for a in all_pristine)
    pristine = with_hole_counter([a for a in all_pristine if len(a) == biggest][
        : args.n_pristine])
    pairs = []
    for a in small_force:
        ia, ib, _, _ = hub_axis(a)
        pairs.append((ia, ib))
    print(f"{len(large)} charged 159-atom, {len(small)} charged 79-atom (subset), "
          f"{len(pristine)} pristine of {biggest}", flush=True)

    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device, weights_only=False).to(
            args.device).eval()
        adopt_model_dtype(model)
        cutoff = graph_cutoff_for(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        for term in (False, True):
            model.image_compensation = bool(term)
            row = dict(model=mp.name, term=bool(term))
            for tag, frs in (("159", large), ("79", small)):
                neff, ratio, depth, free = measure(model, frs, pristine, z, cutoff,
                                                   args.device, ctx)
                row[f"neff_{tag}"] = neff
                row[f"ratio_{tag}"] = ratio
                row[f"depth_{tag}"] = depth
            r, frac, delta_l = r_bound(model, small, large, pristine, z, cutoff,
                                       args.device, ctx)
            row["R_bound"], row["bound_fraction"], row["delta_L"] = r, frac, delta_l
            row["force_loss_79"] = force_loss(model, small_force, pairs, z, cutoff,
                                              args.device, ctx, 1.0)
            rows.append(row)
            print(f"  {mp.name:22s} term={'on ' if term else 'off'}  depth 79 {row['depth_79']:+.4f} "
                  f"159 {row['depth_159']:+.4f}  ratio 79 {row['ratio_79']:.3f} 159 "
                  f"{row['ratio_159']:.3f}  R_bound {r:.3f} (bound {frac:.0%})  "
                  f"F-loss(79) {row['force_loss_79']:.5f}", flush=True)
        model.image_compensation = False
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n=== F15, pooled over models (on minus off) ===")
    def pooled(key):
        on = np.array([r[key] for r in rows if r["term"]])
        off = np.array([r[key] for r in rows if not r["term"]])
        return off, on
    verdict = {}
    for key, label in (("depth_79", "depth from CBM, 79"), ("depth_159", "depth from CBM, 159"),
                       ("ratio_79", "participation ratio, 79"),
                       ("ratio_159", "participation ratio, 159"),
                       ("R_bound", "R_bound"), ("force_loss_79", "force loss, 79")):
        off, on = pooled(key)
        d = on - off
        verdict[key] = dict(off=float(np.nanmean(off)), on=float(np.nanmean(on)),
                            delta=float(np.nanmean(d)), delta_sd=float(np.nanstd(d)),
                            per_model=d.tolist())
        print(f"  {label:28s} off {np.nanmean(off):+.4f}  on {np.nanmean(on):+.4f}  "
              f"delta {np.nanmean(d):+.4f} +- {np.nanstd(d):.4f}  "
              f"(sign per model: {''.join('+' if x > 0 else '-' for x in d)})")
    c1 = verdict["depth_79"]["delta"] > verdict["depth_159"]["delta"] > 0
    c2 = verdict["ratio_79"]["delta"] < 0 and verdict["ratio_159"]["delta"] < 0
    r_on = verdict["R_bound"]["on"]
    c3 = 0.9 <= r_on <= 1.0
    c4 = verdict["force_loss_79"]["delta"] <= 0
    print(f"\n  F15 clauses: depth rises more at 79 than 159: {c1}; ratio falls: {c2}; "
          f"R -> 0.9-1.0: {c3} (R_on {r_on:.3f}); 79-atom force loss does not rise: {c4}")
    print(f"  -> F15 {'HOLDS' if all((c1, c2, c3, c4)) else 'FAILS'} "
          f"({sum((c1, c2, c3, c4))}/4 clauses)")
    args.out.write_text(json.dumps(dict(rows=rows, verdict=verdict,
                                        clauses=dict(depth=c1, ratio=c2, R=c3, force=c4)),
                                   indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
