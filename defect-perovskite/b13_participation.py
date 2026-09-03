#!/usr/bin/env python3
"""Carrier participation with the long-range branch on and off, same seeds, one flag apart.

WHY IT IS ASKED. Arm A of the joint run ends with participation 11.14, 11.19, 10.94 and 5.43 --
one seed roughly twice as localised as its siblings. E_LR switches on at epoch 12 of 20 in that
cohort, so it is a candidate for the spread. The control is the same four seeds with
`USE_LONG_RANGE=False` and nothing else changed, so a difference is attributable to E_LR and to
nothing else.

WHAT IS REPORTED, and why more than one number. `N_eff = 1 / sum alpha_i^2` is a spread over a
FIXED cell, so on its own it is not a statement about boundness -- an earlier cycle measured it
moving from 5.51 to 14.47 as the electronic temperature went from 5 to 100 meV, which is why it
was demoted from a gate to a metric. Three things are printed together:

  * `N_eff` on the charged 159-atom frames, per seed and pooled;
  * the ratio to the run's OWN null channels, which take no gradient and so are that run's
    baseline for what "delocalised" looks like -- the ratio, not N_eff, is the localisation
    statement;
  * the frontier level's depth below the pristine conduction manifold, because a participation
    difference that comes with a depth difference is a different physical claim from one that
    does not.

Paired by seed as well as pooled: with four seeds a pooled mean can hide a swap, and the whole
point is that one seed behaved differently from the other three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b6_depth_edges import align, occupied_count, spectrum  # noqa: E402
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS  # noqa: E402
from ta_band_edge import (capture, channel_of, load_frames, select,  # noqa: E402
                          with_hole_counter)
from tb_edge import neff_and_nulls  # noqa: E402


def measure(model, charged, pristine, z, cutoff, device, ctx):
    """`(N_eff, null ratio, depth below the conduction manifold)` for one model."""
    neff, ratio = [], []
    for b, fr in make_batches(charged, z, cutoff, 1, device):
        try:
            _, out = capture(model, b, ctx=ctx, frames=fr)
            act, nul = neff_and_nulls(out, b, channel_of(b))
            a, n = float(np.nanmean(act)), float(np.nanmean(nul))
            neff.append(a)
            # The nulls take no gradient, so they are THIS run's own baseline for what a
            # delocalised channel looks like. A ratio near 1 is no localisation at all.
            ratio.append(a / max(n, 1e-30))
        except Exception:                       # a diagnostic must not kill the comparison
            continue

    n_pri = occupied_count(pristine)
    pri, cbm = [], []
    for b, fr in make_batches(pristine, z, cutoff, 1, device):
        lam = spectrum(model, b, fr, ctx)
        pri.append(lam)
        cbm.append(float(lam[n_pri]))
    pri_mean = np.mean(pri, axis=0)

    from mace.modules.defect_counting import VALENCE, changed_level_index

    n_def = occupied_count(charged)
    depth = []
    for b, fr in make_batches(charged, z, cutoff, 1, device):
        lam = spectrum(model, b, fr, ctx)
        n_total = sum(VALENCE[int(zz)] for zz in fr[0].get_atomic_numbers())
        k = changed_level_index(n_total, b.carrier_counts.reshape(-1).tolist())
        if not 0 <= k < lam.size:
            continue
        shift, _ = align(lam, n_def, pri_mean, n_pri)
        depth.append(float(np.mean(cbm)) + shift - float(lam[k]))
    return (float(np.nanmean(neff)) if neff else float("nan"),
            float(np.nanmean(ratio)) if ratio else float("nan"),
            float(np.nanmean(depth)) if depth else float("nan"))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", nargs="+", type=Path, required=True)
    ap.add_argument("--off", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--n-pristine", type=int, default=3)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    charged = [a for a in select(load_frames(args.data), charged=True)
               if len(a) == CLEAN_NATOMS][: args.frames]
    all_pristine = select_pristine(ase_read(str(args.data), ":"), 64)
    biggest = max(len(a) for a in all_pristine)
    pristine = with_hole_counter([a for a in all_pristine
                                  if len(a) == biggest][: args.n_pristine])
    print(f"{len(charged)} charged {CLEAN_NATOMS}-atom frames, {len(pristine)} pristine "
          f"of {biggest}", flush=True)

    rows = {"on": [], "off": []}
    for arm, paths in (("on", args.on), ("off", args.off)):
        for mp in paths:
            if not Path(mp).exists():
                print(f"  MISSING {mp}", flush=True)
                continue
            model = torch.load(mp, map_location=args.device,
                               weights_only=False).to(args.device).eval()
            cutoff = max(float(model.r_max),
                         float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
            ctx = ForwardContext.production(model, device=args.device,
                                            eps_inf=args.eps_inf)
            neff, ratio, depth = measure(model, charged, pristine, z, cutoff,
                                         args.device, ctx)
            rows[arm].append(dict(model=Path(mp).name, neff=neff, null_ratio=ratio,
                                  depth_from_cbm=depth,
                                  long_range=bool(getattr(model, "use_long_range", False))))
            print(f"  [{arm:3s}] {Path(mp).name:22s} N_eff {neff:7.3f}   "
                  f"null ratio {ratio:6.3f}   depth {depth:+.3f} eV", flush=True)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if not (rows["on"] and rows["off"]):
        raise SystemExit("one arm produced nothing; there is no comparison to make")

    print("\n=== paired by seed ===")
    print(f"  {'seed':6s} {'N_eff on':>10} {'N_eff off':>10} {'delta':>8}   "
          f"{'ratio on':>9} {'ratio off':>9}   {'depth on':>9} {'depth off':>9}")
    for a, b in zip(rows["on"], rows["off"]):
        print(f"  {a['model'][-8:-6]:6s} {a['neff']:10.3f} {b['neff']:10.3f} "
              f"{a['neff'] - b['neff']:+8.3f}   {a['null_ratio']:9.3f} "
              f"{b['null_ratio']:9.3f}   {a['depth_from_cbm']:+9.3f} "
              f"{b['depth_from_cbm']:+9.3f}")

    for key, label in (("neff", "N_eff"), ("null_ratio", "null ratio"),
                       ("depth_from_cbm", "depth from CBM")):
        on = np.array([r[key] for r in rows["on"]], dtype=float)
        off = np.array([r[key] for r in rows["off"]], dtype=float)
        print(f"\n  {label:16s} on {np.nanmean(on):+8.3f} +- {np.nanstd(on):.3f}   "
              f"off {np.nanmean(off):+8.3f} +- {np.nanstd(off):.3f}   "
              f"difference {np.nanmean(on) - np.nanmean(off):+.3f}")
    on_n = np.array([r["neff"] for r in rows["on"]], dtype=float)
    off_n = np.array([r["neff"] for r in rows["off"]], dtype=float)
    print(f"\n  Seed spread in N_eff: on {np.nanmax(on_n) - np.nanmin(on_n):.3f}, "
          f"off {np.nanmax(off_n) - np.nanmin(off_n):.3f}.")
    print("  If the spread collapses with E_LR off, the branch is what separated the seeds. "
          "If it\n  survives, the outlier is the initialisation and E_LR is not implicated.")
    print("\n  N_eff is a spread over a FIXED cell and is T_el-sensitive (5.51 to 14.47 over "
          "5 to\n  100 meV in an earlier cycle), so it is a metric and not a gate. The ratio "
          "to this run's\n  own null channels is the localisation statement.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
