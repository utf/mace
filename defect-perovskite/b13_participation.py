#!/usr/bin/env python3
"""Carrier participation with the long-range branch on and off, same seeds, one flag apart.

WHY IT IS ASKED. Arm A of the joint run ends with participation 11.14, 11.19, 10.94, 5.43, 5.51
and 5.73 -- the six seeds split three and three, one group roughly twice as localised as the
other. (This docstring first said "one seed of four"; that was written from wave 1 alone, before
a5 and a6 had finished.) E_LR switches on at epoch 12 of 20 in that cohort, so it is a candidate
for the split. The control is seeds 1-4 with `USE_LONG_RANGE=False` and nothing else changed, so
a difference is attributable to E_LR and to nothing else; a5 and a6 have no E_LR-off partner.

WHAT IS REPORTED, and why more than one number. `N_eff = 1 / sum alpha_i^2` is a spread over a
FIXED cell, so on its own it is not a statement about boundness -- an earlier cycle measured it
moving from 5.51 to 14.47 as the electronic temperature went from 5 to 100 meV, which is why it
was demoted from a gate to a metric. Three things are printed together:

  * `N_eff` on the charged 159-atom frames, per seed and pooled;
  * the ratio to the PRISTINE cell scored with the same counters, which is this run's own
    baseline for what "delocalised" looks like -- the ratio, not N_eff, is the localisation
    statement;
  * the frontier level's depth below the pristine conduction manifold, because a participation
    difference that comes with a depth difference is a different physical claim from one that
    does not.

WHY NOT THE NULL CHANNELS, which is what the spectral-era gates used. `neff_and_nulls` compares
the supervised channel against the mean of the other three, and for a four-channel head those
three take no gradient and are a genuine per-run baseline. The counting head has NO channels:
its `alpha` comes from the density-matrix difference and is broadcast across the four slots, so
the "nulls" are copies of the active channel and the ratio is identically 1.000. The training
log shows it plainly -- `partic=[11.137 11.137 11.137 11.137]`, four identical numbers. A
column of 1.000 would have looked like a measurement.

The pristine reference is the honest replacement and it is the same trick Delta_bind uses: the
defect-free cell scored with the SAME counters, which is off-distribution for it by
construction -- a perfect crystal has no carrier to hold -- and that is exactly the point. It
asks where this Hamiltonian would put a carrier if there were no vacancy, which is the
reference a localisation claim needs. Diagnostic only, never a loss.

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
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS  # noqa: E402
from ta_band_edge import (capture, channel_of, load_frames, select,  # noqa: E402
                          with_hole_counter)


def participation(out, batch, channel):
    """`1 / sum_i alpha_i^2` per graph on the supervised channel. The atom count for a
    uniform field, 1 for a fully localised one -- the same quantity the trainer logs as
    `partic`, computed here from the same `carrier_alpha`."""
    alpha = out["carrier_alpha"]
    idx = batch.batch
    per = []
    for g in range(int(batch.num_graphs)):
        v = alpha[idx == g, channel]
        s2 = float((v * v).sum())
        per.append(1.0 / s2 if s2 > 0 else float("nan"))
    return per


def measure(model, charged, pristine, z, cutoff, device, ctx):
    """`(N_eff, pristine ratio, depth below the conduction manifold)` for one model."""
    # The delocalised reference FIRST: the defect-free cell under the same counters. Its
    # participation is what this model does with a carrier when there is no vacancy to bind
    # it, so the charged cell's participation divided by it is a localisation statement that
    # does not depend on the cell size or on T_el the way N_eff alone does.
    free = []
    for b, fr in make_batches(pristine, z, cutoff, 1, device):
        try:
            _, out = capture(model, b, ctx=ctx, frames=fr)
            free += participation(out, b, channel_of(b))
        except Exception:
            continue
    free_mean = float(np.nanmean(free)) if free else float("nan")

    neff, ratio = [], []
    for b, fr in make_batches(charged, z, cutoff, 1, device):
        try:
            _, out = capture(model, b, ctx=ctx, frames=fr)
            a = float(np.nanmean(participation(out, b, channel_of(b))))
            neff.append(a)
            # Below 1 means the vacancy localises the carrier relative to no vacancy at all.
            # Near 1 means it does not.
            ratio.append(a / free_mean if np.isfinite(free_mean) and free_mean > 0
                         else float("nan"))
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
            float(np.nanmean(depth)) if depth else float("nan"),
            free_mean)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", nargs="+", type=Path, required=True)
    ap.add_argument("--off", nargs="+", type=Path, required=True)
    ap.add_argument("--baseline", nargs="*", type=Path, default=[],
                    help="a third cohort scored on the SAME frames, so participation can be "
                    "compared across regimes. The head-only s7 models report N_eff 4.85 +- "
                    "0.25 in their own harness while the joint run logs partic ~11, but "
                    "those came from different frame sets at different points in training, "
                    "so the comparison has never actually been made like for like")
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

    rows = {"on": [], "off": [], "baseline": []}
    for arm, paths in (("on", args.on), ("off", args.off),
                       ("baseline", args.baseline)):
        for mp in paths:
            if not Path(mp).exists():
                print(f"  MISSING {mp}", flush=True)
                continue
            model = torch.load(mp, map_location=args.device,
                               weights_only=False).to(args.device).eval()
            # The batches built for this model must carry ITS dtype: AtomicData uses the
            # process default, which is float32, while the joint run trains at float64.
            adopt_model_dtype(model)
            cutoff = max(float(model.r_max),
                         float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
            ctx = ForwardContext.production(model, device=args.device,
                                            eps_inf=args.eps_inf)
            neff, ratio, depth, free = measure(model, charged, pristine, z, cutoff,
                                               args.device, ctx)
            rows[arm].append(dict(model=Path(mp).name, neff=neff, pristine_ratio=ratio,
                                  pristine_neff=free, depth_from_cbm=depth,
                                  long_range=bool(getattr(model, "use_long_range", False))))
            print(f"  [{arm:8s}] {Path(mp).name:22s} N_eff {neff:7.3f}   pristine "
                  f"{free:7.3f}   ratio {ratio:6.3f}   depth {depth:+.3f} eV", flush=True)
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
              f"{a['neff'] - b['neff']:+8.3f}   {a['pristine_ratio']:9.3f} "
              f"{b['pristine_ratio']:9.3f}   {a['depth_from_cbm']:+9.3f} "
              f"{b['depth_from_cbm']:+9.3f}")

    for key, label in (("neff", "N_eff"), ("pristine_neff", "N_eff, pristine"),
                       ("pristine_ratio", "charged / pristine"),
                       ("depth_from_cbm", "depth from CBM")):
        on = np.array([r[key] for r in rows["on"]], dtype=float)
        off = np.array([r[key] for r in rows["off"]], dtype=float)
        line = (f"\n  {label:18s} E_LR on {np.nanmean(on):+8.3f} +- {np.nanstd(on):.3f}   "
                f"off {np.nanmean(off):+8.3f} +- {np.nanstd(off):.3f}   "
                f"difference {np.nanmean(on) - np.nanmean(off):+.3f}")
        if rows["baseline"]:
            base = np.array([r[key] for r in rows["baseline"]], dtype=float)
            line += (f"\n  {'':18s} head-only baseline {np.nanmean(base):+8.3f} "
                     f"+- {np.nanstd(base):.3f}")
        print(line)
    on_n = np.array([r["neff"] for r in rows["on"]], dtype=float)
    off_n = np.array([r["neff"] for r in rows["off"]], dtype=float)
    print(f"\n  Seed spread in N_eff: on {np.nanmax(on_n) - np.nanmin(on_n):.3f}, "
          f"off {np.nanmax(off_n) - np.nanmin(off_n):.3f}.")
    print("  If the spread collapses with E_LR off, the branch is what separated the seeds. "
          "If it\n  survives, the outlier is the initialisation and E_LR is not implicated.")
    print("\n  N_eff is a spread over a FIXED cell and is T_el-sensitive (5.51 to 14.47 over "
          "5 to\n  100 meV in an earlier cycle), so it is a metric and not a gate. The ratio "
          "to the SAME\n  model's pristine cell under the same counters is the localisation "
          "statement -- the null\n  channels cannot serve, because this head broadcasts one "
          "alpha across all four slots and\n  the ratio to them is identically 1.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
