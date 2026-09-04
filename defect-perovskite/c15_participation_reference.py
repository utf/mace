#!/usr/bin/env python3
"""The participation ratio's denominator is the wrong cell size. Measure it at the right one.

THE DEFECT. `b13_participation.py` reports `pristine_ratio = N_eff(charged) / N_eff(pristine)`
and its own docstring says the ratio "does not depend on the cell size". It cannot deliver
that, because the numerator is measured on **159-atom** charged cells and the denominator on
**80-atom** pristine ones — the dataset holds no defect-free cell bigger than 80, so the
script took the biggest it had. `N_eff` counts atoms, so a delocalised state's `N_eff` grows
with the cell: dividing a 159-atom measurement by an 80-atom reference inflates the ratio by
roughly the size ratio.

This is the third instance of the error §7.2b names — a reference computed on one cell
population and applied to another — after `c_shift_table` and the on-site centre.

THE FIX, measured here rather than argued. Tile a pristine 80-atom cell to 160 atoms, which
is defect-free and the right size, and take `N_eff` on that. The corrected ratio is the
charged 159-atom `N_eff` over the tiled reference.

WHAT SURVIVES REGARDLESS. Every cohort was divided by the same wrong denominator, so
comparisons BETWEEN cohorts are unaffected, and the raw `N_eff` on 159-atom charged frames is
size-matched already. The claim "arm A's carriers are more spread out than the head-only
cohort's" rests on those and does not move. What moves is the absolute ratio and any reading
of it as "a fraction of a delocalised band state".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_participation import participation  # noqa: E402
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from ta_band_edge import (capture, channel_of, load_frames, select,  # noqa: E402
                          with_hole_counter)

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402


def neff_over(model, frames, z, cutoff, device, ctx):
    vals = []
    for b, fr in make_batches(frames, z, cutoff, 1, device):
        try:
            _, out = capture(model, b, ctx=ctx, frames=fr)
            vals += participation(out, b, channel_of(b))
        except Exception:                      # a diagnostic must not kill the comparison
            continue
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return (float(v.mean()), int(v.size)) if v.size else (float("nan"), 0)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--n-pristine", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    charged = [a for a in select(load_frames(args.data), charged=True)
               if len(a) == 159][: args.frames]
    small = select_pristine(ase_read(str(args.data), ":"), 64)
    biggest = max(len(a) for a in small)
    ref_small = with_hole_counter([a for a in small if len(a) == biggest][: args.n_pristine])
    # The same cells tiled 2x1x1: defect-free, and 160 atoms rather than 80.
    ref_big = with_hole_counter([a.repeat((2, 1, 1)) for a in
                                 [b for b in small if len(b) == biggest][: args.n_pristine]])
    print(f"{len(charged)} charged 159-atom frames; reference cells "
          f"{len(ref_small)} of {biggest} and {len(ref_big)} of "
          f"{len(ref_big[0]) if ref_big else 0}", flush=True)

    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        adopt_model_dtype(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        cut = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        n_ch, _ = neff_over(model, charged, z, cut, args.device, ctx)
        n_small, _ = neff_over(model, ref_small, z, cut, args.device, ctx)
        n_big, _ = neff_over(model, ref_big, z, cut, args.device, ctx)
        row = dict(model=mp.name, neff_charged_159=n_ch,
                   neff_pristine_80=n_small, neff_pristine_160=n_big,
                   ratio_as_reported=n_ch / n_small if n_small else None,
                   ratio_size_matched=n_ch / n_big if n_big else None)
        rows.append(row)
        print(f"  {mp.name:16s} N_eff charged(159) {n_ch:6.2f}   "
              f"pristine(80) {n_small:6.2f} -> ratio {row['ratio_as_reported']:.3f}   "
              f"pristine(160) {n_big:6.2f} -> ratio {row['ratio_size_matched']:.3f}",
              flush=True)

    if rows:
        def agg(k):
            v = np.asarray([r[k] for r in rows if r[k] is not None], dtype=float)
            return v.mean(), v.std()
        print("\n=== over seeds ===")
        for k, label in (("neff_charged_159", "N_eff, charged 159 (size-matched already)"),
                         ("neff_pristine_80", "N_eff, pristine 80  (the old denominator)"),
                         ("neff_pristine_160", "N_eff, pristine 160 (the right one)"),
                         ("ratio_as_reported", "ratio as reported by b13"),
                         ("ratio_size_matched", "ratio, size-matched")):
            m, s = agg(k)
            print(f"  {label:44s} {m:8.3f} ± {s:.3f}")
        m80, _ = agg("neff_pristine_80")
        m160, _ = agg("neff_pristine_160")
        print(f"\n  the reference grows {m160 / m80:.2f}x from 80 to 160 atoms "
              f"(a perfectly delocalised state would give 2.00x)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
