#!/usr/bin/env python3
"""F4's follow-up: is the missing half of the slope the HEAD's or the BASE's?

THE PROBLEM. Every wired seed reproduces the sign of the label trend and 40-55% of its
magnitude, with tight per-frame CIs that exclude the -0.134 eV/A reference. Six seeds landing
in the same place with small intervals is a BIAS, not scatter, and "5/6 pass" reports the
threshold while saying nothing about the pattern.

THE DISCRIMINATION. -0.134 eV/A was measured as the slope of `E_DFT - E_base` against d, on
CROSS-FIT bases. These models carry a different, frozen Stage-A base. If that base has its own
d-dependent error, then the slope this head is supposed to reproduce is not -0.134 -- it is
whatever `E_DFT - E_base` slopes at for THIS base. So compute both, per model:

    slope_head      d(delta_sr)/dd            what the head predicts
    slope_residual  d(E_label - E_base)/dd    what this model's base leaves for it

and read the pair:

  head ~ residual, residual short of -0.134  ->  the base is the problem. The head is
                                                reproducing everything left for it, and the
                                                gap is base extrapolation -- which the joint
                                                run removes at source by training the base.
  head short of residual                     ->  the head is the problem, and no amount of
                                                joint training fixes it.

THE GAUGE DOES NOT ENTER. E_base at defect geometries is latent -- the warning about an
under-determined gauge is real -- but a gauge freedom is a CONSTANT offset, and this is a
SLOPE against d. A constant shifts the intercept and leaves the slope untouched.
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
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import (CLEAN_NATOMS, REFERENCE_SLOPE, fit_with_ci,  # noqa: E402
                             hub_separation)
from ta_band_edge import load_frames, select  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == CLEAN_NATOMS]
    d = np.array([hub_separation(a) for a in frames])
    ok = np.isfinite(d)
    print(f"{len(frames)} charged {CLEAN_NATOMS}-atom frames, "
          f"d {d[ok].min():.2f}-{d[ok].max():.2f} A", flush=True)

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
        head, resid = [], []
        for b, fr in make_batches(frames, z, cutoff, 1, args.device):
            with torch.no_grad():
                out = model(ctx.forward_dict(b, fr, requires_grad=False),
                            training=False, compute_force=False)
            head.append(float(out["delta_sr_energy"].reshape(-1)[0]))
            label = getattr(b, "energy", None)
            resid.append(float("nan") if label is None else
                         float(label.reshape(-1)[0]) - float(
                             out["base_energy"].reshape(-1)[0]))
        head, resid = np.array(head), np.array(resid)
        gh = ok & np.isfinite(head)
        gr = ok & np.isfinite(resid)
        if gh.sum() < 5 or gr.sum() < 5:
            print(f"  {Path(mp).name}: not enough usable frames "
                  f"(head {int(gh.sum())}, residual {int(gr.sum())})", flush=True)
            continue
        sh, hlo, hhi, ch = fit_with_ci(d[gh], head[gh])
        sr, rlo, rhi, cr = fit_with_ci(d[gr], resid[gr])
        # Which explanation the pair supports. "head ~ residual" is judged by whether the
        # residual slope sits inside the head slope's own interval, not by a ratio: with
        # n = 16 a ratio of two noisy slopes says less than the interval does.
        verdict = ("base" if hlo <= sr <= hhi else
                   "head" if abs(sh) < abs(sr) else "over")
        rows.append(dict(model=Path(mp).name, slope_head=sh, head_ci=[hlo, hhi],
                         slope_residual=sr, residual_ci=[rlo, rhi], corr_head=ch,
                         corr_residual=cr, verdict=verdict,
                         residual_matches_reference=bool(rlo <= REFERENCE_SLOPE <= rhi)))
        print(f"  {rows[-1]['model']:22s} head {sh:+.4f} [{hlo:+.4f}, {hhi:+.4f}]   "
              f"residual {sr:+.4f} [{rlo:+.4f}, {rhi:+.4f}]   "
              f"reference {REFERENCE_SLOPE:+.4f}   -> {verdict}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        sr = np.array([r["slope_residual"] for r in rows])
        sh = np.array([r["slope_head"] for r in rows])
        v = [r["verdict"] for r in rows]
        print(f"\n  head {sh.mean():+.4f} +- {sh.std():.4f}   "
              f"residual {sr.mean():+.4f} +- {sr.std():.4f}   "
              f"reference {REFERENCE_SLOPE:+.4f}")
        print(f"  verdicts: base {v.count('base')}, head {v.count('head')}, "
              f"over {v.count('over')} of {len(v)}")
        print(f"  this base's residual slope brackets the -0.134 reference in "
              f"{sum(r['residual_matches_reference'] for r in rows)}/{len(rows)} models")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
