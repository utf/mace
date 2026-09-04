#!/usr/bin/env python3
"""Step 2a: `dE_head/dd` on the 16 large frames under three smearing settings.

The decision is already taken -- Gaussian sigma = 0.05 eV, because that is the label
pipeline's own convention (doped, ISMEAR = 0). This is verification and sensitivity, not a
search: it says how much of F4's slope depends on a choice that is now fixed by the labels
rather than by us.

  Gaussian 0.05   the labels' convention, and the head's
  Fermi-Dirac 25 meV   what the head used before, a k_B * 300 K coincidence
  Fermi-Dirac 5 meV    a near-sharp fill, to bound how much the smearing can matter at all

F2, on record before the run: Gaussian 0.05 and FD 25 meV differ by < 5% (they are
tail-equivalent -- verified on a gapped spectrum to 1e-6 in the unit suite); FD 5 meV raises
the slope 10-25%. Reported as sensitivity only. The smearing is no longer a candidate fix for
F4's amplitude, because the labels themselves are smeared at 0.05.
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
from mace.modules.defect_counting import use_smearing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS, REFERENCE_SLOPE, fit_with_ci, hub_separation
from ta_band_edge import load_frames, select  # noqa: E402

SETTINGS = (("gaussian", 0.05), ("fermi", 0.025), ("fermi", 0.005))


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
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"{len(frames)} charged {CLEAN_NATOMS}-atom frames", flush=True)

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        batches = make_batches(frames, z, cutoff, 1, args.device)
        for family, width in SETTINGS:
            # The head reads its family from its own attribute at forward entry, so both are
            # set: the module switch alone would be overwritten by the head itself.
            prev_family = getattr(model.spectral, "smearing_family", "gaussian")
            prev_t = float(model.spectral.t_el)
            model.spectral.smearing_family, model.spectral.t_el = family, float(width)
            previous = use_smearing(family, width)
            try:
                de = []
                for b, fr in batches:
                    with torch.no_grad():
                        out = model(ctx.forward_dict(b, fr, requires_grad=False),
                                    training=False, compute_force=False)
                    de.append(float(out["delta_sr_energy"].reshape(-1)[0]))
            finally:
                use_smearing(*previous)
                model.spectral.smearing_family, model.spectral.t_el = prev_family, prev_t
            de = np.array(de)
            good = ok & np.isfinite(de)
            if good.sum() < 5:
                continue
            slope, lo, hi, corr = fit_with_ci(d[good], de[good])
            rows.append(dict(model=Path(mp).name, family=family, width=width,
                             slope=slope, ci=[lo, hi], corr=corr))
            print(f"  {Path(mp).name:20s} {family:8s} {1000 * width:5.1f} meV  "
                  f"slope {slope:+.4f} [{lo:+.4f}, {hi:+.4f}]  corr {corr:+.3f}",
                  flush=True)
            args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    base = None
    for family, width in SETTINGS:
        sel = [r["slope"] for r in rows if r["family"] == family and r["width"] == width]
        if not sel:
            continue
        m = float(np.mean(sel))
        if base is None:
            base = m
        print(f"\n  {family} {1000 * width:5.1f} meV: slope {m:+.4f} +- "
              f"{float(np.std(sel)):.4f}   ({100 * (m / base - 1):+.1f}% vs the labels' "
              f"convention; reference {REFERENCE_SLOPE:+.3f})")
    print("  F2: < 5% between Gaussian 0.05 and FD 25 meV, 10-25% at FD 5 meV. Sensitivity "
          "only -- the labels are smeared at 0.05, so this is not a candidate fix.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
