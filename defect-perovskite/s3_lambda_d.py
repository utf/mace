#!/usr/bin/env python3
"""F5: `corr(lambda_frontier, d_hub)` on the 159-atom charged subset. Positive expected.

WHAT IT ASKS. The carrier level in a chlorine vacancy is built from the two Pb dangling
orbitals that flank it. Pull those two apart and their coupling falls, so the bonding
combination must RISE. A head that has learned a real defect state reproduces that; a head
that has learned an energy offset attached to the vacancy cannot, because nothing in an
offset knows about `d`.

THE COUNTERS ARE A HOLE, MEASURED NOT ASSUMED: all 16 frames carry (0, 0, 1, 0), which
`spin_targets` reads as h_maj = 1, so `n_maj = ref - 1`. The level in question is the one the
carrier VACATED, not one it entered -- the same defect state either way, which is why the
argument above is unchanged, but the index is not: the naive `n_maj - 1` would have landed a
level below it and measured a valence state's d-dependence instead.

THIS SUPERSEDES THE STAGE-1 FIGURE. That one was measured on the V3 spectral head, whose
`lambda` was a six-state truncation of a different eigenproblem, and it was read in a single
optimiser regime. This is the counting head's own frontier level, over the whole spectrum,
on the clean-label subset -- and the number here is the one that stands.

`lambda_frontier` is the level whose occupation the counters CHANGED, index
`max(n_maj, n_maj_ref) - 1`, which is correct for an added and a removed electron alike.
Taking the gap instead would confound the level with the one above it.

The same 159-atom subset as F4, and for the same reason: the 79-atom energy targets carry
M1b's base-extrapolation slope, and pooling sizes re-imports it.
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
from s3_dehead_trend import CLEAN_NATOMS, fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import capture, load_frames, select  # noqa: E402


def frontier_level(model, batch, frames, ctx) -> float:
    """The highest occupied majority eigenvalue of one graph."""
    from mace.modules.defect_counting import VALENCE, spin_targets

    internals, _ = capture(model, batch, ctx=ctx, frames=frames)
    lam = internals["lam"][0, 0]
    lam = lam[lam < 500.0]                      # strip the padding sentinel
    n_total = sum(VALENCE[int(z)] for z in frames[0].get_atomic_numbers())
    counts = batch.carrier_counts.reshape(-1).tolist()
    n_maj, _ = spin_targets(n_total, counts)
    n_maj_ref = float((n_total + 1) // 2)
    # THE LEVEL WHOSE OCCUPATION CHANGED, which is not the same as the highest occupied one.
    # For an added electron (n_maj = ref + 1) that is index n_maj - 1, the level the carrier
    # went into. For a REMOVED electron (n_maj = ref - 1) the highest occupied level is one
    # below the defect state, and indexing it would measure the d-dependence of a valence
    # level instead -- a plausible flat correlation from the wrong quantity. `max` picks the
    # changed level under both conventions, and the counters are printed so which one the
    # data uses is on record rather than assumed.
    k = int(round(max(n_maj, n_maj_ref))) - 1
    if k < 0 or k >= lam.numel():
        return float("nan")
    return float(lam[k])


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == CLEAN_NATOMS]
    if len(frames) < 5:
        raise SystemExit(f"only {len(frames)} charged {CLEAN_NATOMS}-atom frames found")
    d = np.array([hub_separation(a) for a in frames])
    ok = np.isfinite(d)
    print(f"{len(frames)} charged {CLEAN_NATOMS}-atom frames, "
          f"d range {d[ok].min():.2f}-{d[ok].max():.2f} A", flush=True)

    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    # Provenance for the frontier index: which counter convention these frames actually use.
    seen_counters = sorted({tuple(int(c) for c in b.carrier_counts.reshape(-1).tolist())
                            for b, _ in make_batches(frames, z, 5.0, 1, "cpu")})
    print(f"  counters present: {seen_counters}", flush=True)
    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        # The same production context training used, so the Madelung shift and the graph
        # conventions are identical here and in the run that made this model. Reading a
        # spectrum through a different forward pass is how train and evaluate disagreed four
        # times in this programme.
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        lam = np.array([frontier_level(model, b, fr, ctx)
                        for b, fr in make_batches(frames, z, cutoff, 1, args.device)])
        good = ok & np.isfinite(lam)
        if good.sum() < 5:
            print(f"  {Path(mp).name:26s} only {int(good.sum())} usable frames", flush=True)
            continue
        slope, lo, hi, corr = fit_with_ci(d[good], lam[good])
        passed = bool(corr > 0)
        rows.append(dict(model=Path(mp).name, n=int(good.sum()), corr=corr, slope=slope,
                         slope_ci=[lo, hi], passed=passed,
                         lam_range=[float(lam[good].min()), float(lam[good].max())]))
        print(f"  {rows[-1]['model']:26s} corr {corr:+.3f}  slope {slope:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] eV/A  lambda {lam[good].min():+.3f} to "
              f"{lam[good].max():+.3f} eV  {'PASS' if passed else 'FAIL'}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        c = np.array([r["corr"] for r in rows])
        print(f"\n  corr(lambda_frontier, d) {c.mean():+.3f} +- {c.std():.3f} over "
              f"{len(rows)} seeds; positive in {int((c > 0).sum())}/{len(rows)}")
        print("  F5 asks for the SIGN. A negative correlation would mean the level falls as "
              "the two Pb dangling orbitals separate, which no two-centre picture produces.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
