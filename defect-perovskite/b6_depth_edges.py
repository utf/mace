#!/usr/bin/env python3
"""F9: did widening move the defect level, or the whole spectrum?

THE THING BEING SETTLED. F5's absolute level moved from +1.3..+2.1 eV to +3.58..+3.81 eV
between the two six-seed runs, and that was reported as the widened bound letting the on-site
correction go where it had been pinned. But `lambda_frontier` is an eigenvalue of H, and H
carries `eps0` -- a free per-species constant. A uniform shift of `eps0` moves every level
together and changes NO physical quantity: the transition level, the depth below the
conduction manifold, and the binding are all differences. So a +2 eV move of `lambda` is
compatible with nothing whatever having changed.

The observable that is not gauge-dependent is the DEPTH against the band edges, which is what
this measures on both sides:

    depth_from_VBM = lambda_frontier - VBM_pristine      (aligned)
    depth_from_CBM = CBM_pristine - lambda_frontier      (aligned)

ALIGNMENT, and why it is done by quantiles. The pristine cell and the charged defect cell are
different sizes with different compositions, so their spectra carry different absolute
offsets and different numbers of levels; subtracting one frontier from the other directly
would be subtracting two gauges. The two share a bulk-like valence manifold, so the pristine
spectrum is shifted by the median offset between the two spectra taken at MATCHED QUANTILES of
their occupied levels, over a window that excludes the frontier region where the defect state
lives. That is the standard deep-level alignment, expressed size-agnostically. The shift and
its spread across quantiles are both reported: a large spread means the manifolds do not
actually align and the depths that follow are not to be trusted.

REGIME TAGS ARE MANDATORY HERE. The "before" models were trained at gamma = 1 with
Fermi-Dirac smearing at 25 meV; the "after" models at gamma = 3 with Gaussian at 50 meV.
THREE things changed at once, so this is a regime-to-regime comparison and NOT a measurement
of gamma. It cannot be disentangled after the fact either: eigenvalues do not depend on the
smearing at fixed parameters, so there is no evaluation-side knob to turn -- the smearing
entered only through what training put in the weights. Each row prints its own regime.

F9's forecast, scored here: the move is a whole-spectrum offset and the depths change by
less than 0.2 eV. The decision tree attaches a STOP to the other branch -- a real depth
change larger than that, masked by gates that all passed, means the gate set is not
measuring what it claims and nothing should train until that is reported.
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
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS  # noqa: E402
from ta_band_edge import capture, load_frames, select, with_hole_counter  # noqa: E402

F9_THRESHOLD = 0.2               # eV; a depth move larger than this is not an offset
QUANTILES = np.linspace(0.05, 0.60, 23)     # occupied manifold, well below the frontier


def spectrum(model, batch, frames, ctx):
    internals, _ = capture(model, batch, ctx=ctx, frames=frames)
    lam = internals["lam"][0, 0]
    return torch.sort(lam[lam < 500.0]).values.float().cpu().numpy()


def occupied_count(frames):
    from mace.modules.defect_counting import VALENCE

    return sum(VALENCE[int(z)] for z in frames[0].get_atomic_numbers()) // 2


def align(defect_lam, n_def, pristine_lam, n_pri):
    """Offset that puts the pristine occupied manifold on the defect one, by quantile.

    Returns `(shift, spread)`. The spread is the interquartile range of the per-quantile
    offsets: if the two manifolds really are the same bands, it is small, and if it is not
    then no single shift aligns them and the depths below are meaningless.
    """
    a = defect_lam[:n_def]
    b = pristine_lam[:n_pri]
    if a.size < 20 or b.size < 20:
        return float("nan"), float("nan")
    qa = np.quantile(a, QUANTILES)
    qb = np.quantile(b, QUANTILES)
    off = qa - qb
    return float(np.median(off)), float(np.subtract(*np.percentile(off, [75, 25])))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="TAG=/path/glob pairs, e.g. 'gamma1=/home/alex/runs/s5_models'")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--n-pristine", type=int, default=4)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    charged = [a for a in select(load_frames(args.data), charged=True)
               if len(a) == CLEAN_NATOMS][:args.frames]
    all_pristine = select_pristine(ase_read(str(args.data), ":"), 64)
    biggest = max(len(a) for a in all_pristine)
    pristine = with_hole_counter([a for a in all_pristine
                                  if len(a) == biggest][:args.n_pristine])
    print(f"{len(charged)} charged {CLEAN_NATOMS}-atom frames; {len(pristine)} pristine "
          f"cells of {biggest} atoms (the common-delta_L size)", flush=True)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))

    payload = {"pristine_size": biggest, "n_charged": len(charged), "arms": {}}
    for spec in args.arms:
        tag, _, root = spec.partition("=")
        paths = sorted(Path(root).glob("*.model")) if Path(root).is_dir() else \
            sorted(Path(root).parent.glob(Path(root).name))
        rows = []
        for mp in paths:
            model = torch.load(mp, map_location=args.device,
                               weights_only=False).to(args.device).eval()
            cutoff = max(float(model.r_max),
                         float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
            ctx = ForwardContext.production(model, device=args.device,
                                            eps_inf=args.eps_inf)
            regime = dict(gamma=float(model.spectral.h.on_site_range),
                          width=float(model.spectral.t_el),
                          family=getattr(model.spectral, "smearing_family",
                                         "fermi (attribute absent)"))

            # ---------------------------------------------------------- pristine edges
            n_pri = occupied_count(pristine)
            pri_spec, vbm, cbm = [], [], []
            for b, fr in make_batches(pristine, z, cutoff, 1, args.device):
                lam = spectrum(model, b, fr, ctx)
                pri_spec.append(lam)
                vbm.append(float(lam[n_pri - 1]))
                cbm.append(float(lam[n_pri]))
            pri_mean = np.mean(pri_spec, axis=0)

            # ------------------------------------------------------------- defect side
            from mace.modules.defect_counting import VALENCE, changed_level_index

            n_def = occupied_count(charged)
            rec = []
            for b, fr in make_batches(charged, z, cutoff, 1, args.device):
                lam = spectrum(model, b, fr, ctx)
                n_total = sum(VALENCE[int(zz)] for zz in fr[0].get_atomic_numbers())
                k = changed_level_index(n_total,
                                        b.carrier_counts.reshape(-1).tolist())
                if not 0 <= k < lam.size:
                    continue
                shift, spread = align(lam, n_def, pri_mean, n_pri)
                v = float(np.mean(vbm)) + shift
                c = float(np.mean(cbm)) + shift
                rec.append(dict(lam=float(lam[k]), vbm=v, cbm=c,
                                from_vbm=float(lam[k]) - v,
                                from_cbm=c - float(lam[k]),
                                shift=shift, spread=spread))
            if not rec:
                continue
            row = dict(model=mp.name, regime=regime,
                       pristine_gap=float(np.mean(cbm) - np.mean(vbm)),
                       lam=float(np.mean([r["lam"] for r in rec])),
                       from_vbm=float(np.mean([r["from_vbm"] for r in rec])),
                       from_cbm=float(np.mean([r["from_cbm"] for r in rec])),
                       shift=float(np.mean([r["shift"] for r in rec])),
                       align_spread=float(np.mean([r["spread"] for r in rec])))
            rows.append(row)
            print(f"  [{tag}] {mp.name:20s} gamma {regime['gamma']:.1f} "
                  f"{regime['family'].split()[0]:8s} {regime['width']:.3f}  "
                  f"lambda {row['lam']:+7.3f}  depth: from VBM {row['from_vbm']:+6.3f}  "
                  f"from CBM {row['from_cbm']:+6.3f}  gap {row['pristine_gap']:.3f}  "
                  f"(align shift {row['shift']:+.3f}, iqr {row['align_spread']:.3f})",
                  flush=True)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        payload["arms"][tag] = rows
        args.out.write_text(json.dumps(payload, indent=2, default=float))

    args.out.write_text(json.dumps(payload, indent=2, default=float))
    tags = [t for t, v in payload["arms"].items() if v]
    print("\n=== per arm ===")
    stat = {}
    for t in tags:
        v = payload["arms"][t]
        f = lambda k: np.array([r[k] for r in v])  # noqa: E731
        stat[t] = {k: (f(k).mean(), f(k).std())
                   for k in ("lam", "from_vbm", "from_cbm", "pristine_gap", "shift",
                             "align_spread")}
        reg = v[0]["regime"]
        print(f"  {t:10s} n={len(v)}  gamma {reg['gamma']}  {reg['family']} "
              f"{reg['width']}")
        for k in ("lam", "from_vbm", "from_cbm", "pristine_gap", "align_spread"):
            print(f"      {k:14s} {stat[t][k][0]:+8.4f} +- {stat[t][k][1]:.4f} eV")
    if len(tags) == 2:
        a, b = tags
        d_lam = stat[b]["lam"][0] - stat[a]["lam"][0]
        d_vbm = stat[b]["from_vbm"][0] - stat[a]["from_vbm"][0]
        d_cbm = stat[b]["from_cbm"][0] - stat[a]["from_cbm"][0]
        held = max(abs(d_vbm), abs(d_cbm)) < F9_THRESHOLD
        print(f"\n=== F9: {a} -> {b} ===")
        print(f"  lambda_frontier moved {d_lam:+.3f} eV")
        print(f"  depth from VBM moved  {d_vbm:+.3f} eV")
        print(f"  depth from CBM moved  {d_cbm:+.3f} eV")
        print(f"  -> F9 {'HOLDS' if held else 'FAILS'} "
              f"(forecast: a whole-spectrum offset, depths within {F9_THRESHOLD} eV)")
        if not held:
            print("  DECISION-TREE STOP: a real depth change this size, with every gate "
                  "passing,\n  means the gate set is not measuring what it claims. Report "
                  "before anything trains.")
        print(f"\n  Both arms differ in gamma, smearing family AND width at once. This is a\n"
              f"  regime-to-regime comparison; it does not attribute the move to gamma.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
