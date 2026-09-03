#!/usr/bin/env python3
"""Section 1 of the Stage A' spec: a label-free extrapolation indicator per frame (F16).

    ood_E(frame) = std over fold bases of E_base / N_atoms            [eV/atom]
    ood_F(frame) = rms over atoms of std over fold bases of F_base     [eV/A]
    s_E          = 95th percentile of ood_E over NEUTRAL frames
    w_E(frame)   = min(1, (s_E / ood_E)^2)

WHY DISAGREEMENT BETWEEN FOLD BASES. Four bases trained on disjoint quarters of the neutral
set agree where the training data is dense and disagree where each is extrapolating -- and
the charged 79-atom frames at long d are exactly where b1 found the frozen base
extrapolating (+0.132 eV/A of carrier-free residual slope, +0.364 with a carrier). No label
enters: the indicator is a property of four networks and a geometry.

WHAT IS REPORTED, per the spec. `w_E` against d for the charged 79-atom frames, and the
charged 79-atom residual slope (against the PRODUCTION base) restricted to frames with
`w_E > 0.5`, beside the full-range and window slopes b1 measured. F16 forecasts that the
+0.36 slope is carried by the low-w_E frames and that the w_E > 0.5 frames reproduce the
window value (+0.081 [-0.111, +0.272]).

OUTPUT. A JSON sidecar keyed by `frame_key` (the content hash the base cache uses) holding
ood_E, ood_F, w_E, natoms, charged and d per frame -- the dataset metadata the Stage-B
objective reads to weight the charged 79-atom energies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, evaluate  # noqa: E402
from r1_matrix import adopt_model_dtype  # noqa: E402
from s3_dehead_trend import fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import counts_of  # noqa: E402

from mace.modules.defect_cache import frame_key  # noqa: E402


def load(path, device):
    model = torch.load(path, map_location=device, weights_only=False).to(device).eval()
    adopt_model_dtype(model)
    return model


def per_frame(model, frames, device, batch_size):
    """(E_base [n_frames], F_base list) from one base, in frame order."""
    energies, forces = [], []
    for k in range(0, len(frames), batch_size):
        chunk = frames[k: k + batch_size]
        out = evaluate(model, chunk, device=device)
        energies.extend([float(e) for e in out["base_energy"]])
        forces.extend([np.asarray(f) for f in out["base_forces"]])
    return np.array(energies), forces


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="+", type=Path, required=True,
                    help="the fold bases, e.g. ~/runs/cf_base_f{0..3}/cf_base_f{0..3}.model")
    ap.add_argument("--production", type=Path, required=True,
                    help="the production base the residual slopes are measured against")
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[here / "dataset_pbe" / "train.xyz",
                             here / "dataset_pbe" / "valid.xyz"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = []
    for p in args.data:
        frames.extend(read(str(p), index=":"))
    keys = [frame_key(a.get_atomic_numbers(), a.get_positions(), a.get_cell()[:])
            for a in frames]
    charged = np.array([(counts_of(a) is not None and sum(counts_of(a)) != 0)
                        for a in frames])
    natoms = np.array([len(a) for a in frames])
    d = np.array([hub_separation(a) for a in frames])
    print(f"{len(frames)} frames: {int(charged.sum())} charged, "
          f"{int((~charged).sum())} neutral; sizes {sorted(set(natoms.tolist()))}",
          flush=True)

    # --- the four fold bases -------------------------------------------------------
    e_folds, f_folds = [], []
    for fp in args.folds:
        model = load(fp, args.device)
        e, f = per_frame(model, frames, args.device, args.batch_size)
        e_folds.append(e / natoms)
        f_folds.append(f)
        print(f"  {fp.name}: scored", flush=True)
        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    e_folds = np.stack(e_folds)                                  # [4, n_frames]
    ood_e = e_folds.std(axis=0, ddof=0)
    ood_f = np.array([
        np.sqrt(np.mean(np.stack([f_folds[j][i] for j in range(len(f_folds))]).std(
            axis=0, ddof=0) ** 2))
        for i in range(len(frames))])
    s_e = float(np.percentile(ood_e[~charged], args.percentile))
    w_e = np.minimum(1.0, (s_e / np.maximum(ood_e, 1e-12)) ** 2)

    # --- the production base: residuals for the slopes --------------------------------
    prod = load(args.production, args.device)
    e_prod, _ = per_frame(prod, frames, args.device, args.batch_size)
    labels = np.array([float(a.info["REF_energy"]) for a in frames])
    resid = labels - e_prod

    rows = {}
    for i, a in enumerate(frames):
        rows[str(keys[i])] = dict(
            id=str(a.info.get("id", i)), natoms=int(natoms[i]), charged=bool(charged[i]),
            d=(None if not np.isfinite(d[i]) else float(d[i])),
            ood_E=float(ood_e[i]), ood_F=float(ood_f[i]), w_E=float(w_e[i]),
            resid_E=float(resid[i]))

    # --- report ----------------------------------------------------------------------
    print(f"\n=== indicator ===")
    print(f"  s_E (p{args.percentile:.0f} of ood_E over neutral frames) = {s_e:.5f} eV/atom")
    for label, mask in (("neutral 79/80", (~charged) & (natoms < 100)),
                        ("neutral 159", (~charged) & (natoms >= 100)),
                        ("charged 79", charged & (natoms < 100)),
                        ("charged 159", charged & (natoms >= 100))):
        if mask.sum() == 0:
            continue
        print(f"  {label:14s} n={int(mask.sum()):5d}  ood_E median {np.median(ood_e[mask]):.5f} "
              f"p95 {np.percentile(ood_e[mask], 95):.5f}  ood_F median "
              f"{np.median(ood_f[mask]):.4f}  w_E median {np.median(w_e[mask]):.3f}  "
              f"w_E>0.5: {float((w_e[mask] > 0.5).mean()):.1%}")

    print(f"\n=== w_E against d, charged 79-atom frames ===")
    c79 = charged & (natoms < 100) & np.isfinite(d)
    for lo, hi in ((3.5, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0)):
        m = c79 & (d >= lo) & (d < hi)
        if m.sum():
            print(f"  d {lo:.1f}-{hi:.1f}  n={int(m.sum()):4d}  w_E median {np.median(w_e[m]):.3f}  "
                  f"mean {w_e[m].mean():.3f}  >0.5: {float((w_e[m] > 0.5).mean()):.1%}")

    print(f"\n=== charged 79-atom residual slope d(E_label - E_base)/dd, production base ===")
    slopes = {}
    for label, m in (("full range", c79), ("w_E > 0.5", c79 & (w_e > 0.5)),
                     ("w_E <= 0.5", c79 & (w_e <= 0.5))):
        if m.sum() < 3:
            print(f"  {label:12s} n={int(m.sum())}: too few frames")
            continue
        s, lo, hi, corr = fit_with_ci(d[m], resid[m])
        slopes[label] = dict(n=int(m.sum()), slope=s, ci=[lo, hi], corr=corr)
        print(f"  {label:12s} n={int(m.sum()):4d}  slope {s:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"corr {corr:+.3f}")
    c159 = charged & (natoms >= 100) & np.isfinite(d)
    if c159.sum() >= 3:
        s, lo, hi, corr = fit_with_ci(d[c159], resid[c159])
        slopes["charged 159"] = dict(n=int(c159.sum()), slope=s, ci=[lo, hi], corr=corr)
        print(f"  charged 159  n={int(c159.sum()):4d}  slope {s:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"corr {corr:+.3f}   (the F4 reference against this base)")
    full = slopes.get("full range", {}).get("slope")
    high = slopes.get("w_E > 0.5", {})
    if full is not None and high:
        window = (-0.1108, 0.2720)      # b1's neutral-dense window interval, 79 atoms
        inside = window[0] <= high["slope"] <= window[1]
        print(f"\n  F16: full-range {full:+.4f}; w_E > 0.5 gives {high['slope']:+.4f} "
              f"[{high['ci'][0]:+.4f}, {high['ci'][1]:+.4f}] against the window value "
              f"+0.0806 [{window[0]:+.4f}, {window[1]:+.4f}] -> "
              f"{'HOLDS' if inside and abs(high['slope']) < abs(full) else 'FAILS'}")

    payload = dict(s_E=s_e, percentile=args.percentile, folds=[str(p) for p in args.folds],
                   production=str(args.production), frames=rows, slopes=slopes)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
