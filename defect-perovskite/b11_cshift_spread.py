#!/usr/bin/env python3
"""Why the two drivers disagree about the c-shift by a factor of four.

THE OBSERVATION. On the same dataset and the same base, the harness calibrates the head's
energy zero to +8.70 eV and the production trainer to +36.54 eV. Both compute the median of
`(E_label - E_base - E_head) / Delta_n` over charged frames; they differ only in WHICH charged
frames. The harness takes `select(...)[:48]` -- the first forty-eight in file order, which is
trajectory order, so a few consecutive snapshots of one trajectory. The trainer takes all 944
in the training set.

WHY IT MATTERS. If that ratio were tight across the dataset, the two would agree and the slice
would be harmless. If it is broad, then the harness's c-shift was never "the median energy
mismatch" -- it was the median over an arbitrary slice, and every Stage-3 run in this programme
was initialised from one. `c_shift` is trainable, so a run can recover; what cannot be recovered
is the claim that the initialisation was a property of the data rather than of the file.

WHAT THIS MEASURES. The per-frame ratio on a stratified sample spanning the whole file, and
separately on the first forty-eight, with the two medians and the spread. Cell size is reported
beside it because a size-dependent offset would mean one scalar cannot serve both sizes at
once -- which is what the per-(charge, size) reference constants exist to prevent, and would be
a second finding rather than the same one.

Base only, head excluded: at initialisation `E_head` is small, and including it would make this
depend on which head happened to be built.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, evaluate  # noqa: E402
from ta_band_edge import counts_of, load_frames, select  # noqa: E402

HARNESS_SLICE = 48


def delta_n(atoms) -> float:
    c = counts_of(atoms)
    if c is None:
        return 0.0
    return float(c[0] + c[1] - c[2] - c[3])


def ratios(model, frames, z_table, cutoff, device, batch_size):
    _, base_e = evaluate(model, frames, z_table, cutoff, device, batch_size=batch_size)
    out = []
    for k, a in enumerate(frames):
        lab = a.info.get("REF_energy", a.info.get("energy"))
        dn = delta_n(a)
        if lab is None or dn == 0:
            continue
        out.append(dict(index=k, natoms=len(a),
                        ratio=(float(lab) - float(base_e[k])) / dn))
    return out


def summarise(name, rows):
    v = np.array([r["ratio"] for r in rows], dtype=float)
    if not v.size:
        return {}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"  {name:26s} n {v.size:4d}   median {np.median(v):+9.3f}   "
          f"IQR {q[1]:+8.3f}..{q[3]:+8.3f}   5-95% {q[0]:+8.3f}..{q[4]:+8.3f}")
    return dict(n=int(v.size), median=float(np.median(v)),
                p5=float(q[0]), p25=float(q[1]), p75=float(q[3]), p95=float(q[4]),
                std=float(v.std()))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path,
                    default=Path.home() / "runs" / "e0_base_s1" / "e0_base_s1.model")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--sample", type=int, default=120,
                    help="stratified frames spanning the whole file")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    charged = select(load_frames(args.data), charged=True)
    print(f"{len(charged)} charged frames in {args.data.name}", flush=True)

    first = charged[:HARNESS_SLICE]
    # Stratified by POSITION IN THE FILE, which is what the slice is a slice of. A random
    # sample would answer a different question.
    idx = np.linspace(0, len(charged) - 1, args.sample).astype(int)
    spread = [charged[i] for i in sorted(set(idx.tolist()))]

    model = torch.load(args.base, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    print(f"  base {args.base.name}; the harness slice is the first {HARNESS_SLICE} frames "
          f"(mean {np.mean([len(a) for a in first]):.1f} atoms), the spread sample is "
          f"{len(spread)} frames across the file", flush=True)

    rows_first = ratios(model, first, z, args.cutoff, args.device, args.batch_size)
    rows_spread = ratios(model, spread, z, args.cutoff, args.device, args.batch_size)

    print("\n=== (E_label - E_base) / Delta_n, eV ===")
    a = summarise("first 48 (the harness)", rows_first)
    b = summarise("across the file", rows_spread)
    by_size = {}
    for n in sorted({r["natoms"] for r in rows_spread}):
        by_size[str(n)] = summarise(f"  of which {n} atoms",
                                    [r for r in rows_spread if r["natoms"] == n])

    payload = dict(first=a, spread=b, by_size=by_size,
                   rows_first=rows_first, rows_spread=rows_spread)
    args.out.write_text(json.dumps(payload, indent=2, default=float))

    if a and b:
        print(f"\n  The harness calibrated Stage 3 to {a['median']:+.3f} eV; the whole-set "
              f"median is {b['median']:+.3f}.")
        wide = abs(b["p95"] - b["p5"]) > abs(b["median"]) * 0.5
        print(f"  The ratio is {'BROAD' if wide else 'tight'} across the file "
              f"(5-95% span {abs(b['p95'] - b['p5']):.1f} eV), so the slice "
              f"{'chose the value' if wide else 'was harmless'}.")
        if len(by_size) > 1:
            meds = [v["median"] for v in by_size.values() if v]
            gap = max(meds) - min(meds)
            print(f"  Between cell sizes the medians differ by {gap:.2f} eV. A size-dependent "
                  f"offset\n  would mean one scalar cannot serve both sizes -- which is what "
                  f"the per-(charge, size)\n  reference constants exist to prevent, and would "
                  f"be a SECOND finding, not this one.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
