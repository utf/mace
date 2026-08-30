"""Build the E0 / Stage-A dataset: train the base branch on n = 0 frames only.

E0 asks whether the unpaired data contain any site-resolved signal about where the carrier
goes. The test is to fit a geometry-only base to every frame whose canonical counters are all
zero (pristine + V_Cl0), then look at what it cannot explain on V_Cl+ frames it has never
seen. If the two under-coordinated Pb carry clearly elevated force residuals, the signal is
there and joint training has been destroying it; if the map is flat, no architecture can
conjure it.

Four outputs:

  train.xyz          n = 0 frames from the training split. The correction branch vanishes
                     structurally at n = 0, so this trains the base and nothing else.
  valid.xyz          n = 0 validation frames, for checkpoint selection during that training.
  eval_qp1.xyz       every V_Cl+ frame, from both splits. None was trained on, since the base
                     only ever saw n = 0.
  eval_q0_null.xyz   held-out V_Cl0 frames -- the null control. The (+) residual map only
                     means something as an excess over this, which measures how badly the base
                     extrapolates to defective geometries with no carrier present at all.

The null frames are deliberately excluded from valid.xyz. If they took part in checkpoint
selection the base would be mildly tuned to them, deflating the null and inflating the excess
-- a bias pointing at exactly the conclusion E0 is meant to test.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ase.io import read, write


def counters(atoms):
    c = atoms.info.get("carrier_counts")
    if c is None:
        raise ValueError("frame has no carrier_counts")
    if isinstance(c, str):
        c = [int(x) for x in c.split()]
    return np.asarray(c, dtype=int)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=here / "dataset_pbe")
    ap.add_argument("--out", type=Path, default=here / "dataset_e0")
    ap.add_argument("--null-frames", type=int, default=60,
                    help="held-out V_Cl0 frames reserved for the null control")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_src = read(args.source / "train.xyz", ":")
    valid_src = read(args.source / "valid.xyz", ":")

    def partition(frames):
        neutral, charged = [], []
        for a in frames:
            (neutral if not counters(a).any() else charged).append(a)
        return neutral, charged

    tr_n0, tr_qp1 = partition(train_src)
    va_n0, va_qp1 = partition(valid_src)

    # Reserve part of the validation V_Cl0 pool as the null control, disjoint from anything
    # used for checkpoint selection. Pristine frames stay in validation: they are the least
    # informative about defect extrapolation and the most useful for keeping the base honest.
    va_vac = [a for a in va_n0 if len(a) != 80]
    va_pristine = [a for a in va_n0 if len(a) == 80]
    if len(va_vac) < args.null_frames:
        raise SystemExit(f"only {len(va_vac)} validation V_Cl0 frames, "
                         f"need {args.null_frames} for the null")
    order = rng.permutation(len(va_vac))
    null_idx = set(order[: args.null_frames].tolist())
    null = [a for k, a in enumerate(va_vac) if k in null_idx]
    va_keep = [a for k, a in enumerate(va_vac) if k not in null_idx]

    out_train = tr_n0
    out_valid = va_pristine + va_keep
    out_qp1 = tr_qp1 + va_qp1

    write(args.out / "train.xyz", out_train)
    write(args.out / "valid.xyz", out_valid)
    write(args.out / "eval_qp1.xyz", out_qp1)
    write(args.out / "eval_q0_null.xyz", null)
    shutil.copy(args.source / "band_edges.json", args.out / "band_edges.json")

    manifest = {
        "purpose": "E0 diagnostic / Stage A base training",
        "source": str(args.source),
        "null_split_seed": args.seed,
        "train_n0": len(out_train),
        "valid_n0": len(out_valid),
        "eval_qp1": len(out_qp1),
        "eval_q0_null": len(null),
        "note": ("Null frames are held out of valid.xyz so they take no part in checkpoint "
                 "selection; the (+) map is only meaningful as an excess over this null."),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for a in out_train + out_valid:
        assert not counters(a).any(), "n != 0 frame leaked into base training"
    for a in out_qp1:
        assert counters(a).any(), "n = 0 frame leaked into the charged evaluation set"

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
