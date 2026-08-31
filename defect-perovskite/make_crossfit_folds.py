"""Build the k-fold neutral split for the T2 cross-fit null.

The 60-frame null control could not populate the d(Pb-Pb) window the charged frames occupy:
its median is 4.81 A against the charged 5.49, and it tops out at 5.77 A where the neutral
training set reaches 6.54. So T2 could not run, and E0's excess rested on the same 60 frames.

A single large tail-stratified hold-out would fix the null but starve the base of exactly the
long-d tail it has to model. Cross-fitting avoids the trade: split the neutral DEFECTIVE
frames into k folds, train one diagnostic base per fold on the other k-1, and score every
neutral frame with the base that never saw it. That gives out-of-fold generalisation
residuals across the whole range, with no fold-base losing tail density.

Pristine frames go into every fold's training set: they carry no vacancy, so they are not part
of the d(Pb-Pb) question and removing them would only weaken each base.

The production Stage-A base is untouched. These bases are diagnostics for T2 and the E0
recheck only.
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
    if isinstance(c, str):
        c = [int(x) for x in c.split()]
    return np.asarray(c, dtype=int)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=here / "dataset_pbe")
    ap.add_argument("--out", type=Path, default=here / "dataset_cf")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = read(args.source / "train.xyz", ":") + read(args.source / "valid.xyz", ":")
    neutral = [a for a in frames if not counters(a).any()]
    charged = [a for a in frames if counters(a).any()]
    pristine = [a for a in neutral if len(a) == 80]
    defective = [a for a in neutral if len(a) != 80]

    order = rng.permutation(len(defective))
    fold_of = {int(idx): k % args.folds for k, idx in enumerate(order)}

    manifest = {"folds": args.folds, "seed": args.seed,
                "pristine": len(pristine), "neutral_defective": len(defective),
                "charged": len(charged), "per_fold": {}}

    for k in range(args.folds):
        held = [a for i, a in enumerate(defective) if fold_of[i] == k]
        rest = [a for i, a in enumerate(defective) if fold_of[i] != k]
        # Pristine in every fold: no vacancy, so irrelevant to the d(Pb-Pb) question, and
        # dropping them would only make each diagnostic base worse.
        train = pristine + rest
        d = args.out / f"fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        # A small validation slice from the same training pool, for checkpoint selection
        # only; the held-out fold must never influence the base that will score it.
        cut = max(20, len(train) // 10)
        write(d / "train.xyz", train[cut:])
        write(d / "valid.xyz", train[:cut])
        write(d / "null_oof.xyz", held)
        shutil.copy(args.source / "band_edges.json", d / "band_edges.json")
        manifest["per_fold"][k] = {"train": len(train) - cut, "valid": cut,
                                   "null_oof": len(held)}

    write(args.out / "eval_qp1.xyz", charged)
    manifest["eval_qp1"] = len(charged)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
