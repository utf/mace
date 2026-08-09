#!/usr/bin/env python3
"""The reference floor the model must beat: a linear fit on defect-neighbour geometry.

The delta target is the vertical excitation energy at fixed geometry. A large part of it
is a smooth function of how the six atoms around the divacancy are arranged, so a plain
linear model on their pairwise distances already predicts much of it. That number is the
honest floor for MACEDefect's `RMSE_dE`: anything above it is not learning the physics,
it is learning less than ridge regression on 15 numbers.

Two things this script is careful about, both of which invalidate the comparison if got
wrong:

* it is fitted on the **training** split and scored on the **validation** split, the same
  split the model is scored on. A probe fitted on all data and compared against a model's
  held-out score is not like-for-like, and flatters the probe;
* the defect neighbours are found by coordination deficit, never by index, since atom
  ordering is not guaranteed to be consistent across frames.

    python linear_probe_floor.py --data-dir dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ase.io
import numpy as np

BOND = 2.3  # Si-C is 1.89 A, second shell ~3.1 A


def defect_neighbours(atoms) -> np.ndarray:
    """Indices of the under-coordinated atoms -- the divacancy's first shell."""
    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell())
    fractional = positions @ np.linalg.inv(cell)
    delta = fractional[:, None, :] - fractional[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ cell, axis=-1)
    np.fill_diagonal(distances, np.inf)
    degree = (distances < BOND).sum(axis=1)
    return np.flatnonzero(degree < degree.max())


def features(atoms) -> np.ndarray | None:
    """Sorted pairwise distances among the defect neighbours.

    Sorting makes the descriptor permutation invariant, which matters because the six
    atoms have no canonical order across frames.
    """
    index = defect_neighbours(atoms)
    if len(index) != 6:
        return None
    positions = atoms.get_positions()[index]
    cell = np.asarray(atoms.get_cell())
    fractional = positions @ np.linalg.inv(cell)
    delta = fractional[:, None, :] - fractional[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ cell, axis=-1)
    upper = distances[np.triu_indices(6, k=1)]  # 15 pairwise distances
    return np.sort(upper)


def collect(path: Path, gap: float):
    """Referenced delta targets and descriptors for the excited member of each pair."""
    frames = ase.io.read(str(path), ":")
    by_pair: dict = {}
    for atoms in frames:
        pair = atoms.info.get("pair_id")
        if pair is None:
            continue
        by_pair.setdefault(str(pair), []).append(atoms)

    rows, targets = [], []
    skipped = 0
    for members in by_pair.values():
        if len(members) != 2:
            continue
        counts = [int(np.asarray(a.info["carrier_counts"]).sum()) for a in members]
        ground = members[int(np.argmin(counts))]
        excited = members[int(np.argmax(counts))]
        if counts[0] == counts[1]:
            continue
        descriptor = features(excited)
        if descriptor is None:
            skipped += 1
            continue
        rows.append(descriptor)
        # Same referencing the loader applies: the difference keeps exactly one gap,
        # since ground is (1,0,0,1) and excited (1,1,0,2).
        targets.append(
            float(excited.info["REF_energy"]) - float(ground.info["REF_energy"]) - gap
        )
    return np.asarray(rows), np.asarray(targets), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument(
        "--ridge", type=float, default=1e-6, help="ridge penalty on standardised features"
    )
    args = parser.parse_args()

    import json

    summary = json.loads((args.data_dir / "dataset_summary.json").read_text())
    gap = float(summary["band_edges"]["effective_gap"])

    train_x, train_y, skipped_train = collect(args.data_dir / "train.xyz", gap)
    valid_x, valid_y, skipped_valid = collect(args.data_dir / "valid.xyz", gap)
    print(f"dataset: {args.data_dir}  (gap gauge {gap:.4f} eV)")
    print(f"  train pairs {len(train_y)} (skipped {skipped_train}), "
          f"valid pairs {len(valid_y)} (skipped {skipped_valid})")
    if len(train_y) < 10 or len(valid_y) < 3:
        raise SystemExit("not enough pairs to fit a meaningful probe")

    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.hstack([(train_x - mean) / scale, np.ones((len(train_x), 1))])
    gram = design.T @ design + args.ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ train_y)

    def score(x, y, label):
        d = np.hstack([(x - mean) / scale, np.ones((len(x), 1))])
        residual = y - d @ weights
        rmse = float(np.sqrt(np.mean(residual**2)))
        variance = float(np.var(y))
        r2 = 1.0 - float(np.mean(residual**2)) / variance if variance > 0 else float("nan")
        print(
            f"  {label:5s}: sigma {np.std(y) * 1000:7.2f} meV | "
            f"probe RMSE {rmse * 1000:7.2f} meV | R2 {r2:6.3f}"
        )
        return rmse

    print("\nlinear probe on 15 sorted defect-neighbour distances, fitted on TRAIN:")
    score(train_x, train_y, "train")
    valid_rmse = score(valid_x, valid_y, "valid")
    print(
        f"\nFLOOR for this split: MACEDefect RMSE_dE must beat "
        f"{valid_rmse * 1000:.2f} meV on the validation split to be doing better than "
        f"ridge regression\non 15 numbers. Compare only against a model scored on this "
        f"same split."
    )


if __name__ == "__main__":
    main()
