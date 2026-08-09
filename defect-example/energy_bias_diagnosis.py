#!/usr/bin/env python3
"""Is the energy residual a per-atom constant, a per-cell constant, or neither?

The parity plot shows every subset sitting above its parity line by roughly the same
per-atom amount. That is consistent with two very different faults, and forces cannot
tell them apart because a constant is invisible to a derivative:

* a **per-atom** constant -- the energy reference (``E0``/``atomic_inter_shift``) is off,
  so the residual per atom is flat in cell size and the residual per cell grows like N;
* a **per-cell** constant -- something localised is mis-sized (the defect correction, a
  surface-like term), so the residual per cell is flat and the residual per atom decays
  like 1/N.

Fitting ``residual_per_cell = a * N + b`` separates them: ``a`` is the per-atom part in
eV/atom, ``b`` the per-cell part in eV. Reported per config_type, since ``ideal`` frames
carry ``n = 0`` and so exercise the base model alone.

    python energy_bias_diagnosis.py --model ~/runs/b_128ch_L0_s1/b_128ch_L0_s1.model
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mace import data, tools
from mace.data.defects import load_band_edges
from mace.tools import torch_geometric

CUTOFF = 4.0


def keyspec() -> data.KeySpecification:
    spec = data.KeySpecification()
    spec.info_keys.update(
        {
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "multiplicity": "multiplicity",
            "host": "host",
            "pair_id": "pair_id",
            "stress": "REF_stress",
            "head": "head",
        }
    )
    spec.arrays_keys.update({"forces": "REF_forces"})
    return spec


@torch.no_grad()
def residuals(model, configs, z_table, device, batch_size=4):
    """Per-frame (natoms, residual in eV over the whole cell)."""
    dataset = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=CUTOFF) for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )
    counts, deltas = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.to_dict(), training=False, compute_force=False)
        n = (batch.ptr[1:] - batch.ptr[:-1]).cpu()
        counts.append(n)
        deltas.append((out["energy"].detach() - batch.energy.detach()).cpu())
    return torch.cat(counts).numpy(), torch.cat(deltas).numpy()


def split_fit(natoms: np.ndarray, residual: np.ndarray):
    """Least squares residual_per_cell = a*N + b. Returns (a in meV/atom, b in eV, R^2)."""
    if len(np.unique(natoms)) < 2:
        return float("nan"), float("nan"), float("nan")
    design = np.stack([natoms.astype(float), np.ones_like(natoms, dtype=float)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, residual, rcond=None)
    fitted = design @ np.array([a, b])
    variance = float(np.var(residual))
    r2 = 1.0 - float(np.mean((residual - fitted) ** 2)) / variance if variance > 0 else float("nan")
    return a * 1e3, b, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model = model.float()
    model.eval()
    z_table = tools.AtomicNumberTable([6, 14])

    print(f"model: {args.model.name}")
    for key in ("atomic_inter_shift", "atomic_inter_scale"):
        if hasattr(model, key):
            print(f"  {key}: {getattr(model, key).detach().cpu().numpy()}")
    if hasattr(model, "atomic_energies_fn"):
        e0 = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
        print(f"  E0s (Z={z_table.zs}): {e0}")

    for split in args.splits:
        _, configs = data.load_from_xyz(
            file_path=str(args.data_dir / f"{split}.xyz"),
            key_specification=keyspec(),
            band_edges=load_band_edges(args.data_dir / "band_edges.json"),
        )
        natoms, residual = residuals(model, configs, z_table, device)
        types = np.array([str(c.config_type) for c in configs])

        print(f"\n=== {split}: {len(configs)} frames ===")
        print(f"{'subset':14s} {'n':>4s} {'meV/atom':>9s} {'per-cell a':>11s} "
              f"{'per-cell b':>11s} {'R2':>6s}")
        groups = defaultdict(list)
        for index, t in enumerate(types):
            groups[t].append(index)
        groups["ALL"] = list(range(len(configs)))
        for name, indices in groups.items():
            idx = np.asarray(indices)
            per_atom = residual[idx] / natoms[idx]
            a, b, r2 = split_fit(natoms[idx], residual[idx])
            print(f"{name:14s} {len(idx):4d} {per_atom.mean()*1e3:+9.2f} "
                  f"{a:+11.2f} {b:+11.4f} {r2:6.3f}")

        # A per-atom constant means the mean residual per atom does not move with N.
        print("\n  mean residual per atom (meV/atom) by cell size:")
        for size in sorted(np.unique(natoms)):
            mask = natoms == size
            per_atom = residual[mask] / natoms[mask]
            print(f"    N={int(size):4d}  n={int(mask.sum()):4d}  "
                  f"{per_atom.mean()*1e3:+8.2f}   (per cell {residual[mask].mean():+8.4f} eV)")


if __name__ == "__main__":
    main()
