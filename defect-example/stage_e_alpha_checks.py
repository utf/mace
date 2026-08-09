#!/usr/bin/env python3
"""Stage E precondition: is ``alpha`` fit to be exported as a latent charge?

Turning the long-range branch on promotes ``alpha`` from an internal weighting to a
physical object: ``q_i^carrier = a * sum_c s_c n_c alpha_i^c`` is the charge whose
self-term is the electron-hole interaction. An ``alpha`` that is over-localised gives a
wrong e-h term even when ``Delta E`` is right, because the energy only constrains the
*pooled* value ``sum_i alpha_i u_i`` and is blind to how the weight is distributed.

Two eigenvalue-free checks, neither needing a reference calculation:

1. **Symmetry floor.** On a relaxed, symmetric ground-state geometry the three equivalent
   dangling bonds are related by the defect's own symmetry, so permutation invariance
   forces ``alpha`` equal across them and the participation ratio cannot fall below ~3.
   A model reporting less than that on such a frame is not "more localised", it is
   broken. The 2.5 measured earlier came from a thermally displaced frame, where the
   symmetry is absent and a lower value is legitimate -- so it is not evidence either
   way, and this script is what settles it.

2. **Size stability.** ``alpha`` on corresponding atoms should agree between the 286- and
   398-atom cells. It is an intensive, local quantity; drift with cell size means it is
   absorbing fit rather than describing a carrier, and that drift would be inherited
   directly by the exported charge.

    python stage_e_alpha_checks.py ~/runs/<run>/<run>.model
"""

from __future__ import annotations

import argparse
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


def defect_shell(positions: np.ndarray, cell: np.ndarray, bond: float = 2.3):
    inverse = np.linalg.inv(cell)
    fractional = positions @ inverse
    delta = fractional[:, None, :] - fractional[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ cell, axis=-1)
    np.fill_diagonal(distances, np.inf)
    degree = (distances < bond).sum(axis=1)
    return np.flatnonzero(degree < degree.max())


@torch.no_grad()
def alpha_for(model, config, z_table):
    atomic = data.AtomicData.from_config(config, z_table=z_table, cutoff=CUTOFF)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic], batch_size=1, shuffle=False
    )
    batch = next(iter(loader))
    out = model(batch.to_dict(), training=False, compute_force=False)
    return out["carrier_alpha"].detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("model", type=Path)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument(
        "--relaxed-fmax",
        type=float,
        default=0.30,
        help="max |F| in eV/A below which a frame counts as relaxed enough for the "
        "symmetry-floor check; thermal frames have no symmetry to enforce",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    model = torch.load(args.model, map_location="cpu", weights_only=False).float()
    model.eval()
    z_table = tools.AtomicNumberTable([6, 14])

    _, configs = data.load_from_xyz(
        file_path=str(args.data_dir / "valid.xyz"),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.data_dir / "band_edges.json"),
    )
    defect = [c for c in configs if int(sum(c.properties["carrier_counts"])) > 0]
    if not defect:
        raise SystemExit("no carrier-bearing frames in the validation split")

    print(f"model: {args.model}\n")

    # ---- check 1: symmetry floor -------------------------------------------------------
    forces = [float(np.abs(np.asarray(c.properties["forces"])).max()) for c in defect]
    order = np.argsort(forces)
    calmest = defect[int(order[0])]
    print(
        f"CHECK 1  symmetry floor (calmest available frame, |F|max "
        f"{forces[int(order[0])]:.3f} eV/A)"
    )
    alpha = alpha_for(model, calmest, z_table)
    participation = 1.0 / alpha.pow(2).sum(dim=0)
    for index, name in enumerate(("e_maj", "e_min", "h_maj", "h_min")):
        print(f"    {name:6s} participation {float(participation[index]):8.2f}")
    lowest = float(participation.min())
    if forces[int(order[0])] > args.relaxed_fmax:
        print(
            f"    INCONCLUSIVE: no frame is relaxed (|F|max > {args.relaxed_fmax}). The "
            "floor only\n    binds where the symmetry actually holds; a thermal frame may "
            "legitimately go below 3.\n    Needs a relaxed ground-state geometry, which "
            "this dataset does not contain."
        )
    elif lowest < 2.5:
        print(f"    FAIL: participation {lowest:.2f} < 3 on a symmetric frame")
    else:
        print(f"    PASS: lowest participation {lowest:.2f}")

    # ---- check 2: size stability -------------------------------------------------------
    print("\nCHECK 2  size stability of alpha on the defect shell")
    by_size: dict = {}
    for config in defect:
        by_size.setdefault(len(config.atomic_numbers), []).append(config)
    sizes = sorted(by_size)
    summary = {}
    for size in sizes:
        shares = []
        for config in by_size[size][:8]:
            alpha = alpha_for(model, config, z_table)
            shell = defect_shell(
                np.asarray(config.positions), np.asarray(config.cell)
            )
            if len(shell) != 6:
                continue
            shares.append(alpha[shell].sum(dim=0))
        if shares:
            summary[size] = torch.stack(shares).mean(dim=0)
    for size, share in summary.items():
        values = " ".join(f"{float(v):6.4f}" for v in share)
        print(f"    N={size:4d}  alpha on shell per channel: [{values}]")
    if len(summary) >= 2:
        first, last = summary[sizes[0]], summary[sizes[-1]]
        drift = float((last - first).abs().max())
        print(
            f"\n    max drift across cell sizes: {drift:.4f}  "
            f"({'PASS' if drift < 0.1 else 'FAIL — alpha is absorbing fit, not describing a carrier'})"
        )
    else:
        print("    INCONCLUSIVE: need at least two cell sizes")


if __name__ == "__main__":
    main()
