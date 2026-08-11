#!/usr/bin/env python3
"""Where does the live attention channel actually sit? Audited, not inferred.

The previous read ("uniform over the Cs sublattice") rested on two things that were never
checked: the participation ratio happened to equal the Cs count, and only Pb was separated
from "the rest", so Cs was never distinguished from Cl. It also only ever looked at tiled
supercells built from ONE chosen site of an unrelaxed thermal snapshot -- never at a real
training frame, and never at more than one defect environment. This script fixes all of
that:

* alpha summed **per species**, so Cs and Cl are separated;
* the top individual alpha values with species and distance from the vacancy;
* **real V_Cl+ frames from the training and validation sets**, which is the distribution the
  model was actually fitted on and which contains many inequivalent Cl environments;
* several tiled supercell sites, so a single site cannot stand in for the whole.

The short-range branch is accurate and its pooled <u> is size-stable, so any claim that the
attention is badly delocalised has to survive that fact rather than be asserted past it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst

LIVE = 2  # h_maj: V_Cl+ carries counts (0, 0, 1, 0)
NAMES = ("e_maj", "e_min", "h_maj", "h_min")


def evaluate(model, atoms, z_table, mace_data, torch_geometric):
    config = mace_data.config_from_atoms(
        atoms,
        key_specification=mace_data.KeySpecification(
            info_keys={"carrier_counts": "carrier_counts", "host": "host",
                       "multiplicity": "multiplicity",
                       "m_s_ref_doubled": "m_s_ref_doubled"},
            arrays_keys={},
        ),
    )
    mace_data.canonicalise_config_counters(config)
    atomic = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=5.0)
    batch = next(iter(torch_geometric.dataloader.DataLoader(
        [atomic], batch_size=1, shuffle=False))).to_dict()
    with torch.no_grad():
        out = model(batch, training=False, compute_force=False)
    return out, batch


def report(alpha, symbols, distance, label):
    """One line per species plus the top individual weights."""
    participation = 1.0 / float((alpha ** 2).sum())
    parts = []
    for species in ("Cs", "Pb", "Cl"):
        mask = symbols == species
        parts.append(f"{species} {alpha[mask].sum():6.3f}/{mask.sum():<4d}")
    print(f"  {label:28s} partic {participation:8.2f}  " + "  ".join(parts))
    order = np.argsort(-alpha)[:5]
    top = "  ".join(
        f"{symbols[i]}@{distance[i]:.1f}A:{alpha[i]:.3f}" if distance is not None
        else f"{symbols[i]}:{alpha[i]:.3f}" for i in order)
    print(f"  {'':28s} top5: {top}")
    return participation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--repeats", nargs="+", default=["2,2,2", "3,3,2"])
    parser.add_argument("--sites", type=int, default=3)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from ase.io import read
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location="cpu", weights_only=False).double().eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"model {args.model.name}\n")

    # ---- 1. Real training frames: the distribution the model was fitted on ----------
    print("REAL V_Cl+ FRAMES from the dataset (79 atoms, many inequivalent Cl sites)")
    print("A vacancy is present but its position is not recorded, so distances are")
    print("omitted; the species split is the diagnostic.\n")
    for name in ("train.xyz", "valid.xyz"):
        path = args.data_dir / name
        if not path.exists():
            continue
        frames = [a for a in read(path, ":")
                  if tuple(int(v) for v in a.info.get("carrier_counts", [])) == (0, 0, 1, 0)
                  and len(a) == 79]
        print(f"{name}: {len(frames)} V_Cl+ frames, showing {min(args.frames, len(frames))}")
        values = []
        for atoms in frames[: args.frames]:
            out, _ = evaluate(model, atoms, z_table, mace_data, torch_geometric)
            alpha = out["carrier_alpha"][:, LIVE].numpy()
            symbols = np.array(atoms.get_chemical_symbols())
            values.append(report(alpha, symbols, None, f"frame {len(values)}"))
        if values:
            print(f"  -> participation across frames: min {min(values):.2f} "
                  f"max {max(values):.2f}  (79 atoms; 16 Cs, 16 Pb, 47 Cl)\n")

    # ---- 2. Tiled supercells, several sites -----------------------------------------
    print("\nTILED SUPERCELLS from the unrelaxed primitive, several distinct Cl sites\n")
    pristine = pst.find_pristine(args.data_dir)
    sites = pst.classify_sites(pristine)
    chosen = [members[0] for members in list(sites.values())[: args.sites]]
    for text in args.repeats:
        for number, site_index in enumerate(chosen):
            supercell = pristine.repeat(pst.parse_repeat(text))
            pst.set_state(supercell, pst.PRISTINE)
            vacancy = supercell.get_positions()[site_index].copy()
            defect, _ = pst.make_vacancy(supercell, site_index)
            pst.set_state(defect, pst.VACANCY_QP1)
            out, _ = evaluate(model, defect, z_table, mace_data, torch_geometric)
            alpha = out["carrier_alpha"][:, LIVE].numpy()
            symbols = np.array(defect.get_chemical_symbols())
            cell = defect.cell.array
            fractional = (defect.get_positions() - vacancy) @ np.linalg.inv(cell)
            fractional -= np.rint(fractional)
            distance = np.linalg.norm(fractional @ cell, axis=1)
            report(alpha, symbols, distance, f"N={len(defect)} site {number}")
        print()

    # ---- 3. The pristine 80-atom cell, for contrast ---------------------------------
    print("PRISTINE 80-atom cell at the SAME counter (0,0,1,0) -- no defect to localise on")
    atoms = pristine.copy()
    pst.set_state(atoms, pst.VACANCY_QP1)
    out, _ = evaluate(model, atoms, z_table, mace_data, torch_geometric)
    alpha = out["carrier_alpha"][:, LIVE].numpy()
    report(alpha, np.array(atoms.get_chemical_symbols()), None, "pristine")


if __name__ == "__main__":
    main()
