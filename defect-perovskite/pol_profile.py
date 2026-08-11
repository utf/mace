#!/usr/bin/env python3
"""Is ``q^pol`` a local response, or a spurious ionic lattice?

``q^pol`` is built as a per-atom MLP of the node features and the counter embedding, then
mean-subtracted over the cell. Mean subtraction guarantees ``sum_i q^pol_i = 0`` -- which is
what the docstring claims as the safeguard -- but neutrality is not localisation. For an
atom far from the defect the node features are the bulk values for its species, so the MLP
returns a fixed per-species number and the subtraction removes only the composition-weighted
average. Each species is then left with a fixed, non-decaying residual charge on every atom
in the crystal: a fictitious extra ionic lattice.

The test: bin ``q^pol`` by species and by distance from the vacancy. A genuine polarisation
response decays with distance. A spurious ionic lattice is flat in distance and has a
species-dependent plateau that does not move with cell size.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--repeats", nargs="+", default=["2,2,2", "3,3,2"])
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location="cpu", weights_only=False).double().eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    pristine = pst.find_pristine(args.data_dir)
    sites = pst.classify_sites(pristine)
    site_index = list(sites.values())[0][0]

    print(f"model {args.model.name}   state V_Cl+ (0,0,1,0)\n")

    for text in args.repeats:
        supercell = pristine.repeat(pst.parse_repeat(text))
        pst.set_state(supercell, pst.PRISTINE)
        vacancy_position = supercell.get_positions()[site_index].copy()
        defect, _ = pst.make_vacancy(supercell, site_index)
        pst.set_state(defect, pst.VACANCY_QP1)

        config = mace_data.config_from_atoms(
            defect,
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
        q_pol = (out["latent_charges"] - out["latent_charges_host"]
                 - out["latent_charges_carrier"]).numpy()

        # Minimum-image distance from the vacancy site.
        cell = defect.cell.array
        delta = defect.get_positions() - vacancy_position
        fractional = delta @ np.linalg.inv(cell)
        fractional -= np.rint(fractional)
        distance = np.linalg.norm(fractional @ cell, axis=1)
        symbols = np.array(defect.get_chemical_symbols())

        print(f"N = {len(defect)}   sum q_pol = {q_pol.sum():+.2e}   "
              f"sum q_pol^2 = {(q_pol ** 2).sum():.4f}")
        print(f"  {'species':>8s} {'<3 A':>12s} {'3-6 A':>12s} {'6-10 A':>12s} "
              f"{'>10 A':>12s} {'bulk spread':>13s}")
        edges = [(0, 3), (3, 6), (6, 10), (10, 1e9)]
        for species in ("Cs", "Pb", "Cl"):
            mask = symbols == species
            cells = []
            for low, high in edges:
                selected = mask & (distance >= low) & (distance < high)
                cells.append(f"{q_pol[selected].mean():12.5f}" if selected.any()
                             else f"{'-':>12s}")
            far = mask & (distance >= 10)
            spread = f"{q_pol[far].std():13.6f}" if far.any() else f"{'-':>13s}"
            print(f"  {species:>8s} " + " ".join(cells) + " " + spread)
        print()

    print("A local response decays towards zero in the outer shells. A flat, species-")
    print("dependent plateau with a tiny spread is an ionic lattice: every atom in the")
    print("crystal carries a fixed charge, so sum_i (q^pol_i)^2 -- and hence its Madelung")
    print("energy and its cross term with q^host -- grows in proportion to N.")


if __name__ == "__main__":
    main()
