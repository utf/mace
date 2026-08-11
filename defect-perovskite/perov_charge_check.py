#!/usr/bin/env python3
"""Does the latent charge of a net-charged cell sum to ``a q``, and is E_LR extensive?

``eps_opt`` on the long-range model grows linearly in N and reaches -100 eV, where a
charged cell's image term should *decay* as ``q^2 alpha_M / 2 eps L``. Linear growth is the
signature of an unneutralised Ewald sum: the energy of a periodic array of net charge
diverges with volume unless a compensating background is applied.

4H-SiC could not have exposed this. Every counter there was net neutral, so ``sum_i q_i``
was zero by construction and the background term never mattered. This is the first dataset
with ``q != 0``.
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
    parser.add_argument("--repeats", nargs="+", default=["2,2,2", "3,2,2", "3,3,2"])
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

    print(f"model {args.model.name}\n")
    print(f"{'N':>6s} {'q_cell':>7s} {'sum q_i':>12s} {'a':>8s} {'a*q':>9s} "
          f"{'E_LR':>13s} {'E_LR/N':>11s}")
    history = []
    for text in args.repeats:
        supercell = pristine.repeat(pst.parse_repeat(text))
        pst.set_state(supercell, pst.PRISTINE)
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
            [atomic], batch_size=1, shuffle=False)))
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
        total = out["latent_charges"]
        amplitude = float(out["screening_amplitude"].reshape(-1)[0])
        short_range = float(out["delta_sr_energy"].sum())
        correction = float(out["correction_energy"].sum())
        long_range = correction - short_range
        charge = 1.0  # V_Cl+
        print(f"{len(defect):6d} {charge:7.1f} {float(total.sum()):12.5f} "
              f"{amplitude:8.4f} {amplitude * charge:9.4f} {long_range:13.4f} "
              f"{long_range / len(defect):11.6f}")
        history.append((len(defect), long_range))

    sizes = np.array([h[0] for h in history], float)
    energies = np.array([abs(h[1]) for h in history])
    slope = np.log(energies[-1] / energies[0]) / np.log(sizes[-1] / sizes[0])
    print(f"\n  d(ln|E_LR|)/d(ln N) = {slope:+.3f}")
    print("  expected: -1/3 for a screened monopole image term (q^2/L).")
    print("  +1 means the sum is EXTENSIVE -- a periodic net charge with no compensating")
    print("  background, whose energy grows with the volume.")
    print("\n  sum_i q_i should equal a*q for a correctly assembled latent charge.")


if __name__ == "__main__":
    main()
