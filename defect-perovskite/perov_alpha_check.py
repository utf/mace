#!/usr/bin/env python3
"""Is the attention on V_Cl size-stable? The perovskite analogue of alpha_dilution.py.

The SiC version cannot be reused: it builds a divacancy and looks for a six-atom dangling
shell. Here the defect is a single Cl vacancy, so the "shell" is the Pb atoms that lost a
neighbour, and only one carrier channel is live (``h_maj``, since V_Cl+ = (0,0,1,0)).

That single live channel is the reason this system is a clean test bed: the signed sum
``sum_c s_c n_c alpha_i^c`` reduces to ``+1 * alpha_i^h_maj``, which cannot cancel however
similar the channels become. The channel-collapse degeneracy that silenced ``q^carrier`` in
the SiC long-range model is structurally impossible here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--repeats", nargs="+",
                        default=["1,1,2", "2,2,2", "3,2,2", "3,3,2", "3,3,3"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    import perovskite_size_test as pst

    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).double().eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    pristine = pst.find_pristine(args.data_dir)
    sites = pst.classify_sites(pristine)
    site_index = list(sites.values())[0][0]

    print(f"model {args.model.name}   state V_Cl+ (0,0,1,0)\n")
    print(f"{'N':>6s} {'shell':>6s} " + " ".join(f"{c:>14s}" for c in CHANNELS))
    print(f"{'':>6s} {'':>6s} " + " ".join(f"{'partic  a_shell':>14s}" for _ in CHANNELS))

    history = []
    for text in args.repeats:
        supercell = pristine.repeat(pst.parse_repeat(text))
        pst.set_state(supercell, pst.PRISTINE)
        defect, _ = pst.make_vacancy(supercell, site_index)
        pst.set_state(defect, pst.VACANCY_QP1)

        # The vacancy shell: Pb atoms that lost a Cl neighbour, so are under-coordinated.
        from ase.neighborlist import neighbor_list

        first, _ = neighbor_list("ij", defect, cutoff=3.4)
        degree = np.bincount(first, minlength=len(defect))
        symbols = np.array(defect.get_chemical_symbols())
        lead = symbols == "Pb"
        shell = np.flatnonzero(lead & (degree < degree[lead].max()))

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
        alpha = out["carrier_alpha"].numpy()
        readouts = out["carrier_readouts"].numpy()

        cells = []
        for index in range(len(CHANNELS)):
            column = alpha[:, index]
            cells.append(f"{1.0 / (column**2).sum():7.1f} {column[shell].sum():6.3f}")
        print(f"{len(defect):6d} {len(shell):6d} " + " ".join(f"{c:>14s}" for c in cells))
        history.append((len(defect), alpha[:, 2][shell].sum(),
                        float((alpha[:, 2] * readouts[:, 2]).sum())))

    print(f"\n{'N':>6s} {'a_shell(h_maj)':>16s} {'<u>_h_maj (eV)':>16s}")
    for size, share, pooled in history:
        print(f"{size:6d} {share:16.4f} {pooled:16.4f}")
    first, last = history[0], history[-1]
    print(f"\n  alpha on shell {first[1]:.4f} -> {last[1]:.4f}   "
          f"({'STABLE' if abs(last[1] - first[1]) < 0.05 else 'DILUTING'})")
    print(f"  <u> pooled     {first[2]:.4f} -> {last[2]:.4f}   "
          f"{(last[2] - first[2]) * 1e3:+.1f} meV")
    print(
        "\n<u> is what enters the energy: the correction is n_c <u>_c with n_c = 1 here, so\n"
        "any drift in this column appears directly in eps_opt."
    )


if __name__ == "__main__":
    main()
