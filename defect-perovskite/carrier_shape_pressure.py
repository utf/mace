#!/usr/bin/env python3
"""Does ``carrier^2`` exert a delocalisation pressure at TRAINING cell size?

``carrier^2 = E_Ewald[a alpha]`` splits into an L-independent part and a ``1/L`` image part:

    E(L) = A + k / L

``A`` is the in-cell electrostatic energy of the carrier channel with itself -- the on-site
Gaussian self-term plus the inter-site Hartree between the atoms the carrier occupies. None
of it belongs: a single hole has no Hartree self-repulsion, so the whole in-cell
electrostatic energy of ONE carrier channel is self-interaction error. Only ``k / L`` is
real, because that artefact genuinely exists in the periodic DFT labels.

If ``A`` is larger for a localised carrier than for a delocalised one, the term pays the
model to spread ``alpha`` out, and it is a candidate cause of the Cs collapse.

The previous estimate used a fit obtained on 28-57 A cells extrapolated down to 14.3 A,
outside its range. This measures at the training cell itself, with an isotropic series
(1,1,1 / 2,2,2 / 3,3,3) so the Madelung constant is fixed across the fit.

The analytic self-energy is printed alongside as a check on the decomposition:

    E_self = k_e a^2 sum_i alpha_i^2 / (2 sigma sqrt(pi))
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst
from host_carrier_probe import build, shell_indices

COULOMB = 14.399645  # eV.A


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path,
                        default=Path.home() / "runs/perov_lr_s1/perov_lr_s1.model")
    parser.add_argument("--nolr-model", type=Path,
                        default=Path.home() / "runs/perov_nolr_s1/perov_nolr_s1.model")
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    # Isotropic only, starting at the TRAINING cell (1,1,1 = 79 atoms, L = 14.3 A).
    parser.add_argument("--repeats", nargs="+", default=["1,1,1", "2,2,2", "3,3,3"])
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location="cpu", weights_only=False).double().eval()
    nolr = torch.load(
        args.nolr_model, map_location="cpu", weights_only=False
    ).double().eval()
    ewald = model.latent_ewald
    sigma = float(ewald.sigma)
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    pristine = pst.find_pristine(args.data_dir)
    site_index = list(pst.classify_sites(pristine).values())[0][0]

    rows = []
    for text in args.repeats:
        supercell = pristine.repeat(pst.parse_repeat(text))
        pst.set_state(supercell, pst.PRISTINE)
        defect, _ = pst.make_vacancy(supercell, site_index)
        pst.set_state(defect, pst.VACANCY_QP1)
        batch = build(defect, z_table, mace_data, torch_geometric)
        shell, symbols = shell_indices(defect)
        caesium = np.flatnonzero(symbols == "Cs")
        with torch.no_grad():
            out = model(batch, training=False, compute_force=False)
            out_nolr = nolr(batch, training=False, compute_force=False)
        amplitude = float(out["screening_amplitude"].reshape(-1)[0])
        u_field = out_nolr["carrier_readouts"][:, 2]
        positions, cell = batch["positions"], batch["cell"].view(-1, 3, 3)
        index = batch["batch"]

        record = {"N": len(defect), "L": float(torch.det(cell[0])) ** (1.0 / 3.0)}
        for name, sites in (("Pb shell", shell), ("Cs uniform", caesium)):
            weights = np.zeros(len(defect))
            weights[sites] = 1.0 / len(sites)
            alpha = torch.tensor(weights)
            charge = amplitude * alpha
            with torch.no_grad():
                record[name] = float(ewald.energy(charge, positions, cell, index).sum())
            record[f"ipr[{name}]"] = float((alpha**2).sum())
            record[f"u[{name}]"] = float((alpha * u_field).sum())
        record["a"] = amplitude
        rows.append(record)
        print(f"  N = {len(defect):5d}  L = {record['L']:5.1f} A   done")

    inverse_l = np.array([1.0 / r["L"] for r in rows])
    print("\n\ncarrier^2 = E_Ewald[a alpha], isotropic series (eV)\n")
    print(f"{'N':>6s} {'L (A)':>8s} {'Pb shell':>12s} {'Cs uniform':>12s} "
          f"{'ipr(Pb)':>9s} {'ipr(Cs)':>9s}")
    for r in rows:
        print(f"{r['N']:6d} {r['L']:8.2f} {r['Pb shell']:12.5f} {r['Cs uniform']:12.5f} "
              f"{r['ipr[Pb shell]']:9.5f} {r['ipr[Cs uniform]']:9.5f}")

    print("\n\nDECOMPOSITION  E(L) = A + k/L\n")
    shape = {}
    for name in ("Pb shell", "Cs uniform"):
        values = np.array([r[name] for r in rows])
        k, a_const = np.polyfit(inverse_l, values, 1)
        residual = np.abs(values - (k * inverse_l + a_const)).max()
        shape[name] = a_const
        print(f"  {name:11s}  A = {a_const:+.5f} eV   k = {k:+.4f} eV.A   "
              f"max resid {residual * 1e3:.2f} meV")

    amplitude = rows[0]["a"]
    prefactor = COULOMB * amplitude**2 / (2.0 * sigma * math.sqrt(math.pi))
    print(f"\n  analytic on-site self-energy  k_e a^2 ipr / (2 sigma sqrt(pi)),"
          f"  a = {amplitude:.3f}, sigma = {sigma:.1f} A:")
    for name in ("Pb shell", "Cs uniform"):
        print(f"    {name:11s} ipr = {rows[0][f'ipr[{name}]']:.5f}  ->  "
              f"{prefactor * rows[0][f'ipr[{name}]']:+.4f} eV")

    pressure = shape["Pb shell"] - shape["Cs uniform"]
    binding = rows[0]["u[Pb shell]"] - rows[0]["u[Cs uniform]"]
    print("\n\nTHE COMPETITION at the training cell\n")
    print(f"  shape pressure  A(Pb) - A(Cs) = {pressure:+.4f} eV")
    print(f"  short-range     <u>(Pb) - <u>(Cs) = {binding:+.4f} eV")
    print()
    if pressure > 0:
        print("  A(Pb) > A(Cs): carrier^2 PAYS to delocalise. It is a candidate cause,")
        print(f"  and at {abs(pressure):.3f} eV it is "
              f"{abs(pressure) / max(abs(binding), 1e-9):.1f}x the short-range difference.")
    else:
        print("  A(Pb) < A(Cs): carrier^2 favours the localised shell and is EXONERATED")
        print("  as the driver of delocalisation. Look elsewhere.")
    print("\n  Only k/L is physical -- the image artefact is in the periodic DFT labels.")
    print("  A is in-cell self-interaction of one carrier channel with itself, which a")
    print("  single hole does not have.")


if __name__ == "__main__":
    main()
