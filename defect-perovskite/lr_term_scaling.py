#!/usr/bin/env python3
"""Which term inside ``delta_lr`` is the extensive one?

``delta_lr = E_LR(q_host + pol + carrier) - E_LR(q_host)``. E_LR is quadratic in the
charges, so the host self-energy -- the genuinely extensive Madelung energy of the lattice,
which is physically correct and must not be "fixed" -- cancels exactly by construction. What
survives is five terms:

    pol^2, carrier^2, 2 pol.carrier, 2 host.pol, 2 host.carrier

each obtained by the polarisation identity B(a,b) = [E(a+b) - E(a) - E(b)] / 2. Measuring
all five against N says which one grows, rather than attributing the growth by argument.

Also evaluated here is the *proposed* fix. LES is not a split Ewald: it is a pure
reciprocal-space sum over k != 0 of a Gaussian-smeared charge, so sigma is the physical
smearing rather than a convergence parameter. Dropping k = 0 for a net-charged cell omits
the finite k -> 0 limit of exp(-sigma^2 k^2 / 2)/k^2 * Q^2, whose 1/k^2 piece the jellium
background cancels and whose remainder is

    E_bg = -(2 pi / V) (sigma^2 / 2) Q^2 * norm_factor = -pi sigma^2 Q^2 norm_factor / V

which is the same term as -pi Q^2 / (2 V alpha^2) under alpha = 1/(sigma sqrt 2). It is
O(1/V), i.e. it DECAYS as 1/N. Printing it next to the measured energy settles whether it
can account for growth that goes as N^+1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst


def scaling(sizes: np.ndarray, values: np.ndarray) -> float:
    """d(ln|.|)/d(ln N) across the ladder, by least squares in log-log."""
    good = np.abs(values) > 1e-12
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(sizes[good]), np.log(np.abs(values[good])), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--repeats", nargs="+",
                        default=["2,2,2", "3,2,2", "3,3,2", "3,3,3"])
    parser.add_argument("--site", type=int, default=0)
    parser.add_argument("--per-species", action="store_true",
                        help="patch the trained model to make q^pol neutral WITHIN each "
                             "species. The projection is parameter-free, so it can be "
                             "switched on at inference: if the exponents collapse without "
                             "retraining, the fix is structural rather than fitted.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location="cpu", weights_only=False).double().eval()
    ewald = model.latent_ewald
    sigma = float(ewald.sigma)
    norm_factor = float(ewald.ewald.norm_factor)
    if args.per_species:
        model.latent_charges.per_species_neutral = True
        model.latent_charges.num_species = len(model.atomic_numbers)
        model.per_species_neutral = True
    print(f"model {args.model.name}   per_species_neutral = "
          f"{getattr(model.latent_charges, 'per_species_neutral', False)}")
    print(f"sigma = {sigma}  norm_factor = {norm_factor}\n")

    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    # No relaxation: every size is the same primitive snapshot tiled, which is all the
    # scaling of a quadratic form in the charges needs. Relaxing would only move each
    # size to a slightly different geometry and add hours for no change to an exponent.
    relaxed = pst.find_pristine(args.data_dir)
    sites = pst.classify_sites(relaxed)
    site_index = list(sites.values())[args.site][0]

    rows = []
    for text in args.repeats:
        supercell = relaxed.repeat(pst.parse_repeat(text))
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
            [atomic], batch_size=1, shuffle=False))).to_dict()

        with torch.no_grad():
            out = model(batch, training=False, compute_force=False)
            positions = batch["positions"]
            cell = batch["cell"].view(-1, 3, 3)
            index = batch["batch"]

            q_total = out["latent_charges"]
            q_host = out.get("latent_charges_host")
            q_carrier = out.get("latent_charges_carrier")
            q_pol = None if q_total is None or q_host is None or q_carrier is None \
                else q_total - q_host - q_carrier

        if q_host is None:
            raise SystemExit(
                "model forward does not expose q_host/q_pol/q_carrier; add them to the "
                "output dict before running this diagnostic"
            )

        def energy(charge: torch.Tensor) -> float:
            with torch.no_grad():
                return float(ewald.energy(charge, positions, cell, index).sum())

        def cross(a: torch.Tensor, b: torch.Tensor) -> float:
            """2 B(a,b) by the polarisation identity."""
            return energy(a + b) - energy(a) - energy(b)

        n_atoms = len(defect)
        volume = float(torch.det(cell[0]))
        delta_lr = energy(q_total) - energy(q_host)
        net = float(q_total.sum())

        row = dict(
            N=n_atoms,
            V=volume,
            delta_lr=delta_lr,
            e_host=energy(q_host),
            pol2=energy(q_pol),
            carrier2=energy(q_carrier),
            pol_carrier=cross(q_pol, q_carrier),
            host_pol=cross(q_host, q_pol),
            host_carrier=cross(q_host, q_carrier),
            sum_pol=float(q_pol.sum()),
            sum_carrier=float(q_carrier.sum()),
            sum_host=float(q_host.sum()),
            ss_pol=float((q_pol ** 2).sum()),
            ss_carrier=float((q_carrier ** 2).sum()),
            # the proposed missing background, evaluated for the net charge that is
            # actually present
            bg=-np.pi * sigma ** 2 * net ** 2 * norm_factor / volume,
        )
        rows.append(row)
        print(f"  N = {n_atoms:5d}  delta_lr = {delta_lr:12.4f} eV   done")

    sizes = np.array([r["N"] for r in rows], float)
    print()
    print("Decomposition of delta_lr (eV). host^2 cancels by construction and is shown")
    print("only to make its size visible.\n")
    keys = ["delta_lr", "pol2", "carrier2", "pol_carrier", "host_pol", "host_carrier"]
    print(f"{'N':>6s}" + "".join(f"{k:>15s}" for k in keys))
    for r in rows:
        print(f"{r['N']:6d}" + "".join(f"{r[k]:15.5f}" for k in keys))
    print(f"\n{'scaling':>6s}" + "".join(
        f"{scaling(sizes, np.array([r[k] for r in rows])):+15.3f}" for k in keys))

    print("\n\nCharge bookkeeping. A localised component has size-independent sum-of-")
    print("squares; one that is spread over every atom does not.\n")
    keys2 = ["sum_host", "sum_pol", "sum_carrier", "ss_pol", "ss_carrier"]
    print(f"{'N':>6s}" + "".join(f"{k:>15s}" for k in keys2))
    for r in rows:
        print(f"{r['N']:6d}" + "".join(f"{r[k]:15.6f}" for k in keys2))
    print(f"\n{'scaling':>6s}" + "".join(
        f"{scaling(sizes, np.array([r[k] for r in rows])):+15.3f}" for k in keys2))

    print("\n\nProposed fix: the omitted k = 0 background, -pi sigma^2 Q^2 C / V.\n")
    print(f"{'N':>6s} {'volume':>12s} {'E_bg':>14s} {'delta_lr':>14s} {'ratio':>12s}")
    for r in rows:
        print(f"{r['N']:6d} {r['V']:12.1f} {r['bg']:14.6f} {r['delta_lr']:14.4f} "
              f"{r['bg'] / r['delta_lr']:12.2e}")
    print(f"\n  E_bg scaling  d(ln|.|)/d(ln N) = "
          f"{scaling(sizes, np.array([r['bg'] for r in rows])):+.3f}")
    print("  E_bg is O(1/V) by construction, so it CANNOT cancel a term that grows as N.")


if __name__ == "__main__":
    main()
