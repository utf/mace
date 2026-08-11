#!/usr/bin/env python3
"""Does `host.carrier` outbid short-range binding for the carrier's position?

Under Step A, ``delta_lr = carrier^2 + host.carrier``, and with ``sum_i alpha_i = 1``

    E_host.carrier = a q sum_i alpha_i phi_host(r_i) = a q <phi_host>_alpha

which has **exactly the same pooling form** as ``Delta E_SR = sum_c n_c <u>_alpha``. Two
energy terms, the same alpha, one field learned and one fixed-shape. If the electrostatic
term is the larger, the attention has every reason to sit wherever ``phi_host`` is most
favourable -- the Cs sublattice -- rather than on the vacancy shell.

The measurement pools both energies under the same two attention patterns:

* ``alpha`` localised on the two under-coordinated Pb -- the physically correct answer,
  which ``perov_nolr_s1`` finds unaided;
* ``alpha`` uniform over the Cs sublattice -- the collapsed answer.

and reports

    dE_elec = a q [ <phi_host>_alpha(Cs) - <phi_host>_alpha(Pb shell) ]
    du_SR   = sum_c n_c [ <u>_alpha(Pb shell) - <u>_alpha(Cs) ]

Both are the energy *cost of moving the carrier from the Pb shell to Cs* in their own
channel, so they are directly comparable and their signs say which site each term prefers.

`phi_host` is not an arbitrary offset: LES excludes ``k = 0``, so the cell average of
``phi_host`` is identically zero by construction and the gauge is already fixed. It is the
Madelung potential of the host at the carrier's position.

CAVEAT, stated up front: the only available host field comes from ``perov_lr_s1``, whose
``q^host`` was fitted alongside a broken ``q^pol`` and an already-collapsed attention. Its
*amplitude* is therefore itself suspect -- which is exactly the free scale H1 is about -- so
the implied bulk amplitude is reported rather than assumed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst

LIVE = 2  # h_maj; V_Cl+ carries counts (0, 0, 1, 0) so signed_counts is +1 there


def build(defect, z_table, mace_data, torch_geometric):
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
    return next(iter(torch_geometric.dataloader.DataLoader(
        [atomic], batch_size=1, shuffle=False))).to_dict()


def shell_indices(defect):
    """The Pb that lost a Cl neighbour, i.e. the under-coordinated ones."""
    from ase.neighborlist import neighbor_list

    first, _ = neighbor_list("ij", defect, cutoff=3.4)
    degree = np.bincount(first, minlength=len(defect))
    symbols = np.array(defect.get_chemical_symbols())
    lead = symbols == "Pb"
    return np.flatnonzero(lead & (degree < degree[lead].max())), symbols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--lr-model", type=Path,
                        default=Path.home() / "runs/perov_lr_s1/perov_lr_s1.model")
    parser.add_argument("--nolr-model", type=Path,
                        default=Path.home() / "runs/perov_nolr_s1/perov_nolr_s1.model")
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--repeats", nargs="+", default=["2,2,2", "3,2,2", "3,3,2"])
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    lr = torch.load(args.lr_model, map_location="cpu", weights_only=False).double().eval()
    nolr = torch.load(
        args.nolr_model, map_location="cpu", weights_only=False
    ).double().eval()
    ewald = lr.latent_ewald
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    pristine = pst.find_pristine(args.data_dir)
    sites = pst.classify_sites(pristine)
    site_index = list(sites.values())[0][0]

    print(f"host field + a from : {args.lr_model.name}")
    print(f"u field from        : {args.nolr_model.name}\n")

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
            out_lr = lr(batch, training=False, compute_force=False)
            out_nolr = nolr(batch, training=False, compute_force=False)
        q_host = out_lr["latent_charges_host"]
        amplitude = float(out_lr["screening_amplitude"].reshape(-1)[0])
        u_field = out_nolr["carrier_readouts"][:, LIVE]
        positions, cell = batch["positions"], batch["cell"].view(-1, 3, 3)
        index = batch["batch"]
        n_atoms = len(defect)

        # Two attention patterns, both normalised, neither taken from a trained model so
        # that the comparison is of SITES rather than of two models' quirks.
        patterns = {}
        weights = np.zeros(n_atoms)
        weights[shell] = 1.0 / len(shell)
        patterns["Pb shell"] = weights.copy()
        weights = np.zeros(n_atoms)
        weights[caesium] = 1.0 / len(caesium)
        patterns["Cs uniform"] = weights.copy()

        def energy(charge):
            with torch.no_grad():
                return float(ewald.energy(charge, positions, cell, index).sum())

        record = {"N": n_atoms, "n_shell": len(shell), "n_cs": len(caesium)}
        base_host = energy(q_host)
        for name, weight in patterns.items():
            alpha = torch.tensor(weight)
            q_carrier = amplitude * alpha
            cross = energy(q_host + q_carrier) - base_host - energy(q_carrier)
            record[f"cross[{name}]"] = cross
            record[f"u[{name}]"] = float((alpha * u_field).sum())
        record["dE_elec"] = record["cross[Cs uniform]"] - record["cross[Pb shell]"]
        record["du_SR"] = record["u[Pb shell]"] - record["u[Cs uniform]"]
        record["a"] = amplitude
        record["q_host_rms"] = float(q_host.pow(2).mean().sqrt())
        for species in ("Cs", "Pb", "Cl"):
            mask = symbols == species
            record[f"qh[{species}]"] = float(q_host[mask].mean())
        rows.append(record)
        print(f"  N = {n_atoms:5d}  shell = {len(shell)} Pb, Cs = {len(caesium)}   done")

    print("\n\nPOOLED ENERGIES under each attention pattern (eV)\n")
    print(f"{'N':>6s} {'host.carrier(Pb)':>18s} {'host.carrier(Cs)':>18s} "
          f"{'<u>(Pb)':>10s} {'<u>(Cs)':>10s}")
    for r in rows:
        print(f"{r['N']:6d} {r['cross[Pb shell]']:18.5f} {r['cross[Cs uniform]']:18.5f} "
              f"{r['u[Pb shell]']:10.5f} {r['u[Cs uniform]']:10.5f}")

    print("\n\nTHE COMPETITION -- cost of moving the carrier from the Pb shell to Cs\n")
    print(f"{'N':>6s} {'dE_elec (eV)':>14s} {'du_SR (eV)':>12s} "
          f"{'|dE_elec/du_SR|':>17s}  favours")
    for r in rows:
        ratio = abs(r["dE_elec"]) / max(abs(r["du_SR"]), 1e-12)
        winner = "Cs (electrostatic)" if abs(r["dE_elec"]) > abs(r["du_SR"]) \
            else "Pb shell (short-range)"
        print(f"{r['N']:6d} {r['dE_elec']:14.5f} {r['du_SR']:12.5f} {ratio:17.2f}  {winner}")
    print("\n  dE_elec < 0 means the electrostatic term PREFERS Cs.")
    print("  du_SR   < 0 means the short-range term PREFERS the Pb shell.")
    print("  A sublattice average of a periodic potential is size-independent, so a flat")
    print("  dE_elec across the ladder is the signature of a wrong fixed point rather")
    print("  than of dilution.")

    sizes = np.array([r["N"] for r in rows], float)
    values = np.array([r["dE_elec"] for r in rows])
    if len(sizes) > 1 and np.all(np.abs(values) > 1e-12):
        slope = np.polyfit(np.log(sizes), np.log(np.abs(values)), 1)[0]
        print(f"\n  dE_elec scaling d(ln|.|)/d(ln N) = {slope:+.3f}  "
              f"(spread {values.max() - values.min():+.5f} eV)")

    print("\n\nIMPLIED HOST AMPLITUDE (the free scale H1 is about)\n")
    print(f"{'N':>6s} {'a':>8s} {'rms q_host':>12s} {'<q_host> Cs':>13s} "
          f"{'Pb':>10s} {'Cl':>10s}")
    for r in rows:
        print(f"{r['N']:6d} {r['a']:8.4f} {r['q_host_rms']:12.5f} {r['qh[Cs]']:13.5f} "
              f"{r['qh[Pb]']:10.5f} {r['qh[Cl]']:10.5f}")
    print("\n  q^host and a enter host.carrier only through their product, so one of the")
    print("  two scales is free. This is the amplitude a fix under H1 would pin.")


if __name__ == "__main__":
    main()
