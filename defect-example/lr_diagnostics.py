#!/usr/bin/env python3
"""Why does ``|E_LR|`` grow with cell size? Diagnostics D1-D3, no retraining.

Growth is outside the family of physically possible answers: monopole decays as 1/L,
dipole as 1/L^3, and a residual monopole from a non-neutral cell still *decays*. So neither
the frozen ``eps_inf`` nor the fitted band-edge gauge can be responsible -- both scale
``E_LR`` without touching the sign of an exponent. This looks for an assembly bug.

Runs three things on an existing model:

**D1** ``E_LR`` at ``n = 0``. Both ``q_pol`` and ``q_carrier`` carry an explicit counter
factor, so they vanish identically there and ``delta_lr`` should be *exactly* zero. A
non-zero value puts the bug in the branch (host channel, background, or the Ewald sum) and
excludes the carrier shape.

**D2** channel charge sums at every size. ``q_host`` and ``q_pol`` are mean-subtracted, so
each is neutral by construction; ``q_carrier`` is neutral only through
``sum_i alpha_i = 1`` and ``sum_c s_c n_c = 0``, which is an identity worth checking rather
than assuming. Also reports the dipole, which is well defined precisely because the cell is
neutral.

**D3** the Ewald energy of each channel *separately*, plus the cross terms. ``E`` is
quadratic in the charges, so
``E(A+B) = E(A) + E(B) + 2E_cross(A,B)`` -- evaluating each piece against ``N`` says which
one is growing, which is what a term-by-term Ewald decomposition would tell us without
needing LES internals.

    python lr_diagnostics.py --model ~/runs/esize_128ch_L1_s2/esize_128ch_L1_s2.model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")


def ewald(model, charges, positions, cell, batch):
    return float(model.latent_ewald.energy(charges, positions, cell, batch).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_full")
    parser.add_argument("--repeats", nargs="+",
                        default=["4,4,2", "5,5,2", "6,6,2", "8,8,2", "10,10,3"])
    parser.add_argument("--state", default="excited", choices=["ground", "excited"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    import defect_size_extensivity as ext

    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model = model.double().eval()
    if not getattr(model, "use_long_range", False):
        raise SystemExit("model has no long-range branch")
    z_table = tools.AtomicNumberTable([6, 14])

    ext.load_state_conventions(args.data_dir)
    primitive = ext.find_primitive(args.data_dir, None)
    host = str(primitive.info.get("host", "4H-SiC"))

    print(f"model {args.model.name}\n")
    header = (f"{'N':>6s} {'Sq_host':>11s} {'Sq_pol':>11s} {'Sq_carr':>11s} "
              f"{'Sq_total':>11s} {'|dipole|':>10s}")
    results = {}

    for state_name, counts_spec in (("n=0 (D1)", ext.PRISTINE),
                                    (f"{args.state} (D2/D3)", ext.STATES[args.state])):
        print(f"=== {state_name} ===")
        print(header)
        rows = []
        for text in args.repeats:
            supercell = primitive.repeat(ext.parse_repeat(text))
            ext.set_state(supercell, ext.PRISTINE, host)
            defect, _ = ext.make_divacancy(supercell, "axial")
            ext.set_state(defect, counts_spec, host)

            config = mace_data.config_from_atoms(
                defect,
                key_specification=mace_data.KeySpecification(
                    info_keys={"carrier_counts": "carrier_counts", "host": "host",
                               "multiplicity": "multiplicity"},
                    arrays_keys={},
                ),
            )
            mace_data.canonicalise_config_counters(config)
            atomic = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=4.0)
            batch = next(iter(torch_geometric.dataloader.DataLoader(
                [atomic], batch_size=1, shuffle=False))).to(device)
            with torch.no_grad():
                out = model(batch.to_dict(), training=False, compute_force=False)

                # Rebuild the channels exactly as the model does, so the pieces sum to the
                # charge it actually used rather than to a reimplementation of it.
                counts = batch["carrier_counts"].view(1, -1).to(torch.get_default_dtype())
                q_total = out["latent_charges"]
                q_host = out["latent_charges_host"]
                q_carrier = out["latent_charges_carrier"]
                q_pol = q_total - q_host - q_carrier

                positions = batch["positions"]
                cell = batch["cell"].view(-1, 3)
                index = batch["batch"]
                dipole = float(torch.linalg.norm((q_total.unsqueeze(-1) * positions).sum(0)))

                pieces = {
                    "host": q_host, "pol": q_pol, "carrier": q_carrier, "total": q_total,
                }
                energies = {k: ewald(model, v, positions, cell, index)
                            for k, v in pieces.items()}
                delta_lr = energies["total"] - energies["host"]

            rows.append({
                "N": len(defect),
                "sums": {k: float(v.sum()) for k, v in pieces.items()},
                "dipole": dipole, "energies": energies, "delta_lr": delta_lr,
            })
            s = rows[-1]["sums"]
            print(f"{len(defect):6d} {s['host']:11.2e} {s['pol']:11.2e} "
                  f"{s['carrier']:11.2e} {s['total']:11.2e} {dipole:10.4f}")
        results[state_name] = rows
        print()

    print("=== D1: delta_lr at n = 0 (must be exactly zero) ===")
    for row in results["n=0 (D1)"]:
        verdict = "ok" if abs(row["delta_lr"]) < 1e-12 else "NON-ZERO -- bug in the branch"
        print(f"  N={row['N']:5d}  delta_lr = {row['delta_lr']:+.6e}   {verdict}")

    print(f"\n=== D3: Ewald energy per channel vs N ({args.state}) ===")
    print(f"{'N':>6s} {'E(host)':>13s} {'E(pol)':>13s} {'E(carrier)':>13s} "
          f"{'E(total)':>13s} {'delta_lr':>13s}")
    for row in results[f"{args.state} (D2/D3)"]:
        e = row["energies"]
        print(f"{row['N']:6d} {e['host']:13.6f} {e['pol']:13.6f} {e['carrier']:13.6f} "
              f"{e['total']:13.6f} {row['delta_lr']:13.6f}")

    rows = results[f"{args.state} (D2/D3)"]
    sizes = np.array([r["N"] for r in rows], dtype=float)
    print("\nscaling exponent d(ln|E|)/d(ln N) between the two largest cells "
          "(monopole -1/3, dipole -1, growth = bug):")
    for key in ("host", "pol", "carrier", "total"):
        values = np.array([abs(r["energies"][key]) for r in rows])
        if values[-1] > 0 and values[-2] > 0:
            slope = np.log(values[-1] / values[-2]) / np.log(sizes[-1] / sizes[-2])
            print(f"  E({key:8s}) {values[-1]:12.6f}   exponent {slope:+.3f}")
    cross = np.array([abs(r["delta_lr"]) for r in rows])
    slope = np.log(cross[-1] / cross[-2]) / np.log(sizes[-1] / sizes[-2])
    print(f"  delta_lr    {cross[-1]:12.6f}   exponent {slope:+.3f}")


if __name__ == "__main__":
    main()
