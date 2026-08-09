#!/usr/bin/env python3
"""Does the attention weight on the defect survive growing the cell?

``alpha = softmax(l)`` is normalised over **every atom in the cell**, so the bulk enters
the denominator with a weight proportional to N. With ``k`` defect sites at logit
``l_d`` and ``N - k`` bulk sites at ``l_b``, the total weight landing on the defect is

    alpha_defect = k e^{l_d} / (k e^{l_d} + (N - k) e^{l_b})
                 = 1 / (1 + (N - k)/k * e^{-gap}),        gap = l_d - l_b

which is ~1 while ``N << k e^{gap}`` and falls off as ``1/N`` above it. **The logit gap
does not prevent dilution, it only sets the cell size at which dilution starts.** The
crossover is at

    N* ~ k e^{gap}

so a channel with gap 5 delocalises around a thousand atoms while a channel with gap 13
would need millions. A model can therefore look perfectly localised on its training cells
and come apart on larger ones, with nothing in the training metrics able to see it.

This measures the effect directly: build the divacancy at a ladder of sizes and report,
per carrier channel, the participation ratio and the share of ``alpha`` sitting on the
defect shell, against the prediction above. Geometries are unrelaxed -- localisation is a
property of the attention, not of the last 0.02 eV/A of relaxation, and this way the whole
sweep is single points.

    python alpha_dilution.py --model ~/runs/b_128ch_L1_s1/b_128ch_L1_s1.model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument("--repeats", nargs="+",
                        default=["3,3,1", "4,4,2", "5,5,2", "6,6,2", "8,8,2", "10,10,3"])
    parser.add_argument("--state", default="ground", choices=["ground", "excited"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    import defect_size_extensivity as ext

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model = model.double()
    model.eval()
    z_table = tools.AtomicNumberTable([6, 14])

    ext.load_state_conventions(args.data_dir)
    primitive = ext.find_primitive(args.data_dir, None)
    host = str(primitive.info.get("host", "4H-SiC"))
    counts = ext.STATES[args.state]["counts"]

    print(f"model {args.model.name}   state {args.state} {counts}\n")
    print(f"{'N':>6s} {'channel':>8s} {'participation':>14s} {'alpha on shell':>15s} "
          f"{'implied gap':>12s} {'N* = k e^gap':>13s}")

    rows = []
    for text in args.repeats:
        repeat = ext.parse_repeat(text)
        supercell = primitive.repeat(repeat)
        ext.set_state(supercell, ext.PRISTINE, host)
        defect, _ = ext.make_divacancy(supercell, "axial")
        ext.set_state(defect, ext.STATES[args.state], host)

        # The defect shell: atoms that lost a neighbour, i.e. the dangling bonds.
        from ase.neighborlist import neighbor_list

        first, _ = neighbor_list("ij", defect, cutoff=2.2)
        degree = np.bincount(first, minlength=len(defect))
        shell = np.flatnonzero(degree < degree.max())

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
        loader = torch_geometric.dataloader.DataLoader([atomic], batch_size=1,
                                                       shuffle=False)
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
        alpha = out["carrier_alpha"].detach().cpu().numpy()

        n = len(defect)
        k = max(len(shell), 1)
        for index, name in enumerate(CHANNELS):
            column = alpha[:, index]
            participation = 1.0 / float((column**2).sum())
            on_shell = float(column[shell].sum())
            # Invert alpha_defect = 1/(1 + (N-k)/k e^-gap) for the gap the data implies.
            if 0.0 < on_shell < 1.0:
                gap = float(np.log(on_shell / (1 - on_shell) * (n - k) / k))
                crossover = k * np.exp(gap)
            else:
                gap, crossover = float("nan"), float("nan")
            rows.append({"n": n, "channel": name, "participation": participation,
                         "on_shell": on_shell, "gap": gap, "crossover": crossover})
            print(f"{n:6d} {name:>8s} {participation:14.2f} {on_shell:15.4f} "
                  f"{gap:12.3f} {crossover:13.0f}")
        print()

    print("Reading this table:")
    print("  * participation ~ k means the weight sits on the defect shell; "
          "participation ~ N")
    print("    means it has spread over the whole cell and the correction has become a "
          "bulk average.")
    print("  * `alpha on shell` falling as N grows IS the dilution; a size-stable model "
          "holds it flat.")
    print("  * `implied gap` should be a constant of the model, not of the cell. It is "
          "the logit")
    print("    gap read back out of the measured weight, so a stable value confirms the "
          "softmax")
    print("    picture and `N*` then says where that channel gives out.")

    for name in CHANNELS:
        series = [r for r in rows if r["channel"] == name]
        first, last = series[0], series[-1]
        drop = first["on_shell"] - last["on_shell"]
        gaps = [r["gap"] for r in series if np.isfinite(r["gap"])]
        verdict = "STABLE" if abs(drop) < 0.05 else "DILUTING"
        print(f"\n  {name:6s} alpha on shell {first['on_shell']:.4f} (N={first['n']}) "
              f"-> {last['on_shell']:.4f} (N={last['n']})   {verdict}")
        if gaps:
            print(f"         implied gap {np.mean(gaps):.2f} +- {np.std(gaps):.2f}  "
                  f"=> dilution sets in around N* = {np.mean([r['crossover'] for r in series if np.isfinite(r['crossover'])]):.0f} atoms")


if __name__ == "__main__":
    main()
