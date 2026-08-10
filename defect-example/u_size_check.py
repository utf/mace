#!/usr/bin/env python3
"""Is the pooled site energy ``<u>_alpha`` size-converged, or only ``alpha``?

The size hinge constrains ``f (ubar_w - <u>_alpha)`` -- how much *weight* a larger cell
would divert to bulk. It says nothing about whether ``u`` itself is size-converged.
``alpha`` can be perfectly flat in ``N`` while ``<u>_alpha`` drifts, and the correction
``sum_c n_c <u>_c`` drifts with it.

This is the direct test, and it is the natural suspect for the residual ``E_ZPL`` drift on
the size-constrained model: ``E_f`` and the ground-state correction went flat while
``E_ZPL`` did not, which points at whichever channel the excited counter uses and the
ground counter does not.

Single points on unrelaxed cells, as in alpha_dilution.py -- localisation and site energy
are properties of the attention and the readout, not of the last 0.02 eV/A of relaxation.

    python u_size_check.py --model ~/runs/esize_128ch_L1_s2/esize_128ch_L1_s2.model
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
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_full")
    parser.add_argument("--repeats", nargs="+",
                        default=["4,4,2", "5,5,2", "6,6,2", "8,8,2", "10,10,3", "12,12,3"])
    parser.add_argument("--state", default="ground", choices=["ground", "excited"])
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
    z_table = tools.AtomicNumberTable([6, 14])

    ext.load_state_conventions(args.data_dir)
    primitive = ext.find_primitive(args.data_dir, None)
    host = str(primitive.info.get("host", "4H-SiC"))
    state = ext.STATES[args.state]

    print(f"model {args.model.name}   state {args.state} {state['counts']}\n")
    print(f"{'N':>6s} " + " ".join(f"{c:>12s}" for c in CHANNELS) + "   dE_SR       E_LR")
    print(f"{'':>6s} " + " ".join(f"{'<u>_alpha':>12s}" for _ in CHANNELS))

    history = []
    for text in args.repeats:
        supercell = primitive.repeat(ext.parse_repeat(text))
        ext.set_state(supercell, ext.PRISTINE, host)
        defect, _ = ext.make_divacancy(supercell, "axial")
        ext.set_state(defect, state, host)

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
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([atomic], batch_size=1,
                                                       shuffle=False))
        ).to(device)
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
        alpha = out["carrier_alpha"].detach().cpu().numpy()
        readouts = out["carrier_readouts"].detach().cpu().numpy()
        pooled = (alpha * readouts).sum(axis=0)
        counts = np.asarray(state["counts"], dtype=float)
        correction = float((counts * pooled).sum())
        # The short-range correction is only half the story on a long-range model:
        # correction_energy = delta_sr + delta_lr, so the remainder is E_LR, and screened
        # electrostatics is genuinely size-dependent in a way sum_c n_c <u>_c is not.
        short_range = out.get("delta_sr_energy")
        total = out.get("correction_energy")
        long_range = (
            float(total.detach().sum() - short_range.detach().sum())
            if short_range is not None and total is not None
            else float("nan")
        )
        history.append((len(defect), pooled, correction, long_range))
        print(f"{len(defect):6d} " + " ".join(f"{v:12.5f}" for v in pooled)
              + f"   {correction:9.5f} {long_range:11.5f}")

    print("\ndrift from the smallest to the largest cell:")
    first, last = history[0], history[-1]
    for index, channel in enumerate(CHANNELS):
        change = last[1][index] - first[1][index]
        print(f"  <u>_{channel:6s} {first[1][index]:9.5f} -> {last[1][index]:9.5f}   "
              f"{change * 1e3:+8.1f} meV")
    print(f"  correction  {first[2]:9.5f} -> {last[2]:9.5f}   "
          f"{(last[2] - first[2]) * 1e3:+8.1f} meV")
    print(f"  E_LR        {first[3]:9.5f} -> {last[3]:9.5f}   "
          f"{(last[3] - first[3]) * 1e3:+8.1f} meV")
    print(
        "\nalpha being flat does not make these flat: the hinge bounds the weight a larger\n"
        "cell would divert to bulk, not the size dependence of u itself."
    )


if __name__ == "__main__":
    main()
