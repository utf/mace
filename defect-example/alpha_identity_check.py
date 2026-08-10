#!/usr/bin/env python3
"""Are the per-channel attentions distinct, or has the model collapsed them onto one shape?

``q_i^carrier = a * sum_c s_c n_c alpha_i^c`` is the only channel carrying a monopole and
the only one representing electron-hole *separation*. If every channel converges on the
same spatial distribution, the signed sum collapses: for the excited counter (1,1,0,2) with
signs (-,-,+,+) the weights are (-1,-1,0,+2), which sum to zero, so identical ``alpha``
gives ``q^carrier = 0`` pointwise and the electron-hole term disappears entirely.

This matters because both the logit seed (a channel-independent novelty statistic) and the
size hinge push every channel toward the same defect shell. Neither was designed to make
the channels identical, and nothing in the objective distinguishes them once they are.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")
SIGNS = np.array([-1.0, -1.0, 1.0, 1.0])


def alpha_for(model, data_dir, repeat=(6, 6, 2), state="excited"):
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    import defect_size_extensivity as ext

    primitive = ext.find_primitive(data_dir, None)
    supercell = primitive.repeat(repeat)
    ext.set_state(supercell, ext.PRISTINE, "4H-SiC")
    defect, _ = ext.make_divacancy(supercell, "axial")
    ext.set_state(defect, ext.STATES[state], "4H-SiC")
    config = mace_data.config_from_atoms(
        defect,
        key_specification=mace_data.KeySpecification(
            info_keys={"carrier_counts": "carrier_counts", "host": "host",
                       "multiplicity": "multiplicity"},
            arrays_keys={},
        ),
    )
    mace_data.canonicalise_config_counters(config)
    atomic = mace_data.AtomicData.from_config(
        config, z_table=tools.AtomicNumberTable([6, 14]), cutoff=4.0
    )
    batch = next(iter(torch_geometric.dataloader.DataLoader(
        [atomic], batch_size=1, shuffle=False)))
    with torch.no_grad():
        out = model(batch.to_dict(), training=False, compute_force=False)
    return out["carrier_alpha"].numpy(), np.asarray(ext.STATES[state]["counts"], float)


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--models", nargs="+", required=True, help="name:label pairs")
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_full")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    torch.set_default_dtype(torch.float64)

    for item in args.models:
        name, _, label = item.partition(":")
        path = Path.home() / "runs" / name / f"{name}.model"
        if not path.is_file():
            print(f"missing {path}")
            continue
        model = torch.load(path, map_location="cpu", weights_only=False).double().eval()
        alpha, counts = alpha_for(model, args.data_dir)
        print(f"\n=== {label or name} ===")
        print("  max |alpha_c - alpha_c'| between live channels:")
        for i, j in ((0, 1), (0, 3), (1, 3)):
            print(f"    {CHANNELS[i]:6s} vs {CHANNELS[j]:6s}  "
                  f"{np.abs(alpha[:, i] - alpha[:, j]).max():.3e}")
        carrier_shape = (alpha * (SIGNS * counts)).sum(axis=1)
        print(f"  q^carrier shape, max|.| = {np.abs(carrier_shape).max():.3e}"
              f"   ({'COLLAPSED -- no electron-hole separation' if np.abs(carrier_shape).max() < 1e-6 else 'live'})")


if __name__ == "__main__":
    main()
