#!/usr/bin/env python3
"""Fix plan stage 0.5: the logit-gradient : u-gradient ratio versus cell size.

The concern behind this control is that the attention weights are a softmax over *all*
atoms of the cell, so ``alpha ~ 1/N`` on a delocalised field. If the gradient reaching the
logit networks were suppressed by ``N`` relative to the gradient reaching the energy
readout, the correction would find it progressively harder to localise as cells grow --
and every production cell is larger than the ones trained here.

What is actually expected: the ratio is *N-independent*, because

    d(Delta E_SR) / d(logit_j) = n_c alpha_j (u_j - <u>_alpha)

carries one factor of ``alpha_j`` but the sum over sites restores the normalisation, so
the ratio tracks the spread of ``u`` rather than the atom count. This script measures it
rather than assuming it, on the smallest and largest cells in the dataset.

Both cells are evaluated with the *same* model instance and the same initialisation, so
the comparison isolates the cell size.

    python stage05_gradient_ratio.py [--data dataset/train.xyz]
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import load_band_edges
from mace.modules.defect_models import MACEDefect
from mace.tools import torch_geometric

CUTOFF = 4.0


def keyspec() -> data.KeySpecification:
    spec = data.KeySpecification()
    spec.info_keys.update(
        {
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "multiplicity": "multiplicity",
            "host": "host",
            "pair_id": "pair_id",
            "stress": "REF_stress",
            "head": "head",
        }
    )
    spec.arrays_keys.update({"forces": "REF_forces"})
    return spec


def build_model(z_table: tools.AtomicNumberTable, seed: int, channels: int, max_l: int):
    torch.manual_seed(seed)
    return MACEDefect(
        r_max=CUTOFF,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(
            f"{channels}x0e" if max_l == 0 else f"{channels}x0e + {channels}x1o"
        ),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-7.9, -7.9]),
        avg_num_neighbors=12.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        carrier_feature_dim=16,
        counter_embedding_dim=16,
        carrier_mlp_hidden=32,
        use_long_range=False,
    )


def grad_norm(module: torch.nn.Module) -> float:
    return float(
        sum(
            p.grad.pow(2).sum()
            for p in module.parameters()
            if p.grad is not None
        )
        ** 0.5
    )


def measure(model, config, z_table) -> dict:
    model.zero_grad(set_to_none=True)
    atomic = data.AtomicData.from_config(config, z_table=z_table, cutoff=CUTOFF)
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                dataset=[atomic], batch_size=1, shuffle=False
            )
        )
    )
    out = model(batch.to_dict(), training=True)

    # The delta-energy residual is the term whose gradient we are apportioning.
    target = float(config.properties["delta_energy"])
    loss = torch.square(out["delta_energy"] - target).sum()
    loss.backward()

    alpha = out["carrier_alpha"].detach()
    n_atoms = alpha.shape[0]
    # Participation ratio: N for a uniform field, 1 for a fully localised one.
    participation = float(1.0 / alpha.pow(2).sum(dim=0).max())
    u = out["carrier_readouts"].detach()

    logit_grad = grad_norm(model.carrier_pooling.logit_readouts)
    u_grad = grad_norm(model.carrier_pooling.energy_readouts)
    return {
        "n_atoms": n_atoms,
        "logit_grad": logit_grad,
        "u_grad": u_grad,
        "ratio": logit_grad / u_grad if u_grad > 0 else float("nan"),
        "participation": participation,
        "u_spread": float(u.std()),
        "alpha_max": float(alpha.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here / "dataset" / "train.xyz")
    parser.add_argument(
        "--band-edges", type=Path, default=here / "dataset" / "band_edges.json"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--max-l", type=int, default=0)
    parser.add_argument(
        "--zero-u-init",
        action="store_true",
        help="restore the removed zero-init, to show the ratio collapse it causes",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    _, configs = data.load_from_xyz(
        file_path=str(args.data),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.band_edges),
    )

    # One excited-state frame per distinct cell size: those are the frames that carry a
    # delta target against a charged reference.
    by_size: dict = collections.OrderedDict()
    for config in configs:
        if config.config_type != "paired_ex":
            continue
        by_size.setdefault(len(config.atomic_numbers), config)
    sizes = sorted(by_size)
    if not sizes:
        raise SystemExit("no paired_ex frames found in the dataset")

    z_table = tools.AtomicNumberTable([6, 14])
    model = build_model(z_table, args.seed, args.channels, args.max_l)
    if args.zero_u_init:
        from mace.modules.defect_blocks import zero_last_layer

        for readout in model.carrier_pooling.energy_readouts:
            zero_last_layer(readout)

    print(
        f"model: {args.channels} channels, max_L={args.max_l}, seed={args.seed}, "
        f"zero_u_init={args.zero_u_init}"
    )
    print(
        f"\n{'N':>6} {'|g_logit|':>12} {'|g_u|':>12} {'ratio':>10} "
        f"{'partic.':>10} {'partic./N':>10} {'std(u)':>10} {'max alpha':>10}"
    )
    rows = []
    for size in sizes:
        result = measure(model, by_size[size], z_table)
        rows.append(result)
        print(
            f"{result['n_atoms']:>6} {result['logit_grad']:>12.4e} "
            f"{result['u_grad']:>12.4e} {result['ratio']:>10.4f} "
            f"{result['participation']:>10.1f} "
            f"{result['participation'] / result['n_atoms']:>10.3f} "
            f"{result['u_spread']:>10.4f} {result['alpha_max']:>10.5f}"
        )

    smallest, largest = rows[0], rows[-1]
    size_factor = largest["n_atoms"] / smallest["n_atoms"]
    if smallest["ratio"] > 0 and np.isfinite(largest["ratio"]):
        ratio_factor = largest["ratio"] / smallest["ratio"]
        print(
            f"\ncell size grew {size_factor:.2f}x; the gradient ratio changed "
            f"{ratio_factor:.3f}x"
        )
        print(
            "  N-independent (ratio_factor ~ 1) means the attention does not get harder "
            "to train\n  as cells grow, and the softmax-over-all-atoms form is not by "
            "itself the obstacle."
        )
    else:
        print(
            "\nthe LOGIT gradient is identically zero at every cell size -- the zero-init "
            "pathology.\n  u still receives gradient (see |g_u| above), so u climbs away "
            "from zero and the\n  attention eventually switches on; the cost is a delay, "
            "not a permanent trap."
        )


if __name__ == "__main__":
    main()
