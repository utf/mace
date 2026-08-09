#!/usr/bin/env python3
"""What fraction of the defect loss is each term actually worth?

Choosing ``energy_weight`` by eye is unreliable here because the terms are not
commensurate. ``base_energy`` and the totals energy are normalised **per atom** and then
squared, so they are divided by ``N^2`` -- about 1e5 for a 300-atom cell -- while
``delta_energy`` is a per-frame quantity that is not divided at all. A weight of 1.0 on a
per-atom term is therefore not "a bit smaller" than a weight of 10 on a per-frame term,
it is smaller by five orders of magnitude before the weights are even applied.

This evaluates each term separately through the real ``DefectLoss`` code path -- one loss
object per term with every other weight zeroed -- and reports the budget. It then shows
what the budget becomes at a proposed set of weights, holding the model fixed, which is
the quantity you actually want when picking a multiplier.

    python loss_term_budget.py --model ~/runs/b_128ch_L0_s1/b_128ch_L0_s1.model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mace import data, tools
from mace.data.defects import load_band_edges
from mace.modules.loss import DefectLoss
from mace.tools import torch_geometric

CUTOFF = 4.0

# The production Stage B weights.
PRODUCTION = {
    "energy_weight": 1.0,
    "forces_weight": 100.0,
    "delta_energy_weight": 10.0,
    "delta_forces_weight": 100.0,
    "total_energy_weight": 0.25,
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--propose",
        nargs="*",
        default=["energy_weight=10", "total_energy_weight=10"],
        help="weight=value pairs to re-budget under, e.g. energy_weight=100",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model = model.float()
    model.eval()
    z_table = tools.AtomicNumberTable([6, 14])

    _, configs = data.load_from_xyz(
        file_path=str(args.data_dir / f"{args.split}.xyz"),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.data_dir / "band_edges.json"),
    )
    dataset = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=CUTOFF) for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False
    )

    # One loss per term, every other weight zeroed, so each is evaluated by the code that
    # actually runs in training rather than by a reimplementation. The all-zero object
    # isolates the regularisation, which is added unconditionally and must be subtracted.
    zeros = {k: 0.0 for k in PRODUCTION}
    losses = {"regularisation": DefectLoss(**zeros).to(device)}
    for name, value in PRODUCTION.items():
        losses[name] = DefectLoss(**{**zeros, name: value}).to(device)

    totals = {name: 0.0 for name in losses}
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.to_dict(), training=True, compute_force=True)
        for name, loss_fn in losses.items():
            totals[name] += float(loss_fn(batch, out).detach())
        batches += 1

    base = totals.pop("regularisation") / batches
    means = {name: value / batches - base for name, value in totals.items()}
    grand = sum(means.values())

    print(f"model: {args.model.name}   split: {args.split}   frames: {len(configs)}")
    print(f"regularisation (constant, excluded from the budget): {base:.3e}\n")
    print(f"{'term':22s} {'weight':>8s} {'contribution':>13s} {'share':>8s} {'raw MSE':>11s}")
    raw = {}
    for name, value in sorted(means.items(), key=lambda kv: -kv[1]):
        raw[name] = value / PRODUCTION[name] if PRODUCTION[name] else float("nan")
        print(f"{name:22s} {PRODUCTION[name]:8.2f} {value:13.3e} "
              f"{value / grand:7.2%} {raw[name]:11.3e}")
    print(f"{'TOTAL':22s} {'':8s} {grand:13.3e} {1.0:7.2%}")

    energy_terms = ("energy_weight", "total_energy_weight")
    share = sum(means[k] for k in energy_terms) / grand
    print(f"\nabsolute-energy terms ({' + '.join(energy_terms)}): {share:.3%} of the loss")
    print("delta_energy is a paired difference, so it is blind to a constant offset "
          "and cannot correct one at any weight.")

    if args.propose:
        proposed = dict(PRODUCTION)
        for item in args.propose:
            key, _, value = item.partition("=")
            if key not in proposed:
                raise SystemExit(f"unknown weight {key!r}; choose from {list(PRODUCTION)}")
            proposed[key] = float(value)
        # Holding the model fixed, each contribution scales linearly with its weight.
        new = {name: raw[name] * proposed[name] for name in means}
        new_grand = sum(new.values())
        print(f"\nunder {args.propose}, holding the model fixed:")
        print(f"{'term':22s} {'weight':>8s} {'contribution':>13s} {'share':>8s}")
        for name, value in sorted(new.items(), key=lambda kv: -kv[1]):
            print(f"{name:22s} {proposed[name]:8.2f} {value:13.3e} {value / new_grand:7.2%}")
        share = sum(new[k] for k in energy_terms) / new_grand
        print(f"\nabsolute-energy terms would be {share:.3%} of the loss")


if __name__ == "__main__":
    main()
