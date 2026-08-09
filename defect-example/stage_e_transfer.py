#!/usr/bin/env python3
"""Stage E deliverable: how much energy does ``Delta E_SR`` give up to the long-range term?

The long-range branch is not inert at ``q = 0``. ``q_i^carrier = a (alpha_i^e - alpha_i^h)``
sums to zero but is pointwise non-zero wherever the electron and hole distributions differ,
and its self-term is the electron-hole interaction; ``q_i^pol`` carries a factor
``sum_c n_c``. Part of the observable belongs there, and with the branch off ``Delta E_SR``
has no choice but to absorb it -- fitting the same number through the wrong term.

Two measurements on identical frames:

1. **Within the long-range model**, the size of ``E_LR`` relative to ``Delta E_SR``. This
   is what the branch is actually contributing.
2. **Across models**, ``Delta E_SR`` with the branch on versus off. This is the transfer:
   how much the short-range correction *stopped* claiming once the long-range term was
   available to claim it. A material transfer is the expected result; a null means the
   e-h physics was never in the fit to begin with, or the branch is not reaching it.

    python stage_e_transfer.py --lr-model ~/runs/e_lr_s2/e_lr_s2.model \\
                               --sr-model ~/runs/e1c_anneal_s2/e1c_anneal_s2.model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mace import data, tools
from mace.data.defects import load_band_edges
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


@torch.no_grad()
def evaluate(model, configs, z_table):
    """Per-frame Delta E_SR, E_LR and the total correction."""
    dataset = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=CUTOFF) for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=4, shuffle=False
    )
    sr, lr, total = [], [], []
    for batch in loader:
        out = model(batch.to_dict(), training=False, compute_force=False)
        delta_sr = out["delta_sr_energy"].detach()
        correction = out["correction_energy"].detach()
        sr.append(delta_sr)
        # correction_energy = delta_sr + delta_lr, so the remainder is the long-range part.
        lr.append(correction - delta_sr)
        total.append(correction)
    return (torch.cat(sr), torch.cat(lr), torch.cat(total))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--lr-model", type=Path, required=True)
    parser.add_argument("--sr-model", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument("--n-frames", type=int, default=24)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    z_table = tools.AtomicNumberTable([6, 14])
    _, configs = data.load_from_xyz(
        file_path=str(args.data_dir / "valid.xyz"),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.data_dir / "band_edges.json"),
    )
    defect = [c for c in configs if int(sum(c.properties["carrier_counts"])) > 0]
    defect = defect[: args.n_frames]
    if not defect:
        raise SystemExit("no carrier-bearing frames in the validation split")

    lr_model = torch.load(args.lr_model, map_location="cpu", weights_only=False).float()
    lr_model.eval()
    sr_on, lr_on, total_on = evaluate(lr_model, defect, z_table)

    print(f"frames: {len(defect)} carrier-bearing validation frames")
    print(f"long-range model: {args.lr_model.name}")
    if getattr(lr_model, "use_long_range", False):
        amplitude = None
        try:
            amplitude = float(
                torch.nn.functional.softplus(
                    list(lr_model.latent_charges.amplitude.modules())[-1].bias
                )
            )
        except Exception:  # noqa: BLE001
            pass
        if amplitude is not None:
            print(
                f"  screening amplitude a = {amplitude:.6f} "
                f"(1/a^2 = {1 / amplitude**2:.3f}; frozen, so this is the input gauge, "
                "NOT a fitted result)"
            )
    print()
    print("1. within the long-range model, per frame (eV):")
    print(f"   |Delta E_SR|  mean {sr_on.abs().mean():.4f}   max {sr_on.abs().max():.4f}")
    print(f"   |E_LR|        mean {lr_on.abs().mean():.4f}   max {lr_on.abs().max():.4f}")
    share = float(lr_on.abs().mean() / (sr_on.abs().mean() + lr_on.abs().mean() + 1e-12))
    print(f"   long-range share of the correction magnitude: {share:.1%}")

    if args.sr_model is None:
        return
    sr_model = torch.load(args.sr_model, map_location="cpu", weights_only=False).float()
    sr_model.eval()
    sr_off, lr_off, total_off = evaluate(sr_model, defect, z_table)

    print()
    print("2. transfer: Delta E_SR with the branch on vs off (eV):")
    print(f"   branch off  |Delta E_SR| mean {sr_off.abs().mean():.4f}")
    print(f"   branch on   |Delta E_SR| mean {sr_on.abs().mean():.4f}")
    shift = float((sr_on - sr_off).abs().mean())
    print(f"   mean |change in Delta E_SR| = {shift:.4f} eV")
    print(f"   mean |change in the total correction| = "
          f"{float((total_on - total_off).abs().mean()):.4f} eV")
    if shift < 0.01:
        print(
            "\n   NULL: the short-range term did not give anything up. Either the e-h\n"
            "   physics was never in the fit, or the long-range branch is not reaching it.\n"
            "   Note the two models are independently trained, so seed-to-seed variation\n"
            "   in Delta E_SR sets the floor on what this can resolve."
        )
    else:
        print(
            f"\n   Material transfer ({shift:.3f} eV). Expected: part of the observable\n"
            "   belongs to the electron-hole term and Delta E_SR was absorbing it."
        )


if __name__ == "__main__":
    main()
