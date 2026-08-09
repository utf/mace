#!/usr/bin/env python3
"""Is cuEquivariance conversion safe for a MACEDefect with the long-range branch on?

``run_train`` blocked `--enable_cueq` together with `use_long_range` because the
latent-Ewald branch was untested under conversion. The conversion is not a wrapper: it
**rebuilds** the model with ``source_model.__class__(**extract_config_mace_model(source))``
and then transfers weights, so anything the config extractor omits silently reverts to its
constructor default in the converted model.

An audit of ``MACEDefect.__init__`` against that extractor found four omissions, of which
exactly one diverges in practice:

* ``freeze_amplitude`` -- default False, but stage E passes **True**. Converting a
  frozen-amplitude model therefore produced a *trainable* screening amplitude, which then
  reads as a fitted screening constant instead of the input gauge it is meant to be. This
  is the real hazard the block was protecting against.
* ``high_precision_softmax`` (True), ``zero_u_init`` (init-only), ``correction_trunk``
  (raises for anything but "shared") -- all equal to their defaults in every run, so they
  were no-ops. Captured anyway so they cannot become the next instance of this.

This script is the evidence for lifting the block. It checks, on a real long-range model:

1. the four arguments survive the round trip;
2. the screening amplitude is bit-identical and still frozen;
3. the long-range branch is still present and still contributing (a converted model that
   quietly lost the branch would otherwise pass an energy check that only looks at totals);
4. energy, forces and the correction decomposition agree with e3nn to tight tolerance.

    python verify_cueq_long_range.py --model ~/runs/e_lr_s2/e_lr_s2.model
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

CUTOFF = 4.0


def keyspec():
    from mace import data

    spec = data.KeySpecification()
    spec.info_keys.update({
        "energy": "REF_energy", "carrier_counts": "carrier_counts",
        "multiplicity": "multiplicity", "host": "host", "pair_id": "pair_id",
        "stress": "REF_stress", "head": "head",
    })
    spec.arrays_keys.update({"forces": "REF_forces"})
    return spec


def evaluate(model, batch):
    out = model(batch.to_dict(), training=False, compute_force=True)
    return {
        key: out[key].detach().double().cpu().numpy()
        for key in ("energy", "forces", "delta_sr_energy", "correction_energy",
                    "carrier_alpha")
        if out.get(key) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="eV (energies) / eV/A (forces); float32 kernels differ in "
                             "summation order, so this is not expected to be exact")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.data.defects import load_band_edges
    from mace.tools import torch_geometric

    if not torch.cuda.is_available():
        raise SystemExit("cuEquivariance kernels need a GPU")
    device = "cuda"
    torch.set_default_dtype(torch.float32)

    source = torch.load(args.model, map_location=device, weights_only=False).to(device)
    source = source.float()
    source.eval()
    if not getattr(source, "use_long_range", False):
        raise SystemExit(f"{args.model} has no long-range branch; nothing to verify")

    # Stage E freezes the amplitude. If this model was trained before that wiring was
    # fixed it may not be marked frozen, in which case set it here -- the property under
    # test is whether the flag SURVIVES conversion, not how it got there.
    if not getattr(source, "freeze_amplitude", False):
        print("note: source model is not marked frozen; setting the flag to test the "
              "round trip")
        source.freeze_amplitude = True
        source.latent_charges.freeze_amplitude = True

    print(f"source {args.model.name}: use_long_range={source.use_long_range}, "
          f"freeze_amplitude={source.freeze_amplitude}\n")

    converted = run_e3nn_to_cueq(deepcopy(source), device=device)
    converted.eval()

    failures = []

    # ---- 1. arguments survive the round trip -------------------------------------------
    print("1. constructor arguments after conversion")
    for name, default in (("freeze_amplitude", False), ("high_precision_softmax", True),
                          ("zero_u_init", True), ("correction_trunk", "shared"),
                          ("use_long_range", False)):
        before = getattr(source, name, default)
        after = getattr(converted, name, default)
        ok = before == after
        failures.append(f"{name}: {before} -> {after}") if not ok else None
        print(f"   {name:24s} {str(before):8s} -> {str(after):8s} "
              f"{'ok' if ok else 'LOST'}")

    # ---- 2. the amplitude itself -------------------------------------------------------
    print("\n2. screening amplitude")
    def amplitude_of(model):
        bias = list(model.latent_charges.amplitude.modules())[-1].bias
        return float(torch.nn.functional.softplus(bias)), bool(bias.requires_grad)

    a_source, grad_source = amplitude_of(source)
    a_target, grad_target = amplitude_of(converted)
    print(f"   a = {a_source:.8f} -> {a_target:.8f}   "
          f"(1/a^2 = {1/a_source**2:.4f} -> {1/a_target**2:.4f})")
    print(f"   requires_grad {grad_source} -> {grad_target}")
    if abs(a_source - a_target) > 1e-6:
        failures.append(f"amplitude changed: {a_source} -> {a_target}")
    if grad_target and not grad_source:
        failures.append("amplitude became trainable through conversion")

    # ---- 3 & 4. numerical agreement ----------------------------------------------------
    z_table = tools.AtomicNumberTable([6, 14])
    _, configs = mace_data.load_from_xyz(
        file_path=str(args.data_dir / "valid.xyz"),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.data_dir / "band_edges.json"),
    )
    charged = [c for c in configs
               if int(sum(c.properties["carrier_counts"])) > 0][: args.frames]
    dataset = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=CUTOFF)
               for c in charged]
    loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=len(dataset),
                                                   shuffle=False)
    batch = next(iter(loader)).to(device)

    reference = evaluate(source, batch)
    candidate = evaluate(converted, batch)

    print(f"\n3. long-range branch still contributing ({len(charged)} charged frames)")
    lr = reference["correction_energy"] - reference["delta_sr_energy"]
    lr_after = candidate["correction_energy"] - candidate["delta_sr_energy"]
    print(f"   |E_LR| e3nn {np.abs(lr).mean():.6f} eV   cueq {np.abs(lr_after).mean():.6f} eV")
    if np.abs(lr_after).mean() < 1e-9:
        failures.append("converted model's long-range contribution is identically zero")

    print("\n4. numerical agreement")
    for key in ("energy", "forces", "delta_sr_energy", "correction_energy",
                "carrier_alpha"):
        if key not in reference or key not in candidate:
            continue
        difference = float(np.abs(reference[key] - candidate[key]).max())
        scale = float(np.abs(reference[key]).max()) or 1.0
        ok = difference <= max(args.tol, 1e-6 * scale)
        print(f"   {key:20s} max |diff| = {difference:.3e}   "
              f"(max |value| {scale:.3e})  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{key} differs by {difference:.3e}")

    print()
    if failures:
        print("FAILED:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("PASS: cuEq conversion preserves the long-range branch and its frozen amplitude")


if __name__ == "__main__":
    main()
