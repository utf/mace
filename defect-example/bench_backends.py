#!/usr/bin/env python3
"""Time and peak memory for MACEDefect under e3nn / cuEquivariance / OpenEquivariance.

Memory matters as much as speed here: at 128 channels with ``max_L=1`` the e3nn model
runs out of memory on a 16 GB card at batch 8, so the achievable batch size is part of
the comparison, not a footnote.

Both backends convert the **trunk** only. The correction heads are dense MLPs and are
already as fast as they get -- measured separately, the correction's forward pass costs
nothing and its whole overhead is the two extra ``autograd.grad`` calls for
``delta_forces``.

    CUDA_HOME=$CONDA_PREFIX python bench_backends.py --channels 128 --max-l 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import load_band_edges
from mace.modules.defect_models import MACEDefect
from mace.tools import torch_geometric


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


def build(channels: int, max_l: int, z_table) -> MACEDefect:
    torch.manual_seed(0)
    hidden = f"{channels}x0e" if max_l == 0 else f"{channels}x0e + {channels}x1o"
    return MACEDefect(
        r_max=4.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, len(z_table))),
        avg_num_neighbors=26.9,
        atomic_numbers=z_table.zs,
        correlation=3,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        carrier_feature_dim=32,
        counter_embedding_dim=16,
        carrier_mlp_hidden=32,
        use_long_range=False,
    )


def measure(model, batch, repeats: int = 12):
    """Returns (ms per training step, peak MiB) or (None, None) on OOM."""
    device = next(model.parameters()).device
    payload = batch.to_dict()
    try:
        for _ in range(4):
            out = model(payload, training=True)
            (out["energy"].sum() + out["forces"].sum()).backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        for _ in range(repeats):
            out = model(payload, training=True)
            (out["energy"].sum() + out["forces"].sum()).backward()
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / repeats * 1000
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        return elapsed, peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def make_batch(configs, z_table, size: int):
    chosen = (configs * (size // len(configs) + 1))[:size]
    dataset = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=4.0) for c in chosen
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=size, shuffle=False
    )
    return next(iter(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_beta")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max-l", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16])
    args = parser.parse_args()

    device = "cuda"
    _, configs = data.load_from_xyz(
        file_path=str(args.data_dir / "train.xyz"),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.data_dir / "band_edges.json"),
    )
    defect = [c for c in configs if int(sum(c.properties["carrier_counts"])) > 0]
    z_table = tools.AtomicNumberTable([6, 14])

    backends = [("e3nn", None)]
    try:
        from mace.cli.convert_e3nn_cueq import run as to_cueq

        backends.append(("cueq", to_cueq))
    except ImportError:
        print("cuequivariance not available, skipping")
    try:
        from mace.cli.convert_e3nn_oeq import run as to_oeq

        backends.append(("oeq", to_oeq))
    except ImportError:
        to_oeq = None
        print("openequivariance not available, skipping")

    if to_oeq is not None and len(backends) == 3:
        from mace.cli.convert_e3nn_cueq import run as to_cueq_again

        def to_hybrid(model, device):
            """cuEquivariance first, then OpenEquivariance on top.

            The order is load-bearing. ``extract_config_mace_model`` carries
            ``cueq_config`` through but not ``oeq_config``, so converting cueq -> oeq
            keeps both and produces a genuine hybrid, while oeq -> cueq would silently
            drop the oeq half and give a plain cueq model that merely looks converted.
            """
            return to_oeq(to_cueq_again(model, device=device), device=device)

        backends.append(("cueq+oeq", to_hybrid))

    print(f"MACEDefect {args.channels} channels, max_L={args.max_l}, float32, "
          f"{len(defect[0].atomic_numbers)}-atom cells\n")
    print(f"{'batch':>6} {'backend':>8} {'ms/step':>10} {'peak MiB':>10} {'ms/frame':>10}")
    for size in args.batches:
        batch = make_batch(defect, z_table, size).to(device)
        for name, convert in backends:
            model = build(args.channels, args.max_l, z_table).to(device)
            if convert is not None:
                try:
                    model = convert(model, device=device)
                except Exception as exc:  # noqa: BLE001
                    print(f"{size:>6} {name:>8}   conversion failed: {type(exc).__name__}")
                    continue
            ms, peak = measure(model, batch)
            if ms is None:
                print(f"{size:>6} {name:>8} {'OOM':>10} {'-':>10} {'-':>10}")
            else:
                print(f"{size:>6} {name:>8} {ms:>10.1f} {peak:>10.0f} {ms / size:>10.1f}")
            del model
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
