#!/usr/bin/env python3
"""Inspect a trained MACEDefect checkpoint's carrier correction, per channel.

Serves two purposes at once:

* the **Stage 1 gate** ("participation ratio moves measurably off N in a training run");
* an **autopsy** discriminating why a run is or is not stuck. Three signatures:

  - participation << N but the attention sits on bulk atoms -> lock-in on wrong sites
    (the failure plan section 2.3's structured init exists to prevent);
  - participation ~ N with std(u) still O(0.1) -> delocalised mean-fitting, i.e. the
    correction is carried by the *level* of u rather than by its shape;
  - std(u) -> 0 -> the fit has crushed u back toward the zero-init state.

Channel accounting matters here. In the 4H-SiC dataset only three of the four channels
are ever occupied -- ``e_maj`` and ``h_min`` (ground and excited) and ``e_min`` (excited
only) -- and the synthetic anchors pin only ``e_maj`` and ``h_min``. ``e_min`` is the
channel that distinguishes the two states and it has **no anchor**, so its bulk level is
an unconstrained direction. This script prints per channel so that can be checked.

    python autopsy_checkpoint.py ~/runs/beta_noinit/checkpoints/*.pt
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch

from mace import data
from mace.data.defects import CARRIER_CHANNELS, load_band_edges
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


def coordination_deficit(positions: np.ndarray, cell: np.ndarray, bond: float = 2.3):
    """Atoms with fewer near neighbours than the bulk -- the defect's first shell.

    Diagnostic only. Si-C is 1.89 A and the second shell ~3.1 A, so 2.3 A separates them.
    """
    inverse = np.linalg.inv(cell)
    fractional = positions @ inverse
    delta = fractional[:, None, :] - fractional[None, :, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ cell, axis=-1)
    np.fill_diagonal(distances, np.inf)
    degree = (distances < bond).sum(axis=1)
    return degree < degree.max()


def build_architecture(args, state_dict):
    """Reconstruct MACEDefect so a mid-run state_dict can be loaded into it."""
    from e3nn import o3

    from mace import modules, tools
    from mace.modules.defect_models import MACEDefect

    z_table = tools.AtomicNumberTable([6, 14])
    hidden = (
        f"{args.channels}x0e"
        if args.max_l == 0
        else f"{args.channels}x0e + {args.channels}x1o"
    )
    model = MACEDefect(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=5,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, len(z_table))),  # [n_heads, n_elements]
        # NOT a buffer (blocks.py sets it as a plain attribute), so it does *not* come
        # back from the state dict -- a wrong value here silently corrupts every feature.
        # Take it from the run's log line "Average number of neighbors: ...".
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args.correlation,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        carrier_feature_dim=args.carrier_feature_dim,
        counter_embedding_dim=args.counter_embedding_dim,
        carrier_mlp_hidden=args.carrier_mlp_hidden,
        use_long_range=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    blocking = [k for k in list(missing) + list(unexpected) if "carrier" in k or "counter" in k]
    if blocking:
        raise SystemExit(
            "the checkpoint does not match the architecture given on the command line; "
            f"mismatched correction keys: {blocking[:6]}"
        )
    if missing or unexpected:
        print(
            f"note: {len(missing)} missing / {len(unexpected)} unexpected keys outside "
            "the correction branch (harmless for this measurement)"
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=here / "dataset_beta" / "valid.xyz")
    parser.add_argument(
        "--band-edges", type=Path, default=here / "dataset_beta" / "band_edges.json"
    )
    parser.add_argument("--n-frames", type=int, default=6)
    # Only needed for mid-run checkpoints, which store a state_dict rather than a module.
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--max-l", type=int, default=0)
    parser.add_argument("--r-max", type=float, default=4.0)
    parser.add_argument("--num-radial-basis", type=int, default=4)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--carrier-feature-dim", type=int, default=32)
    parser.add_argument("--counter-embedding-dim", type=int, default=32)
    parser.add_argument("--carrier-mlp-hidden", type=int, default=64)
    parser.add_argument(
        "--avg-num-neighbors",
        type=float,
        default=None,
        help="required for mid-run checkpoints; read it off the run log line "
        "'Average number of neighbors: ...'. It is not stored in the state dict",
    )
    parser.add_argument("--max-ell", type=int, default=3)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = state["model"] if isinstance(state, dict) and "model" in state else state
    if isinstance(model, dict):
        # Mid-run checkpoints hold a state_dict; only the end-of-run *.model file holds
        # the module. Rebuild the architecture and load the weights into it. Buffers such
        # as the atomic energies and the scale/shift come from the state dict, so the
        # placeholders passed here are overwritten.
        if args.avg_num_neighbors is None:
            raise SystemExit(
                "this is a mid-run checkpoint (a state_dict), so the architecture must "
                "be rebuilt; pass --avg-num-neighbors, taken from the run log line "
                "'Average number of neighbors: ...'. It is a plain attribute, not a "
                "buffer, so a wrong value silently corrupts every feature."
            )
        model = build_architecture(args, model)
    model.eval()
    dtype = next(model.parameters()).dtype
    torch.set_default_dtype(dtype)

    _, configs = data.load_from_xyz(
        file_path=str(args.data),
        key_specification=keyspec(),
        band_edges=load_band_edges(args.band_edges),
    )
    excited = [c for c in configs if c.config_type == "paired_ex"][: args.n_frames]
    if not excited:
        raise SystemExit("no paired_ex frames in the given dataset")

    z_table = state.get("z_table") if isinstance(state, dict) else None
    if z_table is None:
        from mace import tools

        z_table = tools.AtomicNumberTable([6, 14])

    print(f"checkpoint: {args.checkpoint}")
    print(f"frames: {len(excited)} paired_ex from {args.data}\n")

    rows = collections.defaultdict(list)
    errors = []
    for config in excited:
        atomic = data.AtomicData.from_config(config, z_table=z_table, cutoff=CUTOFF)
        batch = next(
            iter(
                torch_geometric.dataloader.DataLoader(
                    dataset=[atomic], batch_size=1, shuffle=False
                )
            )
        )
        out = model(batch.to_dict(), training=False)
        alpha = out["carrier_alpha"].detach()
        u = out["carrier_readouts"].detach()
        n_atoms = alpha.shape[0]

        defect_mask = coordination_deficit(
            np.asarray(config.positions), np.asarray(config.cell)
        )
        errors.append(
            float(out["delta_energy"][0]) - float(config.properties["delta_energy"])
        )
        for index, name in enumerate(CARRIER_CHANNELS):
            a = alpha[:, index]
            rows[name].append(
                {
                    "participation": float(1.0 / a.pow(2).sum()),
                    "n_atoms": n_atoms,
                    "u_std": float(u[:, index].std()),
                    "u_mean": float(u[:, index].mean()),
                    "alpha_on_defect": float(a[defect_mask].sum()),
                    "n_defect": int(defect_mask.sum()),
                }
            )

    counts = np.asarray(excited[0].properties["carrier_counts"])
    print(
        f"{'channel':>8} {'occupied':>9} {'anchored':>9} {'partic.':>9} {'partic./N':>10} "
        f"{'std(u)':>9} {'mean(u)':>10} {'a_defect':>9}"
    )
    # Which channels the dataset's anchors actually pin (see module docstring).
    anchored = {"e_maj": True, "e_min": False, "h_maj": False, "h_min": True}
    for index, name in enumerate(CARRIER_CHANNELS):
        entries = rows[name]
        mean = lambda k: float(np.mean([e[k] for e in entries]))  # noqa: E731
        print(
            f"{name:>8} {str(bool(counts[index])):>9} {str(anchored[name]):>9} "
            f"{mean('participation'):>9.1f} "
            f"{mean('participation') / mean('n_atoms'):>10.3f} "
            f"{mean('u_std'):>9.4f} {mean('u_mean'):>10.4f} "
            f"{mean('alpha_on_defect'):>9.4f}"
        )

    n_defect = rows[CARRIER_CHANNELS[0]][0]["n_defect"]
    n_atoms = rows[CARRIER_CHANNELS[0]][0]["n_atoms"]
    print(
        f"\ndefect first shell: {n_defect} atoms of {n_atoms} "
        f"(uniform attention would put {n_defect / n_atoms:.4f} of alpha there)"
    )
    print(
        f"delta_energy error over these frames: "
        f"RMSE {np.sqrt(np.mean(np.square(errors))) * 1000:.2f} meV"
    )
    print(
        "\nRead: participation/N ~ 1 means the attention is still uniform. a_defect well "
        "above\nthe uniform share means it has found the defect; well below means it has "
        "locked onto\nbulk atoms instead."
    )


if __name__ == "__main__":
    main()
