#!/usr/bin/env python3
"""Energy and force parity plots, split by carrier state, after the NEP paper's figure.

Three subsets are shown separately -- pristine, defect ground state, defect excited state
-- because the whole point of the model is that it distinguishes them. In the original
figure those are three *separately trained* NEP models with the electronic state baked
into the atomic species; here they are one model told the state through its carrier
counters, so the three clusters come from a single set of weights.

Each subset is drawn with a fixed offset along x, purely so the clusters do not overlap,
with its own dashed parity line ``y = x - offset``. Offsets are cosmetic and are stated
in the legend.

Energies are per atom and are **band-edge referenced**: the excited state carries two
gaps, the ground state one, so the raw and referenced scales differ by a constant per
state. Forces need no referencing.

    python plot_parity.py --model ~/runs/b_128ch_L1_s1/b_128ch_L1_s1.model
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

# (label, colour, matching config_type prefixes)
GROUPS = [
    ("ideal", "#4d4d4d", ("ideal",)),
    ("defect ground state", "#2c6fbb", ("paired_gs", "unpaired_gs")),
    ("defect excited state", "#c0392b", ("paired_ex", "unpaired_ex")),
]


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


def predict(model, configs, z_table, device, batch_size=4):
    """Per-atom energies and per-component forces, reference and predicted.

    Deliberately NOT under ``torch.no_grad``: the forces are an autograd derivative of
    the energy, so disabling the graph does not merely slow this down, it makes the model
    unable to produce forces at all.
    """
    dataset = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=CUTOFF) for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False
    )
    e_ref, e_pred, f_ref, f_pred = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.to_dict(), training=False, compute_force=True)
        counts = batch.ptr[1:] - batch.ptr[:-1]
        e_ref.append((batch.energy.detach() / counts).cpu())
        e_pred.append((out["energy"].detach() / counts).cpu())
        f_ref.append(batch.forces.detach().reshape(-1).cpu())
        f_pred.append(out["forces"].detach().reshape(-1).cpu())
    return (
        torch.cat(e_ref).numpy(),
        torch.cat(e_pred).numpy(),
        torch.cat(f_ref).numpy(),
        torch.cat(f_pred).numpy(),
    )


def stats(reference: np.ndarray, prediction: np.ndarray):
    """R^2, RMSE and the mean signed residual.

    The bias is returned because for this model it is most of the error: the energy
    residuals are dominated by a single per-atom constant rather than by scatter, so an
    RMSE quoted alone would misdescribe what is wrong.
    """
    residual = prediction - reference
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    variance = float(np.var(reference))
    r2 = 1.0 - float(np.mean(residual**2)) / variance if variance > 0 else float("nan")
    return r2, rmse, bias


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Arial is not installed on this machine; Liberation Sans is metric-compatible with
    # it and is the conventional substitute, so the figure matches the reference style
    # and still renders as Arial wherever Arial exists.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid"],
        help="held-out by default; the original figure shows the fit itself",
    )
    parser.add_argument("--out", type=Path, default=here / "parity.png")
    parser.add_argument("--energy-offset", type=float, default=0.03,
                        help="eV/atom shift between subsets along x, cosmetic only")
    parser.add_argument("--force-offset", type=float, default=0.30,
                        help="eV/A shift between subsets along x, cosmetic only")
    parser.add_argument("--drop-tiny-cells", type=int, default=0,
                        help="exclude cells with fewer than this many atoms; the 8-atom "
                             "primitive cell sits well off the supercell energy scale")
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
    if args.drop_tiny_cells:
        configs = [c for c in configs
                   if len(c.atomic_numbers) >= args.drop_tiny_cells]

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    energy_rows, force_rows = [], []

    for index, (label, colour, prefixes) in enumerate(GROUPS):
        subset = [c for c in configs if str(c.config_type).startswith(prefixes)]
        if not subset:
            continue
        e_ref, e_pred, f_ref, f_pred = predict(model, subset, z_table, device)

        e_shift = index * args.energy_offset
        f_shift = index * args.force_offset

        axes[0].plot(e_ref + e_shift, e_pred, "o", ms=2.2, alpha=0.6,
                     color=colour, mew=0)
        axes[1].plot(f_ref + f_shift, f_pred, "o", ms=0.9, alpha=0.14,
                     color=colour, mew=0)

        # One parity line per subset, y = x - offset, so a point sitting on its own line
        # is an exact prediction despite the cosmetic shift.
        for axis, shift, values in (
            (axes[0], e_shift, e_ref), (axes[1], f_shift, f_ref)
        ):
            span = np.array([values.min(), values.max()])
            pad = 0.06 * (span[1] - span[0])
            span = np.array([span[0] - pad, span[1] + pad])
            axis.plot(span + shift, span, "--", lw=0.7, color=colour, alpha=0.6)

        r2_e, rmse_e, bias_e = stats(e_ref, e_pred)
        r2_f, rmse_f, bias_f = stats(f_ref, f_pred)
        energy_rows.append((label, colour, r2_e, rmse_e * 1e3))
        force_rows.append((label, colour, r2_f, rmse_f * 1e3))
        print(f"{label:22s} E: R2={r2_e:8.5f} RMSE={rmse_e*1e3:7.2f} bias={bias_e*1e3:+7.2f} meV/atom"
              f" | F: R2={r2_f:.4f} RMSE={rmse_f*1e3:5.1f} bias={bias_f*1e3:+5.1f} meV/A")

    def draw_table(axis, rows, unit, r2_fmt, rmse_fmt):
        """R^2 and RMSE only, laid out like the reference figure."""
        x_label, x_r2, x_rmse = 0.60, 0.815, 0.995
        axis.text(x_r2, 0.30, "R$^2$", transform=axis.transAxes,
                  fontsize=9, fontweight="bold", ha="right")
        axis.text(x_rmse, 0.30, "RMSE", transform=axis.transAxes,
                  fontsize=9, fontweight="bold", ha="right")
        axis.text(x_rmse, 0.235, unit, transform=axis.transAxes,
                  fontsize=7.5, ha="right")
        for row_index, (label, colour, r2, rmse) in enumerate(rows):
            y = 0.165 - row_index * 0.068
            axis.text(x_label, y, label, transform=axis.transAxes,
                      fontsize=8, color=colour, ha="right")
            axis.text(x_r2, y, format(r2, r2_fmt), transform=axis.transAxes,
                      fontsize=8, ha="right")
            axis.text(x_rmse, y, format(rmse, rmse_fmt), transform=axis.transAxes,
                      fontsize=8, ha="right")

    axes[0].set_xlabel("DFT energy (eV/atom)", fontsize=9)
    axes[0].set_ylabel("MACEDefect energy (eV/atom)", fontsize=9)
    axes[0].text(0.04, 0.93, "a)", transform=axes[0].transAxes, fontsize=10)
    draw_table(axes[0], energy_rows, "(meV/atom)", ".3f", ".1f")

    axes[1].set_xlabel("DFT force (eV/Å)", fontsize=9)
    axes[1].set_ylabel("MACEDefect force (eV/Å)", fontsize=9)
    axes[1].text(0.04, 0.93, "b)", transform=axes[1].transAxes, fontsize=10)
    draw_table(axes[1], force_rows, "(meV/Å)", ".4f", ".0f")

    for axis in axes:
        axis.set_box_aspect(1)

    figure.tight_layout(pad=0.6, w_pad=1.6)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
