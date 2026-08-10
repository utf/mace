#!/usr/bin/env python3
"""Convergence figure for a short-range model: deviation from the converged value.

The earlier version of this figure was incoherent and its output should be discarded. It
referenced deviations to the *mean over all cells* and then drew linear fits through them.
Those two choices contradict each other: a fit through mean-centred scatter crosses zero in
the middle of the data, so the lines ran to a non-zero value at ``1/N = 0`` while the legend
quoted a limit -- nothing converged at infinity because the construction never arranged for
it to. Worse, the "tail" being fitted was a few tenths of a meV of relaxation scatter.

What is true for this model:

* it is strictly short-ranged, receptive field ``r_max * n_interactions`` = 8 A, so once
  every cell edge exceeds 16 A the images are decoupled and the energy stops depending on
  cell size **exactly**. Cells below that are not finite-size data points, they are cells
  where the defect sees its own image inside the model's cutoff, and they are dropped;
* above the threshold there is no tail to extrapolate, so there is nothing to fit. The
  converged value is a constant and the scatter about it is the error bar.

So: deviations are referenced to the **converged value** (mean over the largest cells,
which agree among themselves), plotted with a zero line and a band at the residual scatter.
Points sitting in the band at small ``1/N`` is what convergence looks like; there are no
fitted lines because there is no model for them to represent.

    python replot_nep_style.py size_convergence_bsize.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ENERGIES = (
    ("formation_energy_eV", r"$E_\mathrm{f}$", "#8ab0e0", "o", 8),
    ("zpl_eV", r"$E_\mathrm{ZPL}$", "#e5c15a", "^", 7),
    ("vertical_absorption_eV", r"$E^\mathrm{abs}_\mathrm{vert}$", "#7fbf7b", "x", 7),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ladder", type=Path)
    parser.add_argument("--r-max", type=float, default=4.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--reference", default="largest",
                        choices=["largest", "mean"],
                        help="what defines the converged value. 'largest' anchors it to "
                             "the biggest cell, so that point sits at exactly zero and "
                             "the plot reads as an approach to it -- which is what "
                             "convergence operationally means. 'mean' averages the top "
                             "few, which suppresses relaxation noise but leaves no point "
                             "at zero and so obscures the claim")
    parser.add_argument("--agreement-cells", type=int, default=3,
                        help="how many of the largest cells to compare for the noise "
                             "estimate reported alongside")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.ladder.read_text())
    rows = sorted(data["rows"], key=lambda r: r["n_host"])
    cellpar = data.get("cellpar") or [3.0815, 0, 10.0605]
    a, c = float(cellpar[0]), float(cellpar[2])
    threshold = 2.0 * args.r_max * args.layers

    kept = [
        r for r in rows
        if min(a * r["repeat"][0], a * r["repeat"][1], c * r["repeat"][2]) > threshold
    ]
    dropped = [r["n_host"] for r in rows if r not in kept]
    if dropped:
        print(f"dropped, images overlap inside the {threshold:.0f} A receptive "
              f"field: N = {dropped}")
    if len(kept) < 2:
        raise SystemExit("not enough decoupled cells to define a converged value")

    reference = kept[-args.agreement_cells:]
    anchor = f"largest cell, N = {kept[-1]['n_host']}" if args.reference == "largest" \
        else f"mean of the largest {len(reference)} cells"
    print(f"converged value: {anchor}")
    print(f"  (agreement among N = {[r['n_host'] for r in reference]} reported as the "
          f"relaxation-noise floor)\n")
    limits, spreads = {}, {}
    for key, label, *_ in ENERGIES + (("delta_q_sqrtDa_A", r"$\Delta q$"),):
        values = np.array([r[key] for r in reference])
        limits[key] = float(kept[-1][key]) if args.reference == "largest" \
            else float(values.mean())
        spreads[key] = float(values.max() - values.min())
        unit, scale = ("sqrt(Da) A", 1.0) if key.startswith("delta_q") else ("eV", 1e3)
        residual = np.array([r[key] for r in kept]) - limits[key]
        print(f"  {label:28s} {limits[key]:9.4f} {unit:11s}  "
              f"agreement among them {spreads[key] * scale:7.4f}, "
              f"largest deviation over the whole decoupled ladder "
              f"{np.abs(residual).max() * scale:7.4f}")

    draw(kept, limits, spreads, args.out or
         args.ladder.with_name(args.ladder.stem + "_nep.png"))


def draw(rows, limits, spreads, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": True,
    })
    sizes = np.array([r["n_host"] for r in rows], dtype=float)
    inverse = 1e3 / sizes

    figure, axes = plt.subplots(
        2, 1, figsize=(4.6, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.08},
    )

    for key, label, colour, marker, size in ENERGIES:
        residual = (np.array([r[key] for r in rows]) - limits[key]) * 1e3
        axes[0].plot(inverse, residual, marker, ms=size, color=colour,
                     mew=1.6 if marker == "x" else 0,
                     label=f"{label} = {limits[key]:.3f} eV")
    band = max(spreads[k] for k, *_ in ENERGIES) * 1e3
    axes[0].axhspan(-band, band, color="0.88", zorder=0,
                    label=f"±{band:.2f} meV")
    axes[0].axhline(0.0, lw=0.8, color="0.4", zorder=1)
    axes[0].set_ylabel(r"$E(N) - E_\mathrm{converged}$ (meV)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=0.5)
    axes[0].text(0.965, 0.06, "a)", transform=axes[0].transAxes, fontsize=11)

    key = "delta_q_sqrtDa_A"
    residual = (np.array([r[key] for r in rows]) - limits[key]) * 1e3
    axes[1].axhspan(-spreads[key] * 1e3, spreads[key] * 1e3, color="0.88", zorder=0)
    axes[1].axhline(0.0, lw=0.8, color="0.4", zorder=1)
    axes[1].plot(inverse, residual, "^", ms=7, color="#a0342c", mew=0,
                 label=rf"$\Delta q$ = {limits[key]:.4f} $\sqrt{{\mathrm{{Da}}}}\,$Å")
    axes[1].set_ylabel(r"$\Delta q - \Delta q_\mathrm{conv}$ ($\times 10^{-3}$)")
    axes[1].set_xlabel(r"Inverse number of atoms $1/N$ ($\times 10^{-3}$)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    axes[1].text(0.965, 0.06, "b)", transform=axes[1].transAxes, fontsize=11)

    for axis in axes:
        axis.set_xlim(-0.05, inverse.max() * 1.08)
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(inverse)
    top.set_xticklabels([f"{int(n)}" for n in sizes], fontsize=8, rotation=45, ha="left")
    top.set_xlabel("Number of atoms $N$", fontsize=9)
    top.tick_params(direction="in")

    figure.suptitle(
        "Short-range model: converged once images decouple. No tail, so no fit.",
        fontsize=9, y=1.06,
    )
    figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
