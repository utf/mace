#!/usr/bin/env python3
"""Analyse a convergence ladder correctly for a short-range model, from its saved JSON.

**A power-law extrapolation is the wrong tool here, and the earlier plots using one should
be discarded.** The reasoning:

A MACE model is strictly short-ranged. Its receptive field is ``r_max * n_interactions``
(8 A here: 4.0 A over two layers), so once every cell edge exceeds twice that, every atom
sits in an environment *identical* to the infinite-cell one and the predicted energies stop
depending on cell size **exactly**, not asymptotically. There is no ``1/N`` tail for such a
model to produce, because there is no interaction left to reach the periodic image.

What the ladder actually contains is therefore a **step, not a tail**: cells below the
decoupling threshold, where images overlap inside the receptive field and the answer is
simply wrong, and cells above it, where the answer is converged. Fitting a power law
through both imposes a decay the model cannot have, lets the sub-threshold points dominate
the fit through their leverage, and returns an intercept that disagrees with the plateau it
was fitted through -- which is exactly what was observed (``Delta q`` extrapolating to 0.676
against a measured 0.647).

So this reports:

* the **decoupling threshold**, computed from the model's own cutoff and layer count;
* the **plateau**: mean and spread over the decoupled cells, the spread being the honest
  uncertainty;
* sub-threshold cells flagged as receptive-field artefacts rather than data points.

**When a power law *is* right.** A long-range branch reintroduces a genuine tail, because
Ewald summation reaches the images however large the cell. Then the power follows the
leading multipole: ``1/L = N^(-1/3)`` for a charged cell, ``1/L^3 = 1/N`` for a neutral cell
carrying a dipole (both carrier states here are net neutral, so ``1/N``). Pass
``--power`` to fit that case, and only that case.

    python replot_ladder.py size_convergence_bsize.json
    python replot_ladder.py size_convergence_esize.json --power 1.0    # long-range model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

QUANTITIES = (
    ("formation_energy_eV", "E_f", "eV"),
    ("zpl_eV", "E_ZPL", "eV"),
    ("vertical_absorption_eV", "E_vert^abs", "eV"),
    ("vertical_emission_eV", "E_vert^em", "eV"),
    ("relaxation_energy_eV", "Delta", "eV"),
    ("delta_correction_eV", "Delta E (correction)", "eV"),
    ("delta_q_sqrtDa_A", "Delta q", "sqrt(Da) A"),
)


def shortest_edge(repeat, a: float, c: float) -> float:
    return min(a * repeat[0], a * repeat[1], c * repeat[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ladder", type=Path)
    parser.add_argument("--r-max", type=float, default=4.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--power", type=float, default=None,
                        help="fit a power law N**-p as well. Only meaningful for a model "
                             "with a long-range branch, which is the only thing that can "
                             "produce a tail; 1.0 for a neutral cell, 1/3 if charged")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.ladder.read_text())
    rows = sorted(data["rows"], key=lambda r: r["n_host"])
    cellpar = data.get("cellpar") or [3.0815, 0, 10.0605]
    a, c = float(cellpar[0]), float(cellpar[2])

    receptive = args.r_max * args.layers
    threshold = 2.0 * receptive
    print(f"receptive field {receptive:.1f} A ({args.r_max} x {args.layers} layers)")
    print(f"images decouple once every cell edge exceeds {threshold:.1f} A\n")

    print(f"{'N':>6s} {'shortest edge':>14s}   state")
    decoupled = []
    for row in rows:
        edge = shortest_edge(row["repeat"], a, c)
        ok = edge > threshold
        decoupled.append(ok)
        print(f"{row['n_host']:6d} {edge:12.1f} A   "
              f"{'decoupled' if ok else 'IMAGES OVERLAP -- receptive-field artefact'}")

    kept = [r for r, ok in zip(rows, decoupled) if ok]
    if len(kept) < 2:
        raise SystemExit("fewer than two decoupled cells; extend the ladder")

    sizes = np.array([r["n_host"] for r in kept], dtype=float)
    print(f"\nplateau over the {len(kept)} decoupled cells "
          f"(N = {int(sizes[0])}-{int(sizes[-1])}):\n")
    print(f"{'quantity':22s} {'mean':>11s} {'spread':>11s} {'drift/decade':>13s}")
    limits = {}
    for key, label, unit in QUANTITIES:
        values = np.array([r[key] for r in kept])
        limits[key] = float(values.mean())
        scale = 1e3 if unit == "eV" else 1.0
        # A residual slope in ln N would betray a tail the model should not have.
        slope = float(np.polyfit(np.log10(sizes), values, 1)[0])
        print(f"{label:22s} {values.mean():11.4f} "
              f"{(values.max() - values.min()) * scale:9.3f}{'m' if unit == 'eV' else ' '} "
              f"{slope * scale:12.3f}{'m' if unit == 'eV' else ' '}")
        if args.power is not None:
            abscissa = sizes ** (-args.power)
            gradient, intercept = np.polyfit(abscissa, values, 1)
            deviation = np.abs(values - (gradient * abscissa + intercept)).max()
            print(f"{'':22s} power law N^-{args.power:.3g}: limit {intercept:.4f}, "
                  f"max dev {deviation * scale:.3f}")

    print(
        "\nUnits: meV for energies. `spread` is the honest uncertainty -- for a short-range\n"
        "model the converged value is a constant, so scatter about it is the error bar.\n"
        "`drift/decade` should be ~0; a systematic slope would mean a tail this model\n"
        "cannot physically have, and would point at the long-range branch."
    )

    out = args.out or args.ladder.with_name(args.ladder.stem + "_plateau.png")
    draw(rows, decoupled, limits, threshold, a, c, out)


def draw(rows, decoupled, limits, threshold, a, c, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    panels = (
        ("zpl_eV", r"$E_\mathrm{ZPL}$ (eV)", 1e3, "meV"),
        ("formation_energy_eV", r"$E_\mathrm{f}$ (eV)", 1e3, "meV"),
        ("delta_q_sqrtDa_A", r"$\Delta q$ ($\sqrt{\mathrm{Da}}\,\mathrm{\AA}$)", 1e3,
         r"$\times10^{-3}$"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4))
    sizes = np.array([r["n_host"] for r in rows], dtype=float)
    mask = np.array(decoupled)
    # The threshold in atoms, for the shaded region: the smallest N that clears it.
    cutoff = sizes[mask][0] if mask.any() else sizes[-1]

    for axis, (key, label, scale, unit) in zip(axes, panels):
        values = np.array([r[key] for r in rows])
        axis.axvspan(sizes.min() * 0.8, cutoff * 0.93, color="0.92", zorder=0)
        axis.axhline(limits[key], lw=1.0, color="#c0392b", zorder=1)
        spread = values[mask].max() - values[mask].min()
        axis.fill_between(
            [sizes.min() * 0.8, sizes.max() * 1.2],
            limits[key] - spread / 2, limits[key] + spread / 2,
            color="#c0392b", alpha=0.13, zorder=0,
        )
        axis.plot(sizes[~mask], values[~mask], "o", ms=5, mfc="white",
                  mec="#4d4d4d", mew=1.2, zorder=3, label="images overlap")
        axis.plot(sizes[mask], values[mask], "o", ms=5, color="#2c6fbb", zorder=3,
                  label="decoupled")
        axis.set_xscale("log")
        # Ticks at the cells actually computed. Matplotlib's log locator puts decade and
        # minor labels on top of each other over this narrow a range, which is what made
        # the previous axis unreadable.
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{int(v)}" for v in sizes], rotation=45, fontsize=7.5)
        axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_xlabel("atoms in cell $N$")
        axis.set_ylabel(label)
        axis.set_xlim(sizes.min() * 0.85, sizes.max() * 1.15)
        axis.set_title(f"plateau {limits[key]:.4f}  ±{spread * scale / 2:.2f} {unit}",
                       fontsize=8.5)
        axis.set_box_aspect(0.85)
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    figure.suptitle(
        "Short-range model: converged exactly once images decouple, not asymptotically. "
        "Shaded = receptive-field overlap.",
        fontsize=9, y=1.0,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
