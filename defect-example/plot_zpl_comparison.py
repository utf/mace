#!/usr/bin/env python3
"""E_ZPL against cell size for several models: which term is still drifting?

The four-model attention chart shows the size hinge holds ``alpha`` flat. This is the
observable consequence, and it separates the two remaining terms: with the long-range
branch on, ``E_ZPL`` drifts whether or not the hinge is present; with it off, the hinge
leaves ``E_ZPL`` flat. Plotted against ``1/N`` so a converged quantity is a horizontal line
meeting the axis at its isolated-limit value.

    python plot_zpl_comparison.py --ladders size_convergence_bfullL1.json:larger data L1 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--ladders", nargs="+", required=True,
                        help="path:label pairs")
    parser.add_argument("--out", type=Path, default=here / "zpl_comparison.png")
    args = parser.parse_args()

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
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    colours = ["#4d4d4d", "#2c6fbb", "#c0392b", "#2e8b57"]
    markers = ["o", "s", "^", "D"]

    rows = []
    for index, item in enumerate(args.ladders):
        path, _, label = item.partition(":")
        target = Path(path)
        if not target.is_absolute():
            target = here / target
        if not target.is_file():
            print(f"  missing {target}, skipping")
            continue
        data = json.loads(target.read_text())
        entries = sorted(data["rows"], key=lambda r: r["n_host"])
        sizes = np.array([r["n_host"] for r in entries], dtype=float)
        for axis, key in ((axes[0], "zpl_eV"), (axes[1], "formation_energy_eV")):
            values = np.array([r[key] for r in entries])
            axis.plot(1e3 / sizes, values, marker=markers[index % 4], ms=4.5, lw=1.4,
                      color=colours[index % 4], label=label or target.stem)
        drift = (
            np.array([r["zpl_eV"] for r in entries])[-1]
            - np.array([r["zpl_eV"] for r in entries])[0]
        )
        rows.append((label or target.stem, sizes[0], sizes[-1], drift * 1e3))

    axes[0].set_ylabel(r"$E_\mathrm{ZPL}$ (eV)")
    axes[1].set_ylabel(r"$E_\mathrm{f}$ (eV)")
    for axis in axes:
        axis.set_xlabel(r"$1/N$ ($\times 10^{-3}$)")
        axis.legend(frameon=False, fontsize=8, loc="best")
        axis.set_box_aspect(0.85)
    figure.suptitle(
        "Horizontal = size-converged. Only the long-range branch still drifts.",
        fontsize=9.5, y=1.0,
    )
    figure.tight_layout()
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}\n")
    print(f"{'model':26s} {'N range':>14s} {'E_ZPL drift':>13s}")
    for label, first, last, drift in rows:
        print(f"{label:26s} {int(first):5d} -> {int(last):5d} {drift:+11.1f} meV")


if __name__ == "__main__":
    main()
