#!/usr/bin/env python3
"""Finite-size figure for V_Cl in CsPbCl3, including the charge transition level.

Same construction as the SiC figure: deviations from the **converged value** (the largest
cell), no fitted lines, with a band at the residual scatter. A short-range model has no tail
to extrapolate -- once the periodic images fall outside the receptive field the answer stops
changing exactly, so a power-law fit would impose a decay the model cannot produce.

The charged state is the interesting difference from SiC. ``V_Cl+`` carries a real monopole,
so the true energy has a ``q^2 alpha_M / 2 eps L`` image tail decaying as ``1/L``. A
short-range model cannot represent that and will return a flat line; a long-range model
should not. Flatness here is therefore a *statement about the model*, not a validation
against nature, and the two are separated in the caption rather than conflated.

    python plot_perov_size.py perov_size_nolr.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PANELS = (
    ("eps_opt_eV", r"$\epsilon_\mathrm{opt}$", "#2c6fbb", "o", 1e3, "meV"),
    ("transition_level_eV", r"$\epsilon(+/0)$", "#c0392b", "s", 1e3, "meV"),
    ("e_relative_eV", r"$E_\mathrm{rel}$", "#4d4d4d", "^", 1e3, "meV"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("ladder", type=Path)
    parser.add_argument("--out", type=Path, default=here / "perov_size.png")
    args = parser.parse_args()

    data = json.loads(args.ladder.read_text())
    rows = sorted(data["rows"], key=lambda r: r["n_host"])
    if len(rows) < 2:
        raise SystemExit("need at least two cell sizes")
    sizes = np.array([r["n_host"] for r in rows], dtype=float)
    inverse = 1e3 / sizes
    relaxed = data.get("relaxed", True)

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
    figure, axes = plt.subplots(
        2, 1, figsize=(4.8, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.08},
    )

    print(f"converged value = largest cell, N = {int(sizes[-1])}\n")
    print(f"{'quantity':22s} {'converged':>12s} {'max deviation':>15s}")
    largest = rows[-1]
    for key, label, colour, marker, scale, unit in PANELS:
        values = np.array([r[key] for r in rows])
        residual = (values - largest[key]) * scale
        axes[0].plot(inverse, residual, marker, ms=6, color=colour, mew=0,
                     label=f"{label} = {largest[key]:.4f} eV")
        print(f"{label:22s} {largest[key]:12.4f} {np.abs(residual).max():12.3f} {unit}")
    axes[0].axhline(0.0, lw=0.8, color="0.4", zorder=0)
    axes[0].set_ylabel(r"$E(N) - E_\mathrm{converged}$ (meV)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    axes[0].text(0.955, 0.06, "a)", transform=axes[0].transAxes, fontsize=11)

    key = "delta_q_sqrtDa_A"
    values = np.array([r[key] for r in rows])
    residual = (values - largest[key]) * 1e3
    axes[1].axhline(0.0, lw=0.8, color="0.4", zorder=0)
    axes[1].plot(inverse, residual, "D", ms=6, color="#a0342c", mew=0,
                 label=rf"$\Delta q$ = {largest[key]:.4f} $\sqrt{{\mathrm{{Da}}}}\,$Å")
    axes[1].set_ylabel(r"$\Delta q - \Delta q_\mathrm{conv}$ ($\times 10^{-3}$)")
    axes[1].set_xlabel(r"Inverse number of atoms $1/N$ ($\times 10^{-3}$)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="best")
    axes[1].text(0.955, 0.06, "b)", transform=axes[1].transAxes, fontsize=11)
    print(f"{'Delta q':22s} {largest[key]:12.4f} {np.abs(residual).max():12.3f} x1e-3")

    for axis in axes:
        axis.set_xlim(-0.15, inverse.max() * 1.1)
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(inverse)
    top.set_xticklabels([f"{int(n)}" for n in sizes], fontsize=8, rotation=45, ha="left")
    top.set_xlabel("Number of atoms $N$", fontsize=9)
    top.tick_params(direction="in")

    long_range = "with" if data.get("long_range") else "without"
    figure.suptitle(
        f"V$_{{\\mathrm{{Cl}}}}$ in CsPbCl$_3$, {long_range} long range"
        f"{'' if relaxed else '  (UNRELAXED)'}"
        f"{'  [dilute]' if data.get('dilute') else ''}\n"
        "charged cell: the true level has a 1/L image tail a short-range model cannot show",
        fontsize=8.5, y=1.04,
    )
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
