"""E(+1) - E(0) against supercell size for the Phi = 0 and W6 models.

Two panels. LEFT: the raw ladder, `dE` against `1/L`, with the least-squares `a + b/L` fit and
the extrapolated `L -> inf` intercept; the exact monopole slope `-alpha_M C / (2 eps_inf)` is
drawn through each intercept for reference. RIGHT: the finite-size error, `dE(L) - a`, against
`L`, before and after subtracting the analytic monopole term -- the curve that is flat is the
one whose finite-size error is accounted for.

The absolute offset between the two models is the per-charge constant `C_Q` absorbs and carries
no information about convergence, so the right panel (which removes each model's own intercept)
is the like-for-like comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

from mace.modules.dscc.ewald import COULOMB, madelung_constant_cubic     # noqa: E402
from mace.modules.dscc.ladder import fit_one_over_l, second_moment_term  # noqa: E402

EPS_INF, R_G = 4.0, 1.0
MONOPOLE = -madelung_constant_cubic() * COULOMB / (2.0 * EPS_INF)        # eV.A, cubic cells
# The monopole term is `-alpha(shape) C / (2 eps L)`, so the finite-size variable is
# `alpha_cell / L` and the expected slope against it is `-C / (2 eps)`, shape-independent.
SLOPE_VS_ALPHA_OVER_L = -COULOMB / (2.0 * EPS_INF)                       # eV.A


def calibration_of(ladder: dict) -> float:
    """`C_Q(+1)` of the model this ladder was run with, from the calibration record. Returns 0
    when the model has none (then the curve carries the head's own zero and only its SHAPE is
    meaningful)."""
    run = Path(ladder.get("model", "")).parent.name
    for f in ("/home/alex/runs/dscc/calibration_v5.json", "/home/alex/runs/dscc/calibration_v5_rest.json"):
        if not Path(f).exists():
            continue
        d = json.load(open(f))
        if run in d:
            return float(d[run]["c_q"]["1"])
    return 0.0


def load(path: Path, calibrated: bool = False):
    doc = json.load(open(path))
    c_q = calibration_of(doc) if calibrated else 0.0
    rows = doc["rows"]
    rows = sorted(rows, key=lambda r: (r["n_atoms"], r["tiling"]))
    return (np.array([r["n_atoms"] for r in rows]),
            np.array([r["L"] for r in rows]),
            np.array([r["dE"] for r in rows]) + c_q,
            np.array([r["alpha_cell"] for r in rows]),
            [r["tiling"] for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--out", default="dscc_convergence.png")
    ap.add_argument("--calibrated", action="store_true",
                    help="add each model's C_Q(+1) so the energies are on the DFT scale")
    args = ap.parse_args()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colours = ["#c0392b", "#2471a3", "#117a65", "#7d3c98"]
    print("| model | tiling | n atoms | L (A) | alpha_cell | dE (eV) | dE - limit (meV) | monopole removed (meV) |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for i, spec in enumerate(args.ladders):
        label, path = spec.rsplit("=", 1)     # labels may contain "=" (e.g. "Phi=0")
        n, L, dE, alpha, tilings = load(Path(path), args.calibrated)
        x = alpha / L                       # the Makov-Payne monopole variable
        A = np.stack([np.ones_like(x), x], axis=1)
        beta, *_ = np.linalg.lstsq(A, dE, rcond=None)
        a, b = float(beta[0]), float(beta[1])
        c = colours[i % len(colours)]
        ax0.plot(x, dE, "o", color=c, label=f"{label}  (fit {b:+.2f}, exact {SLOPE_VS_ALPHA_OVER_L:+.2f} eV·Å)")
        xs = np.linspace(0.0, 1.05 * x.max(), 50)
        ax0.plot(xs, a + b * xs, "-", color=c, lw=1.2)
        ax0.plot(xs, a + SLOPE_VS_ALPHA_OVER_L * xs, ":", color=c, lw=1.0)
        ax0.plot([0.0], [a], "*", color=c, ms=13)
        ax0.annotate(f"limit {a:.3f} eV", (0.0, a), textcoords="offset points", xytext=(8, 8),
                     fontsize=8, color=c)
        err = 1000 * (dE - a)
        corr = 1000 * np.array([SLOPE_VS_ALPHA_OVER_L * xx
                                + second_moment_term(l ** 3, R_G, EPS_INF, 1.0) for xx, l in zip(x, L)])
        # Markers only: two cells share L = 22.73 A with different SHAPES (alpha 2.45 and 1.56),
        # so a line through L would draw a spike that is cell shape, not convergence.
        ax1.plot(L, err, "o", color=c, ms=7, label=f"{label}, raw")
        ax1.plot(L, err - corr, "s", color=c, ms=6, mfc="white", label=f"{label}, monopole removed")
        for xx, yy, tt in zip(L, err, tilings):
            if list(tt) in ([1, 2, 2], [2, 2, 1]):
                ax1.annotate("%d,%d,%d" % tuple(tt), (xx, yy), textcoords="offset points",
                             xytext=(6, -3), fontsize=7, color=c)
        for k in range(len(n)):
            print("| %s | %s | %d | %.2f | %.4f | %.4f | %+.0f | %+.0f |"
                  % (label, tilings[k], n[k], L[k], alpha[k], dE[k], err[k], err[k] - corr[k]))
        print("| %s | **limit** | inf | — | — | **%.4f** | 0 | — |" % (label, a))
        print("  -> %s: fitted monopole slope %+.3f eV.A against the exact %+.3f (%.0f %%); "
              "extrapolated limit %.4f eV; residual after the analytic correction "
              "%+.0f meV at 79 atoms, %+.0f at the largest cell"
              % (label, b, SLOPE_VS_ALPHA_OVER_L, 100 * b / SLOPE_VS_ALPHA_OVER_L, a,
                 (err - corr)[np.argmin(L)], (err - corr)[np.argmax(L)]))
    ax0.set_xlabel(r"$\alpha_{\rm cell}\,/\,L$   (Å$^{-1}$)")
    ax0.set_ylabel("E(+1) − E(0)   (eV%s)" % (", calibrated" if args.calibrated else ", uncalibrated"))
    ax0.set_title("Convergence: dotted = exact monopole slope")
    ax0.legend(fontsize=8, frameon=False)
    ax0.axvline(0.0, color="0.8", lw=0.8)
    ax1.axhline(0.0, color="0.8", lw=0.8)
    ax1.set_xlabel("L = V$^{1/3}$   (Å)")
    ax1.set_ylabel("E(+1) − E(0) − limit   (meV)")
    ax1.set_title("Finite-size error, before and after the monopole term")
    ax1.legend(fontsize=8, frameon=False)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
