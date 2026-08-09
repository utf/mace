#!/usr/bin/env python3
"""Section 6 of the size-extensivity plan: refit the ladder with the right functional form.

The ``1/N`` extrapolation reported earlier fits the wrong model. Finite-size *physics*
(elastic and multipole images) goes as a power of the cell dimension, but the drift here is
attention dilution, and the softmax says exactly what its shape is:

    alpha_d^c(N) = 1 / (1 + (N/k) e^{-gap_c})

which is **logistic in ln N**, not a power law. The observable is then linear in those
occupancies,

    E(N) = C + sum_c A_c * alpha_d^c(N)

with ``gap_c`` already measured by ``alpha_dilution.py`` and held fixed here, so the fit has
no free shape parameters at all -- only the linear coefficients ``C`` and ``A_c``. That is
what makes it a test rather than a curve-fitting exercise: with the shape pinned from an
independent measurement, reproducing held-out sizes is evidence the mechanism is right.

Three things come out that were previously unavailable, at no computational cost:

* the **localised limit** (``alpha_d -> 1``, i.e. ``C + sum_c A_c``) -- what the model says
  with a properly bound carrier, and the honest number to compare against the published
  0.91 eV. The value at any finite cell is contaminated by partial dilution;
* the **diluted limit** ``C`` (``alpha_d -> 0``), which cross-checks the band-edge gauge:
  referencing requires ``sum_c u_bulk^c = 0``, and C says what the model actually has;
* per-channel ``A_c``, the size of each carrier's contribution, not otherwise measured.

Both limits are extrapolations of a 3-parameter fit to a handful of points -- they are the
*model's* asymptotes under a mechanism we have evidence for, not measurements. Treated as
such they are still far more meaningful than a 1/N intercept fitted to a logistic curve.

    python logistic_size_fit.py --ladder size_convergence_bfullL1.json \
        --alpha-log /tmp/L1_alpha.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")
OBSERVABLES = (
    ("zpl_eV", "E_ZPL"),
    ("vertical_absorption_eV", "E_vert^abs"),
    ("formation_energy_eV", "E_f"),
    ("delta_correction_eV", "correction"),
)


def read_gaps(path: Path) -> dict:
    """Per-channel implied gaps from an alpha_dilution.py log."""
    gaps = {}
    text = path.read_text()
    for channel in CHANNELS:
        found = re.search(
            rf"{channel}\s+alpha on shell.*?\n\s+implied gap\s+([-\d.]+)", text
        )
        if found:
            gaps[channel] = float(found.group(1))
    if not gaps:
        raise SystemExit(f"no implied gaps found in {path}")
    return gaps


def occupancies(sizes: np.ndarray, gaps: dict, k: int) -> tuple:
    """alpha_d^c(N) for each live channel; dead channels are dropped, not fitted."""
    live, columns = [], []
    for channel, gap in gaps.items():
        if gap < 0.5:  # a dead channel is uniform at every N and carries no shape
            continue
        live.append(channel)
        columns.append(1.0 / (1.0 + (sizes / k) * np.exp(-gap)))
    return live, np.stack(columns, axis=1)


def fit(sizes: np.ndarray, values: np.ndarray, design: np.ndarray):
    """Least squares for C and A_c, with the shape fixed. Returns (C, A, prediction)."""
    matrix = np.hstack([np.ones((len(sizes), 1)), design])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    return coefficients[0], coefficients[1:], matrix @ coefficients


def leave_one_out(sizes: np.ndarray, values: np.ndarray, design: np.ndarray):
    """Held-out error: the shape is fixed, so this genuinely tests the mechanism."""
    errors = []
    for index in range(len(sizes)):
        mask = np.ones(len(sizes), dtype=bool)
        mask[index] = False
        matrix = np.hstack([np.ones((mask.sum(), 1)), design[mask]])
        coefficients, *_ = np.linalg.lstsq(matrix, values[mask], rcond=None)
        row = np.concatenate([[1.0], design[index]])
        errors.append(abs(float(row @ coefficients) - values[index]))
    return np.array(errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--ladder", type=Path, required=True,
                        help="size_convergence*.json from defect_size_extensivity.py")
    parser.add_argument("--alpha-log", type=Path, required=True,
                        help="alpha_dilution.py log giving the fixed per-channel gaps")
    parser.add_argument("--k", type=int, default=6,
                        help="defect shell size; 6 dangling bonds for the divacancy")
    parser.add_argument("--reference", type=float, default=0.91,
                        help="published E_ZPL to compare the localised limit against")
    args = parser.parse_args()

    ladder = json.loads(args.ladder.read_text())
    rows = sorted(ladder["rows"], key=lambda r: r["n_host"])
    sizes = np.array([r["n_host"] for r in rows], dtype=float)
    gaps = read_gaps(args.alpha_log)
    live, design = occupancies(sizes, gaps, args.k)

    print(f"ladder     {args.ladder.name}  ({len(sizes)} sizes: "
          f"{', '.join(str(int(n)) for n in sizes)})")
    print(f"gaps       " + "  ".join(f"{c}={gaps[c]:.2f}" for c in gaps))
    print(f"fitted     C + sum_c A_c * alpha_d^c(N), shape FIXED, "
          f"live channels: {', '.join(live)}")
    print(f"           {1 + len(live)} free parameters against {len(sizes)} points\n")

    print(f"{'observable':14s} {'diluted C':>10s} {'localised':>10s} "
          f"{'|resid|max':>11s} {'LOO max':>9s}   per-channel A_c")
    for key, label in OBSERVABLES:
        values = np.array([r[key] for r in rows])
        constant, amplitudes, prediction = fit(sizes, values, design)
        residual = float(np.abs(values - prediction).max())
        held_out = leave_one_out(sizes, values, design)
        localised = constant + amplitudes.sum()
        detail = "  ".join(f"{c}={a:+.3f}" for c, a in zip(live, amplitudes))
        print(f"{label:14s} {constant:10.4f} {localised:10.4f} "
              f"{residual * 1e3:10.1f}m {held_out.max() * 1e3:8.1f}m   {detail}")

    # The headline comparison: at a finite cell the value is contaminated by partial
    # dilution, so the localised limit is the number that should face the literature.
    zpl = np.array([r["zpl_eV"] for r in rows])
    constant, amplitudes, _ = fit(sizes, zpl, design)
    print(f"\nE_ZPL")
    print(f"  bound-carrier limit   {constant + amplitudes.sum():.4f} eV   "
          f"(published {args.reference:.2f} eV, "
          f"difference {(constant + amplitudes.sum() - args.reference) * 1e3:+.0f} meV)")
    print(f"  fully diluted limit   {constant:.4f} eV")
    print(f"  smallest cell ({int(sizes[0])})  {zpl[0]:.4f} eV  <- partly diluted "
          f"already; agreement here is compensating error, not validation")

    print("\nGauge cross-check (plan section 5.2): band-edge referencing requires the")
    print("fully diluted correction to sit at 0, since a delocalised carrier is a")
    print("band-to-band transition.")
    correction = np.array([r["delta_correction_eV"] for r in rows])
    constant, amplitudes, _ = fit(sizes, correction, design)
    print(f"  correction diluted limit C = {constant:+.4f} eV  "
          f"(0 required; offset {constant:+.3f} eV)")


if __name__ == "__main__":
    main()
