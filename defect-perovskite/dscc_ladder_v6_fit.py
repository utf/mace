"""Ladder v6 analysis: subtract the EXACT eps_inf monopole and fit only the remainder.

`dE(L) = E_inf - C alpha / (2 eps_inf L) + c / L^3 + ...`, so the remainder
`dE(L) + C alpha / (2 eps_inf L)` should be `E_inf + c / L^3`. Two parameters, so three cells
suffice and four give a check.

`c` is reported three ways: raw; minus the model's OWN C13 second-moment term
`2 pi r_g^2 C / eps_inf` (22.6 eV.A^3 at r_g = 1 A, eps_inf = 4 -- it is part of `E_M` by
construction and would otherwise be misread as core truncation); and, across an `R_core` sweep,
against the truncation term, which scales as `q_ion(R) R^2` while the intrinsic terms do not.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mace.modules.dscc.ewald import COULOMB                              # noqa: E402

EPS_INF, R_G = 4.0, 1.0
SECOND_MOMENT = 2.0 * math.pi * R_G ** 2 * COULOMB / EPS_INF             # eV.A^3, inside E_M


def fit_remainder(rows):
    """`remainder = E_inf + c / L^3` after the exact monopole is removed."""
    L = np.array([r["L"] for r in rows]); al = np.array([r["alpha_cell"] for r in rows])
    dE = np.array([r["dE"] for r in rows])
    rem = dE + COULOMB * al / (2.0 * EPS_INF * L)
    A = np.stack([np.ones_like(L), 1.0 / L ** 3], 1)
    beta, *_ = np.linalg.lstsq(A, rem, rcond=None)
    res = rem - A @ beta
    dof = max(len(L) - 2, 1)
    cov = np.linalg.inv(A.T @ A) * float(res @ res) / dof
    return rem, beta, np.sqrt(np.diag(cov)), res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", nargs="+", required=True)
    ap.add_argument("--drop", nargs="*", default=[], help="tilings to exclude from the fit, e.g. 2,2,2")
    args = ap.parse_args()
    rows = []
    for f in args.ladders:
        rows += json.load(open(f))
    by_core = {}
    for r in rows:
        by_core.setdefault(r["r_core"], []).append(r)
    drop = {tuple(int(x) for x in d.split(",")) for d in args.drop}
    for r_core in sorted(by_core):
        rs = sorted(by_core[r_core], key=lambda r: r["n_atoms"])
        keep = [r for r in rs if tuple(r["tiling"]) not in drop]
        print(f"\n### R_core = {r_core} A   ({len(keep)} cells in the fit"
              + (f", {len(rs)-len(keep)} dropped" if len(rs) > len(keep) else "") + ")")
        print("| tiling | n | L (A) | alpha | dE (eV) | remainder after the exact monopole (eV) |")
        print("|---|--:|--:|--:|--:|--:|")
        rem_all, _, _, _ = fit_remainder(rs)
        for r, rem in zip(rs, rem_all):
            mark = "" if tuple(r["tiling"]) not in drop else "  (dropped)"
            print("| %d,%d,%d | %4d | %.2f | %.4f | %.4f | %.4f%s |"
                  % (*r["tiling"], r["n_atoms"], r["L"], r["alpha_cell"], r["dE"], rem, mark))
        if len(keep) < 2:
            continue
        rem, beta, se, res = fit_remainder(keep)
        print("\n  E_inf = %.4f +- %.4f eV" % (beta[0], se[0]))
        print("  c     = %+.1f +- %.1f eV.A^3   (model's own C13 second moment %+.1f, "
              "so intrinsic c = %+.1f)" % (beta[1], se[1], SECOND_MOMENT, beta[1] - SECOND_MOMENT))
        print("  fit residuals: %s meV" % " ".join("%+.1f" % (1000 * v) for v in res))
        # does a 1/L term survive? (the D13 static-pattern compensation)
        L = np.array([r["L"] for r in keep]); al = np.array([r["alpha_cell"] for r in keep])
        A3 = np.stack([np.ones_like(L), al / L, 1.0 / L ** 3], 1)
        if len(keep) >= 3:
            b3, *_ = np.linalg.lstsq(A3, rem, rcond=None)
            print("  with a free 1/L term as well: E_inf %.4f, residual monopole %+.4f eV.A "
                  "(%.1f %% of the exact %.3f), c %+.1f eV.A^3"
                  % (b3[0], b3[1], 100 * b3[1] / (-COULOMB / (2 * EPS_INF)), -COULOMB / (2 * EPS_INF), b3[2]))
    cores = sorted(by_core)
    if len(cores) > 1:
        print("\n### R_core sweep (the truncation term scales as q_ion(R) R^2; intrinsic terms do not)")
        print("| tiling | " + " | ".join("R=%.0f A" % c for c in cores) + " |")
        print("|---|" + "--:|" * len(cores))
        tilings = sorted({tuple(r["tiling"]) for r in rows}, key=lambda t: t[0] * t[1] * t[2])
        for t in tilings:
            vals = []
            for c in cores:
                m = [r for r in by_core[c] if tuple(r["tiling"]) == t]
                vals.append("%.4f" % m[0]["dE"] if m else "—")
            if any(v != "—" for v in vals):
                print("| %d,%d,%d | %s |" % (*t, " | ".join(vals)))


if __name__ == "__main__":
    main()
