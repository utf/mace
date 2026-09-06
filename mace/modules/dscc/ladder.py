"""Plan section 8 (and the Route B gate of section 5): the model-only tiling ladder.

Pristine supercells with one vacancy, `E(+1) - E(0)` against `1/L` with the Madelung
coefficient of the registered convention, the size-dependent part of `K_LR_ii` against
`-alpha_M C / L`, the active-state count, the `dq` spread and the total force. Under the
neutralising-background convention a monopole `Q` in a cubic cell of side `L` carries
`-alpha_M C Q^2 / (2 eps_inf L)` at leading order (the carrier's own image energy, screened
by `eps_inf`), so the fitted `1/L` coefficient of `E(+1) - E(0)` is compared with
`-alpha_M C / (2 eps_inf)`: the CENTRED Route B pattern leaves it unchanged, an uncentred one
multiplies it by `(1 + 2 z)` (plan section 2.5) -- the negative test.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.modules.dscc.ewald import COULOMB, madelung_constant_cubic


def tiling_ladder(pristine, tilings: Sequence[Tuple[int, int, int]], vacancy_species: int = 17,
                  vacancy_index: int = 0) -> List:
    """One ASE frame per tiling: the pristine cell repeated, with the `vacancy_index`-th atom
    of `vacancy_species` removed (the same site in the first image of every tiling)."""
    out = []
    for t in tilings:
        atoms = pristine.repeat(tuple(int(x) for x in t))
        sites = [i for i, z in enumerate(atoms.get_atomic_numbers()) if int(z) == vacancy_species]
        del atoms[sites[vacancy_index]]
        out.append(atoms)
    return out


def madelung_slope(eps_inf: float) -> float:
    """`-alpha_M C / (2 eps_inf)`: the registered `1/L` coefficient of the monopole term."""
    return -madelung_constant_cubic() * COULOMB / (2.0 * eps_inf)


def fit_one_over_l(lengths: Sequence[float], values: Sequence[float]) -> Tuple[float, float]:
    """`(a, b)` of `value = a + b / L` by least squares."""
    A = np.array([[1.0, 1.0 / L] for L in lengths])
    a, b = np.linalg.lstsq(A, np.asarray(values, dtype=np.float64), rcond=None)[0]
    return float(a), float(b)


def ladder_report(model, frames: Sequence, z_table, r_cut: float, batch_fn, device="cpu",
                  q_plus: Sequence[int] = (0, 0, 1, 0)) -> Dict[str, object]:
    """`E(+1) - E(0)` and the diagnostics per ladder cell; `batch_fn(frames) -> data dict`."""
    from mace.modules.dscc import data as dd
    rows = []
    for atoms in frames:
        L = float(np.cbrt(atoms.get_volume()))
        neutral = atoms.copy(); neutral.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0})
        charged = atoms.copy(); charged.info.update({"carrier_counts": np.array(q_plus), "cell_charge": 1})
        e0 = float(model(batch_fn([neutral]), compute_force=True)["energy"])
        out = model(batch_fn([charged]), compute_force=True)
        d = out["diagnostics"]
        rows.append({"n_atoms": len(atoms), "L": L, "dE": float(out["energy"]) - e0,
                     "dq_max": float(out["dq"].abs().max()), "dq_spread": float(out["dq"].std()),
                     "total_force": float(out["forces"].sum(0).abs().max()),
                     "iterations": d.get("iterations", [None])[0], "converged": d.get("converged", [None])[0]})
    a, b = fit_one_over_l([r["L"] for r in rows], [r["dE"] for r in rows])
    return {"rows": rows, "intercept": a, "slope": b, "madelung_slope": madelung_slope(model.kernel.eps_inf)}
