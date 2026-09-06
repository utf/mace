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
                  q_plus: Sequence[int] = (0, 0, 1, 0), force_atoms: int = 200) -> Dict[str, object]:
    """`E(+1) - E(0)` and the diagnostics per ladder cell; `batch_fn(frames) -> data dict`."""
    from mace.modules.dscc import data as dd
    rows = []
    for atoms in frames:
        L = float(np.cbrt(atoms.get_volume()))
        neutral = atoms.copy(); neutral.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0})
        charged = atoms.copy(); charged.info.update({"carrier_counts": np.array(q_plus), "cell_charge": 1})
        with_forces = len(atoms) <= force_atoms
        e0 = float(model(batch_fn([neutral]), compute_force=with_forces)["energy"])
        out = model(batch_fn([charged]), compute_force=with_forces)
        d = out["diagnostics"]
        rows.append({"n_atoms": len(atoms), "L": L, "dE": float(out["energy"]) - e0,
                     "dq_max": float(out["dq"].abs().max()), "dq_spread": float(out["dq"].std()),
                     "total_force": float(out["forces"].sum(0).abs().max()) if with_forces else None,
                     "iterations": d.get("iterations", [None])[0], "converged": d.get("converged", [None])[0]})
    a, b = fit_one_over_l([r["L"] for r in rows], [r["dE"] for r in rows])
    return {"rows": rows, "intercept": a, "slope": b, "madelung_slope": madelung_slope(model.kernel.eps_inf)}


def local_neutrality_gate(model, unit_cell, tilings: Sequence[Tuple[int, int, int]], z_table,
                          r_cut: float, batch_fn, vacancy_species: int = 17, vacancy_index: int = 0,
                          tolerance: float = 0.05) -> Dict[str, object]:
    """v4.2 (C6): the kernel-level tiling ladder with a FIXED localised carrier (a unit
    charge on one flanking Pb, identified label-free) and the model's own Route B' pattern
    `q0(R)` of each ladder cell: the `1/L` slope of `Phi_cc + E_SF` must equal the Madelung
    value within `tolerance`; the registered negative test is the per-species pattern (the
    pristine species means of `q0`, centred and uncentred), which must fail. To be run on
    the trained Arm-1 `H0` before any Route B' arm opens."""
    from mace.modules.models import ScaleShiftMACE
    from mace.modules.dscc.kernels import (centred_pattern, gamma_lr, gamma_matrix, host_potential,
                                           kernel_components, minimum_image_distances, phi_cc)

    frames = tiling_ladder(unit_cell, tilings, vacancy_species, vacancy_index)
    lengths, values = [], {"q0": [], "species_centred": [], "species_uncentred": []}
    for atoms in frames:
        atoms = atoms.copy()
        atoms.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0})
        data = batch_fn([atoms])
        with torch.no_grad():
            out = ScaleShiftMACE.forward(model.base, model._trunk_data(dict(data)), training=False,
                                         compute_force=False)
            scalars, vectors = model.features(out["node_feats"])
            species = data["node_attrs"].argmax(dim=-1)
            pos, cell = data["positions"], data["cell"].view(3, 3)
            ei = data["edge_index"]
            ev = pos[ei[1]] - pos[ei[0]] + data["unit_shifts"].to(pos.dtype) @ cell
            H = model.h0(scalars, vectors, species, ei, ev)
            numbers = [model.atomic_numbers[int(x)] for x in species.tolist()]
            q0 = model.reference_charges(H, numbers).detach()
            # The fixed carrier: a unit charge on a flanking Pb (first shell of five Cl).
            z = torch.tensor(numbers, device=pos.device)
            r = minimum_image_distances(pos, cell)
            pb = torch.nonzero(z == 82).reshape(-1)
            cl = torch.nonzero(z == 17).reshape(-1)
            d = torch.sort(r[pb][:, cl], dim=1).values
            flank = pb[d[:, 5] > 4.0]
            dq = torch.zeros(len(numbers), dtype=pos.dtype, device=pos.device)
            dq[int(flank[0])] = 1.0
            k_sr = k_lr = None
            k_sr, k_lr = kernel_components(pos, cell, model.kernel)
            gamma = gamma_matrix(k_sr, k_lr, model.lambda_dir(), model.u_eff()[species], model.kernel.eps_inf)
            g_lr = gamma_lr(pos, cell, model.kernel.r_g, model.r_split, model.kernel.eps_inf, tol=model.kernel.tol)
            base = float(phi_cc(gamma, dq))
            species_pattern = model.q0_pristine[species]
            for name, pattern in (("q0", model.pattern_scale() * q0),
                                  ("species_centred", centred_pattern(species_pattern)),
                                  ("species_uncentred", species_pattern)):
                values[name].append(base + float(dq @ host_potential(g_lr, pattern)))
        lengths.append(float(np.cbrt(atoms.get_volume())))
    madelung = madelung_slope(model.kernel.eps_inf)
    slopes = {name: fit_one_over_l(lengths, v)[1] for name, v in values.items()}
    rel = {name: abs(slope - madelung) / abs(madelung) for name, slope in slopes.items()}
    return {"lengths": lengths, "values": values, "slopes": slopes, "madelung_slope": madelung,
            "relative_error": rel,
            "passed": rel["q0"] <= tolerance,
            "negative_test_passed": rel["species_centred"] > tolerance and rel["species_uncentred"] > tolerance}

