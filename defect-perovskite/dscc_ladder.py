"""Plan section 8: the model-only tiling ladder on the static pristine cell -- pristine
supercells with one vacancy (the same site in the first image), `E(+1) - E(0)` against
`1/L` (L = V^(1/3)) with the Madelung coefficient of the registered convention computed
for the cell shape, the size-dependent part of `K_LR_ii` against `-alpha_M C / L`, the
active-state (window) count, the `dq` spread and the total force; dense below
`--dense_max` atoms, the sparse frontier path above, and the dense/sparse agreement at
the largest dense cell.
    python defect-perovskite/dscc_ladder.py --model /home/alex/runs/dscc/<run>/model_dscc.pt --tilings 1,1,1 1,1,2 2,2,1 2,2,2 --out ladder.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ase import Atoms                                                     # noqa: E402
from mace import tools                                                    # noqa: E402
from mace.modules.dscc import data as dd, ewald, sparse                   # noqa: E402
from mace.modules.dscc.kernels import kernel_components, minimum_image_distances   # noqa: E402
from mace.modules.dscc.ladder import fit_one_over_l, tiling_ladder         # noqa: E402
from mace.tools import torch_geometric                                    # noqa: E402


def madelung_of_cell(cell: torch.Tensor) -> float:
    """`alpha` of a unit point charge in this cell shape: `E_self - C/(sqrt(pi) r) = -alpha C / L`."""
    r_g = 0.3
    E = ewald.ewald_matrix(torch.zeros(1, 3, dtype=torch.float64), cell, r_g, background="point")   # a POINT charge's alpha (C13)
    L = float(torch.det(cell).abs() ** (1.0 / 3.0))
    return -float(E[0, 0] - ewald.self_term(r_g)) * L / ewald.COULOMB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
    ap.add_argument("--tilings", nargs="+", default=["1,1,1", "1,1,2", "2,2,1", "2,2,2"])
    ap.add_argument("--dense_max", type=int, default=700)
    ap.add_argument("--reuse", nargs="*", default=[],
                    help="earlier ladder JSONs for the SAME model: any tiling already present is "
                         "copied instead of recomputed. A ladder cell is a deterministic function "
                         "of (model, tiling), so recomputing one is pure waste -- and the large "
                         "cells are the expensive ones.")
    ap.add_argument("--head_only_above", type=int, default=0,
                    help="above this many atoms take dE from the HEAD energy alone and skip the "
                         "neutral forward: E_base cancels exactly in E(+1) - E(0) at fixed "
                         "geometry, so the two full base passes buy nothing and are what runs the "
                         "machine out of memory at a few thousand atoms (a 5119-atom cell died in "
                         "the neutral pass). The charged pass then takes the cached-base path and "
                         "runs block 0 only.")
    ap.add_argument("--force_max", type=int, default=700,
                    help="compute forces only up to this many atoms; above it the row is "
                         "energy-only (the force backward through the base and the head is what "
                         "makes a 2000-atom cell expensive, and the convergence study reads "
                         "energies -- `total_force` stays as a diagnostic on the smaller cells)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    rec = json.load(open(args.static_cell))
    unit = Atoms(numbers=rec["numbers"], positions=rec["positions"], cell=rec["cell"], pbc=True)
    model = torch.load(args.model, weights_only=False, map_location=args.device).to(args.device).eval()
    z_table = tools.AtomicNumberTable(model.atomic_numbers)
    tilings = [tuple(int(x) for x in t.split(",")) for t in args.tilings]
    frames = tiling_ladder(unit, tilings, vacancy_species=17, vacancy_index=0)
    rows = []
    cached = {}
    for path in args.reuse:
        if not Path(path).exists():
            continue
        doc = json.load(open(path))
        if doc.get("model") not in (None, args.model):
            print(f"skipping {path}: it was run with {doc.get('model')}", flush=True)
            continue
        for r in doc.get("rows", []):
            cached[tuple(r["tiling"])] = r
    if cached:
        print(f"reusing {len(cached)} cells: {sorted(tuple(k) for k in cached)}", flush=True)

    def batch(atoms, counts, charge):
        a = atoms.copy(); a.info.update({"carrier_counts": np.array(counts), "cell_charge": charge})
        d = dd.atomic_data([a], z_table, model.r_cut)
        return next(iter(torch_geometric.dataloader.DataLoader(d, batch_size=1))).to(args.device).to_dict()

    for t, atoms in zip(tilings, frames):
        if tuple(t) in cached:
            rows.append(cached[tuple(t)])
            print(f"reused {t} ({cached[tuple(t)]['n_atoms']} atoms)", flush=True)
            continue
        n = len(atoms); cell = torch.tensor(np.array(atoms.get_cell()))
        L = float(atoms.get_volume() ** (1.0 / 3.0))
        t0 = time.time()
        head_only = bool(args.head_only_above) and n > args.head_only_above
        bp = batch(atoms, [0, 0, 1, 0], 1)
        if head_only:
            e0 = None
            bp["dscc_base_energy"] = torch.zeros(1, dtype=torch.float64, device=args.device)
            bp["dscc_base_forces"] = torch.zeros_like(bp["positions"])
        else:
            b0 = batch(atoms, [0, 0, 0, 0], 0)
            with torch.no_grad():           # frontier fills, no density matrices
                e0 = float(model(b0, compute_force=False)["energy"])              # neutral: the base
        row = {"tiling": t, "n_atoms": n, "L": L, "alpha_cell": madelung_of_cell(cell), "E0": e0,
               "head_only": head_only}
        if n <= args.dense_max:
            want_f = n <= args.force_max and not head_only
            if want_f:
                out = model(bp, compute_force=True)
            else:
                with torch.no_grad():       # as above: the eager fills are what cost the memory
                    out = model(bp, compute_force=False)
            # head_only: dE IS the head energy (E_base is identical in the two charge states).
            row.update({"path": "dense",
                        "dE": float(out["head_energy"][0]) if head_only else float(out["energy"]) - e0, "dq_max": float(out["dq"].abs().max()),
                        "dq_spread": float(out["dq"].std()),
                        "total_force": float(out["forces"].sum(0).abs().max()) if want_f else None,
                        "iterations": out["diagnostics"].get("iterations", [None])[0]})
            if n >= 150 or t == tilings[-1] or True:
                # dense/sparse agreement on this cell (the gate)
                try:
                    sp_out = sparse.model_forward_sparse(model, batch(atoms, [0, 0, 1, 0], 1))
                    row["sparse_dE"] = float(sp_out["energy"]) - e0
                    row["dense_sparse_force_diff"] = float((sp_out["forces"] - out["forces"]).abs().max())
                    row["sparse_window"] = sp_out["diagnostics"]["window"]
                except Exception as exc:          # noqa: BLE001 -- a cross-check, not the gate
                    # Route B' and W6 raise NotImplementedError by design (the sparse path has
                    # no reference density); anything else here is still a note, not a reason
                    # to lose the ladder row that IS the measurement.
                    row["sparse_note"] = f"{type(exc).__name__}: {exc}"
        else:
            sp_out = sparse.model_forward_sparse(model, bp)
            row.update({"path": "sparse", "dE": float(sp_out["energy"]) - e0, "dq_max": float(sp_out["dq"].abs().max()),
                        "dq_spread": float(sp_out["dq"].std()), "total_force": float(sp_out["forces"].sum(0).abs().max()),
                        "iterations": sp_out["diagnostics"]["iterations"], "sparse_window": sp_out["diagnostics"]["window"]})
        # K_LR_ii on a Pb site: its size-dependent part against -alpha_cell C / L for THIS cell
        # shape (regime B: K_LR is the Ewald kernel of the broad Gaussian, self term 2C/(sqrt(pi) r_s)).
        pos = torch.tensor(atoms.get_positions()); k_sr, k_lr = kernel_components(pos, cell, model.kernel)
        z = atoms.get_atomic_numbers(); i_pb = int(np.nonzero(z == 82)[0][0])
        row["k_lr_ii_pb"] = float(k_lr[i_pb, i_pb])
        if model.kernel.regime == "B":
            broad_self = 2.0 * ewald.COULOMB / (np.sqrt(np.pi) * model.kernel.r_s)
            row["k_lr_ii_alpha_measured"] = -(row["k_lr_ii_pb"] - broad_self) * L / ewald.COULOMB
            row["k_lr_ii_alpha_rel_err"] = abs(row["k_lr_ii_alpha_measured"] - row["alpha_cell"]) / row["alpha_cell"]
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(json.dumps(row), flush=True)
    # The 1/L fit of E(+1) - E(0) uses the cells of one shape class only (alpha within 2 % of
    # the first cell's), since the Madelung coefficient is a property of the cell shape.
    same = [r for r in rows if abs(r["alpha_cell"] - rows[0]["alpha_cell"]) < 0.02 * rows[0]["alpha_cell"]]
    a, b = fit_one_over_l([r["L"] for r in same], [r["dE"] for r in same]) if len(same) >= 2 else (float("nan"), float("nan"))
    # C13: the density convention's second-moment term (a 1/V piece) is removed before the
    # monopole 1/L reading; both slopes are reported.
    from mace.modules.dscc.ladder import second_moment_term
    kcfg = getattr(model, "kernel", None)
    if kcfg is not None and getattr(kcfg, "background", "density") == "density" and len(same) >= 2:
        corrected = [r["dE"] - second_moment_term(float(r.get("volume", r["L"] ** 3)), float(kcfg.r_g), float(kcfg.eps_inf), 1.0) for r in same]
        a2, b2 = fit_one_over_l([r["L"] for r in same], corrected)
    else:
        a2, b2 = a, b
    alpha_mean = float(np.mean([r["alpha_cell"] for r in same]))
    madelung = -alpha_mean * ewald.COULOMB / (2.0 * model.kernel.eps_inf)
    report = {"rows": rows, "fit_cells": [r["tiling"] for r in same], "dE_intercept": a, "dE_slope": b, "dE_slope_minus_second_moment": b2, "dE_intercept_minus_second_moment": a2,
              "madelung_slope_expected": madelung, "dE_slope_rel_err": abs(b - madelung) / abs(madelung),
              "k_lr_ii_alpha_rel_err_max": max((r.get("k_lr_ii_alpha_rel_err", 0.0) for r in rows), default=None),
              "coupling": bool(model.coupling), "model": args.model,
              "note": "with Phi = 0 the head carries no electrostatics; the 1/L test applies to coupled models"}
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
