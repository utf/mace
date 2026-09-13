"""Ladder v6, part 2: the relaxation ladder (ionic screening).

Starting from the EMBEDDED structure of part 1, only the far field beyond `R_core` has to move,
so BFGS converges quickly. The quantity is

    dE_relax(L) = E_relaxed(L) - E_embedded(L),

the energy of relaxing the far field. Its `1/L` coefficient is `(C alpha / 2)(1/eps_inf -
1/eps_0)`, opposite in sign to the frozen ladder's and, with `eps_0` several times `eps_inf`,
most of its magnitude. Fitting it with `eps_0` as the single free parameter besides the constant
turns the ladder into a measurement of the model's effective static dielectric response -- a test
of the physics the host term was built to supply, needing no labels. A slope near zero would mean
the model's far-field ionic response is missing.

Near-cubic cells only: clipping is physics here, not noise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from dscc_relax import DSCCCalculator                                    # noqa: E402
from dscc_embed_ladder import embed, vacancy_site, relative, KEY_TOL     # noqa: E402
from mace.modules.dscc.ewald import COULOMB, madelung_self               # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                       # noqa: E402

EPS_INF = 4.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--core_key", default="2,2,2")
    ap.add_argument("--r_core", type=float, default=8.0)
    ap.add_argument("--tilings", nargs="+", default=["2,2,3", "3,3,4"])
    ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    doc = json.load(open(args.static_cell))
    unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
    src = json.load(open(args.core))[args.core_key]
    src_t = tuple(int(x) for x in args.core_key.split(","))
    pristine = tiling_ladder(unit, [src_t], vacancy_species=17, vacancy_index=0)[0]
    vac0, cell0 = vacancy_site(unit, src_t)
    p0 = np.array(pristine.get_positions()); p1 = np.array(src["positions"])
    u = p1 - p0
    u -= np.round(u @ np.linalg.inv(cell0)) @ cell0
    rel0 = relative(p0, vac0, cell0)
    core = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(p0))}

    model = torch.load(args.model, weights_only=False, map_location=args.device).to(args.device).eval()
    rows = []
    for t in args.tilings:
        tt = tuple(int(x) for x in t.split(","))
        atoms, moved, missing, inside = embed(unit, tt, core, (vac0, cell0), args.r_core)
        cell = np.array(atoms.get_cell())
        L = float(abs(np.linalg.det(cell))) ** (1 / 3)
        al = -float(madelung_self(torch.tensor(cell, dtype=torch.float64))) * L / COULOMB
        atoms.calc = DSCCCalculator(model, [0, 0, 1, 0], 1, device=args.device)
        e_embedded = atoms.get_potential_energy()
        t0 = time.time()
        opt = BFGS(atoms, logfile="-")
        opt.run(fmax=args.fmax, steps=args.steps)
        e_relaxed = atoms.get_potential_energy()
        f = np.linalg.norm(atoms.get_forces(), axis=1).max()
        rows.append({"tiling": list(tt), "n_atoms": len(atoms), "L": L, "alpha_cell": al,
                     "r_core": args.r_core, "E_embedded": e_embedded, "E_relaxed": e_relaxed,
                     "dE_relax": e_relaxed - e_embedded, "fmax": float(f),
                     "steps": int(opt.get_number_of_steps()), "seconds": time.time() - t0,
                     "core_moved": moved, "core_missing": missing,
                     "positions": atoms.get_positions().tolist()})
        print("%s n=%5d L=%.2f  dE_relax = %+.4f eV  (%d steps, fmax %.3f, %.0f s)"
              % (t, len(atoms), L, rows[-1]["dE_relax"], rows[-1]["steps"], f, rows[-1]["seconds"]), flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)
    if len(rows) >= 2:
        L = np.array([r["L"] for r in rows]); al = np.array([r["alpha_cell"] for r in rows])
        y = np.array([r["dE_relax"] for r in rows])
        A = np.stack([np.ones_like(L), al / L], 1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        # beta[1] = (C/2)(1/eps_inf - 1/eps_0)  ->  eps_0
        k = 2.0 * beta[1] / COULOMB
        eps0 = 1.0 / (1.0 / EPS_INF - k) if (1.0 / EPS_INF - k) > 0 else float("inf")
        print("\ndE_relax(inf) = %.4f eV;  slope %+.4f eV.A  ->  eps_0(model) = %.2f  (eps_inf = %.1f)"
              % (beta[0], beta[1], eps0, EPS_INF))
        json.dump({"rows": rows, "dE_relax_inf": beta[0], "slope": beta[1], "eps0_model": eps0},
                  open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
