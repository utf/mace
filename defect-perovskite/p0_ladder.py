"""Plan P Phase 0.3, route-1 relaxation ladder (and Phase 3's, with the same code).

For a D-SCC checkpoint (W6 as shipped, or the polarisation variant from `p0_variant.py`):
  stage `core`   : relax the +1 vacancy fully in the smallest tiling (fixed cell, BFGS) and
                   store the displacements keyed by minimum-image position from the vacancy;
  stage `ladder` : for each tiling from the base-relaxed pristine cell, E_frozen = E(+1) at the
                   ideal vacancy geometry, then the core displacements embedded within r_core
                   and the FAR FIELD relaxed with the full model (core atoms fixed) -> E_relaxed;
                   dE_relax(L) = E_relaxed - E_frozen. Fixed cell and charge state throughout,
                   so C_Q and E_M cancel: `model.pt`, not the calibrated one.
  fit            : dE_relax = dE_inf + (Q^2 C / 2) x (1/eps_inf - 1/eps0_ladder), x = alpha(h)/L
                   (the cell-shape Madelung constant, as `dscc_embed_ladder.py`), least squares.

    python defect-perovskite/p0_ladder.py --model ~/runs/dscc/p0/w6_s1_polarisation.pt \
        --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --stage core --tilings 2,2,2 --out ~/runs/dscc/p0/ladder_var
    python defect-perovskite/p0_ladder.py --model ... --stage ladder --tilings 2,2,2 2,2,3 3,3,3 3,3,4 --out ...
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                            # noqa: E402
from mace.modules.dscc import data as dd                          # noqa: E402
from mace.modules.dscc.ewald import COULOMB, madelung_self        # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                # noqa: E402
from mace.tools import torch_geometric                            # noqa: E402
from dscc_relax import DSCCCalculator                             # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cell", required=True, help="base-relaxed pristine cell json")
ap.add_argument("--stage", choices=["core", "ladder"], required=True)
ap.add_argument("--tilings", nargs="+", default=["2,2,2"])
ap.add_argument("--core_tiling", default="2,2,2")
ap.add_argument("--r_core", type=float, default=8.0)
ap.add_argument("--fmax", type=float, default=0.02)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--optimizer", default="bfgs", choices=["bfgs", "fire"])
ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--counts", nargs=4, type=int, default=[0, 0, 1, 0])
ap.add_argument("--charge", type=int, default=1)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True, help="output prefix (directory created)")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
out_dir = Path(args.out).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=args.device).to(args.device).eval()
model.base = model.base.double()      # the W6 checkpoints carry a float32 base (base_float32); Phase 0 evaluates in float64
print("base precision:", next(model.base.parameters()).dtype, flush=True)
calc = DSCCCalculator(model, args.counts, args.charge, device=args.device)


def energy_forces(atoms):
    a = atoms.copy(); a.calc = calc
    return float(a.get_potential_energy()), a.get_forces()


def relax(atoms, fixed=None, log=None):
    a = atoms.copy(); a.calc = calc
    if fixed is not None and len(fixed):
        a.set_constraint(FixAtoms(indices=list(fixed)))
    opt = (BFGS(a, logfile=log, maxstep=0.1) if args.optimizer == "bfgs" else FIRE(a, logfile=log))
    t0 = time.time()
    converged = opt.run(fmax=args.fmax, steps=args.steps)
    return a, {"converged": bool(converged), "steps": opt.get_number_of_steps(), "seconds": time.time() - t0,
               "n_calls": calc.n_calls, "fmax_final": float(np.abs(a.get_forces()).max())}


core_file = out_dir / "core.json"
if args.stage == "core":
    tt = tuple(int(x) for x in args.core_tiling.split(","))
    ideal = tiling_ladder(unit, [tt], vacancy_species=17, vacancy_index=0)[0]
    e_frozen, f0 = energy_forces(ideal)
    print(f"core {tt}: n={len(ideal)}, E_frozen {e_frozen:.4f}, |F|max {np.abs(f0).max():.3f}", flush=True)
    relaxed, info = relax(ideal, log=str(out_dir / "core_relax.log"))
    e_rel, _ = energy_forces(relaxed)
    vac, cell = vacancy_site(unit, tt)
    p0 = np.array(ideal.get_positions()); u = np.array(relaxed.get_positions()) - p0
    u -= np.round(u @ np.linalg.inv(cell)) @ cell
    rel0 = relative(p0, vac, cell); r = np.linalg.norm(rel0, axis=1)
    json.dump({"tiling": list(tt), "n_atoms": len(ideal), "E_frozen": e_frozen, "E_relaxed": e_rel, "dE_relax": e_rel - e_frozen,
               "positions": relaxed.get_positions().tolist(), "displacements": u.tolist(), "rel_positions": rel0.tolist(),
               "max_disp_A": float(np.linalg.norm(u, axis=1).max()),
               "disp_by_shell": {f"{a}-{b}": float(np.sqrt((np.linalg.norm(u[(r > a) & (r <= b)], axis=1) ** 2).mean())) if ((r > a) & (r <= b)).any() else None
                                 for a, b in ((0, 4), (4, 8), (8, 12), (12, 100))},
               "relax": info, "model": args.model, "fmax": args.fmax}, open(core_file, "w"), indent=1)
    print(f"core relaxed: dE_relax {e_rel - e_frozen:.4f} eV, max disp {np.linalg.norm(u, axis=1).max():.3f} A, {info}", flush=True)
    sys.exit(0)

src = json.load(open(core_file))
src_t = tuple(src["tiling"])
vac0, cell0 = vacancy_site(unit, src_t)
rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
rows_file = out_dir / "ladder.json"
rows = json.load(open(rows_file)) if rows_file.exists() else []
done = {tuple(r["tiling"]) for r in rows}
for t in args.tilings:
    tt = tuple(int(x) for x in t.split(","))
    if tt in done:
        continue
    t0 = time.time()
    ideal = tiling_ladder(unit, [tt], vacancy_species=17, vacancy_index=0)[0]
    cell = np.array(ideal.get_cell()); L = float(abs(np.linalg.det(cell))) ** (1 / 3)
    al = -float(madelung_self(torch.tensor(cell, dtype=torch.float64))) * L / COULOMB
    e_frozen, f0 = energy_forces(ideal)
    emb, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
    vac, _ = vacancy_site(unit, tt)
    r = np.linalg.norm(relative(np.array(emb.get_positions()), vac, cell), axis=1)
    fixed = np.nonzero(r <= args.r_core)[0]
    e_emb, f_emb = energy_forces(emb)
    print(f"{tt}: n={len(ideal)} L={L:.2f} alpha={al:.4f} E_frozen {e_frozen:.4f}; embedded core {moved} moved/{missing} missing/{inside} inside, "
          f"E_emb {e_emb:.4f}, far-field |F|max {np.abs(f_emb[r > args.r_core]).max():.3f}", flush=True)
    relaxed, info = relax(emb, fixed=fixed, log=str(out_dir / f"relax_{t.replace(',', '')}.log"))
    e_rel, _ = energy_forces(relaxed)
    row = {"tiling": list(tt), "n_atoms": len(ideal), "L": L, "alpha_cell": al, "x": al / L, "E_frozen": e_frozen, "E_embedded": e_emb,
           "E_relaxed": e_rel, "dE_relax": e_rel - e_frozen, "dE_embed_only": e_emb - e_frozen, "core_moved": moved, "core_missing": missing,
           "core_inside": inside, "n_fixed": int(len(fixed)), "relax": info, "seconds": time.time() - t0}
    rows.append(row)
    json.dump(rows, open(rows_file, "w"), indent=1)
    print(f"  -> dE_relax {row['dE_relax']:.4f} eV (embed only {row['dE_embed_only']:.4f}); {info}", flush=True)
    relaxed.write(out_dir / f"relaxed_{t.replace(',', '')}.xyz")

# fit
if len(rows) >= 2:
    x = np.array([r["x"] for r in rows]); y = np.array([r["dE_relax"] for r in rows])
    A = np.stack([np.ones_like(x), x], 1)
    (a, b), res_, *_ = np.linalg.lstsq(A, y, rcond=None)
    q2c2 = args.charge ** 2 * COULOMB / 2
    inv = 1.0 / args.eps_inf - b / q2c2
    eps0 = 1.0 / inv if inv > 0 else float("inf")
    loo = []
    for k in range(len(rows)):
        if len(rows) < 3:
            break
        m = np.ones(len(rows), bool); m[k] = False
        (_, bk), *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        ik = 1.0 / args.eps_inf - bk / q2c2
        loo.append(1.0 / ik if ik > 0 else float("inf"))
    fit = {"dE_relax_inf": a, "slope_eV_A": b, "eps_inf": args.eps_inf, "eps0_ladder": eps0, "leave_one_out_eps0": loo,
           "points": [(r["n_atoms"], r["x"], r["dE_relax"]) for r in rows]}
    json.dump(fit, open(out_dir / "fit.json", "w"), indent=1)
    print("fit:", json.dumps(fit), flush=True)
