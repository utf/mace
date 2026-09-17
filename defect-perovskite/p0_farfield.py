"""Diagnostic (not a gate): the far-field ionic displacement pattern of a relaxed +1 vacancy cell
against the continuum q/(eps0 r^2) form. For each atom, the radial displacement u_r = u . r^ from
the ideal tiling (minimum image from the vacancy), binned by distance and species; fit
log|u_r| = a + b log r over the far field (r > r_core) -> exponent b (continuum: -2), and the
amplitude A = <|u_r| r^2> per species.

    python defect-perovskite/p0_farfield.py --relaxed ~/runs/dscc/p0/ladder2_w6_large/relaxed_334.xyz \
        --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --tiling 3,3,4 --out ~/runs/dscc/p0/farfield_w6_334.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace.modules.dscc.ladder import tiling_ladder            # noqa: E402
from dscc_embed_ladder import relative, vacancy_site           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--relaxed", required=True); ap.add_argument("--cell", required=True); ap.add_argument("--tiling", required=True)
ap.add_argument("--r_core", type=float, default=8.0); ap.add_argument("--out", required=True)
args = ap.parse_args()
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
tt = tuple(int(x) for x in args.tiling.split(","))
ideal = tiling_ladder(unit, [tt], vacancy_species=17, vacancy_index=0)[0]
rel = read(Path(args.relaxed).expanduser())
vac, cell = vacancy_site(unit, tt)
p0 = ideal.get_positions(); u = rel.get_positions() - p0
u -= np.round(u @ np.linalg.inv(cell)) @ cell
rvec = relative(p0, vac, cell); r = np.linalg.norm(rvec, axis=1); rhat = rvec / np.maximum(r, 1e-9)[:, None]
u_r = np.einsum("ij,ij->i", u, rhat); u_t = np.linalg.norm(u - u_r[:, None] * rhat, axis=1)
syms = np.array(ideal.get_chemical_symbols())
res = {"tiling": list(tt), "n_atoms": len(ideal), "r_core": args.r_core, "species": {}, "bins": []}
edges = [args.r_core, 10, 12, 14, 16, 18, 20, 24, 100]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (r > lo) & (r <= hi)
    if m.sum() == 0:
        continue
    row = {"r_lo": lo, "r_hi": hi, "n": int(m.sum()), "rms_u_r_A": float(np.sqrt((u_r[m] ** 2).mean())), "rms_u_t_A": float(np.sqrt((u_t[m] ** 2).mean()))}
    for s in ("Cl", "Cs", "Pb"):
        ms = m & (syms == s)
        if ms.any():
            row[f"mean_u_r_{s}"] = float(u_r[ms].mean()); row[f"A_{s}_eA2"] = float((u_r[ms] * r[ms] ** 2).mean())
    res["bins"].append(row)
far = r > args.r_core
for s in ("Cl", "Cs", "Pb"):
    ms = far & (syms == s) & (np.abs(u_r) > 1e-5)
    if ms.sum() > 10:
        b, a = np.polyfit(np.log(r[ms]), np.log(np.abs(u_r[ms])), 1)
        res["species"][s] = {"n": int(ms.sum()), "exponent": float(b), "mean_sign_u_r": float(np.sign(u_r[ms]).mean()),
                             "A_r2_mean": float((u_r[ms] * r[ms] ** 2).mean()), "rms_u_r": float(np.sqrt((u_r[ms] ** 2).mean()))}
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
for row in res["bins"]:
    print(f"r {row['r_lo']:>4}-{row['r_hi']:<4} n={row['n']:4d} rms u_r {row['rms_u_r_A']:.4f} A, u_t {row['rms_u_t_A']:.4f}")
print("far-field exponents:", {s: round(v["exponent"], 2) for s, v in res["species"].items()}, "(continuum -2)")
print("saved", args.out)
