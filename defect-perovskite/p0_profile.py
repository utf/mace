"""Diagnostic 2: species-signed radial displacement profile of relaxed vacancy cells by shell
(cations out, Cl in, ~r^-2 = polar response; same sign, steeper = the core's elastic field).
    python defect-perovskite/p0_profile.py --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --tiling 3,3,4 \
        --relaxed w6=~/runs/dscc/p0/ladder2_w6_large/relaxed_334.xyz var45=~/runs/dscc/p0/ladder3_var_large/relaxed_334.xyz --out ~/runs/dscc/p0/p0_profile.json
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
ap.add_argument("--cell", required=True); ap.add_argument("--tiling", required=True)
ap.add_argument("--relaxed", nargs="+", required=True, help="label=path"); ap.add_argument("--out", required=True)
args = ap.parse_args()
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
tt = tuple(int(x) for x in args.tiling.split(","))
ideal = tiling_ladder(unit, [tt], vacancy_species=17, vacancy_index=0)[0]
vac, cell = vacancy_site(unit, tt)
p0 = ideal.get_positions(); rvec = relative(p0, vac, cell); r = np.linalg.norm(rvec, axis=1); rhat = rvec / np.maximum(r, 1e-9)[:, None]
syms = np.array(ideal.get_chemical_symbols())
edges = [(8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 24)]
res = {"tiling": list(tt), "arms": {}}
print(f"{'arm':6s} {'shell':7s} " + " ".join(f"{s+' <u_r> mA':>13s} {s+' <u_r>r^2':>11s}" for s in ("Cs", "Pb", "Cl")))
for spec in args.relaxed:
    label, path = spec.split("=", 1)
    rel = read(Path(path).expanduser()); u = rel.get_positions() - p0; u -= np.round(u @ np.linalg.inv(cell)) @ cell
    u_r = np.einsum("ij,ij->i", u, rhat)
    rows = []
    for lo, hi in edges:
        m = (r > lo) & (r <= hi); row = {"r_lo": lo, "r_hi": hi}
        for s in ("Cs", "Pb", "Cl"):
            ms = m & (syms == s)
            row[s] = {"n": int(ms.sum()), "mean_u_r_mA": 1e3 * float(u_r[ms].mean()) if ms.any() else None,
                      "mean_u_r_r2": float((u_r[ms] * r[ms] ** 2).mean()) if ms.any() else None,
                      "frac_outward": float((u_r[ms] > 0).mean()) if ms.any() else None}
        rows.append(row)
        print(f"{label:6s} {lo:>2d}-{hi:<4d} " + " ".join(f"{row[s]['mean_u_r_mA']:>13.2f} {row[s]['mean_u_r_r2']:>11.3f}" for s in ("Cs", "Pb", "Cl")))
    fits = {}
    for s in ("Cs", "Pb", "Cl"):
        ms = (r > 10) & (r <= 24) & (syms == s)
        sgn = np.sign(u_r[ms].mean()); mm = ms & (np.sign(u_r) == sgn) & (np.abs(u_r) > 1e-5)
        b, a = np.polyfit(np.log(r[mm]), np.log(np.abs(u_r[mm])), 1) if mm.sum() > 10 else (float("nan"), float("nan"))
        fits[s] = {"sign": float(sgn), "exponent": float(b), "n": int(mm.sum())}
    res["arms"][label] = {"shells": rows, "fit_10_24": fits}
    print(f"       exponents (10-24 A, majority sign): " + ", ".join(f"{s} {v['sign']:+.0f} r^{v['exponent']:.2f}" for s, v in fits.items()))
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
