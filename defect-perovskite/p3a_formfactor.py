"""Plan P Phase 3a §2.2c: the form factor of base v8 w01's dipoles (P0 A1 procedure with the base as its own provider):
f(q) = |sum_b (qhat . dp_b) e^{iq.R_b}| / |sum_b (qhat . Z*_b u_b) e^{iq.R_b}| for the pattern u_b = A e_b/sqrt(M_b) cos(q.R_b)
(e = the strongest IR mode of the Gamma Hessian, from p3a_lst.npz) in the 3x3x3 tiling, q = n 2 pi/L_axis, n = 1, 2, 3
(lambda = 48 / 24 / 16 A along x and y; 34 / 17 / 11 A along z); dp by central differences of the node dipoles.

    OMP_NUM_THREADS=40 python defect-perovskite/p3a_formfactor.py --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json --lst ~/runs/dscc/p3a/p3a_lst.npz --out ~/runs/dscc/p3a/p3a_formfactor.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from p3a_common import BASE_V8, MEV, Base, load_cell   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--lst", required=True); ap.add_argument("--base", default=BASE_V8)
ap.add_argument("--amp", type=float, default=0.05); ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64); t = time.time()
unit = load_cell(args.cell); N = len(unit); lst = np.load(args.lst); w2, e, M, Z = lst["w2"], lst["e"], lst["M"], lst["Z"]
keep = [m for m in range(3 * N) if w2[m] > 1e-6]
zt = {m: np.einsum("kab,kb->a", Z, e[:, m].reshape(N, 3) / np.sqrt(M)[:, None]) for m in keep}
m_star = max(keep, key=lambda m: (zt[m] ** 2).sum() / w2[m])
print(f"strongest IR mode {m_star}: {math.sqrt(w2[m_star]) * MEV:.2f} meV, Z~ {np.round(zt[m_star], 3)}", flush=True)
B = Base(args.base, args.device)
big = unit.repeat((3, 3, 3)); NB = len(big); cellB = np.array(big.get_cell()); Rb = big.get_positions()
vec_big = np.tile(e[:, m_star].reshape(N, 3) / np.sqrt(M)[:, None], (27, 1)); Zb = np.tile(Z, (27, 1, 1))
res = {"mode": int(m_star), "omega_meV": float(math.sqrt(w2[m_star]) * MEV), "Ztilde": zt[m_star].tolist(), "amp": args.amp, "axes": {}}
for axis, name in enumerate("xyz"):
    L = float(np.linalg.norm(cellB[axis])); rows = []
    for n in args.orders:
        q = np.zeros(3); q[axis] = n * 2 * math.pi / L; qhat = q / np.linalg.norm(q); phase = np.cos(Rb @ q); u = vec_big * phase[:, None]; t0 = time.time()
        dp = (B.dipoles(Atoms(numbers=big.get_atomic_numbers(), positions=Rb + args.amp * u, cell=cellB, pbc=True))
              - B.dipoles(Atoms(numbers=big.get_atomic_numbers(), positions=Rb - args.amp * u, cell=cellB, pbc=True))) / (2 * args.amp)
        ph = np.exp(1j * (Rb @ q))
        S_model = np.sum((dp @ qhat) * ph); S_born = np.sum(np.einsum("a,kab,kb->k", qhat, Zb, u) * ph)
        f = float(abs(S_model) / abs(S_born)) if abs(S_born) > 0 else None
        rows.append({"n": n, "q": float(np.linalg.norm(q)), "lambda": L / n, "f": f, "phase_deg": float(np.degrees(np.angle(S_model / S_born))), "S_born": float(abs(S_born)), "seconds": time.time() - t0})
        print(f"  {name} n={n} lambda={L/n:.1f} A q={np.linalg.norm(q):.3f}: f={f:.4f} ({time.time()-t0:.0f}s)", flush=True)
    qs = np.array([r["q"] for r in rows]); fs = np.array([r["f"] for r in rows]); ok = fs > 0
    w2fit = float(np.linalg.lstsq((qs[ok] ** 2)[:, None], -4 * np.log(fs[ok]), rcond=None)[0][0]); w_eff = math.sqrt(max(w2fit, 0.0))
    res["axes"][name] = {"L": L, "rows": rows, "w_eff_A": w_eff}
    print(f"{name}: w_eff {w_eff:.2f} A (Gaussian fit -4 ln f = w^2 q^2)", flush=True)
res["seconds"] = time.time() - t
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
