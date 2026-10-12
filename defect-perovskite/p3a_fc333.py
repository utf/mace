"""Plan P Phase 3a §2.2b, data: central finite displacement (h) of the 80 atoms of the first image of the 3x3x3 pristine
tiling, base v8 w01 alone; for each displacement the forces and the node dipoles on all atoms. 480 calls; resumable.
Writes G[k, b, b', a] = -dF_{b'a}/du_{kb} (eV/A^2) and Lam[k, b, b', a] = dp_{b'a}/du_{kb} (e) to <out>.npz.

    OMP_NUM_THREADS=48 python defect-perovskite/p3a_fc333.py --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json --out ~/runs/dscc/p3a/fc333_raw
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from p3a_common import BASE_V8, Base, load_cell   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--base", default=BASE_V8); ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--tiling", default="3,3,3"); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64)
unit = load_cell(args.cell); N = len(unit); big = unit.repeat(tuple(int(x) for x in args.tiling.split(","))); NB = len(big)
out = Path(args.out).expanduser().with_suffix(".npz")
if out.exists():
    d = np.load(out); G = d["G"]; Lam = d["Lam"]; done = d["done"]; f0 = d["f0"]; p0 = d["p0"]; e0 = float(d["e0"])
    print(f"resuming: {int(done.sum())}/{N} atoms done", flush=True)
else:
    G = np.zeros((N, 3, NB, 3)); Lam = np.zeros((N, 3, NB, 3)); done = np.zeros(N, bool); f0 = p0 = None; e0 = 0.0
B = Base(args.base, args.device); t0 = time.time()
if f0 is None:
    e0, f0, p0 = B.forces_dipoles(big); print(f"undisplaced: E {e0:.4f} eV, fmax {np.abs(f0).max():.4f}, {time.time()-t0:.0f}s", flush=True)
R0 = big.get_positions()
for k in range(N):
    if done[k]:
        continue
    for b in range(3):
        fp = fm = None
        for sgn in (+1, -1):
            a = big.copy(); pos = R0.copy(); pos[k, b] += sgn * args.h; a.set_positions(pos)
            _, f, p = B.forces_dipoles(a)
            if sgn > 0: fp, pp = f, p
            else: fm, pm = f, p
        G[k, b] = -(fp - fm) / (2 * args.h); Lam[k, b] = (pp - pm) / (2 * args.h)
    done[k] = True
    np.savez(out, G=G, Lam=Lam, done=done, f0=f0, p0=p0, e0=e0, h=args.h, positions=R0, numbers=big.get_atomic_numbers(), cell=np.array(big.get_cell()),
             unit_positions=unit.get_positions(), unit_cell=np.array(unit.get_cell()), tiling=np.array([int(x) for x in args.tiling.split(",")]))
    el = time.time() - t0; nd = int(done.sum())
    print(f"atom {k} ({unit.get_chemical_symbols()[k]}) done: {nd}/{N}, {el/60:.1f} min elapsed, Z*_fd trace/3 {np.trace(Lam[k].sum(1).T)/3:.3f}", flush=True)
print("done", out)
