"""Plan P amendment A1 (replaces 0.1): the provider's plane-wave Born response.

f(q) = |sum_a (q^ . dp_a) e^{i q . R_a}| / |sum_a (q^ . Z*_a u_a) e^{i q . R_a}| for the modulated
pattern u_a = A e_a / sqrt(M_a) cos(q . R_a) (e = the base's strongest IR mode) in the 3x3x3
tiling, q = n 2 pi / L_axis, n = 1..4, along x, y, z; dp by central differences of the provider's
node dipoles. w_eff per axis from -4 ln f = w^2 q^2 (least squares). Rule: r_split = max(4.5,
w_eff) if w_eff exceeds 4.5 A on any axis.

    OMP_NUM_THREADS=48 python defect-perovskite/p0_formfactor.py --base ~/runs/base_v2_prod/base_v2_prod_base.pt \
        --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --device cpu --out ~/runs/dscc/p0/p0_formfactor.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.data import atomic_masses
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                    # noqa: E402
from mace.modules.dscc import data as dd                     # noqa: E402
from mace.modules.dscc.field import load_field_provider      # noqa: E402
from mace.tools import torch_geometric                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--cell", required=True)
ap.add_argument("--amp", type=float, default=0.05)
ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--r_split_floor", type=float, default=4.5)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
N = len(unit); numbers = unit.get_atomic_numbers(); M = np.array([atomic_masses[n] for n in numbers])
z3 = tools.AtomicNumberTable([17, 55, 82])
base = torch.load(Path(args.base).expanduser(), map_location=dev, weights_only=False).double().to(dev).eval()


def batch(a, r_cut=6.0):
    a = a.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "ff", "source_dir": "ff"})
    return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z3, r_cut), batch_size=1))).to(dev).to_dict()


def base_forces(a):
    return base(batch(a), training=False, compute_force=True)["forces"].detach().cpu().numpy()


t = time.time(); pos0 = unit.get_positions()
FC = np.zeros((3 * N, 3 * N))
for i in range(N):
    for c in range(3):
        p, m = unit.copy(), unit.copy()
        pp = pos0.copy(); pp[i, c] += args.h; p.set_positions(pp)
        pm = pos0.copy(); pm[i, c] -= args.h; m.set_positions(pm)
        FC[3 * i + c] = -(base_forces(p) - base_forces(m)).reshape(-1) / (2 * args.h)
FC = 0.5 * (FC + FC.T)
for a_ in range(3):
    for b_ in range(3):
        blk = FC[a_::3, b_::3]; np.fill_diagonal(blk, np.diag(blk) - blk.sum(1)); FC[a_::3, b_::3] = blk
Msq = np.repeat(np.sqrt(M), 3)
w2, e = np.linalg.eigh(FC / np.outer(Msq, Msq))
prov = load_field_provider(args.ckpt, head="mp-dielectric", device=dev)
z79 = tools.AtomicNumberTable([int(z) for z in prov.model.atomic_numbers])
cfg = mace_data.config_from_atoms(unit, key_specification=mace_data.KeySpecification.from_defaults(), head_name="mp-dielectric")
ad = mace_data.AtomicData.from_config(cfg, z_table=z79, cutoff=float(prov.model.r_max), heads=list(prov.model.heads))
b = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
Z = prov.model(b, training=True, compute_force=False, compute_polarization=True, compute_becs=True)["becs"].detach().cpu().numpy().reshape(N, 3, 3)
strength = []
for mm in range(3 * N):
    if w2[mm] <= 1e-6:
        continue
    vec = e[:, mm].reshape(N, 3) / np.sqrt(M)[:, None]
    zt = np.einsum("kab,kb->a", Z, vec)
    strength.append((float((zt ** 2).sum() / w2[mm]), mm, zt))
strength.sort(key=lambda s: -s[0])
_, m_star, zt = strength[0]
MEV = 1000.0 * 6.582119569e-16 * math.sqrt(1.602176634e-19 / (1.66053907e-27 * 1e-20))
print(f"mode {m_star}: {math.sqrt(w2[m_star]) * MEV:.2f} meV, Z~ {np.round(zt, 3)}; {time.time()-t:.0f}s", flush=True)

big = unit.repeat((3, 3, 3)); NB = len(big); cellB = np.array(big.get_cell())
Rb = big.get_positions(); vec_big = np.tile(e[:, m_star].reshape(N, 3) / np.sqrt(M)[:, None], (27, 1)); Zb = np.tile(Z, (27, 1, 1))


def dipoles(a):
    bq = batch(a, r_cut=6.0)
    pos = bq["positions"]; cell_t = bq["cell"].view(3, 3)
    sh = torch.einsum("ei,eij->ej", bq["unit_shifts"], cell_t.expand(bq["edge_index"].shape[1], 3, 3))
    nums = torch.tensor([[17, 55, 82][i] for i in bq["node_attrs"].argmax(-1).tolist()], device=dev)
    with torch.enable_grad():
        p = prov.node_dipoles(pos.detach().requires_grad_(True), cell_t, bq["batch"], bq["ptr"], bq["edge_index"], sh, bq["unit_shifts"], nums)
    return p.detach().cpu().numpy()


res = {"mode": int(m_star), "omega_meV": float(math.sqrt(w2[m_star]) * MEV), "Ztilde": zt.tolist(), "amp": args.amp, "axes": {}}
for axis, name in enumerate("xyz"):
    L = float(np.linalg.norm(cellB[axis])); rows = []
    for n in args.orders:
        q = np.zeros(3); q[axis] = n * 2 * math.pi / L; qhat = q / np.linalg.norm(q)
        phase = np.cos(Rb @ q); u = vec_big * phase[:, None]
        t0 = time.time()
        dp = (dipoles(Atoms(numbers=big.get_atomic_numbers(), positions=Rb + args.amp * u, cell=cellB, pbc=True))
              - dipoles(Atoms(numbers=big.get_atomic_numbers(), positions=Rb - args.amp * u, cell=cellB, pbc=True))) / (2 * args.amp)
        ph = np.exp(1j * (Rb @ q))
        S_model = np.sum([(qhat @ dp[k]) * ph[k] for k in range(NB)])
        S_born = np.sum([(qhat @ (Zb[k] @ u[k])) * ph[k] for k in range(NB)])
        f = abs(S_model) / abs(S_born) if abs(S_born) > 0 else float("nan")
        # also the longitudinal-vs-transverse: report the projection along the mode's polarisation
        rows.append({"n": n, "q": float(np.linalg.norm(q)), "lambda": L / n, "f": f, "phase_deg": float(np.degrees(np.angle(S_model / S_born))) if abs(S_born) > 0 else None,
                     "S_born": abs(S_born), "seconds": time.time() - t0})
        print(f"  {name} n={n} lambda={L/n:.1f} A q={np.linalg.norm(q):.3f} f={f:.4f} ({time.time()-t0:.0f}s)", flush=True)
    qs = np.array([r["q"] for r in rows]); fs = np.array([r["f"] for r in rows])
    ok = fs > 0
    w2fit = float(np.linalg.lstsq((qs[ok] ** 2)[:, None], -4 * np.log(fs[ok]), rcond=None)[0][0]) if ok.any() else float("nan")
    w_eff = math.sqrt(max(w2fit, 0.0))
    res["axes"][name] = {"L": L, "rows": rows, "w_eff": w_eff, "w_eff_per_point": [2 * math.sqrt(max(-math.log(r["f"]), 0)) / r["q"] if r["f"] > 0 else None for r in rows]}
    print(f"axis {name}: w_eff {w_eff:.2f} A (per point {[round(x,2) if x else None for x in res['axes'][name]['w_eff_per_point']]})", flush=True)
w_max = max(v["w_eff"] for v in res["axes"].values())
res["w_eff_max"] = w_max; res["r_split"] = max(args.r_split_floor, w_max)
print(f"w_eff max {w_max:.2f} A -> r_split = max({args.r_split_floor}, w_eff) = {res['r_split']:.2f} A")
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
