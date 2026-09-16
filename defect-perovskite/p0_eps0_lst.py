"""Plan P Phase 0.3: eps0_LST = eps_inf + (4 pi C / Omega) sum_m Z~_m Z~_m^T / omega_m^2 from the
BASE's Gamma-point Hessian (central differences of its forces on the relaxed pristine cell,
acoustic sum rule imposed) and the PROVIDER's Born tensors (mode effective charges
Z~_m,alpha = sum_k sum_beta Z*_k,alpha beta e_k beta,m / sqrt(M_k)). Reported with eps_inf = (R)
the registered 4.0 and (M) the provider's own tensor / polycrystalline mean.

    python defect-perovskite/p0_eps0_lst.py --base ~/runs/base_v2_prod/base_v2_prod_base.pt \
        --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --out ~/runs/dscc/p0/p0_eps0_lst.json
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
from mace.modules.dscc.ewald import COULOMB                  # noqa: E402
from mace.tools import torch_geometric                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--head", default="mp-dielectric")
ap.add_argument("--cell", required=True)
ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--eps_inf_R", type=float, default=4.0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device
doc = json.load(open(Path(args.cell).expanduser()))
atoms = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
N = len(atoms); numbers = atoms.get_atomic_numbers(); M = np.array([atomic_masses[n] for n in numbers])
Omega = abs(np.linalg.det(np.array(atoms.get_cell())))
base = torch.load(Path(args.base).expanduser(), map_location=dev, weights_only=False).double().to(dev).eval()
z3 = tools.AtomicNumberTable([int(x) for x in base.atomic_numbers])


def base_forces(a):
    a = a.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "ph", "source_dir": "ph"})
    b = next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z3, float(base.r_max)), batch_size=1))).to(dev).to_dict()
    return base(b, training=False, compute_force=True)["forces"].detach().cpu().numpy()


t = time.time()
f0 = base_forces(atoms)
FC = np.zeros((3 * N, 3 * N)); pos0 = atoms.get_positions()
for i in range(N):
    for a_ in range(3):
        p, m = atoms.copy(), atoms.copy()
        pp = pos0.copy(); pp[i, a_] += args.h; p.set_positions(pp)
        pm = pos0.copy(); pm[i, a_] -= args.h; m.set_positions(pm)
        FC[3 * i + a_, :] = -(base_forces(p) - base_forces(m)).reshape(-1) / (2 * args.h)
FC = 0.5 * (FC + FC.T)
for a_ in range(3):
    for b_ in range(3):
        blk = FC[a_::3, b_::3]; np.fill_diagonal(blk, np.diag(blk) - blk.sum(1)); FC[a_::3, b_::3] = blk
Msq = np.repeat(np.sqrt(M), 3)
w2, e = np.linalg.eigh(FC / np.outer(Msq, Msq))
order = np.argsort(np.abs(w2)); acoustic = set(order[:3].tolist())
keep = [m for m in range(3 * N) if m not in acoustic]
print(f"Hessian: {N} atoms, |F0|max {np.abs(f0).max():.4f} eV/A, imaginary optical {int(sum(w2[m] < 0 for m in keep))}, {time.time()-t:.0f}s", flush=True)

# provider Z* on the same cell
model = torch.load(args.ckpt, map_location=dev, weights_only=False).double().to(dev).eval()
z79 = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
cfg = mace_data.config_from_atoms(atoms, key_specification=mace_data.KeySpecification.from_defaults(), head_name=args.head)
ad = mace_data.AtomicData.from_config(cfg, z_table=z79, cutoff=float(model.r_max), heads=list(model.heads))
b = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
out = model(b, training=True, compute_force=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
Z = out["becs"].detach().cpu().numpy().reshape(N, 3, 3)          # Z[k, alpha, beta] = d(Omega P_alpha)/dR_k,beta
eps_inf_M = np.eye(3) + out["polarizability"].detach().cpu().numpy()[0]
MEV = 1000.0 * 6.582119569e-16 * math.sqrt(1.602176634e-19 / (1.66053907e-27 * 1e-20))

ionic = np.zeros((3, 3)); modes = []
for m in keep:
    if w2[m] <= 0:
        continue
    vec = e[:, m].reshape(N, 3) / np.sqrt(M)[:, None]                  # e_k / sqrt(M_k)
    zt = np.einsum("kab,kb->a", Z, vec)                                 # Z~_m,alpha
    contrib = np.outer(zt, zt) / w2[m]
    ionic += contrib
    modes.append((float(math.sqrt(w2[m]) * MEV), float(np.trace(contrib) / 3 * 4 * math.pi * COULOMB / Omega)))
ionic *= 4.0 * math.pi * COULOMB / Omega
eps0_R = args.eps_inf_R * np.eye(3) + ionic
eps0_M = eps_inf_M + ionic
res = {"base": args.base, "cell": args.cell, "n_atoms": N, "Omega": Omega, "h": args.h,
       "imaginary_optical_modes": int(sum(w2[m] < 0 for m in keep)),
       "optical_mode_range_meV": [float(math.sqrt(max(min(w2[m] for m in keep), 0)) * MEV), float(math.sqrt(max(w2[m] for m in keep)) * MEV)],
       "Z_species_trace3": {s: float(np.mean([np.trace(Z[k]) / 3 for k in range(N) if numbers[k] == n])) for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))},
       "eps_inf_M_tensor": eps_inf_M.tolist(), "eps_inf_M_poly": float(np.trace(eps_inf_M) / 3),
       "ionic_tensor": ionic.tolist(), "ionic_poly": float(np.trace(ionic) / 3),
       "eps0_LST_R_tensor": eps0_R.tolist(), "eps0_LST_R_poly": float(np.trace(eps0_R) / 3),
       "eps0_LST_M_tensor": eps0_M.tolist(), "eps0_LST_M_poly": float(np.trace(eps0_M) / 3),
       "top_modes_meV_contrib": sorted(modes, key=lambda t: -t[1])[:10],
       "literature_note": "CsPbCl3 cubic eps0 ~ 7.2 (LST, PBE; source of the previously registered value), eps_inf ~ 2.4 in that source"}
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print(f"ionic (poly) {res['ionic_poly']:.3f}; eps0_LST (R, eps_inf 4.0) {res['eps0_LST_R_poly']:.3f}; eps0_LST (M, eps_inf {res['eps_inf_M_poly']:.3f}) {res['eps0_LST_M_poly']:.3f}; "
      f"optical modes {res['optical_mode_range_meV'][0]:.1f}-{res['optical_mode_range_meV'][1]:.1f} meV; top {res['top_modes_meV_contrib'][:4]}")
print("saved", args.out)
