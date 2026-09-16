"""L4: eps_0^model from the base's phonons with rigid latent charges as Born charges.

Gamma-point force constants of the static pristine cell (80 atoms) by central differences of
the base's forces, acoustic sum rule imposed (row sums removed), dynamical matrix
D = FC / sqrt(M_i M_j), the three acoustic modes dropped, and

    eps_0 - eps_inf = (4 pi C / 3 Omega) sum_m sum_alpha | sum_k Z*_k e^alpha_{k m} / sqrt(M_k) |^2 / omega_m^2

with Z*_k = s sqrt(eps_inf) q_LES,k (the plan's Z*_eff; q_LES sign-fixed) -- units: FC in eV/A^2,
M in amu, C = 14.3996 eV A / e^2, so the sum is dimensionless. Folded non-Gamma modes of the
primitive cell carry no net dipole and drop out by themselves. Also reported: the same sum with
the LES model's own dynamical Z* (`compute_bec`, isotropic part per atom) for comparison, and the
IR-active mode list.
    python l4_eps0_phonon.py --base ~/runs/base_v4_lesmh_prod/base_v4_lesmh_prod_base.pt --s 0.5 --sign -1 --out eps0.json
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.data import atomic_masses
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from mace import tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.ewald import COULOMB
from mace.tools import torch_geometric

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
ap.add_argument("--s", type=float, default=0.5, help="the L2 scale s_c (1/sqrt(eps_inf) = 0.5 for a base-only reading)")
ap.add_argument("--sign", type=float, default=1.0, help="Born sign fix of q_LES")
ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--h", type=float, default=0.01, help="FD step, A")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)

doc = json.load(open(args.static_cell))
atoms = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
base = torch.load(os.path.expanduser(args.base), map_location="cpu", weights_only=False).double().to(args.device).eval()
z = tools.AtomicNumberTable([int(x) for x in base.atomic_numbers])
N = len(atoms); numbers = atoms.get_atomic_numbers(); M = np.array([atomic_masses[n] for n in numbers])
Omega = abs(np.linalg.det(np.array(atoms.get_cell())))


def run(a, bec=False):
    a = a.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "ph", "source_dir": "ph"})
    batch = next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z, float(base.r_max)), batch_size=1))).to(args.device).to_dict()
    out = base(batch, training=False, compute_force=True, compute_bec=bec)
    return out


ref = run(atoms, bec=True)
q = args.sign * ref["latent_charges"].detach().cpu().numpy().reshape(-1)
Z_eff = args.s * math.sqrt(args.eps_inf) * q                                   # the plan's Z*_eff per atom
Z_dyn = args.sign * ref["BEC"].detach().cpu().numpy().reshape(N, 3, 3).diagonal(axis1=1, axis2=2).mean(-1)
f0 = ref["forces"].detach().cpu().numpy()
print(f"static cell: {N} atoms, Omega {Omega:.1f} A^3, |F0| max {np.abs(f0).max():.4f} eV/A; "
      f"q_LES species means: " + ", ".join(f"{s}={q[numbers == n].mean():+.4f}" for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))), flush=True)

FC = np.zeros((3 * N, 3 * N))
pos0 = atoms.get_positions()
for i in range(N):
    for a_ in range(3):
        p, m = atoms.copy(), atoms.copy()
        pp = pos0.copy(); pp[i, a_] += args.h; p.set_positions(pp)
        pm = pos0.copy(); pm[i, a_] -= args.h; m.set_positions(pm)
        fp = run(p)["forces"].detach().cpu().numpy(); fm = run(m)["forces"].detach().cpu().numpy()
        FC[3 * i + a_, :] = -(fp - fm).reshape(-1) / (2 * args.h)
FC = 0.5 * (FC + FC.T)
# acoustic sum rule: sum_j FC_{i alpha, j beta} = 0
for a_ in range(3):
    for b_ in range(3):
        blk = FC[a_::3, b_::3]
        np.fill_diagonal(blk, np.diag(blk) - blk.sum(1))
        FC[a_::3, b_::3] = blk
Msq = np.repeat(np.sqrt(M), 3)
Dm = FC / np.outer(Msq, Msq)
w2, e = np.linalg.eigh(Dm)                              # eV / (A^2 amu); e columns = modes
order = np.argsort(np.abs(w2)); acoustic = order[:3]
keep = np.array([m for m in range(3 * N) if m not in set(acoustic)])
print(f"acoustic |omega^2| max {np.abs(w2[acoustic]).max():.2e}, smallest optical {w2[keep].min():.4e} eV/A^2/amu; "
      f"imaginary optical modes: {int((w2[keep] < 0).sum())}", flush=True)
CONV = 4.0 * math.pi * COULOMB / (3.0 * Omega)


def eps0_from(Z):
    contrib = []
    for m in keep:
        if w2[m] <= 0:
            continue
        vec = e[:, m].reshape(N, 3)
        p = (Z[:, None] * vec / np.sqrt(M)[:, None]).sum(0)         # e / sqrt(amu), per direction
        contrib.append((float(w2[m]), float((p ** 2).sum() / w2[m])))
    total = CONV * sum(c for _, c in contrib)
    return args.eps_inf + total, sorted(contrib, key=lambda t: -t[1])[:8]


e0_eff, top_eff = eps0_from(Z_eff)
e0_dyn, top_dyn = eps0_from(Z_dyn)
# mode frequencies in meV for the record: omega = sqrt(w2 [eV/(A^2 amu)]) * hbar-scaling
MEV = 1000.0 * 6.582119569e-16 * math.sqrt(1.602176634e-19 / (1.66053907e-27 * 1e-20))   # sqrt(eV/(A^2 amu)) -> meV
res = {"base": args.base, "s": args.s, "sign": args.sign, "eps_inf": args.eps_inf, "n_atoms": N, "Omega": Omega,
       "Z_eff_species": {s: float(Z_eff[numbers == n].mean()) for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))},
       "Z_dyn_species": {s: float(Z_dyn[numbers == n].mean()) for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))},
       "sum_Z_eff": float(Z_eff.sum()), "sum_Z_dyn": float(Z_dyn.sum()),
       "eps0_model_Zeff": e0_eff, "eps0_model_Zdyn": e0_dyn, "ratio_Zeff": e0_eff / args.eps_inf,
       "imaginary_optical_modes": int((w2[keep] < 0).sum()),
       "top_modes_Zeff_meV_contrib": [(math.sqrt(w) * MEV, CONV * c) for w, c in top_eff],
       "optical_mode_range_meV": [float(math.sqrt(max(w2[keep].min(), 0)) * MEV), float(math.sqrt(w2[keep].max()) * MEV)],
       "registered_eps0": {"value": 7.2, "pm": 0.6, "eps_inf_of_source": 2.4, "ratio": 3.0}}
json.dump(res, open(os.path.expanduser(args.out), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("Z_eff_species", "Z_dyn_species", "eps0_model_Zeff", "eps0_model_Zdyn", "ratio_Zeff",
                                       "imaginary_optical_modes", "optical_mode_range_meV")}, indent=1))
print("saved", args.out)
