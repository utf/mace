"""Diagnostic 1 (Decision 0 attribution): does the E_cP Born force reach the far ions as Z* E_c?

On the core-embedded 2879-atom structure (far field ideal), with the head's dq at that geometry:
  F_born,b = -d/dR_b [ -sum_a p_a(R) . E_c^{detached}(R_a) ]  (autograd through the provider only)
  F_ref,b  = Z*_b^T E_c(R_b), Z*_b the pristine-cell Born tensor of b's tiling counterpart,
  E_c the registered smoothed carrier field (gamma_lr(r_g, r_split) / eps_inf) of dq.
Reported by shell: rms |F_born|, rms |F_ref|, their ratio, the mean cosine, and the full E_cP
force (p and E_c both attached, dq fixed) for context; |E_c| against Q C / (eps_inf r^2).
    OMP_NUM_THREADS=40 python defect-perovskite/p0_bornforce.py --out ~/runs/dscc/p0/p0_bornforce.json
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import data as mace_data, tools                                  # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.ewald import COULOMB                                # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                         # noqa: E402
from mace.modules.dscc.polarisation import carrier_field, carrier_dipole_energy   # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/alex/runs/dscc/p0/w6_s1_pol_r45.pt")
ap.add_argument("--core", default="/home/alex/runs/dscc/p0/ladder3_var_local/core.json")
ap.add_argument("--cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--tiling", default="3,3,4"); ap.add_argument("--r_core", type=float, default=8.0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
tt = tuple(int(x) for x in args.tiling.split(","))
src = json.load(open(Path(args.core).expanduser()))
vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
atoms, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
vac, cell = vacancy_site(unit, tt)
r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1)
N = len(atoms); print(f"embedded {tt}: N={N}, core {moved}/{inside}", flush=True)

model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval()
model.base = model.base.double()
z3 = tools.AtomicNumberTable([17, 55, 82])
a = atoms.copy(); a.info.update({"carrier_counts": np.array([0, 0, 1, 0], float), "cell_charge": 1, "config_type": "bf", "source_dir": "bf"})
b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
with torch.no_grad():
    out = model(dict(b), compute_force=False)
dq = out["dq"].detach(); e_cp_head = float(out["diagnostics"]["e_host"][0])
print(f"head: dq sum {float(dq.sum()):.4f}, E_cP {1e3*e_cp_head:.2f} meV, {time.time()-t0:.0f}s", flush=True)

# provider dipoles with graph, on a positions leaf
pos = b["positions"].detach().clone().requires_grad_(True)
cell_t = b["cell"].view(3, 3)
shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell_t.expand(b["edge_index"].shape[1], 3, 3))
d = dict(b); d["positions"] = pos; d["shifts"] = shifts
species = b["node_attrs"].argmax(-1)
p = model._node_dipoles(d, species)                                        # [N,3], attached to pos
E_c = carrier_field(pos.detach(), cell_t, dq, model.kernel.r_g, model.r_split, model.kernel.eps_inf, tol=model.kernel.tol)  # [N,3] V/A, detached
e_prime = -(p * E_c).sum()
(g_born,) = torch.autograd.grad(e_prime, pos, retain_graph=True)
F_born = (-g_born).detach().cpu().numpy()
e_full = carrier_dipole_energy(pos, cell_t, dq, p, model.kernel.r_g, model.r_split, model.kernel.eps_inf, tol=model.kernel.tol)
(g_full,) = torch.autograd.grad(e_full, pos)
F_full = (-g_full).detach().cpu().numpy()
print(f"E_cP recomputed {1e3*float(e_full):.2f} meV (head {1e3*e_cp_head:.2f}); {time.time()-t0:.0f}s", flush=True)

# pristine Born tensors of the tiling counterparts
prov = model.field_provider
z79 = tools.AtomicNumberTable([int(z) for z in prov.model.atomic_numbers])
cfg = mace_data.config_from_atoms(unit, key_specification=mace_data.KeySpecification.from_defaults(), head_name=prov.head_name)
ad = mace_data.AtomicData.from_config(cfg, z_table=z79, cutoff=float(prov.model.r_max), heads=list(prov.model.heads))
bu = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
Zu = prov.model(bu, training=True, compute_force=False, compute_polarization=True, compute_becs=True)["becs"].detach().cpu().numpy().reshape(len(unit), 3, 3)
rep = unit.repeat(tt); Zrep = np.tile(Zu, (int(np.prod(tt)), 1, 1))
sites = [i for i, z in enumerate(rep.get_atomic_numbers()) if int(z) == 17]
Z = np.delete(Zrep, sites[0], axis=0)                                       # the same atom tiling_ladder removes
E_np = E_c.cpu().numpy()
F_ref = np.einsum("bab,ba->bb", Z, E_np) if False else np.einsum("bij,bi->bj", Z, E_np)   # (Z^T E)_j = sum_i Z_ij E_i
syms = np.array(atoms.get_chemical_symbols())
Q = 1.0; res = {"N": N, "E_cP_head_meV": 1e3 * e_cp_head, "E_cP_meV": 1e3 * float(e_full), "dq_sum": float(dq.sum()), "shells": []}
edges = [(4, 8), (8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 24)]
print(f"{'shell':7s} {'n':>4s} {'|E_c| V/A':>10s} {'QC/eps r2':>10s} {'rms Fborn':>10s} {'rms Fref':>9s} {'ratio':>6s} {'cos':>6s} {'rms Ffull':>10s}   per species ratio (Cs, Pb, Cl)")
for lo, hi in edges:
    m = (r > lo) & (r <= hi)
    if not m.any():
        continue
    rms = lambda x: float(np.sqrt((x[m] ** 2).sum(1).mean()))
    cos = float(np.mean(np.einsum("ij,ij->i", F_born[m], F_ref[m]) / np.maximum(np.linalg.norm(F_born[m], axis=1) * np.linalg.norm(F_ref[m], axis=1), 1e-12)))
    ratio = rms(F_born) / max(rms(F_ref), 1e-12)
    per = {}
    for s in ("Cs", "Pb", "Cl"):
        ms = m & (syms == s)
        per[s] = float(np.sqrt((F_born[ms] ** 2).sum(1).mean()) / max(np.sqrt((F_ref[ms] ** 2).sum(1).mean()), 1e-12)) if ms.any() else None
    row = {"r_lo": lo, "r_hi": hi, "n": int(m.sum()), "E_c_rms_V_A": float(np.sqrt((E_np[m] ** 2).sum(1).mean())),
           "coulomb_field_V_A": float(Q * COULOMB / (model.kernel.eps_inf * ((lo + hi) / 2) ** 2)),
           "rms_F_born_meV_A": 1e3 * rms(F_born), "rms_F_ref_meV_A": 1e3 * rms(F_ref), "ratio": ratio, "cos": cos,
           "rms_F_full_meV_A": 1e3 * rms(F_full), "ratio_by_species": per}
    res["shells"].append(row)
    print(f"{lo:>2d}-{hi:<4d} {row['n']:4d} {row['E_c_rms_V_A']:10.4f} {row['coulomb_field_V_A']:10.4f} {row['rms_F_born_meV_A']:10.2f} {row['rms_F_ref_meV_A']:9.2f} {ratio:6.3f} {cos:6.3f} {row['rms_F_full_meV_A']:10.2f}   " + ", ".join(f"{s} {v:.2f}" if v is not None else f"{s} -" for s, v in per.items()), flush=True)
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out, f"{time.time()-t0:.0f}s")
