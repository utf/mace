"""Check of the E_c factor: (i) the registered smoothed periodic field of a UNIT charge at the
vacancy site through `carrier_field`, against the analytic smeared field C[erf(r/w)/r^2 -
2 e^{-r^2/w^2}/(sqrt(pi) w r)]/eps_inf summed over the nearest images (real space), by shell;
(ii) the head's dq at the embedded 2879-atom geometry: sum within 5 / 8 / 12 A of the vacancy,
participation number 1/sum(dq_i^2) (dq normalised), and the field of the ACTUAL dq against the
unit-charge field, by shell.
    OMP_NUM_THREADS=40 python defect-perovskite/p0_field_check.py --out ~/runs/dscc/p0/p0_field_check.json
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.ewald import COULOMB                                # noqa: E402
from mace.modules.dscc.polarisation import carrier_field, pair_width_cp    # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/alex/runs/dscc/p0/w6_s1_pol_r45.pt")
ap.add_argument("--core", default="/home/alex/runs/dscc/p0/ladder3_var_large/core.json")
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
atoms, *_ = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
vac, cell = vacancy_site(unit, tt)
pos_np = np.array(atoms.get_positions()); rvec = relative(pos_np, vac, cell); r = np.linalg.norm(rvec, axis=1); rhat = rvec / np.maximum(r, 1e-9)[:, None]
N = len(atoms)
model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval(); model.base = model.base.double()
r_g, r_split, eps = model.kernel.r_g, model.r_split, model.kernel.eps_inf; w = pair_width_cp(r_g, r_split)
print(f"kernel: r_g {r_g}, r_split {r_split}, eps_inf {eps}, pair width w = {w:.3f} A, tol {model.kernel.tol}", flush=True)

# (i) unit charge at the vacancy site through the kernel: put a probe "atom" at the vacancy site
# with dq = 1 and all real atoms with dq = 0; the field at the real atoms is the kernel's.
allpos = torch.tensor(np.vstack([vac[None, :], pos_np]), dtype=torch.float64)
cell_t = torch.tensor(cell, dtype=torch.float64)
dq_unit = torch.zeros(N + 1); dq_unit[0] = 1.0
E_unit = carrier_field(allpos, cell_t, dq_unit, r_g, r_split, eps, tol=model.kernel.tol)[1:].numpy()
E_unit_r = np.einsum("ij,ij->i", E_unit, rhat)
# analytic smeared field of the same width, real-space over images within 3 cells
def smeared_field(dvec):
    d = np.linalg.norm(dvec, axis=-1); x = d / w
    mag = COULOMB / eps * (math.erf(0) * 0 + np.vectorize(math.erf)(x) / d ** 2 - 2.0 * np.exp(-x * x) / (math.sqrt(math.pi) * w * d))
    return mag[..., None] * dvec / d[..., None]
imgs = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]) @ cell
E_an = np.zeros((N, 3)); E_an_direct = np.zeros((N, 3))
for L in imgs:
    f = smeared_field(rvec + L); E_an += f
    if np.allclose(L, 0):
        E_an_direct = f
E_an_r = np.einsum("ij,ij->i", E_an, rhat); E_dir_r = np.einsum("ij,ij->i", E_an_direct, rhat)
coul = COULOMB / (eps * r ** 2)

# (ii) the head's dq at this geometry
z3 = tools.AtomicNumberTable([17, 55, 82])
a = atoms.copy(); a.info.update({"carrier_counts": np.array([0, 0, 1, 0], float), "cell_charge": 1, "config_type": "fc", "source_dir": "fc"})
b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
out = model(dict(b), compute_force=False); dq = out["dq"].detach().cpu().numpy()
dqn = dq / dq.sum(); part = 1.0 / float((dqn ** 2).sum())
loc = {f"within_{R}A": float(dq[r <= R].sum()) for R in (3, 5, 8, 12, 16)}
top = np.argsort(-np.abs(dq))[:8]
E_dq = carrier_field(torch.tensor(pos_np), cell_t, torch.tensor(dq), r_g, r_split, eps, tol=model.kernel.tol).numpy(); E_dq_r = np.einsum("ij,ij->i", E_dq, rhat)
print(f"dq: sum {dq.sum():.4f}, participation {part:.1f} atoms, " + ", ".join(f"{k} {v:+.3f}" for k, v in loc.items()) + f"; largest: " + ", ".join(f"{atoms.get_chemical_symbols()[i]}@{r[i]:.1f}A {dq[i]:+.3f}" for i in top), flush=True)
res = {"w": w, "r_g": r_g, "r_split": r_split, "eps_inf": eps, "dq_sum": float(dq.sum()), "participation": part, "dq_localisation": loc, "shells": []}
print(f"{'shell':7s} {'n':>4s} {'Coulomb':>8s} {'smeared direct':>14s} {'smeared+images':>14s} {'kernel unit q':>13s} {'kernel/analytic':>15s} {'field of dq':>11s} {'dq/unit':>8s}")
for lo, hi in [(4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 24)]:
    m = (r > lo) & (r <= hi)
    if not m.any():
        continue
    row = {"r_lo": lo, "r_hi": hi, "n": int(m.sum()), "coulomb": float(coul[m].mean()), "smeared_direct": float(E_dir_r[m].mean()),
           "smeared_images": float(E_an_r[m].mean()), "kernel_unit": float(E_unit_r[m].mean()), "field_of_dq": float(E_dq_r[m].mean())}
    row["kernel_over_analytic"] = row["kernel_unit"] / row["smeared_images"]; row["dq_over_unit"] = row["field_of_dq"] / row["kernel_unit"]
    res["shells"].append(row)
    print(f"{lo:>2d}-{hi:<4d} {row['n']:4d} {row['coulomb']:8.4f} {row['smeared_direct']:14.4f} {row['smeared_images']:14.4f} {row['kernel_unit']:13.4f} {row['kernel_over_analytic']:15.3f} {row['field_of_dq']:11.4f} {row['dq_over_unit']:8.3f}", flush=True)
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out, f"{time.time()-t0:.0f}s")
