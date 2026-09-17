"""Carrier localisation of the head's fill against cell size and fill smearing.
For each tiling (core embedded within r_core, far field ideal) and a 79-atom training frame:
sum of dq within 3/5/8/12/16 A of the vacancy, participation number, the two largest sites;
at the largest tiling also with sigma_s x 0.5 and x 0.25.
    OMP_NUM_THREADS=40 python defect-perovskite/p0_localisation.py --out ~/runs/dscc/p0/p0_localisation.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                         # noqa: E402
from mace.modules.dscc.train import vacancy_centre_full                    # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/alex/runs/dscc/p0/w6_s1_pol_r45.pt")
ap.add_argument("--core", default="/home/alex/runs/dscc/p0/ladder3_var_large/core.json")
ap.add_argument("--cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--tilings", nargs="+", default=["2,2,2", "2,2,3", "3,3,3", "3,3,4"])
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz"))
ap.add_argument("--sigma_factors", type=float, nargs="+", default=[0.5, 0.25])
ap.add_argument("--r_core", type=float, default=8.0); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
src = json.load(open(Path(args.core).expanduser()))
vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval(); model.base = model.base.double()
z3 = tools.AtomicNumberTable([17, 55, 82]); sigma0 = float(model.sigma_s)


def stats(atoms, r, label, sigma=None):
    if sigma is not None:
        model.sigma_s = sigma
    a = atoms.copy(); a.info.update({"carrier_counts": np.array([0, 0, 1, 0], float), "cell_charge": 1, "config_type": "loc", "source_dir": "loc"})
    b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
    with torch.no_grad():
        out = model(dict(b), compute_force=False)
    model.sigma_s = sigma0
    dq = out["dq"].detach().cpu().numpy(); dqn = dq / dq.sum(); part = 1.0 / float((dqn ** 2).sum())
    syms = atoms.get_chemical_symbols(); top = np.argsort(-np.abs(dq))[:4]
    row = {"label": label, "n_atoms": len(atoms), "sigma_s": float(model.sigma_s if sigma is None else sigma), "dq_sum": float(dq.sum()), "participation": part,
           "within": {str(R): float(dq[r <= R].sum()) for R in (3, 5, 8, 12, 16)}, "top": [(syms[i], float(r[i]), float(dq[i])) for i in top],
           "e_host_meV": 1e3 * float(out["diagnostics"]["e_host"][0]), "head_energy": float(out["head_energy"][0])}
    print(f"{label:22s} N={len(atoms):5d} sigma_s {row['sigma_s']:.3f}: part {part:5.1f}, within 3/5/8/12/16 A " + " ".join(f"{row['within'][k]:.3f}" for k in ("3", "5", "8", "12", "16"))
          + f", E_cP {row['e_host_meV']:+.1f} meV; top " + ", ".join(f"{s}@{d:.1f} {q:+.3f}" for s, d, q in row["top"][:2]) + f" [{time.time()-t0:.0f}s]", flush=True)
    return row


rows = []
# 79-atom charged training frame (first +1 frame at 79 atoms)
fr = next(a for a in read(args.frames, index=":") if int(a.info.get("cell_charge", 0)) == 1 and len(a) == 79)
pos = torch.tensor(fr.get_positions()); cell_t = torch.tensor(np.array(fr.get_cell()))
found = vacancy_centre_full(pos, cell_t, fr.get_atomic_numbers())
r79 = found[0].numpy() if found is not None else np.full(len(fr), np.nan)
rows.append(stats(fr, r79, "train 79 (+1)"))
for t in args.tilings:
    tt = tuple(int(x) for x in t.split(","))
    atoms, *_ = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
    vac, cell = vacancy_site(unit, tt); r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1)
    rows.append(stats(atoms, r, f"embedded {t}"))
    if t == args.tilings[-1]:
        for f in args.sigma_factors:
            rows.append(stats(atoms, r, f"embedded {t} sigma x{f}", sigma=sigma0 * f))
json.dump({"sigma_s_default": sigma0, "rows": rows}, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
