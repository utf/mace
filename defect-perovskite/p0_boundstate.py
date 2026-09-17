"""C5 bound-state precondition of the head at each embedded ladder geometry (and a 79-atom
training frame): the reference-fill separation eps[HOMO+1] - eps[HOMO] against Delta_c and the
Phi = 0 hole's participation N_eff against N_loc, plus the top of the H0 spectrum.
    OMP_NUM_THREADS=40 python defect-perovskite/p0_boundstate.py --out ~/runs/dscc/p0/p0_boundstate.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.precondition import PreconditionConfig, bound_state_precondition   # noqa: E402
from mace.modules.dscc.species import S_REF, neutral_count                 # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, vacancy_site                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/alex/runs/dscc/p0/w6_s1_pol_r45.pt")
ap.add_argument("--core", default="/home/alex/runs/dscc/p0/ladder3_var_large/core.json")
ap.add_argument("--cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--tilings", nargs="+", default=["2,2,2", "2,2,3", "3,3,3", "3,3,4"])
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz"))
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
z3 = tools.AtomicNumberTable([17, 55, 82]); cfg = PreconditionConfig()
print(f"precondition config: {cfg}, sigma_s {model.sigma_s}", flush=True)


def batch_of(atoms):
    a = atoms.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "bs", "source_dir": "bs"})
    return next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()


def top_levels(atoms, k=6):
    """eps around the reference HOMO from the head's H0 (eV, relative to the HOMO)."""
    data = batch_of(atoms)
    with torch.no_grad():
        out = model.base_forward(model._trunk_data(dict(data)), training=False, compute_force=False)
        scalars, vectors = model.features(out["node_feats"]); species = data["node_attrs"].argmax(dim=-1)
        positions, cell = data["positions"], data["cell"].view(-1, 3, 3)
        sender, receiver = data["edge_index"][0], data["edge_index"][1]
        shifts = torch.einsum("ei,eij->ej", data["unit_shifts"].to(positions.dtype), cell[data["batch"][sender]])
        edge_vector = positions[receiver] - positions[sender] + shifts
        H = model.h0(scalars, vectors, species, data["edge_index"], edge_vector)
        eps = torch.linalg.eigvalsh(H)
    numbers = [model.atomic_numbers[int(x)] for x in species.tolist()]
    n_up, _ = S_REF.counts(neutral_count(numbers)); homo = n_up - 1
    return (eps[homo - k:homo + k + 1] - eps[homo]).tolist(), data


rows = []
fr = next(a for a in read(args.frames, index=":") if int(a.info.get("cell_charge", 0)) == 0 and len(a) == 79)
lv, data = top_levels(fr); rec = bound_state_precondition(model, [data], cfg)["records"][0]
rows.append({"label": "train 79 (neutral vacancy)", "n": 79, **rec, "levels_rel_HOMO": lv})
print(f"train 79: separation {rec['separation']:.3f} eV, N_eff {rec['n_eff']:.1f}, pass {rec['passed']}; levels rel HOMO {np.round(lv, 3).tolist()} [{time.time()-t0:.0f}s]", flush=True)
for t in args.tilings:
    tt = tuple(int(x) for x in t.split(","))
    atoms, *_ = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
    lv, data = top_levels(atoms); rec = bound_state_precondition(model, [data], cfg)["records"][0]
    rows.append({"label": f"embedded {t}", "n": len(atoms), **rec, "levels_rel_HOMO": lv})
    print(f"embedded {t} N={len(atoms)}: separation {rec['separation']:.3f} eV, N_eff {rec['n_eff']:.1f}, pass {rec['passed']}; levels rel HOMO {np.round(lv, 3).tolist()} [{time.time()-t0:.0f}s]", flush=True)
json.dump({"config": cfg.__dict__, "sigma_s": float(model.sigma_s), "rows": rows}, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
