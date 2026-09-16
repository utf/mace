"""G-D1.1: student vs teacher on the held 10 % of D0, per family (energy meV/atom, forces
per-component meV/A, stress meV/A^3), through the student's OWN forward (the Ewald term in).
    python d1_read.py --model ~/runs/d1_light/d1_light.model --out d1_light_read.json [--ablate]
--ablate zeroes the LES readouts first (G-D1.3: the change on tilings / polar is the reading)."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from ase.io import read
from mace import data as mace_data
from mace.tools import AtomicNumberTable, torch_geometric
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--valid", default=str(HERE / "dataset_d0/valid.xyz"))
ap.add_argument("--device", default="cuda"); ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--ablate", action="store_true"); ap.add_argument("--template", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", required=True)
args = ap.parse_args()
from d1_common import load_checkpoint
model, kind = load_checkpoint(args.model, args.template, args.device)
torch.set_default_dtype(next(model.parameters()).dtype)
if args.ablate:
    with torch.no_grad():
        for r in model.les_readouts:
            final = getattr(r, "linear_2", None) or getattr(r, "linear", None)
            for p in final.parameters():
                p.zero_()
zt = AtomicNumberTable([int(z) for z in model.atomic_numbers]); heads = list(getattr(model, "heads", ["Default"]))
frames = read(args.valid, ":");  frames = frames[: args.limit] if args.limit else frames
by_size = {}
for i, a in enumerate(frames):
    by_size.setdefault(len(a), []).append(i)
res = {}; t0 = time.time(); n_done = 0
for n, idx in by_size.items():
    for s in range(0, len(idx), args.batch_size):
        chunk = idx[s:s + args.batch_size]
        cfgs = [mace_data.config_from_atoms(frames[i]) for i in chunk]
        ds = [mace_data.AtomicData.from_config(c, z_table=zt, cutoff=float(model.r_max), heads=heads) for c in cfgs]
        batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds)))).to(args.device).to_dict()
        with torch.no_grad():
            pass
        out = model(batch, training=False, compute_force=True, compute_stress=True)
        ptr = batch["ptr"].cpu().numpy()
        for g, i in enumerate(chunk):
            a = frames[i]; fam = a.info["d0_family"]; r = res.setdefault(fam, {"n": 0, "e2": 0.0, "f2": 0.0, "nf": 0, "s2": 0.0, "eles": []})
            r["n"] += 1
            r["e2"] += ((float(out["energy"][g]) - a.info["REF_energy"]) / len(a)) ** 2
            df = out["forces"][ptr[g]:ptr[g + 1]].detach().cpu().numpy() - a.arrays["REF_forces"]
            r["f2"] += float((df ** 2).sum()); r["nf"] += df.size
            ds_ = out["stress"][g].detach().cpu().numpy().reshape(-1) - np.asarray(a.info["REF_stress"]).reshape(-1)
            r["s2"] += float((ds_ ** 2).mean())
            r["eles"].append(1e3 * float(out["les_energy"][g]) / len(a))
        n_done += len(chunk)
table = {fam: {"n": r["n"], "energy_meV_per_atom": 1e3 * math.sqrt(r["e2"] / r["n"]), "forces_meV_per_A": 1e3 * math.sqrt(r["f2"] / r["nf"]),
               "stress_meV_per_A3": 1e3 * math.sqrt(r["s2"] / r["n"]), "E_LES_meV_per_atom_mean": float(np.mean(r["eles"]))}
         for fam, r in res.items()}
allf = math.sqrt(sum(r["f2"] for r in res.values()) / sum(r["nf"] for r in res.values())) * 1e3
alle = math.sqrt(sum(r["e2"] for r in res.values()) / sum(r["n"] for r in res.values())) * 1e3
out = {"model": args.model, "ablated": args.ablate, "n": n_done, "seconds": time.time() - t0, "all": {"energy": alle, "forces": allf},
       "families": table, "G-D1.1": {fam: (v["energy_meV_per_atom"] <= 1.0 and v["forces_meV_per_A"] <= 5.0) for fam, v in table.items()}}
json.dump(out, open(os.path.expanduser(args.out), "w"), indent=1)
print(f"{'family':20s} {'n':>5s} {'E meV/at':>9s} {'F meV/A':>8s} {'S meV/A3':>9s} {'E_LES':>7s}")
for fam, v in table.items():
    print(f"{fam:20s} {v['n']:5d} {v['energy_meV_per_atom']:9.3f} {v['forces_meV_per_A']:8.2f} {v['stress_meV_per_A3']:9.3f} {v['E_LES_meV_per_atom_mean']:7.2f}")
print(f"all: {alle:.3f} meV/atom, {allf:.2f} meV/A; ablated={args.ablate}; saved {args.out}")
