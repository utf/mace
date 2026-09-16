"""Plan P Phase 1 gates on base v5 (registered 2026-09-16; A1 note: the relative-robustness gate
is the one that matters most).

  1. neutral validation forces <= 11.2 meV/A per component, energies <= 1.5 meV/atom
     (read from the run's best-validation record; also recomputed here on valid.xyz)
  2. pristine-cell Z* (per species, Frobenius) and polycrystalline eps_inf within 10 % of the
     checkpoint's 0.2 values on the relaxed cell (p0_locality.json)
  3. base-alone force error on the charged frames by shell not worse than base v2
     (33.2 total / 74.9 / 30.4 / 15.3 / 8.9 / 8.0 in the registered shells) by more than 3 meV/A

    python defect-perovskite/p1_gates.py --base ~/runs/dscc/p1/base_v5/base_v5_base.pt \
        --p0 ~/runs/dscc/p0/p0_locality.json --out ~/runs/dscc/p1/base_v5/P1_GATES.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.io import read
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W); sys.path.insert(0, W + "/defect-perovskite")
from mace import data as mace_data, tools                       # noqa: E402
from mace.tools import torch_geometric as tg                     # noqa: E402
from w13_eval import forces_energies, shell_masks, SPECIES        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="single-head, pruned MACEField base (w3_make_base.py output)")
ap.add_argument("--p0", default="/home/alex/runs/dscc/p0/p0_locality.json")
ap.add_argument("--cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--valid", default=f"{W}/defect-perovskite/dataset_e0/valid.xyz")
ap.add_argument("--charged", default=f"{W}/defect-perovskite/dataset_pbe/train.xyz")
ap.add_argument("--base_v2_shells", default="33.2,74.9,30.4,15.3,8.9,8.0", help="registered base v2: total, 2-4, 4-8, 8-10, 10-12, 12+")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
model = torch.load(os.path.expanduser(args.base), map_location=dev, weights_only=False).double().to(dev).eval()
zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
res = {"base": args.base, "pp_r_split": float(getattr(model, "pp_r_split", 0.0))}

# ---- gate 1: neutral validation
val = read(args.valid, index=":")
for a in val:
    a.info.setdefault("carrier_counts", np.zeros(4))
F, E = forces_energies(model, val, zt, float(model.r_max), dev, 4)
dF = np.concatenate([f - a.arrays["REF_forces"] for f, a in zip(F, val)])
dE = np.array([(e - a.info["REF_energy"]) / len(a) for e, a in zip(E, val)])
res["gate1"] = {"rmse_F_meV_A": 1e3 * float(np.sqrt((dF ** 2).mean())), "rmse_E_meV_atom": 1e3 * float(np.sqrt((dE ** 2).mean())),
                "mean_E_offset_meV_atom": 1e3 * float(dE.mean()), "n": len(val)}
res["gate1"]["pass"] = res["gate1"]["rmse_F_meV_A"] <= 11.2 and res["gate1"]["rmse_E_meV_atom"] <= 1.5
print("gate1", res["gate1"], flush=True)

# ---- gate 2: response on the relaxed pristine cell
doc = json.load(open(os.path.expanduser(args.cell)))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
cfg = mace_data.config_from_atoms(unit, key_specification=mace_data.KeySpecification.from_defaults(), head_name=list(model.heads)[-1])
ad = mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(model.r_max), heads=list(model.heads))
b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
out = model(b, training=True, compute_force=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
n = len(unit); Z = out["becs"].detach().cpu().numpy().reshape(n, 3, 3); al = out["polarizability"].detach().cpu().numpy()[0]
syms = unit.get_chemical_symbols()
ref = json.load(open(os.path.expanduser(args.p0)))["response_0_2"]
g2 = {"eps_inf_poly": float(1 + np.trace(al) / 3), "eps_inf_ref": ref["eps_inf_poly"], "species": {}}
g2["eps_inf_rel"] = abs(g2["eps_inf_poly"] - ref["eps_inf_poly"]) / ref["eps_inf_poly"]
ok = g2["eps_inf_rel"] <= 0.10
for s in sorted(set(syms)):
    idx = [k for k, x in enumerate(syms) if x == s]
    fro = float(np.mean([np.linalg.norm(Z[k]) for k in idx])); r = ref["species"][s]["frobenius_mean"]
    g2["species"][s] = {"frobenius": fro, "ref": r, "rel": abs(fro - r) / r, "trace3": float(np.mean([np.trace(Z[k]) / 3 for k in idx]))}
    ok = ok and abs(fro - r) / r <= 0.10
g2["pass"] = ok; res["gate2"] = g2
print("gate2", g2, flush=True)

# ---- gate 3: charged frames by shell
ch = [a for a in read(args.charged, index=":") if int(a.info.get("cell_charge", 0)) != 0]
for a in ch:
    a.info.setdefault("carrier_counts", np.zeros(4))
F, E = forces_energies(model, ch, zt, float(model.r_max), dev, 4)
shell_names = ["2-4", "4-8", "8-10", "10-12", "12+"]; edges = [(2, 4), (4, 8), (8, 10), (10, 12), (12, 1e9)]
sq = {k: [0.0, 0] for k in ["total"] + shell_names}
for f, a in zip(F, ch):
    d = f - a.arrays["REF_forces"]
    sq["total"][0] += float((d ** 2).sum()); sq["total"][1] += d.size
    rad, _cats, _d = shell_masks(a)
    if rad is None:
        continue
    r = np.asarray(rad)
    for name, (lo, hi) in zip(shell_names, edges):
        m = (r > lo) & (r <= hi)
        if m.any():
            sq[name][0] += float((d[m] ** 2).sum()); sq[name][1] += int(m.sum()) * 3
v2 = [float(x) for x in args.base_v2_shells.split(",")]
g3 = {"n_frames": len(ch), "shells": {}}
ok = True
for (k, (s, c)), ref_v in zip(sq.items(), v2):
    val_ = 1e3 * float(np.sqrt(s / c)) if c else None
    g3["shells"][k] = {"rmse_meV_A": val_, "base_v2": ref_v, "excess": (val_ - ref_v) if val_ is not None else None}
    ok = ok and (val_ is not None and val_ - ref_v <= 3.0)
g3["pass"] = ok; res["gate3"] = g3
print("gate3", g3, flush=True)
res["all_pass"] = bool(res["gate1"]["pass"] and res["gate2"]["pass"] and res["gate3"]["pass"])
json.dump(res, open(os.path.expanduser(args.out), "w"), indent=1)
print("ALL PASS" if res["all_pass"] else "GATES FAILED", "saved", args.out)
