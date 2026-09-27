"""Phase 1 head choice for a new provider (the A2 rule): for each candidate head of the checkpoint,
the initial neutral-validation E / F and the head's response on the relaxed pristine cell (the
gate-2 reference for that head). Rule: the head with the lower initial force error, unless the
other's response on the relaxed cell is closer to the Phase 0 reference (mp-dielectric of 1.0.2:
eps_inf 3.547; Z* Frobenius Cl 3.520 / Cs 2.235 / Pb 6.209) by more than the force difference warrants.

  python defect-perovskite/p1v8_head_choice.py --ckpt ~/runs/mh1f2/deliver/MACE-MH-1-Field-A2B.model --heads omat_pbe,mp-dielectric --out ~/runs/dscc/p1v8/head_choice.json
"""
import argparse, json, sys
import numpy as np, torch
from ase import Atoms
from ase.io import read
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--heads", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--valid", default=f"{W}/defect-perovskite/dataset_e0/valid.xyz"); ap.add_argument("--cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
m = torch.load(args.ckpt, map_location=dev, weights_only=False).double().to(dev).eval()
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
val = read(args.valid, index=":")
doc = json.load(open(args.cell)); unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
REF = {"eps_inf": 3.5472, "Z": {"Cl": 3.5201, "Cs": 2.2354, "Pb": 6.2088}}
res = {"ckpt": args.ckpt, "reference": REF, "heads": {}}
for head in args.heads.split(","):
    dE, dF = [], []
    for a in val:
        cfg = mace_data.config_from_atoms(a, key_specification=ks, head_name=head)
        ad = mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(m.r_max), heads=list(m.heads))
        b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
        o = m(b, training=False, compute_force=True)
        dE.append((o["energy"].item() - a.info["REF_energy"]) / len(a)); dF.append(o["forces"].detach().cpu().numpy() - a.arrays["REF_forces"])
    dF = np.concatenate(dF); dE = np.array(dE)
    cfg = mace_data.config_from_atoms(unit, key_specification=ks, head_name=head)
    ad = mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(m.r_max), heads=list(m.heads))
    b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
    o = m(b, training=True, compute_force=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
    Z = o["becs"].detach().cpu().numpy().reshape(len(unit), 3, 3); al = o["polarizability"].detach().cpu().numpy()[0]; syms = unit.get_chemical_symbols()
    eps = al + np.eye(3); w = np.linalg.eigvalsh(eps)
    frob = {s: float(np.mean([np.linalg.norm(Z[i]) for i in range(len(unit)) if syms[i] == s])) for s in ("Cl", "Cs", "Pb")}
    rel = {s: abs(frob[s] - REF["Z"][s]) / REF["Z"][s] for s in frob}
    res["heads"][head] = {"valid_rmse_F_meV_A": 1e3 * float(np.sqrt((dF ** 2).mean())), "valid_rmse_E_meV_atom": 1e3 * float(np.sqrt((dE ** 2).mean())), "valid_mean_E_offset": 1e3 * float(dE.mean()),
                         "eps_inf_poly": float(np.mean(w)), "eps_inf_rel": abs(float(np.mean(w)) - REF["eps_inf"]) / REF["eps_inf"], "Z_frobenius": frob, "Z_rel_to_ref": rel, "response_within_10pct": bool(max(rel.values()) <= 0.10 and abs(float(np.mean(w)) - REF["eps_inf"]) / REF["eps_inf"] <= 0.10)}
    print(head, json.dumps(res["heads"][head]), flush=True)
json.dump(res, open(args.out, "w"), indent=1); print("saved", args.out)
