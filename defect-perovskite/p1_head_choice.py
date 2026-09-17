"""A2: which MACE-Field head to fine-tune from -- initial neutral-validation E/F per head, and
each head's response on the relaxed pristine cell (the gate-2 reference for that head)."""
import json, sys
import numpy as np, torch
from ase import Atoms
from ase.io import read
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace import data as mace_data, tools
from mace.tools import torch_geometric as tg
torch.set_default_dtype(torch.float64); dev = "cuda"
m = torch.load("/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model", map_location=dev, weights_only=False).double().to(dev).eval()
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
val = read(f"{W}/defect-perovskite/dataset_e0/valid.xyz", index=":")
doc = json.load(open("/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")); unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
res = {}
for head in ("pt_head", "mp-dielectric"):
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
    res[head] = {"valid_rmse_F_meV_A": 1e3 * float(np.sqrt((dF ** 2).mean())), "valid_rmse_E_meV_atom": 1e3 * float(np.sqrt((dE ** 2).mean())), "valid_mean_E_offset": 1e3 * float(dE.mean()),
                 "eps_inf_poly": float(1 + np.trace(al) / 3),
                 "Z_frobenius": {s: float(np.mean([np.linalg.norm(Z[k]) for k in range(len(unit)) if syms[k] == s])) for s in ("Cl", "Cs", "Pb")},
                 "Z_trace3": {s: float(np.mean([np.trace(Z[k]) / 3 for k in range(len(unit)) if syms[k] == s])) for s in ("Cl", "Cs", "Pb")}}
    print(head, {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in res[head].items()}, flush=True)
json.dump(res, open("/home/alex/runs/dscc/p1/head_choice.json", "w"), indent=1)
