"""Phase 1 data assembly for a new provider checkpoint (base v8 = A2B provider), mirroring
p1_build_data.py: training file = labelled neutral train + three isolated atoms at E0s estimated
from the provider head on the neutral training set (config_weight 1000, response weights 0);
validation file copied; replay = the base v2 replay frames relabelled by the provider head
(E/F/S only, head `pt_head`).

  python defect-perovskite/p1v8_build.py --dir ~/runs/dscc/p1v8 --ckpt ~/runs/mh1f2/deliver/MACE-MH-1-Field-A2B.model --head mp-dielectric
"""
import argparse, json, sys
import numpy as np, torch
from ase import Atoms
from ase.io import read, write
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace import data as mace_data, tools                     # noqa: E402
from mace.data.utils import estimate_e0s_from_foundation      # noqa: E402
from mace.tools.scripts_utils import remove_pt_head           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--head", required=True)
ap.add_argument("--device", default="cuda")
args = ap.parse_args(); D = args.dir
torch.set_default_dtype(torch.float64)
m = torch.load(args.ckpt, map_location=args.device, weights_only=False)
single = remove_pt_head(m, args.head).double().to(args.device).eval()
zt = tools.AtomicNumberTable([17, 55, 82])
train = read(f"{D}/train_v8.xyz", index=":")
ks = mace_data.KeySpecification(info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"})
cfgs = [mace_data.config_from_atoms(a, key_specification=ks, head_name="Default") for a in train]
full = [int(z) for z in single.atomic_numbers]
found = {z: float(single.atomic_energies_fn.atomic_energies.reshape(-1, len(full))[0, full.index(z)]) for z in zt.zs}
est = estimate_e0s_from_foundation(single, found, cfgs, zt, device=args.device)
print("foundation E0s (head %s):" % args.head, {z: round(v, 4) for z, v in found.items()}); print("estimated E0s:", {int(z): round(float(v), 4) for z, v in est.items()})
json.dump({"ckpt": args.ckpt, "head": args.head, "foundation": found, "estimated": {int(z): float(v) for z, v in est.items()}}, open(f"{D}/e0s_estimated.json", "w"), indent=1)
iso = []
for z, sym in ((17, "Cl"), (55, "Cs"), (82, "Pb")):
    a = Atoms(sym, positions=[[0, 0, 0]], cell=[20.0, 21.0, 22.0], pbc=True)
    a.info.update({"REF_energy": float(est[z]), "config_type": "IsolatedAtom", "config_weight": 1000.0,
                   "REF_stress": np.zeros((3, 3)), "REF_polarization": np.zeros(3), "REF_polarizability": np.zeros((3, 3)), "REF_electric_field": np.zeros(3),
                   "config_polarization_weight": 0.0, "config_becs_weight": 0.0, "config_polarizability_weight": 0.0, "config_stress_weight": 0.0})
    a.arrays["REF_forces"] = np.zeros((1, 3)); a.arrays["REF_becs"] = np.zeros((1, 9))
    iso.append(a)
write(f"{D}/train_v8_iso.xyz", train + iso, format="extxyz"); print(f"train_v8_iso.xyz: {len(train)} frames + 3 isolated atoms")
rep = read(f"{D}/replay_v8_raw.xyz", index=":")
for a in rep: a.info["head"] = "pt_head"
write(f"{D}/replay_v8.xyz", rep, format="extxyz"); print(f"replay_v8.xyz: {len(rep)} frames (E/F/S from {args.head})")
