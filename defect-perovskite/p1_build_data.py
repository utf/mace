"""A2 data assembly: replay = labelled base v2 replay + labelled MACE-Field Cl/Cs/Pb structures
minus duplicates; training file = labelled neutral train + three isolated atoms at the estimated
E0s (foundation head, corrected on the training set), config_weight 1000, response weights 0.
    python defect-perovskite/p1_build_data.py --dir ~/runs/dscc/p1v2 --head mp-dielectric
"""
import argparse, json, sys
import numpy as np, torch
from ase import Atoms
from ase.io import read, write
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace import data as mace_data, tools
from mace.data.utils import estimate_e0s_from_foundation
from mace.tools.scripts_utils import remove_pt_head

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--head", default="mp-dielectric")
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
D = args.dir


def fingerprint(a):
    d = a.get_all_distances(mic=True); tri = np.sort(d[np.triu_indices(len(a), 1)])
    return (a.get_chemical_formula(), len(a), round(abs(np.linalg.det(a.cell.array)), 1), tuple(np.round(tri, 2)))


replay = read(f"{D}/replay_sel10k_mf.xyz", index=":"); combos = read(f"{D}/mf_combos_mf.xyz", index=":")
fps = {}
for a in replay:
    f = fingerprint(a)
    if f[1] <= 200:
        fps[f] = True
kept, dropped = [], 0
for a in combos:
    f = fingerprint(a)
    # duplicate: same formula/N, volume within 0.1 %, distance fingerprint within 0.01 A
    dup = any(f[0] == g[0] and f[1] == g[1] and abs(f[2] - g[2]) <= 1e-3 * max(f[2], 1) and np.allclose(f[3], g[3], atol=0.01) for g in fps if g[0] == f[0] and g[1] == f[1])
    if dup:
        dropped += 1
    else:
        kept.append(a)
for a in replay + kept:
    a.info["head"] = "pt_head"
write(f"{D}/replay_v2.xyz", replay + kept, format="extxyz")
print(f"replay: {len(replay)} base v2 structures + {len(kept)} MACE-Field Cl/Cs/Pb structures ({dropped} duplicates dropped) -> replay_v2.xyz")

# E0s: estimated from the foundation head on the neutral training set
torch.set_default_dtype(torch.float64)
m = torch.load(args.ckpt, map_location=args.device, weights_only=False)
single = remove_pt_head(m, args.head).double().to(args.device).eval()
zt = tools.AtomicNumberTable([17, 55, 82])
train = read(f"{D}/train_mf.xyz", index=":")
ks = mace_data.KeySpecification(info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"})
cfgs = [mace_data.config_from_atoms(a, key_specification=ks, head_name="Default") for a in train]
full = [int(z) for z in single.atomic_numbers]
found = {z: float(single.atomic_energies_fn.atomic_energies.reshape(-1, len(full))[0, full.index(z)]) for z in zt.zs}
est = estimate_e0s_from_foundation(single, found, cfgs, zt, device=args.device)
print("foundation E0s (head %s):" % args.head, {z: round(v, 4) for z, v in found.items()}); print("estimated E0s:", {int(z): round(float(v), 4) for z, v in est.items()})
json.dump({"head": args.head, "foundation": found, "estimated": {int(z): float(v) for z, v in est.items()}}, open(f"{D}/e0s_estimated.json", "w"), indent=1)
iso = []
for z, sym in ((17, "Cl"), (55, "Cs"), (82, "Pb")):
    a = Atoms(sym, positions=[[0, 0, 0]], cell=[20.0, 21.0, 22.0], pbc=True)
    a.info.update({"REF_energy": float(est[z]), "config_type": "IsolatedAtom", "config_weight": 1000.0,
                   "REF_stress": np.zeros((3, 3)), "REF_polarization": np.zeros(3), "REF_polarizability": np.zeros((3, 3)), "REF_electric_field": np.zeros(3),
                   "config_polarization_weight": 0.0, "config_becs_weight": 0.0, "config_polarizability_weight": 0.0, "config_stress_weight": 0.0})
    a.arrays["REF_forces"] = np.zeros((1, 3)); a.arrays["REF_becs"] = np.zeros((1, 9))
    iso.append(a)
write(f"{D}/train_v2_iso.xyz", train + iso, format="extxyz"); print(f"train_v2_iso.xyz: {len(train)} frames + 3 isolated atoms")
valid = read(f"{D}/valid_mf.xyz", index=":"); write(f"{D}/valid_v2.xyz", valid, format="extxyz"); print(f"valid_v2.xyz: {len(valid)} frames")
