"""A2: label structures with the MACE-Field checkpoint (one head): REF_energy, REF_forces,
REF_stress (eV/A^3, 3x3), REF_polarization (e/A^2), REF_becs (N,3,3), REF_polarizability
(3x3, per unit eps0), REF_electric_field = 0. Existing REF_* keys are overwritten unless
--keep_ef is given (then E/F/stress are kept and only the response is added).
    python defect-perovskite/p1_label.py --head pt_head --in replay.xyz --out replay_labelled.xyz
"""
import argparse, sys, time
import numpy as np, torch
from ase.io import read, write
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace import data as mace_data, tools
from mace.tools import torch_geometric as tg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--head", required=True); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--keep_ef", action="store_true", help="keep the file's REF_energy/forces/stress; add the response only")
ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="float64")
args = ap.parse_args()
torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32); dev = args.device
m = torch.load(args.ckpt, map_location=dev, weights_only=False); m = (m.double() if args.dtype == "float64" else m.float()).to(dev).eval()
for p in m.parameters():
    p.requires_grad_(False)
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); ks = mace_data.KeySpecification(info_keys={}, arrays_keys={})
frames = read(args.inp, index=":"); t0 = time.time(); out_frames = []
allowed = set(int(z) for z in m.atomic_numbers)
skipped = sum(1 for a in frames if not set(int(z) for z in a.get_atomic_numbers()) <= allowed)
frames = [a for a in frames if set(int(z) for z in a.get_atomic_numbers()) <= allowed]
print(f"{skipped} structures with elements outside the checkpoint's table skipped; {len(frames)} to label", flush=True)
for i, a in enumerate(frames):
    cfg = mace_data.config_from_atoms(a, key_specification=ks, head_name=args.head)
    ad = mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(m.r_max), heads=list(m.heads))
    b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
    try:
        o = m(b, training=True, compute_force=True, compute_stress=True, compute_polarization=True, compute_becs=True, compute_polarizability=True)
    except torch.OutOfMemoryError:
        # a large structure's second derivatives do not fit the card: do this one on the CPU
        torch.cuda.empty_cache()
        if not hasattr(m, "_cpu_twin"):
            import copy
            m._cpu_twin = copy.deepcopy(m).cpu()
        bc = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b.items()}
        o = m._cpu_twin(bc, training=True, compute_force=True, compute_stress=True, compute_polarization=True, compute_becs=True, compute_polarizability=True)
        print(f"  structure {i} ({len(a)} atoms) labelled on the CPU", flush=True)
    o = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in o.items()}
    torch.cuda.empty_cache()
    a = a.copy()
    if not args.keep_ef:
        a.info["REF_energy"] = float(o["energy"][0]); a.arrays["REF_forces"] = o["forces"].numpy().astype(np.float64)
        a.info["REF_stress"] = o["stress"].numpy().reshape(3, 3).astype(np.float64)
    a.info["REF_polarization"] = o["polarization"].numpy().reshape(3).astype(np.float64)
    a.arrays["REF_becs"] = o["becs"].numpy().reshape(len(a), 9).astype(np.float64)
    a.info["REF_polarizability"] = o["polarizability"].numpy().reshape(3, 3).astype(np.float64)
    a.info["REF_electric_field"] = np.zeros(3)
    for k in list(a.info):
        if k.startswith("config_") and k.endswith("_weight") and k not in ("config_weight",):
            pass
    out_frames.append(a)
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(frames)} {time.time()-t0:.0f}s", flush=True); write(args.out, out_frames, format="extxyz")
write(args.out, out_frames, format="extxyz"); print(f"labelled {len(out_frames)} frames with head {args.head} -> {args.out} ({time.time()-t0:.0f}s)")
