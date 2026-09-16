"""Plan P Phase 0.3, matching check: the forces of E_PP alone (p from the MACE-Field provider,
dp/dR included) on the neutral and charged training frames at a given r_split, RMS per
component, by family. Threshold (registered): <= 3 meV/A, else raise r_split.

    python defect-perovskite/p0_matching.py --r_split 3.6 --out ~/runs/dscc/p0/p0_matching_r36.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                       # noqa: E402
from mace.modules.dscc import data as dd                     # noqa: E402
from mace.modules.dscc.field import load_field_provider      # noqa: E402
from mace.modules.dscc.polarisation import dipole_dipole_energy   # noqa: E402
from mace.tools import torch_geometric                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--head", default="mp-dielectric")
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz"))
ap.add_argument("--r_split", type=float, nargs="+", required=True)
ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device
prov = load_field_provider(args.ckpt, head=args.head, device=dev)
frames = read(args.frames, index=":")
if args.max_frames:
    rng = np.random.default_rng(0); frames = [frames[i] for i in sorted(rng.choice(len(frames), args.max_frames, replace=False))]
z3 = tools.AtomicNumberTable([17, 55, 82])


def family(a):
    q = int(a.info.get("cell_charge", 0)); n = len(a)
    return f"{'charged' if q != 0 else ('pristine' if n % 80 == 0 else 'neutral_vacancy')}_{n}"


res = {"r_split": {}, "n_frames": len(frames), "frames": args.frames, "eps_inf": args.eps_inf}
t0 = time.time()
per = {rs: {} for rs in args.r_split}
for i, a in enumerate(frames):
    a = a.copy(); a.info.setdefault("carrier_counts", np.zeros(4))
    b = next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z3, 6.0), batch_size=1))).to(dev).to_dict()
    pos = b["positions"].clone().requires_grad_(True)
    cell = b["cell"].view(3, 3)
    shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell.expand(b["edge_index"].shape[1], 3, 3))
    numbers = torch.tensor([[17, 55, 82][k] for k in b["node_attrs"].argmax(-1).tolist()], device=dev)
    p = prov.node_dipoles(pos, cell, b["batch"], b["ptr"], b["edge_index"], shifts, b["unit_shifts"], numbers)
    fam = family(a)
    for rs in args.r_split:
        e = dipole_dipole_energy(pos, cell, p, rs, args.eps_inf)
        (g,) = torch.autograd.grad(e, pos, retain_graph=True)
        d = per[rs].setdefault(fam, {"sq": 0.0, "n": 0, "e": [], "fmax": 0.0})
        d["sq"] += float((g ** 2).sum()); d["n"] += g.numel(); d["e"].append(float(e) / len(a)); d["fmax"] = max(d["fmax"], float(g.abs().max()))
    del p, pos
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(frames)} {time.time()-t0:.0f}s", flush=True)
for rs in args.r_split:
    row = {}
    for fam, d in per[rs].items():
        row[fam] = {"rms_F_PP_meV_per_A": 1e3 * float(np.sqrt(d["sq"] / d["n"])), "n_frames": len(d["e"]),
                    "E_PP_per_atom_meV_mean": 1e3 * float(np.mean(d["e"])), "E_PP_per_atom_meV_std": 1e3 * float(np.std(d["e"])),
                    "fmax_meV_per_A": 1e3 * d["fmax"]}
    tot_sq = sum(d["sq"] for d in per[rs].values()); tot_n = sum(d["n"] for d in per[rs].values())
    row["all"] = {"rms_F_PP_meV_per_A": 1e3 * float(np.sqrt(tot_sq / tot_n))}
    res["r_split"][str(rs)] = row
    print(f"r_split {rs}: " + "; ".join(f"{f}: F_PP rms {v['rms_F_PP_meV_per_A']:.2f}" + (f" (E_PP {v['E_PP_per_atom_meV_mean']:+.2f}+-{v['E_PP_per_atom_meV_std']:.2f} meV/atom)" if 'E_PP_per_atom_meV_mean' in v else "") for f, v in row.items()))
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print("saved", args.out, f"{time.time()-t0:.0f}s")
