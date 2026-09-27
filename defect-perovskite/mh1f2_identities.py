"""Two-model plan §7: identities on 100 validation structures (ΣZ* = 0 per frame, α symmetric), float64.

  python defect-perovskite/mh1f2_identities.py --model RUN/MACE-MH-1-Field-A.model --head mp-dielectric --out RUN/identities.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--head", default="mp-dielectric"); ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="/home/alex/runs/mh1f2/data_mh0/dielectric_valid_seed123.xyz"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max_atoms", type=int, default=30)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
m = torch.load(args.model, map_location=dev, weights_only=False).to(dev).double().eval()
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
fr = [a for a in read(args.frames, ":") if len(a) <= args.max_atoms and set(a.numbers.tolist()) <= set(zt.zs)]
idx = np.random.default_rng(0).choice(len(fr), min(args.n, len(fr)), replace=False); fr = [fr[i] for i in sorted(idx)]
sum_z, asym = 0.0, 0.0
for a in fr:
    ad = mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=args.head), z_table=zt, cutoff=float(m.r_max), heads=list(m.heads))
    b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev)
    o = m(b.to_dict(), training=False, compute_force=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)
    Z = o["becs"].detach().view(-1, 3, 3); A = o["polarizability"].detach().view(3, 3)
    sum_z = max(sum_z, float(Z.sum(0).abs().max())); asym = max(asym, float((A - A.T).abs().max()))
res = {"model": args.model, "head": args.head, "n": len(fr), "max_abs_sum_Zstar": sum_z, "max_alpha_asymmetry": asym, "pass_1e-10": sum_z < 1e-10 and asym < 1e-10}
Path(args.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))
