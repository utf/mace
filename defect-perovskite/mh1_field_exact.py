"""MACE-MH-1-Field, plan §2: exactness at E = 0 against MH-1 on a fixed set of 2000 structures
(1400 from the MH-1 replay pool + 600 CsPbCl3 defect frames), every head, in float32:
max |dE|/atom, max |dF|, max |dsigma|; and the field-coupling residual is exactly zero at E = 0.

  python defect-perovskite/mh1_field_exact.py --model ~/runs/mh1field/MACE-MH-1-Field-init.model --out ~/runs/mh1field/exact_before.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--mh1", default="/home/alex/.cache/mace/macemh1model")
ap.add_argument("--replay", default="/home/alex/.cache/mace/replay_mh1_omat_pbe_sel10k.xyz"); ap.add_argument("--n_replay", type=int, default=1400)
ap.add_argument("--defect", default=str(HERE / "dataset_pbe" / "train.xyz")); ap.add_argument("--n_defect", type=int, default=600)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
torch.set_default_dtype(torch.float32); dev = args.device; t0 = time.time()
new = torch.load(args.model, map_location=dev, weights_only=False).to(dev).float().eval()
ref = torch.load(args.mh1, map_location=dev, weights_only=False).to(dev).float().eval()
rng = np.random.default_rng(0)
rp = read(args.replay, ":"); rp = [rp[i] for i in sorted(rng.choice(len(rp), args.n_replay, replace=False))]
df = read(args.defect, ":"); df = [df[i] for i in sorted(rng.choice(len(df), min(args.n_defect, len(df)), replace=False))]
frames = rp + df
zt = tools.AtomicNumberTable([int(z) for z in ref.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
res = {"n_structures": len(frames), "n_replay": len(rp), "n_defect": len(df), "heads": {}}
worst = {}
for head in list(ref.heads):
    dE = dF = dS = 0.0; resid = 0.0
    for start in range(0, len(frames), args.batch):
        ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=head), z_table=zt, cutoff=float(ref.r_max), heads=list(ref.heads)) for a in frames[start:start + args.batch]]
        b = next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev).to_dict()
        o_ref = ref(dict(b), training=False, compute_force=True, compute_stress=True)
        o_new = new(dict(b), training=False, compute_force=True, compute_stress=True)
        n_at = (b["ptr"][1:] - b["ptr"][:-1]).float()
        dE = max(dE, float(((o_new["energy"] - o_ref["energy"]) / n_at).abs().max()))
        dF = max(dF, float((o_new["forces"] - o_ref["forces"]).abs().max()))
        dS = max(dS, float((o_new["stress"] - o_ref["stress"]).abs().max()))
    res["heads"][head] = {"max_dE_meV_per_atom": 1e3 * dE, "max_dF_meV_per_A": 1e3 * dF, "max_dstress_meV_per_A3": 1e3 * dS}
    print(f"{head:18s} max|dE| {1e3*dE:.3e} meV/atom  max|dF| {1e3*dF:.3e} meV/A  max|dsigma| {1e3*dS:.3e} meV/A^3  ({time.time()-t0:.0f}s)", flush=True)
# the field-coupling residual at E = 0 on one batch: field_linear(field_feats(feats, 0)) == 0 exactly
ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=ref.heads[0]), z_table=zt, cutoff=float(ref.r_max), heads=list(ref.heads)) for a in frames[:4]]
b = next(iter(tg.dataloader.DataLoader(ads, batch_size=4))).to(dev).to_dict()
feats = torch.randn(b["positions"].shape[0], new.field_feats[0].irreps_in1.dim, device=dev)
zero_field = torch.zeros(b["positions"].shape[0], 3, device=dev)
resid = float(new.field_linear[0](new.field_feats[0](feats, zero_field)).abs().max())
res["field_residual_at_zero_field"] = resid
res["pass"] = all(v["max_dE_meV_per_atom"] <= 1e-5 * 10 and v["max_dF_meV_per_A"] <= 1e-4 * 10 for v in res["heads"].values())   # float32 round-off band (10x the plan's figures is the tolerance actually applied; both reported)
res["pass_strict"] = all(v["max_dE_meV_per_atom"] <= 1e-5 and v["max_dF_meV_per_A"] <= 1e-4 for v in res["heads"].values())
json.dump(res, open(args.out, "w"), indent=1)
print(f"field residual at E = 0: {resid:.3e}; strict pass (1e-5 meV/atom, 1e-4 meV/A): {res['pass_strict']}; saved {args.out}")
