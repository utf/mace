"""Plan P: convert a Phase 1 MACE-Field multi-head fine-tune into the single-head, {Cl, Cs, Pb}
MACEField base the head takes. The density interaction block's skip_tp carries per-element
weights that `prune_elements` refuses, so the pruning goes through MACE's own element-filtered
loader (`load_foundations_elements`: skip_tp / embeddings sliced and rescaled) and is CHECKED:
E, F, P, Z*, alpha on kept-element frames must equal the unpruned Default head's.

    python defect-perovskite/p1_make_base.py --model ~/runs/dscc/p1/base_v5/base_v5.model --out ~/runs/dscc/p1/base_v5/base_v5_base.pt
"""
import argparse, os, sys
import numpy as np, torch
from ase.io import read
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace import data as mace_data, tools
from mace.modules.extensions import MACEField
from mace.tools import torch_geometric as tg
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model, remove_pt_head

SPECIES = [17, 55, 82]
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--head", default="Default")
ap.add_argument("--check", default=f"{W}/defect-perovskite/dataset_e0/valid.xyz"); ap.add_argument("--n_check", type=int, default=6)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device

m = torch.load(os.path.expanduser(args.model), map_location="cpu", weights_only=False)
if len(list(getattr(m, "heads", []))) > 1:
    m = remove_pt_head(m, args.head)
m = m.double().to(dev).eval()
cfg = extract_config_mace_model(m)
full_z = [int(z) for z in m.atomic_numbers]
idx = [full_z.index(z) for z in SPECIES]
cfg["atomic_numbers"] = SPECIES; cfg["num_elements"] = len(SPECIES)
ae = np.asarray(cfg["atomic_energies"]); cfg["atomic_energies"] = ae[..., idx] if ae.ndim > 1 else ae[idx]
new = MACEField(**cfg).double().to(dev)
load_foundations_elements(new, m, tools.AtomicNumberTable(SPECIES), load_readout=True, use_shift=True, use_scale=True,
                          max_L=cfg["hidden_irreps"].lmax)
new = new.eval()
print("pruned:", type(new).__name__, "heads", new.heads, "elements", new.atomic_numbers.tolist(), "pp_r_split", new.pp_r_split, flush=True)


def run(model, atoms_list):
    zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers]); ks = mace_data.KeySpecification.from_defaults()
    outs = []
    for a in atoms_list:
        cfg_ = mace_data.config_from_atoms(a, key_specification=ks, head_name=model.heads[-1])
        ad = mace_data.AtomicData.from_config(cfg_, z_table=zt, cutoff=float(model.r_max), heads=list(model.heads))
        b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
        o = model(b, training=True, compute_force=True, compute_polarization=True, compute_becs=True, compute_polarizability=True)
        outs.append({k: o[k].detach().cpu().numpy() for k in ("energy", "forces", "polarization", "becs", "polarizability")})
    return outs


frames = read(os.path.expanduser(args.check), f":{args.n_check}")
a_, b_ = run(m, frames), run(new, frames)
worst = {k: max(float(np.abs(x[k] - y[k]).max()) for x, y in zip(a_, b_)) for k in a_[0]}
print("max |pruned - full| per key:", {k: f"{v:.2e}" for k, v in worst.items()}, flush=True)
tol = {"energy": 1e-6, "forces": 1e-7, "polarization": 1e-8, "becs": 1e-6, "polarizability": 1e-6}
bad = {k: v for k, v in worst.items() if v > tol[k]}
if bad:
    raise SystemExit(f"pruned model does not reproduce the full model: {bad}")
torch.save(new.cpu(), os.path.expanduser(args.out)); print("saved", args.out)
