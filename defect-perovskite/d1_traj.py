"""D1 trajectory reader: |q|, E_LES, species means and LES readout norms from a run's checkpoint
(raw weights, cueq layout squeezed) or a saved .model (EMA), on 8 neutral 80-atom D0 valid frames.
    python d1_traj.py --model ~/runs/d1_light/checkpoints/d1_light_run-1_epoch-7.pt --template ~/runs/d1_light/d1_light.model
(--template: a saved model of the same architecture to load the checkpoint's state dict into; omit for a .model)"""
import argparse, copy, json, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from ase.io import read
from mace import data as mace_data
from mace.tools import AtomicNumberTable, torch_geometric
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--template", default="")
ap.add_argument("--valid", default=str(HERE / "dataset_d0/valid.xyz")); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=8)
args = ap.parse_args(); from d1_common import load_checkpoint
m, kind = load_checkpoint(args.model, args.template, args.device)
torch.set_default_dtype(next(m.parameters()).dtype)
m = m.eval()
DT = next(m.parameters()).dtype
def fnorm(readouts):
    out = []
    for r in readouts:
        final = getattr(r, "linear_2", None) or getattr(r, "linear", None); out.append(sum(float((p ** 2).sum()) for p in final.parameters()) ** 0.5)
    return out
zt = AtomicNumberTable([int(z) for z in m.atomic_numbers]); heads = list(getattr(m, "heads", ["Default"]))
frames = [a for a in read(args.valid, ":") if a.info.get("d0_family") == "neutral" and len(a) == 80][: args.n]
qs, el = [], []
for a in frames:
    ds = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a), z_table=zt, cutoff=float(m.r_max), heads=heads)]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(args.device).to_dict()
    o = m(b, training=False, compute_force=False); qs.append(o["latent_charges"].detach().cpu().numpy()); el.append(1e3 * float(o["les_energy"]) / 80)
q = np.concatenate(qs); num = np.concatenate([a.get_atomic_numbers() for a in frames])
res = {"model": args.model, "kind": kind, "les_readout_norms": fnorm(m.les_readouts), "energy_readout_norms": fnorm(m.readouts),
       "abs_q_mean": float(np.abs(q).mean()), "q_species": {s: float(q[num == z].mean()) for s, z in (("Cl", 17), ("Cs", 55), ("Pb", 82))},
       "net_q_per_cell": float(q.sum() / len(frames)), "E_LES_meV_per_atom": float(np.mean(el))}
print(json.dumps(res))
