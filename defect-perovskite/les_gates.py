"""L0 gates on a converted single-head MACELES base (G-L0.3 Born check; the recorded-not-gated
net latent charge, per-species means, E_LES). Protocol = the LES report's: `compute_bec=True`
through the model's OWN forward, isotropic part tr(Z*)/3 per atom, species means over the first
`--n` neutral 80-atom frames of `--frames`. The global sign of q_LES (and so of Z*, linear in q)
is free -- the energy is quadratic in q -- and is FIXED here so that Z*(Cs) > 0; the sign found
is recorded so the same flip is applied wherever q_LES is consumed.
Gate: after the flip Cl < 0, Cs > 0, Pb > 0; |Pb/Cs| in [2, 3.5]; |sum Z*| <= 0.3 e per cell.
"""
import argparse, json, os, sys
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from ase.io import read
from mace.modules.dscc import data as dd
from mace.tools import AtomicNumberTable, torch_geometric as tg

SPECIES = [17, 55, 82]; NAMES = {17: "Cl", 55: "Cs", 82: "Pb"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--frames", default=f"{W}/defect-perovskite/dataset_e0/valid.xyz")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--n_atoms", type=int, default=80)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)

model = torch.load(os.path.expanduser(args.model), map_location="cpu", weights_only=False).double().to(args.device).eval()
assert type(model).__name__ == "MACELES"
frames = [a for a in read(args.frames, ":") if int(a.info.get("cell_charge", 0)) == 0 and len(a) == args.n_atoms][:args.n]
assert len(frames) == args.n, len(frames)
zt = AtomicNumberTable(SPECIES)
rows = []
for a in frames:
    b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], zt, float(model.r_max)), batch_size=1)))
    batch = b.to(args.device).to_dict()
    out = model(batch, training=False, compute_force=True, compute_bec=True)
    Z = out["BEC"].detach().cpu()                                   # [n_atoms, 3, 3]
    iso = Z.diagonal(dim1=-2, dim2=-1).mean(-1).numpy()
    q = out["latent_charges"].detach().cpu().numpy()
    rows.append({"numbers": a.get_atomic_numbers(), "iso": iso, "q": q,
                 "e_les_meV_per_atom": 1e3 * float(out["les_energy"]) / len(a),
                 "sum_iso": float(iso.sum()), "net_q": float(q.sum())})
num = np.concatenate([r["numbers"] for r in rows]); iso = np.concatenate([r["iso"] for r in rows]); q = np.concatenate([r["q"] for r in rows])
zs = {NAMES[z]: float(iso[num == z].mean()) for z in SPECIES}
sign = 1.0 if zs["Cs"] > 0 else -1.0
zf = {k: sign * v for k, v in zs.items()}
sum_z = sign * float(np.mean([r["sum_iso"] for r in rows]))
ratio = abs(zf["Pb"] / zf["Cs"]) if zf["Cs"] != 0 else float("inf")
res = {"model": args.model, "n_frames": len(frames), "n_atoms": args.n_atoms,
       "sign_fix": sign, "Z_star_iso_raw": zs, "Z_star_iso_signfixed": zf,
       "abs_Pb_over_Cs": ratio, "sum_Z_star_per_cell_signfixed": sum_z,
       "net_latent_charge_per_cell_raw": float(np.mean([r["net_q"] for r in rows])),
       "q_species_mean_raw": {NAMES[z]: float(q[num == z].mean()) for z in SPECIES},
       "abs_q_mean": float(np.abs(q).mean()),
       "e_les_meV_per_atom": float(np.mean([r["e_les_meV_per_atom"] for r in rows]))}
gate = {"signs": zf["Cl"] < 0 and zf["Cs"] > 0 and zf["Pb"] > 0,
        "ratio_in_2_3.5": 2.0 <= ratio <= 3.5, "asr_le_0.3": abs(sum_z) <= 0.3}
res["gate"] = gate; res["G-L0.3"] = "PASS" if all(gate.values()) else "FAIL"
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(os.path.expanduser(args.out), "w"), indent=1)
