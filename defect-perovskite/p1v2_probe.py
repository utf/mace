"""Plan P A3 probe: for a multi-head Phase 1 model (or the raw foundation checkpoint with the
estimated E0 shift), evaluate the Default head on the neutral validation set and a replay subset
with pt_head, and report (i) per-atom energy error split into mean offset and scatter, by frame
size, (ii) force RMSE, (iii) the universal_field loss per term with the run weights, so the
gradient budget between target and replay is visible. Usage: p1v2_probe.py MODEL [--raw]"""
import argparse, json, sys, numpy as np, torch
from ase.io import read
from mace import data as mace_data, tools
from mace.modules.loss import UniversalFieldLoss
from mace.tools import torch_geometric as tg

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--raw", action="store_true")
ap.add_argument("--valid", default="/home/alex/runs/dscc/p1v2/valid_v2.xyz")
ap.add_argument("--replay", default="/home/alex/runs/dscc/p1v2/replay_v2.xyz"); ap.add_argument("--n_replay", type=int, default=160)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_default_dtype(torch.float32); dev = torch.device(args.device)
m = torch.load(args.model, map_location="cpu", weights_only=False).to(dev).float().eval()
heads = list(m.heads); print("heads", heads, flush=True)
e0 = json.load(open("/home/alex/runs/dscc/p1v2/e0s_estimated.json"))
shift = {int(z): e0["estimated"][z] - e0["foundation"][z] for z in e0["estimated"]} if args.raw else {}
zt = tools.AtomicNumberTable([int(z) for z in m.atomic_numbers])
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
W = dict(energy_weight=10, forces_weight=10, stress_weight=1, polarization_weight=1000, becs_weight=100, polarizability_weight=100)
# one loss object per term (only that weight non-zero) -> weighted per-term contributions
term_losses = {k: UniversalFieldLoss(**{kk: (v if kk == k else 0.0) for kk, v in W.items()}).to(dev) for k in W}

def evaluate(frames, head):
    rows = []; terms = {}
    for a in frames:
        cfg = mace_data.config_from_atoms(a, key_specification=ks, head_name=head)
        ad = mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(m.r_max), heads=heads)
        b = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev)
        o = m(b.to_dict(), training=True, compute_force=True, compute_stress=True, compute_polarization=True, compute_becs=True, compute_polarizability=True)
        n = len(a); e_pred = float(o["energy"]) + sum(shift.get(int(z), 0.0) for z in a.numbers)
        de = (e_pred - a.info["REF_energy"]) / n
        df = o["forces"].detach().cpu().numpy() - a.arrays["REF_forces"]
        rows.append((n, de, float(np.sqrt((df**2).mean())), a.info.get("config_type", "")))
        if not args.raw:
            with torch.no_grad():
                for k, lf in term_losses.items():
                    terms[k] = terms.get(k, 0.0) + float(lf(b, o)) / len(frames)
    return rows, terms

def summarise(name, rows):
    n = np.array([r[0] for r in rows]); de = np.array([r[1] for r in rows]) * 1e3; fr = np.array([r[2] for r in rows]) * 1e3
    print(f"{name}: N={len(rows)} E/atom rmse {np.sqrt((de**2).mean()):.2f} meV = offset {de.mean():+.2f} + scatter {de.std():.2f}; F rmse {np.sqrt((fr**2).mean()):.1f} meV/A")
    for s in sorted(set(n)):
        k = n == s; print(f"   size {s:5d}: n={k.sum():4d} offset {de[k].mean():+.2f} scatter {de[k].std():.2f} F {np.sqrt((fr[k]**2).mean()):.1f}")
    return {"rmse": float(np.sqrt((de**2).mean())), "offset": float(de.mean()), "scatter": float(de.std()), "F": float(np.sqrt((fr**2).mean()))}

res = {}
head_d = "Default" if "Default" in heads else "mp-dielectric"
va = read(args.valid, ":"); r, t = evaluate(va, head_d); res["valid"] = summarise("valid/Default", r); res["valid_terms"] = t; print("  loss terms:", {k: f"{v:.2e}" for k, v in t.items()}, "total", f"{sum(t.values()):.2e}")
iso = [a for a in read("/home/alex/runs/dscc/p1v2/train_v2_iso.xyz", ":") if len(a) == 1]
r, _ = evaluate(iso, head_d); print("isolated atoms E error (meV):", [f"{x[3] or a.get_chemical_formula()} {x[1]*1e3:+.1f}" for x, a in zip(r, iso)]); res["iso"] = [x[1] * 1e3 for x in r]
if not args.raw:
    rp = read(args.replay, f"::{max(1, 9376 // args.n_replay)}")[: args.n_replay]
    r, t = evaluate(rp, "pt_head"); res["replay"] = summarise("replay/pt_head", r); res["replay_terms"] = t; print("  loss terms:", {k: f"{v:.2e}" for k, v in t.items()}, "total", f"{sum(t.values()):.2e}")
if args.out: json.dump(res, open(args.out, "w"), indent=1)
