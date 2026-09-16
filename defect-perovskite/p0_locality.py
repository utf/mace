"""Plan P Phase 0.1 -- locality of the field response (decides r_split), and 0.2 -- the pristine
response (Z*, alpha, eps_inf) of the MACE-Field checkpoint. No training.

    python defect-perovskite/p0_locality.py --cell ~/runs/dscc/p0/pristine_relaxed_v2.json \
        --out ~/runs/dscc/p0/p0_locality.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.data import chemical_symbols
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                       # noqa: E402
from mace.tools import torch_geometric                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--head", default="mp-dielectric")
ap.add_argument("--cell", required=True, help="relaxed pristine cell json (numbers, positions, cell)")
ap.add_argument("--tiling", default="2,2,2")
ap.add_argument("--field", type=float, default=0.01, help="node field, V/A")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device

doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
model = torch.load(args.ckpt, map_location=dev, weights_only=False).double().to(dev).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
keyspec = mace_data.KeySpecification.from_defaults()


def batch_of(atoms):
    cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=args.head)
    ad = mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max), heads=list(model.heads))
    return next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()


def min_image_dist(atoms, i):
    d = atoms.get_distances(i, np.arange(len(atoms)), mic=True)
    return d


def r90(dist, weight):
    order = np.argsort(dist)
    c = np.cumsum(weight[order]); c = c / c[-1]
    return float(dist[order][np.searchsorted(c, 0.9)])


result = {"cell": str(args.cell), "tiling": args.tiling, "field_V_per_A": args.field, "head": args.head}
# ---------------- 0.2 on the relaxed pristine cell
t = time.time()
b = batch_of(unit)
out = model(b, training=True, compute_force=True, compute_polarization=True, compute_becs=True, compute_polarizability=True)
n = len(unit); Z = out["becs"].detach().cpu().numpy().reshape(n, 3, 3); al = out["polarizability"].detach().cpu().numpy()[0]
F = out["forces"].detach().cpu().numpy()
syms = unit.get_chemical_symbols()
resp = {"n_atoms": n, "fmax_eV_per_A": float(np.abs(F).max()), "species": {}}
for s in sorted(set(syms)):
    idx = [k for k, x in enumerate(syms) if x == s]
    Zs = Z[idx]
    resp["species"][s] = {"count": len(idx), "Z_mean": Zs.mean(0).tolist(), "trace_over_3_mean": float(np.mean([np.trace(z) / 3 for z in Zs])),
                          "frobenius_mean": float(np.mean([np.linalg.norm(z) for z in Zs])),
                          "frobenius_std": float(np.std([np.linalg.norm(z) for z in Zs])),
                          "diag_range": [float(np.min(Zs.diagonal(axis1=1, axis2=2))), float(np.max(Zs.diagonal(axis1=1, axis2=2)))]}
eps = np.eye(3) + al
ev = np.linalg.eigvalsh(0.5 * (eps + eps.T))
resp["sum_Z_diag"] = Z.sum(0).diagonal().tolist()
resp["alpha"] = al.tolist(); resp["eps_inf_tensor"] = eps.tolist(); resp["eps_inf_poly"] = float(np.trace(eps) / 3)
resp["eps_inf_eigen"] = ev.tolist(); resp["eps_inf_anisotropy"] = float((ev.max() - ev.min()) / np.mean(ev))
result["response_0_2"] = resp
print(f"0.2: eps_inf {eps.diagonal()} poly {np.trace(eps)/3:.3f} aniso {(ev.max()-ev.min())/np.mean(ev):.3%}; "
      + "; ".join(f"{s}: tr/3 {v['trace_over_3_mean']:+.3f} |Z| {v['frobenius_mean']:.3f}" for s, v in resp["species"].items())
      + f"; sum Z diag {np.round(Z.sum(0).diagonal(), 3)}; {time.time()-t:.1f}s")

# ---------------- 0.1 locality on the tiling
rep = tuple(int(x) for x in args.tiling.split(","))
big = unit.repeat(rep)
N = len(big); syms_b = big.get_chemical_symbols()
bb = batch_of(big)
t = time.time()


def run_node(e_node, force=True):
    """p (no graph kept) and, separately, the forces at the node field: two light passes
    instead of one double-backward pass (the 640-atom cell OOMs the 16 GB card otherwise)."""
    e = torch.as_tensor(e_node, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    o = model(bb, training=False, compute_force=False, electric_field_node=e)
    (g,) = torch.autograd.grad(o["interaction_energy"].sum(), e)
    p = (-g).detach().cpu().numpy(); ne = o["node_energy"].detach().cpu().numpy(); en = o["energy"].item()
    del o, g
    f = None
    if force:
        o = model(bb, training=False, compute_force=True, electric_field_node=e.detach())
        f = o["forces"].detach().cpu().numpy(); del o
    torch.cuda.empty_cache()
    return p, f, ne, en


p0, f0, ne0, e0 = run_node(np.zeros((N, 3)))
print(f"0.1: tiling {rep} N={N}; baseline |p| rms {np.sqrt((p0**2).sum(1).mean()):.4f} eA, fmax {np.abs(f0).max():.4f}; {time.time()-t:.1f}s")
loc = {"N": N, "species": {}}
for s in sorted(set(syms_b)):
    i = syms_b.index(s)                             # one atom of each species
    dist = min_image_dist(big, i)
    per_dir = []
    for c in range(3):
        e = np.zeros((N, 3)); e[i, c] = args.field
        p1, f1, _, e1 = run_node(e)
        dp = p1 - p0; df = f1 - f0
        wdp = np.linalg.norm(dp, axis=1); wdf = np.linalg.norm(df, axis=1)
        per_dir.append({"dir": "xyz"[c], "r90_dp": r90(dist, wdp), "r90_dF": r90(dist, wdf),
                        "dp_self": float(wdp[i]), "dp_total": float(wdp.sum()), "dF_total": float(wdf.sum()),
                        "dp_shell": {f"{r:.0f}": float(wdp[dist <= r].sum() / wdp.sum()) for r in (3, 4, 5, 6, 8, 10, 12)},
                        "dF_shell": {f"{r:.0f}": float(wdf[dist <= r].sum() / wdf.sum()) for r in (3, 4, 5, 6, 8, 10, 12)}})
    loc["species"][s] = {"atom": i, "per_direction": per_dir,
                         "r90_dp_mean": float(np.mean([d["r90_dp"] for d in per_dir])),
                         "r90_dp_max": float(np.max([d["r90_dp"] for d in per_dir])),
                         "r90_dF_mean": float(np.mean([d["r90_dF"] for d in per_dir]))}
    print(f"  {s}: r90(dp) per dir {[round(d['r90_dp'],2) for d in per_dir]} A, r90(dF) {[round(d['r90_dF'],2) for d in per_dir]} A, "
          f"self share {[round(d['dp_self']/d['dp_total'],3) for d in per_dir]}")
r90_all = max(v["r90_dp_max"] for v in loc["species"].values())
loc["r90_max_over_species"] = r90_all
loc["r_split_rule"] = float(max(3.0, 1.2 * r90_all))
result["locality_0_1"] = loc
print(f"  r90 max over species {r90_all:.2f} A -> r_split = max(3.0, 1.2 x r90) = {max(3.0, 1.2*r90_all):.2f} A")

# ---------------- partition ambiguity: p'[a] = -d(node_energy_a)/dE_global on the same cell
t = time.time()
E = torch.zeros(1, 3, dtype=torch.float64, device=dev, requires_grad=True)
o = model(bb, training=True, compute_force=False, electric_field=E)
ne = o["node_energy"]                                  # [N], attached
pprime = np.zeros((N, 3))
chunk = 8
for s0 in range(0, N, chunk):
    rows = torch.zeros(min(chunk, N - s0), N, dtype=torch.float64, device=dev)
    for r_, a in enumerate(range(s0, min(s0 + chunk, N))):
        rows[r_, a] = 1.0
    g = torch.autograd.grad(ne, E, grad_outputs=rows, retain_graph=True, is_grads_batched=True)[0]   # [chunk, 1, 3]
    pprime[s0:s0 + rows.shape[0]] = -g.reshape(-1, 3).cpu().numpy()
diff = p0 - pprime
amb = {"rms_p": float(np.sqrt((p0**2).sum(1).mean())), "rms_pprime": float(np.sqrt((pprime**2).sum(1).mean())),
       "rms_diff": float(np.sqrt((diff**2).sum(1).mean())), "sum_p": p0.sum(0).tolist(), "sum_pprime": pprime.sum(0).tolist()}
amb["ratio"] = amb["rms_diff"] / amb["rms_p"] if amb["rms_p"] > 0 else None
result["partition_ambiguity"] = amb
print(f"  partition ambiguity: rms p {amb['rms_p']:.4f}, rms p' {amb['rms_pprime']:.4f}, rms(p-p')/rms p = {amb['ratio']:.3f}; "
      f"sum p {np.round(p0.sum(0),4)} sum p' {np.round(pprime.sum(0),4)}; {time.time()-t:.1f}s")
json.dump(result, open(Path(args.out).expanduser(), "w"), indent=1)
print("saved", args.out)
