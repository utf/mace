"""Two-model plan §4: float64 memory / time probes for the atom budget, before the trainer is
designed. Cases: (1) the largest dielectric frame alone (Z* + alpha), (2) the largest
ferroelectric frame alone (P + E + F), (3) mixed random batches at a few atom budgets, each
as Model A (field maps + new-head readout slices trainable) and Model B (all weights). Reports
peak memory and seconds per step, and projects the epoch time from the per-atom cost.

  python defect-perovskite/mh1f2_probe.py --init ~/runs/mh1f2/MACE-MH-1-Field-init.model --out ~/runs/mh1f2/probe_b3.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools          # noqa: E402
from mace.tools import torch_geometric as tg       # noqa: E402
from mace.modules.loss import UniversalFieldLoss   # noqa: E402

S = "/home/alex/src/2025-04-mace-field/scripts"
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
ap.add_argument("--budgets", default="16,32,48,64,96"); ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
model = torch.load(args.init, map_location=dev, weights_only=False).to(dev).double().train()
zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability", electric_field="REF_electric_field")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
D = "/home/alex/runs/mh1f2/data"; diel = read(f"{D}/dielectric_train_seed123.xyz", ":"); ferro = read(f"{D}/ferroelectric_train_seed123.xyz", ":")
for a in diel: a.info["_head"] = "mp-dielectric"
for a in ferro: a.info["_head"] = "mp-ferroelectric"
rng = np.random.default_rng(0)
loss_fn = UniversalFieldLoss(energy_weight=1.0, forces_weight=100.0, stress_weight=1.0, polarization_weight=100.0, becs_weight=100.0, polarizability_weight=100.0).to(dev)


def make_batch(frames):
    ads = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=a.info["_head"]), z_table=zt, cutoff=float(model.r_max), heads=list(model.heads)) for a in frames]
    return next(iter(tg.dataloader.DataLoader(ads, batch_size=len(ads)))).to(dev)


def set_mode(mode):
    for n, p in model.named_parameters():
        p.requires_grad_(mode == "B" or n.startswith("field_") or n.startswith("readouts"))


def step(b, needs):
    kw = dict(training=True, compute_force=needs["F"], compute_stress=False, compute_polarization=needs["P"], compute_becs=needs["Z"], compute_polarizability=needs["A"])
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    out = model(b.to_dict(), **kw); loss = loss_fn(pred=out, ref=b); loss.backward(); model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = time.time() - t0; del out, loss
    return dt, torch.cuda.max_memory_allocated() / 2 ** 30


def needs_of(frames):
    return {"F": any("polar_mpid" in a.info for a in frames), "P": any("polar_mpid" in a.info for a in frames), "Z": any("REF_becs" in a.arrays for a in frames), "A": any("REF_becs" in a.arrays for a in frames)}


res = {"device": torch.cuda.get_device_name(0), "dtype": "float64", "cases": []}
big_d = max(diel, key=len); big_f = max(ferro, key=len)
cases = [("largest dielectric frame", [big_d]), ("largest ferroelectric frame", [big_f])]
for budget in [int(x) for x in args.budgets.split(",")]:
    pool = diel + ferro; order = rng.permutation(len(pool)); sel, n = [], 0
    for i in order:
        if n + len(pool[i]) > budget and sel: break
        sel.append(pool[i]); n += len(pool[i])
    cases.append((f"mixed random batch, budget {budget} ({len(sel)} frames, {n} atoms)", sel))
for name, frames in cases:
    b = make_batch(frames); needs = needs_of(frames); n_at = sum(len(a) for a in frames)
    for mode in ("A", "B"):
        set_mode(mode)
        try:
            ts, ms = [], []
            for _ in range(args.repeats):
                dt, mem = step(b, needs); ts.append(dt); ms.append(mem)
            r = {"case": name, "mode": mode, "n_atoms": n_at, "n_frames": len(frames), "needs": needs, "s_per_step": float(np.median(ts)), "s_per_atom": float(np.median(ts)) / n_at, "peak_GB": float(max(ms))}
        except torch.cuda.OutOfMemoryError:
            r = {"case": name, "mode": mode, "n_atoms": n_at, "n_frames": len(frames), "needs": needs, "oom": True}
            model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); res["cases"].append(r); print(r, flush=True)
    del b
atoms_epoch = {"dielectric_train": sum(len(a) for a in diel), "ferroelectric_train": sum(len(a) for a in ferro)}
res["atoms_per_epoch"] = atoms_epoch; res["frames_per_epoch"] = {"dielectric_train": len(diel), "ferroelectric_train": len(ferro)}
Path(args.out).write_text(json.dumps(res, indent=1)); print("atoms per epoch:", atoms_epoch, "saved", args.out)
