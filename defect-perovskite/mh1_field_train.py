"""MACE-MH-1-Field, plan §3-§5: train the field-coupling parameters only, on the MACE-Field
dielectric + ferroelectric dataset (response labels only), validate P (folded) / Z* / alpha
against the MH-0 MACE-Field on the same validation split.

Split (recorded deviation): the MH-0 model's split indices are not published (the paper says
"splits are defined at the material level", details in its SI); the repo README recipe uses
`--valid_fraction 0.2`. Here: ferroelectric frames are split by material (polar_mpid, 20 % of
materials), dielectric frames at random (20 %), seed 123. Loss weights: the README recipe
(P 1, Z* 100, alpha 100; E/F/S 0). Adam lr 1e-3 (raised once to 3e-3 at epoch 5 if the validation
loss is still falling steadily), reduce-on-plateau on validation Z*, EMA 0.999, clip 1.0, no
weight decay; stop when Z* and alpha have not improved for 10 epochs (max 60).

  python defect-perovskite/mh1_field_train.py --init ~/runs/mh1field/MACE-MH-1-Field-init.model --run ~/runs/mh1field/run1
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
from torch_ema import ExponentialMovingAverage
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                       # noqa: E402
from mace.modules.loss import UniversalFieldLoss                 # noqa: E402
from mace.tools import torch_geometric as tg                     # noqa: E402
from mace.tools.train import evaluate                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--data", default="/home/alex/src/dielectric_ferroelectric_combined.xyz")
ap.add_argument("--head", default="mp_pbe_refit_add"); ap.add_argument("--ref", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model"); ap.add_argument("--ref_head", default="mp-dielectric")
ap.add_argument("--valid_fraction", type=float, default=0.2); ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lr_raise", type=float, default=3e-3); ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--w_p", type=float, default=1.0); ap.add_argument("--w_z", type=float, default=100.0); ap.add_argument("--w_a", type=float, default=100.0)
ap.add_argument("--atoms_per_batch", type=int, default=48); ap.add_argument("--max_frames_per_batch", type=int, default=8); ap.add_argument("--valid_batch", type=int, default=2)
ap.add_argument("--max_atoms", type=int, default=120, help="frames above this size are excluded (alpha's nested derivatives: ~1 GB per 16 atoms on the 16 GB card); recorded")
ap.add_argument("--max_epochs", type=int, default=60); ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--plateau_patience", type=int, default=5); ap.add_argument("--plateau_factor", type=float, default=0.8)
ap.add_argument("--device", default="cuda"); ap.add_argument("--max_frames", type=int, default=0, help="debug: truncate the dataset")
args = ap.parse_args()
run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(run / "train.log"), logging.StreamHandler(sys.stdout)])
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_default_dtype(torch.float32); dev = torch.device(args.device)

# ------------------------------------------------------------------ data and split
frames_all = read(args.data, ":")
if args.max_frames: frames_all = frames_all[:: max(1, len(frames_all) // args.max_frames)][: args.max_frames]   # debug: strided, both subsets
excluded = [i for i, a in enumerate(frames_all) if len(a) > args.max_atoms]
frames = [a for a in frames_all if len(a) <= args.max_atoms]
# Units (recorded): the combined dataset stores the polarisability as the Gaussian polarisability
# density chi / 4 pi (MgO 0.197 -> eps_inf 3.5, Si 0.935 -> 12.7, diamond 0.446 -> 6.6), while
# MACE-Field (fork and port alike) predicts chi = (dP/dE) / (V eps0) directly (CsPbCl3 2.55 ->
# eps_inf 3.55). Labels are converted to chi here; the MH-0 reference is read on the same converted labels.
FOUR_PI = 4.0 * np.pi
for a in frames:
    if "REF_polarizability" in a.info:
        a.info["REF_polarizability"] = np.asarray(a.info["REF_polarizability"], dtype=float) * FOUR_PI
logging.info("polarisability labels converted chi = 4 pi x REF_polarizability (dataset convention chi / 4 pi)")
logging.info("frames %d; excluded %d above %d atoms (%d with P labels, %d with Z*/alpha labels)", len(frames), len(excluded), args.max_atoms,
             sum(1 for i in excluded if "REF_polarization" in frames_all[i].info), sum(1 for i in excluded if "REF_becs" in frames_all[i].arrays))
ferro = [i for i, a in enumerate(frames) if "polar_mpid" in a.info]; diel = [i for i in range(len(frames)) if i not in set(ferro)]
rng = np.random.default_rng(args.seed)
mats = sorted(set(frames[i].info["polar_mpid"] for i in ferro)); rng.shuffle(mats)
valid_mats = set(mats[: int(round(args.valid_fraction * len(mats)))])
valid = [i for i in ferro if frames[i].info["polar_mpid"] in valid_mats]
diel_perm = list(diel); rng.shuffle(diel_perm); n_v = int(args.valid_fraction * len(diel_perm))
valid += diel_perm[:n_v]
valid_set = set(valid); train = [i for i in range(len(frames)) if i not in valid_set]
json.dump({"seed": args.seed, "valid_fraction": args.valid_fraction, "n_train": len(train), "n_valid": len(valid), "ferro_valid_materials": sorted(valid_mats),
           "train": train, "valid": valid, "rule": "ferroelectric by material (polar_mpid), dielectric random", "max_atoms": args.max_atoms, "n_excluded": len(excluded),
           "note": "indices refer to the size-filtered frame list"}, open(run / "split.json", "w"))
logging.info("split: train %d, valid %d (ferro valid materials %d of %d; dielectric valid %d of %d)", len(train), len(valid), len(valid_mats), len(mats), n_v, len(diel))

ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability", electric_field="REF_electric_field")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")


def dataset(model, head, indices):
    zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    out = []
    for i in indices:
        cfg = mace_data.config_from_atoms(frames[i], key_specification=ks, head_name=head)
        out.append(mace_data.AtomicData.from_config(cfg, z_table=zt, cutoff=float(model.r_max), heads=list(model.heads)))
    return out


class AtomBudgetBatches:
    """Size-sorted batches under an atom budget (alpha's nested derivatives); order reshuffled per epoch."""
    def __init__(self, sizes, budget, max_frames, seed):
        order = np.argsort(sizes); self.batches = []; cur = []; n = 0
        for i in order:
            if cur and (n + sizes[i] > budget or len(cur) >= max_frames):
                self.batches.append(cur); cur = []; n = 0
            cur.append(int(i)); n += sizes[i]
        if cur: self.batches.append(cur)
        self.rng = np.random.default_rng(seed)
    def __iter__(self):
        perm = self.rng.permutation(len(self.batches))
        for j in perm: yield self.batches[j]
    def __len__(self): return len(self.batches)


model = torch.load(args.init, map_location=dev, weights_only=False).to(dev).float()
params = [p for p in model.parameters() if p.requires_grad]
logging.info("trainable parameters: %d (of %d)", sum(p.numel() for p in params), sum(p.numel() for p in model.parameters()))
train_ds, valid_ds = dataset(model, args.head, train), dataset(model, args.head, valid)
sizes = [int(d.positions.shape[0]) for d in train_ds]
sampler = AtomBudgetBatches(sizes, args.atoms_per_batch, args.max_frames_per_batch, args.seed)
train_loader = tg.dataloader.DataLoader(train_ds, batch_size=1, shuffle=False, drop_last=False, batch_sampler=sampler)
valid_loader = tg.dataloader.DataLoader(valid_ds, batch_size=args.valid_batch, shuffle=False)
loss_fn = UniversalFieldLoss(energy_weight=0.0, forces_weight=0.0, stress_weight=0.0, polarization_weight=args.w_p, becs_weight=args.w_z, polarizability_weight=args.w_a).to(dev)
OUT = {"forces": False, "virials": False, "stress": False, "polarization": True, "becs": True, "polarizability": True}
KW = dict(training=True, compute_force=False, compute_stress=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)


def metrics(m, loader):
    m.eval(); _, aux = evaluate(m, loss_fn, loader, OUT, dev); m.train()
    return {k: (float(aux[k]) if aux.get(k) is not None else float("nan")) for k in ("rmse_polarization", "rmse_becs", "rmse_polarizability")}


# reference: the MH-0 MACE-Field on the same validation structures (its own table and head)
ref = torch.load(args.ref, map_location=dev, weights_only=False).to(dev).float()
ref_valid = dataset(ref, args.ref_head, valid)
ref_m = metrics(ref, tg.dataloader.DataLoader(ref_valid, batch_size=args.valid_batch, shuffle=False)); del ref, ref_valid; torch.cuda.empty_cache()
logging.info("reference MH-0 MACE-Field (%s) on this validation split: %s", args.ref_head, {k: round(v, 5) for k, v in ref_m.items()})
init_m = metrics(model, valid_loader)
logging.info("MH-1-Field at init (zero field coupling): %s", {k: round(v, 5) for k, v in init_m.items()})

opt = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
ema = ExponentialMovingAverage(params, decay=args.ema)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=args.plateau_factor, patience=args.plateau_patience)
history = [{"epoch": -1, "valid": init_m, "ref": ref_m}]
best = {"becs": float("inf"), "alpha": float("inf"), "epoch": -1}; since = 0; raised = False
t_start = time.time()
for epoch in range(args.max_epochs):
    t0 = time.time(); tot = 0.0; n_b = 0; peak = 0
    torch.cuda.reset_peak_memory_stats()
    for b in train_loader:
        b = b.to(dev); opt.zero_grad(set_to_none=True)
        out = model(b.to_dict(), **KW); loss = loss_fn(pred=out, ref=b); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); ema.update()
        tot += float(loss); n_b += 1; del out, loss
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    with ema.average_parameters():
        vm = metrics(model, valid_loader)
    key = vm["rmse_becs"] if vm["rmse_becs"] == vm["rmse_becs"] else vm["rmse_polarization"]
    sched.step(key)
    improved = vm["rmse_becs"] < best["becs"] or vm["rmse_polarizability"] < best["alpha"] or (vm["rmse_becs"] != vm["rmse_becs"] and vm["rmse_polarization"] < best.get("pol", float("inf")))
    if vm["rmse_polarization"] < best.get("pol", float("inf")): best["pol"] = vm["rmse_polarization"]
    if improved:
        best.update(becs=min(best["becs"], vm["rmse_becs"]), alpha=min(best["alpha"], vm["rmse_polarizability"]), epoch=epoch); since = 0
        with ema.average_parameters():
            model.field_config = getattr(model, "field_config", {}); model.training_record = {"split": str(run / "split.json"), "head": args.head, "weights": {"P": args.w_p, "Z": args.w_z, "alpha": args.w_a}, "epoch": epoch, "valid": vm, "ref_mh0": ref_m}
            torch.save(model, run / "MACE-MH-1-Field.model")
    else:
        since += 1
    entry = {"epoch": epoch, "train_loss": tot / max(n_b, 1), "valid": vm, "lr": opt.param_groups[0]["lr"], "s_per_epoch": time.time() - t0, "s_per_step": (time.time() - t0) / max(n_b, 1), "peak_mem_GB": peak, "best_epoch": best["epoch"]}
    history.append(entry); json.dump(history, open(run / "history.json", "w"), indent=1)
    logging.info("epoch %d loss %.4e valid P %.5f Z* %.5f alpha %.5f (ref %.5f / %.5f / %.5f) lr %.1e %.0fs (%.2f s/step, %.1f GB)",
                 epoch, entry["train_loss"], vm["rmse_polarization"], vm["rmse_becs"], vm["rmse_polarizability"], ref_m["rmse_polarization"], ref_m["rmse_becs"], ref_m["rmse_polarizability"], entry["lr"], entry["s_per_epoch"], entry["s_per_step"], peak)
    if epoch == 5 and not raised:
        h = [e["train_loss"] for e in history[1:]]
        if len(h) >= 5 and h[-1] < 0.7 * h[-4]:
            for g in opt.param_groups: g["lr"] = args.lr_raise
            raised = True; logging.info("loss still falling steadily at epoch 5: lr raised once to %.1e", args.lr_raise)
    if since >= args.patience:
        logging.info("no improvement in Z* or alpha for %d epochs: stop", args.patience); break
logging.info("done: best epoch %d, Z* %.5f, alpha %.5f (ref %.5f / %.5f); target 1.1x -> %s; %.0f s total",
             best["epoch"], best["becs"], best["alpha"], ref_m["rmse_becs"], ref_m["rmse_polarizability"],
             "PASS" if (best["becs"] <= 1.1 * ref_m["rmse_becs"] and best["alpha"] <= 1.1 * ref_m["rmse_polarizability"]) else "MISS", time.time() - t_start)
json.dump({"best": best, "ref": ref_m, "history": history}, open(run / "summary.json", "w"), indent=1)
