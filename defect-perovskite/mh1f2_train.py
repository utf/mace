"""MACE-MH-1-Field, two-model plan §4-§6: the trainer for Model A (energy-exact) and Model B
(published recipe, backbone unfrozen).

Data: the recipe's seed-123 split of the full released MP-Dielectric (head mp-dielectric: Z*, alpha)
and MP-Ferroelectric (head mp-ferroelectric: P, E, F) files; Model B adds the 9000-frame OMAT-PBE
replay pseudolabelled by MH-1 through omat_pbe (E, F, sigma) and is read on the 1000-frame DFT
hold-out. Batches: random across every pool, every batch, under an atom budget (never size-sorted).
Loss universal_field (Huber delta 0.01, P normalized_metric as the script), weights E 1 / F 100 /
sigma 1 / P 100 / Z* 100 / alpha 100 (Table I; A: E/F/sigma 0). Adam + AMSGrad, EMA 0.995, weight
decay 1e-8, gradient clipping at 10 (the script's), ReduceLROnPlateau on validation Z*, early stop
at patience 20 on validation Z*, cap 150 epochs. float64. DDP by manual gradient all-reduce
(torchrun; rank-disjoint shards; validation on rank 0, metric broadcast).

Model A: only the field maps and the two new heads' readout slices receive gradient (masks on the
shared readout tensors; weight decay applied by hand to the masked-in entries so Adam's internal
L2 cannot touch MH-1's entries); the frozen entries are asserted bit-identical at the end.

  python defect-perovskite/mh1f2_train.py --model A --init ~/runs/mh1f2/MACE-MH-1-Field-init.model --run ~/runs/mh1f2/A --lr 1e-3
  torchrun --nproc-per-node 2 defect-perovskite/mh1f2_train.py --model B --init ... --run ~/runs/mh1f2/B --lr 3e-4
"""
import argparse, json, logging, os, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
from torch_ema import ExponentialMovingAverage
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                 # noqa: E402
from mace.tools import torch_geometric as tg              # noqa: E402
from mace.modules.loss import UniversalFieldLoss, fold_polarization   # noqa: E402

D = "/home/alex/runs/mh1f2/data"
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["A", "B"], required=True); ap.add_argument("--init", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--data_dir", default=D); ap.add_argument("--replay_train", default=f"{D}/replay_train_pseudolabel_mh1.xyz"); ap.add_argument("--replay_holdout", default=f"{D}/replay_holdout_dft.xyz")
ap.add_argument("--lr", type=float, required=True); ap.add_argument("--ema", type=float, default=0.995); ap.add_argument("--weight_decay", type=float, default=1e-8); ap.add_argument("--clip_grad", type=float, default=10.0)
ap.add_argument("--w_e", type=float, default=1.0); ap.add_argument("--w_f", type=float, default=100.0); ap.add_argument("--w_s", type=float, default=1.0)
ap.add_argument("--w_p", type=float, default=100.0); ap.add_argument("--w_z", type=float, default=100.0); ap.add_argument("--w_a", type=float, default=100.0); ap.add_argument("--huber_delta", type=float, default=0.01)
ap.add_argument("--atoms_per_batch", type=int, required=True, help="atom budget of one random batch"); ap.add_argument("--max_frames_per_batch", type=int, default=64); ap.add_argument("--valid_atoms", type=int, default=64)
ap.add_argument("--response_atoms", type=int, default=96, help="atom budget of the dielectric (Z*/alpha) sub-pass; a batch is split into a dielectric sub-pass (nested derivatives) and an E/F/P sub-pass so the nested derivatives run only where a label needs them")
ap.add_argument("--max_atoms_response", type=int, default=96, help="dielectric frames above this size are excluded (alpha nested derivatives: 160 atoms OOM on 24 GB in float64); recorded")
ap.add_argument("--max_epochs", type=int, default=150); ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--plateau_patience", type=int, default=5); ap.add_argument("--plateau_factor", type=float, default=0.8)
ap.add_argument("--cueq", type=int, default=1, help="1: cuequivariance kernels (gradient check passed to 3e-8, 2.2x faster in float64); the field layers' weights are copied across MACE's converter, which re-initialises them"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--device", default="cuda"); ap.add_argument("--max_frames", type=int, default=0, help="debug: truncate every pool")
ap.add_argument("--epoch_frames", type=int, default=0, help="debug: frames per epoch (0 = all)"); ap.add_argument("--debug_max_atoms", type=int, default=0, help="debug: drop frames above this size from every pool")
ap.add_argument("--ferro_efs", type=int, default=0, help="1: train on the ferroelectric frames' E and F too; default 0 mirrors the published run (its mp-ferroelectric head saw polarisation only: log counts energy 0, forces 0)")
args = ap.parse_args()

# ------------------------------------------------------------------ DDP / setup
ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
if ddp:
    torch.distributed.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
else:
    dev = torch.device(args.device)
run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
handlers = [logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(run / "train.log")] if rank == 0 else [])
logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s %(message)s", handlers=handlers)
torch.manual_seed(args.seed + rank); np.random.seed(args.seed); torch.set_default_dtype(torch.float64)
log = logging.info

# ------------------------------------------------------------------ data
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability", electric_field="REF_electric_field")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
model = torch.load(args.init, map_location="cpu", weights_only=False).double()
zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers]); heads = list(model.heads)


def to_cueq(m):
    """MACE's e3nn -> cueq converter plus the field layers' state (the converter re-instantiates them)."""
    from mace.cli.convert_e3nn_cueq import run as _conv
    mc = _conv(m, device="cpu", return_model=True).double()
    sd_e, sd_c = m.state_dict(), mc.state_dict()
    mc.load_state_dict({k: v.reshape(sd_c[k].shape) for k, v in sd_e.items() if k.startswith("field_") and k in sd_c}, strict=False)
    for k in sd_e:
        if k.startswith("field_") and k not in sd_c and sd_e[k].numel() > 0 and not k.endswith("output_mask"):
            assert float(sd_e[k].abs().max()) == 0.0, ("non-zero field tensor dropped by the converter", k)
    for attr in ("field_config", "build_record"):
        if hasattr(m, attr): setattr(mc, attr, getattr(m, attr))
    return mc


def load(path, head):
    fr = read(path, ":")
    if head == "mp-ferroelectric" and not args.ferro_efs:
        for a in fr: a.info["config_energy_weight"] = 0.0; a.info["config_forces_weight"] = 0.0
    if args.debug_max_atoms: fr = [a for a in fr if len(a) <= args.debug_max_atoms]
    if args.max_frames: fr = fr[:: max(1, len(fr) // args.max_frames)][: args.max_frames]
    for a in fr: a.info["head"] = head
    return fr


def to_data(frames):
    return [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=ks, head_name=a.info["head"]), z_table=zt, cutoff=float(model.r_max), heads=heads) for a in frames]


pools = {"dielectric": load(f"{args.data_dir}/dielectric_train_seed123.xyz", "mp-dielectric"), "ferroelectric": load(f"{args.data_dir}/ferroelectric_train_seed123.xyz", "mp-ferroelectric")}
valids = {"dielectric": load(f"{args.data_dir}/dielectric_valid_seed123.xyz", "mp-dielectric"), "ferroelectric": load(f"{args.data_dir}/ferroelectric_valid_seed123.xyz", "mp-ferroelectric")}
excluded = {k: [a.info.get("material_id") for a in v if len(a) > args.max_atoms_response] for k, v in (("dielectric_train", pools["dielectric"]), ("dielectric_valid", valids["dielectric"]))}
pools["dielectric"] = [a for a in pools["dielectric"] if len(a) <= args.max_atoms_response]; valids["dielectric"] = [a for a in valids["dielectric"] if len(a) <= args.max_atoms_response]
log("dielectric frames above %d atoms excluded (alpha memory): %s", args.max_atoms_response, {k: len(v) for k, v in excluded.items()})
if args.model == "B":
    pools["replay"] = load(args.replay_train, "omat_pbe"); valids["replay_holdout"] = load(args.replay_holdout, "omat_pbe")
train_frames = [a for p in pools.values() for a in p]
train_ds = to_data(train_frames); sizes = np.array([len(a) for a in train_frames])
kinds = np.array([0 if "REF_becs" in a.arrays else (1 if "REF_polarization" in a.info else 2) for a in train_frames])   # 0 dielectric, 1 ferro, 2 replay
valid_ds = {k: to_data(v) for k, v in valids.items()}
log("model %s | pools: %s | validation: %s | %d train frames, %d atoms per epoch", args.model, {k: len(v) for k, v in pools.items()}, {k: len(v) for k, v in valids.items()}, len(train_ds), int(sizes.sum()))


def epoch_batches(epoch):
    """Random over every pool every batch, greedy fill under the atom budget; rank-disjoint shards of equal length."""
    rng = np.random.default_rng(args.seed * 1000 + epoch); perm = rng.permutation(len(train_ds))
    if args.epoch_frames: perm = perm[: args.epoch_frames]
    batches, cur, n, n_resp = [], [], 0, 0
    for i in perm:
        is_resp = kinds[i] == 0
        if cur and (n + sizes[i] > args.atoms_per_batch or len(cur) >= args.max_frames_per_batch or (is_resp and n_resp + sizes[i] > args.response_atoms)):
            batches.append(cur); cur, n, n_resp = [], 0, 0
        cur.append(int(i)); n += int(sizes[i]); n_resp += int(sizes[i]) if is_resp else 0
    if cur: batches.append(cur)
    per_rank = len(batches) // world
    return batches[rank * per_rank:(rank + 1) * per_rank]


# ------------------------------------------------------------------ model, masks, optimiser
n_heads = len(heads); i_new = [heads.index("mp-dielectric"), heads.index("mp-ferroelectric")]
masks = {}
if args.model == "A":
    # masks in the e3nn layout (flat (mul_in, mul_out) per instruction, head-major blocks of the non-linear readout)
    for n, p in model.named_parameters():
        if not n.startswith("readouts"): continue
        m = torch.zeros_like(p)
        if n == "readouts.0.linear.weight":
            mv = m.view(-1, n_heads); mv[:, i_new] = 1.0
        elif n == "readouts.1.linear_1.weight":
            mv = m.view(-1, n_heads, p.numel() // (n_heads * model.readouts[1].linear_1.irreps_in.dim)); mv[:, i_new, :] = 1.0
        elif n == "readouts.1.linear_2.weight":
            c = model.readouts[1].hidden_irreps.dim // n_heads; mv = m.view(n_heads, c, n_heads)
            for j in i_new: mv[j, :, j] = 1.0
        else:
            raise RuntimeError(f"unexpected readout parameter {n}")
        masks[n] = m
    if args.cueq:
        # carry the masks through the converter itself: a copy of the model whose readout weights ARE the masks
        import copy
        probe = copy.deepcopy(model)
        with torch.no_grad():
            for n, p in probe.named_parameters():
                if n in masks: p.copy_(masks[n])
        probe_c = to_cueq(probe); masks_c = {n: (p.detach() != 0).double() for n, p in probe_c.named_parameters() if n.startswith("readouts")}
        assert set(masks_c) == set(masks) and all(int(masks_c[n].sum()) == int(masks[n].sum()) for n in masks), "readout mask did not survive the cueq conversion"
        masks = masks_c
if args.cueq:
    model = to_cueq(model); log("cueq kernels: model converted (field layers' state copied across)")
model = model.to(dev).train()
if args.model == "A":
    masks = {n: m.to(dev) for n, m in masks.items()}
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("field_") or n.startswith("readouts"))
    frozen_ref = {n: (p.detach().clone(), 1.0 - masks[n]) for n, p in model.named_parameters() if n in masks}
else:
    for p in model.parameters(): p.requires_grad_(True)
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]; params = [p for _, p in named]
field_p = [p for n, p in named if n.startswith("field_")]; other_p = [p for n, p in named if not n.startswith("field_")]
groups = [{"params": field_p, "weight_decay": args.weight_decay}]
if other_p: groups.append({"params": other_p, "weight_decay": 0.0 if args.model == "A" else args.weight_decay})   # A: decay by hand on masked-in entries
opt = torch.optim.Adam(groups, lr=args.lr, amsgrad=True)
ema = ExponentialMovingAverage(params, decay=args.ema)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=args.plateau_factor, patience=args.plateau_patience)
n_train = sum(p.numel() for p in params) if args.model == "B" else sum(p.numel() for n, p in named if n.startswith("field_")) + sum(int(m.sum()) for m in masks.values())
log("trainable parameters: %d of %d (%s)", n_train, sum(p.numel() for p in model.parameters()), "field maps + new-head readout slices" if args.model == "A" else "all")
efs = args.model == "B"
loss_fn = UniversalFieldLoss(energy_weight=args.w_e if efs else 0.0, forces_weight=args.w_f if efs else 0.0, stress_weight=args.w_s if efs else 0.0,
                             polarization_weight=args.w_p, becs_weight=args.w_z, polarizability_weight=args.w_a, huber_delta=args.huber_delta).to(dev)


def passes(ids):
    """The random batch split into a dielectric sub-pass (nested derivatives) and an E/F/P sub-pass, each with only the derivatives its labels need."""
    d = [i for i in ids if kinds[i] == 0]; o = [i for i in ids if kinds[i] != 0]
    out = []
    if d: out.append((d, dict(compute_force=False, compute_stress=False, compute_polarization=True, compute_becs=True, compute_polarizability=True)))
    if o:
        k = set(int(kinds[i]) for i in o)
        out.append((o, dict(compute_force=efs and (2 in k or (1 in k and bool(args.ferro_efs))), compute_stress=efs and 2 in k, compute_polarization=1 in k, compute_becs=False, compute_polarizability=False)))
    return out


class Acc:
    """Sum of squared errors per target (all components; P folded), reported as RMSE."""
    def __init__(self): self.s = {}; self.n = {}
    def add(self, key, d):
        d = d.detach(); self.s[key] = self.s.get(key, 0.0) + float((d ** 2).sum()); self.n[key] = self.n.get(key, 0) + d.numel()
    def update(self, out, b):
        if out.get("becs") is not None and hasattr(b, "becs"):
            w = (b.becs_weight.view(-1, 9).mean(1) > 0).repeat_interleave(b.ptr[1:] - b.ptr[:-1])   # becs_weight is [n_graphs, 3, 3]
            if w.any(): self.add("becs", (out["becs"].view(-1, 3, 3) - b.becs.view(-1, 3, 3))[w])
        if out.get("polarizability") is not None and hasattr(b, "polarizability"):
            w = b.polarizability_weight.view(-1, 9).mean(1) > 0   # [n_graphs, 3, 3]
            if w.any(): self.add("polarizability", (out["polarizability"].view(-1, 3, 3) - b.polarizability.view(-1, 3, 3))[w])
        if out.get("polarization") is not None and hasattr(b, "polarization"):
            w = b.polarization_weight.view(-1, 3).mean(1) > 0
            if w.any():
                dP, _ = fold_polarization(out["polarization"].detach(), b.polarization.view(-1, 3), b.cell); self.add("polarization", dP[w])
        if out.get("energy") is not None and hasattr(b, "energy"):
            w = b.energy_weight.view(-1) > 0; n_at = (b.ptr[1:] - b.ptr[:-1]).double()
            if w.any(): self.add("e_per_atom", ((out["energy"] - b.energy) / n_at)[w])
        if out.get("forces") is not None and hasattr(b, "forces"):
            w = b.forces_weight.repeat_interleave(b.ptr[1:] - b.ptr[:-1]) > 0
            if w.any(): self.add("f", (out["forces"] - b.forces)[w])
        if out.get("stress") is not None and hasattr(b, "stress"):
            w = b.stress_weight.view(-1) > 0
            if w.any(): self.add("stress", (out["stress"] - b.stress.view(-1, 3, 3))[w])
    def rmse(self):
        return {k: float(np.sqrt(self.s[k] / self.n[k])) for k in self.s}


def evaluate(ds, kind):
    """Validation on one pool (rank 0). Returns all-component RMSEs (P folded, E per atom); replay in eV units."""
    model.eval(); acc = Acc(); k = {kind}
    order = np.argsort([int(d.positions.shape[0]) for d in ds]); cur, n = [], 0
    def flush(cur):
        b = next(iter(tg.dataloader.DataLoader([ds[i] for i in cur], batch_size=len(cur)))).to(dev)
        out = model(b.to_dict(), training=kind == 1 and bool(args.ferro_efs), compute_force=kind == 2 or (kind == 1 and bool(args.ferro_efs)), compute_stress=kind == 2, compute_polarization=kind != 2, compute_becs=kind == 0, compute_polarizability=kind == 0)
        acc.update(out, b); del out
    for i in order:
        if cur and n + int(ds[i].positions.shape[0]) > args.valid_atoms: flush(cur); cur, n = [], 0
        cur.append(int(i)); n += int(ds[i].positions.shape[0])
    if cur: flush(cur)
    model.train(); return acc.rmse()


def validate():
    with ema.average_parameters():
        vm = {"dielectric": evaluate(valid_ds["dielectric"], 0), "ferroelectric": evaluate(valid_ds["ferroelectric"], 1)}
        if "replay_holdout" in valid_ds: vm["replay_holdout"] = evaluate(valid_ds["replay_holdout"], 2)
    return vm


def fmt(vm):
    d, f = vm["dielectric"], vm["ferroelectric"]
    s = f"Z* {d.get('becs', float('nan')):.4f} alpha {d.get('polarizability', float('nan')):.4f} P {f.get('polarization', float('nan')):.5f}"
    if "e_per_atom" in f: s += f" ferro E {1e3*f['e_per_atom']:.1f} F {1e3*f['f']:.1f}"
    if "replay_holdout" in vm: r = vm["replay_holdout"]; s += f" | replay E {1e3*r['e_per_atom']:.2f} meV/atom F {1e3*r['f']:.2f} sigma {1e3*r['stress']:.3f}"
    return s


# ------------------------------------------------------------------ training
history = []; best = float("inf"); best_epoch = -1; since = 0; t_start = time.time()
if rank == 0:
    vm0 = validate(); log("epoch -1 (init, EMA=weights): %s", fmt(vm0)); history.append({"epoch": -1, "valid": vm0})
for epoch in range(args.max_epochs):
    t0 = time.time(); acc = Acc(); tot = 0.0; n_b = 0; n_at_epoch = 0
    torch.cuda.reset_peak_memory_stats(dev)
    for ids in epoch_batches(epoch):
        opt.zero_grad(set_to_none=True); n_tot = len(ids)
        for sub, kw in passes(ids):
            b = next(iter(tg.dataloader.DataLoader([train_ds[i] for i in sub], batch_size=len(sub)))).to(dev)
            out = model(b.to_dict(), training=True, **kw); loss = loss_fn(pred=out, ref=b) * (len(sub) / n_tot); loss.backward()
            acc.update(out, b); tot += float(loss); n_at_epoch += int(b.positions.shape[0]); del out, loss, b
        n_b += 1
        if ddp:
            for p in params:
                if p.grad is None: p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
        for n, p in named:
            if n in masks and p.grad is not None: p.grad.mul_(masks[n])
        if args.clip_grad: torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        if args.model == "A":
            for n, p in named:
                if n in masks and p.grad is not None: p.grad.add_(p.detach() * masks[n], alpha=args.weight_decay)
        opt.step(); ema.update()
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30; t_train = time.time() - t0
    key = torch.zeros(2, device=dev)
    if rank == 0:
        vm = validate(); key[0] = vm["dielectric"]["becs"]
        if key[0] < best: best = float(key[0]); best_epoch = epoch; since = 0
        else: since += 1
        key[1] = float(since >= args.patience)
    if ddp: torch.distributed.broadcast(key, 0)
    sched.step(float(key[0]))
    if rank == 0:
        if best_epoch == epoch:
            with ema.average_parameters():
                model.training_record = {"model": args.model, "plan": "MACE-MH-1-Field two-model plan", "args": vars(args), "epoch": epoch, "valid": vm, "split": f"{args.data_dir}/split_record.json",
                                         "replay": args.replay_train if args.model == "B" else None, "trainable_params": n_train}
                torch.save(model, run / f"MACE-MH-1-Field-{args.model}.model")
        entry = {"epoch": epoch, "train_loss": tot / max(n_b, 1), "train_rmse": acc.rmse(), "valid": vm, "lr": opt.param_groups[0]["lr"], "s_train": t_train, "s_epoch": time.time() - t0,
                 "s_per_step": t_train / max(n_b, 1), "steps": n_b, "atoms": n_at_epoch, "peak_GB": peak, "best_epoch": best_epoch, "since": since}
        history.append(entry); (run / "history.json").write_text(json.dumps(history, indent=1))
        tr = acc.rmse()
        log("epoch %d loss %.4e | train Z* %.4f alpha %.4f P %.5f%s | valid %s | lr %.2e | %.0f s (%.2f s/step, %d steps, %.1f GB) best %d since %d", epoch, entry["train_loss"],
            tr.get("becs", float("nan")), tr.get("polarizability", float("nan")), tr.get("polarization", float("nan")),
            (f" E {1e3*tr['e_per_atom']:.1f} F {1e3*tr['f']:.1f}" if "f" in tr else ""), fmt(vm), entry["lr"], entry["s_epoch"], entry["s_per_step"], n_b, peak, best_epoch, since)
    if key[1] > 0:
        log("validation Z* has not improved for %d epochs: stop (best epoch %d, Z* %.4f)", args.patience, best_epoch, best); break
if args.model == "A" and rank == 0:
    bad = [n for n, (ref, fm) in frozen_ref.items() if not torch.equal(dict(model.named_parameters())[n].detach() * fm, ref * fm)]
    log("frozen readout entries bit-identical to the init: %s", "YES" if not bad else f"NO {bad}")
if rank == 0:
    (run / "summary.json").write_text(json.dumps({"best_valid_becs": best, "best_epoch": best_epoch, "epochs": len(history) - 1, "hours": (time.time() - t_start) / 3600, "args": vars(args)}, indent=1))
    log("done: best epoch %d, validation Z* %.4f, %.1f h", best_epoch, best, (time.time() - t_start) / 3600)
if ddp: torch.distributed.destroy_process_group()
