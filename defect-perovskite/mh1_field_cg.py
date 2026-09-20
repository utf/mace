"""MACE-MH-1-Field, option 3: solve the frozen-backbone response fit as the linear least-squares
problem it is. With the first field map fixed (random init) and everything else frozen, Z* and P
are LINEAR in the second map W (field_linear): J v = Z*(W = v) exactly. Conjugate gradient on the
normal equations (J^T J + lambda I) w = J^T y over the training split's Z* and folded-P labels
(per-component weights w_Z, w_P), with the Krylov iteration count as the regulariser: the
validation Z* / P / alpha are read every `--eval_every` iterations and the best iterate is kept.
Each CG iteration costs one pass over the training set (forward + backward per batch).

  python defect-perovskite/mh1_field_cg.py --init ~/runs/mh1field/MACE-MH-1-Field-init.model --run ~/runs/mh1field/run_cg
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                       # noqa: E402
from mace.modules.loss import UniversalFieldLoss                 # noqa: E402
from mace.modules.utils import fold_polarization                 # noqa: E402
from mace.tools import torch_geometric as tg                     # noqa: E402
from mace.tools.train import evaluate                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--data", default="/home/alex/src/dielectric_ferroelectric_combined.xyz"); ap.add_argument("--head", default="mp_pbe_refit_add")
ap.add_argument("--split", default="/home/alex/runs/mh1field/run1_p1_lr1e-3/split.json", help="the registered split (same frame filtering)")
ap.add_argument("--w_z", type=float, default=1.0); ap.add_argument("--w_p", type=float, default=10.0); ap.add_argument("--lam", type=float, default=0.0)
ap.add_argument("--iters", type=int, default=24); ap.add_argument("--eval_every", type=int, default=3)
ap.add_argument("--atoms_per_batch", type=int, default=64); ap.add_argument("--valid_batch", type=int, default=2); ap.add_argument("--max_atoms", type=int, default=120)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(run / "cg.log"), logging.StreamHandler(sys.stdout)])
torch.set_default_dtype(torch.float32); dev = torch.device(args.device)
frames_all = read(args.data, ":"); frames = [a for a in frames_all if len(a) <= args.max_atoms]
for a in frames:
    if "REF_polarizability" in a.info: a.info["REF_polarizability"] = np.asarray(a.info["REF_polarizability"], dtype=float) * 4 * np.pi
split = json.load(open(args.split)); train, valid = split["train"], split["valid"]
ks = mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability", electric_field="REF_electric_field")
ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
model = torch.load(args.init, map_location=dev, weights_only=False).to(dev).float().eval()
for p in model.parameters(): p.requires_grad_(False)
W = model.field_linear[0].weight; W.requires_grad_(True)
zt = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
def ds(idx):
    return [mace_data.AtomicData.from_config(mace_data.config_from_atoms(frames[i], key_specification=ks, head_name=args.head), z_table=zt, cutoff=float(model.r_max), heads=list(model.heads)) for i in idx]
train_ds, valid_ds = ds(train), ds(valid)
sizes = [int(d.positions.shape[0]) for d in train_ds]; order = np.argsort(sizes); batches = []; cur = []; n = 0
for i in order:
    if cur and n + sizes[i] > args.atoms_per_batch: batches.append(cur); cur = []; n = 0
    cur.append(int(i)); n += sizes[i]
if cur: batches.append(cur)
loader = tg.dataloader.DataLoader(train_ds, batch_size=1, shuffle=False, batch_sampler=batches)
valid_loader = tg.dataloader.DataLoader(valid_ds, batch_size=args.valid_batch, shuffle=False)
KW = dict(training=True, compute_force=False, compute_stress=False, compute_polarization=True, compute_becs=True, compute_polarizability=False)
loss_fn = UniversalFieldLoss(0, 0, 0, 1000, 100, 100).to(dev)
OUT = {"forces": False, "virials": False, "stress": False, "polarization": True, "becs": True, "polarizability": True}
logging.info("CG over W = field_linear.0.weight (%d unknowns); train %d frames in %d batches; valid %d; w_Z %.1f w_P %.1f lambda %.2e", W.numel(), len(train_ds), len(batches), len(valid_ds), args.w_z, args.w_p, args.lam)


def set_w(v):
    with torch.no_grad(): W.copy_(v)


def predict(b):
    o = model(b.to_dict(), **KW)
    return o["becs"].reshape(-1), o["polarization"].reshape(-1, 3)


def targets(b, p_pred):
    """Z* labels (masked by the per-config becs weight) and the folded-P residual target."""
    zw = b.becs_weight.reshape(b.num_graphs, -1)[:, 0]; mask_z = torch.repeat_interleave(zw > 0, (b.ptr[1:] - b.ptr[:-1]) * 9)
    pw = b.polarization_weight.reshape(b.num_graphs, -1)[:, 0] > 0
    yz = b.becs.reshape(-1); yp = b.polarization.reshape(-1, 3)
    dP, _ = fold_polarization(p_pred.detach(), yp, b.cell)          # folded difference pred - label
    yp_fold = p_pred.detach() - dP                                    # the label on the branch nearest the prediction
    return yz, mask_z.float(), yp_fold, pw.float()


def apply_A(v, yz_pass=False):
    """A v = w_Z J_Z^T J_Z v + w_P J_P^T J_P v + lambda v (and, with yz_pass, b = J^T y instead)."""
    set_w(v if not yz_pass else torch.zeros_like(v))
    acc = torch.zeros_like(W)
    for b in loader:
        b = b.to(dev)
        z, p = predict(b)                                            # linear in W: J v when W = v
        yz, mz, yp, mp = targets(b, p)
        if yz_pass:
            tz, tp = yz * mz, yp * mp.unsqueeze(-1)
        else:
            tz, tp = z.detach() * mz, p.detach() * mp.unsqueeze(-1)
        s_ = args.w_z * (z * tz).sum() + args.w_p * (p * tp).sum()
        (g,) = torch.autograd.grad(s_, W)
        acc += g
    return acc + (0.0 if yz_pass else args.lam * v)


def validate(v, tag):
    set_w(v); model.eval()
    with torch.enable_grad():
        _, aux = evaluate(model, loss_fn, valid_loader, OUT, dev)
    m = {k: float(aux[k]) for k in ("rmse_polarization", "rmse_becs", "rmse_polarizability")}
    logging.info("%s: valid P %.5f Z* %.5f alpha %.5f (ref 0.00177 / 0.21009 / 1.10125)", tag, m["rmse_polarization"], m["rmse_becs"], m["rmse_polarizability"])
    return m


t0 = time.time()
zero = torch.zeros_like(W)
bvec = apply_A(zero, yz_pass=True); logging.info("b = J^T y formed (%.0f s), |b| %.3e", time.time() - t0, float(bvec.norm()))
x = torch.zeros_like(W); r = bvec.clone(); pdir = r.clone(); rs = float((r * r).sum())
best = {"becs": float("inf"), "iter": -1}; hist = [{"iter": 0, "valid": validate(x, "iter 0")}]
for k in range(1, args.iters + 1):
    t1 = time.time(); Ap = apply_A(pdir); alpha = rs / float((pdir * Ap).sum())
    x = x + alpha * pdir; r = r - alpha * Ap; rs_new = float((r * r).sum())
    logging.info("CG iter %d: |r| %.3e alpha %.3e |x| %.3e (%.0f s)", k, rs_new ** 0.5, alpha, float(x.norm()), time.time() - t1)
    if k % args.eval_every == 0 or k == args.iters:
        m = validate(x, f"iter {k}"); hist.append({"iter": k, "valid": m, "resid": rs_new ** 0.5})
        if m["rmse_becs"] < best["becs"]:
            best = {"becs": m["rmse_becs"], "iter": k, "valid": m}; set_w(x); torch.save(model, run / "MACE-MH-1-Field-cg.model")
        json.dump({"best": best, "history": hist}, open(run / "summary.json", "w"), indent=1)
    pdir = r + (rs_new / rs) * pdir; rs = rs_new
set_w(x); logging.info("done: best iter %d Z* %.5f (target 0.231); %.0f s", best["iter"], best["becs"], time.time() - t0)
