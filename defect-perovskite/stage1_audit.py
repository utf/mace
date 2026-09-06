"""v8.1 addendum, section 8: the loss-path audit of a completed Stage-1 run (items 1-4).

Runs INSIDE the trainer: `train_defect_model.sh` is invoked with the run's own recipe and
`MACE_TRAIN_MODULE=stage1_audit`, so `mace.cli.run_train` builds the data, weights, null
gate, size upweights, loss, optimiser, parameter groups, clipping and hooks exactly as the
run did; this module replaces the training loop with the audit and exits.

    AUDIT_CHECKPOINT   the run's pre-final checkpoint (model + optimizer + scheduler state)
    AUDIT_OUT          output JSON path
    AUDIT_EPOCH        the epoch the checkpoint ends (its replay is epoch AUDIT_EPOCH + 1)
    AUDIT_REPLAY_STEPS how many optimiser steps of that epoch to replay (default: all)
    AUDIT_DRAWS_BEFORE_EPOCH0  randperm draws the ORIGINAL run consumed from the loader's
                       generator before epoch 0 (default 4: profile probe, base-cache build,
                       protocol init batch, c-table calibration); the audit's own startup
                       consumed one fewer when the base cache loaded from disk.

Item 1: with the head at the checkpoint, perturb the 159-atom constant c[0,1] by +-1 eV and
+-0.1 eV on every batch that carries a 159-atom charged frame, compare the scalar-loss change
with the analytic form of the implemented per-atom-squared totals term, and compare autodiff
dL/dc with that analytic derivative; report the split of dL/dc into the totals-energy term,
the force terms and the rest, for c[0,1] and c[0,0].
Item 2: trace every charged 159-atom frame's weights (weight, energy_weight,
base_energy_weight, delta_energy_weight, forces_weight), its mask value, its atom count,
dtype, its parameter group and the checkpoint restore of the constant.
Item 3: replay optimiser steps of epoch AUDIT_EPOCH + 1 from the checkpoint's model and
optimizer state with the seeded batch order reconstructed; per step record the pre-clip
gradient on c (split), the clip coefficient, Adam's moments, lr, weight decay and the update;
compare the epoch's net change of c with the run's logged trajectory.
Item 4: the analytic within-stratum profiler recovers an injected constant offset.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch


def _c_param(model):
    head = model.spectral
    for name, p in model.named_parameters():
        if p is head.c_shift_table:
            return name, p
    raise RuntimeError("no c_shift_table parameter")


def _group_of(optimizer, param):
    for gi, g in enumerate(optimizer.param_groups):
        for p in g["params"]:
            if p is param:
                return gi, {k: v for k, v in g.items() if k != "params"}
    return None, None


def _charged_mask(batch):
    return batch["carrier_counts"].reshape(int(batch.num_graphs), -1).sum(dim=-1) > 0


def _sizes(batch):
    return (batch.ptr[1:] - batch.ptr[:-1]).to(torch.float64)


def _loss_pieces(loss_fn, pred, batch):
    """The implemented totals-energy term, the force terms and the remainder, recomputed
    from `pred`/`batch` with the loss's own weights (mirrors DefectLoss.forward)."""
    from mace.modules.loss import mean_squared_error_forces_field, reduce_loss, \
        _spread_to_atoms
    n = _sizes(batch)
    mask = _charged_mask(batch).to(batch.weight.dtype)
    total = pred["energy"]
    if getattr(loss_fn, "detach_base_in_totals", False):
        total = pred["base_energy"].detach() + pred["correction_energy"]
    per_frame = batch.weight * batch.energy_weight * mask * torch.square(
        (batch["energy"] - total) / n)
    e_tot = loss_fn.total_energy_weight * reduce_loss(per_frame, False)
    f_base = loss_fn.forces_weight * mean_squared_error_forces_field(
        batch, pred, "base_forces", "base_forces_weight", ddp=False)
    f_delta = loss_fn.delta_forces_weight * mean_squared_error_forces_field(
        batch, pred, "delta_forces", "delta_forces_weight", ddp=False)
    raw_tot_f = (_spread_to_atoms(batch.weight, batch)
                 * _spread_to_atoms(batch.forces_weight * mask, batch)
                 * torch.square(batch["forces"] - pred["forces"]))
    f_tot = loss_fn.forces_weight * reduce_loss(raw_tot_f, False)
    return dict(e_tot=e_tot, forces=f_base + f_delta + f_tot, per_frame=per_frame, n=n,
                mask=mask, total=total)


def _grad_wrt(scalar, param, retain=True):
    g = torch.autograd.grad(scalar, [param], retain_graph=retain, allow_unused=True)[0]
    return torch.zeros_like(param) if g is None else g.detach().clone()


def audit_train(model, loss_fn, train_loader, valid_loaders, optimizer, lr_scheduler,
                start_epoch, max_num_epochs, patience, checkpoint_handler, logger,
                eval_interval, output_args, device, log_errors, swa=None, ema=None,
                max_grad_norm=10.0, epoch_hook=None, post_step_hook=None, **_):
    t_start = time.time()
    out: Dict[str, Any] = {"items": {}}
    ck_path = os.environ["AUDIT_CHECKPOINT"]
    out_path = os.environ["AUDIT_OUT"]
    ck_epoch = int(os.environ.get("AUDIT_EPOCH", "20"))
    replay_steps = int(os.environ.get("AUDIT_REPLAY_STEPS", "0") or 0)
    draws_hist = int(os.environ.get("AUDIT_DRAWS_BEFORE_EPOCH0", "4"))
    dev = torch.device(device)

    # ------------------------------------------------------------ environment / item 2 (a)
    c_name, c_param = _c_param(model)
    gi, group = _group_of(optimizer, c_param)
    out["environment"] = dict(
        loss=repr(loss_fn), total_energy_weight=float(loss_fn.total_energy_weight),
        forces_weight=float(loss_fn.forces_weight), energy_weight=float(loss_fn.energy_weight),
        detach_base_in_totals=bool(getattr(loss_fn, "detach_base_in_totals", False)),
        c_param=c_name, c_requires_grad=bool(c_param.requires_grad), c_param_group=gi,
        c_group_settings={k: (float(v) if isinstance(v, (int, float)) else str(v))
                          for k, v in (group or {}).items()},
        optimizer=type(optimizer).__name__, max_grad_norm=max_grad_norm,
        dtype=str(c_param.dtype), device=str(dev), batch_size=int(train_loader.batch_size),
        drop_last=bool(train_loader.drop_last), n_train=len(train_loader.dataset),
        n_batches=len(train_loader), post_step_hook=str(post_step_hook),
        pristine_atoms=int(getattr(model.spectral, "pristine_atoms", 0) or 0),
        c_at_startup=c_param.detach().cpu().tolist())

    # ------------------------------------------------------------ checkpoint restore
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    optimizer.load_state_dict(ck["optimizer"])
    lr_scheduler.load_state_dict(ck["lr_scheduler"])
    out["checkpoint"] = dict(path=ck_path, epoch=ck_epoch, missing=list(missing),
                             unexpected=list(unexpected),
                             c_restored=c_param.detach().cpu().tolist(),
                             c_in_state_dict=ck["model"][c_name].tolist(),
                             adam_state_keys=sorted(str(k) for k in
                                                    optimizer.state[c_param].keys()),
                             lr_after_restore=[float(g["lr"]) for g in optimizer.param_groups])
    st = optimizer.state[c_param]
    out["checkpoint"]["adam_step"] = float(st["step"]) if "step" in st else None
    out["checkpoint"]["exp_avg"] = st["exp_avg"].cpu().tolist() if "exp_avg" in st else None
    out["checkpoint"]["exp_avg_sq"] = st["exp_avg_sq"].cpu().tolist() if "exp_avg_sq" in st else None
    # the epoch hook at the replayed epoch: warmup overlay (a no-op past the warmup) and the
    # class-table refresh on the restored head, as the run did at the start of that epoch
    if epoch_hook is not None:
        epoch_hook(ck_epoch + 1, model)
    model.train()

    # ------------------------------------------------------------ item 2: the frames
    ds = train_loader.dataset
    rows = []
    for i, d in enumerate(ds):
        n = int(d.positions.shape[0])
        charged = int(d.carrier_counts.reshape(-1).sum()) != 0
        if not charged:
            continue
        rows.append(dict(index=i, natoms=n, weight=float(d.weight),
                         energy_weight=float(d.energy_weight),
                         base_energy_weight=float(getattr(d, "base_energy_weight", float("nan"))),
                         delta_energy_weight=float(getattr(d, "delta_energy_weight", float("nan"))),
                         forces_weight=float(d.forces_weight),
                         base_forces_weight=float(getattr(d, "base_forces_weight", float("nan"))),
                         delta_forces_weight=float(getattr(d, "delta_forces_weight", float("nan"))),
                         energy=float(d.energy), frame_key=int(getattr(d, "frame_key", -1)),
                         dtype=str(d.energy.dtype)))
    big = [r for r in rows if r["natoms"] >= 120]
    small = [r for r in rows if r["natoms"] < 120]
    out["items"]["2_frames"] = dict(
        n_charged=len(rows), n_charged_large=len(big), n_charged_small=len(small),
        large=big,
        small_energy_weight_nonzero=sum(1 for r in small if r["energy_weight"] != 0.0),
        small_energy_weight_zero=sum(1 for r in small if r["energy_weight"] == 0.0),
        small_delta_energy_weight_nonzero=sum(1 for r in small
                                              if r["delta_energy_weight"] not in (0.0,)
                                              and not math.isnan(r["delta_energy_weight"])),
        large_indices=[r["index"] for r in big])
    big_idx = set(r["index"] for r in big)

    # ------------------------------------------------------------ item 1: dL/dc
    # Every batch containing a 159-atom charged frame, in the REPLAYED order of the
    # checkpoint's next epoch (so items 1 and 3 read the same batches).
    gen = train_loader.generator
    n_ds = len(ds)
    # the audit's own startup: probe batch, protocol init batch, c calibration (3), plus the
    # base-cache build when the cache did not load from disk (the launcher counts it)
    draws_done = int(os.environ.get("AUDIT_DRAWS_DONE", "3"))
    for _ in range(draws_hist - draws_done):
        torch.randperm(n_ds, generator=gen)
    for _ in range(ck_epoch + 1):            # epochs 0 .. ck_epoch
        torch.randperm(n_ds, generator=gen)
    order = torch.randperm(n_ds, generator=gen).tolist()
    bs = int(train_loader.batch_size)
    n_full = n_ds // bs if train_loader.drop_last else (n_ds + bs - 1) // bs
    batches_idx = [order[k * bs:(k + 1) * bs] for k in range(n_full)]
    out["replay"] = dict(epoch=ck_epoch + 1, draws_hist=draws_hist, draws_done_here=draws_done,
                         n_batches=len(batches_idx),
                         batches_with_large_charged=[k for k, b in enumerate(batches_idx)
                                                     if any(i in big_idx for i in b)])
    from mace.tools import torch_geometric
    collate = torch_geometric.dataloader.Collater([], [])

    def make_batch(idx):
        return collate([ds[i] for i in idx]).to(dev)

    def forward_loss(batch, c_override=None):
        with torch.no_grad():
            saved = c_param.detach().clone()
            if c_override is not None:
                c_param.copy_(c_override)
        d = batch.to_dict()
        pred = model(d, training=True, compute_force=True,
                     compute_virials=bool(output_args.get("virials", False)),
                     compute_stress=bool(output_args.get("stress", False)))
        loss = loss_fn(pred=pred, ref=batch)
        if c_override is not None:
            with torch.no_grad():
                c_param.copy_(saved)
        return pred, loss

    item1 = []
    for k in out["replay"]["batches_with_large_charged"][:6]:
        batch = make_batch(batches_idx[k])
        pred, loss = forward_loss(batch)
        pieces = _loss_pieces(loss_fn, pred, batch)
        g_full = _grad_wrt(loss, c_param)
        g_etot = _grad_wrt(pieces["e_tot"], c_param)
        g_forces = _grad_wrt(pieces["forces"], c_param, retain=False)
        # dE_pred/dc per graph by a central difference in c[0,1] (linear in c)
        c0 = c_param.detach().clone()
        h = 1.0e-3
        with torch.no_grad():
            cp, cm = c0.clone(), c0.clone()
            cp[0, 1] += h
            cm[0, 1] -= h
        pred_p, loss_p = forward_loss(batch, cp)
        pred_m, loss_m = forward_loss(batch, cm)
        dE_dc = ((pred_p["energy"] - pred_m["energy"]) / (2 * h)).detach()
        n = pieces["n"]
        mask = pieces["mask"]
        resid = (batch["energy"] - pieces["total"]).detach()
        analytic = (loss_fn.total_energy_weight / float(batch.num_graphs)
                    * (batch.weight * batch.energy_weight * mask * 2.0 * resid / n
                       * (-dE_dc) / n)).sum()
        row = dict(batch=k, n_graphs=int(batch.num_graphs),
                   sizes=n.tolist(), charged=mask.tolist(),
                   energy_weight=batch.energy_weight.tolist(), weight=batch.weight.tolist(),
                   resid_eV=resid.tolist(), dE_dc01=dE_dc.tolist(),
                   loss=float(loss), e_tot_term=float(pieces["e_tot"]),
                   dL_dc_autograd=g_full.tolist(), dLetot_dc_autograd=g_etot.tolist(),
                   dLforces_dc_autograd=g_forces.tolist(),
                   dLrest_dc_autograd=(g_full - g_etot - g_forces).tolist(),
                   dLetot_dc01_analytic=float(analytic),
                   fd=[])
        for delta in (1.0, -1.0, 0.1, -0.1):
            with torch.no_grad():
                cd = c0.clone()
                cd[0, 1] += delta
            _, loss_d = forward_loss(batch, cd)
            # quadratic prediction of the totals term alone under the linear E(c)
            with torch.no_grad():
                tot_d = pieces["total"] + dE_dc * delta
                per_d = (batch.weight * batch.energy_weight * mask
                         * torch.square((batch["energy"] - tot_d) / n))
                e_tot_d = loss_fn.total_energy_weight * per_d.mean()
            row["fd"].append(dict(delta=delta, loss=float(loss_d),
                                  dloss=float(loss_d - loss),
                                  dloss_totals_predicted=float(e_tot_d - pieces["e_tot"])))
        item1.append(row)
        logging.info("audit item 1, batch %d: dL/dc01 autograd %.3e (totals %.3e, forces "
                     "%.3e, rest %.3e); analytic totals %.3e; dE/dc01 on the 159 frames %s",
                     k, g_full[0, 1], g_etot[0, 1], g_forces[0, 1],
                     (g_full - g_etot - g_forces)[0, 1], float(analytic),
                     [round(x, 4) for x, m in zip(dE_dc.tolist(), mask.tolist()) if m > 0])
    out["items"]["1_gradient"] = item1

    # ------------------------------------------------------------ item 3: replay
    steps = replay_steps if replay_steps > 0 else len(batches_idx)
    beta1, beta2 = optimizer.param_groups[gi]["betas"]
    eps = float(optimizer.param_groups[gi]["eps"])
    wd = float(optimizer.param_groups[gi].get("weight_decay", 0.0))
    lr = float(optimizer.param_groups[gi]["lr"])
    trace = []
    c_before_epoch = c_param.detach().cpu().clone()
    resid_rows: List[Dict[str, Any]] = []
    for s in range(steps):
        batch = make_batch(batches_idx[s])
        optimizer.zero_grad(set_to_none=True)
        pred, loss = forward_loss(batch)
        pieces = _loss_pieces(loss_fn, pred, batch)
        g_etot = _grad_wrt(pieces["e_tot"], c_param)
        g_forces = _grad_wrt(pieces["forces"], c_param)
        loss.backward()
        g_pre = c_param.grad.detach().clone()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        clip_coef = min(1.0, float(max_grad_norm) / (float(total_norm) + 1e-6))
        g_post = c_param.grad.detach().clone()
        c_prev = c_param.detach().clone()
        stp = optimizer.state[c_param]
        m_prev = stp["exp_avg"].clone()
        v_prev = stp["exp_avg_sq"].clone()
        optimizer.step()
        if post_step_hook is not None:
            post_step_hook(model)
        c_new = c_param.detach().clone()
        # residuals of the charged frames in this batch (for item 4 and the recalibration)
        with torch.no_grad():
            for g in range(int(batch.num_graphs)):
                if pieces["mask"][g] > 0:
                    resid_rows.append(dict(index=int(batches_idx[s][g]),
                                           natoms=int(pieces["n"][g]),
                                           resid=float(batch["energy"][g] - pieces["total"][g])))
        trace.append(dict(
            step=s, sizes=pieces["n"].tolist(), n_charged_large=int(sum(
                1 for g in range(int(batch.num_graphs))
                if pieces["mask"][g] > 0 and pieces["n"][g] >= 120)),
            loss=float(loss), total_grad_norm=float(total_norm), clip_coef=clip_coef,
            g_pre=[g_pre[0, 0].item(), g_pre[0, 1].item()],
            g_etot=[g_etot[0, 0].item(), g_etot[0, 1].item()],
            g_forces=[g_forces[0, 0].item(), g_forces[0, 1].item()],
            g_post=[g_post[0, 0].item(), g_post[0, 1].item()],
            m_prev=[m_prev[0, 0].item(), m_prev[0, 1].item()],
            v_prev=[v_prev[0, 0].item(), v_prev[0, 1].item()],
            dc=[(c_new - c_prev)[0, 0].item(), (c_new - c_prev)[0, 1].item()],
            c=[c_new[0, 0].item(), c_new[0, 1].item()]))
        if s % 40 == 0:
            logging.info("audit replay step %d/%d: c00 %.5f c01 %.5f  dc %s  g_pre %s clip %.3f",
                         s, steps, c_new[0, 0], c_new[0, 1],
                         [f"{x:+.2e}" for x in trace[-1]["dc"]],
                         [f"{x:+.2e}" for x in trace[-1]["g_pre"]], clip_coef)
    c_after = c_param.detach().cpu().clone()
    out["items"]["3_replay"] = dict(
        steps=steps, lr=lr, betas=[beta1, beta2], eps=eps, weight_decay=wd,
        c_before=[c_before_epoch[0, 0].item(), c_before_epoch[0, 1].item()],
        c_after=[c_after[0, 0].item(), c_after[0, 1].item()],
        net_dc=[(c_after - c_before_epoch)[0, 0].item(), (c_after - c_before_epoch)[0, 1].item()],
        steps_with_nonzero_g01=sum(1 for t in trace if t["g_pre"][1] != 0.0),
        steps_with_large_charged=sum(1 for t in trace if t["n_charged_large"] > 0),
        steps_clipped=sum(1 for t in trace if t["clip_coef"] < 1.0),
        sum_g01_etot=sum(t["g_etot"][1] for t in trace),
        sum_g01_forces=sum(t["g_forces"][1] for t in trace),
        sum_g01_pre=sum(t["g_pre"][1] for t in trace),
        sum_g00_pre=sum(t["g_pre"][0] for t in trace),
        sum_abs_g01_pre=sum(abs(t["g_pre"][1]) for t in trace),
        sum_abs_g00_pre=sum(abs(t["g_pre"][0]) for t in trace),
        trace=trace)

    # ------------------------------------------------------------ item 4: the profiler
    by_size: Dict[int, List[float]] = {}
    for r in resid_rows:
        by_size.setdefault(r["natoms"], []).append(r["resid"])
    prof = {}
    for n, vals in by_size.items():
        v = np.asarray(vals)
        inj = 0.7
        c_star = -v.mean()
        c_star_inj = -(v + inj).mean()
        prof[str(n)] = dict(n=int(v.size), raw_mean=float(v.mean()),
                            profiled_mean=float((v + c_star).mean()),
                            within_sd=float(v.std(ddof=1)) if v.size > 1 else None,
                            injected=inj, recovered_shift=float(c_star_inj - c_star),
                            recovery_error=float(abs((c_star_inj - c_star) + inj)))
    out["items"]["4_profiler"] = prof
    out["seconds"] = time.time() - t_start
    text = json.dumps(out, indent=1, default=lambda o: (o.item() if hasattr(o, "item")
                                                         else str(o)))
    with open(out_path, "w") as f:
        f.write(text)
    logging.info("audit written to %s (%.0f s)", out_path, out["seconds"])
    raise SystemExit(0)


def main():
    import mace.tools as tools
    from mace.cli import run_train

    tools.train = audit_train           # run_train calls tools.train(...) by attribute
    run_train.tools.train = audit_train
    run_train.main()


if __name__ == "__main__":
    main()
