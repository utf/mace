"""One cached training step, timed by component, with and without the frontier term.

Loads a saved model and the base cache its training run wrote, builds a batch of training
frames the way the trainer's loader does (frame keys attached, so the trunk is served from
the cache), and profiles forward + backward with the component marks on. Prints the
inclusive wall (CPU) and GPU time of each marked region and of the whole step, for the
model as saved and for the model with the frontier term switched off.

Usage: python step_timing.py --model ~/runs/s13ra_s1/s13ra_s1.model \
           --cache ~/runs/s13ra_s1/checkpoints/s13ra_s1_basecache_*.pt --device cuda:6
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.modules import defect_cache, defect_profile
from mace.modules.defect_context import ForwardContext
from mace.tools import torch_geometric

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arm_gates import KEYSPEC  # noqa: E402

MARKS = ["head/loop", "head/assemble", "head/eigh", "head/fills", "head/response",
         "head/forces", "ewald/madelung", "ewald/frontier", "model/grad_corr",
         "model/grad_corr_ref"]


def make_batch(frames, model, ctx, device):
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    prepare_defect_configurations(configs)
    ds = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=ctx.cutoff)
          for c in configs]
    defect_cache.attach_frame_keys(ds, z_table=z_table)
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds), shuffle=False)
    return next(iter(loader)).to(device)


def one_step(model, ctx, batch, params):
    d = ctx.forward_dict(batch, requires_grad=True)
    out = model(d, training=True, compute_force=True)
    loss = out["energy"].sum() + (out["forces"] ** 2).sum()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    del grads
    return out


def _dev(ev, name):
    """torch >= 2.4 names the accelerator totals `device_*`; older builds `cuda_*`."""
    return float(getattr(ev, name, getattr(ev, name.replace("device", "cuda"), 0.0)))


def timed(model, ctx, batch, params, device, steps, label):
    from torch.profiler import ProfilerActivity, profile

    cuda = torch.device(device).type == "cuda"
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    out = one_step(model, ctx, batch, params)                    # warm-up
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=activities) as prof:
        for _ in range(steps):
            one_step(model, ctx, batch, params)
        if cuda:
            torch.cuda.synchronize()
    wall = (time.time() - t0) / steps
    rows = {}
    for ev in prof.key_averages():
        if ev.key in MARKS:
            rows[ev.key] = (ev.cpu_time_total / 1e6 / steps,
                            (_dev(ev, "device_time_total") if cuda else 0.0) / 1e6 / steps,
                            ev.count / steps)
    total_cuda = sum(_dev(ev, "self_device_time_total") for ev in prof.key_averages()) \
        / 1e6 / steps if cuda else 0.0
    n_kernels = sum(ev.count for ev in prof.key_averages()
                    if cuda and _dev(ev, "self_device_time_total") > 0) / steps
    print(f"\n== {label}: {wall:.3f} s/step (GPU busy {total_cuda:.3f} s, "
          f"~{n_kernels:.0f} kernel launches)")
    print(f"{'region':22s} {'wall s':>8s} {'gpu s':>8s} {'calls':>6s}")
    for k in MARKS:
        if k in rows:
            w, g, c = rows[k]
            print(f"{k:22s} {w:8.3f} {g:8.3f} {c:6.0f}")
    return wall, out


def plain(model, ctx, batch, params, device, steps):
    """Wall time per step with no profiler and the marks off: what training pays."""
    cuda = torch.device(device).type == "cuda"
    was = defect_profile.enable(False)
    one_step(model, ctx, batch, params)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one_step(model, ctx, batch, params)
    if cuda:
        torch.cuda.synchronize()
    defect_profile.enable(was)
    return (time.time() - t0) / steps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--data", type=Path, default=HERE / "dataset_pbe" / "train.xyz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--plain_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_159", type=int, default=0,
                   help="replace this many picked frames with charged 159-atom frames")
    args = p.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    device = args.device
    model = torch.load(args.model, map_location=device, weights_only=False)
    model = model.to(device)
    model.train()
    from mace.modules.defect_protocol import trainable_mask

    for name, par in model.named_parameters():      # the Stage-3 protocol: head only
        par.requires_grad_(trainable_mask(name))
    params = [q for q in model.parameters() if q.requires_grad]
    cache_path = sorted(glob.glob(str(Path(args.cache).expanduser())))[-1]
    cache = defect_cache.BaseOutputCache.load(cache_path, defect_cache.base_checksum(model),
                                              device=device)
    if cache is None:
        raise SystemExit(f"base cache {cache_path} does not match the model")
    model.set_base_cache(cache)
    ctx = ForwardContext.production(model, device=device)
    frames = read(args.data, ":")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(frames), size=args.batch_size, replace=False)
    sel = [frames[i] for i in pick]
    if args.with_159:
        big = [a for a in frames if len(a) == 159
               and int(np.asarray(a.info["carrier_counts"]).sum()) != 0]
        for j in range(args.with_159):
            sel[-1 - j] = big[int(rng.integers(len(big)))]
    batch = make_batch(sel, model, ctx, device)
    sizes = [len(a) for a in sel]
    charged = [int(np.asarray(a.info["carrier_counts"]).sum()) != 0 for a in sel]
    print(f"batch of {len(sel)}: sizes {sizes}, charged {charged}, cache entries {len(cache)}")
    plain_a = plain(model, ctx, batch, params, device, args.plain_steps)
    ewald = model.frontier_ewald
    model.frontier_ewald = None
    plain_b = plain(model, ctx, batch, params, device, args.plain_steps)
    model.frontier_ewald = ewald
    print(f"unprofiled: {plain_a:.3f} s/step as saved, {plain_b:.3f} s/step with the "
          f"frontier term off -> frontier term {plain_a - plain_b:+.3f} s/step "
          f"({100 * (plain_a - plain_b) / plain_a:.0f}%)")
    defect_profile.enable(True)
    wall_a, out = timed(model, ctx, batch, params, device, args.steps, "as saved")
    w = out["frontier_w"].detach().cpu().tolist()
    print(f"frontier evaluated on {sum(1 for x in w if x >= 0)} of {len(w)} graphs")
    ewald = model.frontier_ewald
    model.frontier_ewald = None
    wall_b, _ = timed(model, ctx, batch, params, device, args.steps, "frontier term off")
    model.frontier_ewald = ewald
    print(f"\nfrontier term: {wall_a - wall_b:+.3f} s/step of {wall_a:.3f} "
          f"({100 * (wall_a - wall_b) / wall_a:.0f}%)")


if __name__ == "__main__":
    main()
