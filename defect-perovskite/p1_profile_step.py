#!/usr/bin/env python3
"""Section 1.3: where one training step's time goes, by component.

WHY BEFORE ANY REWRITE. The Stage A' report measured a 3.45 s step with CUDA self time
1.74 s and CPU self time 5.7 s, and concluded "CPU-bound in the head's per-graph eigensolve
and Ewald loop" from the operator table alone. An operator table cannot say which of those
two it is, nor how much of the CPU time is the Fermi bisection's GPU->CPU syncs rather than
arithmetic. Section 1.1 and section 1.2 are different rewrites of different sizes; this
decides how far down each is worth going, and it is the before-half of F24.

WHAT IS MEASURED. The marks in `mace/modules/defect_profile.py`, which bracket the trunk,
the head's per-graph loop and its parts (H assembly, eigh, the four fills, the bisection,
the density response, the force contraction), the Madelung site potential, the long-range
branch and the image term. `record_function` regions nest, so a region's *total* time
includes its children and its *self* time does not; both are reported and the difference is
the reading.

THE LOSS IS A SURROGATE and says so: `sum E^2 + sum F^2`, differentiated to the trainable
parameters. What matters for the timing is the SHAPE of the graph -- a second derivative
through the head, taken to every head parameter -- and that is identical to the real
objective's. The energy/force weights change the numbers in the loss, not the work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules import defect_profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

MARKS = ("step/forward", "trunk", "head/loop", "head/assemble", "head/eigh", "head/fills",
         "head/bisect", "head/response", "head/forces", "ewald/madelung", "ewald/lr",
         "ewald/image", "model/grad_corr", "model/grad_corr_ref",
         "model/outputs", "step/backward")


def one_step(model, batch, params):
    with defect_profile.mark("step/forward"):
        out = model(batch.to_dict(), training=True, compute_force=True)
        loss = out["energy"].pow(2).sum() + out["forces"].pow(2).sum()
    with defect_profile.mark("step/backward"):
        grads = torch.autograd.grad(loss, params, allow_unused=True)
    del grads


def profile(model, batch, params, device, steps: int):
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    activities = [ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    one_step(model, batch, params)                      # warm-up, outside the window
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    previous = defect_profile.enable(True)
    try:
        with torch_profile(activities=activities) as prof:
            for _ in range(steps):
                one_step(model, batch, params)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
    finally:
        defect_profile.enable(previous)
    wall = (time.time() - t0) / max(steps, 1)
    return prof, wall


def marks_table(prof, steps: int):
    """Per-mark total and self time, in seconds per step. CPU and CUDA both.

    ONE MARK IS TWO EVENTS. `key_averages()` returns a host-side row for the annotation
    (which carries the CPU times and, as `device_time_total`, the kernels launched inside
    it) and a device-side row of the same key (which carries `self_device_time_total`).
    Keying a dict by `ev.key` overwrites the first with the second and reports every CPU
    column as zero -- which is what the first run of this script did, and the reason the
    numbers are accumulated rather than assigned.
    """
    rows = {}
    for ev in prof.key_averages():
        if ev.key not in MARKS:
            continue
        r = rows.setdefault(ev.key, dict(count=0.0, cpu_total=0.0, cpu_self=0.0,
                                         cuda_total=0.0, cuda_self=0.0))
        r["count"] = max(r["count"], int(ev.count) / steps)
        r["cpu_total"] += float(ev.cpu_time_total) * 1e-6 / steps
        r["cpu_self"] += float(getattr(ev, "self_cpu_time_total", 0.0)) * 1e-6 / steps
        r["cuda_total"] += float(getattr(ev, "device_time_total", 0.0)) * 1e-6 / steps
        r["cuda_self"] += float(getattr(ev, "self_device_time_total", 0.0)) * 1e-6 / steps
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-batches", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="before")
    args = ap.parse_args()

    _assert_repo()
    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    model.train()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    cutoff = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
    frames = load_frames(args.data)
    # A batch of CHARGED frames: the neutral ones skip the density response entirely, so a
    # mixed batch would time a step the training set only sometimes takes.
    charged = select(frames, charged=True)
    pool = charged if len(charged) >= args.batch_size else frames
    print(f"  {len(charged)} charged of {len(frames)} frames; batching from "
          f"{'charged' if pool is charged else 'all'}", flush=True)
    batches = make_batches(pool[: args.batch_size * args.n_batches], z, cutoff,
                           args.batch_size, args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"  {len(params)} trainable tensors, batch {args.batch_size}", flush=True)

    results = []
    for i, (batch, _) in enumerate(batches):
        prof, wall = profile(model, batch, params, args.device, args.steps)
        rows = marks_table(prof, args.steps)
        results.append(dict(batch=i, wall=wall, marks=rows))
        print(f"\n=== batch {i}: {wall:.3f} s per step ===")
        print(f"  {'mark':16s} {'n':>5s} {'cpu tot':>9s} {'cpu self':>9s} "
              f"{'cuda tot':>9s} {'cuda self':>9s}")
        for k in MARKS:
            r = rows.get(k)
            if r is None:
                print(f"  {k:16s}     -   (region did not run)")
                continue
            print(f"  {k:16s} {r['count']:5.1f} {r['cpu_total']:9.3f} "
                  f"{r['cpu_self']:9.3f} {r['cuda_total']:9.3f} {r['cuda_self']:9.3f}")
        if i == 0:
            table = prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20)
            print("\n--- top 20 by self CPU ---\n" + table)
            results[-1]["table"] = table

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        dict(label=args.label, model=str(args.model), batch_size=args.batch_size,
             steps=args.steps, results=results), indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
