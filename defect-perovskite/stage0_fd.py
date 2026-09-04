#!/usr/bin/env python3
"""Stage 0 of plan v8, section 7.2 on the trained model: the per-term FD status, recorded.

Runs `defect_fd.force_check` and `defect_fd.strain_check` on arm A seed 1 -- converted to
the uniform float64 policy, because a float32 trunk contributes ~1e-5 eV of noise on a
500 eV cell, which at h = 1e-4 A is 0.1 eV/A of `epsilon / h` and would drown every
verdict -- over the frames section 7.2 names: an ordinary charged frame, the charged frame
whose frontier gap is smallest among a pool (the model's own spectrum selects it), the
neutral vacancy and a pristine cell, in both gauges. The projector-window frame is
reported as not applicable until the projectors of Stage 0.9 exist.

The atoms sampled are the ones carrying the most carrier density by the model's own
`carrier_alpha` (a model output, not a label) plus random ones, so the head's force is
exercised where it is largest.

Writes `~/runs/stage0_fd.json` with every point and prints the status table. This is a
MEASUREMENT of the v6 functional; the plan's Stage 0 acceptance is that it is recorded.

    python defect-perovskite/stage0_fd.py --model ~/runs/arma_models/arma_s1.model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace.modules import defect_fd as fd
from mace.modules.defect_context import ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from stage0_golden import DATA, frame_selection, git_sha, make_batch  # noqa: E402


def load_uniform(path):
    model = torch.load(path, map_location="cpu", weights_only=False).to("cpu").eval()
    model.precision_policy = "uniform"
    model.double()
    return model


def charged_pool(n: int):
    """The first `n` charged 79-atom frames, for the gap-based selection."""
    from ase.io import read

    out = []
    for a in read(str(DATA / "eval_qp1.xyz"), index=f":{4 * n}"):
        c = a.info.get("carrier_counts")
        if len(a) == 79 and c is not None and int(np.asarray(c).sum()) != 0:
            out.append(a)
            if len(out) == n:
                break
    return out


def gaps_of(model, ctx, frames):
    gaps = []
    with torch.no_grad():
        for a in frames:
            out = model(ctx.forward_dict(make_batch([a], ctx.cutoff)), training=False,
                        compute_force=False)
            gaps.append(float(out["logit_gap"][0, 0]))
    return gaps


def components_for(model, ctx, batch, n_random: int = 2, n_carrier: int = 2, seed: int = 0):
    """Atoms with the most carrier density (the model's own alpha) plus random ones."""
    with torch.no_grad():
        out = model(ctx.forward_dict(batch), training=False, compute_force=False)
    alpha = out["carrier_alpha"][:, 0]
    rng = np.random.default_rng(seed)
    top = torch.argsort(alpha, descending=True)[:n_carrier].tolist()
    n = int(alpha.shape[0])
    rest = [i for i in range(n) if i not in top]
    picked = top + [int(x) for x in rng.choice(rest, size=n_random, replace=False)]
    return [(int(i), int(rng.integers(3))) for i in picked]


def run_frame(model, ctx, atoms, label, gauge, args):
    model.gauge = gauge
    batch = make_batch([atoms], ctx.cutoff)
    data = ctx.forward_dict(batch)
    comps = components_for(model, ctx, batch)
    t0 = time.time()
    force = fd.force_check(model, data, comps, tol=args.force_tol)
    strain = fd.strain_check(model, data, tol=args.stress_tol)
    dt = time.time() - t0
    rows = []
    for r in force + strain:
        rows.append(dict(frame=label, gauge=gauge, term=r.term, kind=r.kind,
                         status=r.status, fit=r.fit, tol=r.tol))
        print(f"  {label:16s} {gauge:9s} {r.kind:6s} {r.term:16s} {r.status:19s} "
              f"min {r.fit['min_error']:.2e}  floor {r.fit.get('floor', float('nan')):.2e}")
    print(f"  ({dt:.0f} s)")
    return rows, fd.as_records(force + strain), comps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="~/runs/stage0_fd.json")
    p.add_argument("--pool", type=int, default=24)
    p.add_argument("--force-tol", type=float, default=1e-4)
    p.add_argument("--stress-tol", type=float, default=1e-5)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args(argv)
    _assert_repo()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    model = load_uniform(args.model)
    ctx = ForwardContext.production(model, device="cpu")
    frames = frame_selection()
    pool = charged_pool(args.pool)
    gaps = gaps_of(model, ctx, pool)
    sel = fd.select_frames(gaps, n_ordinary=1, n_crossing=1)
    chosen = {
        "charged_ordinary": pool[sel["ordinary"][0]],
        "charged_crossing": pool[sel["near_crossing"][0]],
        "vcl0_79": frames["vcl0_79"][0][2],
        "pristine_80": frames["pristine"][0][2],
    }
    print(f"frames: ordinary gap {gaps[sel['ordinary'][0]]:.3f} eV, near-crossing gap "
          f"{gaps[sel['near_crossing'][0]]:.3f} eV; projector-window frame: not applicable "
          "(Stage 0.9)")
    summary, detail = [], {}
    for label, atoms in chosen.items():
        for gauge in ("periodic", "isolated"):
            rows, records, comps = run_frame(model, ctx, atoms, label, gauge, args)
            summary.extend(rows)
            detail[f"{label}/{gauge}"] = dict(components=comps, reports=records)
    out = Path(args.out).expanduser()
    out.write_text(json.dumps(dict(
        git_sha=git_sha(), model=str(args.model), precision_policy="uniform float64",
        h_values=list(fd.H_VALUES), strains=list(fd.STRAINS), force_tol=args.force_tol,
        stress_tol=args.stress_tol, gaps=gaps, selection=sel,
        projector_window="not applicable until Stage 0.9",
        summary=summary, detail=detail), indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    main()
