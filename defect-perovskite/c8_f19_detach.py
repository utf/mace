#!/usr/bin/env python3
"""F19 (spec section 2.2): forces with and without the density detach, ten charged frames of
each size, RMS difference < 1 meV/A.

The detach removes the gradient of E_LR with respect to the carrier density; the forces
are dE_LR/dR at fixed q either way, so the two force fields differ only through the
density's own position dependence dq/dR entering E_LR's gradient. Measured on the trained
Stage B seeds by toggling `lr_detach_density` on the same weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, graph_cutoff_for, make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402


def forces(model, frames, z, cutoff, device, ctx):
    out = []
    for batch, frs in make_batches(frames, z, cutoff, 1, device):
        d = ctx.forward_dict(batch, frs, requires_grad=True)
        o = model(d, training=False, compute_force=True)
        out.append(o["forces"].detach().double().cpu().numpy())
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable([17, 55, 82])
    charged = select(load_frames(args.data), charged=True)
    rng = np.random.default_rng(0)
    sets = {}
    for n_at in (79, 159):
        pool = [a for a in charged if len(a) == n_at]
        idx = sorted(rng.choice(len(pool), min(args.n, len(pool)), replace=False))
        sets[n_at] = [pool[i] for i in idx]
    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device, weights_only=False).to(
            args.device).eval()
        adopt_model_dtype(model)
        cutoff = graph_cutoff_for(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        row = dict(model=mp.name)
        for n_at, frs in sets.items():
            model.lr_detach_density = True
            f_on = forces(model, frs, z, cutoff, args.device, ctx)
            model.lr_detach_density = False
            f_off = forces(model, frs, z, cutoff, args.device, ctx)
            model.lr_detach_density = True
            d = np.concatenate([(a - b).reshape(-1) for a, b in zip(f_on, f_off)])
            row[f"rms_{n_at}"] = float(np.sqrt(np.mean(d ** 2)))
            row[f"max_{n_at}"] = float(np.abs(d).max())
        rows.append(row)
        print(f"  {mp.name:16s} rms |F_detach - F_full|  79: {row['rms_79'] * 1000:.4f} meV/A "
              f"(max {row['max_79'] * 1000:.4f})   159: {row['rms_159'] * 1000:.4f} meV/A "
              f"(max {row['max_159'] * 1000:.4f})", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    worst = max(max(r["rms_79"], r["rms_159"]) for r in rows)
    print(f"\n  F19: worst rms {worst * 1000:.4f} meV/A over {len(rows)} models -> "
          f"{'HOLDS' if worst < 1e-3 else 'FAILS'} (< 1 meV/A)")
    args.out.write_text(json.dumps(dict(rows=rows, worst_rms=worst, holds=bool(worst < 1e-3)),
                                   indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
