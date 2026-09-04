#!/usr/bin/env python3
"""Section 5.2: how far a charged frame's atoms sit from the neutral training set, in the
head's own feature space. REPORT ONLY -- no training role, by the spec.

WHAT IT ASKS. `w_E` (F16, failed twice) tried to detect extrapolation by the disagreement of
four fold bases, and found none: the folds agree on the charged 79-atom frames to about
1 meV/atom, below their disagreement on the neutral ones. That is a statement about the
ENSEMBLE, and an ensemble of four networks trained on nearly the same data can be
confidently wrong together. This asks a different question with no ensemble in it: for each
atom of a charged frame, how far is its first-block feature from the nearest atom of the
NEUTRAL training frames? An atom whose environment has no neighbour in the base's training
distribution is one the base is extrapolating for, whatever four folds happen to agree on.

    d_i        = min over neutral training atoms of ||x_i - x_j||   (first-block features)
    D(frame)   = max over atoms i of d_i

Reported against the hub separation d, for the charged 79- and 159-atom frames and, as the
scale, for the neutral frames themselves (whose own leave-one-out distance is the floor).

THE FEATURES ARE THE HEAD'S, not the trunk's raw block-0 output: the head reads
`defect_feature_readouts[0](block0)[:, :spectral_feature_dim]`, and a distance in the wrong
space is a distance in the wrong space. They are taken from one model, so this measures the
geometry of that model's representation and not a property of the data alone -- which is
stated rather than implied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import hub_separation  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402


def features(model, frames, z_table, cutoff, device, ctx, batch_size=8):
    """`[n_atoms, feature_dim]` per frame, in the head's feature space."""
    out = []
    for batch, fr in make_batches(frames, z_table, cutoff, batch_size, device):
        with torch.no_grad():
            res = model(ctx.forward_dict(batch, fr, requires_grad=False),
                        training=False, compute_force=False)
        feats = res["defect_features"][:, : model.spectral_feature_dim]
        ptr = batch.ptr
        for g in range(int(batch.num_graphs)):
            out.append(feats[ptr[g]: ptr[g + 1]].float())
    return out


def nearest(query: torch.Tensor, reference: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Min distance from each row of `query` to any row of `reference`, chunked."""
    best = torch.full((query.shape[0],), float("inf"), device=query.device)
    for start in range(0, reference.shape[0], chunk):
        block = reference[start: start + chunk]
        d = torch.cdist(query, block)
        best = torch.minimum(best, d.min(dim=1).values)
    return best


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-neutral", type=int, default=400,
                    help="neutral frames sampled for the reference cloud")
    ap.add_argument("--n-charged", type=int, default=400)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    rng = np.random.default_rng(args.seed)
    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    cutoff = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
    ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))

    frames = load_frames(args.data)
    neutral = select(frames, charged=False)
    charged = select(frames, charged=True)
    pick = lambda pool, n: [pool[i] for i in rng.choice(  # noqa: E731
        len(pool), size=min(n, len(pool)), replace=False)]
    # THE NEUTRAL FLOOR MUST BE HELD OUT. Scoring a neutral frame that is itself in the
    # reference cloud gives zero by construction, and a floor of zero makes every charged
    # number look enormous. The pool is split: `n_neutral` frames build the cloud and a
    # DISJOINT hundred are queried against it, so the floor is an honest "how far is a
    # neutral frame from other neutral frames".
    order = rng.permutation(len(neutral))
    ref_frames = [neutral[i] for i in order[: args.n_neutral]]
    held_out = [neutral[i] for i in order[args.n_neutral: args.n_neutral + 100]]
    if not held_out:
        raise SystemExit("not enough neutral frames to hold any out of the cloud")
    print(f"  reference cloud: {len(ref_frames)} neutral frames; "
          f"{len(held_out)} held out as the floor", flush=True)

    ref = torch.cat(features(model, ref_frames, z, cutoff, args.device, ctx))
    print(f"  {ref.shape[0]} reference atoms, {ref.shape[1]} features", flush=True)

    rows = []
    for label, pool in (("neutral_held_out", held_out),
                        ("charged_79", [a for a in charged if len(a) == 79]),
                        ("charged_159", [a for a in charged if len(a) == 159])):
        pool = pick(pool, args.n_charged) if label.startswith("charged") else pool
        if not pool:
            continue
        feats = features(model, pool, z, cutoff, args.device, ctx)
        for atoms, f in zip(pool, feats):
            d = nearest(f, ref)
            rows.append(dict(population=label, natoms=len(atoms),
                             d=float(hub_separation(atoms)),
                             knn_max=float(d.max()),
                             knn_median=float(d.median())))
        vals = np.array([r["knn_max"] for r in rows if r["population"] == label])
        print(f"  {label:12s} n={len(vals):4d}  max-over-atoms kNN distance: "
              f"median {np.median(vals):.4f}  p95 {np.percentile(vals, 95):.4f}",
              flush=True)

    print("\n=== max-over-atoms kNN distance against d, charged 79-atom frames ===")
    c79 = [r for r in rows if r["population"] == "charged_79" and np.isfinite(r["d"])]
    if c79:
        dd = np.array([r["d"] for r in c79])
        kk = np.array([r["knn_max"] for r in c79])
        for lo, hi in ((4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.5)):
            m = (dd >= lo) & (dd < hi)
            if m.sum():
                print(f"  d {lo:.1f}-{hi:.1f}  n={int(m.sum()):4d}  "
                      f"kNN median {np.median(kk[m]):.4f}")
        if len(dd) > 3:
            corr = float(np.corrcoef(dd, kk)[0, 1])
            print(f"  correlation of kNN distance with d: {corr:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(model=str(args.model), rows=rows), indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
