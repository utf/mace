#!/usr/bin/env python3
"""Section 2.2's required readout: the per-site charge deviation, per seed.

WHAT §2.2 ASKS FOR. The Madelung charges are no longer one number per element:

    Z_i = Z0[s(i)] + zeta * tanh( z(x_i) - z(xbar_{s(i)}) ),   zeta = 1.0 e

with the per-cell sum projected back onto neutrality. The spec says to "report the per-site
deviation magnitude per seed", and nothing else in the suite does: c4 reports the hopping
stops and L_b, c7 reports the ON-SITE correction channel (a different MLP, on a different
head), and b10/b13 never touch the charges. This is that readout.

WHAT IS REPORTED, per model and per species:

    max |dev|   the largest single-site departure from the element baseline, in e
    rms |dev|   the root-mean-square over sites
    |sum dev|   the per-graph sum AFTER centring -- must be ~0, and is the check that the
                centring is doing its job on the deviation as well as the projection on Z0

A channel reading zero everywhere is dead (the site MLP never left its initialisation); a
channel at zeta = 1.0 on many sites is at its stop and the bound is the binding constraint.
Both are failure modes worth naming, and neither is visible from the loss.

REPORT ONLY -- no gate hangs on it this cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402

SPECIES = {17: "Cl", 55: "Cs", 82: "Pb"}
SAT = 0.98


def deviations(model, batch, frames, ctx):
    """`(dev [n_atoms], atomic numbers [n_atoms], per-graph sums)` for one batch, in e."""
    mad = getattr(model, "madelung", None)
    if mad is None or getattr(mad, "site", None) is None:
        return None
    with torch.no_grad():
        out = model(ctx.forward_dict(batch, frames, requires_grad=False),
                    training=False, compute_force=False)
        feats = out["defect_features"][:, : model.spectral_feature_dim]
        species = batch.node_attrs.argmax(dim=-1)
        centre = model.pristine_centre(feats.dtype)
        dev = mad.deviation(species, feats, centre, batch.batch, int(batch.num_graphs))
    if dev is None:
        return None
    numbers = np.asarray([int(n) for n in model.atomic_numbers])
    z = numbers[species.cpu().numpy()]
    idx = batch.batch.cpu().numpy()
    d = dev.cpu().numpy()
    sums = [float(d[idx == g].sum()) for g in range(int(batch.num_graphs))]
    return d, z, sums


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    # No per-host default: eps_inf comes off the model unless the caller overrides it.
    ap.add_argument("--eps-inf", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z_table = tools.AtomicNumberTable([17, 55, 82])
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == 159][: args.frames]
    if not frames:
        raise SystemExit("no charged 159-atom frames")

    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        adopt_model_dtype(model)
        zeta = float(getattr(getattr(model, "madelung", None), "site_zeta", 0.0) or 0.0)
        if zeta == 0.0:
            print(f"  {mp.name}: site channel off (zeta = 0); skipping")
            continue
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        cut = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        dev_all, z_all, sums = [], [], []
        for batch, frs in make_batches(frames, z_table, cut, 4, args.device):
            got = deviations(model, batch, frs, ctx)
            if got is None:
                break
            d, z, s = got
            dev_all.append(d)
            z_all.append(z)
            sums.extend(s)
        if not dev_all:
            print(f"  {mp.name}: no site channel on the loaded model; skipping")
            continue
        dev = np.concatenate(dev_all)
        zz = np.concatenate(z_all)
        row = dict(model=str(mp), zeta=zeta,
                   max_abs=float(np.abs(dev).max()),
                   rms=float(np.sqrt((dev ** 2).mean())),
                   at_stop=float((np.abs(dev) > SAT * zeta).mean()),
                   graph_sum_max=float(np.max(np.abs(sums))),
                   per_species={})
        for num, name in SPECIES.items():
            m = zz == num
            if m.any():
                row["per_species"][name] = dict(
                    n=int(m.sum()), max_abs=float(np.abs(dev[m]).max()),
                    rms=float(np.sqrt((dev[m] ** 2).mean())),
                    mean=float(dev[m].mean()))
        rows.append(row)
        print(f"  {mp.name:20s} zeta {zeta:.2f} e   max |dev| {row['max_abs']:.4f} e   "
              f"rms {row['rms']:.4f} e   at stop {row['at_stop']:.1%}   "
              f"|sum| <= {row['graph_sum_max']:.2e}")
        print("      " + "   ".join(
            f"{k} max {v['max_abs']:.4f} rms {v['rms']:.4f}"
            for k, v in row["per_species"].items()))

    if rows:
        print("\n=== §2.2 per-site deviation, over seeds ===")
        for key in ("max_abs", "rms", "at_stop"):
            v = np.array([r[key] for r in rows])
            print(f"  {key:8s} {v.mean():.4f} +- {v.std():.4f}   "
                  f"(min {v.min():.4f}, max {v.max():.4f})")
        big = sum(1 for r in rows if r["at_stop"] > 0.0)
        print(f"  seeds with any site at the zeta stop: {big}/{len(rows)}")
        dead = sum(1 for r in rows if r["max_abs"] < 1e-6)
        print(f"  seeds with a dead channel (max |dev| < 1e-6 e): {dead}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows), indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
