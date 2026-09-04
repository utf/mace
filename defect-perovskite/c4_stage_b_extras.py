#!/usr/bin/env python3
"""Stage B gates 3 and 7 (spec section 4), read off the trained models.

  * gate 3 (report): c(79), c(159) from the per-(charge, size) table, their difference, and
    the predicted size difference -- the mean E_LR per carrier at each size plus the
    difference of the Ewald G = 0 (jellium background) constant, both measured on the
    charged frames of each size with the model's own kernel;
  * gate 7: the learned decay lengths L_b per integral type, and the fraction of hub bonds
    at the modulation stop under the LOG form (|tanh g| > 0.98), which must not exceed
    one seed in six for any type.

Forward-only, on the sixteen charged 159-atom and a subset of charged 79-atom training
frames. The stop statistic is recovered from the head's own pre-tanh audit, not from a
ratio that assumes the linear form.
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
from s3_dilution import hub_axis  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402
from mace.modules.defect_counting import BOND_TYPES  # noqa: E402

AT_BOUND = 0.98


def hub_stops(model, frames, z_table, cutoff, device, ctx):
    """Per bond type: mean tanh g on the vacancy-flanking Pb-Pb edge, and the fraction at
    the stop, pooled over frames."""
    head = model.spectral.h
    per_type = {b: [] for b in BOND_TYPES}
    for atoms in frames:
        try:
            ia, ib, _, _ = hub_axis(atoms)
        except ValueError:
            continue
        batch, frs = make_batches([atoms], z_table, cutoff, 1, device)[0]
        d = ctx.forward_dict(batch, frs, requires_grad=False)
        bin_ = head.audit(True)
        try:
            with torch.no_grad():
                model(d, training=False, compute_force=False)
        finally:
            head.audit(False)
        pre = torch.cat(bin_["hop"]).reshape(-1, len(BOND_TYPES))
        # The audit stores every edge of every head call in order; the edge list of the
        # head graph is what the model saw, so recover the hub edge by index.
        src, dst = batch.edge_index
        sel = ((src == ia) & (dst == ib)) | ((src == ib) & (dst == ia))
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0 or pre.shape[0] < int(src.numel()):
            continue
        # The first head call is the un-referenced one; its rows are the first n_edges.
        rows = pre[: int(src.numel())][idx.cpu()]
        g = torch.tanh(rows)
        for j, b in enumerate(BOND_TYPES):
            per_type[b].extend(g[:, j].tolist())
    out = {}
    for b, vals in per_type.items():
        v = np.array(vals)
        out[b] = dict(mean_tanh=float(v.mean()) if v.size else float("nan"),
                      at_bound=float((np.abs(v) > AT_BOUND).mean()) if v.size else float("nan"),
                      n=int(v.size))
    return out


def lr_constants(model, frames, z_table, cutoff, device, ctx):
    """Mean E_LR (delta_lr) per carrier and the Ewald G = 0 background constant, per frame."""
    e_lr, bg = [], []
    ew = getattr(model, "latent_ewald", None)
    for batch, frs in make_batches(frames, z_table, cutoff, 1, device):
        d = ctx.forward_dict(batch, frs, requires_grad=False)
        with torch.no_grad():
            out = model(d, training=False, compute_force=False)
            delta_lr = (out["correction_energy"] - out["delta_sr_energy"]).reshape(-1)[0]
            e_lr.append(float(delta_lr))
            if ew is not None:
                # The kernel's OWN G = 0 constant, as LatentEwald.energy adds it:
                # E_bg = -norm_factor * sigma^2 * Q^2 / (2 V), for the net carrier charge
                # the frame carries (|Delta_n| times the frozen amplitude).
                sigma = float(ew.sigma)
                norm = float(ew.ewald.norm_factor)
                vol = float(torch.det(batch.cell.reshape(3, 3)).abs())
                amp = out.get("screening_amplitude")
                q_net = float(amp.reshape(-1)[0]) if amp is not None else 1.0
                bg.append(float(-norm * sigma ** 2 * q_net ** 2 / (2.0 * vol)))
    return (float(np.mean(e_lr)) if e_lr else float("nan"),
            float(np.mean(bg)) if bg else float("nan"))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-small", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable([17, 55, 82])
    frames = load_frames(args.data)
    charged = select(frames, charged=True)
    large = [a for a in charged if len(a) == 159]
    rng = np.random.default_rng(0)
    small_all = [a for a in charged if len(a) == 79]
    small = [small_all[i] for i in sorted(rng.choice(len(small_all), args.n_small,
                                                     replace=False))]
    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device, weights_only=False).to(
            args.device).eval()
        adopt_model_dtype(model)
        cutoff = graph_cutoff_for(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        head = model.spectral
        table = head.c_shift_table.detach().cpu()
        # The charge class the dataset's charged frames actually occupy (Delta_n < 0 here:
        # the counter is a hole), read off the frames rather than assumed.
        from mace.modules.defect_counting import c_shift_classes
        counts0 = torch.tensor([large[0].info["carrier_counts"]], dtype=torch.float64)
        charge_cls = int(c_shift_classes(counts0, torch.tensor([159]))[0][0])
        c_small, c_large = float(table[charge_cls, 0]) + float(head.c_shift), \
            float(table[charge_cls, 1]) + float(head.c_shift)
        lb = head.h.decay_lengths().detach().cpu().tolist()
        e_lr_small, bg_small = lr_constants(model, small, z, cutoff, args.device, ctx)
        e_lr_large, bg_large = lr_constants(model, large, z, cutoff, args.device, ctx)
        stops = hub_stops(model, large + small, z, cutoff, args.device, ctx)
        row = dict(model=mp.name, charge_class=charge_cls, c_79=c_small, c_159=c_large,
                   dc=c_large - c_small,
                   e_lr_79=e_lr_small, e_lr_159=e_lr_large, bg_79=bg_small, bg_159=bg_large,
                   predicted_dc=(e_lr_large - e_lr_small) + (bg_large - bg_small),
                   decay_lengths=dict(zip(BOND_TYPES, lb)), stops=stops,
                   hop_form=str(getattr(head.h, "hop_form", "linear")))
        rows.append(row)
        print(f"  {mp.name:16s} c(79) {c_small:+.4f}  c(159) {c_large:+.4f}  dc {row['dc']:+.4f}"
              f"  predicted dc {row['predicted_dc']:+.4f}  (E_LR {e_lr_small:+.4f}/"
              f"{e_lr_large:+.4f}, bg {bg_small:+.4f}/{bg_large:+.4f})")
        print("     L_b  " + "  ".join(f"{b} {v:.3f}" for b, v in zip(BOND_TYPES, lb)))
        print("     stop " + "  ".join(f"{b} {s['at_bound']:.0%} (tanh {s['mean_tanh']:+.2f})"
                                       for b, s in stops.items()))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n=== gate 7: seeds with any hub bond of a type at its stop ===")
    verdict = {}
    for b in BOND_TYPES:
        n_hit = sum(1 for r in rows if r["stops"][b]["at_bound"] > 0)
        verdict[b] = n_hit
        print(f"  {b:9s} {n_hit}/{len(rows)} seeds  {'OK' if n_hit <= 1 else 'FAIL'}")
    lb = np.array([[r["decay_lengths"][b] for b in BOND_TYPES] for r in rows])
    print("  learned L_b (A), mean +- sd over seeds: " + "  ".join(
        f"{b} {lb[:, j].mean():.3f} +- {lb[:, j].std():.3f}" for j, b in enumerate(BOND_TYPES)))
    print("\n=== gate 3 (report): c per size ===")
    dc = np.array([r["dc"] for r in rows]); pdc = np.array([r["predicted_dc"] for r in rows])
    print(f"  c(79) {np.mean([r['c_79'] for r in rows]):+.4f} +- {np.std([r['c_79'] for r in rows]):.4f}"
          f"   c(159) {np.mean([r['c_159'] for r in rows]):+.4f} +- "
          f"{np.std([r['c_159'] for r in rows]):.4f}   dc {dc.mean():+.4f} +- {dc.std():.4f}"
          f"   predicted {pdc.mean():+.4f} +- {pdc.std():.4f}")
    args.out.write_text(json.dumps(dict(rows=rows, gate7=verdict), indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
