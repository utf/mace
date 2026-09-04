#!/usr/bin/env python3
"""F10 on the CENTRED on-site correction (spec section 2.1), scored on what the model applies.

b4 reads the on-site MLP's raw pre-tanh output and reports `gamma * tanh(h(x_i))`. Under the
centred form the model applies

    corr_i = gamma * [ tanh h(x_i) - tanh h(xbar_s(i)) ]

so b4's number is not the correction any more, and a raw output driven deep into
saturation (pre-tanh +7 on chlorine, seen on the Stage B seeds) means a DEAD channel
there: tanh(7.08) - tanh(7.03) is zero. This script computes corr_i per atom on the
charged 159-atom frames, classifies the shells as b4 does, and reports per shell the mean
correction, the within-shell spread, and the saturation fraction of the raw output --
plus F10's two conditions: ligand-Cl minus bulk-Cl larger than two within-shell sd AND
larger than 50 meV (the amended rule), in >= 4/6 seeds for gate 4.
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
from b4_per_atom_corrections import classify  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, graph_cutoff_for, make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402

SHELLS = ("hub_Pb", "ligand_Cl", "bulk_Cl", "bulk_Pb", "Cs")
SAT = 0.98


def centred_correction(model, batch, frames, ctx):
    """Per atom: (corr_s [eV], raw tanh at x_i, raw tanh at xbar_s) for one graph.

    THE FORM IS THE MODEL'S, NOT THIS SCRIPT'S. Two are in circulation:

        "output"    corr = gamma [ tanh h(x_i) - tanh h(xbar_s) ]   Stage A'
        "argument"  corr = gamma   tanh[ h(x_i) - h(xbar_s) ]       this cycle

    and scoring one cohort through the other's form reports numbers that model never
    computed -- which is exactly the mistake this script was written to fix when it took
    over from b4. It is read off the head, with the pickle-era default ("output") for a
    model that predates the attribute.

    The two raw tanh values come back in both cases, because the saturation diagnostic is
    about `h` itself and is what tells a dead channel from a small one.
    """
    head = model.spectral.h
    form = getattr(head, "centre_form", "output")
    with torch.no_grad():
        out = model(ctx.forward_dict(batch, frames, requires_grad=False),
                    training=False, compute_force=False)
        feats = out["defect_features"][:, : model.spectral_feature_dim]
        species = batch.node_attrs.argmax(dim=-1)
        e = head.elem(species)
        pre = head.site(torch.cat([feats, e], dim=-1))                      # [n, 2]
        centre = model.pristine_centre(feats.dtype)
        pre_ref = head.site(torch.cat([centre.to(feats.dtype)[species], e], dim=-1))
        raw, ref = torch.tanh(pre), torch.tanh(pre_ref)
        corr = float(head.on_site_range) * (
            torch.tanh(pre - pre_ref) if form == "argument" else raw - ref)
    return (corr[:, 0].cpu().numpy(), raw[:, 0].cpu().numpy(), ref[:, 0].cpu().numpy())


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable([17, 55, 82])
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == 159][: args.frames]
    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device, weights_only=False).to(
            args.device).eval()
        adopt_model_dtype(model)
        if not getattr(model, "on_site_centred", False):
            print(f"  {mp.name}: not centred; skipping")
            continue
        cutoff = graph_cutoff_for(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        per_shell = {s: [] for s in SHELLS}
        sat_raw = {s: [] for s in SHELLS}
        sat_ref = {s: [] for s in SHELLS}
        for batch, frs in make_batches(frames, z, cutoff, 1, args.device):
            cls = classify(frs[0])
            if cls is None:
                continue
            shell, _ = cls
            corr, raw, ref = centred_correction(model, batch, frs, ctx)
            for s in SHELLS:
                sel = shell == s
                per_shell[s].extend(corr[sel].tolist())
                sat_raw[s].extend((np.abs(raw[sel]) > SAT).tolist())
                sat_ref[s].extend((np.abs(ref[sel]) > SAT).tolist())
        row = dict(model=mp.name, shells={})
        for s in SHELLS:
            v = np.array(per_shell[s])
            row["shells"][s] = dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std()),
                                    sat_raw=float(np.mean(sat_raw[s])),
                                    sat_centre=float(np.mean(sat_ref[s])))
        lig, blk = row["shells"]["ligand_Cl"], row["shells"]["bulk_Cl"]
        diff = lig["mean"] - blk["mean"]
        row["ligand_minus_bulk"] = diff
        row["resolved"] = bool(abs(diff) > 2.0 * blk["sd"])
        row["above_floor"] = bool(abs(diff) > 0.05)
        row["f10"] = bool(row["resolved"] and row["above_floor"])
        rows.append(row)
        print(f"  {mp.name}")
        for s in SHELLS:
            d = row["shells"][s]
            print(f"    {s:10s} n {d['n']:4d}  corr {d['mean']:+.4f} +- {d['sd']:.4f} eV  "
                  f"raw saturated {d['sat_raw']:.0%}  centre saturated {d['sat_centre']:.0%}")
        print(f"    ligand-Cl - bulk-Cl {diff:+.4f} eV  (2 sd = {2 * blk['sd']:.4f})  "
              f"resolved {row['resolved']}  > 50 meV {row['above_floor']}  -> F10 "
              f"{'holds' if row['f10'] else 'fails'}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    n_pass = sum(r["f10"] for r in rows)
    print(f"\n=== gate 4 / F10 on the centred correction: {n_pass}/{len(rows)} seeds "
          f"(rule >= 4/6) -> {'PASS' if n_pass >= 4 else 'FAIL'} ===")
    for s in SHELLS:
        m = np.array([r["shells"][s]["mean"] for r in rows])
        sr = np.array([r["shells"][s]["sat_raw"] for r in rows])
        print(f"  {s:10s} corr {m.mean():+.4f} +- {m.std():.4f} eV over seeds;  raw output "
              f"saturated in {sr.mean():.0%} of atoms (seed range {sr.min():.0%}-{sr.max():.0%})")
    args.out.write_text(json.dumps(dict(rows=rows, n_pass=n_pass), indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
