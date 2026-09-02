#!/usr/bin/env python3
"""Which SPECIES saturate? The audit's 59.7% is suspiciously exact across five seeds.

95 / 159 = 0.5975, and a 159-atom V_Cl cell holds exactly 95 Cl (32 formula units minus the
vacancy), 32 Cs, 32 Pb. Five seeds landing on the same fraction to three figures is not five
seeds learning the same thing -- it is a set determined by species rather than by training.
If it is the anion, the recommendation changes: widen the on-site bound where the
missing-anion Madelung shift acts, not uniformly.

Prints the saturated fraction per species per seed, and the signed mean of the pre-tanh
argument, since a bound pinned at +1 eV and one pinned at -1 eV are different failures.
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
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

SYMBOL = {0: "Cl", 1: "Cs", 2: "Pb"}
SATURATED = 2.0


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--atoms", type=int, default=159)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == args.atoms][:1]
    if not frames:
        raise SystemExit(f"no charged {args.atoms}-atom frames")
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    counts = {SYMBOL[i]: int((np.array([{17: 0, 55: 1, 82: 2}[int(z)]
                                        for z in frames[0].get_atomic_numbers()]) == i).sum())
              for i in range(3)}
    print(f"{args.atoms}-atom cell composition: {counts}", flush=True)

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        batch, frs = make_batches(frames, z_table, cutoff, 1, args.device)[0]
        head = model.spectral
        head.h._audit_bin = {}
        bin_ = head.h.audit(True)
        try:
            with torch.no_grad():
                model(ctx.forward_dict(batch, frs, requires_grad=False),
                      training=False, compute_force=False)
            # The FIRST call only. `_carrier_head` runs twice per forward -- once at this
            # frame's counters and once at the reference -- so the buffer holds two passes
            # over the same atoms, and concatenating them would double-count.
            first = bin_["site"][0]
        finally:
            head.h.audit(False)
        species = batch.node_attrs.argmax(dim=-1).detach().cpu()
        n_atoms = int(species.numel())
        if first.numel() % n_atoms:
            raise SystemExit(f"site channel has {first.numel()} entries for {n_atoms} "
                             "atoms; the per-atom mapping is wrong")
        # Several channels per atom (s and p, over the head's channel slots). An atom counts
        # as saturated if ANY of its on-site corrections is pinned -- the bound binds per
        # channel, and one pinned channel is one frozen parameter.
        per_atom = first.reshape(n_atoms, -1)
        site = per_atom.abs().max(dim=-1).values * torch.sign(
            per_atom.gather(-1, per_atom.abs().argmax(dim=-1, keepdim=True)).reshape(-1))
        row = {"model": Path(mp).name, "species": {}}
        for i, sym in SYMBOL.items():
            v = site[species == i]
            if v.numel() == 0:
                continue
            row["species"][sym] = dict(
                n=int(v.numel()),
                saturated_fraction=float((v.abs() > SATURATED).float().mean()),
                signed_mean=float(v.mean()), abs_max=float(v.abs().max()))
        rows.append(row)
        print(f"  {row['model']:20s} " + "   ".join(
            f"{k} {100 * d['saturated_fraction']:5.1f}% (mean {d['signed_mean']:+.2f})"
            for k, d in row["species"].items()), flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    for sym in SYMBOL.values():
        f = np.array([r["species"][sym]["saturated_fraction"] for r in rows
                      if sym in r["species"]])
        m = np.array([r["species"][sym]["signed_mean"] for r in rows if sym in r["species"]])
        if f.size:
            print(f"\n  {sym}: saturated {100 * f.mean():.1f}% +- {100 * f.std():.1f}%, "
                  f"signed mean of the pre-tanh argument {m.mean():+.3f} +- {m.std():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
