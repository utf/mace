#!/usr/bin/env python3
"""Section 4: the electronic-temperature scan. Forward-only; the output is a config choice.

`T_el = 25 meV` was inherited, never chosen. It sets the width of the Fermi function that
fills the spectrum, and it enters three things that pull in different directions:

  the gradient bound  the Daleckii-Krein divided difference is bounded by `1/(4 T_el)`, so a
                      small `T_el` makes the density response stiff and a large one blurs it;
  the fill            with a level spacing of ~0.01 eV near the frontier -- measured, not
                      assumed -- `T_el = 25 meV` smears over SEVERAL levels, so the "bound"
                      state is partly a thermal average rather than one state;
  the entropy         `-T_el S` is a real term in the free energy the head returns.

What is scanned, per model and per `T_el`: the head energy, the frontier gap, the
participation `N_eff`, and the occupation of the changed level. A `T_el` that leaves the
carrier level only fractionally occupied is filling neighbours instead, which is visible here
and invisible in any single-temperature number.

NO RETRAINING and no edit. The decision this feeds is a config parameter for section 5, and
the scan is attached to it.
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
from mace.modules.defect_context import ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import capture, load_frames, select  # noqa: E402

T_GRID = (0.005, 0.010, 0.025, 0.050, 0.100)


def probe(model, batch, frames, ctx, t_el):
    """One forward at a given T_el, reading the head's own spectrum and energy."""
    from mace.modules.defect_counting import (VALENCE, changed_level_index, fermi_fill,
                                              resolve_fills)

    old = model.spectral.t_el
    model.spectral.t_el = float(t_el)
    try:
        internals, out = capture(model, batch, ctx=ctx, frames=frames)
    finally:
        model.spectral.t_el = old
    lam = internals["lam"][0, 0]
    lam = torch.sort(lam[lam < 500.0]).values
    n_total = sum(VALENCE[int(z)] for z in frames[0].get_atomic_numbers())
    counts = batch.carrier_counts.reshape(-1).tolist()
    k = changed_level_index(n_total, counts)
    (n_maj, _), _ = resolve_fills(n_total, counts)
    f = fermi_fill(lam.double(), float(n_maj), float(t_el))
    alpha = out["carrier_alpha"][:, 0]
    mass = alpha / alpha.sum().clamp_min(1e-30)
    return dict(
        t_el=float(t_el),
        e_head=float(out["delta_sr_energy"].reshape(-1)[0]),
        gap=float(lam[k + 1] - lam[k]) if k + 1 < lam.numel() else float("nan"),
        occ_changed=float(f[k]) if k < f.numel() else float("nan"),
        # How many levels the smearing straddles: the count with a fractional occupation.
        n_fractional=int(((f > 1e-3) & (f < 1.0 - 1e-3)).sum()),
        neff=float(1.0 / (mass ** 2).sum().clamp_min(1e-30)),
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--atoms", type=int, default=159)
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == args.atoms][:3]
    if not frames:
        raise SystemExit(f"no charged {args.atoms}-atom frames in {args.data}")
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"{len(frames)} charged {args.atoms}-atom frames; T_el grid "
          f"{[1000 * t for t in T_GRID]} meV", flush=True)

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        batches = make_batches(frames, z_table, cutoff, 1, args.device)
        for t in T_GRID:
            per = [probe(model, b, fr, ctx, t) for b, fr in batches]
            row = dict(model=Path(mp).name)
            for key in ("e_head", "gap", "occ_changed", "n_fractional", "neff"):
                row[key] = float(np.mean([p[key] for p in per]))
            row["t_el"] = float(t)
            rows.append(row)
            print(f"  {row['model']:20s} T_el {1000 * t:6.1f} meV  "
                  f"E_head {row['e_head']:+8.4f} eV  gap {row['gap']:.4f} eV  "
                  f"f(changed) {row['occ_changed']:.3f}  "
                  f"fractional levels {row['n_fractional']:5.1f}  "
                  f"N_eff {row['neff']:6.2f}", flush=True)
            args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        print("\n  averaged over seeds:")
        for t in T_GRID:
            sel = [r for r in rows if r["t_el"] == t]
            if not sel:
                continue
            f = np.mean([r["occ_changed"] for r in sel])
            nf = np.mean([r["n_fractional"] for r in sel])
            ne = np.mean([r["neff"] for r in sel])
            print(f"    T_el {1000 * t:6.1f} meV   f(changed level) {f:.3f}   "
                  f"levels with fractional occupation {nf:5.1f}   N_eff {ne:6.2f}")
        print("  Read the SECOND column: the carrier is one state only where that count is "
              "near 1. Where it is large the fill is spread over neighbours and 'the bound "
              "state' is a thermal average, which no single-temperature number reveals.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
