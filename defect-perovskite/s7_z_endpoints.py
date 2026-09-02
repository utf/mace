#!/usr/bin/env python3
"""Step 2c: Z endpoints old vs new, and the eps0-versus-correction degeneracy.

TWO QUESTIONS, both cheap and both read-only.

Z ENDPOINTS. The corrected kernel changed the per-species potential the charges are fitting,
so Z should re-settle. Z_Cl is watched in particular: it sets the Madelung well depth at the
vacancy, and it was the one species whose cross-size on-site error was positive.

THE DEGENERACY. `eps0[s]` is a free per-species parameter and the bounded correction
`gamma * tanh(h)` is another; a uniform per-species shift can be carried by either. The audit
found every chlorine pinned at the correction's bound while `eps0[Cl]` sat wherever it
landed, which is what a flat direction looks like when the optimiser walks along it and stops
at a wall. If the two are redundant, widening gamma moves the wall without removing the
degeneracy -- and the joint run would have a flat direction that no gate in the current set
would detect. Measured as: across seeds, does `eps0[Cl]` move opposite to the mean Cl
correction, and does their SUM vary less than either part?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)

SYMBOL = {0: "Cl", 1: "Cs", 2: "Pb"}


def summarise(paths, label):
    rows = []
    for mp in paths:
        if not Path(mp).exists():
            continue
        m = torch.load(mp, map_location="cpu", weights_only=False)
        z = [float(x) for x in m.madelung.z.detach()]
        eps0 = m.spectral.h.eps0.detach()
        rows.append(dict(model=Path(mp).name, Z=z,
                         eps0_s=[float(x) for x in eps0[:, 0]],
                         eps0_p=[float(x) for x in eps0[:, 1]],
                         on_site_range=float(getattr(m.spectral.h, "on_site_range", 1.0))))
    if rows:
        z = np.array([r["Z"] for r in rows])
        print(f"\n{label}  ({len(rows)} models, gamma = {rows[0]['on_site_range']})")
        for i, sym in SYMBOL.items():
            print(f"    Z_{sym:2s} {z[:, i].mean():+.4f} +- {z[:, i].std():.4f}")
        for shell in ("eps0_s", "eps0_p"):
            e = np.array([r[shell] for r in rows])
            print(f"    {shell}: " + "  ".join(
                f"{SYMBOL[i]} {e[:, i].mean():+.3f}+-{e[:, i].std():.3f}" for i in SYMBOL))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", nargs="+", type=Path, required=True)
    ap.add_argument("--new", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    old = summarise(args.old, "BEFORE the kernel correction (self-image subtracted)")
    new = summarise(args.new, "AFTER the kernel correction (full sum)")

    out = {"old": old, "new": new}
    if old and new:
        zo = np.array([r["Z"] for r in old])
        zn = np.array([r["Z"] for r in new])
        print("\n  Z shift, new - old:")
        for i, sym in SYMBOL.items():
            d = zn[:, i].mean() - zo[:, i].mean()
            pooled = float(np.hypot(zo[:, i].std(), zn[:, i].std()))
            print(f"    {sym:2s} {d:+.4f} eV against a pooled seed spread of {pooled:.4f}"
                  + ("   MOVED" if abs(d) > 2 * pooled else "   within spread"))
        out["z_shift"] = {SYMBOL[i]: float(zn[:, i].mean() - zo[:, i].mean())
                          for i in SYMBOL}

        # The degeneracy, on the new models: does eps0 absorb what the correction does not?
        es = np.array([r["eps0_s"] for r in new])
        ep = np.array([r["eps0_p"] for r in new])
        print("\n  eps0 spread across seeds (a flat direction shows up as a LARGE spread in "
              "the parts with a small spread in what they sum to):")
        for i, sym in SYMBOL.items():
            print(f"    {sym:2s}  eps0_s {es[:, i].std():.4f}   eps0_p {ep[:, i].std():.4f}")
        out["eps0_spread"] = {SYMBOL[i]: dict(s=float(es[:, i].std()),
                                              p=float(ep[:, i].std())) for i in SYMBOL}
        print("  The correction is pinned at -gamma for every Cl, so its contribution is the "
              "same constant in every seed. A large eps0[Cl] spread against that constant is "
              "the flat direction; a small one means eps0 is pinned by something else.")
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
