#!/usr/bin/env python3
"""Can a prefactor on the image term keep gate 10 and pass gate 2?

WHERE THIS COMES FROM. Arm B removes 86.8% of the tiling drift (gate 10, 16/16) and fails
gate 2 at F4 = −0.3200 against a band of [−0.142, −0.063]. `c19_image_slope` showed the term
contributes −0.2439 eV/Å of that slope and the head alone is at −0.0761, inside the band. So
the defect is a MAGNITUDE, and §2.5 has no magnitude: `comp = −s φ_img / ε∞` carries no
amplitude.

If the response were linear in a prefactor λ, F4(λ) ≈ −0.0761 − 0.2439 λ, so λ ≈ 0.27 would
sit at the band's edge and λ ≈ 0.10 near its middle. The question the next cycle actually
needs answered is what the DRIFT does over that range: if the drift reduction is also linear,
λ = 0.27 buys only about a quarter of the 86.8% and the term is not worth its 76% cost; if it
saturates, most of the benefit survives at a magnitude that passes gate 2.

This measures the curve rather than assuming either. `image_potential` is wrapped to return
`scale * phi` on the same trained models, and the tiling drift is re-measured at each scale.
No weight is touched, so this is the term's own contribution at fixed weights -- a model
trained at a given λ would differ, and that is the experiment this one is meant to justify or
rule out before anyone spends a cycle on it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])


def _child(scale: float, models: list[str], out: str, device: str,
           reps: list[str], thermal: int) -> None:
    sys.path.insert(0, REPO)
    sys.path.insert(0, REPO + "/defect-perovskite")
    import mace  # noqa: F401

    import mace.modules.defect_image as di

    original = di.image_potential

    def scaled(*a, **k):
        return original(*a, **k) * float(scale)

    di.image_potential = scaled
    print(f"image_potential SCALED by {scale}", flush=True)

    sys.argv = (["c3_tiling_drift.py", "--models"] + models
                + ["--reps"] + reps
                + ["--thermal", str(thermal), "--device", device, "--out", out])
    import c3_tiling_drift

    c3_tiling_drift.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    ap.add_argument("--reps", nargs="+", default=["1", "3"])
    ap.add_argument("--thermal", type=int, default=1)
    ap.add_argument("--out-prefix", default=str(Path.home() / "runs" / "image_scale"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--child", type=float, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child is not None:
        _child(args.child, args.models, f"{args.out_prefix}_{args.child}.json",
               args.device, args.reps, args.thermal)
        return

    for scale in args.scales:
        print(f"\n{'=' * 70}\n=== image term scaled by {scale}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, __file__, "--child", str(scale),
                        "--models"] + args.models
                       + ["--reps"] + args.reps
                       + ["--thermal", str(args.thermal), "--device", args.device,
                          "--out-prefix", args.out_prefix], check=False)

    print("\n=== drift ratio against the prefactor ===")
    print("  scale   mean |D_with| / D0   (arm B at scale 1.0 was 0.132)")
    import numpy as np
    for scale in args.scales:
        p = Path(f"{args.out_prefix}_{scale}.json")
        if not p.exists():
            print(f"  {scale:5.2f}   (did not run)")
            continue
        v = json.loads(p.read_text()).get("verdicts", [])
        if v:
            r = np.mean([abs(x["ratio"]) for x in v])
            print(f"  {scale:5.2f}   {r:.3f}   ({100 * (1 - r):.1f}% of the drift removed, "
                  f"{sum(x['passes'] for x in v)}/{len(v)} inside the gate)")
    print("\n  Read against the F4 line: F4(scale) is about -0.0761 - 0.2439*scale if the "
          "response\n  is linear, so the band's edge is near scale 0.27.")


if __name__ == "__main__":
    main()
