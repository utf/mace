#!/usr/bin/env python3
"""Which term makes arm A's level deep? Knockouts at FIXED WEIGHTS.

Arm A's frontier level sits 0.41 eV below the CBM against Stage B's 0.05. Two learned
channels could be responsible: §2.1's on-site correction `gamma tanh(h(x) - h(xbar))` and
§2.2's per-site charges, whose deviations reach 0.66 e on Pb and enter H through
`phi_i / eps_inf`.

Each is disabled in turn on the SAME trained models and the level re-measured:

    site    the per-site charge channel off (`deviation` returns None)
    gamma0  the on-site correction off (`on_site_range` -> 0 for the call; `eps0` and the
            Madelung shift untouched, so it is the learned correction that is removed and
            not the on-site energy itself)
    both    both

CAVEAT, stated rather than implied. A knockout at fixed weights answers "how much of the
level's present position does this term contribute", not "where would the level have settled
had the model been trained without it". The second question needs an arm, not a probe. What
the answer is good for is attribution: if the level returns to the earlier cohort's value
when a term is removed, that term is where the difference lives.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
MODES = ("as_trained", "site", "gamma0", "both")


def _child(mode: str, models_dir: str, out_dir: str, device: str) -> None:
    sys.path.insert(0, REPO)
    sys.path.insert(0, REPO + "/defect-perovskite")
    import mace  # noqa: F401

    if mode in ("site", "both"):
        from mace.modules.defect_madelung import MadelungOnSite

        MadelungOnSite.deviation = lambda self, *a, **k: None
        print("site channel FORCED OFF", flush=True)
    if mode in ("gamma0", "both"):
        from mace.modules.defect_counting import SlaterKosterH

        original = SlaterKosterH.on_site

        def zero_corr(self, *a, **k):
            saved = self.on_site_range
            self.on_site_range = 0.0
            try:
                return original(self, *a, **k)
            finally:
                self.on_site_range = saved

        SlaterKosterH.on_site = zero_corr
        print("on-site CORRECTION range FORCED TO ZERO", flush=True)

    sys.argv = ["b6_depth_edges.py", "--arms", f"arm={models_dir}",
                "--device", device, "--out", f"{out_dir}/depth_knockout_{mode}.json"]
    import b6_depth_edges

    b6_depth_edges.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", required=True,
                    help="a directory of .model files, as b6_depth_edges --arms takes")
    ap.add_argument("--out-dir", default=str(Path.home() / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        _child(args.child, args.models_dir, args.out_dir, args.device)
        return

    for mode in MODES:
        print(f"\n{'=' * 70}\n=== {mode}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, __file__, "--child", mode,
                        "--models-dir", args.models_dir, "--out-dir", args.out_dir,
                        "--device", args.device], check=False)
    print("\nEach block's `depth: from CBM` column is the number; the difference between "
          "`as_trained` and `both` is what the two learned on-site channels contribute.")


if __name__ == "__main__":
    main()
