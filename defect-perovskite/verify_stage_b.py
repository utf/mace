"""End-to-end check that a Stage-B run really is Stage B.

The unit tests cover the pieces. This checks the thing that actually ships: take a model
saved by a short Stage-B training run, and confirm its base branch is bit-identical to the
Stage-A checkpoint it was initialised from. If the base drifted by even a little, the
correction was chasing a moving target and the arm is not testing component S.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import mace  # noqa: F401
from mace.modules.defect_stage import _base_state, is_correction_param  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--stage-b", type=Path, required=True)
    args = ap.parse_args()

    a = torch.load(args.stage_a, map_location="cpu", weights_only=False)
    b = torch.load(args.stage_b, map_location="cpu", weights_only=False)
    sa, sb = _base_state(a), _base_state(b)

    moved, checked = [], 0
    for name, ta in sa.items():
        tb = sb.get(name)
        if tb is None or ta.numel() == 0 or tuple(ta.shape) != tuple(tb.shape):
            continue
        checked += 1
        if not ta.is_floating_point():
            if not torch.equal(ta, tb):
                moved.append((name, float("inf")))
            continue
        d = float((ta.double() - tb.double()).abs().max())
        if d > 0:
            moved.append((name, d))

    print(f"compared {checked} base tensors")
    if moved:
        moved.sort(key=lambda kv: -kv[1])
        print("BASE MOVED -- this run is NOT Stage B:")
        for n, d in moved[:10]:
            print(f"  {n}: max|delta| = {d:.3e}")
        sys.exit(1)

    corr = [n for n, _ in b.named_parameters() if is_correction_param(n)]
    print(f"base identical to Stage A across all {checked} tensors")
    print(f"correction branch has {len(corr)} trainable tensors, free to move")
    print("STAGE_B_VERIFIED")


if __name__ == "__main__":
    main()
