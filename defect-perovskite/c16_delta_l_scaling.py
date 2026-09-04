#!/usr/bin/env python3
"""Audit item four: is `delta_L` measured on the population it is applied to?

`bound_switch(depth) = sigmoid((depth/delta_L - 2)/0.5)` divides by the pristine frontier
level spacing, which `collect_pristine_centre` records from the **80-atom** stoichiometric
cells and which the model then applies to 159-atom frames. That is the pattern §7.2b names:
a reference computed on one cell population and used on another.

The expectation going in was 1/N scaling -- a band's mean level spacing halves when the cell
doubles -- which would make the 80-atom value about twice the 160-atom one and understate
`depth/delta_L` for a big cell by the same factor. It is measured here rather than assumed,
by recording `delta_L` from the 80-atom cells and from the same cells tiled 2x1x1 to 160.

The measurement contradicts the expectation, and the entry is recorded as the audit's first
BENIGN finding: the mismatch is real, it is 0.72x rather than 2x, it pushes `s` toward 1, and
it changes nothing while `s` is saturated at 1.000 on every model this programme has
produced. It would matter the moment a level came within ~50 meV of the continuum -- the
regime the switch exists for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402

from mace import tools  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    every = select_pristine(ase_read(str(args.data), ":"), 64)
    biggest = max(len(a) for a in every)
    small = [a for a in every if len(a) == biggest][: args.n_cells]
    big = [a.repeat((2, 1, 1)) for a in small]
    print(f"{len(small)} cells of {biggest} atoms, tiled to {len(big[0])}", flush=True)

    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        adopt_model_dtype(model)
        cut = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        got = {}
        for label, frames in ((str(biggest), small), (str(len(big[0])), big)):
            batches = [b for b, _ in make_batches(frames, z, cut, 1, args.device)]
            model.collect_pristine_centre(batches, device=args.device)
            got[label] = float(model.pristine_level_spacing)
        keys = sorted(got, key=int)
        rows.append(dict(model=mp.name, small=got[keys[0]], big=got[keys[1]],
                         ratio=got[keys[0]] / got[keys[1]]))
        print(f"  {mp.name:16s} delta_L({keys[0]}) {got[keys[0]]:.5f} eV   "
              f"delta_L({keys[1]}) {got[keys[1]]:.5f} eV   "
              f"ratio {rows[-1]['ratio']:.2f}x", flush=True)

    if rows:
        a = np.array([r["small"] for r in rows])
        b = np.array([r["big"] for r in rows])
        ratio = float((a / b).mean())
        print(f"\n  delta_L, small cells  {a.mean():.5f} ± {a.std():.5f} eV")
        print(f"  delta_L, tiled cells  {b.mean():.5f} ± {b.std():.5f} eV")
        print(f"  the small-cell value is {ratio:.2f}x the tiled one "
              "(1/N scaling would give 2.00x)")
        print(f"\n  consequence: a big-cell frame's depth/delta_L is {ratio:.2f}x what it "
              "should be, so `s` is pushed toward 1. Invisible while `s` saturates.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
