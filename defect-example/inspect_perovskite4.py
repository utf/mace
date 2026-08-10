#!/usr/bin/env python3
"""Spin state per class and charge: what multiplicity should the metadata carry?

MACEDefect requires a multiplicity, and it must be the real one -- the counter
canonicalisation is defined against it. A cell-level magmom near zero means an
unpolarised calculation (multiplicity 1); near one means a doublet.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from inspect_perovskite2 import CLASSES, ROOT, read

for charge in ("01_0", "02_+1"):
    print(f"\n=== {charge} ===")
    grouped = defaultdict(list)
    for split in ("train", "val"):
        path = ROOT / charge / f"{split}_pbe.xyz"
        if not path.is_file():
            continue
        for row in read(path):
            if row["magmom"] is not None:
                grouped[CLASSES.get(row["composition"], row["composition"])].append(
                    row["magmom"]
                )
    for name, values in sorted(grouped.items()):
        array = np.abs(np.array(values))
        print(
            f"  {name:22s} n={len(array):5d}  |magmom| mean {array.mean():.4f}  "
            f"max {array.max():.4f}  >0.5: {(array > 0.5).sum()}"
        )
