#!/usr/bin/env python3
"""What exactly overlaps between the two charge-state directories?

Geometry-keyed matching found 464 shared geometries in train, every one with an identical
energy in both files. That rules out fixed-geometry charge-state pairs, so the remaining
question is what the overlap actually is -- shared pristine reference frames included in
both training sets would be a sensible dataset design, whereas a partial copy of the
defect frames would be contamination.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from inspect_perovskite2 import CLASSES, ROOT, read


def main() -> None:
    for split in ("train", "val"):
        neutral = read(ROOT / "01_0" / f"{split}_pbe.xyz")
        charged = read(ROOT / "02_+1" / f"{split}_pbe.xyz")
        left = {r["key"]: r for r in neutral}
        right = {r["key"]: r for r in charged}
        shared = set(left) & set(right)

        def label(row):
            return CLASSES.get(row["composition"], row["composition"])

        print(f"\n=== {split} ===")
        print(f"  shared geometries by class: "
              f"{dict(Counter(label(left[k]) for k in shared))}")
        for name, rows, other in (("01_0", left, right), ("02_+1", right, left)):
            unique = set(rows) - set(other)
            print(f"  only in {name:6s} ({len(unique):5d}): "
                  f"{dict(Counter(label(rows[k]) for k in unique))}")

        # If every pristine frame is shared, the overlap is a common reference set.
        pristine_left = {k for k in left if label(left[k]).startswith("pristine")}
        print(f"  pristine in 01_0: {len(pristine_left)}, "
              f"of which shared: {len(pristine_left & shared)}")

        # Energy ranges per class, per charge state: a +1 cell should sit higher than the
        # same neutral one, and if the defect frames are disjoint that is the only handle
        # left on whether the two directories really are different charge states.
        for name, rows in (("01_0", left), ("02_+1", right)):
            print(f"  {name} energy per class:")
            by_class: dict = {}
            for row in rows.values():
                by_class.setdefault(label(row), []).append(row["energy"])
            for cls, values in sorted(by_class.items()):
                print(f"    {cls:22s} n={len(values):5d}  "
                      f"[{min(values):9.3f}, {max(values):9.3f}] eV")


if __name__ == "__main__":
    main()
