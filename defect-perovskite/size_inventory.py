"""Test 2 step 1: how many charged frames exist at each cell size, and at what d(Pb-Pb)?

Cell-size variation is the only observable in hand that identifies physical boundness: a bound
carrier's force on its own atoms is size-invariant, while a band state's hub amplitude -- and
so the hub force -- halves from 79 to 159 atoms. Everything in Test 2 depends on there being
enough charged 159-atom frames to measure that ratio per d bin, so count them before building
the comparison.

Also reports whether the large cells are single-vacancy 2x1x1 supercells (dilution factor 2)
rather than two-vacancy cells, since the ratio's interpretation depends on it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.geometry import get_distances
from ase.io import read

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vacancy_site import locate_vacancy  # noqa: E402


def counters(atoms):
    c = atoms.info.get("carrier_counts")
    if isinstance(c, str):
        c = [int(x) for x in c.split()]
    return np.asarray(c, dtype=int)


def hub_d(atoms):
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    a, b = int(site.shell[0]), int(site.shell[1])
    pos = atoms.get_positions()
    _, dist = get_distances(pos[a][None], pos[b][None], cell=atoms.get_cell(), pbc=atoms.pbc)
    return float(dist[0, 0])


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe")
    ap.add_argument("--out", type=Path, default=here / "size_inventory.json")
    args = ap.parse_args()

    frames = read(args.data / "train.xyz", ":") + read(args.data / "valid.xyz", ":")
    charged = [a for a in frames if counters(a).any()]
    neutral = [a for a in frames if not counters(a).any()]

    report = {"total": len(frames), "charged": len(charged), "neutral": len(neutral),
              "charged_sizes": dict(Counter(len(a) for a in charged)),
              "neutral_sizes": dict(Counter(len(a) for a in neutral)), "by_size": {}}

    print(f"frames {len(frames)}  charged {len(charged)}  neutral {len(neutral)}")
    print(f"charged sizes: {report['charged_sizes']}")
    print(f"neutral sizes: {report['neutral_sizes']}")

    for size in sorted(report["charged_sizes"]):
        group = [a for a in charged if len(a) == size]
        ds = [d for d in (hub_d(a) for a in group) if d is not None]
        cells = [np.round(np.linalg.norm(a.get_cell(), axis=1), 2).tolist() for a in group[:3]]
        entry = dict(n=len(group), locatable=len(ds), example_cells=cells)
        if ds:
            entry.update(d_min=float(np.min(ds)), d_med=float(np.median(ds)),
                         d_max=float(np.max(ds)),
                         d_hist={f"{lo:.1f}-{lo+0.2:.1f}":
                                 int(((np.array(ds) >= lo) & (np.array(ds) < lo + 0.2)).sum())
                                 for lo in np.arange(4.6, 7.0, 0.2)})
        report["by_size"][str(size)] = entry
        print(f"\n--- charged, {size} atoms: {len(group)} frames "
              f"({len(ds)} with a locatable vacancy) ---")
        if ds:
            print(f"    d(Pb-Pb) min {np.min(ds):.2f}  median {np.median(ds):.2f}  "
                  f"max {np.max(ds):.2f}")
            print(f"    example cell lengths: {cells}")

    # A 2x1x1 single-vacancy supercell has one vacancy in twice the volume; two vacancies in
    # one cell would make the size ratio mean something different.
    for size in sorted(report["neutral_sizes"]):
        print(f"neutral {size} atoms: {report['neutral_sizes'][size]} frames")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
