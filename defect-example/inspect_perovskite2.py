#!/usr/bin/env python3
"""Perovskite dataset: composition classes, duplicates, and geometry-keyed pairing.

Supersedes the id-keyed pass, which was misleading twice over: ids repeat within a file
(1576 frames, 1106 distinct ids), so looking one up finds an arbitrary frame; and an id
match is not a pairing anyway. What MACEDefect needs is the *same geometry* evaluated in
two charge states, so the geometry is the key here and the identifier is only reported.

A pair whose two energies are identical is not a charge state -- it is the same frame
present in both files -- so those are counted separately rather than folded in.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/alex/src/mace/defect-perovskite")
# Pristine CsPbCl3 2x2x2 is Cs16Pb16Cl48; one Cl short is the vacancy.
CLASSES = {
    "Cl48Cs16Pb16": "pristine (80)",
    "Cl47Cs16Pb16": "V_Cl (79)",
    "Cl96Cs32Pb32": "pristine large (160)",
    "Cl95Cs32Pb32": "V_Cl large (159)",
}


def frames(path: Path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            try:
                natoms = int(line.strip())
            except ValueError:
                return
            comment = handle.readline()
            positions = np.empty((natoms, 3), dtype=float)
            species = []
            for index in range(natoms):
                parts = handle.readline().split()
                species.append(parts[0])
                positions[index] = [float(v) for v in parts[1:4]]
            yield comment, positions, species


def read(path: Path):
    """Frames keyed by rounded geometry, so identity is decided by coordinates."""
    out = []
    for comment, positions, species in frames(path):
        identifier = re.search(r'\bid=(\S+)', comment)
        energy = re.search(r'\bREF_energy=(\S+)', comment)
        magmom = re.search(r'\bmagmom=(\S+)', comment)
        composition = "".join(f"{s}{c}" for s, c in sorted(Counter(species).items()))
        out.append(
            {
                "id": identifier.group(1) if identifier else None,
                "energy": float(energy.group(1)) if energy else None,
                "magmom": float(magmom.group(1)) if magmom else None,
                "composition": composition,
                "natoms": len(species),
                # 1e-4 A: far below any thermal displacement, far above float noise.
                "key": (composition, np.round(positions, 4).tobytes()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavour", default="pbe", choices=["pbe", "soc"])
    args = parser.parse_args()

    data = {}
    for charge_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for split in ("train", "val"):
            path = charge_dir / f"{split}_{args.flavour}.xyz"
            if path.is_file():
                data[(charge_dir.name, split)] = read(path)

    print(f"=== composition classes ({args.flavour.upper()}) ===")
    for (charge, split), rows in data.items():
        counts = Counter(CLASSES.get(r["composition"], r["composition"]) for r in rows)
        print(f"  {charge}/{split:5s} {len(rows):5d} frames   {dict(counts)}")

    print("\n=== duplicates within each file (identical geometry) ===")
    for (charge, split), rows in data.items():
        seen = Counter(r["key"] for r in rows)
        repeats = sum(v - 1 for v in seen.values() if v > 1)
        print(f"  {charge}/{split:5s} {len(seen):5d} distinct geometries, "
              f"{repeats} duplicate frames")

    print("\n=== magmom (does the charge state carry a spin?) ===")
    for (charge, split), rows in data.items():
        values = Counter(r["magmom"] for r in rows if r["magmom"] is not None)
        print(f"  {charge}/{split:5s} {dict(values)}")

    print("\n=== pairing: same geometry, both charge states ===")
    charges = sorted({c for c, _ in data})
    for split in ("train", "val"):
        left = {r["key"]: r for r in data.get((charges[0], split), [])}
        right = {r["key"]: r for r in data.get((charges[1], split), [])}
        shared = set(left) & set(right)
        genuine, copied = [], 0
        for key in shared:
            difference = right[key]["energy"] - left[key]["energy"]
            if abs(difference) < 1e-9:
                copied += 1
            else:
                genuine.append((difference, left[key], right[key]))
        print(f"\n  {split}: {len(left)} vs {len(right)} distinct geometries, "
              f"{len(shared)} shared")
        print(f"    genuine pairs (energies differ): {len(genuine)}")
        print(f"    same energy in both files      : {copied}  <- duplicated frames")
        if genuine:
            deltas = np.array([d for d, _, _ in genuine])
            by_class = defaultdict(list)
            for difference, row, _ in genuine:
                by_class[CLASSES.get(row["composition"], row["composition"])].append(
                    difference
                )
            print(f"    E(+1) - E(0): [{deltas.min():.3f}, {deltas.max():.3f}] eV, "
                  f"median {np.median(deltas):.3f}")
            for name, values in sorted(by_class.items()):
                array = np.array(values)
                print(f"      {name:22s} n={len(array):4d}  "
                      f"median {np.median(array):+.3f} eV")


if __name__ == "__main__":
    main()
