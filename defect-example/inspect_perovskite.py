#!/usr/bin/env python3
"""What is in the perovskite dataset, and is any of it paired?

Pairing is the question that matters most for MACEDefect: the delta targets are
fixed-geometry charge-state differences, and they only exist where the *same* geometry
appears under two different charge states. Two charge-state directories are not by
themselves evidence of that -- they may be independent relaxations of the same defect,
which look similar and are not paired at all.

So this checks geometry identity, not just matching identifiers, and it reads only the
comment lines plus positions rather than building any graphs.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/alex/src/mace/defect-perovskite")


def frames(path: Path):
    """Yield (natoms, comment, positions, species) without ASE, so this stays fast."""
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
            yield natoms, comment, positions, species


def info(comment: str) -> dict:
    out = {}
    for key in ("id", "REF_energy", "energy", "magmom", "charge", "spin",
                "total_charge", "config_type"):
        found = re.search(rf'\b{key}=("[^"]*"|\S+)', comment)
        if found:
            out[key] = found.group(1).strip('"')
    return out


def summarise(path: Path, limit: int | None):
    ids, sizes, compositions, energies = [], [], [], []
    keys_seen: Counter = Counter()
    geometries = {}
    for index, (natoms, comment, positions, species) in enumerate(frames(path)):
        if limit and index >= limit:
            break
        meta = info(comment)
        keys_seen.update(meta.keys())
        ids.append(meta.get("id"))
        sizes.append(natoms)
        compositions.append("".join(f"{s}{c}" for s, c in sorted(Counter(species).items())))
        if "REF_energy" in meta:
            energies.append(float(meta["REF_energy"]))
        if meta.get("id"):
            geometries.setdefault(meta["id"], positions)
    return {
        "n": len(sizes),
        "ids": ids,
        "sizes": Counter(sizes),
        "compositions": Counter(compositions),
        "energies": energies,
        "keys": keys_seen,
        "geometries": geometries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavour", default="pbe", choices=["pbe", "soc"])
    parser.add_argument("--limit", type=int, default=None,
                        help="frames per file, for a quick look")
    args = parser.parse_args()

    files = {}
    for charge_dir in sorted(ROOT.iterdir()):
        if not charge_dir.is_dir():
            continue
        for split in ("train", "val"):
            path = charge_dir / f"{split}_{args.flavour}.xyz"
            if path.is_file():
                files[(charge_dir.name, split)] = path

    summaries = {}
    print(f"=== {args.flavour.upper()} files ===")
    for key, path in files.items():
        data = summarise(path, args.limit)
        summaries[key] = data
        charge, split = key
        named = sum(1 for i in data["ids"] if i)
        print(f"\n{charge}/{split}: {data['n']} frames, {named} with an id")
        print(f"  cell sizes      {dict(data['sizes'])}")
        print(f"  compositions    {dict(list(data['compositions'].items())[:3])}")
        if data["energies"]:
            values = np.array(data["energies"])
            print(f"  REF_energy      [{values.min():.3f}, {values.max():.3f}] eV")
        print(f"  comment keys    {sorted(data['keys'])}")

    # ---- pairing ----------------------------------------------------------------------
    print("\n=== pairing between charge states ===")
    charges = sorted({c for c, _ in files})
    if len(charges) < 2:
        print("  only one charge state present")
        return
    for split in ("train", "val"):
        left = summaries.get((charges[0], split))
        right = summaries.get((charges[1], split))
        if not left or not right:
            continue
        shared = set(left["geometries"]) & set(right["geometries"])
        print(f"\n  {split}: {len(left['geometries'])} vs {len(right['geometries'])} "
              f"unique ids, {len(shared)} shared")
        if not shared:
            print("    no shared ids -- nothing can be paired by identifier")
            continue
        # An identifier match is not a pairing. Only identical coordinates are.
        identical, moved = 0, []
        for name in sorted(shared):
            a, b = left["geometries"][name], right["geometries"][name]
            if a.shape != b.shape:
                continue
            delta = float(np.abs(a - b).max())
            if delta < 1e-6:
                identical += 1
            else:
                moved.append((name, delta))
        print(f"    identical geometry: {identical} / {len(shared)}")
        if moved:
            worst = sorted(moved, key=lambda kv: -kv[1])[:3]
            print(f"    differing:          {len(moved)}  "
                  f"(largest shifts {[(n, round(d, 3)) for n, d in worst]} A)")
        # Same id AND same energy in both files would mean the file was duplicated, not
        # that a charge state was computed -- worth separating from genuine pairing.
        if identical:
            same_energy = 0
            for name in sorted(shared)[:200]:
                ea = left["energies"][left["ids"].index(name)] if name in left["ids"] else None
                eb = right["energies"][right["ids"].index(name)] if name in right["ids"] else None
                if ea is not None and eb is not None and abs(ea - eb) < 1e-6:
                    same_energy += 1
            print(f"    of those, identical REF_energy too: {same_energy} "
                  f"(a charge state that changed nothing is a copied file, not a pair)")


if __name__ == "__main__":
    main()
