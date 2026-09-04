#!/usr/bin/env python3
"""Standing rule 2: turn the A' reference into the null file the trainer gates on.

WHAT A "NEUTRAL NULL" IS, and the whole rule turns on it. Take out-of-fold neutral frames
of one cell size, fit the residual of the frozen base against the hub separation d, and read
the slope. That slope is what the base does at that size WITH NO CARRIER PRESENT. Where it
is consistent with zero, a charged frame's residual at that size is carrier physics and the
head can be fitted to it. Where it is resolved and non-zero, the residual is the base's own
extrapolation, and fitting the head to it teaches the head the base's artefact -- which is
the +0.37 eV/A the 79-atom charged frames carry and the reason the rule exists.

The numbers, from `aprime_reference.json` (Stage A' folds `aprime_f0..3`, out-of-fold):

    159 atoms   +0.0243 [-0.0697, +0.1184] eV/A   brackets zero  -> nulled
     79 atoms   +0.1147 [+0.0947, +0.1348] eV/A   resolved       -> not nulled

THE RULE LIVES IN THE TRAINER, NOT HERE. `mace.data.two_size.nulled_sizes_from` decides
which sizes qualify, by asking whether the interval brackets zero; this script only carries
the evidence and the provenance into the file the trainer reads, so that "has a neutral
null" means one thing everywhere. The 79-atom entry is written out even though it fails,
because a file that omits it cannot be told apart from one that never measured it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Which key of the A' reference carries which size's neutral null. The 159 entry is the
# out-of-fold null; the 79 one is the null over the neutral-dense window, which is where the
# two populations overlap in d and therefore the only place the comparison is meaningful.
SOURCES = {
    159: ("neutral_159", "neutral_159_ci"),
    79: ("neutral_79_window", "neutral_79_window_ci"),
}


def build(reference: dict) -> dict:
    nulls = {}
    for size, (slope_key, ci_key) in SOURCES.items():
        if slope_key not in reference or ci_key not in reference:
            continue
        nulls[str(size)] = {
            "slope": float(reference[slope_key]),
            "ci": [float(x) for x in reference[ci_key]],
            "source_key": slope_key,
        }
    return {
        "nulls": nulls,
        "provenance": {
            "reference": reference.get("source"),
            "production_base": reference.get("production"),
            "rule": ("a size class is nulled when its out-of-fold neutral residual slope's "
                     "95% interval brackets zero; the decision is made by "
                     "mace.data.two_size.nulled_sizes_from, not by this file"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True,
                    help="aprime_reference.json, as c5_references.py writes it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    reference = json.loads(args.reference.read_text())
    payload = build(reference)
    if not payload["nulls"]:
        raise SystemExit(f"{args.reference} carries no neutral null under {sorted(SOURCES)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    from mace.data.two_size import nulled_sizes_from

    admitted = nulled_sizes_from(payload)
    print(f"wrote {args.out}")
    for size, entry in sorted(payload["nulls"].items(), key=lambda kv: int(kv[0])):
        lo, hi = entry["ci"]
        verdict = "NULLED" if int(size) in admitted else "resolved, not nulled"
        print(f"  {size:>4s} atoms  {entry['slope']:+.4f} [{lo:+.4f}, {hi:+.4f}] eV/A"
              f"   {verdict}")
    print(f"  charged energies will enter the loss for sizes {admitted}")


if __name__ == "__main__":
    main()
