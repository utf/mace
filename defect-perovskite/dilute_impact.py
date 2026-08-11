#!/usr/bin/env python3
"""What does the ``dilute`` option actually change, and does it scale correctly?

``dilute`` is not a switch between two evaluators: it *adds* an isolated-limit correction
on top of the periodic long-range energy. So the right question is not "which is right" but
"is the correction the right size and does it scale like an image term".

An image correction for a charged cell should go as ``q^2 alpha_M / 2 eps L``, i.e. decay as
``N^(-1/3)``. The main term should decay too. Measuring both separates a correctly behaving
correction from a broken quantity it is being added to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The ladder files are ARGUMENTS, not hardcoded names. A fixed default meant that analysing
# a new model silently reported the previous one's numbers: the script reads whatever is on
# disk, exits 0, and the report renders stale results under a new heading. That is the same
# stale-rendered-as-current failure the stage status table exists to catch, and no default
# can be trusted to fail loudly.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("periodic", type=Path, help="ladder JSON, periodic evaluation")
parser.add_argument("dilute", type=Path, help="ladder JSON, --dilute evaluation")
args = parser.parse_args()
for path in (args.periodic, args.dilute):
    if not path.exists():
        raise SystemExit(f"missing ladder file: {path}")

print(f"periodic: {args.periodic.name}")
print(f"dilute:   {args.dilute.name}")
print()
periodic = {r["n_host"]: r for r in json.loads(args.periodic.read_text())["rows"]}
dilute = {r["n_host"]: r for r in json.loads(args.dilute.read_text())["rows"]}
sizes = sorted(set(periodic) & set(dilute))
if not sizes:
    raise SystemExit("the two ladders share no cell sizes")

print(f"{'N':>6s} {'periodic':>11s} {'dilute':>11s} {'correction':>13s}")
differences = []
for n in sizes:
    difference = periodic[n]["eps_opt_eV"] - dilute[n]["eps_opt_eV"]
    differences.append(abs(difference))
    print(f"{n:6d} {periodic[n]['eps_opt_eV']:11.4f} {dilute[n]['eps_opt_eV']:11.4f} "
          f"{difference * 1e3:10.1f} meV")

array = np.array(sizes, dtype=float)
values = np.array(differences)
main = np.array([abs(periodic[n]["eps_opt_eV"]) for n in sizes])
print()
print(f"  correction scaling  d(ln|.|)/d(ln N) = "
      f"{np.log(values[-1] / values[0]) / np.log(array[-1] / array[0]):+.3f}")
print(f"  main term  scaling  d(ln|.|)/d(ln N) = "
      f"{np.log(main[-1] / main[0]) / np.log(array[-1] / array[0]):+.3f}")
print()
print("  -1/3 is the 1/L image scaling; +1 is extensive, i.e. an unneutralised sum.")
