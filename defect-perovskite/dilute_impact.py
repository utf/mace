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

import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
periodic = {r["n_host"]: r for r in
            json.loads((here / "perov_size_lr.json").read_text())["rows"]}
dilute = {r["n_host"]: r for r in
          json.loads((here / "perov_size_lr_dilute.json").read_text())["rows"]}
sizes = sorted(set(periodic) & set(dilute))

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
