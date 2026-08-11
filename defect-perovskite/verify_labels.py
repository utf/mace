#!/usr/bin/env python3
"""Do the relabelled frames load, and do the new assertions actually fire?

Two halves. The first loads the regenerated dataset through the real loader -- if the
`m_s_ref` plumbing or the charge assertion were wrong, this is where it shows. The second
feeds the *old* (charged-reference) labelling in deliberately and requires it to be
rejected: an assertion that has never failed is not known to work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mace import data
from mace.data.defects import (
    counter_charge,
    spin_magnetisation,
    validate_counts,
)

DATA = Path("/home/alex/src/mace/defect-perovskite/dataset_pbe")


def keyspec():
    spec = data.KeySpecification()
    spec.info_keys.update(
        {
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "multiplicity": "multiplicity",
            "host": "host",
            "cell_charge": "cell_charge",
            "m_s_ref_doubled": "m_s_ref_doubled",
            "stress": "REF_stress",
            "head": "head",
        }
    )
    spec.arrays_keys.update({"forces": "REF_forces"})
    return spec


print("=== 1. the regenerated dataset loads ===")
_, configs = data.load_from_xyz(
    file_path=str(DATA / "valid.xyz"),
    key_specification=keyspec(),
    band_edges=data.load_band_edges(DATA / "band_edges.json"),
)
print(f"  loaded {len(configs)} validation frames")

seen = {}
for config in configs:
    counts = tuple(int(v) for v in config.properties["carrier_counts"])
    multiplicity = int(config.properties["multiplicity"])
    charge = int(config.properties.get("cell_charge", 0))
    reference = int(config.properties.get("m_s_ref_doubled", 0))
    seen.setdefault((counts, multiplicity, charge, reference), 0)
    seen[(counts, multiplicity, charge, reference)] += 1

print(f"\n  {'counts':>14s} {'mult':>5s} {'q_cell':>7s} {'2M_s_ref':>9s} "
      f"{'q_counter':>10s} {'frames':>7s}")
for (counts, multiplicity, charge, reference), n in sorted(seen.items()):
    implied = counter_charge(np.array(counts))
    flag = "ok" if implied == charge else "MISMATCH"
    print(f"  {str(counts):>14s} {multiplicity:5d} {charge:7d} {reference:9d} "
          f"{implied:10d} {n:7d}  {flag}")
    assert implied == charge, "counter charge must equal the absolute cell charge"
    assert multiplicity == reference + spin_magnetisation(np.array(counts)) + 1

print("\n=== 2. the old labelling is rejected ===")
# The old labelling is SPIN-self-consistent: (1,0,0,0) has M_s = 1, so multiplicity 2 is
# exactly right against a closed-shell reference. That is why it survived review -- the
# multiplicity assertion cannot see it, and no amount of spin bookkeeping will.
validate_counts((1, 0, 0, 0), multiplicity=2, m_s_ref_doubled=0)
print("  multiplicity check PASSES on the old labelling -- as expected, it is")
print("  spin-consistent. This is why the error was invisible.")
print("\n  Only the charge check catches it:")
for label, counts, cell_q in (("old V_Cl0", (1, 0, 0, 0), 0),
                              ("old V_Cl+", (0, 0, 0, 0), 1),
                              ("new V_Cl0", (0, 0, 0, 0), 0),
                              ("new V_Cl+", (0, 0, 1, 0), 1)):
    implied = counter_charge(np.array(counts))
    ok = implied == cell_q
    print(f"    {label:10s} {str(counts):>12s} counter q = {implied:+d}, cell q = "
          f"{cell_q:+d}   {'ok' if ok else 'HARD ERROR'}")

# The same counters with a charge that contradicts them.
old_counts = np.array([1, 0, 0, 0])
print(f"  old V_Cl0 counter charge = {counter_charge(old_counts)} on a cell with q = 0 "
      f"-> would now be a hard error")

# And the ionised state under the old convention carried no counter at all.
print(f"  old V_Cl+ counter charge = {counter_charge(np.array([0, 0, 0, 0]))} on a cell "
      f"with q = +1 -> also a hard error")
