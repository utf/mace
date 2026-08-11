#!/usr/bin/env python3
"""Which Cl site does each supercell size end up removing?

``<u>`` came out non-monotone in cell size (-4.277, -4.226, -3.905, -4.277) while ``alpha``
stayed exactly 1.0000, which rules out dilution. A value that returns to its starting point
is not a size trend; the natural suspect is that "the Cl nearest the cell centre" is not the
same crystallographic site at every repeat. Orthorhombic CsPbCl3 has inequivalent Cl sites
(axial and equatorial in the tilted octahedra), and their vacancies are different defects.
"""

from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

import perovskite_size_test as pst

pristine = pst.find_pristine(Path("dataset_pbe"))
print("pristine cell lengths (A):", np.round(pristine.cell.cellpar()[:3], 2))
print()
for repeat in [(1, 1, 2), (2, 2, 2), (3, 2, 2), (3, 3, 2), (3, 3, 3)]:
    supercell = pristine.repeat(repeat)
    pst.set_state(supercell, pst.PRISTINE)
    _, victim = pst.make_vacancy(supercell)
    first, second = neighbor_list("ij", supercell, cutoff=3.4)
    neighbours = sorted(
        supercell.get_chemical_symbols()[b] for a, b in zip(first, second) if a == victim
    )
    # Distance to the two nearest Pb: the axial/equatorial distinction shows up here.
    positions = supercell.get_positions()
    lead = np.flatnonzero(np.array(supercell.get_chemical_symbols()) == "Pb")
    separations = np.sort(np.linalg.norm(positions[lead] - positions[victim], axis=1))[:2]
    edges = np.round(supercell.cell.cellpar()[:3], 1)
    print(f"  {str(repeat):>9s} N={len(supercell):5d} edges={edges}  Cl#{victim:5d}  "
          f"neighbours={neighbours}  d(Pb)={np.round(separations, 3)}")
