"""Plan P: is the base-v2-relaxed pristine cell the same tilt pattern as the static one?
Pb-Cl-Pb angles and Pb-Cl bond lengths before and after (recorded, not gated)."""
import json, sys
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
doc = json.load(open(sys.argv[1]))
def stats(pos):
    a = Atoms(numbers=doc["numbers"], positions=pos, cell=doc["cell"], pbc=True)
    i, j, d, D = neighbor_list("ijdD", a, 3.4)
    z = a.get_atomic_numbers()
    pbcl = [(ii, jj, dd, DD) for ii, jj, dd, DD in zip(i, j, d, D) if z[ii] == 82 and z[jj] == 17]
    bonds = np.array([dd for _, _, dd, _ in pbcl])
    # Pb-Cl-Pb angles: for each Cl, its two Pb neighbours
    angles = []
    byCl = {}
    for ii, jj, dd, DD in pbcl:
        byCl.setdefault(jj, []).append(-DD)      # vector Cl -> Pb
    for cl, vs in byCl.items():
        if len(vs) == 2:
            c = np.dot(vs[0], vs[1]) / np.linalg.norm(vs[0]) / np.linalg.norm(vs[1])
            angles.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return bonds, np.array(angles), len(byCl)
for name, pos in (("static", doc["positions_static"]), ("relaxed_v2", doc["positions"])):
    b, ang, ncl = stats(pos)
    print(f"{name}: Pb-Cl bonds n={len(b)} {b.min():.3f}-{b.max():.3f} mean {b.mean():.3f} A; Pb-Cl-Pb angles n={len(ang)} "
          f"min {ang.min():.1f} mean {ang.mean():.1f} max {ang.max():.1f} deg; Cl with 2 Pb: {ncl}")
