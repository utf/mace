"""Ladder v6, part 1: the frozen-core embedding ladder.

WHY. A charge in a periodic cell with FROZEN ions follows the eps_inf Madelung law, which is
exactly what `E_M` encodes. Once the ions relax, the lattice screens the charge too and the
leading coefficient becomes the STATIC one, `-C alpha / 2 eps_0`; with eps_0 several times
eps_inf the fully relaxed slope is a fraction of the eps_inf value. In 16-45 A cells the
relaxation field is clipped, so every cell sits at a different point between the two laws and no
single `1/L` coefficient exists to fit. That is why the relaxed ladder fitted badly and why its
77 % should never have been read against the eps_inf reference.

This ladder removes the ambiguity: the defect core is relaxed ONCE (in the 639-atom cell) and
then embedded, rigid, in larger pristine tilings. The local geometry is then identical in every
cell and inside the training domain; the only thing that changes with size is the periodic
environment, so the leading term is the exact eps_inf monopole whatever `R_core` is -- the core's
polarisation is compensated by a bound surface charge at `R_core`, so the object seen from
outside still carries `+1` and its images interact through the unrelaxed medium.

The core's truncation does add a second moment of order `q_ion R_core^2`, which enters as a
`1/L^3` (Makov-Payne `Q Q_2`) term. It is separated from the intrinsic `1/L^3` terms (the hole's
finite extent, the D13 compensation cloud) by sweeping `R_core`: the truncation term scales as
`q_ion(R) R^2` and the intrinsic ones do not.

Energies only, no forces, no gradients -- the force path is what costs 70 GB.

    python defect-perovskite/dscc_embed_ladder.py --model .../model.pt \
        --core /home/alex/runs/dscc/relax159/relaxed_structures.json --core_key 2,2,2 \
        --r_core 8 --tilings 2,2,2 2,2,3 3,3,3 3,3,4 --out embed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                                   # noqa: E402
from mace.modules.dscc import data as dd                                 # noqa: E402
from mace.modules.dscc.ewald import COULOMB, madelung_self               # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                       # noqa: E402
from mace.tools import torch_geometric                                   # noqa: E402

KEY_TOL = 1e-4          # A; positions of two tilings of one lattice agree to float precision


def vacancy_site(unit: Atoms, tiling, species=17, index=0):
    """The position the removed atom occupied, and the cell, for a tiling."""
    rep = unit.repeat(tuple(int(x) for x in tiling))
    sites = [i for i, z in enumerate(rep.get_atomic_numbers()) if int(z) == species]
    return np.array(rep.get_positions()[sites[index]]), np.array(rep.get_cell())


def relative(pos, origin, cell):
    """Minimum-image position relative to `origin`."""
    d = pos - origin
    return d - np.round(d @ np.linalg.inv(cell)) @ cell


def embed(unit: Atoms, tiling, core_disp, core_origin_cell, r_core: float):
    """A pristine tiling with the relaxed core displacements applied within `r_core`."""
    atoms = tiling_ladder(unit, [tuple(int(x) for x in tiling)], vacancy_species=17, vacancy_index=0)[0]
    vac, cell = vacancy_site(unit, tiling)
    pos = np.array(atoms.get_positions())
    rel = relative(pos, vac, cell)
    r = np.linalg.norm(rel, axis=1)
    moved, missing = 0, 0
    for i in np.nonzero(r <= r_core)[0]:
        key = tuple(np.round(rel[i] / KEY_TOL).astype(np.int64))
        u = core_disp.get(key)
        if u is None:
            missing += 1
            continue
        pos[i] += u
        moved += 1
    atoms.set_positions(pos)
    return atoms, moved, missing, int((r <= r_core).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--core", required=True, help="relaxed_structures.json")
    ap.add_argument("--core_key", default="2,2,2")
    ap.add_argument("--r_core", type=float, nargs="+", default=[8.0])
    ap.add_argument("--tilings", nargs="+", default=["2,2,2", "2,2,3", "3,3,3", "3,3,4"])
    ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    doc = json.load(open(args.static_cell))
    unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)

    # The relaxed core: displacement of every atom of the source cell, keyed by its
    # minimum-image position relative to the vacancy (identical across tilings of one lattice).
    src = json.load(open(args.core))[args.core_key]
    src_t = tuple(int(x) for x in args.core_key.split(","))
    pristine = tiling_ladder(unit, [src_t], vacancy_species=17, vacancy_index=0)[0]
    vac0, cell0 = vacancy_site(unit, src_t)
    p0 = np.array(pristine.get_positions()); p1 = np.array(src["positions"])
    u = p1 - p0
    u -= np.round(u @ np.linalg.inv(cell0)) @ cell0                    # unwrap
    rel0 = relative(p0, vac0, cell0)
    core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(p0))}
    print(f"core from {args.core_key}: {len(core_disp)} atoms, max |u| = {np.linalg.norm(u, axis=1).max():.4f} A",
          flush=True)

    model = torch.load(args.model, weights_only=False, map_location=args.device).to(args.device).eval()
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    def energy(atoms, counts, charge):
        a = atoms.copy()
        a.info.update({"carrier_counts": np.array(counts), "cell_charge": charge,
                       "config_type": "embed", "source_dir": "embed"})
        ds = dd.atomic_data([a], z_table, model.r_cut)
        batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(args.device).to_dict()
        with torch.no_grad():                       # energies only: frontier fills, no densities
            out = model(batch, compute_force=False)
        # the neutral cell short-circuits to the base and carries no diagnostics
        return float(out["energy"][0]), out.get("diagnostics", {})

    rows = []
    for r_core in args.r_core:
        for t in args.tilings:
            tt = tuple(int(x) for x in t.split(","))
            t0 = time.time()
            atoms, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), r_core)
            cell = np.array(atoms.get_cell())
            L = float(abs(np.linalg.det(cell))) ** (1 / 3)
            al = -float(madelung_self(torch.tensor(cell, dtype=torch.float64))) * L / COULOMB
            e1, diag = energy(atoms, [0, 0, 1, 0], 1)
            e0, _ = energy(atoms, [0, 0, 0, 0], 0)
            row = {"tiling": list(tt), "r_core": r_core, "n_atoms": len(atoms), "L": L,
                   "alpha_cell": al, "x": al / L, "E0": e0, "E1": e1, "dE": e1 - e0,
                   "core_atoms_moved": moved, "core_atoms_missing": missing, "core_atoms_inside": inside,
                   "seconds": time.time() - t0}
            for k in ("q0_sum", "e_host", "e_madelung", "dq_sum"):
                if k in diag:
                    row[k] = diag[k][0] if isinstance(diag[k], list) else diag[k]
            rows.append(row)
            print("R_core %4.1f  %s  n=%5d  L=%6.2f  alpha=%.4f  dE=%.4f eV  "
                  "(core %d moved, %d missing of %d; %.0f s)"
                  % (r_core, t, len(atoms), L, al, row["dE"], moved, missing, inside, row["seconds"]),
                  flush=True)
            json.dump(rows, open(args.out, "w"), indent=1)
    json.dump(rows, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
