"""Dump the alpha and site-energy maps behind an AMBIGUOUS gate verdict (plan section 9.1).

An AMBIGUOUS model is neither localised nor cleanly sublattice-uniform, so the aggregate
numbers do not say what it settled on. a0_s6 and a0_s8 landed at shell mass 0.486/0.488 and
N_eff 9.54/9.56 from different seeds -- close enough that it looks like a genuine attractor
rather than noise, which is worth resolving rather than filing under "unclear".

Prints, per model: where the attention actually sits by species and by distance, and how
concentrated it is on its own support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_gates import KEYSPEC, evaluate  # noqa: E402
from vacancy_site import distance_to_vacancy, locate_vacancy  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    frames = [a for a in read(args.data, ":")
              if np.asarray([int(x) for x in a.info["carrier_counts"].split()]
                            if isinstance(a.info["carrier_counts"], str)
                            else a.info["carrier_counts"]).any()][: args.frames]
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    for path in args.models:
        model = torch.load(path, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        rows = evaluate(model, frames, z_table, 5.0, args.device)

        by_species, by_dist, top_frac, eps_by_species = [], [], [], []
        for r in rows:
            active = int(np.argmax(r["counts"]))
            a = r["alpha"][:, active]
            a = a / a.sum()
            atoms = r["atoms"]
            try:
                site = locate_vacancy(atoms)
            except ValueError:
                continue
            d = distance_to_vacancy(atoms, site)
            sym = np.array(atoms.get_chemical_symbols())

            by_species.append([a[sym == s].sum() for s in ("Cs", "Pb", "Cl")])
            edges = [0, 4, 6, 8, 10, 100]
            by_dist.append([a[(d >= lo) & (d < hi)].sum()
                            for lo, hi in zip(edges[:-1], edges[1:])])
            # How much of the mass sits on its top 10 atoms: separates "spread over a
            # sublattice" from "concentrated on a handful of sites".
            top_frac.append(np.sort(a)[::-1][:10].sum())
            if r["site"] is not None:
                eps = r["site"][:, active]
                eps_by_species.append([eps[sym == s].std() for s in ("Cs", "Pb", "Cl")])

        sp = np.mean(by_species, axis=0)
        di = np.mean(by_dist, axis=0)
        print(f"\n=== {path.stem} ===")
        print(f"  alpha by species : Cs {sp[0]:.3f}  Pb {sp[1]:.3f}  Cl {sp[2]:.3f}")
        print(f"  alpha by distance: <4A {di[0]:.3f}  4-6 {di[1]:.3f}  6-8 {di[2]:.3f}  "
              f"8-10 {di[3]:.3f}  >10 {di[4]:.3f}")
        print(f"  mass on top 10 atoms: {np.mean(top_frac):.3f}")
        if eps_by_species:
            e = np.mean(eps_by_species, axis=0)
            print(f"  within-species eps spread: Cs {e[0]:.3f}  Pb {e[1]:.3f}  "
                  f"Cl {e[2]:.3f} eV")


if __name__ == "__main__":
    main()
