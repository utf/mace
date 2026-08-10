#!/usr/bin/env python3
"""Attach MACEDefect metadata to the CsPbCl3 V_Cl dataset (PBE) and write train/valid.

Source: Mosquera-Lois & Walsh, *Dynamic Vacancy Levels in CsPbCl3 Obey Equilibrium Defect
Thermodynamics*, PRX Energy 4, 043008 (2025). Chloride vacancy in orthorhombic CsPbCl3,
sampled by MD at 300 K, in two charge states.

WHAT THE SOURCE DATA IS
-----------------------
Two directories, one per charge state, each holding PBE and HSE+SOC copies. ``_pbe`` and
``_soc`` are **levels of theory**, not spin treatments -- the paper trains a multi-head
model on cheap PBE plus a small high-fidelity HSE+SOC set. Only PBE is used here.

Composition classes, from the atom counts:

    Cs16Pb16Cl48  (80 atoms)   pristine
    Cs16Pb16Cl47  (79 atoms)   V_Cl
    Cs32Pb32Cl95  (159 atoms)  V_Cl, larger cell

THERE ARE NO PAIRED FRAMES, AND THAT IS BY DESIGN
-------------------------------------------------
Keyed on geometry rather than on the ``id`` field -- ids repeat within a file, so an id
match proves nothing -- the two charge states share 464 geometries in train, and every one
has an *identical* energy in both files. Those are the pristine reference frames, included
in both training sets. The V_Cl frames are completely disjoint: 1057 neutral against 928
charged, zero in common.

That follows from how the paper works: it trains **separate models per charge state** and
evaluates the optical level afterwards as ``eps_opt = E+(R0) - E0(R0)`` on frames from the
neutral trajectory. The fixed-geometry pairing exists in their *analysis*, not in their
training data.

Consequence for MACEDefect: ``delta_energy`` and ``delta_forces`` have no support here, so
the paired terms are inert and supervision has to come from ``L_base`` plus ``L_tot``. The
``eps_opt`` curve the paper reports is then a genuine held-out test rather than a fit --
which is the appealing part, since one charge-conditioned model would replace their two.

COUNTER CONVENTION, AND WHY IT CANNOT BE THE OBVIOUS ONE
--------------------------------------------------------
Measured cell magnetisation: V_Cl neutral |m| = 1.004 (a doublet, one unpaired electron),
V_Cl charged |m| = 0.010 and pristine 0.004 (both closed shell).

A neutral doublet cannot be expressed with ``q = 0``. With ``q = (n2+n3) - (n0+n1)`` and
``M_s = (n0-n2) - (n1-n3)``, setting ``q = 0`` forces ``n0+n1 = n2+n3``, which makes
``M_s`` even and the multiplicity odd. So the reference state must be the **ionised**
``V_Cl+``, and the neutral vacancy carries one electron relative to it:

    pristine     n = (0, 0, 0, 0)   multiplicity 1    q_cell  0
    V_Cl+        n = (0, 0, 0, 0)   multiplicity 1    q_cell +1
    V_Cl0        n = (1, 0, 0, 0)   multiplicity 2    q_cell  0     (q = -1 vs reference)

One thing to be explicit about: this puts a neutral pristine cell and a +1 charged vacancy
cell both at ``n = 0``, i.e. both on the base surface, even though their absolute charges
differ. They differ in composition, so the model can tell them apart and the E0 fit absorbs
the difference -- but "base" now means *the reference charge state for that composition*
rather than "neutral", and any transition level read off this model inherits that
convention. Stated here because nothing downstream can infer it.

BAND EDGES
----------
The referencing constant is ``n_electrons * e_cbm - n_holes * e_vbm``; with one electron
and no holes only ``e_cbm`` enters, so the gauge here is a single number. As in the SiC
pipeline this is a fitted gauge rather than a DFT band structure -- pass ``--gap`` to set
it, and treat transition levels from the resulting model as gauge-dependent until real
PBE edges for orthorhombic CsPbCl3 replace it.

    python extract_perovskite_dataset.py --out dataset_pbe
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write

HOST = "CsPbCl3"
DATASET_VERSION = "perovskite-v1"
PRISTINE = (0, 0, 0, 0)
IONISED = (0, 0, 0, 0)
NEUTRAL_VACANCY = (1, 0, 0, 0)


def classify(atoms) -> str:
    counts = Counter(atoms.get_chemical_symbols())
    chlorine, lead = counts.get("Cl", 0), counts.get("Pb", 0)
    if chlorine == 3 * lead:
        return "pristine"
    if chlorine == 3 * lead - 1:
        return "vacancy"
    return "unknown"


def geometry_key(atoms) -> tuple:
    """Identity by coordinates, not by the ``id`` field, which repeats within a file."""
    symbols = "".join(sorted(atoms.get_chemical_symbols()))
    return symbols, np.round(atoms.get_positions(), 4).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=here)
    parser.add_argument("--out", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--flavour", default="pbe", choices=["pbe", "soc"])
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="fitted gauge gap in eV; e_cbm = +gap/2, e_vbm = -gap/2. Required, and "
        "deliberately has no default: guessing it silently would put an unrecorded "
        "constant into every transition level the model produces",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.gap is None:
        raise SystemExit(
            "--gap is required. The PBE gap of orthorhombic CsPbCl3 is roughly 2.4 eV "
            "(PBE underestimates; the paper uses HSE+SOC for levels). Pass it explicitly "
            "so the value lands in dataset_summary.json and travels with the data."
        )

    charge_states = {
        "01_0": {"counts": NEUTRAL_VACANCY, "multiplicity": 2, "cell_charge": 0,
                 "tag": "q0"},
        "02_+1": {"counts": IONISED, "multiplicity": 1, "cell_charge": 1, "tag": "qp1"},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    written: dict = {"train": [], "valid": []}
    seen: dict = {"train": set(), "valid": set()}
    tally: dict = defaultdict(Counter)

    for directory, spec in charge_states.items():
        for split, name in (("train", "train"), ("valid", "val")):
            path = args.source / directory / f"{name}_{args.flavour}.xyz"
            if not path.is_file():
                print(f"  missing {path}, skipping")
                continue
            frames = ase_read(str(path), index=":")
            if args.max_frames:
                frames = frames[: args.max_frames]
            for atoms in frames:
                kind = classify(atoms)
                if kind == "unknown":
                    tally[directory]["unknown-composition"] += 1
                    continue

                key = geometry_key(atoms)
                if key in seen[split]:
                    # The pristine frames appear in both charge-state files with identical
                    # energies. Keeping both copies would double their weight in the base
                    # fit for no new information.
                    tally[directory]["duplicate-dropped"] += 1
                    continue
                seen[split].add(key)

                if kind == "pristine":
                    counts, multiplicity, cell_charge = PRISTINE, 1, 0
                    config_type = "ideal"
                else:
                    counts = spec["counts"]
                    multiplicity = spec["multiplicity"]
                    cell_charge = spec["cell_charge"]
                    config_type = f"vcl_{spec['tag']}"

                atoms.info.update(
                    {
                        "carrier_counts": np.asarray(counts, dtype=int),
                        "multiplicity": int(multiplicity),
                        "host": HOST,
                        "cell_charge": int(cell_charge),
                        "config_type": config_type,
                        "source_dir": directory,
                        "natoms": len(atoms),
                        # No pair partner exists anywhere in this dataset; recorded
                        # explicitly so the loader reports "unpaired" rather than looking
                        # like the field was forgotten.
                        "pair_id": "",
                        "e_cbm_cell": 0.5 * args.gap,
                        "e_vbm_cell": -0.5 * args.gap,
                    }
                )
                # REF_forces already exists in the source; REF_energy and REF_stress too.
                if "REF_energy" not in atoms.info and atoms.info.get("energy") is not None:
                    atoms.info["REF_energy"] = atoms.info["energy"]
                written[split].append(atoms)
                tally[directory][f"{split}:{config_type}"] += 1

    for split in ("train", "valid"):
        target = args.out / f"{split}.xyz"
        ase_write(str(target), written[split], format="extxyz")
        print(f"wrote {target}  ({len(written[split])} frames)")

    (args.out / "band_edges.json").write_text(
        json.dumps({HOST: {"e_cbm_cell": 0.5 * args.gap,
                           "e_vbm_cell": -0.5 * args.gap}}, indent=2)
    )
    summary = {
        "dataset_version": DATASET_VERSION,
        "host": HOST,
        "level_of_theory": args.flavour.upper(),
        "source": "Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025)",
        "counter_vectors": {
            "pristine": list(PRISTINE),
            "ionised_vacancy_qp1": list(IONISED),
            "neutral_vacancy_q0": list(NEUTRAL_VACANCY),
        },
        "multiplicities": {"pristine": 1, "vcl_qp1": 1, "vcl_q0": 2},
        "reference_state": (
            "V_Cl+ (ionised). A neutral doublet cannot be expressed at q = 0, since "
            "q = 0 forces M_s even; so the reference is the closed-shell charged state "
            "and the neutral vacancy carries one electron relative to it."
        ),
        "pairing": (
            "NONE. Geometry-keyed matching finds the two charge states share only "
            "pristine frames, with identical energies, i.e. copies. The V_Cl frames are "
            "disjoint because the paper trains separate models per charge state. "
            "delta_energy/delta_forces therefore have no support; supervision is "
            "L_base + L_tot."
        ),
        "band_edges": {
            "fitted_gauge_gap": args.gap,
            "note": "A fitted gauge, not PBE band edges. Only e_cbm enters, since the "
                    "counters here carry electrons and no holes.",
        },
        "counts": {k: dict(v) for k, v in tally.items()},
        "n_train": len(written["train"]),
        "n_valid": len(written["valid"]),
    }
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'dataset_summary.json'}")
    for directory, counter in tally.items():
        print(f"  {directory}: {dict(counter)}")


if __name__ == "__main__":
    main()
