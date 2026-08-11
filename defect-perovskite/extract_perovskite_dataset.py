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

COUNTER CONVENTION: THE REFERENCE MUST BE NEUTRAL
--------------------------------------------------
Measured cell magnetisation: V_Cl neutral |m| = 1.004 (a doublet, one unpaired electron),
V_Cl charged 0.010 and pristine 0.004 (both closed shell).

    pristine   n = (0,0,0,0)   M_s_ref 0     multiplicity 1   cell q  0
    V_Cl0      n = (0,0,0,0)   M_s_ref 1/2   multiplicity 2   cell q  0   <- reference
    V_Cl+      n = (0,0,1,0)   dM_s = -1/2   multiplicity 1   cell q +1

``h_maj``, not ``h_min``: removing the unpaired **majority** electron takes M_s from 1/2 to
0, which is what the measured 0.010 for V_Cl+ confirms. Under this assignment the counter
charge equals the absolute cell charge on every frame, which is asserted here.

Why not the closed-shell reference (V_Cl+), which is what a naive reading of the parity rule
suggests. ``q = 0`` forces ``n0+n1 == n2+n3``, hence even ``M_s`` and odd multiplicity, so a
neutral doublet is inexpressible at ``q = 0`` *against a closed-shell reference*. The
resolution is to move the reference, not to bend the counters: ``M_s_ref`` becomes
composition-dependent. Referencing against V_Cl+ instead would cause three measurable
failures --

* **the long-range branch inverts.** ``sum_i q_i = a q`` uses the *counter* charge, so the
  physically neutral V_Cl0 would get q = -1 and the charged V_Cl+ q = 0. With
  ``a^2 = 1/eps_inf ~ 0.22`` and L ~ 11 A that misplaces a Madelung term of order 0.4 eV,
  applied to the wrong state and omitted from the right one; in ``eps_opt`` the two errors
  add rather than cancel;
* **E_base is asked to represent a non-local quantity.** A charged periodic cell carries
  ``-q^2 alpha_M / 2 eps L``, which depends on cell size rather than on local environments,
  so a sum of local atomic energies cannot express it and exact base extensivity must break;
* **the fit resolves the tension the wrong way.** Nothing pins ``a``, and the cheapest way
  to remove a spurious size-dependent term is ``a -> 0`` -- disabling the long-range branch
  on the first dataset able to validate it.

A benefit of the polarised reference: ``h_maj`` and ``h_min`` are now physically distinct
states with different multiplicities, so the time-reversal canonicalisation that left
``e_min`` unidentifiable in 4H-SiC does not apply. The magnetisation identifies the channel
directly. Canonicalisation is therefore disabled whenever ``m_s_ref_doubled != 0``.

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
# Neutral reference, per composition. `m_s_ref_doubled` is 2*M_s of that reference: 0 for
# the even-electron pristine cell, 1 for the odd-electron vacancy whose neutral state is a
# doublet (measured |m| = 1.004).
PRISTINE = (0, 0, 0, 0)
NEUTRAL_VACANCY = (0, 0, 0, 0)     # the reference for this composition
IONISED = (0, 0, 1, 0)             # h_maj: removing the unpaired MAJORITY electron


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
    # 2*M_s of the neutral reference, by composition.
    M_S_REF = {"pristine": 0, "vacancy": 1}

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
                m_s_ref = M_S_REF[kind]

                # Hard error, not a warning: under a neutral reference the counter charge
                # IS the absolute cell charge, and the original labelling of this dataset
                # violated exactly this.
                implied = int(counts[2] + counts[3] - counts[0] - counts[1])
                if implied != cell_charge:
                    raise SystemExit(
                        f"counter {counts} implies charge {implied} but the cell carries "
                        f"{cell_charge}"
                    )
                expected_multiplicity = (
                    m_s_ref + (counts[0] - counts[2]) - (counts[1] - counts[3]) + 1
                )
                if expected_multiplicity != multiplicity:
                    raise SystemExit(
                        f"counter {counts} with reference 2*M_s = {m_s_ref} implies "
                        f"multiplicity {expected_multiplicity}, not {multiplicity}"
                    )

                atoms.info.update(
                    {
                        "carrier_counts": np.asarray(counts, dtype=int),
                        "multiplicity": int(multiplicity),
                        "host": HOST,
                        "cell_charge": int(cell_charge),
                        "m_s_ref_doubled": int(m_s_ref),
                        "config_type": config_type,
                        "source_dir": directory,
                        "natoms": len(atoms),
                        # No pair partner exists anywhere in this dataset. The key is
                        # OMITTED rather than set empty: an empty extxyz info value does
                        # not round-trip -- it merges with the following key on read, and
                        # every frame then arrives with pair_id='e_cbm_cell=1.2'.
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
            "neutral_vacancy_q0": list(NEUTRAL_VACANCY),
            "ionised_vacancy_qp1": list(IONISED),
        },
        "multiplicities": {"pristine": 1, "vcl_q0": 2, "vcl_qp1": 1},
        "m_s_ref_doubled": {"pristine": 0, "vacancy": 1},
        "q_ref": 0,
        "reference_state": (
            "NEUTRAL, per composition. 'base' means the neutral surface for that "
            "composition, not a closed-shell one: pristine is a closed-shell singlet "
            "(2*M_s_ref = 0) while V_Cl0 is a doublet (2*M_s_ref = 1). The counter charge "
            "equals the absolute cell charge on every frame, which is asserted at "
            "extraction."
        ),
        "parity_argument": (
            "q = 0 forces n0+n1 == n2+n3, hence M_s even and multiplicity odd, so a "
            "neutral doublet is inexpressible at q = 0 against a closed-shell reference. "
            "The reference moves rather than the counters bending: M_s_ref is "
            "composition-dependent. Referencing against the charged V_Cl+ instead would "
            "invert the long-range branch (sum_i q_i = a q uses the counter charge), ask "
            "E_base to represent a Madelung term that no sum of local energies can "
            "express, and let the fit drive a -> 0 to remove the resulting size drift."
        ),
        "canonicalisation": (
            "Disabled where m_s_ref_doubled != 0: time reversal flips the reference spin "
            "too, so (0,0,1,0) and (0,0,0,1) are distinct states with different "
            "multiplicities. h_maj vs h_min is identified by the measured magnetisation."
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
