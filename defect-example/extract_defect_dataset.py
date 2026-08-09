#!/usr/bin/env python3
"""Build a MACEDefect training set from the 4H-SiC divacancy ASE databases.

The three databases that accompany "Optical line shapes of color centers in solids
from classical autocorrelation functions" hold, per frame, a PBEsol total energy,
forces and stress:

* ``ideal.db``                 -- pristine 4H-SiC supercells (59 frames)
* ``defect_ground_state.db``   -- the neutral divacancy, electronic ground state (641)
* ``defect_excited_state.db``  -- the same defect in the first excited state (641)

Three things have to happen before MACEDefect can read them.

**Species relabelling is undone.** The published databases encode the electronic state
in the *chemical species* of the six defect neighbours -- ``Si(gs)`` is written as P,
``C(gs)`` as N, ``Si(ex)`` as S, ``C(ex)`` as O -- because the NEP model they were built
for has no other channel for it. MACEDefect's trunk is deliberately geometry-only and
the electronic state enters through the carrier counters, so P/N/S/O are mapped back to
Si/C. This is not cosmetic: the pair join asserts that paired frames share their atomic
numbers, so without the remap no ground/excited pair would ever match.

**The pairing is recovered.** 504 of the 641 frames in each defect database carry a
``run`` key and are the *same geometry* evaluated in both electronic states -- verified
here to bit-identical positions. Those become ``pair_id`` groups, which is what enables
the paired difference loss ``L_delta``, the highest-weighted term in ``DefectLoss``. The
remaining 137 frames per database are independent trajectories with no partner; the
ground-state ones supervise the base branch, the excited-state ones train through
``L_tot``.

**Carriers are counted from the closed-shell surface, not from the system's own ground
state.** The neutral divacancy ground state is a triplet, so it is *not* ``n = 0``: it is
``n = (1, 0, 0, 1)`` (``q = 0``, ``M_s = 2``), and the excitation, a further promotion
within the minority spin channel, is ``n = (1, 1, 0, 2)``. Only the pristine cells carry
``n = 0``. Consequences worth stating plainly: no defect geometry supplies a base-branch
label any more (there is no ``M_s = 0`` calculation at those geometries); and the paired
difference is now taken between two carrier states rather than against the reference
state, which is what ``n_ref`` in the model exists to evaluate.

``L_tot`` was switched off for a while because of that missing reference, which left total
forces at defect geometries supervised by nothing. It is on again: ``E_base`` there is a
*gauge* rather than an unobservable, and the correction cannot absorb a bulk-wide base
error because it is intensive (``sum_i alpha_i = 1``) while ``E_base`` is extensive. See
the forward plan's stage A5. The unpaired frames exist for that term.

**Band edges are fitted, because none were published.** MACEDefect references a frame by
``n_e E_CBM - n_h E_VBM``. Both defect states here have equal electron and hole counts --
one pair in the ground state, two in the excited state -- so they are referenced by one
and two gaps respectively and the paired difference keeps exactly one. The only
load-bearing number is therefore the effective gap, estimated from the data as

    gap = <E_ex - E_gs>_paired + margin

with ``<.> = 0.943 eV`` here and a small positive margin (the "slight shift"), so that
the defect excitation sits *inside* the gap and the referenced delta target is the
(negative) exciton binding energy plus its geometry-dependent fluctuation. The two edges
are written symmetrically about zero. **They are a fitted gauge, not PBEsol band edges**:
only their difference enters anything, and no absolute meaning should be read into either
value or into energies referenced with them.

Outputs, into ``--out-dir``:

    train.xyz             extended XYZ, one frame per DFT calculation
    valid.xyz             held out at the level of whole pair groups, never split pairs
    band_edges.json       the ``--band_edges_file`` table, keyed on host
    dataset_summary.json  the sampling decisions, counts and fitted constants

Example
-------
    python extract_defect_dataset.py --n-structures 200 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import ase.io
import numpy as np
from ase import Atoms
from ase.db import connect

# notes.md: the defect environment is encoded in the species of the six neighbours.
# Undo it -- MACEDefect learns the electronic state from the counters, not the labels.
SPECIES_REMAP = {"P": "Si", "N": "C", "S": "Si", "O": "C"}

# Carrier labelling, relative to the closed-shell M_s = 0 reference surface (plan 2.2).
#
# The neutral divacancy ground state is a *triplet*: two unpaired electrons occupying
# localised gap levels. Relative to the closed-shell surface that is one majority-spin
# electron promoted into a gap level plus the minority-spin hole it leaves behind, so
# n = (1, 0, 0, 1) with q = 0 and M_s = 2. Labelling it n = 0 -- as the first version of
# this script did -- asserts that the triplet *is* the reference surface, which is what
# broke the counter algebra: it made the reference state itself spin-polarised, and it
# put the entire ground-state binding energy outside the model's reach.
#
# The excitation promotes a further electron between two localised levels within the
# minority (beta) channel, adding one minority electron and one minority hole:
# n = (1, 1, 0, 2), again q = 0 and M_s = 2.
#
# Both are already canonical (M_s > 0 fixes the time-reversal gauge). Multiplicity is
# 2 S + 1 = M_s + 1 = 3 for both, and is now written to every frame -- the loader makes
# it mandatory for spin-polarised frames, and that assertion is what would have caught
# the original mislabelling.
# Bumped whenever the frame composition changes: metrics are not comparable across
# versions, since loss composition and epoch length both shift.
#   v1  original (mislabelled: triplet ground state written as n = 0)
#   v2  plan-convention labels + synthetic zero anchors
#   v3  anchors removed (see below); labels unchanged from v2
#   v4  unpaired defect frames restored, since L_tot is live again (plan A5.3)
DATASET_VERSION = "v4"

COUNTS_PRISTINE = (0, 0, 0, 0)
COUNTS_GROUND = (1, 0, 0, 1)
COUNTS_EXCITED = (1, 1, 0, 2)
MULTIPLICITY_PRISTINE = 1
MULTIPLICITY_DEFECT = 3

# Synthetic zero-anchor frames were tried here and are REJECTED -- do not re-add them.
# Two reasons. Only one frame in ideal.db is a relaxed ideal lattice (the 8-atom cell), so
# in practice every anchor would sit on a thermally displaced cell; and there the zero
# target is simply false -- a strained pristine cell's band edge genuinely shifts by the
# deformation potential, which is a real effect plan 3.2 wants u to carry, not an error to
# be pinned away. The level-mode freedom they were meant to close is handled instead by
# the optional gauge penalty (--defect_gauge_weight), which constrains a cell-level pooled
# scalar rather than asserting a per-frame energy nobody computed.

# Paired frames must be the same geometry; the loader's own tolerance is 1e-8 A.
POSITION_TOLERANCE = 1e-10


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


class Frame:
    """One DFT calculation, with the provenance needed to stratify and to pair."""

    def __init__(self, atoms: Atoms, meta: Dict[str, object]) -> None:
        self.atoms = atoms
        self.meta = meta

    @property
    def stratum(self) -> str:
        return f"{self.meta['pool']}|{self.meta['sampling']}|{self.meta['natoms']}"


class Group:
    """The unit of sampling and of the train/valid split.

    A ground/excited pair is one group of two frames: splitting it would both destroy
    the delta target and leak the geometry across the split.
    """

    def __init__(self, frames: Sequence[Frame], stratum: str, pool: str) -> None:
        self.frames = list(frames)
        self.stratum = stratum
        self.pool = pool

    def __len__(self) -> int:
        return len(self.frames)


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def remap_species(atoms: Atoms) -> Atoms:
    symbols = [SPECIES_REMAP.get(symbol, symbol) for symbol in atoms.get_chemical_symbols()]
    atoms = atoms.copy()
    atoms.set_chemical_symbols(symbols)
    return atoms


def read_database(path: Path) -> List[Dict[str, object]]:
    """Read an ASE sqlite database into plain dicts, with the species remap applied."""
    rows = []
    with connect(str(path)) as database:
        for row in database.select():
            atoms = remap_species(row.toatoms())
            # toatoms() attaches a SinglePointCalculator; drop it so the extxyz writer
            # emits only the explicit REF_* keys and never a duplicate 'energy' column.
            atoms.calc = None
            key_values = dict(row.key_value_pairs)
            rows.append(
                {
                    "atoms": atoms,
                    "energy": float(row.energy),
                    "forces": np.asarray(row.forces, dtype=float),
                    "stress": (
                        np.asarray(row.stress, dtype=float)
                        if "stress" in row
                        else None
                    ),
                    "id": int(row.id),
                    "run": key_values.get("run"),
                    "sampling": key_values.get("sampling", key_values.get("category")),
                    "natoms": len(atoms),
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# Pool construction
# --------------------------------------------------------------------------------------


def build_pairs(
    ground: List[Dict[str, object]], excited: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Match ground- and excited-state frames that share a run, in database order.

    The geometry equality is asserted rather than assumed: it is the whole content of
    the pairing, and a silent mismatch would train a charge-state difference between
    two different structures.
    """
    by_run_ground = collections.defaultdict(list)
    by_run_excited = collections.defaultdict(list)
    for row in ground:
        if row["run"] is not None:
            by_run_ground[row["run"]].append(row)
    for row in excited:
        if row["run"] is not None:
            by_run_excited[row["run"]].append(row)

    if set(by_run_ground) != set(by_run_excited):
        missing = set(by_run_ground) ^ set(by_run_excited)
        raise ValueError(f"runs present in only one database: {sorted(missing)}")

    pairs = []
    for run in sorted(by_run_ground):
        gs_rows, ex_rows = by_run_ground[run], by_run_excited[run]
        if len(gs_rows) != len(ex_rows):
            raise ValueError(
                f"run '{run}' has {len(gs_rows)} ground-state frames but "
                f"{len(ex_rows)} excited-state frames"
            )
        for index, (gs_row, ex_row) in enumerate(zip(gs_rows, ex_rows)):
            gs_atoms, ex_atoms = gs_row["atoms"], ex_row["atoms"]
            if not np.array_equal(gs_atoms.numbers, ex_atoms.numbers):
                raise ValueError(f"run '{run}' frame {index}: atomic numbers differ")
            position_error = np.abs(gs_atoms.positions - ex_atoms.positions).max()
            cell_error = np.abs(gs_atoms.cell.array - ex_atoms.cell.array).max()
            if max(position_error, cell_error) > POSITION_TOLERANCE:
                raise ValueError(
                    f"run '{run}' frame {index}: geometries differ by "
                    f"{position_error:.2e} A (cell {cell_error:.2e} A); the frames are "
                    "not the same structure and cannot be paired"
                )
            pairs.append({"run": run, "index": index, "ground": gs_row, "excited": ex_row})
    return pairs


def sanitise(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def make_frame(
    row: Dict[str, object],
    counts: Sequence[int],
    pool: str,
    config_type: str,
    source: str,
    multiplicity: int,
    pair_id: Optional[str] = None,
    positions_from: Optional[Atoms] = None,
    host: str = "4H-SiC",
    with_stress: bool = True,
) -> Frame:
    """Attach the MACE info/arrays keys to one frame."""
    atoms = row["atoms"].copy()
    atoms.calc = None
    if positions_from is not None:
        # Write both members of a pair from one geometry, so the join's exact position
        # comparison cannot be defeated by extxyz's 8-decimal output.
        atoms.set_positions(positions_from.get_positions())
        atoms.set_cell(positions_from.get_cell())

    atoms.info["REF_energy"] = row["energy"]
    atoms.arrays["REF_forces"] = np.asarray(row["forces"], dtype=float)
    if with_stress and row["stress"] is not None:
        atoms.info["REF_stress"] = np.asarray(row["stress"], dtype=float)
    atoms.info["carrier_counts"] = np.asarray(counts, dtype=int)
    # Mandatory for spin-polarised frames; the loader asserts M_s == multiplicity - 1.
    atoms.info["multiplicity"] = int(multiplicity)
    atoms.info["host"] = host
    if pair_id is not None:
        atoms.info["pair_id"] = pair_id
    atoms.info["config_type"] = config_type
    atoms.info["source_db"] = source
    atoms.info["source_id"] = int(row["id"])
    atoms.info["sampling"] = str(row["sampling"])

    meta = {
        "pool": pool,
        "sampling": str(row["sampling"]),
        "natoms": int(row["natoms"]),
        "config_type": config_type,
    }
    return Frame(atoms, meta)


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def largest_remainder_quotas(sizes: Dict[str, int], target: int) -> Dict[str, int]:
    """Proportional allocation of ``target`` over strata, ties broken deterministically."""
    total = sum(sizes.values())
    if total == 0 or target <= 0:
        return {name: 0 for name in sizes}
    exact = {name: target * size / total for name, size in sizes.items()}
    quotas = {name: int(math.floor(value)) for name, value in exact.items()}
    shortfall = target - sum(quotas.values())
    order = sorted(sizes, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:shortfall]:
        quotas[name] += 1
    return quotas


def stratified_sample(
    items: Sequence, target: int, key: Callable[[object], str], rng: random.Random
) -> List:
    """Random sampling, allocated proportionally across strata.

    Still random -- but the paired pool spans five sampling protocols whose mean
    vertical excitation differs by 0.14 eV (the Stokes shift), and ``config_coord`` is
    both the smallest stratum and the one with the widest spread. Plain global sampling
    at n ~ 70 would leave that balance to chance.
    """
    if target <= 0:
        return []
    if target >= len(items):
        return list(items)

    strata: Dict[str, List] = collections.defaultdict(list)
    for item in items:
        strata[key(item)].append(item)
    for members in strata.values():
        rng.shuffle(members)

    sizes = {name: len(members) for name, members in strata.items()}
    quotas = largest_remainder_quotas(sizes, target)
    # Clamp to availability, then hand the deficit back to strata with spare capacity.
    quotas = {name: min(count, sizes[name]) for name, count in quotas.items()}
    deficit = target - sum(quotas.values())
    while deficit > 0:
        spare = [name for name in sorted(sizes) if quotas[name] < sizes[name]]
        if not spare:
            break
        for name in spare:
            if deficit == 0:
                break
            quotas[name] += 1
            deficit -= 1

    selected = []
    for name in sorted(strata):
        selected.extend(strata[name][: quotas[name]])
    rng.shuffle(selected)
    return selected


def report_e0_fit(
    frames: Sequence[Frame], e_cbm_cell: float, e_vbm_cell: float
) -> Dict[str, object]:
    """Reproduce ``--E0s average`` for information.

    ``mace.data.compute_average_E0s`` fits the isolated-atom energies by least squares on
    composition over **every training frame**, not over the ``n = 0`` subset, and it does
    so *after* band-edge referencing has been applied -- so the referencing constant is
    subtracted here too, or the prediction would not match the training log.

    Every structure here -- pristine and divacancy alike -- has equal Si and C counts, so
    the design matrix is rank 1 and ``lstsq`` returns the minimum-norm solution: the two
    E0s come out identical. That is a gauge choice, not two independently determined
    atomic energies. It is harmless at fixed stoichiometry, but it will look odd in the
    training log, so it is reported here.
    """
    if not frames:
        return {}
    species = sorted({int(z) for frame in frames for z in frame.atoms.numbers})
    design = np.array(
        [[int(np.count_nonzero(frame.atoms.numbers == z)) for z in species] for frame in frames],
        dtype=float,
    )
    targets = np.array(
        [
            frame.atoms.info["REF_energy"]
            - referencing_constant(
                frame.atoms.info["carrier_counts"], e_cbm_cell, e_vbm_cell
            )
            for frame in frames
        ],
        dtype=float,
    )
    solution, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    return {
        "species": species,
        "e0": [float(value) for value in solution],
        "rank": int(rank),
        "n_frames": len(frames),
    }


def referencing_constant(
    counts: Sequence[int], e_cbm_cell: float, e_vbm_cell: float
) -> float:
    """What the loader subtracts from a raw energy (mace.data.defects)."""
    counts = np.asarray(counts, dtype=int)
    electrons = int(counts[0] + counts[1])
    holes = int(counts[2] + counts[3])
    return electrons * e_cbm_cell - holes * e_vbm_cell


def split_groups(
    groups: Sequence[Group], valid_fraction: float, rng: random.Random
) -> tuple:
    """Hold out whole groups, pool by pool, so pairs stay intact and unsplit."""
    train: List[Group] = []
    valid: List[Group] = []
    by_pool: Dict[str, List[Group]] = collections.defaultdict(list)
    for group in groups:
        by_pool[group.pool].append(group)
    for pool in sorted(by_pool):
        members = list(by_pool[pool])
        rng.shuffle(members)
        n_valid = int(round(valid_fraction * len(members)))
        if valid_fraction > 0 and n_valid == 0 and len(members) > 1:
            n_valid = 1
        valid.extend(members[:n_valid])
        train.extend(members[n_valid:])
    return train, valid


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here, help="directory holding the .db files")
    parser.add_argument("--out-dir", type=Path, default=here / "dataset", help="output directory")
    parser.add_argument(
        "--n-structures",
        type=int,
        default=200,
        help="approximate total number of frames written, train and valid combined",
    )
    parser.add_argument(
        "--frac-paired",
        type=float,
        default=0.70,
        help="share of the budget spent on ground/excited pairs (two frames each); "
        "these carry the delta targets and are the most informative data available",
    )
    parser.add_argument(
        "--frac-unpaired",
        type=float,
        default=0.15,
        help="share spent on unpaired defect frames, split evenly between ground state "
        "(base branch) and excited state (L_tot)",
    )
    parser.add_argument(
        "--frac-ideal",
        type=float,
        default=0.15,
        help="share spent on pristine 4H-SiC supercells, which anchor the base branch "
        "and the E0 fit",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.1, help="held-out share of groups")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--host", type=str, default="4H-SiC", help="host label written to every frame")
    parser.add_argument(
        "--gap-margin",
        type=float,
        default=0.2,
        help="eV added to the mean vertical excitation energy to place the effective "
        "CBM-VBM gap above it; equals the resulting mean exciton binding energy",
    )
    parser.add_argument(
        "--band-gap",
        type=float,
        default=None,
        help="effective E_CBM - E_VBM in eV; overrides the fit from the data entirely",
    )
    parser.add_argument(
        "--min-natoms",
        type=int,
        default=0,
        help="drop frames smaller than this (the 8-atom primitive cell in ideal.db)",
    )
    parser.add_argument(
        "--max-natoms",
        type=int,
        default=0,
        help="drop frames larger than this (0 = no limit). Cost per frame grows with the "
        "atom count, so capping at 290 keeps only the smallest defect cells (286) and "
        "their pristine partners (288) -- the fast configuration for debugging. The "
        "8-atom pristine cell is always kept, since it is the only exact anchor",
    )
    parser.add_argument(
        "--defect-natoms",
        type=int,
        default=0,
        help="keep only defect frames with exactly this many atoms (0 = no filter). "
        "Applies to the ground/excited databases only, so the pristine pool can be held "
        "fixed while the defect cell size is varied -- which is what a cell-size "
        "comparison needs, since removing the pristine frames removes all n = 0 labels "
        "and with them every base-branch loss term",
    )
    parser.add_argument(
        "--ideal-natoms",
        type=int,
        default=0,
        help="keep only pristine frames with exactly this many atoms (0 = no filter)",
    )
    parser.add_argument("--no-stratify", action="store_true", help="sample uniformly at random instead")
    parser.add_argument("--no-stress", action="store_true", help="omit REF_stress from the output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading databases from {args.data_dir}")
    ideal = read_database(args.data_dir / "ideal.db")
    ground = read_database(args.data_dir / "defect_ground_state.db")
    excited = read_database(args.data_dir / "defect_excited_state.db")
    print(f"  ideal {len(ideal)}, ground state {len(ground)}, excited state {len(excited)}")

    if args.min_natoms > 0:
        keep = lambda rows: [r for r in rows if r["natoms"] >= args.min_natoms]  # noqa: E731
        ideal, ground, excited = keep(ideal), keep(ground), keep(excited)
    if args.max_natoms > 0:
        # The 8-atom pristine cell is exempt: it is the only frame in the whole set at a
        # relaxed ideal lattice, so it is the only anchor whose zero target is exact.
        cap = lambda rows: [  # noqa: E731
            r for r in rows if r["natoms"] <= args.max_natoms or r["natoms"] == 8
        ]
        ideal, ground, excited = cap(ideal), cap(ground), cap(excited)
        print(
            f"  capped at {args.max_natoms} atoms: ideal {len(ideal)}, "
            f"ground {len(ground)}, excited {len(excited)}"
        )
    if args.defect_natoms > 0:
        ground = [r for r in ground if r["natoms"] == args.defect_natoms]
        excited = [r for r in excited if r["natoms"] == args.defect_natoms]
        print(
            f"  defect cells fixed at {args.defect_natoms} atoms: "
            f"ground {len(ground)}, excited {len(excited)}"
        )
    if args.ideal_natoms > 0:
        ideal = [r for r in ideal if r["natoms"] == args.ideal_natoms]
        print(f"  pristine cells fixed at {args.ideal_natoms} atoms: ideal {len(ideal)}")

    pairs = build_pairs(ground, excited)
    paired_ground_ids = {pair["ground"]["id"] for pair in pairs}
    paired_excited_ids = {pair["excited"]["id"] for pair in pairs}
    unpaired_ground = [row for row in ground if row["id"] not in paired_ground_ids]
    unpaired_excited = [row for row in excited if row["id"] not in paired_excited_ids]
    print(
        f"  {len(pairs)} ground/excited pairs (geometry verified), "
        f"{len(unpaired_ground)} unpaired ground, {len(unpaired_excited)} unpaired excited"
    )

    # ---- band-edge gauge -------------------------------------------------------------
    vertical = np.array(
        [pair["excited"]["energy"] - pair["ground"]["energy"] for pair in pairs]
    )
    mean_vertical = float(vertical.mean())
    band_gap = args.band_gap if args.band_gap is not None else mean_vertical + args.gap_margin
    e_cbm_cell, e_vbm_cell = 0.5 * band_gap, -0.5 * band_gap
    print(
        f"\nVertical excitation over all {len(vertical)} pairs: "
        f"mean {mean_vertical:.4f} eV, std {vertical.std():.4f}, "
        f"range [{vertical.min():.4f}, {vertical.max():.4f}]"
    )
    by_sampling = collections.defaultdict(list)
    for pair, value in zip(pairs, vertical):
        by_sampling[pair["ground"]["sampling"]].append(value)
    for name in sorted(by_sampling):
        values = np.array(by_sampling[name])
        print(f"    {name:<14s} n={len(values):<4d} mean {values.mean():.4f}  std {values.std():.4f}")
    print(
        f"  effective gap {band_gap:.4f} eV "
        f"(= mean + margin {args.gap_margin:.3f})"
        if args.band_gap is None
        else f"  effective gap {band_gap:.4f} eV (given)"
    )
    print(f"  gauge edges: e_cbm_cell {e_cbm_cell:+.4f}, e_vbm_cell {e_vbm_cell:+.4f} eV")
    print("  (a fitted gauge; only their difference enters the referencing)")

    # ---- budget ----------------------------------------------------------------------
    weights = np.array([args.frac_paired, args.frac_unpaired, args.frac_ideal], dtype=float)
    if weights.sum() <= 0:
        raise SystemExit("at least one of --frac-paired/--frac-unpaired/--frac-ideal must be > 0")
    requested = weights.copy()
    weights = weights / weights.sum()
    n_pairs = int(round(args.n_structures * weights[0] / 2))
    n_unpaired = int(round(args.n_structures * weights[1]))
    n_ideal = max(0, args.n_structures - 2 * n_pairs - n_unpaired)
    n_unpaired_ground = n_unpaired // 2
    n_unpaired_excited = n_unpaired - n_unpaired_ground

    print(f"\nBudget for {args.n_structures} frames")
    if abs(requested.sum() - 1.0) > 1e-9:
        # Passing one --frac-* without the others silently rescales the other two, which
        # is not what the flag name suggests. Say so rather than let it pass unnoticed.
        print(
            f"  --frac-* were given as {requested.tolist()} (sum {requested.sum():.3f}) "
            f"and renormalised to {[round(float(w), 3) for w in weights]}; "
            "pass all three to control the split exactly"
        )
    print(
        f"  paired {2 * n_pairs} frames ({n_pairs} pairs), unpaired {n_unpaired} "
        f"({n_unpaired_ground} ground + {n_unpaired_excited} excited), ideal {n_ideal}"
    )
    for label, wanted, available in (
        ("pairs", n_pairs, len(pairs)),
        ("unpaired ground", n_unpaired_ground, len(unpaired_ground)),
        ("unpaired excited", n_unpaired_excited, len(unpaired_excited)),
        ("ideal", n_ideal, len(ideal)),
    ):
        if wanted > available:
            print(f"  CAPPED: asked {wanted} {label} but only {available} exist")

    stratify = not args.no_stratify

    def sample(items: Sequence, target: int, key: Callable[[object], str]) -> List:
        if not stratify:
            target = min(target, len(items))
            return rng.sample(list(items), target)
        return stratified_sample(items, target, key, rng)

    pair_key = lambda pair: f"{pair['ground']['sampling']}|{pair['ground']['natoms']}"  # noqa: E731
    row_key = lambda row: f"{row['sampling']}|{row['natoms']}"  # noqa: E731

    chosen_pairs = sample(pairs, n_pairs, pair_key)
    chosen_unpaired_ground = sample(unpaired_ground, n_unpaired_ground, row_key)
    chosen_unpaired_excited = sample(unpaired_excited, n_unpaired_excited, row_key)
    chosen_ideal = sample(ideal, n_ideal, row_key)

    # ---- frames ----------------------------------------------------------------------
    groups: List[Group] = []
    with_stress = not args.no_stress

    for pair in chosen_pairs:
        pair_id = f"{sanitise(pair['run'])}_{pair['index']:03d}"
        reference = pair["ground"]["atoms"]
        gs_frame = make_frame(
            pair["ground"], COUNTS_GROUND, "paired", "paired_gs", "ground_state",
            MULTIPLICITY_DEFECT,
            pair_id=pair_id, positions_from=reference, host=args.host, with_stress=with_stress,
        )
        ex_frame = make_frame(
            pair["excited"], COUNTS_EXCITED, "paired", "paired_ex", "excited_state",
            MULTIPLICITY_DEFECT,
            pair_id=pair_id, positions_from=reference, host=args.host, with_stress=with_stress,
        )
        groups.append(Group([gs_frame, ex_frame], gs_frame.stratum, "paired"))

    for row in chosen_unpaired_ground:
        frame = make_frame(
            row, COUNTS_GROUND, "unpaired_gs", "unpaired_gs", "ground_state",
            MULTIPLICITY_DEFECT, host=args.host, with_stress=with_stress,
        )
        groups.append(Group([frame], frame.stratum, "unpaired_gs"))

    for row in chosen_unpaired_excited:
        frame = make_frame(
            row, COUNTS_EXCITED, "unpaired_ex", "unpaired_ex", "excited_state",
            MULTIPLICITY_DEFECT, host=args.host, with_stress=with_stress,
        )
        groups.append(Group([frame], frame.stratum, "unpaired_ex"))

    for row in chosen_ideal:
        # Pristine frames are ungrouped: they are the only n = 0 labels, so they train the
        # base branch, and they supply the geometries the optional gauge penalty uses.
        frame = make_frame(
            row, COUNTS_PRISTINE, "ideal", "ideal", "ideal",
            MULTIPLICITY_PRISTINE,
            host=args.host, with_stress=with_stress,
        )
        groups.append(Group([frame], frame.stratum, "ideal"))

    train_groups, valid_groups = split_groups(groups, args.valid_fraction, rng)
    train_frames = [frame for group in train_groups for frame in group.frames]
    valid_frames = [frame for group in valid_groups for frame in group.frames]

    train_path = args.out_dir / "train.xyz"
    valid_path = args.out_dir / "valid.xyz"
    ase.io.write(str(train_path), [frame.atoms for frame in train_frames], format="extxyz")
    ase.io.write(str(valid_path), [frame.atoms for frame in valid_frames], format="extxyz")

    edges_path = args.out_dir / "band_edges.json"
    edges_path.write_text(
        json.dumps(
            {args.host: {"e_cbm_cell": e_cbm_cell, "e_vbm_cell": e_vbm_cell}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ---- report ----------------------------------------------------------------------
    def counts_by(frames: Iterable[Frame], field: str) -> Dict[str, int]:
        counter = collections.Counter(str(frame.meta[field]) for frame in frames)
        return dict(sorted(counter.items()))

    print(f"\nWrote {len(train_frames)} train and {len(valid_frames)} valid frames "
          f"({len(train_frames) + len(valid_frames)} total) to {args.out_dir}")
    print("  train by config_type:", counts_by(train_frames, "config_type"))
    print("  valid by config_type:", counts_by(valid_frames, "config_type"))
    print("  train by supercell size:", counts_by(train_frames, "natoms"))
    print("  train by sampling protocol:", counts_by(train_frames, "sampling"))

    delta = np.array(
        [
            group.frames[1].atoms.info["REF_energy"]
            - group.frames[0].atoms.info["REF_energy"]
            - band_gap
            for group in groups
            if group.pool == "paired"
        ]
    )
    if delta.size:
        print(
            f"  referenced delta targets: mean {delta.mean():+.4f} eV, "
            f"std {delta.std():.4f}, range [{delta.min():+.4f}, {delta.max():+.4f}]"
        )

    e0_fit = report_e0_fit(train_frames, e_cbm_cell, e_vbm_cell)
    if e0_fit:
        pairs_of = ", ".join(
            f"Z={z}: {value:+.6f} eV" for z, value in zip(e0_fit["species"], e0_fit["e0"])
        )
        print(f"\n  --E0s average would fit ({e0_fit['n_frames']} train frames): {pairs_of}")
        if e0_fit["rank"] < len(e0_fit["species"]):
            print(
                f"    rank {e0_fit['rank']} of {len(e0_fit['species'])}: Si and C counts are "
                "equal in every structure, so the two E0s are a gauge, not independent values"
            )

    summary = {
        "seed": args.seed,
        "n_structures_requested": args.n_structures,
        "n_train": len(train_frames),
        "n_valid": len(valid_frames),
        "fractions": {
            "paired": weights[0],
            "unpaired": weights[1],
            "ideal": weights[2],
        },
        "stratified": stratify,
        "host": args.host,
        "dataset_version": DATASET_VERSION,
        "counter_vectors": {
            "pristine": list(COUNTS_PRISTINE),
            "ground": list(COUNTS_GROUND),
            "excited": list(COUNTS_EXCITED),
        },
        "multiplicities": {
            "pristine": MULTIPLICITY_PRISTINE,
            "defect": MULTIPLICITY_DEFECT,
        },
        "anchors": {
            "enabled": False,
            "note": (
                "Synthetic zero-anchor frames were tried and removed. Only one frame in "
                "ideal.db is a relaxed ideal lattice, so anchors would in practice sit on "
                "thermally displaced cells, where the zero target is false: a strained "
                "pristine cell's band edge genuinely shifts by the deformation potential. "
                "Use --defect_gauge_weight instead if the level mode needs closing."
            ),
        },
        "species_remap": SPECIES_REMAP,
        "band_edges": {
            "e_cbm_cell": e_cbm_cell,
            "e_vbm_cell": e_vbm_cell,
            "effective_gap": band_gap,
            "mean_vertical_excitation": mean_vertical,
            "gap_margin": args.gap_margin if args.band_gap is None else None,
            "note": (
                "A fitted gauge, not PBEsol band edges. Only e_cbm_cell - e_vbm_cell "
                "enters the referencing of the defect frames, since the ground state "
                "(1,0,0,1) and the excited state (1,1,0,2) each have equal electron and "
                "hole counts -- they are referenced by one and two gaps respectively, so "
                "the paired difference keeps exactly one. The split of the gap between "
                "the two edges is therefore a pure gauge with no observable consequence. "
                "The GAP ITSELF is not a gauge: it is fitted from the mean vertical "
                "excitation rather than computed as E(N+-1) - E(N) per supercell, so it "
                "is a genuine error source across cell sizes. See HANDOFF.md."
            ),
        },
        "pool_sizes_available": {
            "pairs": len(pairs),
            "unpaired_ground": len(unpaired_ground),
            "unpaired_excited": len(unpaired_excited),
            "ideal": len(ideal),
        },
        "pool_sizes_sampled": {
            "pairs": len(chosen_pairs),
            "unpaired_ground": len(chosen_unpaired_ground),
            "unpaired_excited": len(chosen_unpaired_excited),
            "ideal": len(chosen_ideal),
        },
        "train_by_config_type": counts_by(train_frames, "config_type"),
        "valid_by_config_type": counts_by(valid_frames, "config_type"),
        "e0_least_squares_fit": e0_fit,
    }
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
