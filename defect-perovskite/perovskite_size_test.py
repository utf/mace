#!/usr/bin/env python3
"""Finite-size behaviour of V_Cl in CsPbCl3: vertical level, transition level, Delta q.

The SiC script cannot be reused: that defect is a neutral divacancy with an excited state,
this one is a single vacancy in two *charge* states, so the quantities differ.

Per cell size, two relaxations (q = 0 and q = +1) and two single points (each state at the
other's geometry):

    eps_opt      = E+(R0) - E0(R0)              vertical, the paper's optical level
    eps(+/0)     = E0(R0) - E+(R+)              thermodynamic transition level
    E_rel        = E(V_Cl) - E(pristine)        formation energy up to the Cl reservoir
    Delta q      = sqrt(sum_i m_i |R+ - R0|^2)  configuration coordinate

**Energy scale.** `--energy-scale referenced` is the default *here*, unlike the SiC script,
and the reason is worth stating. The referencing constant is
``n_e e_cbm - n_h e_vbm``; the hole counter of V_Cl+ makes it ``-e_vbm``, so working on the
referenced scale measures the charged state *from the valence band edge* -- which is
precisely the reference a transition level is quoted against. ``eps(+/0)`` on this scale is
therefore already relative to the VBM. It is relative to the **fitted gauge** VBM, not a DFT
one, so it is only as good as ``--gap`` was at extraction; both scales are reported so the
constant is visible rather than buried.

**Charged cells have a real finite-size tail.** Unlike the neutral SiC divacancy, V_Cl+
carries a monopole, so its periodic images interact as ``q^2 alpha_M / 2 eps L``, decaying
as ``1/L = N^(-1/3)``. That is genuine physics and a *short-range model cannot represent
it*: the model's receptive field is finite, so it will predict a size-independent answer
where the truth converges as 1/L. Deviation here is therefore expected, and is the point of
the measurement rather than a defect in it.

    python perovskite_size_test.py --model ~/runs/perov_nolr_s1/perov_nolr_s1.model
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

CUTOFF = 5.0
# Neutral reference per composition, as written by extract_perovskite_dataset.py.
PRISTINE = {"counts": [0, 0, 0, 0], "multiplicity": 1, "m_s_ref": 0, "charge": 0}
VACANCY_Q0 = {"counts": [0, 0, 0, 0], "multiplicity": 2, "m_s_ref": 1, "charge": 0}
VACANCY_QP1 = {"counts": [0, 0, 1, 0], "multiplicity": 1, "m_s_ref": 1, "charge": 1}


def parse_repeat(text: str) -> tuple:
    parts = [int(v) for v in text.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'n1,n2,n3', got {text!r}")
    return tuple(parts)


def find_pristine(data_dir: Path):
    """Smallest pristine frame: the 80-atom Cs16Pb16Cl48 cell."""
    from ase.io import read

    for name in ("valid.xyz", "train.xyz"):
        path = data_dir / name
        if not path.is_file():
            continue
        frames = [
            atoms for atoms in read(str(path), ":")
            if str(atoms.info.get("config_type", "")).startswith("ideal")
        ]
        if frames:
            return min(frames, key=len)
    raise SystemExit(f"no pristine frame under {data_dir}")


def set_state(atoms, state: dict, host: str = "CsPbCl3") -> None:
    atoms.info["carrier_counts"] = list(state["counts"])
    atoms.info["multiplicity"] = int(state["multiplicity"])
    atoms.info["m_s_ref_doubled"] = int(state["m_s_ref"])
    atoms.info["cell_charge"] = int(state["charge"])
    atoms.info["host"] = host


def classify_sites(pristine):
    """Group the Cl atoms of the primitive cell into crystallographic sites.

    Orthorhombic CsPbCl3 has inequivalent Cl positions -- axial and equatorial in the
    tilted octahedra -- and their vacancies are genuinely different defects, with different
    Pb-Cl distances and different levels. The signature is the sorted pair of nearest Pb
    distances.
    """
    from ase.neighborlist import neighbor_list  # noqa: F401  (kept for parity of imports)

    positions = pristine.get_positions()
    symbols = np.array(pristine.get_chemical_symbols())
    lead = np.flatnonzero(symbols == "Pb")
    groups: dict = {}
    for index in np.flatnonzero(symbols == "Cl"):
        delta = positions[lead] - positions[index]
        # Minimum image, so a Cl near a face is not mistaken for a distant site.
        fractional = delta @ np.linalg.inv(pristine.cell.array)
        fractional -= np.rint(fractional)
        separations = np.sort(np.linalg.norm(fractional @ pristine.cell.array, axis=1))[:2]
        groups.setdefault(tuple(np.round(separations, 2)), []).append(int(index))
    return dict(sorted(groups.items()))


def make_vacancy(supercell, site_index: int):
    """Remove a **fixed** atom index, so the same crystallographic site is used at every size.

    Selecting "the Cl nearest the cell centre" does not do this: at different repeats it
    lands on different sites, and the measured level then tracks the site rather than the
    cell size. Observed directly -- two ladder points that happened to pick the same site
    gave identical <u> to four decimals while an intermediate one differed by 320 meV.

    A fixed index is safe because ``Atoms.repeat`` tiles the original cell contiguously, so
    index ``k`` of the primitive cell is index ``k`` of every supercell, and in a periodic
    cell the defect's position is irrelevant anyway.
    """
    defect = supercell.copy()
    del defect[site_index]
    return defect, site_index


def relax(atoms, calculator, fmax: float, steps: int):
    from ase.optimize import BFGS, LBFGS

    atoms.calc = calculator
    driver = BFGS if len(atoms) < 500 else LBFGS
    with driver(atoms, logfile=None) as optimisation:
        optimisation.run(fmax=fmax, steps=steps)
    residual = float(np.abs(atoms.get_forces()).max())
    return float(atoms.get_potential_energy()), residual, residual <= fmax


def single_point(atoms, state: dict, calculator) -> float:
    probe = atoms.copy()
    set_state(probe, state)
    probe.calc = calculator
    return float(probe.get_potential_energy())


def configuration_coordinate(a, b) -> float:
    from ase.geometry import find_mic

    delta, _ = find_mic(b.positions - a.positions, a.cell, pbc=True)
    return float(np.sqrt((a.get_masses() * (delta**2).sum(axis=1)).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--repeats", type=parse_repeat, nargs="+",
                        default=[(2, 2, 2), (3, 2, 2), (3, 3, 2), (3, 3, 3)])
    parser.add_argument("--energy-scale", default="referenced",
                        choices=["referenced", "raw"])
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--fmax-bulk", type=float, default=0.02,
                        help="tolerance for the primitive-cell relaxation that every "
                             "supercell is then built from")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dilute", action="store_true",
                        help="evaluate the long-range branch in its ISOLATED limit rather "
                             "than the periodic one. For a charged cell this is the "
                             "substantive choice, not a detail: the periodic evaluation "
                             "includes the compensating background and the q^2/L image "
                             "term, the dilute one is what an isolated defect would give. "
                             "The difference between them IS the finite-size correction "
                             "the model is applying, so running both measures it directly")
    parser.add_argument("--site", type=int, default=0,
                        help="which crystallographic Cl site to vacate; sites are grouped "
                             "by their two nearest Pb distances and listed at run time. "
                             "Must be held fixed across the ladder or the measurement "
                             "compares different defects")
    parser.add_argument("--no-relax", action="store_true",
                        help="single points on the ideal vacancy geometry only; gives "
                             "eps_opt and the base-extensivity check without the cost of "
                             "two relaxations per size")
    parser.add_argument("--out-json", type=Path, default=here / "perov_size.json")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    import torch
    from mace.calculators import MACECalculator
    from mace.data.defects import load_band_edges, registry_key

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    edges = load_band_edges(args.data_dir / "band_edges.json")["CsPbCl3"]
    pristine_seed = find_pristine(args.data_dir)
    sizes_needed = set()
    for repeat in args.repeats:
        count = len(pristine_seed) * int(np.prod(repeat))
        sizes_needed.update({count, count - 1})
    registry = {
        registry_key("CsPbCl3", size): {
            "e_cbm_cell": edges.e_cbm_cell, "e_vbm_cell": edges.e_vbm_cell
        }
        for size in sizes_needed
    }

    calculator = MACECalculator(
        model_paths=str(args.model), device=device, default_dtype="float64",
        energy_scale=args.energy_scale, band_edges=registry, dilute=args.dilute,
    )
    print(f"model        {args.model.name}")
    print(f"device       {device}   energy scale: {args.energy_scale}   "
          f"dilute: {args.dilute}")
    print(f"gauge        e_vbm = {edges.e_vbm_cell:+.4f} eV, "
          f"e_cbm = {edges.e_cbm_cell:+.4f} eV  (FITTED, not DFT)")
    print(f"pristine     {pristine_seed.get_chemical_formula()}  "
          f"{len(pristine_seed)} atoms")

    # RELAX THE PRIMITIVE FIRST. The source frames are 300 K MD snapshots, not the ideal
    # lattice, so tiling one and relaxing the supercell makes every size relax a different
    # frozen-in disorder pattern into a different minimum: observed as Delta q ~ 15
    # sqrt(Da) A and E_rel swinging 23 eV across the ladder, neither of which is a
    # finite-size effect. Building every supercell from a single relaxed primitive makes
    # the sizes comparable by construction.
    if not args.no_relax:
        from ase.filters import FrechetCellFilter
        from ase.optimize import BFGS

        set_state(pristine_seed, PRISTINE)
        pristine_seed.calc = calculator
        with BFGS(FrechetCellFilter(pristine_seed), logfile=None) as optimisation:
            optimisation.run(fmax=args.fmax_bulk, steps=args.steps)
        residual = float(np.abs(pristine_seed.get_forces()).max())
        lengths = pristine_seed.cell.cellpar()[:3]
        print(f"bulk relaxed a,b,c = {lengths[0]:.3f}, {lengths[1]:.3f}, "
              f"{lengths[2]:.3f} A   |F|max {residual:.4f} eV/A"
              f"{'' if residual <= args.fmax_bulk else '   NOT CONVERGED'}")

    sites = classify_sites(pristine_seed)
    print(f"\nCl sites in the primitive cell, by their two nearest Pb distances:")
    for number, (signature, members) in enumerate(sites.items()):
        mark = " <- selected" if number == args.site else ""
        print(f"  site {number}: d(Pb) = {signature} A, {len(members)} equivalent "
              f"atoms{mark}")
    if args.site >= len(sites):
        raise SystemExit(f"--site {args.site} but only {len(sites)} sites exist")
    site_index = list(sites.values())[args.site][0]
    print(f"  vacating atom index {site_index}, held FIXED across the ladder so every "
          f"size removes the same site\n")

    header = (f"{'repeat':>10s} {'N':>6s} {'eps_opt':>9s} {'eps(+/0)':>9s} "
              f"{'E_rel':>10s} {'Dq':>8s} {'base':>9s}")
    print(header)
    print(f"{'':>10s} {'':>6s} {'(eV)':>9s} {'(eV)':>9s} {'(eV)':>10s} "
          f"{'(VDa A)':>8s} {'(meV/at)':>9s}")

    rows = []
    e_bulk = None
    for repeat in args.repeats:
        supercell = pristine_seed.repeat(repeat)
        set_state(supercell, PRISTINE)
        supercell.calc = calculator
        e_host = float(supercell.get_potential_energy())
        if e_bulk is None:
            e_bulk = e_host / len(supercell)
        base_drift = (e_host / len(supercell) - e_bulk) * 1e3

        template, removed = make_vacancy(supercell, site_index)

        neutral = template.copy()
        set_state(neutral, VACANCY_Q0)
        if args.no_relax:
            neutral.calc = calculator
            e_neutral = float(neutral.get_potential_energy())
            charged = template.copy()
            set_state(charged, VACANCY_QP1)
            charged.calc = calculator
            e_charged = float(charged.get_potential_energy())
            e_charged_at_neutral = e_charged
            dq = 0.0
            converged = True
        else:
            e_neutral, _, ok0 = relax(neutral, calculator, args.fmax, args.steps)
            charged = template.copy()
            set_state(charged, VACANCY_QP1)
            e_charged, _, okp = relax(charged, calculator, args.fmax, args.steps)
            e_charged_at_neutral = single_point(neutral, VACANCY_QP1, calculator)
            dq = configuration_coordinate(neutral, charged)
            converged = ok0 and okp

        eps_opt = e_charged_at_neutral - e_neutral
        # Thermodynamic level: the Fermi energy at which the two charge states are
        # degenerate, both structures relaxed. On the referenced scale this is already
        # measured from the (gauge) valence band edge.
        transition = e_neutral - e_charged
        relative = e_neutral - e_host

        rows.append({
            "repeat": list(repeat), "n_host": len(supercell), "n_defect": len(neutral),
            "e_host": e_host, "e_neutral": e_neutral, "e_charged": e_charged,
            "eps_opt_eV": eps_opt, "transition_level_eV": transition,
            "e_relative_eV": relative, "delta_q_sqrtDa_A": dq,
            "base_drift_meV_per_atom": base_drift, "converged": bool(converged),
            "removed_index": removed,
        })
        print(f"{str(repeat):>10s} {len(supercell):6d} {eps_opt:9.4f} {transition:9.4f} "
              f"{relative:10.4f} {dq:8.4f} {base_drift:+9.3f}"
              f"{'' if converged else '  UNCONVERGED'}")

    args.out_json.write_text(json.dumps(
        {"model": str(args.model), "energy_scale": args.energy_scale,
         "e_vbm_gauge": edges.e_vbm_cell, "relaxed": not args.no_relax,
         "rows": rows}, indent=1))
    print(f"\nwrote {args.out_json}")

    if len(rows) >= 2:
        print("\ndrift from the smallest to the largest cell:")
        for key, label, scale in (("eps_opt_eV", "eps_opt", 1e3),
                                  ("transition_level_eV", "eps(+/0)", 1e3),
                                  ("e_relative_eV", "E_rel", 1e3),
                                  ("delta_q_sqrtDa_A", "Delta q", 1.0)):
            change = rows[-1][key] - rows[0][key]
            print(f"  {label:10s} {rows[0][key]:9.4f} -> {rows[-1][key]:9.4f}   "
                  f"{change * scale:+8.2f} {'meV' if scale > 1 else ''}")
        print(
            "\nA charged cell has a genuine 1/L image tail that a short-range model cannot\n"
            "represent, so eps(+/0) and E_rel are expected to drift where a neutral\n"
            "quantity would not. eps_opt is a difference at FIXED geometry between two\n"
            "charge states, so it carries the same tail once and does not cancel it."
        )


if __name__ == "__main__":
    main()
