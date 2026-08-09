#!/usr/bin/env python3
"""Finite-size convergence of the divacancy: E_f, E_ZPL, E_vert^abs and Delta q.

Reproduces the convergence figure of the NEP paper for a MACEDefect model. Four
quantities are followed up a ladder of supercells and extrapolated to the isolated limit:

    E_f          = E_gs(R_gs) - E_host + mu_SiC          formation energy
    E_ZPL        = E_ex(R_ex) - E_gs(R_gs)               zero-phonon line
    E_vert^abs   = E_ex(R_gs) - E_gs(R_gs)               vertical absorption
    Delta        = E_vert^abs - E_ZPL                    excited-state relaxation
    Delta q      = sqrt(sum_i m_i |R_ex,i - R_gs,i|^2)   configuration coordinate

so each size needs two relaxations (ground and excited) and two single points (each state
at the other's geometry).

**This is also an extensivity test of the model**, which is why it is worth running on a
defect potential rather than only on a converged one. The architecture is a sum of an
*extensive* base (a sum of local atomic energies, growing with N) and an *intensive*
correction (``sum_c n_c sum_i alpha_i u_i`` with ``sum_i alpha_i = 1``, a weighted
average). Nothing forces ``alpha`` to stay localised; if it spreads over the cell, the
correction becomes a bulk average of ``u`` and every quantity above drifts with N instead
of converging. Training cannot see this -- it is done at one or two cell sizes.

Two diagnostics separate a model failure from real finite-size physics:

* ``E_host/N - e_bulk`` is **base extensivity**. The host supercell is the relaxed
  primitive repeated, so an extensive base gives identically zero at every N. Anything
  else, and the energies below are resting on a base model that is not size-consistent.
* ``Delta E`` (the correction alone, ``E(n) - E(0)`` at one geometry) removes the base
  exactly. It is the intensive term, so it should be flat in N.

Energy scale. **``raw`` is required and is the default.** The labels were band-edge
referenced, and the constant is ``n_electrons*e_cbm - n_holes*e_vbm``: 0.9430 eV for the
ground state and 1.8861 eV for the excited state on this dataset. It therefore does *not*
cancel in ``E_ZPL`` or ``E_vert^abs`` -- computing them on the referenced scale would
shift both by exactly the gauge gap, 0.9430 eV, which is the same order as the answer.
``--energy-scale referenced`` is offered only for diagnostics.

Band edges are keyed ``host|natoms`` in the training registry, and this script visits cell
sizes that were never trained on. The recorded table holds one entry per host with no size
dependence (it is a fitted gauge in which only ``e_cbm - e_vbm`` matters), so the entry is
replicated to every size needed. If a genuinely size-dependent table is ever recorded,
this replication is wrong and must be revisited -- it is reported at run time for that
reason.

    python defect_size_extensivity.py --model ~/runs/b_128ch_L1_s1/b_128ch_L1_s1.model
    python defect_size_extensivity.py --model <final>.model \
        --repeats 5,5,2 6,6,2 8,8,2 10,10,3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

# Plan-convention carrier counters (n_e_maj, n_e_min, n_h_maj, n_h_min). Overridden by
# dataset_summary.json when one is found, so this file cannot drift from the labels the
# model was trained on.
STATES = {
    "ground": {"counts": [1, 0, 0, 1], "multiplicity": 3},
    "excited": {"counts": [1, 1, 0, 2], "multiplicity": 3},
}
PRISTINE = {"counts": [0, 0, 0, 0], "multiplicity": 1}

# The paper's ladder where it overlaps ours: 400, 576, 1024, 2400 atoms.
DEFAULT_REPEATS = [(4, 4, 2), (5, 5, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3)]


def load_state_conventions(data_dir: Path) -> None:
    """Adopt the counter vectors recorded when the dataset was generated, if present."""
    summary = data_dir / "dataset_summary.json"
    if not summary.is_file():
        return
    try:
        recorded = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return
    vectors = recorded.get("counter_vectors") or {}
    multiplicities = recorded.get("multiplicities") or {}
    for name in ("ground", "excited"):
        if name in vectors:
            STATES[name]["counts"] = list(vectors[name])
            STATES[name]["multiplicity"] = int(multiplicities.get("defect", 3))
    if "pristine" in vectors:
        PRISTINE["counts"] = list(vectors["pristine"])
        PRISTINE["multiplicity"] = int(multiplicities.get("pristine", 1))


def parse_repeat(text: str) -> tuple:
    parts = [int(v) for v in text.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'n1,n2,n3', got {text!r}")
    return tuple(parts)


def find_primitive(data_dir: Path, explicit: Path | None):
    """The smallest pristine cell available: the 4H-SiC primitive, 8 atoms."""
    from ase.io import read

    if explicit is not None:
        return read(str(explicit), index=0)
    for name in ("train.xyz", "valid.xyz"):
        path = data_dir / name
        if not path.is_file():
            continue
        frames = [
            atoms
            for atoms in read(str(path), ":")
            if str(atoms.info.get("config_type", "")).startswith("ideal")
        ]
        if frames:
            return min(frames, key=len)
    raise SystemExit(
        f"no pristine frame found under {data_dir}; pass --primitive explicitly"
    )


def build_registry(data_dir: Path, host: str, sizes) -> dict:
    """Replicate the host's band edges to every cell size this run will visit.

    The registry is keyed ``host|natoms`` but the recorded table has a single size-free
    entry per host, so without this every cell size off the training ladder raises
    KeyError under ``--energy-scale raw``.
    """
    from mace.data.defects import load_band_edges, registry_key

    path = data_dir / "band_edges.json"
    if not path.is_file():
        raise SystemExit(
            f"no band_edges.json under {data_dir}; needed for --energy-scale raw"
        )
    table = load_band_edges(path)
    if host not in table:
        raise SystemExit(f"host '{host}' not in {path}; known: {sorted(table)}")
    edges = table[host]
    return {
        registry_key(host, size): {
            "e_cbm_cell": edges.e_cbm_cell,
            "e_vbm_cell": edges.e_vbm_cell,
        }
        for size in sizes
    }


def set_state(atoms, state: dict, host: str) -> None:
    atoms.info["carrier_counts"] = list(state["counts"])
    atoms.info["multiplicity"] = int(state["multiplicity"])
    atoms.info["host"] = host


def make_divacancy(supercell, pair: str, bond_tol: float = 0.25):
    """Remove the nearest-neighbour Si-C pair closest to the cell centre.

    ``axial`` takes the bond most nearly parallel to c, ``basal`` the most perpendicular;
    in 4H-SiC these are genuinely inequivalent defects, so the choice is recorded rather
    than left to whichever atom happened to come first in the file.
    """
    from ase.neighborlist import neighbor_list

    symbols = np.array(supercell.get_chemical_symbols())
    # A neighbour list rather than the full distance matrix: at 2400 atoms the N^2 matrix
    # is 46 M entries and the O(N) list is the difference between seconds and minutes.
    first, second, vectors = neighbor_list("ijD", supercell, cutoff=2.6)
    is_si_c = (symbols[first] == "Si") & (symbols[second] == "C")
    if not is_si_c.any():
        raise SystemExit("no Si-C bonds found; is this SiC?")
    first, second, vectors = first[is_si_c], second[is_si_c], vectors[is_si_c]
    lengths = np.linalg.norm(vectors, axis=1)
    bond = float(lengths.min())

    keep = lengths < bond + bond_tol
    first, second, vectors = first[keep], second[keep], vectors[keep]
    along_c = np.abs(vectors[:, 2]) / np.linalg.norm(vectors, axis=1)
    axial = along_c > 0.9
    candidates = np.flatnonzero(axial if pair == "axial" else ~axial)
    if len(candidates) == 0:
        raise SystemExit(
            f"no {pair} Si-C nearest-neighbour bond found (bond length {bond:.3f} A)"
        )

    centre = supercell.cell.array.sum(axis=0) / 2.0
    midpoints = supercell.positions[first[candidates]] + vectors[candidates] / 2.0
    best = candidates[int(np.argmin(np.linalg.norm(midpoints - centre, axis=1)))]
    i, j = int(first[best]), int(second[best])

    defect = supercell.copy()
    del defect[[max(i, j), min(i, j)]]  # highest index first
    return defect, {"pair": pair, "removed": [i, j], "bond_length": bond}


def relax(atoms, calculator, fmax: float, steps: int, optimizer: str, cell: bool = False):
    """Relax to ``fmax``; ``cell=True`` also relaxes the cell (bulk only)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS, FIRE, LBFGS

    atoms.calc = calculator
    target = FrechetCellFilter(atoms) if cell else atoms
    if optimizer == "auto":
        # BFGS converges in fewer steps but stores a dense (3N)^2 Hessian: 415 MB at
        # 2400 atoms, and growing as N^2. LBFGS is the only sane choice up there.
        optimizer = "bfgs" if len(atoms) < 500 else "lbfgs"
    driver = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}[optimizer]
    with driver(target, logfile=None) as optimisation:
        optimisation.run(fmax=fmax, steps=steps)
    residual = float(np.abs(atoms.get_forces()).max())
    return float(atoms.get_potential_energy()), residual, residual <= fmax


def single_point(atoms, state: dict, host: str, calculator) -> float:
    """Energy of ``atoms`` held fixed, in a different carrier state."""
    probe = atoms.copy()
    set_state(probe, state, host)
    probe.calc = calculator
    return float(probe.get_potential_energy())


def configuration_coordinate(ground, excited) -> float:
    """Delta q = sqrt(sum_i m_i |dR_i|^2) in sqrt(Da) Angstrom.

    The displacement is taken with the minimum-image convention: relaxation can carry an
    atom across a cell face, and the raw coordinate difference would then report a whole
    lattice vector of motion that did not happen.
    """
    from ase.geometry import find_mic

    delta, _ = find_mic(excited.positions - ground.positions, ground.cell, pbc=True)
    masses = ground.get_masses()
    return float(np.sqrt((masses * (delta**2).sum(axis=1)).sum()))


def fit_limit(sizes: np.ndarray, values: np.ndarray, power: float):
    """Extrapolate to N -> inf against ``N**-power``; returns (intercept, max deviation)."""
    if len(sizes) < 2:
        return float(values[-1]), float("nan")
    abscissa = sizes.astype(float) ** (-power)
    slope, intercept = np.polyfit(abscissa, values, 1)
    deviation = float(np.abs(values - (slope * abscissa + intercept)).max())
    return float(intercept), deviation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset")
    parser.add_argument("--primitive", type=Path, default=None,
                        help="pristine cell to build from; default: smallest ideal frame")
    parser.add_argument("--repeats", type=parse_repeat, nargs="+",
                        default=DEFAULT_REPEATS,
                        help="supercell repeats, e.g. 5,5,2 6,6,2 8,8,2 10,10,3")
    parser.add_argument("--pair", default="axial", choices=["axial", "basal"])
    parser.add_argument("--host", default=None,
                        help="host label for the band-edge table; default: from the "
                             "primitive frame, else '4H-SiC'")
    parser.add_argument("--fmax", type=float, default=0.02, help="eV/A")
    parser.add_argument("--fmax-bulk", type=float, default=0.005, help="eV/A, cell relax")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--optimizer", default="auto",
                        choices=["auto", "bfgs", "lbfgs", "fire"])
    parser.add_argument("--device", default=None, help="cuda / cpu; default: auto")
    parser.add_argument("--energy-scale", default="raw",
                        choices=["raw", "referenced"],
                        help="raw undoes the band-edge referencing and is REQUIRED for "
                             "E_ZPL to mean anything; see the module docstring")
    parser.add_argument("--energy-power", type=float, default=1.0,
                        help="extrapolate energies against N**-p; 1.0 (the paper's "
                             "choice) makes image-interaction terms a straight line")
    parser.add_argument("--dq-power", type=float, default=1 / 3,
                        help="extrapolate Delta q against N**-p; the elastic field at "
                             "the defect converges with the linear cell dimension")
    parser.add_argument("--dilute", action="store_true",
                        help="evaluate the long-range branch in its isolated limit. OFF "
                             "by default and it must stay off for this test: the dilute "
                             "term is size-independent by construction, so switching it "
                             "on would make a long-range model pass trivially instead of "
                             "being measured")
    parser.add_argument("--no-host-relax", action="store_true",
                        help="skip the host relaxation; it is a symmetry-fixed minimum "
                             "at the relaxed bulk cell, so this is normally a no-op")
    parser.add_argument("--out-json", type=Path, default=here / "size_convergence.json")
    parser.add_argument("--out-plot", type=Path, default=here / "size_convergence.png")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    import torch
    from mace.calculators import MACECalculator

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    load_state_conventions(args.data_dir)

    primitive_probe = find_primitive(args.data_dir, args.primitive)
    host_label = args.host or str(primitive_probe.info.get("host", "4H-SiC"))
    sizes = {len(primitive_probe)}
    for repeat in args.repeats:
        count = len(primitive_probe) * int(np.prod(repeat))
        sizes.update({count, count - 2})

    registry = None
    if args.energy_scale == "raw":
        registry = build_registry(args.data_dir, host_label, sorted(sizes))

    calculator = MACECalculator(
        model_paths=str(args.model),
        device=device,
        default_dtype="float64",
        energy_scale=args.energy_scale,
        band_edges=registry,
        dilute=args.dilute,
    )
    if not calculator.is_defect_model:
        raise SystemExit(f"{args.model} is not a MACEDefect model")

    primitive = primitive_probe
    set_state(primitive, PRISTINE, host_label)
    print(f"model         {args.model.name}")
    print(f"device        {device}   energy scale: {args.energy_scale}   "
          f"defect: {args.pair} divacancy")
    if registry is not None:
        from mace.data.defects import referencing_constant
        edges = build_registry(args.data_dir, host_label, [1])
        entry = next(iter(edges.values()))
        from mace.data.defects import BandEdges
        band = BandEdges(entry["e_cbm_cell"], entry["e_vbm_cell"])
        print(f"referencing   ground {referencing_constant(np.array(STATES['ground']['counts']), band):+.4f} eV, "
              f"excited {referencing_constant(np.array(STATES['excited']['counts']), band):+.4f} eV "
              f"(size-independent table, replicated to {len(sizes)} cell sizes)")
    print(f"primitive     {primitive.get_chemical_formula()}  {len(primitive)} atoms")

    # --- bulk reference -----------------------------------------------------------------
    energy_bulk, residual, converged = relax(
        primitive, calculator, args.fmax_bulk, args.steps, args.optimizer, cell=True
    )
    e_bulk = energy_bulk / len(primitive)
    cellpar = primitive.cell.cellpar()
    print(f"bulk relaxed  a = {cellpar[0]:.4f} A, c = {cellpar[2]:.4f} A, "
          f"c/a = {cellpar[2] / cellpar[0]:.4f}  "
          f"({'converged' if converged else f'NOT CONVERGED |F|max {residual:.3f}'})")
    print(f"              e_bulk = {e_bulk:.6f} eV/atom, mu_SiC = {2 * e_bulk:.6f} eV\n")

    header = (f"{'repeat':>10s} {'N':>6s} {'E_f':>9s} {'E_ZPL':>8s} {'E_abs':>8s} "
              f"{'Delta':>7s} {'Dq':>7s} {'dE_corr':>8s} {'base':>7s}")
    print(header)
    print(f"{'':>10s} {'':>6s} {'(eV)':>9s} {'(eV)':>8s} {'(eV)':>8s} {'(eV)':>7s} "
          f"{'(VDa A)':>7s} {'(eV)':>8s} {'(meV/at)':>7s}")

    rows = []
    for repeat in args.repeats:
        supercell = primitive.repeat(repeat)
        set_state(supercell, PRISTINE, host_label)
        n_host = len(supercell)

        if args.no_host_relax:
            supercell.calc = calculator
            energy_host, host_ok = float(supercell.get_potential_energy()), True
        else:
            energy_host, _, host_ok = relax(
                supercell, calculator, args.fmax, args.steps, args.optimizer
            )
        base_drift = (energy_host / n_host - e_bulk) * 1e3

        template, provenance = make_divacancy(supercell, args.pair)

        ground = template.copy()
        set_state(ground, STATES["ground"], host_label)
        e_gs_at_gs, fmax_gs, gs_ok = relax(
            ground, calculator, args.fmax, args.steps, args.optimizer
        )

        excited = template.copy()
        set_state(excited, STATES["excited"], host_label)
        e_ex_at_ex, fmax_ex, ex_ok = relax(
            excited, calculator, args.fmax, args.steps, args.optimizer
        )

        # Each state evaluated at the other's geometry: the vertical transitions.
        e_ex_at_gs = single_point(ground, STATES["excited"], host_label, calculator)
        e_gs_at_ex = single_point(excited, STATES["ground"], host_label, calculator)
        # The correction alone at one fixed geometry; E(n) - E(0) cancels E_base exactly.
        delta_correction = e_gs_at_gs - single_point(
            ground, PRISTINE, host_label, calculator
        )

        formation = e_gs_at_gs - energy_host + 2 * e_bulk
        zpl = e_ex_at_ex - e_gs_at_gs
        vertical_absorption = e_ex_at_gs - e_gs_at_gs
        vertical_emission = e_ex_at_ex - e_gs_at_ex
        relaxation = vertical_absorption - zpl
        dq = configuration_coordinate(ground, excited)

        converged_all = host_ok and gs_ok and ex_ok
        rows.append({
            "repeat": list(repeat), "n_host": n_host, "n_defect": len(ground),
            "e_host": energy_host, "e_gs_at_gs": e_gs_at_gs,
            "e_ex_at_ex": e_ex_at_ex, "e_ex_at_gs": e_ex_at_gs,
            "e_gs_at_ex": e_gs_at_ex,
            "formation_energy_eV": formation, "zpl_eV": zpl,
            "vertical_absorption_eV": vertical_absorption,
            "vertical_emission_eV": vertical_emission,
            "relaxation_energy_eV": relaxation,
            "delta_q_sqrtDa_A": dq,
            "delta_correction_eV": delta_correction,
            "base_drift_meV_per_atom": base_drift,
            "converged": bool(converged_all),
            "fmax_gs": fmax_gs, "fmax_ex": fmax_ex,
            **provenance,
        })
        print(f"{str(repeat):>10s} {n_host:6d} {formation:9.4f} {zpl:8.4f} "
              f"{vertical_absorption:8.4f} {relaxation:7.4f} {dq:7.4f} "
              f"{delta_correction:8.4f} {base_drift:+7.3f}"
              f"{'' if converged_all else '  UNCONVERGED'}")

    # --- extrapolation ------------------------------------------------------------------
    sizes_array = np.array([r["n_host"] for r in rows], dtype=float)
    limits = {}
    print(f"\nextrapolated to N -> infinity:")
    for key, label, power in (
        ("formation_energy_eV", "E_f", args.energy_power),
        ("zpl_eV", "E_ZPL", args.energy_power),
        ("vertical_absorption_eV", "E_vert^abs", args.energy_power),
        ("vertical_emission_eV", "E_vert^em", args.energy_power),
        ("relaxation_energy_eV", "Delta", args.energy_power),
        ("delta_correction_eV", "Delta E (correction)", args.energy_power),
        ("delta_q_sqrtDa_A", "Delta q", args.dq_power),
    ):
        values = np.array([r[key] for r in rows])
        intercept, deviation = fit_limit(sizes_array, values, power)
        limits[key] = intercept
        unit = "sqrt(Da) A" if key.startswith("delta_q") else "eV"
        print(f"  {label:22s} {intercept:9.4f} {unit:11s} "
              f"(N^-{power:.3g} fit, max deviation {deviation * (1 if unit != 'eV' else 1e3):.3f}"
              f" {'meV' if unit == 'eV' else unit})")

    drift = float(np.abs(np.array([r["delta_correction_eV"] for r in rows])
                         - rows[-1]["delta_correction_eV"]).max())
    base = float(np.abs(np.array([r["base_drift_meV_per_atom"] for r in rows])).max())
    print(f"\nextensivity: base drift |E_host/N - e_bulk| <= {base:.4f} meV/atom "
          f"({'PASS' if base < 0.01 else 'FAIL -- base branch is not size-consistent'})")
    print(f"             correction spread {drift:.4f} eV across the ladder "
          f"({'flat' if drift < 0.05 else 'DRIFTING -- alpha may be delocalising'})")

    args.out_json.write_text(json.dumps(
        {"model": str(args.model), "energy_scale": args.energy_scale,
         "e_bulk_per_atom": e_bulk, "cellpar": list(cellpar),
         "limits": limits, "rows": rows}, indent=1,
    ))
    print(f"\nwrote {args.out_json}")
    make_plot(rows, limits, args)


def make_plot(rows, limits, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": True,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    sizes = np.array([r["n_host"] for r in rows], dtype=float)
    order = np.argsort(1.0 / sizes)
    inverse = (1.0 / sizes)[order] * 1e3  # plotted in units of 1e-3

    figure, axes = plt.subplots(
        2, 1, figsize=(4.4, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.06},
    )

    # Panel a: each energy as its deviation from its own extrapolated limit, in meV, so
    # three quantities of very different absolute size share one axis.
    series = (
        ("formation_energy_eV", r"$E_\mathrm{f}$", "#8ab0e0", "o", 8),
        ("zpl_eV", r"$E_\mathrm{ZPL}$", "#e5c15a", "^", 7),
        ("vertical_absorption_eV", r"$E^\mathrm{abs}_\mathrm{vert}$", "#7fbf7b", "x", 7),
    )
    for key, label, colour, marker, size in series:
        values = np.array([r[key] for r in rows])[order]
        deviation = (values - limits[key]) * 1e3
        fitted = np.polyfit(inverse, deviation, 1)
        span = np.array([0.0, inverse.max() * 1.05])
        axes[0].plot(span, np.polyval(fitted, span), "-", lw=1.6, color=colour, alpha=0.9)
        axes[0].plot(inverse, deviation, marker, ms=size, color=colour,
                     mew=1.6 if marker == "x" else 0,
                     label=f"{label} $\\rightarrow$ {limits[key]:.2f} eV")
    axes[0].axhline(0.0, lw=0.6, color="0.75", zorder=0)
    axes[0].set_ylabel(r"$\Delta E$ (meV)")
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left", handletextpad=0.4)
    axes[0].text(0.04, 0.93, "a)", transform=axes[0].transAxes, fontsize=11)

    # Panel b: Delta q absolutely, with its own (different) convergence power.
    values = np.array([r["delta_q_sqrtDa_A"] for r in rows])[order]
    colour = "#a0342c"
    grid = np.linspace(0.0, inverse.max() * 1.05, 200)
    # Fit in the variable it actually converges in, then draw it against 1/N, which is
    # why this line is curved where panel a's are straight.
    fitted = np.polyfit((sizes[order]) ** (-args.dq_power), values, 1)
    with np.errstate(divide="ignore"):
        curve = np.polyval(fitted, (grid / 1e3) ** args.dq_power)
    curve[grid == 0] = fitted[1]
    axes[1].plot(grid, curve, "-", lw=1.6, color=colour, alpha=0.9)
    axes[1].plot(inverse, values, "^", ms=7, color=colour, mew=0,
                 label=f"$\\Delta q \\rightarrow$ {limits['delta_q_sqrtDa_A']:.2f} "
                       r"$\sqrt{\mathrm{Da}}$ $\mathrm{\AA}$")
    axes[1].set_ylabel(r"Displ. $\Delta q$ ($\sqrt{\mathrm{Da}}\,\mathrm{\AA}$)")
    axes[1].set_xlabel(r"Inverse number of atoms $1/N$ ($\times 10^{-3}$)")
    axes[1].legend(frameon=False, fontsize=9.5, loc="lower right", handletextpad=0.4)
    axes[1].text(0.04, 0.93, "b)", transform=axes[1].transAxes, fontsize=11)
    axes[1].set_xlim(-0.05, inverse.max() * 1.05)

    # Top axis labelled by N, as in the original.
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(inverse)
    top.set_xticklabels([f"{int(n)}" for n in sizes[order]], fontsize=9)
    top.set_xlabel("Number of atoms $N$", fontsize=9)
    top.tick_params(direction="in")

    figure.savefig(args.out_plot, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out_plot}")


if __name__ == "__main__":
    main()
