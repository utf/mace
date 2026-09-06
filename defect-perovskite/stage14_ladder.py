#!/usr/bin/env python3
"""Plan v8 section 7.5, the tiling ladder (model only), periodic gauge -- Stage 1.4.

One ideal Cl vacancy in the 1x, 2x, 3x isotropic tilings of the IDEAL pristine cell (79,
639, 2159 atoms), at its own charge state (V_Cl+, the counter of a real charged frame) and
neutral; the same construction with a THERMAL charged 79-atom frame embedded in the first
sub-cell of each tiling; every frame under both kernels. What is required (7.5):

  * forces within a fixed radius of the vacancy converge with L;
  * the correction converges after switching to G_inf;
  * the total force is zero;
  * the volume-scaled defect stress (the vacancy cell's virial minus the tiled pristine's)
    converges;
  * no discontinuity from any switch: E at three steepnesses of w (delta_p) and three
    widths of the projectors (Delta_s), per size;
  * the leading exponent of E_PBC - E_inf per component, fitted, nothing assumed; the
    monopole components must go as 1/L and the monopole-neutral ones must move away from it;
  * the thermal-displacement image contribution, reported separately.

The ideal pristine cell is the POPULATION MEAN of the stoichiometric training frames: the
mean cell, and the mean fractional position of each site after matching every frame's atoms
to the first frame's by species and minimum-image distance (the frames are not
index-aligned). Written to `golden/ideal_pristine_80.xyz` with the matching residuals.

Before Stage 4 the only gauge-dependent term is Phi_FF (the Madelung term inside H is
periodic in both gauges), so E_PBC - E_inf is Phi_FF itself; the band term and the trunk
are reported per size so their own convergence is visible.

    python defect-perovskite/stage14_ladder.py --model ~/runs/arma_models/arma_s1.model \\
        --reps 1 2 3 --out defect-perovskite/golden/stage14_ladder_arma_s1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import mace  # noqa: F401
from ase import Atoms
from ase.io import read, write

from mace.modules.defect_context import ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from stage0_fd import charged_pool, load_uniform  # noqa: E402
from stage0_golden import DATA, ensure_table, frame_selection, git_sha, make_batch  # noqa: E402

HERE = Path(__file__).resolve().parent
CARRIER_INFO_KEYS = ("carrier_counts", "multiplicity", "m_s_ref_doubled", "cell_charge",
                     "config_type", "host", "e_cbm_cell", "e_vbm_cell")
DOMAIN_RADIUS = 0.8      # A: frames whose every atom matches within this are one domain
RADIUS = 6.0


# ------------------------------------------------------------------ geometry


def _min_image(delta, cell):
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def match_to(reference: Atoms, atoms: Atoms):
    """Index map `perm` with `atoms[perm[i]]` the atom of `atoms` matched to `reference[i]`
    (same species, minimum-image nearest, one to one), and the matched distances."""
    cell = np.array(reference.get_cell())
    ref_pos, pos = reference.get_positions(), atoms.get_positions()
    ref_z, z = reference.get_atomic_numbers(), atoms.get_atomic_numbers()
    perm = np.full(len(reference), -1)
    dist = np.zeros(len(reference))
    for species in sorted(set(ref_z.tolist())):
        i_ref = np.nonzero(ref_z == species)[0]
        i_at = np.nonzero(z == species)[0]
        d = np.linalg.norm(_min_image(pos[i_at][None, :, :] - ref_pos[i_ref][:, None, :], cell),
                           axis=-1)
        rows, cols = linear_sum_assignment(d)
        perm[i_ref[rows]] = i_at[cols]
        dist[i_ref[rows]] = d[rows, cols]
    return perm, dist


def ideal_pristine(frames, out_path: Path, stride: int = 10):
    """The population mean of the stoichiometric frames of ONE domain, matched atom by atom.

    The training set's pristine frames are not one crystal: matched to any one of them,
    most others sit 1.5-4 A away (different runs, origins and tilt domains), so a mean over
    all of them is not a structure. The domain is the largest set of frames whose every
    atom matches a reference frame within DOMAIN_RADIUS; the reference is the frame (of
    every `stride`-th candidate) with the most such neighbours, and the ideal cell is the
    mean cell and mean matched fractional position over that domain."""
    candidates = list(range(0, len(frames), stride))
    best = None
    for k in candidates:
        worst = np.array([match_to(frames[k], a)[1].max() for a in frames])
        members = np.nonzero(worst < DOMAIN_RADIUS)[0]
        if best is None or len(members) > len(best[1]):
            best = (k, members)
    k, members = best
    ref = frames[k]
    domain = [frames[i] for i in members]
    cells = np.array([np.array(a.get_cell()) for a in domain])
    mean_cell = cells.mean(axis=0)
    ref_frac = ref.get_scaled_positions()
    acc = np.zeros_like(ref_frac)
    worst, means = [], []
    for a in domain:
        perm, dist = match_to(ref, a)
        frac = a.get_scaled_positions()[perm]
        frac = frac - np.round(frac - ref_frac)        # unwrap onto the reference's image
        acc += frac
        worst.append(float(dist.max()))
        means.append(float(dist.mean()))
    mean_frac = acc / len(domain)
    ideal = Atoms(numbers=ref.get_atomic_numbers(), scaled_positions=mean_frac, cell=mean_cell,
                  pbc=True)
    ideal.info = {k_: v for k_, v in ref.info.items() if k_ not in CARRIER_INFO_KEYS}
    stats = dict(n_frames_total=len(frames), reference_frame=int(k), n_domain=len(domain),
                 domain_radius=DOMAIN_RADIUS, matched_distance_mean=float(np.mean(means)),
                 matched_distance_worst=float(np.max(worst)),
                 cell=mean_cell.tolist(), cell_sd=cells.std(axis=0).tolist())
    write(str(out_path), ideal)
    return ideal, stats


def closest_thermal(pristine: Atoms, frames79):
    """The 79-atom frame (any counter; the bookkeeping is copied from a charged frame) whose
    atoms match the ideal cell minus one Cl most closely -- the thermal snapshot that lives
    in the ideal cell's own domain."""
    best = None
    for i, a in enumerate(frames79):
        perm, dist = match_to(pristine, a)
        missing = [j for j in range(len(pristine)) if perm[j] < 0]
        if len(missing) != 1 or pristine.get_atomic_numbers()[missing[0]] != 17:
            continue
        worst = float(dist[perm >= 0].max())
        if best is None or worst < best[0]:
            best = (worst, i)
    return frames79[best[1]], best[1], best[0]


def vacancy_index(pristine: Atoms, thermal: Atoms):
    """Which Cl of the ideal cell the thermal 79-atom frame lacks: match the frame's atoms to
    the ideal cell's (one Cl row stays unmatched). Returns the index, the match and the
    matched distances -- the same site is used for every ideal tiling, so the ideal and the
    thermal ladders are the same defect."""
    perm, dist = match_to(pristine, thermal)
    missing = [i for i in range(len(pristine)) if perm[i] < 0]
    if len(missing) != 1 or pristine.get_atomic_numbers()[missing[0]] != 17:
        raise RuntimeError(f"the thermal frame does not match the ideal cell minus one Cl: "
                           f"unmatched {missing}")
    return int(missing[0]), perm, dist[perm >= 0]


def _bookkeeping(atoms: Atoms, reference: Atoms):
    atoms.info = {k: v for k, v in atoms.info.items() if k not in CARRIER_INFO_KEYS}
    for k in CARRIER_INFO_KEYS:
        if k in reference.info:
            atoms.info[k] = reference.info[k]
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    atoms.info["REF_energy"] = 0.0
    return atoms


def vacancy_cell(pristine: Atoms, reps: int, reference: Atoms, vac_index: int):
    """The tiled pristine cell (the original cell is the first sub-cell of `repeat`) with
    Cl `vac_index` of that sub-cell removed, carrying a real frame's counter bookkeeping.
    Returns the cell and the vacancy's position."""
    atoms = pristine.repeat((reps, reps, reps))
    site = atoms.get_positions()[vac_index].copy()
    del atoms[vac_index]
    return _bookkeeping(atoms, reference), site


def embed_thermal(pristine: Atoms, reps: int, thermal: Atoms, reference: Atoms,
                  vac_index: int, perm):
    """The ideal vacancy tiling with the first sub-cell's atoms displaced to the thermal
    charged frame's fractional positions: one thermal cell inside an ideal periodic
    environment (at 1x it is the thermal frame itself, in the ideal cell)."""
    tiled = pristine.repeat((reps, reps, reps))
    one = np.array(pristine.get_cell())
    positions = tiled.get_positions().copy()
    id_frac = pristine.get_scaled_positions()
    th_frac = thermal.get_scaled_positions()
    for i in range(len(pristine)):
        if perm[i] < 0:
            continue
        d = th_frac[perm[i]] - id_frac[i]
        d -= np.round(d)
        positions[i] = positions[i] + d @ one
    site = tiled.get_positions()[vac_index].copy()
    keep = [i for i in range(len(tiled)) if i != vac_index]
    atoms = Atoms(numbers=tiled.get_atomic_numbers()[keep], positions=positions[keep],
                  cell=tiled.get_cell(), pbc=True)
    return _bookkeeping(atoms, reference), site


# ------------------------------------------------------------------ evaluation


def classes_for(model, ctx, ideal, atoms_list):
    """The ladder's own class table: the IDEAL pristine cell is the reference, the ideal
    vacancy cell at each size its class's reference geometry -- so no size carries a
    reference another size lacks. Replaces the model's (golden-frame) table for the run."""
    from mace.modules.defect_cache import attach_frame_keys
    from mace.modules.defect_composition import describe, ensure_class_table, lookup_class

    from stage0_golden import Z_TABLE, atomic_data

    pristine = ideal.copy()
    pristine.info["carrier_counts"] = np.zeros(4, dtype=int)
    pristine.info["multiplicity"] = 1
    pristine.info["m_s_ref_doubled"] = 0
    pristine.info["cell_charge"] = 0
    pristine.arrays["REF_forces"] = np.zeros((len(pristine), 3))
    pristine.info["REF_energy"] = 0.0
    new = atomic_data([pristine] + list(atoms_list), ctx.cutoff)
    attach_frame_keys(new, z_table=Z_TABLE)
    t0 = time.time()
    model.composition_classes = None
    table = ensure_class_table(model, new, log=False)
    # plan v8.1: the gauge reference is the ladder's own pristine cell (the model's previous
    # reference, if any, was cleared with its table above)
    from mace.modules.defect_models import establish_spectral_gauge

    if hasattr(getattr(model, "spectral", None), "assemble_hamiltonian"):
        model.gauge_reference = None
        establish_spectral_gauge(model, new, log=False)
        table = model.composition_classes
    for a in atoms_list:
        rec = lookup_class(table, a.get_atomic_numbers())
        print(f"  class {rec.key}: {describe(rec)}")
    print(f"  ({time.time() - t0:.0f} s)")
    return table


def evaluate(model, ctx, atoms, gauge, stress=True):
    model.gauge = gauge
    batch = make_batch([atoms], ctx.cutoff)
    d = ctx.forward_dict(batch, requires_grad=True)
    t0 = time.time()
    out = model(d, training=False, compute_force=True, compute_stress=stress)
    out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    model.gauge = "periodic"
    alpha = out["carrier_alpha"][:, 0]
    rec = dict(energy=float(out["energy"][0]), base=float(out["base_trunk_energy"][0]),
               band=float(out["delta_sr_energy"][0]), frontier=float(out["frontier_energy"][0]),
               w=float(out["frontier_w"][0]), w_ref=float(out["frontier_w_ref"][0]),
               p=float(out["frontier_p"][0]), q_F=float(out["frontier_q_F"][0]),
               n_eff=(float(1.0 / (alpha ** 2).sum()) if float((alpha ** 2).sum()) > 0
                      else None),
               total_force=out["forces"].sum(dim=0).tolist(),
               seconds=time.time() - t0, n_atoms=len(atoms))
    if stress and out.get("stress") is not None:
        volume = float(abs(np.linalg.det(np.array(atoms.get_cell()))))
        rec["virial"] = (out["stress"][0] * volume).tolist()
    return rec, out["forces"].cpu().numpy()


def near_vacancy(atoms, site, radius=RADIUS):
    cell = np.array(atoms.get_cell())
    d = np.linalg.norm(_min_image(atoms.get_positions() - site, cell), axis=1)
    return np.nonzero(d < radius)[0], d


def force_map(a_small, site_small, a_big, site_big, radius=RADIUS):
    """Pairs (i_small, i_big) of atoms within `radius` of the vacancy, matched by their
    displacement from it."""
    i_s, _ = near_vacancy(a_small, site_small, radius)
    i_b, _ = near_vacancy(a_big, site_big, radius)
    rel_s = _min_image(a_small.get_positions()[i_s] - site_small, np.array(a_small.get_cell()))
    rel_b = _min_image(a_big.get_positions()[i_b] - site_big, np.array(a_big.get_cell()))
    pairs = []
    for k, r in enumerate(rel_s):
        d = np.linalg.norm(rel_b - r, axis=1)
        j = int(np.argmin(d))
        if d[j] < 0.5 and a_small.get_atomic_numbers()[i_s[k]] == a_big.get_atomic_numbers()[i_b[j]]:
            pairs.append((int(i_s[k]), int(i_b[j])))
    return pairs


def fit_exponent(lengths, values):
    """Leading exponent of |v| ~ L^s by least squares in log-log; also the 1/L fit residual."""
    L = np.asarray(lengths, dtype=float)
    v = np.asarray(values, dtype=float)
    if np.any(np.abs(v) < 1e-12):
        return dict(exponent=None, note="a value is zero")
    slope, intercept = np.polyfit(np.log(L), np.log(np.abs(v)), 1)
    # the 1/L model: v = a / L + b
    A = np.stack([1.0 / L, np.ones_like(L)], axis=1)
    coef, res, _, _ = np.linalg.lstsq(A, v, rcond=None)
    return dict(exponent=float(slope), inv_L_coefficient=float(coef[0]),
                inv_L_offset=float(coef[1]),
                inv_L_residual=float(np.sqrt(res[0] / len(L))) if res.size else 0.0)


# ------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="~/runs/arma_models/arma_s1.model")
    p.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--out", default=str(HERE / "golden" / "stage14_ladder.json"))
    p.add_argument("--data", type=Path, default=HERE / "dataset_pbe" / "train.xyz")
    p.add_argument("--no-thermal", action="store_true")
    p.add_argument("--no-switch-sweep", action="store_true")
    args = p.parse_args(argv)
    _assert_repo()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    model = load_uniform(str(Path(args.model).expanduser()))
    ctx = ForwardContext.production(model, device="cpu")
    frames = frame_selection()

    pristine_frames = [a for a in read(str(args.data), index=":") if len(a) == 80]
    ideal, stats = ideal_pristine(pristine_frames, HERE / "golden" / "ideal_pristine_80.xyz")
    print(f"ideal pristine: domain of {stats['n_domain']} of {stats['n_frames_total']} frames "
          f"(reference frame {stats['reference_frame']}, radius {DOMAIN_RADIUS} A), matched "
          f"distance mean {stats['matched_distance_mean']:.3f} A worst "
          f"{stats['matched_distance_worst']:.3f} A, cell diag "
          f"{np.round(np.diag(stats['cell']), 3).tolist()}")
    # Section 7.5: a SINGLE C_Q, no size-dependent reference. The head's c table carries one
    # constant per (charge, size) class; for the ladder every size class reads the
    # 1x column, so the only thing that changes between tilings is the box.
    head = model.spectral
    table = getattr(head, "c_shift_table", None)
    if table is not None:
        with torch.no_grad():
            c_original = table.detach().clone()
            table[:, 1:] = table[:, :1]
        print(f"c table collapsed to the 1x column per charge class: "
              f"{[round(float(x), 3) for x in table[:, 0]]} (was columns "
              f"{[[round(float(x), 3) for x in row] for row in c_original]})")
    charged_ref = frames["qp1_79"][0][2]
    neutral_ref = frames["vcl0_79"][0][2]
    frames79 = [a for a in read(str(args.data), index=":") if len(a) == 79]
    thermal, thermal_index, thermal_worst = closest_thermal(ideal, frames79)
    vac_index, perm, mdist = vacancy_index(ideal, thermal)
    result_thermal_match = dict(train_index_79=int(thermal_index), vacancy_index=vac_index,
                                counter=[int(x) for x in thermal.info["carrier_counts"]],
                                matched_distance_mean=float(mdist.mean()),
                                matched_distance_worst=float(mdist.max()))
    print(f"thermal frame: 79-atom train frame {thermal_index} (counter "
          f"{thermal.info['carrier_counts']}), vacancy at ideal Cl {vac_index}; matched "
          f"displacement mean {mdist.mean():.3f} A worst {mdist.max():.3f} A")

    result = dict(git_sha=git_sha(), model=str(args.model), ideal_pristine=stats,
                  regime="CPU, uniform float64, %d threads" % args.threads, radius=RADIUS,
                  sizes={}, thermal={}, switch_sweep={}, thermal_match=result_thermal_match)
    cells = {}
    ideal_frames = {}
    # the ladder's own compositions, counted against the same pristine reference
    ladder_cells = [vacancy_cell(ideal, reps, neutral_ref, vac_index)[0] for reps in args.reps]
    print("class table for the ladder (ideal pristine reference):")
    result["class_table"] = classes_for(model, ctx, ideal, ladder_cells)
    for reps in args.reps:
        L = float(np.cbrt(abs(np.linalg.det(np.array(ideal.get_cell()) * reps))))
        entry = dict(reps=reps, L=L)
        for label, ref in (("charged", charged_ref), ("neutral", neutral_ref)):
            atoms, site = vacancy_cell(ideal, reps, ref, vac_index)
            per, f_per = evaluate(model, ctx, atoms, "periodic")
            iso, f_iso = evaluate(model, ctx, atoms, "isolated", stress=False)
            entry[label] = dict(periodic=per, isolated=iso,
                                e_pbc_minus_e_inf=per["energy"] - iso["energy"],
                                correction_isolated=iso["band"] + iso["frontier"],
                                correction_periodic=per["band"] + per["frontier"])
            ideal_frames[(reps, label)] = (atoms, site, f_per, f_iso)
            print(f"{reps}x {label:7s} N {len(atoms)} L {L:.1f}: E {per['energy']:.4f} "
                  f"band {per['band']:+.4f} Phi {per['frontier']:+.4f} (w_ref {per['w_ref']:.3f}) "
                  f"E_PBC-E_inf {per['energy'] - iso['energy']:+.4f}  |sum F| "
                  f"{np.linalg.norm(per['total_force']):.2e}  N_eff "
                  f"{per['n_eff'] if per['n_eff'] is None else round(per['n_eff'], 1)}  "
                  f"{per['seconds']:.0f}+{iso['seconds']:.0f} s", flush=True)
        # the tiled pristine's virial, for the defect stress
        pr = ideal.repeat((reps, reps, reps))
        pr.info = dict(charged_ref.info)
        pr.info["carrier_counts"] = np.zeros(4, dtype=int)
        pr.info["multiplicity"] = 1
        pr.info["m_s_ref_doubled"] = 0
        pr.info["cell_charge"] = 0
        pr.arrays["REF_forces"] = np.zeros((len(pr), 3))
        pr.info["REF_energy"] = 0.0
        pris, _ = evaluate(model, ctx, pr, "periodic")
        entry["pristine"] = pris
        for label in ("charged", "neutral"):
            v = np.array(entry[label]["periodic"]["virial"]) - np.array(pris["virial"])
            entry[label]["defect_virial"] = v.tolist()
            entry[label]["defect_virial_trace"] = float(np.trace(v))
        cells[reps] = entry
        result["sizes"][str(reps)] = entry

    # force convergence within RADIUS of the vacancy, against the largest size
    big = max(args.reps)
    for label in ("charged", "neutral"):
        a_big, s_big, f_big, fi_big = ideal_frames[(big, label)]
        for reps in args.reps:
            a, s, f, fi = ideal_frames[(reps, label)]
            pairs = force_map(a, s, a_big, s_big)
            dper = max(float(np.linalg.norm(f[i] - f_big[j])) for i, j in pairs)
            diso = max(float(np.linalg.norm(fi[i] - fi_big[j])) for i, j in pairs)
            fmax = max(float(np.linalg.norm(f[i])) for i, _ in pairs)
            result["sizes"][str(reps)][label]["force_convergence"] = dict(
                n_atoms_within_radius=len(pairs), max_dF_periodic_vs_largest=dper,
                max_dF_isolated_vs_largest=diso, max_F_within_radius=fmax)
            print(f"  {label:7s} {reps}x vs {big}x: {len(pairs)} atoms within {RADIUS} A, "
                  f"max |dF| periodic {dper:.2e} isolated {diso:.2e} (|F| up to {fmax:.2f})")

    # exponents of E_PBC - E_inf and of Phi_FF
    if len(args.reps) >= 2:
        Ls = [cells[r]["L"] for r in args.reps]
        for label in ("charged", "neutral"):
            comp = {
                "e_pbc_minus_e_inf": [cells[r][label]["e_pbc_minus_e_inf"] for r in args.reps],
                "frontier": [cells[r][label]["periodic"]["frontier"] for r in args.reps],
                "band": [cells[r][label]["periodic"]["band"] for r in args.reps],
                "correction_isolated": [cells[r][label]["correction_isolated"] for r in args.reps],
            }
            fits = {k: dict(values=v, **fit_exponent(Ls, v)) for k, v in comp.items()}
            result["sizes"].setdefault("fits", {})[label] = fits
            for k, f in fits.items():
                print(f"  {label:7s} {k:22s} " + " ".join(f"{x:+.4f}" for x in f["values"])
                      + (f"  exponent {f['exponent']:+.2f}" if f.get("exponent") is not None
                         else "  (zero)"))

    # the thermal frame embedded at each size
    if not args.no_thermal:
        for reps in args.reps:
            atoms, site = embed_thermal(ideal, reps, thermal, charged_ref, vac_index, perm)
            mstats = result_thermal_match
            per, f_per = evaluate(model, ctx, atoms, "periodic", stress=False)
            iso, _ = evaluate(model, ctx, atoms, "isolated", stress=False)
            ideal_per = cells[reps]["charged"]["periodic"]
            result["thermal"][str(reps)] = dict(
                match=mstats, periodic=per, isolated=iso,
                e_pbc_minus_e_inf=per["energy"] - iso["energy"],
                thermal_image_contribution=(per["energy"] - iso["energy"])
                - cells[reps]["charged"]["e_pbc_minus_e_inf"])
            print(f"thermal {reps}x N {len(atoms)}: "
                  f"Phi {per['frontier']:+.4f} (ideal {ideal_per['frontier']:+.4f}) "
                  f"E_PBC-E_inf {per['energy'] - iso['energy']:+.4f}; thermal image contribution "
                  f"{result['thermal'][str(reps)]['thermal_image_contribution']:+.4f} eV", flush=True)

    # the switch steepness sweep: no discontinuity from w or Delta_s
    if not args.no_switch_sweep:
        f = model.functional
        base_dp, base_ds = f["delta_p"], f["delta_s"]
        for reps in args.reps:
            atoms, site, _, _ = ideal_frames[(reps, "charged")]
            rows = {}
            for dp in (0.5 * base_dp, base_dp, 2.0 * base_dp):
                for ds in (0.5 * base_ds, base_ds, 2.0 * base_ds):
                    f["delta_p"], f["delta_s"] = float(dp), float(ds)
                    per, _ = evaluate(model, ctx, atoms, "periodic", stress=False)
                    rows[f"dp={dp:.4f},ds={ds:.4f}"] = dict(energy=per["energy"],
                                                            frontier=per["frontier"],
                                                            w_ref=per["w_ref"])
            f["delta_p"], f["delta_s"] = base_dp, base_ds
            result["switch_sweep"][str(reps)] = rows
            phis = [r["frontier"] for r in rows.values()]
            print(f"switch sweep {reps}x: Phi over 9 (delta_p, Delta_s) settings "
                  f"{min(phis):+.4f} .. {max(phis):+.4f} eV", flush=True)

    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
