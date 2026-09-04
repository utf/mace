#!/usr/bin/env python3
"""The joint run's adoption rule, scored. Criteria 1-4 and 6; criterion 5 has its own scripts.

THE RULE IS A LEAKAGE DETECTOR FIRST, and that is why it is not simply "did the fit improve".
In the joint run the base branch is no longer frozen, so it can remove M1b's +0.36 eV/A
small-cell artefact in either of two ways: by learning the long-d region it used to
extrapolate into, which is the point, or by absorbing the carrier itself, which would improve
every aggregate number while destroying the decomposition the whole programme rests on. Both
look like success in the loss. They differ in exactly one place.

    THE DETECTOR: `d(E_label - E_base)/dd` on the seventeen charged 159-atom frames, measured
    against the TRAINED model's own base branch. Before the joint run it is -0.1338
    [-0.1446, -0.1230] against a carrier-free null of +0.0800 [-0.0503, +0.2102] -- so it is
    carrier physics, not base error. If the base ate the carrier, that slope shrinks toward
    -0.06 because the residual no longer contains what the carrier does.

Criteria, all required:

  1. LEAKAGE. Charged 159-atom energy slope within the reference CI of -0.134, and the force
     analogue within its CI of -0.1901. Shrinkage = not adopted, regardless of total fit.
  2. NEUTRAL 159. The carrier-free slope moves toward zero from +0.0800 and does not grow.
     Reported IN-SAMPLE and labelled as such: the joint run trains on these frames, so this is
     weaker than b1's out-of-fold null and "does not grow" is the part that carries weight.
  3. NEUTRAL 79, inside the neutral-dense window. Energy and force error not degraded against
     the cross-fit bases. Also in-sample for the joint model; the comparison is to a base that
     never saw the frame, so a joint model merely matching it is already doing well.
  4. F4 AS A FITTED TARGET. `d(delta_sr)/dd` on the charged 159-atom frames within 1.5x of
     -0.134. No 79-atom slope gate: b1 proved that mixture uninterpretable.
  6. DEPTH. The level stays a shallow donor -- 0.10-0.13 eV below the conduction manifold,
     moving by no more than 0.2 eV.

Every arm is regime-tagged from the model itself, and both joint-run arms are scored
separately: a from-scratch control that leaks and a staged one that does not is a result about
staging, not about the head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_label_slope_by_size import (charged_frames, fit_arm,  # noqa: E402
                                    neutral_dense_window, rows_for, show)
from b6_depth_edges import align, occupied_count, spectrum  # noqa: E402
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo, evaluate  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS, fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import capture, counts_of, load_frames, select, with_hole_counter  # noqa: E402

# The pre-joint references, every one of them measured (b1, b6), not assumed.
REF_ENERGY_SLOPE = -0.1338
REF_ENERGY_CI = (-0.1446, -0.1230)
REF_FORCE_SLOPE = -0.1901
REF_FORCE_CI = (-0.2103, -0.1700)
REF_NEUTRAL_159 = 0.0800        # b1's out-of-fold cross-fit null
# The same slope measured against the PRODUCTION base, i.e. the one a Stage-A-initialised
# joint model starts from. Criterion 2's registered reference is the cross-fit number above,
# and it stays that; this is reported beside it because it is the like-for-like comparison --
# a joint model is scored against its own base, and the two bases do not agree here. Measured
# on the frozen pre-joint cohort by this same script, which is also its null control: on a
# model that has not been jointly trained it returns the pre-joint references (charged energy
# -0.1308 against b1's -0.1338, force -0.1868 against -0.1901) and correctly refuses to adopt.
REF_NEUTRAL_159_PRODUCTION_BASE = 0.0968
LEAK_FLOOR = -0.10          # magnitude below this = the base ate the carrier (F14)
F4_TOLERANCE = 1.5          # criterion 4: within 1.5x of the reference
DEPTH_BAND = (0.10, 0.13)   # eV below the conduction manifold, pre-joint
DEPTH_MOVE_MAX = 0.2


def neutral_frames(paths, natoms=None):
    out = []
    for p in paths:
        for a in ase_read(str(p), index=":"):
            c = counts_of(a)
            if (c is None or sum(c) == 0) and (natoms is None or len(a) == natoms):
                out.append(a)
    return out


def errors_on(model, frames, z_table, cutoff, device, batch_size=4):
    """RMSE of the TOTAL energy per atom and of the forces, for criterion 3."""
    e_err, f_sq, f_n = [], 0.0, 0
    for s in range(0, len(frames), batch_size):
        chunk = frames[s:s + batch_size]
        batch, frs = make_batches(chunk, z_table, cutoff, len(chunk), device)[0]
        d = batch.to_dict()
        d["positions"].requires_grad_(True)
        out = model(d, training=False, compute_force=True)
        pred_e = out["energy"].detach().cpu().numpy().reshape(-1)
        idx = batch.batch.detach().cpu().numpy()
        pred_f = out["forces"].detach().cpu().numpy()
        ref_f = batch.forces.detach().cpu().numpy()
        for k, atoms in enumerate(frs):
            lab = atoms.info.get("REF_energy", atoms.info.get("energy"))
            if lab is not None:
                e_err.append((float(lab) - float(pred_e[k])) / len(atoms))
        f_sq += float(((pred_f - ref_f) ** 2).sum())
        f_n += ref_f.size
    return (float(np.sqrt(np.mean(np.square(e_err)))) if e_err else float("nan"),
            float(np.sqrt(f_sq / max(f_n, 1))))


def inside(value, ci):
    return bool(ci[0] <= value <= ci[1])


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--fold-prefix", default="cf_base_f",
                    help="fold bases are <cf-runs>/<prefix><k>/<prefix><k>.model")
    ap.add_argument("--reference-json", type=Path, default=None,
                    help="override the pre-joint reference constants with the ones a "
                    "re-derivation wrote (keys: energy_slope, energy_ci, force_slope, "
                    "force_ci, neutral_159, neutral_159_production_base)")
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[here / "dataset_pbe" / "train.xyz",
                             here / "dataset_pbe" / "valid.xyz"])
    ap.add_argument("--cf-runs", type=Path, default=Path.home() / "runs")
    ap.add_argument("--cf-dir", type=Path,
                    default=Path(__file__).resolve().parent / "dataset_cf",
                    help="the cross-fit folds, each holding the frames its base never saw")
    ap.add_argument("--n-neutral-79", type=int, default=200)
    ap.add_argument("--n-pristine", type=int, default=3)
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.reference_json is not None:
        # Module-level constants by design (the pre-joint rule was registered as literals);
        # a re-derived reference replaces them here, once, before anything is scored.
        ref = json.load(open(args.reference_json))
        g = globals()
        g["REF_ENERGY_SLOPE"] = float(ref["energy_slope"])
        g["REF_ENERGY_CI"] = tuple(float(x) for x in ref["energy_ci"])
        g["REF_FORCE_SLOPE"] = float(ref["force_slope"])
        g["REF_FORCE_CI"] = tuple(float(x) for x in ref["force_ci"])
        g["REF_NEUTRAL_159"] = float(ref["neutral_159"])
        g["REF_NEUTRAL_159_PRODUCTION_BASE"] = float(ref.get(
            "neutral_159_production_base", ref["neutral_159"]))
        # The leakage floor scales with the reference: the registered -0.10 was 0.75 x the
        # pre-joint -0.1338, and a literal floor above a smaller reference would flag the
        # frozen base's own slope as a leak.
        g["LEAK_FLOOR"] = float(ref.get("leak_floor", 0.75 * ref["energy_slope"]))
        print(f"reference constants from {args.reference_json}: leak floor {LEAK_FLOOR:+.4f}; "
              f"energy {REF_ENERGY_SLOPE:+.4f} "
              f"{REF_ENERGY_CI}, force {REF_FORCE_SLOPE:+.4f} {REF_FORCE_CI}, neutral "
              f"{REF_NEUTRAL_159:+.4f} / {REF_NEUTRAL_159_PRODUCTION_BASE:+.4f}", flush=True)

    _assert_repo()
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    charged = charged_frames(args.data)
    big_charged = [a for a in charged if len(a) == CLEAN_NATOMS]
    big_neutral = neutral_frames(args.data, CLEAN_NATOMS)
    dense, _ = neutral_dense_window(args.data, 79)
    # Criterion 3's frames come from each fold's OWN held-out pool, keyed by fold, not from a
    # `i % 4` partition of the whole neutral set. The partition was wrong in a way that made
    # the baseline look better than it is: fold base k has seen three quarters of any such
    # partition, so most of "its" frames were in its training set and the reference error was
    # partly in-sample. `null_oof.xyz` is what each base actually never saw.
    small_by_fold = {}
    for k in range(4):
        pool = args.cf_dir / f"fold{k}" / "null_oof.xyz"
        if not pool.exists():
            continue
        mine = [a for a in ase_read(str(pool), index=":")
                if len(a) == 79 and dense
                and dense[0] <= hub_separation(a) <= dense[1]]
        small_by_fold[k] = mine[:max(args.n_neutral_79 // 4, 1)]
    small_neutral = [a for k in sorted(small_by_fold) for a in small_by_fold[k]]
    if not small_neutral:
        raise SystemExit(
            f"no out-of-fold neutral 79-atom frames inside {dense}; criterion 3 has no "
            "honest baseline and would otherwise be scored against an in-sample one")
    all_pristine = select_pristine(ase_read(str(args.data[0]), ":"), 64)
    biggest = max(len(a) for a in all_pristine)
    pristine = with_hole_counter([a for a in all_pristine
                                  if len(a) == biggest][:args.n_pristine])
    print(f"{len(big_charged)} charged and {len(big_neutral)} neutral "
          f"{CLEAN_NATOMS}-atom frames; {len(small_neutral)} neutral 79-atom inside "
          f"{dense[0]:.2f}-{dense[1]:.2f} A; {len(pristine)} pristine of {biggest}",
          flush=True)

    # ------------------------------------------------- criterion 3's baseline: the cf bases
    cf_e, cf_f = [], []
    for k in range(4):
        path = args.cf_runs / f"{args.fold_prefix}{k}" / f"{args.fold_prefix}{k}.model"
        if not path.exists():
            continue
        base = torch.load(path, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        # The batches built for this model must carry ITS dtype: AtomicData uses the
        # process default, which is float32, while the joint run trains at float64.
        adopt_model_dtype(base)
        mine = small_by_fold.get(k, [])
        if mine:
            e, f = errors_on(base, mine, z, 5.0, args.device)
            cf_e.append(e)
            cf_f.append(f)
        del base
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    baseline = (float(np.mean(cf_e)) if cf_e else float("nan"),
                float(np.mean(cf_f)) if cf_f else float("nan"))
    print(f"  cross-fit baseline on the neutral-dense window: "
          f"E/atom {baseline[0] * 1000:.1f} meV, F {baseline[1] * 1000:.1f} meV/A "
          f"(genuinely out-of-fold, {len(small_neutral)} frames)", flush=True)

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        # The batches built for this model must carry ITS dtype: AtomicData uses the
        # process default, which is float32, while the joint run trains at float64.
        adopt_model_dtype(model)
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        head = getattr(model, "spectral", None)
        row = {"model": Path(mp).name,
               "regime": dict(gamma=float(head.h.on_site_range),
                              hop_form=getattr(head.h, "hop_form", "linear"),
                              envelope=getattr(head.h, "envelope", "exp"),
                              family=getattr(head, "smearing_family", "?"),
                              width=float(head.t_el),
                              c_shift=float(head.c_shift))}

        # ------------------------------------------------------ criteria 1 and 2
        for tag, frames in (("charged_159", big_charged), ("neutral_159", big_neutral)):
            got = rows_for(model, frames, z, cutoff, args.device, 4)
            row[f"{tag}_energy"] = fit_arm(got, "de", tag)
            row[f"{tag}_force"] = fit_arm(got, "axial", tag)

        # -------------------------------------------------------------- criterion 3
        e, f = errors_on(model, small_neutral, z, cutoff, args.device)
        row["neutral_79_window"] = dict(energy_per_atom=e, force=f,
                                        baseline_energy=baseline[0],
                                        baseline_force=baseline[1])

        # -------------------------------------------------------------- criterion 4
        sr, dd = [], []
        for b, fr in make_batches(big_charged, z, cutoff, 1, args.device):
            _, out = capture(model, b, ctx=ctx, frames=fr)
            sr.append(float(out["delta_sr_energy"].reshape(-1)[0]))
            dd.append(hub_separation(fr[0]))
        sr, dd = np.array(sr), np.array(dd)
        good = np.isfinite(sr) & np.isfinite(dd)
        s4, lo4, hi4, c4 = fit_with_ci(dd[good], sr[good])
        row["f4"] = dict(slope=s4, ci=[lo4, hi4], corr=c4)

        # -------------------------------------------------------------- criterion 6
        n_pri = occupied_count(pristine)
        pri, vbm, cbm = [], [], []
        for b, fr in make_batches(pristine, z, cutoff, 1, args.device):
            lam = spectrum(model, b, fr, ctx)
            pri.append(lam)
            vbm.append(float(lam[n_pri - 1]))
            cbm.append(float(lam[n_pri]))
        pri_mean = np.mean(pri, axis=0)
        from mace.modules.defect_counting import VALENCE, changed_level_index

        n_def = occupied_count(big_charged)
        depths = []
        for b, fr in make_batches(big_charged, z, cutoff, 1, args.device):
            lam = spectrum(model, b, fr, ctx)
            n_total = sum(VALENCE[int(zz)] for zz in fr[0].get_atomic_numbers())
            k = changed_level_index(n_total, b.carrier_counts.reshape(-1).tolist())
            if not 0 <= k < lam.size:
                continue
            shift, _ = align(lam, n_def, pri_mean, n_pri)
            depths.append(float(np.mean(cbm)) + shift - float(lam[k]))
        row["depth"] = dict(from_cbm=float(np.mean(depths)) if depths else float("nan"),
                            pristine_gap=float(np.mean(cbm) - np.mean(vbm)))

        # ------------------------------------------------------------- the scoring
        ce, cf_ = row["charged_159_energy"], row["charged_159_force"]
        nn = row["neutral_159_energy"]
        verdict = dict(
            c1_leakage=bool(inside(ce["slope"], REF_ENERGY_CI)
                            and inside(cf_["slope"], REF_FORCE_CI)),
            # F14's clause, and the sign needs care: the slope is negative, so "shrunk"
            # means LESS negative than the floor. `<= -0.10` is the healthy case.
            c1_not_shrunk=bool(ce["slope"] <= LEAK_FLOOR),
            c2_neutral_toward_zero=bool(abs(nn["slope"]) <= abs(REF_NEUTRAL_159)),
            c3_not_degraded=bool(e <= baseline[0] * 1.1 and f <= baseline[1] * 1.1),
            c4_f4_in_band=bool(s4 < 0 and abs(s4) >= abs(REF_ENERGY_SLOPE) / F4_TOLERANCE
                               and abs(s4) <= abs(REF_ENERGY_SLOPE) * F4_TOLERANCE),
            c6_shallow=bool(np.isfinite(row["depth"]["from_cbm"])
                            and abs(row["depth"]["from_cbm"]
                                    - np.mean(DEPTH_BAND)) <= DEPTH_MOVE_MAX),
            c5_gap=bool(abs(row["depth"]["pristine_gap"] - 2.4) <= 0.1))
        verdict["adopted"] = bool(all(v for k, v in verdict.items()
                                      if k.startswith(("c1_leakage", "c2", "c3", "c4",
                                                       "c5", "c6"))))
        row["verdict"] = verdict
        rows.append(row)

        print(f"\n  {row['model']}   c_shift {row['regime']['c_shift']:+.4f}")
        print(f"    1  charged 159 energy {ce['slope']:+.4f} "
              f"[{ce['ci'][0]:+.4f}, {ce['ci'][1]:+.4f}] vs {REF_ENERGY_SLOPE:+.4f} "
              f"{'OK' if verdict['c1_leakage'] else 'OUTSIDE THE REFERENCE CI'}"
              f"{'' if verdict['c1_not_shrunk'] else '  <- SHRUNK past -0.10: F14 leakage'}")
        print(f"       charged 159 force  {cf_['slope']:+.4f} vs {REF_FORCE_SLOPE:+.4f}")
        print(f"    2  neutral 159 energy {nn['slope']:+.4f} vs {REF_NEUTRAL_159:+.4f} "
              f"(cross-fit) / {REF_NEUTRAL_159_PRODUCTION_BASE:+.4f} (production base)  "
              f"{'toward zero' if verdict['c2_neutral_toward_zero'] else 'GREW'}")
        print(f"    3  neutral-79 window  E {e * 1000:.1f} meV/atom "
              f"(base {baseline[0] * 1000:.1f}), F {f * 1000:.1f} meV/A "
              f"(base {baseline[1] * 1000:.1f})  "
              f"{'OK' if verdict['c3_not_degraded'] else 'DEGRADED'}")
        print(f"    4  F4 delta_sr slope  {s4:+.4f} [{lo4:+.4f}, {hi4:+.4f}]  "
              f"{'OK' if verdict['c4_f4_in_band'] else 'OUT OF BAND'}")
        print(f"    5  pristine gap {row['depth']['pristine_gap']:.3f} eV  "
              f"{'OK' if verdict['c5_gap'] else 'FAIL'}")
        print(f"    6  depth from CBM {row['depth']['from_cbm']:+.3f} eV  "
              f"{'shallow donor' if verdict['c6_shallow'] else 'MOVED'}")
        print(f"    -> {'ADOPTED' if verdict['adopted'] else 'NOT ADOPTED'}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        print("\n=== pooled ===")
        for key, label, ref in (("charged_159_energy", "charged 159 energy slope",
                                 REF_ENERGY_SLOPE),
                                ("charged_159_force", "charged 159 force slope",
                                 REF_FORCE_SLOPE),
                                ("neutral_159_energy", "neutral 159 energy slope",
                                 REF_NEUTRAL_159)):
            v = np.array([r[key]["slope"] for r in rows])
            print(f"  {label:26s} {v.mean():+.4f} +- {v.std():.4f}   "
                  f"(pre-joint {ref:+.4f})")
        v = np.array([r["f4"]["slope"] for r in rows])
        print(f"  {'F4 delta_sr slope':26s} {v.mean():+.4f} +- {v.std():.4f}")
        n = sum(r["verdict"]["adopted"] for r in rows)
        print(f"\n  adopted by all six criteria: {n}/{len(rows)}")
        print("  Criterion 1 is the one that cannot be traded against the others: a base that "
              "removed\n  M1b's artefact by absorbing the carrier improves every aggregate "
              "number and destroys\n  the decomposition. Shrinkage there is not adopted "
              "regardless of total fit.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
