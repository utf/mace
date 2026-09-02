#!/usr/bin/env python3
"""F6: do the 79-atom labels contain the 159-atom d-trend at all?

WHY THIS EXISTS. F4 compares the head's own energy slope against -0.134 eV/A, a number
measured on the SEVENTEEN charged 159-atom frames. Everything downstream then treated that
reference as size-transferable, and the retraction on record says plainly that it was never
measured: the 79-atom frames are 98% of the charged training set, so if THEY carry a slope
near zero the head is fitting a mixture in which the 159-atom trend is a rounding error, and
"the head is short" is the wrong diagnosis.

TWO OBSERVABLES, DELIBERATELY NOT ONE.

  * FORCE. The axial pair-force residual (R0's statistic), `0.5 * [(F_DFT - F_base)_b -
    (F_DFT - F_base)_a] . axis`, regressed on d. This is what the plan asked for, and it has
    never been fitted as a slope per size -- T2 binned it, and a binned median hides a trend.
  * ENERGY. `E_label - E_base` regressed on d. This is the observable -0.134 was measured in,
    so it is the one the decision tree's "79-atom labels carry ~ -0.13" can be scored against.
    Quoting a force slope against an energy reference would be comparing eV/A to eV/A and
    meaning two different things by it.

CROSS-FIT BASES, per the plan. Each frame is scored by one of the four T2 fold bases,
assigned `i mod 4` WITHIN its size, so no frame is scored by a base that saw it and each base
carries equal weight at both sizes. A size-level assignment would put the entire 17-frame arm
on one base and confound base identity with size. Charged frames are out-of-sample for every
base regardless -- none was trained on a charged frame -- so the partition is about weighting,
not leakage. The 159-atom arm is ALSO fitted against the production base `e0_base_s1`, because
that is the base -0.134 was measured against and swapping it silently would make the two
numbers incomparable.

MATCHED-d. The two sizes overlap on roughly 4.9-6.8 A but not beyond, and a slope fitted over
a wider range at one size than the other is a range difference wearing a size label. The
matched window is the intersection; both the full-range and matched-window fits are reported.

PRE-REGISTERED SCORING RULE, fixed before any number exists, because n ~ 1030 against n = 17
makes "materially smaller" slippery: F6 HOLDS if the two sizes' 95% slope intervals are
DISJOINT and the 79-atom |slope| is the smaller. Overlapping intervals score as NOT RESOLVED,
which is a third outcome and not a pass.

Evaluation-only: the vacancy assignment defines the axis and never reaches a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, evaluate  # noqa: E402
from r0_pair_force import axis_and_residuals  # noqa: E402
from s3_dehead_trend import fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import counts_of  # noqa: E402

SIZES = (79, 159)
REFERENCE_ENERGY_SLOPE = -0.134       # M1b, 159-atom charged subset, production base


def charged_frames(paths):
    out = []
    for p in paths:
        for a in read(str(p), index=":"):
            c = counts_of(a)
            if c is not None and sum(c) != 0:
                out.append(a)
    return out


def neutral_dense_window(paths, natoms, lo_q=5.0, hi_q=95.0):
    """The d range the NEUTRAL defective frames actually populate, at one cell size.

    M1b's finding is that the 79-atom charged energy residual correlates with d at +0.447
    over the full range and only +0.062 inside the neutral-dense window -- i.e. most of the
    apparent trend is the frozen base extrapolating into geometries its training set does not
    cover, not label content. A slope fitted over the full range therefore measures base error
    and reports it as a property of the labels. This computes the window from the neutral
    frames' own d distribution rather than hardcoding M1b's numbers, so it travels if the
    dataset does.
    """
    d = []
    for p in paths:
        for a in read(str(p), index=":"):
            c = counts_of(a)
            if (c is None or sum(c) == 0) and len(a) == natoms:
                v = hub_separation(a)
                if np.isfinite(v):
                    d.append(v)
    if len(d) < 50:
        return None, 0
    d = np.asarray(d)
    return (float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q))), int(d.size)


def rows_for(model, frames, z_table, cutoff, device, batch_size):
    """Per-frame d, axial force residual, and energy residual, for one base."""
    base_f, base_e = evaluate(model, frames, z_table, cutoff, device,
                              batch_size=batch_size)
    out = []
    for k, a in enumerate(frames):
        r = axis_and_residuals(a, base_f[k])
        if r is None:
            continue
        e_label = a.info.get("REF_energy", a.info.get("energy"))
        if e_label is None:
            continue
        out.append(dict(natoms=len(a), d=float(r["d_pbpb"]), axial=float(r["axial"]),
                        de=float(e_label) - float(base_e[k])))
    return out


def fit_arm(rows, key, label, window=None):
    d = np.array([r["d"] for r in rows], dtype=float)
    v = np.array([r[key] for r in rows], dtype=float)
    good = np.isfinite(d) & np.isfinite(v)
    if window is not None:
        good &= (d >= window[0]) & (d <= window[1])
    if int(good.sum()) < 5:
        return dict(label=label, n=int(good.sum()), slope=float("nan"),
                    ci=[float("nan"), float("nan")], corr=float("nan"))
    slope, lo, hi, corr = fit_with_ci(d[good], v[good])
    return dict(label=label, n=int(good.sum()), slope=slope, ci=[lo, hi], corr=corr,
                d_range=[float(d[good].min()), float(d[good].max())])


def show(fit):
    print(f"    {fit['label']:36s} n {fit['n']:5d}  slope {fit['slope']:+.4f} "
          f"[{fit['ci'][0]:+.4f}, {fit['ci'][1]:+.4f}]  corr {fit['corr']:+.3f}", flush=True)


def disjoint(a, b):
    """Do two 95% intervals fail to overlap?"""
    if not all(np.isfinite(x) for x in (*a["ci"], *b["ci"])):
        return False
    return a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0]


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[here / "dataset_pbe" / "train.xyz",
                             here / "dataset_pbe" / "valid.xyz"])
    ap.add_argument("--runs", type=Path, default=Path.home() / "runs")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--production-base", default="e0_base_s1")
    ap.add_argument("--null-dir", type=Path,
                    default=Path(__file__).resolve().parent / "dataset_cf",
                    help="cross-fit folds holding each base's held-out NEUTRAL frames")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    frames = charged_frames(args.data)
    by_size = {n: [a for a in frames if len(a) == n] for n in SIZES}
    print(f"{len(frames)} charged frames: " +
          ", ".join(f"{n} atoms x{len(v)}" for n, v in by_size.items()), flush=True)

    # --------------------------------------------------------------- cross-fit arms
    rows = []
    for k in range(args.folds):
        path = args.runs / f"cf_base_f{k}" / f"cf_base_f{k}.model"
        if not path.exists():
            raise SystemExit(f"missing fold base {path}; run run_crossfit_bases.sh first")
        model = torch.load(path, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        for n in SIZES:
            mine = [a for i, a in enumerate(by_size[n]) if i % args.folds == k]
            if not mine:
                continue
            got = rows_for(model, mine, z_table, args.cutoff, args.device, args.batch_size)
            rows += got
            print(f"  fold {k}, {n} atoms: {len(got)}/{len(mine)} usable", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- the neutral null
    #
    # THE CONTROL THE -0.134 REFERENCE HAS NEVER HAD. Both label arms are E_label - E_base,
    # so both inherit whatever the base gets wrong at those geometries, and the 79-atom arm
    # demonstrably does: its slope collapses from +0.36 to consistent-with-zero once the fit
    # is restricted to the window where the base's own training set is dense. Only 4 of the
    # 17 large frames sit inside that window, so the same question has to be asked of them --
    # and it cannot be asked by restricting the range, because there is nothing left to fit.
    #
    # It can be asked with a null. The neutral defective cells carry the same vacancy, the
    # same d(Pb-Pb) and no carrier, so ANY d-trend in their residual is base error by
    # construction. If the large-cell null is flat while the charged arm is -0.134, the
    # reference is carrier physics. If the null carries the same slope, F4 has been scored
    # against an artefact.
    #
    # The frames come from the fold's own held-out pool, so each is scored by a base that
    # never saw it -- an in-sample null would report the base's memory, not its error.
    null_rows = []
    if args.null_dir is not None and args.null_dir.exists():
        for k in range(args.folds):
            pool = args.null_dir / f"fold{k}" / "null_oof.xyz"
            path = args.runs / f"cf_base_f{k}" / f"cf_base_f{k}.model"
            if not pool.exists() or not path.exists():
                continue
            model = torch.load(path, map_location=args.device,
                               weights_only=False).to(args.device).eval()
            frames_k = read(str(pool), index=":")
            for n in SIZES:
                mine = [a for a in frames_k if len(a) == n]
                if not mine:
                    continue
                null_rows += rows_for(model, mine, z_table, args.cutoff, args.device,
                                      args.batch_size)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        print(f"  neutral out-of-fold null: " + ", ".join(
            f"{n} atoms x{sum(1 for r in null_rows if r['natoms'] == n)}" for n in SIZES),
            flush=True)

    # ------------------------------------------------------- production-base 159 arm
    prod_rows = []
    prod = args.runs / args.production_base / f"{args.production_base}.model"
    if prod.exists():
        model = torch.load(prod, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        prod_rows = rows_for(model, by_size[159], z_table, args.cutoff, args.device,
                             args.batch_size)
        print(f"  production base {args.production_base}, 159 atoms: "
              f"{len(prod_rows)}/{len(by_size[159])} usable", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        print(f"  production base {prod} absent; the tie to the {REFERENCE_ENERGY_SLOPE} "
              "reference is UNMEASURED, not passed", flush=True)

    # -------------------------------------------------------------------- the window
    per_size = {n: [r for r in rows if r["natoms"] == n] for n in SIZES}
    spans = {n: (min(r["d"] for r in v), max(r["d"] for r in v))
             for n, v in per_size.items() if v}
    if len(spans) < 2:
        raise SystemExit(f"only one size produced usable rows: {list(spans)}")
    window = (max(s[0] for s in spans.values()), min(s[1] for s in spans.values()))
    print("\n  d spans: " + "  ".join(f"{n}: {lo:.2f}-{hi:.2f}"
                                      for n, (lo, hi) in spans.items()))
    print(f"  matched window {window[0]:.2f}-{window[1]:.2f} A", flush=True)

    dense, n_neutral = neutral_dense_window(args.data, 79)
    if dense:
        print(f"  neutral-dense window at 79 atoms {dense[0]:.2f}-{dense[1]:.2f} A "
              f"(5th-95th percentile of {n_neutral} neutral defective frames)", flush=True)

    payload = {"window": list(window),
               "neutral_dense_window_79": list(dense) if dense else None,
               "n_neutral_79": n_neutral,
               "n_by_size": {str(n): len(v) for n, v in per_size.items()},
               "reference_energy_slope": REFERENCE_ENERGY_SLOPE}
    verdicts = {}
    for key, name, unit in (("axial", "axial pair-force residual", "(eV/A) per A"),
                            ("de", "E_label - E_base", "eV/A")):
        print(f"\n=== {name} vs d, slope in {unit} ===")
        fits = {}
        for n in SIZES:
            if not per_size[n]:
                continue
            fits[f"{n}_full"] = fit_arm(per_size[n], key, f"{n} atoms, full range")
            fits[f"{n}_matched"] = fit_arm(per_size[n], key, f"{n} atoms, matched window",
                                           window)
            show(fits[f"{n}_full"])
            show(fits[f"{n}_matched"])
            if n == 79 and dense:
                fits["79_neutral_dense"] = fit_arm(per_size[n], key,
                                                   "79 atoms, neutral-dense window", dense)
                show(fits["79_neutral_dense"])
        if prod_rows:
            fits["159_prod"] = fit_arm(prod_rows, key, "159 atoms, production base")
            show(fits["159_prod"])
        for n in SIZES:
            mine = [r for r in null_rows if r["natoms"] == n]
            if len(mine) >= 5:
                fits[f"{n}_neutral_null"] = fit_arm(
                    mine, key, f"{n} atoms, NEUTRAL null (no carrier)")
                show(fits[f"{n}_neutral_null"])
        payload[key] = fits

        a, b = fits.get("79_matched"), fits.get("159_matched")
        if a and b and np.isfinite(a["slope"]) and np.isfinite(b["slope"]):
            dis = disjoint(a, b)
            smaller = abs(a["slope"]) < abs(b["slope"])
            # A THIRD OUTCOME THE PRE-REGISTERED RULE DID NOT ANTICIPATE, kept separate
            # rather than folded into "fails". F6 was written assuming the small cells carry
            # a weaker version of the same trend, so it scores only magnitude. Opposite SIGNS
            # are neither "holds" nor "79 carries it": they say the two sizes disagree about
            # the direction, which is a different finding and has to be reported as one.
            opposite = np.sign(a["slope"]) != np.sign(b["slope"])
            verdict = ("OPPOSITE SIGN (F6's premise fails)" if (dis and opposite) else
                       "F6 HOLDS" if (dis and smaller) else
                       "F6 FAILS (79 carries it)" if dis else
                       "NOT RESOLVED (intervals overlap)")
            verdicts[key] = verdict
            print(f"    -> matched-window intervals "
                  f"{'DISJOINT' if dis else 'OVERLAP'}; "
                  f"79 {a['slope']:+.4f} vs 159 {b['slope']:+.4f}  ->  {verdict}")
    payload["verdicts"] = verdicts
    # The null's verdict on the reference, stated rather than left to the reader.
    null159 = payload.get("de", {}).get("159_neutral_null")
    chg159 = payload.get("de", {}).get("159_prod")
    if null159 and chg159 and np.isfinite(null159["slope"]):
        clean = not (null159["ci"][0] <= chg159["slope"] <= null159["ci"][1])
        payload["reference_survives_null"] = bool(clean)
        print("\n  THE NULL ON THE REFERENCE: neutral 159 slope "
              f"{null159['slope']:+.4f} [{null159['ci'][0]:+.4f}, "
              f"{null159['ci'][1]:+.4f}] against the charged {chg159['slope']:+.4f}")
        print("  -> the -0.134 reference is " + (
            "CARRIER PHYSICS (the null does not contain it)" if clean
            else "INSIDE THE NULL -- it may be base error"))
    args.out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {args.out}")
    print("\n  F6's scoring rule was fixed before the run: disjoint 95% intervals with the\n"
          "  79-atom |slope| smaller. Overlap is NOT RESOLVED, not a pass -- with 17 frames\n"
          "  in the large arm the interval is wide enough that an overlap says little.")


if __name__ == "__main__":
    main()
