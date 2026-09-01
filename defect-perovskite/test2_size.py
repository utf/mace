"""Test 2 (DFT side): does the axial hub residual DILUTE when the cell doubles?

This is the only observable in hand that identifies physical boundness. A bound carrier's
force on its own atoms is size-invariant: R_DFT = |axial|_79 / |axial|_159 ~ 1. A band state's
hub amplitude halves when the cell doubles, and so does the hub force: R_DFT ~ 2.

Model-independent by construction -- F_DFT minus the cross-fit diagnostic bases -- so the
branch decision cannot inherit any model-side defect. R_model is computed separately, and only
once the head's overshoot is understood.

Design forced by the inventory: 17 charged frames at 159 atoms against 1030 at 79, and the
large frames sit long (median d 6.11 A vs 5.49 A). Per-bin ratios would rest on 1-2 samples,
so instead each 159-atom frame is MATCHED to the 79-atom frames within +/- 0.05 A of its own
d(Pb-Pb) -- with ~1000 candidates every frame finds matches -- and compared against that
matched set's median. The 17 ratios are then bootstrapped for a CI. That controls d without
needing bin populations.

The neutral out-of-fold null runs through the identical pipeline at both sizes: base
extrapolation error must not itself change with cell size, or the charged ratio means nothing.

Readings (plan section 4):
  R_DFT ~ 1        -> bound
  R_DFT ~ 2        -> band-like
  R_DFT ~ 1.3-1.6  -> partial binding; Branch B item 1 only (upweight two-size frames),
                      NOT the dilution constraint, which asserts full boundness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from ase.geometry import get_distances
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, evaluate  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402


def counters_of(atoms):
    c = atoms.info.get("carrier_counts")
    if isinstance(c, str):
        c = [int(x) for x in c.split()]
    return np.asarray(c, dtype=int)


def axial_residual(atoms, base_forces):
    """0.5 * (dF_b - dF_a) . axis, r0_pair_force's convention. Positive = pushed apart."""
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    a, b = int(site.shell[0]), int(site.shell[1])
    pos = atoms.get_positions()
    vec, dist = get_distances(pos[a][None], pos[b][None], cell=atoms.get_cell(), pbc=atoms.pbc)
    axis = vec[0, 0] / max(float(dist[0, 0]), 1e-12)
    dF = atoms.arrays["REF_forces"] - base_forces
    return dict(d=float(dist[0, 0]), n_atoms=len(atoms),
                axial=0.5 * (float(np.dot(dF[b], axis)) - float(np.dot(dF[a], axis))))


def score(model, frames, z_table, cutoff, device, batch=4):
    base, _ = evaluate(model, frames, z_table, cutoff, device, batch_size=batch)
    return [r for r in (axial_residual(a, base[k]) for k, a in enumerate(frames))
            if r is not None]


def matched_ratios(big, small, tol, rng, n_boot=10000):
    """Each large-cell frame against the median |axial| of same-d small-cell frames."""
    sd = np.array([r["d"] for r in small])
    sa = np.abs([r["axial"] for r in small])
    ratios, detail = [], []
    for r in big:
        m = np.abs(sd - r["d"]) <= tol
        if m.sum() < 3:
            detail.append(dict(d=r["d"], n_match=int(m.sum()), ratio=None))
            continue
        ref = float(np.median(sa[m]))
        if ref <= 0:
            continue
        ratio = ref / max(abs(r["axial"]), 1e-30)      # |axial|_79 / |axial|_159
        ratios.append(ratio)
        detail.append(dict(d=r["d"], n_match=int(m.sum()), small_med=ref,
                           big=abs(r["axial"]), ratio=ratio))
    ratios = np.array(ratios)
    if len(ratios) == 0:
        return None
    boot = np.array([np.median(rng.choice(ratios, len(ratios), replace=True))
                     for _ in range(n_boot)])
    return dict(n=len(ratios), median=float(np.median(ratios)),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                mean=float(ratios.mean()), detail=detail)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe")
    ap.add_argument("--folds-dir", type=Path, default=here / "dataset_cf")
    ap.add_argument("--runs", type=Path, default=Path.home() / "runs")
    ap.add_argument("--folds", type=int, default=4)
    # +/- 0.10 A recovers the four long-d large-cell frames that found no match at 0.05.
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=here / "test2_size.json")
    args = ap.parse_args()

    _assert_repo()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    rng = np.random.default_rng(args.seed)

    bases = []
    for k in range(args.folds):
        p = args.runs / f"cf_base_f{k}" / f"cf_base_f{k}.model"
        if not p.exists():
            raise SystemExit(f"missing fold base {p}")
        bases.append(torch.load(p, map_location=args.device,
                                weights_only=False).to(args.device).eval())

    frames = read(args.data / "train.xyz", ":") + read(args.data / "valid.xyz", ":")
    charged = [a for a in frames if counters_of(a).any()]

    # Charged frames are out-of-sample for every fold base (no fold trained on any charged
    # frame), so partitioning across folds keeps each frame counted once with equal weight.
    ch_rows = []
    for k in range(args.folds):
        mine = [a for i, a in enumerate(charged) if i % args.folds == k]
        ch_rows += score(bases[k], mine, z_table, args.cutoff, args.device)
        print(f"  charged fold {k}: {len(mine)} frames", flush=True)

    # Neutral null: each frame scored by the base that held it out.
    nl_rows = []
    for k in range(args.folds):
        held = read(args.folds_dir / f"fold{k}" / "null_oof.xyz", ":")
        nl_rows += score(bases[k], held, z_table, args.cutoff, args.device)
        print(f"  null fold {k}: {len(held)} frames", flush=True)

    out = {}
    for name, rows in (("charged", ch_rows), ("null", nl_rows)):
        big = [r for r in rows if r["n_atoms"] >= 100]
        small = [r for r in rows if r["n_atoms"] < 100]
        res = matched_ratios(big, small, args.tol, rng)
        out[name] = dict(n_big=len(big), n_small=len(small), result=res)
        print(f"\n=== {name}: {len(big)} large-cell frames, {len(small)} small ===")
        if res is None:
            print("  no matched pairs")
            continue
        print(f"  matched |axial|_79 / |axial|_159 over {res['n']} frames "
              f"(+/- {args.tol} A in d)")
        print(f"  median {res['median']:.2f}   95% CI [{res['lo']:.2f}, {res['hi']:.2f}]"
              f"   mean {res['mean']:.2f}")

    # The control that actually protects R_DFT: how large is base extrapolation error at 159
    # atoms COMPARED WITH the charged signal there? The null's own 79/159 ratio is a ratio of
    # two small noisy numbers, so its wide CI is expected and is not evidence that base error
    # scales with size. This magnitude comparison is the meaningful check.
    def med_abs(rows, big):
        sel = [abs(r["axial"]) for r in rows
               if (r["n_atoms"] >= 100) == big]
        return float(np.median(sel)) if sel else float("nan")

    mag = dict(charged_159=med_abs(ch_rows, True), null_159=med_abs(nl_rows, True),
               charged_79=med_abs(ch_rows, False), null_79=med_abs(nl_rows, False))
    mag["null_over_charged_159"] = mag["null_159"] / max(mag["charged_159"], 1e-30)
    mag["null_over_charged_79"] = mag["null_79"] / max(mag["charged_79"], 1e-30)
    out["magnitudes"] = mag
    print("\n=== null magnitude control (must be << 1; target < 0.2) ===")
    print(f"  at 159 atoms: |null| {mag['null_159']:.4f}  |charged| {mag['charged_159']:.4f}"
          f"   ratio {mag['null_over_charged_159']:.3f}")
    print(f"  at  79 atoms: |null| {mag['null_79']:.4f}  |charged| {mag['charged_79']:.4f}"
          f"   ratio {mag['null_over_charged_79']:.3f}")

    r = out["charged"]["result"]
    if r:
        print("\n=== Test 2 (DFT side) reading ===")
        print(f"  R_DFT = {r['median']:.2f}  CI [{r['lo']:.2f}, {r['hi']:.2f}]")
        verdict = ("BOUND (size-invariant)" if r["hi"] < 1.3 else
                   "BAND-LIKE (dilutes ~2x)" if r["lo"] > 1.7 else
                   "PARTIAL -- Branch B item 1 only, no dilution constraint"
                   if r["lo"] > 1.2 else "INCONCLUSIVE at this n")
        print(f"  -> {verdict}")
        n = out["null"]["result"]
        if n:
            print(f"  null control R = {n['median']:.2f} CI [{n['lo']:.2f}, {n['hi']:.2f}]"
                  "  (must be ~1: base error may not itself scale with size)")

    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
