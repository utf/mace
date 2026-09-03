#!/usr/bin/env python3
"""Section 3, gate 2: does the head's OWN hub force dilute when the cell doubles?

THE ONLY OBSERVABLE THAT IDENTIFIES BOUNDNESS. A bound carrier's force on its own atoms is
size-invariant: R = |axial|_79 / |axial|_159 ~ 1. A band state's hub amplitude halves when the
cell doubles, and its hub force with it: R ~ 2. N_eff cannot make this distinction -- it
measures spread over a fixed cell and says nothing about what happens when the cell grows.

R_MODEL, NOT R_DFT. `test2_size.py` scores the DFT residual against cross-fit bases and is
model-independent by construction. This scores the MODEL's own carrier force, `forces -
base_forces`, on the same frames through the same matching -- so the two are directly
comparable and a disagreement is a statement about the head.

THE MATCHING IS `test2_size.matched_ratios`, VERBATIM. Seventeen 159-atom charged frames
against 1030 at 79, and the large frames sit long (median d 6.11 A against 5.49 A), so
per-bin ratios would rest on one or two samples. Each large frame is matched to the small
frames within +/- tol of its own d(Pb-Pb) and compared to that set's median; the ratios are
then bootstrapped. Reusing the function rather than reimplementing it is the point: a
difference between R_DFT and R_model must be the models, not two matching schemes.

BOUND-CONDITIONED. A ratio only means "bound versus band" for a frame whose defect level is
actually split off. The condition is `depth / delta_L > 2`:

  depth    lam[k+1] - lam[k] on the CHARGED frame, where k is the level whose occupation the
           counters changed. For a state split off below a band, the separation from the next
           level up IS its depth below the band edge.
  delta_L  the median level spacing just above the frontier of this model's PRISTINE
           spectrum -- the trained model's own, not the initialisation's, since delta_L is a
           property of the bands the head ended up with.

Frames that fail it are reported, not silently dropped: the bound fraction is a result.

Gate: bound-conditioned median R <= 1.3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.geometry import get_distances
from ase.io import read as ase_read

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS  # noqa: E402
from ta_band_edge import capture, load_frames, select  # noqa: E402
from test2_size import matched_ratios  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

SMALL_NATOMS = 79
BOUND_MIN = 2.0          # depth in level spacings
GATE_MAX_RATIO = 1.3


def hub_axis(atoms):
    """(index_a, index_b, unit axis, separation) for the two Pb flanking the vacancy."""
    site = locate_vacancy(atoms)
    a, b = int(site.shell[0]), int(site.shell[1])
    pos = atoms.get_positions()
    vec, dist = get_distances(pos[a][None], pos[b][None], cell=atoms.get_cell(),
                              pbc=atoms.pbc)
    d = float(dist[0, 0])
    return a, b, vec[0, 0] / max(d, 1e-12), d


def model_axial(model, frames, z_table, cutoff, device, ctx, batch=4):
    """The HEAD's axial hub force per frame, in `axial_residual`'s convention.

    `forces - base_forces` is the correction branch alone. Stage 3 has the long-range branch
    off, so that is the counting head's own force and nothing else.
    """
    rows = []
    for b, fr in make_batches(frames, z_table, cutoff, batch, device):
        d = ctx.forward_dict(b, fr, requires_grad=True)
        out = model(d, training=False, compute_force=True)
        corr = (out["forces"] - out["base_forces"]).detach().cpu().numpy()
        graph = b.batch.detach().cpu().numpy()
        for g, atoms in enumerate(fr):
            sel = np.nonzero(graph == g)[0]
            try:
                ia, ib, axis, dist = hub_axis(atoms)
            except ValueError:
                continue
            dF = corr[sel]
            rows.append(dict(d=dist, n_atoms=len(atoms),
                             axial=0.5 * (float(np.dot(dF[ib], axis))
                                          - float(np.dot(dF[ia], axis)))))
    return rows


def level_spacing(model, pristine, z_table, cutoff, device, ctx) -> float:
    """`delta_L`: median spacing just above the pristine frontier, this model's own.

    THE COMMON-delta_L CONVENTION. `delta_L` must come from ONE cell size for every frame the
    bound flag is applied to, and that size is the LARGE one. The labels are k-sampled on the
    doubled axis, so their continuum discretisation is size-invariant, while the model's
    Gamma-only continuum is 2x sparser at 79 atoms than at 159. Taking `delta_L` from a small
    pristine cell and `depth` from a large charged one -- which this did -- compares a depth
    against a level spacing from a different Brillouin-zone sampling, and inflates the bound
    fraction at the small size for a reason that has nothing to do with binding.

    The caller passes pristine frames already filtered to the large size; `pristine_size` is
    reported so the convention travels with the number.
    """
    from mace.modules.defect_counting import VALENCE

    spac = []
    for b, fr in make_batches(pristine, z_table, cutoff, 1, device):
        internals, _ = capture(model, b, ctx=ctx, frames=fr)
        lam = internals["lam"][0, 0]
        lam = torch.sort(lam[lam < 500.0]).values
        n_el = sum(VALENCE[int(z)] for z in fr[0].get_atomic_numbers()) // 2
        top = min(n_el + 11, int(lam.numel()))
        if top - n_el < 3:
            continue
        spac.append(float(torch.diff(lam[n_el:top]).median()))
    return float(np.median(spac)) if spac else float("nan")


def depths(model, frames, z_table, cutoff, device, ctx):
    """`lam[k+1] - lam[k]` per charged frame, k the level whose occupation changed."""
    from mace.modules.defect_counting import VALENCE, spin_targets

    out = []
    for b, fr in make_batches(frames, z_table, cutoff, 1, device):
        internals, _ = capture(model, b, ctx=ctx, frames=fr)
        lam = internals["lam"][0, 0]
        lam = torch.sort(lam[lam < 500.0]).values
        n_total = sum(VALENCE[int(z)] for z in fr[0].get_atomic_numbers())
        n_maj, _ = spin_targets(n_total, b.carrier_counts.reshape(-1).tolist())
        k = int(round(max(n_maj, float((n_total + 1) // 2)))) - 1
        out.append(float(lam[k + 1] - lam[k]) if 0 <= k < lam.numel() - 1 else float("nan"))
    return np.array(out)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--tol", type=float, default=0.10, help="d-matching window, A")
    ap.add_argument("--n-pristine", type=int, default=4)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    charged = select(load_frames(args.data), charged=True)
    big = [a for a in charged if len(a) == CLEAN_NATOMS]
    small_all = [a for a in charged if len(a) == SMALL_NATOMS]
    if len(big) < 5 or len(small_all) < 50:
        raise SystemExit(f"need both sizes; have {len(big)} big and {len(small_all)} small")

    big_d = np.array([hub_axis(a)[3] for a in big])
    # Only the small frames the matching can actually use. Exact, not a subsample:
    # `matched_ratios` looks at nothing outside these windows, and scoring all 1030 would be
    # an hour of forwards per model for the same numbers.
    small = [a for a in small_all
             if np.min(np.abs(big_d - hub_axis(a)[3])) <= args.tol]
    print(f"{len(big)} frames of {CLEAN_NATOMS} atoms, d {big_d.min():.2f}-{big_d.max():.2f} A;"
          f" {len(small)} of {SMALL_NATOMS} inside the +-{args.tol} A windows "
          f"(of {len(small_all)})", flush=True)

    # The largest pristine cells available, for the common-delta_L convention above.
    all_pristine = select_pristine(ase_read(str(args.data), ":"), 64)
    if not all_pristine:
        raise SystemExit("no pristine frames found")
    biggest = max(len(a) for a in all_pristine)
    pristine = [a for a in all_pristine if len(a) == biggest][: args.n_pristine]
    print(f"delta_L from {len(pristine)} pristine cells of {biggest} atoms "
          f"(common-delta_L convention: one size for every frame the bound flag touches)",
          flush=True)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
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

        delta_l = level_spacing(model, pristine, z, cutoff, args.device, ctx)
        dep = depths(model, big, z, cutoff, args.device, ctx)
        bound = np.isfinite(dep) & (dep / delta_l > BOUND_MIN)

        big_rows = model_axial(model, big, z, cutoff, args.device, ctx)
        small_rows = model_axial(model, small, z, cutoff, args.device, ctx)
        rng = np.random.default_rng(args.seed)
        allr = matched_ratios(big_rows, small_rows, args.tol, rng)
        bnd = matched_ratios([r for r, k in zip(big_rows, bound) if k], small_rows,
                             args.tol, rng) if bound.any() else None

        row = dict(model=Path(mp).name, delta_L=delta_l, pristine_size=int(biggest),
                   depth_med=float(np.nanmedian(dep)),
                   bound_fraction=float(bound.mean()), all=allr, bound=bnd,
                   passed=bool(bnd is not None and bnd["median"] <= GATE_MAX_RATIO))
        rows.append(row)
        b_txt = ("n/a" if bnd is None
                 else f"{bnd['median']:.2f} [{bnd['lo']:.2f}, {bnd['hi']:.2f}] (n={bnd['n']})")
        print(f"  {row['model']:26s} delta_L {delta_l:.3f} eV  depth {row['depth_med']:.3f} eV"
              f"  bound {100 * row['bound_fraction']:3.0f}%  R_bound {b_txt}"
              f"  {'PASS' if row['passed'] else 'FAIL'}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    scored = [r for r in rows if r["bound"] is not None]
    if scored:
        med = np.array([r["bound"]["median"] for r in scored])
        bf = np.array([r["bound_fraction"] for r in rows])
        print(f"\n  bound-conditioned R {med.mean():.2f} +- {med.std():.2f} over "
              f"{len(scored)} seeds; gate <= {GATE_MAX_RATIO} met by "
              f"{sum(r['passed'] for r in rows)}/{len(rows)}")
        print(f"  bound fraction {100 * bf.mean():.0f}% +- {100 * bf.std():.0f}% of the "
              f"{len(big)} large frames (depth > {BOUND_MIN} level spacings)")
        print("  R ~ 1 is a bound carrier, R ~ 2 a band state whose hub amplitude halved "
              "with the cell. Anything between is partial binding and is NOT a pass.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
