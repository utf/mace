#!/usr/bin/env python3
"""Section 2.3 adoption test: the frontier-weighted on-site potential across tilings.

A single chlorine vacancy in the 1x, 2x and 3x isotropic tilings of a pristine supercell,
scored with an s7 head-only model, with and without the image-compensation term:

    D(L) = sum_i rho_i [ madelung_i + comp_i ]

with rho the carrier density from the head. Without the term the periodic images of the
carrier shift the level as ~1/L (D0 = D(1x) - D(3x)); with it the drift must be at most
0.3 * D0. Ideal geometries -- the pristine cell from the dataset with one Cl removed and no
relaxation -- so the only thing that changes between tilings is the box.

The 3x tiling holds 2159 atoms; the dense eigensolve is 8636 x 8636 in float64, which the
A4000 handles in seconds. This is why "1x, 2x, 3x" is read isotropically.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, graph_cutoff_for, make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402


CARRIER_INFO_KEYS = ("carrier_counts", "multiplicity", "m_s_ref_doubled", "cell_charge",
                     "config_type", "host")


def vacancy_in_tiling(pristine, reps: int, reference):
    """The tiled pristine cell with one Cl removed, carrying a REAL charged frame's counter
    bookkeeping (counts, multiplicity, spin reference) so the counter validator accepts it."""
    atoms = pristine.repeat((reps, reps, reps))
    cl = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 17]
    # The Cl nearest the cell centre, so the vacancy sits away from the (periodic) boundary
    # of the tiled box in the same way at every size.
    centre = atoms.get_cell().sum(axis=0) / 2
    pos = atoms.get_positions()[cl]
    pick = cl[int(np.argmin(np.linalg.norm(pos - centre, axis=1)))]
    del atoms[pick]
    atoms.info = {k: v for k, v in atoms.info.items() if k not in CARRIER_INFO_KEYS}
    for k in CARRIER_INFO_KEYS:
        if k in reference.info:
            atoms.info[k] = reference.info[k]
    return atoms


def frontier_potential(model, atoms, z_table, cutoff, device, ctx):
    batch, frs = make_batches([atoms], z_table, cutoff, 1, device)[0]
    d = ctx.forward_dict(batch, frs, requires_grad=False)
    with torch.no_grad():
        out = model(d, training=False, compute_force=False)
        rho = out["carrier_alpha"][:, 0].reshape(-1)
        species = batch.node_attrs.argmax(dim=-1)
        madelung = model.madelung.on_site_shift(
            model.latent_ewald, species, batch.positions.to(rho.dtype),
            batch.cell.to(rho.dtype), batch.batch, eps_inf=model.madelung_eps_inf
        ).reshape(-1)
        comp = out.get("image_compensation")
        comp = torch.zeros_like(madelung) if comp is None else comp.reshape(-1).to(rho.dtype)
    return dict(
        n_atoms=int(len(atoms)),
        n_eff=float(1.0 / (rho ** 2).sum()),
        madelung_weighted=float((rho * madelung).sum()),
        comp_weighted=float((rho * comp).sum()),
        total=float((rho * (madelung + comp)).sum()),
        comp_mean=float(comp.mean()), comp_std=float(comp.std()),
        gap=float(out["logit_gap"].reshape(-1)[0]),
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z_table = tools.AtomicNumberTable([17, 55, 82])
    pristine = [a for a in select_pristine(load_frames(args.data), 64)]
    biggest = max(len(a) for a in pristine)
    pristine = [a for a in pristine if len(a) == biggest][0]
    reference = [a for a in select(load_frames(args.data), charged=True) if len(a) == 79][0]
    print(f"pristine cell: {len(pristine)} atoms, cell {np.diag(pristine.get_cell()).round(3)}; "
          f"counter bookkeeping from a charged 79-atom frame: "
          f"{ {k: reference.info.get(k) for k in CARRIER_INFO_KEYS} }", flush=True)

    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device, weights_only=False).to(
            args.device).eval()
        adopt_model_dtype(model)
        cutoff = graph_cutoff_for(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        for term in (False, True):
            model.image_compensation = bool(term)
            for reps in args.reps:
                atoms = vacancy_in_tiling(pristine, reps, reference)
                r = frontier_potential(model, atoms, z_table, cutoff, args.device, ctx)
                r.update(model=mp.name, term=bool(term), reps=int(reps),
                         L=float(np.cbrt(atoms.get_volume())))
                rows.append(r)
                print(f"  {mp.name:22s} term={'on ' if term else 'off'} {reps}x "
                      f"({r['n_atoms']:5d} atoms, L {r['L']:.2f} A)  N_eff {r['n_eff']:7.2f}  "
                      f"rho.madelung {r['madelung_weighted']:+.4f}  rho.comp "
                      f"{r['comp_weighted']:+.4f}  D {r['total']:+.4f} eV  gap {r['gap']:.3f}",
                      flush=True)
        model.image_compensation = False
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n=== drift D(1x) - D(3x), per model ===")
    verdicts = []
    for mp in args.models:
        sub = {(r["term"], r["reps"]): r for r in rows if r["model"] == mp.name}
        lo, hi = min(args.reps), max(args.reps)
        d0 = sub[(False, lo)]["total"] - sub[(False, hi)]["total"]
        d1 = sub[(True, lo)]["total"] - sub[(True, hi)]["total"]
        ok = abs(d1) <= 0.3 * abs(d0)
        verdicts.append(dict(model=mp.name, D0=d0, D_with=d1, ratio=(abs(d1) / abs(d0)
                                                                    if d0 else float("nan")),
                             passes=bool(ok)))
        print(f"  {mp.name:22s} D0 {d0:+.4f} eV   with term {d1:+.4f} eV   "
              f"|ratio| {abs(d1) / max(abs(d0), 1e-12):.3f}   {'PASS' if ok else 'FAIL'}"
              f"   (gate: <= 0.3 D0)")
    n_pass = sum(v["passes"] for v in verdicts)
    print(f"  {n_pass}/{len(verdicts)} models within 0.3 D0")
    args.out.write_text(json.dumps(dict(rows=rows, verdicts=verdicts), indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
