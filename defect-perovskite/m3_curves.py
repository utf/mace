#!/usr/bin/env python3
"""M3 + per-frame curves: separate R-A (dynamic level), R-B (cage route), R-C (flip inert).

Measurement 1 (M3): autograd d|H_ab|/dd on REAL frames -- exact, not the zero-feature profile
slope F2 had to use. H_ab's gradient w.r.t. both hub positions, projected on the pair axis:
    t' = 0.5 * (grad_b - grad_a) . axis
  R-A/R-B predict ON |t'| -> 0.10-0.20 eV/A, OFF at prior. R-C predicts ON ~ OFF ~ prior.

Measurement 2: N_eff and Delta_bind per FRAME against d(Pb-Pb), per cell.
  R-A predicts N_eff falling as d falls (binds when the pair dimerises) and Delta_bind rising
  at short d. R-B predicts both flat in d.

Both run on saved checkpoints; no training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from ase.geometry import get_distances
from ase.io import read

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, make_batch  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402


def analyse(model, atoms, z, cutoff, device):
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    ia, ib = int(site.shell[0]), int(site.shell[1])
    pos = atoms.get_positions()
    vec, dd = get_distances(pos[ia][None], pos[ib][None], cell=atoms.get_cell(),
                            pbc=atoms.pbc)
    axis = vec[0, 0] / max(float(dd[0, 0]), 1e-12)

    batch = make_batch([atoms], z, cutoff, device)
    d = batch.to_dict()
    d["positions"] = d["positions"].detach().requires_grad_(True)
    p = d["positions"]
    g = {}
    head = model.spectral
    orig = head.forward
    head.forward = lambda *A, **K: orig(*A, **dict(K, internals=g))
    try:
        out = model(d, training=False, compute_force=False)
    finally:
        head.forward = orig

    loc = g["local"]
    n = len(atoms)
    slots = loc[:n].long()
    ch = int(torch.argmax(d["carrier_counts"].reshape(-1)).item())
    H = g["H"][0, ch][slots][:, slots]
    hab = H[ia, ib]
    grad = torch.autograd.grad(hab, p, retain_graph=False, allow_unused=True)[0]
    if grad is None:
        tprime = float("nan")
    else:
        gr = grad.detach().cpu().numpy()
        tprime = 0.5 * float(np.dot(gr[ib] - gr[ia], axis))

    alpha = out["carrier_alpha"][:, ch].detach().cpu().numpy()
    s2 = float((alpha ** 2).sum())
    neff = 1.0 / s2 if s2 > 0 else float("nan")
    lam = g["lam"][0, ch]
    lam = float(lam[lam < 500.0].min())
    return dict(d=float(dd[0, 0]), tprime=tprime, abs_tprime=abs(tprime),
                hab=float(abs(hab)), neff=neff, lam=lam)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path.home() / "runs" / "s1_charged.xyz")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = read(str(args.data), ":")[: args.limit]
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        per = [r for r in (analyse(model, a, z, cutoff, args.device) for a in frames)
               if r is not None]
        if not per:
            continue
        d = np.array([r["d"] for r in per])
        ne = np.array([r["neff"] for r in per])
        lam = np.array([r["lam"] for r in per])
        tp = np.array([r["abs_tprime"] for r in per])
        row = dict(model=Path(mp).name, n=len(per),
                   abs_tprime=float(np.mean(tp)), hab=float(np.mean(
                       [r["hab"] for r in per])),
                   neff=float(np.mean(ne)),
                   corr_neff_d=float(np.corrcoef(d, ne)[0, 1]) if len(d) > 2 else np.nan,
                   corr_lam_d=float(np.corrcoef(d, lam)[0, 1]) if len(d) > 2 else np.nan,
                   d_range=[float(d.min()), float(d.max())])
        rows.append(row)
        print(f"  {row['model']:34s} |t'| {row['abs_tprime']:.4f}  |H_ab| {row['hab']:.4f}  "
              f"N_eff {row['neff']:6.2f}  corr(N_eff,d) {row['corr_neff_d']:+.3f}  "
              f"corr(lam,d) {row['corr_lam_d']:+.3f}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))
    args.out.write_text(json.dumps(rows, indent=2, default=float))
    for arm in ("on", "off"):
        g = [r for r in rows if f"_{arm}_" in r["model"]]
        if g:
            print(f"\n  {arm.upper()} n={len(g)}  |t'| {np.mean([r['abs_tprime'] for r in g]):.4f}"
                  f"  corr(N_eff,d) {np.nanmean([r['corr_neff_d'] for r in g]):+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
