#!/usr/bin/env python3
"""D-2: bin the charge-state force error by resonance and by electrostatic depth.

TWO BINNINGS, both from quantities the model already has.

1. **Resonance.** The frontier level's distance to the nearest delocalised state, in units of
   the pristine supercell's own level spacing: `depth / delta_L`. Below 1 the defect level is
   degenerate with the continuum on the scale the finite cell can resolve, and the state it
   describes is not a bound state at all. **Prediction: the error concentrates in the
   resonant bin.**

2. **Electrostatic depth.** `|phi_LR|` at the defect-near atoms, from a NOMINAL point-charge
   proxy (Cs +1, Pb +2, Cl -1) rather than from anything learned -- the point is to bin by a
   fixed external quantity, so that a correlation is not the model correlating with itself.
   The vacancy leaves the cell at net +1 and LatentEwald's background handles that; the
   convention is E_LR's own, which is the same one Edit 1 will use.

**The lambda-d anomaly is carried here.** `corr(lambda, d_hub) < 0` in 10 of 12 A/B cells --
the defect level falls as the vacancy-flanking pair separates, the reverse of expectation. If
it localises to the resonant bin, it is recorded as explained-by-resonance pending Edit 4 and
stops being an open architectural question. So the correlation is reported per bin, not
pooled.

Stage 1's gate is a re-run of this script with the new models, so it takes a model list and
nothing else that changes between stages.
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
from mace.modules.defect_madelung import self_potential_of, site_potential
from mace.modules.latent_ewald import LatentEwald

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo, make_batch  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

# Nominal formal charges, in the model's atomic-number order. A proxy, not a fit.
NOMINAL = {17: -1.0, 55: 1.0, 82: 2.0}


def spectrum(model, atoms, z, cutoff, device):
    """Physical eigenvalues of the active channel, ascending."""
    batch = make_batch([atoms], z, cutoff, device)
    d = batch.to_dict()
    grabbed: dict = {}
    head = model.spectral
    original = head.forward
    head.forward = lambda *a, **k: original(*a, **dict(k, internals=grabbed))
    try:
        with torch.no_grad():
            out = model(d, training=False, compute_force=False)
    finally:
        head.forward = original
    ch = int(torch.argmax(d["carrier_counts"].reshape(-1)).item())
    lam = grabbed["lam"][0, ch]
    lam = lam[lam < 500.0]
    return torch.sort(lam).values.detach().cpu().numpy(), out, ch


def level_spacing(model, frames, z, cutoff, device):
    """delta_L: the pristine supercell's mean adjacent level spacing near the frontier."""
    gaps = []
    for a in frames:
        lam, _, _ = spectrum(model, a, z, cutoff, device)
        if lam.size > 3:
            # Near the frontier, which for this head is the bottom of the manifold.
            k = max(2, min(8, lam.size - 1))
            gaps.append(float(np.mean(np.diff(lam[:k]))))
    return float(np.mean(gaps)) if gaps else float("nan")


def phi_proxy(atoms, ewald, device, radius=5.0):
    """mean |phi| over the defect-near atoms, from nominal point charges.

    Diagnostic, so the vacancy assignment is permitted here -- it selects which atoms are
    reported, and never enters a model input.
    """
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return float("nan")
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64,
                        device=device).reshape(1, 3, 3)
    q = torch.tensor([NOMINAL[int(n)] for n in atoms.get_atomic_numbers()],
                     dtype=torch.float64, device=device)
    b = torch.zeros(len(atoms), dtype=torch.long, device=device)
    with torch.no_grad():
        self_pot = self_potential_of(ewald, cell)
        phi = site_potential(ewald, q, pos, cell, b, self_potential=self_pot)
    centre = atoms.get_positions()[list(site.shell)].mean(axis=0)[None]
    _, dd = get_distances(centre, atoms.get_positions(), cell=atoms.get_cell(),
                          pbc=atoms.pbc)
    near = dd[0] <= radius
    if not near.any():
        return float("nan")
    return float(np.abs(phi.detach().cpu().numpy()[near]).mean())


def frame_row(model, atoms, z, cutoff, device, ewald, delta_l):
    lam, _, ch = spectrum(model, atoms, z, cutoff, device)
    if lam.size < 2:
        return None
    depth = float(lam[1] - lam[0])

    batch = make_batch([atoms], z, cutoff, device)
    d = batch.to_dict()
    d["positions"] = d["positions"].detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(d, training=False, compute_force=True)
    ref = batch.forces.detach().cpu().numpy()
    err = (out["forces"].detach().cpu().numpy() - ref)
    err_b = (out["base_forces"].detach().cpu().numpy() - ref)

    try:
        site = locate_vacancy(atoms)
        pos = atoms.get_positions()
        _, dd = get_distances(pos[int(site.shell[0])][None], pos[int(site.shell[1])][None],
                              cell=atoms.get_cell(), pbc=atoms.pbc)
        d_hub = float(dd[0, 0])
    except (ValueError, IndexError):
        d_hub = float("nan")

    return dict(lam1=float(lam[0]), depth=depth,
                resonance=depth / delta_l if delta_l == delta_l else float("nan"),
                phi=phi_proxy(atoms, ewald, device),
                rmse=float(np.sqrt((err ** 2).mean())) * 1000.0,
                rmse_base=float(np.sqrt((err_b ** 2).mean())) * 1000.0,
                d_hub=d_hub, channel=ch)


def binned(rows, key, edges):
    out = []
    vals = np.array([r[key] for r in rows], dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, v in zip(rows, vals) if lo <= v < hi]
        if not sel:
            out.append(dict(lo=lo, hi=hi, n=0))
            continue
        lam = np.array([r["lam1"] for r in sel])
        dh = np.array([r["d_hub"] for r in sel])
        ok = np.isfinite(lam) & np.isfinite(dh)
        out.append(dict(
            lo=float(lo), hi=float(hi), n=len(sel),
            rmse=float(np.mean([r["rmse"] for r in sel])),
            rmse_base=float(np.mean([r["rmse_base"] for r in sel])),
            corr_lam_d=(float(np.corrcoef(dh[ok], lam[ok])[0, 1])
                        if ok.sum() > 2 else float("nan"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path.home() / "runs" / "s1_charged.xyz")
    ap.add_argument("--pristine", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--n-pristine", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = read(str(args.data), ":")[: args.limit]
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    pristine = select_pristine(read(str(args.pristine), ":"), args.n_pristine)

    ewald = LatentEwald(None).to(args.device).double()

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            print(f"  MISSING {mp}", flush=True)
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        delta_l = level_spacing(model, pristine, z, cutoff, args.device)
        per = [r for r in (frame_row(model, a, z, cutoff, args.device, ewald, delta_l)
                           for a in frames) if r is not None]
        if not per:
            continue
        phis = np.array([r["phi"] for r in per], dtype=float)
        finite = phis[np.isfinite(phis)]
        phi_edges = ([float(finite.min()), float(np.median(finite)),
                      float(finite.max()) + 1e-9] if finite.size > 2 else [0.0, 1.0])
        row = dict(model=Path(mp).name, n=len(per), delta_L=delta_l,
                   resonance_bins=binned(per, "resonance", [0.0, 1.0, 2.0, np.inf]),
                   phi_bins=binned(per, "phi", phi_edges),
                   frames=per)
        rows.append(row)
        res = row["resonance_bins"]
        txt = "  ".join(f"[{b['lo']:.0f},{b['hi']:.0f}) n={b['n']}"
                        + (f" rmse={b['rmse']:.1f} corr={b['corr_lam_d']:+.2f}"
                           if b["n"] else "")
                        for b in res)
        print(f"  {row['model']:34s} dL={delta_l:.4f}  {txt}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
