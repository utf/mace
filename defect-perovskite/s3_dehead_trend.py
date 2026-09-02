#!/usr/bin/env python3
"""Stage 3's fourth gate: `corr(dE_head, d)` on the 159-atom subset, against -0.134 eV/A.

THE SUBSET IS THE POINT. M1b established that the 79-atom energy targets carry a
base-extrapolation slope of +0.37 eV/A that the frozen base cannot help, and that restricting
to the neutral-dense window collapses their correlation from +0.447 to +0.062. The 159-atom
charged frames are the cut that speaks for the labels: n = 17, corr -0.989, slope
**-0.134 eV/A**. Pooling sizes here would re-import the contamination the subset exists to
avoid, so this script refuses to.

`dE_head` is the model's own carrier correction (`delta_sr_energy`), not a re-derived
quantity, and `d` is the vacancy-flanking Pb-Pb separation. The gate asks whether the
counting head reproduces the sign and rough magnitude of the label trend -- the first time in
this programme that the energy channel has been checked against a label slope rather than
against itself.
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

from mace import tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

REFERENCE_SLOPE = -0.134          # eV/A, M1b's 159-atom charged subset
CLEAN_NATOMS = 159


def hub_separation(atoms) -> float:
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return float("nan")
    pos = atoms.get_positions()
    _, dd = get_distances(pos[int(site.shell[0])][None], pos[int(site.shell[1])][None],
                          cell=atoms.get_cell(), pbc=atoms.pbc)
    return float(dd[0, 0])


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == CLEAN_NATOMS]
    if len(frames) < 5:
        raise SystemExit(
            f"only {len(frames)} charged {CLEAN_NATOMS}-atom frames found; this gate is "
            "defined on that subset and pooling sizes would re-import the contamination "
            "M1b measured")
    d = np.array([hub_separation(a) for a in frames])
    ok = np.isfinite(d)
    print(f"{len(frames)} charged {CLEAN_NATOMS}-atom frames, "
          f"d range {d[ok].min():.2f}-{d[ok].max():.2f} A", flush=True)

    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        de = []
        for batch, fr in make_batches(frames, z, cutoff, 1, args.device):
            with torch.no_grad():
                out = model(batch.to_dict(), training=False, compute_force=False)
            de.append(float(out["delta_sr_energy"].reshape(-1)[0]))
        de = np.array(de)
        good = ok & np.isfinite(de)
        if good.sum() < 5:
            continue
        slope = float(np.polyfit(d[good], de[good], 1)[0])
        corr = float(np.corrcoef(d[good], de[good])[0, 1])
        rows.append(dict(model=Path(mp).name, n=int(good.sum()), slope=slope, corr=corr,
                         de_range=[float(de[good].min()), float(de[good].max())]))
        print(f"  {rows[-1]['model']:26s} slope {slope:+.4f} eV/A  corr {corr:+.3f}  "
              f"(reference {REFERENCE_SLOPE:+.3f})", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        sl = np.array([r["slope"] for r in rows])
        print(f"\n  mean slope {sl.mean():+.4f} +- {sl.std():.4f} eV/A against "
              f"{REFERENCE_SLOPE:+.3f}; correct sign in "
              f"{int((sl < 0).sum())}/{len(sl)} cells")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
