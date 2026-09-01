#!/usr/bin/env python3
"""M1: does the LABEL's energy channel carry a d(Pb-Pb) trend, and with which sign?

The corrected claim is that the per-frame target dE(R) = E+_DFT(R) - E_base(R) is the vertical
ionisation energy and DECREASES in d: short d -> deeper bonding level -> harder to remove. The
head under OFF fits +(lambda_min + mu) = eps - t(d) + mu, whose d-derivative is -t' > 0, wrong
at any parameter values because t' < 0 is structural. So if the label has no d-trend, the
claim has nothing to stand on.

This is the first gate of the post-F2 plan and it uses no trained head at all: labels, a
frozen base, and the vacancy geometry.
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


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path,
                    default=Path.home() / "runs" / "e0_base_s1" / "e0_base_s1.model")
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[here / "dataset_pbe" / "valid.xyz",
                             here / "dataset_pbe" / "train.xyz"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    base = torch.load(args.base, map_location=args.device,
                      weights_only=False).to(args.device).eval()
    cutoff = float(base.r_max)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))

    d_list, de_list, n_list = [], [], []
    for path in args.data:
        for a in read(str(path), ":"):
            c = a.info.get("carrier_counts")
            if c is None or int(np.asarray(c).sum()) == 0:
                continue
            try:
                site = locate_vacancy(a)
            except ValueError:
                continue
            ia, ib = int(site.shell[0]), int(site.shell[1])
            pos = a.get_positions()
            _, dd = get_distances(pos[ia][None], pos[ib][None],
                                  cell=a.get_cell(), pbc=a.pbc)
            batch = make_batch([a], z, cutoff, args.device)
            with torch.no_grad():
                out = base(batch.to_dict(), training=False, compute_force=False)
            e_base = float(out["energy"][0])
            d_list.append(float(dd[0, 0]))
            de_list.append(float(a.info["REF_energy"]) - e_base)
            n_list.append(len(a))

    d = np.array(d_list); de = np.array(de_list); n = np.array(n_list)
    res = {}
    for label, sel in (("all", np.ones(len(d), bool)),
                       ("79-atom", n == 79), ("159-atom", n == 159)):
        if sel.sum() < 3:
            continue
        dd, ee = d[sel], de[sel]
        r = float(np.corrcoef(dd, ee)[0, 1])
        slope, intercept = np.polyfit(dd, ee, 1)
        res[label] = dict(n=int(sel.sum()), corr=r, slope=float(slope),
                          d_range=[float(dd.min()), float(dd.max())],
                          dE_range=[float(ee.min()), float(ee.max())],
                          swing=float(ee.max() - ee.min()))
        print(f"  {label:9s} n={int(sel.sum()):4d}  corr {r:+.3f}  slope {slope:+.4f} eV/A  "
              f"d {dd.min():.2f}-{dd.max():.2f} A  dE swing {ee.max() - ee.min():.3f} eV")

    a = res.get("all", {})
    verdict = ("PASS: negative d-trend of eV scale" if a.get("corr", 0) < -0.2
               and a.get("swing", 0) > 0.5 else
               "FAIL: no eV-scale negative d-trend in the labels")
    res["verdict"] = verdict
    print(f"\n  {verdict}")
    args.out.write_text(json.dumps(res, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
