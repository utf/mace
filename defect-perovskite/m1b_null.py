#!/usr/bin/env python3
"""M1b: which cell-size subset speaks for the labels?

M1 split: 159-atom frames give corr -0.989 (the predicted sign), 1030 cell-limited 79-atom
frames give +0.447. The proposed explanation is ours, not the label's: the frozen base
extrapolates on charged long-d frames, and an extrapolating network flattens steep surfaces,
so under-predicting E0 at long d inflates dE with an error that GROWS with d.

Four cuts, no training:

  1. base-error null   E0_DFT - E_base on NEUTRAL defective frames vs d. If the base's own
                       error rises with d, the charged trend is contaminated by it.
  2. windowed          79-atom regression restricted to d <= 5.5, the neutral-dense window.
  3. orientation       79-atom split by vacancy axis (geometry only, no labels).
  4. corrected         79-atom trend on dE - null(d).

CAVEAT, stated because it weakens cut 1: no per-fold bases exist, so the null is IN-SAMPLE --
the base was trained on these neutral frames. In-sample error understates extrapolation error,
so a positive null slope here is a LOWER bound on the contamination and a flat null does not
prove the absence of it. Out-of-fold bases would be needed for the clean test.
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


def collect(base, frames, z, cutoff, device, charged):
    d, de, n, axis_c = [], [], [], []
    for a in frames:
        c = a.info.get("carrier_counts")
        has = c is not None and int(np.asarray(c).sum()) > 0
        if has != charged:
            continue
        try:
            site = locate_vacancy(a)
        except ValueError:
            continue
        ia, ib = int(site.shell[0]), int(site.shell[1])
        pos = a.get_positions()
        vec, dd = get_distances(pos[ia][None], pos[ib][None], cell=a.get_cell(), pbc=a.pbc)
        u = vec[0, 0] / max(float(dd[0, 0]), 1e-12)
        batch = make_batch([a], z, cutoff, device)
        with torch.no_grad():
            out = base(batch.to_dict(), training=False, compute_force=False)
        d.append(float(dd[0, 0]))
        de.append(float(a.info["REF_energy"]) - float(out["energy"][0]))
        n.append(len(a))
        axis_c.append(abs(float(u[2])))          # |cos| with c: geometry only
    return map(np.array, (d, de, n, axis_c))


def fit(d, e, label):
    if len(d) < 3:
        return None
    r = float(np.corrcoef(d, e)[0, 1])
    s = float(np.polyfit(d, e, 1)[0])
    print(f"  {label:34s} n={len(d):5d}  corr {r:+.3f}  slope {s:+.4f} eV/A")
    return dict(n=int(len(d)), corr=r, slope=s)


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
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    cutoff = float(base.r_max)
    frames = [a for p in args.data for a in read(str(p), ":")]

    res = {}
    print("\n  [1] base-error null, NEUTRAL defective frames (IN-SAMPLE -- lower bound)")
    dn, en, nn_, _ = collect(base, frames, z, cutoff, args.device, charged=False)
    res["null_all"] = fit(dn, en, "null, all neutral")
    m79 = nn_ == 79
    res["null_79"] = fit(dn[m79], en[m79], "null, 79-atom neutral")
    if m79.sum() > 3:
        long_ = m79 & (dn > 5.5)
        if long_.sum() > 3:
            res["null_79_long"] = fit(dn[long_], en[long_], "null, 79-atom d > 5.5")

    print("\n  [2,3,4] charged frames")
    dc, ec, nc, ax = collect(base, frames, z, cutoff, args.device, charged=True)
    c79 = nc == 79
    res["charged_79"] = fit(dc[c79], ec[c79], "charged 79, all d")
    w = c79 & (dc <= 5.5)
    res["charged_79_windowed"] = fit(dc[w], ec[w], "charged 79, d <= 5.5 (window)")
    hi, lo = c79 & (ax > 0.7), c79 & (ax <= 0.7)
    res["charged_79_axis_c"] = fit(dc[hi], ec[hi], "charged 79, axis ~ c")
    res["charged_79_axis_ab"] = fit(dc[lo], ec[lo], "charged 79, axis ~ a/b")
    if res.get("null_79"):
        corr = ec - np.polyval([res["null_79"]["slope"],
                                float(np.mean(en[m79]) - res["null_79"]["slope"]
                                      * np.mean(dn[m79]))], dc)
        res["charged_79_corrected"] = fit(dc[c79], corr[c79], "charged 79, null-corrected")
    res["charged_159"] = fit(dc[nc == 159], ec[nc == 159], "charged 159 (reference)")

    ns = res.get("null_79", {}).get("slope", float("nan"))
    verdict = ("null is FLAT -> small-cell trend is real label physics -> §2 falsified"
               if abs(ns) < 0.05 else
               "null RISES with d -> small-cell trend is contaminated by base error -> "
               "the 159-atom subset speaks for the labels")
    res["null_79_slope"] = ns
    res["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    print("  (in-sample null: a positive slope is a lower bound; a flat one is not proof)")
    args.out.write_text(json.dumps(res, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
