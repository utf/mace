#!/usr/bin/env python3
"""Section 1 of the Stage A' spec: the re-derived references, in the scorer's format (F17).

Reads the JSON `b1_label_slope_by_size.py` wrote against the A' folds and production base,
adds the one number b1 does not compute -- the neutral 159-atom energy slope against the
PRODUCTION base, the like-for-like reference for criterion 2 -- and writes the reference
JSON `b10_adoption.py --reference-json` consumes:

    energy_slope, energy_ci        charged 159, d(E_label - E_base)/dd, production base
    force_slope, force_ci          charged 159, axial pair-force residual, production base
    neutral_159                    out-of-fold neutral null (cross-fit)
    neutral_159_production_base    neutral 159 against the production base

Then scores F17 against the pre-joint numbers:
    the 159 null within +-0.05 of zero;
    the charged 159 residual within the OLD interval [-0.1447, -0.1232], or shifted by less
    than the old null's width (0.26);
    the 79-atom charged residual magnitude < 0.2.
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
from e0_residual_maps import _assert_repo, evaluate  # noqa: E402
from r1_matrix import adopt_model_dtype, graph_cutoff_for  # noqa: E402
from s3_dehead_trend import fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import counts_of  # noqa: E402

from mace import tools  # noqa: E402

OLD_ENERGY_CI = (-0.1447, -0.1232)
OLD_NULL_WIDTH = 0.2102 - (-0.0503)       # the old out-of-fold null's interval width
OLD_ENERGY_SLOPE = -0.1338


def neutral_159_against(model_path, frames, device):
    model = torch.load(model_path, map_location=device, weights_only=False).to(device).eval()
    adopt_model_dtype(model)
    z = tools.AtomicNumberTable([17, 55, 82])
    sel = [a for a in frames if len(a) == 159
           and counts_of(a) is not None and sum(counts_of(a)) == 0]
    _, e = evaluate(model, sel, z, graph_cutoff_for(model), device, 4)
    d = np.array([hub_separation(a) for a in sel])
    lab = np.array([float(a.info["REF_energy"]) for a in sel])
    s, lo, hi, corr = fit_with_ci(d, lab - np.asarray(e))
    return dict(n=len(sel), slope=s, ci=[lo, hi], corr=corr)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b1-json", type=Path, required=True)
    ap.add_argument("--production", type=Path, required=True)
    ap.add_argument("--data", nargs="+", type=Path,
                    default=[here / "dataset_pbe" / "train.xyz",
                             here / "dataset_pbe" / "valid.xyz"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    b1 = json.load(open(args.b1_json))
    de, ax = b1["de"], b1["axial"]
    frames = []
    for p in args.data:
        frames.extend(read(str(p), index=":"))
    n159 = neutral_159_against(args.production, frames, args.device)

    ref = dict(
        energy_slope=de["159_prod"]["slope"], energy_ci=de["159_prod"]["ci"],
        force_slope=ax["159_prod"]["slope"], force_ci=ax["159_prod"]["ci"],
        neutral_159=de["159_neutral_null"]["slope"],
        neutral_159_ci=de["159_neutral_null"]["ci"],
        neutral_159_production_base=n159["slope"],
        neutral_159_production_base_ci=n159["ci"],
        charged_79_full=de["79_full"]["slope"], charged_79_full_ci=de["79_full"]["ci"],
        charged_79_window=de.get("79_neutral_dense", {}).get("slope"),
        neutral_79_window=de.get("79_neutral_null", {}).get("slope"),
        neutral_79_window_ci=de.get("79_neutral_null", {}).get("ci"),
        neutral_159_force=ax.get("159_neutral_null", {}).get("slope"),
        source=str(args.b1_json), production=str(args.production))

    print("=== Stage A' references ===")
    for k in ("energy_slope", "energy_ci", "force_slope", "force_ci", "neutral_159",
              "neutral_159_ci", "neutral_159_production_base", "neutral_159_production_base_ci",
              "charged_79_full", "charged_79_full_ci", "charged_79_window",
              "neutral_79_window", "neutral_79_window_ci", "neutral_159_force"):
        print(f"  {k:32s} {ref[k]}")

    # ---- F17 ---------------------------------------------------------------------
    null_ok = abs(ref["neutral_159"]) <= 0.05
    chg = ref["energy_slope"]
    inside_old = OLD_ENERGY_CI[0] <= chg <= OLD_ENERGY_CI[1]
    shift = abs(chg - OLD_ENERGY_SLOPE)
    chg_ok = inside_old or shift < OLD_NULL_WIDTH
    small_ok = abs(ref["charged_79_full"]) < 0.2
    print("\n=== F17 ===")
    print(f"  159 null {ref['neutral_159']:+.4f} within +-0.05 of zero: {null_ok}")
    print(f"  charged 159 {chg:+.4f}: inside the old interval {OLD_ENERGY_CI}: {inside_old}; "
          f"shift from {OLD_ENERGY_SLOPE:+.4f} = {shift:.4f} < old null width "
          f"{OLD_NULL_WIDTH:.3f}: {shift < OLD_NULL_WIDTH}  -> {chg_ok}")
    print(f"  79-atom charged residual |{ref['charged_79_full']:+.4f}| < 0.2: {small_ok}")
    holds = null_ok and chg_ok and small_ok
    print(f"  -> F17 {'HOLDS' if holds else 'FAILS'} "
          f"({int(null_ok) + int(chg_ok) + int(small_ok)}/3 clauses)")
    ref["f17"] = dict(null=null_ok, charged_159=chg_ok, charged_79=small_ok, holds=holds)
    args.out.write_text(json.dumps(ref, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
