#!/usr/bin/env python3
"""Section 4: how far the kernel correction moves the on-site energies, per species.

The number section 5 is read against. Under the retired convention `phi_LR` had
`A_ii * Z_i` subtracted; `A_ii` is the Makov-Payne self-image potential and is `-alpha_M/L`,
so the correction adds `+A_ii * Z_i` back to `phi` and therefore shifts the on-site energy of
species s by

    delta_eps_s = -A_ii * Z_s / eps_inf

which is species-proportional and does NOT cancel between species. At the 79-atom cells the
plan expects order 0.5-1 eV. Measured here per model rather than predicted, because `Z` is
learned and differs by seed.

INTERPRETIVE ONLY. These weights were fitted under the old kernel, so what this reports is
how far from its own optimum each model now sits -- not a prediction about the retrained
models. The rerun is what settles that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT
from mace.modules.defect_madelung import self_potential_of

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

SYMBOL = {0: "Cl", 1: "Cs", 2: "Pb"}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    charged = select(load_frames(args.data), charged=True)
    sizes = {}
    for a in charged:
        sizes.setdefault(len(a), a)
    wanted = [n for n in (79, 159) if n in sizes]
    print(f"cell sizes present: {sorted(sizes)}; scoring {wanted}", flush=True)

    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        z_learned = [float(x) for x in model.madelung.z.detach().cpu()]
        entry = {"model": Path(mp).name, "Z": z_learned, "sizes": {}}
        for n in wanted:
            batch, _ = make_batches([sizes[n]], z_table, cutoff, 1, args.device)[0]
            cell = batch.cell.reshape(1, 3, 3).to(args.device)
            a_ii = float(self_potential_of(model.latent_ewald, cell)[0])
            # delta_eps_s = -A_ii Z_s / eps_inf: what the correction adds back.
            deltas = {SYMBOL[i]: -a_ii * z / args.eps_inf
                      for i, z in enumerate(z_learned)}
            entry["sizes"][str(n)] = {"A_ii": a_ii, "delta_eps": deltas}
            print(f"  {entry['model']:20s} n={n:3d}  A_ii {a_ii:+.4f} eV  "
                  + "  ".join(f"d(eps_{k}) {v:+.4f}" for k, v in deltas.items()),
                  flush=True)
        if len(wanted) == 2:
            small = entry["sizes"][str(wanted[0])]["delta_eps"]
            big = entry["sizes"][str(wanted[1])]["delta_eps"]
            entry["cross_size"] = {k: big[k] - small[k] for k in small}
            print(f"  {'':20s} CROSS-SIZE (the defect that motivated the change): "
                  + "  ".join(f"{k} {v:+.4f} eV" for k, v in entry["cross_size"].items()),
                  flush=True)
        rows.append(entry)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows and "cross_size" in rows[0]:
        for sp in SYMBOL.values():
            v = np.array([r["cross_size"][sp] for r in rows if "cross_size" in r])
            print(f"\n  {sp}: cross-size on-site error under the retired convention "
                  f"{v.mean():+.4f} +- {v.std():.4f} eV")
        print("  This is the quantity that was wrong between the two training cell sizes, "
              "per species, at the learned Z. It is now zero by construction.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
