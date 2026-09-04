#!/usr/bin/env python3
"""Is §2.1's on-site correction a defect correction, or a bulk shift?

THE OBSERVATION. Arm A's per-shell corrections put −0.551 eV on 680 bulk Cl atoms against
−0.232 eV on 16 hub Pb. Weighted by count that is a bulk-wide shift with a defect-sized
rounding error, and it explains three otherwise unrelated results at once: the level moved
0.28 eV deeper (a uniform shift of the Cl sublattice moves the frontier eigenvalue), the
carrier DELOCALISED from n_eff 13 to 19 (a uniform shift is not a potential well), and F10
stayed unresolved (a uniform shift does not separate ligand from bulk).

THE HYPOTHESIS. The correction is `γ tanh(h(x_i) − h(x̄_{s(i)}))` and `x̄_s` is the
per-species mean first-block feature over the PRISTINE reference cells — 80-atom,
stoichiometric, thermal. It is then subtracted from atoms in 159-atom defective cells. If
bulk Cl in a big cell does not sit at the small-cell mean, `h(x_i) − h(x̄_s)` has a non-zero
mean over bulk atoms and the correction becomes a bulk-wide constant. That is a property of
the CENTRE's population, not of the defect.

THE DISCRIMINATOR. Measure the same per-shell correction on **neutral** 159-atom frames,
which contain no defect at all. If bulk Cl carries the same large offset there, the shift is
an artefact of centring across cell sizes. If it vanishes, the shift is defect physics and
this hypothesis is wrong.

A second, cleaner control: neutral 80-atom frames, the very population the centre is built
from. The correction there should be near zero by construction, and how near says how much
of the offset is thermal sampling rather than size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_per_atom_corrections import classify  # noqa: E402
from c7_centred_f10 import centred_correction  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402
from mace.modules.defect_context import ForwardContext  # noqa: E402

SHELLS = ("hub_Pb", "ligand_Cl", "bulk_Cl", "bulk_Pb", "Cs")


def shell_labels(atoms) -> np.ndarray:
    """Per-atom shell label. `classify` needs a vacancy to find, so on a STOICHIOMETRIC
    cell -- the very population the centre is built from -- it returns None. There the
    labels are the species alone, which is the right answer: with no vacancy there is no
    hub and no ligand cage, and every Cl is a bulk Cl."""
    got = classify(atoms)
    if got is not None:
        return np.asarray(got[0], dtype=object)
    sym = np.array(atoms.get_chemical_symbols())
    return np.array(["bulk_" + s if s in ("Cl", "Pb") else "Cs" for s in sym],
                    dtype=object)


def per_shell(model, frames, z_table, cutoff, device, ctx, batch_size=4):
    """Mean correction per shell over a population, and the atom counts."""
    acc = {s: [] for s in SHELLS}
    for batch, frs in make_batches(frames, z_table, cutoff, batch_size, device):
        corr, _, _ = centred_correction(model, batch, frs, ctx)
        ptr = batch.ptr.cpu().numpy()
        for g, atoms in enumerate(frs):
            sl = slice(int(ptr[g]), int(ptr[g + 1]))
            shell = shell_labels(atoms)
            block = corr[sl]
            for s in SHELLS:
                m = shell == s
                if m.any():
                    acc[s].append(float(np.mean(block[m])))
    return {s: (float(np.mean(v)), float(np.std(v)), len(v)) for s, v in acc.items() if v}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    z_table = tools.AtomicNumberTable([17, 55, 82])
    frames = load_frames(args.data)
    neutral, charged = select(frames, charged=False), select(frames, charged=True)
    pops = {
        "charged 159 (the defect)": [a for a in charged if len(a) == 159][: args.frames],
        "neutral 159 (no defect)": [a for a in neutral if len(a) == 159][: args.frames],
        "neutral 80 (the centre's own population)":
            [a for a in neutral if len(a) == 80][: args.frames],
        "neutral 79 (a vacancy, no carrier)":
            [a for a in neutral if len(a) == 79][: args.frames],
    }
    for k, v in pops.items():
        print(f"  {k}: {len(v)} frames")

    rows = []
    for mp in args.models:
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        adopt_model_dtype(model)
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        cut = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        entry = {"model": mp.name, "populations": {}}
        print(f"\n=== {mp.name} (centre_form "
              f"{getattr(model.spectral.h, 'centre_form', 'output')}) ===")
        for label, pool in pops.items():
            if not pool:
                continue
            res = per_shell(model, pool, z_table, cut, args.device, ctx)
            entry["populations"][label] = res
            print(f"  {label}")
            for s in SHELLS:
                if s in res:
                    mean, sd, n = res[s]
                    print(f"      {s:11s} {mean:+8.4f} ± {sd:.4f} eV over {n} frames")
        rows.append(entry)

    print("\n=== the discriminator: bulk Cl, by population ===")
    for label in pops:
        vals = [r["populations"][label]["bulk_Cl"][0] for r in rows
                if label in r["populations"] and "bulk_Cl" in r["populations"][label]]
        if vals:
            print(f"  {label:44s} {np.mean(vals):+8.4f} ± {np.std(vals):.4f} eV")
    print("\n  If the neutral 159 offset matches the charged one, the shift is an artefact")
    print("  of centring an 80-atom reference against a 159-atom cell, not defect physics.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
