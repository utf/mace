"""Does the correction reproduce the axial residual it is fitting, including its SIGN?

D1's decomposition showed every head producing a negative axial force on the hub pair (the
two Pb pulled together) while T2 reported a DFT residual with signed mean +0.1035 eV/A (pushed
apart). Those two numbers are not comparable: one is a median of the head's own force, the
other a mean against the cross-fit diagnostic bases rather than the Stage-A base these runs
actually used.

This makes the comparison like-for-like -- same frames, same model, same base:

    target_i     = F_DFT,i - F_base,i          what the correction is supposed to supply
    correction_i = F_total,i - F_base,i        what it actually supplies

both taken from ONE forward pass of the trained model, then projected on the hub axis with
r0_pair_force's convention (positive = the two hub Pb pushed apart). If the signs disagree
systematically, the head is anti-correlated with its own target and no amount of localisation
work would help.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo, make_batch  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402


def axial(fa, fb, axis):
    return 0.5 * (float(np.dot(fb, axis)) - float(np.dot(fa, axis)))



def graph_cutoff_for(model, override=None):
    """Neighbour-list radius the head was TRAINED with.

    run_train builds the graph at the spectral Hamiltonian's range (10 A here) and filters the
    trunk back to r_max inside the model. An analysis that rebuilds batches at r_max instead
    silently drops every 5-10 A edge, so the head is handed a truncated H: the hub pair at a
    median 5.43 A then has no edge at all and reports |H_ab| = 0 with no hopping term, which
    is an artefact of the analysis and not a property of the model. Derived from the model so
    it cannot drift from what training used.
    """
    if override:
        return float(override)
    return max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_cf" / "eval_qp1.xyz")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cutoff", type=float, default=None,
                    help="override; default = the model's own graph cutoff")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    _assert_repo()
    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    cutoff = graph_cutoff_for(model, args.cutoff)
    print(f"  graph cutoff {cutoff:.1f} A (model r_max {float(model.r_max):.1f})")
    frames = read(args.data, ":")[: args.limit]

    tgt, cor, skipped = [], [], 0
    for atoms in frames:
        try:
            site = locate_vacancy(atoms)
        except ValueError:
            skipped += 1
            continue
        a, b = int(site.shell[0]), int(site.shell[1])
        vec, dist = get_distances(atoms.get_positions()[a][None],
                                  atoms.get_positions()[b][None],
                                  cell=atoms.get_cell(), pbc=atoms.pbc)
        axis = vec[0, 0] / max(float(dist[0, 0]), 1e-12)

        batch = make_batch([atoms], z_table, cutoff, args.device)
        out = model(batch.to_dict(), training=False, compute_force=True)
        total = out["forces"].detach().cpu().numpy()
        base = out["base_forces"].detach().cpu().numpy()
        ref = atoms.arrays["REF_forces"]

        tgt.append(axial(ref[a] - base[a], ref[b] - base[b], axis))
        cor.append(axial(total[a] - base[a], total[b] - base[b], axis))

    tgt, cor = np.array(tgt), np.array(cor)
    if len(tgt) == 0:
        raise SystemExit("no frames analysable")

    agree = float(np.mean(np.sign(tgt) == np.sign(cor)))
    r = float(np.corrcoef(tgt, cor)[0, 1]) if len(tgt) > 2 else float("nan")

    print(f"\n=== D1-sign {args.label or args.model.stem}  "
          f"({len(tgt)} frames, {skipped} skipped) ===")
    print("  positive = the two hub Pb pushed apart (r0_pair_force convention)")
    print(f"  target   (F_DFT - F_base):   mean {tgt.mean():+.4f}  "
          f"median {np.median(tgt):+.4f}  frac>0 {float((tgt > 0).mean()):.2f}")
    print(f"  correction (F_tot - F_base): mean {cor.mean():+.4f}  "
          f"median {np.median(cor):+.4f}  frac>0 {float((cor > 0).mean()):.2f}")
    print(f"  sign agreement {agree:.2f}   Pearson r {r:+.3f}")
    print(f"  mean |target| {np.abs(tgt).mean():.4f}   "
          f"mean |correction| {np.abs(cor).mean():.4f}   "
          f"ratio {np.abs(cor).mean() / max(np.abs(tgt).mean(), 1e-12):.3f}")

    if args.out:
        args.out.write_text(json.dumps(dict(
            label=args.label or args.model.stem, n=len(tgt),
            target_mean=float(tgt.mean()), target_median=float(np.median(tgt)),
            corr_mean=float(cor.mean()), corr_median=float(np.median(cor)),
            sign_agreement=agree, pearson=r,
            target=tgt.tolist(), correction=cor.tolist()), indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
