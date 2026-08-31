"""R0 -- is there a Pb-Pb bond fingerprint in the residual forces?

The last cycle localised the problem: not compactness, but SITE SELECTION between the hub
(the two vacancy-sharing Pb) and the spokes (their ten ligand Cl). Every compact failure sat
inside the vacancy neighbourhood; the head simply preferred the cage.

A spoke carrier reproduces a Pb-dominated force pattern automatically, because each Pb is
first-shell to five ligands, and does so with five times the fitting freedom of a two-atom
state. So "the residual is large on Pb" cannot separate the hypotheses. What can is the
DIRECTION: a hole in a Pb-Pb sigma-like state along the vacancy axis pushes the two Pb along
that axis, and the push should scale with their separation. A carrier delocalised over the
ligand cage has no reason to produce a coherent axial component.

Measured here, on the E0 base (neutral-only, so it has never seen a carrier):

  1. the residual force on each shell Pb projected onto the Pb-Pb axis -- sign, magnitude,
     and regression against d(Pb-Pb);
  2. residual magnitude on the ten topological ligands against the two Pb;

both against the neutral null control, which carries the same geometry and no carrier.

If the axial component is comparable to the null, one ensemble cannot separate hub from
spokes and no head will fix it -- decision tree section 10, midpoint antisymmetric reference.
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
# e0_residual_maps.evaluate returns per-frame BASE forces, which is what the residual is
# defined against. arm_gates.evaluate returns scored rows built from total forces.
from e0_residual_maps import evaluate  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402


def axis_and_residuals(atoms, base_forces):
    """Axial residual on the two hub Pb, plus hub/cage/bulk magnitudes."""
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    if site.cage is None or len(site.cage) == 0:
        return None

    a, b = int(site.shell[0]), int(site.shell[1])
    pos = atoms.get_positions()
    cell, pbc = atoms.get_cell(), atoms.pbc

    # Minimum-image Pb->Pb vector, so the axis is right across a boundary.
    vec, dist = get_distances(pos[a][None], pos[b][None], cell=cell, pbc=pbc)
    axis = vec[0, 0] / max(float(dist[0, 0]), 1e-12)
    d_pbpb = float(dist[0, 0])

    dF = atoms.arrays["REF_forces"] - base_forces
    mag = np.linalg.norm(dF, axis=1)

    # Positive = the two Pb pushed APART along their common axis (outward), which is the
    # sign expected for a hole in a Pb-Pb antibonding-derived level.
    axial = 0.5 * (float(np.dot(dF[b], axis)) - float(np.dot(dF[a], axis)))
    perp_a = float(np.linalg.norm(dF[a] - np.dot(dF[a], axis) * axis))
    perp_b = float(np.linalg.norm(dF[b] - np.dot(dF[b], axis) * axis))

    sym = np.array(atoms.get_chemical_symbols())
    hub = np.array([a, b])
    cage = np.asarray(site.cage, dtype=int)
    bulk = np.setdiff1d(np.flatnonzero(sym == "Pb"), hub)

    return dict(
        d_pbpb=d_pbpb, axial=axial,
        perp=0.5 * (perp_a + perp_b),
        hub_mag=float(mag[hub].mean()),
        cage_mag=float(mag[cage].mean()) if len(cage) else np.nan,
        bulk_pb_mag=float(mag[bulk].mean()) if len(bulk) else np.nan,
        n_cage=len(cage),
    )


def collect(model, frames, z_table, cutoff, device):
    base, _ = evaluate(model, frames, z_table, cutoff, device)
    rows = [axis_and_residuals(a, base[k]) for k, a in enumerate(frames)]
    return [r for r in rows if r is not None]


def summarise(name, rows):
    ax = np.array([r["axial"] for r in rows])
    d = np.array([r["d_pbpb"] for r in rows])
    hub = np.array([r["hub_mag"] for r in rows])
    cage = np.array([r["cage_mag"] for r in rows])
    bulk = np.array([r["bulk_pb_mag"] for r in rows])
    perp = np.array([r["perp"] for r in rows])

    print(f"\n=== {name} ({len(rows)} frames, mean cage size "
          f"{np.mean([r['n_cage'] for r in rows]):.1f}) ===")
    print(f"  axial residual (+ = Pb pushed apart):")
    print(f"    mean {ax.mean():+.4f}  median {np.median(ax):+.4f}  "
          f"std {ax.std():.4f} eV/A")
    print(f"    fraction positive: {float((ax > 0).mean()):.2f}")
    print(f"    |axial| median {np.median(np.abs(ax)):.4f}  vs perpendicular "
          f"{np.median(perp):.4f} eV/A  -> axial/perp {np.median(np.abs(ax))/max(np.median(perp),1e-12):.2f}")
    print(f"  magnitudes: hub Pb {hub.mean():.4f}   cage Cl {np.nanmean(cage):.4f}   "
          f"bulk Pb {np.nanmean(bulk):.4f} eV/A")
    print(f"    hub/cage {hub.mean()/max(np.nanmean(cage),1e-12):.2f}   "
          f"hub/bulk {hub.mean()/max(np.nanmean(bulk),1e-12):.2f}")

    if len(rows) > 4:
        k, c = np.polyfit(d, ax, 1)
        r = np.corrcoef(d, ax)[0, 1]
        print(f"  axial vs d(Pb-Pb): slope {k:+.4f} eV/A per A, intercept {c:+.4f}, "
              f"r = {r:+.3f}")
    return dict(axial_mean=float(ax.mean()), axial_median=float(np.median(ax)),
                axial_abs_median=float(np.median(np.abs(ax))),
                perp_median=float(np.median(perp)),
                frac_positive=float((ax > 0).mean()),
                hub=float(hub.mean()), cage=float(np.nanmean(cage)),
                bulk=float(np.nanmean(bulk)),
                slope=float(np.polyfit(d, ax, 1)[0]) if len(rows) > 4 else None,
                r=float(np.corrcoef(d, ax)[0, 1]) if len(rows) > 4 else None)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_e0")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    out = {}
    for label, fname in (("V_Cl+ (charged)", "eval_qp1.xyz"),
                         ("V_Cl0 (null control)", "eval_q0_null.xyz")):
        frames = read(args.data / fname, ":")[: args.limit]
        rows = collect(model, frames, z_table, args.cutoff, args.device)
        out[label] = summarise(label, rows)

    p, n = out["V_Cl+ (charged)"], out["V_Cl0 (null control)"]
    print("\n=== R0 verdict ===")
    print(f"  axial |residual|  charged {p['axial_abs_median']:.4f}   "
          f"null {n['axial_abs_median']:.4f}   "
          f"excess x{p['axial_abs_median']/max(n['axial_abs_median'],1e-12):.2f}")
    print(f"  signed axial mean charged {p['axial_mean']:+.4f} "
          f"(fraction positive {p['frac_positive']:.2f}) vs null {n['axial_mean']:+.4f} "
          f"({n['frac_positive']:.2f})")
    print(f"  hub/cage ratio    charged {p['hub']/max(p['cage'],1e-12):.2f}   "
          f"null {n['hub']/max(n['cage'],1e-12):.2f}")
    # The plan's criterion is "a clear distance-dependent axial component", judged against
    # the null. An earlier version of this test also demanded a consistent SIGN, which is
    # wrong here: the regression crosses zero at d ~ 5.56 A, essentially the median Pb-Pb
    # separation, so a genuine linear relation produces ~50% positive by construction. Sign
    # consistency would only be expected if the equilibrium separation sat outside the
    # sampled range, and requiring it labelled a 15x signal "weak".
    excess = p["axial_abs_median"] / max(n["axial_abs_median"], 1e-12)
    anisotropy = (p["axial_abs_median"] / max(p["perp_median"], 1e-12)) / max(
        n["axial_abs_median"] / max(n["perp_median"], 1e-12), 1e-12)
    strong = (excess > 3.0 and abs(p["r"] or 0.0) > 0.3 and anisotropy > 2.0)
    print(f"  axial/perp anisotropy vs null: x{anisotropy:.2f}")
    print(f"  distance regression r: charged {p['r']:+.3f}  null {n['r']:+.3f}")
    print(f"  axial bond fingerprint: {'PRESENT' if strong else 'WEAK'}")
    print("  WEAK means one ensemble cannot separate hub from spokes; see plan section 10")
    print("  (midpoint antisymmetric reference) before building further heads.")

    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
