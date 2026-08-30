"""E0: does the unpaired data contain site-resolved information about where the carrier goes?

A base trained only on n = 0 frames (pristine + V_Cl0) has never seen a carrier. Evaluate it
on V_Cl+ frames and the residual dF_i = F_DFT - F_base is whatever the geometry-only surface
cannot explain. If the carrier leaves a site-resolved fingerprint in the labels, that residual
should concentrate on the two under-coordinated Pb.

The comparison that matters is against a NULL CONTROL: the same residual map on held-out
V_Cl0 frames, which the base also never trained on. That measures how badly the base
extrapolates to defective geometries with no carrier present. Only the excess of (+) over null
is evidence of carrier signal; the raw (+) map on its own would mostly measure extrapolation
error near a vacancy, which is large for reasons that have nothing to do with a hole.

PASS (indicative, judge by the maps): shell Pb carry >= 3x the bulk-Pb median residual, in
excess of the null, decaying with distance. FAIL: (+) indistinguishable from null.

The vacancy position is used to bin and classify atoms and for nothing else. It never touches
the model, which is evaluated purely on geometry and counters.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (must precede e3nn: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)


def _assert_repo(strict=True):
    """Fail loudly if `mace` did not resolve to the tree this script lives in.

    Python puts the launch directory first on sys.path, so running this from
    `defect-perovskite/` -- a directory with no `mace/` in it -- silently falls through to the
    editable install, which points at the main checkout on a different branch. That happened
    on the first run here and produced a canonicalisation error from code that is not under
    test. Comparisons attributed to the wrong revision have bitten this project before, so
    check rather than assume.
    """
    want = Path(__file__).resolve().parent.parent
    got = Path(mace.__file__).resolve().parent.parent
    if got != want:
        msg = (f"`mace` resolved to {got}, not {want}.\n"
               f"Run from the repo root, e.g.:\n"
               f"  cd {want} && python defect-perovskite/{Path(__file__).name} ...\n"
               f"or set PYTHONPATH={want}.")
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
    head = subprocess.run(["git", "-C", str(want), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(want), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip().splitlines()
    print(f"code: {want}  @ {head}  ({len(dirty)} dirty files)")


_assert_repo(strict=not os.environ.get("E0_ALLOW_ANY_REPO"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data.defects import prepare_defect_configurations
from mace.tools import torch_geometric

from vacancy_site import distance_to_vacancy, locate_vacancy

KEYSPEC = mace_data.KeySpecification(
    info_keys={"energy": "REF_energy", "carrier_counts": "carrier_counts",
               "multiplicity": "multiplicity", "host": "host",
               "m_s_ref_doubled": "m_s_ref_doubled", "cell_charge": "cell_charge",
               "e_cbm_cell": "e_cbm_cell", "e_vbm_cell": "e_vbm_cell"},
    arrays_keys={"forces": "REF_forces"})


def make_batch(frames, z_table, cutoff, device):
    configs = [mace_data.config_from_atoms(a, key_specification=KEYSPEC) for a in frames]
    prepare_defect_configurations(configs)
    atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff)
              for c in configs]
    loader = torch_geometric.dataloader.DataLoader(atomic, batch_size=len(atomic),
                                                   shuffle=False)
    return next(iter(loader)).to(device)


def evaluate(model, frames, z_table, cutoff, device, batch_size=4):
    """Per-atom base forces and per-frame base energy for every frame."""
    out_f, out_e = [], []
    for s in range(0, len(frames), batch_size):
        chunk = frames[s:s + batch_size]
        batch = make_batch(chunk, z_table, cutoff, device)
        res = model(batch.to_dict(), training=False, compute_force=True)
        bf = res["base_forces"].detach().cpu().numpy()
        be = res["base_energy"].detach().cpu().numpy()
        idx = batch.batch.detach().cpu().numpy()
        for k in range(len(chunk)):
            out_f.append(bf[idx == k])
        out_e.extend(np.atleast_1d(be).tolist())
    return out_f, np.array(out_e)


def check_base_is_geometry_only(model, data_dir, z_table, cutoff, device):
    """The whole analysis reads `base_forces` on charged frames. Verify it ignores counters.

    E_base is meant to be a function of geometry alone, so the residual F_DFT - F_base is the
    part of the charged label the geometry-only surface cannot express. If base_forces
    depended on the carrier counters at all, that reading would be wrong and every number
    downstream would be contaminated. Structural claims deserve a measurement.
    """
    from ase.io import read

    frames = read(data_dir / "eval_qp1.xyz", ":4")
    charged = evaluate(model, frames, z_table, cutoff, device)[0]

    # Establish the numerical noise floor first, by evaluating the SAME inputs twice. In
    # float32 this is ~8e-7 eV/A: changing the counters changes batch contents and hence
    # reduction order, so a fixed absolute tolerance below this floor fails on round-off. An
    # earlier 1e-6 threshold did exactly that. Confirmed by repeating in float64, where both
    # this floor and the counter-change deviation collapse to ~1.4e-15.
    repeat = evaluate(model, frames, z_table, cutoff, device)[0]
    noise = max(float(np.abs(a - b).max()) for a, b in zip(charged, repeat))

    neutralised = []
    for a in frames:
        b = a.copy()
        b.info = dict(a.info)
        b.arrays = dict(a.arrays)
        # Relabel as the neutral state at the same geometry. A V_Cl0 cell is a doublet, so
        # multiplicity must move with the counters or the consistency check rejects it.
        b.info["carrier_counts"] = np.array([0, 0, 0, 0])
        b.info["multiplicity"] = 2
        b.info["cell_charge"] = 0
        neutralised.append(b)
    neutral = evaluate(model, neutralised, z_table, cutoff, device)[0]

    worst = max(float(np.abs(c - n).max()) for c, n in zip(charged, neutral))
    scale = max(float(np.abs(c).max()) for c in charged)
    tol = max(4.0 * noise, 1e-6 * scale)
    print(f"base_forces counter-independence: deviation {worst:.3e} eV/A "
          f"vs noise floor {noise:.3e} (tol {tol:.3e}, force scale {scale:.3f} eV/A)")
    if worst > tol:
        raise SystemExit(
            f"base_forces changed by {worst:.3e} eV/A when only the counters changed, well "
            f"above the {noise:.3e} eV/A numerical noise floor. E_base is not geometry-only, "
            "so the E0 residual cannot be read as carrier signal. Investigate before "
            "trusting any number here.")


def analyse(frames, base_forces, base_energy, label):
    """Residual statistics, classified by species and by distance to the vacancy."""
    rows, dumps, skipped = [], [], 0
    for k, atoms in enumerate(frames):
        try:
            site = locate_vacancy(atoms)
        except ValueError:
            skipped += 1
            continue
        d = distance_to_vacancy(atoms, site)
        sym = np.array(atoms.get_chemical_symbols())
        dF = atoms.arrays["REF_forces"] - base_forces[k]
        mag = np.linalg.norm(dF, axis=1)

        shell = np.zeros(len(atoms), dtype=bool)
        shell[site.shell] = True
        is_pb = sym == "Pb"
        bulk_pb = is_pb & ~shell
        if not bulk_pb.any():
            skipped += 1
            continue

        bulk_med = float(np.median(mag[bulk_pb]))
        rows.append(dict(
            frame=k,
            shell_mean=float(mag[shell].mean()),
            bulk_pb_median=bulk_med,
            ratio=float(mag[shell].mean() / bulk_med) if bulk_med > 0 else np.nan,
            shell_fraction=float((mag[shell] ** 2).sum() / (mag ** 2).sum()),
            e_residual=float(atoms.info["REF_energy"] - base_energy[k]),
            cs=float(mag[sym == "Cs"].mean()), cl=float(mag[sym == "Cl"].mean()),
            pb=float(mag[is_pb].mean()), all_mean=float(mag.mean()),
            margin=site.margin,
        ))
        dumps.append(dict(d=d.astype(np.float32), mag=mag.astype(np.float32),
                          sym=sym, shell=shell))

    if not rows:
        raise SystemExit(f"{label}: no frames could be analysed")
    keys = rows[0].keys()
    arr = {k: np.array([r[k] for r in rows]) for k in keys}

    # |dF| against distance to the vacancy, pooled over frames.
    alld = np.concatenate([x["d"] for x in dumps])
    allm = np.concatenate([x["mag"] for x in dumps])
    edges = np.array([0, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 12.0, 100.0])
    which = np.digitize(alld, edges) - 1
    profile = [(float(edges[b]), float(edges[b + 1]), int((which == b).sum()),
                float(np.median(allm[which == b])) if (which == b).any() else np.nan)
               for b in range(len(edges) - 1)]

    return arr, profile, dumps, skipped


def summarise(name, arr, profile, skipped, n):
    print(f"\n=== {name} ===")
    print(f"  frames analysed {len(arr['ratio'])}/{n} (vacancy unlocatable on {skipped})")
    r = arr["ratio"][np.isfinite(arr["ratio"])]
    print(f"  shell/bulk-Pb |dF| ratio: median {np.median(r):.2f}  "
          f"mean {r.mean():.2f}  p25 {np.percentile(r,25):.2f}  p75 {np.percentile(r,75):.2f}")
    print(f"  shell |dF| {arr['shell_mean'].mean():.4f} eV/A  "
          f"bulk-Pb median {arr['bulk_pb_median'].mean():.4f} eV/A")
    print(f"  shell energy fraction (2 of N atoms): {arr['shell_fraction'].mean():.4f}")
    print(f"  per-species mean |dF|: Pb {arr['pb'].mean():.4f}  "
          f"Cl {arr['cl'].mean():.4f}  Cs {arr['cs'].mean():.4f} eV/A")
    print(f"  energy residual: mean {arr['e_residual'].mean():.4f} eV  "
          f"std {arr['e_residual'].std():.4f} eV")
    print("  |dF| vs distance to vacancy (median, eV/A):")
    for lo, hi, cnt, med in profile:
        if cnt:
            print(f"     {lo:5.1f}-{hi:5.1f} A  n={cnt:6d}  {med:.4f}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_e0")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = torch.load(args.model, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    check_base_is_geometry_only(model, args.data, z_table, args.cutoff, args.device)

    results = {}
    for label, fname in (("V_Cl+ (charged)", "eval_qp1.xyz"),
                         ("V_Cl0 (null control)", "eval_q0_null.xyz")):
        frames = read(args.data / fname, ":")
        if args.limit:
            frames = frames[: args.limit]
        bf, be = evaluate(model, frames, z_table, args.cutoff, args.device)
        arr, profile, dumps, skipped = analyse(frames, bf, be, label)
        summarise(label, arr, profile, skipped, len(frames))
        results[label] = (arr, profile, dumps)

    plus = results["V_Cl+ (charged)"][0]
    null = results["V_Cl0 (null control)"][0]
    rp = np.median(plus["ratio"][np.isfinite(plus["ratio"])])
    rn = np.median(null["ratio"][np.isfinite(null["ratio"])])
    print("\n=== E0 verdict ===")
    print(f"  shell/bulk ratio  charged {rp:.2f}   null {rn:.2f}   excess {rp - rn:+.2f}")
    print(f"  shell energy fraction  charged {plus['shell_fraction'].mean():.4f}"
          f"   null {null['shell_fraction'].mean():.4f}")
    verdict = ("PASS" if rp >= 3.0 and rp > 1.5 * rn else
               "FAIL" if rp < 1.5 or rp <= 1.15 * rn else "MARGINAL")
    print(f"  indicative verdict: {verdict}  "
          f"(PASS needs charged >= 3.0 and clearly above null)")

    # Does the verdict rest on frames where the vacancy site was confidently identified?
    # A handful of frames have near-tied bridges; if excluding them moves the answer, the
    # result is really about the locator rather than about the physics.
    for tag, arr in (("charged", plus), ("null", null)):
        conf = arr["margin"] > 0.5
        r_all = arr["ratio"][np.isfinite(arr["ratio"])]
        r_conf = arr["ratio"][np.isfinite(arr["ratio"]) & conf]
        if len(r_conf) and len(r_conf) != len(r_all):
            print(f"  {tag}: ratio {np.median(r_all):.2f} over all "
                  f"{len(r_all)} frames, {np.median(r_conf):.2f} over the "
                  f"{len(r_conf)} with locator margin > 0.5 A")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            **{f"{tag}_{k}": v for tag, (arr, _, _) in
               (("plus", results["V_Cl+ (charged)"]),
                ("null", results["V_Cl0 (null control)"]))
               for k, v in arr.items()})
        meta = {"model": str(args.model), "verdict": verdict,
                "ratio_charged": float(rp), "ratio_null": float(rn),
                "profile_charged": results["V_Cl+ (charged)"][1],
                "profile_null": results["V_Cl0 (null control)"][1]}
        args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"\n  dumped to {args.out}")


if __name__ == "__main__":
    main()
