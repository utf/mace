#!/usr/bin/env python3
"""Where does the on-site correction actually go? Per atom, by shell and by distance.

WHY IT MATTERS NOW. The species audit at gamma = 3 reports every species at 0% saturated with
a pre-tanh mean near zero -- the opposite of the gamma = 1 picture, where every chlorine was
pinned. But a species MEAN near zero is consistent with two very different heads: one whose
correction genuinely varies from atom to atom around zero, and one whose correction is a
species constant that `eps0` already supplies, leaving the feature-dependent channel inert.
The species statistic cannot tell them apart. The per-atom SPREAD, and its structure against
distance to the vacancy, can.

WHAT IS MEASURED, per seed, over the charged 159-atom frames:

  * the pre-tanh argument and the correction it produces, `gamma * tanh(pre)` in eV, for the
    two hub Pb, their ligand-Cl shell, and the bulk of each species;
  * the within-species spread, which is the quantity that says whether the channel carries
    any structure at all;
  * the profile against distance to the vacancy, which says whether what structure there is
    is defect-local or ambient.

F10 is scored here: with a working correction channel, the ligand Cl near the vacancy should
separate from bulk Cl. The criterion has TWO conditions, both fixed before the numbers exist,
because either alone is gameable. RESOLVED: the ligand-Cl mean sits more than two within-shell
standard deviations from the bulk-Cl mean. MATERIAL: the two corrections differ by more than
50 meV. If the channel is inert the within-shell sd collapses towards zero and the first test
passes on a 2 meV difference, which would be a pass certifying that nothing happened.

DIAGNOSTIC SCOPE. The vacancy position and the hub/ligand assignment are evaluation
machinery. They pick which atoms to PRINT and never enter the model, the loss or any input.
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
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import adopt_model_dtype, make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

SYMBOL = {0: "Cl", 1: "Cs", 2: "Pb"}
SHELLS = ("hub_Pb", "ligand_Cl", "bulk_Cl", "bulk_Pb", "Cs")
BINS = np.array([0.0, 3.0, 4.5, 6.0, 8.0, 10.0, 99.0])
MEANINGFUL_EV = 0.05      # a correction difference smaller than this decides nothing


def classify(atoms):
    """Shell label and distance-to-vacancy per atom. Returns None if the site is unclear."""
    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    sym = np.array(atoms.get_chemical_symbols())
    shell = np.array(["bulk_" + s if s in ("Cl", "Pb") else "Cs" for s in sym],
                     dtype=object)
    hub = np.asarray(site.shell, dtype=int)
    shell[hub] = "hub_Pb"
    if site.cage is not None and len(site.cage):
        shell[np.asarray(site.cage, dtype=int)] = "ligand_Cl"
    _, dd = get_distances(np.asarray(site.position)[None], atoms.get_positions(),
                          cell=atoms.get_cell(), pbc=atoms.pbc)
    return shell, dd[0]


def pre_tanh(model, batch, frames, ctx):
    """The on-site MLP's pre-tanh output, `[n_atoms, 2]`, for one graph.

    The FIRST pass only: `_carrier_head` runs twice per forward, once at this frame's
    counters and once at the reference, and the buffer would otherwise hold two passes over
    the same atoms.
    """
    head = model.spectral
    head.h._audit_bin = {}
    bin_ = head.h.audit(True)
    try:
        with torch.no_grad():
            model(ctx.forward_dict(batch, frames, requires_grad=False),
                  training=False, compute_force=False)
        first = bin_["site"][0]
    finally:
        head.h.audit(False)
    n = int(batch.node_attrs.shape[0])
    if first.numel() % n:
        raise SystemExit(f"site channel has {first.numel()} entries for {n} atoms")
    return first.reshape(n, -1).float().numpy()


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == CLEAN_NATOMS][:args.frames]
    tagged = [(a, classify(a)) for a in frames]
    tagged = [(a, c) for a, c in tagged if c is not None]
    print(f"{len(tagged)} charged {CLEAN_NATOMS}-atom frames with a located vacancy",
          flush=True)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        # The batches built for this model must carry ITS dtype: AtomicData uses the
        # process default, which is float32, while the joint run trains at float64.
        adopt_model_dtype(model)
        gamma = float(model.spectral.h.on_site_range)
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)

        pre_by_shell = {s: [] for s in SHELLS}
        pre_by_bin = {i: [] for i in range(len(BINS) - 1)}
        for atoms, (shell, dist) in tagged:
            batch, fr = make_batches([atoms], z, cutoff, 1, args.device)[0]
            pre = pre_tanh(model, batch, fr, ctx)
            # The s shell carries the level the two-centre argument speaks about; the p
            # shell is reported alongside so a channel-specific effect is visible.
            for s in SHELLS:
                m = shell == s
                if m.any():
                    pre_by_shell[s].append(pre[m])
            which = np.digitize(dist, BINS) - 1
            for i in pre_by_bin:
                m = which == i
                if m.any():
                    pre_by_bin[i].append(pre[m])

        row = {"model": Path(mp).name, "gamma": gamma, "shells": {}, "bins": {}}
        for s in SHELLS:
            if not pre_by_shell[s]:
                continue
            v = np.concatenate(pre_by_shell[s], axis=0)
            row["shells"][s] = dict(
                n=int(v.shape[0]),
                pre_s_mean=float(v[:, 0].mean()), pre_s_std=float(v[:, 0].std()),
                pre_p_mean=float(v[:, 1].mean()), pre_p_std=float(v[:, 1].std()),
                corr_s_mean=float(gamma * np.tanh(v[:, 0]).mean()),
                corr_s_std=float(gamma * np.tanh(v[:, 0]).std()))
        for i, lo in enumerate(BINS[:-1]):
            if not pre_by_bin[i]:
                continue
            v = np.concatenate(pre_by_bin[i], axis=0)
            row["bins"][f"{lo:.1f}-{BINS[i + 1]:.1f}"] = dict(
                n=int(v.shape[0]), pre_s_mean=float(v[:, 0].mean()),
                pre_s_std=float(v[:, 0].std()))
        rows.append(row)

        print(f"\n  {row['model']}  (gamma {gamma} eV)")
        print(f"    {'shell':11s} {'n':>6}  {'pre-tanh s':>18}  {'correction s (eV)':>20}")
        for s in SHELLS:
            d = row["shells"].get(s)
            if d:
                print(f"    {s:11s} {d['n']:6d}  {d['pre_s_mean']:+9.4f} "
                      f"+- {d['pre_s_std']:.4f}  {d['corr_s_mean']:+11.4f} "
                      f"+- {d['corr_s_std']:.4f}")
        args.out.write_text(json.dumps(rows, indent=2, default=float))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    if rows:
        print("\n=== pooled over seeds ===")
        print(f"  {'shell':11s} {'pre-tanh s':>20}  {'correction s (eV)':>22}")
        pooled = {}
        for s in SHELLS:
            m = np.array([r["shells"][s]["pre_s_mean"] for r in rows if s in r["shells"]])
            sd = np.array([r["shells"][s]["pre_s_std"] for r in rows if s in r["shells"]])
            c = np.array([r["shells"][s]["corr_s_mean"] for r in rows if s in r["shells"]])
            if m.size:
                pooled[s] = (m.mean(), sd.mean(), c.mean())
                print(f"  {s:11s} {m.mean():+9.4f} +- {m.std():.4f} "
                      f"(within-shell sd {sd.mean():.4f})  {c.mean():+11.4f} eV")
        if "ligand_Cl" in pooled and "bulk_Cl" in pooled:
            lig, bulk = pooled["ligand_Cl"], pooled["bulk_Cl"]
            resolved = abs(lig[0] - bulk[0]) > 2.0 * bulk[1]
            gap_ev = abs(lig[2] - bulk[2])
            material = gap_ev > MEANINGFUL_EV
            print(f"\n  F10: ligand-Cl mean {lig[0]:+.4f} against bulk-Cl "
                  f"{bulk[0]:+.4f} +- {bulk[1]:.4f} (within-shell sd); the two corrections "
                  f"differ by {gap_ev:.4f} eV")
            # TWO conditions, and the second is not decoration. With a within-shell sd near
            # zero the 2-sigma test is a division by almost nothing: a 2 meV difference
            # clears it while meaning nothing about the physics. A separation has to be
            # resolved AND large enough to matter, so both are required and both printed.
            print(f"  -> statistically resolved: {'yes' if resolved else 'no'};  "
                  f"larger than {MEANINGFUL_EV} eV: {'yes' if material else 'no'}")
            print(f"  -> F10 {'HOLDS' if (resolved and material) else 'FAILS'}")
        print("\n  A within-shell sd near zero means the channel is INERT, not balanced: "
              "the MLP is\n  returning a species constant that eps0 already supplies, and "
              "carries no structure.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
