#!/usr/bin/env python3
"""Which radial envelope should the rerun use? Measured at INITIALISATION, before any seed.

WHY AT INITIALISATION, and not on the trained models. `b3_hopping_channel.py` swaps the
envelope on trained weights and reports the result as a sensitivity bound, correctly: those
`v0` values were fitted under the exponential, so they have already absorbed part of what the
envelope failed to supply, and the swap is off-distribution. The rerun does not start there.
It starts from Harrison initialisation, where `v0` is EXACTLY the universal rule and the two
envelopes differ in one factor and nothing else. That is the comparison that predicts the
rerun, so it is the one that decides the config.

WHAT WOULD DISQUALIFY A CHOICE, fixed before the numbers exist. A longer-ranged envelope
strengthens every bond in the graph, not only the hub bond, so the risk is not subtle: the
bandwidth grows, the pristine spectrum stops looking like bands, and the frontier gap the
Stage-3 gate is defined on moves out of reach of `loss_gap`. So the pristine spectrum is
scored FIRST and a candidate that fails the initialisation gate is out regardless of what it
does for the hub bond.

    initialisation gate   band-edge spacings <= E_gap/2 and bandwidth >= 2 E_gap
    pristine frontier gap eps_{N+1} - eps_N on a defect-free cell, against the 2.40 eV target
    split fraction        (lam_2 - lam_1)/(lam_last - lam_1); the superatom failure

and only then

    t / t_Harrison        on the real hub bond, the quantity F7 fires on
    d(delta_sr)/dd        the clean derivative, hub Pb displaced along their own axis
    dlambda/dd            the level the two-centre argument speaks about

`decay_length` is scanned alongside the power law because "adjust the global length constant"
is the plan's own first option and it deserves to lose on a number rather than on an argument:
1.40 A matches Harrison's log-slope at `d_ref`, 3.02 A matches it at the median hub separation,
and the fact that those are different is the case against a retune.

Nothing here trains. Every model is built, Harrison-initialised, measured and discarded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_hopping_channel import (HubProbe, derivative, hub_of,  # noqa: E402
                                observables)
from d1_sensitivity import select_pristine  # noqa: E402
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS, hub_separation  # noqa: E402
from stage_run import build, measure_t_ref, pristine_spectrum_check  # noqa: E402
from ta_band_edge import capture, load_frames, select, with_hole_counter  # noqa: E402

ETA_PP_SIGMA = 3.24
HBAR2_OVER_M = 7.62
PP_SIGMA = 2

CANDIDATES = (
    ("exp_L1.0", dict(counting_envelope="exp", counting_decay_length=1.0)),
    ("exp_L1.4", dict(counting_envelope="exp", counting_decay_length=1.4)),
    ("exp_L2.0", dict(counting_envelope="exp", counting_decay_length=2.0)),
    ("exp_L3.0", dict(counting_envelope="exp", counting_decay_length=3.0)),
    ("power", dict(counting_envelope="power", counting_decay_length=1.0)),
)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--n-pristine", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--e-gap", type=float, default=2.4)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    charged = [a for a in select(load_frames(args.data), charged=True)
               if len(a) == CLEAN_NATOMS][:args.frames]
    keep = [(a, hub_of(a)) for a in charged]
    keep = [(a, p) for a, p in keep if p is not None]
    all_pristine = select_pristine(ase_read(str(args.data), ":"), 64)
    biggest = max(len(a) for a in all_pristine)
    pristine = with_hole_counter([a for a in all_pristine
                                  if len(a) == biggest][: args.n_pristine])
    d_hub = np.array([hub_separation(a) for a, _ in keep])
    print(f"{len(keep)} charged {CLEAN_NATOMS}-atom frames (d {d_hub.min():.2f}-"
          f"{d_hub.max():.2f} A), {len(pristine)} pristine cells of {biggest} atoms",
          flush=True)

    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    quiet = lambda *a, **k: None                       # noqa: E731
    rows = []
    for tag, overrides in CANDIDATES:
        per_seed = []
        for seed in range(1, args.seeds + 1):
            # t_ref is measured on the data, once per build, exactly as stage_run does it --
            # the Harrison initialisation is calibrated at the MEASURED bond length and a
            # different value here would make this scan describe a different model.
            probe_batch = make_batches([keep[0][0]], z, 10.0, 1, args.device)[0][0]
            t_ref = measure_t_ref(probe_batch)
            model = build(args.arch, args.base, seed, args.device, 3, True,
                          args.eps_inf, t_ref, quiet, counting_overrides=overrides)
            model.eval()
            head = model.spectral
            cutoff = max(float(model.r_max),
                         float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
            ctx = ForwardContext.production(model, device=args.device,
                                            eps_inf=args.eps_inf)
            pbatches = make_batches(pristine, z, cutoff, 1, args.device)

            # ------------------------------------------------ the disqualifying block
            from mace.modules.defect_counting import VALENCE, initialisation_gate

            internals, _ = capture(model, pbatches[0][0], ctx=ctx, frames=pbatches[0][1])
            lam0 = internals["lam"][0, 0]
            lam0 = lam0[lam0 < 500.0]
            n_el = sum(VALENCE[int(zz)]
                       for zz in pbatches[0][1][0].get_atomic_numbers()) / 2.0
            gate = initialisation_gate(lam0, n_el, args.e_gap)
            spec = pristine_spectrum_check(model, pbatches, quiet)

            # ------------------------------------------------------- the hub channel
            hub_t, sr, lam = [], [], []
            for atoms, pair in keep:
                _, _, cap = observables(model, atoms, z, cutoff, args.device, ctx, pair)
                hub_t.append(cap.get("hub_v", np.full(4, np.nan)))
                a, b = derivative(model, atoms, z, cutoff, args.device, ctx, pair,
                                  args.step)
                sr.append(a)
                lam.append(b)
            hub_t = np.array(hub_t, dtype=float)
            harr = ETA_PP_SIGMA * HBAR2_OVER_M / d_hub ** 2
            ratio = hub_t[:, PP_SIGMA] / harr

            per_seed.append(dict(
                seed=seed,
                init_gate=bool(gate["passed"]),
                bandwidth=float(gate["bandwidth"]),
                edge_below=float(gate["edge_spacing_below"]),
                edge_above=float(gate["edge_spacing_above"]),
                pristine_gap=float(spec.get("pristine_frontier_gap", np.nan)),
                split_fraction=float(spec.get("split_fraction", np.nan)),
                t_over_harrison=float(np.nanmean(ratio)),
                d_delta_sr_dd=float(np.nanmean(sr)),
                d_lambda_dd=float(np.nanmean(lam))))
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        def agg(key):
            v = np.array([p[key] for p in per_seed], dtype=float)
            return float(np.nanmean(v)), float(np.nanstd(v))

        row = dict(tag=tag, overrides=overrides, seeds=per_seed,
                   gate_pass=sum(p["init_gate"] for p in per_seed),
                   n_seeds=len(per_seed),
                   **{k: agg(k) for k in ("bandwidth", "pristine_gap", "split_fraction",
                                          "t_over_harrison", "d_delta_sr_dd",
                                          "d_lambda_dd")})
        rows.append(row)
        print(f"  {tag:10s} init gate {row['gate_pass']}/{row['n_seeds']}  "
              f"bandwidth {row['bandwidth'][0]:7.2f}  gap {row['pristine_gap'][0]:6.3f}  "
              f"split {row['split_fraction'][0]:.3f}  |  t/Harrison "
              f"{row['t_over_harrison'][0]:.3f}  d(delta_sr)/dd "
              f"{row['d_delta_sr_dd'][0]:+.4f}  dlambda/dd "
              f"{row['d_lambda_dd'][0]:+.4f}", flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    ok = [r for r in rows if r["gate_pass"] == r["n_seeds"]]
    print("\n=== the choice ===")
    if not ok:
        print("  NO candidate passes the initialisation gate on every seed. The envelope is "
              "not the\n  lever it looked like, and nothing should be rerun on this basis.")
    else:
        best = max(ok, key=lambda r: abs(r["d_lambda_dd"][0]))
        base = next((r for r in rows if r["tag"] == "exp_L1.0"), None)
        print(f"  candidates passing the gate: {', '.join(r['tag'] for r in ok)}")
        print(f"  largest |dlambda/dd| among them: {best['tag']} at "
              f"{best['d_lambda_dd'][0]:+.4f} eV/A")
        if base is not None and abs(base["d_lambda_dd"][0]) > 0:
            print(f"  against the current exp_L1.0 at {base['d_lambda_dd'][0]:+.4f} eV/A "
                  f"-> x{abs(best['d_lambda_dd'][0] / base['d_lambda_dd'][0]):.2f}")
        print("\n  The gate comes first on purpose: a candidate that lifts the hub coupling "
              "and loses\n  the pristine band structure has bought a d-slope with the thing "
              "the whole head is\n  meant to preserve.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
