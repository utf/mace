#!/usr/bin/env python3
"""Where the harness's c-shift and the trainer's actually diverge.

THE STANDING FACTS. On the same data and base, the harness calibrates the head's energy zero to
+8.70 eV and the trainer to +36.54. b11 tested the obvious explanation -- that the harness's
`select(...)[:48]` slices the file in trajectory order and lands somewhere unrepresentative --
and FALSIFIED it: the raw `(E_label - E_base)/Delta_n` is +3.756 over the first forty-eight
against +3.587 across the whole file, a 5-95% span of 0.6 eV. The slice was harmless.

So the difference is not WHICH frames. It is what each driver computes on them, and there are
exactly three candidates, all testable on one model:

  A. THE FORWARD PATH. The harness calls the model through `ForwardContext.forward_dict`; the
     trainer calls `batch.to_dict()` directly. If those disagree about `base_energy` or
     `delta_sr_energy` then train and evaluate disagree about the forward pass, which is the
     failure this project has already paid for four times, and it would matter far beyond the
     c-shift.
  B. THE ENERGY LABEL. b11 read `atoms.info["REF_energy"]`; both calibrations read
     `batch.energy`, which has been through `prepare_defect_configurations` and its
     per-(charge, size) referencing. That is a different number by construction, and it is why
     b11's +3.6 matches neither calibration.
  C. THE HEAD AT INITIALISATION. `E_head` is a free-energy difference of order the frontier
     level, tens of eV, and the trainer zero-initialises the on-site channel while the harness
     does not. Both are legitimate; they are different starting points.

A is a bug if it fires. B and C are not bugs -- they are the two drivers doing different,
defensible things -- but which one dominates decides whether the harness's Stage-3
initialisation and the joint run's are comparable at all.

One model, one batch, both paths, all three terms printed.
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
from mace.modules.defect_context import ForwardContext
from mace.modules.defect_protocol import c_shift_terms, zero_on_site_correction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from stage_run import build, measure_t_ref  # noqa: E402
from ta_band_edge import counts_of, load_frames, select  # noqa: E402


def terms(model, batch, frames, ctx, label):
    """`base_energy`, `delta_sr_energy` and the c-shift ratio through one forward path."""
    with torch.no_grad():
        d = (ctx.forward_dict(batch, frames, requires_grad=False) if ctx is not None
             else batch.to_dict())
        out = model(d, training=False, compute_force=False)
    ratios = c_shift_terms(out, getattr(batch, "energy", None), batch.carrier_counts)
    return dict(path=label,
                base_energy=float(out["base_energy"].reshape(-1)[0]),
                delta_sr=float(out["delta_sr_energy"].reshape(-1)[0]),
                batch_energy=float(getattr(batch, "energy").reshape(-1)[0]),
                c_median=(float(torch.median(ratios)) if ratios is not None
                          and ratios.numel() else float("nan")))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path,
                    default=Path.home() / "runs" / "r2_h3_anneal_s5"
                    / "r2_h3_anneal_s5.model")
    ap.add_argument("--base", type=Path,
                    default=Path.home() / "runs" / "e0_base_s1" / "e0_base_s1.model")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=0,
                    help="take every Nth charged frame instead of the first --frames. The "
                    "decisive test: if the c-shift median over a spread sample matches the "
                    "trainer's whole-set value rather than the harness's first-48 value, the "
                    "difference is the frame set after all -- in the term b11 could not see, "
                    "which is E_head")
    ap.add_argument("--eps-inf", type=float, required=True,
                    help="per-host input; this script builds a model rather than loading one, so there is nothing to read it off and no default to fall back on")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    charged_all = select(load_frames(args.data), charged=True)
    frames = (charged_all[:: args.stride] if args.stride > 0
              else charged_all[: args.frames])
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"{len(frames)} charged frames of {len(charged_all)}, mean "
          f"{np.mean([len(a) for a in frames]):.1f} atoms, "
          f"{'stride ' + str(args.stride) if args.stride else 'first ' + str(args.frames)}",
          flush=True)
    dn = np.array([float(sum(c[:2]) - sum(c[2:]))
                   for c in (counts_of(a) for a in frames)])
    print(f"  Delta_n: {dict(zip(*[x.tolist() for x in np.unique(dn, return_counts=True)]))}",
          flush=True)

    quiet = lambda *a, **k: None                    # noqa: E731
    probe = make_batches(frames, z, 10.0, len(frames), args.device)[0][0]
    t_ref = measure_t_ref(probe)
    model = build(args.arch, args.base, 1, args.device, 3, True, args.eps_inf, t_ref, quiet)
    model.eval()
    cutoff = max(float(model.r_max), float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
    ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
    batch, frs = make_batches(frames, z, cutoff, len(frames), args.device)[0]

    rows = []
    # A -- the two forward paths, on the SAME model and batch.
    rows.append(terms(model, batch, frs, ctx, "harness: ForwardContext"))
    rows.append(terms(model, batch, frs, None, "trainer: batch.to_dict()"))
    # C -- the same model with the on-site channel zeroed, as the joint run starts it.
    zero_on_site_correction(model)
    rows.append(terms(model, batch, frs, ctx, "zero-init on-site, ForwardContext"))
    rows.append(terms(model, batch, frs, None, "zero-init on-site, to_dict()"))

    print("\n=== one model, one batch, first graph ===")
    print(f"  {'path':34s} {'base_energy':>14} {'delta_sr':>12} {'batch.energy':>14} "
          f"{'c median':>10}")
    for r in rows:
        print(f"  {r['path']:34s} {r['base_energy']:14.4f} {r['delta_sr']:12.4f} "
              f"{r['batch_energy']:14.4f} {r['c_median']:10.4f}")

    # B -- the referenced label against the raw one, which is what b11 read.
    raw = [float(a.info.get("REF_energy", a.info.get("energy", float("nan"))))
           for a in frs]
    ref = batch.energy.reshape(-1).tolist()
    delta = np.array(ref) - np.array(raw)
    print(f"\n  batch.energy - REF_energy: {delta.mean():+.4f} +- {delta.std():.4f} eV "
          f"over {len(raw)} frames")

    same_path = abs(rows[0]["c_median"] - rows[1]["c_median"])
    zero_effect = abs(rows[0]["c_median"] - rows[2]["c_median"])
    print(f"\n  A  forward paths differ by {same_path:.6f} eV in the c-shift  -> "
          f"{'THE SAME' if same_path < 1e-6 else 'THEY DISAGREE -- this is a bug'}")
    print(f"  B  referencing moves the label by {delta.mean():+.3f} eV, which is why b11's "
          f"+3.6 matches neither driver")
    print(f"  C  zero-initialising the on-site channel moves the c-shift by "
          f"{zero_effect:.3f} eV")
    args.out.write_text(json.dumps(
        dict(rows=rows, label_shift=float(delta.mean()),
             forward_path_gap=float(same_path), zero_init_gap=float(zero_effect)),
        indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
