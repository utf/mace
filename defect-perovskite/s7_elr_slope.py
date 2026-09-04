#!/usr/bin/env python3
"""Step 2b / F3: what is `d(E_LR)/dd` on the 16 large frames, with the branch switched on?

WHY IT DECIDES SOMETHING. F4's amplitude shortfall is ~0.073 eV/A -- the head reproduces
-0.061 against a residual slope of -0.131 -- which integrates to about 0.16 eV across the
d-range in hand. E_LR is a candidate for the missing piece only if its own d-slope is of that
order. The eFNV estimate is ~60 meV per cell, and a term whose TOTAL is 60 meV cannot supply
a 160 meV variation, so the expectation is that it cannot be the factor.

F3, on record: `|d(E_LR)/dd| < 0.01 eV/A` takes E_LR off the candidate list.

MEASURED, NOT ESTIMATED. `use_long_range` is switched on for the forward and `lr_start_epoch`
lowered so the branch actually evaluates -- the models were TRAINED with it off, so this is
"what would E_LR contribute if enabled", not "what these models predict". That is exactly the
question: whether re-enabling it in the joint run could close F4.

The branch is restored afterwards, so a model object is not left mutated for whatever runs
next in the same process.
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
from mace.tools.scripts_utils import extract_config_mace_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from s3_dehead_trend import CLEAN_NATOMS, fit_with_ci, hub_separation  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

F3_THRESHOLD = 0.01          # eV/A


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--eps-inf", type=float, default=None,
                    help="per-host input; read off the model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == CLEAN_NATOMS]
    d = np.array([hub_separation(a) for a in frames])
    ok = np.isfinite(d)
    z = tools.AtomicNumberTable(sorted({17, 55, 82}))
    print(f"{len(frames)} charged {CLEAN_NATOMS}-atom frames, "
          f"d {d[ok].min():.2f}-{d[ok].max():.2f} A", flush=True)

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        trained = torch.load(mp, map_location=args.device, weights_only=False)
        # The branch cannot be switched on by a flag: these models were BUILT with
        # use_long_range=False, so `latent_charges` does not exist on them and setting the
        # attribute raises. The model is rebuilt with the branch present and the trained
        # weights copied in.
        #
        # WHAT THAT MEANS FOR THE ANSWER, stated because it bounds the claim: the long-range
        # branch's own parameters arrive at INITIALISATION, not trained. So this measures the
        # d-slope E_LR would contribute on day one of a staged re-enable, not what it would
        # contribute after the joint run had fitted it. A slope that is already large would
        # settle F3; a slope near zero is suggestive and not conclusive.
        cfg = extract_config_mace_model(trained)
        cfg["use_long_range"] = True
        cfg["lr_start_epoch"] = 0
        model = trained.__class__(**cfg)
        missing, unexpected = model.load_state_dict(trained.state_dict(), strict=False)
        model = model.to(args.device).eval()
        print(f"  {Path(mp).name:20s} rebuilt with E_LR; {len(missing)} parameters at "
              f"initialisation, {len(unexpected)} unused", flush=True)
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        batches = make_batches(frames, z, cutoff, 1, args.device)

        prev_lr = bool(getattr(model, "use_long_range", False))
        prev_start = int(getattr(model, "lr_start_epoch", 0))
        # `current_epoch` is a registered BUFFER, so it takes a tensor and not an int --
        # assigning an int raises rather than silently coercing, which is the good outcome.
        prev_epoch = model.current_epoch.detach().clone()
        model.use_long_range = True
        model.lr_start_epoch = 0
        with torch.no_grad():
            model.current_epoch.fill_(10 ** 6)
        try:
            lr_energy, head = [], []
            for b, fr in batches:
                with torch.no_grad():
                    out = model(ctx.forward_dict(b, fr, requires_grad=False),
                                training=False, compute_force=False)
                # E_LR is the correction minus the head: both are in the output, and taking
                # the difference avoids assuming which key holds which branch.
                corr = float(out["correction_energy"].reshape(-1)[0])
                sr = float(out["delta_sr_energy"].reshape(-1)[0])
                lr_energy.append(corr - sr)
                head.append(sr)
        finally:
            model.use_long_range = prev_lr
            model.lr_start_epoch = prev_start
            with torch.no_grad():
                model.current_epoch.copy_(prev_epoch)

        lr_energy = np.array(lr_energy)
        good = ok & np.isfinite(lr_energy)
        if good.sum() < 5 or float(np.abs(lr_energy[good]).max()) == 0.0:
            print(f"  {Path(mp).name:20s} E_LR is identically zero on this model -- the "
                  "branch did not evaluate; F3 is UNSCORED here, not passed", flush=True)
            rows.append(dict(model=Path(mp).name, evaluated=False))
            continue
        slope, lo, hi, corr = fit_with_ci(d[good], lr_energy[good])
        passed = bool(abs(slope) < F3_THRESHOLD)
        rows.append(dict(model=Path(mp).name, evaluated=True, slope=slope, ci=[lo, hi],
                         corr=corr, passed=passed,
                         e_lr_range=[float(lr_energy[good].min()),
                                     float(lr_energy[good].max())]))
        print(f"  {Path(mp).name:20s} dE_LR/dd {slope:+.5f} [{lo:+.5f}, {hi:+.5f}] eV/A  "
              f"corr {corr:+.3f}  E_LR spans {lr_energy[good].min():+.4f} to "
              f"{lr_energy[good].max():+.4f} eV  "
              f"{'F3 PASS (off the list)' if passed else 'F3 FAIL (stays a candidate)'}",
              flush=True)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    scored = [r for r in rows if r.get("evaluated")]
    if scored:
        sl = np.array([r["slope"] for r in scored])
        print(f"\n  dE_LR/dd {sl.mean():+.5f} +- {sl.std():.5f} eV/A over {len(scored)} "
              f"models; F3 (|slope| < {F3_THRESHOLD}) met by "
              f"{sum(r['passed'] for r in scored)}/{len(scored)}")
        print("  Against F4's shortfall of ~0.073 eV/A. E_LR is a candidate for the missing "
              "amplitude only if its own d-slope is of that order.")
    else:
        print("\n  NOTHING SCORED: the long-range branch did not evaluate on any model, so "
              "F3 is unscored. Do not read that as a pass.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
