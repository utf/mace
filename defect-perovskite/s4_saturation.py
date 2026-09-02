#!/usr/bin/env python3
"""Section 4: the saturation audit of Edit 3's bounded elements. Forward-only.

Edit 3 replaced floors and a learned decay with bounded forms:

    t   = V0[s_i, s_j] * f(r) * (1 + 0.5 tanh g_theta)
    eps = eps0[s]      - phi/eps_inf + 1 eV * tanh h_theta

The bounds are what stopped the superatom collapse, and they cost nothing while the tanh
arguments stay in their linear region. A SATURATED tanh is a different matter: its gradient
is `sech^2`, which is ~0.07 at |x| = 2 and ~0.0002 at |x| = 5, so a saturated element is
effectively frozen -- the model looks like it is learning and that parameter is not. It also
means the bound is binding, i.e. the fit wants a value the bound forbids, which is a
statement about whether the bound is set right.

Reported per model:
  * the distribution of |g| and |h| over the real graph, and the fraction beyond |x| = 2;
  * the resulting gradient attenuation `sech^2`, which is the number that matters;
  * the realised hopping envelope: t at the nearest-neighbour distance and its decay, so
    the radial exponent family can be judged against what the model actually uses.

NO EDIT. The plan is explicit that the outcome is recorded as a config decision for the
rerun, not as a change to the head.
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
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

SATURATED = 2.0          # |x| beyond which sech^2 has lost >90% of its gradient


def collect(model, batch, frames, ctx):
    """The raw tanh arguments on a real graph, from the head's OWN audit buffer.

    Not a spy on guessed attribute names. The previous version looked for submodules called
    `g`, `h`, `g_mlp` and so on, matched none, and reported a silent null -- which is the
    worst possible failure for an audit, since "no saturation found" and "nothing was
    measured" print the same. `SlaterKosterH.audit()` now stores every pre-tanh tensor from
    the code that computes it.

    The audited class is `SlaterKosterH`, NOT `BoundedLocalHead`: Stage 3 replaced the
    spectral head wholesale, so these models contain no BoundedLocalHead at all and auditing
    it would have audited a module that is not there.
    """
    head = model.spectral
    head.h._audit_bin = {}
    bin_ = head.h.audit(True)
    try:
        with torch.no_grad():
            model(ctx.forward_dict(batch, frames, requires_grad=False),
                  training=False, compute_force=False)
        grabbed = {k: torch.cat(v) for k, v in bin_.items() if v}
    finally:
        head.h.audit(False)
    return grabbed


def envelope(model, r_grid):
    """`t(r)` on the model's own radial form, at the learned scales."""
    h = model.spectral.h
    out = {}
    for name in ("radial", "f", "envelope", "decay"):
        fn = getattr(h, name, None)
        if callable(fn):
            try:
                with torch.no_grad():
                    out[name] = [float(fn(torch.tensor([[r]],
                                                       dtype=torch.get_default_dtype(),
                                                       device=h.v0_raw.device)).reshape(-1)[0])
                                 for r in r_grid]
            except Exception:
                continue
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--atoms", type=int, default=159)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    frames = [a for a in select(load_frames(args.data), charged=True)
              if len(a) == args.atoms][:2]
    if not frames:
        raise SystemExit(f"no charged {args.atoms}-atom frames in {args.data}")
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    r_grid = [2.8, 3.5, 4.5, 5.6, 6.8, 8.0, 10.0]

    rows = []
    for mp in args.models:
        if not Path(mp).exists():
            continue
        model = torch.load(mp, map_location=args.device,
                           weights_only=False).to(args.device).eval()
        cutoff = max(float(model.r_max),
                     float(getattr(model, "spectral_r_cut", 0.0) or 0.0))
        ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf)
        batch, frs = make_batches(frames, z_table, cutoff, 1, args.device)[0]
        raw = collect(model, batch, frs, ctx)
        row = {"model": Path(mp).name, "channels": {}}
        if not raw:
            row["note"] = ("the audit buffer came back empty -- SlaterKosterH.audit() did "
                           "not fire, so nothing was measured. This is a FAILURE of the "
                           "audit, not a finding of no saturation.")
            print(f"  {row['model']:20s} AUDIT DID NOT FIRE", flush=True)
        for name, v in raw.items():
            v = v.float()
            sat = float((v.abs() > SATURATED).float().mean())
            att = float(torch.cosh(v.clamp(-20, 20)).pow(-2).mean())
            row["channels"][name] = dict(
                n=int(v.numel()), abs_mean=float(v.abs().mean()),
                abs_p95=float(v.abs().quantile(0.95)), abs_max=float(v.abs().max()),
                saturated_fraction=sat, mean_gradient_attenuation=att)
            print(f"  {row['model']:20s} {name:8s} |x| mean {v.abs().mean():.3f} "
                  f"p95 {float(v.abs().quantile(0.95)):.3f} max {float(v.abs().max()):.3f}  "
                  f"saturated {100 * sat:5.1f}%  mean sech^2 {att:.4f}", flush=True)
        # The learned hopping scales, which bound how large t can be at all.
        row["v0_abs_max"] = float(model.spectral.h.v0_raw.detach().abs().max())
        row["eps0"] = [[float(x) for x in r]
                       for r in model.spectral.h.eps0.detach().cpu()]
        row["envelope"] = {"r": r_grid, **envelope(model, r_grid)}
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    args.out.write_text(json.dumps(rows, indent=2, default=float))
    names = sorted({k for r in rows for k in r["channels"]})
    for name in names:
        sat = np.array([r["channels"][name]["saturated_fraction"] for r in rows
                        if name in r["channels"]])
        att = np.array([r["channels"][name]["mean_gradient_attenuation"] for r in rows
                        if name in r["channels"]])
        print(f"\n  {name}: saturated fraction {100 * sat.mean():.1f}% +- "
              f"{100 * sat.std():.1f}%, mean gradient attenuation {att.mean():.4f}")
    if names:
        print("  A large saturated fraction means the bound is BINDING -- the fit wants a "
              "value it forbids -- and those parameters are frozen at sech^2 ~ 0.07 or "
              "below. A small one means the bounds cost nothing and should stay as they are.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
