#!/usr/bin/env python3
"""Plan v8 section 7.1, continuity, on the trained model (Stage 1.2).

Two paths on arma_s1, uniform float64, CPU:

  * an ON-SITE SWEEP on the charged 79-atom vacancy frame: the on-site levels of the Pb site
    carrying the most carrier density (the model's own `carrier_alpha`) are shifted by
    `lambda` from -3 to +3 eV, which drives the frontier level from deep in the gap through
    the projector windows and into a band;
  * a GEOMETRIC PATH between the two closest-by-displacement thermal charged 79-atom frames
    of the pool (linear interpolation of the minimum-image displacement in the first frame's
    cell).

Along each: `E`, `F`, `Phi_FF`, `w` (at the state and at the reference fill), `q_F` and the
per-frame counts. The continuity
statement is section 7.1's: no jump above the FD floor. Measured as in the toy test -- the
largest consecutive change at step `h` against the one at `h / 4`; a discontinuity survives
the refinement, a continuous curve's change falls with it. The integers must not move.

Writes `golden/stage12_continuity.json`.

    export PYTHONPATH=<repo>; python defect-perovskite/stage12_continuity.py \\
        --model ~/runs/arma_models/arma_s1.model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401
from mace.modules import defect_composition as dc
from mace.modules.defect_context import ForwardContext
from mace.modules.defect_state import StateBatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from stage0_fd import charged_pool, load_uniform  # noqa: E402
from stage0_golden import ensure_table, frame_selection, git_sha, make_batch  # noqa: E402

HERE = Path(__file__).resolve().parent


def forward(model, ctx, atoms, forces=True):
    d = ctx.forward_dict(make_batch([atoms], ctx.cutoff), requires_grad=forces)
    with torch.enable_grad() if forces else torch.no_grad():
        out = model(d, training=False, compute_force=forces)
    out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    rec = dict(E=float(out["energy"][0]), phi=float(out["frontier_energy"][0]),
               w=float(out["frontier_w"][0]), p=float(out["frontier_p"][0]),
               q_F=float(out["frontier_q_F"][0]), band=float(out["delta_sr_energy"][0]),
               w_ref=float(out["frontier_w_ref"][0]),
               min_weight=float(out["frontier_min_weight"][0]))
    if forces:
        rec["F"] = out["forces"].detach().cpu().numpy()
    return rec


def sweep(model, ctx, atoms, site, lambdas):
    head = model.spectral
    original = head.h.on_site
    out = []
    try:
        for lam in lambdas:
            def shifted(feats, species, madelung, centre=None, _s=float(lam)):
                levels = original(feats, species, madelung, centre=centre)
                bump = torch.zeros_like(levels)
                bump[site] = _s
                return levels + bump
            head.h.on_site = shifted
            out.append(forward(model, ctx, atoms))
    finally:
        head.h.on_site = original
    return out


def _min_image(delta, cell):
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def closest_pair(frames):
    """The two frames of the pool with the smallest largest per-atom displacement under
    the minimum image: two snapshots of the SAME vacancy, so a linear path between them
    moves every atom by less than a bond length. Two frames of different vacancies (a
    different Cl removed) are not index-aligned, and a path between them collides atoms."""
    best = None
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a, b = frames[i], frames[j]
            if (a.get_atomic_numbers() != b.get_atomic_numbers()).any():
                continue
            d = _min_image(b.positions - a.positions, np.array(a.get_cell()))
            worst = float(np.linalg.norm(d, axis=1).max())
            if best is None or worst < best[0]:
                best = (worst, i, j)
    return best


def path(model, ctx, a, b, ts):
    delta = _min_image(b.positions - a.positions, np.array(a.get_cell()))
    out = []
    for t in ts:
        x = a.copy()
        x.positions = a.positions + t * delta
        out.append(forward(model, ctx, x))
    return out


def jumps(records):
    keys = ("E", "phi", "w", "w_ref", "band")
    res = {k: max(abs(y[k] - x[k]) for x, y in zip(records[:-1], records[1:])) for k in keys}
    res["F"] = max(float(np.abs(y["F"] - x["F"]).max())
                   for x, y in zip(records[:-1], records[1:]))
    return res


def ratio_table(coarse, fine):
    jc, jf = jumps(coarse), jumps(fine)
    return {k: dict(coarse=jc[k], fine=jf[k], ratio=(jf[k] / jc[k] if jc[k] > 0 else 0.0))
            for k in jc}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="~/runs/arma_models/arma_s1.model")
    p.add_argument("--out", default=str(HERE / "golden" / "stage12_continuity.json"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--n-coarse", type=int, default=25)
    args = p.parse_args(argv)
    _assert_repo()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    model = load_uniform(str(Path(args.model).expanduser()))
    ctx = ForwardContext.production(model, device="cpu")
    frames = frame_selection()
    ensure_table(model, ctx, frames, log=False)

    charged = frames["qp1_79"][0][2]
    rec = dc.lookup_class(model.composition_classes, charged.get_atomic_numbers())
    state = StateBatch.from_counts(torch.tensor([[float(x) for x in charged.info["carrier_counts"]]]))
    counts = dc.frame_counts(rec, state, 0)
    with torch.no_grad():
        out = model(ctx.forward_dict(make_batch([charged], ctx.cutoff)), training=False,
                    compute_force=False)
    alpha = out["carrier_alpha"][:, 0].cpu().numpy()
    pb = [i for i, z in enumerate(charged.get_atomic_numbers()) if z == 82]
    site = int(max(pb, key=lambda i: alpha[i]))
    print(f"on-site sweep: V_Cl+ 79-atom frame, class {rec.key}, counts (n_e, n_h, q_F) = "
          f"{counts}, site {site} (Pb, alpha {alpha[site]:.3f})")

    n_c = args.n_coarse
    coarse = np.linspace(-3.0, 3.0, n_c)
    fine = np.linspace(-3.0, 3.0, 4 * (n_c - 1) + 1)
    sc = sweep(model, ctx, charged, site, coarse)
    sf = sweep(model, ctx, charged, site, fine)
    on_site = dict(site=site, lambdas_coarse=coarse.tolist(), lambdas_fine=fine.tolist(),
                   ratios=ratio_table(sc, sf),
                   fine=[{k: v for k, v in r.items() if k != "F"} for r in sf],
                   q_F_values=sorted({r["q_F"] for r in sf}),
                   counts=list(counts), q_core=rec.q_core,
                   weight_range=[min(r["min_weight"] for r in sf),
                                 max(r["min_weight"] for r in sf)])
    for k, v in on_site["ratios"].items():
        print(f"  {k:5s} max consecutive change  h: {v['coarse']:.3e}  h/4: {v['fine']:.3e}  "
              f"ratio {v['ratio']:.2f}")
    print(f"  q_F along the sweep: {on_site['q_F_values']}; projector weight range "
          f"{on_site['weight_range'][0]:.3f} .. {on_site['weight_range'][1]:.3f}")

    pool = charged_pool(12)
    worst, i, j = closest_pair(pool)
    a, b = pool[i], pool[j]
    print(f"geometric path between pool frames {i} and {j} (largest atom displacement "
          f"{worst:.3f} A):")
    pc = path(model, ctx, a, b, np.linspace(0, 1, 9))
    pf = path(model, ctx, a, b, np.linspace(0, 1, 33))
    geometric = dict(pair=[i, j], largest_displacement_A=worst, ratios=ratio_table(pc, pf),
                     fine=[{k: v for k, v in r.items() if k != "F"} for r in pf],
                     q_F_values=sorted({r["q_F"] for r in pf}))
    for k, v in geometric["ratios"].items():
        print(f"  {k:5s} max consecutive change  h: {v['coarse']:.3e}  h/4: {v['fine']:.3e}  "
              f"ratio {v['ratio']:.2f}")

    verdict = all(v["ratio"] <= 0.5 for v in on_site["ratios"].values()) and \
        all(v["ratio"] <= 0.5 for v in geometric["ratios"].values()) and \
        len(on_site["q_F_values"]) == 1 and len(geometric["q_F_values"]) == 1
    print("continuity:", "pass" if verdict else "FAIL")
    Path(args.out).write_text(json.dumps(dict(
        git_sha=git_sha(), model=str(args.model),
        regime="arma_s1, uniform float64, CPU, 8 threads", on_site=on_site,
        geometric=geometric, status="pass" if verdict else "FAIL"), indent=1))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
