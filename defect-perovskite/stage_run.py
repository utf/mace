#!/usr/bin/env python3
"""Head-only retrains for Stages 1-3, behind one harness.

ONE harness for all three stages, deliberately. Stage 2's gate is "force parity within the
seed spread of Stage 1" and Stage 3's readouts are compared to Stage 2's; those comparisons
only mean something if the data, the seeds, the batching, the optimiser and the evaluation
are held fixed and the ONLY thing that changes is the edit under test. A second script would
make that a claim rather than a property.

    --stage 1   Edit 1: Madelung on-site, response channel gone
    --stage 2   + Edit 3: bounded elements, floors and t_min out
    --stage 3   + Edit 4: electron-counting head, uniform s+p

`--madelung off` is Stage 1's control arm: the same code, the same seeds, the term absent.
Force parity is read against it, not against a remembered number from a previous programme.

WHAT IS NOT HERE, and why. No `channel_sign`, no `mu_c`, no sign-aware initialiser, no
band-edge `L_edge` and no soft gauge anchor -- the one-manifold programme is archived, and
carrying its machinery forward "in case" is how a retired mechanism keeps influencing runs.
The head is the V3 local head with default conventions.

The loss is FORCES ONLY for stages 1 and 2. The 79-atom energy targets carry the
base-extrapolation slope M1b measured (+0.37 eV/A on the frozen base), which the plan removes
at source by training the base jointly in the final run -- not here. Stage 3 adds `loss_gap`,
which is a pristine-spectrum constraint and not an energy label.

Everything forward-facing goes through ForwardContext: production refuses the defect masks,
and train/evaluate/capture cannot disagree about the graph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext
from mace.modules.defect_stage import is_correction_param

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import evaluate, fresh_model, make_batches, masks_for  # noqa: E402
from ta_band_edge import capture, channel_of, load_frames, select  # noqa: E402
from tb_edge import neff_and_nulls  # noqa: E402

# Pristine stoichiometry of CsPbCl3 in AtomicNumberTable order (Cl 17, Cs 55, Pb 82).
# A property of the training set's formula unit, not of the defect.
COMPOSITION = [3.0, 1.0, 1.0]
Z_INIT = [-1.0, 1.0, 2.0]          # nominal, projected to neutrality at construction


def log_to(path):
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log


def measure_t_ref(batch) -> float:
    """Median nearest-neighbour distance, measured. Calibrates the initial hopping scale
    once at construction; never enters the functional form."""
    ei = batch.edge_index
    vec = batch.positions[ei[1]] - batch.positions[ei[0]]
    if getattr(batch, "shifts", None) is not None and batch.shifts.numel() == vec.numel():
        vec = vec + batch.shifts
    lengths = torch.linalg.norm(vec, dim=-1)
    nn = torch.full((int(batch.num_nodes),), float("inf"), device=lengths.device)
    nn = nn.index_reduce(0, ei[0], lengths, "amin", include_self=True)
    return float(np.median(nn[torch.isfinite(nn)].detach().cpu().numpy()))


def build(arch_path, base_path, seed, device, stage, madelung, eps_inf, t_ref, log):
    from mace.modules.defect_spectral_v3 import install_local_head

    # Nominal charges as the initialisation, not zeros. At Z = 0 the Madelung term is
    # identically absent at epoch 0, so the run starts from no donor well at all and has to
    # discover the ionicity of a rock-salt-like crystal from force residuals. (-1, 1, 2) is
    # already exactly neutral against the 3:1:1 composition, so the projection is a no-op
    # here and the A1 table describes the state the run actually starts in.
    model, _ = fresh_model(arch_path, base_path, seed, device,
                           madelung=COMPOSITION if madelung else None, eps_inf=eps_inf,
                           z_init=Z_INIT if madelung else None)
    install_local_head(model, t_ref_r=t_ref)
    if stage >= 2:
        from mace.modules.defect_bounded import install_bounded_elements

        install_bounded_elements(model)
    if stage >= 3:
        raise NotImplementedError(
            "stage 3 needs the counting head; build it before asking for it rather than "
            "letting this harness silently run stage 2 under a stage-3 label")
    model = model.to(device)
    log(f"      stage {stage}, madelung {'ON' if madelung else 'OFF'}, "
        f"eps_inf {eps_inf}, seed {seed}")
    return model


def run_cell(arch_path, base_path, seed, batches, frame_masks, device, epochs, lr,
             stage, madelung, eps_inf, t_ref, log):
    model = build(arch_path, base_path, seed, device, stage, madelung, eps_inf, t_ref, log)
    ctx = ForwardContext.production(model, device=device, eps_inf=eps_inf,
                                    stage=stage, madelung=bool(madelung), seed=seed)
    # The in-loop reach assertion, on a batch this run actually consumes. Skipping it is how
    # a 20-seed screen was voided.
    rep = ctx.assert_reach(batches[0][0])
    ctx.assert_coupling(batches[0][0])
    log(f"      reach: max edge {rep.get('max_edge', float('nan')):.2f} A "
        f"against cutoff {ctx.cutoff:.1f} A")

    model.train()
    for n, p in model.named_parameters():
        p.requires_grad_(is_correction_param(n) or ".spectral." in n
                         or n.startswith("madelung."))
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    hist = []
    for ep in range(epochs):
        f_sum, n_step = 0.0, 0
        for batch, frames in batches:
            d = ctx.forward_dict(batch, frames, requires_grad=True)
            out = model(d, training=True, compute_force=True)
            loss = ((out["forces"] - batch.forces) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            # Z lives on the pristine composition hyperplane. Without the projection the
            # species charges drift as a group, which is a gauge on phi and a slow
            # divergence -- a post-step hook, so it cannot be skipped by a branch.
            if getattr(model, "madelung", None) is not None:
                model.madelung.project_()
            f_sum += float(loss)
            n_step += 1
        hist.append(f_sum / max(n_step, 1))
        if ep % 5 == 0 or ep == epochs - 1:
            z = (model.madelung.z.tolist() if getattr(model, "madelung", None) is not None
                 else None)
            log(f"      epoch {ep:3d}  force {hist[-1]:.5f}"
                + (f"  Z {['%.3f' % v for v in z]}" if z else ""))

    model.eval()
    metrics = evaluate(model, batches, frame_masks, device, ctx=ctx)
    channel = channel_of(batches[0][0])
    neff, ratio = float("nan"), float("nan")
    try:
        internals, out = capture(model, batches[0][0], ctx=ctx, frames=batches[0][1])
        act, nul = neff_and_nulls(out, batches[0][0], channel)
        neff = float(np.nanmean(act))
        # The nulls take no gradient, so they are this run's own baseline for what
        # "delocalised" looks like. The ratio, not N_eff, is the localisation criterion.
        ratio = float(np.nanmean(act) / max(np.nanmean(nul), 1e-30))
    except Exception as exc:                      # diagnostics must not kill a run
        log(f"      capture failed: {exc}")
    metrics.update(seed=seed, stage=stage, madelung=bool(madelung), neff=neff,
                   null_ratio=ratio, force_final=hist[-1], force_first=hist[0],
                   context=ctx.as_dict())
    if getattr(model, "madelung", None) is not None:
        metrics["z"] = [float(v) for v in model.madelung.z]
    return model, metrics


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--madelung", choices=("on", "off"), default="on")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    log = log_to(str(args.out) + ".log")
    log(f"stage {args.stage}, madelung {args.madelung}, eps_inf {args.eps_inf}")

    frames = select(load_frames(args.data), charged=True)[: args.frames]
    log(f"  {len(frames)} charged frames, mean {np.mean([len(a) for a in frames]):.1f} atoms")
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    probe = torch.load(args.arch, map_location="cpu", weights_only=False)
    cutoff = max(float(probe.r_max), float(getattr(probe, "spectral_r_cut", 0.0) or 0.0))
    del probe
    batches = make_batches(frames, z_table, cutoff, args.batch_size, args.device)
    t_ref = measure_t_ref(batches[0][0])
    log(f"  cutoff {cutoff:.1f} A, measured nearest-neighbour {t_ref:.3f} A")

    rng = np.random.default_rng(0)
    frame_masks = {}
    for a in frames:
        m = masks_for(a, rng)
        if m is not None:
            frame_masks[id(a)] = m
    log(f"  masks for {len(frame_masks)}/{len(frames)} frames (evaluation only)")

    rows = []
    for k in range(args.seeds):
        seed = args.seed_start + k
        log(f"  --- seed {seed} ---")
        try:
            model, met = run_cell(args.arch, args.base, seed, batches, frame_masks,
                                  args.device, args.epochs, args.lr, args.stage,
                                  args.madelung == "on", args.eps_inf, t_ref, log)
        except Exception as exc:
            log(f"      FAILED: {exc}")
            rows.append(dict(seed=seed, error=str(exc)))
            args.out.write_text(json.dumps(rows, indent=2, default=float))
            continue
        rows.append(met)
        log(f"      axial_red {met['axial_red']:+.3f}  rmse_all {met['rmse_all']:.1f}  "
            f"rmse_nbhd {met['rmse_nbhd']:.1f}  N_eff {met['neff']:.2f}")
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            tag = f"s{args.stage}_{args.madelung}_s{seed}"
            torch.save(model, args.save_dir / f"{tag}.model")
        args.out.write_text(json.dumps(rows, indent=2, default=float))

    good = [r for r in rows if "error" not in r]
    if good:
        log(f"  MEAN over {len(good)}: axial_red "
            f"{np.mean([r['axial_red'] for r in good]):+.3f}  rmse_all "
            f"{np.mean([r['rmse_all'] for r in good]):.1f}  rmse_nbhd "
            f"{np.mean([r['rmse_nbhd'] for r in good]):.1f}  N_eff "
            f"{np.mean([r['neff'] for r in good]):.2f}")
    args.out.write_text(json.dumps(rows, indent=2, default=float))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
