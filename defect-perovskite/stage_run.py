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
    # Truncate, do not append. An appended log mixes a discarded run's lines with a live
    # one's, and reading a void run's numbers as current has already happened twice today.
    # The JSON is authoritative either way; this stops the log lying.
    open(path, "w").close()

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
                           z_init=Z_INIT if madelung else None,
                           counting=(stage >= 3))
    if stage < 3:
        install_local_head(model, t_ref_r=t_ref)
        if stage >= 2:
            from mace.modules.defect_bounded import install_bounded_elements

            install_bounded_elements(model)
    if stage >= 3:
        # Section 3: Harrison term values, not a zero initialisation. eps0 = 0 makes every
        # site degenerate -- the atomic limit, where the bond order vanishes and (on the
        # frozen-P gradient) nothing could move the hoppings at all. The measured bond length
        # is passed in so the head stays host-agnostic.
        from mace.modules.defect_counting import harrison_initialise

        harrison_initialise(model.spectral, [int(z) for z in model.atomic_numbers],
                            bond_length=t_ref)
    model = model.to(device)
    log(f"      stage {stage}, madelung {'ON' if madelung else 'OFF'}, "
        f"eps_inf {eps_inf}, seed {seed}")
    return model


def pristine_spectrum_check(model, pristine_batches, log):
    """Stage 2's gate: on a DEFECT-FREE cell the spectrum must be bands, not a superatom.

    A perfect crystal has no bound state, so its lowest level must sit among the others --
    `(lam_2 - lam_1)` of order the level spacing, small against the bandwidth. The superatom
    failure is the opposite: one in-phase mode over the whole cell, split far below
    everything else. E2 measured exactly that at a 10 A reach, so it is a real available
    minimum and not a hypothetical.

    Reported as `split_fraction = (lam_2 - lam_1) / (lam_last - lam_1)`.

    `lam_last` is the head's LAST COMPUTED state, not the band top: the solve returns
    `num_states` levels (6 here), so this is a local measure at the bottom of the spectrum and
    not a true bandwidth. That is what makes the reference value concrete rather than
    arbitrary -- evenly spaced levels give exactly `1 / (num_states - 1) = 0.2`, while a
    single mode split off below the rest drives it towards 1. The threshold is the
    even-spacing value, so "BANDS" means "no more split off than uniform spacing would give".
    """
    from ta_band_edge import capture as _capture

    fracs, gaps, _spectra = [], [], []
    for batch, frames in pristine_batches:
        internals, _ = _capture(model, batch)
        lam = internals["lam"]
        for g in range(lam.shape[0]):
            for c in range(lam.shape[1]):
                v = lam[g, c]
                v = torch.sort(v[v < 500.0]).values
                if v.numel() < 3:
                    continue
                span = float(v[-1] - v[0])
                if span <= 1e-9:
                    continue
                fracs.append(float(v[1] - v[0]) / span)
                gaps.append(float(v[1] - v[0]))
                _spectra.append(v)
    if not fracs:
        return dict(split_fraction=float("nan"), pristine_gap=float("nan"))
    # From the REALISED spectrum length, not a configured num_states: the counting
    # head returns the whole 4N spectrum and advertises num_states = -1, which
    # would make the reference 1.0 and the gate vacuous.
    n_states = max(int(np.median([int(s.numel()) for s in _spectra])), 2)
    even = 1.0 / (n_states - 1)
    out = dict(split_fraction=float(np.mean(fracs)),
               # NOT the band gap. This is lam_2 - lam_1 at the BOTTOM of the spectrum, which
               # is what the superatom check needs. A reader who sees "pristine_gap" and
               # thinks "band gap" would compare it to E_gap and draw nonsense, so the name
               # says which one it is.
               pristine_split_lam2_lam1=float(np.mean(gaps)),
               split_fraction_even=float(even))
    log(f"      pristine spectrum: split fraction {out['split_fraction']:.4f}, "
        f"lam2-lam1 {out['pristine_split_lam2_lam1']:.4f} eV "
        f"({'BANDS' if out['split_fraction'] < even else 'SUPERATOM RISK'}, "
        f"even spacing would give {even:.4f})")
    return out


def run_cell(arch_path, base_path, seed, batches, frame_masks, device, epochs, lr,
             stage, madelung, eps_inf, t_ref, log, pristine_batches=None,
             freeze_z=False, e_gap=2.4, w_gap=1.0):
    model = build(arch_path, base_path, seed, device, stage, madelung, eps_inf, t_ref, log)
    ctx = ForwardContext.production(model, device=device, eps_inf=eps_inf,
                                    stage=stage, madelung=bool(madelung), seed=seed)
    # The in-loop reach assertion, on a batch this run actually consumes. Skipping it is how
    # a 20-seed screen was voided.
    rep = ctx.assert_reach(batches[0][0])
    ctx.assert_coupling(batches[0][0])
    log(f"      reach: max edge {rep.get('max_edge', float('nan')):.2f} A "
        f"against cutoff {ctx.cutoff:.1f} A")

    metrics_init = {}
    if stage >= 3:
        from mace.modules.defect_counting import VALENCE, initialisation_gate

        b0, f0 = batches[0]
        with torch.no_grad():
            out0 = model(ctx.forward_dict(b0, f0), training=False, compute_force=False)
            target = getattr(b0, "energy", None)
            dn = b0.carrier_counts.reshape(int(b0.num_graphs), -1)
            dn = (dn[:, 0] + dn[:, 1] - dn[:, 2] - dn[:, 3]).to(
                out0["delta_sr_energy"].dtype)
            sel = dn.abs() > 0
            if target is not None and bool(sel.any()):
                # E_head moves by c * Delta_n and by nothing else, so one scalar solves the
                # median mismatch exactly. This is mu's old initialisation role, without
                # mu's per-channel bookkeeping.
                resid = target - out0["base_energy"] - out0["delta_sr_energy"]
                c = float(torch.median(resid[sel] / dn[sel]))
                model.spectral.c_shift.fill_(c)
                log(f"      c-shift calibrated to {c:+.4f} eV on the init batch")
        pb0, pf0 = (pristine_batches or [(b0, f0)])[0]
        internals, _ = capture(model, pb0, ctx=ctx, frames=pf0)
        lam0 = internals["lam"][0, 0]
        lam0 = lam0[lam0 < 500.0]
        n_el = sum(VALENCE[int(z)] for z in pf0[0].get_atomic_numbers()) / 2.0
        gate = initialisation_gate(lam0, n_el, e_gap)
        log(f"      init gate: edges {gate['edge_spacing_below']:.3f}/"
            f"{gate['edge_spacing_above']:.3f} eV (need <= {0.5 * e_gap:.2f}), bandwidth "
            f"{gate['bandwidth']:.2f} eV (need >= {2 * e_gap:.2f}) -> "
            f"{'PASS' if gate['passed'] else 'TRIP'}")
        metrics_init = {f"init_{k}": v for k, v in gate.items()}

    model.train()
    for n, p in model.named_parameters():
        train_it = (is_correction_param(n) or ".spectral." in n
                    or n.startswith("madelung."))
        if freeze_z and n.startswith("madelung."):
            # Diagnostic arm: Z pinned at the nominal charges. Answers whether the LEARNED
            # scale of Z buys anything, or whether the two seeds that inflated it to
            # (-2.6, +2.3, +5.4) simply went down a bad path and dragged the fit with them.
            train_it = False
        p.requires_grad_(train_it)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    # 5-epoch linear warmup. The counting head's correction is eV-scale at step 0, so the
    # first few steps see gradients three orders larger than the converged ones; going in at
    # full rate is how a seed gets thrown somewhere it cannot return from.
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1.0, (e + 1) / warmup) if stage >= 3 else 1.0)

    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        f_sum, n_step = 0.0, 0
        for batch, frames in batches:
            d = ctx.forward_dict(batch, frames, requires_grad=True)
            out = model(d, training=True, compute_force=True)
            loss = ((out["forces"] - batch.forces) ** 2).mean()
            if stage >= 3 and pristine_batches:
                # loss_gap, Stage 3 only. A constraint on the PRISTINE spectrum -- the
                # frontier gap of a defect-free cell must be the host band gap -- so it is
                # not an energy label and does not carry M1b's base-extrapolation slope.
                # One pristine draw per step, ensemble mean over its graphs.
                pb, pfr = pristine_batches[int(rng.integers(len(pristine_batches)))]
                pout = model(ctx.forward_dict(pb, pfr, requires_grad=False),
                             training=True, compute_force=False)
                gap_res = pout["logit_gap"][:, 0].mean() - e_gap
                loss = loss + w_gap * gap_res ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0 if stage >= 3 else 10.0)
            opt.step()
            # Z lives on the pristine composition hyperplane. Without the projection the
            # species charges drift as a group, which is a gauge on phi and a slow
            # divergence -- a post-step hook, so it cannot be skipped by a branch.
            if getattr(model, "madelung", None) is not None:
                model.madelung.project_()
            f_sum += float(loss)
            n_step += 1
        sched.step()
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
    if pristine_batches:
        metrics.update(pristine_spectrum_check(model, pristine_batches, log))
        # Stage 3's gate of record: the pristine FRONTIER gap, eps_{N+1} - eps_N, which is
        # what loss_gap drives toward E_gap. A different quantity from the split check above,
        # persisted separately so the two can never be read as one another.
        with torch.no_grad():
            fg = [float(model(ctx.forward_dict(pb, pf), training=False,
                              compute_force=False)["logit_gap"][:, 0].mean())
                  for pb, pf in pristine_batches]
        metrics["pristine_frontier_gap"] = float(np.mean(fg))
        log(f"      pristine frontier gap {metrics['pristine_frontier_gap']:.3f} eV "
            f"(target {e_gap:.2f}, gate |delta| <= 0.10)")
        # The coadvisor's Stage-3 watch item. Edit 3 narrows the superatom route but does not
        # close it -- four of twelve Stage-2 seeds found it anyway, and those were exactly
        # the seeds that "fitted". A Stage-3 seed above this line is to be reseeded, not
        # interpreted, so it is marked here rather than left to a reader's judgement.
        sf = metrics.get("split_fraction", float("nan"))
        if sf == sf and sf > 0.3:
            metrics["superatom_reseed"] = True
            log(f"      *** SUPERATOM WATCH: split fraction {sf:.3f} > 0.30 -- this seed is "
                "to be reseeded, not interpreted ***")
    metrics.update(**metrics_init)
    metrics.update(seed=seed, stage=stage, madelung=bool(madelung), freeze_z=bool(freeze_z),
                   neff=neff,
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
    ap.add_argument("--train", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--n-pristine", type=int, default=8)
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--madelung", choices=("on", "off"), default="on")
    ap.add_argument("--freeze-z", action="store_true",
                    help="pin Z at the nominal charges; diagnostic arm for the Z runaway")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--e-gap", type=float, default=2.4,
                    help="host band gap for loss_gap (stage 3); sensitivity 2.2")
    ap.add_argument("--w-gap", type=float, default=1.0)
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

    # Stage 2's gate needs a DEFECT-FREE spectrum: a perfect crystal must show bands, not
    # one in-phase mode split off below everything else. Selected by composition, which is a
    # property of the formula unit and not of the defect.
    from d1_sensitivity import select_pristine
    from ase.io import read as _read
    pristine = select_pristine(_read(str(args.train), ":"), args.n_pristine)
    pristine_batches = make_batches(pristine, z_table, cutoff, args.batch_size, args.device)
    log(f"  {len(pristine)} pristine frames of {len(pristine[0])} atoms for the "
        f"band-versus-superatom check")

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
                                  args.madelung == "on", args.eps_inf, t_ref, log,
                                  pristine_batches=pristine_batches,
                                  freeze_z=args.freeze_z, e_gap=args.e_gap,
                                  w_gap=args.w_gap)
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
            # The tag carries EVERY arm-distinguishing flag, not just stage and madelung.
            # It did not, and the Z-frozen diagnostic arm silently overwrote the Stage-1 ON
            # checkpoints -- same stage, same madelung setting, different model. Anything
            # that changes what is trained belongs in the filename.
            tag = (f"s{args.stage}_{args.madelung}"
                   + ("_frz" if args.freeze_z else "") + f"_s{seed}")
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
