#!/usr/bin/env python3
"""T-B: does a band-edge inequality select the compact solution?

T-A asks whether the models fail Delta_bind > 0. T-B asks the causal question: if the
inequality is imposed, does the carrier localise, and does it do so honestly?

    L_edge = w_e * max(0, m - Delta_bind)^2
    Delta_bind = lambda_1(8 pristine frames, hole counter) - lambda_1(each charged frame)

Frozen Stage-A base, head + response channel, no clamp (`free`), 40 epochs, 3 seeds. No DCL
and no tiling: this is the whole point of the plan -- if a spectral inequality the head
already computes does the job, the dilution machinery is retired rather than debugged.

TWO HEAD VARIANTS.

V1 is the current feature-based on-site term. It tests whether the inequality alone
localises -- and, just as importantly, whether it does so by THE ESCAPE. The head can satisfy
Delta_bind without binding anything, by lowering eps on defect-cell atoms wholesale through
the trunk's receptive field: the defect cell's whole spectrum drops, lambda_1 with it, and the
inequality is met by a state no more localised than before. That is not a bound level, it is a
relabelled origin. The escape check (eps far from the vacancy, against pristine) is what
distinguishes them, and it is reported every cell.

V2 is the airtight version: eps_i = e(z_i) + sum_j phi(z_i, z_j, r_ij), a counted on-site term
built from element embeddings and pair distances with no trunk features in eps at all.
The escape needs eps to know it is in a defect cell; a term that can only count neighbours
knows only the local coordination, which is the same in bulk whether or not a vacancy sits
20 A away. Hopping and the response channel are untouched.

WHAT `m` CAN AND CANNOT DO. m sets the minimum depth and hence the maximum localisation
length. The level will sit at ~ m, because nothing here pins the absolute pristine edge:
without charged-pristine labels the model is free to place the continuum wherever it likes and
satisfy the inequality by construction. That is the known limit of the label-free version, not
a failure of it -- and it is why the decision rule treats "V2 reaches m without the escape" as
grounds to go and measure the depth, not as a measurement of it.

LABEL-FREE. The pristine frames entering L_edge are the existing 80-atom perfect cells; they
carry no vacancy information. The vacancy position appears only in the evaluation-time escape
check and the hub metric, never in the loss.
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
from mace.modules.defect_stage import is_correction_param

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import (evaluate, fresh_model, graph_cutoff_for, make_batches,  # noqa: E402
                       masks_for)
from ta_band_edge import (FAR_CUTOFF, HOLE_COUNTER, PRISTINE_NATOMS,  # noqa: E402
                          capture, channel_of, eps_by_species, far_from_vacancy,
                          load_frames, select, with_hole_counter)

from mace import tools  # noqa: E402  (after mace)


# --------------------------------------------------------------- differentiable lambda_1


def lambda1_grad(model, batch_dict, channel):
    """lambda_1 per graph, WITH the graph attached.

    `lambda1_of` in defect_bind runs under no_grad because it is a diagnostic; L_edge needs
    the gradient, so the internals hook is used directly. The head documents its internals as
    "STILL ATTACHED TO THE GRAPH", which is exactly what makes this possible without a second
    implementation of the Hamiltonian.

    Padded slots sit at ~1e3 by construction and are masked out rather than min'd over, or a
    cell with fewer atoms than num_states reports the padding energy as its ground state.
    """
    grabbed: dict = {}
    head = model.spectral
    original = head.forward

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    try:
        model(batch_dict, training=True, compute_force=False)
    finally:
        head.forward = original
    lam = grabbed["lam"][:, channel, :]                     # [G, m]
    masked = torch.where(lam < 500.0, lam, torch.full_like(lam, float("inf")))
    return masked.min(dim=-1).values                        # [G]


def localisation_length(psi, w, positions, batch, channel, ng, cell=None):
    """Amplitude-weighted RMS radius of the carrier density, under periodic boundaries.

    The first version of this took a plain Cartesian centroid and plain differences, with no
    minimum-image convention. Under PBC that is not a slightly worse estimate, it is a wrong
    one: it reported 1.10 A for a V1 state whose N_eff was 51, i.e. a state spread over most
    of the cell. Every number it produced has been discarded.

    A periodic centroid has no arithmetic mean, so each fractional axis is averaged on the
    circle (weighted mean of the unit vector at angle 2*pi*f, back through atan2) and
    displacements are wrapped into [-0.5, 0.5) before being taken back to Cartesian. A band
    state then returns ~ the cell radius instead of an artefact, which is the behaviour that
    makes the metric readable at all.
    """
    dens = (psi[:, channel].pow(2) * w[:, channel].unsqueeze(-2)).sum(-1)   # [G, N]
    out = []
    for g in range(ng):
        sel = (batch == g).nonzero(as_tuple=True)[0]
        p = positions[sel]
        d = dens[g][: sel.numel()]
        s = float(d.sum())
        if s <= 0:
            out.append(float("nan"))
            continue
        wgt = (d / s)
        if cell is None:
            centroid = (wgt.unsqueeze(-1) * p).sum(0)
            r2 = ((p - centroid) ** 2).sum(-1)
            out.append(float(torch.sqrt((wgt * r2).sum())))
            continue
        C = cell[3 * g:3 * g + 3] if cell.dim() == 2 and cell.shape[0] == 3 * ng else cell[g]
        inv = torch.linalg.inv(C.to(p.dtype))
        frac = p @ inv                                        # [n, 3]
        ang = 2.0 * torch.pi * frac
        cbar = (wgt.unsqueeze(-1) * torch.cos(ang)).sum(0)
        sbar = (wgt.unsqueeze(-1) * torch.sin(ang)).sum(0)
        f0 = torch.atan2(sbar, cbar) / (2.0 * torch.pi)        # circular mean, in [-0.5, 0.5]
        df = frac - f0
        df = df - torch.round(df)                              # minimum image
        disp = df @ C.to(p.dtype)
        out.append(float(torch.sqrt((wgt * (disp ** 2).sum(-1)).sum())))
    return out


def neff_and_nulls(out, batch, channel):
    """N_eff of the supervised channel and the mean N_eff of the three ungradiented ones.

    The nulls take no gradient and so form a per-run baseline for what "delocalised" looks
    like in this model at this epoch; the ratio is the localisation criterion the gates use.
    """
    alpha = out["carrier_alpha"]
    idx = batch.batch
    ng = int(batch.num_graphs)
    act, nul = [], []
    for g in range(ng):
        sel = idx == g
        per = []
        for c in range(alpha.shape[1]):
            v = alpha[sel, c]
            s2 = float((v * v).sum())
            per.append(1.0 / s2 if s2 > 0 else float("nan"))
        act.append(per[channel])
        nul.append(float(np.nanmean([per[c] for c in range(len(per)) if c != channel])))
    return act, nul


# --------------------------------------------------------------------------- one cell


def run_cell(arch_path, base_path, seed, batches, frame_masks, pristine_pool, device,
             epochs, lr, m, w_e, variant, log):
    model, _ = fresh_model(arch_path, base_path, seed, device, response=True)
    if variant == "v2":
        from mace.modules.defect_onsite import install_rigid_onsite
        install_rigid_onsite(model)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad_(is_correction_param(n))
    if variant == "v2":
        for n, p in model.named_parameters():
            if ".rigid_onsite." in n:
                p.requires_grad_(True)
    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=lr)

    rng = np.random.default_rng(seed)
    channel = channel_of(batches[0][0])
    hist = []

    for ep in range(epochs):
        f_sum = f_n = 0.0
        e_sum = d_sum = d_n = 0.0
        for batch, frames in batches:
            d = batch.to_dict()
            d["positions"] = batch.positions.detach().clone().requires_grad_(True)
            d["_clamp_mask"] = None
            out = model(d, training=True, compute_force=True)
            f_loss = ((out["forces"] - batch.forces) ** 2).mean()

            # Delta_bind against a pristine draw. Drawn per step, per the plan: the gradient
            # is then stochastic but unbiased in the reference, rather than tied to one
            # arbitrary set of 8 host frames.
            pb = pristine_pool[int(rng.integers(len(pristine_pool)))]
            lam_p = lambda1_grad(model, pb, channel).mean()
            lam_d = lambda1_grad(model, d, channel)
            delta = lam_p - lam_d                              # [G]
            edge = (torch.clamp(m - delta, min=0.0) ** 2).mean()
            loss = f_loss + w_e * edge

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            f_sum += float(f_loss); f_n += 1
            e_sum += float(edge)
            d_sum += float(delta.mean()); d_n += 1
        row = dict(epoch=ep, force=f_sum / max(f_n, 1), edge=e_sum / max(f_n, 1),
                   delta_bind=d_sum / max(d_n, 1))
        hist.append(row)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"      ep{ep:3d}  force {row['force']:.4f}  edge {row['edge']:.5f}  "
                f"Delta_bind {row['delta_bind']:+.4f}  (m={m}, w_e={w_e:.3g})")

    metrics = evaluate(model, batches, frame_masks, device)
    metrics.update(seed=seed, variant=variant, m=m, w_e=w_e,
                   arch=str(arch_path), history=hist)

    # --- localisation, against a FIXED reference so the reported number does not track the
    # per-step sampling the loss deliberately uses.
    model.eval()
    act, nul, lens, dvals = [], [], [], []
    ref = []
    with torch.no_grad():
        for pb in pristine_pool:
            ref.append(lambda1_grad(model, pb, channel).detach().cpu().numpy())
    ref_mean = float(np.mean(np.concatenate(ref)))
    for batch, frames in batches:
        internals, out = capture(model, batch)
        a, n = neff_and_nulls(out, batch, channel)
        act += a; nul += n
        lens += localisation_length(internals["psi"], internals["w"], batch.positions,
                                    batch.batch, channel, int(batch.num_graphs))
        lam_d = internals["lam"][:, channel, :]
        lam_d = torch.where(lam_d < 500.0, lam_d, torch.full_like(lam_d, float("inf")))
        dvals += (ref_mean - lam_d.min(dim=-1).values).detach().cpu().numpy().tolist()
    metrics["neff"] = float(np.nanmean(act))
    metrics["neff_nulls"] = float(np.nanmean(nul))
    metrics["null_ratio"] = float(np.nanmean(act) / max(np.nanmean(nul), 1e-12))
    metrics["loc_length"] = float(np.nanmean(lens))
    metrics["delta_bind_final"] = float(np.nanmean(dvals))
    metrics["lambda1_pristine_final"] = ref_mean
    return model, metrics


def escape_report(model, big_charged, big_pristine, z_table, cutoff, device, channel):
    """eps far from the vacancy against pristine, raw and gauge-matched. Evaluation only."""
    if not big_charged or not big_pristine:
        return None, None
    from collections import defaultdict
    bb = make_batches(big_charged, z_table, cutoff, 1, device)
    bp = make_batches(big_pristine, z_table, cutoff, 1, device)
    raw_d, mat_d, raw_p, mat_p = (defaultdict(list) for _ in range(4))
    for batch, frames in bb:
        fm = [far_from_vacancy(f) for f in frames]
        if any(x is None for x in fm):
            continue
        internals, _ = capture(model, batch)
        syms = [np.array(f.get_chemical_symbols()) for f in frames]
        r, mt, _ = eps_by_species(internals, batch, syms, channel, far_masks=fm)
        for k, v in r.items():
            raw_d[k].append(v)
        for k, v in mt.items():
            mat_d[k].append(v)
    for batch, frames in bp:
        internals, _ = capture(model, batch)
        syms = [np.array(f.get_chemical_symbols()) for f in frames]
        r, mt, _ = eps_by_species(internals, batch, syms, channel)
        for k, v in r.items():
            raw_p[k].append(v)
        for k, v in mt.items():
            mat_p[k].append(v)
    sp = sorted(set(raw_d) & set(raw_p))
    return ({k: float(np.mean(raw_d[k]) - np.mean(raw_p[k])) for k in sp},
            {k: float(np.mean(mat_d[k]) - np.mean(mat_p[k])) for k in sp})


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", type=Path, required=True)
    ap.add_argument("--base", type=Path,
                    default=Path.home() / "runs" / "e0_base_s1" / "e0_base_s1.model")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--train", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=1,
                    help="first seed index; lets a second machine add seeds 4-6 to another's "
                         "1-3 instead of recomputing them")
    ap.add_argument("--variants", nargs="+", default=["v1"])
    ap.add_argument("--margins", nargs="+", type=float, default=[0.2, 0.5])
    ap.add_argument("--w-edge", type=float, default=None,
                    help="default: calibrated so a violation of m/2 costs one frame's force loss")
    ap.add_argument("--n-pristine", type=int, default=64)
    ap.add_argument("--pristine-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()

    def log(msg):
        print(msg, flush=True)

    arch = torch.load(args.arch, map_location="cpu", weights_only=False)
    cutoff = graph_cutoff_for(arch)
    z_table = tools.AtomicNumberTable([int(z) for z in arch.atomic_numbers])
    log(f"  graph cutoff {cutoff:.1f} A (model r_max {float(arch.r_max):.1f})")

    # Charged frames, exactly as the clamp harness builds them.
    frames_all = load_frames(args.data)
    rng = np.random.default_rng(0)
    charged, frame_masks = [], {}
    for a in frames_all:
        c = a.info.get("carrier_counts")
        if c is None or int(np.asarray(c).sum()) == 0:
            continue
        fm = masks_for(a, rng)
        if fm is None:
            continue
        charged.append(a)
        frame_masks[id(a)] = fm
        if len(charged) >= args.frames:
            break
    log(f"  {len(charged)} charged frames usable")
    batches = make_batches(charged, z_table, cutoff, args.batch_size, args.device)

    # Pristine pool for L_edge: the existing 80-atom perfect cells under the hole counter.
    train = load_frames(args.train)
    pristine = with_hole_counter(
        select(train, natoms=PRISTINE_NATOMS, charged=False)[:args.n_pristine])
    pool_batches = make_batches(pristine, z_table, cutoff, args.pristine_batch, args.device)
    pristine_pool = [b.to_dict() for b, _ in pool_batches]
    log(f"  pristine pool: {len(pristine)} frames in {len(pristine_pool)} draws of "
        f"{args.pristine_batch}")

    big_charged = select(train, natoms=159, charged=True) + \
        select(frames_all, natoms=159, charged=True)
    big_pristine = with_hole_counter(
        select(train, natoms=159, charged=False) + select(frames_all, natoms=159, charged=False))

    # w_e calibration: a violation of m/2 should cost about one charged frame's force loss.
    # Measured from this harness's own epoch-0 forward, not assumed.
    if args.w_edge is None:
        probe, _ = fresh_model(args.arch, args.base, 1, args.device, response=True)
        probe.eval()
        tot, n = 0.0, 0
        for batch, _ in batches:
            d = batch.to_dict()
            d["positions"] = batch.positions.detach().clone().requires_grad_(True)
            d["_clamp_mask"] = None
            with torch.enable_grad():
                o = probe(d, training=False, compute_force=True)
            tot += float(((o["forces"] - batch.forces) ** 2).mean()); n += 1
        f0 = tot / max(n, 1)
        del probe
        log(f"  epoch-0 force loss {f0:.4f} eV^2/A^2")
    else:
        f0 = None

    rows = []
    for variant in args.variants:
        for m in args.margins:
            w_e = args.w_edge if args.w_edge is not None else f0 / (0.5 * m) ** 2
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                t0 = time.time()
                log(f"  --- {variant}  m={m}  seed {seed}  (w_e={w_e:.4g})")
                model, met = run_cell(args.arch, args.base, seed, batches, frame_masks,
                                      pristine_pool, args.device, args.epochs, args.lr,
                                      m, w_e, variant, log)
                ch = channel_of(batches[0][0])
                raw, matched = escape_report(model, big_charged, big_pristine, z_table,
                                             cutoff, args.device, ch)
                met["escape_raw"] = raw
                met["escape_gauge_matched"] = matched
                met["seconds"] = time.time() - t0
                rows.append(met)
                log(f"    {variant} m={m} s{seed}  N_eff {met['neff']:6.2f}  "
                    f"ratio {met['null_ratio']:.3f}  Delta_bind {met['delta_bind_final']:+.4f}  "
                    f"axial_red {met['axial_red']:+.3f}  rmse_nbhd {met['rmse_nbhd']:.1f}  "
                    f"loc_len {met['loc_length']:.2f} A")
                args.out.write_text(json.dumps(rows, indent=2, default=float))
                del model
                torch.cuda.empty_cache()
    args.out.write_text(json.dumps(rows, indent=2, default=float))
    log(f"wrote {args.out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
