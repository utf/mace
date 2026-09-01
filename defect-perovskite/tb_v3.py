#!/usr/bin/env python3
"""T-B on V3: gap-referenced constraints on a Hamiltonian that is local by construction.

V2's result was that Delta_bind >= m is necessary but not sufficient: lambda_1(pristine)
floated over ~8 eV, so the inequality was met by sliding the host origin rather than by
splitting a level off the continuum. V3 removes the freedom (block-1 detached descriptor, so
the far block of H is pinned to the host) and the constraint gains an upper cap:

    L_edge = w_e [ max(0, m - Delta_bind)^2 + max(0, Delta_bind - E_gap)^2 ],  m = f_m * E_gap
    L_gap  = w_g ( lambda_1^e + mu_e + lambda_1^h + mu_h - E_gap )^2      on pristine frames

The cap is the escape detector and it is not decorative: V1 reached Delta_bind = +11.9 eV
against m = 0.5, which no level inside a 2.2 eV gap can do. Any breach means a route out
still exists and the run stops rather than producing a number.

NO PER-DEFECT TARGETS. The only per-host reference is the PBE band gap. Nothing here knows
where the defect level should sit, only that it must sit inside the gap.

GATES ARE REGION-RELATIVE. The candidate region is the set of atoms within r_max of the
vacancy, so `N_eff <= 8` -- a criterion for a region half the cell's size -- is not used.

MEASURED, and it is not what the plan assumed: at r_max = 5.0 A the region holds 14.2 atoms
of 89 on these frames, not the ~40 of 79 the plan states. That is simple geometry -- a 5 A
sphere is ~19% of a ~2800 A^3 cell -- and it makes the region gate roughly three times
TIGHTER than intended, so it is reported rather than quietly adopted. If the plan meant a
region of ~40 atoms, r_max is not the radius that produces it.

What is asked is that the state lives inside the candidate region and survives dilution:
retained mass >= 0.9 when the cell is tiled 2x1x1 with a pristine block, the criterion Test 2
validated.
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
from mace.data.dilution import (interface_mass, read_dilution, retained_mass,
                                tile_with_pristine)
from mace.modules.defect_stage import is_correction_param

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import (evaluate, fresh_model, graph_cutoff_for, make_batches,
                       masks_for, stacked)  # noqa: E402
from ta_band_edge import (PRISTINE_NATOMS, capture, channel_of, load_frames,  # noqa: E402
                          select, with_hole_counter)
from tb_edge import lambda1_grad, localisation_length, neff_and_nulls  # noqa: E402
from vacancy_site import locate_vacancy  # noqa: E402

from mace import tools  # noqa: E402

E_MAJ, H_MAJ = 0, 2          # channel indices: (e_maj, e_min, h_maj, h_min)


# --------------------------------------------------------------------------- gap provenance

GAP_SOURCES = {
    # DEFAULT. The dataset's own band edges (band_edges.json, e_cbm_cell +1.2 / e_vbm_cell
    # -1.2) are the reference the energy labels are already stated against, so they have
    # better provenance here than a literature number that may be for the cubic phase.
    # Orthorhombic tilting is expected to widen the PBE gap slightly over cubic.
    "dataset_band_edges": 2.4,
    # Sensitivity arm. Literature PBE, scalar-relativistic, no SOC -- matching the labels,
    # which are the paper's low-fidelity PBE set. Deliberately NOT the paper's Table III
    # 3.08 eV (tuned HSE + SOC, ~0.9 eV too high against these labels), NOT PBE+SOC (~1.3 eV;
    # SOC moves the Pb-6p band by ~1 eV and the labels are scalar-relativistic), NOT the
    # experimental 2.93 eV.
    "literature_pbe_nosoc": 2.2,
}
# The 0.2 eV between them moves m by 0.04 eV at f_m = 0.2 and the cap is soft, so no decision
# turns on the choice. Both are recorded; replace with the thermal-mean Kohn-Sham gap from the
# Zenodo bulk outputs if those are ever downloaded.


def resolve_gap(value, source):
    if value is not None:
        return float(value), source or "explicit"
    return GAP_SOURCES["dataset_band_edges"], "dataset_band_edges(+/-1.2 eV -> 2.4)"


# --------------------------------------------------------------------------- candidate region


def candidate_mask(atoms, r_max):
    """Atoms within r_max of the vacancy site -- the region a level may occupy.

    EVALUATION ONLY. The vacancy position never enters the loss; it is used here to say which
    atoms the answer is allowed to live on, and to build the far block below.
    """
    from ase.geometry import get_distances

    try:
        site = locate_vacancy(atoms)
    except ValueError:
        return None
    if site.cage is None or len(site.cage) == 0:
        return None
    pos = atoms.get_positions()
    cell, pbc = atoms.get_cell(), atoms.pbc
    a, b = int(site.shell[0]), int(site.shell[1])
    vec, _ = get_distances(pos[a][None], pos[b][None], cell=cell, pbc=pbc)
    centre = pos[a] + 0.5 * vec[0, 0]
    dv, _ = get_distances(centre[None], pos, cell=cell, pbc=pbc)
    return np.linalg.norm(dv[0], axis=-1) <= float(r_max)


def init_mu_for_gap(model, pristine_batch, e_gap, log):
    """Set mu_e + mu_h so L_gap is satisfied exactly at step 0.

    L_gap is a single scalar equality that mu_e + mu_h can zero outright, so it collapses
    immediately and is inert with respect to H -- its job is to pin the two channel offsets to
    each other, which only becomes informative in a host where both channels carry data.
    Leaving it to find that by gradient descent means starting orders of magnitude above the
    force loss and briefly destroying the fit for nothing. The shift is split evenly so any
    existing difference between the two offsets survives.
    """
    with torch.no_grad():
        lam_e = float(lambda1_grad(model, pristine_batch, E_MAJ).mean())
        lam_h = float(lambda1_grad(model, pristine_batch, H_MAJ).mean())
        mu = model.spectral.mu
        current = lam_e + float(mu[E_MAJ]) + lam_h + float(mu[H_MAJ])
        shift = (float(e_gap) - current) / 2.0
        mu[E_MAJ] += shift
        mu[H_MAJ] += shift
    log(f"      mu init: lambda_e {lam_e:+.3f}, lambda_h {lam_h:+.3f}, shifted each mu by "
        f"{shift:+.3f} -> L_gap residual 0 at step 0")


def far_block_lambda1(internals, batch, near_masks, channel):
    """lambda_1 of H restricted to atoms BEYOND r_max of the vacancy.

    The defect cell's own continuum edge, read off the Hamiltonian the head assembled rather
    than a re-derived one. With V3 this must coincide with lambda_1(pristine): the far block's
    matrix elements are identical to the host's by construction. Their agreement is therefore
    a check on the construction, not an assumption -- which is why both are reported.
    """
    H = internals["H"]                                   # [G, C, n_max, n_max]
    idx = batch.batch.detach().cpu().numpy()
    out = []
    for g, near in enumerate(near_masks):
        if near is None:
            out.append(float("nan"))
            continue
        n = int((idx == g).sum())
        far = ~near[:n]
        if far.sum() < 2:
            out.append(float("nan"))
            continue
        sel = torch.as_tensor(np.where(far)[0], device=H.device)
        sub = H[g, channel][sel][:, sel]
        ev = torch.linalg.eigvalsh(sub)
        ev = ev[ev < 500.0]
        out.append(float(ev.min()) if ev.numel() else float("nan"))
    return out


# --------------------------------------------------------------------------- constraints


def edge_loss(delta, m, e_gap, w_e):
    """Two-sided: a floor at m and a cap at the other band edge."""
    below = torch.clamp(m - delta, min=0.0) ** 2
    above = torch.clamp(delta - e_gap, min=0.0) ** 2
    return w_e * (below + above).mean()


def gap_loss(model, pristine_batch, e_gap, w_g):
    """(lambda_1^e + mu_e) + (lambda_1^h + mu_h) = E_gap on pristine frames.

    Ties the two channels' edges to each other rather than pinning either one. Here only the
    hole channel carries data, so this is close to a regulariser; in a two-carrier host it is
    the term that makes the pair of edges consistent.
    """
    lam_e = lambda1_grad(model, pristine_batch, E_MAJ)
    lam_h = lambda1_grad(model, pristine_batch, H_MAJ)
    mu = model.spectral.mu
    total = lam_e + mu[E_MAJ] + lam_h + mu[H_MAJ]
    resid = (total - e_gap)
    # The raw residual is returned alongside the weighted loss because the weighted number is
    # uninformative on its own: an untrained head puts the two edges tens of eV apart, so
    # L_gap starts orders of magnitude above the force loss and dominates until it is
    # satisfied. Whether that is converging is readable from the residual, not from w_g*r^2.
    return w_g * (resid ** 2).mean(), float(resid.abs().mean())


# --------------------------------------------------------------------------- one cell


PROFILE_R = (2.85, 5.6, 8.0, 10.0)          # where the realised hopping profile is reported


def log_profile(model, log, when):
    """Report the realised |t| at the profile distances -- init, and end of run.

    The end-of-run values are the effective-coupling-length diagnostic: what the head actually
    settled on, as opposed to what it was initialised to.
    """
    prof = model.spectral.hopping_profile(PROFILE_R)
    first = prof[0] if prof[0] > 0 else float("nan")
    txt = "  ".join(f"t({r:.2f})={v:.4f} ({100 * v / first:5.2f}%)"
                    for r, v in zip(PROFILE_R, prof))
    log(f"      hopping profile [{when}]: {txt}")
    return {f"t_{r}": float(v) for r, v in zip(PROFILE_R, prof)}


def run_cell(arch_path, base_path, seed, batches, frame_masks, near_masks, pristine_pool,
             pristine_frames, pristine_far_masks, device, epochs, lr, f_m, e_gap,
             w_e, w_g, t_ref_r, log, sign="off", clamp=None):
    from mace.modules.defect_spectral_v3 import install_local_head

    model, _ = fresh_model(arch_path, base_path, seed, device, response=True)
    install_local_head(model, t_ref_r=t_ref_r, single_manifold=True)
    # The single variable under test. install_local_head sets (+1,+1,-1,-1); OFF restores the
    # all-ones convention so the two arms differ in exactly this buffer and nothing else.
    with torch.no_grad():
        if sign == "off":
            model.spectral.channel_sign.fill_(1.0)
    log(f"      arm {sign.upper()}: channel_sign "
        f"{model.spectral.channel_sign.tolist()}"
        + (f", clamp={clamp}" if clamp else ""))
    prof_init = log_profile(model, log, "init")
    # L_gap is a scalar equality mu can satisfy outright; do that rather than spend the first
    # epochs letting it dominate the loss on its way to the same place.
    init_mu_for_gap(model, pristine_pool[0], e_gap, log)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad_(is_correction_param(n) or ".spectral." in n)
    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=lr)

    m = float(f_m) * float(e_gap)
    rng = np.random.default_rng(seed)
    channel = channel_of(batches[0][0])
    hist, breaches = [], 0

    for ep in range(epochs):
        f_sum = e_sum = g_sum = d_sum = r_sum = 0.0
        n_step = 0
        ep_delta = []
        for batch, frames in batches:
            d = batch.to_dict()
            d["positions"] = batch.positions.detach().clone().requires_grad_(True)
            d["_clamp_mask"] = (None if clamp is None
                                else stacked([frame_masks[id(f)] for f in frames],
                                             clamp, device))
            out = model(d, training=True, compute_force=True)
            f_loss = ((out["forces"] - batch.forces) ** 2).mean()

            pb = pristine_pool[int(rng.integers(len(pristine_pool)))]
            lam_p = lambda1_grad(model, pb, channel).mean()
            lam_d = lambda1_grad(model, d, channel)
            delta = lam_p - lam_d
            l_edge = edge_loss(delta, m, e_gap, w_e)
            l_gap, gap_res = gap_loss(model, pb, e_gap, w_g)
            loss = f_loss + l_edge + l_gap

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            f_sum += float(f_loss); e_sum += float(l_edge); g_sum += float(l_gap)
            r_sum += gap_res
            d_sum += float(delta.mean()); n_step += 1
            ep_delta += delta.detach().cpu().numpy().tolist()

        # Test 5: a breach of the cap is the escape signature. Judged on the epoch MEDIAN and
        # on consecutive epochs, because a single 8-frame pristine draw can spike Delta_bind
        # without any escape being real -- but every individual breach is still logged.
        med = float(np.median(ep_delta)) if ep_delta else float("nan")
        n_over = int(np.sum(np.asarray(ep_delta) > e_gap))
        if n_over:
            log(f"      ep{ep:3d}  CAP: {n_over}/{len(ep_delta)} frames over E_gap={e_gap}, "
                f"median {med:+.4f}")
        if med > e_gap:
            breaches += 1
            if breaches >= 2:
                raise RuntimeError(
                    f"E_GAP BREACH: epoch-median Delta_bind {med:.4f} exceeded E_gap "
                    f"{e_gap} in two consecutive epochs. An escape route remains open; the "
                    f"decision tree says find it (tests 1-3) before anything else, so this "
                    f"run is aborting rather than producing a number.")
        else:
            breaches = 0

        row = dict(epoch=ep, force=f_sum / max(n_step, 1), edge=e_sum / max(n_step, 1),
                   gap=g_sum / max(n_step, 1), gap_residual=r_sum / max(n_step, 1),
                   delta_bind=d_sum / max(n_step, 1),
                   delta_median=med)
        hist.append(row)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"      ep{ep:3d}  force {row['force']:.4f}  edge {row['edge']:.5f}  "
                f"gap {row['gap']:.4g} (resid {row['gap_residual']:.3f} eV)  "
                f"Delta_bind {row['delta_bind']:+.4f}")

    metrics = evaluate(model, batches, frame_masks, device, clamp=clamp)
    metrics.update(seed=seed, f_m=f_m, m=m, e_gap=e_gap, w_e=w_e, w_g=w_g,
                   arch=str(arch_path), history=hist)

    # ---- gates, all evaluation-only
    model.eval()
    ref, ref_trunc = [], []
    with torch.no_grad():
        for k, pb in enumerate(pristine_pool):
            ref.append(lambda1_grad(model, pb, channel).detach().cpu().numpy())
    # LIKE-WITH-LIKE reference. lambda_1 of the defect cell's far block is the lowest
    # eigenvalue of a TRUNCATED matrix, and interlacing puts that above the full matrix's
    # regardless of how the elements compare -- so differencing it against the full pristine
    # lambda_1 measures the truncation, not the binding. The pristine cells are therefore
    # truncated the same way (the same exclusion radius about a fixed site) and the two
    # truncated values are compared, so the truncation term cancels. Element-level identity
    # beyond r_max, which test 2 asserts, remains the constructive guarantee.
    for pb, fm in zip(pristine_pool, pristine_far_masks):
        internals_p = {}
        head = model.spectral
        orig = head.forward

        def _wrap(*a, **k):
            k["internals"] = internals_p
            return orig(*a, **k)

        head.forward = _wrap
        try:
            with torch.no_grad():
                model(pb, training=False, compute_force=False)
        finally:
            head.forward = orig
        bidx = pb["batch"].detach().cpu().numpy()
        H = internals_p["H"]
        for g, far in enumerate(fm):
            n = int((bidx == g).sum())
            f = far[:n]
            if f.sum() < 2:
                continue
            sel = torch.as_tensor(np.where(f)[0], device=H.device)
            ev = torch.linalg.eigvalsh(H[g, channel][sel][:, sel])
            ev = ev[ev < 500.0]
            if ev.numel():
                ref_trunc.append(float(ev.min()))
    ref_mean = float(np.mean(np.concatenate(ref)))
    ref_trunc_mean = float(np.mean(ref_trunc)) if ref_trunc else float("nan")

    act, nul, lens, d_pri, d_far, region, occ = [], [], [], [], [], [], []
    gi = 0
    for batch, frames in batches:
        cm = (None if clamp is None
              else stacked([frame_masks[id(f)] for f in frames], clamp, device))
        internals, out = capture(model, batch, clamp_mask=cm)
        a, n = neff_and_nulls(out, batch, channel)
        act += a; nul += n
        lens += localisation_length(internals["psi"], internals["w"], batch.positions,
                                    batch.batch, channel, int(batch.num_graphs),
                                    cell=getattr(batch, "cell", None))
        nm = [near_masks[id(f)] for f in frames]
        lam_d = internals["lam"][:, channel, :]
        lam_d = torch.where(lam_d < 500.0, lam_d, torch.full_like(lam_d, float("inf")))
        lam_d = lam_d.min(dim=-1).values.detach().cpu().numpy()
        d_pri += (ref_mean - lam_d).tolist()
        fb = far_block_lambda1(internals, batch, nm, channel)
        # Truncated-to-truncated, so the truncation offset cancels.
        d_far += [float(ref_trunc_mean - f) for f in fb]

        # Mass inside the candidate region, and the region's size.
        alpha = out["carrier_alpha"][:, channel].detach().cpu().numpy()
        idx = batch.batch.detach().cpu().numpy()
        for g, msk in enumerate(nm):
            sel = idx == g
            av = alpha[sel]
            if msk is None:
                region.append(float("nan")); occ.append(float("nan")); continue
            region.append(float(msk.sum()))
            tot = float(av.sum())
            occ.append(float(av[msk[:av.size]].sum() / tot) if tot > 0 else float("nan"))
        gi += int(batch.num_graphs)

    metrics["neff"] = float(np.nanmean(act))
    metrics["null_ratio"] = float(np.nanmean(act) / max(np.nanmean(nul), 1e-12))
    metrics["loc_length"] = float(np.nanmean(lens))
    metrics["delta_bind_vs_pristine"] = float(np.nanmean(d_pri))
    metrics["delta_bind_vs_far_block"] = float(np.nanmean(d_far))
    metrics["delta_bind_agreement"] = abs(metrics["delta_bind_vs_pristine"]
                                          - metrics["delta_bind_vs_far_block"])
    metrics["candidate_region_size"] = float(np.nanmean(region))
    metrics["mass_in_region"] = float(np.nanmean(occ))
    metrics["lambda1_pristine_final"] = ref_mean
    metrics["lambda1_pristine_truncated"] = ref_trunc_mean

    # ---- retained mass under 2x1x1 dilution with a pristine block
    metrics["retained_mass"], metrics["interface_mass"] = dilution_gate(
        model, batches, pristine_frames, device, channel, rng)
    metrics["dilution_reading"] = read_dilution(metrics["retained_mass"],
                                                metrics["interface_mass"])

    metrics["profile_init"] = prof_init
    metrics["profile_final"] = log_profile(model, log, "final")

    # ---- gate verdicts, region-relative and stated explicitly
    reg = metrics["candidate_region_size"]
    metrics["gate_neff"] = bool(metrics["neff"] <= reg) if np.isfinite(reg) else False
    metrics["gate_ratio"] = bool(metrics["null_ratio"] <= 0.15)
    metrics["gate_delta"] = bool(m <= metrics["delta_bind_vs_pristine"] <= e_gap)
    metrics["gate_retained"] = bool(metrics["retained_mass"] >= 0.9)
    metrics["gates_passed"] = int(metrics["gate_neff"]) + int(metrics["gate_ratio"]) \
        + int(metrics["gate_delta"]) + int(metrics["gate_retained"])
    return model, metrics


def dilution_gate(model, batches, pristine_frames, device, channel, rng, n_frames=8):
    """Retained carrier mass when the cell is tiled 2x1x1 with a pristine block.

    The criterion Test 2 validated: a bound state keeps ~all of its amplitude in the original
    block, a band state keeps ~half. Size-invariant, and it needs no defect label beyond
    knowing which half of the tiled cell the original was.
    """
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = graph_cutoff_for(model)
    vals, iface = [], []
    frames = [f for _, fr in batches for f in fr][:n_frames]
    for f in frames:
        try:
            tiled, mask, axis = tile_with_pristine(f, pristine_frames, rng)
        except Exception:                                   # noqa: BLE001
            continue
        if tiled is None:
            continue
        try:
            tb = make_batches([tiled], z_table, cutoff, 1, device)
        except Exception:                                   # noqa: BLE001
            continue
        for b, _ in tb:
            with torch.no_grad():
                o = model(b.to_dict(), training=False, compute_force=False)
            a = o["carrier_alpha"][:, channel].detach().cpu().numpy()
            vals.append(retained_mass(a, mask[:a.size]))
            # Retained mass ALONE is ambiguous and was reported without this once already:
            # ~0.5 with low interface mass is a genuinely band-like state, ~0.5 with high
            # interface mass is a straddling artefact sitting on the join and must not be
            # read as band-like. The two are only separable with both numbers.
            iface.append(interface_mass(a, tiled.get_positions(), tiled.get_cell(),
                                        axis, len(f)))
    return (float(np.nanmean(vals)) if vals else float("nan"),
            float(np.nanmean(iface)) if iface else float("nan"))


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
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--f-m", nargs="+", type=float, default=[0.2])
    ap.add_argument("--e-gap", type=float, default=None)
    ap.add_argument("--gap-source", default=None)
    ap.add_argument("--w-edge", type=float, default=None)
    ap.add_argument("--n-pristine", type=int, default=64)
    ap.add_argument("--pristine-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sign", choices=["on", "off"], default="off",
                    help="ON = hole channels contribute -(Lambda + mu_c). The ONLY difference "
                         "between the A/B arms: both use the counter-free single manifold, so "
                         "'bit-identical otherwise' holds exactly. For this dataset every "
                         "charged frame carries the same counter (0,0,1,0), so the counter "
                         "embedding is a constant vector and dropping it shifts eps by a "
                         "constant the gauge absorbs -- which is why OFF here still reproduces "
                         "the F1/F2 phenomenology measured on counter-dependent heads.")
    ap.add_argument("--clamp", default=None,
                    help="C1: restrict the carrier to a mask (e.g. hub2). Fixes hub mass by "
                         "construction so the energy-channel claim is tested in isolation "
                         "from placement.")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="write each cell's trained model here. Follow-up diagnostics "
                         "(the hopping-offset comparison, T-A on the trained head) need the "
                         "model, and it was not saved for V1/V2 -- so those analyses cannot "
                         "be run on them at all without a full re-run.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()

    def log(msg):
        print(msg, flush=True)

    e_gap, gap_src = resolve_gap(args.e_gap, args.gap_source)
    log(f"  E_gap {e_gap} eV  (source: {gap_src})")

    arch = torch.load(args.arch, map_location="cpu", weights_only=False)
    cutoff = graph_cutoff_for(arch)
    r_max = float(arch.r_max)
    z_table = tools.AtomicNumberTable([int(z) for z in arch.atomic_numbers])
    log(f"  graph cutoff {cutoff:.1f} A (r_max {r_max:.1f}) -> candidate region = "
        f"atoms within {r_max:.1f} A of the vacancy")

    frames_all = load_frames(args.data)
    rng = np.random.default_rng(0)
    charged, frame_masks, near_masks = [], {}, {}
    for a in frames_all:
        c = a.info.get("carrier_counts")
        if c is None or int(np.asarray(c).sum()) == 0:
            continue
        fm = masks_for(a, rng)
        nm = candidate_mask(a, r_max)
        if fm is None or nm is None:
            continue
        charged.append(a); frame_masks[id(a)] = fm; near_masks[id(a)] = nm
        if len(charged) >= args.frames:
            break
    log(f"  {len(charged)} charged frames; candidate region "
        f"{np.mean([near_masks[id(a)].sum() for a in charged]):.1f} of "
        f"{np.mean([len(a) for a in charged]):.1f} atoms")
    batches = make_batches(charged, z_table, cutoff, args.batch_size, args.device)

    # Test 4, on a batch this harness will actually consume rather than on a rebuilt one.
    # Aborts if the flanking separations are not coupled: nothing downstream would be
    # interpretable, which is exactly how a 20-seed screen was voided before.
    from mace.modules.defect_reach import assert_coupling_envelope
    rep = assert_coupling_envelope(batches[0][0], r_couple=cutoff)
    log(f"  coupling envelope: f_env(6.8 A) = {rep['f_env_at_window_far_end']:.3f}, "
        f"{rep['frames_with_a_window_edge']:.0%} of frames carry a window edge")

    train = load_frames(args.train)
    pristine_frames = select(train, natoms=PRISTINE_NATOMS, charged=False)[:args.n_pristine]
    pristine = with_hole_counter(pristine_frames)
    pool = make_batches(pristine, z_table, cutoff, args.pristine_batch, args.device)
    pristine_pool = [b.to_dict() for b, _ in pool]
    log(f"  pristine pool: {len(pristine)} frames in {len(pristine_pool)} draws")

    # Amplitude calibration reference: the MEASURED median nearest-neighbour distance, not a
    # constant. Used once at construction to place the initial hopping scale; it never enters
    # the functional form, so no material-specific length is introduced.
    b0 = batches[0][0]
    _ei = b0.edge_index
    _vec = b0.positions[_ei[1]] - b0.positions[_ei[0]]
    if getattr(b0, "shifts", None) is not None and b0.shifts.numel() == _vec.numel():
        _vec = _vec + b0.shifts
    _len = torch.linalg.norm(_vec, dim=-1)
    _nn = torch.full((int(b0.num_nodes),), float("inf"), device=_len.device)
    _nn = _nn.index_reduce(0, _ei[0], _len, "amin", include_self=True)
    t_ref_r = float(np.median(_nn[torch.isfinite(_nn)].detach().cpu().numpy()))
    log(f"  measured median nearest-neighbour distance {t_ref_r:.3f} A "
        f"(hopping amplitude calibrated here; not a cutoff)")

    # Pristine frames truncated the SAME way as the defect far block, so the truncation
    # cancels in the comparison. A fixed site per frame keeps it deterministic.
    from ase.geometry import get_distances
    pristine_far_masks = []
    for _b, _fr in pool:
        _masks = []
        for _a in _fr:
            _pos = _a.get_positions()
            _dv, _ = get_distances(_pos[0][None], _pos, cell=_a.get_cell(), pbc=_a.pbc)
            _masks.append(np.linalg.norm(_dv[0], axis=-1) > r_max)
        pristine_far_masks.append(_masks)

    # w_e from this harness's own epoch-0 force loss, as before.
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
    for f_m in args.f_m:
        m = f_m * e_gap
        w_e = args.w_edge if args.w_edge is not None else f0 / (0.5 * m) ** 2
        w_g = w_e
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            t0 = time.time()
            log(f"  --- V3  f_m={f_m} (m={m:.3f} eV)  E_gap={e_gap}  seed {seed}  "
                f"(w_e={w_e:.4g})")
            try:
                model, met = run_cell(args.arch, args.base, seed, batches, frame_masks,
                                      near_masks, pristine_pool, pristine_frames,
                                      pristine_far_masks, args.device, args.epochs,
                                      args.lr, f_m, e_gap, w_e, w_g, t_ref_r, log,
                                      sign=args.sign, clamp=args.clamp)
            except RuntimeError as exc:
                if "E_GAP BREACH" not in str(exc):
                    raise
                log(f"    ABORTED: {exc}")
                rows.append(dict(seed=seed, f_m=f_m, e_gap=e_gap, error=str(exc)))
                args.out.write_text(json.dumps(rows, indent=2, default=float))
                continue
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                dest = args.save_dir / f"v3_{args.sign}_fm{f_m}_gap{e_gap}_s{seed}.model"
                torch.save(model, dest)
                met["saved_model"] = str(dest)
            met["sign_arm"] = args.sign
            met["clamp"] = args.clamp
            met["gap_source"] = gap_src
            met["seconds"] = time.time() - t0
            rows.append(met)
            log(f"    V3 f_m={f_m} s{seed}  N_eff {met['neff']:6.2f}/"
                f"{met['candidate_region_size']:.0f}  ratio {met['null_ratio']:.3f}  "
                f"D_bind {met['delta_bind_vs_pristine']:+.4f} (far {met['delta_bind_vs_far_block']:+.4f}, "
                f"agree {met['delta_bind_agreement']:.4f})  region_mass {met['mass_in_region']:.3f}  "
                f"retained {met['retained_mass']:.3f} (iface {met['interface_mass']:.3f}, "
                f"{met['dilution_reading']})  axial {met['axial_red']:+.3f}  "
                f"loc {met['loc_length']:.2f} A  gates {met['gates_passed']}/4")
            args.out.write_text(json.dumps(rows, indent=2, default=float))
            del model
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(rows, indent=2, default=float))
    log(f"wrote {args.out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
