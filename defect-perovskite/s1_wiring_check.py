#!/usr/bin/env python3
"""Section 1's on-batch gate: the density response is wired, and it changed nothing it must not.

Five checks on REAL frames with a real Stage-3 model, because the unit tests pin the algebra
on toys and the plumbing is where this has gone wrong before:

  A  the explicit `-Tr((P - P_ref) dH/dR)` reproduces the model's own head force. This is the
     `P - P_ref` bug class: using `P` alone would return the force of the whole valence
     manifold, which the base potential already carries, and would be an eV/A-scale double
     count on every frame.
  B  forces are BIT-IDENTICAL with the wiring on and off. Not a tolerance -- the response is
     `D - D.detach()` contracted with `dH/dR`, exactly zero, so anything but `torch.equal`
     here means something else moved.
  C  the force-loss gradient with respect to the hopping scales is DIFFERENT. Together with
     B this is the whole claim: same predictions, better gradient. Either half alone would
     pass for a no-op or for a silent prediction change.
  D  a pristine frame builds no response at all and feels no head force. The head vanishes at
     zero counters by construction, and this asserts it at the FORCE level, where it never
     has been.
  E  the per-step cost, on and off. Recorded now so a later "training got slower" has its
     baseline; the extra backward through the H builder plus one float64 eigensolve is real.

Run before the six-seed rerun, not after.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_matrix import make_batches  # noqa: E402
from stage_run import COMPOSITION, build, measure_t_ref  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402


def capture_hamiltonians(model):
    """Spy on `head_energy_hf` to keep each graph's H and its two densities.

    A spy rather than a new return value: the explicit force must be reconstructed from what
    the head ACTUALLY built on this batch, and a debug hook in the production forward is a
    place for train and evaluate to diverge.
    """
    from mace.modules import defect_counting as dc

    grabbed = []
    original = dc.head_energy_hf

    def wrapped(H, n_total, counts, t_el=dc.T_EL, occupation=None):
        out = original(H, n_total, counts, t_el, occupation=occupation)
        grabbed.append((H, out[3], out[4]))
        return out

    dc.head_energy_hf = wrapped
    return grabbed, (lambda: setattr(dc, "head_energy_hf", original))


def head_force_explicit(grabbed, positions):
    """`-Tr((P - P_ref) dH/dR)` summed over the graphs of one batch."""
    live = [(H, p, q) for H, p, q in grabbed if float((p - q).abs().max()) > 0]
    if not live:
        return torch.zeros_like(positions)
    grad = torch.autograd.grad(
        [H for H, _, _ in live], [positions],
        grad_outputs=[(p - q).to(H.dtype) for H, p, q in live],
        retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(positions) if grad is None else -grad


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--train", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--atoms", type=int, default=159,
                    help="frame size for the charged checks; the big frames are the ones "
                         "float32 used to fail on")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    probe = torch.load(args.arch, map_location="cpu", weights_only=False)
    cutoff = max(float(probe.r_max), float(getattr(probe, "spectral_r_cut", 0.0) or 0.0))
    del probe

    charged = [a for a in select(load_frames(args.data), charged=True)
               if len(a) == args.atoms][:2]
    if not charged:
        raise SystemExit(f"no charged frames of {args.atoms} atoms in {args.data}")
    batches = make_batches(charged, z_table, cutoff, len(charged), args.device)

    from ase.io import read as _read
    from d1_sensitivity import select_pristine
    pristine = select_pristine(_read(str(args.train), ":"), 2)
    pris_batches = make_batches(pristine, z_table, cutoff, len(pristine), args.device)

    t_ref = measure_t_ref(batches[0][0])
    model = build(args.arch, args.base, args.seed, args.device, 3, True,
                  args.eps_inf, t_ref, log)
    ctx = ForwardContext.production(model, device=args.device, eps_inf=args.eps_inf,
                                    stage=3, madelung=True, seed=args.seed)
    model.train()
    log(f"  {len(charged)} charged frames of {args.atoms} atoms, "
        f"{len(pristine)} pristine of {len(pristine[0])}, cutoff {cutoff:.1f} A, "
        f"t_ref {t_ref:.3f} A")

    batch, frames = batches[0]
    fails = []

    # ---- A: the explicit route reproduces the model's head force --------------------
    grabbed, restore = capture_hamiltonians(model)
    try:
        d = ctx.forward_dict(batch, frames, requires_grad=True)
        out = model(d, training=True, compute_force=True)
        head_auto = (out["forces"] - out["base_forces"]).detach()
        explicit = head_force_explicit(grabbed, d["positions"]).detach()
    finally:
        restore()
    gap = float((head_auto - explicit).abs().max())
    scale = float(head_auto.abs().max())
    log(f"\nA  explicit vs autograd head force: max |delta| {gap:.3e} eV/A "
        f"against a head force of {scale:.3e} eV/A")
    if not gap <= 1e-6:
        fails.append(f"A: explicit and autograd head forces differ by {gap:.3e} eV/A")

    # ---- B and C: same forces, different gradient ------------------------------------
    #
    # The response is asserted to be an EXACT zero on its own tensor, which is deterministic.
    # Comparing two forward passes instead would not be: MACE's scatter reductions use
    # atomics, so the same batch through the same weights twice differs in the last bits, and
    # a `torch.equal` between passes measures that floor rather than the wiring. The floor is
    # measured here and the on-versus-off difference is required to sit inside it.
    seen = {}

    def one_pass(wired, keep_response=False):
        model.spectral.wants_positions = bool(wired)
        model.zero_grad(set_to_none=True)
        d = ctx.forward_dict(batch, frames, requires_grad=True)
        if keep_response:
            # Keep a REFERENCE to the model's own dict; substituting one of our own would
            # capture the response and simultaneously stop it reaching the forces, which is
            # exactly the silent no-op this check exists to detect.
            handles = []
            original = model._carrier_head

            def spy(**kw):
                if kw.get("force_out") is not None:
                    handles.append(kw["force_out"])
                return original(**kw)

            model._carrier_head = spy
            try:
                out = model(d, training=True, compute_force=True)
            finally:
                model._carrier_head = original
            seen["response"] = handles[0].get("force_response") if handles else None
        else:
            out = model(d, training=True, compute_force=True)
        loss = ((out["forces"] - batch.forces) ** 2).mean()
        g = torch.autograd.grad(loss, model.spectral.h.v0_raw, retain_graph=False)[0]
        return out["forces"].detach().clone(), g.detach().clone()

    f_on, g_on = one_pass(True, keep_response=True)
    f_on2, _ = one_pass(True)
    f_off, g_off = one_pass(False)
    model.spectral.wants_positions = True

    resp = seen.get("response")
    resp_max = float("nan") if resp is None else float(resp.abs().max())
    floor = float((f_on - f_on2).abs().max())
    swing = float((f_on - f_off).abs().max())
    log(f"\nB  the response tensor itself: max |F_resp| {resp_max:.3e} eV/A (must be 0)")
    log(f"   forces on-vs-off {swing:.3e} against a repeat-run floor of {floor:.3e} eV/A "
        f"(scatter atomics)")
    if resp is None or resp_max != 0.0:
        fails.append(f"B: the response is not an identity zero (max {resp_max:.3e})")
    if swing > max(10.0 * floor, 1e-9):
        fails.append(f"B: forces moved by {swing:.3e} eV/A, well outside the {floor:.3e} "
                     "run-to-run floor -- the wiring changed predictions")

    delta = float((g_on - g_off).abs().max())
    rel = delta / max(float(g_off.abs().max()), 1e-30)
    log(f"C  d(force loss)/d(v0_raw): frozen-P max {float(g_off.abs().max()):.3e}, "
        f"wired max {float(g_on.abs().max()):.3e}, response term {delta:.3e} "
        f"({rel:.1f}x the frozen-P term)")
    if not (np.isfinite(delta) and delta > 1e-12):
        fails.append(f"C: the response added nothing to the gradient (delta {delta:.3e})")
    if not torch.isfinite(g_on).all():
        fails.append("C: the wired gradient is not finite")

    # ---- D: a pristine frame is untouched --------------------------------------------
    pb, pf = pris_batches[0]
    d = ctx.forward_dict(pb, pf, requires_grad=True)
    out = model(d, training=True, compute_force=True)
    head_pristine = float((out["forces"] - out["base_forces"]).abs().max())
    e_pristine = float(out["delta_sr_energy"].abs().max())
    log(f"\nD  pristine head force {head_pristine:.3e} eV/A, head energy "
        f"{e_pristine:.3e} eV")
    if not head_pristine < 1e-10:
        fails.append(f"D: a pristine frame feels a head force of {head_pristine:.3e} eV/A")

    # ---- E: the cost of the extra backward -------------------------------------------
    def timed(wired):
        model.spectral.wants_positions = bool(wired)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            model.zero_grad(set_to_none=True)
            d = ctx.forward_dict(batch, frames, requires_grad=True)
            out = model(d, training=True, compute_force=True)
            ((out["forces"] - batch.forces) ** 2).mean().backward()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.steps

    timed(True)                       # warm the allocator, discard
    t_on, t_off = timed(True), timed(False)
    model.spectral.wants_positions = True
    log(f"\nE  {1000 * t_off:.0f} ms/step frozen-P, {1000 * t_on:.0f} ms/step wired "
        f"({t_on / max(t_off, 1e-9):.2f}x) on {args.atoms}-atom frames, "
        f"batch {len(charged)}, {args.device}")

    log("\n" + ("SECTION 1 WIRING: PASS" if not fails else "SECTION 1 WIRING: FAIL"))
    for f in fails:
        log(f"  {f}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
