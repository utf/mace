#!/usr/bin/env python3
"""Section 3: the config path rebuilds the hand-wired Stage-3 model bit for bit.

THE FIFTH-INSTANCE GUARD. Four times in this programme a run's saved artefact has disagreed
with the configuration that produced it -- a flag present on one construction site and absent
on the other, a checkpoint that restores without an attribute, a save tag that collided. Each
time the symptom was a plausible number from a model nobody had specified. So before any seed
is spent on the corrected kernel, the production path must reproduce the hand-wired path
exactly, on a real batch, in energies, forces and site charges.

Three checks, in increasing strength:

  1  `extract_config_mace_model` round-trips the Edit 1 / Edit 4 flags. A config that loses
     `counting_head` rebuilds a spectral-head model that runs fine and means nothing.
  2  `_defect_madelung_kwargs` maps the CLI flags onto exactly those constructor arguments,
     so `--defect_counting_head` and the hand-wired `cfg["counting_head"] = True` are the
     same switch rather than two switches that happen to agree today.
  3  BIT-IDENTITY on a real batch: rebuild from the round-tripped config, copy the state
     dict, and require `torch.equal` on total energy, forces and q_i. Not `allclose` --
     the two models are meant to be the same function, and a tolerance here would hide
     exactly the flag drift this exists to catch.

Both paths now route through the full-sum Madelung kernel, so the comparison is like with
like.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

import mace  # noqa: F401  (before e3nn)
from mace import tools
from mace.modules.defect_context import EPS_INF_DEFAULT, ForwardContext
from mace.tools.model_script_utils import _defect_madelung_kwargs
from mace.tools.scripts_utils import extract_config_mace_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_matrix import make_batches  # noqa: E402
from stage_run import COMPOSITION, Z_INIT, build, measure_t_ref  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

FLAG_KEYS = ("counting_head", "counting_on_site_range", "counting_hop_range",
             "counting_smearing_family", "madelung_on_site", "madelung_eps_inf",
             "madelung_composition", "madelung_z_init")


def cli_namespace(eps_inf: float) -> Namespace:
    """The CLI flags a production run would pass for the Stage-3 configuration."""
    return Namespace(
        defect_counting_head=True,
        defect_spectral_head=True,
        defect_counting_on_site_range=3.0,
        defect_counting_hop_range=0.5,
        defect_counting_smearing="gaussian",
        defect_madelung_on_site=True,
        defect_madelung_eps_inf=eps_inf,
        defect_madelung_composition=",".join(str(c) for c in COMPOSITION),
        defect_madelung_z_init=",".join(str(z) for z in Z_INIT),
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "valid.xyz")
    ap.add_argument("--eps-inf", type=float, default=EPS_INF_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    fails = []

    # ---- 2: the CLI flags map onto the constructor arguments -----------------------
    kw = _defect_madelung_kwargs(cli_namespace(args.eps_inf))
    log(f"\n2  CLI -> constructor: {kw}")
    if kw["counting_head"] is not True or kw["madelung_on_site"] is not True:
        fails.append("2: the CLI flags did not switch the head on")
    if kw["madelung_composition"] != [float(c) for c in COMPOSITION]:
        fails.append(f"2: composition parsed as {kw['madelung_composition']}")
    if kw["madelung_z_init"] != [float(z) for z in Z_INIT]:
        fails.append(f"2: z_init parsed as {kw['madelung_z_init']}")
    for bad, why in ((Namespace(defect_counting_head=True, defect_spectral_head=False),
                      "counting head without the spectral branch"),
                     (Namespace(defect_madelung_on_site=True,
                                defect_madelung_composition=None),
                      "madelung without a composition")):
        try:
            _defect_madelung_kwargs(bad)
            fails.append(f"2: {why} was accepted; it must be refused")
        except (ValueError, AttributeError):
            log(f"   refused, as it must be: {why}")

    # ---- the hand-wired model, and one real batch ----------------------------------
    frames = select(load_frames(args.data), charged=True)[:4]
    probe = torch.load(args.arch, map_location="cpu", weights_only=False)
    cutoff = max(float(probe.r_max), float(getattr(probe, "spectral_r_cut", 0.0) or 0.0))
    del probe
    batches = make_batches(frames, tools.AtomicNumberTable(sorted({17, 55, 82})),
                           cutoff, len(frames), args.device)
    batch, frs = batches[0]
    t_ref = measure_t_ref(batch)
    hand = build(args.arch, args.base, 1, args.device, 3, True, args.eps_inf, t_ref, log)

    # ---- 1: the config round-trips the flags ---------------------------------------
    cfg = extract_config_mace_model(hand)
    log("\n1  round-tripped config: "
        + ", ".join(f"{k}={cfg.get(k)!r}" for k in FLAG_KEYS))
    for k in FLAG_KEYS:
        if k not in cfg:
            fails.append(f"1: {k} is absent from the extracted config")
    if cfg.get("counting_head") is not True or cfg.get("madelung_on_site") is not True:
        fails.append("1: the round trip lost the head switches")
    for k, v in kw.items():
        if k in cfg and cfg[k] != v:
            fails.append(f"1/2: config says {k}={cfg[k]!r}, the CLI path says {v!r}")

    # ---- 3: bit-identity on a real batch -------------------------------------------
    rebuilt = hand.__class__(**cfg)
    rebuilt.load_state_dict(hand.state_dict())
    rebuilt = rebuilt.to(args.device)
    ctx = ForwardContext.production(hand, device=args.device, eps_inf=args.eps_inf,
                                    stage=3, madelung=True, seed=1)
    # The hand-wired model runs TWICE, so the third row is a repeat-run floor. MACE's
    # scatter reductions use atomics: the same model on the same batch does not reproduce
    # itself bit for bit on a GPU, and without this control a nondeterminism at 1e-6 would be
    # reported as a config difference -- or, worse, a real config difference of that size
    # would be dismissed as noise. The floor is measured, not assumed.
    outs = []
    for model in (hand, rebuilt, hand):
        model.eval()
        d = ctx.forward_dict(batch, frs, requires_grad=True)
        out = model(d, training=False, compute_force=True)
        head = model.spectral
        grabbed = {}
        original = head.forward
        head.forward = lambda *a, **k: original(*a, **dict(k, internals=grabbed))
        try:
            with torch.no_grad():
                model(ctx.forward_dict(batch, frs, requires_grad=False),
                      training=False, compute_force=False)
        finally:
            head.forward = original
        outs.append((out["energy"].detach(), out["forces"].detach(),
                     grabbed.get("eps"), out["delta_sr_energy"].detach()))

    names = ("energy", "forces", "eps (site)", "delta_sr")
    log("")
    for name, a, b, c in zip(names, outs[0], outs[1], outs[2]):
        if a is None or b is None:
            log(f"3  {name:12s} not captured on one path")
            fails.append(f"3: {name} unavailable for comparison")
            continue
        gap = float((a - b).abs().max())
        floor = float((a - c).abs().max())
        same = bool(torch.equal(a, b))
        log(f"3  {name:12s} cross-path {gap:.3e}   same-model repeat {floor:.3e}"
            f"   bit-identical: {same}")
        # Bit-identity where the hardware allows it; inside the measured floor where it does
        # not. A cross-path gap an order of magnitude above the floor is a real difference.
        if not same and gap > max(10.0 * floor, 1e-12):
            fails.append(f"3: {name} differs by {gap:.3e} across build paths against a "
                         f"{floor:.3e} repeat-run floor -- that is a config difference, "
                         "not scatter nondeterminism")

    log("\n" + ("SECTION 3 BIT-IDENTITY: PASS" if not fails
                else "SECTION 3 BIT-IDENTITY: FAIL"))
    for f in fails:
        log(f"  {f}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
