"""Stage B: train the correction branch against a frozen, pre-trained base.

The problem this solves (component S of the localisation plan)
--------------------------------------------------------------
The perovskite dataset has no paired charge-state geometries: the V_Cl0 and V_Cl+ ensembles
are geometrically disjoint, so there is no frame where the same geometry appears with and
without a carrier. Joint training therefore becomes a race. A geometry-only base is perfectly
capable of fitting each charge state's force field separately, and if it does, the correction
sees a near-constant residual with no site information left in it -- the signal is laundered
away before the carrier branch can claim it. Which branch wins is decided by the seed, which
is the observed behaviour: localisation succeeds on roughly half of seeds, and the failures
commit at the species level rather than the site level.

Staging removes the race. Train the base to convergence on n = 0 frames only (Stage A), freeze
it, then train the correction with that base held fixed. The frozen base evaluated at a
charged geometry acts as a surrogate for the missing neutral pair partner: the residual
approximates the vertical surface, and its force residual concentrates on the shell. That is
delta-force-like supervision recovered from unpaired data.

E0 measured that this residual really does carry the signal: the two under-coordinated Pb
carry 32% of the total squared force residual, against 14-16% for a carrier-free null control,
with a distance profile falling by a factor of 20 against the null's 2.8.

What "frozen" means here
------------------------
Two independent things, both required:

  * the base WEIGHTS come from the Stage-A checkpoint (this module), and
  * the base LEARNING RATE is zero (`--base_lr_factor 0.0`), so they stay there.

Doing only the first trains a correction against a base that immediately drifts; doing only
the second freezes a randomly initialised base, which is worse than not staging at all.
`assert_base_frozen` checks the pair actually held after an optimiser step.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

__all__ = ["CORRECTION_PREFIXES", "is_correction_param", "load_stage_a_base",
           "assert_base_frozen"]


# Names owned by the carrier correction rather than by the geometry-only base. Verified by
# construction: on an n = 0 batch exactly these tensors receive zero gradient from the base
# objective, and the complement is what E_base is built from.
#
# `novelty_*` are the logit-seeding scales, written by defect_seed and consumed only to seed
# the attention logits. They are registered CONDITIONALLY, so a Stage-A model trained with
# seeding off does not have them at all -- which is how they were caught: classified as base,
# they made every Stage-B load fail as an architecture mismatch. They are correction state and
# should be left at Stage B's own initialisation.
#
# `spectral` is the carrier Hamiltonian: it produces dE_SR and exists only when the spectral
# head is enabled, so it is correction state by definition. Left unclassified it was treated
# as base, which made every Stage-A load demand a Hamiltonian from a checkpoint that predates
# it -- and, worse, would have FROZEN the Hamiltonian in any staged run, silently turning the
# decisive arm into an untrained head. Same failure shape as novelty_ above.
#
# `madelung` is on this list for the same reason `spectral` is. The learned species charges Z
# are correction state introduced by Edit 1, not base weights: `stage_run` trains them,
# `defect_protocol.trainable_mask` returns True for them, and the post-step projection exists
# to keep them on the composition hyperplane. Omitting the prefix made `_base_state` classify
# them as base, so `load_stage_a_base` demanded a Stage-A checkpoint contain `madelung.z` --
# and no Stage-A base ever will, because Stage A has no Madelung term. The effect was that
# --defect_base_init and --defect_madelung_on_site could not be combined AT ALL from config,
# which is the joint run's own configuration. Found by running the production trainer end to
# end; every unit test passed throughout, because each of them exercised one flag.
# `pristine_` is the centred on-site correction's reference (Stage A' section 2.1): the
# species-mean first-block feature over the pristine cell, set by the trainer after the
# base is loaded. Correction state, not base weights; a Stage-A checkpoint never has it.
CORRECTION_PREFIXES = ("carrier_", "counter_", "latent_charges", "logit",
                       "defect_", "delta_", "novelty_", "spectral", "madelung",
                       "pristine_")


# Training bookkeeping: neither base weights nor correction state. `current_epoch` drives the
# epoch-dependent schedules (long-range gate, size warmup, seed anneal), and the training loop
# rewrites it every epoch. It must be excluded from BOTH operations here: copying it would
# seed a Stage-B run with Stage A's final epoch number, and checking it would report the base
# as unfrozen simply because training progressed -- which is how it was noticed.
# `trunk_avg_num_neighbors` is the serialised copy of the blocks' plain floats (Stage A'
# section 5.1). The loader carries the floats themselves, explicitly, below; the buffer is
# refreshed from them whenever a state dict is written, so it is neither copied nor
# checked here -- and a Stage-A checkpoint written before it existed must still load.
BOOKKEEPING_NAMES = ("current_epoch", "base_cache_checksum", "trunk_avg_num_neighbors")


def is_correction_param(name: str) -> bool:
    return any(name.startswith(p) or f".{p}" in name for p in CORRECTION_PREFIXES)


def is_bookkeeping(name: str) -> bool:
    return name.split(".")[-1] in BOOKKEEPING_NAMES


def _base_state(model) -> dict:
    """Base-branch parameters AND buffers, keyed by name."""
    out = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not is_correction_param(name) and not is_bookkeeping(name):
            out[name] = tensor
    return out


def load_stage_a_base(model, path, device="cpu", strict=True):
    """Copy the base branch of a saved Stage-A model into `model`, in place.

    Buffers are copied as well as parameters, deliberately. `atomic_energies` and the
    scale/shift are buffers derived from whichever training set computed them; Stage A used
    n = 0 frames while Stage B trains on all frames, so leaving them at Stage B's values would
    pair Stage A's weights with a different energy reference and silently change what E_base
    means. Copying them makes the frozen base numerically identical to the one E0 measured.
    """
    source = torch.load(path, map_location=device, weights_only=False)
    src = _base_state(source)
    dst = _base_state(model)

    missing = [k for k in dst if k not in src]
    extra = [k for k in src if k not in dst]
    mismatched = [k for k in dst if k in src and tuple(dst[k].shape) != tuple(src[k].shape)]

    if strict and (missing or mismatched):
        raise RuntimeError(
            "Stage-A base does not match this model's base branch.\n"
            f"  missing from checkpoint: {missing[:6]}{' ...' if len(missing) > 6 else ''}\n"
            f"  shape mismatches:        {mismatched[:6]}"
            f"{' ...' if len(mismatched) > 6 else ''}\n"
            "The trunk architecture must be identical between Stage A and Stage B "
            "(channels, max_L, interactions, cutoff, element set).")

    # `avg_num_neighbors` is NOT a parameter and NOT a buffer -- it is a plain float on each
    # interaction block, dividing every message. So it is absent from `state_dict`, which
    # means the name/shape check above cannot see it and the copy below cannot carry it, and
    # a Stage-B model built with a different value loads Stage A's weights into a trunk that
    # normalises them differently. The loader reports success and the base branch silently
    # stops reproducing Stage A.
    #
    # It is not a hypothetical mismatch. Stage A was trained at r_max = 5.0 with no carrier
    # head and got 14.08; a Stage-B run with the spectral head builds its graph at the
    # CARRIER cutoff of 10 A and computes 112.5 on the same data. Eight times the divisor on
    # every message. It was found because the c-shift calibration -- the median energy
    # mismatch, which is a direct read on E_base -- disagreed between the two drivers by a
    # factor of four, and every other candidate had been eliminated.
    src_ann = [float(b.avg_num_neighbors) for b in getattr(source, "interactions", [])]
    dst_ann = [float(b.avg_num_neighbors) for b in getattr(model, "interactions", [])]
    if src_ann and len(src_ann) == len(dst_ann):
        moved = [(a, b) for a, b in zip(dst_ann, src_ann) if abs(a - b) > 1e-6]
        if moved:
            with torch.no_grad():
                for block, value in zip(model.interactions, src_ann):
                    block.avg_num_neighbors = value
            logging.warning(
                "Stage B: avg_num_neighbors differed between this model and the Stage-A "
                "checkpoint (%s vs %s) and has been RESET to Stage A's. It is a plain float "
                "on each interaction block, not a buffer, so it is invisible to state_dict "
                "and the base branch would otherwise have normalised every message "
                "differently from the base whose weights it just loaded.",
                [round(a, 3) for a, _ in moved], [round(b, 3) for _, b in moved])
    elif src_ann:
        raise RuntimeError(
            f"Stage-A checkpoint has {len(src_ann)} interaction blocks and this model has "
            f"{len(dst_ann)}; avg_num_neighbors cannot be carried across and the base branch "
            "would not reproduce Stage A.")

    copied = 0
    with torch.no_grad():
        for name, tensor in dst.items():
            if name in src and tuple(tensor.shape) == tuple(src[name].shape):
                tensor.copy_(src[name].to(tensor.device, tensor.dtype))
                copied += 1

    logging.info(
        f"Stage B: loaded {copied} base tensors from {path} "
        f"({len(extra)} unused in checkpoint, {len(missing)} left at initialisation). "
        f"Correction branch is untouched and trains from scratch.")
    return copied


def assert_base_frozen(model, reference: dict, tol: float = 0.0) -> None:
    """Verify the base really did not move. Call after an optimiser step.

    A learning-rate factor of zero is an easy thing to believe and a cheap thing to check.
    If a base group were missed by the factor -- or a second optimiser touched it, or weight
    decay applied without a gradient -- the correction would be chasing a moving target again
    and the arm would silently stop being Stage B.
    """
    moved = []
    for name, tensor in _base_state(model).items():
        if name not in reference:
            continue
        # Some buffers are legitimately empty (an unset gauge probe, for instance). `.max()`
        # raises on a zero-element tensor, so skip them rather than let the freeze check die
        # on a tensor that cannot move.
        if tensor.numel() == 0:
            continue
        current = tensor.detach().cpu()
        # Bool and integer buffers (masks, counters, element tables) have no meaningful
        # difference, and subtracting bools raises. For those, "moved" means "not identical".
        if not current.is_floating_point():
            if not torch.equal(current, reference[name]):
                moved.append((name, float("inf")))
            continue
        delta = float((current - reference[name]).abs().max())
        if delta > tol:
            moved.append((name, delta))
    if moved:
        moved.sort(key=lambda kv: -kv[1])
        raise RuntimeError(
            "Base branch moved during Stage B training, so the base is not frozen:\n"
            + "\n".join(f"  {n}: max|delta| = {d:.3e}" for n, d in moved[:8])
            + "\nExpected --base_lr_factor 0.0 to hold every base group fixed.")


def snapshot_base(model) -> dict:
    """Detached CPU copy of the base branch, for `assert_base_frozen`."""
    return {name: tensor.detach().cpu().clone()
            for name, tensor in _base_state(model).items()}
