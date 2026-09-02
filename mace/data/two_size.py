"""Upweight the charged frames at the second cell size so their boundness signal survives.

Test 2 established from DFT that the axial hub residual does not dilute when the cell volume
doubles (R_DFT = 0.95, CI [0.66, 1.34]). That is the measurement which says the carrier is
bound rather than band-like -- and it lives entirely in 17 of 1047 charged frames. At natural
weighting those 17 contribute ~1.6% of the charged force loss, so the one signal in the data
that distinguishes a bound carrier from a delocalised one is numerically invisible.

This scales their per-configuration `forces_weight` so their share of the charged force loss
reaches a target fraction. Cell size is a property of the simulation box, not a defect label:
nothing here says where the vacancy is, or that a carrier is bound. It makes the model see
evidence it already had.

The realised share is returned and logged every run, because a weight chosen from frame counts
alone would drift if the loss weights or the charged/neutral mix ever changed.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np

__all__ = ["apply_two_size_upweight", "apply_neutral_size_upweight"]


def _n_atoms(d) -> int:
    return int(d.positions.shape[0])


def _is_charged(d) -> bool:
    counts = getattr(d, "carrier_counts", None)
    if counts is None:
        return False
    return bool(counts.abs().sum() > 0)


def apply_two_size_upweight(dataset: Sequence, target_share: float = 0.25,
                            size_threshold: int = 100) -> Tuple[float, float]:
    """Scale `forces_weight` on large charged frames to reach `target_share`.

    Share is computed in units of (weight x atom count), which is what the force loss sums
    over, rather than frame count -- a 159-atom frame already contributes twice the terms of
    a 79-atom one, and ignoring that would overshoot by 2x.

    Returns (factor, realised_share).
    """
    small_mass = large_mass = 0.0
    large = []
    for d in dataset:
        if not _is_charged(d):
            continue
        w = float(getattr(d, "forces_weight", 1.0))
        m = w * _n_atoms(d)
        if _n_atoms(d) >= size_threshold:
            large.append(d)
            large_mass += m
        else:
            small_mass += m

    if not large or large_mass <= 0:
        logging.warning("Two-size upweight: no large charged frames found; nothing applied")
        return 1.0, 0.0

    # Solve f * L / (f * L + S) = target  ->  f = target * S / ((1 - target) * L)
    target = float(np.clip(target_share, 1e-6, 0.95))
    factor = target * small_mass / max((1.0 - target) * large_mass, 1e-30)
    for d in large:
        d.forces_weight = d.forces_weight * factor

    realised = (factor * large_mass) / (factor * large_mass + small_mass)
    logging.info(
        f"Two-size upweight: {len(large)} charged frames at >= {size_threshold} atoms "
        f"scaled by {factor:.1f}x -> {realised:.1%} of the charged force loss "
        f"(target {target:.0%}). Cell size is not a defect label.")
    return float(factor), float(realised)


def apply_neutral_size_upweight(dataset: Sequence, target_share: float = 0.25,
                                size_threshold: int = 100) -> Tuple[float, float]:
    """The same treatment for the NEUTRAL frames at the large size.

    WHY THEY MATTER SEPARATELY. The seventeen neutral 159-atom cells are the base branch's
    only direct constraint at large d: the neutral training set is otherwise 79- and 80-atom,
    its d distribution stops around 6.0 A, and above that the base extrapolates. That
    extrapolation is not hypothetical -- it is the measured +0.132 eV/A slope in the 79-atom
    carrier-free residual, and it is why the small-cell charged labels carry a +0.36 artefact
    that points opposite to the physics.

    In the joint run the base is no longer frozen, so this is the one lever that lets it
    LEARN the long-d region instead of extrapolating into it. Leaving these seventeen at
    natural weight while upweighting the charged seventeen would ask the correction to absorb
    a base error the base was never given the chance to fix -- which is exactly the leakage
    the adoption rule tests for.

    Charged and neutral shares are computed within their own populations, so the two
    upweights do not compete for one budget.
    """
    small_mass = large_mass = 0.0
    large = []
    for d in dataset:
        if _is_charged(d):
            continue
        w = float(getattr(d, "forces_weight", 1.0))
        m = w * _n_atoms(d)
        if _n_atoms(d) >= size_threshold:
            large.append(d)
            large_mass += m
        else:
            small_mass += m

    if not large or large_mass <= 0:
        logging.warning(
            "Neutral two-size upweight: NO large neutral frames in the training set. The "
            "base has no direct constraint at large d and will extrapolate there; the "
            "leakage detector in the adoption rule is the thing to watch.")
        return 1.0, 0.0

    target = float(np.clip(target_share, 1e-6, 0.95))
    factor = target * small_mass / max((1.0 - target) * large_mass, 1e-30)
    for d in large:
        d.forces_weight = d.forces_weight * factor

    realised = (factor * large_mass) / (factor * large_mass + small_mass)
    logging.info(
        f"Neutral two-size upweight: {len(large)} neutral frames at >= {size_threshold} "
        f"atoms scaled by {factor:.1f}x -> {realised:.1%} of the neutral force loss "
        f"(target {target:.0%}).")
    return float(factor), float(realised)
