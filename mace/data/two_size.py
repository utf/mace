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

__all__ = ["apply_two_size_upweight", "apply_neutral_size_upweight",
           "apply_size_upweight", "realised_shares", "apply_energy_weights_from_json",
           "weight_column"]


def _n_atoms(d) -> int:
    return int(d.positions.shape[0])


def _frame_weight(d) -> float:
    """The frame-level `weight` (config_type_weights) every loss term multiplies."""
    w = getattr(d, "weight", None)
    return 1.0 if w is None else float(w)


def weight_column(population: str, channel: str) -> str:
    """The AtomicData column the loss actually reads for this population and channel.

    THE COLUMNS DIFFER BY POPULATION, and getting this wrong is silent. `DefectLoss` scores a
    neutral (n = 0) frame through its BASE terms, which read `base_energy_weight` and
    `base_forces_weight`; it scores a charged frame through its TOTALS terms, which read the
    generic `energy_weight` and `forces_weight`. The generic columns exist on a neutral frame
    too -- and no term reads them. The first neutral upweight scaled those, logged a realised
    25% share of a loss that never saw it, and every unit test passed because each of them
    read the column it had written. Found by printing `base_forces_weight` before and after.
    """
    if population == "neutral":
        return "base_forces_weight" if channel == "forces" else "base_energy_weight"
    if population == "charged":
        return "forces_weight" if channel == "forces" else "energy_weight"
    raise ValueError(f"unknown population {population!r}")


def _mass(d, channel: str, population: str = "charged") -> float:
    """What one frame contributes to a channel's loss normalisation.

    Forces: `weight * <column> * n_atoms` (every atom is a term). Energy: the loss is a
    per-atom MSE with one term per frame, so `weight * <column>` -- atom count does not
    enter. The frame-level `weight` is included in both: pristine frames carry 5.0 in this
    dataset, and a share computed without it is a share of a loss that is not the one being
    minimised. `<column>` is `weight_column(population, channel)`.
    """
    col = float(getattr(d, weight_column(population, channel), 1.0))
    if channel == "forces":
        return _frame_weight(d) * col * _n_atoms(d)
    if channel == "energy":
        return _frame_weight(d) * col
    raise ValueError(f"unknown channel {channel!r}")


def _in_population(d, population: str) -> bool:
    if population == "charged":
        return _is_charged(d)
    if population == "neutral":
        return not _is_charged(d)
    raise ValueError(f"unknown population {population!r}")


def apply_energy_weights_from_json(dataset: Sequence, path, z_table=None,
                                   population: str = "charged"):
    """Multiply `energy_weight` on the frames of `population` by the `w_E` recorded for
    their `frame_key` in the JSON `c1_ood_indicator.py` writes. Returns
    (n_matched, n_in_population, mean_w_applied)."""
    import json

    from mace.modules.defect_cache import frame_key

    payload = json.load(open(path))
    table = payload.get("frames", payload)
    hit = total = 0
    applied = []
    for d in dataset:
        if not _in_population(d, population):
            continue
        total += 1
        numbers = d.node_attrs.argmax(dim=-1).cpu().numpy()
        if z_table is not None:
            numbers = np.array([z_table.zs[int(i)] for i in numbers], dtype=np.int64)
        key = str(frame_key(numbers, d.positions.detach().cpu().numpy(),
                            d.cell.detach().cpu().numpy()))
        entry = table.get(key)
        if entry is None:
            continue
        w = float(entry["w_E"] if isinstance(entry, dict) else entry)
        d.energy_weight = d.energy_weight * w
        applied.append(w)
        hit += 1
    return hit, total, (float(np.mean(applied)) if applied else float("nan"))


def realised_shares(dataset: Sequence, population: str = "neutral",
                    size_threshold: int = 100, channels=("energy", "forces")) -> dict:
    """The large-cell share of each channel's loss mass within `population`, as it stands."""
    out = {}
    for ch in channels:
        large = small = 0.0
        for d in dataset:
            if not _in_population(d, population):
                continue
            m = _mass(d, ch, population)
            if _n_atoms(d) >= size_threshold:
                large += m
            else:
                small += m
        out[ch] = large / (large + small) if (large + small) > 0 else 0.0
    return out


def apply_size_upweight(dataset: Sequence, population: str = "neutral",
                        target_share: float = 0.25, size_threshold: int = 100,
                        channels=("energy", "forces")) -> dict:
    """Section 1 of the Stage A' spec: the SAME target share in the energy loss AND the
    force loss, each solved on its own mass. Returns {channel: (factor, realised_share)}.

    Scales `energy_weight` and `forces_weight` on the large frames of `population`. The two
    channels have different masses (atom count enters the force loss and not the energy
    loss) so the two factors differ; both realised shares are returned and should be logged
    every epoch, because a weight chosen from frame counts alone drifts if the loss weights
    or the mix ever change.
    """
    target = float(np.clip(target_share, 1e-6, 0.95))
    result = {}
    for ch in channels:
        large, small_mass, large_mass = [], 0.0, 0.0
        for d in dataset:
            if not _in_population(d, population):
                continue
            m = _mass(d, ch, population)
            if _n_atoms(d) >= size_threshold:
                large.append(d)
                large_mass += m
            else:
                small_mass += m
        if not large or large_mass <= 0:
            logging.warning("Size upweight (%s, %s): no large frames; nothing applied",
                            population, ch)
            result[ch] = (1.0, 0.0)
            continue
        factor = target * small_mass / max((1.0 - target) * large_mass, 1e-30)
        attr = weight_column(population, ch)
        for d in large:
            setattr(d, attr, getattr(d, attr) * factor)
        realised = (factor * large_mass) / (factor * large_mass + small_mass)
        logging.info(
            f"Size upweight ({population}, {ch}): {len(large)} frames at >= "
            f"{size_threshold} atoms scaled by {factor:.1f}x -> {realised:.1%} of the "
            f"{population} {ch} loss (target {target:.0%}).")
        result[ch] = (float(factor), float(realised))
    return result


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
        w = float(getattr(d, "base_forces_weight", 1.0))
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
        # The BASE column: it is the one the loss reads for an n = 0 frame. Scaling the
        # generic `forces_weight` here, as this function first did, changed nothing.
        d.base_forces_weight = d.base_forces_weight * factor

    realised = (factor * large_mass) / (factor * large_mass + small_mass)
    logging.info(
        f"Neutral two-size upweight: {len(large)} neutral frames at >= {size_threshold} "
        f"atoms scaled by {factor:.1f}x -> {realised:.1%} of the neutral force loss "
        f"(target {target:.0%}).")
    return float(factor), float(realised)
