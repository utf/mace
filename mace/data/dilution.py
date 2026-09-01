"""Tile a charged defective cell with a pristine block, for the dilution constraint.

Test 2 measured from DFT that the axial hub force does not dilute when the cell volume
doubles (R_DFT = 0.95, CI [0.66, 1.34]): the carrier is bound, not band-like. The dilution
constraint turns that measurement into a training signal. Under a 2x1x1 tiling a BAND state
loses about half its amplitude to the added block, while a BOUND state loses none, so

    L_dil = w * (1 - sum_{i in original block} alpha_i)^2

separates the two without anyone saying where the defect is or that it is bound.

Design choices that matter, and why:

* **The added block is a PRISTINE MD frame, not the same frame with the vacancy refilled.**
  A refilled Cl would sit unrelaxed between two Pb that had relaxed apart, so the trunk would
  see a compressed local defect at exactly the place the constraint needs to be bulk.
* **Random block and random axis per batch.** A fixed block or axis makes the interface atoms
  on the original side the same atoms every time, with a learnable outward environment -- the
  head could then build a compact-but-wrong well anchored to that interface and satisfy the
  constraint without binding anything.
* **Interface-matched sampling.** Of K random pristine candidates, take the one whose atoms
  near the two interface planes best match the defective frame's boundary displacements. This
  keeps the randomness while reducing unphysical interface strain, which is the one genuine
  artefact of tiling. Selection is geometric only -- no model is evaluated.

The vacancy assignment is not used anywhere here. "Original block" is defined by construction
order, not by where the defect is.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from ase import Atoms

__all__ = ["tile_with_pristine", "retained_mass", "interface_rms"]


def interface_rms(defective: Atoms, pristine: Atoms, axis: int, width: float = 2.0) -> float:
    """RMS mismatch between the two blocks' atoms near the planes that will meet.

    Cheap proxy for interface strain: compare the fractional-coordinate displacement pattern
    of atoms within `width` of the joining faces. Lower is a smoother join.
    """
    la = float(defective.get_cell()[axis, axis])
    if la <= 0:
        return float("inf")
    da = defective.get_positions()[:, axis] % la
    pa = pristine.get_positions()[:, axis] % la
    near_d = np.concatenate([da[da < width], da[da > la - width]])
    near_p = np.concatenate([pa[pa < width], pa[pa > la - width]])
    n = min(len(near_d), len(near_p))
    if n == 0:
        return float("inf")
    return float(np.sqrt(np.mean((np.sort(near_d)[:n] - np.sort(near_p)[:n]) ** 2)))


def tile_with_pristine(defective: Atoms, pristine_pool: Sequence[Atoms], rng,
                       axis: Optional[int] = None, candidates: int = 8
                       ) -> Tuple[Atoms, np.ndarray, int]:
    """Return (tiled cell, boolean mask of the ORIGINAL block's atoms, axis used).

    The original block occupies the first `len(defective)` entries, so the mask is simply the
    leading slice -- but it is returned explicitly rather than assumed, because the constraint
    reads it and a silent ordering change would invert the meaning of the loss.
    """
    if axis is None:
        axis = int(rng.integers(0, 3))

    idx = rng.choice(len(pristine_pool), size=min(candidates, len(pristine_pool)),
                     replace=False)
    best, best_score = None, float("inf")
    for j in idx:
        cand = pristine_pool[int(j)]
        if not np.allclose(cand.get_cell(), defective.get_cell(), atol=0.5):
            continue                     # different box; not a valid tiling partner
        score = interface_rms(defective, cand, axis)
        if score < best_score:
            best, best_score = cand, score
    if best is None:
        best = pristine_pool[int(idx[0])]

    cell = np.array(defective.get_cell())
    shift = cell[axis].copy()

    pos = np.vstack([defective.get_positions(), best.get_positions() + shift])
    numbers = np.concatenate([defective.get_atomic_numbers(), best.get_atomic_numbers()])
    new_cell = cell.copy()
    new_cell[axis] = cell[axis] * 2.0

    tiled = Atoms(numbers=numbers, positions=pos, cell=new_cell, pbc=defective.pbc)
    # Carry the counters over unchanged: the tiled cell holds the SAME one carrier in twice
    # the volume, which is the whole point.
    for key in ("carrier_counts", "cell_charge", "multiplicity", "m_s_ref_doubled",
                "host", "pair_id", "e_cbm_cell", "e_vbm_cell"):
        if key in defective.info:
            tiled.info[key] = defective.info[key]

    mask = np.zeros(len(tiled), dtype=bool)
    mask[: len(defective)] = True
    return tiled, mask, int(axis)


def retained_mass(alpha: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the carrier's amplitude still inside the original block.

    ~1.0 for a bound state, ~len(original)/len(tiled) for a state spread over the whole cell.
    """
    total = float(alpha.sum())
    if total <= 0:
        return float("nan")
    return float(alpha[mask].sum() / total)
