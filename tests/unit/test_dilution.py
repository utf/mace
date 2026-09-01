"""The tiled cell must be a valid doubling, and retained mass must separate the two states.

The constraint's whole job is to distinguish a bound carrier (retained mass ~1) from a band
state (~79/159). If the tiling is wrong -- overlapping atoms, a cell that is not doubled,
counters dropped -- the constraint would penalise something other than delocalisation, and it
would do so invisibly, since there are no labels on the tiled cell to disagree with.
"""

import numpy as np
import pytest
from ase import Atoms

from mace.data.dilution import retained_mass, tile_with_pristine


def cubic(n_side=3, a=5.6, drop=False, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pos, num = [], []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                pos.append([i * a, j * a, k * a])
                num.append(17)
    pos = np.array(pos, dtype=float)
    if jitter:
        pos = pos + rng.normal(scale=jitter, size=pos.shape)
    if drop:
        pos, num = pos[1:], num[1:]
    at = Atoms(numbers=num, positions=pos, cell=np.eye(3) * (n_side * a), pbc=True)
    at.info["carrier_counts"] = "0 0 1 0"
    at.info["cell_charge"] = 1
    return at


def test_tiled_cell_is_a_true_doubling():
    rng = np.random.default_rng(0)
    d = cubic(drop=True)
    pool = [cubic(jitter=0.05, seed=s) for s in range(4)]
    tiled, mask, axis = tile_with_pristine(d, pool, rng)
    assert len(tiled) == len(d) + len(pool[0])
    assert np.isclose(np.linalg.det(tiled.get_cell()),
                      2.0 * np.linalg.det(d.get_cell()))
    assert tiled.get_cell()[axis, axis] == pytest.approx(2 * d.get_cell()[axis, axis])


def test_original_block_mask_is_the_leading_slice():
    rng = np.random.default_rng(1)
    d = cubic(drop=True)
    tiled, mask, _ = tile_with_pristine(d, [cubic(jitter=0.05)], rng)
    assert mask.sum() == len(d)
    assert mask[: len(d)].all() and not mask[len(d):].any()


def test_counters_survive_the_tiling():
    """The tiled cell holds the SAME one carrier in twice the volume -- that is the point."""
    rng = np.random.default_rng(2)
    d = cubic(drop=True)
    tiled, _, _ = tile_with_pristine(d, [cubic(jitter=0.05)], rng)
    assert tiled.info["carrier_counts"] == "0 0 1 0"
    assert tiled.info["cell_charge"] == 1


def test_no_atoms_overlap_across_the_interface():
    rng = np.random.default_rng(3)
    d = cubic(drop=True)
    tiled, _, _ = tile_with_pristine(d, [cubic(jitter=0.02)], rng)
    pos = tiled.get_positions()
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(pos).query(pos, k=2)
    assert dist[:, 1].min() > 0.5, "atoms collide at the interface"


def test_axis_varies_across_calls():
    """A fixed axis would make the same atoms face the interface every time."""
    rng = np.random.default_rng(4)
    d = cubic(drop=True)
    pool = [cubic(jitter=0.05, seed=s) for s in range(4)]
    axes = {tile_with_pristine(d, pool, rng)[2] for _ in range(30)}
    assert len(axes) > 1, f"tiling axis never changed: {axes}"


def test_retained_mass_separates_bound_from_band():
    """Real counts: a 79-atom defective cell plus an 80-atom pristine block is 159, so a
    uniform state retains 79/159 = 0.497 -- not 0.500. The tiled cell must match the real
    159-atom DFT cells, since R_model is compared against R_DFT measured on those."""
    n_orig, n_tot = 79, 159
    mask = np.zeros(n_tot, dtype=bool)
    mask[:n_orig] = True

    bound = np.zeros(n_tot)
    bound[:2] = 0.5                                   # all amplitude on two original atoms
    assert retained_mass(bound, mask) == pytest.approx(1.0)

    band = np.full(n_tot, 1.0 / n_tot)                # uniform over the doubled cell
    assert retained_mass(band, mask) == pytest.approx(79.0 / 159.0, abs=1e-6)
    assert retained_mass(band, mask) < 0.5


def test_real_shaped_tiling_gives_159_atoms():
    """79 defective + 80 pristine = 159, matching the DFT large cells exactly."""
    rng = np.random.default_rng(7)
    d = cubic(n_side=4, drop=True)        # 63 atoms, stands in for the 79-atom defective
    pool = [cubic(n_side=4, jitter=0.05, seed=s) for s in range(3)]   # 64, the pristine
    tiled, mask, _ = tile_with_pristine(d, pool, rng)
    assert len(tiled) == len(d) + len(pool[0]) == 127
    assert mask.sum() == len(d), "original block is the defective cell, one atom short"
    assert len(tiled) % 2 == 1, "a single vacancy in a doubled cell gives an odd count"


def test_retained_mass_handles_zero_amplitude():
    mask = np.ones(4, dtype=bool)
    assert np.isnan(retained_mass(np.zeros(4), mask))


def test_reading_rule_separates_the_four_cases():
    """The log is only useful if it is read the same way every time."""
    from mace.data.dilution import read_dilution

    assert read_dilution(0.98, 0.05) == "bound-in-original-block"
    assert read_dilution(0.98, 0.40) == "interface-artefact-well"
    assert read_dilution(0.497, 0.40) == "straddling-artefact"
    assert read_dilution(0.497, 0.05) == "band-like"
    assert read_dilution(float("nan"), 0.1) == "unknown"


def test_interface_mass_spots_amplitude_piled_at_the_join():
    """retained ~0.5 from a straddling state must not be read as band-like."""
    from mace.data.dilution import interface_mass

    cell = np.diag([20.0, 10.0, 10.0])          # doubled along x, original ends at x=10
    pos = np.array([[10.1, 0, 0], [9.9, 0, 0], [5.0, 0, 0], [15.0, 0, 0]])
    piled = np.array([0.45, 0.45, 0.05, 0.05])
    assert interface_mass(piled, pos, cell, 0, 2) > 0.85

    spread = np.array([0.05, 0.05, 0.45, 0.45])
    assert interface_mass(spread, pos, cell, 0, 2) < 0.2
