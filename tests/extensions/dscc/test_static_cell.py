"""D8: the static pristine cell from thermal frames -- the index-median with a trapped
distortion removed by symmetrising over the generated group."""
import numpy as np
import pytest
from ase import Atoms

from mace.modules.dscc import static_cell as sc


def _perovskite(rattle, seed, delta=0.0):
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
                                              [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * 5.6, pbc=True).repeat((2, 2, 2))
    if delta:
        frac = atoms.get_scaled_positions()
        cl = atoms.get_atomic_numbers() == 17
        a_cell = (frac[:, 0] < 0.5) & (frac[:, 1] < 0.5) & (frac[:, 2] < 0.5)
        b_cell = (frac[:, 0] >= 0.5) & (frac[:, 1] < 0.5) & (frac[:, 2] < 0.5)
        atoms.positions[cl & a_cell, 2] += delta
        atoms.positions[cl & b_cell, 2] -= delta
    atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def test_median_and_group_recover_the_ideal_lattice():
    frames = [_perovskite(0.05, s, delta=0.12) for s in range(1, 13)]
    frames[3].positions += np.array([2.0, -1.0, 0.5])      # one frame rigidly shifted then wrapped
    frames[3].wrap()
    out = sc.static_pristine_cell(frames, log=False)
    c = out["construction"]
    assert c["n_symmetries"] == 384 and c["symmetry_rounds"] <= 3 and c["cell_snapped_orthogonal"]
    assert 0.06 < c["symmetrisation_shift"] < 0.16           # the +-0.12 A sub-cell distortion removed (orbit-averaged)
    pos, cell, numbers = np.array(out["positions"]), np.array(out["cell"]), np.array(out["numbers"])
    pb, cl = pos[numbers == 82], pos[numbers == 17]
    d = pb[:, None, :] - cl[None, :, :]
    d -= np.round(d @ np.linalg.inv(cell)) @ cell
    nearest = np.sort(np.linalg.norm(d, axis=-1), axis=1)[:, :6]
    assert np.abs(nearest - 2.8).max() < 0.02
    # Atom order and origin are irrelevant: permuted, shifted frames give the same cell.
    rng = np.random.default_rng(5)
    scrambled = [f[rng.permutation(len(f))] for f in frames]
    for f in scrambled[::3]:
        f.positions += rng.normal(size=3) * 3.0
        f.wrap()
    out2 = sc.static_pristine_cell(scrambled, log=False)
    assert out2["construction"]["n_symmetries"] == 384
    pos2 = np.array(out2["positions"]); numbers2 = np.array(out2["numbers"])
    pb2 = pos2[numbers2 == 82]
    # Same Pb sublattice up to a rigid shift and a permutation: compare pair-distance sets.
    def pair_set(p):
        d = p[:, None, :] - p[None, :, :]; d -= np.round(d @ np.linalg.inv(cell)) @ cell
        return np.sort(np.linalg.norm(d, axis=-1).reshape(-1))
    assert np.abs(pair_set(pb2) - pair_set(pb)).max() < 1e-6
    with pytest.raises(ValueError):
        sc.static_pristine_cell([frames[0], frames[1][:-1]], log=False)
