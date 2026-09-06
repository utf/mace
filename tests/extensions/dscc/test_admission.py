"""Plan section 6: the energy-admission machinery (coverage, out-of-fold s0 +- SE, sQ)."""
import numpy as np
import torch
from ase import Atoms

from mace.modules.dscc import admission as adm


def _vacancy_frame(remove=0):
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
                                              [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * 5.6, pbc=True).repeat((2, 2, 2))
    cl = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 17]
    if remove is not None:
        del atoms[cl[remove]]
    return atoms


def test_collective_coordinate_is_the_flanking_pb_distance():
    atoms = _vacancy_frame()
    d = adm.collective_coordinate(torch.tensor(atoms.get_positions()), torch.tensor(np.array(atoms.get_cell())),
                                  atoms.get_atomic_numbers())
    assert abs(d - 5.6) < 1e-9                        # Pb-Cl-Pb straight through the vacancy
    pristine = _vacancy_frame(remove=None)
    assert adm.collective_coordinate(torch.tensor(pristine.get_positions()), torch.tensor(np.array(pristine.get_cell())),
                                     pristine.get_atomic_numbers()) is None


def test_slope_with_se_and_admission_decisions():
    rng = np.random.default_rng(0)
    x = np.linspace(4.8, 6.8, 60)
    s, se, a = adm.slope_with_se(x, 0.5 + 0.01 * x + rng.normal(0, 1e-3, x.size))
    assert abs(s - 0.01) < 5 * se and se < 1e-3
    cfg = adm.AdmissionConfig(bin_width=0.5, n_min=2, s_tol=0.05, z=2.0)
    charged_d = {1: x.tolist(), 2: x.tolist()}
    charged_r = {1: (0.02 * x).tolist(), 2: (0.02 * x).tolist()}
    # Size 1: neutral coverage stops at 6.0 A -> the last bins are empty -> not admitted.
    n1 = x[x < 6.0]
    # Size 2: full coverage and a flat null -> admitted.
    n2 = x
    table = adm.admission_table(charged_d, charged_r, {1: n1.tolist(), 2: n2.tolist()},
                                {1: (1e-3 * rng.normal(size=n1.size)).tolist(), 2: (1e-3 * rng.normal(size=n2.size)).tolist()}, cfg)
    assert not table[1].admitted and not table[1].coverage and "coverage" in table[1].reason
    assert table[2].admitted and table[2].coverage and abs(table[2].s0) < 0.01
    assert abs(table[2].sQ - 0.02) < 1e-9
    # A sloped null fails the s_tol test even with coverage.
    table3 = adm.admission_table({2: x.tolist()}, {2: (0.02 * x).tolist()}, {2: x.tolist()}, {2: (0.1 * x).tolist()}, cfg)
    assert not table3[2].admitted and "s_tol" in table3[2].reason
    rec = adm.table_record(table, cfg, fold=0)
    assert rec["fold"] == 0 and set(rec["sizes"]) == {"1", "2"}
