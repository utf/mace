"""Plan section 6 (v4.2): the force path of the objective -- weights, units, reductions --
and the fold construction; the loss-path audit reduced to what the loss contains."""
import numpy as np
import pytest
import torch

from mace.modules.dscc import data as dd
from mace.modules.dscc import train as tr


def test_force_loss_is_the_stratum_weighted_per_atom_mean():
    ptr = torch.tensor([0, 3, 5])
    batch = {"ptr": ptr, "batch": torch.tensor([0, 0, 0, 1, 1]),
             "forces": torch.zeros(5, 3, dtype=torch.float64)}
    forces = torch.zeros(5, 3, dtype=torch.float64)
    forces[0, 0] = 1.0                         # graph 0: one atom off by 1 eV/A -> mean 1/3
    forces[3, 1] = 2.0                         # graph 1: one atom off by 2 -> mean 4/2 = 2
    out = {"forces": forces}
    loss = tr.force_loss(out, batch, torch.tensor([0.5, 2.0], dtype=torch.float64))
    assert float(loss) == pytest.approx(0.5 * (1.0 / 3) + 2.0 * 2.0)
    # An injected constant force error scales the loss quadratically (units: (eV/A)^2).
    out2 = {"forces": forces * 2}
    assert float(tr.force_loss(out2, batch, torch.tensor([0.5, 2.0], dtype=torch.float64))) == pytest.approx(4 * float(loss))


def test_frame_weights_normalise_within_frozen_stratum_totals(tmp_path):
    from ase import Atoms
    frames, metas = [], []
    for i in range(6):
        a = Atoms("CsPbCl3", positions=np.random.default_rng(i).random((5, 3)) * 5, cell=np.eye(3) * 5.6, pbc=True)
        a.info.update({"carrier_counts": np.array([0, 0, 1, 0]), "cell_charge": 1, "config_type": "t",
                       "source_dir": "A" if i < 4 else "B"})
        frames.append(a); metas.append(dd.frame_meta(i, a, "CsPbCl3", 5))
    fold_of = {i: (0 if i == 5 else 1) for i in range(6)}
    cfg = tr.TrainConfig(fold=0, stratum_weights={tr.stratum_id(metas[0]): 3.0}, device="cpu")

    class _M:                                  # a stand-in with what Trainer.__init__ reads
        atomic_numbers = [17, 55, 82]
    trainer = tr.Trainer(_M(), cfg, frames, metas, fold_of, str(tmp_path))
    assert sorted(trainer.train_idx) == [0, 1, 2, 3, 4] and trainer.held_idx == [5]
    w_a = [trainer.frame_weight[i] for i in range(4)]
    assert all(abs(w - 3.0 / 4) < 1e-12 for w in w_a)       # stratum A: total 3 over 4 frames
    assert trainer.frame_weight[4] == pytest.approx(1.0)      # stratum B: default total 1, one frame


def test_vacancy_centre_takes_the_empty_side_of_a_two_image_pair():
    """Two Pb 5.3 A apart along an 11.1 A axis share two sites: a bridging Cl on the short
    path and the vacancy on the long one. The minimum-image midpoint would sit on the Cl."""
    cell = torch.diag(torch.tensor([16.0, 16.0, 11.1], dtype=torch.float64))
    pos, numbers = [], []
    for z0 in (0.0, 5.3):                                       # the flanking pair along z
        pos.append([8.0, 8.0, z0]); numbers.append(82)
        for dx, dy in ((2.8, 0.0), (-2.8, 0.0), (0.0, 2.8), (0.0, -2.8)):      # four equatorial Cl each
            pos.append([8.0 + dx, 8.0 + dy, z0]); numbers.append(17)
    pos.append([8.0, 8.0, 2.65]); numbers.append(17)            # the occupied bridge on the short path
    pos.append([4.0, 4.0, 8.2]); numbers.append(55)             # a Cs off to the side
    pos = torch.tensor(pos, dtype=torch.float64)
    rad = tr.vacancy_centre(pos, cell, numbers)
    assert rad is not None
    # the vacancy site is at z = 8.2 (the long path), 2.9 A from each flanking Pb
    assert rad[0] == pytest.approx(2.9, abs=1e-6) and rad[5] == pytest.approx(2.9, abs=1e-6)
    assert rad[10] == pytest.approx(11.1 - 8.2 + 2.65, abs=1e-6)  # the bridging Cl is 5.55 A away, not 0
    # a centred pair (no second image path) reproduces the minimum-image midpoint
    cell2 = torch.diag(torch.tensor([16.0, 16.0, 22.2], dtype=torch.float64))
    rad2 = tr.vacancy_centre(pos, cell2, numbers)
    assert rad2[10] == pytest.approx(0.0, abs=1e-6)             # now the only shared site is the bridge: it is the "vacancy"
