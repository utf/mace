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


def test_near_field_categories_split_the_flanking_pb_from_the_first_shell_cl():
    """W0.2 (v5): inside the 2-4 A shell the flanking Pb pair, the Cl within 3.5 A of a
    flanking Pb, and the remainder are separate readings."""
    cell = torch.diag(torch.tensor([16.0, 16.0, 11.1], dtype=torch.float64))
    pos, numbers = [], []
    for z0 in (0.0, 5.3):
        pos.append([8.0, 8.0, z0]); numbers.append(82)
        for dx, dy in ((2.8, 0.0), (-2.8, 0.0), (0.0, 2.8), (0.0, -2.8)):
            pos.append([8.0 + dx, 8.0 + dy, z0]); numbers.append(17)
    pos.append([8.0, 8.0, 2.65]); numbers.append(17)          # the occupied bridge
    pos.append([4.0, 4.0, 8.2]); numbers.append(55)           # a Cs off to the side
    pos = torch.tensor(pos, dtype=torch.float64)
    found = tr.vacancy_centre_full(pos, cell, numbers)
    assert found is not None
    rad, flank = found
    assert sorted(int(i) for i in flank) == [0, 5]            # the two Pb
    cats = tr.near_field_categories(pos, cell, numbers, flank)
    assert bool(cats["pb_flank"][0]) and bool(cats["pb_flank"][5])
    assert int(cats["pb_flank"].sum()) == 2
    # every equatorial Cl is 2.8 A from its Pb, and the bridge 2.65 A: all first shell
    assert int(cats["cl_first"].sum()) == 9
    assert not bool(cats["cl_first"][11]) and bool(cats["other"][11])     # the Cs
    assert int((cats["pb_flank"] & cats["cl_first"]).sum()) == 0          # the categories partition
    assert int((cats["pb_flank"] | cats["cl_first"] | cats["other"]).sum()) == len(numbers)


def test_parameter_average_is_uniform_over_the_window_and_restores_the_last_epoch():
    """W0.4: the evaluation model is the uniform average of the trainable parameters over the
    window; the last-epoch parameters are restored after it is read."""
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    frozen = torch.nn.Parameter(torch.ones(2, dtype=torch.float64), requires_grad=False)
    model.register_parameter("frozen", frozen)
    values = [1.0, 2.0, 6.0]
    avg_sum, n = None, 0
    for v in values:
        with torch.no_grad():
            model.weight.fill_(v); model.bias.fill_(-v); model.frozen.fill_(v)
        avg_sum = tr.average_into(avg_sum, model); n += 1
    assert "frozen" not in avg_sum                               # only the trainable parameters
    saved = tr.load_average(model, avg_sum, n)
    assert float(model.weight[0, 0]) == pytest.approx(3.0)       # (1 + 2 + 6) / 3
    assert float(model.bias[0]) == pytest.approx(-3.0)
    assert float(model.frozen[0]) == pytest.approx(6.0)          # untouched
    tr.restore_parameters(model, saved)
    assert float(model.weight[0, 0]) == pytest.approx(6.0) and float(model.bias[0]) == pytest.approx(-6.0)
