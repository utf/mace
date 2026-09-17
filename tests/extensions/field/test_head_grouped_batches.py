"""Head-homogeneous batches and per-batch field-derivative skipping (Plan P A5)."""
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from mace import data
from mace.modules.loss import UniversalFieldLoss
from mace.tools import torch_geometric
from mace.tools.head_batch_sampler import HeadGroupedBatchSampler, concat_group_ids
from mace.tools.train import batch_field_flags, take_step

import importlib.util, os
_spec = importlib.util.spec_from_file_location('_mf_helpers', os.path.join(os.path.dirname(__file__), 'test_macefield_node_field.py'))
_mf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mf)
_cell, _small_model = _mf._cell, _mf._small_model


def test_sampler_never_mixes_groups_covers_all_and_reshuffles():
    ids = [0] * 23 + [1] * 7
    s = HeadGroupedBatchSampler(ids, batch_size=4, drop_last=False, seed=3)
    epoch1 = list(s)
    epoch2 = list(s)
    assert len(epoch1) == len(s) == 6 + 2
    for b in epoch1:
        assert len({ids[i] for i in b}) == 1
    assert sorted(i for b in epoch1 for i in b) == list(range(30))
    assert epoch1 != epoch2  # generator advances between iterations
    assert sorted(i for b in epoch2 for i in b) == list(range(30))
    s2 = HeadGroupedBatchSampler(ids, batch_size=4, drop_last=True, seed=3)
    assert len(list(s2)) == len(s2) == 5 + 1
    assert all(len(b) == 4 for b in s2)


def _atomic_data(model, table, atoms, head, with_response):
    keyspec = data.KeySpecification.from_defaults()
    keyspec.info_keys.update(energy="REF_energy", polarization="REF_polarization", polarizability="REF_polarizability")
    keyspec.arrays_keys.update(forces="REF_forces", becs="REF_becs")
    atoms.info["REF_energy"] = -1.0
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    if with_response:
        atoms.info["REF_polarization"] = np.zeros(3)
        atoms.info["REF_polarizability"] = np.eye(3)
        atoms.arrays["REF_becs"] = np.zeros((len(atoms), 9))
    cfg = data.config_from_atoms(atoms, key_specification=keyspec, head_name=head)
    return data.AtomicData.from_config(cfg, z_table=table, cutoff=float(model.r_max), heads=list(model.heads))


def test_flags_and_loss_skip_for_unlabelled_batches_and_take_step_runs():
    model, table = _small_model()
    head = model.heads[-1]
    labelled = [_atomic_data(model, table, _cell(seed=i), head, True) for i in range(2)]
    unlabelled = [_atomic_data(model, table, _cell(seed=10 + i), head, False) for i in range(3)]
    ds = ConcatDataset([labelled, unlabelled])
    assert concat_group_ids(ds) == [0, 0, 1, 1, 1]
    loader = torch_geometric.dataloader.DataLoader(
        ds, batch_size=1, shuffle=False, drop_last=False,
        batch_sampler=HeadGroupedBatchSampler(concat_group_ids(ds), batch_size=2, drop_last=False, seed=0),
    )
    out_args = {"forces": True, "virials": False, "stress": True, "polarization": True, "becs": True, "polarizability": True}
    loss_fn = UniversalFieldLoss(energy_weight=1, forces_weight=1, polarization_weight=10, becs_weight=1, polarizability_weight=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    seen = {True: 0, False: 0}
    for batch in loader:
        flags = batch_field_flags(batch, out_args)
        has_labels = bool(torch.any(batch.becs_weight != 0))
        assert all(v == has_labels for v in flags.values())
        seen[has_labels] += 1
        # the loss accepts a prediction without the field keys when the batch has no labels
        if not has_labels:
            out = model(batch.to_dict(), training=True, compute_force=True, compute_stress=True)
            assert "becs" not in out or out["becs"] is None
            l = loss_fn(pred=out, ref=batch)
            assert torch.isfinite(l)
        take_step(model=model, loss_fn=loss_fn, batch=batch, optimizer=opt, ema=None,
                  output_args=out_args, max_grad_norm=None, device=torch.device("cpu"))
    assert seen == {True: 1, False: 2}


def test_flags_true_for_mixed_batch():
    model, table = _small_model()
    head = model.heads[-1]
    ads = [_atomic_data(model, table, _cell(seed=1), head, True), _atomic_data(model, table, _cell(seed=2), head, False)]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ads, batch_size=2)))
    out_args = {"polarization": True, "becs": True, "polarizability": False}
    assert batch_field_flags(batch, out_args) == {"polarization": True, "becs": True, "polarizability": False}
