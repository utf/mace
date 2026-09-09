"""v5 W1.2: element pruning of a foundation base leaves its predictions on kept-element frames unchanged."""
import os
import numpy as np
import pytest
import torch
from ase import Atoms

from mace.modules.dscc.prune import prune_elements
from mace.tools import torch_geometric, AtomicNumberTable
from mace.data import AtomicData, config_from_atoms

MH1 = os.path.expanduser("~/.cache/mace/macemh1model")


def _predict(model, atoms):
    z = AtomicNumberTable([int(x) for x in model.atomic_numbers])
    data = AtomicData.from_config(config_from_atoms(atoms), z_table=z, cutoff=float(model.r_max))
    batch = next(iter(torch_geometric.dataloader.DataLoader([data], batch_size=1))).to_dict()
    out = model(batch, training=False, compute_force=True)
    return float(out["energy"]), out["forces"].detach()


@pytest.mark.skipif(not os.path.exists(MH1), reason="MACE-MH-1 checkpoint not cached")
def test_pruned_mh1_reproduces_predictions_on_kept_elements():
    torch.set_default_dtype(torch.float64)
    from mace.tools.scripts_utils import remove_pt_head
    full = remove_pt_head(torch.load(MH1, map_location="cpu", weights_only=False), "omat_pbe").double().eval()
    pruned = prune_elements(full, [82, 17, 55]).eval()
    assert pruned.atomic_numbers.tolist() == [17, 55, 82]
    rng = np.random.default_rng(0)
    atoms = Atoms("CsPbCl3" * 2, positions=rng.random((10, 3)) * 5.6 + np.array([[0, 0, 0]] * 5 + [[0, 0, 5.6]] * 5), cell=np.diag([5.6, 5.6, 11.2]), pbc=True)
    e0, f0 = _predict(full, atoms); e1, f1 = _predict(pruned, atoms)
    assert abs(e0 - e1) < 1e-10 and float((f0 - f1).abs().max()) < 1e-10
    with pytest.raises(ValueError):
        prune_elements(full, [17, 55, 82, 200])
