"""The field provider on the head's graph equals MACE-Field on its own graph."""
import os
import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data as mace_data, tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.field import load_field_provider
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
CKPT = "/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model"
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="MACE-Field checkpoint not present")


def _atoms(seed=0):
    a0 = 5.6
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
                  cell=np.eye(3) * a0, pbc=True).repeat((2, 2, 1))
    atoms.positions += np.random.default_rng(seed).normal(scale=0.05, size=atoms.positions.shape)
    return atoms


def test_provider_dipoles_match_direct_model():
    prov = load_field_provider(CKPT)
    atoms = _atoms()
    # the head's graph (r_cut 7 > provider r_max 6), three-element table
    atoms.info["carrier_counts"] = np.zeros(4)
    z3 = tools.AtomicNumberTable([17, 55, 82])
    ds = dd.atomic_data([atoms], z3, 7.0)
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to_dict()
    numbers = torch.tensor([[17, 55, 82][i] for i in b["node_attrs"].argmax(-1).tolist()])
    pos = b["positions"].clone().requires_grad_(True)
    p = prov.node_dipoles(pos, b["cell"], b["batch"], b["ptr"], b["edge_index"], b["shifts"], b["unit_shifts"], numbers)
    # direct: MACE-Field on its own graph
    keyspec = mace_data.KeySpecification.from_defaults()
    z79 = tools.AtomicNumberTable([int(z) for z in prov.model.atomic_numbers])
    cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name="mp-dielectric")
    ad = mace_data.AtomicData.from_config(cfg, z_table=z79, cutoff=6.0, heads=list(prov.model.heads))
    bd = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to_dict()
    n = len(atoms)
    e_node = torch.zeros(n, 3, requires_grad=True)
    out = prov.model(bd, training=True, compute_force=False, electric_field_node=e_node, compute_node_dipoles=True,
                     compute_polarization=True)
    assert torch.allclose(p, out["node_dipoles"], atol=1e-9), (p - out["node_dipoles"]).abs().max()
    vol = abs(np.linalg.det(atoms.cell.array))
    assert torch.allclose(p.sum(0), out["polarization"][0] * vol, atol=1e-9)
    # attached to the positions: dp/dR is available
    (g,) = torch.autograd.grad(p.sum(), pos)
    assert float(g.abs().max()) > 0
    # the Born pattern of CsPbCl3: sign of the on-site charge from the dipole response is not
    # what p encodes; p itself is O(0.1 e A) after a 0.05 A rattle
    assert float(p.abs().max()) < 5.0
