"""MACE-Field port and the Plan P per-node field (registered tests, not experiments).

1. A per-node field equal to the broadcast global field reproduces the global-field path's
   energy, forces, P, Z*, alpha bit-for-bit (same tensor products, same graph).
2. `sum_a node_dipoles[a] = Omega P` from the global-field derivative.
3. e3nn -> cueq -> e3nn round trip reproduces P, Z*, alpha to 1e-8 (GPU only).
"""
import os
import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules
from mace.modules.extensions import MACEField
from mace.tools import torch_geometric, utils

torch.set_default_dtype(torch.float64)
CKPT = "/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model"


def _small_model(seed=0):
    torch.manual_seed(seed)
    table = utils.AtomicNumberTable([17, 55, 82])
    cfg = dict(
        r_max=5.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu,
        atomic_energies=np.array([-1.0, -2.0, -3.0]), avg_num_neighbors=8.0,
        atomic_numbers=table.zs, correlation=2, radial_type="bessel",
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
    )
    model = MACEField(**cfg)
    # the field modules start at zero in a fresh model? no -- random init; make them O(1)
    return model, table


def _cell(rattle=0.05, seed=1, rep=(1, 1, 1)):
    a0 = 5.6
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
                  cell=np.eye(3) * a0, pbc=True).repeat(rep)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(scale=rattle, size=atoms.positions.shape)
    return atoms


def _batch(model, table, atoms_list, field=None):
    keyspec = data.KeySpecification.from_defaults()
    ads = []
    for a in atoms_list:
        if field is not None:
            a.info["REF_electric_field"] = list(field)
        cfg = data.config_from_atoms(a, key_specification=keyspec, head_name=model.heads[-1])
        ads.append(data.AtomicData.from_config(cfg, z_table=table, cutoff=float(model.r_max), heads=list(model.heads)))
    return next(iter(torch_geometric.dataloader.DataLoader(ads, batch_size=len(ads)))).to_dict()


def _run(model, batch, **kw):
    return model(batch, training=True, compute_force=True, compute_stress=False,
                 compute_polarization=True, compute_becs=True, compute_polarizability=True, **kw)


@pytest.mark.parametrize("field", [(0.0, 0.0, 0.0), (0.02, -0.01, 0.03)])
def test_node_field_reproduces_global_field_bit_for_bit(field):
    model, table = _small_model()
    atoms = [_cell(seed=1), _cell(seed=2, rep=(1, 1, 2))]
    batch = _batch(model, table, atoms, field=field)
    ref = _run(model, dict(batch))
    n = batch["positions"].shape[0]
    e_node = torch.tensor(field, dtype=torch.float64).expand(n, 3).clone().requires_grad_(True)
    out = _run(model, dict(batch), electric_field_node=e_node, compute_node_dipoles=True)
    for key in ("energy", "forces", "polarization", "becs", "polarizability"):
        assert torch.equal(ref[key], out[key]), key
    # sum of node dipoles = Omega P (float64: exact up to summation order)
    vol = torch.linalg.det(batch["cell"].view(-1, 3, 3)).abs()
    omega_p = ref["polarization"] * vol.view(-1, 1)
    summed = torch.zeros_like(omega_p).index_add_(0, batch["batch"], out["node_dipoles"])
    assert torch.allclose(summed, omega_p, atol=1e-12, rtol=0)
    assert float(out["node_dipoles"].abs().max()) > 0


def test_node_dipoles_are_local_derivatives():
    """d E / d E_node[a] by finite differences on one atom."""
    model, table = _small_model()
    batch = _batch(model, table, [_cell(seed=3)])
    n = batch["positions"].shape[0]
    e_node = torch.zeros(n, 3, dtype=torch.float64, requires_grad=True)
    out = _run(model, dict(batch), electric_field_node=e_node, compute_node_dipoles=True)
    p = out["node_dipoles"].detach()
    h = 1e-5
    for a, c in [(0, 2), (3, 0)]:
        ep = torch.zeros(n, 3, dtype=torch.float64); ep[a, c] = h
        em = -ep
        e1 = model(dict(batch), training=False, compute_force=False, electric_field_node=ep)["energy"]
        e2 = model(dict(batch), training=False, compute_force=False, electric_field_node=em)["energy"]
        fd = -(e1 - e2).item() / (2 * h)
        assert abs(fd - p[a, c].item()) < 1e-6 * max(1.0, abs(fd)), (a, c, fd, p[a, c].item())


@pytest.mark.skipif(not os.path.exists(CKPT), reason="MACE-Field checkpoint not present")
def test_checkpoint_loads_and_node_field_matches():
    model = torch.load(CKPT, map_location="cpu", weights_only=False).double()
    table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    batch = _batch(model, table, [_cell(seed=5)], field=(0.01, 0.0, 0.0))
    ref = _run(model, dict(batch))
    n = batch["positions"].shape[0]
    e_node = torch.tensor([0.01, 0.0, 0.0]).expand(n, 3).clone().requires_grad_(True)
    out = _run(model, dict(batch), electric_field_node=e_node)
    for key in ("energy", "forces", "polarization", "becs", "polarizability"):
        assert torch.equal(ref[key], out[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cueq round trip needs a GPU")
def test_cueq_round_trip_reproduces_response():
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    from mace.cli.convert_cueq_e3nn import run as to_e3nn
    model, table = _small_model()
    batch = _batch(model, table, [_cell(seed=7)], field=(0.0, 0.01, 0.0))
    ref = _run(model, dict(batch))
    cu = to_cueq(model, device="cuda")
    batch_c = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out_c = _run(cu, batch_c)
    back = to_e3nn(cu, device="cpu")
    out_b = _run(back, dict(batch))
    for key in ("energy", "forces", "polarization", "becs", "polarizability"):
        assert torch.allclose(ref[key], out_c[key].cpu(), atol=1e-8, rtol=0), ("cueq", key, (ref[key] - out_c[key].cpu()).abs().max())
        assert torch.allclose(ref[key], out_b[key], atol=1e-8, rtol=0), ("back", key)
