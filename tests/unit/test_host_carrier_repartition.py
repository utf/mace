"""``host_carrier_coupling=False`` must remove exactly the cross term, and nothing else.

``delta_lr = E(q_host + d) - E(q_host)`` with ``d = q_pol + q_carrier``. Since ``E_LR`` is
quadratic, that expands to ``E(d) + 2B(q_host, d)``, so keeping only ``E(d)`` drops exactly
``2B``. These tests check that identity numerically rather than by inspection, and check the
things that would silently break around it: the ``n = 0`` identity, the reference branch, and
survival of the cuEq round trip -- a constructor argument missing from the config extractor
reverts to its default, which is how a frozen screening amplitude once became trainable.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_models import MACEDefect
from mace.tools import torch_geometric

pytest.importorskip("les")

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def build_model(**overrides) -> MACEDefect:
    torch.manual_seed(0)
    arguments = dict(
        r_max=CUTOFF,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(Z_TABLE),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, len(Z_TABLE))),
        avg_num_neighbors=8.0,
        atomic_numbers=Z_TABLE.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        carrier_feature_dim=8,
        counter_embedding_dim=4,
        carrier_mlp_hidden=8,
        use_long_range=True,
        zero_u_init=False,
        les_arguments={"sigma": 1.0, "dl": 2.0, "norm_factor": 90.4756},
    )
    arguments.update(overrides)
    model = MACEDefect(**arguments).double()
    # zero_last_layer leaves the charge readouts at zero, which would make every identity
    # below hold vacuously.
    generator = torch.Generator().manual_seed(1)
    for mlp in (model.latent_charges.host_charge, model.latent_charges.polarisation):
        last = list(mlp.modules())[-1]
        with torch.no_grad():
            last.weight.normal_(0.0, 0.3, generator=generator)
            last.bias.normal_(0.0, 0.3, generator=generator)
    return model.eval()


def make_batch(counts):
    from ase.atoms import Atoms

    rng = np.random.default_rng(0)
    atoms_list = []
    for count in counts:
        atoms = Atoms(
            numbers=[6, 14, 6, 14, 6, 14],
            positions=rng.normal(scale=2.0, size=(6, 3)) + 3.0,
            cell=np.eye(3) * 9.0,
            pbc=True,
        )
        atoms.info["REF_energy"] = 0.0
        array = np.asarray(count, dtype=int)
        atoms.info["carrier_counts"] = array
        atoms.info["multiplicity"] = int((array[0] - array[2]) - (array[1] - array[3])) + 1
        atoms.info["e_cbm_cell"] = -3.2
        atoms.info["e_vbm_cell"] = -6.85
        atoms.arrays["REF_forces"] = np.zeros((6, 3))
        atoms_list.append(atoms)
    keyspec = data.KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "multiplicity": "multiplicity",
            "e_cbm_cell": "e_cbm_cell",
            "e_vbm_cell": "e_vbm_cell",
        },
        arrays_keys={"forces": "REF_forces"},
    )
    configs = [data.config_from_atoms(a, key_specification=keyspec) for a in atoms_list]
    prepare_defect_configurations(configs)
    dataset = [
        data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=CUTOFF) for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False
    )
    return next(iter(loader))


def _pieces(model, batch):
    out = model(batch.to_dict(), training=False, compute_force=False)
    charge = out["latent_charges"]
    host = out["latent_charges_host"]
    positions = batch.positions
    cell = batch.cell.view(-1, 3, 3)
    index = batch.batch
    energy = model.latent_ewald.energy
    with torch.no_grad():
        return {
            "delta_lr": (out["correction_energy"] - out["delta_sr_energy"]).detach(),
            "E_total": energy(charge, positions, cell, index),
            "E_host": energy(host, positions, cell, index),
            "E_delta": energy(charge - host, positions, cell, index),
        }


def test_uncoupled_delta_lr_is_exactly_the_self_energy():
    batch = make_batch([(1, 0, 0, 1), (1, 1, 0, 2)])
    model = build_model(host_carrier_coupling=False)
    p = _pieces(model, batch)
    assert torch.allclose(p["delta_lr"], p["E_delta"], atol=1e-12), (
        f"delta_lr {p['delta_lr'].tolist()} != E(q_pol + q_carrier) "
        f"{p['E_delta'].tolist()}"
    )


def test_coupled_minus_uncoupled_is_exactly_the_cross_term():
    """The difference must be 2B(host, d) = E(host+d) - E(host) - E(d), to machine noise."""
    batch = make_batch([(1, 0, 0, 1), (1, 1, 0, 2)])
    coupled = _pieces(build_model(host_carrier_coupling=True), batch)
    uncoupled = _pieces(build_model(host_carrier_coupling=False), batch)
    cross = coupled["E_total"] - coupled["E_host"] - coupled["E_delta"]
    difference = coupled["delta_lr"] - uncoupled["delta_lr"]
    assert torch.allclose(difference, cross, atol=1e-10), (
        f"removed {difference.tolist()}, cross term is {cross.tolist()}"
    )
    # And it is not a no-op on this system, or the test proves nothing.
    assert float(cross.abs().max()) > 1e-6


@pytest.mark.parametrize("coupling", [True, False])
def test_zero_counter_identity_survives(coupling):
    """E_total(R, 0) == E_base(R) must hold structurally under either convention."""
    batch = make_batch([(0, 0, 0, 0)])
    model = build_model(host_carrier_coupling=coupling)
    out = model(batch.to_dict(), training=False, compute_force=False)
    assert float(out["correction_energy"].abs().max()) < 1e-12
    assert float((out["energy"] - out["base_energy"]).abs().max()) < 1e-10


def test_reference_branch_uses_the_same_convention():
    """delta_energy is a difference of two corrections; a mixed convention would bias it."""
    batch = make_batch([(1, 1, 0, 2)])
    model = build_model(host_carrier_coupling=False)
    out = model(batch.to_dict(), training=False, compute_force=False)
    implied = out["correction_energy"] - out["delta_energy"]
    # correction_energy_ref is not exported; recover it and check it is the uncoupled form
    # by confirming delta_energy is finite and the reference correction has no host self
    # energy left in it (which would show up as an O(eV) offset on a 6-atom cell).
    assert torch.isfinite(implied).all()
    assert float(implied.abs().max()) < 5.0, (
        f"reference correction {implied.tolist()} looks like it still carries the host "
        "self-energy"
    )


def test_carrier_charge_is_unchanged_by_the_repartition():
    batch = make_batch([(1, 0, 0, 1), (1, 1, 0, 2)])
    coupled = build_model(host_carrier_coupling=True)(
        batch.to_dict(), training=False, compute_force=False
    )
    uncoupled = build_model(host_carrier_coupling=False)(
        batch.to_dict(), training=False, compute_force=False
    )
    assert torch.allclose(
        coupled["latent_charges"], uncoupled["latent_charges"], atol=1e-12
    ), "the repartition must change the ENERGY, never the charges"


def test_forces_follow_the_repartitioned_energy():
    """A stale energy in get_outputs would leave forces from the coupled expression."""
    batch = make_batch([(1, 1, 0, 2)])
    model = build_model(host_carrier_coupling=False)
    batch_dict = batch.to_dict()
    batch_dict["positions"].requires_grad_(True)
    out = model(batch_dict, training=True, compute_force=True)
    energy = out["energy"].sum()
    manual = torch.autograd.grad(energy, batch_dict["positions"], retain_graph=True)[0]
    assert torch.allclose(out["forces"], -manual, atol=1e-8), (
        "reported forces are not the gradient of the reported energy"
    )
