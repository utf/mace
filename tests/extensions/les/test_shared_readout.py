"""D2: one latent-charge readout shared by every head of a multi-head MACELES."""
import numpy as np, pytest, torch
from e3nn import o3
from ase import Atoms
from mace import data as mace_data, modules
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scripts_utils import remove_pt_head

pytest.importorskip("les")
torch.set_default_dtype(torch.float64)


def _model(shared):
    from mace.modules.extensions import MACELES
    torch.manual_seed(0)
    return MACELES(les_shared_readout=shared, r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
                   interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
                   interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
                   num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
                   MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu, atomic_energies=np.zeros((2, 3)),
                   avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
                   atomic_inter_scale=[1.0, 1.0], atomic_inter_shift=[0.0, 0.0], heads=["pt_head", "Default"]).double()


def _batch(model, head):
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * 5.6, pbc=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=1)
    cfg = mace_data.config_from_atoms(atoms); cfg.head = head
    ds = [mace_data.AtomicData.from_config(cfg, z_table=AtomicNumberTable([17, 55, 82]), cutoff=4.0, heads=list(model.heads))]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to_dict()


def test_shared_readout_gives_head_independent_charges_and_survives_head_removal():
    m = _model(True)
    assert all((getattr(r, "num_heads", 1) == 1) for r in m.les_readouts)
    q0 = m(_batch(m, "pt_head"), compute_force=False)["latent_charges"]
    q1 = m(_batch(m, "Default"), compute_force=False)["latent_charges"]
    assert torch.allclose(q0, q1) and float(q0.abs().max()) > 0
    e0 = m(_batch(m, "pt_head"), compute_force=False)["energy"]; e1 = m(_batch(m, "Default"), compute_force=False)["energy"]
    assert not torch.allclose(e0, e1)            # energy readouts stay per head
    single = remove_pt_head(m, "Default")
    assert single.les_shared_readout
    q_single = single(_batch(single, "Default"), compute_force=False)["latent_charges"]
    assert torch.allclose(q_single, q1, atol=1e-10)


def test_unshared_readout_is_head_dependent():
    m = _model(False)
    q0 = m(_batch(m, "pt_head"), compute_force=False)["latent_charges"]
    q1 = m(_batch(m, "Default"), compute_force=False)["latent_charges"]
    assert not torch.allclose(q0, q1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cueq round trip needs CUDA")
def test_shared_readout_survives_the_cueq_round_trip():
    pytest.importorskip("cuequivariance")
    from mace.cli.convert_cueq_e3nn import run as to_e3nn
    from mace.cli.convert_e3nn_cueq import run as to_cueq
    m = _model(True)
    q = m(_batch(m, "Default"), compute_force=False)["latent_charges"]
    c = to_cueq(m, device="cuda")
    assert c.les_shared_readout and c.les_readouts[0].linear.irreps_out.dim == 1
    e = to_e3nn(c.cpu(), device="cpu").double()
    assert e.les_shared_readout
    q2 = e(_batch(e, "Default"), compute_force=False)["latent_charges"]
    assert torch.allclose(q, q2, atol=1e-8)
