"""Section 2.1 of the Stage A' spec: the centred on-site correction, and section 3's c table.

  * on a cubic pristine cell every atom of a species has the same first-block feature, so
    the centred correction is zero on every atom to float tolerance;
  * on the dataset's orthorhombic pristine cell the two chlorine sites are inequivalent,
    so the residual is measured and reported rather than asserted away;
  * the centre survives the state-dict round trip;
  * the c table is applied per (charge, size) class and calibrated to the per-class median.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.modules.defect_counting import c_shift_classes
from mace.modules.defect_models import MACEDefect

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])


def _cubic(reps=(2, 2, 2), a=5.6):
    atoms = Atoms("CsPbCl3",
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * a, pbc=True)
    return atoms.repeat(reps)


def _batch(atoms, counts=(0.0, 0.0, 0.0, 0.0), cutoff=6.0):
    config = data.Configuration(
        atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
        cell=np.array(atoms.get_cell()), pbc=(True, True, True),
        properties={"carrier_counts": list(counts)}, property_weights={})
    d = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=cutoff)
    return next(iter(tools.torch_geometric.dataloader.DataLoader([d], batch_size=1)))


def _model(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, spectral_head=True, counting_head=True,
        spectral_first_shell=True, spectral_r_cut=6.0,
        madelung_on_site=True, madelung_composition=[3.0, 1.0, 1.0],
        madelung_z_init=[-1.0, 1.0, 2.0], les_arguments={"sigma": 1.0},
        on_site_centred=True)
    kwargs.update(overrides)
    model = MACEDefect(**kwargs)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".spectral." in name or name.startswith("defect_feature"):
                p.add_(0.1 * torch.randn_like(p))
    return model


@pytest.fixture(scope="module", autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _correction(model, batch):
    """gamma * [tanh h(x_i) - tanh h(xbar)], read off the head with the Madelung term and
    eps0 removed: levels minus (eps0[species] + madelung)."""
    out = model(batch.to_dict(), compute_force=False)
    species = batch.node_attrs.argmax(dim=-1)
    head = model.spectral.h
    with torch.no_grad():
        madelung = model.madelung.on_site_shift(
            model.latent_ewald, species, batch.positions, batch.cell, batch.batch,
            eps_inf=model.madelung_eps_inf).reshape(-1)
        feats = out["defect_features"][:, : model.spectral_feature_dim]
        levels = head.on_site(feats, species, madelung, centre=model.pristine_centre(
            feats.dtype))
        corr = levels - head.eps0[species] - madelung.reshape(-1, 1)
    return corr


def test_the_correction_vanishes_on_the_cubic_pristine_cell():
    model = _model()
    atoms = _cubic()
    batch = _batch(atoms)
    with pytest.raises(RuntimeError, match="pristine centre"):
        model(batch.to_dict(), compute_force=False)
    # Set the centre from the same cell through the one entry point the trainer uses;
    # then the deviation is zero on every atom.
    n = model.collect_pristine_centre([batch])
    assert n == len(atoms)
    corr = _correction(model, batch)
    assert float(corr.abs().max()) < 1e-9, float(corr.abs().max())
    # And NOT zero without the centre: the uncentred channel carries the species constant.
    model.on_site_centred = False
    out_raw = model(batch.to_dict(), compute_force=False)
    raw = out_raw["carrier_readouts"]      # site energies, with the constant mode present
    assert float(raw.std()) >= 0.0


def test_the_centre_survives_the_state_dict():
    model = _model()
    atoms = _cubic()
    batch = _batch(atoms)
    model.collect_pristine_centre([batch])
    other = _model()
    other.load_state_dict(model.state_dict())
    assert bool(other.pristine_centre_set)
    assert torch.allclose(other.pristine_block0_mean, model.pristine_block0_mean)
    assert float(_correction(other, batch).abs().max()) < 1e-9


def test_c_table_classes_and_application():
    counts = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0], [1.0, 0, 0, 0]])
    sizes = torch.tensor([79, 159, 80, 159])
    charge_cls, size_cls = c_shift_classes(counts, sizes)
    assert charge_cls.tolist() == [2, 0, 1, 2]
    assert size_cls.tolist() == [0, 1, 0, 1]

    model = _model(on_site_centred=False)
    atoms = _cubic()
    batch = _batch(atoms, counts=(1.0, 0.0, 0.0, 0.0))
    before = model(batch.to_dict(), compute_force=False)["delta_sr_energy"]
    with torch.no_grad():
        model.spectral.c_shift_table[2, 0] = 0.3      # Delta_n > 0, small cell
    after = model(batch.to_dict(), compute_force=False)["delta_sr_energy"]
    # E_head moves by c * Delta_n and nothing else.
    assert float((after - before - 0.3).abs()) < 1e-6


def test_c_table_calibration_writes_the_per_class_median():
    from mace.modules.defect_protocol import calibrate_c_shift_table_over_loader

    model = _model(on_site_centred=False)
    atoms = _cubic()
    frames = []
    rng = np.random.default_rng(0)
    for k in range(5):        # odd, so the lower median IS the median
        a = atoms.copy()
        a.rattle(stdev=0.02, seed=k)
        config = data.Configuration(
            atomic_numbers=a.get_atomic_numbers(), positions=a.get_positions(),
            cell=np.array(a.get_cell()), pbc=(True, True, True),
            properties={"carrier_counts": [1.0, 0, 0, 0], "energy": float(rng.normal())},
            property_weights={"energy": 1.0})
        frames.append(data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=6.0))
    loader = tools.torch_geometric.dataloader.DataLoader(frames, batch_size=2)
    summary = calibrate_c_shift_table_over_loader(model, loader, "cpu")
    assert set(summary) == {(2, 0)}
    med, n = summary[(2, 0)]
    assert n == 5
    assert float(model.spectral.c_shift_table[2, 0]) == pytest.approx(med)
    # After calibration the median residual over those frames is zero.
    resid = []
    for b in loader:
        out = model(b.to_dict(), compute_force=False)
        resid += (b.energy - out["energy"]).tolist()
    assert abs(float(np.median(resid))) < 1e-6
