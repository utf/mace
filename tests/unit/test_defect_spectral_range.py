"""The carrier Hamiltonian has its own range, longer than the trunk's.

At the trunk's 5.0 A cutoff the two under-coordinated Pb that share the hole -- a median
5.32 A apart, bridged in bulk by the atom that is now the vacancy -- have no edge at all in
64% of frames, and the smooth envelope gives the remaining 36% a median weight of 0.00000. The
physically correct two-site state was therefore not representable. The head's range is a
property of the carrier, not of message passing; the two coincided only because the plan
reused the trunk's neighbour list.

The graph is now built at the head's cutoff and the trunk filtered back to r_max. That filter
is an optimisation, not a correctness fix -- the radial cutoff already zeroes anything beyond
r_max -- and the test that matters is that the trunk's output is unchanged by the larger graph.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data as mace_data
from mace import modules
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_models import MACEDefect
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scripts_utils import extract_config_mace_model

R_MAX = 4.0
NUM_INTERACTIONS = 2


def build(spectral, r_cut=0.0, seed=0):
    torch.manual_seed(seed)
    return MACEDefect(
        r_max=R_MAX, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=NUM_INTERACTIONS, num_elements=3,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, spectral_head=spectral, spectral_num_states=4,
        spectral_r_cut=r_cut,
    )


def make_batch(cutoff):
    from ase.atoms import Atoms

    rng = np.random.default_rng(0)
    atoms = Atoms(numbers=[17, 55, 82] * 4,
                  positions=rng.uniform(0, 12, size=(12, 3)),
                  cell=np.eye(3) * 12.0, pbc=True)
    atoms.info.update({"REF_energy": 0.0, "carrier_counts": np.array([0, 0, 1, 0]),
                       "multiplicity": 1, "m_s_ref_doubled": 1, "cell_charge": 1,
                       "host": "CsPbCl3", "e_cbm_cell": -3.2, "e_vbm_cell": -6.8})
    atoms.arrays["REF_forces"] = np.zeros((12, 3))
    keyspec = mace_data.KeySpecification(
        info_keys={"energy": "REF_energy", "carrier_counts": "carrier_counts",
                   "multiplicity": "multiplicity", "host": "host",
                   "m_s_ref_doubled": "m_s_ref_doubled", "cell_charge": "cell_charge",
                   "e_cbm_cell": "e_cbm_cell", "e_vbm_cell": "e_vbm_cell"},
        arrays_keys={"forces": "REF_forces"})
    config = mace_data.config_from_atoms(atoms, key_specification=keyspec)
    prepare_defect_configurations([config])
    atomic = mace_data.AtomicData.from_config(
        config, z_table=AtomicNumberTable([17, 55, 82]), cutoff=cutoff)
    return next(iter(torch_geometric.dataloader.DataLoader([atomic], batch_size=1)))


def test_default_range_is_the_receptive_field():
    model = build(True)
    assert model.spectral_r_cut == pytest.approx(R_MAX * NUM_INTERACTIONS)
    assert model.spectral.r_cut == pytest.approx(R_MAX * NUM_INTERACTIONS)


def test_trunk_is_unchanged_by_the_larger_graph():
    """The load-bearing claim. Base energy and base forces must be identical whether the graph
    was built at r_max or at the head's longer cutoff, because the radial cutoff sends
    everything beyond r_max to exactly zero. If this drifts, every arm's base differs from
    every other arm's and nothing is comparable.

    Run in float64. This asserts exactness, and in float32 the accumulated round-off from
    summing 144 edges instead of 26 exceeds any tolerance worth calling exact -- which is a
    statement about the arithmetic, not about the filter. In float64 the energy difference is
    identically zero and the forces differ by 6e-16 relative, i.e. machine epsilon.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build(True, r_cut=R_MAX * NUM_INTERACTIONS).double()
        small = model(make_batch(R_MAX).to_dict(), training=False, compute_force=True)
        large = model(make_batch(R_MAX * NUM_INTERACTIONS).to_dict(),
                      training=False, compute_force=True)

        de = float((small["base_energy"] - large["base_energy"]).abs().max())
        df = float((small["base_forces"] - large["base_forces"]).abs().max())
        scale = float(small["base_forces"].abs().max())
        assert de == 0.0, f"base energy moved by {de:.3e}"
        assert df / max(scale, 1e-30) < 1e-12, f"base forces moved by {df:.3e} (rel)"
    finally:
        torch.set_default_dtype(prev)


def test_trunk_filter_keeps_exactly_the_short_edges():
    """The filter must select precisely the edges the small graph would have had."""
    small = make_batch(R_MAX)
    large = make_batch(R_MAX * NUM_INTERACTIONS)

    def lengths(b):
        vec = b.positions[b.edge_index[1]] - b.positions[b.edge_index[0]] + b.shifts
        return vec.norm(dim=-1)

    n_small = small.edge_index.shape[1]
    n_kept = int((lengths(large) <= R_MAX).sum())
    assert n_kept == n_small, f"{n_kept} edges survive the filter, small graph has {n_small}"
    assert large.edge_index.shape[1] > n_small, "the larger cutoff added no edges at all"


def test_head_actually_sees_the_longer_edges():
    """The correction must CHANGE with the larger graph -- that is the entire point. If it
    did not, the head would still be blind to the shell Pb pair."""
    model = build(True, r_cut=R_MAX * NUM_INTERACTIONS)
    small = model(make_batch(R_MAX).to_dict(), training=False)
    large = model(make_batch(R_MAX * NUM_INTERACTIONS).to_dict(), training=False)
    assert not torch.allclose(small["delta_sr_energy"], large["delta_sr_energy"]), (
        "the head saw the same Hamiltonian at both cutoffs, so the longer edges are "
        "not reaching it")


def test_attention_head_is_untouched():
    """Non-spectral arms must be unaffected: no filter, no behaviour change."""
    model = build(False)
    a = model(make_batch(R_MAX).to_dict(), training=False, compute_force=True)
    assert torch.isfinite(a["energy"]).all()
    assert model.spectral is None


def test_range_survives_config_extraction():
    model = build(True, r_cut=7.5)
    config = extract_config_mace_model(model)
    assert config["spectral_r_cut"] == pytest.approx(7.5)
    rebuilt = model.__class__(**config)
    assert rebuilt.spectral_r_cut == pytest.approx(7.5)
    assert rebuilt.spectral.r_cut == pytest.approx(7.5)
