"""Section 5.1 of the Stage A' spec: non-parameter floats that touch the forward travel.

`avg_num_neighbors` divides every message in the trunk and is a plain float on each
interaction block -- not a parameter, not a buffer, absent from `state_dict`. A Stage-B trunk
once normalised every message by eight times Stage A's value while the loader reported
success (LEDGER.md entry 10). The standing rule that came out of it has two clauses, and
each is a test here:

  * the value is carried explicitly across a stage boundary (the buffer, refreshed at write
    time and written back at load time);
  * the PREMISE is asserted -- that the float really is invisible to state_dict -- so that if
    upstream ever turns it into a buffer, the special case is known to be redundant rather
    than left to disagree with it.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.defect_models import MACEDefect


def _model(avg_num_neighbors: float, **overrides) -> MACEDefect:
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
        avg_num_neighbors=avg_num_neighbors, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False)
    kwargs.update(overrides)
    return MACEDefect(**kwargs)


def test_the_premise_avg_num_neighbors_is_invisible_to_state_dict():
    """If this ever fails, upstream made it a buffer and the sync hooks are redundant."""
    model = _model(14.08)
    block_keys = set(model.interactions[0].state_dict().keys())
    assert not any("avg_num_neighbors" in k for k in block_keys), (
        "avg_num_neighbors is now in the interaction block's state_dict; the explicit "
        "carry in MACEDefect is redundant and should be retired, not left to disagree")


def test_state_dict_carries_the_blocks_current_value_not_the_construction_value():
    model = _model(8.0)
    # The trainer rescales the plain float after construction (run_train.py), and the
    # Stage-A loader resets it. The state dict must record what the blocks hold NOW.
    for block in model.interactions:
        block.avg_num_neighbors = 14.08
    sd = model.state_dict()
    assert torch.allclose(sd["trunk_avg_num_neighbors"],
                          torch.tensor([14.08, 14.08], dtype=torch.float64))


def test_loading_a_state_dict_writes_the_value_back_onto_the_blocks():
    source = _model(14.08)
    target = _model(112.5)      # the carrier-cutoff count the Stage-B trunk once computed
    target.load_state_dict(source.state_dict())
    assert [float(b.avg_num_neighbors) for b in target.interactions] == pytest.approx(
        [14.08, 14.08])


def test_the_round_trip_changes_the_forward():
    """The carry is not bookkeeping: the divisor changes every message, so a trunk holding
    the wrong value computes a different energy from the same weights."""
    from mace import data, tools

    table = tools.AtomicNumberTable([17, 55, 82])
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 6, size=(12, 3))
    config = data.Configuration(
        atomic_numbers=np.array([17] * 8 + [55] * 2 + [82] * 2),
        positions=positions, cell=np.eye(3) * 8.0, pbc=(True, True, True),
        properties={}, property_weights={})
    batch = data.AtomicData.from_config(config, z_table=table, cutoff=4.0)
    batch = next(iter(tools.torch_geometric.dataloader.DataLoader([batch], batch_size=1)))

    source = _model(14.08)
    wrong = _model(112.5)
    wrong.load_state_dict(source.state_dict())          # value restored -> identical
    for block in wrong.interactions:
        block.avg_num_neighbors = 112.5                  # value dropped -> different
    e_right = source(batch.to_dict(), compute_force=False)["base_energy"]
    e_wrong = wrong(batch.to_dict(), compute_force=False)["base_energy"]
    assert not torch.allclose(e_right, e_wrong), (
        "the divisor did not change the base energy; the test is not exercising the trunk")
    _restore = MACEDefect.__mro__  # noqa: F841 -- keep the name for the reader
    from mace.modules.defect_models import _restore_trunk_constants_from_buffer
    _restore_trunk_constants_from_buffer(wrong, None)    # the load-hook, applied by hand
    e_back = wrong(batch.to_dict(), compute_force=False)["base_energy"]
    assert torch.allclose(e_right, e_back)


def test_the_cache_checksum_buffer_round_trips():
    source = _model(14.08)
    with torch.no_grad():
        source.base_cache_checksum.copy_(torch.arange(32, dtype=torch.uint8))
    target = _model(14.08)
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.base_cache_checksum, torch.arange(32, dtype=torch.uint8))
