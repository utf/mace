"""The spectral head inside MACEDefect: same contract, different physics.

Both heads must expose the same six outputs, or the three call sites in `forward` and every
downstream diagnostic (per-species alpha mass, N_eff, the gap column in the training log)
would need a head-specific branch and would quietly disagree between arms.

The conversion test matters most. `run_e3nn_to_cueq` rebuilds the model from
`extract_config_mace_model`, so any constructor argument the extractor omits reverts to its
default. For `spectral_head` that default is False -- and because the two heads have entirely
different parameters, there is no shape mismatch to catch it. A converted spectral arm would
come back as an attention model with the softmax the arm exists to replace.
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


def build(spectral, seed=0):
    torch.manual_seed(seed)
    return MACEDefect(
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
        use_long_range=False, spectral_head=spectral, spectral_num_states=4,
    )


def make_batch(counts=(0, 0, 1, 0), multiplicity=1, n_cells=1):
    from ase.atoms import Atoms

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n_cells):
        atoms = Atoms(numbers=[17, 55, 82, 17, 55, 82, 17, 82],
                      positions=rng.normal(scale=2.0, size=(8, 3)) + 5.0,
                      cell=np.eye(3) * 11.0, pbc=True)
        atoms.info.update({"REF_energy": 0.0, "carrier_counts": np.array(counts),
                           "multiplicity": multiplicity, "m_s_ref_doubled": 1,
                           "cell_charge": int(counts[2] + counts[3] - counts[0] - counts[1]),
                           "host": "CsPbCl3", "e_cbm_cell": -3.2, "e_vbm_cell": -6.8})
        atoms.arrays["REF_forces"] = np.zeros((8, 3))
        frames.append(atoms)

    keyspec = mace_data.KeySpecification(
        info_keys={"energy": "REF_energy", "carrier_counts": "carrier_counts",
                   "multiplicity": "multiplicity", "host": "host",
                   "m_s_ref_doubled": "m_s_ref_doubled", "cell_charge": "cell_charge",
                   "e_cbm_cell": "e_cbm_cell", "e_vbm_cell": "e_vbm_cell"},
        arrays_keys={"forces": "REF_forces"})
    configs = [mace_data.config_from_atoms(a, key_specification=keyspec) for a in frames]
    prepare_defect_configurations(configs)
    z_table = AtomicNumberTable([17, 55, 82])
    atomic = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=4.0)
              for c in configs]
    return next(iter(torch_geometric.dataloader.DataLoader(
        atomic, batch_size=len(atomic), shuffle=False)))


def test_forward_produces_the_same_output_contract():
    batch = make_batch()
    keys = ("energy", "forces", "delta_sr_energy", "carrier_alpha", "carrier_readouts")
    outs = {}
    for spectral in (False, True):
        out = build(spectral)(batch.to_dict(), training=True, compute_force=True)
        for k in keys:
            assert out[k] is not None, f"{k} missing with spectral_head={spectral}"
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        outs[spectral] = out

    a = outs[True]["carrier_alpha"]
    assert a.shape == outs[False]["carrier_alpha"].shape
    assert float(a.sum(0).min()) > 0.99, "spectral alpha is not a per-cell distribution"


def test_both_heads_return_identically_shaped_outputs():
    """Shape parity across the whole six-output contract, on a multi-cell batch.

    The adapter returned delta_u summed over all nodes -- shape [C] where the attention head
    gives [n_graphs, C]. Nothing in the model objected; the training diagnostics did, several
    minutes into a real run, with "shape '[8, -1]' is invalid for input of size 4". Per-graph
    quantities that collapse the graph axis only show up on a batch with more than one cell,
    which is why this test uses three.
    """
    batch = make_batch(n_cells=3)
    shapes = {}
    # Follow the global default dtype rather than hardcoding .float(): several test modules
    # set the default to float64 at import time, so under the full suite this ran a float64
    # model against float32 inputs and failed on dtype, while passing in isolation.
    dt = torch.get_default_dtype()
    for spectral in (False, True):
        model = build(spectral)
        counts = batch.carrier_counts.view(3, -1).to(dt)
        out = model._carrier_head(
            node_feats=torch.randn(len(batch.batch), model.carrier_pooling.feature_dim,
                                   dtype=dt),
            counter_emb=model.counter_embedding(counts),
            counts=counts,
            batch=batch.batch, num_graphs=3,
            edge_index=batch.edge_index,
            edge_length=torch.rand(batch.edge_index.shape[1], 1, dtype=dt) * 3 + 1,
        )
        shapes[spectral] = [tuple(t.shape) for t in out]

    for i, (a, b) in enumerate(zip(shapes[False], shapes[True])):
        assert a == b, f"output {i} differs: attention {a} vs spectral {b}"


def test_alpha_normalised_per_cell_in_a_multi_cell_batch():
    batch = make_batch(n_cells=3)
    out = build(True)(batch.to_dict(), training=False)
    alpha = out["carrier_alpha"]
    for g in range(3):
        m = batch.batch == g
        for c in range(alpha.shape[1]):
            assert np.isclose(float(alpha[m, c].sum()), 1.0, atol=1e-5)


def test_zero_counts_give_zero_correction():
    batch = make_batch(counts=(0, 0, 0, 0), multiplicity=2)  # V_Cl0 is a doublet
    out = build(True)(batch.to_dict(), training=False)
    assert float(out["delta_sr_energy"].abs().max()) == 0.0
    assert float((out["energy"] - out["base_energy"]).abs().max()) == 0.0


def test_forces_flow_through_the_eigensolver():
    """Hellmann-Feynman: the carrier energy must produce real forces on the atoms its state
    lives on, or the existing force labels cannot supervise localisation."""
    batch = make_batch()
    model = build(True)
    out = model(batch.to_dict(), training=True, compute_force=True)
    out["delta_sr_energy"].sum().backward()
    grads = [p.grad for p in model.spectral.parameters() if p.grad is not None]
    assert grads, "no gradient reached the Hamiltonian"
    assert max(float(g.abs().max()) for g in grads) > 0


def test_config_extractor_preserves_the_head():
    """The trap this file exists for: a dropped flag rebuilds a spectral arm as attention."""
    model = build(True)
    config = extract_config_mace_model(model)
    assert config["spectral_head"] is True
    assert config["spectral_num_states"] == 4

    rebuilt = model.__class__(**config)
    assert rebuilt.spectral_head is True
    assert rebuilt.spectral is not None

    assert extract_config_mace_model(build(False))["spectral_head"] is False


def test_spectral_parameters_reach_an_optimizer_group():
    """Every spectral parameter must be in a group, or the Hamiltonian never trains.

    This is the failure the repo's orphan guard caught on the first real run: the head's 12
    tensors were in no group, so eps and t would have stayed at initialisation for the whole
    arm. That state -- near-flat site energies, small hopping -- is exactly what "no bound
    state anywhere" looks like, so E2 and E3 would have run to completion and reported that
    component H does not localise. A silent wrong answer, not a crash.
    """
    import argparse

    from mace.tools.arg_parser import build_default_arg_parser
    from mace.tools.scripts_utils import get_params_options

    model = build(True)
    # Take every default from the real parser rather than listing the handful this function
    # happens to read today: that list has already grown twice while writing this test, and
    # a stale hand-built namespace fails on an unrelated attribute rather than on the thing
    # under test.
    defaults = {a.dest: a.default for a in build_default_arg_parser()._actions}
    args = argparse.Namespace(**defaults)
    args.lr = 0.01
    options = get_params_options(args, model)

    covered = {id(p) for g in options["params"] for p in g["params"]}
    orphans = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in covered]
    assert not orphans, f"parameters in no optimizer group: {orphans}"

    spectral_ids = {id(p) for p in model.spectral.parameters()}
    assert spectral_ids <= covered, "spectral head is not covered"

    # And it must not land in a base group, or Stage B's frozen base would freeze it too.
    base_groups = {"embedding", "interactions_decay", "interactions_no_decay",
                   "products", "readouts"}
    for g in options["params"]:
        if g.get("name") in base_groups:
            assert not (spectral_ids & {id(p) for p in g["params"]}), \
                "spectral head is in a base group and would be frozen by Stage B"


def test_checkpoint_predating_the_spectral_head_still_runs():
    """Models are persisted by pickling the module, so a checkpoint written before
    `spectral_head` existed restores without that attribute. Every model trained up to the
    point this head was added is in that category -- A0, the Stage-A bases, every historical
    run -- and plain attribute access raises AttributeError on all of them, which is how this
    was found: scoring A0 against the gates failed to load a single model.
    """
    model = build(False)
    del model.spectral_head
    del model.spectral

    batch = make_batch()
    out = model(batch.to_dict(), training=False)
    assert torch.isfinite(out["energy"]).all()
    assert out["carrier_alpha"] is not None


@pytest.mark.parametrize("spectral", [False, True])
def test_state_dict_round_trips(spectral):
    src = build(spectral, seed=0)
    dst = build(spectral, seed=1)
    dst.load_state_dict(src.state_dict())
    batch = make_batch()
    a = src(batch.to_dict(), training=False)["delta_sr_energy"]
    b = dst(batch.to_dict(), training=False)["delta_sr_energy"]
    assert torch.allclose(a, b, atol=1e-10)
