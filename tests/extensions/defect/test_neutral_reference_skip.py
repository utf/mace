"""Section 1.1: skipping the neutral reference branch is an identity, not an approximation.

THE CLAIM. When the reference counter is the neutral one, every term of the reference
branch is exactly zero: the counting head's energy is a difference from that same fill, and
both long-range reference charges carry a `counts` factor. `MACEDefect.forward` therefore
skips a whole head pass, an Ewald evaluation and a force gradient.

WHAT THIS TEST HAS TO CHECK, and why energies are not enough. A removed subgraph changes
nothing in the value while changing the PARAMETER GRADIENT if any of those terms was not
actually zero -- and the parameter gradient is what training consumes. So every trainable
tensor's gradient is compared, not just the outputs. The long-range branch is exercised
both frozen (host charges zero, polarisation off) and live (host charges and polarisation
on, and again with the gated polarisation and the isolated carrier self-term), because
`E[latent_ref - q_host] = E[0]` is only interesting when `q_host` is not itself zero.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn import o3

from mace import data, modules, tools
from mace.modules.defect_models import MACEDefect

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])


@pytest.fixture(scope="module", autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _perovskite(reps=(2, 2, 2), rattle=0.02, seed=0):
    a = 5.6
    cell = bulk("Cs", "sc", a=a)
    atoms = Atoms("CsPbCl3",
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=cell.cell, pbc=True).repeat(reps)
    atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def _batch(frames, counts, cutoff=6.0):
    ds = []
    for atoms, c in zip(frames, counts):
        config = data.Configuration(
            atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
            cell=np.array(atoms.get_cell()), pbc=(True, True, True),
            properties={"carrier_counts": c}, property_weights={})
        ds.append(data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=cutoff))
    loader = tools.torch_geometric.dataloader.DataLoader(ds, batch_size=len(ds))
    return next(iter(loader))


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
        use_long_range=True, spectral_head=True, counting_head=True,
        spectral_first_shell=True, spectral_r_cut=6.0,
        madelung_on_site=True, madelung_composition=[3.0, 1.0, 1.0],
        madelung_z_init=[-1.0, 1.0, 2.0], les_arguments={"sigma": 1.0})
    kwargs.update(overrides)
    model = MACEDefect(**kwargs)
    # Give the host-charge readout something to say, so `E[q_host]` is not accidentally
    # zero and the uncoupled reference's `E[latent_ref - q_host]` is a real cancellation.
    with torch.no_grad():
        for module in model.latent_charges.host_charge.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.mul_(1.0)
                module.bias.add_(0.05)
    return model


def _outputs_and_grads(model, batch, skip: bool):
    model.skip_neutral_reference = skip
    model.zero_grad(set_to_none=True)
    out = model(batch.to_dict(), training=True, compute_force=True)
    loss = (out["energy"].pow(2).sum() + out["forces"].pow(2).sum()
            + out["delta_forces"].pow(2).sum() + out["base_forces"].pow(2).sum()
            + out["delta_energy"].pow(2).sum())
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    keys = ("energy", "forces", "delta_forces", "base_forces", "delta_energy",
            "delta_sr_energy")
    values = {k: (None if out.get(k) is None else out[k].detach().clone()) for k in keys}
    return values, {n: (None if g is None else g.detach().clone())
                    for (n, _), g in zip(params, grads)}


CONFIGS = {
    "frozen (Stage B, the plan v8 default)": dict(),
    "live host charges": dict(use_polarisation=False),
    "polarisation on": dict(use_polarisation=True),
    "isolated carrier self": dict(use_polarisation=True, carrier_self_isolated=True),
    "host-carrier coupling": dict(use_polarisation=True, host_carrier_coupling=True),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_skipping_the_neutral_reference_changes_nothing(name):
    model = _model(**CONFIGS[name])
    frames = [_perovskite(seed=s) for s in (1, 2)]
    batch = _batch(frames, [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    skipped, grad_skipped = _outputs_and_grads(model, batch, True)
    full, grad_full = _outputs_and_grads(model, batch, False)
    for key in skipped:
        a, b = skipped[key], full[key]
        if a is None and b is None:
            continue
        worst = float((a - b).abs().max())
        assert worst < 1e-12, f"{name}: {key} differs by {worst:.3e}"
    moved = []
    for k in grad_full:
        a, b = grad_skipped[k], grad_full[k]
        if a is None and b is None:
            continue
        # `allow_unused` returns None for a parameter that is not in the graph at all.
        # Removing a subgraph that contributed exactly zero takes a parameter out of the
        # graph, which is the same thing as a zero gradient for the optimiser -- but only
        # if the other side really is zero, so that is asserted rather than assumed.
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        worst = float((a - b).abs().max())
        if worst > 1e-12:
            moved.append(f"{k}: {worst:.3e}")
    assert not moved, f"{name}: parameter gradients moved -- " + "; ".join(moved)
    # The gradient must be non-trivial, or the comparison above is vacuous.
    assert any(g is not None and float(g.abs().max()) > 0 for g in grad_full.values())


def test_a_supplied_reference_counter_disables_the_skip():
    """A caller that states its own reference is not the neutral case, and must get the
    full branch even when the flag is on."""
    model = _model()
    frames = [_perovskite(seed=7)]
    batch = _batch(frames, [[0.0, 0.0, 1.0, 0.0]])
    d = batch.to_dict()
    d["carrier_counts_ref"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                           dtype=torch.get_default_dtype())
    model.skip_neutral_reference = True
    out = model(d, training=True, compute_force=True)
    # A non-neutral reference gives a non-zero reference correction, so `delta_energy` and
    # the total's correction differ; if the skip had fired they would coincide.
    assert float((out["delta_energy"] - out["delta_sr_energy"]).abs().max()) > 0.0
