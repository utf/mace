"""``q^host`` must be able to move. A zero init on a quadratic term traps it forever.

Under ``host_carrier_coupling=False`` the only place ``q^host`` enters is ``E_LR[q^host]``,
which is quadratic in the charges. At ``q^host = 0`` that term is zero *with zero gradient*,
so a readout initialised to output exactly zero sits on a stationary point it can never
leave. Measured on a real run before the fix: ``|q^host| max = 0.000e+00`` after four epochs,
``E_LR[q^host] = 0.0 eV``, and 0.000% of the base-energy trunk gradient flowing through it --
a component that had never been exercised at all.

The ``n = 0`` identity is checked alongside, because the zero init was originally justified
by it. That justification was wrong: the identity comes from the ``counts *`` prefactor in
``delta_sr`` and from ``q^pol``'s counter factor, and ``q^host`` is geometry-only, so it
cannot break the identity however it is initialised.
"""

import pytest
import torch

from tests.unit.test_host_carrier_repartition import build_model, make_batch

pytest.importorskip("les")


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_q_host_is_not_identically_zero_at_init():
    # A bare block, not build_model: that helper randomises the charge readouts, which
    # would make this pass regardless of what the initialiser does.
    from mace.modules.defect_blocks import StructuredLatentCharges

    torch.manual_seed(0)
    block = StructuredLatentCharges(feature_dim=8, counter_dim=4, hidden_dim=8)
    feats = torch.randn(12, 8)
    counter = torch.randn(1, 4)
    q, q_host, _, _, _ = block(
        node_feats=feats,
        counter_emb=counter,
        counts=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        alpha=torch.full((12, 4), 1.0 / 12.0),
        batch=torch.zeros(12, dtype=torch.long),
        num_graphs=1,
    )
    assert float(q_host.abs().max()) > 1e-6, (
        "q^host is identically zero at init; on a quadratic term that is a stationary "
        "point it can never leave"
    )
    # Still exactly neutral -- the mean subtraction is what guarantees that, not the init.
    assert abs(float(q_host.sum())) < 1e-12


def test_q_host_receives_gradient():
    """The point of the fix: the readout must actually be trainable through E_LR."""
    from mace.modules.defect_blocks import StructuredLatentCharges
    from mace.modules.latent_ewald import LatentEwald

    torch.manual_seed(0)
    block = StructuredLatentCharges(feature_dim=8, counter_dim=4, hidden_dim=8)
    ewald = LatentEwald({"sigma": 1.0, "dl": 2.0, "norm_factor": 90.4756})
    positions = torch.rand(12, 3) * 8.0
    cell = torch.eye(3).unsqueeze(0) * 8.0
    batch = torch.zeros(12, dtype=torch.long)
    _, q_host, _, _, _ = block(
        node_feats=torch.randn(12, 8),
        counter_emb=torch.randn(1, 4),
        counts=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        alpha=torch.full((12, 4), 1.0 / 12.0),
        batch=batch,
        num_graphs=1,
    )
    energy = ewald.energy(q_host, positions, cell, batch).sum()
    grads = torch.autograd.grad(
        energy, list(block.host_charge.parameters()), allow_unused=True
    )
    norm = sum(float((g**2).sum()) for g in grads if g is not None) ** 0.5
    assert norm > 1e-10, f"E_LR[q^host] gives the host readout no gradient (|g| = {norm})"


@pytest.mark.parametrize("coupling", [True, False])
def test_zero_counter_identity_is_unaffected(coupling):
    """The identity never depended on the q^host init, and must still hold without it."""
    batch = make_batch([(0, 0, 0, 0)])
    model = build_model(host_carrier_coupling=coupling)
    out = model(batch.to_dict(), training=False, compute_force=False)
    assert float(out["correction_energy"].abs().max()) < 1e-12
    assert float((out["energy"] - out["base_energy"]).abs().max()) < 1e-10
