"""The long-range branch must be genuinely absent before ``lr_start_epoch``, then present.

"Absent" has to mean the model is bit-identical to the short-range model, not merely that
some term is small: the whole point is to let attention settle under the short-range
objective, which reliably finds the vacancy shell within a few epochs, before the long-range
branch can capture it.
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


def _energies(model, batch):
    out = model(batch.to_dict(), training=False, compute_force=False)
    return {
        "energy": out["energy"].detach().clone(),
        "base": out["base_energy"].detach().clone(),
        "correction": out["correction_energy"].detach().clone(),
        "charges": out["latent_charges"],
    }


def test_before_the_start_epoch_the_model_is_the_short_range_model():
    batch = make_batch([(1, 1, 0, 2)])
    gated = build_model(use_long_range=True, lr_start_epoch=30)
    plain = build_model(use_long_range=False)
    with torch.no_grad():
        gated.current_epoch.fill_(0)
    a, b = _energies(gated, batch), _energies(plain, batch)
    for key in ("energy", "base", "correction"):
        assert torch.allclose(a[key], b[key], atol=1e-12), (
            f"{key} differs before the start epoch: {a[key].tolist()} vs {b[key].tolist()}"
        )
    assert a["charges"] is None, "latent charges must not be produced before the gate"


def test_at_and_after_the_start_epoch_the_branch_is_live():
    batch = make_batch([(1, 1, 0, 2)])
    model = build_model(use_long_range=True, lr_start_epoch=30)
    with torch.no_grad():
        model.current_epoch.fill_(29)
    before = _energies(model, batch)
    with torch.no_grad():
        model.current_epoch.fill_(30)
    after = _energies(model, batch)
    assert after["charges"] is not None, "branch did not switch on at the start epoch"
    assert not torch.allclose(before["energy"], after["energy"], atol=1e-9), (
        "energy unchanged across the switch; the branch is contributing nothing"
    )


def test_default_is_always_on():
    """lr_start_epoch = 0 must leave existing behaviour untouched at epoch 0."""
    batch = make_batch([(1, 1, 0, 2)])
    model = build_model(use_long_range=True)
    with torch.no_grad():
        model.current_epoch.fill_(0)
    assert _energies(model, batch)["charges"] is not None


def test_the_epoch_survives_a_state_dict_round_trip():
    """It is a buffer so it travels with the model; a plain attribute would not."""
    model = build_model(use_long_range=True, lr_start_epoch=30)
    with torch.no_grad():
        model.current_epoch.fill_(41)
    rebuilt = build_model(use_long_range=True, lr_start_epoch=30)
    rebuilt.load_state_dict(model.state_dict())
    assert int(rebuilt.current_epoch) == 41


def test_zero_counter_identity_holds_on_both_sides_of_the_gate():
    batch = make_batch([(0, 0, 0, 0)])
    model = build_model(use_long_range=True, lr_start_epoch=30)
    for epoch in (0, 30):
        with torch.no_grad():
            model.current_epoch.fill_(epoch)
        out = model(batch.to_dict(), training=False, compute_force=False)
        assert float(out["correction_energy"].abs().max()) < 1e-12
        assert float((out["energy"] - out["base_energy"]).abs().max()) < 1e-10
