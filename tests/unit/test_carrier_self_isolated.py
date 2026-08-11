"""``E_LR = E_periodic[Q] - sum_c E_isolated[Q^c]`` must remove the in-cell term and only it.

The in-cell electrostatic energy of ONE carrier channel with itself is self-interaction
error: a single hole has no Hartree self-repulsion. Measured at the training cell it pays
+0.104 eV to spread the attention out, and removing it is what let two of three control seeds
reach the correct vacancy-shell solution.

What must survive: the image interaction (the artefact genuinely present in the periodic DFT
labels) and the CROSS-channel terms (the electron-hole interaction on 4H-SiC is real physics).
"""

import numpy as np
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


def _delta_lr(model, batch):
    out = model(batch.to_dict(), training=False, compute_force=False)
    return (out["correction_energy"] - out["delta_sr_energy"]).detach(), out


def test_subtracts_exactly_the_isolated_self_energy():
    batch = make_batch([(1, 0, 0, 1)])
    plain = build_model(host_carrier_coupling=False, carrier_self_isolated=False)
    corrected = build_model(host_carrier_coupling=False, carrier_self_isolated=True)
    before, out = _delta_lr(plain, batch)
    after, _ = _delta_lr(corrected, batch)

    alpha = out["carrier_alpha"]
    amplitude = out["screening_amplitude"]
    counts = batch.carrier_counts.view(int(batch.num_graphs), -1).to(alpha.dtype)
    signs = plain.latent_charges.carrier_signs.unsqueeze(0) * counts
    index = batch.batch
    expected = torch.zeros(int(batch.num_graphs), dtype=alpha.dtype)
    for channel in range(4):
        charge = amplitude[index] * alpha[:, channel] * signs[index, channel]
        expected = expected + plain.latent_ewald.isolated_energy(
            charge, batch.positions, index, int(batch.num_graphs)
        )
    assert torch.allclose(before - after, expected, atol=1e-10), (
        f"removed {(before - after).tolist()}, expected {expected.tolist()}"
    )
    assert float(expected.abs().max()) > 1e-6, "no-op on this system; test proves nothing"


def test_cross_channel_terms_survive():
    """Only the WITHIN-channel self-energy is removed; electron-hole must remain."""
    batch = make_batch([(1, 0, 0, 1)])  # one electron and one hole: a live cross term
    model = build_model(host_carrier_coupling=False, carrier_self_isolated=True)
    delta, out = _delta_lr(model, batch)

    alpha = out["carrier_alpha"]
    amplitude = out["screening_amplitude"]
    counts = batch.carrier_counts.view(int(batch.num_graphs), -1).to(alpha.dtype)
    signs = model.latent_charges.carrier_signs.unsqueeze(0) * counts
    index = batch.batch
    channels = [
        amplitude[index] * alpha[:, c] * signs[index, c] for c in range(4)
    ]
    isolated = model.latent_ewald.isolated_energy
    n = int(batch.num_graphs)
    # counts (1, 0, 0, 1) makes channels 0 (e_maj) and 3 (h_min) live; 1 and 2 carry no
    # charge at all, so a cross term built from them would be identically zero and the
    # test would pass while checking nothing.
    cross = (
        isolated(channels[0] + channels[3], batch.positions, index, n)
        - isolated(channels[0], batch.positions, index, n)
        - isolated(channels[3], batch.positions, index, n)
    )
    assert float(cross.abs().max()) > 1e-8, "no cross term on this system to test"
    # delta_lr still contains it: removing it too would leave delta_lr smaller by `cross`.
    assert not torch.allclose(delta, delta - cross, atol=1e-12)


@pytest.mark.parametrize("isolated", [True, False])
def test_zero_counter_identity_survives(isolated):
    batch = make_batch([(0, 0, 0, 0)])
    model = build_model(host_carrier_coupling=False, carrier_self_isolated=isolated)
    out = model(batch.to_dict(), training=False, compute_force=False)
    assert float(out["correction_energy"].abs().max()) < 1e-12
    assert float((out["energy"] - out["base_energy"]).abs().max()) < 1e-10


def test_charges_are_untouched():
    batch = make_batch([(1, 0, 0, 1), (1, 1, 0, 2)])
    plain = build_model(host_carrier_coupling=False, carrier_self_isolated=False)(
        batch.to_dict(), training=False, compute_force=False
    )
    corrected = build_model(host_carrier_coupling=False, carrier_self_isolated=True)(
        batch.to_dict(), training=False, compute_force=False
    )
    assert torch.allclose(
        plain["latent_charges"], corrected["latent_charges"], atol=1e-12
    ), "the correction must change the ENERGY, never the charges"


def test_forces_follow_the_corrected_energy():
    batch = make_batch([(1, 1, 0, 2)])
    model = build_model(host_carrier_coupling=False, carrier_self_isolated=True)
    batch_dict = batch.to_dict()
    batch_dict["positions"].requires_grad_(True)
    out = model(batch_dict, training=True, compute_force=True)
    manual = torch.autograd.grad(
        out["energy"].sum(), batch_dict["positions"], retain_graph=True
    )[0]
    assert torch.allclose(out["forces"], -manual, atol=1e-8)


def test_removes_the_shape_pressure_between_two_attention_patterns():
    """The point of the fix: a localised and a spread carrier must stop differing by the
    in-cell term. Hand-set attention, so the trained alpha cannot confound it."""
    from mace.modules.latent_ewald import LatentEwald

    ewald = LatentEwald({"sigma": 1.0, "dl": 2.0, "norm_factor": 90.4756})
    positions = torch.rand(24, 3) * 12.0
    cell = torch.eye(3).unsqueeze(0) * 12.0
    index = torch.zeros(24, dtype=torch.long)
    amplitude = 0.5

    shapes = {}
    for name, sites in (("localised", [0, 1]), ("spread", list(range(12)))):
        weights = torch.zeros(24)
        weights[sites] = 1.0 / len(sites)
        charge = amplitude * weights
        periodic = float(ewald.energy(charge, positions, cell, index).sum())
        isolated = float(ewald.isolated_energy(charge, positions, index, 1).sum())
        shapes[name] = (periodic, periodic - isolated)

    raw = abs(shapes["localised"][0] - shapes["spread"][0])
    corrected = abs(shapes["localised"][1] - shapes["spread"][1])
    assert corrected < 0.5 * raw, (
        f"in-cell pressure not reduced: raw gap {raw:.4f} eV, corrected {corrected:.4f} eV"
    )
