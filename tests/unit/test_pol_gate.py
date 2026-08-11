"""Size consistency of the carrier-gated polarisation channel, in both regimes.

The failure this replaces was a whole-cell mean subtraction, which gives neutrality but not
locality: a bulk atom keeps a fixed per-species residual, and the resulting fictitious ionic
lattice has a Madelung energy (``pol^2``) and a cross term with ``q^host`` that both grow in
proportion to N.

Gating fixes it structurally, and the two regimes have *different* correct answers, so a
test that only checks one of them proves little:

* **localised** carrier -- the gate decays away from it, so ``pol^2`` and ``host.pol`` are
  both size-independent;
* **delocalised** carrier (``alpha = 1/N``) -- the gate is ``~C/N`` uniform, so ``pol^2``
  goes as ``N (1/N)^2 = 1/N`` and *decays*, while ``host.pol`` goes as
  ``N A_host (1/N) = O(1)`` and is *constant*. Constant is correct here: an intensive shift
  for a delocalised carrier is a band-edge shift, which is physical.

``host.pol`` was the larger term in the real failure (-52 eV against -21.5 eV), so it is
checked in both regimes rather than ``pol^2`` alone.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_blocks import StructuredLatentCharges


@pytest.fixture(autouse=True)
def _double_precision():
    """Per-test, not at import.

    Setting the default dtype at module import is not enough: pytest imports every module
    before running anything, and another module setting float32 mid-session then changes
    the numerics here. Observed exactly that -- these tests passed alone and
    `test_ungated_mean_subtraction_is_the_bug` reported +0.014 instead of ~+1.0 when run
    after the model tests.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)



def _cubic(repeat: int, spacing: float = 3.0):
    """Simple cubic lattice and its r_max = 5 edge list, minimum image."""
    grid = np.stack(
        np.meshgrid(*[np.arange(repeat)] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions = grid * spacing
    box = repeat * spacing
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= box * np.rint(delta / box)
    distance = np.linalg.norm(delta, axis=-1)
    sender, receiver = np.nonzero((distance > 1e-9) & (distance < 5.0))
    return (
        torch.tensor(positions),
        torch.tensor(np.stack([sender, receiver])),
        torch.tensor(distance[sender, receiver]),
        float(box),
        torch.tensor(grid),
    )


# lambda = 4 A over one hop reaches the six nearest neighbours of a 3 A simple cubic
# lattice and nothing further. Production uses 6 A over two hops (~10 A), but a 10 A gate
# in a 9-15 A test box wraps onto itself, so the "localised" cloud would fill every cell
# and the test would measure the lattice rather than the gate. Repeats start at 4 (12 A
# box) so every cell is more than twice the gate range.
GATE = {"pol_gate_lambda": 4.0, "pol_gate_hops": 1}
# Even repeats only, so the two-species pattern below tiles the cell exactly and the
# defect's local environment is identical at every size. With a repeat that breaks the
# sublattice period, each size presents the defect a different neighbourhood and the
# measured scaling is that inconsistency rather than the gate's behaviour.
REPEATS = (4, 6, 8)


def _block(**kwargs):
    if kwargs.get("pol_gate"):
        kwargs = {**GATE, **kwargs}
    # Seed the GLOBAL rng, not just the generator below. `_mlp` initialises its first layer
    # from the global state, so without this the hidden layer depends on whatever tests ran
    # before -- these passed alone and failed after the model tests, reporting +0.014 where
    # ~+1.0 was expected.
    torch.manual_seed(0)
    block = StructuredLatentCharges(feature_dim=4, counter_dim=3, hidden_dim=8, **kwargs)
    # Non-trivial readouts: zero_last_layer leaves p identically zero, which would pass
    # every scaling check vacuously.
    generator = torch.Generator().manual_seed(0)
    for mlp in (block.polarisation, block.host_charge):
        last = list(mlp.modules())[-1]
        with torch.no_grad():
            last.weight.normal_(0.0, 0.5, generator=generator)
            last.bias.normal_(0.0, 0.5, generator=generator)
    return block


def _charges(block, repeat, localised: bool):
    positions, edge_index, lengths, _, grid = _cubic(repeat)
    n_nodes = positions.shape[0]
    batch = torch.zeros(n_nodes, dtype=torch.long)
    # THREE species, which the bug requires. With a single species the whole-cell mean is
    # dominated by the bulk value and each bulk atom is left only -(defect excess)/N, which
    # decays -- so a one-species lattice cannot reproduce the failure at all. With several,
    # the mean is composition-weighted and each species keeps a fixed non-zero residual on
    # every one of its atoms: the fictitious ionic lattice measured on CsPbCl3 as
    # Cs -0.067, Pb +0.109, Cl -0.015 e.
    species = grid[:, 0] % 2  # geometric, so it tiles with the lattice
    feats = torch.ones(n_nodes, 4) * (
        1.0 + species.unsqueeze(-1).to(torch.get_default_dtype())
    )
    # Two "defect" atoms with a perturbed environment, as a real descriptor would give.
    feats[:2] += 0.5
    counts = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    counter = torch.tensor([[0.3, -0.2, 0.1]])

    alpha = torch.zeros(n_nodes, 4)
    if localised:
        alpha[0, 2] = 0.5
        alpha[1, 2] = 0.5
    else:
        alpha[:, 2] = 1.0 / n_nodes

    q, q_host, q_carrier, _, _ = block(
        node_feats=feats,
        counter_emb=counter,
        counts=counts,
        alpha=alpha,
        batch=batch,
        num_graphs=1,
        edge_index=edge_index,
        edge_lengths=lengths,
    )
    return q - q_host - q_carrier, q_host, n_nodes


@pytest.mark.parametrize("localised", [True, False])
def test_gate_keeps_q_pol_exactly_neutral(localised):
    block = _block(pol_gate=True)
    for repeat in REPEATS[:2]:
        q_pol, _, _ = _charges(block, repeat, localised)
        assert abs(float(q_pol.sum())) < 1e-12


def test_gate_is_identically_zero_without_carriers():
    """The n = 0 identity has to be structural, not approximate."""
    block = _block(pol_gate=True)
    positions, edge_index, lengths, _, _ = _cubic(4)
    n_nodes = positions.shape[0]
    q, q_host, q_carrier, _, _ = block(
        node_feats=torch.ones(n_nodes, 4),
        counter_emb=torch.tensor([[0.3, -0.2, 0.1]]),
        counts=torch.zeros(1, 4),
        alpha=torch.zeros(n_nodes, 4),
        batch=torch.zeros(n_nodes, dtype=torch.long),
        num_graphs=1,
        edge_index=edge_index,
        edge_lengths=lengths,
    )
    assert torch.all((q - q_host - q_carrier) == 0.0)


def test_localised_carrier_gives_size_independent_q_pol():
    """A localised carrier: the gate decays, so sum q_pol^2 stops changing with N."""
    block = _block(pol_gate=True)
    values = {}
    for repeat in REPEATS:
        q_pol, _, n_nodes = _charges(block, repeat, localised=True)
        values[n_nodes] = float((q_pol**2).sum())
    sizes = np.array(sorted(values))
    series = np.array([values[n] for n in sizes])
    exponent = np.polyfit(np.log(sizes), np.log(series), 1)[0]
    assert exponent < 0.1, f"sum q_pol^2 grows with N (exponent {exponent:+.3f})"


def test_delocalised_carrier_gives_decaying_q_pol():
    """alpha = 1/N: the gate is ~C/N uniform, so sum q_pol^2 must go as 1/N."""
    block = _block(pol_gate=True)
    values = {}
    for repeat in REPEATS:
        q_pol, _, n_nodes = _charges(block, repeat, localised=False)
        values[n_nodes] = float((q_pol**2).sum())
    sizes = np.array(sorted(values))
    series = np.array([values[n] for n in sizes])
    exponent = np.polyfit(np.log(sizes), np.log(series), 1)[0]
    assert exponent < -0.5, f"expected ~1/N decay, got exponent {exponent:+.3f}"


def test_ungated_mean_subtraction_is_the_bug():
    """Guard the diagnosis itself: without gating, sum q_pol^2 grows in proportion to N.

    If this ever stops failing the way it does today, the gated results above lose their
    meaning -- they would no longer be fixing anything.
    """
    block = _block(pol_gate=False)
    values = {}
    for repeat in REPEATS:
        q_pol, _, n_nodes = _charges(block, repeat, localised=True)
        values[n_nodes] = float((q_pol**2).sum())
    sizes = np.array(sorted(values))
    series = np.array([values[n] for n in sizes])
    exponent = np.polyfit(np.log(sizes), np.log(series), 1)[0]
    assert exponent > 0.8, f"expected extensive growth, got exponent {exponent:+.3f}"


def test_ablation_removes_the_channel_entirely():
    block = _block(use_polarisation=False)
    for localised in (True, False):
        q_pol, _, _ = _charges(block, 4, localised)
        # Reconstructed as q - q_host - q_carrier, which is not bit-exact in floating
        # point even when the channel contributes exactly nothing.
        assert float(q_pol.abs().max()) < 1e-15


def test_gate_energy_scaling_in_both_regimes():
    """The energies themselves, through the real Ewald sum, in both regimes.

    ``host.pol`` is checked as well as ``pol^2``: it was the larger term in the failure
    (-52 eV against -21.5 eV), so a check covering only ``pol^2`` misses what broke.

    The two regimes need different instruments. For a **localised** carrier the charge
    distribution is bit-identical at every size (verified below), so the energy *converges*
    -- image interactions die off -- rather than following any power law. Fitting an
    exponent to a converging sequence reads its approach to the limit as growth: these
    energies go -0.002634, -0.003295, -0.003361, which is a 10x reduction in successive
    differences and fits to +0.118. Convergence is therefore tested as convergence. For a
    **delocalised** carrier there is a genuine power law and the exponent is the right tool.
    """
    pytest.importorskip("les")
    from mace.modules.latent_ewald import LatentEwald

    ewald = LatentEwald({"sigma": 1.0, "dl": 2.0, "norm_factor": 90.4756})
    block = _block(pol_gate=True)

    def terms(repeat, localised):
        positions, _, _, box, _ = _cubic(repeat)
        q_pol, q_host, n_nodes = _charges(block, repeat, localised)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        cell = torch.eye(3).unsqueeze(0) * box

        def energy(charge):
            return float(ewald.energy(charge, positions, cell, batch).sum())

        pol_squared = energy(q_pol)
        cross = energy(q_host + q_pol) - energy(q_host) - energy(q_pol)
        return pol_squared, cross, float((q_pol**2).sum())

    # --- localised: identical charges at every size, energies converging ---------------
    pols, crosses, sums = [], [], []
    for repeat in REPEATS:
        pol_squared, cross, sum_squares = terms(repeat, localised=True)
        pols.append(pol_squared)
        crosses.append(cross)
        sums.append(sum_squares)
    assert max(sums) - min(sums) < 1e-12, (
        f"gated q_pol should be size-independent for a localised carrier, got {sums}"
    )
    for name, series in (("pol^2", pols), ("host.pol", crosses)):
        first = abs(series[1] - series[0])
        second = abs(series[2] - series[1])
        assert second < 0.5 * first + 1e-12, (
            f"localised {name} is not converging: steps {first:.3e} then {second:.3e}"
        )
        assert second < 1e-3, f"localised {name} still moving by {second:.3e} eV"

    # --- delocalised: a genuine power law, so fit it -----------------------------------
    sizes, pols, crosses = [], [], []
    for repeat in REPEATS:
        pol_squared, cross, _ = terms(repeat, localised=False)
        sizes.append(repeat**3)
        pols.append(abs(pol_squared))
        crosses.append(abs(cross))
    logs = np.log(np.array(sizes, dtype=float))
    pol_exponent = np.polyfit(logs, np.log(pols), 1)[0]
    cross_exponent = np.polyfit(logs, np.log(crosses), 1)[0]
    assert pol_exponent < -0.5, f"delocalised pol^2 exponent {pol_exponent:+.3f}, want ~-1"
    # Constant is the correct answer here -- an intensive shift for a delocalised carrier
    # is a band-edge shift -- so this bounds growth rather than requiring decay.
    assert cross_exponent < 0.35, (
        f"delocalised host.pol exponent {cross_exponent:+.3f}, want ~0"
    )
