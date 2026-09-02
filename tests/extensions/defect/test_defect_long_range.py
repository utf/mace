"""Long-range branch of MACEDefect: latent Ewald, structured charges, dilute limit.

These tests need the LES library (plan section 3.4 builds on it). Prefactors are pinned
against analytic results *before* any absolute energy is trusted: a prefactor error would
otherwise be absorbed into the amplitude ``a`` and surface as a plausible but wrong
``epsilon_inf``.
"""

import math

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import prepare_defect_configurations
from mace.tools import torch_geometric
from tests.helpers import LES_AVAILABLE

pytestmark = [
    pytest.mark.les,
    pytest.mark.skipif(not LES_AVAILABLE, reason="LES library is not available"),
]

if LES_AVAILABLE:
    from mace.modules.defect_models import MACEDefect
    from mace.modules.latent_ewald import LatentEwald

torch.set_default_dtype(torch.float64)

# e^2 / (4 pi eps_0), in eV Angstrom. LES stores 2 pi times this as its norm_factor.
COULOMB_CONSTANT = 14.399645
MADELUNG_NACL = 1.747564594633

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0
# Small sigma keeps the smeared kernel close to the point-charge one; dl must then be
# fine enough to converge the reciprocal-space sum.
SHARP = {"sigma": 0.5, "dl": 0.6}


def rocksalt(lattice_constant):
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    positions, charges = [], []
    for site in basis:
        positions.append(np.array(site) * lattice_constant)
        charges.append(1.0)
        positions.append((np.array(site) + np.array([0.5, 0, 0])) * lattice_constant)
        charges.append(-1.0)
    return torch.tensor(np.array(positions)), torch.tensor(charges)


class TestPrefactors:
    """Pin the absolute scale before trusting any energy (plan section 9.4)."""

    def test_isolated_pair_matches_the_smeared_coulomb_kernel(self):
        evaluator = LatentEwald({"sigma": 1.0, "dl": 1.0})
        batch = torch.zeros(2, dtype=torch.long)
        for distance in (3.0, 5.0, 7.0):
            positions = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
            charges = torch.tensor([1.0, -1.0])
            energy = evaluator.isolated_energy(charges, positions, batch, 1)
            # The kernel exp(-sigma^2 k^2 / 2) / k^2 transforms to erf(r / (sigma sqrt2)) / r.
            expected = (
                -COULOMB_CONSTANT
                / distance
                * math.erf(distance / (1.0 * math.sqrt(2.0)))
            )
            assert float(energy) == pytest.approx(expected, abs=1e-5)

    def test_periodic_sum_reproduces_the_nacl_madelung_constant(self):
        lattice_constant = 5.64
        evaluator = LatentEwald(SHARP)
        positions, charges = rocksalt(lattice_constant)
        cell = torch.eye(3).unsqueeze(0) * lattice_constant
        energy = evaluator.energy(
            charges, positions, cell, torch.zeros(8, dtype=torch.long)
        )
        # Four ion pairs in the conventional cell.
        expected = -4 * MADELUNG_NACL * COULOMB_CONSTANT / (lattice_constant / 2)
        assert float(energy) == pytest.approx(expected, rel=1e-6)

    def test_single_isolated_charge_has_no_self_energy(self):
        evaluator = LatentEwald(SHARP)
        energy = evaluator.isolated_energy(
            torch.tensor([1.0]),
            torch.zeros(1, 3),
            torch.zeros(1, dtype=torch.long),
            1,
        )
        assert float(energy) == pytest.approx(0.0, abs=1e-12)


class TestDecomposition:
    def test_cross_term_carries_an_explicit_factor_of_two(self):
        """Against a two-charge analytic case: each single charge has zero self-energy,
        so the whole pair energy must sit in ``2 E_cross``."""
        evaluator = LatentEwald({"sigma": 0.5, "dl": 0.5})
        distance = 4.0
        positions = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        q_env = torch.tensor([1.0, 0.0])
        q_carrier = torch.tensor([0.0, -1.0])
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.zeros(1, 3, 3)  # isolated

        env, cross, carrier = evaluator.decompose(
            q_env, q_carrier, positions, cell, batch
        )
        pair = -COULOMB_CONSTANT / distance * math.erf(distance / (0.5 * math.sqrt(2)))
        assert float(env) == pytest.approx(0.0, abs=1e-12)
        assert float(carrier) == pytest.approx(0.0, abs=1e-12)
        assert float(2 * cross) == pytest.approx(pair, rel=1e-5)

    def test_decomposition_reproduces_the_total(self):
        """E_LR is quadratic in the structure factor, so the polarisation identity is
        exact -- including the self-energy subtraction."""
        generator = torch.Generator().manual_seed(0)
        evaluator = LatentEwald(SHARP)
        positions = torch.rand((6, 3), generator=generator) * 5.0
        q_env = torch.randn(6, generator=generator)
        q_env = q_env - q_env.mean()
        q_carrier = torch.randn(6, generator=generator)
        cell = torch.eye(3).unsqueeze(0) * 5.0
        batch = torch.zeros(6, dtype=torch.long)

        env, cross, carrier = evaluator.decompose(
            q_env, q_carrier, positions, cell, batch
        )
        total = evaluator.energy(q_env + q_carrier, positions, cell, batch)
        assert float(env + 2 * cross + carrier) == pytest.approx(float(total), abs=1e-9)

    def test_periodic_converges_to_isolated_under_cell_growth(self):
        """The two evaluators share the smearing and self-energy paths, so their
        difference is purely image-plus-background and must vanish with cell size."""
        evaluator = LatentEwald({"sigma": 0.5, "dl": 0.5})
        positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        charges = torch.tensor([1.0, -1.0])  # neutral, so no background term
        batch = torch.zeros(2, dtype=torch.long)
        isolated = float(evaluator.isolated_energy(charges, positions, batch, 1))

        errors = []
        for length in (10.0, 20.0, 40.0):
            cell = torch.eye(3).unsqueeze(0) * length
            periodic = float(evaluator.energy(charges, positions, cell, batch))
            errors.append(abs(periodic - isolated))
        assert errors[-1] < errors[0]
        assert errors[-1] < 1e-2


def build_model(seed=0, **overrides):
    torch.manual_seed(seed)
    config = dict(
        r_max=CUTOFF,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(Z_TABLE),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=4,
        atomic_numbers=Z_TABLE.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        carrier_feature_dim=8,
        counter_embedding_dim=8,
        carrier_mlp_hidden=16,
        use_long_range=True,
        les_arguments=SHARP,
    )
    config.update(overrides)
    model = MACEDefect(**config)
    randomise_correction(model)
    return model


def randomise_correction(model, seed=1):
    """The correction heads start at zero so training starts at the base potential; the
    identities under test must survive arbitrary parameters."""
    generator = torch.Generator().manual_seed(seed)
    modules_to_randomise = list(model.carrier_pooling.energy_readouts) + list(
        model.carrier_pooling.logit_readouts
    )
    if model.use_long_range:
        modules_to_randomise += [
            model.latent_charges.host_charge,
            model.latent_charges.polarisation,
        ]
    for block in modules_to_randomise:
        for parameter in block.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)


def make_atoms(counts, seed=0, rattle=0.1, cell_size=6.0):
    rng = np.random.default_rng(seed)
    atoms = Atoms(
        numbers=[6, 14, 6, 14],
        positions=np.array(
            [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [0.0, 0.0, 2.1]]
        ),
        cell=np.eye(3) * cell_size,
        pbc=True,
    )
    if rattle:
        atoms.positions += rng.normal(scale=rattle, size=atoms.positions.shape)
    atoms.info["REF_energy"] = 0.0
    atoms.info["carrier_counts"] = np.asarray(counts, dtype=int)
    # Multiplicity is REQUIRED for every spin-polarised frame -- the guard that catches the
    # 4H-SiC labelling class, where a triplet ground state written as n = 0 had nothing to
    # contradict it. These fixtures predate that guard and were failing on it, which left
    # the whole long-range branch untested. M_s = 2 S_z, so multiplicity = |M_s| + 1.
    m_s = int((counts[0] - counts[2]) - (counts[1] - counts[3]))
    atoms.info["multiplicity"] = abs(m_s) + 1
    atoms.info["e_cbm_cell"] = -3.2
    atoms.info["e_vbm_cell"] = -6.85
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


def make_batch(atoms_list):
    keyspec = data.KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "multiplicity": "multiplicity",
            "e_cbm_cell": "e_cbm_cell",
            "e_vbm_cell": "e_vbm_cell",
        },
        arrays_keys={"forces": "REF_forces"},
    )
    configs = [
        data.config_from_atoms(atoms, key_specification=keyspec) for atoms in atoms_list
    ]
    prepare_defect_configurations(configs)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)
            for config in configs
        ],
        batch_size=len(configs),
        shuffle=False,
    )
    return next(iter(loader))


def run(model, atoms_list, **kwargs):
    return model(make_batch(atoms_list).to_dict(), **kwargs)


class TestStructuredCharges:
    def test_host_and_polarisation_channels_are_exactly_neutral(self):
        model = build_model()
        batch = make_batch(
            [make_atoms((1, 0, 0, 0), seed=0), make_atoms((0, 0, 1, 1), seed=1)]
        )
        out = model(batch.to_dict())
        for key in ("latent_charges_host", "polarisation"):
            totals = torch.zeros(2, dtype=torch.float64)
            totals.index_add_(0, batch.batch, out[key].double())
            assert torch.allclose(totals, torch.zeros_like(totals), atol=1e-12), key

    @pytest.mark.parametrize(
        "counts,charge",
        [((0, 0, 0, 0), 0), ((1, 0, 0, 0), -1), ((0, 0, 0, 1), 1), ((0, 0, 1, 1), 2)],
    )
    def test_net_latent_charge_is_the_screened_monopole(self, counts, charge):
        """sum_i q_i == a q, *not* q: asserting the latter would force a = 1, destroy the
        screening and make every dilute-limit number wrong by a factor of epsilon_inf."""
        model = build_model()
        batch = make_batch([make_atoms(counts)])
        out = model(batch.to_dict())
        total = float(out["latent_charges"].sum())
        amplitude = float(out["screening_amplitude"][0])
        assert total == pytest.approx(amplitude * charge, abs=1e-10)

    def test_amplitude_is_one_per_graph_and_independent_of_the_counters(self):
        """``a`` is a host property: it must not inherit the defect's local environment
        or depend on n."""
        model = build_model()
        atoms = make_atoms((0, 0, 0, 0), seed=2)
        amplitudes = []
        for counts in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 2), (1, 0, 1, 0)]:
            charged = atoms.copy()
            charged.info = dict(atoms.info)
            charged.info["carrier_counts"] = np.asarray(counts, dtype=int)
            # Multiplicity travels WITH the counts. Copying it from the neutral frame and
            # overriding only the counts is the labelling error the guard exists to catch,
            # reproduced in a test fixture.
            m_s = int((counts[0] - counts[2]) - (counts[1] - counts[3]))
            charged.info["multiplicity"] = abs(m_s) + 1
            charged.arrays["REF_forces"] = np.zeros((len(charged), 3))
            out = run(model, [charged])
            assert out["screening_amplitude"].shape == (1,)
            amplitudes.append(float(out["screening_amplitude"][0]))
        assert all(a == pytest.approx(amplitudes[0], abs=1e-12) for a in amplitudes)

    def test_amplitude_starts_at_the_requested_dielectric_gauge(self):
        model = build_model(eps_inf_init=4.0)
        out = run(model, [make_atoms((1, 0, 0, 0))])
        assert float(out["screening_amplitude"][0]) == pytest.approx(0.5, abs=1e-9)


class TestReferenceStateIdentityWithLongRange:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_total_equals_base_at_the_reference_state(self, seed):
        model = build_model(seed=seed)
        out = run(model, [make_atoms((0, 0, 0, 0), seed=seed)])
        assert float(out["delta_energy"].abs().max()) == 0.0
        assert torch.allclose(out["energy"], out["base_energy"], atol=0.0, rtol=0.0)

    def test_dilute_equals_base_at_the_reference_state(self):
        model = build_model()
        out = run(model, [make_atoms((0, 0, 0, 0))], dilute=True)
        assert torch.allclose(out["energy"], out["base_energy"], atol=0.0, rtol=0.0)
        assert float(out["dilute_correction"].abs().max()) == 0.0

    def test_base_energy_includes_the_host_long_range_term(self):
        """E_LR[q_host] belongs to the base branch: it is geometry-only and present at
        n = 0. If it were left out, L_tot's detached base would be wrong."""
        model = build_model()
        atoms = make_atoms((0, 0, 0, 0))
        with_lr = run(model, [atoms])
        short_range_only = build_model(use_long_range=False)
        # Copy the shared parameters so only the long-range term differs.
        short_range_only.load_state_dict(
            {
                key: value
                for key, value in model.state_dict().items()
                if key in short_range_only.state_dict()
            },
            strict=False,
        )
        assert not torch.allclose(
            with_lr["base_energy"], run(short_range_only, [atoms])["base_energy"]
        )


class TestDiluteLimit:
    def test_dilute_differs_from_total_only_by_the_carrier_self_term(self):
        model = build_model()
        atoms = make_atoms((0, 0, 0, 1), seed=3)
        periodic = run(model, [atoms])
        dilute = run(model, [atoms], dilute=True)

        batch = make_batch([atoms])
        evaluator = model.latent_ewald
        q_carrier = periodic["latent_charges_carrier"].detach()
        cell = batch.cell
        expected = float(
            evaluator.isolated_energy(q_carrier, batch.positions, batch.batch, 1)
            - evaluator.energy(q_carrier, batch.positions, cell, batch.batch)
        )
        assert float(dilute["energy"] - periodic["energy"]) == pytest.approx(
            expected, abs=1e-9
        )

    def test_dilute_correction_scales_as_the_amplitude_squared(self):
        """E_total - E_dilute is proportional to a^2, which is why a 20% error in a is a
        44% error in the image energy removed."""
        model = build_model()
        atoms = make_atoms((0, 0, 0, 1), seed=4)
        first = float(run(model, [atoms], dilute=True)["dilute_correction"][0])

        amplitude = float(run(model, [atoms])["screening_amplitude"][0])
        with torch.no_grad():
            bias = list(model.latent_charges.amplitude.modules())[-1].bias
            bias.fill_(math.log(math.expm1(2 * amplitude)))
        second = float(run(model, [atoms], dilute=True)["dilute_correction"][0])
        assert second == pytest.approx(4 * first, rel=1e-6)


class TestGradientsWithLongRange:
    def finite_difference(self, model, atoms, index, axis, step=1e-5, **kwargs):
        energies = []
        for sign in (1.0, -1.0):
            image = atoms.copy()
            image.info = dict(atoms.info)
            image.positions[index, axis] += sign * step
            image.arrays["REF_forces"] = np.zeros((len(image), 3))
            energies.append(float(run(model, [image], **kwargs)["energy"]))
        return -(energies[0] - energies[1]) / (2 * step)

    @pytest.mark.parametrize("dilute", [False, True])
    def test_forces_match_finite_differences(self, dilute):
        model = build_model()
        atoms = make_atoms((1, 0, 0, 0), seed=5)
        out = run(model, [atoms], dilute=dilute)
        for index in range(len(atoms)):
            for axis in range(3):
                expected = self.finite_difference(
                    model, atoms, index, axis, dilute=dilute
                )
                assert float(out["forces"][index, axis]) == pytest.approx(
                    expected, abs=1e-5
                )

    def test_delta_forces_match_finite_differences_of_the_delta_energy(self):
        model = build_model()
        atoms = make_atoms((0, 0, 0, 1), seed=6)
        out = run(model, [atoms])
        step = 1e-5
        for index in range(len(atoms)):
            for axis in range(3):
                energies = []
                for sign in (1.0, -1.0):
                    image = atoms.copy()
                    image.info = dict(atoms.info)
                    image.positions[index, axis] += sign * step
                    image.arrays["REF_forces"] = np.zeros((len(image), 3))
                    energies.append(float(run(model, [image])["delta_energy"]))
                expected = -(energies[0] - energies[1]) / (2 * step)
                assert float(out["delta_forces"][index, axis]) == pytest.approx(
                    expected, abs=1e-5
                )

    def test_stress_is_available_with_the_long_range_branch(self):
        model = build_model()
        out = run(model, [make_atoms((1, 0, 0, 0))], compute_stress=True)
        assert out["stress"] is not None
        assert bool(torch.isfinite(out["stress"]).all())


class TestCalculatorDiluteLimit:
    """The dilute evaluator through the ASE calculator: this is how configuration
    coordinate diagrams and isolated-limit relaxations are actually run."""

    def calculator(self, model=None, **kwargs):
        from mace.calculators import MACECalculator

        return MACECalculator(models=model or build_model(), device="cpu", **kwargs)

    def test_dilute_changes_the_energy_but_not_at_the_reference_state(self):
        model = build_model()
        periodic = self.calculator(model)
        dilute = self.calculator(model, dilute=True)

        charged = make_atoms((0, 0, 0, 1), seed=8)
        charged.calc = periodic
        periodic_energy = charged.get_potential_energy()
        charged = make_atoms((0, 0, 0, 1), seed=8)
        charged.calc = dilute
        assert charged.get_potential_energy() != periodic_energy

        neutral = make_atoms((0, 0, 0, 0), seed=8)
        neutral.calc = periodic
        neutral_periodic = neutral.get_potential_energy()
        neutral = make_atoms((0, 0, 0, 0), seed=8)
        neutral.calc = dilute
        assert neutral.get_potential_energy() == pytest.approx(
            neutral_periodic, abs=1e-12
        )

    def test_relaxation_runs_under_the_isolated_evaluator(self):
        """E_dilute is a modification of the trained model, not a single-point shift, so
        the lattice must be able to respond to the image-free field."""
        from ase.optimize import BFGS

        atoms = make_atoms((0, 0, 0, 1), seed=9)
        atoms.calc = self.calculator(dilute=True)
        start = atoms.get_potential_energy()
        BFGS(atoms, logfile=None).run(fmax=0.5, steps=3)
        assert atoms.get_potential_energy() <= start

    def test_saved_model_keeps_its_band_edge_registry(self, tmp_path):
        import torch as _torch

        model = build_model()
        model.band_edge_registry = {"GaN|4": {"e_cbm_cell": -3.2, "e_vbm_cell": -6.85}}
        path = tmp_path / "defect.model"
        _torch.save(model, path)

        from mace.calculators import MACECalculator

        calc = MACECalculator(model_paths=str(path), device="cpu", energy_scale="raw")
        atoms = make_atoms((1, 0, 0, 0), seed=10)
        atoms.info["host"] = "GaN"
        atoms.calc = calc
        raw = atoms.get_potential_energy()

        atoms = make_atoms((1, 0, 0, 0), seed=10)
        atoms.info["host"] = "GaN"
        atoms.calc = MACECalculator(model_paths=str(path), device="cpu")
        assert raw == pytest.approx(atoms.get_potential_energy() - 3.2, abs=1e-9)
