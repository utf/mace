"""Unit tests for MACEDefect's short-range carrier correction (plan sections 3.2, 3.3).

The identities checked here must hold for *any* parameters, so the correction heads are
randomised away from their zero initialisation first. Nothing here needs LES: the
long-range branch is tested separately.
"""

import argparse
import json

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.defect_blocks import NUM_CARRIER_CHANNELS, segment_softmax
from mace.modules.defect_models import MACEDefect
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_params_options
from mace.tools.torch_tools import supports_float64, to_high_precision

torch.set_default_dtype(torch.float64)

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0


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
        use_long_range=False,
    )
    config.update(overrides)
    model = MACEDefect(**config)
    randomise_correction(model)
    return model


def randomise_correction(model, seed=1):
    """The energy readouts are initialised at zero so training starts at the base
    potential; the identities under test must survive arbitrary values."""
    generator = torch.Generator().manual_seed(seed)
    for readout in model.carrier_pooling.energy_readouts:
        for parameter in readout.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    for readout in model.carrier_pooling.logit_readouts:
        for parameter in readout.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)


def make_atoms(counts, seed=0, rattle=0.1, repeat=1, cell_size=4.2):
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
    if repeat > 1:
        atoms = atoms.repeat((repeat, 1, 1))
    atoms.info["REF_energy"] = 0.0
    atoms.info["carrier_counts"] = np.asarray(counts, dtype=int)
    atoms.info["e_cbm_cell"] = -3.2
    atoms.info["e_vbm_cell"] = -6.85
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


def make_batch(atoms_list):
    keyspec = data.KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "e_cbm_cell": "e_cbm_cell",
            "e_vbm_cell": "e_vbm_cell",
            "pair_id": "pair_id",
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
    batch = make_batch(atoms_list)
    return model(batch.to_dict(), **kwargs)


class TestReferenceStateIdentity:
    """E_total(R, 0) == E_base(R) exactly, for any parameters (plan section 8.1)."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_neutral_energy_equals_base_energy(self, seed):
        model = build_model(seed=seed)
        out = run(model, [make_atoms((0, 0, 0, 0), seed=seed)])
        assert torch.allclose(out["energy"], out["base_energy"], atol=0.0, rtol=0.0), (
            "the correction must vanish identically, not approximately"
        )
        assert float(out["delta_energy"].abs().max()) == 0.0

    def test_neutral_delta_forces_vanish(self):
        model = build_model()
        out = run(model, [make_atoms((0, 0, 0, 0))])
        assert float(out["delta_forces"].abs().max()) == 0.0

    def test_charged_state_actually_changes_the_energy(self):
        """Guards against the identity passing because the branch is dead."""
        model = build_model()
        neutral = run(model, [make_atoms((0, 0, 0, 0))])
        charged = run(model, [make_atoms((1, 0, 0, 0))])
        assert not torch.allclose(neutral["energy"], charged["energy"])
        assert torch.allclose(neutral["base_energy"], charged["base_energy"])

    def test_base_branch_is_independent_of_the_counters(self):
        model = build_model()
        energies = [
            run(model, [make_atoms(counts)])["base_energy"]
            for counts in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 2), (1, 0, 1, 0)]
        ]
        for energy in energies[1:]:
            assert torch.allclose(energy, energies[0], atol=0.0, rtol=0.0)


class TestOptimizerCoverage:
    """Every trainable parameter must reach the optimizer.

    ``get_params_options`` builds its groups from an explicit whitelist of submodules,
    so a model with a submodule nobody added there trains with that branch frozen and
    reports nothing unusual. For MACEDefect this was silent and total: ``MLP_u`` is
    initialised at zero, so a frozen correction branch predicts exactly zero, the
    ``E_total(R, 0) == E_base(R)`` identity holds trivially, and the reported ``RMSE dE``
    sits exactly on the target standard deviation -- a green error table with 69% of the
    model never updated.
    """

    @staticmethod
    def _args():
        return argparse.Namespace(
            lr=0.005,
            weight_decay=5e-7,
            amsgrad=True,
            beta=0.9,
            lr_params_factors=json.dumps({}),
            freeze=None,
            train_one_body_contribution=False,
        )

    @pytest.mark.parametrize("use_long_range", [False, True])
    def test_every_defect_parameter_is_optimised(self, use_long_range):
        model = build_model(use_long_range=use_long_range)
        options = get_params_options(self._args(), model)
        covered = {id(p) for group in options["params"] for p in group["params"]}
        missing = [n for n, p in model.named_parameters() if id(p) not in covered]
        assert not missing
        # The correction heads specifically, not merely "some group exists".
        names = {group["name"] for group in options["params"]}
        assert {
            "counter_embedding",
            "carrier_pooling",
            "defect_feature_readouts",
        } <= names

    @pytest.mark.parametrize("train_one_body", [True, False])
    def test_untrained_parameters_stay_covered_at_zero_lr(self, train_one_body):
        """Not training a parameter must not mean dropping it from the optimizer.

        ``onebody_magmombasis_coeffs`` exists on the magnetic models whether or not
        ``--train_one_body_contribution`` is set, so excluding it when the flag is off
        would make a legitimate config indistinguishable from a forgotten submodule.
        """
        model = build_model()
        model.onebody_magmombasis_coeffs = torch.nn.Parameter(torch.randn(2, 3, 1))
        args = self._args()
        args.train_one_body_contribution = train_one_body

        options = get_params_options(args, model)
        group = next(
            g for g in options["params"] if g["name"] == "onebody_magmombasis_coeffs"
        )
        assert group["lr"] == (args.lr if train_one_body else 0.0)

    def test_an_unregistered_submodule_is_rejected(self):
        """The guard has to fire, or it is decoration."""
        model = build_model()
        model.stowaway = torch.nn.Linear(4, 4)
        with pytest.raises(RuntimeError, match="no optimizer group"):
            get_params_options(self._args(), model)


class TestPrecisionHelpers:
    """MPS has no float64, so an unconditional upcast is a hard error there."""

    def test_upcasts_on_cpu(self):
        tensor = torch.zeros(3, dtype=torch.float32)
        assert supports_float64(tensor.device)
        assert to_high_precision(tensor).dtype == torch.float64

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="requires an MPS device"
    )
    def test_is_a_noop_on_mps(self):
        tensor = torch.zeros(3, dtype=torch.float32, device="mps")
        assert not supports_float64(tensor.device)
        assert to_high_precision(tensor).dtype == torch.float32


class TestPooling:
    def test_attention_weights_sum_to_one_per_cell_and_channel(self):
        model = build_model()
        batch = make_batch(
            [
                make_atoms((1, 0, 0, 0), seed=0),
                make_atoms((0, 0, 0, 1), seed=1, repeat=2),
                make_atoms((1, 0, 1, 0), seed=2, repeat=3),
            ]
        )
        out = model(batch.to_dict())
        totals = torch.zeros(3, 4, dtype=torch.float64)
        totals.index_add_(0, batch.batch, out["carrier_alpha"].double())
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-12)

    @pytest.mark.parametrize("high_precision", [True, False])
    def test_softmax_normalisation_without_the_float64_accumulation(
        self, high_precision
    ):
        """Bound the accuracy lost when the float64 upcast is unavailable.

        ``to_high_precision`` is a no-op on MPS, which has no float64 at all, so a model
        running there takes the ``high_precision=False`` path whatever the flag says.
        Exercising that path explicitly keeps the deviation measured on every machine
        rather than only on a Mac. The invariant itself must survive: no epsilon is added
        to the denominator, so ``sum_i alpha_i == 1`` holds up to rounding, not
        approximately.
        """
        generator = torch.Generator().manual_seed(0)
        # Deliberately wide logits: the gap is where a low-precision softmax would drift.
        logits = 12.0 * torch.randn(
            64, NUM_CARRIER_CHANNELS, generator=generator, dtype=torch.float32
        )
        batch = torch.repeat_interleave(torch.arange(4), 16)

        alpha, _ = segment_softmax(
            logits, batch, num_graphs=4, high_precision=high_precision
        )
        assert alpha.dtype == torch.float32

        totals = torch.zeros(4, NUM_CARRIER_CHANNELS, dtype=torch.float64)
        totals.index_add_(0, batch, alpha.double())
        # float64 accumulation lands within float32's own representable spacing of 1;
        # without it the error grows with the number of summands but stays tiny.
        tolerance = 1e-7 if high_precision else 1e-5
        assert torch.allclose(totals, torch.ones_like(totals), atol=tolerance)

    def test_batching_does_not_couple_cells(self):
        """A batch-wide softmax would pass the single-cell tests and fail this one."""
        model = build_model()
        atoms_list = [
            make_atoms((1, 0, 0, 0), seed=0),
            make_atoms((0, 0, 0, 1), seed=1, repeat=2),
            make_atoms((1, 0, 0, 1), seed=2, repeat=3),
        ]
        batched = run(model, atoms_list)
        for index, atoms in enumerate(atoms_list):
            single = run(model, [atoms])
            assert torch.allclose(
                batched["energy"][index], single["energy"][0], atol=1e-10
            )
            assert torch.allclose(
                batched["delta_energy"][index], single["delta_energy"][0], atol=1e-10
            )

    @pytest.mark.parametrize("repeat", [2, 3])
    def test_correction_is_intensive_on_pristine_cells(self, repeat):
        """The pooled correction is exact on pristine cells of any size: the number of
        formula units cancels identically (plan section 3.2)."""
        model = build_model()
        small = run(model, [make_atoms((1, 0, 0, 0), rattle=0.0)])
        large = run(model, [make_atoms((1, 0, 0, 0), rattle=0.0, repeat=repeat)])
        assert torch.allclose(small["delta_energy"], large["delta_energy"], atol=1e-10)
        # The base branch, by contrast, is extensive.
        assert float(large["base_energy"]) == pytest.approx(
            repeat * float(small["base_energy"]), rel=1e-8
        )

    def test_permutation_invariance(self):
        """Worth testing explicitly because the pooling introduces a global reduction."""
        model = build_model()
        atoms = make_atoms((1, 0, 0, 1), seed=3)
        permuted = atoms[[2, 0, 3, 1]]
        permuted.info = dict(atoms.info)
        permuted.arrays["REF_forces"] = np.zeros((len(permuted), 3))
        reference = run(model, [atoms])
        shuffled = run(model, [permuted])
        assert torch.allclose(
            reference["delta_energy"], shuffled["delta_energy"], atol=1e-12
        )
        assert torch.allclose(reference["energy"], shuffled["energy"], atol=1e-12)

    def test_translation_invariance(self):
        model = build_model()
        atoms = make_atoms((0, 0, 0, 1), seed=4)
        moved = atoms.copy()
        moved.positions += np.array([0.37, -1.1, 0.6])
        moved.info = dict(atoms.info)
        moved.arrays["REF_forces"] = np.zeros((len(moved), 3))
        assert torch.allclose(
            run(model, [atoms])["energy"], run(model, [moved])["energy"], atol=1e-10
        )

    def test_multi_carrier_scaling_is_not_hard_wired(self):
        """Delta E_SR is linear in n_c at fixed alpha and u, but both depend on the full
        counter vector, so the total need not be linear in the carrier count."""
        model = build_model()
        one = float(run(model, [make_atoms((1, 0, 0, 0))])["delta_energy"])
        two = float(run(model, [make_atoms((2, 0, 0, 0))])["delta_energy"])
        assert abs(two) > 0
        assert two != pytest.approx(2 * one, rel=1e-6)

    def test_logit_gap_is_reported_per_graph_and_channel(self):
        model = build_model()
        out = run(model, [make_atoms((1, 0, 0, 0)), make_atoms((0, 0, 0, 1), repeat=2)])
        assert out["logit_gap"].shape == (2, 4)
        assert bool((out["logit_gap"] >= 0).all())


class TestGradients:
    def finite_difference(self, model, atoms, key, index, axis, step=1e-5):
        plus, minus = atoms.copy(), atoms.copy()
        plus.info, minus.info = dict(atoms.info), dict(atoms.info)
        for image, sign in ((plus, 1.0), (minus, -1.0)):
            image.positions[index, axis] += sign * step
            image.arrays["REF_forces"] = np.zeros((len(image), 3))
        upper = float(run(model, [plus])[key])
        lower = float(run(model, [minus])[key])
        return -(upper - lower) / (2 * step)

    @pytest.mark.parametrize(
        "key,forces_key", [("energy", "forces"), ("delta_energy", "delta_forces")]
    )
    def test_autograd_matches_finite_differences(self, key, forces_key):
        model = build_model()
        atoms = make_atoms((1, 0, 0, 1), seed=5)
        out = run(model, [atoms])
        for index in range(len(atoms)):
            for axis in range(3):
                expected = self.finite_difference(model, atoms, key, index, axis)
                assert float(out[forces_key][index, axis]) == pytest.approx(
                    expected, abs=1e-6
                )

    def test_total_and_delta_forces_are_both_available_in_one_pass(self):
        """Both branches must come out of a single forward pass, since the loss needs
        them together (plan section 4)."""
        model = build_model()
        out = run(model, [make_atoms((1, 0, 0, 0))], training=True)
        assert out["forces"] is not None and out["delta_forces"] is not None
        assert out["forces"].shape == out["delta_forces"].shape

    def test_stress_is_available(self):
        model = build_model()
        out = run(
            model,
            [make_atoms((1, 0, 0, 0))],
            compute_stress=True,
        )
        assert out["stress"] is not None
        assert out["stress"].shape == (1, 3, 3)


class TestConfiguration:
    def test_separate_trunk_is_rejected_until_implemented(self):
        with pytest.raises(NotImplementedError, match="correction_trunk"):
            build_model(correction_trunk="separate")

    def test_shared_logits_across_spin_channels(self):
        model = build_model(share_logits_across_spin=True)
        assert len(model.carrier_pooling.logit_readouts) == 2
        out = run(model, [make_atoms((1, 0, 0, 0))])
        assert out["energy"] is not None

    def test_correction_starts_at_the_base_potential(self):
        """Fresh models must predict the base potential, so a warm-started base branch
        is not disturbed before the correction heads have settled."""
        torch.manual_seed(0)
        model = MACEDefect(
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
            use_long_range=False,
        )
        out = run(model, [make_atoms((1, 0, 0, 1))])
        assert float(out["delta_energy"].abs().max()) == 0.0


class TestTrainingStep:
    """The model, the loss and the optimiser have to work together before there is any
    data; this exercises the whole path on a synthetic paired batch."""

    def paired_batch(self):
        neutral = make_atoms((0, 0, 0, 0), seed=7)
        charged = make_atoms((1, 0, 0, 0), seed=7)
        for index, atoms in enumerate((neutral, charged)):
            atoms.info["pair_id"] = "g"
            atoms.info["REF_energy"] = -10.0 - index
            atoms.arrays["REF_forces"] = np.full((len(atoms), 3), 0.1 * (index + 1))
        return make_batch([neutral, charged])

    def test_loss_backward_reaches_every_branch(self):
        from mace.modules.loss import DefectLoss

        model = build_model()
        batch = self.paired_batch()
        loss_fn = DefectLoss()
        output = model(batch.to_dict(), training=True)
        loss = loss_fn(ref=batch, pred=output)
        loss.backward()

        def grad_norm(module):
            return sum(
                float(p.grad.abs().sum())
                for p in module.parameters()
                if p.grad is not None
            )

        assert grad_norm(model.carrier_pooling.energy_readouts) > 0
        assert grad_norm(model.carrier_pooling.logit_readouts) > 0
        assert grad_norm(model.counter_embedding) > 0
        # The base branch is trained by L_base, so it must receive gradient too.
        assert grad_norm(model.interactions) > 0

    def test_optimiser_step_reduces_the_loss(self):
        from mace.modules.loss import DefectLoss

        model = build_model()
        batch = self.paired_batch()
        loss_fn = DefectLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(batch.to_dict(), training=True)
            loss = loss_fn(ref=batch, pred=output)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        assert losses[-1] < losses[0]

    def test_metrics_collect_the_delta_errors(self):
        from mace.modules.loss import DefectLoss
        from mace.tools.train import MACELoss

        model = build_model()
        batch = self.paired_batch()
        metrics = MACELoss(loss_fn=DefectLoss())
        output = model(batch.to_dict(), training=True)
        metrics.update(batch, output)
        _, aux = metrics.compute()
        assert aux["rmse_delta_e"] > 0
        assert aux["rmse_delta_f"] > 0


class TestFoundationWarmStart:
    """The finetuning path used to raise for any model that adds a child module the
    foundation does not have (the documented MACELES xfail)."""

    def foundation(self):
        torch.manual_seed(0)
        return modules.ScaleShiftMACE(
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
        )

    def test_defect_model_can_be_warm_started_from_a_plain_mace(self):
        model = build_model(seed=3)
        loaded = tools.load_foundations_elements(
            model, self.foundation(), table=Z_TABLE, max_L=1
        )
        assert torch.allclose(
            loaded.interactions[0].linear_up.weight,
            self.foundation().interactions[0].linear_up.weight,
        )

    def test_correction_heads_keep_their_initialisation(self):
        """There is nothing to copy into them, and they must start at the base
        potential so the warm-started base branch is not disturbed."""
        model = build_model(seed=3)
        before = [
            p.detach().clone()
            for p in model.carrier_pooling.energy_readouts.parameters()
        ]
        tools.load_foundations_elements(
            model, self.foundation(), table=Z_TABLE, max_L=1
        )
        after = list(model.carrier_pooling.energy_readouts.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))
