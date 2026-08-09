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
    # Required for spin-polarised frames (plan 1.1.2); consistent by construction here.
    counts_array = np.asarray(counts, dtype=int)
    atoms.info["multiplicity"] = int(
        (counts_array[0] - counts_array[2]) - (counts_array[1] - counts_array[3])
    ) + 1
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


def _configs(atoms_list):
    """Configurations with the defect targets derived, for loader-level tests."""
    keyspec = data.KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "multiplicity": "multiplicity",
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
    return configs


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

    def test_correction_is_live_without_the_zero_init(self):
        """Without the zero-init, a fresh model predicts a non-zero correction at n != 0.

        This pins the *mechanism*, not the default, so it constructs the model with
        ``zero_u_init=False`` explicitly. The shipped default is ``True``: the zero-init
        is kept for optimisation conditioning, not because the ``E_total(R, 0) ==
        E_base(R)`` identity needs it -- that identity is structural via the counter
        prefactor and holds for any parameters (see TestReferenceStateIdentity), which is
        exactly why removing the zero-init on those grounds made the fit worse rather
        than better.
        """
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
            zero_u_init=False,
        )
        out = run(model, [make_atoms((1, 0, 0, 1))])
        assert float(out["delta_energy"].abs().max()) > 0.0


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

    def test_delta_forces_alone_reach_the_correction_heads(self):
        """A loss built from ``delta_forces`` *only* must train the correction.

        ``delta_forces`` is a second derivative path: it is itself produced by
        ``autograd.grad``, so it only carries gradient back to the parameters if that
        call was made with ``create_graph=True``. If it is not, the delta-force term
        silently contributes nothing and ``RMSE_dF`` never moves -- which is exactly the
        symptom under investigation (fix plan stage 0.2), and both earlier bugs in this
        family were plumbing rather than physics.

        Built with ``zero_u_init=False`` so that the logit networks have a live gradient
        to check: under the shipped zero-init they are legitimately starved at step 0
        (that is the subject of the next test), which would mask a genuine break in the
        second-derivative path.
        """
        model = build_model(zero_u_init=False)
        batch = self.paired_batch()
        output = model(batch.to_dict(), training=True)

        loss = torch.square(output["delta_forces"]).sum()
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
        assert grad_norm(model.defect_feature_readouts) > 0

    def test_zero_u_init_would_starve_the_logits(self):
        """Regression guard for the init that was removed (plan 9.6, amended).

        With ``u`` identically zero, ``Delta E_SR = sum_c n_c sum_i alpha_i^c u_i^c`` is
        zero for every ``alpha``, so its derivative with respect to the logits vanishes
        identically. Zeroing ``MLP_u`` therefore froze the attention at initialisation.
        This test pins the mechanism, so that reinstating the zero-init cannot look
        harmless.
        """
        from mace.modules.defect_blocks import zero_last_layer

        model = build_model()
        for readout in model.carrier_pooling.energy_readouts:
            zero_last_layer(readout)

        batch = self.paired_batch()
        output = model(batch.to_dict(), training=True)
        loss = torch.square(output["delta_energy"]).sum()
        loss.backward()

        logit_grad = sum(
            float(p.grad.abs().sum())
            for p in model.carrier_pooling.logit_readouts.parameters()
            if p.grad is not None
        )
        assert logit_grad == 0.0, (
            "with u == 0 the attention must be starved of gradient; if this ever "
            "becomes non-zero the diagnosis in defect_correction_branch_diagnosis.md "
            "needs revisiting"
        )

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

    def test_dead_channels_are_never_updated(self):
        """A channel with ``n_c == 0`` on every frame must not move during training.

        Two things depend on this. It is the control that makes the shrinkage of *live*
        channels interpretable -- if a never-occupied channel also drifted, the shrinkage
        would be weight decay or a leak rather than a gradient response. And it is a
        gradient-leak detector: the counter prefactor is what confines each channel to
        the frames that occupy it, so any path that bypasses it shows up here.

        The 4H-SiC dataset has exactly this situation: `h_maj` is zero in every frame.
        """
        from mace.modules.loss import DefectLoss

        model = build_model()
        batch = self.paired_batch()
        # h_maj (index 2) is zero on every frame of this batch, as in the real dataset.
        assert float(batch.carrier_counts.view(-1, 4)[:, 2].abs().sum()) == 0.0

        dead = [readout for readout in model.carrier_pooling.energy_readouts][2]
        before = [p.detach().clone() for p in dead.parameters()]

        loss_fn = DefectLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(3):
            optimizer.zero_grad()
            output = model(batch.to_dict(), training=True)
            loss_fn(ref=batch, pred=output).backward()
            optimizer.step()

        after = list(dead.parameters())
        for old, new in zip(before, after):
            assert torch.equal(old, new), (
                "a carrier channel that is unoccupied on every frame was updated; the "
                "counter prefactor is supposed to make its gradient identically zero"
            )

    def test_totals_train_the_base_branch_at_defect_geometries(self):
        """``L_tot`` must reach the base branch, and must cover paired frames too.

        Both halves matter (plan A5.3). Detaching ``E_base`` here was what left total
        forces at defect geometries supervised by nothing; and restricting the term to
        *unpaired* frames would leave the paired majority of the dataset uncovered, which
        is where the force error actually lives.
        """
        from mace.modules.loss import DefectLoss

        model = build_model()
        batch = self.paired_batch()
        # Every frame in this batch is paired and carries carriers, so under the old
        # unpaired-only mask this loss would have had no totals term at all.
        assert float(batch.carrier_counts.sum()) > 0

        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=1.0,
            delta_energy_weight=0.0,
            delta_forces_weight=0.0,
            total_energy_weight=1.0,
        )
        output = model(batch.to_dict(), training=True)
        loss = loss_fn(ref=batch, pred=output)
        assert float(loss) > 0.0, "L_tot contributed nothing on a fully paired batch"
        loss.backward()

        def grad_norm(module):
            return sum(
                float(p.grad.abs().sum())
                for p in module.parameters()
                if p.grad is not None
            )

        # `readouts` is the base-branch energy head. The trunk itself is shared with the
        # correction, so a gradient there would prove nothing; these parameters are on
        # the base energy path alone.
        assert grad_norm(model.readouts) > 0, (
            "the base branch received no gradient from L_tot; E_base is being detached"
        )

    def test_detach_flag_restores_the_old_behaviour(self):
        """The ablation switch must actually sever the base-branch gradient."""
        from mace.modules.loss import DefectLoss

        model = build_model()
        batch = self.paired_batch()
        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            delta_energy_weight=0.0,
            delta_forces_weight=0.0,
            total_energy_weight=1.0,
            detach_base_in_totals=True,
        )
        output = model(batch.to_dict(), training=True)
        loss_fn(ref=batch, pred=output).backward()
        # With forces off and the base detached, nothing reaches the base energy head.
        # The shared trunk still moves, through the correction, so it is not the thing to
        # assert on here.
        base_grad = sum(
            float(p.grad.abs().sum())
            for p in model.readouts.parameters()
            if p.grad is not None
        )
        correction_grad = sum(
            float(p.grad.abs().sum())
            for p in model.carrier_pooling.parameters()
            if p.grad is not None
        )
        assert base_grad == 0.0
        assert correction_grad > 0.0

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


class TestTiedAttention:
    """Plan stage D: alpha = softmax(-beta u), derived from the site energy itself."""

    @staticmethod
    def tied_model(**kwargs):
        return build_model(alpha_mode="tied", beta=10.0, **kwargs)

    def test_no_logit_network_exists(self):
        """The tie is meant to remove MLP_l, not merely bypass it."""
        model = self.tied_model()
        assert len(model.carrier_pooling.logit_readouts) == 0
        assert (
            sum(p.numel() for p in model.carrier_pooling.logit_readouts.parameters())
            == 0
        )

    def test_alpha_is_softmax_of_minus_beta_u(self):
        """The defining relation, checked against a direct computation."""
        model = self.tied_model()
        out = run(model, [make_atoms((1, 1, 0, 2))])
        alpha = out["carrier_alpha"]
        u = out["carrier_readouts"]
        expected = torch.softmax(-model.carrier_pooling.beta * u, dim=0)
        assert torch.allclose(alpha, expected, atol=1e-6)

    def test_reference_state_identity_survives_the_tie(self):
        """The n = 0 identity is structural and must not depend on how alpha is made."""
        model = self.tied_model()
        out = run(model, [make_atoms((0, 0, 0, 0))])
        assert torch.allclose(out["energy"], out["base_energy"], atol=0.0, rtol=0.0)
        assert float(out["delta_energy"].abs().max()) == 0.0

    def test_attention_still_sums_to_one(self):
        model = self.tied_model()
        out = run(model, [make_atoms((1, 0, 0, 1)), make_atoms((1, 1, 0, 2), seed=3)])
        batch = make_batch([make_atoms((1, 0, 0, 1)), make_atoms((1, 1, 0, 2), seed=3)])
        from mace.tools.scatter import scatter_sum

        totals = scatter_sum(
            out["carrier_alpha"], batch.batch, dim=0, dim_size=int(batch.num_graphs)
        )
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-9)

    def test_correction_is_live_at_zero_u(self):
        """The tie needs no seeding: d(Delta E)/du = n_c alpha_j at u == 0.

        This is why the zero-init is safe here even though it starves the *untied*
        attention -- there is no separate attention left to starve.
        """
        from mace.modules.defect_blocks import zero_last_layer

        model = self.tied_model()
        for readout in model.carrier_pooling.energy_readouts:
            zero_last_layer(readout)
        batch = make_batch([make_atoms((1, 1, 0, 2))])
        out = model(batch.to_dict(), training=True)
        # A *linear* functional, deliberately. delta_energy is exactly zero at u == 0, so
        # a squared loss would have zero gradient by construction and the test would pass
        # or fail for reasons unrelated to the mechanism being probed.
        out["delta_energy"].sum().backward()
        grad = sum(
            float(p.grad.abs().sum())
            for p in model.carrier_pooling.energy_readouts.parameters()
            if p.grad is not None
        )
        assert grad > 0.0

    def test_delta_u_equals_logit_gap_over_beta(self):
        """Under the tie the reported gap is a physical energy, not a logit scale."""
        model = self.tied_model()
        out = run(model, [make_atoms((1, 1, 0, 2))])
        beta = model.carrier_pooling.beta
        assert torch.allclose(
            out["logit_gap"], beta * out["delta_u"], atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("counts", [(1, 0, 0, 1), (1, 1, 0, 2)])
    def test_forces_match_finite_differences(self, counts):
        """Forces now carry a -beta n_c Cov_alpha(u, grad u) term; check them numerically."""
        # Save and restore, never hardcode: this module sets float64 at import, so a
        # finally that forces float32 silently corrupts every test that runs after it.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            model = self.tied_model().double()
            atoms = make_atoms(counts, seed=5)
            batch = make_batch([atoms])
            out = model(batch.to_dict(), training=True)
            analytic = out["delta_forces"].detach().clone()

            step = 1e-5
            numerical = torch.zeros_like(analytic)
            for i in range(len(atoms)):
                for axis in range(3):
                    shifted = []
                    for sign in (+1, -1):
                        moved = atoms.copy()
                        moved.info.update(atoms.info)
                        moved.arrays["REF_forces"] = atoms.arrays["REF_forces"]
                        positions = moved.get_positions()
                        positions[i, axis] += sign * step
                        moved.set_positions(positions)
                        energy = model(
                            make_batch([moved]).to_dict(), training=False
                        )["delta_energy"]
                        shifted.append(float(energy))
                    numerical[i, axis] = -(shifted[0] - shifted[1]) / (2 * step)
            assert torch.allclose(analytic, numerical, atol=1e-6, rtol=1e-4)
        finally:
            torch.set_default_dtype(previous)


class TestLogitSeeding:
    """Plan D7.1, logit route: logit_i^c += gamma_c * s_hat_i."""

    def test_seed_is_live_and_gamma_is_trainable(self):
        model = build_model(logit_seed_gamma=1.5)
        assert model.logit_seed_gamma.requires_grad
        assert model.logit_seed_gamma.shape == (NUM_CARRIER_CHANNELS,)
        out = run(model, [make_atoms((1, 1, 0, 2), seed=4)])
        assert float(out["delta_energy"].abs().max()) > 0.0

    def test_reference_state_identity_survives(self):
        """The seed is an energy term, so the n = 0 identity has to be rechecked."""
        model = build_model(logit_seed_gamma=1.5)
        out = run(model, [make_atoms((0, 0, 0, 0), seed=4)])
        assert torch.allclose(out["energy"], out["base_energy"], atol=0.0, rtol=0.0)
        assert float(out["delta_energy"].abs().max()) == 0.0

    def test_forces_differentiate_the_descriptor(self):
        """The trap: a cached descriptor would silently break force consistency.

        s_hat enters the energy through the attention, so it must be differentiated with
        respect to the positions. It is rebuilt in-graph from the trunk's edge basis for
        exactly this reason, and this test is what says so.
        """
        # Save and restore, never hardcode: this module sets float64 at import, so a
        # finally that forces float32 silently corrupts every test that runs after it.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            model = build_model(logit_seed_gamma=1.5).double()
            atoms = make_atoms((1, 1, 0, 2), seed=4, rattle=0.2)
            analytic = model(make_batch([atoms]).to_dict(), training=True)["delta_forces"]
            analytic = analytic.detach().clone()

            step = 1e-5
            numerical = torch.zeros_like(analytic)
            for i in range(len(atoms)):
                for axis in range(3):
                    shifted = []
                    for sign in (+1, -1):
                        moved = atoms.copy()
                        moved.info.update(atoms.info)
                        moved.arrays["REF_forces"] = atoms.arrays["REF_forces"]
                        positions = moved.get_positions()
                        positions[i, axis] += sign * step
                        moved.set_positions(positions)
                        shifted.append(
                            float(
                                model(make_batch([moved]).to_dict(), training=False)[
                                    "delta_energy"
                                ]
                            )
                        )
                    numerical[i, axis] = -(shifted[0] - shifted[1]) / (2 * step)
            assert torch.allclose(analytic, numerical, atol=1e-6, rtol=1e-4)
        finally:
            torch.set_default_dtype(previous)

    def test_gamma_zero_reproduces_the_unseeded_model(self):
        """Disabling the seed must not perturb anything else."""
        seeded = build_model(seed=3, logit_seed_gamma=0.0)
        plain = build_model(seed=3)
        atoms = [make_atoms((1, 0, 0, 1), seed=6)]
        assert torch.allclose(
            run(seeded, atoms)["delta_energy"],
            run(plain, atoms)["delta_energy"],
            atol=0.0,
            rtol=0.0,
        )

    def test_anneal_reaches_exactly_zero_by_the_configured_epoch(self):
        """The schedule is keyed to an absolute epoch, not a fraction of the run.

        Under early stopping the run length is not known in advance, so a fractional
        schedule can let a model converge and stop with the seed still active -- shipping
        an inference-time descriptor, which is the one outcome the anneal exists to
        prevent.
        """
        from mace.modules.defect_seed import anneal_logit_seed, calibrate_novelty
        from mace.tools import torch_geometric

        model = build_model(logit_seed_gamma=1.5)
        dataset = [
            data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)
            for config in _configs([make_atoms((1, 1, 0, 2), seed=s) for s in (0, 1)])
        ]
        loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset, batch_size=2, shuffle=False
        )
        calibrate_novelty(model, loader, torch.device("cpu"))
        gamma_init = model.logit_seed_gamma.detach().clone()

        history = []
        for epoch in range(12):
            anneal_logit_seed(
                model=model,
                data_loader=loader,
                device=torch.device("cpu"),
                epoch=epoch,
                zero_by_epoch=10,
                gamma_init=gamma_init,
            )
            history.append(model.logit_seed_gamma.detach().clone())

        assert float(history[-1].abs().max()) == 0.0, "gamma must be exactly zero"
        assert float(history[10].abs().max()) == 0.0, "zero at the configured epoch"
        # Monotone non-increasing per channel: a hand-over, not an oscillation.
        for earlier, later in zip(history, history[1:]):
            assert torch.all(later <= earlier + 1e-9)

    def test_annealed_model_is_bias_free(self):
        """Once gamma is zero the model must be identical to an unseeded one.

        This is what makes the converged model safe to ship: no descriptor to carry, no
        derivative term to forget, and the production finite-difference check unambiguous.
        """
        seeded = build_model(seed=5, logit_seed_gamma=1.5)
        with torch.no_grad():
            seeded.logit_seed_gamma.zero_()
        plain = build_model(seed=5)
        atoms = [make_atoms((1, 1, 0, 2), seed=7)]
        assert torch.allclose(
            run(seeded, atoms)["delta_energy"],
            run(plain, atoms)["delta_energy"],
            atol=0.0,
            rtol=0.0,
        )

    def test_anneal_survives_an_optimiser_that_also_moves_gamma(self):
        """gamma must end at exactly zero even if something else is updating it.

        Left trainable during annealing, the optimizer moves gamma after each epoch's
        schedule update and the ratchet locks in the result -- observed finishing at
        -0.092 on one channel, a live and sign-inverted bias. This test pins the
        invariant that matters: after the schedule reaches its terminal epoch, gamma is
        zero regardless of what else touched it.
        """
        from mace.modules.defect_seed import anneal_logit_seed, calibrate_novelty
        from mace.tools import torch_geometric

        model = build_model(logit_seed_gamma=1.5)
        dataset = [
            data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)
            for config in _configs([make_atoms((1, 1, 0, 2), seed=s) for s in (0, 1)])
        ]
        loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset, batch_size=2, shuffle=False
        )
        calibrate_novelty(model, loader, torch.device("cpu"))
        gamma_init = model.logit_seed_gamma.detach().clone()

        for epoch in range(9):
            anneal_logit_seed(
                model=model,
                data_loader=loader,
                device=torch.device("cpu"),
                epoch=epoch,
                zero_by_epoch=6,
                gamma_init=gamma_init,
            )
            # Stand in for the optimizer: perturb gamma between schedule updates.
            with torch.no_grad():
                model.logit_seed_gamma.add_(
                    torch.tensor([0.05, -0.05, 0.05, -0.05])
                )

        anneal_logit_seed(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            epoch=9,
            zero_by_epoch=6,
            gamma_init=gamma_init,
        )
        assert float(model.logit_seed_gamma.abs().max()) == 0.0


class TestLongRangeBranch:
    """Plan stage E: the long-range branch, with the screening amplitude frozen."""

    def test_freeze_amplitude_actually_freezes_it(self):
        """A construction flag that silently fails to arrive is worse than no flag.

        This was real: the CLI parsed `--freeze_amplitude=True`, the value reached
        `args`, and a no-op edit meant it never reached the constructor -- so stage E
        trained a *fitted* `a` while every log said it was frozen. The two seeds then
        disagreed by a factor of 1.7 in `1/a^2`, which is the unidentifiability the freeze
        exists to prevent, discovered only by reading the saved weights.
        """
        model = build_model(use_long_range=True, eps_inf_init=6.5, freeze_amplitude=True)
        assert model.freeze_amplitude is True
        amplitude = model.latent_charges.amplitude
        assert not any(p.requires_grad for p in amplitude.parameters())

        out = run(model, [make_atoms((1, 0, 0, 1), seed=2)])
        expected = 1.0 / 6.5**0.5
        assert torch.allclose(
            out["screening_amplitude"],
            torch.full_like(out["screening_amplitude"], expected),
            atol=1e-6,
        ), "a must sit exactly at 1/sqrt(eps_inf)"

    def test_amplitude_is_trainable_when_not_frozen(self):
        model = build_model(use_long_range=True, eps_inf_init=6.5, freeze_amplitude=False)
        assert model.freeze_amplitude is False
        assert any(p.requires_grad for p in model.latent_charges.amplitude.parameters())

    def test_latent_charge_invariants(self):
        """sum_i q_host = sum_i q_pol = 0 per cell, and sum_i q_i = a*q."""
        from mace.tools.scatter import scatter_sum

        model = build_model(use_long_range=True, eps_inf_init=6.5, freeze_amplitude=True)
        atoms = [make_atoms((1, 0, 0, 1), seed=2), make_atoms((1, 1, 0, 2), seed=3)]
        batch = make_batch(atoms)
        out = model(batch.to_dict(), training=True)
        num_graphs = int(batch.num_graphs)

        host = scatter_sum(
            out["latent_charges_host"], batch.batch, dim=0, dim_size=num_graphs
        )
        assert torch.allclose(host, torch.zeros_like(host), atol=1e-6)

        counts = batch.carrier_counts.view(num_graphs, -1)
        charge = counts[:, 2] + counts[:, 3] - counts[:, 0] - counts[:, 1]
        total = scatter_sum(
            out["latent_charges"], batch.batch, dim=0, dim_size=num_graphs
        )
        expected = out["screening_amplitude"] * charge
        assert torch.allclose(total, expected, atol=1e-6)
