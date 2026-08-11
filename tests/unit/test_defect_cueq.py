"""cuEquivariance conversion of MACEDefect must not change any output.

The conversion rebuilds the model with ``cueq_config`` set and transfers weights. For
MACEDefect the correction heads are plain dense MLPs, so nothing there is converted -- but
``defect_feature_readouts`` *are* e3nn ``LinearReadoutBlock``s reading the trunk irreps,
and those are rebuilt under a different internal layout. This test is what says the
weight transfer got that right; without it, a layout mismatch would show up as a silently
wrong correction rather than an error.
"""

import copy

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules, tools
from mace.modules.defect_models import MACEDefect

pytest.importorskip("cuequivariance_torch")

# cuequivariance initialises CUDA even when the conversion targets the CPU, so these
# cannot run on a CPU-only machine (or one whose GPU is already full).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuequivariance requires a CUDA device"
)

from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq  # noqa: E402

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0


def build_model(seed: int = 0, **overrides) -> MACEDefect:
    torch.manual_seed(seed)
    arguments = dict(
        r_max=CUTOFF,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(Z_TABLE),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, len(Z_TABLE))),
        avg_num_neighbors=8.0,
        atomic_numbers=Z_TABLE.zs,
        correlation=3,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        carrier_feature_dim=16,
        counter_embedding_dim=8,
        carrier_mlp_hidden=16,
        use_long_range=False,
        # The correction must be non-trivial, or the test would pass on zeros.
        zero_u_init=False,
    )
    arguments.update(overrides)
    return MACEDefect(**arguments)


def make_batch(counts):
    from ase.atoms import Atoms

    from mace import data
    from mace.data.defects import prepare_defect_configurations
    from mace.tools import torch_geometric

    rng = np.random.default_rng(0)
    atoms_list = []
    for count in counts:
        atoms = Atoms(
            numbers=[6, 14, 6, 14],
            positions=np.array(
                [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [0.0, 0.0, 2.1]]
            )
            + rng.normal(scale=0.1, size=(4, 3)),
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        atoms.info["REF_energy"] = 0.0
        atoms.info["carrier_counts"] = np.asarray(count, dtype=int)
        array = np.asarray(count, dtype=int)
        atoms.info["multiplicity"] = (
            int((array[0] - array[2]) - (array[1] - array[3])) + 1
        )
        atoms.info["e_cbm_cell"] = -3.2
        atoms.info["e_vbm_cell"] = -6.85
        atoms.arrays["REF_forces"] = np.zeros((4, 3))
        atoms_list.append(atoms)

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
        data.config_from_atoms(a, key_specification=keyspec) for a in atoms_list
    ]
    prepare_defect_configurations(configs)
    dataset = [
        data.AtomicData.from_config(c, z_table=Z_TABLE, cutoff=CUTOFF) for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False
    )
    return next(iter(loader))


class TestDefectCueqEquivalence:
    def test_outputs_match_e3nn(self):
        torch.set_default_dtype(torch.float64)
        model = build_model().double()
        converted = run_e3nn_to_cueq(model, device="cpu").double()

        batch = make_batch([(1, 0, 0, 1), (1, 1, 0, 2)])
        reference = model(batch.to_dict(), training=False)
        result = converted(batch.to_dict(), training=False)

        for key in ("energy", "base_energy", "delta_energy", "forces", "delta_forces"):
            assert reference[key] is not None
            assert torch.allclose(
                reference[key], result[key], atol=1e-8, rtol=1e-8
            ), f"cueq conversion changed '{key}'"

    def test_the_correction_survives_conversion(self):
        """Guards against a conversion that silently zeroes the correction branch."""
        torch.set_default_dtype(torch.float64)
        converted = run_e3nn_to_cueq(build_model().double(), device="cpu").double()
        out = converted(make_batch([(1, 1, 0, 2)]).to_dict(), training=False)
        assert float(out["correction_energy"].abs().max()) > 0.0


class TestConversionPreservesConstructorArguments:
    """The conversion rebuilds the model, so the extracted config is load-bearing.

    ``run_e3nn_to_cueq`` does not wrap the model, it calls
    ``source.__class__(**extract_config_mace_model(source))`` and transfers weights. Any
    constructor argument the extractor omits therefore reverts to its **default** in the
    converted model, silently and with no shape mismatch to catch it.

    That is not hypothetical: ``freeze_amplitude`` was omitted, and it defaults to False
    while stage E passes True, so converting a frozen-amplitude long-range model returned
    one whose screening amplitude was trainable again. The amplitude is meant to be an
    input gauge (``a = 1/sqrt(eps_inf)``); once trainable it drifts to absorb the
    electron-hole energy and is then indistinguishable from a fitted screening constant.
    """

    @pytest.mark.parametrize(
        "name, value",
        [
            ("freeze_amplitude", True),
            ("high_precision_softmax", False),
            ("zero_u_init", True),
            ("correction_trunk", "shared"),
            # A buffer rather than a plain attribute, so it also has to survive the
            # state-dict transfer with the right shape -- if the rebuilt model allocated
            # the empty default the load would fail or the gauge probe would come back
            # silently switched off.
            ("gauge_counters", [[1, 0, 0, 1], [1, 1, 0, 2]]),
            # The q^pol controls. use_polarisation=False is the case that matters most:
            # it defaults to True, so an extractor that dropped it would rebuild an
            # ABLATED model with the channel switched back on -- the same silent revert
            # that turned a frozen screening amplitude into a trainable one.
            ("use_polarisation", False),
            ("pol_gate", True),
            ("pol_gate_lambda", 5.5),
            ("pol_gate_hops", 3),
        ],
    )
    def test_argument_survives_extraction(self, name, value):
        from mace.tools.scripts_utils import extract_config_mace_model

        model = build_model(use_long_range=True, **{name: value})
        assert extract_config_mace_model(model)[name] == value

    def test_gauge_probe_survives_conversion(self):
        counters = [[1, 0, 0, 1], [1, 1, 0, 2]]
        model = build_model(use_long_range=True, gauge_counters=counters)
        converted = run_e3nn_to_cueq(copy.deepcopy(model), device="cuda")
        assert converted.gauge_counters.detach().cpu().tolist() == counters
        # The batch has to follow the model onto the device. This test is CUDA-gated, so
        # the omission only surfaced once a run happened on a machine with a GPU.
        batch = make_batch([(0, 0, 0, 0)])
        batch = batch.to("cuda")
        out = converted(batch.to_dict(), training=False)
        assert out["gauge_mean_u"] is not None
        assert out["gauge_mean_u"].shape[1] == len(counters)

    def test_frozen_amplitude_stays_frozen_through_conversion(self):
        model = build_model(use_long_range=True, freeze_amplitude=True)
        amplitude = list(model.latent_charges.amplitude.modules())[-1].bias
        assert not amplitude.requires_grad, "precondition: source amplitude is frozen"
        before = float(torch.nn.functional.softplus(amplitude))

        converted = run_e3nn_to_cueq(copy.deepcopy(model), device="cuda")

        assert converted.freeze_amplitude is True
        assert converted.use_long_range is True
        after_bias = list(converted.latent_charges.amplitude.modules())[-1].bias
        assert not after_bias.requires_grad, "amplitude became trainable via conversion"
        assert float(torch.nn.functional.softplus(after_bias)) == pytest.approx(
            before, abs=1e-7
        )
