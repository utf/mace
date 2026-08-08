"""Inference-side behaviour of MACEDefect through the ASE calculator (plan sections 7, 8).

The two options that matter are ``energy_scale`` -- which decides whether the affine map
back to raw energies is applied, and where its constants come from -- and ``dilute``,
which selects the isolated-limit evaluator. Counter canonicalisation must happen here
too: the calculator is an entry point like any other.
"""

import json

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import modules, tools
from mace.calculators import MACECalculator
from mace.modules.defect_models import MACEDefect

torch.set_default_dtype(torch.float64)

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0
CBM_CELL = -3.2
VBM_CELL = -6.85
REGISTRY = {"GaN|4": {"e_cbm_cell": CBM_CELL, "e_vbm_cell": VBM_CELL}}


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
    generator = torch.Generator().manual_seed(1)
    for readout in model.carrier_pooling.energy_readouts:
        for parameter in readout.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    return model


def make_atoms(counts, host="GaN"):
    atoms = Atoms(
        numbers=[6, 14, 6, 14],
        positions=np.array(
            [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [0.0, 0.0, 2.1]]
        ),
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    atoms.info["carrier_counts"] = np.asarray(counts, dtype=int)
    if host is not None:
        atoms.info["host"] = host
    return atoms


def calculator(model=None, **kwargs):
    return MACECalculator(models=model or build_model(), device="cpu", **kwargs)


class TestCounterConditioning:
    def test_energy_depends_on_the_counters(self):
        calc = calculator()
        neutral = make_atoms((0, 0, 0, 0))
        charged = make_atoms((1, 0, 0, 0))
        neutral.calc = calc
        charged.calc = calc
        assert neutral.get_potential_energy() != charged.get_potential_energy()

    def test_counters_are_canonicalised_at_this_entry_point(self):
        """(0,1,0,0) and (1,0,0,0) are the same physical state; the calculator must not
        be a hole through which a non-canonical vector reaches the network."""
        calc = calculator()
        canonical = make_atoms((1, 0, 0, 0))
        swapped = make_atoms((0, 1, 0, 0))
        canonical.calc = calc
        swapped.calc = calc
        assert canonical.get_potential_energy() == pytest.approx(
            swapped.get_potential_energy(), abs=1e-12
        )

    def test_open_shell_singlet_labellings_agree(self):
        calc = calculator()
        first = make_atoms((1, 0, 1, 0))
        second = make_atoms((0, 1, 0, 1))
        first.calc = calc
        second.calc = calc
        assert first.get_potential_energy() == pytest.approx(
            second.get_potential_energy(), abs=1e-12
        )

    def test_missing_counters_are_treated_as_the_reference_state(self):
        calc = calculator()
        bare = make_atoms((0, 0, 0, 0))
        del bare.info["carrier_counts"]
        neutral = make_atoms((0, 0, 0, 0))
        bare.calc = calc
        neutral.calc = calc
        assert bare.get_potential_energy() == pytest.approx(
            neutral.get_potential_energy(), abs=1e-12
        )

    def test_invalid_counters_are_rejected(self):
        calc = calculator()
        atoms = make_atoms((1, 0, 0, 0))
        atoms.info["carrier_counts"] = np.array([-1, 0, 0, 0])
        atoms.calc = calc
        with pytest.raises(ValueError, match="non-negative"):
            atoms.get_potential_energy()


class TestEnergyScale:
    def test_referenced_is_the_default(self):
        calc = calculator()
        assert calc.energy_scale == "referenced"

    def test_raw_scale_undoes_the_band_edge_referencing(self):
        atoms = make_atoms((1, 0, 0, 0))
        referenced = calculator()
        raw = calculator(band_edges=REGISTRY, energy_scale="raw")

        atoms.calc = referenced
        referenced_energy = atoms.get_potential_energy()
        atoms = make_atoms((1, 0, 0, 0))
        atoms.calc = raw
        raw_energy = atoms.get_potential_energy()
        # E_raw = E_target + n_e * E_CBM
        assert raw_energy == pytest.approx(referenced_energy + CBM_CELL, abs=1e-9)

    def test_raw_scale_is_a_no_op_at_the_reference_state(self):
        atoms = make_atoms((0, 0, 0, 0))
        referenced = calculator()
        atoms.calc = referenced
        referenced_energy = atoms.get_potential_energy()

        atoms = make_atoms((0, 0, 0, 0))
        atoms.calc = calculator(band_edges=REGISTRY, energy_scale="raw")
        assert atoms.get_potential_energy() == pytest.approx(
            referenced_energy, abs=1e-12
        )

    def test_forces_are_identical_under_either_scale(self):
        """The referencing term is a per-configuration constant, so relaxation and MD do
        not care which scale is selected."""
        first = make_atoms((0, 0, 0, 1))
        first.calc = calculator()
        second = make_atoms((0, 0, 0, 1))
        second.calc = calculator(band_edges=REGISTRY, energy_scale="raw")
        assert np.allclose(first.get_forces(), second.get_forces(), atol=1e-12)

    def test_raw_scale_reads_the_registry_shipped_with_the_model(self):
        model = build_model()
        model.band_edge_registry = REGISTRY
        atoms = make_atoms((1, 0, 0, 0))
        atoms.calc = calculator(model=model, energy_scale="raw")
        assert np.isfinite(atoms.get_potential_energy())

    def test_raw_scale_without_any_table_is_refused(self):
        with pytest.raises(ValueError, match="band edges used in training"):
            calculator(energy_scale="raw")

    def test_missing_registry_entry_is_an_error_not_a_silent_zero(self):
        """The failure this guards against is a uniform shift of every transition level,
        which nothing downstream can see."""
        atoms = make_atoms((1, 0, 0, 0), host="ZnO")
        atoms.calc = calculator(band_edges=REGISTRY, energy_scale="raw")
        with pytest.raises(KeyError, match="no band edges recorded"):
            atoms.get_potential_energy()

    def test_band_edges_can_be_given_as_a_json_file(self, tmp_path):
        path = tmp_path / "edges.json"
        path.write_text(
            json.dumps({"GaN|4": {"e_cbm_cell": CBM_CELL, "e_vbm_cell": VBM_CELL}}),
            encoding="utf-8",
        )
        atoms = make_atoms((1, 0, 0, 0))
        atoms.calc = calculator(band_edges=str(path), energy_scale="raw")
        assert np.isfinite(atoms.get_potential_energy())

    def test_invalid_scale_is_rejected(self):
        with pytest.raises(ValueError, match="energy_scale"):
            calculator(energy_scale="nonsense")


class TestOptionGuards:
    def plain_mace(self):
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

    @pytest.mark.parametrize("options", [{"dilute": True}, {"energy_scale": "raw"}])
    def test_defect_options_are_refused_for_plain_models(self, options):
        with pytest.raises(ValueError, match="only meaningful for MACEDefect"):
            MACECalculator(models=self.plain_mace(), device="cpu", **options)

    def test_plain_models_still_work_unchanged(self):
        atoms = make_atoms((0, 0, 0, 0))
        atoms.calc = MACECalculator(models=self.plain_mace(), device="cpu")
        assert np.isfinite(atoms.get_potential_energy())

    def test_dilute_needs_the_long_range_branch(self):
        """Without it there is no carrier charge distribution to change the boundary
        condition on, so the flag would silently do nothing."""
        atoms = make_atoms((1, 0, 0, 0))
        calc = calculator(dilute=True)
        atoms.calc = calc
        # Short-range only: the dilute correction is absent, so this must equal the
        # periodic result rather than pretend to be an isolated-limit number.
        periodic = make_atoms((1, 0, 0, 0))
        periodic.calc = calculator()
        assert atoms.get_potential_energy() == pytest.approx(
            periodic.get_potential_energy(), abs=1e-12
        )


class TestResultCaching:
    def test_changing_only_the_counters_invalidates_the_cache(self):
        """ASE compares geometry, not atoms.info, so without an override a shared
        calculator would hand back the first charge state's energy for the second."""
        calc = calculator()
        atoms = make_atoms((0, 0, 0, 0))
        atoms.calc = calc
        neutral = atoms.get_potential_energy()

        atoms.info["carrier_counts"] = np.array([1, 0, 0, 0])
        charged = atoms.get_potential_energy()
        assert neutral != charged

    def test_repeated_evaluation_of_the_same_state_is_stable(self):
        calc = calculator()
        atoms = make_atoms((1, 0, 0, 0))
        atoms.calc = calc
        assert atoms.get_potential_energy() == atoms.get_potential_energy()

    def test_configuration_coordinate_scan_gives_distinct_curves(self):
        """The workflow the cache bug would have broken: two charge states evaluated
        along one set of geometries through a single calculator."""
        calc = calculator()
        curves = {(0, 0, 0, 0): [], (1, 0, 0, 0): []}
        for displacement in (0.0, 0.05, 0.1):
            for counts in curves:
                atoms = make_atoms(counts)
                atoms.positions[0, 0] += displacement
                atoms.calc = calc
                curves[counts].append(atoms.get_potential_energy())
        assert not np.allclose(curves[(0, 0, 0, 0)], curves[(1, 0, 0, 0)])
