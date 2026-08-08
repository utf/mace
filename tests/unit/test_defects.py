"""Unit tests for mace/data/defects.py (charge-aware defect data).

Covers the carrier counter algebra and canonicalisation of the plan's section 2.1,
bounded enumeration (section 7.3), band-edge referencing of energy labels
(section 2.3), the pair join and the base/delta target construction (section 4),
and the round-trip of the derived fields into AtomicData. No model, no LES, no
network.
"""

import itertools
import json

import numpy as np
import pytest
from ase.atoms import Atoms

from mace.data.atomic_data import AtomicData
from mace.data.defects import (
    BandEdges,
    canonicalise_counts,
    enumerate_counts,
    load_band_edges,
    net_charge,
    prepare_defect_configurations,
    referencing_constant,
    spin_magnetisation,
    swap_spin_channels,
    validate_counts,
)
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable

# (state, counts, q, M_s) from the worked examples in section 2.1 of the plan.
WORKED_EXAMPLES = [
    ("closed-shell neutral", (0, 0, 0, 0), 0, 0),
    ("donor q=+1", (0, 0, 0, 1), 1, 1),
    ("acceptor q=-1", (1, 0, 0, 0), -1, 1),
    ("neutral triplet", (1, 0, 0, 1), 0, 2),
    ("neutral open-shell singlet", (1, 0, 1, 0), 0, 0),
    ("q=+2 M_s=0", (0, 0, 1, 1), 2, 0),
    ("q=+2 M_s=2", (0, 0, 0, 2), 2, 2),
]

EDGES = BandEdges(e_cbm_cell=-3.20, e_vbm_cell=-6.85)


def keyspec():
    return KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "stress": "REF_stress",
            "head": "head",
            "carrier_counts": "carrier_counts",
            "host": "host",
            "pair_id": "pair_id",
            "multiplicity": "multiplicity",
            "e_cbm_cell": "e_cbm_cell",
            "e_vbm_cell": "e_vbm_cell",
        },
        arrays_keys={"forces": "REF_forces"},
    )


def make_config(counts, energy, forces=None, positions=None, **info):
    positions = (
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        if positions is None
        else np.asarray(positions)
    )
    atoms = Atoms(numbers=[31, 7], positions=positions, cell=np.eye(3) * 6.0, pbc=True)
    atoms.info["REF_energy"] = energy
    atoms.info["carrier_counts"] = np.asarray(counts, dtype=int)
    atoms.arrays["REF_forces"] = (
        np.zeros((2, 3)) if forces is None else np.asarray(forces, dtype=float)
    )
    atoms.info.update(info)
    return config_from_atoms(atoms, key_specification=keyspec())


class TestCounterAlgebra:
    @pytest.mark.parametrize("name,counts,charge,magnetisation", WORKED_EXAMPLES)
    def test_worked_examples(self, name, counts, charge, magnetisation):
        counts = np.array(counts)
        assert net_charge(counts) == charge, name
        assert spin_magnetisation(counts) == magnetisation, name

    @pytest.mark.parametrize("name,counts,charge,magnetisation", WORKED_EXAMPLES)
    def test_worked_examples_are_canonical(self, name, counts, charge, magnetisation):
        assert np.array_equal(canonicalise_counts(counts), np.array(counts)), name

    def test_swap_preserves_charge_and_flips_magnetisation(self):
        for counts in itertools.product(range(3), repeat=4):
            counts = np.array(counts)
            swapped = swap_spin_channels(counts)
            assert net_charge(swapped) == net_charge(counts)
            assert spin_magnetisation(swapped) == -spin_magnetisation(counts)

    def test_canonicalisation_is_time_reversal_invariant(self):
        """The property the whole scheme rests on: both labellings of a physical state
        must reach the network as the same input."""
        for counts in itertools.product(range(4), repeat=4):
            counts = np.array(counts)
            assert np.array_equal(
                canonicalise_counts(counts),
                canonicalise_counts(swap_spin_channels(counts)),
            )

    def test_canonicalisation_is_idempotent_and_charge_preserving(self):
        for counts in itertools.product(range(3), repeat=4):
            canonical = canonicalise_counts(counts)
            assert np.array_equal(canonicalise_counts(canonical), canonical)
            assert net_charge(canonical) == net_charge(np.array(counts))
            assert spin_magnetisation(canonical) >= 0

    def test_magnetisation_clause_precedes_lexicographic_tie_break(self):
        """A bare lexicographic rule would pick (0,0,1,0) and hand back M_s = -1."""
        assert np.array_equal(canonicalise_counts((0, 0, 1, 0)), np.array([0, 0, 0, 1]))

    def test_rejects_malformed_counters(self):
        with pytest.raises(ValueError, match="non-negative"):
            canonicalise_counts((-1, 0, 0, 0))
        with pytest.raises(ValueError, match="integers"):
            canonicalise_counts((0.5, 0, 0, 0))
        with pytest.raises(ValueError, match="4 entries"):
            canonicalise_counts((0, 0, 0))

    def test_validate_rejects_non_canonical(self):
        with pytest.raises(ValueError, match="not canonical"):
            validate_counts((0, 1, 0, 0))

    def test_validate_multiplicity_cross_check(self):
        validate_counts((1, 0, 0, 1), multiplicity=3)  # triplet
        with pytest.raises(ValueError, match="multiplicity"):
            validate_counts((1, 0, 0, 1), multiplicity=1)


class TestEnumeration:
    @pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
    def test_candidates_are_canonical_and_at_the_right_charge(self, charge):
        candidates = enumerate_counts(charge)
        assert candidates
        for counts in candidates:
            assert net_charge(counts) == charge
            assert np.array_equal(canonicalise_counts(counts), counts)
        keys = [tuple(c.tolist()) for c in candidates]
        assert len(keys) == len(set(keys))

    def test_neutral_set_contains_the_three_physical_singlet_triplet_states(self):
        keys = {tuple(c.tolist()) for c in enumerate_counts(0)}
        assert (0, 0, 0, 0) in keys  # closed-shell singlet
        assert (1, 0, 1, 0) in keys  # open-shell singlet
        assert (1, 0, 0, 1) in keys  # triplet

    def test_bound_is_respected(self):
        for counts in enumerate_counts(1, max_carriers=1):
            assert counts.sum() <= 1


class TestBandEdges:
    def test_load_table(self, tmp_path):
        path = tmp_path / "edges.json"
        path.write_text(
            json.dumps({"GaN_216": {"e_cbm_cell": -3.2, "e_vbm_cell": -6.85}}),
            encoding="utf-8",
        )
        table = load_band_edges(path)
        assert table["GaN_216"] == BandEdges(-3.2, -6.85)

    def test_load_table_rejects_incomplete_entry(self, tmp_path):
        path = tmp_path / "edges.json"
        path.write_text(json.dumps({"GaN": {"e_cbm_cell": -3.2}}), encoding="utf-8")
        with pytest.raises(ValueError, match="e_cbm_cell"):
            load_band_edges(path)

    def test_referencing_constant_is_linear_in_the_counters(self):
        assert referencing_constant((1, 0, 0, 0), EDGES) == pytest.approx(-3.20)
        assert referencing_constant((0, 0, 0, 1), EDGES) == pytest.approx(6.85)
        assert referencing_constant((1, 0, 0, 1), EDGES) == pytest.approx(3.65)
        assert referencing_constant((2, 0, 0, 0), EDGES) == pytest.approx(-6.40)

    def test_band_edge_carrier_references_to_exactly_zero(self):
        """On a pristine host the carrier sits at the band edge, so Delta E must vanish
        identically -- the check in the table of section 2.3."""
        pristine = -1000.0
        for counts in [(0, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
            raw = pristine + referencing_constant(counts, EDGES)
            configs = [
                make_config((0, 0, 0, 0), pristine, pair_id="p", host="GaN"),
                make_config(counts, raw, pair_id="p", host="GaN"),
            ]
            prepare_defect_configurations(configs, band_edges={"GaN": EDGES})
            assert configs[1].properties["delta_energy"] == pytest.approx(0.0, abs=1e-9)


class TestPrepareConfigurations:
    def make_pair(self, **overrides):
        neutral = make_config(
            (0, 0, 0, 0), -1234.50, forces=np.full((2, 3), 0.1), pair_id="g", host="GaN"
        )
        charged = make_config(
            (1, 0, 0, 0),
            -1238.70,
            forces=np.full((2, 3), 0.3),
            pair_id="g",
            host="GaN",
            **overrides,
        )
        return [neutral, charged]

    def test_referencing_and_delta_targets(self):
        configs = self.make_pair()
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})
        neutral, charged = configs

        # E_target = E_raw - n_e * E_CBM = -1238.70 - (-3.20)
        assert charged.properties["energy"] == pytest.approx(-1235.50)
        assert charged.properties["raw_energy"] == pytest.approx(-1238.70)
        # The carrier is bound 1 eV below the CBM.
        assert charged.properties["delta_energy"] == pytest.approx(-1.00)
        assert np.allclose(charged.properties["delta_forces"], 0.2)
        assert charged.properties["base_energy"] == pytest.approx(-1234.50)

        # The reference state is untouched by referencing and has no delta target.
        assert neutral.properties["energy"] == pytest.approx(-1234.50)
        assert neutral.property_weights["delta_energy"] == 0.0

    def test_base_branch_is_supervised_once_per_geometry(self):
        neutral = make_config((0, 0, 0, 0), -1234.50, pair_id="g", host="GaN")
        charged_a = make_config((1, 0, 0, 0), -1238.70, pair_id="g", host="GaN")
        charged_b = make_config((0, 0, 0, 1), -1227.90, pair_id="g", host="GaN")
        configs = [neutral, charged_a, charged_b]
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})

        weights = [c.property_weights["base_energy"] for c in configs]
        assert weights == [1.0, 0.0, 0.0]
        # ... but every member still carries the label and a delta target.
        assert all(
            c.properties["base_energy"] == pytest.approx(-1234.50) for c in configs
        )
        assert charged_a.property_weights["delta_energy"] == 1.0
        assert charged_b.property_weights["delta_energy"] == 1.0

    def test_unpaired_charged_frame_trains_through_totals_only(self):
        configs = [make_config((1, 0, 0, 0), -1238.70, host="GaN")]
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})
        config = configs[0]
        assert config.properties["delta_energy"] is None
        assert config.property_weights["delta_energy"] == 0.0
        assert config.property_weights["base_energy"] == 0.0
        assert config.properties["energy"] == pytest.approx(-1235.50)

    def test_unpaired_reference_frame_supervises_the_base_branch(self):
        configs = [make_config((0, 0, 0, 0), -1234.50, host="GaN")]
        prepare_defect_configurations(configs)
        assert configs[0].property_weights["base_energy"] == 1.0
        assert configs[0].properties["base_energy"] == pytest.approx(-1234.50)

    def test_per_frame_edges_override_the_table(self):
        configs = self.make_pair(e_cbm_cell=-2.0, e_vbm_cell=-6.0)
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})
        assert configs[1].properties["energy"] == pytest.approx(-1236.70)

    def test_group_without_reference_state_is_rejected(self):
        configs = [
            make_config((1, 0, 0, 0), -1238.70, pair_id="g", host="GaN"),
            make_config((0, 0, 0, 1), -1227.90, pair_id="g", host="GaN"),
        ]
        with pytest.raises(ValueError, match="0 reference-state"):
            prepare_defect_configurations(configs, band_edges={"GaN": EDGES})

    def test_group_with_two_reference_states_is_rejected(self):
        configs = [
            make_config((0, 0, 0, 0), -1234.50, pair_id="g", host="GaN"),
            make_config((0, 0, 0, 0), -1234.40, pair_id="g", host="GaN"),
        ]
        with pytest.raises(ValueError, match="2 reference-state"):
            prepare_defect_configurations(configs, band_edges={"GaN": EDGES})

    def test_group_with_mismatched_geometry_is_rejected(self):
        """The failure a structurally-paired file format cannot even express: two states
        that were relaxed separately and are not the same geometry."""
        neutral = make_config((0, 0, 0, 0), -1234.50, pair_id="g", host="GaN")
        charged = make_config(
            (1, 0, 0, 0),
            -1238.70,
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]],
            pair_id="g",
            host="GaN",
        )
        with pytest.raises(ValueError, match="not the same geometry"):
            prepare_defect_configurations([neutral, charged], band_edges={"GaN": EDGES})

    def test_charged_frame_without_edges_is_rejected(self):
        configs = [make_config((1, 0, 0, 0), -1238.70, host="GaN")]
        with pytest.raises(ValueError, match="no band edges"):
            prepare_defect_configurations(configs)

    def test_neutral_frame_needs_no_edges(self):
        configs = [make_config((0, 0, 0, 0), -1234.50, host="GaN")]
        prepare_defect_configurations(configs)
        assert configs[0].properties["energy"] == pytest.approx(-1234.50)

    def test_counters_are_canonicalised_at_load(self):
        configs = [
            make_config(
                (0, 1, 0, 0), -1238.70, host="GaN", e_cbm_cell=-3.2, e_vbm_cell=-6.85
            )
        ]
        prepare_defect_configurations(configs)
        assert np.array_equal(
            configs[0].properties["carrier_counts"], np.array([1, 0, 0, 0])
        )

    def test_multi_host_frames_use_their_own_edges(self):
        table = {"GaN": EDGES, "ZnO": BandEdges(e_cbm_cell=-4.05, e_vbm_cell=-7.31)}
        gan = make_config((1, 0, 0, 0), -1238.70, host="GaN")
        zno = make_config((1, 0, 0, 0), -900.00, host="ZnO")
        prepare_defect_configurations([gan, zno], band_edges=table)
        assert gan.properties["energy"] == pytest.approx(-1235.50)
        assert zno.properties["energy"] == pytest.approx(-895.95)

    def test_frame_with_edges_but_no_counters_is_rejected(self):
        atoms = Atoms(
            numbers=[31, 7], positions=[[0, 0, 0], [0, 0, 1.1]], cell=np.eye(3) * 6
        )
        atoms.info["REF_energy"] = -1238.70
        atoms.info["e_cbm_cell"] = -3.2
        atoms.info["e_vbm_cell"] = -6.85
        configs = [config_from_atoms(atoms, key_specification=keyspec())]
        with pytest.raises(ValueError, match="no carrier_counts"):
            prepare_defect_configurations(configs)


class TestAtomicDataRoundTrip:
    def test_defect_fields_reach_atomic_data(self):
        configs = [
            make_config(
                (0, 0, 0, 0),
                -1234.50,
                forces=np.full((2, 3), 0.1),
                pair_id="g",
                host="GaN",
            ),
            make_config(
                (1, 0, 0, 0),
                -1238.70,
                forces=np.full((2, 3), 0.3),
                pair_id="g",
                host="GaN",
            ),
        ]
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})
        z_table = AtomicNumberTable([7, 31])
        data = [
            AtomicData.from_config(config, z_table=z_table, cutoff=4.0)
            for config in configs
        ]

        assert data[1].carrier_counts.shape == (1, 4)
        assert data[1].carrier_counts.tolist() == [[1.0, 0.0, 0.0, 0.0]]
        assert float(data[1].delta_energy) == pytest.approx(-1.00)
        assert np.allclose(data[1].delta_forces.numpy(), 0.2)
        assert float(data[1].base_energy) == pytest.approx(-1234.50)
        assert float(data[1].delta_energy_weight) == 1.0
        assert float(data[1].base_energy_weight) == 0.0
        assert float(data[0].base_energy_weight) == 1.0

    def test_ordinary_configurations_are_unaffected(self):
        """A dataset with no carrier data must produce zero weights, not zero labels
        that look supervised."""
        atoms = Atoms(
            numbers=[31, 7], positions=[[0, 0, 0], [0, 0, 1.1]], cell=np.eye(3) * 6
        )
        atoms.info["REF_energy"] = -12.0
        config = config_from_atoms(atoms, key_specification=keyspec())
        data = AtomicData.from_config(
            config, z_table=AtomicNumberTable([7, 31]), cutoff=4.0
        )
        assert data.carrier_counts.tolist() == [[0.0, 0.0, 0.0, 0.0]]
        assert float(data.delta_energy_weight) == 0.0
        assert float(data.base_energy_weight) == 0.0

    def test_hdf5_preprocessing_carries_the_derived_targets(self, tmp_path):
        """The targets are built at Configuration level precisely so that the HDF5
        preprocessing path needs no changes of its own."""
        import h5py

        from mace.data.hdf5_dataset import HDF5Dataset
        from mace.data.utils import save_configurations_as_HDF5

        configs = [
            make_config(
                (0, 0, 0, 0),
                -1234.50,
                forces=np.full((2, 3), 0.1),
                pair_id="g",
                host="GaN",
            ),
            make_config(
                (1, 0, 0, 0),
                -1238.70,
                forces=np.full((2, 3), 0.3),
                pair_id="g",
                host="GaN",
            ),
        ]
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})

        path = tmp_path / "configs.h5"
        with h5py.File(path, "w") as handle:
            save_configurations_as_HDF5(configs, 0, handle)

        dataset = HDF5Dataset(path, r_max=4.0, z_table=AtomicNumberTable([7, 31]))
        charged = dataset[1]
        assert charged.carrier_counts.tolist() == [[1.0, 0.0, 0.0, 0.0]]
        assert float(charged.delta_energy) == pytest.approx(-1.00)
        assert float(charged.delta_energy_weight) == 1.0
        assert float(charged.base_energy) == pytest.approx(-1234.50)

    def test_batching_keeps_counters_per_graph(self):
        from mace.tools.torch_geometric.batch import Batch

        configs = [
            make_config((0, 0, 0, 0), -1234.50, host="GaN"),
            make_config((1, 0, 0, 0), -1238.70, host="GaN"),
            make_config((0, 0, 0, 1), -1227.90, host="GaN"),
        ]
        prepare_defect_configurations(configs, band_edges={"GaN": EDGES})
        z_table = AtomicNumberTable([7, 31])
        batch = Batch.from_data_list(
            [
                AtomicData.from_config(config, z_table=z_table, cutoff=4.0)
                for config in configs
            ]
        )
        assert batch.carrier_counts.shape == (3, 4)
        assert batch.carrier_counts.tolist() == [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
