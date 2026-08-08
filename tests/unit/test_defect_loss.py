"""Unit tests for DefectLoss and the MACEDefect training plumbing (plan section 4).

Synthetic batches only: what is checked here is which subset each term acts on, that the
base branch is detached in the unpaired-totals term, and that the regularisers reach the
readouts they are supposed to.
"""

import numpy as np
import pytest
import torch
from ase.atoms import Atoms

from mace import data, tools
from mace.data.defects import prepare_defect_configurations
from mace.modules.loss import DefectLoss
from mace.tools import torch_geometric
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.scripts_utils import (
    get_loss_fn,
    reference_state_configurations,
    reference_state_data_loader,
)

torch.set_default_dtype(torch.float64)

Z_TABLE = tools.AtomicNumberTable([6, 14])
CUTOFF = 4.0
EDGES = {"GaN": data.BandEdges(e_cbm_cell=-3.2, e_vbm_cell=-6.85)}


def keyspec():
    return data.KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "carrier_counts": "carrier_counts",
            "host": "host",
            "pair_id": "pair_id",
        },
        arrays_keys={"forces": "REF_forces"},
    )


def make_atoms(counts, energy, forces=0.0, pair_id=None):
    atoms = Atoms(
        numbers=[6, 14],
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    atoms.info["REF_energy"] = energy
    atoms.info["carrier_counts"] = np.asarray(counts, dtype=int)
    atoms.info["host"] = "GaN"
    if pair_id is not None:
        atoms.info["pair_id"] = pair_id
    atoms.arrays["REF_forces"] = np.full((2, 3), forces)
    return atoms


def make_batch(atoms_list):
    configs = [
        data.config_from_atoms(atoms, key_specification=keyspec())
        for atoms in atoms_list
    ]
    prepare_defect_configurations(configs, band_edges=EDGES)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)
            for config in configs
        ],
        batch_size=len(configs),
        shuffle=False,
    )
    return next(iter(loader))


def perfect_prediction(batch, **overrides):
    """A prediction that reproduces every label exactly."""
    prediction = {
        "energy": batch.energy.clone(),
        "base_energy": batch.base_energy.clone(),
        "delta_energy": batch.delta_energy.clone(),
        "forces": batch.forces.clone(),
        "base_forces": batch.base_forces.clone(),
        "delta_forces": batch.delta_forces.clone(),
        "stress": batch.stress.clone(),
    }
    prediction.update(overrides)
    return prediction


PAIR = [
    make_atoms((0, 0, 0, 0), -1234.50, forces=0.1, pair_id="g"),
    make_atoms((1, 0, 0, 0), -1238.70, forces=0.3, pair_id="g"),
]


class TestTermSelection:
    def test_perfect_prediction_gives_zero_loss(self):
        batch = make_batch(PAIR)
        loss_fn = DefectLoss(u_l2=0.0, p_l2=0.0, qhost_l2=0.0)
        assert float(loss_fn(ref=batch, pred=perfect_prediction(batch))) == 0.0

    def test_delta_error_is_penalised(self):
        batch = make_batch(PAIR)
        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            total_energy_weight=0.0,
            delta_forces_weight=0.0,
            delta_energy_weight=1.0,
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
        )
        pred = perfect_prediction(batch)
        pred["delta_energy"] = pred["delta_energy"] + torch.tensor([0.0, 0.5])
        # Two configurations in the batch, one of which carries a delta label.
        assert float(loss_fn(ref=batch, pred=pred)) == pytest.approx(0.25 / 2)

    def test_delta_energy_is_not_normalised_per_atom(self):
        """A charge-state difference is intensive; dividing by the atom count would make
        it vanish from the objective in exactly the large cells that matter."""
        small = make_batch(PAIR)
        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            total_energy_weight=0.0,
            delta_forces_weight=0.0,
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
        )
        pred = perfect_prediction(small)
        pred["delta_energy"] = pred["delta_energy"] + torch.tensor([0.0, 0.5])
        loss_small = float(loss_fn(ref=small, pred=pred))

        # The same error in a cell with twice the atoms must cost the same.
        big_atoms = [atoms.repeat((2, 1, 1)) for atoms in PAIR]
        for atoms, source in zip(big_atoms, PAIR):
            atoms.info = dict(source.info)
            atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
        big = make_batch(big_atoms)
        pred_big = perfect_prediction(big)
        pred_big["delta_energy"] = pred_big["delta_energy"] + torch.tensor([0.0, 0.5])
        assert float(loss_fn(ref=big, pred=pred_big)) == pytest.approx(loss_small)

    def test_base_energy_is_supervised_once_per_geometry(self):
        """Only the frame carrying base weight contributes, so a geometry with several
        charge states does not count its base target several times."""
        batch = make_batch(PAIR)
        loss_fn = DefectLoss(
            energy_weight=1.0,
            forces_weight=0.0,
            delta_energy_weight=0.0,
            delta_forces_weight=0.0,
            total_energy_weight=0.0,
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
        )
        pred = perfect_prediction(batch)
        pred["base_energy"] = pred["base_energy"] + torch.tensor([2.0, 2.0])
        # Per-atom normalisation, two atoms, one supervised frame, mean over two frames.
        assert float(loss_fn(ref=batch, pred=pred)) == pytest.approx((1.0**2) / 2)


class TestUnpairedTotals:
    def unpaired_batch(self):
        return make_batch(
            [
                make_atoms((0, 0, 0, 0), -1234.50, forces=0.1),
                make_atoms((1, 0, 0, 0), -1238.70, forces=0.3),
            ]
        )

    def consistent_prediction(self, batch, **overrides):
        """An unpaired charged frame has no base label -- only the sum is constrained --
        so a prediction that costs nothing must be built from the total, not the labels."""
        prediction = perfect_prediction(batch)
        prediction["base_energy"] = batch.energy - prediction["delta_energy"]
        prediction.update(overrides)
        return prediction

    def test_totals_term_applies_only_to_unpaired_charged_frames(self):
        batch = self.unpaired_batch()
        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            delta_energy_weight=0.0,
            delta_forces_weight=0.0,
            total_energy_weight=1.0,
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
        )
        # Perturb the reference-state frame: it has no partner either, but it is not
        # charged, so the totals term must ignore it.
        pred = self.consistent_prediction(batch)
        pred["base_energy"] = pred["base_energy"] + torch.tensor([3.0, 0.0])
        assert float(loss_fn(ref=batch, pred=pred)) == 0.0

        pred = self.consistent_prediction(batch)
        pred["delta_energy"] = pred["delta_energy"] + torch.tensor([0.0, 2.0])
        assert float(loss_fn(ref=batch, pred=pred)) > 0.0

    def test_paired_frames_are_excluded_from_the_totals_term(self):
        batch = make_batch(PAIR)
        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            delta_energy_weight=0.0,
            delta_forces_weight=0.0,
            total_energy_weight=1.0,
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
        )
        pred = perfect_prediction(batch)
        pred["delta_energy"] = pred["delta_energy"] + torch.tensor([0.0, 2.0])
        assert float(loss_fn(ref=batch, pred=pred)) == 0.0

    def test_base_branch_is_detached_in_the_totals_term(self):
        """Without the detachment, unpaired charged data reshapes the base potential and
        the residual decomposition stops meaning what it is supposed to."""
        batch = self.unpaired_batch()
        loss_fn = DefectLoss(
            energy_weight=0.0,
            forces_weight=0.0,
            delta_energy_weight=0.0,
            delta_forces_weight=0.0,
            total_energy_weight=1.0,
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
        )
        base = (batch.energy - batch.delta_energy).clone().requires_grad_(True)
        delta = batch.delta_energy.clone().requires_grad_(True)
        pred = self.consistent_prediction(
            batch, base_energy=base, delta_energy=delta + 1.0
        )
        loss = loss_fn(ref=batch, pred=pred)
        loss.backward()
        assert base.grad is None or float(base.grad.abs().sum()) == 0.0
        assert float(delta.grad.abs().sum()) > 0.0


class TestRegularisation:
    def test_readout_penalties_are_applied(self):
        batch = make_batch(PAIR)
        pred = perfect_prediction(batch)
        pred.update(
            {
                "carrier_readouts": torch.full((2, 4), 2.0),
                "polarisation": torch.full((2,), 3.0),
                "latent_charges_host": torch.full((2,), 4.0),
            }
        )
        loss_fn = DefectLoss(u_l2=0.5, p_l2=0.25, qhost_l2=0.125)
        expected = 0.5 * 4.0 + 0.25 * 9.0 + 0.125 * 16.0
        assert float(loss_fn(ref=batch, pred=pred)) == pytest.approx(expected)

    def test_dielectric_prior_pulls_the_amplitude_to_the_dfpt_value(self):
        batch = make_batch(PAIR)
        pred = perfect_prediction(batch)
        pred["screening_amplitude"] = torch.tensor([0.25])
        loss_fn = DefectLoss(
            u_l2=0.0,
            p_l2=0.0,
            qhost_l2=0.0,
            eps_inf=4.0,
            eps_inf_prior_weight=2.0,
        )
        # 1/sqrt(4) = 0.5, so the residual is 0.25.
        assert float(loss_fn(ref=batch, pred=pred)) == pytest.approx(2.0 * 0.0625)

    def test_regularisation_is_skipped_without_the_long_range_branch(self):
        batch = make_batch(PAIR)
        pred = perfect_prediction(batch)
        pred["carrier_readouts"] = torch.full((2, 4), 1.0)
        loss_fn = DefectLoss(u_l2=1.0, p_l2=1.0, qhost_l2=1.0)
        # polarisation and host charges are absent for a short-range-only model.
        assert float(loss_fn(ref=batch, pred=pred)) == pytest.approx(1.0)


class TestPlumbing:
    def test_loss_factory_builds_the_defect_loss(self):
        args = build_default_arg_parser().parse_args(
            [
                "--name=x",
                "--train_file=f.xyz",
                "--model=MACEDefect",
                "--loss=defect",
                "--delta_energy_weight=7.0",
                "--defect_u_l2=0.5",
            ]
        )
        loss_fn = get_loss_fn(args, dipole_only=False, compute_dipole=False)
        assert isinstance(loss_fn, DefectLoss)
        assert float(loss_fn.delta_energy_weight) == 7.0
        assert loss_fn.u_l2 == 0.5

    def test_statistics_use_the_reference_state_subset(self):
        configs = [
            data.config_from_atoms(atoms, key_specification=keyspec())
            for atoms in [
                make_atoms((0, 0, 0, 0), -1234.50),
                make_atoms((1, 0, 0, 0), -1238.70),
                make_atoms((0, 0, 0, 1), -1227.90),
            ]
        ]
        prepare_defect_configurations(configs, band_edges=EDGES)
        filtered = reference_state_configurations(configs)
        assert len(filtered) == 1
        assert int(filtered[0].properties["carrier_counts"].sum()) == 0

    def test_reference_state_loader_filters_charged_cells(self):
        batch_atoms = [
            make_atoms((0, 0, 0, 0), -1234.50),
            make_atoms((1, 0, 0, 0), -1238.70),
            make_atoms((0, 0, 0, 0), -1234.10),
        ]
        configs = [
            data.config_from_atoms(atoms, key_specification=keyspec())
            for atoms in batch_atoms
        ]
        prepare_defect_configurations(configs, band_edges=EDGES)
        dataset = [
            data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)
            for config in configs
        ]
        loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=1)
        filtered = reference_state_data_loader(loader)
        assert len(filtered.dataset) == 2

    def test_reference_state_loader_falls_back_when_everything_is_charged(self):
        configs = [
            data.config_from_atoms(
                make_atoms((1, 0, 0, 0), -1238.70), key_specification=keyspec()
            )
        ]
        prepare_defect_configurations(configs, band_edges=EDGES)
        dataset = [
            data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=CUTOFF)
            for config in configs
        ]
        loader = torch_geometric.dataloader.DataLoader(dataset, batch_size=1)
        assert reference_state_data_loader(loader) is loader
