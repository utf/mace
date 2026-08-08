"""End-to-end training run for MACEDefect (charge-aware defect model).

Runs the real CLI in a subprocess on a tiny synthetic dataset with a known
charge-state difference, so the whole chain -- band-edge referencing, the pair join,
the delta targets, the loss and the DefectRMSE table -- is exercised together.
"""

import json

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from tests.helpers import base_mace_params, run_mace_train

# The charged frames sit 3.1 eV above the neutral ones, and the CBM reference is
# -3.2 eV, so every referenced charge-state difference is exactly +0.1 eV.
CBM_CELL = -3.2
VBM_CELL = -6.85
RAW_OFFSET = -3.1
EXPECTED_DELTA = RAW_OFFSET - CBM_CELL


def defect_configs(num_geometries=8):
    rng = np.random.default_rng(0)
    frames = []
    for index in range(num_geometries):
        base = Atoms(
            numbers=[6, 14, 6, 14],
            positions=np.array(
                [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [0.0, 0.0, 2.1]]
            )
            + rng.normal(scale=0.1, size=(4, 3)),
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        neutral_energy = -20.0 + 0.1 * index
        for counts, energy in (
            ((0, 0, 0, 0), neutral_energy),
            ((1, 0, 0, 0), neutral_energy + RAW_OFFSET),
        ):
            atoms = base.copy()
            atoms.info["REF_energy"] = energy
            atoms.info["carrier_counts"] = np.asarray(counts)
            atoms.info["host"] = "GaN"
            atoms.info["pair_id"] = f"geom{index}"
            atoms.arrays["REF_forces"] = rng.normal(scale=0.05, size=(4, 3))
            frames.append(atoms)

    for number, energy in ((6, -1.0), (14, -3.0)):
        isolated = Atoms(
            numbers=[number],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        isolated.info["REF_energy"] = energy
        isolated.info["config_type"] = "IsolatedAtom"
        isolated.info["carrier_counts"] = np.zeros(4, dtype=int)
        frames.append(isolated)
    return frames


@pytest.fixture(name="defect_dataset")
def fixture_defect_dataset(tmp_path):
    train = tmp_path / "train.xyz"
    ase.io.write(train, defect_configs())
    edges = tmp_path / "band_edges.json"
    edges.write_text(
        json.dumps({"GaN": {"e_cbm_cell": CBM_CELL, "e_vbm_cell": VBM_CELL}}),
        encoding="utf-8",
    )
    return train, edges


def defect_params(tmp_path, train, edges):
    params = base_mace_params()
    params.update(
        {
            "name": "MACEDefect",
            "train_file": str(train),
            "band_edges_file": str(edges),
            "model": "MACEDefect",
            "loss": "defect",
            "error_table": "DefectRMSE",
            "use_long_range": "False",
            "valid_fraction": 0.25,
            "max_num_epochs": 2,
            "batch_size": 2,
            "valid_batch_size": 2,
            "num_channels": 8,
            "max_L": 0,
            "num_interactions": 2,
            "r_max": 4.0,
            "num_radial_basis": 4,
            "default_dtype": "float64",
            "device": "cpu",
            "seed": 1,
            "checkpoints_dir": str(tmp_path),
            "model_dir": str(tmp_path),
            "log_dir": str(tmp_path),
            "results_dir": str(tmp_path),
            "work_dir": str(tmp_path),
            "eval_interval": 1,
        }
    )
    for key in ("swa", "start_swa", "ema", "ema_decay", "loss_type"):
        params.pop(key, None)
    return params


@pytest.mark.slow
def test_defect_training_run(tmp_path, defect_dataset):
    train, edges = defect_dataset
    result = run_mace_train(
        defect_params(tmp_path, train, edges), capture_output=True, text=True
    )
    output = result.stdout

    assert (tmp_path / "MACEDefect.model").exists()
    # The error table must report the charge-state difference columns...
    assert "RMSE dE / meV" in output
    # ... and the untrained correction predicts zero, so the reported dE error is the
    # known synthetic difference. This checks the referencing arithmetic, the join and
    # the metric together rather than merely that training ran.
    for line in output.splitlines():
        if line.startswith("|") and "train_Default" in line:
            reported = float(line.split("|")[4])
            assert reported == pytest.approx(abs(EXPECTED_DELTA) * 1000, rel=0.2)
            break
    else:
        raise AssertionError("no train_Default row in the error table")
