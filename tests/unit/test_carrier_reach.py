"""The reach check must fire on a short graph and pass on a correct one.

This is the fourth guard written for this failure mode, and the previous three measured a
reimplementation instead of the real path. So the check operates on a batch from the training
DataLoader itself, and these tests exercise it on real neighbour lists built by the same
AtomicData path training uses -- not on hand-made tensors, which is how a check ends up
agreeing with a bug.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import mace  # noqa: F401  (before e3nn)
from ase.io import read

from mace import data as mace_data
from mace import tools
from mace.data import AtomicData
from mace.modules.defect_reach import assert_carrier_reach, reach_report

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "defect-perovskite" / "dataset_pbe" / "valid.xyz"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="perovskite dataset not present")


def build(cutoff, n=3):
    from mace.data.defects import prepare_defect_configurations

    z_table = tools.AtomicNumberTable([17, 55, 82])
    frames = read(DATA, ":")[:n]
    out = []
    for atoms in frames:
        cfg = mace_data.config_from_atoms(atoms)
        prepare_defect_configurations([cfg])
        out.append(AtomicData.from_config(cfg, z_table=z_table, cutoff=cutoff))
    loader = tools.torch_geometric.dataloader.DataLoader(out, batch_size=len(out),
                                                         shuffle=False)
    return next(iter(loader))


def test_a_5A_graph_is_rejected_when_10A_was_requested():
    """The exact fault that voided the screen: loader built at r_max, head wants 10 A."""
    batch = build(5.0)
    with pytest.raises(RuntimeError, match="CARRIER REACH FAILURE"):
        assert_carrier_reach(batch, r_max=5.0, cutoff=10.0)


def test_a_10A_graph_passes():
    batch = build(10.0)
    report = assert_carrier_reach(batch, r_max=5.0, cutoff=10.0)
    assert report["max_edge"] > 9.0
    assert report["frac_beyond_r_max"] > 0.0
    assert report["edges_in_hub_window"] > 0, "no edges at the hub separation"


def test_plain_mace_is_untouched():
    """With no long-range head, cutoff == r_max and the check must never fire."""
    batch = build(5.0)
    report = assert_carrier_reach(batch, r_max=5.0, cutoff=5.0)
    assert report["max_edge"] <= 5.01


def test_the_report_is_label_free():
    """No vacancy assignment may reach the training loop; the check is structural only."""
    import ast
    import inspect

    from mace.modules import defect_reach

    tree = ast.parse(inspect.getsource(defect_reach))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names.add(mod.split(".")[-1])
            for alias in node.names:
                names.add(alias.name.split(".")[-1])
    forbidden = {"locate_vacancy", "vacancy_site", "hub", "shell", "cage"}
    assert not (forbidden & names), f"reach check reads {sorted(forbidden & names)}"


def test_reach_report_survives_an_edgeless_batch():
    batch = build(10.0)
    batch.edge_index = torch.zeros((2, 0), dtype=torch.long)
    assert reach_report(batch, 5.0, 10.0) == {}
