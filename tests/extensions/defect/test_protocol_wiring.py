"""The protocol is wired into the trainer, and there is exactly ONE implementation of it.

WHAT THESE TESTS ARE AND ARE NOT. They do NOT claim step-level weight identity between
`defect-perovskite/stage_run.py` and `mace.cli.run_train`. The two drivers batch differently
-- the harness builds its own fixed list of graphs, the trainer shuffles a DataLoader -- and a
different permutation gives different weights after one epoch for reasons that have nothing to
do with the protocol. Chasing that number would be measuring the shuffle.

What CAN be asserted, and is what "the joint run comes from config" actually needs:

  * every protocol piece has one implementation, called by both drivers (no inlined copy
    survives in the harness);
  * fed the same inputs, each piece returns the same thing regardless of caller;
  * the config knobs the protocol depends on survive a save/rebuild round trip, so a model
    reloaded from disk trains under the settings its artefact records.

The last one is not hypothetical. A run trained at Gaussian sigma = 0.025 under a 0.05 banner
because the family and the width came from different places, and no grep could have found it:
every file said "gaussian" and "0.05" somewhere. Only the assembled object knows.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from mace.modules import defect_protocol
from mace.modules.defect_counting import ENVELOPES, SlaterKosterH

REPO = Path(__file__).resolve().parents[3]
STAGE_RUN = REPO / "defect-perovskite" / "stage_run.py"
RUN_TRAIN = REPO / "mace" / "cli" / "run_train.py"


# --------------------------------------------------------------------- one implementation


def _calls(path: Path) -> set:
    """Every name a file reaches for as `<something>.<name>` or calls directly.

    Attribute REFERENCES count, not only calls: `post_step` is handed to the trainer as a
    hook rather than invoked at the call site, and a check that only looked for `(` would
    report the one piece that runs on every optimiser step as missing.
    """
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out


PROTOCOL_PIECES = ("calibrate_c_shift", "trainable_mask", "warmup_factor", "post_step",
                   "apply_harrison", "initialisation_report")


@pytest.mark.parametrize("piece", PROTOCOL_PIECES)
def test_the_harness_calls_the_protocol_rather_than_its_own_copy(piece):
    assert piece in _calls(STAGE_RUN), (
        f"stage_run.py does not call defect_protocol.{piece}. Every Stage-3 result came "
        "from that harness, so a copy there that agrees with the package today is a copy "
        "that can drift from it tomorrow.")


@pytest.mark.parametrize("piece", ("calibrate_c_shift", "trainable_mask", "warmup_factor",
                                   "post_step", "apply_harrison"))
def test_the_trainer_calls_the_protocol(piece):
    assert piece in _calls(RUN_TRAIN), f"run_train.py does not call {piece}"


def test_the_harness_no_longer_inlines_the_trainable_mask():
    """The specific predicate that used to be written out in the harness's training loop."""
    text = STAGE_RUN.read_text()
    assert "is_correction_param(n)" not in text, (
        "stage_run.py still spells out the head-only mask inline; that is the second "
        "implementation this refactor exists to delete.")


# ------------------------------------------------------------------------- the pieces


def test_c_shift_is_none_rather_than_zero_when_no_frame_carries_a_carrier():
    """A silent 0.0 would be indistinguishable from a calibration that happened."""
    out = {"delta_sr_energy": torch.zeros(3), "base_energy": torch.zeros(3)}
    counts = torch.zeros(3, 4)
    assert defect_protocol.calibrate_c_shift(out, torch.ones(3), counts) is None


def test_c_shift_solves_the_median_mismatch_exactly():
    out = {"delta_sr_energy": torch.tensor([0.0, 0.0, 0.0]),
           "base_energy": torch.tensor([0.0, 0.0, 0.0])}
    # Delta_n = e_maj + e_min - h_maj - h_min. One electron each, residuals 1, 2, 3 eV.
    counts = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float64)
    c = defect_protocol.calibrate_c_shift(out, torch.tensor([1.0, 2.0, 3.0]), counts)
    assert c == pytest.approx(2.0)


def test_warmup_reaches_one_exactly_at_the_end_and_stays_there():
    assert defect_protocol.warmup_factor(0, 5) == pytest.approx(0.2)
    assert defect_protocol.warmup_factor(4, 5) == pytest.approx(1.0)
    assert defect_protocol.warmup_factor(9, 5) == pytest.approx(1.0)
    # Zero warmup is off, not a division by zero.
    assert defect_protocol.warmup_factor(0, 0) == 1.0


def test_trainable_mask_covers_the_head_the_corrections_and_z():
    assert defect_protocol.trainable_mask("spectral.h.eps0")
    assert defect_protocol.trainable_mask("madelung.z")
    assert not defect_protocol.trainable_mask("interactions.0.linear.weight")
    assert not defect_protocol.trainable_mask("madelung.z", freeze_z=True)
    # freeze_z touches ONLY the charges; the head must stay trainable in that arm or the
    # diagnostic stops being a diagnostic and becomes a different experiment.
    assert defect_protocol.trainable_mask("spectral.h.eps0", freeze_z=True)


def test_protocol_summary_reads_the_head_and_not_the_module_default():
    """The mismatch that trained six seeds at the wrong width, one layer up."""
    from mace.modules import defect_counting

    class _Head:
        smearing_family = "fermi"
        t_el = 0.123

    class _Model:
        spectral = _Head()

    summary = defect_protocol.protocol_summary(3, 2.4, 1.0, 5, 1.0, False, model=_Model())
    assert summary["smearing_family"] == "fermi"
    assert summary["smearing_width"] == pytest.approx(0.123)
    # And the module default is genuinely different, so the assertion above has teeth.
    assert defect_counting.smearing()[0] != "fermi"


# ------------------------------------------------------------------------ the envelope


@pytest.mark.parametrize("envelope", ENVELOPES)
def test_the_envelope_agrees_with_harrison_at_d_ref_whichever_family(envelope):
    """Both families are 1 at d_ref, which is where v0 is initialised. If they were not,
    switching the envelope would silently rescale every hopping in the model."""
    h = SlaterKosterH(num_elements=3, feature_dim=8, envelope=envelope)
    r = torch.tensor([h.d_ref], dtype=torch.float64)
    taper = (1.0 - (h.d_ref / h.r_cut) ** 6) ** 2
    assert float(h.radial(r)[0]) == pytest.approx(taper, rel=1e-9)


def test_the_power_envelope_is_the_bigger_one_at_the_hub_separation():
    """The measurement F7 rests on: at 6 A the exponential is several times smaller."""
    a = SlaterKosterH(num_elements=3, feature_dim=8, envelope="exp", decay_length=1.0)
    b = SlaterKosterH(num_elements=3, feature_dim=8, envelope="power")
    r = torch.tensor([6.0], dtype=torch.float64)
    assert float(b.radial(r)) / float(a.radial(r)) > 4.0


def test_an_unknown_envelope_is_refused_rather_than_silently_exponential():
    with pytest.raises(ValueError, match="unknown radial envelope"):
        SlaterKosterH(num_elements=3, feature_dim=8, envelope="gaussian")


def test_models_pickled_before_the_envelope_existed_still_evaluate():
    """`radial` reads the attribute through getattr for exactly this case."""
    h = SlaterKosterH(num_elements=3, feature_dim=8)
    del h.envelope
    r = torch.tensor([4.0], dtype=torch.float64)
    assert torch.isfinite(h.radial(r)).all()


# ------------------------------------------------------------- the composition selector


def _fake_batch(sizes, composition=(3, 1, 1)):
    """Graphs with the given per-species counts, as one-hot node_attrs plus a batch index."""
    node_attrs, batch = [], []
    for g, counts in enumerate(sizes):
        for species, n in enumerate(counts):
            for _ in range(n):
                row = [0.0] * len(composition)
                row[species] = 1.0
                node_attrs.append(row)
                batch.append(g)

    class _Batch:
        pass

    b = _Batch()
    b.node_attrs = torch.tensor(node_attrs)
    b.batch = torch.tensor(batch, dtype=torch.long)
    b.num_graphs = len(sizes)
    b.weight = torch.ones(len(sizes))
    return b


def test_stoichiometric_mask_separates_pristine_from_defective_by_composition_alone():
    from mace.modules.loss import DefectLoss

    loss = DefectLoss(gap_weight=1.0, e_gap=2.4, gap_composition=[3.0, 1.0, 1.0])
    # (Cl, Cs, Pb): an 80-atom pristine cell, a 79-atom V_Cl cell, a 159-atom V_Cl cell.
    b = _fake_batch([(48, 16, 16), (47, 16, 16), (95, 32, 32)])
    mask = loss.stoichiometric_mask(b)
    assert mask.tolist() == [True, False, False], (
        "the selector must call the vacancy cells defective from their composition alone -- "
        "it never sees a defect label, and a 159-atom cell is the biggest frame in the set")


def test_the_gap_term_is_zero_when_no_pristine_cell_is_in_the_batch():
    from mace.modules.loss import DefectLoss

    loss = DefectLoss(gap_weight=1.0, e_gap=2.4, gap_composition=[3.0, 1.0, 1.0])
    b = _fake_batch([(47, 16, 16)])
    pred = {"logit_gap": torch.tensor([[1.0]])}
    assert float(loss.gap_penalty(b, pred)) == 0.0
    assert loss.gap_steps_total == 1 and loss.gap_steps_with_term == 0


def test_the_gap_term_fires_and_is_counted_when_one_is():
    from mace.modules.loss import DefectLoss

    loss = DefectLoss(gap_weight=2.0, e_gap=2.4, gap_composition=[3.0, 1.0, 1.0])
    b = _fake_batch([(48, 16, 16), (47, 16, 16)])
    pred = {"logit_gap": torch.tensor([[1.4], [9.9]])}
    # Only the pristine graph contributes: 2 * (1.4 - 2.4)^2 = 2.0. If the defective graph
    # leaked in, the mean would be 5.65 and the value 21.
    assert float(loss.gap_penalty(b, pred)) == pytest.approx(2.0)
    assert loss.gap_steps_with_term == 1


def test_the_gap_term_is_off_without_a_composition():
    """Without one, nothing can tell a defect-free cell from a defective one without a
    label -- so the term must not silently apply itself to every graph."""
    from mace.modules.loss import DefectLoss

    loss = DefectLoss(gap_weight=1.0, e_gap=2.4)
    b = _fake_batch([(48, 16, 16)])
    assert float(loss.gap_penalty(b, {"logit_gap": torch.tensor([[0.0]])})) == 0.0


# ------------------------------------------------------------------- the trainer's hook


def test_train_passes_the_post_step_hook_to_take_step():
    """The projection must run on the values the optimiser just wrote."""
    import importlib

    # The MODULE, not the `train` function `mace.tools` re-exports.
    train_mod = importlib.import_module("mace.tools.train")

    assert "post_step_hook" in inspect.signature(train_mod.train).parameters
    assert "post_step_hook" in inspect.signature(train_mod.take_step).parameters
    src = inspect.getsource(train_mod.take_step)
    step = src.index("optimizer.step()")
    hook = src.index("post_step_hook(model)")
    ema = src.index("if ema is not None")
    assert step < hook < ema, (
        "the hook must run after the optimiser and before the EMA update; averaging a "
        "shadow copy of unprojected weights would put them back into the evaluated model")
