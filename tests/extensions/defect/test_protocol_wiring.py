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


@pytest.mark.parametrize("piece", ("calibrate_c_shift_over_loader", "trainable_mask",
                                   "warmup_factor", "post_step", "apply_harrison",
                                   "zero_on_site_correction"))
def test_the_trainer_calls_the_protocol(piece):
    """The trainer calibrates over the WHOLE loader, not `calibrate_c_shift` on one batch.

    A single-batch calibration makes the head's energy zero depend on the shuffle, and in the
    degenerate case -- a first batch that happens to be all neutral -- skips it entirely,
    which is what the production smoke hit. The single-batch entry point stays for callers
    that genuinely have one batch (the harness) and is covered separately.
    """
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

        @staticmethod
        def named_parameters():
            return iter(())

    summary = defect_protocol.protocol_summary(3, 2.4, 1.0, 5, 1.0, False, model=_Model())
    assert summary["smearing_family"] == "fermi"
    assert summary["smearing_width"] == pytest.approx(0.123)
    # And the module default is genuinely different, so the assertion above has teeth.
    assert defect_counting.smearing()[0] != "fermi"


def test_protocol_summary_records_whether_the_c_shift_HAPPENED():
    """Not whether it was intended. A run whose first batch carries no net carrier skips the
    calibration and starts at c = 0; the summary used to say "calibrated: true" anyway,
    because it reported `stage >= 3`. Caught by the first end-to-end trainer run."""
    skipped = defect_protocol.protocol_summary(3, 2.4, 1.0, 5, 1.0, False, c_shift=None)
    assert skipped["c_shift_calibrated"] is False and skipped["c_shift"] is None
    done = defect_protocol.protocol_summary(3, 2.4, 1.0, 5, 1.0, False, c_shift=-1.25)
    assert done["c_shift_calibrated"] is True
    assert done["c_shift"] == pytest.approx(-1.25)


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


# ------------------------------------------------- the deterministic c-shift and the forms


def test_c_shift_over_a_loader_is_the_median_of_ALL_terms_not_of_per_batch_medians():
    """Two batches with 1 and 3 charged frames. A median of medians weights the lone frame
    equally with the three; the median over all four terms does not."""
    import torch as _t

    class _Batch:
        def __init__(self, resid, counts):
            self.energy = _t.tensor(resid, dtype=_t.float64)
            self.carrier_counts = _t.tensor(counts, dtype=_t.float64)
            self._n = len(resid)

        def to(self, _device):
            return self

        def to_dict(self):
            return {}

    def _out(batch):
        n = batch._n
        return {"delta_sr_energy": _t.zeros(n, dtype=_t.float64),
                "base_energy": _t.zeros(n, dtype=_t.float64)}

    one = _Batch([100.0], [[1, 0, 0, 0]])
    three = _Batch([1.0, 2.0, 3.0], [[1, 0, 0, 0]] * 3)

    class _M:
        training = False

        def train(self, _mode=True):
            pass

        def eval(self):
            pass

    c, n = defect_protocol.calibrate_c_shift_over_loader(
        _M(), [one, three], "cpu", forward=_out)
    assert n == 4
    # torch.median takes the LOWER of the two middle values on an even-length tensor, so
    # [1, 2, 3, 100] gives 2.0 rather than numpy's 2.5. Kept as torch's, because the
    # single-batch entry point has always used torch.median and the two must agree.
    # The point stands either way: a median of per-batch medians would be 51.
    assert c == pytest.approx(2.0)


def test_c_shift_over_a_loader_reports_none_when_nothing_is_charged():
    import torch as _t

    class _Batch:
        energy = _t.zeros(2, dtype=_t.float64)
        carrier_counts = _t.zeros(2, 4, dtype=_t.float64)

        def to(self, _device):
            return self

    class _M:
        training = False

        def train(self, _mode=True):
            pass

        def eval(self):
            pass

    c, n = defect_protocol.calibrate_c_shift_over_loader(
        _M(), [_Batch()], "cpu",
        forward=lambda b: {"delta_sr_energy": _t.zeros(2, dtype=_t.float64),
                           "base_energy": _t.zeros(2, dtype=_t.float64)})
    assert c is None and n == 0


def test_harrison_reports_whether_it_ran():
    """`protocol_summary` used to infer `harrison_init` from the stage number."""
    class _NoHead:
        spectral = None

    assert defect_protocol.apply_harrison(_NoHead(), [17], 2.8) is False
    summary = defect_protocol.protocol_summary(3, 2.4, 1.0, 5, 1.0, False,
                                               harrison_applied=False)
    assert summary["harrison_init"] is False


def test_both_modulation_forms_are_exactly_one_on_a_bulk_like_bond():
    """`pre = 0` must give factor 1 in both, or switching forms would rescale every hopping
    in the model rather than only the ones the head is straining on."""
    for form in ("linear", "log"):
        h = SlaterKosterH(num_elements=3, feature_dim=8, hop_form=form)
        assert float(h.modulation(torch.zeros(1))) == pytest.approx(1.0)


def test_the_log_form_is_symmetric_in_log_space_and_the_linear_form_is_not():
    lin = SlaterKosterH(num_elements=3, feature_dim=8, hop_form="linear", hop_range=0.5)
    log = SlaterKosterH(num_elements=3, feature_dim=8, hop_form="log")
    big = torch.tensor([12.0])            # tanh saturates
    up_lin = float(lin.modulation(big))
    dn_lin = float(lin.modulation(-big))
    up_log = float(log.modulation(big))
    dn_log = float(log.modulation(-big))
    assert up_lin == pytest.approx(1.5, abs=1e-6) and dn_lin == pytest.approx(0.5, abs=1e-6)
    assert up_log * dn_log == pytest.approx(1.0, rel=1e-6)   # symmetric in log space
    assert up_log == pytest.approx(3.0, rel=1e-6)
    # And the log form is positive everywhere, so widening cannot flip a hopping's sign.
    assert float(log.modulation(torch.tensor([-50.0]))) > 0.0


def test_an_unknown_modulation_form_is_refused():
    with pytest.raises(ValueError, match="unknown hopping modulation"):
        SlaterKosterH(num_elements=3, feature_dim=8, hop_form="tanh")


def test_models_pickled_before_the_modulation_form_existed_still_evaluate():
    h = SlaterKosterH(num_elements=3, feature_dim=8)
    del h.hop_form
    assert float(h.modulation(torch.zeros(1))) == pytest.approx(1.0)


def test_zero_init_removes_the_on_site_channel_output_entirely():
    """Weights AND bias: zeroing only the weight leaves a constant per-channel offset, which
    is precisely the uniform gauge mode being removed."""
    class _M:
        pass

    m = _M()
    m.spectral = type("H", (), {})()
    m.spectral.h = SlaterKosterH(num_elements=3, feature_dim=8)
    last = [x for x in m.spectral.h.site.modules() if isinstance(x, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.fill_(0.3)
        if last.bias is not None:
            last.bias.fill_(0.7)
    assert defect_protocol.zero_on_site_correction(m) is True
    assert float(last.weight.abs().max()) == 0.0
    assert last.bias is None or float(last.bias.abs().max()) == 0.0


def test_zero_init_says_so_when_there_is_nothing_to_zero():
    class _M:
        spectral = None

    assert defect_protocol.zero_on_site_correction(_M()) is False


def test_the_neutral_upweight_touches_neutral_frames_only():
    from mace.data.two_size import apply_neutral_size_upweight

    class _D:
        def __init__(self, n, charged):
            self.positions = torch.zeros(n, 3)
            self.carrier_counts = (torch.tensor([1.0, 0, 0, 0]) if charged
                                   else torch.zeros(4))
            self.forces_weight = 1.0

    small = [_D(79, False) for _ in range(40)]
    large = [_D(159, False) for _ in range(2)]
    charged_large = [_D(159, True) for _ in range(2)]
    factor, share = apply_neutral_size_upweight(small + large + charged_large,
                                                target_share=0.25)
    assert factor > 1.0 and share == pytest.approx(0.25, abs=1e-6)
    assert all(d.forces_weight == 1.0 for d in charged_large), (
        "the neutral upweight must not touch charged frames -- the two shares are computed "
        "within their own populations so they do not compete for one budget")
    assert all(d.forces_weight == 1.0 for d in small)
