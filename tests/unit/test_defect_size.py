"""The size-extensivity probe and hinge (size-extensivity plan, section 4).

These are written to discriminate, not to cover. The properties that matter are the ones
that distinguish this term from every localisation penalty the plan rejected: it must be
*exactly* zero when the site energies carry no defect contrast, and its gradient must push
the shell logits **up**. A test that only checks the loss is finite would pass on a term
that does neither.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_size import segment_logsumexp, size_extensivity_probe

CHANNELS = 4


def two_level_cell(
    num_bulk: int = 60,
    num_shell: int = 6,
    gap: float = 5.0,
    u_bulk: float = 0.0,
    u_shell: float = -1.0,
    num_species: int = 2,
):
    """One cell: ``num_shell`` atoms at a raised logit, the rest at a flat bulk level.

    This is the field the model actually produces -- beyond ``r_max * n_layers`` every
    atom has an identical descriptor and hence an identical logit, so the logit field is
    two-level rather than an exponentially decaying envelope. That is precisely why weight
    leaks to bulk in proportion to N.
    """
    total = num_bulk + num_shell
    logits = torch.zeros(total, CHANNELS, dtype=torch.float64)
    logits[:num_shell] = gap
    readouts = torch.full((total, CHANNELS), u_bulk, dtype=torch.float64)
    readouts[:num_shell] = u_shell
    batch = torch.zeros(total, dtype=torch.long)
    # Alternate species so both are present and the median reference is well defined.
    node_attrs = torch.zeros(total, num_species, dtype=torch.float64)
    node_attrs[torch.arange(total) % num_species, ] = 0.0
    for index in range(total):
        node_attrs[index, index % num_species] = 1.0
    alpha = torch.softmax(logits, dim=0)
    return logits, readouts, alpha, batch, node_attrs


class TestProbe:
    def test_no_contrast_gives_exactly_zero_drift(self):
        """The central claim: a state with no defect contrast is not penalised at all.

        A genuine band-edge or shallow carrier *should* be delocalised. Here ``u`` is
        uniform while the logits are strongly peaked, so ``f`` is large -- and the drift
        must still be exactly zero, because the padding would capture attention but no
        energy. Entropy and participation penalties are large in exactly this situation,
        which is why they were rejected.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell(
            u_bulk=0.3, u_shell=0.3
        )
        f, drift, _ = size_extensivity_probe(
            logits, readouts, alpha, batch, node_attrs, num_graphs=1, ratio=1e4
        )
        assert float(f.max()) > 0.5, "precondition: the padding should capture weight"
        # Analytically identically zero; in floating point it lands at rounding on
        # quantities of order u itself (~1e-15 relative). Bounded far below the 1e-3
        # tolerance the hinge uses, so "exactly zero" holds for every practical purpose.
        assert float(drift.abs().max()) < 1e-12

    def test_drift_grows_with_contrast(self):
        """With contrast present the drift is non-zero and scales with it."""
        results = []
        for contrast in (0.5, 1.0, 2.0):
            logits, readouts, alpha, batch, node_attrs = two_level_cell(
                u_shell=-contrast
            )
            _, drift, _ = size_extensivity_probe(
                logits, readouts, alpha, batch, node_attrs, num_graphs=1, ratio=1e4
            )
            results.append(float(drift.abs().max()))
        assert results[0] < results[1] < results[2]

    def test_the_lnA_path_raises_the_shell_logits(self):
        """The intended mechanism, isolated.

        Descending the penalty must **raise** the shell logits, so that the current
        cell's normalisation outgrows the hypothetical padding. With ``<u>_alpha``
        detached the only route left is through ``lnA``, and its sign must be negative
        (gradient descent then increases the logit). Isolating it matters: the combined
        gradient is dominated by the other path, which is measured separately below.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=2.0)
        logits = logits.clone().requires_grad_(True)
        _, drift, _ = size_extensivity_probe(
            logits, readouts, torch.softmax(logits.detach(), dim=0), batch,
            node_attrs, num_graphs=1, ratio=1e4,
        )
        drift.abs().sum().backward()
        assert float(logits.grad[:6].sum()) < 0.0

    @pytest.mark.parametrize("ratio, dominant", [(1e4, "contrast"), (10.0, "gap")])
    def test_which_gradient_path_dominates(self, ratio, dominant):
        """``f = sigmoid(lnB - lnA)`` saturates, and a saturated sigmoid has no gradient.

        This records a real property of the term as specified rather than asserting a
        preference. At large ``R`` the padding outweighs the cell so overwhelmingly that
        ``f -> 1``, its derivative ``f(1-f) -> 0``, and the ``lnA`` path -- the only one
        that raises the gap -- is suppressed by orders of magnitude. What survives is the
        gradient through ``<u>_alpha``, which reduces the drift by flattening the defect
        contrast instead. The two exchange dominance around the point where ``f`` comes
        off its bound, i.e. where ``gap ~ ln(R N / k)``.

        Consequence for tuning: ``R`` is not a free statement of intent. It sets where the
        term has any gradient at all, and at the plan's default of 1e4 that is only once
        the gap is already near target.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=5.0)

        live = logits.clone().requires_grad_(True)
        size_extensivity_probe(
            live, readouts, torch.softmax(live.detach(), dim=0), batch, node_attrs,
            1, ratio,
        )[1].abs().sum().backward()
        via_gap = abs(float(live.grad[:6].sum()))

        live_u = logits.clone().requires_grad_(True)
        size_extensivity_probe(
            live_u.detach(), readouts, torch.softmax(live_u, dim=0), batch, node_attrs,
            1, ratio,
        )[1].abs().sum().backward()
        via_contrast = abs(float(live_u.grad[:6].sum()))

        if dominant == "gap":
            assert via_gap > via_contrast
        else:
            assert via_contrast > via_gap

    def test_survives_logits_at_the_clamp(self):
        """``e^40`` overflows float32; the term must live entirely in log space."""
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=40.0)
        logits = logits.to(torch.float32)
        readouts = readouts.to(torch.float32)
        node_attrs = node_attrs.to(torch.float32)
        alpha = torch.softmax(logits, dim=0)
        f, drift, clamped = size_extensivity_probe(
            logits, readouts, alpha, batch, node_attrs, num_graphs=1, ratio=1e4,
            logit_clamp=40.0,
        )
        assert torch.isfinite(f).all() and torch.isfinite(drift).all()
        # 6 shell atoms of 66 are sitting on the bound.
        assert float(clamped[0, 0]) == pytest.approx(6 / 66, abs=1e-6)

    def test_median_reference_tracks_bulk_not_the_shell(self):
        """A mean bulk reference would be dominated by a handful of ``e^{12}`` sites.

        The plan calls the median load-bearing rather than cosmetic, and this is that
        claim, tested directly on the reference itself rather than on the drift. The
        padding-weighted site energy is recoverable from the returned quantities, since
        ``drift = f (ubar_w - <u>_alpha)``, and it must sit on the **bulk** value even
        though the shell is far more extreme.
        """
        shell_u, bulk_u = -5.0, 0.25
        logits, readouts, alpha, batch, node_attrs = two_level_cell(
            num_shell=6, gap=12.0, u_shell=shell_u, u_bulk=bulk_u
        )
        f, drift, _ = size_extensivity_probe(
            logits, readouts, alpha, batch, node_attrs, num_graphs=1, ratio=1e4
        )
        pooled = float((alpha * readouts).sum(dim=0)[0])
        recovered = float(drift[0, 0] / f[0, 0]) + pooled
        assert recovered == pytest.approx(bulk_u, abs=1e-9), (
            "the padded reference must be the bulk site energy; a mean would be dragged "
            f"toward {shell_u}"
        )

    def test_segment_logsumexp_matches_torch(self):
        values = torch.randn(40, CHANNELS, dtype=torch.float64)
        batch = torch.repeat_interleave(torch.arange(4), 10)
        result = segment_logsumexp(values, batch, num_graphs=4)
        for graph in range(4):
            expected = torch.logsumexp(values[batch == graph], dim=0)
            assert torch.allclose(result[graph], expected, atol=1e-12)

    def test_ratio_must_exceed_one(self):
        with pytest.raises(ValueError, match="ratio must exceed 1"):
            size_extensivity_probe(
                *two_level_cell(), num_graphs=1, ratio=1.0
            )


class TestHinge:
    """The loss wrapper: tolerance, dead channels and the warmup gate."""

    @staticmethod
    def build_loss(**kwargs):
        from mace.modules.loss import DefectLoss

        defaults = dict(
            energy_weight=0.0, forces_weight=0.0, delta_energy_weight=0.0,
            delta_forces_weight=0.0, total_energy_weight=0.0, p_l2=0.0, qhost_l2=0.0,
            size_weight=1.0, size_tol=0.0, size_warmup_epochs=0, size_ratio=1e4,
        )
        defaults.update(kwargs)
        return DefectLoss(**defaults)

    @staticmethod
    def make_inputs(counts=(1, 0, 0, 1)):
        logits, readouts, alpha, batch, node_attrs = two_level_cell()
        ref = {
            "batch": batch,
            "node_attrs": node_attrs,
            "carrier_counts": torch.tensor([counts], dtype=torch.long),
        }

        class FakeBatch(dict):
            num_graphs = 1
            weight = torch.ones(1, dtype=torch.float64)

        pred = {
            "carrier_logits": logits,
            "carrier_readouts": readouts,
            "carrier_alpha": alpha,
        }
        return FakeBatch(ref), pred

    def test_active_term_is_positive(self):
        ref, pred = self.make_inputs()
        value = self.build_loss().size_penalty(ref, pred)
        assert float(value) > 0.0

    def test_tolerance_switches_it_off(self):
        ref, pred = self.make_inputs()
        loose = self.build_loss(size_tol=10.0).size_penalty(ref, pred)
        assert float(loose) == 0.0

    def test_warmup_gate(self):
        ref, pred = self.make_inputs()
        loss = self.build_loss(size_warmup_epochs=20)
        loss.current_epoch = 5
        assert float(loss.size_penalty(ref, pred)) == 0.0
        # A restart lands the trainer's absolute epoch straight into the active region;
        # the term must be on immediately rather than serving the warmup again.
        loss.current_epoch = 50
        assert float(loss.size_penalty(ref, pred)) > 0.0

    def test_dead_channels_contribute_nothing(self):
        """``n_c = 0`` zeroes a channel through the ``n_c^2`` weight, for free."""
        ref_live, pred = self.make_inputs(counts=(1, 1, 1, 1))
        ref_dead, _ = self.make_inputs(counts=(1, 0, 0, 0))
        loss = self.build_loss()
        assert float(loss.size_penalty(ref_live, pred)) > float(
            loss.size_penalty(ref_dead, pred)
        )

    def test_zero_weight_is_exactly_off(self):
        ref, pred = self.make_inputs()
        assert float(self.build_loss(size_weight=0.0).size_penalty(ref, pred)) == 0.0
