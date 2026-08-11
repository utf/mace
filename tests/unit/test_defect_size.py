"""The size-extensivity probe and hinge (size plan section 4, as amended).

Written to discriminate rather than to cover. The properties that matter are the ones
separating this term from every localisation penalty the plan rejected: it must be exempt
when the site energies carry no defect contrast, its gradient must raise the gap and must
*not* be able to flatten contrast, and it must be blind to a uniform shift of the logits.
A test that only checks the loss is finite would pass on a term that does none of these.
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
    """``num_shell`` atoms at a raised logit, the rest on a flat bulk plateau.

    This is the field the model actually produces: beyond ``r_max * n_layers`` every atom
    has an identical descriptor and hence an identical logit, so the logit field is
    two-level rather than an exponentially decaying envelope. That flat plateau is what
    leaks weight in proportion to N, and it was never physical.
    """
    total = num_bulk + num_shell
    logits = torch.zeros(total, CHANNELS, dtype=torch.float64)
    logits[:num_shell] = gap
    readouts = torch.full((total, CHANNELS), u_bulk, dtype=torch.float64)
    readouts[:num_shell] = u_shell
    batch = torch.zeros(total, dtype=torch.long)
    node_attrs = torch.zeros(total, num_species, dtype=torch.float64)
    for index in range(total):
        node_attrs[index, index % num_species] = 1.0
    alpha = torch.softmax(logits, dim=0)
    return logits, readouts, alpha, batch, node_attrs


def build_loss(**kwargs):
    from mace.modules.loss import DefectLoss

    defaults = dict(
        energy_weight=0.0, forces_weight=0.0, delta_energy_weight=0.0,
        delta_forces_weight=0.0, total_energy_weight=0.0, p_l2=0.0, qhost_l2=0.0,
        size_weight=1.0, size_tol=1e-3, size_warmup_epochs=0, size_ratio=1e4,
    )
    defaults.update(kwargs)
    return DefectLoss(**defaults)


def make_inputs(counts=(1, 0, 0, 1), **cell):
    logits, readouts, alpha, batch, node_attrs = two_level_cell(**cell)

    class FakeBatch(dict):
        num_graphs = 1
        weight = torch.ones(1, dtype=torch.float64)

    ref = FakeBatch(
        {
            "batch": batch,
            "node_attrs": node_attrs,
            "carrier_counts": torch.tensor([counts], dtype=torch.long),
        }
    )
    pred = {
        "carrier_logits": logits,
        "carrier_readouts": readouts,
        "carrier_alpha": alpha,
    }
    return ref, pred


class TestZeroSetEquivalence:
    """The reformulation must not move the feasible set, only the coordinate."""

    def test_sigma_x_c_le_tol_iff_x_le_threshold(self):
        generator = torch.Generator().manual_seed(0)
        for _ in range(500):
            x = float(torch.randn(1, generator=generator) * 6.0)
            contrast = float(torch.rand(1, generator=generator)) * 2.0
            tol = float(torch.rand(1, generator=generator)) * 0.5
            energy_space = torch.sigmoid(torch.tensor(x)) * contrast <= tol
            t = tol / max(contrast, 1e-12)
            if t >= 1.0:
                log_space = True  # threshold is +inf: no x can violate
            else:
                log_space = x <= float(np.log(t / (1.0 - t)))
            assert bool(energy_space) == bool(log_space), (x, contrast, tol)


class TestGradient:
    def test_lnA_gradient_survives_saturation(self):
        """The failure the reformulation exists to fix.

        At ``f > 0.99`` the energy-space form's gradient carried a factor ``f(1-f)`` and
        effectively vanished. Here the gradient is linear in the violation, so it must be
        of order one even when ``f`` is pinned.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=2.0)
        logits = logits.clone().requires_grad_(True)
        ref, pred = make_inputs()
        pred["carrier_logits"] = logits
        loss = build_loss()
        loss.contrast_ema = torch.full((CHANNELS,), 1.0, dtype=torch.float64)
        probe_f = torch.sigmoid(
            size_extensivity_probe(
                logits, readouts, alpha, batch, node_attrs, 1, 1e4
            ).x
        )
        assert float(probe_f.min()) > 0.99, "precondition: f is saturated"

        loss.size_penalty(ref, pred).backward()
        gradient = float(logits.grad[:6].sum())
        assert gradient < 0.0, "descent must RAISE the shell logits"
        assert abs(gradient) > 1e-2, (
            f"gradient {gradient} is vanishing; the f(1-f) suppression is back"
        )

    def test_contrast_carries_no_gradient(self):
        """``c`` is detached, so the term cannot buy the constraint by flattening it.

        Differentiating the *whole penalty*, not just ``x``: ``x`` has no functional
        dependence on ``u`` at all, so backpropagating through it alone would prove
        nothing. The threshold does depend on ``|c|``, and that is the route that has to
        be dead.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell()
        logits = logits.clone().requires_grad_(True)
        readouts = readouts.clone().requires_grad_(True)
        ref, pred = make_inputs()
        pred.update(
            {"carrier_logits": logits, "carrier_readouts": readouts,
             "carrier_alpha": alpha}
        )
        loss = build_loss()
        loss.contrast_ema = torch.full((CHANNELS,), 1.0, dtype=torch.float64)
        value = loss.size_penalty(ref, pred)
        assert float(value) > 0.0, "precondition: the term must be active"
        value.backward()
        assert readouts.grad is None or float(readouts.grad.abs().max()) == 0.0
        assert float(logits.grad.abs().max()) > 0.0

    def test_uniform_logit_shift_leaves_x_and_the_gradient_alone(self):
        """``x`` is shift-invariant as a function; the gradient must be too.

        With a *detached* bulk reference ``dx/dl_i = -alpha_i``, which sums to -1, so the
        optimiser would read uniform logit inflation as descent even though ``x`` does not
        move. Averaging the reference live makes the sum exactly zero.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=3.0)
        base = size_extensivity_probe(
            logits, readouts, alpha, batch, node_attrs, 1, 1e4
        ).x
        shifted = size_extensivity_probe(
            logits + 7.5, readouts, alpha, batch, node_attrs, 1, 1e4
        ).x
        assert torch.allclose(base, shifted, atol=1e-9)

        live = logits.clone().requires_grad_(True)
        size_extensivity_probe(
            live, readouts, alpha, batch, node_attrs, 1, 1e4
        ).x.sum().backward()
        assert float(live.grad.sum(dim=0).abs().max()) < 1e-9

    def test_log_space_mean_bias_is_bounded_and_safe(self):
        """Why a plain mean is admissible on the ``l`` side at all.

        Averaging ``e^l`` would be dominated by the shell: ``ln(mean e^l)`` overshoots the
        bulk level by ``gap - ln(N_Z/k_Z)``. Averaging ``l`` instead caps the pull at
        ``k_Z gap / N_Z``, which shrinks as ``1/N`` -- and the sign matters as much as the
        size, because overestimating ``ln B`` tightens the constraint and can never loosen
        it.
        """
        num_bulk, num_shell, gap = 280, 6, 12.0
        logits, *_ = two_level_cell(num_bulk=num_bulk, num_shell=num_shell, gap=gap)
        total = num_bulk + num_shell
        # Species 0 holds every other atom, so it carries half the shell.
        column = logits[torch.arange(total) % 2 == 0][:, 0]
        shell_in_species = int((column > 0).sum())
        log_space = float(column.mean())
        linear_space = float(torch.log(torch.exp(column).mean()))

        predicted = shell_in_species * gap / len(column)
        assert log_space == pytest.approx(predicted, rel=1e-6)
        assert log_space < 0.4, "log-space bias must stay small"
        assert linear_space > 7.0, "linear-space averaging is catastrophic, as claimed"
        assert log_space > 0.0, "bias must tighten ln B, never loosen it"

    def test_uniform_state_gradient_is_null(self):
        """On a uniform cell the term cannot *create* a gap, only deepen one.

        A second protection for genuinely delocalised carriers, alongside the ``|c| <=
        tol`` exemption -- and the reason the warmup must sit after the escape phase.
        """
        logits, readouts, alpha, batch, node_attrs = two_level_cell(
            num_shell=0, num_bulk=64, u_bulk=0.2, u_shell=0.2
        )
        live = logits.clone().requires_grad_(True)
        size_extensivity_probe(
            live, readouts, torch.softmax(live, dim=0), batch, node_attrs, 1, 1e4
        ).x.sum().backward()
        assert float(live.grad.abs().max()) < 1e-9


class TestExemption:
    def test_no_contrast_is_exempt_when_the_carrier_is_localised(self):
        """Zero contrast exempts a channel -- but only if its attention is localised.

        The exemption's reasoning holds for the energy: if the weighted pool is within
        ``tol`` of the mean pool, no amount of dilution can move the energy further. The
        default cell here has 6 shell atoms of 66, participation 6.8, which is a real
        localised carrier and keeps the exemption.
        """
        ref, pred = make_inputs(u_bulk=0.3, u_shell=0.3)
        loss = build_loss()
        loss.size_penalty(ref, pred)  # primes the EMA
        assert float(loss.size_penalty(ref, pred)) == 0.0
        assert loss.last_size_exempt > 0

    def test_collapsed_attention_is_refused_the_exemption(self):
        """A flat channel must NOT be exempted, or the constraint disables itself.

        The exemption is self-reinforcing: flat attention drives the contrast to zero, the
        channel is exempted, the hinge switches off, and nothing pulls the attention back.
        Observed end to end -- the perovskite long-range run spent its whole life at
        ``size_f = 1.000`` on all four channels with ``|c| = 0.000``, and finished with its
        live channel uniform over the Cs sublattice (participation 16.1 of 79 atoms, where
        a sublattice is 16) and zero weight on the vacancy shell. The short-range run, whose
        contrast stayed at 0.090, kept participation at 2.0 on the correct two atoms.
        """
        # gap = 0 makes alpha exactly uniform, so the contrast is zero AND the carrier is
        # spread over the whole cell -- the degenerate state, not a delocalised-but-fine one.
        ref, pred = make_inputs(gap=0.0, u_bulk=0.3, u_shell=0.3)
        loss = build_loss()
        loss.size_penalty(ref, pred)
        loss.size_penalty(ref, pred)
        assert loss.last_size_exempt == 0, (
            "a channel whose attention covers the cell must keep the constraint"
        )

    def test_threshold_is_continuous_approaching_t_equals_one(self):
        """``x* -> +inf`` smoothly as ``|c| -> tol``, rather than switching at a branch."""
        loss = build_loss(size_tol=1e-3)
        counts = torch.tensor([[1, 0, 0, 1]])
        previous = -float("inf")
        for magnitude in (2e-3, 1.5e-3, 1.1e-3, 1.01e-3, 1.001e-3):
            loss.contrast_ema = torch.full((CHANNELS,), magnitude, dtype=torch.float64)
            value = float(loss.size_threshold(counts)[0, 0])
            assert value > previous, "threshold must rise monotonically toward +inf"
            previous = value
        loss.contrast_ema = torch.full((CHANNELS,), 1e-3, dtype=torch.float64)
        assert not torch.isfinite(loss.size_threshold(counts)[0, 0])

    def test_zero_tolerance_is_maximally_strict_not_silently_off(self):
        """``tol = 0`` must be the tightest setting, not a disabled one.

        ``t -> 0`` sends ``ln(t/(1-t)) -> -inf``, which is non-finite in the same way the
        exemption's ``+inf`` is. A caller filtering on ``isfinite`` would then skip the
        channel and switch the term off precisely when it was demanded most. Caught in a
        real run, where ``size_viol`` sat at 0 with ``--defect_size_tol 0``.
        """
        ref, pred = make_inputs()
        loss = build_loss(size_tol=0.0)
        loss.size_penalty(ref, pred)  # primes the EMA
        value = float(loss.size_penalty(ref, pred))
        assert value > 0.0, "zero tolerance must produce an active, finite penalty"
        threshold = loss.size_threshold(ref["carrier_counts"])
        assert bool(torch.isfinite(threshold).all())

    def test_tolerance_is_divided_by_the_carrier_count(self):
        """``n_c`` sets a per-channel drift budget rather than an outer weight."""
        loss = build_loss()
        loss.contrast_ema = torch.full((CHANNELS,), 1.0, dtype=torch.float64)
        one = float(loss.size_threshold(torch.tensor([[1, 0, 0, 0]]))[0, 0])
        two = float(loss.size_threshold(torch.tensor([[2, 0, 0, 0]]))[0, 0])
        assert two < one, "two carriers must get a tighter threshold"


class TestPlumbing:
    def test_warmup_gate_is_restart_safe(self):
        ref, pred = make_inputs()
        loss = build_loss(size_warmup_epochs=20)
        loss.current_epoch = 5
        assert float(loss.size_penalty(ref, pred)) == 0.0
        loss.current_epoch = 50
        assert float(loss.size_penalty(ref, pred)) > 0.0

    def test_zero_weight_is_exactly_off(self):
        ref, pred = make_inputs()
        assert float(build_loss(size_weight=0.0).size_penalty(ref, pred)) == 0.0

    def test_dead_channels_are_excluded(self):
        ref_dead, pred = make_inputs(counts=(0, 0, 0, 0))
        assert float(build_loss().size_penalty(ref_dead, pred)) == 0.0

    def test_survives_logits_at_the_clamp(self):
        """``e^40`` overflows float32; the term must live entirely in log space."""
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=40.0)
        probe = size_extensivity_probe(
            logits.to(torch.float32), readouts.to(torch.float32),
            torch.softmax(logits.to(torch.float32), dim=0), batch,
            node_attrs.to(torch.float32), 1, 1e4, logit_clamp=40.0,
        )
        assert torch.isfinite(probe.x).all() and torch.isfinite(probe.contrast).all()
        assert float(probe.clamped[0, 0]) == pytest.approx(6 / 66, abs=1e-6)

    def test_segment_logsumexp_matches_torch(self):
        values = torch.randn(40, CHANNELS, dtype=torch.float64)
        batch = torch.repeat_interleave(torch.arange(4), 10)
        result = segment_logsumexp(values, batch, num_graphs=4)
        for graph in range(4):
            expected = torch.logsumexp(values[batch == graph], dim=0)
            assert torch.allclose(result[graph], expected, atol=1e-12)

    def test_ratio_must_exceed_one(self):
        with pytest.raises(ValueError, match="ratio must exceed 1"):
            size_extensivity_probe(*two_level_cell(), num_graphs=1, ratio=1.0)


class TestSaturationRegressionGuard:
    """Retained from the superseded f-space form.

    It encodes a failure that is invisible in the loss value: at ``R = 1e4`` the sigmoid
    pins at 1, so any term whose gradient carries an ``f(1-f)`` factor is silently inert
    while appearing to train. If anyone reformulates back into energy space this fires.
    """

    def test_f_is_saturated_at_the_default_ratio(self):
        logits, readouts, alpha, batch, node_attrs = two_level_cell(gap=5.0)
        probe = size_extensivity_probe(
            logits, readouts, alpha, batch, node_attrs, 1, 1e4
        )
        assert float(probe.f.min()) > 0.99
        suppression = float((probe.f * (1 - probe.f)).max())
        assert suppression < 1e-2, (
            "f(1-f) is the factor the energy-space gradient carried; it is ~0 here, which "
            "is why the term is formulated in x-space instead"
        )
