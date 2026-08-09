###########################################################################################
# Blocks for charge-aware defect models
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Carrier conditioning and pooling blocks.

The pooling in :class:`CarrierAttentionPooling` is what makes the charge correction
*intensive*: the attention weights are normalised per cell, so the correction scales with
the number of carriers rather than with the number of atoms (plan section 3.2).
"""

import math
from typing import List, Optional, Tuple

import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_mean, scatter_sum
from mace.tools.torch_tools import to_high_precision

NUM_CARRIER_CHANNELS = 4


def segment_softmax(
    logits: torch.Tensor,  # [n_nodes, n_channels]
    batch: torch.Tensor,  # [n_nodes]
    num_graphs: int,
    high_precision: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over the atoms of each cell, independently per channel.

    Returns the attention weights and the per-graph maximum logit. Training batches hold
    several cells, so a plain softmax over the batch would couple them.

    No epsilon is added to the denominator: after subtracting the maximum, the largest
    term contributes exactly 1, so the sum is bounded below by 1 for any non-empty cell
    and ``sum_i alpha_i == 1`` holds to full precision. An epsilon would bias alpha
    exactly where the logit gap is large, which is where the extensivity bound lives.
    """
    dtype = logits.dtype
    # On MPS the upcast is a no-op, so the accumulation runs at the working dtype and
    # sum_i alpha_i == 1 holds only to float32 rounding. See test_defects.py.
    values = to_high_precision(logits) if high_precision else logits

    index = batch.unsqueeze(-1).expand_as(values)
    maxima = torch.full(
        (num_graphs, values.shape[-1]),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    maxima = maxima.scatter_reduce(0, index, values, reduce="amax", include_self=False)

    weights = torch.exp(values - maxima[batch])
    totals = scatter_sum(weights, batch, dim=0, dim_size=num_graphs)
    # Empty graphs (padding) would divide by zero; their weights are empty anyway.
    totals = torch.where(totals > 0, totals, torch.ones_like(totals))
    alpha = weights / totals[batch]
    return alpha.to(dtype), maxima.to(dtype)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )


def zero_last_layer(mlp: torch.nn.Module) -> None:
    """Start a readout at zero, so the model begins at the base potential."""
    last = list(mlp.modules())[-1]
    torch.nn.init.zeros_(last.weight)
    if last.bias is not None:
        torch.nn.init.zeros_(last.bias)


@compile_mode("script")
class CounterEmbedding(torch.nn.Module):
    """Embed the carrier counter vector, once per configuration.

    The embedding is continuous in ``n`` -- no lookup table per discrete state -- so
    fractional counters remain available if ensemble-DFT labels are added later. It is
    deliberately *not* mixed into the message-passing trunk: the trunk must see geometry
    alone for ``E_total(R, 0) == E_base(R)`` to hold for any parameters.
    """

    def __init__(self, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.out_dim = out_dim
        hidden = hidden_dim if hidden_dim is not None else max(out_dim, 16)
        # Four counters plus the derived (q, M_s), which the network would otherwise
        # have to learn to form.
        self.mlp = _mlp(NUM_CARRIER_CHANNELS + 2, hidden, out_dim)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """counts: [n_graphs, 4] -> [n_graphs, out_dim]"""
        charge = (counts[:, 2] + counts[:, 3] - counts[:, 0] - counts[:, 1]).unsqueeze(
            -1
        )
        magnetisation = (
            (counts[:, 0] - counts[:, 2]) - (counts[:, 1] - counts[:, 3])
        ).unsqueeze(-1)
        return self.mlp(torch.cat([counts, charge, magnetisation], dim=-1))


@compile_mode("script")
class CarrierAttentionPooling(torch.nn.Module):
    """Per-channel attention pooling and the short-range charge correction.

    For each carrier channel ``c`` the block predicts a logit and an energy readout per
    atom, normalises the logits over the atoms of the cell, and returns

        ``Delta E_SR = sum_c n_c sum_i alpha_i^c u_i^c``.

    Both networks see the full counter vector, not just their own channel, so
    multi-carrier effects (the on-site U, exchange between co-located carriers) are
    representable and one carrier's spatial profile may depend on the presence of others.
    """

    def __init__(
        self,
        feature_dim: int,
        counter_dim: int,
        hidden_dim: int = 64,
        share_logits_across_spin: bool = False,
        logit_clamp: float = 40.0,
        high_precision_softmax: bool = True,
        zero_u_init: bool = True,
        alpha_mode: str = "logits",
        beta: float = 10.0,
    ):
        super().__init__()
        if alpha_mode not in ("logits", "tied"):
            raise ValueError(
                f"alpha_mode must be 'logits' or 'tied', got '{alpha_mode}'"
            )
        self.share_logits_across_spin = share_logits_across_spin
        self.logit_clamp = logit_clamp
        self.high_precision_softmax = high_precision_softmax
        # 'tied' derives the attention from the carrier site energy itself,
        # alpha = softmax(-beta u), instead of from a separate logit network (plan D1).
        # beta is a fixed gauge constant recorded with the model, like sigma and k_c --
        # not a learnable parameter and not a tuning knob.
        self.alpha_mode = alpha_mode
        self.tied = alpha_mode == "tied"
        self.beta = beta

        in_dim = feature_dim + counter_dim
        self.feature_dim = feature_dim
        self.energy_readouts = torch.nn.ModuleList(
            [_mlp(in_dim, hidden_dim, 1) for _ in range(NUM_CARRIER_CHANNELS)]
        )
        # Under the tie there is no separate logit network at all: an empty ModuleList
        # keeps the attribute (and the optimizer group) valid without holding parameters.
        num_logit_networks = 0 if self.tied else (2 if share_logits_across_spin else NUM_CARRIER_CHANNELS)
        self.logit_readouts = torch.nn.ModuleList(
            [_mlp(in_dim, hidden_dim, 1) for _ in range(num_logit_networks)]
        )
        # u is zero-initialised by default (plan section 9.6 -- KEEP, but for a different
        # reason than the one originally stated).
        #
        # The stated reason was to make a fresh model reproduce the base potential. That
        # reason is wrong: the n = 0 identity is *structural*, coming from the `counts *`
        # prefactor in `delta_sr`, and holds for any parameters whatsoever. Acting on the
        # stated reason, the zero-init was removed -- and the fit got worse.
        #
        # The real reason is optimisation conditioning. At u == 0 the attention has no
        # gradient at all (d(Delta E_SR)/d(logit) vanishes identically), which looks like
        # the problem but is not: u itself keeps its gradient, and from zero it grows
        # *along* the gradient, so the contrast it develops is aligned with the target
        # from the first step. A random u is instead structureless noise contributing a
        # spurious Delta E; the cheapest early loss reduction is to destroy it, so the
        # optimiser travels back to u ~ 0 anyway and arrives with the attention still
        # uniform and nothing to grow from. Measured over three seeds on dataset_beta:
        # escape from the plateau in 3/3 runs with the zero-init, 1/3 without.
        #
        # Do not remove this again by re-deriving from the n = 0 identity.
        self.zero_u_init = zero_u_init
        if zero_u_init:
            for readout in self.energy_readouts:
                zero_last_layer(readout)

    def counter_input_l2(self) -> torch.Tensor:
        """Squared norm of the ``z(n)`` input columns of every ``MLP_u``'s first layer.

        These are the weights through which the carrier site energy depends on the
        counter vector, and they are what makes ``u`` *nonlinear* in ``n``.

        The rationale is subtle and is not gauge suppression (plan A5.4). The base-branch
        gauge is killed by counter *dependency*: ``g(n) = sum_c n_c s_c`` is linear, so an
        observed counter that is the sum of two other observed counters forces
        ``g(n1 + n2) = 2`` against the required ``g = 1``, over-determining the system and
        pinning ``f = 0``. That cancellation is exact only while the correction is linear
        in ``n``; genuine multi-carrier interaction makes it approximate. This penalty
        exists to keep the nonlinearity small enough for the cancellation to bite, while
        still leaving real electron-hole physics representable.
        """
        total = torch.zeros((), device=self.energy_readouts[0][0].weight.device)
        for readout in self.energy_readouts:
            first = readout[0]
            total = total + first.weight[:, self.feature_dim :].pow(2).sum()
        return total

    def forward(
        self,
        node_feats: torch.Tensor,  # [n_nodes, feature_dim] invariant
        counter_emb: torch.Tensor,  # [n_graphs, counter_dim]
        counts: torch.Tensor,  # [n_graphs, 4]
        batch: torch.Tensor,  # [n_nodes]
        num_graphs: int,
        logit_bias: Optional[torch.Tensor] = None,  # [n_nodes, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (Delta E_SR, alpha, u, logit gap, delta_u), the last three per channel."""
        features = torch.cat([node_feats, counter_emb[batch]], dim=-1)

        readout_list: List[torch.Tensor] = []
        for readout in self.energy_readouts:
            readout_list.append(readout(features).squeeze(-1))
        energies = torch.stack(readout_list, dim=-1)  # [n_nodes, 4]

        if self.tied:
            # alpha = softmax(-beta u): the carrier localises where its own site energy is
            # lowest, by construction. This removes the additive gauge freedom in the
            # logits and forbids incoherent states -- bound but delocalised, or localised
            # but unbound -- which the independent networks could represent.
            #
            # Note the correction stays live at u == 0: with alpha uniform,
            # d(Delta E_SR)/du_j = n_c alpha_j [1 - beta (u_j - <u>_alpha)] = n_c / N,
            # so nothing needs seeding and the zero-init costs no gradient here.
            logits = -self.beta * energies
        else:
            logit_list: List[torch.Tensor] = []
            for readout in self.logit_readouts:
                logit_list.append(readout(features).squeeze(-1))
            if self.share_logits_across_spin:
                logits = torch.stack(
                    [logit_list[0], logit_list[0], logit_list[1], logit_list[1]], dim=-1
                )
            else:
                logits = torch.stack(logit_list, dim=-1)  # [n_nodes, 4]
            # Not applied under the tie: clamping would break alpha == softmax(-beta u)
            # exactly where u is largest, which is the bound site the model is supposed
            # to find. segment_softmax subtracts the per-cell maximum anyway, so the
            # large-logit case is already numerically safe.
            logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        if logit_bias is not None:
            # Additive novelty bias (plan D7.1, logit route). Applied after the clamp so
            # the seed is never truncated, and outside the tied branch because there is
            # no separate logit to bias when alpha is derived from u.
            logits = logits + logit_bias

        alpha, maxima = segment_softmax(
            logits, batch, num_graphs, high_precision=self.high_precision_softmax
        )
        pooled = scatter_sum(
            alpha * energies, batch, dim=0, dim_size=num_graphs
        )  # [n_graphs, 4]

        # Every n-dependent term carries an explicit counter factor, so the correction
        # vanishes identically at n = 0 for any parameters.
        delta_sr = (counts * pooled).sum(dim=-1)

        # Diagnostic only (plan section 3.2): with one defect site among N bulk atoms the
        # mean logit tracks the bulk value, so max - mean approximates the logit gap that
        # suppresses the size drift.
        mean_logits = scatter_mean(logits, batch, dim=0, dim_size=num_graphs)
        logit_gap = maxima - mean_logits

        # Delta u: the bulk-minus-bound carrier site energy, in eV. Under the tie this is
        # a physical binding energy rather than a statement about an arbitrary logit
        # scale, and logit_gap == beta * delta_u identically. It is computed the same way
        # in both modes so the two are directly comparable.
        mean_u = scatter_mean(energies, batch, dim=0, dim_size=num_graphs)
        minima = torch.full(
            (num_graphs, energies.shape[-1]),
            float("inf"),
            dtype=energies.dtype,
            device=energies.device,
        )
        minima = minima.scatter_reduce(
            0,
            batch.unsqueeze(-1).expand_as(energies),
            energies,
            reduce="amin",
            include_self=False,
        )
        delta_u = mean_u - minima

        return delta_sr, alpha, energies, logit_gap, delta_u


@compile_mode("script")
class StructuredLatentCharges(torch.nn.Module):
    """Assemble the latent charge from three channels (plan section 3.4, addition 1).

        ``q_i = q_i^host + q_i^pol + q_i^carrier``,  ``sum_i q_i = a q``

    An unstructured per-atom readout conditioned on a global carrier count would give
    every atom a charge shift, so ``sum_i q_i`` would scale with cell size. Instead:

    * ``q^host`` is geometry-only and exactly neutral by mean subtraction;
    * ``q^pol`` is exactly neutral too, carries an explicit counter factor, and is free
      in sign -- it supplies the sign-changing near-field structure that a sign-definite
      carrier channel cannot represent, and hence the multipole content the far field
      needs. It supplies *shape*, never monopole;
    * ``q^carrier`` reuses the attention weights for its shape and a single amplitude
      ``a`` for its magnitude, so it alone carries the monopole.

    ``a`` is the screened-monopole amplitude, ``a = 1/sqrt(epsilon_inf)``, read off the
    trained model. It is a function of the plain mean-pooled host descriptor -- not of an
    attention-weighted pool, which would let the defect's local environment leak into a
    host property -- and it must not depend on ``n``.
    """

    def __init__(
        self,
        feature_dim: int,
        counter_dim: int,
        hidden_dim: int = 64,
        eps_inf_init: float = 1.0,
    ):
        super().__init__()
        self.host_charge = _mlp(feature_dim, hidden_dim, 1)
        self.polarisation = _mlp(feature_dim + counter_dim, hidden_dim, 1)
        self.amplitude = _mlp(feature_dim, hidden_dim, 1)

        # Start at the base potential: no host charges, no polarisation.
        zero_last_layer(self.host_charge)
        zero_last_layer(self.polarisation)
        # softplus(0) ~ 0.69 would put the initial amplitude at an implied eps_inf of
        # about 2.1 by accident; start from a stated gauge instead.
        zero_last_layer(self.amplitude)
        target = 1.0 / math.sqrt(eps_inf_init)
        last = list(self.amplitude.modules())[-1]
        with torch.no_grad():
            last.bias.fill_(math.log(math.expm1(target)))

        # s_c: electron channels are negative, hole channels positive.
        self.register_buffer("carrier_signs", torch.tensor([-1.0, -1.0, 1.0, 1.0]))

    def forward(
        self,
        node_feats: torch.Tensor,  # [n_nodes, feature_dim] invariant, geometry only
        counter_emb: torch.Tensor,  # [n_graphs, counter_dim]
        counts: torch.Tensor,  # [n_graphs, 4]
        alpha: torch.Tensor,  # [n_nodes, 4]
        batch: torch.Tensor,  # [n_nodes]
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (q, q_host, q_carrier, p, a)."""
        host_raw = self.host_charge(node_feats).squeeze(-1)
        host_mean = scatter_mean(host_raw, batch, dim=0, dim_size=num_graphs)
        q_host = host_raw - host_mean[batch]

        polar_raw = self.polarisation(
            torch.cat([node_feats, counter_emb[batch]], dim=-1)
        ).squeeze(-1)
        polar_mean = scatter_mean(polar_raw, batch, dim=0, dim_size=num_graphs)
        polarisation = polar_raw - polar_mean[batch]
        total_carriers = counts.sum(dim=-1)  # [n_graphs]
        q_pol = total_carriers[batch] * polarisation

        host_descriptor = scatter_mean(node_feats, batch, dim=0, dim_size=num_graphs)
        amplitude = torch.nn.functional.softplus(
            self.amplitude(host_descriptor).squeeze(-1)
        )  # [n_graphs]

        signed_counts = self.carrier_signs.unsqueeze(0) * counts  # [n_graphs, 4]
        q_carrier = amplitude[batch] * (alpha * signed_counts[batch]).sum(dim=-1)

        return q_host + q_pol + q_carrier, q_host, q_carrier, polarisation, amplitude
