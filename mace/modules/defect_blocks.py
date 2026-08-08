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
    ):
        super().__init__()
        self.share_logits_across_spin = share_logits_across_spin
        self.logit_clamp = logit_clamp
        self.high_precision_softmax = high_precision_softmax

        in_dim = feature_dim + counter_dim
        self.energy_readouts = torch.nn.ModuleList(
            [_mlp(in_dim, hidden_dim, 1) for _ in range(NUM_CARRIER_CHANNELS)]
        )
        num_logit_networks = 2 if share_logits_across_spin else NUM_CARRIER_CHANNELS
        self.logit_readouts = torch.nn.ModuleList(
            [_mlp(in_dim, hidden_dim, 1) for _ in range(num_logit_networks)]
        )
        # u starts at zero so the correction starts at zero; the logits do not, since a
        # zero logit field is a uniform (maximally delocalised) carrier.
        for readout in self.energy_readouts:
            zero_last_layer(readout)

    def forward(
        self,
        node_feats: torch.Tensor,  # [n_nodes, feature_dim] invariant
        counter_emb: torch.Tensor,  # [n_graphs, counter_dim]
        counts: torch.Tensor,  # [n_graphs, 4]
        batch: torch.Tensor,  # [n_nodes]
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (Delta E_SR [n_graphs], alpha, u [n_nodes, 4], logit gap [n_graphs, 4])."""
        features = torch.cat([node_feats, counter_emb[batch]], dim=-1)

        readout_list: List[torch.Tensor] = []
        for readout in self.energy_readouts:
            readout_list.append(readout(features).squeeze(-1))
        energies = torch.stack(readout_list, dim=-1)  # [n_nodes, 4]

        logit_list: List[torch.Tensor] = []
        for readout in self.logit_readouts:
            logit_list.append(readout(features).squeeze(-1))
        if self.share_logits_across_spin:
            logits = torch.stack(
                [logit_list[0], logit_list[0], logit_list[1], logit_list[1]], dim=-1
            )
        else:
            logits = torch.stack(logit_list, dim=-1)  # [n_nodes, 4]
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)

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

        return delta_sr, alpha, energies, logit_gap


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
