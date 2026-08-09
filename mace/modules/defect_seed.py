###########################################################################################
# Data-derived contrast seeding for the carrier readouts (forward plan D7.1)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Seed ``MLP_u`` so the attention starts with a geometry-derived, correctly-directed
contrast instead of a flat field.

Why this exists. The gradient reaching site ``i`` is gated by ``alpha_i``, so atoms the
attention never visits are gradient-starved: whichever atoms hold attention when contrast
first develops are the only ones that can later be refined. The plateau is that
exploration, and the zero-init makes it long because ``d(Delta E_SR)/d(logit)`` is
identically zero at ``u == 0``.

What makes this seed different from the one that was tried and withdrawn. It is
**centred**: ``mean(u) = 0`` per frame reproduces the zero-init's *energy* under uniform
attention, so nothing in the loss pressures the seed's destruction -- which is exactly how
the earlier random seed failed, the optimiser spending its first epochs erasing a field
that told it nothing. Meanwhile ``d(Delta E_SR)/d(logit_j) = n_c u_j / N`` is live and
correctly directed from step 0.

The descriptor is a **power spectrum** of the model's own one-particle basis, with the
species embedding replaced by one-hot. Three properties matter:

* it is rotation-invariant *by construction* -- contracting over ``m`` before taking the
  norm. Using equivariant components raw would seed rotated copies of a frame differently;
* it is whitened per ``(n, l, Z)`` channel before the norm, because low-``n``/low-``l``
  blocks dominate by orders of magnitude and an unweighted norm collapses to a
  coordination count, discarding the angular information that distinguishes a dangling
  bond from a vacancy-adjacent-but-saturated site;
* the angular terms exist even at ``max_L = 0``, since the edge basis computes them
  regardless of the output irrep order. The seed can therefore see directional asymmetry
  the readout itself cannot.

It is training-only: not an input, not part of the energy, absent at inference. Plan
section 3.6 (no defect-position or novelty *inputs*) is untouched.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools.scatter import scatter_mean, scatter_sum


@torch.no_grad()
def node_power_spectrum(model: torch.nn.Module, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``p_i^(n,l,Z) = sum_m | sum_j R_n(r_ij) Y_l^m(rhat_ij) delta_{z_j,Z} |^2``.

    Returns ``[n_nodes, num_bessel * (max_ell + 1) * n_species]``.
    """
    positions = data["positions"]
    edge_index = data["edge_index"]
    sender, receiver = edge_index[0], edge_index[1]
    num_nodes = positions.shape[0]

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions, edge_index=edge_index, shifts=data["shifts"]
    )
    # The model's own radial basis and cutoff envelope, so no second notion of locality
    # is introduced. Species-agnostic radial bases ignore node_attrs; the species
    # dependence enters below through the one-hot neighbour channel instead.
    edge_feats, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], edge_index, model.atomic_numbers
    )
    if cutoff is not None:
        edge_feats = edge_feats * cutoff
    edge_sh = model.spherical_harmonics(vectors)  # [n_edges, (max_ell+1)^2]

    num_radial = edge_feats.shape[-1]
    num_species = data["node_attrs"].shape[-1]
    # One-hot of the *neighbour* species: delta_{z_j, Z}.
    neighbour_species = data["node_attrs"][sender]  # [n_edges, n_species]

    max_ell = int(round(edge_sh.shape[-1] ** 0.5)) - 1
    blocks: List[torch.Tensor] = []
    offset = 0
    for ell in range(max_ell + 1):
        width = 2 * ell + 1
        sh_block = edge_sh[:, offset : offset + width]  # [n_edges, 2l+1]
        offset += width
        # outer product over (radial, m, species), summed over the neighbours of each atom
        contribution = (
            edge_feats[:, :, None, None]
            * sh_block[:, None, :, None]
            * neighbour_species[:, None, None, :]
        )  # [n_edges, n_radial, 2l+1, n_species]
        accumulated = scatter_sum(
            contribution.reshape(contribution.shape[0], -1),
            receiver,
            dim=0,
            dim_size=num_nodes,
        ).view(num_nodes, num_radial, width, num_species)
        # Contract over m *before* the norm: this is what makes p rotation-invariant.
        blocks.append(accumulated.pow(2).sum(dim=2))  # [n_nodes, n_radial, n_species]
    return torch.cat([b.reshape(num_nodes, -1) for b in blocks], dim=-1)


@torch.no_grad()
def calibrate_novelty(model: torch.nn.Module, data_loader, device: torch.device) -> Dict[str, float]:
    """Fix the two scale constants that define ``s_hat``, from the training set.

    They are dataset statistics, not learned parameters: the per-``(n,l,Z)`` whitening
    scale and the global scale of the resulting norm. Stored as buffers so they travel
    with the model -- ``s_hat`` is an *input* under logit seeding, so it must be
    reproducible at inference, and its constants are part of the model definition.
    """
    spectra: List[torch.Tensor] = []
    species: List[torch.Tensor] = []
    frames: List[torch.Tensor] = []
    offset = 0
    for batch in data_loader:
        batch = batch.to(device)
        payload = batch.to_dict()
        spectra.append(node_power_spectrum(model, payload).cpu())
        species.append(payload["node_attrs"].argmax(dim=-1).cpu())
        frames.append(batch.batch.cpu() + offset)
        offset += int(batch.num_graphs)

    spectrum = torch.cat(spectra, dim=0)
    element = torch.cat(species, dim=0)
    frame = torch.cat(frames, dim=0)

    channel_scale = spectrum.std(dim=0)
    channel_scale = torch.where(
        channel_scale > 0, channel_scale, torch.ones_like(channel_scale)
    )
    whitened = spectrum / channel_scale

    num_species = int(element.max()) + 1
    key = frame * num_species + element
    means = scatter_mean(whitened, key, dim=0, dim_size=offset * num_species)
    raw = (whitened - means[key]).norm(dim=-1)
    global_scale = float(raw.std())
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    model.novelty_channel_scale.copy_(channel_scale.to(model.novelty_channel_scale))
    model.novelty_global_scale.fill_(global_scale)
    logging.info(
        f"Calibrated the novelty descriptor: {spectrum.shape[-1]} channels over "
        f"{spectrum.shape[0]} atoms in {offset} frames, global scale {global_scale:.4f}"
    )
    return {"global_scale": global_scale, "n_channels": int(spectrum.shape[-1])}


def novelty_from_basis(
    edge_feats: torch.Tensor,
    edge_sh: torch.Tensor,
    edge_index: torch.Tensor,
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    channel_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """``s_hat_i``, built from the trunk's *already computed* edge basis.

    Differentiable in the positions, because it is assembled from the same
    ``edge_feats``/``edge_sh`` the interactions consume rather than from a cached array.
    That is not an optimisation: an additive logit bias is part of the energy, so a
    descriptor held constant would silently break force consistency. Being an analytic
    function of the positions, backpropagating through it is exact.

    Returns per-frame centred values, so a pristine cell -- where every atom looks alike
    -- gives ~0 and leaves the attention uniform (plan 3.2 exactness at init).
    """
    sender, receiver = edge_index[0], edge_index[1]
    num_nodes = node_attrs.shape[0]
    num_radial = edge_feats.shape[-1]
    num_species = node_attrs.shape[-1]
    neighbour_species = node_attrs[sender]

    max_ell = int(round(edge_sh.shape[-1] ** 0.5)) - 1
    blocks: List[torch.Tensor] = []
    offset = 0
    for ell in range(max_ell + 1):
        width = 2 * ell + 1
        sh_block = edge_sh[:, offset : offset + width]
        offset += width
        contribution = (
            edge_feats[:, :, None, None]
            * sh_block[:, None, :, None]
            * neighbour_species[:, None, None, :]
        )
        accumulated = scatter_sum(
            contribution.reshape(contribution.shape[0], -1),
            receiver,
            dim=0,
            dim_size=num_nodes,
        ).view(num_nodes, num_radial, width, num_species)
        blocks.append(accumulated.pow(2).sum(dim=2))
    spectrum = torch.cat([b.reshape(num_nodes, -1) for b in blocks], dim=-1)

    whitened = spectrum / channel_scale
    element = node_attrs.argmax(dim=-1)
    key = batch * num_species + element
    means = scatter_mean(whitened, key, dim=0, dim_size=num_graphs * num_species)
    raw = (whitened - means[key]).norm(dim=-1)
    novelty = raw / global_scale
    return novelty - scatter_mean(novelty, batch, dim=0, dim_size=num_graphs)[batch]


@torch.no_grad()
def anneal_logit_seed(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    zero_by_epoch: int,
    gamma_init: torch.Tensor,
) -> Dict[str, float]:
    """Hand the attention back to ``MLP_l``, per channel, on readiness rather than epochs.

    Why anneal at all, given the descriptor is already differentiated in-graph and the
    forces are FD-exact? Because leaving ``gamma > 0`` makes the seed a permanent
    inference-time term: every downstream consumer must then carry the descriptor and its
    constants, and must differentiate it. Any future path that caches ``s_hat`` as a
    constant -- an obvious optimisation -- silently drops the ``dE/ds_hat * ds_hat/dR``
    term and stops the forces being the gradient of the energy. That is fatal for MD,
    phonons and NEB, and it fails silently. Annealing to zero dissolves the whole class
    of problem: the converged model is bias-free and the production FD check is
    unambiguous.

    Safe here, and *only* here, because ``MLP_l`` was kept. Once ``u`` has contrast,
    ``d(Delta E)/d(logit_j) = n_c alpha_j (u_j - <u>_alpha)`` is non-zero even at uniform
    attention -- at ``alpha_j = 1/N`` it is ``n_c (u_j - <u>)/N`` -- so annealing early
    costs a brief re-plateau with ``u`` already trained, not permanent lock-in. Under the
    tie the same move would be unrecoverable, since there is no separate logit to hand
    back to.

    The schedule keeps the total gap roughly constant: ``gamma_c`` falls as the channel's
    *intrinsic* gap rises, so it is a hand-over, not a cliff. A forced linear ramp runs
    underneath it so ``gamma`` reaches zero by ``zero_by_epoch`` whatever the readiness
    says.

    ``zero_by_epoch`` is an **absolute epoch, not a fraction of the run**. A fractional
    schedule silently breaks under early stopping: the run ends when validation stops
    improving, not at ``max_num_epochs``, so "two thirds of training" is a length nobody
    knows in advance and a model can converge and stop with ``gamma`` still non-zero --
    i.e. with the seed baked into the shipped model, which is the one outcome the anneal
    exists to prevent.
    """
    batch = next(iter(data_loader)).to(device)
    output = model(batch.to_dict(), training=False, compute_force=False)
    intrinsic = output.get("logit_gap_intrinsic")
    seeded = output.get("logit_gap")
    if intrinsic is None or seeded is None:
        return {}

    counts = batch.carrier_counts.view(int(batch.num_graphs), -1)
    live = counts.sum(dim=0) > 0
    intrinsic_mean = intrinsic.mean(dim=0)
    # Target: the gap the seed was supplying at the start.
    target = float(seeded.mean(dim=0)[live].mean()) if bool(live.any()) else 1.0
    target = max(target, 1e-6)

    forced = max(0.0, 1.0 - epoch / max(1.0, float(zero_by_epoch)))
    readiness = (intrinsic_mean / target).clamp(0.0, 1.0)
    scale = torch.minimum(torch.full_like(readiness, forced), 1.0 - readiness)
    updated = (gamma_init.to(scale.device) * scale.clamp(min=0.0)).to(
        model.logit_seed_gamma.dtype
    )
    # Ratchet: the hand-over never hands back. The readiness term is built from a
    # reference gap that itself shrinks as gamma falls, so an unconstrained schedule can
    # let gamma tick back up when the intrinsic gap dips -- re-imposing a prior the model
    # was in the middle of outgrowing.
    #
    # Clamped at zero, because `min` alone *preserves* a negative gamma: min(0, -0.09) is
    # -0.09, not 0. A negative gain does not merely weaken the prior, it inverts it --
    # pushing attention away from the novel atoms. That is how a real run finished at
    # [0.0, -0.0013, 0.0, -0.092] despite a schedule that had reached its terminal epoch.
    updated = torch.minimum(
        updated, model.logit_seed_gamma.detach().clamp(min=0.0)
    ).clamp(min=0.0)
    # Unconditional, and last: at and past the terminal epoch the model must be bias-free
    # whatever the ratchet or anything else has done to gamma.
    if epoch >= zero_by_epoch:
        updated = torch.zeros_like(updated)
    model.logit_seed_gamma.copy_(updated)
    return {
        "gamma": [round(float(v), 4) for v in updated],
        "gap_intrinsic": [round(float(v), 3) for v in intrinsic_mean],
        "forced": round(forced, 3),
    }
