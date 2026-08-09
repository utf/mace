"""Size-extensivity probe for the carrier correction (size-extensivity plan, section 4).

``alpha = g / sum_j g_j`` with a strictly positive local gate ``g`` cannot be
size-consistent: the denominator carries ``(N - k) * g_bulk``, which grows without bound,
so attention on the defect decays as ``1/N`` once ``N`` passes ``k * e^{gap}``. The gap does
not prevent this, it only sets where it starts.

The fix asks for invariance over a **ratio** of cell sizes rather than an absolute size.
Padding a cell with ideal bulk has a closed form, so the constraint needs no new structures:
adding ``m_Z = (R-1) * N_Z`` atoms of each species preserves stoichiometry, and every added
atom contributes the bulk gate and the bulk site energy. Per channel,

    ln A = logsumexp_i(l_i)                    # the cell as it is
    ln B = logsumexp_Z(ln m_Z + lbar_Z)        # the hypothetical padding
    f    = sigmoid(ln B - ln A)                # share of the weight the padding would take
    ubar_w = sum_Z softmax_Z(ln m_Z + lbar_Z) * ubar_Z

so the correction at the padded size is ``(1-f) <u>_alpha + f ubar_w`` and the drift the
padding would introduce is ``f (ubar_w - <u>_alpha)``.

Why this quantity and not a localisation penalty: **it penalises dilution, not
delocalisation**. If ``u`` carries no defect contrast -- a genuine band-edge or shallow
carrier, which *should* be spread out -- then ``ubar_w == <u>_alpha`` and the drift is
exactly zero for any ``f``. Only the intermediate regime, weight pinned on a shell while
the remainder leaks to bulk, is penalised. Entropy and participation penalties cannot make
that distinction and would force every state onto the defect.

Deliberately a plain function rather than part of the model: ``MACEDefect`` is
``@compile_mode("script")``, and this never needs to run at inference. Keeping it out of the
forward costs nothing at training time and keeps the exported model untouched.
"""

from typing import Tuple

import torch

from mace.tools.scatter import scatter_sum


def segment_logsumexp(
    values: torch.Tensor,  # [n_nodes, n_channels]
    batch: torch.Tensor,  # [n_nodes]
    num_graphs: int,
) -> torch.Tensor:
    """``log sum_i exp(values_i)`` over the nodes of each graph, per channel.

    The maximum is subtracted before exponentiating and is **detached**: it cancels
    analytically in ``d(lnA)/dl_j = softmax_j``, so detaching it changes no gradient while
    removing a spurious path through an argmax. Detaching ``lnA`` itself would remove the
    gradient this whole term exists to deliver, so only the shift is detached.
    """
    index = batch.unsqueeze(-1).expand_as(values)
    maxima = torch.full(
        (num_graphs, values.shape[-1]),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    maxima = maxima.scatter_reduce(0, index, values, reduce="amax", include_self=False)
    # Empty graphs (padding in a compiled batch) would give -inf and poison the log.
    shift = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima)).detach()
    totals = scatter_sum(
        torch.exp(values - shift[batch]), batch, dim=0, dim_size=num_graphs
    )
    tiny = torch.finfo(values.dtype).tiny
    return shift + torch.log(totals.clamp_min(tiny))


def size_extensivity_probe(
    logits: torch.Tensor,  # [n_nodes, n_channels]
    readouts: torch.Tensor,  # [n_nodes, n_channels], the site energies u
    alpha: torch.Tensor,  # [n_nodes, n_channels]
    batch: torch.Tensor,  # [n_nodes]
    node_attrs: torch.Tensor,  # [n_nodes, n_species] one-hot
    num_graphs: int,
    ratio: float,
    logit_clamp: float = 40.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(f, drift, clamped_fraction)``, each ``[num_graphs, n_channels]``.

    ``drift = f * (ubar_w - <u>_alpha)`` is the energy error an ``R``-fold larger cell would
    show, in eV, and is the quantity the loss penalises.

    Gradients are routed deliberately. ``lbar_Z`` and ``ubar_Z`` are **detached**, so the
    term pushes the shell logits up rather than dragging the bulk reference down to meet
    them -- and it avoids the sparse gradient of a median selection. ``lnA`` and
    ``<u>_alpha`` stay live.

    ``clamped_fraction`` reports how much of the logit field is sitting on the
    ``logit_clamp`` bound. This term pushes shell logits up, and a logit pinned at the clamp
    receives no gradient, so a run can stall there silently; the hinge should stop the push
    long before that, and this is how you check it did.
    """
    if ratio <= 1.0:
        raise ValueError(f"size ratio must exceed 1, got {ratio}")

    detached_logits = logits.detach()
    detached_readouts = readouts.detach()
    num_species = node_attrs.shape[-1]
    num_channels = logits.shape[-1]
    species = node_attrs.argmax(dim=-1)

    # Per-graph species counts, hence the padding that preserves stoichiometry.
    counts = scatter_sum(node_attrs, batch, dim=0, dim_size=num_graphs)  # [G, S]
    padding = (ratio - 1.0) * counts

    # Per-graph, per-species medians. A mean would be wrong here, not merely noisier: a
    # handful of shell atoms sitting at e^{12} relative to bulk would dominate it, and the
    # bulk reference is exactly what must not move with the defect. The double loop is over
    # (graphs x species) -- order ten iterations per batch -- against a message-passing
    # forward, so its cost is irrelevant.
    shape = (num_graphs, num_species, num_channels)
    median_logits = torch.zeros(shape, dtype=logits.dtype, device=logits.device)
    median_readouts = torch.zeros(shape, dtype=logits.dtype, device=logits.device)
    for graph in range(num_graphs):
        in_graph = batch == graph
        for element in range(num_species):
            selection = in_graph & (species == element)
            if not bool(selection.any()):
                continue
            median_logits[graph, element] = detached_logits[selection].median(dim=0).values
            median_readouts[graph, element] = (
                detached_readouts[selection].median(dim=0).values
            )

    tiny = torch.finfo(logits.dtype).tiny
    log_padding = torch.log(padding.clamp_min(tiny)).unsqueeze(-1)  # [G, S, 1]
    terms = log_padding + median_logits  # [G, S, C]
    # Species absent from a cell contribute nothing rather than log(0).
    terms = terms.masked_fill((padding <= 0).unsqueeze(-1), float("-inf"))

    log_a = segment_logsumexp(logits, batch, num_graphs)  # gradient flows through this
    log_b = torch.logsumexp(terms, dim=1)  # detached: built from medians and counts
    f = torch.sigmoid(log_b - log_a)

    weights = torch.softmax(terms, dim=1)
    padded_u = (weights * median_readouts).sum(dim=1)  # [G, C]
    pooled_u = scatter_sum(alpha * readouts, batch, dim=0, dim_size=num_graphs)
    drift = f * (padded_u - pooled_u)

    at_clamp = (detached_logits.abs() >= logit_clamp - 1e-4).to(logits.dtype)
    clamped = scatter_sum(at_clamp, batch, dim=0, dim_size=num_graphs) / (
        scatter_sum(torch.ones_like(at_clamp), batch, dim=0, dim_size=num_graphs).clamp_min(1.0)
    )
    return f, drift, clamped
