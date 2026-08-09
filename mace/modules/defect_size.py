"""Size-extensivity probe for the carrier correction (size plan section 4, as amended).

``alpha = g / sum_j g_j`` with a strictly positive local gate ``g`` cannot be
size-consistent: the denominator carries ``(N - k) * g_bulk``, which grows without bound,
so attention on the defect decays as ``1/N`` once ``N`` passes ``k * e^{gap}``. The gap does
not prevent this, it only sets where it starts.

The constraint needs no new structures, because padding a cell with ideal bulk has a closed
form: adding ``m_Z = (R-1) N_Z`` atoms of each species preserves stoichiometry, and every
added atom contributes the bulk gate and the bulk site energy. Per channel,

    ln A = logsumexp_i(l_i)                    # the cell as it is
    ln B = logsumexp_Z(ln m_Z + lbar_Z)        # the hypothetical padding
    x    = ln B - ln A                         # log-ratio: how much the padding outweighs it
    c    = ubar_w - <u>_alpha                  # the site-energy contrast being diluted

The **energy** drift the padding would cause is ``sigma(x) * c``. Penalising that directly
is what the first version of this module did, and it does not train: ``sigma`` saturates,
and at ``f = sigma(x) -> 1``

    dD/d(lnA) = -f(1-f) c        # raises the gap -- the intent
    dD/dc     = f                # flattens defect contrast -- the pathology

whose ratio is exactly ``(1-f)``. At ``R = 1e4`` on 286-atom cells ``f`` pins at 1.000, the
useful path is suppressed ~50-fold at gap 5 and ~10^4-fold at gap 2, and the feasible set
``|f c| <= tol`` degenerates to ``|c| <= tol``: the term's unique optimum becomes the very
failure it exists to prevent. Lowering ``R`` would restore the gradient but ``R`` states the
validity range, so that trades the specification for the optimiser's convenience.

Because ``sigma`` is monotone, ``sigma(x)|c| <= tol`` iff ``x <= ln(t/(1-t))`` with
``t = tol/|c|``. Penalising in ``x``-space keeps the zero-set exactly and makes the gradient
linear in the gap instead of vanishing with it. This module therefore reports ``x`` and
``c`` separately and leaves the hinge to the loss, which owns the tolerance and the EMA.
"""

from typing import NamedTuple

import torch

from mace.tools.scatter import scatter_sum


class SizeProbe(NamedTuple):
    """Per graph, per channel.

    ``x`` carries gradient; ``contrast`` never does. That asymmetry is the whole point of
    the reformulation -- the term may only act by moving the log-ratio, never by flattening
    the defect contrast to buy the constraint.
    """

    x: torch.Tensor  # [n_graphs, n_channels] log-ratio ln B - ln A
    contrast: torch.Tensor  # [n_graphs, n_channels] ubar_w - <u>_alpha, detached
    f: torch.Tensor  # [n_graphs, n_channels] sigmoid(x); saturation indicator
    clamped: torch.Tensor  # [n_graphs, n_channels] fraction of logits on the clamp bound


def segment_logsumexp(
    values: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """``log sum_i exp(values_i)`` over the nodes of each graph, per channel.

    The maximum is subtracted before exponentiating and is **detached**: it cancels
    analytically in ``d(lnA)/dl_j = softmax_j``, so detaching it changes no gradient while
    removing a spurious path through an argmax.
    """
    index = batch.unsqueeze(-1).expand_as(values)
    maxima = torch.full(
        (num_graphs, values.shape[-1]),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    maxima = maxima.scatter_reduce(0, index, values, reduce="amax", include_self=False)
    shift = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima)).detach()
    totals = scatter_sum(
        torch.exp(values - shift[batch]), batch, dim=0, dim_size=num_graphs
    )
    tiny = torch.finfo(values.dtype).tiny
    return shift + torch.log(totals.clamp_min(tiny))


def _bulk_reference_logits(values: torch.Tensor) -> torch.Tensor:
    """Plain arithmetic mean of the logits: **not** detached, trimmed, or a median.

    Two distinct properties are needed of ``dx/dl_i = w_Z (dlbar_Z/dl_i) - alpha_i``:

    * **shift-invariance**, ``sum_i dx/dl_i = 0``, so uniform logit inflation is not a
      descent direction;
    * the **uniform-state null**, ``dx/dl_i = 0`` *pointwise* on a uniform cell, so the
      term cannot manufacture structure where none exists.

    The plain mean gives ``dlbar_Z/dl_i = 1/N_Z`` for every atom of the species, hence
    ``dx/dl_i = w_Z/N_Z - alpha_i``, which sums to ``sum_Z w_Z - 1 = 0`` unconditionally
    and equals ``1/N - 1/N = 0`` on a uniform cell with matched composition. In the
    localised regime it still points the right way: defect atoms get ``1/N - alpha_i``,
    strongly negative, so their logits rise; bulk atoms get a small positive value, so
    theirs fall.

    The three tempting alternatives each break one of the two. **Detaching** gives
    ``-alpha_i``, summing to ``-1``: uniform inflation becomes a false descent direction.
    **Trimming** gives ``(1/N_Z^kept) 1[i kept]``, which sums to zero but is nonzero
    pointwise on a uniform state -- measured at exactly ``-1/N`` on the trimmed atoms --
    and since two-sided trimming selects by extremal ``l``, arbitrary under ties, it would
    promote seed-determined atoms and reintroduce premature commitment. A **median** has
    the same defect in sharper form.

    Why a mean is safe here at all: the usual objection is that shell atoms at ``e^l ~
    e^{12}`` dominate an average, and that is true of averaging in *linear* space, where
    ``ln(mean_i e^{l_i})`` overestimates ``l_bulk`` by ``gap - ln(N_Z/k_Z)`` ~ 8 at gap 12.
    In **log** space the shift is only ``k_Z gap / N_Z`` -- about 0.25 at N = 286 and gap
    12, rising to 0.38 at gap 18. Bounded, shrinking as ``1/N``, and biased so that ``ln B``
    is over- rather than under-estimated, which tightens the constraint slightly and can
    never loosen it.

    ``ubar_Z`` keeps its median: that side is detached, so robustness there is free and
    costs no gradient property. The asymmetry is the point -- the ``u`` side needs a robust
    value, the ``l`` side needs a clean gradient.
    """
    return values.mean(dim=0)


def size_extensivity_probe(
    logits: torch.Tensor,  # [n_nodes, n_channels]
    readouts: torch.Tensor,  # [n_nodes, n_channels], the site energies u
    alpha: torch.Tensor,  # [n_nodes, n_channels]
    batch: torch.Tensor,  # [n_nodes]
    node_attrs: torch.Tensor,  # [n_nodes, n_species] one-hot
    num_graphs: int,
    ratio: float,
    logit_clamp: float = 40.0,
) -> SizeProbe:
    """The log-ratio ``x``, the contrast ``c``, and two diagnostics.

    ``clamped`` reports how much of the logit field sits on the ``logit_clamp`` bound. This
    term pushes shell logits up and a logit pinned at the clamp receives no gradient, so a
    run can stall there with nothing else to show for it.
    """
    if ratio <= 1.0:
        raise ValueError(f"size ratio must exceed 1, got {ratio}")

    num_species = node_attrs.shape[-1]
    species = node_attrs.argmax(dim=-1)
    detached_readouts = readouts.detach()

    counts = scatter_sum(node_attrs, batch, dim=0, dim_size=num_graphs)  # [G, S]
    padding = (ratio - 1.0) * counts

    # Per-graph, per-species references. The loop runs over (graphs x species) -- order ten
    # iterations per batch -- against a message-passing forward, so its cost is irrelevant.
    reference_logits, reference_u = [], []
    for graph in range(num_graphs):
        in_graph = batch == graph
        per_species_logits, per_species_u = [], []
        for element in range(num_species):
            selection = in_graph & (species == element)
            if not bool(selection.any()):
                zero = torch.zeros(
                    logits.shape[-1], dtype=logits.dtype, device=logits.device
                )
                per_species_logits.append(zero)
                per_species_u.append(zero)
                continue
            per_species_logits.append(_bulk_reference_logits(logits[selection]))
            # u only ever enters the detached contrast, so a plain median is right here.
            per_species_u.append(detached_readouts[selection].median(dim=0).values)
        reference_logits.append(torch.stack(per_species_logits, dim=0))
        reference_u.append(torch.stack(per_species_u, dim=0))
    mean_logits = torch.stack(reference_logits, dim=0)  # [G, S, C]
    median_u = torch.stack(reference_u, dim=0)  # [G, S, C]

    tiny = torch.finfo(logits.dtype).tiny
    log_padding = torch.log(padding.clamp_min(tiny)).unsqueeze(-1)
    terms = log_padding + mean_logits
    terms = terms.masked_fill((padding <= 0).unsqueeze(-1), float("-inf"))

    log_a = segment_logsumexp(logits, batch, num_graphs)
    log_b = torch.logsumexp(terms, dim=1)
    x = log_b - log_a

    # The contrast is detached in full. The term must not be able to satisfy itself by
    # flattening the defect, which is exactly what the superseded f-space form did.
    weights = torch.softmax(terms.detach(), dim=1)
    padded_u = (weights * median_u).sum(dim=1)
    pooled_u = scatter_sum(
        alpha.detach() * detached_readouts, batch, dim=0, dim_size=num_graphs
    )
    contrast = (padded_u - pooled_u).detach()

    at_clamp = (logits.detach().abs() >= logit_clamp - 1e-4).to(logits.dtype)
    totals = scatter_sum(
        torch.ones_like(at_clamp), batch, dim=0, dim_size=num_graphs
    ).clamp_min(1.0)
    clamped = scatter_sum(at_clamp, batch, dim=0, dim_size=num_graphs) / totals
    return SizeProbe(x=x, contrast=contrast, f=torch.sigmoid(x), clamped=clamped)
