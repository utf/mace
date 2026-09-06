"""Plan v8 sections 2.1, 2.2 and 2.8: the frontier-frontier image term `Phi_FF`.

    Phi_FF = 1/2 B_img[w rho_F,loc, w rho_F,loc],    B_img = G_PBC - G_inf   (section 2.8)
    B_img[rho_F,ext, .] := 0                          (band electrons have no isolated
                                                       counterpart; stated choice)

with `rho_F,loc` the channel-normalised frontier density of section 2.1 -- built from the
head's own Hamiltonian, spectrum and fills through the smooth edge projectors, the
per-channel normalisations and the participation switch `w(P)` -- and the counts the
per-frame integers of the composition class. The energy the model carries is section 2.2's
`Phi_F(Q) - Phi_F(0)`: the term at the frame's state minus the term at the reference fill of
the SAME Hamiltonian (the fills the head already made), so the head correction is exactly
zero at `S = S_ref` and a vacancy frame's own reference cloud (`V_Cl^0` carries one frontier
electron at `Q = 0`) is subtracted rather than counted.

Its derivative in the density matrix reaches the forces by the divided-difference route
(section 2.5): every channel object is a matrix function of `H` through
`defect_counting.channel_matrix`, whose backward is the Daleckii-Krein map, so the autograd
of `Phi_FF` through `H` to the positions IS `Tr(P~ dH/dR)`. Nothing here is detached, and no
eigenvector is differentiated. What `Phi_FF` does NOT yet do is enter `H`: `V_FF = dPhi/dP`
is Stage 5's, and the registry says so.

Gauge: the image part is a periodic-kernel object and `Phi_FF = 0` identically under the
isolated gauge (section 2.6). The Ewald evaluator is the model's own, at
`sigma = r_res` -- the Gaussian width of the density-difference objects (section 2.7) --
so `B_img` is a property of the physical density, not of the numerical `ewald_sigma`
(section 7.3's invariance holds by construction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mace.modules.defect_carriers import UnsupportedStateError
from mace.modules.defect_composition import (ClassRecord, composition_key, frame_counts,
                                             lookup_class)
from mace.modules.defect_counting import ORBITALS_PER_ATOM, channel_matrix
from mace.modules.defect_density import edge_projectors, participation_fraction

# The fills' order inside the head's `occupations` / `mu`: (maj, min, maj_ref, min_ref).
FILL_AT_STATE = (0, 1)
FILL_AT_REFERENCE = (2, 3)
# A channel whose total projector weight falls below this is a level merging into a band:
# logged, not acted on (section 2.1, "the class counts are not changed by it").
LOG_BELOW = 0.5
# The normalisation floor. On a real spectrum the band-edge leakage of the projectors keeps
# every channel's weight far above it; it exists so a synthetic spectrum with NO states in
# a window is a warning and a finite density rather than a division by zero.
WEIGHT_FLOOR = 1e-12
NOT_COMPUTED = -1.0


@dataclass
class FrontierEntry:
    """What the head hands back per graph: the ATTACHED Hamiltonian and its DETACHED
    spectrum and fills, all belonging to that one matrix."""

    H: torch.Tensor                 # [n, n], attached
    lam: torch.Tensor               # [n], float64, detached
    U: torch.Tensor                 # [n, n], float64, detached
    occupations: torch.Tensor       # [4, n], detached
    mu: torch.Tensor                # [4], detached
    # The rigid shift of this graph's levels relative to S_ref (the head's c table for the
    # frame's charge class, section 2.2's per-charge-state constant on the levels). The
    # class edges were aligned at S_ref, so the projectors are applied at `edges + shift`.
    level_shift: torch.Tensor = torch.zeros(())


def channel_site_density(entry: FrontierEntry, fill: int, hole: bool, edges, t_el: float,
                         n_sites: int):
    """`diag(Q^c)` summed per site for one channel of one fill: the unnormalised
    site-resolved density `sum_k g_k |psi_k|^2_i`, and its total. Differentiable in `H`."""
    vbm_al, cbm_al, delta, delta_s = edges
    shift = float(entry.level_shift)
    s_e, s_h = edge_projectors(entry.lam, vbm_al + shift, cbm_al + shift, delta, delta_s)
    s = s_h if hole else s_e
    slope = s * (1.0 - s) / delta_s * (-1.0 if hole else 1.0)
    Q = channel_matrix(entry.H, entry.lam, entry.U, entry.occupations[fill], entry.mu[fill],
                       s, slope, hole, t_el)
    site = torch.diagonal(Q).reshape(n_sites, ORBITALS_PER_ATOM).sum(dim=-1)
    return site, site.sum()


def frontier_site_charges(entry: FrontierEntry, fills: Sequence[int], n_e: Sequence[int],
                          n_h: Sequence[int], edges, t_el: float, n_sites: int,
                          p_star: float, delta_p: float, label: str = "") -> Dict[str, Any]:
    """`rho_F,loc` as site charges (`[n_sites]`, signed, integrating to `q_F`), with the
    participation switch `w` and the per-channel projector weights. The electron and hole
    channels are built separately and never from the sign of `q_F`."""
    device, dtype = entry.H.device, entry.H.dtype
    site = torch.zeros(n_sites, device=device, dtype=dtype)
    weights: Dict[str, float] = {}
    for spin in range(2):
        for count, hole, sign, name in ((int(n_e[spin]), False, -1.0, "e"),
                                        (int(n_h[spin]), True, 1.0, "h")):
            if count == 0:
                continue
            dens, total = channel_site_density(entry, fills[spin], hole, edges, t_el,
                                               n_sites)
            weight = float(total.detach())
            weights[f"{name}{spin}"] = weight
            if weight < LOG_BELOW:
                logging.info("frontier %s channel, spin %d%s: projector weight %.3f < %.1f "
                             "(a level merging into a band); counts unchanged",
                             "hole" if hole else "electron", spin,
                             f" ({label})" if label else "", weight, LOG_BELOW)
            site = site + sign * float(count) * dens / total.clamp_min(WEIGHT_FLOOR)
    q_f = float(sum(int(x) for x in n_h) - sum(int(x) for x in n_e))
    if q_f == 0.0 and not weights:
        p = torch.ones((), device=device, dtype=dtype)
    else:
        p = participation_fraction(site)
    w = torch.sigmoid((p_star - p) / delta_p)
    return {"site": site, "w": w, "p": p, "q_F": q_f, "weights": weights}


def frontier_channels(entry: FrontierEntry, fills: Sequence[int], n_e: Sequence[int],
                      n_h: Sequence[int], edges, t_el: float, n_sites: int, cfg,
                      m_vb: Sequence[int], positions: torch.Tensor, cell: torch.Tensor,
                      label: str = "") -> Tuple[List[Any], Dict[str, Any]]:
    """The active channels of one state as `defect_image.Channel`s (addendum 5.1/5.2).

    Built the explicit way: compact-support windows `B_c = b_c(H_fix)`, the continued-valence
    projector `P_V^fix = Pi_{M_VB}(H_fix)`, the positive excess `r_+(+-Delta P)` of the
    state's own density matrix, `D_c = B_c r_+ B_c`, its normalised site density and the
    exact-plateau localisation weight `w_c = W(N_eff) W(R_eff)` (`defect_windows`). Each
    channel carries its own weight, so electron and hole participation are evaluated
    separately and opposite signs cannot cancel. `edges` are the class's aligned
    `(VBM_al, CBM_al)`; `cfg` is the `WindowConfig`; `m_vb` the per-spin valence rank.

    The gates fire here, before any density is handed on: the projector's gap, the
    occupation tails, the trace bounds -- each an `UnsupportedStateError`.
    """
    from mace.modules import defect_windows as dw
    from mace.modules.defect_counting import fermi_density_difference
    from mace.modules.defect_image import Channel

    H = entry.H
    lam, U = entry.lam.double(), entry.U.double()
    shift = float(entry.level_shift)
    vbm_al, cbm_al = float(edges[0]) + shift, float(edges[1]) + shift
    b_e, b_e_slope = dw.electron_window(lam, vbm_al, cbm_al, cfg)
    b_h, b_h_slope = dw.hole_window(lam, vbm_al, cbm_al, cfg)
    B_e = dw.spectral_function(H, b_e, b_e_slope, spectrum=(lam, U))
    B_h = dw.spectral_function(H, b_h, b_h_slope, spectrum=(lam, U))
    channels: List[Any] = []
    diagnostics: Dict[str, Any] = {}
    for spin in range(2):
        count_e, count_h = int(n_e[spin]), int(n_h[spin])
        if count_e == 0 and count_h == 0:
            continue
        where = f" ({label}, spin {spin})" if label else f" (spin {spin})"
        f = entry.occupations[fills[spin]].double()
        # P_sigma, attached to H by the Daleckii-Krein route of the fill it came from.
        P = fermi_density_difference(H, (float(f.sum()),), (1.0,), t_el, spectrum=(lam, U),
                                     occupations=f.unsqueeze(0),
                                     mus=(entry.mu[fills[spin]],))
        P_V = dw.valence_projector(H, lam, U, int(m_vb[spin]), cfg.gap_floor, label=where)
        R_e, R_h = dw.positive_excess_pair(P - P_V, cfg.eta)
        for count, hole, sign, name, B, R in ((count_e, False, -1.0, "e", B_e, R_e),
                                               (count_h, True, 1.0, "h", B_h, R_h)):
            D = B @ R @ B
            kind = "hole" if hole else "electron"
            # Both channels pass the background gates: the inactive one must be empty
            # inside its window, the active one must be captured by it.
            leakage = dw.background_gates(R, D, count, cfg.leakage_tol,
                                          label=f"{where}, {kind}")
            if count == 0:
                continue
            rho_hat, trace = dw.channel_density(D, n_sites, count, cfg,
                                                label=f"{where}, {kind}")
            w, n_eff, r_eff = dw.localisation(rho_hat, positions, cell, cfg)
            key = f"{name}{spin}"
            channels.append(Channel(name=key, density=rho_hat, count=count, sign=sign,
                                    weight=w))
            diagnostics[key] = dw.ChannelDiagnostics(
                trace=float(trace.detach()), n_eff=float(n_eff.detach()),
                r_eff=float(r_eff.detach()), leakage=float(leakage))
    return channels, diagnostics


def _composition_keys(node_species: torch.Tensor, batch: torch.Tensor, num_graphs: int,
                      atomic_numbers: Sequence[int]) -> List[str]:
    """One class key per graph from a species histogram: no per-atom host round trip."""
    n_species = len(atomic_numbers)
    hist = torch.zeros(num_graphs, n_species, dtype=torch.long, device=batch.device)
    hist.index_put_((batch, node_species.long()), torch.ones_like(batch), accumulate=True)
    hist = hist.cpu().tolist()
    keys = []
    for row in hist:
        numbers = []
        for z, c in zip(atomic_numbers, row):
            numbers.extend([int(z)] * int(c))
        keys.append(composition_key(numbers))
    return keys


def frontier_energy(model, entries: Sequence[FrontierEntry], state, state_ref_spec,
                    node_species: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
                    batch: torch.Tensor, num_graphs: int, gauge: str,
                    label: str = "", training: bool = False) -> Dict[str, Any]:
    """`Phi_FF(S) - Phi_FF(S_ref)` per graph, `[B]`, with the per-graph diagnostics.

    `training`: an uncounted class is a transient of the untrained head (decision 21) --
    the class table is refreshed every epoch and the class is counted once the head has a
    gap -- so in a training forward, or anywhere inside a training session (the trainer
    sets `model.uncounted_class_policy = "zero"` for the base-cache build and the
    validation passes and resets it on the model it saves), its graphs contribute an exact
    zero with a warning; otherwise an uncounted class is refused, as the plan says.

    `entries` are the head's per-graph records for THIS pass; `state` the pass's state batch;
    `state_ref_spec` the model's reference state, which decides per graph whether the term
    is an exact zero (the graph is at `S_ref`) or is evaluated. The class table is required
    for every graph that is evaluated and is not consulted otherwise, so a model without a
    table still runs every reference-state batch.
    """
    device, dtype = positions.device, positions.dtype
    zeros = torch.zeros(num_graphs, device=device, dtype=dtype)
    sentinel = torch.full((num_graphs,), NOT_COMPUTED, device=device, dtype=dtype)
    out: Dict[str, Any] = {"energy": zeros, "w": sentinel.clone(), "p": sentinel.clone(),
                           "q_F": sentinel.clone(), "w_ref": sentinel.clone(),
                           "min_weight": sentinel.clone(), "evaluated": 0}
    off_reference = (~state.is_reference(state_ref_spec)).clone()
    if gauge == "isolated" or not bool(off_reference.any()):
        # Isolated gauge: no image part, identically. All at reference: the difference is
        # an exact zero with no graph, which is what the bit-identity at S_ref needs.
        return out
    table = getattr(model, "composition_classes", None)
    if not table or not table.get("classes"):
        raise RuntimeError(
            "the frontier term needs the composition-class table and this model carries "
            "none: build it with defect_composition.ensure_class_table(model, frames) "
            "(the trainer does so before the first epoch) or evaluate the frame at its "
            "reference state")
    f = model.functional
    delta, delta_s = float(f["delta"]), float(f["delta_s"])
    p_star, delta_p = float(f["p_star"]), float(f["delta_p"])
    t_el = float(model.spectral.t_el)
    keys = _composition_keys(node_species, batch, num_graphs, model.atomic_numbers)
    sizes = torch.bincount(batch, minlength=num_graphs).tolist()
    off = off_reference.tolist()
    charges_now: List[torch.Tensor] = []
    charges_ref: List[torch.Tensor] = []
    w_now = [None] * num_graphs
    w_ref = [None] * num_graphs
    for g in range(num_graphs):
        n_g = int(sizes[g])
        if not off[g]:
            charges_now.append(torch.zeros(n_g, device=device, dtype=dtype))
            charges_ref.append(torch.zeros(n_g, device=device, dtype=dtype))
            continue
        record: ClassRecord = lookup_class(table, _numbers_from_key(keys[g]))
        tolerate = training or getattr(model, "uncounted_class_policy", "refuse") == "zero"
        if tolerate and not record.counted:
            _warn_uncounted(record)
            charges_now.append(torch.zeros(n_g, device=device, dtype=dtype))
            charges_ref.append(torch.zeros(n_g, device=device, dtype=dtype))
            off_reference[g] = False
            continue
        n_e, n_h, q_f = frame_counts(record, state, g)
        edges = (float(record.vbm_al), float(record.cbm_al), delta, delta_s)
        entry = entries[g]
        now = frontier_site_charges(entry, FILL_AT_STATE, n_e, n_h, edges, t_el, n_g,
                                    p_star, delta_p, label=f"{label} graph {g}")
        ref = frontier_site_charges(entry, FILL_AT_REFERENCE, record.n_e, record.n_h, edges,
                                    t_el, n_g, p_star, delta_p,
                                    label=f"{label} graph {g} reference")
        charges_now.append(now["w"] * now["site"])
        charges_ref.append(ref["w"] * ref["site"])
        w_now[g], w_ref[g] = now, ref
        out["evaluated"] += 1
    q_now = torch.cat(charges_now)
    q_ref = torch.cat(charges_ref)
    ewald = model.frontier_ewald
    # 1/2 B_img[w rho, w rho] for both charge sets in two Ewald passes over the batch: the
    # energy is quadratic and per-graph separable, so scaling the charges by `w` beforehand
    # is `w^2` on the energy.
    e_now = 0.5 * (ewald.energy(q_now, positions, cell, batch)
                   - ewald.isolated_energy(q_now, positions, batch, num_graphs))
    e_ref = 0.5 * (ewald.energy(q_ref, positions, cell, batch)
                   - ewald.isolated_energy(q_ref, positions, batch, num_graphs))
    energy = e_now - e_ref
    # Graphs at the reference state contributed zero charge, so their image energies are
    # exact zeros on both sides; written as the zero it is, without a graph.
    mask = off_reference.to(device)
    out["energy"] = torch.where(mask, energy, zeros)
    for g in range(num_graphs):
        if w_now[g] is None:
            continue
        out["w"][g] = w_now[g]["w"].detach().to(dtype)
        out["p"][g] = w_now[g]["p"].detach().to(dtype)
        out["q_F"][g] = float(w_now[g]["q_F"])
        out["w_ref"][g] = w_ref[g]["w"].detach().to(dtype)
        weights = list(w_now[g]["weights"].values()) + list(w_ref[g]["weights"].values())
        out["min_weight"][g] = min(weights) if weights else NOT_COMPUTED
    return out


_WARNED: set = set()


def _warn_uncounted(record: ClassRecord) -> None:
    """Once per (class, reason): the training-time zero is logged, never silent."""
    key = (record.key, record.reason)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logging.warning("frontier term: class %s is uncounted (%s); its graphs contribute ZERO "
                    "in this training forward until the epoch refresh counts it (decision "
                    "21). Evaluation refuses the class.", record.key, record.reason)


def _numbers_from_key(key: str) -> List[int]:
    numbers: List[int] = []
    for part in key.split(","):
        z, c = part.split("x")
        numbers.extend([int(z)] * int(c))
    return numbers


def entries_from_head(records: Optional[Sequence[Optional[Dict[str, Any]]]]
                      ) -> List[Optional[FrontierEntry]]:
    """The per-graph records on the head's output, as entries (None for an empty graph)."""
    if records is None:
        return []
    return [None if e is None else FrontierEntry(**e) for e in records]
