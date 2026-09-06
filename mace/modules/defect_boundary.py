"""Addendum section 8, Stage 4: the forward-only two-boundary diagnostic.

    H_fix = H_local,   P^(0) = count_fill(H_fix, N_S)
    E_head(S) = [Phi_SF^{inf,LR}[P^(0)(S)] + Phi_img^B[P^(0)(S)]] - [the same at S_ref]

evaluated per graph on the model's OWN fill (no potential is fed back into `H`: neither
`dPhi_SF / dP` nor `dPhi_img / dP` reaches the Hamiltonian at this stage -- that is Stage 5's
stationary solve), while the complete response `dP^(0) / d(R, h)` is retained in the forces
and cell derivatives by the divided-difference route the frontier term already uses (every
channel object is a matrix function of the attached `H`).

WHAT IS GATED, IN ORDER, BEFORE ANY ENERGY. The carrier multiplicity of the requested and of
the reference state (`m_F <= 1`, addendum 3.4); the residual-monopole support of the static
density (addendum 4.1); the canonical lift's branch (addendum 4.2 -- a vanishing circular
moment raises, since `G_inf` is called). A failure is an `UnsupportedStateError`, never a
silently evaluated number. The full `IsoOK` predicate is NOT required for this PBC diagnostic
(the addendum reserves it for a boundary-complete claim); its clearance evidence is reported.

WHAT IS REPORTED. `phi_sf` (boundary-common), `phi_img` (what enters the energy under the
active boundary: the periodic image functional, or zero under the isolated boundary) and
`phi_img_pbc` (the periodic value under EITHER boundary, which is the two-boundary diagnostic
and, at `S_ref`, the section-6.3 reference completion), each at the state and at the
reference; `q_img`; the smallest active channel weight; the lift fingerprint and the mass of
`rho_img` inside the cut buffers.

The static density is built against the constructor's own pristine geometry stored in the
class table (`pristine_reference`), with the class's frozen correspondence, and re-registered
on every frame (covariant, differentiated). This module holds no separate image potential
and no pairwise patch: the periodic image addition is `Phi_img` and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mace.modules.defect_carriers import (UnsupportedStateError, assert_supported_multiplicity,
                                          carrier_multiplicity)
from mace.modules.defect_composition import (ClassRecord, frame_counts,
                                             frame_static_densities, lookup_class)
from mace.modules.defect_frontier import (FILL_AT_REFERENCE, FILL_AT_STATE, NOT_COMPUTED,
                                          FrontierEntry, _composition_keys,
                                          _numbers_from_key, _warn_uncounted,
                                          channels_from_density, fill_densities,
                                          window_operators)
from mace.modules.defect_image import (ImageDensity, boundary_functional,
                                       image_active_density)
from mace.modules.defect_lift import clearance_report

__all__ = ["GraphBoundary", "graph_boundary", "phi_b", "boundary_potential", "stage4_terms"]


@dataclass
class GraphBoundary:
    """Everything of one graph that `Phi_B[P]` needs besides `P`: the class record, the
    attached Hamiltonian entry and its compact windows (matrix functions of `H_fix`), the
    static density with its lift record and correspondence, the geometry, the evaluator and
    the registered settings. Built once per graph per forward; `P` varies (the head's own
    fill at Stage 4, the SCF iterate and the stationary solution at Stage 5)."""

    record: ClassRecord
    entry: FrontierEntry
    static: Any
    lift: Any
    windows: Tuple[torch.Tensor, torch.Tensor]
    cfg: Any
    positions: torch.Tensor
    cell: torch.Tensor
    ewald: Any
    r_split: float
    eps_inf: float
    boundary: str
    t_el: float
    correspondence: Dict[str, Any]
    #: The atoms' component anchors for the lift (their matched pristine sites, fractional).
    anchors: Optional[torch.Tensor] = None
    label: str = ""

    @property
    def n_sites(self) -> int:
        return int(self.positions.shape[0])


def graph_boundary(model, record: ClassRecord, entry: FrontierEntry, charges: torch.Tensor,
                   positions: torch.Tensor, cell: torch.Tensor, species: torch.Tensor,
                   boundary: str, label: str = "") -> GraphBoundary:
    """The P-independent part of one graph's boundary functional. Gates: the residual
    support and the lift branch (`frame_static_densities(lift=True)`), both
    `UnsupportedStateError`."""
    from mace.modules.defect_windows import WindowConfig

    f = model.functional
    cfg = WindowConfig.from_functional(f)
    dens = frame_static_densities(model, record, None, charges, positions, cell,
                                  r_res=float(f["r_res"]), z0=model.madelung.z, lift=True,
                                  species=species)
    edges = (float(record.vbm_al), float(record.cbm_al))
    return GraphBoundary(
        record=record, entry=entry, static=dens["static"], lift=dens["lift"],
        windows=window_operators(entry, edges, cfg), cfg=cfg, positions=positions, cell=cell,
        ewald=model.image_ewald, r_split=float(f["r_split"]),
        eps_inf=float(model.madelung_eps_inf), boundary=boundary,
        t_el=float(model.spectral.t_el), correspondence=dens["correspondence"],
        anchors=dens["anchors"], label=label)


def phi_b(ctx: GraphBoundary, P: Sequence[torch.Tensor], n_e: Sequence[int], n_h: Sequence[int],
          label: str = "") -> Tuple[Dict[str, torch.Tensor], ImageDensity, Dict[str, Any]]:
    """`Phi_B[P]` (section 6.2) of one graph for an INDEPENDENT `P` (one matrix per spin),
    with the image-active density and the channel diagnostics: the channels from `P`, the
    signed image density, the boundary functional."""
    channels, diagnostics = channels_from_density(
        ctx.entry, P, n_e, n_h, ctx.record.m_vb, ctx.windows, ctx.cfg, ctx.n_sites,
        ctx.positions, ctx.cell, label=label or ctx.label)
    rho = image_active_density(ctx.static, channels, ctx.positions, ctx.record.q_core,
                               anchors=ctx.anchors)
    phi = boundary_functional(rho, ctx.lift, ctx.ewald, r_split=ctx.r_split,
                              eps_inf=ctx.eps_inf, boundary=ctx.boundary)
    return phi, rho, diagnostics


def boundary_potential(ctx: GraphBoundary, P: Sequence[torch.Tensor], n_e: Sequence[int],
                       n_h: Sequence[int], label: str = ""
                       ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], ImageDensity,
                                  Dict[str, Any]]:
    """`V_B,sigma = dPhi_B / dP_sigma` at `P` (section 6.2), the exact functional derivative
    of the reported `Phi_B` by autograd through the channel construction (windows and the
    continued-valence projector are fixed matrix functions of `H_fix`; the excess `r_+`,
    the normalisation, the localisation weights and the lifted-density response are all
    differentiated), symmetrised; an exact zero for a spin that carries no channel. `P` is
    taken detached: the potential is a function of the iterate, not of its history.
    Returns `(V, phi, rho, diagnostics)` with `phi` evaluated at this `P`.
    """
    leaves = [p.detach().clone().requires_grad_(True) for p in P]
    # The SCF runs under no_grad (it is a solver); the potential still needs its own graph.
    with torch.enable_grad():
        phi, rho, diagnostics = phi_b(ctx, leaves, n_e, n_h, label=label)
        grads = torch.autograd.grad(phi["phi"], leaves, allow_unused=True)
    V = []
    for p, g in zip(leaves, grads):
        if g is None:
            V.append(torch.zeros_like(p))
        else:
            V.append(0.5 * (g + g.transpose(-1, -2)))
    detached = {k: v.detach() for k, v in phi.items()}
    return V, detached, rho, diagnostics


def _zeros(num_graphs: int, device, dtype) -> torch.Tensor:
    return torch.zeros(num_graphs, device=device, dtype=dtype)


def stage4_terms(model, entries: Sequence[Optional[FrontierEntry]], state, state_ref_spec,
                 node_species: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
                 batch: torch.Tensor, num_graphs: int, boundary: str,
                 feats: Optional[torch.Tensor] = None, centre: Optional[torch.Tensor] = None,
                 label: str = "", training: bool = False) -> Dict[str, Any]:
    """`Phi_B[P^(0)(S)] - Phi_B[P^(0)(S_ref)]` per graph, `[B]`, with its parts and gates.

    `boundary` is the model's gauge (`"periodic"` or `"isolated"`). Graphs at the reference
    state contribute an exact zero with no graph, as the frontier term does.
    """
    device, dtype = positions.device, positions.dtype
    zeros = _zeros(num_graphs, device, dtype)
    sentinel = torch.full((num_graphs,), NOT_COMPUTED, device=device, dtype=dtype)
    out: Dict[str, Any] = {
        "energy": zeros, "phi_sf": zeros.clone(), "phi_img": zeros.clone(),
        "phi_img_pbc": zeros.clone(), "phi_sf_ref": zeros.clone(),
        "phi_img_ref": zeros.clone(), "phi_img_pbc_ref": zeros.clone(),
        "q_img": sentinel.clone(), "q_img_ref": sentinel.clone(),
        "w_min": sentinel.clone(), "clearance_mass": sentinel.clone(),
        "n_eff_max": sentinel.clone(), "r_eff_max": sentinel.clone(),
        "registration_max_displacement": sentinel.clone(),
        "registration_flagged": sentinel.clone(),
        "lift_fingerprint": [None] * num_graphs, "evaluated": 0}
    off_reference = (~state.is_reference(state_ref_spec)).clone()
    if not bool(off_reference.any()):
        return out
    table = getattr(model, "composition_classes", None)
    if not table or not table.get("classes"):
        raise RuntimeError(
            "the boundary functional needs the composition-class table and this model "
            "carries none: build it with defect_composition.ensure_class_table(model, frames) "
            "(the trainer does so before the first epoch) or evaluate the frame at its "
            "reference state")
    madelung = getattr(model, "madelung", None)
    if madelung is None:
        raise RuntimeError("the boundary functional needs the static charges "
                           "(madelung_on_site) to build rho_S")
    t_el = float(model.spectral.t_el)
    keys = _composition_keys(node_species, batch, num_graphs, model.atomic_numbers)
    off = off_reference.tolist()
    # The live static charges of every atom (with any per-site deviation, centred per
    # graph), attached: the parameter gradient of rho_S flows through them.
    charges_all = madelung.charges(node_species, feats, centre, batch, num_graphs)
    z0 = madelung.z
    cells = cell.view(-1, 3, 3)
    energy = zeros.clone()
    for g in range(num_graphs):
        if not off[g]:
            continue
        record: ClassRecord = lookup_class(table, _numbers_from_key(keys[g]))
        tolerate = training or getattr(model, "uncounted_class_policy", "refuse") == "zero"
        if tolerate and not record.counted:
            _warn_uncounted(record)
            off_reference[g] = False
            continue
        n_e, n_h, _ = frame_counts(record, state, g)
        # Addendum 3.4 / section 8 Stage 4: the multiplicity guard on BOTH states, before
        # any density or energy is built.
        assert_supported_multiplicity(carrier_multiplicity(n_e, n_h),
                                      f"{label} graph {g}, requested state")
        assert_supported_multiplicity(record.m_f, f"{label} graph {g}, reference state")
        mask = batch == g
        pos_g = positions[mask]
        cell_g = cells[g]
        n_g = int(pos_g.shape[0])
        # rho_S with the class's frozen correspondence, re-registered on this frame, plus
        # the lift record and the residual-support verdict (both raise when unsupported).
        ctx = graph_boundary(model, record, entries[g], charges_all[mask], pos_g, cell_g,
                             node_species[mask], boundary, label=f"{label} graph {g}")
        lift = ctx.lift
        # P^(0) = count_fill(H_fix, N_S): the head's own fills, attached to H by the
        # Daleckii-Krein route (the density response is in the forces, section 8 Stage 4).
        phi_now, rho_now, diag_now = phi_b(
            ctx, fill_densities(entries[g], FILL_AT_STATE, t_el), n_e, n_h,
            label=f"{label} graph {g}")
        phi_ref, rho_ref, diag_ref = phi_b(
            ctx, fill_densities(entries[g], FILL_AT_REFERENCE, t_el), record.n_e, record.n_h,
            label=f"{label} graph {g} reference")
        energy[g] = (phi_now["phi"] - phi_ref["phi"]).to(dtype)
        for key in ("phi_sf", "phi_img", "phi_img_pbc"):
            out[key][g] = phi_now[key].to(dtype)
            out[f"{key}_ref"][g] = phi_ref[key].to(dtype)
        out["q_img"][g] = rho_now.q_img.detach().to(dtype)
        out["q_img_ref"][g] = rho_ref.q_img.detach().to(dtype)
        weights = [float(w) for w in rho_now.weights.values()] + \
                  [float(w) for w in rho_ref.weights.values()]
        out["w_min"][g] = min(weights) if weights else NOT_COMPUTED
        diags = list(diag_now.values()) + list(diag_ref.values())
        out["n_eff_max"][g] = max(d.n_eff for d in diags) if diags else NOT_COMPUTED
        out["r_eff_max"][g] = max(d.r_eff for d in diags) if diags else NOT_COMPUTED
        img = rho_now.img
        report = clearance_report(img.centres.detach() @ torch.linalg.inv(cell_g.detach()),
                                  img.charges.detach(), cell_g.detach(), lift,
                                  anchors=img.component_anchors())
        out["clearance_mass"][g] = float(report["buffer_mass"])
        corr = ctx.correspondence
        out["registration_max_displacement"][g] = float(corr["max_matched_displacement"])
        out["registration_flagged"][g] = float(corr["flagged"])
        out["lift_fingerprint"][g] = lift.fingerprint
        out["evaluated"] += 1
    mask = off_reference.to(device)
    out["energy"] = torch.where(mask, energy, zeros)
    return out
