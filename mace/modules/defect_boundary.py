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
from typing import Any, Dict, List, Optional, Sequence

import torch

from mace.modules.defect_carriers import (UnsupportedStateError, assert_supported_multiplicity,
                                          carrier_multiplicity)
from mace.modules.defect_composition import (ClassRecord, frame_counts,
                                             frame_static_densities, lookup_class)
from mace.modules.defect_frontier import (FILL_AT_REFERENCE, FILL_AT_STATE, NOT_COMPUTED,
                                          FrontierEntry, _composition_keys,
                                          _numbers_from_key, _warn_uncounted,
                                          frontier_channels)
from mace.modules.defect_image import boundary_functional, image_active_density
from mace.modules.defect_lift import clearance_report

__all__ = ["stage4_terms"]


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
    f = model.functional
    delta, delta_s = float(f["delta"]), float(f["delta_s"])
    p_star, delta_p = float(f["p_star"]), float(f["delta_p"])
    r_split, r_res = float(f["r_split"]), float(f["r_res"])
    eps_inf = float(model.madelung_eps_inf)
    t_el = float(model.spectral.t_el)
    ewald = model.image_ewald
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
        dens = frame_static_densities(model, record, None, charges_all[mask], pos_g, cell_g,
                                      r_res=r_res, z0=z0, lift=True)
        static, lift = dens["static"], dens["lift"]
        edges = (float(record.vbm_al), float(record.cbm_al), delta, delta_s)
        entry = entries[g]
        now = frontier_channels(entry, FILL_AT_STATE, n_e, n_h, edges, t_el, n_g, p_star,
                                delta_p, label=f"{label} graph {g}")
        ref = frontier_channels(entry, FILL_AT_REFERENCE, record.n_e, record.n_h, edges, t_el,
                                n_g, p_star, delta_p, label=f"{label} graph {g} reference")
        rho_now = image_active_density(static, now, pos_g, record.q_core)
        rho_ref = image_active_density(static, ref, pos_g, record.q_core)
        phi_now = boundary_functional(rho_now, lift, ewald, r_split=r_split, eps_inf=eps_inf,
                                      boundary=boundary)
        phi_ref = boundary_functional(rho_ref, lift, ewald, r_split=r_split, eps_inf=eps_inf,
                                      boundary=boundary)
        energy[g] = (phi_now["phi"] - phi_ref["phi"]).to(dtype)
        for key in ("phi_sf", "phi_img", "phi_img_pbc"):
            out[key][g] = phi_now[key].to(dtype)
            out[f"{key}_ref"][g] = phi_ref[key].to(dtype)
        out["q_img"][g] = rho_now.q_img.detach().to(dtype)
        out["q_img_ref"][g] = rho_ref.q_img.detach().to(dtype)
        weights = [float(w) for w in rho_now.weights.values()] + \
                  [float(w) for w in rho_ref.weights.values()]
        out["w_min"][g] = min(weights) if weights else NOT_COMPUTED
        img = rho_now.img
        report = clearance_report(img.centres.detach() @ torch.linalg.inv(cell_g.detach()),
                                  img.charges.detach(), cell_g.detach(), lift)
        out["clearance_mass"][g] = float(report["buffer_mass"])
        out["lift_fingerprint"][g] = lift.fingerprint
        out["evaluated"] += 1
    mask = off_reference.to(device)
    out["energy"] = torch.where(mask, energy, zeros)
    return out
