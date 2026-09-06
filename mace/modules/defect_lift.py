###########################################################################################
# Canonical isolated-space lift and the IsoOK predicate (v8.1 addendum, section 4.2)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Unwrap a periodic density onto R^3 before any isolated kernel touches it.

A periodic model density lives on the cell torus; ``G_inf`` acts on a density in ``R^3``.
Passing a wrapped density to an isolated kernel is not an approximation, it is a category
error -- the same object has different multipoles depending on where the cell boundary
happens to fall. Every requested/reference evaluation bundle therefore goes through one
registered, component-preserving lift that unwraps ``rho_S``, every signed frontier channel
and their sum *with the same branch choices*, so the algebraic S--S / S--F / F--F
decomposition survives the lift.

**The branch may not be chosen from the density being lifted.** Two reasons, both fatal if
ignored. A branch selected from the variational density would depend on ``P``, adding an
omitted ``delta U / delta P`` term to every derivative; and it would be a hard switch inside
the SCF, so an iterate could flip it and the "converged" solution would be a limit cycle.
The branch therefore comes from an occupation-independent, nonnegative, cancellation-free
support envelope built once at the frozen class-reference geometry out of constructor
topology -- present-minus-pristine species baselines, plus the same departure signal the
residual shape uses. It is mapped covariantly to the current cell, but contains no thermal
displacement, no ``P``, no occupation and no spectral projector.

Cancellation-free matters: a signed density can integrate to zero over a region that is
plainly where the defect is, and its circular moment would then be an arbitrary direction.
The envelope is built from ``sqrt(x^2 + eps^2) - eps`` of the signed topology plus a
nonnegative departure term, so contributions add rather than cancel.

The lift is only half the contract. A Gaussian has no compact support, so the cut always
slices *something*; :func:`clearance_report` measures what, and :func:`iso_ok` refuses the
isolated output unless every clause holds -- not just the frontier being compact, which is
the mistake the addendum was written to correct.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Branch-selection defaults (addendum 4.2). All are recorded in the lift record and enter
# every cache key of a functional that calls G_inf.
EPS_ETA = 1e-8          # smooths the signed topology into a nonnegative envelope
LAMBDA_LIFT = 1.0       # weight of the departure term in the envelope
R_LIFT = 1.0            # A, width of the envelope kernels
Z_MIN = 1e-3            # minimum |circular moment| for a defined support centre
D_CLEAR = 2.0           # A, physical buffer either side of each cut

LIFT_ALGORITHM = "circular-moment/topology-envelope/v1"


class LiftError(RuntimeError):
    """The isolated-space lift is undefined; no isolated output may be produced."""


@dataclass(frozen=True)
class LiftRecord:
    """Everything that makes a lift reproducible, and part of every G_inf cache key."""

    algorithm: str
    centre: Tuple[float, float, float]        # fractional support centre
    cut: Tuple[float, float, float]           # fractional cut position, half a cell away
    moments: Tuple[float, float, float]       # |z_alpha| per direction
    eps_eta: float
    lambda_lift: float
    r_lift: float
    z_min: float
    d_clear: float
    constructor_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({k: (round(v, 12) if isinstance(v, float) else v)
                           for k, v in self.to_dict().items()},
                          sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def support_envelope(topology_charges: torch.Tensor, departure: Optional[torch.Tensor] = None,
                     eps_eta: float = EPS_ETA, lambda_lift: float = LAMBDA_LIFT
                     ) -> torch.Tensor:
    """``zeta = sqrt(rho_topo^2 + eps^2) - eps + lambda * departure``, per class site.

    Nonnegative and cancellation-free by construction: a vacancy and an interstitial in the
    same cell reinforce rather than cancel, so the circular moment below points at the
    defect region instead of at whatever asymmetry survives the cancellation.
    """
    if eps_eta <= 0.0 or lambda_lift <= 0.0:
        raise LiftError(f"eps_eta and lambda_lift must be positive, got {eps_eta}, "
                        f"{lambda_lift}")
    zeta = torch.sqrt(topology_charges * topology_charges + eps_eta * eps_eta) - eps_eta
    if departure is not None:
        if departure.shape[0] != zeta.shape[0]:
            raise LiftError(f"departure has {departure.shape[0]} entries for "
                            f"{zeta.shape[0]} class sites")
        zeta = zeta + float(lambda_lift) * departure.clamp_min(0.0)
    return zeta


def circular_moments(weights: torch.Tensor, scaled_positions: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(|z_alpha|, centre_alpha)`` from the first circular moment per direction.

    A plain mean of fractional coordinates is wrong on a torus -- two sites either side of
    the boundary average to the middle of the cell. The circular mean is the correct
    statistic, and its magnitude doubles as the confidence that a centre exists at all:
    weight spread uniformly around the ring gives ``|z| -> 0`` and no defined centre.
    """
    total = weights.sum()
    if float(total) <= 0.0:
        raise LiftError("the support envelope is identically zero; no centre is defined")
    phase = torch.exp(2j * math.pi * scaled_positions.to(torch.float64))
    z = (weights.to(torch.float64).unsqueeze(-1) * phase).sum(dim=0) / total.to(torch.float64)
    centre = torch.angle(z) / (2.0 * math.pi)
    return z.abs(), centre % 1.0


def build_lift(topology_charges: torch.Tensor, scaled_positions: torch.Tensor,
               departure: Optional[torch.Tensor] = None, *,
               eps_eta: float = EPS_ETA, lambda_lift: float = LAMBDA_LIFT,
               r_lift: float = R_LIFT, z_min: float = Z_MIN, d_clear: float = D_CLEAR,
               constructor_hash: str = "") -> LiftRecord:
    """Select the branch once, from constructor topology alone."""
    zeta = support_envelope(topology_charges, departure, eps_eta, lambda_lift)
    magnitude, centre = circular_moments(zeta, scaled_positions)
    if bool((magnitude < z_min).any()):
        raise LiftError(
            f"circular moment {[round(float(m), 5) for m in magnitude]} below z_min "
            f"{z_min} in at least one direction: the support is not localised enough for a "
            "unique cut, so no isolated output is defined")
    # The cut goes half a cell from the centre -- the point furthest from the support.
    cut = (centre + 0.5) % 1.0
    return LiftRecord(
        algorithm=LIFT_ALGORITHM,
        centre=tuple(float(c) for c in centre),
        cut=tuple(float(c) for c in cut),
        moments=tuple(float(m) for m in magnitude),
        eps_eta=float(eps_eta), lambda_lift=float(lambda_lift), r_lift=float(r_lift),
        z_min=float(z_min), d_clear=float(d_clear), constructor_hash=constructor_hash)


#: A component anchor within this fractional distance of the cut (on either side) is "on
#: the cut" and is assigned to the POSITIVE side of the centre, deterministically.
#: Ideal-lattice sites of a centrosymmetric lattice sit half a cell from a site-centred
#: envelope (the antipodal plane of a site is a site plane), so the tie is generic, not
#: measure-zero, and the envelope's centre sits a few 1e-4 off the site (the departure
#: tails); resolving the side by that offset would make the assignment depend on the
#: frame's representation and flip under rewrapping or a finite-difference step. The
#: margin is far below the clearance buffer (`D_CLEAR`), which is what flags such a
#: component for the isolated claim; here only determinism is at stake.
CUT_TIE = 0.01


def image_assignment(anchors: torch.Tensor, record: LiftRecord) -> torch.Tensor:
    """Integer image shift per COMPONENT anchor, fixed with respect to ``P``.

    Everything on the far side of the cut from the centre is pulled back by one cell. The
    result is an *integer* array held constant through the SCF and through the admissible
    finite-difference neighbourhood, so ``d U / d P = 0`` while ``U (d rho / d P)`` is
    retained. The assignment is read on the component ANCHORS (a present atom's matched
    pristine site, addendum 4.2's "component-preserving" lift), never on the thermal
    positions themselves, so a displaced atom and its site are one image.
    """
    centre = torch.tensor(record.centre, dtype=torch.float64, device=anchors.device)
    # Minimum image about the SUPPORT CENTRE: every component is pulled into the half-cell
    # window centred on the support, which is the same thing as unwrapping through a cut
    # placed half a cell away. Measuring the offset from the cut instead is wrong -- a site
    # just past the centre is then more than half a cell from the cut and gets shifted,
    # splitting a compact object across two images. An anchor within CUT_TIE of the cut,
    # whichever side its representation puts it on, goes to the positive side.
    delta = anchors.to(torch.float64) - centre
    image = -torch.round(delta)
    offset = delta + image                                  # in [-0.5, 0.5]
    return image + (offset < -0.5 + CUT_TIE).to(image.dtype)


def lift_positions(scaled_positions: torch.Tensor, cell: torch.Tensor,
                   record: LiftRecord, anchors: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cartesian positions unwrapped onto ``R^3`` under the record's branch, each primitive
    by the image of its component anchor (its own position when it has none): the lifted
    anchor plus the primitive's minimum-image displacement from its anchor, so a primitive
    stored in any periodic representation (wrapped or not) lands with its component."""
    positions = scaled_positions.to(torch.float64)
    if anchors is None:
        shifted = positions + image_assignment(positions, record)
    else:
        anchors = anchors.to(torch.float64)
        relative = positions - anchors
        relative = relative - torch.round(relative.detach())
        shifted = anchors + image_assignment(anchors, record) + relative
    return shifted @ cell.to(torch.float64)


def clearance_report(scaled_positions: torch.Tensor, weights: torch.Tensor,
                     cell: torch.Tensor, record: LiftRecord,
                     anchors: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """How much density sits in the buffer either side of each cut, per COMPONENT.

    A Gaussian has no compact support, so the cut always slices something; the question is
    only whether what it slices is below the registered absolute tolerance. Absolute, not
    relative: a per-atom tolerance would loosen as the cell grows, which is precisely the
    regime the isolated claim is about. With `anchors`, primitives sharing an anchor are one
    component (a thermal atom/site pair, its residual probe, the carrier density on it) and
    the component's mass is the absolute value of its NET charge: a pair that cancels to its
    thermal dipole is not counted twice at `|Z|`. Without anchors every primitive is a
    component (the old, cruder measure).
    """
    lengths = torch.linalg.norm(cell.to(torch.float64), dim=-1)
    cut = torch.tensor(record.cut, dtype=torch.float64, device=scaled_positions.device)
    anchors = scaled_positions if anchors is None else anchors
    anchors = anchors.to(torch.float64)
    # Components: primitives with the same anchor (to 1e-6 fractional).
    key = torch.round(anchors * 1e6)
    _, component = torch.unique(key, dim=0, return_inverse=True)
    n_components = int(component.max()) + 1 if component.numel() else 0
    net = torch.zeros(n_components, dtype=torch.float64, device=anchors.device)
    net.index_add_(0, component, weights.to(torch.float64))
    where = torch.zeros(n_components, 3, dtype=torch.float64, device=anchors.device)
    where.index_copy_(0, component, anchors)
    distance = (where - cut + 0.5) % 1.0 - 0.5                                # signed, [-.5,.5)
    physical = distance.abs() * lengths                                       # A, per axis
    inside = (physical < record.d_clear).any(dim=-1)
    total = float(net.abs().sum())
    mass = float(net.abs()[inside].sum())
    return {"buffer_mass": mass, "total_mass": total,
            "buffer_fraction": (mass / total) if total > 0 else 0.0,
            "n_in_buffer": int(inside.sum()), "n_components": n_components,
            "d_clear": record.d_clear}


@dataclass(frozen=True)
class IsoStatus:
    """The conjunctive predicate's verdict, clause by clause.

    Deliberately not a bool: when an isolated output is withheld, the caller has to be able
    to say WHICH clause failed, and a PBC label-boundary output may still be valid.
    """

    constructor: bool = False
    multiplicity: bool = False
    frontier_compact: bool = False
    registration: bool = False
    localisation: bool = False
    lift: bool = False
    tails: bool = False
    stationary: bool = False
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all((self.constructor, self.multiplicity, self.frontier_compact,
                    self.registration, self.localisation, self.lift, self.tails,
                    self.stationary))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        d["ok"] = self.ok
        return d


def iso_ok(*, constructor_ok: bool, m_f: int, channel_weights: Sequence[float],
           registration_ok: bool, localisation_ok: bool, lift_record: Optional[LiftRecord],
           tail_bounds: Optional[Dict[str, float]] = None,
           tail_tolerances: Optional[Dict[str, float]] = None,
           stationary_ok: bool = True, max_multiplicity: int = 1) -> IsoStatus:
    """The conjunctive ``IsoOK`` predicate of addendum 4.2.

    Every clause is required. Frontier compactness alone is *not* sufficient -- that is the
    specific error this replaces: a carrier-free charged state with an extended static
    density would otherwise pass on a vacuously satisfied frontier clause, so clauses 1 and
    3-6 stay mandatory when there is no active channel.
    """
    reasons: List[str] = []

    if not constructor_ok:
        reasons.append("the constructor/exact-count/static-support gates did not pass")
    multiplicity = int(m_f) <= int(max_multiplicity)
    if not multiplicity:
        reasons.append(f"m_F = {int(m_f)} exceeds {int(max_multiplicity)}")

    # Vacuous when no channel is active -- but the other clauses still gate the output.
    weights = [float(w) for w in channel_weights]
    frontier_compact = all(abs(w - 1.0) <= 1e-12 for w in weights)
    if not frontier_compact:
        reasons.append(f"an active frontier channel has w != 1 ({weights})")

    if not registration_ok:
        reasons.append("the pristine/defect registration is not valid up to the "
                       "equivalence rule")
    if not localisation_ok:
        reasons.append("a constituent density failed the independent localisation test; "
                       "compactness of a difference is not a substitute")

    lift = lift_record is not None
    if not lift:
        reasons.append("no canonical lift: the branch is undefined or unstable")

    tails = True
    if tail_bounds is not None:
        tolerances = tail_tolerances or {}
        for name, value in tail_bounds.items():
            limit = tolerances.get(name)
            if limit is not None and float(value) > float(limit):
                tails = False
                reasons.append(f"tail bound {name} = {value:.3e} exceeds {limit:.3e}")
    if not stationary_ok:
        reasons.append("the boundary-specific stationary/stability gates did not pass")

    return IsoStatus(constructor=bool(constructor_ok), multiplicity=multiplicity,
                     frontier_compact=frontier_compact, registration=bool(registration_ok),
                     localisation=bool(localisation_ok), lift=lift, tails=tails,
                     stationary=bool(stationary_ok), reasons=tuple(reasons))
