###########################################################################################
# Tier-1 rank-certified gap verifier (transition plan v8.1 addendum, section 3.2)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Verify a valence rank established independently; never infer one from the spectrum.

The v8 Tier-1 test asked whether an eigenvalue lay close to the aligned VBM. That tests the
*density of the folded valence manifold*, not whether the manifold is separable from the
frontier space -- so it becomes more likely to fail as the supercell grows even when the
physical valence-frontier gap is unchanged. Folding a 2x1x1 cell puts more valence states
near the edge; nothing physical has changed.

This module replaces it. Tier 1 is a cheap *verifier* of a rank that was established
elsewhere:

* a pristine class's rank is exact by electron count;
* a homologous defect class's rank comes from an accepted anchor at another size,
  transported by the pristine-rank increment

      M_pred(L2) = M_acc(L1) + [M_pris(L2) - M_pris(L1)],

  which is the only size dependence the raw rank has -- the *offset* ``d_sigma`` and
  ``Q_core`` are invariant, while ``M_VB`` itself is extensive and must differ between sizes.

Tier 1 then checks that the consecutive gap **at that predicted rank** is real. It never
searches for a different, better-looking gap: doing so is how a frontier-frontier gap gets
mistaken for the valence-frontier boundary. Any failure routes to Tier 2, and reading a
compatible anchor fingerprint is not itself a Tier-2 continuation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Registered numerical margin added to the measured alignment envelope, in eV.
DEFAULT_TAU_AL_NUM = 1e-3

# Registered numerical gap floor: below this an eigensolver cannot separate two levels.
DEFAULT_G_NUM = 1e-4

# Tolerance for comparisons the registered numbers are meant to satisfy with equality.
_BOUNDARY_TOL = 1e-12


class RankRoutingError(RuntimeError):
    """Tier 1 cannot decide; the caller must route to Tier 2."""


@dataclass(frozen=True)
class RankAnchor:
    """An independently certified spin-resolved rank at one size.

    Only two things may become an anchor: a pristine class (exact by electron count) and a
    class accepted by Tier 2, or by another independently certified route. An older
    VBM-threshold result is *not* grandfathered -- it becomes an anchor only after agreeing
    with an accepted Tier-2 result.
    """

    homology: str                    # composition-difference / homology signature
    n_atoms: int
    m_vb: Tuple[int, ...]            # accepted M_VB,sigma at this size
    pristine_rank: Tuple[int, ...]   # M_VB,sigma^pristine for the same tiling
    source: str                      # "pristine" | "tier2" | "certified"
    source_hash: str                 # provenance of the accepting record
    invariant_key: str = ""          # common-record invariant partition hash

    @property
    def offset(self) -> Tuple[int, ...]:
        """``M_VB - M_pristine``: the size-invariant part of the rank."""
        return tuple(int(m) - int(p) for m, p in zip(self.m_vb, self.pristine_rank))


@dataclass(frozen=True)
class AlignmentBound:
    """``u_al`` with the provenance that makes it reconstructible.

    A deterministic engineering envelope over a fingerprinted constructor domain, not a
    probabilistic confidence bound -- and explicitly not a sample IQR, which is a diagnostic.
    A target class may not certify itself by first enlarging this envelope: it contributes
    only after an independent Tier-2 acceptance, and then only to *later* verifier records.
    """

    value: float
    tau_num: float
    n_records: int
    sources: Tuple[str, ...]
    residuals: Tuple[float, ...] = field(default_factory=tuple)

    @property
    def provenance_hash(self) -> str:
        payload = {
            "sources": list(self.sources),
            "residuals": [round(float(r), 12) for r in self.residuals],
            "tau_num": round(float(self.tau_num), 12),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Tier1Result:
    """The verifier's decision, with every quantity the class record must store."""

    accepted: bool
    m_vb: Optional[Tuple[int, ...]]
    reason: str
    predicted: Tuple[int, ...] = ()
    thresholds: Tuple[float, ...] = ()
    gaps: Tuple[float, ...] = ()
    window: Tuple[float, float] = (float("nan"), float("nan"))
    u_al: float = float("nan")
    delta_search: float = float("nan")
    gap_required: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def predict_rank(anchor: RankAnchor, pristine_rank_target: Sequence[int]) -> Tuple[int, ...]:
    """Transport an accepted rank to another size of the same homologous family.

    ``M_pred(L2) = M_acc(L1) + [M_pris(L2) - M_pris(L1)]``. Equivalently the offset is
    carried across unchanged, which is the statement that actually has physical content.
    """
    if len(pristine_rank_target) != len(anchor.m_vb):
        raise RankRoutingError(
            f"anchor has {len(anchor.m_vb)} spin channels but the target pristine rank has "
            f"{len(pristine_rank_target)}")
    return tuple(int(m) + int(pt) - int(p)
                 for m, p, pt in zip(anchor.m_vb, anchor.pristine_rank, pristine_rank_target))


def select_anchor(anchors: Sequence[RankAnchor], homology: str,
                  pristine_rank_target: Sequence[int],
                  invariant_key: str = "") -> RankAnchor:
    """The unique fingerprint-compatible anchor, or route to Tier 2.

    Two eligible anchors that disagree about the transported rank mean the family record is
    inconsistent; Tier 1 has no authority to pick one, so it hands over.
    """
    eligible = [a for a in anchors
                if a.homology == homology
                and (not invariant_key or not a.invariant_key or a.invariant_key == invariant_key)]
    if not eligible:
        raise RankRoutingError(
            f"no fingerprint-compatible accepted anchor for homology '{homology}'")
    predictions = {predict_rank(a, pristine_rank_target) for a in eligible}
    if len(predictions) > 1:
        raise RankRoutingError(
            f"eligible anchors disagree about the transported rank for '{homology}': "
            f"{sorted(predictions)}")
    # Prefer an exact pristine anchor, then Tier 2, deterministically.
    order = {"pristine": 0, "tier2": 1, "certified": 2}
    return sorted(eligible, key=lambda a: (order.get(a.source, 9), a.n_atoms, a.source_hash))[0]


def alignment_bound(residuals: Sequence[float], sources: Sequence[str],
                    tau_num: float = DEFAULT_TAU_AL_NUM) -> AlignmentBound:
    """``u_al = max_{j,sigma,delta} |eps_{M_acc} - E_v_al| + tau_al_num``.

    ``residuals`` come from calibration records whose ranks were established *independently*
    of this bound -- registered pristine cells and Tier-2-accepted defect classes -- each
    evaluated over the registered precision and small-geometry perturbations.
    """
    residuals = [float(r) for r in residuals]
    if not residuals:
        raise RankRoutingError(
            "no applicable independently ranked calibration records: u_al cannot be "
            "certified, so the class routes to Tier 2")
    return AlignmentBound(value=max(abs(r) for r in residuals) + float(tau_num),
                          tau_num=float(tau_num), n_records=len(residuals),
                          sources=tuple(sources), residuals=tuple(residuals))


def verify(spectrum: Sequence[float], predicted: Sequence[int], *,
           vbm_aligned: float, gap_host: float, smearing: float,
           u_al: float, delta_search: Optional[float] = None,
           g_num: float = DEFAULT_G_NUM,
           perturbed_spectra: Optional[Sequence[Sequence[float]]] = None) -> Tier1Result:
    """Check the consecutive gap at ``predicted``; accept or route to Tier 2.

    ``perturbed_spectra`` are the registered search-window, precision and small-geometry
    perturbation replicas. Condition 4 requires the *same* rank to stay separated across all
    of them; without replicas the caller has not run the stability test and condition 4 is
    reported as untested rather than silently passed.
    """
    eig = np.sort(np.asarray(spectrum, dtype=np.float64))
    k_total = eig.size
    predicted = tuple(int(k) for k in predicted)

    # Condition 5: the host gap must itself be resolvable at this smearing.
    if gap_host < 4.0 * smearing:
        return Tier1Result(False, None,
                           f"pristine gap {gap_host:.4f} eV < 4 x smearing "
                           f"{4 * smearing:.4f} eV", predicted=predicted, u_al=u_al)

    if delta_search is None:
        delta_search = u_al + 2.0 * smearing
    # The addendum permits equality, and the registered numbers hit it exactly
    # (0.30 = 0.20 + 2 x 0.05), where binary floating point rounds the sum upward.
    required_search = u_al + 2.0 * smearing
    if delta_search < required_search - _BOUNDARY_TOL * max(1.0, abs(required_search)):
        return Tier1Result(False, None,
                           f"delta_search {delta_search:.4f} eV violates the registered "
                           f"inequality u_al + 2 s_smear = {u_al + 2 * smearing:.4f} eV",
                           predicted=predicted, u_al=u_al, delta_search=delta_search)

    gap_required = max(4.0 * smearing, 2.0 * u_al, g_num)
    thresholds: List[float] = []
    gaps: List[float] = []

    # The window must hold for EVERY edge shift in [-u_al, +u_al]. Shifting the edge shifts
    # the window, so the operative test is the intersection over that interval.
    # Non-degenerate by construction: lo - hi = 2(u_al - delta_search) - g_host/2, and the
    # enforced delta_search >= u_al + 2 s_smear makes that at most -4 s_smear - g_host/2 < 0.
    # An oversized u_al is therefore caught by the delta_search inequality above -- which is
    # exactly the addendum's "u_al <= 0.20 eV at s_smear = 0.05, otherwise route to Tier 2"
    # once a registered delta_search (0.30 eV for the current host) is supplied.
    lo = vbm_aligned + u_al - delta_search
    hi = vbm_aligned - u_al + 0.5 * gap_host + delta_search

    for k in predicted:
        # Condition 1.
        if not 1 <= k < k_total:
            return Tier1Result(False, None,
                               f"predicted rank {k} outside 1..{k_total - 1}",
                               predicted=predicted, u_al=u_al, delta_search=delta_search,
                               window=(lo, hi), gap_required=gap_required)
        gap = float(eig[k] - eig[k - 1])
        threshold = 0.5 * float(eig[k] + eig[k - 1])
        thresholds.append(threshold)
        gaps.append(gap)
        # Condition 3.
        if gap < gap_required:
            return Tier1Result(False, None,
                               f"gap {gap:.5f} eV at the certified rank {k} is below the "
                               f"required {gap_required:.5f} eV",
                               predicted=predicted, thresholds=tuple(thresholds),
                               gaps=tuple(gaps), window=(lo, hi), u_al=u_al,
                               delta_search=delta_search, gap_required=gap_required)
        # Condition 2.
        if not lo <= threshold <= hi:
            return Tier1Result(False, None,
                               f"threshold {threshold:.4f} eV leaves the window "
                               f"[{lo:.4f}, {hi:.4f}] under a +-{u_al:.4f} eV edge shift",
                               predicted=predicted, thresholds=tuple(thresholds),
                               gaps=tuple(gaps), window=(lo, hi), u_al=u_al,
                               delta_search=delta_search, gap_required=gap_required)

    # Condition 4: the same rank stays separated under every registered perturbation.
    if perturbed_spectra is None:
        return Tier1Result(False, None,
                           "the registered stability perturbations were not supplied, so "
                           "condition 4 is untested; Tier 1 may not accept",
                           predicted=predicted, thresholds=tuple(thresholds),
                           gaps=tuple(gaps), window=(lo, hi), u_al=u_al,
                           delta_search=delta_search, gap_required=gap_required)
    for index, replica in enumerate(perturbed_spectra):
        rep = np.sort(np.asarray(replica, dtype=np.float64))
        for k in predicted:
            if not 1 <= k < rep.size:
                return Tier1Result(False, None,
                                   f"perturbation {index} has too few states for rank {k}",
                                   predicted=predicted, thresholds=tuple(thresholds),
                                   gaps=tuple(gaps), window=(lo, hi), u_al=u_al,
                                   delta_search=delta_search, gap_required=gap_required)
            if float(rep[k] - rep[k - 1]) < gap_required:
                return Tier1Result(False, None,
                                   f"perturbation {index} closes the gap at rank {k} to "
                                   f"{float(rep[k] - rep[k - 1]):.5f} eV",
                                   predicted=predicted, thresholds=tuple(thresholds),
                                   gaps=tuple(gaps), window=(lo, hi), u_al=u_al,
                                   delta_search=delta_search, gap_required=gap_required)

    return Tier1Result(True, predicted, "", predicted=predicted,
                       thresholds=tuple(thresholds), gaps=tuple(gaps), window=(lo, hi),
                       u_al=u_al, delta_search=delta_search, gap_required=gap_required)


def pristine_anchor(homology: str, n_atoms: int, n_sigma: Sequence[int],
                    source_hash: str = "", invariant_key: str = "") -> RankAnchor:
    """A pristine class: ``M_VB,sigma = N_sigma(S_ref)`` exactly, ``Q_core = m_F = 0``.

    This is the extensive rank the transport equation uses, so it is also the pristine
    reference of its own anchor.
    """
    ranks = tuple(int(n) for n in n_sigma)
    return RankAnchor(homology=homology, n_atoms=int(n_atoms), m_vb=ranks,
                      pristine_rank=ranks, source="pristine",
                      source_hash=source_hash or "exact_electron_count",
                      invariant_key=invariant_key)
