###########################################################################################
# Split constructor cache keys (transition plan v8.1 addendum, section 3.5)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Two cache keys, so a cheap verifier change does not throw away an expensive anchor.

The composition constructor establishes integers by two very different routes. Tier 1 is a
cheap gap verifier; Tier 2 is an expensive valence-subspace continuation along two paths at
two step schedules. Keying both on one fingerprint means that retuning a Tier-1 search
window invalidates the Tier-2 anchors too, and they get recomputed for nothing -- or worse,
someone avoids retuning because recomputing is expensive.

So the common record is partitioned by *what must be true across a homologous size family*:

* :class:`InvariantFields` -- must match **exactly** between two sizes of the same family.
  Note the rank-core schema is here but the Tier-1 router/verifier schema deliberately is
  not: the verifier is allowed to change without invalidating anchors.
* :class:`TargetFields` -- intentionally target-specific. Exact geometry, cell, composition
  and tiling hashes are *expected to differ* between sizes and are never compared for
  equality across them; they are related by the registered tiling and rank-increment
  predicates instead.

Cross-size reuse is therefore **field-wise**, never a single-hash comparison: a single
fingerprint over everything would make every larger cell a cache miss, which is exactly the
reuse the transport equation exists to permit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class InvariantFields:
    """Fields that must be identical across a homologous size family.

    ``rank_core_schema`` versions the counting core. The Tier-1 router/verifier schema is
    explicitly excluded -- it lives on the verifier key -- so that changing the verifier
    cannot invalidate a Tier-2 anchor.
    """

    rank_core_schema: str
    constructor_checkpoint: str
    parameter_hash: str
    host_reference: str
    basis_convention: str
    correspondence_algorithm: str
    homology_algorithm: str
    homology_signature: str
    smearing_convention: str

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class TargetFields:
    """Intentionally target-specific fields; expected to differ between sizes."""

    composition_hash: str
    tiling: Tuple[int, int, int]
    pristine_rank: Tuple[int, ...]
    geometry_hash: str
    cell_hash: str
    n_atoms: int = 0
    composition_difference: Tuple[Tuple[int, int], ...] = ()   # (Z, count) vs pristine

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class NumericalRegime:
    """Eigensolver and arithmetic regime; part of both keys."""

    eigensolver: str
    backend: str
    dtype: str
    precision: str
    tolerances: Tuple[Tuple[str, float], ...] = ()

    @classmethod
    def of(cls, eigensolver: str, backend: str, dtype: str, precision: str,
           tolerances: Optional[Mapping[str, float]] = None) -> "NumericalRegime":
        items = tuple(sorted((str(k), float(v)) for k, v in (tolerances or {}).items()))
        return cls(eigensolver, backend, dtype, precision, items)

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class Tier2AnchorKey:
    """Key of an expensive, still-compatible continuation record."""

    invariant: InvariantFields
    target: TargetFields
    tier2_schema: str
    continuation_paths: Tuple[str, ...]
    step_sizes: Tuple[float, ...]
    e_sink: float
    eta: float
    endpoint_contiguity: bool
    regime: NumericalRegime

    @property
    def digest(self) -> str:
        return _digest({
            "invariant": self.invariant.digest,
            "target": self.target.digest,
            "tier2_schema": self.tier2_schema,
            "paths": list(self.continuation_paths),
            "steps": [round(float(s), 12) for s in self.step_sizes],
            "e_sink": round(float(self.e_sink), 9),
            "eta": round(float(self.eta), 12),
            "contiguity": bool(self.endpoint_contiguity),
            "regime": self.regime.digest,
        })


@dataclass(frozen=True)
class Tier1VerifierKey:
    """Key of the cheap verifier record; carries its own schema version."""

    verifier_schema: str
    invariant: InvariantFields
    target: TargetFields
    source_anchor_hash: str
    pristine_rank_increment: Tuple[int, ...]
    u_al: float
    u_al_provenance: str
    delta_search: float
    perturbations: Tuple[str, ...]
    g_num: float
    regime: NumericalRegime

    @property
    def digest(self) -> str:
        return _digest({
            "verifier_schema": self.verifier_schema,
            "invariant": self.invariant.digest,
            "target": self.target.digest,
            "source_anchor": self.source_anchor_hash,
            "increment": list(self.pristine_rank_increment),
            "u_al": round(float(self.u_al), 12),
            "u_al_provenance": self.u_al_provenance,
            "delta_search": round(float(self.delta_search), 12),
            "perturbations": list(self.perturbations),
            "g_num": round(float(self.g_num), 12),
            "regime": self.regime.digest,
        })


def tiling_volume(tiling: Sequence[int]) -> int:
    volume = 1
    for factor in tiling:
        volume *= int(factor)
    return volume


def compatible_across_sizes(source: Tier2AnchorKey, target: TargetFields,
                            target_invariant: Optional[InvariantFields] = None
                            ) -> Tuple[bool, str]:
    """Field-wise cross-size compatibility (addendum section 3.5).

    Every invariant field must match exactly. The target-specific fields must satisfy the
    registered *relations* -- same composition difference, a consistent pristine tiling, and
    a pristine rank that scales with the tiling volume -- and are never required to be
    equal, because they are expected to differ.
    """
    invariant = target_invariant if target_invariant is not None else source.invariant
    if invariant.digest != source.invariant.digest:
        return False, "an invariant constructor field differs; the records are incompatible"

    if tuple(target.composition_difference) != tuple(source.target.composition_difference):
        return False, ("the composition-difference signature differs, so the classes are not "
                       "homologous")

    src_volume = tiling_volume(source.target.tiling)
    tgt_volume = tiling_volume(target.tiling)
    if src_volume < 1 or tgt_volume < 1:
        return False, "a tiling factor is not a positive integer"

    # The pristine rank is extensive in the tiling, which is exactly what makes the
    # transport equation M_pred = M_acc + [M_pris(L2) - M_pris(L1)] meaningful.
    for src_rank, tgt_rank in zip(source.target.pristine_rank, target.pristine_rank):
        if int(src_rank) * tgt_volume != int(tgt_rank) * src_volume:
            return False, (f"the pristine rank does not scale with the tiling volume: "
                           f"{src_rank} at {source.target.tiling} against {tgt_rank} at "
                           f"{target.tiling}")

    if len(target.pristine_rank) != len(source.target.pristine_rank):
        return False, "the two records disagree about the number of spin channels"

    return True, ""


def pristine_rank_increment(source: TargetFields, target: TargetFields) -> Tuple[int, ...]:
    """``M_pris(L2) - M_pris(L1)``, the only size dependence the raw rank has."""
    return tuple(int(t) - int(s)
                 for s, t in zip(source.pristine_rank, target.pristine_rank))


def invalidates_tier2(old: Tier2AnchorKey, new: Tier2AnchorKey) -> bool:
    """Whether a settings change throws away the expensive anchor."""
    return old.digest != new.digest


def invalidates_tier1(old: Tier1VerifierKey, new: Tier1VerifierKey) -> bool:
    return old.digest != new.digest


def route(anchors: Mapping[str, Any], verifier_key: Tier1VerifierKey,
          verifier_records: Mapping[str, Any]) -> Tuple[str, str]:
    """``(route, reason)`` -- ``"tier1_cached"``, ``"tier1"`` or ``"tier2"``.

    Routing always runs the cheap verifier first and stops when it accepts. Reading a
    compatible anchor fingerprint is a dictionary lookup, not a continuation: this function
    never triggers Tier 2 merely because an anchor exists.
    """
    if verifier_key.digest in verifier_records:
        return "tier1_cached", "a compatible verifier record already exists"
    if verifier_key.source_anchor_hash and verifier_key.source_anchor_hash in anchors:
        return "tier1", "a compatible anchor exists; run the cheap verifier"
    return "tier2", "no compatible anchor; the continuation must establish the rank"
