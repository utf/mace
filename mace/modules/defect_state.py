"""Section 2.1 of transition plan v8: the electronic state as a boundary condition.

WHAT THIS SEPARATES. Until now the carrier head was handed four integers -- `(e_maj, e_min,
h_maj, h_min)` -- and everything it knew about the electronic state was folded into that
vector: the formal charge, the per-spin electron counts, and, implicitly, the rule that fills
the spectrum. Three different things travelled in one tensor, and the head's null ("the
correction is exactly zero on the reference") was written as `counts == 0`, which is a
statement about the counter and not about the state.

The state is now a value of its own:

    ElectronicStateSpec S = (state_id, Q_formal, delta_n_sigma, occupation_policy, payload)

with `Q_formal` the exact formal charge, `delta_n_sigma` the exact per-spin electron-count
change relative to the designated reference `S_ref`, and the occupation policy the rule that
turns a spectrum and those counts into occupations. `state_id` is bookkeeping -- it is
excluded from equality on purpose, so two frames carrying different labels for the same
physical state are the same state, and a label can never leak into a number.

THE NULL IS DEFINED ON THE PHYSICAL KEY. `physical_key()` is `(schema version, Q_formal,
delta_n_sigma, policy, payload)`. The head correction is identically zero exactly when a
state's key equals `S_ref`'s. `Q_formal == 0` is not the condition -- a state with the
reference's charge under a different occupation policy is a different state -- and neither
is `counts == 0`, which is the reference only because the current adapter maps it there.

THE ONE PRODUCTION POLICY. `count_fill` is the count-filled smeared occupation the head has
always used, and it is dispatched to the legacy implementation itself
(`head_energy_hf` and `batched_head_energy_hf`) rather than to a rewrite: the plan accepts
Stage 0 only if chemical potentials, occupations, density matrices, band free energies,
energies, forces and stresses are bit-identical, and the only implementation that is
bit-identical to the legacy path is the legacy path. Nothing else is reachable from a model
configuration: `MACEDefect` refuses any other policy key at construction, and the registry
accepts test-only policies solely through `register_test_policy`, which no production module
calls (asserted on the AST in `test_electronic_state.py`).

THE ADAPTER. `StateBatch.from_counts` maps the loader's counter tensor to a batch of states:
`delta_n = (e_maj - h_maj, e_min - h_min)`, `Q_formal = -(delta_n_maj + delta_n_min)` with
`Q_ref = 0`. These are the same integers `spin_targets` has always formed, so the fill the
policy produces is the fill it always produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

__all__ = ["STATE_SCHEMA_VERSION", "COUNT_FILL", "PRODUCTION_POLICIES",
           "ElectronicStateSpec", "StateBatch", "reference_state", "OccupationPolicy",
           "CountFillPolicy", "dispatch", "register_test_policy", "policy_version"]

#: Bumped whenever the meaning or layout of the physical key changes. Part of the key, so a
#: cache written under one schema can never be read under another.
STATE_SCHEMA_VERSION = 1
COUNT_FILL = "count_fill"
#: The policies a model may be CONFIGURED with in Stages 0-6. One entry, deliberately.
PRODUCTION_POLICIES = (COUNT_FILL,)


def _canonical_payload(payload: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    """A hashable, order-independent form of the payload. Empty for `count_fill`."""
    if not payload:
        return ()
    out = []
    for k in sorted(payload):
        v = payload[k]
        if isinstance(v, (list, tuple)):
            v = tuple(float(x) for x in v)
        out.append((str(k), v))
    return tuple(out)


@dataclass(frozen=True)
class ElectronicStateSpec:
    """One electronic state, immutable. See the module docstring for what each field is."""

    q_formal: int
    delta_n: Tuple[int, int]
    occupation_policy: str = COUNT_FILL
    occupation_payload: Tuple[Tuple[str, Any], ...] = ()
    state_id: str = ""
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self):
        if len(self.delta_n) != 2:
            raise ValueError(f"delta_n must be (maj, min), got {self.delta_n!r}")
        if any(int(x) != x for x in self.delta_n) or int(self.q_formal) != self.q_formal:
            raise ValueError("Q_formal and delta_n are exact integers; got "
                             f"{self.q_formal!r}, {self.delta_n!r}")
        object.__setattr__(self, "q_formal", int(self.q_formal))
        object.__setattr__(self, "delta_n", (int(self.delta_n[0]), int(self.delta_n[1])))
        object.__setattr__(self, "occupation_payload",
                           _canonical_payload(dict(self.occupation_payload)
                                              if self.occupation_payload else None))
        if self.occupation_policy == COUNT_FILL and self.occupation_payload:
            raise ValueError("count_fill takes no occupation payload; a payload here would "
                             "be an alternate policy wearing the production key")
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(f"state schema {self.schema_version} is not the current "
                             f"{STATE_SCHEMA_VERSION}")

    # ------------------------------------------------------------------ identity

    def physical_key(self) -> tuple:
        """`(schema, Q_formal, delta_n, policy, payload)`. `state_id` is excluded."""
        return (self.schema_version, self.q_formal, self.delta_n, self.occupation_policy,
                self.occupation_payload)

    def key_digest(self) -> str:
        """A short stable hash of the physical key, for cache keys and file names."""
        return hashlib.blake2b(repr(self.physical_key()).encode(),
                               digest_size=8).hexdigest()

    def is_reference(self, ref: "ElectronicStateSpec") -> bool:
        return self.physical_key() == ref.physical_key()

    def check_against(self, ref: "ElectronicStateSpec") -> None:
        """`Q_formal(S) - Q_ref = -sum_sigma delta_n_sigma(S)`, the identity of section 2.1.

        Raised, not repaired: a state whose charge and counts disagree is a labelling error
        somewhere upstream, and a silent fix here would hide it.
        """
        if ref.delta_n != (0, 0):
            raise ValueError(f"the reference state must have delta_n = (0, 0); got {ref}")
        if self.q_formal - ref.q_formal != -(self.delta_n[0] + self.delta_n[1]):
            raise ValueError(
                f"Q_formal - Q_ref = {self.q_formal - ref.q_formal} but -sum(delta_n) = "
                f"{-(self.delta_n[0] + self.delta_n[1])} for state {self}")

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        return dict(schema_version=self.schema_version, q_formal=self.q_formal,
                    delta_n=list(self.delta_n), occupation_policy=self.occupation_policy,
                    occupation_payload={k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in self.occupation_payload},
                    state_id=self.state_id)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ElectronicStateSpec":
        return cls(q_formal=int(d["q_formal"]), delta_n=tuple(int(x) for x in d["delta_n"]),
                   occupation_policy=str(d.get("occupation_policy", COUNT_FILL)),
                   occupation_payload=_canonical_payload(d.get("occupation_payload")),
                   state_id=str(d.get("state_id", "")),
                   schema_version=int(d.get("schema_version", STATE_SCHEMA_VERSION)))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def reference_state() -> ElectronicStateSpec:
    """`S_ref` for the current programme: the count-filled neutral state, `Q_ref = 0`.

    One reference for every composition class in Stages 0-6. The composition-class
    constructor of section 2.1 will designate per-class references when a class needs one;
    until then this is the designated reference and `MACEDefect` carries it as config.
    """
    return ElectronicStateSpec(q_formal=0, delta_n=(0, 0), occupation_policy=COUNT_FILL,
                               state_id="S_ref")


# ---------------------------------------------------------------------------- batches

@dataclass(frozen=True)
class StateBatch:
    """The states of one batch of graphs, as tensors plus the shared policy.

    `q_formal` is `[B]` long, `delta_n` is `[B, 2]` long. One policy per batch: the
    production forward never mixes policies, and a batch that did would need per-graph
    dispatch inside the eigensolve loop for no case that exists.
    """

    q_formal: torch.Tensor
    delta_n: torch.Tensor
    occupation_policy: str = COUNT_FILL
    occupation_payload: Tuple[Tuple[str, Any], ...] = ()
    state_ids: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_counts(cls, counts: torch.Tensor, policy: str = COUNT_FILL,
                    state_ids: Optional[Sequence[str]] = None) -> "StateBatch":
        """The adapter from `(e_maj, e_min, h_maj, h_min)` per graph.

        `delta_n = (e_maj - h_maj, e_min - h_min)`, `Q_formal = -(dn_maj + dn_min)`: the
        integers `spin_targets` forms, so the count fill is unchanged by the adapter.
        """
        c = counts.reshape(counts.shape[0], -1)
        if c.shape[1] != 4:
            raise ValueError(f"expected [B, 4] counters, got {tuple(counts.shape)}")
        c = torch.round(c).long()
        dn = torch.stack([c[:, 0] - c[:, 2], c[:, 1] - c[:, 3]], dim=-1)
        q = -(dn[:, 0] + dn[:, 1])
        ids = tuple(state_ids) if state_ids is not None else tuple("" for _ in range(c.shape[0]))
        return cls(q_formal=q, delta_n=dn, occupation_policy=str(policy), state_ids=ids)

    def __len__(self) -> int:
        return int(self.q_formal.shape[0])

    def spec(self, g: int) -> ElectronicStateSpec:
        return ElectronicStateSpec(
            q_formal=int(self.q_formal[g]), delta_n=(int(self.delta_n[g, 0]),
                                                     int(self.delta_n[g, 1])),
            occupation_policy=self.occupation_policy,
            occupation_payload=self.occupation_payload,
            state_id=self.state_ids[g] if g < len(self.state_ids) else "")

    def specs(self) -> Tuple[ElectronicStateSpec, ...]:
        return tuple(self.spec(g) for g in range(len(self)))

    def is_reference(self, ref: ElectronicStateSpec) -> torch.Tensor:
        """`[B]` bool: which graphs sit at the reference state's physical key."""
        same_policy = (self.occupation_policy == ref.occupation_policy
                       and self.occupation_payload == ref.occupation_payload)
        if not same_policy:
            return torch.zeros(len(self), dtype=torch.bool, device=self.q_formal.device)
        ref_dn = torch.as_tensor(ref.delta_n, device=self.delta_n.device)
        return (self.q_formal == ref.q_formal) & (self.delta_n == ref_dn).all(dim=-1)

    def all_reference(self, ref: ElectronicStateSpec) -> bool:
        """The batch-level null condition: every graph at `S_ref`."""
        return bool(self.is_reference(ref).all())

    def key_digests(self) -> Tuple[str, ...]:
        return tuple(s.key_digest() for s in self.specs())

    def check_against(self, ref: ElectronicStateSpec) -> None:
        for s in self.specs():
            s.check_against(ref)


# ---------------------------------------------------------------------------- policies

class OccupationPolicy:
    """The dispatch interface. A policy turns `(H, N_total, state)` into a solved fill.

    `solve` handles one graph and `solve_batched` a size-uniform batch, with EXACTLY the
    return shapes of `head_energy_hf` and `batched_head_energy_hf`, because for the one
    production policy they are those functions.
    """

    key: str = ""
    version: int = 0

    def solve(self, H, n_total, counts, t_el, response_out=None, internals=None):
        raise NotImplementedError

    def solve_batched(self, H, n_total, counts, t_el, response_out=None, internals=None):
        raise NotImplementedError


class CountFillPolicy(OccupationPolicy):
    """`count_fill`: the smeared count fill, dispatched to the legacy implementation.

    Looked up through the module at call time rather than bound at import, so anything that
    instruments `defect_counting.head_energy_hf` (the golden capture does) sees this call.
    """

    key = COUNT_FILL
    version = 1

    def solve(self, H, n_total, counts, t_el, response_out=None, internals=None):
        from mace.modules import defect_counting as dc

        return dc.head_energy_hf(H, n_total, counts, t_el, occupation=None,
                                 response_out=response_out, internals=internals)

    def solve_batched(self, H, n_total, counts, t_el, response_out=None, internals=None):
        from mace.modules import defect_counting as dc

        return dc.batched_head_energy_hf(H, n_total, counts, t_el, occupation=None,
                                         response_out=response_out, internals=internals)


_REGISTRY: Dict[str, OccupationPolicy] = {COUNT_FILL: CountFillPolicy()}


def dispatch(policy_key: str) -> OccupationPolicy:
    """The policy object for a key. Unknown keys raise; nothing is defaulted."""
    try:
        return _REGISTRY[str(policy_key)]
    except KeyError:
        raise KeyError(
            f"no occupation policy {policy_key!r} is registered; the production programme "
            f"has exactly {PRODUCTION_POLICIES}") from None


def policy_version(policy_key: str) -> int:
    return int(dispatch(policy_key).version)


def register_test_policy(policy: OccupationPolicy) -> None:
    """Register a TEST-ONLY policy. Refused for any production key.

    A registered test policy is reachable through `dispatch` and nowhere else: `MACEDefect`
    refuses every key outside `PRODUCTION_POLICIES` at construction, so no configuration
    can route a forward through it.
    """
    if policy.key in PRODUCTION_POLICIES:
        raise ValueError(f"{policy.key!r} is a production policy and cannot be replaced")
    if not policy.key:
        raise ValueError("a policy needs a non-empty key")
    _REGISTRY[policy.key] = policy


def registered_policies() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
