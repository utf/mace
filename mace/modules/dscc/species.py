"""Plan section 1: the species table and the electronic state.

`n0[Z]` is the neutral valence count of species `Z` in the s+p basis -- Cs 1, Pb 4, Cl 7 --
a convention of the basis, not a host quantity. `N_ref = sum_i n0[Z_i]` is the reference
(neutral) electron count of a frame, split majority-channel: `N_ref_up = ceil(N_ref / 2)`,
`N_ref_dn = floor(N_ref / 2)`. The state `S = {Q, dN_up, dN_dn}` gives `N_S_sigma =
N_ref_sigma + dN_sigma` and `Q = -(dN_up + dN_dn)`; `S_ref = {0, 0, 0}` is the exact
neutral short-circuit (plan section 2.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch

STATE_SCHEMA_VERSION = 1
OCCUPATION_POLICY = "count_fill"

# Plan section 1 / section 11. Registered 2026-09-06.
N0: Dict[int, int] = {55: 1, 82: 4, 17: 7}
# Plan section 2.4: `U_max[Z]` from the GFN1-xTB hardness `GAM` (bounds only, never values):
# Cl 0.519712, Cs 0.085110, Pb 1.000000 Hartree (param_gfn1-xtb.txt, grimme-lab/xtb,
# doi 10.1021/acs.jctc.7b00118), in eV. Registered 2026-09-06.
HARTREE = 27.211386
U_MAX_GFN1: Dict[int, float] = {17: 0.519712 * HARTREE, 55: 0.085110 * HARTREE,
                                82: 1.000000 * HARTREE}


def neutral_count(atomic_numbers: Sequence[int], n0: Mapping[int, int] = N0) -> int:
    """`N_ref = sum_i n0[Z_i]`."""
    total = 0
    for z in atomic_numbers:
        z = int(z)
        if z not in n0:
            raise KeyError(f"species Z={z} has no entry in the species table {dict(n0)}")
        total += n0[z]
    return total


def reference_split(n_ref: int) -> Tuple[int, int]:
    """Majority-channel split of the reference count: `(ceil(N/2), floor(N/2))`."""
    n_ref = int(n_ref)
    return (n_ref + 1) // 2, n_ref // 2


@dataclass(frozen=True)
class State:
    """The electronic state `{Q, dN_up, dN_dn}` (plan section 2.2). Frozen and hashable so
    that canonical equality to `S_ref` is a value comparison, never a label."""
    Q: int
    dN_up: int
    dN_dn: int
    schema_version: int = STATE_SCHEMA_VERSION
    occupation_policy: str = OCCUPATION_POLICY

    def __post_init__(self) -> None:
        if self.Q != -(self.dN_up + self.dN_dn):
            raise ValueError(f"inconsistent state: Q={self.Q} but -(dN_up + dN_dn)="
                             f"{-(self.dN_up + self.dN_dn)}")
        if self.occupation_policy != OCCUPATION_POLICY:
            raise ValueError(f"unknown occupation policy {self.occupation_policy!r}")

    @property
    def is_reference(self) -> bool:
        return self == S_REF

    def counts(self, n_ref: int) -> Tuple[int, int]:
        """`(N_S_up, N_S_dn)` for a frame with reference count `n_ref`."""
        up, dn = reference_split(n_ref)
        return up + self.dN_up, dn + self.dN_dn

    def as_dict(self) -> Dict[str, object]:
        return {"schema_version": self.schema_version, "Q": self.Q, "dN_up": self.dN_up,
                "dN_dn": self.dN_dn, "occupation_policy": self.occupation_policy}


S_REF = State(0, 0, 0)


def state_from_carrier_counts(counts: Sequence[int]) -> State:
    """Decision D2: the dataset's `carrier_counts = (e_maj, e_min, h_maj, h_min)` --
    an electron adds one to its channel, a hole removes one -- give `dN_up = e_maj - h_maj`,
    `dN_dn = e_min - h_min`."""
    e_maj, e_min, h_maj, h_min = (int(c) for c in counts)
    d_up, d_dn = e_maj - h_maj, e_min - h_min
    return State(-(d_up + d_dn), d_up, d_dn)


def states_from_batch(carrier_counts: torch.Tensor) -> Tuple[State, ...]:
    """One `State` per graph from a `[n_graphs, 4]` counter tensor."""
    return tuple(state_from_carrier_counts(row) for row in carrier_counts.detach().cpu().tolist())
