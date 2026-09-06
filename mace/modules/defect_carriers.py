###########################################################################################
# Carrier-count algebra and the multiplicity support guard (v8.1 addendum, sections 3.3-3.4)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Counts that stay valid across electron/hole crossings, and the ``m_F <= 1`` guard.

Everything here is derived from one signed quantity per spin channel::

    d_sigma(S)  = N_sigma(S) - M_VB,sigma^class
    n_e,sigma   = max(d_sigma, 0)          n_h,sigma = max(-d_sigma, 0)
    q_F(S)      = sum_sigma (n_h - n_e) = -sum_sigma d_sigma
    Q_core      = Q_formal(S_ref) - q_F(S_ref)
    Q_formal(S) = Q_core + q_F(S)                      (asserted, never assumed)

The signed form matters because the alternative -- keeping separate electron and hole
counters and updating them additively as charge is added or removed -- goes *negative* the
moment a charge sequence crosses the valence rank. Recomputing from ``d_sigma`` cannot: one
number crosses zero and the two nonnegative counts are read off it.

The absolute multiplicity

    m_F(S) = sum_sigma [ n_e,sigma(S) + n_h,sigma(S) ]

counts every frontier carrier *present in the state*, not the size of a charge difference.
A state with one electron and one hole has ``m_F = 2`` even though ``q_F = 0``.

**Why the guard exists.** The functional currently contains the boundary-common
static-frontier interaction and the complete image interaction of the declared image-active
density, but it does **not** contain a boundary-common, self-interaction-controlled
frontier-frontier functional. Until one exists, any state carrying two or more frontier
carriers would have a distinct-carrier interaction that nothing accounts for -- so the
static/frontier partition, and ``Q_core`` in particular, stops being a harmless bookkeeping
gauge. Such states are refused outright rather than evaluated approximately.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# Until a boundary-common frontier-frontier functional is implemented and validated.
MAX_FRONTIER_MULTIPLICITY = 1

# The reserved ledger slot for that functional, fixed to zero under the guard (addendum 3.4).
PHI_FF_COMMON = 0.0


class UnsupportedStateError(RuntimeError):
    """The state lies outside the validated domain; no energy, force or stress is returned."""


def signed_excess(m_vb: Sequence[int], n_sigma: Sequence[int]) -> Tuple[int, ...]:
    """``d_sigma = N_sigma - M_VB,sigma``, the one quantity the counts are read from."""
    if len(m_vb) != len(n_sigma):
        raise ValueError(f"{len(m_vb)} valence ranks against {len(n_sigma)} electron counts")
    return tuple(int(n) - int(m) for m, n in zip(m_vb, n_sigma))


def counts_from_excess(excess: Sequence[int]
                       ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """``(n_e,sigma, n_h,sigma, q_F)`` from the signed excess."""
    n_e = tuple(max(int(d), 0) for d in excess)
    n_h = tuple(max(-int(d), 0) for d in excess)
    q_f = int(sum(n_h) - sum(n_e))
    return n_e, n_h, q_f


def carrier_multiplicity(n_e: Sequence[int], n_h: Sequence[int]) -> int:
    """``m_F`` -- every frontier carrier present, both signs, both spins."""
    return int(sum(int(x) for x in n_e) + sum(int(x) for x in n_h))


def core_charge(q_formal_ref: int, q_f_ref: int) -> int:
    """``Q_core = Q_formal(S_ref) - q_F(S_ref)``.

    Well defined for a nonzero formal reference too, and that case is retained as a
    synthetic identity test -- but see :func:`require_neutral_reference` for why it may not
    reach the production energy path.
    """
    return int(q_formal_ref) - int(q_f_ref)


def require_neutral_reference(q_formal_ref: int) -> None:
    """The production reference must be neutral.

    The frozen base is calibrated to a neutral reference state, so a charged reference would
    silently reinterpret every base energy. The algebra above is still valid for such a
    state, which is why it stays available as a counting test.
    """
    if int(q_formal_ref) != 0:
        raise UnsupportedStateError(
            f"the production reference state has Q_formal = {int(q_formal_ref)}; the frozen "
            "base is calibrated to a neutral reference, so a charged reference is refused "
            "on the energy path (addendum section 3.3)")


def assert_charge_consistency(q_formal: int, q_core: int, q_f: int, context: str = "") -> None:
    """``Q_formal = Q_core + q_F`` on every frame -- asserted, not assumed."""
    if int(q_formal) != int(q_core) + int(q_f):
        where = f" ({context})" if context else ""
        raise AssertionError(
            f"Q_formal = {int(q_formal)} but Q_core + q_F = {int(q_core)} + {int(q_f)} = "
            f"{int(q_core) + int(q_f)}{where}: the counters and the class integers disagree")


def assert_supported_multiplicity(m_f: int, context: str = "",
                                  limit: int = MAX_FRONTIER_MULTIPLICITY) -> None:
    """Refuse ``m_F > limit`` before any energy, force or stress is built.

    Checked on the *absolute* counts of both the requested and the reference state, never on
    ``|Q_formal(S) - Q_formal(S_ref)|``: a state can carry two carriers while differing from
    the reference by zero net charge.
    """
    if int(m_f) > int(limit):
        where = f" ({context})" if context else ""
        raise UnsupportedStateError(
            f"m_F = {int(m_f)} exceeds the supported multiplicity {int(limit)}{where}: the "
            "boundary-common frontier-frontier functional is not implemented, so this state "
            "has a distinct-carrier interaction nothing accounts for (addendum section 3.4)")


def class_counts(m_vb: Sequence[int], n_sigma: Sequence[int], q_formal_ref: int = 0
                 ) -> dict:
    """The full composition-class inventory, in one place.

    Returns ``d_sigma``, ``n_e``, ``n_h``, ``q_F``, ``Q_core`` and ``m_F`` for the reference
    state. The caller decides whether to enforce neutrality and the multiplicity guard, so
    that synthetic counting tests can exercise the algebra without the production gates.
    """
    excess = signed_excess(m_vb, n_sigma)
    n_e, n_h, q_f = counts_from_excess(excess)
    return {
        "d_sigma": excess,
        "n_e": n_e,
        "n_h": n_h,
        "q_f": q_f,
        "q_core": core_charge(q_formal_ref, q_f),
        "m_f": carrier_multiplicity(n_e, n_h),
    }
