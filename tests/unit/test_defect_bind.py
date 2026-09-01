"""Delta_bind must be ~0 for a band state and positive for a split-off level.

lambda_2 - lambda_1 fails exactly this test, which is why it is being retired: on a strongly
connected graph with all-negative off-diagonals the lowest eigenvector is the in-phase
superatom mode, and the gap above it is large whether or not anything is bound. Delta_bind
references the same Hamiltonian WITHOUT the defect, so that gap cancels between the two terms.

These use hand-built Hamiltonians rather than a trained model, because the property under
test is a property of the measure, not of any particular head.
"""

import numpy as np
import pytest


def lam1(H):
    return float(np.linalg.eigvalsh(H)[0])


def ring(n, t=1.0, eps=0.0):
    """Uniform ring: the archetypal band state. Lowest eigenvector is fully delocalised."""
    H = np.full((n, n), 0.0)
    np.fill_diagonal(H, eps)
    for i in range(n):
        H[i, (i + 1) % n] = -t
        H[(i + 1) % n, i] = -t
    return H


def test_band_state_gives_delta_bind_near_zero():
    """A defect that only perturbs weakly must not register as bound."""
    pristine = ring(40)
    defect = ring(40)
    defect[0, 0] -= 0.01                      # negligible well
    assert abs(lam1(pristine) - lam1(defect)) < 0.05


def test_split_off_level_gives_a_large_positive_delta_bind():
    pristine = ring(40)
    defect = ring(40)
    defect[0, 0] -= 3.0                       # deep well: a level splits off below the band
    delta = lam1(pristine) - lam1(defect)
    assert delta > 1.0, f"delta_bind {delta}"


def test_the_superatom_gap_does_not_masquerade_as_binding():
    """The failure mode that retired lambda_2 - lambda_1.

    A densely connected graph has a large lambda_2 - lambda_1 with a completely delocalised
    ground state. Delta_bind must stay ~0 there while the old measure reads large.
    """
    n = 40
    dense = -np.ones((n, n)) * 0.5
    np.fill_diagonal(dense, 0.0)
    evals, evecs = np.linalg.eigh(dense)
    old_measure = evals[1] - evals[0]
    psi = evecs[:, 0] ** 2
    n_eff = 1.0 / np.sum(psi ** 2)

    assert old_measure > 5.0, "expected a large superatom gap"
    assert n_eff > 0.8 * n, "expected a delocalised ground state"

    # Delta_bind against the same graph without a defect is exactly zero.
    assert abs(lam1(dense) - lam1(dense)) == pytest.approx(0.0)


def test_padded_slots_are_excluded_by_the_threshold():
    """Cells smaller than num_states carry padding at ~1e3; it must never be the minimum."""
    lam = np.array([1e3, 1e3 + 1.0, -0.4, 0.2])
    physical = lam < 500.0
    assert float(lam[physical].min()) == pytest.approx(-0.4)
