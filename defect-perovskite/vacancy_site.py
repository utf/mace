"""Locate the Cl vacancy in a CsPbCl3 V_Cl cell, robustly on thermally distorted frames.

EVALUATION ONLY. The vacancy position and the identity of the two shell Pb are metrics
machinery: they must never enter the model, the loss, or any model input. They exist so that
"did the carrier localise on the right atoms?" can be asked of a model that was never told
where the defect is.

Why not coordination counting
-----------------------------
The obvious locator -- the Pb whose Cl coordination is below the cell maximum -- was written
for vacancies constructed on an ideal supercell and does not survive MD snapshots. On the
validation frames it identifies exactly two under-coordinated Pb in only 106 of 222 frames
(best case, 3.6 A cutoff); tightening or loosening the cutoff makes it worse in one direction
or the other. Thermal distortion moves bulk Pb-Cl bonds past any fixed cutoff.

Why not "the emptiest Pb-Pb bridge"
-----------------------------------
In ABX3 perovskite every Cl bridges a nearest-neighbour Pb-Pb pair, so the vacancy looks like
a bridge with no Cl near its midpoint. But octahedral tilting displaces occupied-bridge Cl by
a median 0.58 A and up to 2.78 A, while the vacancy bridge's nearest Cl sits at ~3.3 A. Those
distributions overlap, so a per-bridge threshold misidentifies the site on ~13% of frames.

What this does instead
----------------------
Global assignment. A cell with N_Pb lead atoms has exactly 3*N_Pb octahedral bridges and, with
one vacancy, 3*N_Pb - 1 chlorines. Match chlorines to bridges one-to-one, minimising total
squared displacement (Hungarian algorithm). A tilted Cl still wins its own bridge because
every other bridge is claimed by its own Cl; the single unmatched bridge is the vacancy. The
decision is made by the global structure rather than by one distance against a threshold.
"""

from __future__ import annotations

import numpy as np
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list

__all__ = ["locate_vacancy", "VacancySite"]


class VacancySite:
    """Where the vacancy is, which Pb form its shell, and how trustworthy that is."""

    def __init__(self, position, shell, cost, runner_up, n_bridges):
        self.position = position          # cartesian, may lie outside the cell
        self.shell = shell                # the two Pb indices bridging the empty site
        self.cost = cost                  # nearest Cl to the vacancy bridge, A
        self.runner_up = runner_up        # same for the next-emptiest bridge, A
        self.n_bridges = n_bridges

    @property
    def margin(self):
        """Separation between the chosen site and the next candidate. Larger is safer."""
        return self.cost - self.runner_up

    def __repr__(self):
        return (f"VacancySite(shell={self.shell.tolist()}, cost={self.cost:.2f} A, "
                f"margin={self.margin:.2f} A)")


def _octahedral_bridges(atoms, pb):
    """The 3*N_Pb nearest-neighbour Pb-Pb pairs, as (index_i, index_j, midpoint)."""
    # Axis neighbours sit at 5.3-6.0 A and the next Pb shell at 7.3-8.0 A, a clean gap. Use a
    # generous cutoff and select the six shortest per atom rather than thresholding: a 7.0 A
    # cutoff truncates before six are found on strongly distorted frames, which silently
    # dropped a bridge and produced the wrong bridge count on 8% of frames.
    i, j, D, S = neighbor_list("ijDS", atoms, cutoff=9.0)
    sym = np.array(atoms.get_chemical_symbols())
    sel = (sym[i] == "Pb") & (sym[j] == "Pb")
    i, j, D, S = i[sel], j[sel], D[sel], S[sel]

    r = np.linalg.norm(D, axis=1)

    # Deduplicate to undirected bridges. Each is seen from both ends as (i, j, S) and
    # (j, i, -S), and a pair can legitimately appear twice with different image offsets in a
    # cell short enough that both images are neighbours -- so the key includes S. S is an
    # integer image index, so this is exact; keying on a rounded cartesian vector split
    # bridges whose components landed on a rounding boundary.
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    orient = np.where((i < j)[:, None], S, -S)          # canonical direction, low -> high
    key = np.concatenate([lo[:, None], hi[:, None], orient], axis=1)
    _, uniq = np.unique(key, axis=0, return_index=True)
    i, j, D, r = i[uniq], j[uniq], D[uniq], r[uniq]

    # Select the octahedral axes GLOBALLY rather than per atom. Taking each Pb's six shortest
    # neighbours independently is not a mutual relation under thermal distortion -- A can have
    # B among its six while B does not have A -- so the union overshoots 3*N_Pb and produced 49
    # bridges instead of 48 on 8% of frames. Instead accept bridges shortest-first while both
    # endpoints still have room, which builds an exactly 6-regular graph: every Pb ends with
    # its six octahedral axes and the total is 3*N_Pb by construction.
    order = np.argsort(r)
    degree = np.zeros(len(atoms), dtype=int)
    chosen = []
    for e in order:
        a, b = i[e], j[e]
        if degree[a] < 6 and degree[b] < 6:
            chosen.append(e)
            degree[a] += 1
            degree[b] += 1
    chosen = np.array(chosen, dtype=int)

    pos = atoms.get_positions()
    mid = pos[i[chosen]] + 0.5 * D[chosen]
    return i[chosen], j[chosen], mid


def locate_vacancy(atoms):
    """Locate the single Cl vacancy. Raises ValueError if the cell is not as expected."""
    from scipy.optimize import linear_sum_assignment

    sym = np.array(atoms.get_chemical_symbols())
    pb = np.flatnonzero(sym == "Pb")
    cl = np.flatnonzero(sym == "Cl")
    if len(pb) == 0:
        raise ValueError("no Pb in cell")

    bi, bj, mid = _octahedral_bridges(atoms, pb)
    expected = 3 * len(pb)
    if len(mid) != expected:
        raise ValueError(f"found {len(mid)} octahedral bridges, expected {expected}")
    if len(cl) != expected - 1:
        raise ValueError(f"{len(cl)} Cl for {expected} bridges; expected exactly one vacancy")

    _, dist = get_distances(mid, atoms.get_positions()[cl],
                            cell=atoms.get_cell(), pbc=atoms.pbc)

    # Assign every Cl to a distinct bridge, minimising total squared displacement. Squared
    # rather than linear so one badly tilted Cl cannot buy its way out by displacing several
    # others slightly.
    rows, cols = linear_sum_assignment((dist ** 2).T)   # rows index Cl, cols index bridges
    unmatched = np.setdiff1d(np.arange(len(mid)), cols)
    if len(unmatched) != 1:
        raise ValueError(f"{len(unmatched)} unmatched bridges, expected 1")
    v = int(unmatched[0])

    nearest = dist.min(axis=1)
    others = np.delete(nearest, v)
    return VacancySite(position=mid[v], shell=np.array([bi[v], bj[v]]),
                       cost=float(nearest[v]), runner_up=float(others.max()),
                       n_bridges=len(mid))


def distance_to_vacancy(atoms, site):
    """Minimum-image distance from every atom to the vacancy site."""
    _, d = get_distances(atoms.get_positions(), site.position[None],
                         cell=atoms.get_cell(), pbc=atoms.pbc)
    return d[:, 0]
