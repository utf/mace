"""`avg_num_neighbors` must count the edges the TRUNK uses, not the ones the carrier head does.

The trunk passes messages only over edges inside `r_max`, and divides every message by
`avg_num_neighbors`. With a carrier head the loader builds its graph at the carrier cutoff
instead -- 10 A against r_max 5.0 in this programme -- so a count taken off that graph includes
edges the trunk never touches and comes back roughly eight times too large.

Two independent repairs, because they cover different runs:

  * a Stage-B run inherits the value from its checkpoint (`test_stage_b_normalisation.py`);
  * a FROM-SCRATCH run has no checkpoint to inherit from, so the count itself is rescaled by
    the measured edge-count ratio. Without that the staging control would differ from the
    staged arm in trunk normalisation, which has nothing to do with staging.

`count_edges_within` is the measurement the second repair rests on.
"""

from __future__ import annotations

import pytest
import torch

from mace.modules.defect_reach import count_edges_within


class _Batch:
    """Minimal stand-in: `defect_reach._edge_lengths` reads positions, edge_index, shifts."""

    def __init__(self, lengths):
        n = len(lengths)
        # One atom at the origin and n partners strung out along x, so each edge's length is
        # exactly the value given.
        self.positions = torch.tensor([[0.0, 0.0, 0.0]]
                                      + [[float(v), 0.0, 0.0] for v in lengths])
        self.edge_index = torch.stack([torch.zeros(n, dtype=torch.long),
                                       torch.arange(1, n + 1)])
        self.shifts = torch.zeros(n, 3)
        self.unit_shifts = torch.zeros(n, 3)


def test_it_counts_both_populations():
    b = _Batch([1.0, 3.0, 4.9, 5.1, 7.0, 9.9])
    total, inside = count_edges_within(b, 5.0)
    assert total == 6
    assert inside == 3


def test_the_boundary_is_inclusive():
    """An edge exactly at r_max is one the trunk uses, so it counts."""
    total, inside = count_edges_within(_Batch([5.0]), 5.0)
    assert (total, inside) == (1, 1)


def test_an_empty_graph_is_zero_and_not_a_division_by_zero():
    class _Empty:
        positions = torch.zeros(1, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        shifts = torch.zeros(0, 3)
        unit_shifts = torch.zeros(0, 3)

    assert count_edges_within(_Empty(), 5.0) == (0, 0)


def test_the_ratio_is_the_correction_and_it_is_large_at_this_programmes_cutoffs():
    """Not a volume argument. The graph is built per frame with periodic images, so
    `(10/5)^3 = 8` is an estimate and the measured ratio is the number that matters -- but it
    should land in that neighbourhood, or the premise is wrong."""
    # A shell-like distribution: many more edges between 5 and 10 A than inside 5 A, as a
    # roughly uniform density gives.
    inside_lengths = [1.0, 2.0, 3.0, 4.0, 4.5]
    outside_lengths = [5.5 + 0.1 * i for i in range(35)]
    total, inside = count_edges_within(_Batch(inside_lengths + outside_lengths), 5.0)
    ratio = inside / total
    assert 0.05 < ratio < 0.2
    assert 1.0 / ratio == pytest.approx(8.0, rel=0.1)
