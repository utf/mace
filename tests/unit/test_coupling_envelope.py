"""V3 test 4: the flanking separations are coupled, asserted without a defect label.

The plan asks for "the flanking pair coupled with f_env >= 0.3 in >= 99% of charged frames".
Taken literally that needs the vacancy assignment inside the training loop, which hard rule 1
forbids. The check here is the structural equivalent: the envelope at the far end of the
5.2-6.8 A window clears the floor (a pure function of r_couple), and essentially every frame
carries an edge in that window. Neither statement says where the defect is.

The duplicated envelope is the risk this file exists to cover -- a reach test evaluating its
own formula rather than the head's is the exact pattern that has produced four passing guards
over broken paths here, so the two are asserted equal.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_reach import assert_coupling_envelope, envelope_at
from mace.modules.defect_spectral_v3 import LocalSpectralHead


def test_envelope_matches_the_head():
    """The standalone envelope must be the head's, not a lookalike."""
    h = LocalSpectralHead(feature_dim=4, counter_dim=2, num_elements=2,
                          r_max=5.0, r_couple=10.0, hidden=8, num_channels=1,
                          num_states=2, smearing=0.02).double()
    for r in (2.0, 5.3, 6.8, 9.5, 10.0, 12.0):
        got = float(h._envelope(torch.tensor([r], dtype=torch.float64))[0])
        assert got == pytest.approx(envelope_at(r, 10.0), abs=1e-12), f"mismatch at {r} A"


def test_ten_angstrom_reach_clears_the_floor():
    """r_couple = 10 A must leave the far end of the flanking window well above 0.3."""
    assert envelope_at(6.8, 10.0) > 0.3
    assert envelope_at(5.3, 10.0) > 0.3
    # and a graph that only just reaches the window would not
    assert envelope_at(6.8, 7.0) < 0.3


class _Batch(dict):
    pass


def _batch(lengths, graph_of_edge, n_nodes=8):
    lengths = torch.as_tensor(lengths, dtype=torch.float64)
    n_e = lengths.numel()
    src = torch.arange(n_e) % n_nodes
    dst = (src + 1) % n_nodes
    pos = torch.zeros(n_nodes, 3, dtype=torch.float64)
    # Place each edge's endpoints so the measured length is the requested one.
    b = _Batch(edge_index=torch.stack([src, dst]), positions=pos,
               shifts=torch.zeros(n_e, 3, dtype=torch.float64),
               batch=torch.as_tensor(graph_of_edge, dtype=torch.long))
    # _edge_lengths derives lengths from positions; override by giving explicit offsets.
    b["positions"] = torch.zeros(n_nodes, 3, dtype=torch.float64)
    b["shifts"] = torch.zeros(n_e, 3, dtype=torch.float64)
    b["shifts"][:, 0] = lengths
    return b


def test_fires_when_no_frame_has_a_window_edge():
    """A graph too short for the flanking separations must abort, not warn."""
    b = _batch([3.0] * 8, [0] * 8)
    with pytest.raises(RuntimeError, match="COUPLING FAILURE"):
        assert_coupling_envelope(b, r_couple=10.0)


def test_passes_when_every_frame_has_one():
    b = _batch([6.0] * 8, [0, 0, 0, 0, 1, 1, 1, 1])
    rep = assert_coupling_envelope(b, r_couple=10.0)
    assert rep["frames_with_a_window_edge"] == pytest.approx(1.0)
    assert rep["f_env_at_window_far_end"] > 0.3


def test_fires_when_the_envelope_is_too_short():
    b = _batch([6.0] * 8, [0] * 8)
    with pytest.raises(RuntimeError, match="COUPLING FAILURE"):
        assert_coupling_envelope(b, r_couple=7.0)
