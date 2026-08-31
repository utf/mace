"""T5: pin the site-energy gauge with a penalty, not a subtraction.

Both approaches kill the same uniform mode. The difference is what they do to cell-size
dependence.

Subtracting the per-frame mean is exact but couples lambda to N: for a well localised on a
fixed number of sites the mean is ~depth*k/N, so lambda picks up an O(1/N) term -- measured
as exactly -3*(1 - 1/N) in the toy below, and ~2.6 meV across the 640-5120 ladder against a
<= 1 meV gate.

Penalising the mean in the loss leaves lambda alone. This checks both halves: that the
subtraction really does introduce the drift, and that the penalty variant really does not.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_spectral import SpectralCarrierHead


def head(gauge_penalty):
    torch.manual_seed(0)
    h = SpectralCarrierHead(feature_dim=8, counter_dim=4, hidden=16, num_channels=2,
                            num_states=1, smearing=0.02, r_cut=6.0,
                            eps_init_scale=0.0, gauge_penalty=gauge_penalty).double()
    with torch.no_grad():
        h.site[-1].bias.zero_()
        h.hop[-1].weight.zero_()
        h.hop[-1].bias.fill_(0.3)
    return h


def chain(n):
    feats = torch.zeros(n, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(n, dtype=torch.long)
    e = []
    for i in range(n - 1):
        e += [(i, i + 1), (i + 1, i)]
    ei = torch.tensor(e, dtype=torch.long).T
    r = torch.full((ei.shape[1],), 3.0, dtype=torch.float64)
    bias = torch.zeros(n, 2, dtype=torch.float64)
    bias[0] = -3.0                      # one deep site: a localised well
    return feats, counter, counts, batch, ei, r, bias


def lowest(h, n):
    feats, counter, counts, batch, ei, r, bias = chain(n)
    out = h(feats, counter, counts, batch, 1, ei, r, site_bias=bias)
    return float(out.eigenvalues[0, 0, 0]), float(out.eps_mean[0, 0])


@pytest.mark.parametrize("n", [6, 10, 18])
def test_subtraction_introduces_a_1_over_n_drift(n):
    """The behaviour T5 exists to remove, pinned down so the fix is checkable."""
    lam, _ = lowest(head(gauge_penalty=False), n)
    # eps on the deep site becomes -3 * (1 - 1/n) once the mean is removed.
    assert abs(lam - (-3.0 * (1 - 1.0 / n))) < 0.02, (
        f"n={n}: lambda {lam:.4f}, expected ~{-3.0 * (1 - 1.0 / n):.4f}")


def test_penalty_variant_has_no_size_drift():
    """With the mean reported rather than subtracted, lambda is N-independent."""
    h = head(gauge_penalty=True)
    lams = [lowest(h, n)[0] for n in (6, 10, 18)]
    spread = max(lams) - min(lams)
    assert spread < 1e-3, f"lambda still drifts with cell size: {lams}"
    assert abs(lams[0] - (-3.0)) < 0.02, (
        f"lambda {lams[0]:.4f} should sit at the bare well depth, not a rescaled one")


def test_mean_is_reported_for_the_loss_to_penalise():
    """The penalty needs the mean; it must come out of the head under both settings."""
    for gp in (False, True):
        _, mean = lowest(head(gauge_penalty=gp), 10)
        assert np.isfinite(mean)
    # With a single -3 well on 10 sites the raw mean is -0.3.
    _, mean_pen = lowest(head(gauge_penalty=True), 10)
    assert abs(mean_pen - (-0.3)) < 1e-9, f"raw mean {mean_pen:.6f}, expected -0.3"


def test_penalty_is_zero_when_the_gauge_is_already_centred():
    """No well, no offset: the penalty must not fight a model that is already fine."""
    h = head(gauge_penalty=True)
    feats, counter, counts, batch, ei, r, _ = chain(10)
    out = h(feats, counter, counts, batch, 1, ei, r)
    assert abs(float(out.eps_mean[0, 0])) < 1e-12
