"""Bandwidth anneal: start with a wide band and narrow it, so a level can separate.

The head begins with no bound state by design -- near-flat eps, small hopping -- and has to
pull a level out of the band. Scaling every hopping by s(e) = s0^(1 - e/E_a) starts the band
wide, so early training sees a well-mixed spectrum, and narrows it to s = 1 by E_a, at which
point a site-energy contrast that could not compete with the wide band can split a level off.

The schedule matters only if it actually reaches 1: an anneal that leaves the hoppings
permanently scaled would change the converged model, not just its path.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_spectral import SpectralCarrierHead


def anneal_scale(epoch, s0=4.0, e_a=20):
    """s(e) = s0^(1 - e/E_a) for e <= E_a, then 1."""
    if s0 <= 0:
        return 1.0
    return float(s0 ** (1.0 - min(epoch, e_a) / e_a))


def head(**kw):
    torch.manual_seed(0)
    d = dict(feature_dim=8, counter_dim=4, hidden=16, num_channels=2, num_states=3,
             smearing=0.02, r_cut=10.0, use_decay=True, num_elements=3)
    d.update(kw)
    return SpectralCarrierHead(**d).double()


def chain(n=8):
    feats = torch.randn(n, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(n, dtype=torch.long)
    e = []
    for i in range(n - 1):
        e += [(i, i + 1), (i + 1, i)]
    ei = torch.tensor(e, dtype=torch.long).T
    r = torch.full((ei.shape[1],), 3.0, dtype=torch.float64)
    sp = torch.zeros(n, dtype=torch.long)
    return feats, counter, counts, batch, ei, r, sp


def test_schedule_starts_wide_and_reaches_one():
    assert np.isclose(anneal_scale(0), 4.0)
    assert np.isclose(anneal_scale(10), 2.0)
    assert np.isclose(anneal_scale(20), 1.0)
    # And STAYS at one: a schedule that never returns to 1 changes the converged model.
    assert np.isclose(anneal_scale(50), 1.0)
    assert np.isclose(anneal_scale(140), 1.0)


def test_scale_multiplies_the_hopping():
    h = head()
    feats, counter, counts, batch, ei, r, sp = chain()
    base = h.hopping(feats[ei[0]], feats[ei[1]], r, sp[ei[0]], sp[ei[1]])
    with torch.no_grad():
        h.hop_scale.fill_(3.0)
    scaled = h.hopping(feats[ei[0]], feats[ei[1]], r, sp[ei[0]], sp[ei[1]])
    assert torch.allclose(scaled, 3.0 * base)


def test_wider_band_at_high_scale():
    """The point of the anneal: a larger scale gives a broader spectrum."""
    h = head()
    feats, counter, counts, batch, ei, r, sp = chain()

    def width():
        out = h(feats, counter, counts, batch, 1, ei, r, node_species=sp)
        lam = out.eigenvalues[0, 0]
        return float(lam.max() - lam.min())

    with torch.no_grad():
        h.hop_scale.fill_(1.0)
    narrow = width()
    with torch.no_grad():
        h.hop_scale.fill_(4.0)
    wide = width()
    assert wide > 1.5 * narrow, f"scale 4 gave width {wide:.4f} vs {narrow:.4f} at 1"


def test_scale_one_is_the_unannealed_head():
    """At s = 1 the head must be exactly what it would have been without the anneal."""
    h = head()
    feats, counter, counts, batch, ei, r, sp = chain()
    a = h(feats, counter, counts, batch, 1, ei, r, node_species=sp)
    with torch.no_grad():
        h.hop_scale.fill_(1.0)
    b = h(feats, counter, counts, batch, 1, ei, r, node_species=sp)
    assert torch.equal(a.delta_sr, b.delta_sr)


def test_scale_is_a_buffer_so_it_travels_with_the_model():
    """Not a hidden global: it must be in the state dict and move with .to()."""
    h = head()
    assert "hop_scale" in dict(h.named_buffers())
    assert "hop_scale" in h.state_dict()
