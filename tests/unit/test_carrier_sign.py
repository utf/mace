"""The hole's level is a maximum of the occupied manifold, not a minimum.

The head computes `lambda_min` of a bonding-signed H. That is an ADDED ELECTRON's level,
eps - t(d): it FALLS as a pair closes, so the resulting force is attractive. A hole removed
from a bonding state has energy -eps + t(d), which RISES as the pair closes, giving a
repulsive force -- and no minimum eigenvalue can represent it, because a minimum lies at or
below the smallest diagonal (Rayleigh) while the hole's level lies ABOVE its on-site energy
by t.

So the sign has to come from the counter rather than from the solver. These tests pin the
consequence that matters physically -- the direction of the hub force -- rather than the
sign of a coefficient, because a coefficient can be right while the force it produces is not.

Two independent sign conventions exist and must not be confused: `channel_sign` on the
ENERGY, (+1, +1, -1, -1), and `carrier_signs` on the CHARGE in StructuredLatentCharges,
(-1, -1, +1, +1). A hole has positive charge AND a negated energy contribution; applying
either to the other quantity would be the double application the plan warns about.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_blocks import StructuredLatentCharges  # noqa: F401
from mace.modules.defect_spectral_v3 import LocalSpectralHead

E_MAJ, H_MAJ = 0, 2


def head(single_manifold=True, **kw):
    torch.manual_seed(0)
    d = dict(feature_dim=8, counter_dim=4, num_elements=3, r_max=5.0, r_couple=10.0,
             hidden=16, num_channels=4, num_states=2, smearing=0.02,
             single_manifold=single_manifold)
    d.update(kw)
    return LocalSpectralHead(**d).double()


def two_site(h, d, counts):
    """delta_sr for a two-atom cell at separation d, with the given counter."""
    feats = torch.zeros(2, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    c = torch.tensor([counts], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    ei = torch.tensor([[0, 1], [1, 0]])
    r = torch.full((2,), float(d), dtype=torch.float64)
    vec = torch.zeros(2, 3, dtype=torch.float64)
    vec[0, 0], vec[1, 0] = float(d), -float(d)
    sp = torch.tensor([2, 2])
    out = h(feats, counter, c, batch, 1, ei, r, node_species=sp, edge_vector=vec)
    return float(out.delta_sr[0]), float(out.eigenvalues[0, H_MAJ].min())


def test_default_channel_sign_is_ones():
    """Every pre-existing head must be untouched: 24 saved V3 checkpoints depend on it."""
    h = head(single_manifold=False)
    assert torch.equal(h.channel_sign, torch.ones(4, dtype=h.channel_sign.dtype))


def test_single_manifold_negates_the_hole_channels():
    h = head()
    assert h.channel_sign.tolist() == [1.0, 1.0, -1.0, -1.0]


def test_hole_energy_rises_and_electron_energy_falls_as_the_pair_closes():
    """The physical statement, by finite differences on the separation.

    Closing the pair raises t. The electron's level eps - t falls; the hole's -eps + t rises.
    Asserted on the head's own delta_sr, so it covers the counter -> sign path and not just
    the eigenvalue.
    """
    h = head()
    near, far = 4.6, 5.4
    e_near, lam_near = two_site(h, near, [1, 0, 0, 0])     # one electron
    e_far, lam_far = two_site(h, far, [1, 0, 0, 0])
    hole_near, _ = two_site(h, near, [0, 0, 1, 0])         # one hole
    hole_far, _ = two_site(h, far, [0, 0, 1, 0])

    assert lam_near < lam_far, "lambda_min must fall as the pair closes (t grows)"
    assert e_near < e_far, "the electron's energy must FALL as the pair closes"
    assert hole_near > hole_far, "the hole's energy must RISE as the pair closes"
    # and the two must move oppositely, which is the whole point
    assert (e_near - e_far) * (hole_near - hole_far) < 0


def test_hub_force_directions_are_opposite():
    """Force, not energy: -dE/dd > 0 means the pair is pushed apart.

    The hole must be pushed apart (repulsive, matching the DFT residual's outward sign) and
    the electron pulled together.
    """
    h = head()
    d0, eps = 5.0, 1e-4
    for counts, want_outward in (([0, 0, 1, 0], True), ([1, 0, 0, 0], False)):
        e_minus, _ = two_site(h, d0 - eps, counts)
        e_plus, _ = two_site(h, d0 + eps, counts)
        force = -(e_plus - e_minus) / (2 * eps)
        if want_outward:
            assert force > 0, f"hole force {force:+.4g} should push the pair apart"
        else:
            assert force < 0, f"electron force {force:+.4g} should pull the pair together"


def test_energy_sign_and_charge_sign_are_separate_conventions():
    """The two must not be conflated -- applying either twice would cancel the fix.

    A hole carries POSITIVE charge (so it polarises the lattice like a positive defect) and a
    NEGATED energy contribution (because its level is a maximum). Both are true at once.
    """
    h = head()
    assert h.channel_sign.tolist() == [1.0, 1.0, -1.0, -1.0]          # energy
    charge = StructuredLatentCharges.__init__
    assert charge is not None
    # the charge convention lives on its own buffer with the opposite pattern
    import mace.modules.defect_blocks as blocks
    src = blocks.__file__
    with open(src) as fh:
        text = fh.read()
    assert "carrier_signs" in text and "[-1.0, -1.0, 1.0, 1.0]" in text, (
        "the charge-sign buffer changed shape or value; the energy/charge conventions are "
        "no longer independently pinned")


def test_H_is_counter_independent_in_single_manifold():
    """One manifold means the counter cannot enter H at all.

    That is what lets pristine frames be scored natively for Delta_bind -- the hole-counter
    override, and with it the canonicalisation and m_s_ref_doubled bookkeeping, drops out of
    that path entirely.
    """
    h = head()
    grabbed_a, grabbed_b = {}, {}
    # Fixed inputs, generated ONCE: the only thing allowed to differ between the two calls is
    # the counter, including the counter embedding it feeds.
    torch.manual_seed(7)
    feats = torch.randn(3, 8, dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    r = torch.full((4,), 3.0, dtype=torch.float64)
    vec = torch.zeros(4, 3, dtype=torch.float64)
    vec[:, 0] = torch.tensor([3.0, -3.0, 3.0, -3.0], dtype=torch.float64)
    sp = torch.tensor([0, 1, 2])

    def run(counts, counter_emb, grabbed):
        c = torch.tensor([counts], dtype=torch.float64)
        h(feats, counter_emb, c, batch, 1, ei, r, node_species=sp,
          edge_vector=vec, internals=grabbed)

    # Different counters AND different counter embeddings; H must not notice either.
    run([0, 0, 0, 0], torch.zeros(1, 4, dtype=torch.float64), grabbed_a)
    run([0, 0, 1, 0], torch.randn(1, 4, dtype=torch.float64), grabbed_b)
    assert torch.equal(grabbed_a["H"], grabbed_b["H"]), (
        "H moved with the counter; the manifold is not counter-free")


def test_four_channels_still_present_at_the_interface():
    """Broadcasting one manifold must not change the shape anything downstream sees.

    channel_of, the D1 scripts and the four-channel training log all index a C = 4 axis.
    """
    h = head()
    _, _ = two_site(h, 5.0, [0, 0, 1, 0])
    feats = torch.zeros(2, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    c = torch.tensor([[0, 0, 1, 0]], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    ei = torch.tensor([[0, 1], [1, 0]])
    r = torch.full((2,), 5.0, dtype=torch.float64)
    vec = torch.zeros(2, 3, dtype=torch.float64)
    vec[0, 0], vec[1, 0] = 5.0, -5.0
    out = h(feats, counter, c, batch, 1, ei, r, node_species=torch.tensor([2, 2]),
            edge_vector=vec)
    assert out.eigenvalues.shape[1] == 4
    assert out.alpha.shape[1] == 4
    assert out.site_energy.shape[1] == 4
