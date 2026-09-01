"""V2's on-site term must be incapable of the band-edge escape, not merely discouraged from it.

T-B imposes Delta_bind >= m. A feature-based eps can satisfy that without binding anything,
by lowering site energies across the whole defect cell -- the trunk's receptive field tells
every atom it is in a defect cell, so the spectrum drops bodily and the inequality is met by a
state no more localised than before. V2 is supposed to remove the representation rather than
tax it, so what needs asserting is a pair of independence properties, not a loss curve:

  * eps does not depend on trunk features at all, so "which cell am I in" is unreachable;
  * eps is local, so an atom outside the on-site cutoff cannot shift eps_i however far the
    trunk's receptive field reaches.

Plus the property that makes the switch safe to ship: a head that never had the term
installed runs the production path bit-identically.
"""

import torch

from mace.modules.defect_onsite import RigidOnSite, install_rigid_onsite
from mace.modules.defect_spectral import SpectralCarrierHead


def head(**kw):
    torch.manual_seed(0)
    defaults = dict(feature_dim=8, counter_dim=4, hidden=16, num_channels=2,
                    num_states=3, smearing=0.02, r_cut=6.0, num_elements=3,
                    use_decay=True)
    defaults.update(kw)
    return SpectralCarrierHead(**defaults).double()


def chain(n=6, spacing=2.5):
    """A 1-D chain: edges between consecutive atoms, so 'far' is unambiguous."""
    feats = torch.randn(n, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.zeros(1, 2, dtype=torch.float64)
    counts[0, 0] = 1.0
    batch = torch.zeros(n, dtype=torch.long)
    src, dst = [], []
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    edge_index = torch.tensor([src, dst])
    pos = torch.arange(n, dtype=torch.float64) * spacing
    length = (pos[edge_index[1]] - pos[edge_index[0]]).abs()
    species = torch.tensor([i % 3 for i in range(n)])
    return feats, counter, counts, batch, edge_index, length, species


def test_absent_by_default_and_production_path_unchanged():
    h = head()
    assert getattr(h, "rigid_onsite", None) is None
    feats, counter, counts, batch, ei, r, sp = chain()
    a = h(feats, counter, counts, batch, 1, ei, r, node_species=sp)
    b = h(feats, counter, counts, batch, 1, ei, r, node_species=sp)
    assert torch.equal(a.site_energy, b.site_energy)
    # and the feature-based path DOES depend on features, which is what V2 changes
    feats2 = feats.clone()
    feats2[0] += 1.0
    c = h(feats2, counter, counts, batch, 1, ei, r, node_species=sp)
    assert not torch.allclose(a.site_energy, c.site_energy)


def test_eps_ignores_trunk_features():
    """The escape route runs through node_feats; V2 must not be able to read them."""
    h = head()
    h.rigid_onsite = RigidOnSite(num_elements=3, num_channels=2, radial_dim=h.radial_dim,
                                 r_cut=6.0).double()
    feats, counter, counts, batch, ei, r, sp = chain()
    base = h(feats, counter, counts, batch, 1, ei, r, node_species=sp).site_energy

    perturbed = feats.clone() + 5.0          # a large, global feature change
    after = h(perturbed, counter, counts, batch, 1, ei, r, node_species=sp).site_energy
    assert torch.equal(base, after), "eps moved when only trunk features changed"


def _far_lengths(ei, r):
    """Lengthen only the bonds among atoms 3-5, all outside atom 0's 6 A cutoff."""
    r_far = r.clone()
    for k in range(r.shape[0]):
        if min(int(ei[0, k]), int(ei[1, k])) >= 3:
            r_far[k] = r[k] + 4.0
    return r_far


def test_eps_is_local_under_gauge_penalty():
    """An atom beyond the on-site cutoff cannot shift eps_i.

    This is the property that blocks 'there is a vacancy 20 A away'. Asserted with
    gauge_penalty=True, which is what the runs actually use: there eps IS eps_raw and the
    locality of the counted form survives to the Hamiltonian.
    """
    h = head(gauge_penalty=True)
    h.rigid_onsite = RigidOnSite(num_elements=3, num_channels=2, radial_dim=h.radial_dim,
                                 r_cut=6.0).double()
    with torch.no_grad():
        for p in h.rigid_onsite.phi.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    feats, counter, counts, batch, ei, r, sp = chain(n=6, spacing=2.5)
    eps0 = h(feats, counter, counts, batch, 1, ei, r, node_species=sp).site_energy[0].clone()
    eps0_far = h(feats, counter, counts, batch, 1, ei, _far_lengths(ei, r),
                 node_species=sp).site_energy[0]
    assert torch.allclose(eps0, eps0_far, atol=1e-12), "eps_0 moved with a distant atom"


def test_mean_subtraction_breaks_locality_without_gauge_penalty():
    """The counted form is local; the head's OTHER gauge is not, and that is worth pinning.

    With gauge_penalty=False the head returns eps_raw - mean_eps, and the per-frame mean
    couples every atom to every other one however far apart. eps_i then moves when a distant
    atom moves, through the mean rather than through phi.

    This is not a defect of V2 and it does not reopen the escape -- a projected-out mean is
    exactly a uniform shift, so the wholesale lowering V2 exists to prevent is impossible in
    that gauge for a different reason. It is recorded because the two gauges block the escape
    by different mechanisms, and a future run that flips gauge_penalty would otherwise inherit
    a locality guarantee this test shows it does not have.
    """
    h = head(gauge_penalty=False)
    h.rigid_onsite = RigidOnSite(num_elements=3, num_channels=2, radial_dim=h.radial_dim,
                                 r_cut=6.0).double()
    with torch.no_grad():
        for p in h.rigid_onsite.phi.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    feats, counter, counts, batch, ei, r, sp = chain(n=6, spacing=2.5)
    eps0 = h(feats, counter, counts, batch, 1, ei, r, node_species=sp).site_energy[0].clone()
    eps0_far = h(feats, counter, counts, batch, 1, ei, _far_lengths(ei, r),
                 node_species=sp).site_energy[0]
    assert not torch.allclose(eps0, eps0_far, atol=1e-12)

    # ...and the raw counted term underneath it IS local, which is what V2 guarantees.
    raw = h.rigid_onsite(sp, ei, r, feats.shape[0])[0]
    raw_far = h.rigid_onsite(sp, ei, _far_lengths(ei, r), feats.shape[0])[0]
    assert torch.allclose(raw, raw_far, atol=1e-12)


def test_neighbour_count_is_visible():
    """The term must still see local coordination -- that is the physics a bound level needs.

    Removing a neighbour of atom 0 is exactly what a vacancy does to its shell, and eps_0
    has to respond or V2 cannot represent a well at all.
    """
    h = head()
    h.rigid_onsite = RigidOnSite(num_elements=3, num_channels=2, radial_dim=h.radial_dim,
                                 r_cut=6.0).double()
    # give phi some non-trivial weights so the response is not zero by initialisation
    with torch.no_grad():
        for p in h.rigid_onsite.phi.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    feats, counter, counts, batch, ei, r, sp = chain()
    full = h(feats, counter, counts, batch, 1, ei, r, node_species=sp).site_energy[0].clone()

    keep = ~(((ei[0] == 0) & (ei[1] == 1)) | ((ei[0] == 1) & (ei[1] == 0)))
    cut = h(feats, counter, counts, batch, 1, ei[:, keep], r[keep],
            node_species=sp).site_energy[0]
    assert not torch.allclose(full, cut), "eps_0 ignored the loss of a neighbour"


def test_install_on_model_switches_the_branch():
    """install_rigid_onsite must change what the head actually computes, not just attach."""
    h = head()
    feats, counter, counts, batch, ei, r, sp = chain()
    before = h(feats, counter, counts, batch, 1, ei, r, node_species=sp).site_energy.clone()

    class Stub:
        spectral = h
        atomic_numbers = torch.tensor([17, 55, 82])

    install_rigid_onsite(Stub, r_cut=6.0)
    assert h.rigid_onsite is not None
    after = h(feats, counter, counts, batch, 1, ei, r, node_species=sp).site_energy
    assert not torch.allclose(before, after), "installing the term changed nothing"
