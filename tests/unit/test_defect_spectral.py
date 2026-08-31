"""Pre-flight tests for the spectral carrier head (plan section 11).

The head replaces a softmax with an eigendecomposition, which brings failure modes the old
head could not have: a non-symmetric H makes `eigh`'s gradients quietly wrong, degenerate
eigenvalues make them singular, padding can leak states into the low spectrum, and a
zero-initialised hopping never trains because it sits at a stationary point.

The two-site case is the anchor: a 2x2 symmetric Hamiltonian has a closed form, so eigenvalues,
eigenvectors, the smeared energy and the resulting alpha can all be checked against algebra
rather than against the implementation's own output.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_spectral import SpectralCarrierHead


def head(**kw):
    torch.manual_seed(0)
    defaults = dict(feature_dim=8, counter_dim=4, hidden=16, num_channels=2,
                    num_states=2, smearing=0.02, r_cut=5.0)
    defaults.update(kw)
    return SpectralCarrierHead(**defaults).double()


def two_site_inputs(r=3.0, n_channels=2):
    feats = torch.randn(2, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, n_channels, dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    length = torch.full((2,), r, dtype=torch.float64)
    return feats, counter, counts, batch, edge_index, length


def test_two_site_matches_closed_form():
    """For H = [[a, -t], [-t, b]] the eigenvalues are (a+b)/2 -/+ sqrt(((a-b)/2)^2 + t^2)."""
    h = head()
    feats, counter, counts, batch, ei, r = two_site_inputs()
    out = h(feats, counter, counts, batch, 1, ei, r)

    eps = out.site_energy[:, 0]
    t = h.hopping(feats[ei[0]], feats[ei[1]], r)[0, 0]

    a, b = float(eps[0]), float(eps[1])
    tt = float(t)
    mid, half = (a + b) / 2.0, np.sqrt(((a - b) / 2.0) ** 2 + tt ** 2)
    expected = np.array([mid - half, mid + half])

    assert np.allclose(out.eigenvalues[0, 0].detach().numpy(), expected, atol=1e-10)

    # The smeared energy must follow from those same eigenpairs. dE_SR sums over ALL channels
    # -- each has its own site energies and so its own spectrum -- so the comparison is
    # against sum_c n_c sum_k w_kc lambda_kc, not against one channel scaled.
    w = torch.softmax(-out.eigenvalues[0] / h.smearing, dim=-1)          # [C, m]
    expected_energy = float((w * out.eigenvalues[0]).sum(-1).sum())
    assert np.allclose(expected_energy, float(out.delta_sr[0]), atol=1e-10)
    assert np.allclose(float(out.alpha[:, 0].sum()), 1.0, atol=1e-10)


def test_hopping_is_exactly_symmetric():
    """t_ij == t_ji must hold identically, not approximately: it is what makes H symmetric,
    and `eigh` assumes symmetry rather than checking it."""
    h = head()
    fi, fj = torch.randn(7, 8, dtype=torch.float64), torch.randn(7, 8, dtype=torch.float64)
    r = torch.rand(7, dtype=torch.float64) * 4 + 0.5
    assert torch.equal(h.hopping(fi, fj, r), h.hopping(fj, fi, r))


def test_hopping_and_envelope_vanish_smoothly_at_the_cutoff():
    h = head()
    fi, fj = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
    at_cut = h.hopping(fi, fj, torch.tensor([5.0], dtype=torch.float64))
    assert float(at_cut.abs().max()) < 1e-12

    # And the derivative goes to zero too, or forces would jump as a neighbour crosses r_cut.
    r = torch.tensor([4.999], dtype=torch.float64, requires_grad=True)
    h.hopping(fi, fj, r).sum().backward()
    assert float(r.grad.abs()) < 1e-4


def test_zero_counts_give_exactly_zero_correction():
    """dE_SR = sum_c n_c (...) must vanish structurally at n = 0, to numerical precision."""
    h = head()
    feats, counter, _, batch, ei, r = two_site_inputs()
    counts = torch.zeros(1, 2, dtype=torch.float64)
    out = h(feats, counter, counts, batch, 1, ei, r)
    assert float(out.delta_sr.abs().max()) == 0.0


def test_padding_does_not_leak_into_the_low_spectrum():
    """Two cells of different sizes in one batch. The smaller is padded, and those padded
    slots must not appear among the lowest states or carry any alpha."""
    h = head(num_states=3)
    feats = torch.randn(7, 8, dtype=torch.float64)
    counter = torch.zeros(2, 4, dtype=torch.float64)
    counts = torch.ones(2, 2, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1])          # 5 atoms then 2
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 5, 6],
                               [1, 0, 2, 1, 3, 2, 4, 3, 6, 5]])
    r = torch.full((edge_index.shape[1],), 3.0, dtype=torch.float64)
    out = h(feats, counter, counts, batch, 2, edge_index, r)

    assert torch.isfinite(out.eigenvalues).all()

    # The 2-atom cell genuinely has only 2 states, so asking for 3 must include padding.
    # What matters is that padding carries no weight and never reaches a reported quantity.
    assert float(out.weights[1, :, 2].max()) < 1e-12, "padded state carries weight"
    assert float(out.gap.max()) < 1e3, "gap was read off a padded state"
    for g in (0, 1):
        mass = float(out.alpha[batch == g, 0].sum())
        assert np.isclose(mass, 1.0, atol=1e-8), f"cell {g} alpha sums to {mass}, not 1"


def test_eigh_converges_on_a_realistically_padded_batch():
    """A batch mixing 79- and 159-atom cells, the real dataset's two sizes.

    Padding every slot with the SAME energy made that block exactly degenerate, and `eigh`
    refused an 80-fold repeated eigenvalue with "ill-conditioned or has too many repeated
    eigenvalues". The earlier padding test used 5- and 2-atom cells, where the degenerate
    block is three rows and LAPACK copes -- too small to expose it.
    """
    h = head(num_states=6)
    sizes = [79, 159, 79]
    n = sum(sizes)
    feats = torch.randn(n, 8, dtype=torch.float64)
    counter = torch.zeros(len(sizes), 4, dtype=torch.float64)
    counts = torch.ones(len(sizes), 2, dtype=torch.float64)
    batch = torch.cat([torch.full((s,), g, dtype=torch.long)
                       for g, s in enumerate(sizes)])

    # A connected chain within each cell, so every cell has a real spectrum.
    edges = []
    off = 0
    for s in sizes:
        for i in range(s - 1):
            edges.append((off + i, off + i + 1))
            edges.append((off + i + 1, off + i))
        off += s
    ei = torch.tensor(edges, dtype=torch.long).T
    r = torch.full((ei.shape[1],), 3.0, dtype=torch.float64)

    out = h(feats, counter, counts, batch, len(sizes), ei, r)
    assert torch.isfinite(out.eigenvalues).all()
    assert torch.isfinite(out.delta_sr).all()
    for g, s in enumerate(sizes):
        assert np.isclose(float(out.alpha[batch == g, 0].sum()), 1.0, atol=1e-8)
        # No padded state may reach the occupied window of a cell that has real states.
        assert float(out.eigenvalues[g].max()) < 0.5 * 1.0e3


def test_alpha_is_normalised_per_cell_and_channel():
    h = head()
    feats, counter, counts, batch, ei, r = two_site_inputs()
    out = h(feats, counter, counts, batch, 1, ei, r)
    for c in range(2):
        assert np.isclose(float(out.alpha[:, c].sum()), 1.0, atol=1e-10)


def test_gradients_are_finite_through_a_degenerate_spectrum():
    """Identical sites make the two eigenvalues degenerate, where eigendecomposition
    derivatives are singular. The smearing has to keep this finite rather than produce NaN."""
    h = head()
    f = torch.randn(1, 8, dtype=torch.float64)
    feats = torch.cat([f, f]).requires_grad_(True)        # exactly identical sites
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    ei = torch.tensor([[0, 1], [1, 0]])
    r = torch.full((2,), 3.0, dtype=torch.float64)

    out = h(feats, counter, counts, batch, 1, ei, r)
    out.delta_sr.sum().backward()
    assert torch.isfinite(feats.grad).all(), "non-finite gradient at a degenerate spectrum"


def test_hopping_is_nonzero_at_init_and_receives_gradient():
    """The zero-init audit the plan asks for.

    A zero hopping disconnects the graph, makes every eigenvector a delta function, and sits
    at a stationary point where the gradient with respect to t vanishes -- so it would never
    train, silently. Initialisation must be small but genuinely nonzero, and must receive
    gradient on a representative charged batch.
    """
    h = head()
    feats, counter, counts, batch, ei, r = two_site_inputs()
    t0 = h.hopping(feats[ei[0]], feats[ei[1]], r)
    assert float(t0.abs().max()) > 1e-3, "hopping initialised too close to zero to train"

    out = h(feats, counter, counts, batch, 1, ei, r)
    out.delta_sr.sum().backward()
    grads = [p.grad.abs().max() for p in h.hop.parameters() if p.grad is not None]
    assert grads and max(float(g) for g in grads) > 0, "hopping receives no gradient"


def test_site_energy_is_near_flat_at_init():
    """No bound state at initialisation: the data must pull a level out of the band, rather
    than the head starting with one and merely relocating it."""
    h = head()
    feats = torch.randn(40, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    batch = torch.zeros(40, dtype=torch.long)
    eps = h.site(torch.cat([feats, counter[batch]], dim=-1))
    assert float(eps.std()) < 0.15, f"site energies already structured at init: {eps.std()}"


def test_permutation_invariance():
    """Relabelling atoms must permute alpha the same way and leave the energy unchanged."""
    h = head()
    feats = torch.randn(4, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(4, dtype=torch.long)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    r = torch.full((6,), 3.0, dtype=torch.float64)
    out = h(feats, counter, counts, batch, 1, ei, r)

    perm = torch.tensor([2, 0, 3, 1])
    inv = torch.argsort(perm)
    out_p = h(feats[perm], counter, counts, batch, 1, inv[ei], r)

    assert np.isclose(float(out.delta_sr[0]), float(out_p.delta_sr[0]), atol=1e-10)
    assert torch.allclose(out.alpha[perm], out_p.alpha, atol=1e-10)


@pytest.mark.parametrize("n_extra", [0, 4, 12])
def test_bound_state_energy_is_size_independent(n_extra):
    """The property softmax could not deliver: a bound level's energy does not drift as the
    cell grows. A deep site is embedded in a chain of identical bulk sites, and lambda_0 must
    not move as bulk sites are appended."""
    h = head(num_states=1, eps_init_scale=0.0)
    with torch.no_grad():
        h.site[-1].bias.zero_()
        h.hop[-1].weight.zero_()
        h.hop[-1].bias.fill_(0.3)

    n = 6 + n_extra
    feats = torch.zeros(n, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(n, dtype=torch.long)
    ei = torch.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                       [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
    r = torch.full((ei.shape[1],), 3.0, dtype=torch.float64)

    bias = torch.zeros(n, 2, dtype=torch.float64)
    bias[0] = -3.0                                    # one deep site: a bound state
    out = h(feats, counter, counts, batch, 1, ei, r, site_bias=bias)

    assert float(out.alpha[0, 0]) > 0.8, "bound state not localised on the deep site"
    test_bound_state_energy_is_size_independent.energies = getattr(
        test_bound_state_energy_is_size_independent, "energies", {})
    test_bound_state_energy_is_size_independent.energies[n_extra] = float(
        out.eigenvalues[0, 0, 0])

    seen = test_bound_state_energy_is_size_independent.energies
    if len(seen) > 1:
        spread = max(seen.values()) - min(seen.values())
        assert spread < 1e-3, f"bound-state energy drifts with cell size: {seen}"
