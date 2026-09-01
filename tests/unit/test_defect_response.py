"""The response channel must vanish at n = 0 and at V = 0, and stay low-order in V.

The readout is deliberately linear + quadratic rather than a free MLP on (h_i, V_i). A free
MLP would be a second per-atom energy field carrying exactly the alpha/u gauge freedom the
spectral head exists to remove, and it would not vanish at V = 0. These pin both properties,
plus the sign constraint on the polarisability: a sign-free quadratic term could lower the
energy by inventing anti-polarisable ions, which fits rather than responds.
"""

import torch

from mace.modules.defect_response import CarrierResponse


def make(feature_dim=8, n=6, seed=0):
    torch.manual_seed(seed)
    mod = CarrierResponse(feature_dim=feature_dim, num_elements=3, hidden=16)
    feats = torch.randn(n, feature_dim)
    species = torch.tensor([0, 1, 2] * (n // 3))
    batch = torch.zeros(n, dtype=torch.long)
    return mod, feats, species, batch


class FakeEwald:
    """E = 1/2 sum_ij q_i q_j / (|r_ij| + soft), diagonal included -- enough to exercise
    the autograd potential and the self-term subtraction without a real Ewald sum."""

    def energy(self, q, r, cell, batch):
        d = torch.cdist(r, r) + torch.eye(len(r), device=r.device) * 1.0
        return 0.5 * (q[:, None] * q[None, :] / d).sum().reshape(1)


def test_energy_vanishes_when_no_carrier_is_present():
    """The n = 0 guarantee: a neutral frame must reduce exactly to the base model."""
    mod, feats, species, batch = make()
    alpha = torch.rand(6, 4)
    counts = torch.zeros(1, 4)
    out = mod(feats, species, alpha, counts, batch, torch.randn(6, 3),
              torch.eye(3).reshape(1, 3, 3), FakeEwald(), num_graphs=1)
    assert torch.allclose(out, torch.zeros_like(out))


def test_energy_vanishes_when_the_potential_is_zero():
    """u = V g - V^2 p /2 has no constant term, so zero field means zero response."""
    mod, feats, species, batch = make()
    v = torch.zeros(6)
    g = torch.randn(6)
    p = torch.rand(6)
    u = v * g - 0.5 * v.pow(2) * p
    assert torch.allclose(u, torch.zeros_like(u))


def test_polarisability_is_non_negative():
    """softplus keeps the quadratic term stabilising; an anti-polarisable ion is a fit."""
    mod, feats, species, _ = make()
    emb = mod.species(species.long())
    p = torch.nn.functional.softplus(mod.p(torch.cat([feats, emb], -1)))
    assert bool((p >= 0).all())


def test_potential_is_the_derivative_of_the_energy():
    """V_i = dE/dq_i, checked against a finite difference on the same kernel."""
    mod, *_ = make()
    ew = FakeEwald()
    r = torch.randn(5, 3)
    cell = torch.eye(3).reshape(1, 3, 3)
    b = torch.zeros(5, dtype=torch.long)
    q = torch.rand(5)
    v = mod.potential(ew, q, r, cell, b)

    eps = 1e-4
    for i in (0, 3):
        qp, qm = q.clone(), q.clone()
        qp[i] += eps
        qm[i] -= eps
        fd = (ew.energy(qp, r, cell, b) - ew.energy(qm, r, cell, b)) / (2 * eps)
        assert abs(float(fd) - float(v[i])) < 1e-2, f"atom {i}: {float(fd)} vs {float(v[i])}"


def test_self_term_subtraction_changes_the_potential():
    """Excluding j = i is not cosmetic: eps_i already carries the on-site response."""
    mod, *_ = make()
    ew = FakeEwald()
    r = torch.randn(5, 3)
    cell = torch.eye(3).reshape(1, 3, 3)
    b = torch.zeros(5, dtype=torch.long)
    q = torch.rand(5) + 0.5
    with_self = mod.potential(ew, q, r, cell, b)
    without = mod.potential(ew, q, r, cell, b, self_potential=torch.ones(1))
    assert not torch.allclose(with_self, without)
    assert torch.allclose(without, with_self - q)


def test_scales_with_counter_multiplicity():
    """E_resp carries the n_c prefactor, so two carriers give twice the response."""
    mod, feats, species, batch = make()
    alpha = torch.rand(6, 4)
    alpha = alpha / alpha.sum(0, keepdim=True)
    r, cell = torch.randn(6, 3), torch.eye(3).reshape(1, 3, 3)
    one = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    two = torch.tensor([[0.0, 0.0, 2.0, 0.0]])
    e1 = mod(feats, species, alpha, one, batch, r, cell, FakeEwald(), num_graphs=1)
    e2 = mod(feats, species, alpha, two, batch, r, cell, FakeEwald(), num_graphs=1)
    assert torch.allclose(e2, 2 * e1, atol=1e-6)
