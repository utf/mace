"""The R1 clamped-state probe: solving the eigenproblem on a subspace.

R1 asks a representational question -- "if the carrier is FORCED onto these atoms, how well
can this head fit the forces?" -- for the hub (two vacancy-sharing Pb) against the cage (their
ten ligand Cl). Answering it honestly requires clamping by restriction rather than by
projection: the state must stay an eigenvector of H restricted to the mask, so its orbital
composition is optimal there and Hellmann-Feynman forces remain valid.

DIAGNOSTIC ONLY. It uses the vacancy assignment, so production configs must refuse it.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_spectral import SpectralCarrierHead


def head(**kw):
    torch.manual_seed(0)
    defaults = dict(feature_dim=8, counter_dim=4, hidden=16, num_channels=2,
                    num_states=3, smearing=0.02, r_cut=6.0)
    defaults.update(kw)
    return SpectralCarrierHead(**defaults).double()


def chain(n=10):
    feats = torch.randn(n, 8, dtype=torch.float64)
    counter = torch.zeros(1, 4, dtype=torch.float64)
    counts = torch.ones(1, 2, dtype=torch.float64)
    batch = torch.zeros(n, dtype=torch.long)
    e = []
    for i in range(n - 1):
        e += [(i, i + 1), (i + 1, i)]
    ei = torch.tensor(e, dtype=torch.long).T
    r = torch.full((ei.shape[1],), 3.0, dtype=torch.float64)
    return feats, counter, counts, batch, ei, r


def test_clamped_state_lives_only_on_the_mask():
    h = head()
    feats, counter, counts, batch, ei, r = chain()
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[2, 3]] = True

    out = h(feats, counter, counts, batch, 1, ei, r, clamp_mask=mask)
    alpha = out.alpha[:, 0]
    assert float(alpha[mask].sum()) > 1 - 1e-8, "mask does not hold the state"
    assert float(alpha[~mask].abs().max()) < 1e-8, "state leaked outside the mask"


def test_different_masks_give_different_energies():
    """The whole point: the head must be able to prefer one site set over another."""
    h = head()
    feats, counter, counts, batch, ei, r = chain()
    m1 = torch.zeros(10, dtype=torch.bool); m1[[2, 3]] = True
    m2 = torch.zeros(10, dtype=torch.bool); m2[[6, 7]] = True

    e1 = h(feats, counter, counts, batch, 1, ei, r, clamp_mask=m1).delta_sr
    e2 = h(feats, counter, counts, batch, 1, ei, r, clamp_mask=m2).delta_sr
    assert not torch.allclose(e1, e2)


def test_clamped_energy_matches_an_explicit_submatrix_solve():
    """Restriction, not projection: the clamped eigenvalue must equal the eigenvalue of H
    restricted to the mask, computed independently."""
    h = head(num_states=1, num_channels=1)
    feats, counter, counts, batch, ei, r = chain(n=8)
    counts = torch.ones(1, 1, dtype=torch.float64)
    idx = [2, 3, 4]
    mask = torch.zeros(8, dtype=torch.bool); mask[idx] = True

    out = h(feats, counter, counts, batch, 1, ei, r, clamp_mask=mask)

    with torch.no_grad():
        eps_raw = h.site(torch.cat([feats, counter[batch]], dim=-1))[:, 0]
        eps = eps_raw - eps_raw.mean()
        t = h.hopping(feats[ei[0]], feats[ei[1]], r,
                      None, None)[:, 0] if not h.use_decay else None
        H = torch.zeros(8, 8, dtype=torch.float64)
        H[ei[0], ei[1]] = -t
        H = 0.5 * (H + H.T)
        H[range(8), range(8)] = eps
        sub = H[np.ix_(idx, idx)]
        lam_sub = float(torch.linalg.eigvalsh(sub)[0])

    lam_clamped = float(out.eigenvalues[0, 0, 0])
    assert abs(lam_clamped - lam_sub) < 1e-9, (
        f"clamped eigenvalue {lam_clamped:.9f} != submatrix eigenvalue {lam_sub:.9f}")


def test_gradients_flow_under_clamping():
    """Hellmann-Feynman must still work, or R1 cannot compare force fits across masks."""
    h = head()
    feats, counter, counts, batch, ei, r = chain()
    feats = feats.requires_grad_(True)
    mask = torch.zeros(10, dtype=torch.bool); mask[[2, 3]] = True

    out = h(feats, counter, counts, batch, 1, ei, r, clamp_mask=mask)
    out.delta_sr.sum().backward()
    assert torch.isfinite(feats.grad).all()
    assert float(feats.grad.abs().max()) > 0


def test_unclamped_is_unchanged():
    """Passing no mask must reproduce the ordinary head exactly."""
    h = head()
    feats, counter, counts, batch, ei, r = chain()
    a = h(feats, counter, counts, batch, 1, ei, r)
    b = h(feats, counter, counts, batch, 1, ei, r, clamp_mask=None)
    assert torch.equal(a.delta_sr, b.delta_sr)
    assert torch.equal(a.alpha, b.alpha)
