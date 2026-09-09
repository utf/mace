"""v5 W2 closing item: the active-window partial solve against the full diagonalisation."""
import numpy as np
import pytest
import torch

from mace.modules.dscc import scf, window as win
from mace.modules.dscc.fill import SIGMA_S

torch.set_default_dtype(torch.float64)


def _h(n_atoms=20, seed=0, gap=True):
    rng = np.random.default_rng(seed); n = n_atoms * 4
    A = torch.tensor(rng.standard_normal((n, n))) * 0.3; A = 0.5 * (A + A.T)
    diag = torch.linspace(-4.0, 4.0, n)
    if gap:                                   # a 1.5 eV gap around the 40th level
        diag[n // 2:] += 1.5
    return A + torch.diag(diag)


def test_inertia_below_matches_the_spectrum():
    H = _h(seed=1); eps = torch.linalg.eigvalsh(H)
    for e in (-3.0, 0.0, float(eps[39]) + 1e-6, float(eps[40]) - 1e-6, 5.0):
        assert win.inertia_below(H, e) == int((eps < e).sum())


def test_partial_pass_reproduces_the_full_fillings_and_certifies():
    H0 = _h(seed=2); n_s, n_ref = (39.0, 40.0), (40.0, 40.0)
    with torch.no_grad():
        full = scf.two_fillings(H0, n_s, n_ref)
    eps, U = full.fills[0].eps, full.fills[0].U
    w0 = win.build_window(eps, U, float(full.fills[0].mu), float(full.fills[2].mu))
    assert 0 < w0.S.shape[1] < H0.shape[0]
    # a perturbed H (a site potential of a few tens of meV, as one Newton step moves it)
    V = torch.tensor(np.random.default_rng(3).standard_normal(20)) * 0.03
    H1 = H0 - torch.diag(V.repeat_interleave(4))
    out = win.partial_fillings(H1, w0, n_s, n_ref)
    assert out is not None
    energy, dq, fills, frontier, w1 = out
    with torch.no_grad():
        ref = scf.two_fillings(H1, n_s, n_ref)
    assert float((dq - ref.dq).abs().max()) < 1e-10
    assert abs(float(energy) - float(ref.energy)) < 1e-10
    dP = sum((Ua * da.unsqueeze(0)) @ Ua.T for Ua, da in frontier)
    assert float((dP - ref.dP).abs().max()) < 1e-9
    assert w1.n_below == w0.n_below and w1.S.shape == w0.S.shape


def test_partial_pass_falls_back_when_a_level_enters_the_window():
    H0 = _h(seed=4); n_s, n_ref = (39.0, 40.0), (40.0, 40.0)
    with torch.no_grad():
        full = scf.two_fillings(H0, n_s, n_ref)
    w0 = win.build_window(full.fills[0].eps, full.fills[0].U, float(full.fills[0].mu), float(full.fills[2].mu))
    # push a level from far below into the window's range: a large potential on one site
    V = torch.zeros(20); V[0] = -6.0
    H1 = H0 - torch.diag(V.repeat_interleave(4))
    assert win.partial_fillings(H1, w0, n_s, n_ref) is None
