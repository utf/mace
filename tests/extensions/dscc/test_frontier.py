"""v5 W2 items 2 and 5: the frontier-only fillings agree with the eager fills, the identical
spin channel is skipped exactly, the shared chemical-potential solve equals separate solves,
and the hole response with reused potentials equals the one that re-solves them."""
import numpy as np
import pytest
import torch

from mace.modules.dscc import fill as fl
from mace.modules.dscc import scf

torch.set_default_dtype(torch.float64)


def _random_h(n_atoms=9, seed=0):
    rng = np.random.default_rng(seed)
    n = n_atoms * scf.ORBITALS
    A = torch.tensor(rng.standard_normal((n, n)))
    H = 0.5 * (A + A.T) + torch.diag(torch.linspace(-3.0, 3.0, n))
    return H


def test_frontier_fillings_match_the_eager_fills():
    H = _random_h(); n_s, n_ref = (17.0, 18.0), (18.0, 18.0)      # one hole in the up channel
    with torch.enable_grad():
        eager = scf.two_fillings(H.clone().requires_grad_(True), n_s, n_ref)
    with torch.no_grad():
        front = scf.two_fillings(H, n_s, n_ref)
    assert front.fills[0].P is None and front._dP is None             # nothing dense was formed
    assert float((front.dq - eager.dq).abs().max()) < 1e-10
    assert abs(float(front.energy) - float(eager.energy)) < 1e-10
    assert float((front.dP - eager.dP.detach()).abs().max()) < 1e-10  # formed on demand from the active levels
    assert front.fills[1] is front.fills[3]                          # the unaffected channel: one fill, reused
    assert float(front.dq.sum()) == pytest.approx(1.0, abs=1e-9)


def test_eager_channel_skip_is_exact():
    H = _random_h(seed=1); n_s, n_ref = (17.0, 18.0), (18.0, 18.0)
    Hg = H.clone().requires_grad_(True)
    sol = scf.two_fillings(Hg, n_s, n_ref)
    assert sol.fills[1] is sol.fills[3]
    # dF/dH = dP: the band-form energy's H-derivative is the density difference
    (g,) = torch.autograd.grad(sol.energy, Hg)
    assert float((g - sol.dP.detach()).abs().max()) < 1e-10


def test_shared_chemical_potential_equals_separate_solves():
    H = _random_h(seed=2); eps = torch.linalg.eigvalsh(H)
    mus = fl.chemical_potential(eps.unsqueeze(0).expand(3, -1), torch.tensor([17.0, 18.0, 19.0]), fl.SIGMA_S)
    for k, n in enumerate((17.0, 18.0, 19.0)):
        assert float(mus[k]) == pytest.approx(float(fl.chemical_potential(eps, n, fl.SIGMA_S)), abs=1e-14)


def test_hole_response_with_reused_potentials():
    H = _random_h(seed=3); n_s, n_ref = (17.0, 18.0), (18.0, 18.0)
    with torch.no_grad():
        sol = scf.two_fillings(H, n_s, n_ref)
    eps, U = sol.fills[0].eps, sol.fills[0].U
    m_a = scf.hole_response(H, n_s, n_ref, spectrum=(eps, U))
    m_b = scf.hole_response(H, n_s, n_ref, spectrum=(eps, U), mus=tuple(f.mu for f in sol.fills))
    assert float((m_a - m_b).abs().max()) < 1e-12


def test_batched_frontier_matches_per_frame():
    Hs = torch.stack([_random_h(seed=s) for s in (4, 5)])
    n_s = (torch.tensor([17.0, 18.0]), torch.tensor([18.0, 17.0])); n_ref = (torch.tensor([18.0, 18.0]), torch.tensor([18.0, 18.0]))
    with torch.no_grad():
        both = scf.two_fillings(Hs, n_s, n_ref)
        for b in range(2):
            one = scf.two_fillings(Hs[b], (float(n_s[0][b]), float(n_s[1][b])), (float(n_ref[0][b]), float(n_ref[1][b])))
            assert float((both.dq[b] - one.dq).abs().max()) < 1e-12
            assert abs(float(both.energy[b]) - float(one.energy)) < 1e-12
            assert float((both.dP[b] - one.dP).abs().max()) < 1e-12


def test_mixed_precision_batched_solve_reaches_the_float64_fixed_point():
    """v5 W2 item 7: the float32 pre-stage changes the iteration count, not the fixed point."""
    Hs = torch.stack([_random_h(seed=s) for s in (6, 7)])
    rng = np.random.default_rng(8); n_atoms = 9
    G = torch.tensor(rng.standard_normal((2, n_atoms, n_atoms))); gamma = 0.05 * (G @ G.transpose(1, 2)) / n_atoms + 0.3 * torch.eye(n_atoms)
    n_s = (torch.tensor([17.0, 17.0]), torch.tensor([18.0, 18.0])); n_ref = (torch.tensor([18.0, 18.0]), torch.tensor([18.0, 18.0]))
    with torch.no_grad():
        mixed = scf.solve_dscc_batched(Hs, gamma, n_s, n_ref, options=scf.ScfOptions())
        plain = scf.solve_dscc_batched(Hs, gamma, n_s, n_ref, options=scf.ScfOptions(mixed_precision=False))
        ramp_m = scf.continuation_solve_batched(Hs, gamma, n_s, n_ref, options=scf.ScfOptions())
        ramp_p = scf.continuation_solve_batched(Hs, gamma, n_s, n_ref, options=scf.ScfOptions(mixed_precision=False))
    assert all(mixed.converged) and all(plain.converged)
    assert mixed.pre_fills and sum(mixed.pre_fills) > 0
    assert float((mixed.dq - plain.dq).abs().max()) < 1e-10 and float((mixed.energy - plain.energy).abs().max()) < 1e-10
    assert float((ramp_m.dq - ramp_p.dq).abs().max()) < 1e-10 and float((ramp_m.energy - ramp_p.energy).abs().max()) < 1e-10
