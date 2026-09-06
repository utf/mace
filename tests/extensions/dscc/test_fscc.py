"""Plan section 2.9 (Arm 4): the F-SCC solver -- its response Jacobian against finite
differences, convergence, charge conservation, and the zero-kernel limit (two fillings)."""
import pytest
import torch

from mace.modules.dscc import fscc, scf
from tests.extensions.dscc.test_scf import N_REF, N_S, _toy

torch.set_default_dtype(torch.float64)
N0 = torch.tensor([1.0, 4.0] + [7.0] * 6)      # toy species table for 8 atoms (sum 46: not the toy's 20;
                                              # the absolute charges only need a fixed n0)


def test_single_fill_response_matches_finite_differences():
    H0, gamma, _, _, _ = _toy(seed=11)
    n = 8
    V0 = 0.05 * torch.randn(n, generator=torch.Generator().manual_seed(12))
    H = H0 - scf.site_potential_matrix(V0)
    M = fscc.single_fill_response(H, N_REF[0], N_REF[1])

    def charges(V):
        Hv = H0 - scf.site_potential_matrix(V)
        e, U = torch.linalg.eigh(Hv)
        P = fscc.fill(Hv, float(N_REF[0]), fscc.SIGMA_S, (e, U)).P + fscc.fill(Hv, float(N_REF[1]), fscc.SIGMA_S, (e, U)).P
        return fscc.absolute_charges(P, N0).detach()
    h = 1e-5
    for j in (0, 4, 7):
        e = torch.zeros(n); e[j] = h
        fd = (charges(V0 + e) - charges(V0 - e)) / (2 * h)
        assert torch.allclose(M[:, j], fd, atol=1e-6, rtol=1e-5), (j, M[:, j], fd)


def test_fscc_solves_and_conserves_charge():
    H0, gamma, _, _, _ = _toy(seed=13)
    opt = scf.ScfOptions(tol_q=1e-11)
    head, state, ref = fscc.fscc_head(H0, 0.3 * gamma, N0, N_S, N_REF, options=opt)
    assert state.converged and ref.converged
    assert float(ref.dq.sum()) == pytest.approx(float(N0.sum() - sum(N_REF)), abs=1e-10)
    assert float(state.dq.sum()) == pytest.approx(float(N0.sum() - sum(N_S)), abs=1e-10)
    assert torch.isfinite(head)
    # Zero kernel: the head is the two-fillings band difference of H0.
    head0, _, _ = fscc.fscc_head(H0, 0.0 * gamma, N0, N_S, N_REF, options=opt)
    two = scf.two_fillings(H0, N_S, N_REF)
    assert float(head0 - two.energy) == pytest.approx(0.0, abs=1e-9)
    assert fscc.excess_trace_norm(state.P, ref.P, 1) >= 0.0
