"""Plan section 5 gates that need more than one evaluation each: the central-difference
force ladder at steps {1e-2, 1e-3, 1e-4} A with the discrepancy decreasing with the SCF
tolerance, and all six strain components; regime B (primary), Route A and Route B."""
import numpy as np
import pytest
import torch

from mace.modules.dscc.model import MACEDSCC
from mace.modules.dscc.kernels import KernelConfig
from mace.modules.dscc.scf import ScfOptions
from tests.extensions.dscc.test_model import VACP, _base, _batch, _frame, _perovskite

torch.set_default_dtype(torch.float64)


def _model(route_b, tol_q):
    m = MACEDSCC(_base(0), r_cut=6.0, directional=True, hidden=16, coupling=True, route_b=route_b,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0),
                 scf=ScfOptions(tol_q=tol_q, tol_E=max(tol_q, 1e-11)))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
        m.lambda_raw.fill_(0.0)
        m.u_raw.fill_(-1.0)
    pristine = _batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])
    m.set_pristine_centre([pristine])
    if route_b:
        m.initialise_route_b([pristine])
    return m


def _energy(m, atoms):
    return float(m(_batch([atoms]), compute_force=False)["energy"])


@pytest.mark.parametrize("route_b", [False, True])
def test_force_step_ladder_and_scf_tolerance(route_b):
    """|F_analytic - F_FD| at h = 1e-2, 1e-3, 1e-4 A is below 1e-4 eV/A at the tight
    tolerance and does not grow when the tolerance is tightened further."""
    atom, comp = 7, 2
    worst = {}
    for tol_q in (1e-6, 1e-10):
        m = _model(route_b, tol_q)
        analytic = float(m(_batch([VACP]), compute_force=True)["forces"][atom, comp])
        errs = []
        for h in (1e-2, 1e-3, 1e-4):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            fd = -(_energy(m, plus) - _energy(m, minus)) / (2 * h)
            errs.append(abs(analytic - fd))
        worst[tol_q] = errs
    # At the tight tolerance every step is within the gate; the 1e-2 step carries the O(h^2)
    # truncation, the 1e-4 step the SCF noise, and both are below 1e-4 eV/A.
    assert max(worst[1e-10]) < 1e-4, worst
    # Tightening the tolerance does not worsen the small-step discrepancy (SCF noise).
    assert worst[1e-10][2] <= worst[1e-6][2] + 1e-6, worst


def test_all_six_strain_components():
    m = _model(False, 1e-10)
    out = m(_batch([VACP]), compute_force=False, compute_stress=True)
    volume = VACP.get_volume()
    h = 1e-4
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        e = []
        for sign in (1.0, -1.0):
            strained = VACP.copy()
            eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = sign * h
            strained.set_cell(np.array(VACP.get_cell()) @ (np.eye(3) + eps), scale_atoms=True)
            e.append(_energy(m, strained))
        fd = (e[0] - e[1]) / (2 * h) / volume
        expected = float(out["stress"][0, i, j] + (out["stress"][0, j, i] if i != j else 0.0))
        assert expected == pytest.approx(fd, abs=1e-7), (i, j)
