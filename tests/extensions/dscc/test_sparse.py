"""Plan section 8: the sparse frontier path agrees with the dense reference."""
import numpy as np
import pytest
import torch

from mace.modules.dscc import sparse as sp
from mace.modules.dscc.scf import two_fillings
from mace.modules.dscc.species import S_REF, State, neutral_count
from tests.extensions.dscc.test_hamiltonian import _inputs, _model, _perovskite, ZS
from mace.modules.dscc import graph as gr

torch.set_default_dtype(torch.float64)


def test_csr_hamiltonian_equals_dense_and_frontier_fillings_agree():
    atoms = _perovskite(rattle=0.05)
    cl = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 17]
    del atoms[cl[0]]                                             # a vacancy: a gap state
    m = _model()
    with torch.no_grad():
        m.h0 if False else None
    species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
    ev = gr.edge_vectors(positions, cell, ei, S)
    H_dense = m(scalars, vectors, species, ei, ev).detach()
    H_csr = sp.csr_hamiltonian(m, scalars, vectors, species, ei, ev)
    assert np.abs(H_csr.toarray() - H_dense.numpy()).max() < 1e-12
    numbers = [ZS[int(s)] for s in species.tolist()]
    n_ref = neutral_count(numbers)
    n_s, n_r = State(1, -1, 0).counts(n_ref), S_REF.counts(n_ref)
    eps = torch.linalg.eigvalsh(H_dense)
    sigma = 0.5 * float(eps[n_r[0] - 1] + eps[n_r[0]])
    inertia, lu = sp.inertia_below(H_csr, sigma)
    assert inertia.below == n_r[0] and inertia.residual < 1e-8
    dense = two_fillings(H_dense, n_s, n_r)
    J, (psi, w), dq, window = sp.two_fillings_sparse(H_csr, n_s, n_r, sigma=sigma, k_buffer=12)
    assert window.tail_bound_charge < 1e-8
    assert float(J - dense.energy) == pytest.approx(0.0, abs=1e-8)
    assert float((dq - dense.dq.detach()).abs().max()) < 1e-8
    assert float(dq.sum()) == pytest.approx(1.0, abs=1e-8)


def test_sparse_scf_and_frontier_forces_match_the_dense_model():
    """The frontier-window solve and the rank-k force contraction reproduce the dense
    model's coupled energy and forces (Route A, regime B) on the toy vacancy."""
    from mace.modules.dscc.kernels import KernelConfig, gamma_matrix, kernel_components
    from mace.modules.dscc.scf import ScfOptions
    from tests.extensions.dscc.test_model import VACP, _batch, _coupled
    m = _coupled(regime="B", route_b=False)
    m.scf_options = ScfOptions(tol_q=1e-10, tol_E=1e-11, continuation_steps=0)
    batch = _batch([VACP])
    dense = m(dict(batch), compute_force=True)
    # Rebuild the pieces for the sparse path.
    from mace.modules.models import ScaleShiftMACE
    data = dict(_batch([VACP])); positions = data["positions"].requires_grad_(True)
    cell = data["cell"].view(3, 3)
    out = ScaleShiftMACE.forward(m.base, m._trunk_data(data), compute_force=False)
    scalars, vectors = m.features(out["node_feats"]); species = data["node_attrs"].argmax(-1)
    ei = data["edge_index"]; ev = positions[ei[1]] - positions[ei[0]] + data["unit_shifts"] @ cell
    H_csr = sp.csr_hamiltonian(m.h0, scalars.detach(), vectors.detach(), species, ei, ev.detach())
    k_sr, k_lr = kernel_components(positions, cell, m.kernel)
    gamma = gamma_matrix(k_sr, k_lr, m.lambda_dir(), m.u_eff()[species], m.kernel.eps_inf)
    numbers = [ZS[int(s)] for s in species.tolist()]; n_ref = neutral_count(numbers)
    n_s, n_r = State(1, -1, 0).counts(n_ref), S_REF.counts(n_ref)
    eps = torch.linalg.eigvalsh(torch.tensor(H_csr.toarray()))
    sigma = 0.5 * float(eps[n_r[0] - 1] + eps[n_r[0]])
    sol = sp.solve_dscc_sparse(H_csr, gamma.detach(), n_s, n_r, sigma, k_buffer=12, tol_q=1e-10)
    assert sol["converged"]
    assert float(sol["energy"] - dense["head_energy"][0]) == pytest.approx(0.0, abs=1e-7)
    assert float((sol["dq"] - dense["dq"]).abs().max()) < 1e-7
    forces = sp.frontier_forces(m.h0, scalars, vectors, species, ei, ev, positions, sol["psi"], sol["w"], gamma, sol["dq"])
    # dense forces = base forces + head forces; compare the head part.
    base = ScaleShiftMACE.forward(m.base, m._trunk_data(dict(_batch([VACP]))), compute_force=True)["forces"]
    head_dense = dense["forces"] - base
    assert float((forces - head_dense).abs().max()) < 1e-6, float((forces - head_dense).abs().max())


@pytest.mark.parametrize("coupling", [False, True])
def test_model_forward_sparse_matches_dense(coupling):
    from mace.modules.dscc.scf import ScfOptions
    from tests.extensions.dscc.test_model import VACP, _batch, _coupled, model as _dense_fixture  # noqa: F401
    m = _coupled(regime="B", route_b=False)
    m.coupling = coupling
    m.scf_options = ScfOptions(tol_q=1e-10, tol_E=1e-11, continuation_steps=0)
    dense = m(_batch([VACP]), compute_force=True)
    out = sp.model_forward_sparse(m, _batch([VACP]), k_buffer=12, tol_q=1e-10)
    assert out["diagnostics"]["converged"]
    assert float((out["energy"] - dense["energy"][0]).abs()) < 1e-7
    assert float((out["forces"] - dense["forces"]).abs().max()) < 1e-6
    assert float((out["dq"] - dense["dq"]).abs().max()) < 1e-7
