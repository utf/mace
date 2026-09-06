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
