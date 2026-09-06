"""Plan section 4 gate: `H0` invariant under translation, rotation, permutation and
rewrapping with covariant derivatives; `Q_i = 0` at centrosymmetric sites; the scalar-only
control is the directional block switched off."""
import numpy as np
import pytest
import torch
from ase import Atoms

from mace.modules.dscc import graph as gr
from mace.modules.dscc.hamiltonian import H0, quadrupole_descriptor

torch.set_default_dtype(torch.float64)
ZS = [17, 55, 82]


def _perovskite(reps=(2, 2, 2), a=5.6, rattle=0.0, seed=0):
    atoms = Atoms("CsPbCl3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
                                              [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                  cell=np.eye(3) * a, pbc=True).repeat(reps)
    if rattle:
        atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def _inputs(atoms, r_cut, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(atoms)
    species = torch.tensor([ZS.index(int(z)) for z in atoms.get_atomic_numbers()])
    scalars = 0.3 * torch.randn(n, 8, generator=g)
    vectors = 0.3 * torch.randn(n, 6, 3, generator=g)
    positions = torch.tensor(atoms.get_positions())
    cell = torch.tensor(np.array(atoms.get_cell()))
    ei, S = gr.neighbour_list(atoms.get_positions(), np.array(atoms.get_cell()), r_cut)
    return species, scalars, vectors, positions, cell, torch.tensor(ei), torch.tensor(S)


def _model(directional=True, seed=1):
    torch.manual_seed(seed)
    m = H0(ZS, feature_dim=8, n_vectors=6, r_cut=6.0, directional=directional, hidden=16)
    with torch.no_grad():
        m.vector_mix.normal_(0.0, 0.5)
        m.alpha.fill_(0.7)
        m.beta.fill_(-0.4)
    m.set_centre(torch.zeros(4, 8), torch.tensor([0, 1, 2, 2]))
    return m


def _h(m, species, scalars, vectors, positions, cell, ei, S):
    return m(scalars, vectors, species, ei, gr.edge_vectors(positions, cell, ei, S))


class TestDescriptor:
    def test_quadrupole_vanishes_at_cubic_sites_and_is_uniaxial_on_the_bridging_anion(self):
        """A traceless rank-2 tensor is even under inversion, so `Q_i` vanishes at sites of
        CUBIC symmetry (the ideal Pb and Cs sites), not at every centrosymmetric one: the
        Cl site (D4h, linear Pb-Cl-Pb) carries the uniaxial quadrupole that splits its
        p_sigma from its p_pi -- the crystal-field sign Arm 1 (iii) tests."""
        atoms = _perovskite()
        species, _, _, positions, cell, ei, S = _inputs(atoms, 4.5)
        Q = quadrupole_descriptor(ei, gr.edge_vectors(positions, cell, ei, S), len(atoms), 4.5)
        cubic = species != ZS.index(17)
        assert float(Q[cubic].abs().max()) < 1e-12
        assert float(torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-12
        # Cl at (a/2, a/2, 0): Pb pair along z at 2.8 A; the in-plane Cs and Cl shells at
        # 3.96 A cancel each other exactly on the cubic lattice.
        w = (1 - (2.8 / 4.5) ** 6) ** 2
        cl0 = int(torch.nonzero(species == ZS.index(17)).reshape(-1)[0])
        expected = 2 * w * torch.diag(torch.tensor([-1 / 3, -1 / 3, 2 / 3]))
        assert torch.allclose(Q[cl0], expected, atol=1e-12)
        atoms.positions[2] += [0.3, 0.0, 0.0]                   # a Cl off its site
        species, _, _, positions, cell, ei, S = _inputs(atoms, 4.5)
        Q = quadrupole_descriptor(ei, gr.edge_vectors(positions, cell, ei, S), len(atoms), 4.5)
        assert float(Q.abs().max()) > 1e-3
        assert torch.allclose(Q, Q.transpose(-1, -2)) and float(torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-12


class TestInvariances:
    def test_symmetric_translation_rewrapping_permutation(self):
        atoms = _perovskite(rattle=0.05)
        m = _model()
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        H = _h(m, species, scalars, vectors, positions, cell, ei, S)
        assert torch.allclose(H, H.T, atol=1e-14) and H.dtype == torch.float64
        # Translation and rewrapping: same graph, same vectors.
        shifted = atoms.copy(); shifted.positions += [1.1, -0.7, 2.3]
        wrapped = atoms.copy(); wrapped.positions[5] += np.array(atoms.get_cell())[1]
        for other in (shifted, wrapped):
            sp2, _, _, pos2, cell2, ei2, S2 = _inputs(other, m.r_cut)
            H2 = _h(m, sp2, scalars, vectors, pos2, cell2, ei2, S2)
            assert float((torch.linalg.eigvalsh(H) - torch.linalg.eigvalsh(H2)).abs().max()) < 1e-10
        # Permutation of the atoms permutes the orbital blocks.
        perm = torch.randperm(len(atoms), generator=torch.Generator().manual_seed(3))
        permuted = atoms[perm.numpy()]
        sp3, _, _, pos3, cell3, ei3, S3 = _inputs(permuted, m.r_cut)
        H3 = _h(m, sp3, scalars[perm], vectors[perm], pos3, cell3, ei3, S3)
        orb = (perm.unsqueeze(-1) * 4 + torch.arange(4)).reshape(-1)
        assert float((H3 - H[orb][:, orb]).abs().max()) < 1e-12

    def test_rotation_is_covariant(self):
        """Rotating the frame and the polar-vector features rotates the p orbitals:
        `H' = D H D^T`, `D = blockdiag(1, R)` per atom."""
        atoms = _perovskite(rattle=0.05)
        m = _model()
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        H = _h(m, species, scalars, vectors, positions, cell, ei, S)
        R = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(4)))[0]
        if torch.det(R) < 0:
            R = -R
        rotated = atoms.copy()
        rotated.set_cell(np.array(atoms.get_cell()) @ R.T.numpy(), scale_atoms=False)
        rotated.positions = atoms.get_positions() @ R.T.numpy()
        sp2, _, _, pos2, cell2, ei2, S2 = _inputs(rotated, m.r_cut)
        H2 = _h(m, sp2, scalars, vectors @ R.T, pos2, cell2, ei2, S2)
        D = torch.block_diag(*[torch.block_diag(torch.ones(1, 1), R) for _ in range(len(atoms))])
        assert float((H2 - D @ H @ D.T).abs().max()) < 1e-10

    def test_position_derivatives_of_the_spectrum(self):
        atoms = _perovskite(rattle=0.05)
        m = _model()
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, m.r_cut)
        pos = positions.clone().requires_grad_(True)
        H = m(scalars, vectors, species, ei, gr.edge_vectors(pos, cell, ei, S))
        eps = torch.linalg.eigvalsh(H)
        (grad,) = torch.autograd.grad(eps[:60].sum(), pos)
        h = 1e-5
        for atom, comp in ((0, 0), (3, 2), (11, 1)):
            plus, minus = positions.clone(), positions.clone()
            plus[atom, comp] += h; minus[atom, comp] -= h
            fd = (float(torch.linalg.eigvalsh(m(scalars, vectors, species, ei, gr.edge_vectors(plus, cell, ei, S)))[:60].sum())
                  - float(torch.linalg.eigvalsh(m(scalars, vectors, species, ei, gr.edge_vectors(minus, cell, ei, S)))[:60].sum())) / (2 * h)
            assert float(grad[atom, comp]) == pytest.approx(fd, abs=1e-7, rel=1e-6)


class TestControl:
    def test_scalar_only_control_has_no_directional_block(self):
        atoms = _perovskite(rattle=0.05)
        full, scalar_only = _model(directional=True), _model(directional=False)
        species, scalars, vectors, positions, cell, ei, S = _inputs(atoms, full.r_cut)
        Hf = _h(full, species, scalars, vectors, positions, cell, ei, S)
        Hs = _h(scalar_only, species, scalars, None, positions, cell, ei, S)
        block = full.directional_block(vectors, species, ei, gr.edge_vectors(positions, cell, ei, S))
        assert float((Hf - Hs - block).abs().max()) < 1e-12
        assert float(block.abs().max()) > 1e-3
        # Harrison initialisation: anion p below the cation p levels at start.
        eps0 = full.sk.eps0.detach()
        assert eps0[ZS.index(17), 1] < eps0[ZS.index(82), 1] < eps0[ZS.index(55), 1]
        assert not bool(H0(ZS, feature_dim=8, n_vectors=6, hidden=16).centre_set)


class TestBatched:
    def test_batched_h0_equals_the_per_graph_matrices(self):
        atoms = [_perovskite(rattle=0.05, seed=s) for s in (1, 2, 3)]
        m = _model()
        singles, feats = [], []
        for a in atoms:
            species, scalars, vectors, positions, cell, ei, S = _inputs(a, m.r_cut, seed=int(a.get_positions()[0, 0] * 1000) % 97)
            singles.append(_h(m, species, scalars, vectors, positions, cell, ei, S))
            feats.append((species, scalars, vectors, positions, cell, ei, S))
        n = len(atoms[0])
        species = torch.cat([f[0] for f in feats]); scalars = torch.cat([f[1] for f in feats])
        vectors = torch.cat([f[2] for f in feats])
        batch = torch.arange(3).repeat_interleave(n)
        ei = torch.cat([f[5] + g * n for g, f in enumerate(feats)], dim=1)
        ev = torch.cat([gr.edge_vectors(f[3], f[4], f[5], f[6]) for f in feats])
        H = m.batched(scalars, vectors, species, ei, ev, batch, 3, n)
        assert H.shape == (3, 4 * n, 4 * n)
        for g in range(3):
            assert float((H[g] - singles[g]).abs().max()) < 1e-12
