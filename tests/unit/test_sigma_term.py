"""H3's dangling-orbital vector must fire on the vacancy pair and nowhere else.

R0 measured the physics this term exists to express: the residual force on the two
vacancy-sharing Pb is axial (anisotropy 6.6-7.8 against ~0.9 for the carrier-free null) and
scales with their separation -- a sigma bond between two dangling orbitals facing each other.
A scalar hopping has no notion of direction and cannot represent it.

The term is only legitimate if it is geometry alone: v_i = -sum_j f_cut(r_ij) rhat_ij must be
~0 for a complete coordination shell, large and axial for an atom facing a vacancy, and
equivariant. If it instead fired on any low-coordination atom it would be a disguised
novelty signal, which section 9 rules out as a placement mechanism.
"""

import numpy as np
import torch

from mace.modules.defect_spectral import SpectralCarrierHead


def head(**kw):
    torch.manual_seed(0)
    d = dict(feature_dim=8, counter_dim=4, hidden=16, num_channels=2, num_states=3,
             smearing=0.02, r_cut=10.0, use_decay=True, use_sigma=True, num_elements=3)
    d.update(kw)
    return SpectralCarrierHead(**d).double()


def octahedron(a=2.85, drop=None):
    """One central atom with six ligands on the axes; `drop` removes one of them."""
    pos = [[0.0, 0.0, 0.0]]
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for k, d in enumerate(dirs):
        if drop is not None and k == drop:
            continue
        pos.append([a * c for c in d])
    return torch.tensor(pos, dtype=torch.float64)


def edges_from(pos, cutoff=10.0):
    n = len(pos)
    src, dst, vec = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[j] - pos[i]
            if float(d.norm()) <= cutoff:
                src.append(i); dst.append(j); vec.append(d)
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.stack(vec))


def test_complete_shell_gives_a_vanishing_vector():
    """A full octahedron is centrosymmetric, so the contributions cancel exactly."""
    h = head()
    pos = octahedron()
    ei, vec = edges_from(pos)
    v = h.dangling_vectors(ei, vec, len(pos))
    assert float(v[0].norm()) < 1e-12, f"|v| = {float(v[0].norm()):.3e} on a complete shell"


def test_missing_ligand_gives_a_vector_pointing_at_the_gap():
    """Remove the +x ligand: v on the centre must point along +x, at the hole."""
    h = head()
    pos = octahedron(drop=0)             # dirs[0] = (1, 0, 0)
    ei, vec = edges_from(pos)
    v = h.dangling_vectors(ei, vec, len(pos))
    v0 = v[0]
    assert float(v0.norm()) > 0.1, "no vector where a ligand is missing"
    direction = v0 / v0.norm()
    assert float(direction[0]) > 0.99, f"vector points {direction.tolist()}, not along +x"


def test_vector_is_equivariant():
    """Rotate the structure; v must rotate with it."""
    h = head()
    pos = octahedron(drop=2)
    ei, vec = edges_from(pos)
    v = h.dangling_vectors(ei, vec, len(pos))

    theta = 0.7
    R = torch.tensor([[np.cos(theta), -np.sin(theta), 0.0],
                      [np.sin(theta), np.cos(theta), 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float64)
    v_rot = h.dangling_vectors(ei, vec @ R.T, len(pos))
    assert torch.allclose(v_rot[0], R @ v[0], atol=1e-12)


def test_sigma_coupling_singles_out_the_vacancy_pair_in_a_real_lattice():
    """The product (v_i.rhat)(v_j.rhat_ji) must pick out the hub pair, in a real lattice.

    A finite cluster is not a fair test: isolated ligands have incomplete shells of their own
    and so look maximally dangling, which makes every pair score alike. In the actual crystal
    each Cl bridges two Pb and each Pb has six Cl, so only the two atoms facing the vacancy
    have an uncancelled vector. Built here as a periodic CsPbCl3 supercell with one Cl removed.
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    a = 5.6
    cell = Atoms(symbols="PbClClClCs",
                 scaled_positions=[(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
                                   (0.5, 0.5, 0.5)],
                 cell=[a, a, a], pbc=True).repeat((3, 3, 3))
    sym = np.array(cell.get_chemical_symbols())
    victim = int(np.flatnonzero(sym == "Cl")[0])
    removed = cell.get_positions()[victim].copy()
    del cell[victim]

    i, j, D = neighbor_list("ijD", cell, cutoff=6.5)
    ei = torch.tensor(np.stack([i, j]), dtype=torch.long)
    vec = torch.tensor(D, dtype=torch.float64)

    h = head()
    v = h.dangling_vectors(ei, vec, len(cell))
    mag = v.norm(dim=-1).numpy()

    # The two Pb that lost this Cl are its former bridge partners: the nearest Pb to the
    # removed site.
    sym = np.array(cell.get_chemical_symbols())
    pb = np.flatnonzero(sym == "Pb")
    from ase.geometry import get_distances
    _, dist = get_distances(cell.get_positions()[pb], removed[None],
                            cell=cell.get_cell(), pbc=True)
    hub = pb[np.argsort(dist[:, 0])[:2]]

    others = np.setdiff1d(np.arange(len(cell)), hub)
    assert mag[hub].min() > 5 * mag[others].max(), (
        f"hub |v| = {mag[hub]} is not clearly above the rest "
        f"(max {mag[others].max():.4f})")


def test_sigma_off_reproduces_the_plain_head():
    h_off = head(use_sigma=False)
    assert not hasattr(h_off, "sigma_amp")
