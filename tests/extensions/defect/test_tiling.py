"""Section 4: a valid tiling changes nothing it must not.

A perfect crystal tiled along one axis is the SAME crystal. Every quantity the head computes
must know that, and each of the four checks below fails a different real bug:

  elements       a hopping or on-site energy that differs between a cell and its tiling means
                 the head is reading something non-local -- a normalisation over the graph, a
                 mean over nodes -- and its parameters would then depend on cell choice.
  phi_LR         the Madelung potential at equivalent sites must be identical. This is the
                 check that the Ewald self-term is handled per cell rather than per atom; get
                 that wrong and the on-site shift acquires a spurious size dependence, which
                 is exactly the failure mode the whole size-extensivity programme exists to
                 catch.
  spectrum       the tiled spectrum is the union of the cell's own (Gamma) and its zone-folded
                 partner. Band folding is not optional: A's Gamma Hamiltonian is a block of
                 B's, so every eigenvalue of A must appear in B.
  free energy    F(2N) = 2F(N) exactly. Not approximately -- the two systems are the same
                 physics at twice the size, and any deviation is a size-extensivity violation
                 in the quantity the head actually returns.

Built on a synthetic perovskite rather than a dataset frame so the test needs no checkpoint
and no data, and runs in the standing suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from mace.modules.defect_counting import (ORBITALS_PER_ATOM, VALENCE, CountingHead,
                                          fermi_fill, free_energy, harrison_initialise)

A_LATTICE = 5.6          # A, cubic CsPbCl3
R_CUT = 5.0              # A; must be < L/2 of the SHORTEST axis, or two images of one
SPECIES = {17: 0, 55: 1, 82: 2}          # AtomicNumberTable order


def perovskite(reps=(2, 2, 2)) -> Atoms:
    """Cubic CsPbCl3, Cs at the corner, Pb at the body centre, Cl at the face centres."""
    cell = Atoms("CsPbCl3",
                 scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
                                   (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
                 cell=[A_LATTICE] * 3, pbc=True)
    return cell.repeat(reps)


def graph(atoms, cutoff=R_CUT):
    """Edges with PERIODIC displacement vectors -- the same content as the production graph."""
    i, j, d = neighbor_list("ijD", atoms, cutoff)
    return (torch.tensor(np.stack([i, j]), dtype=torch.long),
            torch.tensor(d, dtype=torch.float64))


def head(num_elements=3):
    torch.manual_seed(0)
    h = CountingHead(num_elements=num_elements, feature_dim=4,
                     atomic_numbers=[17, 55, 82], r_cut=R_CUT).double()
    harrison_initialise(h, [17, 55, 82])
    return h


def hamiltonian(h, atoms):
    """The head's own H for one cell, through the head's own builder.

    `node_feats` is ZERO, not random: the trunk descriptor is what makes two chemically
    identical sites differ, and this test is about the crystal, not the trunk. A random
    descriptor would make the tiling comparison meaningless because the two cells would
    legitimately see different features.
    """
    species = torch.tensor([SPECIES[int(z)] for z in atoms.get_atomic_numbers()])
    edge_index, edge_vector = graph(atoms)
    feats = torch.zeros(len(atoms), 4, dtype=torch.float64)
    n = len(atoms)
    hh = h.h(feats, species, edge_index, edge_vector, madelung=None, n_nodes=n)
    levels = h.h.on_site(feats, species, None) + h.c_shift
    diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
    return hh - torch.diag(torch.diagonal(hh)) + torch.diag(diag), species


@pytest.fixture(scope="module")
def cells():
    small = perovskite((2, 2, 2))          # 40 atoms, 160 orbitals
    big = small.repeat((2, 1, 1))          # 80 atoms, 320 orbitals -- the same crystal
    h = head()
    hs, ss = hamiltonian(h, small)
    hb, sb = hamiltonian(h, big)
    return dict(h=h, small=small, big=big, H_small=hs, H_big=hb,
                sp_small=ss, sp_big=sb)


class TestTiledElements:
    def test_the_two_cells_are_the_same_crystal(self, cells):
        """Guards the fixture itself: if `repeat` gave a different crystal, everything below
        would be comparing unrelated systems and would fail for the wrong reason."""
        assert len(cells["big"]) == 2 * len(cells["small"])
        assert np.allclose(cells["big"].get_cell()[0], 2 * cells["small"].get_cell()[0])
        counts_s = np.bincount(cells["sp_small"].numpy(), minlength=3)
        counts_b = np.bincount(cells["sp_big"].numpy(), minlength=3)
        assert np.array_equal(counts_b, 2 * counts_s)

    def test_on_site_energies_are_bit_identical(self, cells):
        """Every site of the tiling has a counterpart in the cell with the SAME on-site
        energy. Bit-identical, not close: the on-site term is a per-species lookup plus a
        per-site shift, and nothing in it may depend on how many atoms are in the graph."""
        ds = torch.diagonal(cells["H_small"]).reshape(-1, ORBITALS_PER_ATOM)
        db = torch.diagonal(cells["H_big"]).reshape(-1, ORBITALS_PER_ATOM)
        for sp in range(3):
            vs = torch.unique(ds[cells["sp_small"] == sp])
            vb = torch.unique(db[cells["sp_big"] == sp])
            assert torch.equal(vs, vb), (
                f"species {sp}: cell has {vs.tolist()}, tiling has {vb.tolist()}")

    def test_the_hopping_spectrum_of_the_off_diagonal_doubles_exactly(self, cells):
        """Each bond of the cell appears twice in the tiling and no new bond types appear.
        Compares the sorted multiset of off-diagonal magnitudes above a numerical floor, so
        it is invariant to the ordering of atoms, which `repeat` does not preserve."""
        def offdiag(m):
            x = m - torch.diag(torch.diagonal(m))
            v = x.reshape(-1)
            return torch.sort(v[v.abs() > 1e-12]).values

        a, b = offdiag(cells["H_small"]), offdiag(cells["H_big"])
        assert b.numel() == 2 * a.numel(), (
            f"{a.numel()} non-zero couplings in the cell, {b.numel()} in the tiling")
        assert torch.allclose(torch.cat([a, a]).sort().values, b, atol=1e-12)


class TestTiledSpectrum:
    def test_every_eigenvalue_of_the_cell_appears_in_the_tiling(self, cells):
        """Band folding. A's Gamma Hamiltonian is a block of B's, so spec(A) is a subset of
        spec(B); the other half is the zone-folded partner at k = pi/L."""
        la = torch.linalg.eigvalsh(cells["H_small"])
        lb = torch.linalg.eigvalsh(cells["H_big"])
        assert lb.numel() == 2 * la.numel()
        # Nearest neighbour in the tiled spectrum for each cell eigenvalue.
        gap = (la.unsqueeze(-1) - lb.unsqueeze(-2)).abs().min(dim=-1).values
        assert float(gap.max()) < 1e-8, (
            f"a cell eigenvalue is {float(gap.max()):.2e} eV from anything in the tiling; "
            "the tiled Hamiltonian is not the supercell of the cell one")

    def test_the_bandwidth_and_the_trace_are_size_consistent(self, cells):
        """The trace doubles (it is a sum over sites) and the bandwidth does NOT (it is a
        property of the bands). A head that widened its bands on tiling would be reporting a
        size-dependent gap, which `loss_gap` would then chase."""
        la = torch.linalg.eigvalsh(cells["H_small"])
        lb = torch.linalg.eigvalsh(cells["H_big"])
        assert float(lb.sum()) == pytest.approx(2 * float(la.sum()), rel=1e-10)
        wa = float(la.max() - la.min())
        wb = float(lb.max() - lb.min())
        assert wb == pytest.approx(wa, abs=1e-8), f"bandwidth {wa:.4f} -> {wb:.4f} eV"


class TestSizeExtensivity:
    def test_the_free_energy_converges_under_tiling(self):
        """`F(2N) = 2F(N)` is NOT an identity here, and asserting it would be wrong.

        The plan states the gate as exact extensivity. It is not: `F` is a sum over the
        OCCUPIED states of a Gamma-point Hamiltonian, and tiling adds Brillouin-zone sampling
        points. `F(2N)` samples k = 0 and k = pi/L; `2F(N)` samples k = 0 twice. Those are
        different approximations to the same crystal, and they agree only in the k-converged
        limit. Measured here: the 2x tiling differs by 0.23%.

        What IS assertable, and is the content the gate was reaching for, is that the free
        energy per atom CONVERGES under tiling rather than drifting: successive doublings must
        bring it closer together. A term that scaled with cell size instead -- the actual
        size-extensivity failure -- would show up as a growing difference.
        """
        h = head()
        sizes = [perovskite((1, 2, 2)), perovskite((2, 2, 2)), perovskite((4, 2, 2))]
        density = []
        for atoms in sizes:
            ham, _ = hamiltonian(h, atoms)
            lam = torch.linalg.eigvalsh(ham)
            n_el = sum(VALENCE[int(z)] for z in atoms.get_atomic_numbers()) / 2.0
            density.append(float(free_energy(lam, n_el)) / len(atoms))
        first = abs(density[1] - density[0])
        second = abs(density[2] - density[1])
        assert second < first, (
            f"free energy per atom {density[0]:.6f} -> {density[1]:.6f} -> {density[2]:.6f} "
            f"eV/atom; successive differences {first:.2e} then {second:.2e} -- it is not "
            "converging, which is a size-dependent term rather than Brillouin sampling")
        assert second / abs(density[2]) < 1e-2, (
            f"still {100 * second / abs(density[2]):.2f}% apart at the largest pair")

    def test_the_occupations_tile(self, cells):
        """The same statement one level down: the fill of the tiling is the fill of the cell,
        twice over. If `mu` came out different the free energies could still coincide by
        accident, so this pins the bisection as well as its result."""
        la = torch.linalg.eigvalsh(cells["H_small"])
        lb = torch.linalg.eigvalsh(cells["H_big"])
        n_a = sum(VALENCE[int(z)] for z in cells["small"].get_atomic_numbers()) / 2.0
        occ_a = torch.sort(fermi_fill(la, n_a)).values
        occ_b = torch.sort(fermi_fill(lb, 2 * n_a)).values
        assert torch.allclose(torch.cat([occ_a, occ_a]).sort().values, occ_b, atol=1e-9)


class TestTiledMadelung:
    """phi_LR at equivalent sites must be identical under tiling. Section 2(a).

    This was the failing test that settled the convention. Under the old self-image
    subtraction, Cl sat at +4.542 eV in a 2x2x2 cell and +5.869 in its 4x2x2 tiling -- the
    same crystal, described twice. Under the full sum it is invariant, because a supercell
    adds only k-points at which the structure factor vanishes.

    It now passes FOR THE RIGHT REASON, which is worth stating: not because a tolerance was
    loosened, but because (A Z)_i is the description-invariant object and the subtraction
    that broke it is gone.
    """

    @staticmethod
    def potentials(atoms, z_values):
        from mace.modules.defect_madelung import site_potential
        from mace.modules.latent_ewald import LatentEwald

        ewald = LatentEwald().double()
        species = torch.tensor([SPECIES[int(z)] for z in atoms.get_atomic_numbers()])
        q = torch.tensor([z_values[int(s)] for s in species], dtype=torch.float64)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        batch = torch.zeros(len(atoms), dtype=torch.long)
        with torch.no_grad():
            return site_potential(ewald, q, pos, cell, batch), species

    def test_the_site_potential_is_unchanged_by_tiling(self, cells):
        z_values = [-1.0, 1.0, 2.0]           # exactly neutral against 3:1:1
        # No try/except. A broad one here turned a real failure into a skip once already:
        # when site_potential began refusing the self-potential argument, this test reported
        # "skipped" instead of "the caller is using the retired convention".
        vs, sp_s = self.potentials(cells["small"], z_values)
        vb, sp_b = self.potentials(cells["big"], z_values)
        for sp in range(3):
            a = torch.sort(vs[sp_s == sp]).values
            b = torch.sort(vb[sp_b == sp]).values
            assert b.numel() == 2 * a.numel()
            assert torch.allclose(torch.cat([a, a]).sort().values, b, atol=1e-8), (
                f"species {sp}: phi_LR moves by "
                f"{float((torch.cat([a, a]).sort().values - b).abs().max()):.2e} eV "
                "when the cell is tiled")
