"""Section 2: the full-sum convention, pinned.

The settled convention is that `phi_LR` is the potential of the infinite periodic ion lattice
with only the TRUE self term (`j = i`, `R = 0`) excluded. The infinite lattice is one set of
charges under any supercell description, so that potential is description-invariant; only its
partition into "in-cell" and "image" pieces depends on the box. The retired convention also
subtracted `A_ii`, which is measured below to be the Makov-Payne self-IMAGE potential --
ion i's own periodic images, real atoms of the crystal that a carrier on site i feels.

These tests exist because that error was invisible at a single cell size and changed the
on-site energies by ~0.3 eV * Z_i between the two sizes the dataset contains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from mace.modules.defect_madelung import (MadelungOnSite, self_potential_of,
                                          site_potential)
from mace.modules.latent_ewald import LatentEwald

torch.set_default_dtype(torch.float64)

A_LATTICE = 5.6
SPECIES = {17: 0, 55: 1, 82: 2}
Z_NOMINAL = [-1.0, 1.0, 2.0]              # exactly neutral against 3:1:1
COULOMB = 14.399645                        # eV.A
ALPHA_M_SC = 2.8372974                     # simple-cubic Madelung constant, jellium


def perovskite(reps=(2, 2, 2)) -> Atoms:
    cell = Atoms("CsPbCl3",
                 scaled_positions=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5),
                                   (0, .5, .5)],
                 cell=[A_LATTICE] * 3, pbc=True)
    return cell.repeat(reps)


@pytest.fixture(scope="module")
def ewald():
    return LatentEwald().double()


def phi(ewald, atoms):
    species = torch.tensor([SPECIES[int(z)] for z in atoms.get_atomic_numbers()])
    q = torch.tensor([Z_NOMINAL[int(s)] for s in species], dtype=torch.float64)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
    batch = torch.zeros(len(atoms), dtype=torch.long)
    with torch.no_grad():
        return site_potential(ewald, q, pos, cell, batch), species


def per_species(v, species):
    """The sorted potentials of each species. Tiling permutes atom order; this does not care."""
    return {s: torch.sort(v[species == s]).values for s in range(3)}


class TestTilingInvariance:
    """Section 2(a). `(A Z)_i` is invariant under any tiling, to 1e-8."""

    @pytest.mark.parametrize("reps", [(2, 1, 1), (1, 3, 1), (2, 2, 1)])
    def test_the_site_potential_is_invariant_under_tiling(self, ewald, reps):
        """Three tilings, including two non-trivial ones. Under the retired subtraction the
        2x case moved Cl from +4.542 to +5.869 eV; under the full sum it does not move."""
        base = perovskite((2, 2, 2))
        tiled = base.repeat(reps)
        mult = int(np.prod(reps))
        vb, sb = phi(ewald, base)
        vt, st = phi(ewald, tiled)
        for s, a in per_species(vb, sb).items():
            b = per_species(vt, st)[s]
            assert b.numel() == mult * a.numel()
            expect = torch.cat([a] * mult).sort().values
            assert torch.allclose(expect, b, atol=1e-8), (
                f"species {s} under {reps}: phi moves by "
                f"{float((expect - b).abs().max()):.2e} eV")

    def test_it_fails_under_the_retired_convention(self, ewald):
        """The counterfactual, so the test above cannot pass vacuously. Subtracting A_ii by
        hand reproduces the eV-scale drift that motivated the change -- which is also the
        record of what the old numbers were computed with."""
        base, tiled = perovskite((2, 2, 2)), perovskite((2, 2, 2)).repeat((2, 1, 1))
        drift = []
        for atoms in (base, tiled):
            v, sp = phi(ewald, atoms)
            cell = torch.tensor(np.array(atoms.get_cell()),
                                dtype=torch.float64).reshape(1, 3, 3)
            q = torch.tensor([Z_NOMINAL[int(s)] for s in sp], dtype=torch.float64)
            old = v - self_potential_of(ewald, cell)[0] * q
            drift.append(float(old[sp == 0].mean()))
        assert abs(drift[1] - drift[0]) > 1.0, (
            "the retired convention should move Cl by ~1.3 eV under this tiling; if it no "
            "longer does, self_potential_of has changed and the calibration below is stale")


class TestKernelCalibration:
    """Section 2(b). One analytic check on the function E_LR itself calls."""

    def test_a_lone_charge_feels_the_makov_payne_potential(self, ewald):
        """`A_ii` against `-alpha_M / L`, the potential at a point charge in jellium.

        This is the calibration of the shared kernel's `G = 0` convention, and it is what
        establishes that `A_ii` is a self-IMAGE term rather than a self-energy: a Gaussian
        self-energy would add +11.49 eV/e at sigma = 1 and swamp the table. Agreement is
        ~1.5%, the residue of the sigma = 1 A smearing against a point charge.
        """
        for reps in (1, 2, 3, 4, 6):
            length = A_LATTICE * reps
            cell = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * length
            measured = float(self_potential_of(ewald, cell)[0])
            analytic = -ALPHA_M_SC * COULOMB / length
            assert measured == pytest.approx(analytic, rel=0.03), (
                f"L = {length:.1f} A: kernel {measured:+.4f}, -alpha_M/L {analytic:+.4f}")

    def test_phi_and_e_lr_go_through_the_same_object(self, ewald):
        """Shared BY IDENTITY, not by agreement. `site_potential` is `dE/dq` of the very
        `ewald.energy` the long-range branch evaluates, so the two cannot drift apart in
        smearing, in the `G = 0` treatment or in the background."""
        atoms = perovskite((2, 2, 2))
        species = torch.tensor([SPECIES[int(z)] for z in atoms.get_atomic_numbers()])
        q = torch.tensor([Z_NOMINAL[int(s)] for s in species], dtype=torch.float64)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        batch = torch.zeros(len(atoms), dtype=torch.long)
        v = site_potential(ewald, q, pos, cell, batch)
        energy = ewald.energy(q, pos, cell.reshape(1, 3, 3), batch).sum()
        # E = q^T A q / 2 exactly, which holds only if v is dE/dq of THIS energy.
        assert float(0.5 * (q * v).sum()) == pytest.approx(float(energy), rel=1e-10)


class TestPristineCrossSize:
    """Section 2(d). Ideal lattices at the two dataset sizes must agree, strictly."""

    def test_on_site_energies_match_across_the_two_cell_sizes(self, ewald):
        """The 80-atom pristine cell and its 160-atom doubling are the same crystal, and the
        on-site shift Edit 1 applies must be identical at corresponding sites. This is the
        statement the retired convention violated by ~0.3 eV * Z_i, at exactly these sizes.
        """
        small = perovskite((2, 2, 4))              # 80 atoms, the dataset's pristine size
        big = small.repeat((2, 1, 1))              # 160
        head = MadelungOnSite(num_elements=3, composition=[3.0, 1.0, 1.0],
                              z_init=Z_NOMINAL).double()
        out = []
        for atoms in (small, big):
            species = torch.tensor([SPECIES[int(z)] for z in atoms.get_atomic_numbers()])
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
            cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
            batch = torch.zeros(len(atoms), dtype=torch.long)
            with torch.no_grad():
                shift = head.on_site_shift(ewald, species, pos, cell, batch, eps_inf=4.0)
            out.append((shift, species))
        for s, a in per_species(out[0][0], out[0][1]).items():
            b = per_species(out[1][0], out[1][1])[s]
            assert torch.allclose(torch.cat([a, a]).sort().values, b, atol=1e-8), (
                f"species {s}: the on-site shift differs by "
                f"{float((torch.cat([a, a]).sort().values - b).abs().max()):.2e} eV "
                "between the two pristine cell sizes")

    def test_the_shift_refuses_a_self_potential(self, ewald):
        """The retired convention cannot be reinstated by a caller passing the old argument."""
        head = MadelungOnSite(num_elements=3, composition=[3.0, 1.0, 1.0],
                              z_init=Z_NOMINAL).double()
        atoms = perovskite((1, 1, 1))
        species = torch.tensor([SPECIES[int(z)] for z in atoms.get_atomic_numbers()])
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        with pytest.raises(ValueError, match="no longer accepts a self-potential"):
            head.on_site_shift(ewald, species, pos, cell,
                               torch.zeros(len(atoms), dtype=torch.long), eps_inf=4.0,
                               self_potential=torch.zeros(1, dtype=torch.float64))


class TestRecordedProhibitions:
    """Section 2(c) and the isolated-mode ban, enforced against the source rather than argued.

    (c) NO CONSTANT-DIFFERENCE TEST ACROSS DEFECT CELL SIZES. Between the 79- and 159-atom
    DEFECT cells the lattice potential differs by the jellium-compensated potential of the
    net charge. The per-(charge, size) reference constants absorb the `G = 0` (uniform) part
    only; the `O(Q / eps L)` non-uniform part is physical and is present in the labels. A
    test asserting a constant difference there would fail for the right reason, and adding
    one would look like a regression when it is the physics. The prohibition is recorded
    here so that anyone tempted to write it finds this first.
    """

    @staticmethod
    def _module_source() -> str:
        root = Path(__file__).resolve().parents[3]
        return (root / "mace" / "modules" / "defect_madelung.py").read_text()

    def test_no_isolated_branch_is_reachable_in_the_phi_path(self):
        """H carries the periodic ion-lattice potential in EVERY mode; only E_LR switches.

        A grep, deliberately: the guarantee is that no such branch exists to be reached, and
        a runtime check could only observe the branches that happen to execute.
        """
        import ast

        tree = ast.parse(self._module_source())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
        # The AST, not a text grep: the module's PROSE says "isolated" several times, on
        # purpose, and a text search would fail on the very docstring that records the rule.
        banned = {n for n in names
                  if "isolated" in n.lower() or "cell_les" in n.lower()}
        assert not banned, (
            f"{sorted(banned)} appear as code in the Madelung module; the phi_LR path must "
            "not have an isolated-mode branch -- image corrections belong to E_LR's switch "
            "and the per-(charge, size) reference constants, and nowhere else")

    def test_the_statement_of_record_is_present(self):
        """The convention is load-bearing for every number downstream, so it travels with the
        code rather than only in a plan document."""
        src = self._module_source()
        assert "STATEMENT OF RECORD" in src
        assert "H is gauge-invariant" in src

    def test_the_self_image_subtraction_is_gone_from_the_production_path(self):
        root = Path(__file__).resolve().parents[3]
        models = (root / "mace" / "modules" / "defect_models.py").read_text()
        assert "self_potential_of" not in models, (
            "defect_models.py still imports the self-image subtraction; the full-sum "
            "convention means the production forward must not call it at all")
