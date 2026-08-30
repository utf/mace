"""The vacancy locator decides every localisation metric, so it needs its own test.

`vacancy_site.locate_vacancy` supplies the shell-Pb identity and the distance-to-vacancy axis
behind the acceptance gates: shell attention mass, Cs mass, the |dF| profile. If it silently
picks the wrong site, a model that localised correctly would be scored as having failed, or
the reverse. It is evaluation-only machinery -- it never touches the model -- but a wrong
answer here is indistinguishable from a wrong answer from the model.

Built on a synthetic cubic perovskite rather than on dataset frames, so the ground truth is
exact: the site removed is known, and the locator must recover it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import get_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "defect-perovskite"))

from vacancy_site import distance_to_vacancy, locate_vacancy  # noqa: E402

A = 5.6  # cubic CsPbCl3 lattice constant, near enough for a geometric test


def perovskite(reps=(2, 2, 2)):
    """Ideal cubic CsPbCl3: Pb at the origin, Cl on the axes, Cs body-centred."""
    cell = [A, A, A]
    atoms = Atoms(
        symbols="PbClClClCs",
        scaled_positions=[(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
                          (0.5, 0.5, 0.5)],
        cell=cell, pbc=True)
    return atoms.repeat(reps)


def make_vacancy(atoms, which=0):
    cl = np.flatnonzero(np.array(atoms.get_chemical_symbols()) == "Cl")
    victim = int(cl[which])
    removed = atoms.get_positions()[victim].copy()
    defect = atoms.copy()
    del defect[victim]
    return defect, removed


def offset(atoms, a, b):
    _, d = get_distances(np.asarray(a)[None], np.asarray(b)[None],
                         cell=atoms.get_cell(), pbc=True)
    return float(d[0, 0])


@pytest.mark.parametrize("reps", [(2, 2, 2), (3, 2, 2), (3, 3, 3)])
def test_finds_the_removed_site(reps):
    pristine = perovskite(reps)
    defect, removed = make_vacancy(pristine)
    site = locate_vacancy(defect)

    assert site.n_bridges == 3 * sum(np.array(defect.get_chemical_symbols()) == "Pb")
    assert offset(defect, site.position, removed) < 0.1


@pytest.mark.parametrize("which", [0, 5, 11])
def test_finds_whichever_site_was_removed(which):
    pristine = perovskite((3, 3, 2))
    defect, removed = make_vacancy(pristine, which)
    site = locate_vacancy(defect)
    assert offset(defect, site.position, removed) < 0.1


def test_shell_is_the_two_neighbouring_pb():
    """The shell must be the two Pb that lost a Cl, at ~a/2 from the empty site."""
    pristine = perovskite((3, 3, 3))
    defect, removed = make_vacancy(pristine)
    site = locate_vacancy(defect)

    assert len(site.shell) == 2
    sym = np.array(defect.get_chemical_symbols())
    assert set(sym[site.shell]) == {"Pb"}

    d = distance_to_vacancy(defect, site)
    assert np.allclose(d[site.shell], A / 2, atol=0.15)

    # Nothing else is that close: the shell is the two nearest atoms, unambiguously.
    others = np.delete(d, site.shell)
    assert others.min() > A / 2


def test_survives_thermal_distortion():
    """Real frames are MD snapshots. Displacements of ~0.2 A must not move the answer.

    This is the case that defeats coordination counting, which on the real dataset finds
    exactly two under-coordinated Pb in fewer than half of all frames.
    """
    pristine = perovskite((3, 3, 3))
    defect, removed = make_vacancy(pristine)
    rng = np.random.default_rng(0)
    for _ in range(5):
        rattled = defect.copy()
        rattled.set_positions(rattled.get_positions()
                              + rng.normal(scale=0.2, size=(len(rattled), 3)))
        site = locate_vacancy(rattled)
        assert offset(rattled, site.position, removed) < 0.6


def test_rejects_a_cell_with_no_vacancy():
    """A stoichiometric cell has no site to find and must raise rather than invent one."""
    with pytest.raises(ValueError, match="expected exactly one vacancy"):
        locate_vacancy(perovskite((2, 2, 2)))


def test_rejects_a_cell_with_two_vacancies():
    pristine = perovskite((3, 3, 3))
    defect, _ = make_vacancy(pristine)
    defect2, _ = make_vacancy(defect)
    with pytest.raises(ValueError, match="expected exactly one vacancy"):
        locate_vacancy(defect2)


def test_margin_is_large_on_an_ideal_cell():
    """On an undistorted cell the empty bridge should stand far clear of every other."""
    defect, _ = make_vacancy(perovskite((3, 3, 3)))
    site = locate_vacancy(defect)
    assert site.margin > 1.0
