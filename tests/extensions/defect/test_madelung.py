"""A1: the Madelung sign ships with its toy table.

`eps_i = eps_local_i - phi_LR_i / eps_inf` is one minus sign, and it is the whole edit. Get it
backwards and the model builds a barrier where the donor well should be, fits the labels
anyway through the local term, and every downstream number is wrong in a way no gate catches
-- the run does not fail, it succeeds at the wrong thing.

So the sign is checked on cases whose answers are known independently of this code:

1. a rock-salt toy where removing an anion must produce a well at the neighbouring cation;
2. pristine CsPbCl3 with nominal charges, where the anion band must come out below the
   cation band and the gap must widen;
3. the G = 0 convention, checked against the energy kernel by an exact identity rather than
   by re-deriving the background term -- that re-derivation is the class of drift that has
   cost this project real time before.

Plus the finite-difference force check, run in EVAL mode. Keying `create_graph` to
`module.training` rather than to the ambient grad mode is a silent force bug: forces are
still returned at evaluation, just without the Madelung contribution. Only an eval-mode check
sees it.
"""

import numpy as np
import pytest
import torch

from mace.modules.defect_madelung import (MadelungOnSite, project_neutral_,
                                          self_potential_of, site_potential)
from mace.modules.latent_ewald import LatentEwald

torch.set_default_dtype(torch.float64)

EPS_INF = 4.0
NOMINAL_CSPBCL3 = {"Cs": 1.0, "Pb": 2.0, "Cl": -1.0}


@pytest.fixture(scope="module")
def ewald():
    return LatentEwald(None).double()


def phi_of(ewald, charges, positions, cell, drop_self=False):
    """`drop_self` is retained as a keyword and refused: the self-image subtraction is gone.

    Kept rather than deleted so that an archived call site fails loudly instead of silently
    computing the other convention.
    """
    if drop_self:
        raise ValueError("the self-image subtraction was removed; phi_LR is the full sum")
    q = torch.as_tensor(charges, dtype=torch.float64)
    r = torch.as_tensor(positions, dtype=torch.float64)
    c = torch.as_tensor(cell, dtype=torch.float64).reshape(1, 3, 3)
    b = torch.zeros(len(q), dtype=torch.long)
    # site_potential keeps the graph whenever grad is enabled -- that is the point of
    # keying create_graph to the ambient mode -- so a value-only helper detaches here.
    return site_potential(ewald, q, r, c, b).detach()


# --------------------------------------------------------------------------- toy structures


def rock_salt(a=5.6, reps=2):
    """A 2x2x2 rock-salt cell: cations at the FCC sites, anions offset by a/2."""
    pos, kind = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                base = np.array([i, j, k], dtype=float) * a
                pos.append(base)
                kind.append("cation")
                pos.append(base + np.array([a / 2, 0.0, 0.0]))
                kind.append("anion")
    return np.array(pos), kind, np.eye(3) * (a * reps)


def cspbcl3(a=5.6, reps=2):
    """Cubic CsPbCl3: Pb at the corner, Cs at the body centre, Cl at the face midpoints."""
    motif = [("Pb", [0.0, 0.0, 0.0]), ("Cs", [0.5, 0.5, 0.5]),
             ("Cl", [0.5, 0.0, 0.0]), ("Cl", [0.0, 0.5, 0.0]), ("Cl", [0.0, 0.0, 0.5])]
    pos, sym = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                for s, f in motif:
                    pos.append((np.array([i, j, k], dtype=float) + np.array(f)) * a)
                    sym.append(s)
    return np.array(pos), sym, np.eye(3) * (a * reps)


# --------------------------------------------------------------------------- A1 test 1


class TestRockSaltDonorWell:
    """Removing an anion must LOWER the electron on-site energy at the nearby cation."""

    def test_the_vacancy_raises_phi_at_the_neighbouring_cation(self, ewald):
        pos, kind, cell = rock_salt()
        charges = np.array([1.0 if k == "cation" else -1.0 for k in kind])
        anion = next(i for i, k in enumerate(kind) if k == "anion")

        keep = [i for i in range(len(pos)) if i != anion]
        phi_vac = phi_of(ewald, charges[keep], pos[keep], cell).numpy()

        # Nearest and farthest cation to the hole left behind, in the SAME cell. Comparing
        # across cells would compare two different G = 0 offsets: removing a Z = -1 ion
        # leaves the cell at net +1, and the neutralising background shifts every phi by one
        # constant. Contrast within a cell is free of that.
        sub = np.array(pos)[keep]
        cat = np.array([i for i, j in enumerate(keep) if kind[j] == "cation"])
        d = np.linalg.norm(
            (sub[cat] - pos[anion] + np.diag(cell) / 2) % np.diag(cell) - np.diag(cell) / 2,
            axis=1)
        near, far = cat[np.argmin(d)], cat[np.argmax(d)]

        assert phi_vac[near] > phi_vac[far], (
            "removing an anion must RAISE the electrostatic potential at the cation next to "
            "it -- the compensating negative charge is gone")

    def test_the_on_site_shift_is_a_well_not_a_barrier(self, ewald):
        pos, kind, cell = rock_salt()
        charges = np.array([1.0 if k == "cation" else -1.0 for k in kind])
        anion = next(i for i, k in enumerate(kind) if k == "anion")
        keep = [i for i in range(len(pos)) if i != anion]
        phi_vac = phi_of(ewald, charges[keep], pos[keep], cell)

        shift = (-phi_vac / EPS_INF).numpy()
        sub = np.array(pos)[keep]
        cat = np.array([i for i, j in enumerate(keep) if kind[j] == "cation"])
        d = np.linalg.norm(
            (sub[cat] - pos[anion] + np.diag(cell) / 2) % np.diag(cell) - np.diag(cell) / 2,
            axis=1)
        near, far = cat[np.argmin(d)], cat[np.argmax(d)]

        assert shift[near] < shift[far], (
            "THE SIGN IS BACKWARDS. eps_i = eps_local_i - phi_i/eps_inf, and the electron "
            "carries charge -1, so a raised potential LOWERS the on-site energy. The anion "
            "vacancy must be a donor well below the cation band, not a barrier above it.")

    def test_pristine_cations_are_equivalent(self, ewald):
        """Control: with no vacancy the cation sites are symmetry-equivalent, so any
        contrast the test above finds must come from the vacancy and not from the lattice."""
        pos, kind, cell = rock_salt()
        charges = np.array([1.0 if k == "cation" else -1.0 for k in kind])
        phi = phi_of(ewald, charges, pos, cell).numpy()
        cat = [i for i, k in enumerate(kind) if k == "cation"]
        assert np.ptp(phi[cat]) < 1e-8


# --------------------------------------------------------------------------- A1 test 2


class TestPristinePerovskiteBandOrder:
    """Nominal Z on pristine CsPbCl3: anion on-sites below cation on-sites, gap widened."""

    def test_anion_sits_below_cation(self, ewald):
        pos, sym, cell = cspbcl3()
        charges = np.array([NOMINAL_CSPBCL3[s] for s in sym])
        shift = (-phi_of(ewald, charges, pos, cell) / EPS_INF).numpy()
        sym = np.array(sym)

        cl = shift[sym == "Cl"].mean()
        pb = shift[sym == "Pb"].mean()
        cs = shift[sym == "Cs"].mean()

        assert cl < pb, (
            f"Cl-derived on-site {cl:.3f} eV must lie BELOW Pb-derived {pb:.3f} eV: the "
            "anion sits in the positive potential of its cation neighbours")
        assert cl < cs
        # The split is the gap-widening statement: without it the Madelung term is doing
        # nothing a per-species constant could not do.
        assert (pb - cl) > 1.0

    def test_the_split_scales_inversely_with_eps_inf(self, ewald):
        """Screening is division, so doubling eps_inf must halve the split exactly."""
        pos, sym, cell = cspbcl3()
        charges = np.array([NOMINAL_CSPBCL3[s] for s in sym])
        phi = phi_of(ewald, charges, pos, cell).numpy()
        sym = np.array(sym)
        split = lambda e: ((-phi / e)[sym == "Pb"].mean() - (-phi / e)[sym == "Cl"].mean())
        assert split(2 * EPS_INF) == pytest.approx(0.5 * split(EPS_INF), rel=1e-12)


# --------------------------------------------------------------------------- A1 test 3


class TestG0Convention:
    """The G = 0 treatment is checked against the energy kernel, not re-derived."""

    def test_potential_reproduces_the_energy(self, ewald):
        """Euler on a quadratic form: E = (1/2) sum_i q_i V_i, with V INCLUDING the self
        term. This pins the whole convention -- smearing, background, G = 0 -- in one
        assertion, without this file knowing what any of them are."""
        pos, sym, cell = cspbcl3()
        charges = np.array([NOMINAL_CSPBCL3[s] for s in sym])
        q = torch.as_tensor(charges)
        r = torch.as_tensor(pos)
        c = torch.as_tensor(cell).reshape(1, 3, 3)
        b = torch.zeros(len(q), dtype=torch.long)

        v_full = site_potential(ewald, q, r, c, b).detach()
        energy = ewald.energy(q, r, c, b).sum()
        assert float(0.5 * (q * v_full).sum()) == pytest.approx(float(energy), rel=1e-10)

    def test_the_background_is_a_uniform_per_cell_offset(self, ewald):
        """The neutralising background contributes the SAME shift to every site, so it can
        never produce contrast -- which is why Stages 1 and 2 are insensitive to it under
        the difference gauge, and why Stage 3 can rely on it being one constant per
        (charge, volume)."""
        pos, sym, cell = cspbcl3()
        charges = np.array([NOMINAL_CSPBCL3[s] for s in sym])
        # A net-charged cell: drop one Cl, as the vacancy does.
        drop = int(np.where(np.array(sym) == "Cl")[0][0])
        keep = [i for i in range(len(pos)) if i != drop]
        q = torch.as_tensor(charges[keep])
        r = torch.as_tensor(pos[keep])
        c = torch.as_tensor(cell).reshape(1, 3, 3)
        b = torch.zeros(len(q), dtype=torch.long)

        # Both legs keep the self term. Subtracting it from one and not the other would put
        # the self energy into the "background" and the test would fail for a reason that has
        # nothing to do with the G = 0 convention.
        with_bg = site_potential(ewald, q, r, c, b)

        def raw_energy(qq):
            e, _, _ = ewald.ewald(q=qq, r=r, cell=c, batch=b)
            return e.sum()

        qq = q.clone().requires_grad_(True)
        no_bg = torch.autograd.grad(raw_energy(qq), qq)[0]

        offset = (with_bg - no_bg).detach().numpy()
        assert np.ptp(offset) < 1e-9, "the background must not vary from site to site"

        net = float(q.sum())
        volume = float(torch.det(c).abs())
        analytic = -ewald.ewald.norm_factor * ewald.sigma ** 2 * net / volume
        assert offset.mean() == pytest.approx(analytic, rel=1e-8)

    def test_the_offset_differs_between_cell_sizes_but_not_between_frames(self, ewald):
        """The property the plan asks for: one constant per (charge, size). Two cell sizes
        at the same net charge, several rattled frames each."""
        rng = np.random.default_rng(0)
        offsets = {}
        for reps in (2, 3):
            pos, sym, cell = cspbcl3(reps=reps)
            charges = np.array([NOMINAL_CSPBCL3[s] for s in sym])
            drop = int(np.where(np.array(sym) == "Cl")[0][0])
            keep = [i for i in range(len(pos)) if i != drop]
            per_frame = []
            for _ in range(4):
                jitter = pos[keep] + rng.normal(scale=0.05, size=(len(keep), 3))
                q = torch.as_tensor(charges[keep])
                c = torch.as_tensor(cell).reshape(1, 3, 3)
                volume = float(torch.det(c).abs())
                per_frame.append(-ewald.ewald.norm_factor * ewald.sigma ** 2
                                 * float(q.sum()) / volume)
                # the potential itself is frame-dependent; the OFFSET must not be
                assert phi_of(ewald, charges[keep], jitter, cell).shape[0] == len(keep)
            offsets[reps] = np.array(per_frame)
            assert np.ptp(offsets[reps]) < 1e-12, "offset drifted between frames"
        assert offsets[2].mean() != pytest.approx(offsets[3].mean())


# --------------------------------------------------------------------------- forces


class TestForcesInEvalMode:
    """The create_graph trap: at evaluation the Madelung force must not silently vanish."""

    @staticmethod
    def _energy(ewald, charges, positions, cell, weights):
        phi = site_potential(
            ewald, torch.as_tensor(charges), positions,
            torch.as_tensor(cell).reshape(1, 3, 3),
            torch.zeros(len(charges), dtype=torch.long))
        return (torch.as_tensor(weights) * (-phi / EPS_INF)).sum()

    def test_autograd_matches_finite_differences(self, ewald):
        pos, sym, cell = cspbcl3()
        charges = np.array([NOMINAL_CSPBCL3[s] for s in sym])
        rng = np.random.default_rng(1)
        weights = rng.uniform(0.0, 1.0, size=len(sym))
        weights /= weights.sum()

        # eval mode: no module is in training(), and this is where the bug hides.
        r = torch.as_tensor(pos).clone().requires_grad_(True)
        with torch.enable_grad():
            e = self._energy(ewald, charges, r, cell, weights)
            grad = torch.autograd.grad(e, r)[0].numpy()

        assert np.abs(grad).max() > 1e-6, (
            "the Madelung force is identically zero at evaluation -- this is the "
            "create_graph=self.training bug, not a converged structure")

        h = 1e-5
        for j in (0, 5, 11):
            for c in range(3):
                plus, minus = pos.copy(), pos.copy()
                plus[j, c] += h
                minus[j, c] -= h
                with torch.no_grad():
                    ep = float(self._energy(ewald, charges, torch.as_tensor(plus), cell,
                                            weights))
                    em = float(self._energy(ewald, charges, torch.as_tensor(minus), cell,
                                            weights))
                assert grad[j, c] == pytest.approx((ep - em) / (2 * h), abs=1e-6)


# --------------------------------------------------------------------------- neutrality


class TestNeutralityProjection:
    def test_projection_makes_the_cell_neutral_in_the_pristine_composition(self):
        composition = torch.tensor([3.0, 1.0, 1.0])          # Cl, Cs, Pb per formula unit
        z = torch.tensor([-0.7, 1.4, 2.2])
        project_neutral_(z, composition)
        assert float(torch.dot(composition, z)) == pytest.approx(0.0, abs=1e-12)

    def test_projection_is_idempotent(self):
        composition = torch.tensor([3.0, 1.0, 1.0])
        z = torch.tensor([-0.7, 1.4, 2.2])
        project_neutral_(z, composition)
        once = z.clone()
        project_neutral_(z, composition)
        assert torch.allclose(once, z, atol=1e-14)

    def test_the_module_projects_at_construction_and_on_demand(self):
        m = MadelungOnSite(3, [3.0, 1.0, 1.0], z_init=[-1.0, 1.0, 2.0])
        assert float(torch.dot(m.composition, m.z)) == pytest.approx(0.0, abs=1e-12)
        with torch.no_grad():
            m.z += 0.3
        assert abs(float(torch.dot(m.composition, m.z))) > 1e-3
        m.project_()
        assert float(torch.dot(m.composition, m.z)) == pytest.approx(0.0, abs=1e-12)

    def test_composition_length_is_checked(self):
        with pytest.raises(ValueError, match="composition has"):
            MadelungOnSite(3, [3.0, 1.0])
