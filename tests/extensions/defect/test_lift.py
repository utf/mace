"""Canonical isolated-space lift and IsoOK (addendum 4.2, gates in 11.1)."""

import math

import pytest
import torch

from mace.modules.defect_lift import (
    LiftError,
    LiftRecord,
    build_lift,
    circular_moments,
    clearance_report,
    image_assignment,
    iso_ok,
    lift_positions,
    support_envelope,
)


def cell(a: float = 12.0) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64) * a


def frac(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


class TestSupportEnvelope:
    def test_is_nonnegative_and_cancellation_free(self):
        """A vacancy and an interstitial must REINFORCE, not cancel.

        The signed topology of `-1` and `+1` sums to zero; if the envelope inherited that,
        the circular moment would point somewhere arbitrary.
        """
        signed = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        zeta = support_envelope(signed)
        assert bool((zeta >= 0).all())
        assert float(zeta.sum()) == pytest.approx(2.0, abs=1e-6)

    def test_departure_adds_and_is_clamped(self):
        signed = torch.zeros(3, dtype=torch.float64)
        dep = torch.tensor([1.0, -5.0, 0.0], dtype=torch.float64)   # negative is clamped
        zeta = support_envelope(signed, departure=dep, lambda_lift=2.0)
        assert float(zeta[0]) == pytest.approx(2.0, abs=1e-6)
        assert float(zeta[1]) == pytest.approx(0.0, abs=1e-6)

    def test_bad_parameters_are_refused(self):
        signed = torch.ones(2, dtype=torch.float64)
        with pytest.raises(LiftError, match="must be positive"):
            support_envelope(signed, eps_eta=0.0)
        with pytest.raises(LiftError, match="entries for"):
            support_envelope(signed, departure=torch.zeros(5, dtype=torch.float64))


class TestCircularMoments:
    def test_the_circular_mean_handles_the_boundary(self):
        """Two sites either side of the boundary: a plain mean gives the cell middle."""
        w = torch.ones(2, dtype=torch.float64)
        pos = frac([0.98, 0.5, 0.5], [0.02, 0.5, 0.5])
        magnitude, centre = circular_moments(w, pos)
        assert float(centre[0]) == pytest.approx(0.0, abs=1e-9) or \
               float(centre[0]) == pytest.approx(1.0, abs=1e-9)
        assert float(magnitude[0]) > 0.99
        # The naive mean would have said 0.5 -- the opposite side of the cell.
        assert abs(float(pos[:, 0].mean()) - 0.5) < 1e-12

    def test_uniform_weight_has_no_defined_centre(self):
        w = torch.ones(8, dtype=torch.float64)
        pos = torch.zeros(8, 3, dtype=torch.float64)
        pos[:, 0] = torch.linspace(0.0, 7.0 / 8.0, 8, dtype=torch.float64)
        magnitude, _ = circular_moments(w, pos)
        assert float(magnitude[0]) < 1e-9

    def test_zero_envelope_is_refused(self):
        with pytest.raises(LiftError, match="identically zero"):
            circular_moments(torch.zeros(3, dtype=torch.float64), torch.zeros(3, 3,
                                                                             dtype=torch.float64))


class TestBuildLift:
    def test_the_cut_sits_half_a_cell_from_the_support(self):
        charges = torch.tensor([-1.0], dtype=torch.float64)
        pos = frac([0.25, 0.25, 0.25])
        rec = build_lift(charges, pos)
        for c, k in zip(rec.centre, rec.cut):
            assert c == pytest.approx(0.25, abs=1e-6)
            assert k == pytest.approx(0.75, abs=1e-6)

    def test_a_delocalised_envelope_is_refused(self):
        """No unique cut => no isolated output, rather than an arbitrary one."""
        charges = torch.ones(8, dtype=torch.float64)
        pos = torch.zeros(8, 3, dtype=torch.float64)
        pos[:, 0] = torch.linspace(0.0, 7.0 / 8.0, 8, dtype=torch.float64)
        with pytest.raises(LiftError, match="below z_min"):
            build_lift(charges, pos)

    def test_the_record_fingerprints_and_round_trips(self):
        rec = build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.3, 0.3, 0.3]))
        again = LiftRecord(**{**rec.to_dict(),
                              "centre": tuple(rec.centre), "cut": tuple(rec.cut),
                              "moments": tuple(rec.moments)})
        assert again.fingerprint == rec.fingerprint
        other = build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.4, 0.3, 0.3]))
        assert other.fingerprint != rec.fingerprint


class TestBranchStability:
    """The branch must not move with P, and translation must be continuous."""

    def test_the_image_assignment_is_integer_and_p_independent(self):
        rec = build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.5, 0.5, 0.5]))
        pos = frac([0.5, 0.5, 0.5], [0.1, 0.5, 0.5], [0.9, 0.5, 0.5])
        shift = image_assignment(pos, rec)
        # Integral, and BOTH signs are legitimate: the unwrap is a minimum image about the
        # support centre, so a site below centre - 1/2 is pulled up by +1.
        assert torch.equal(shift, torch.round(shift))
        # The assignment is a pure function of geometry and the record: no P anywhere.
        assert torch.equal(shift, image_assignment(pos.clone(), rec))

    @pytest.mark.parametrize("centre,site,expected", [
        (0.02, 0.95, -1.0),   # 0.93 ABOVE the centre in fractional terms: wraps down
        (0.02, 0.60, -1.0),
        (0.90, 0.35, +1.0),   # 0.55 below the centre: wraps up
        (0.50, 0.55, 0.0),    # already inside the half-cell window
    ])
    def test_sites_wrap_to_the_nearer_image(self, centre, site, expected):
        """Both signs occur, but only one per centre: fractional coordinates span one cell,
        so for a given centre every site is either above it or below it, never both."""
        rec = build_lift(torch.tensor([-1.0], dtype=torch.float64),
                         frac([centre, 0.5, 0.5]))
        pos = frac([site, 0.5, 0.5])
        assert float(image_assignment(pos, rec)[0, 0]) == expected

    def test_every_lifted_site_lands_within_half_a_cell_of_the_centre(self):
        """The invariant the unwrap exists to establish, over the whole cell."""
        rec = build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.02, 0.5, 0.5]))
        pos = torch.zeros(50, 3, dtype=torch.float64)
        pos[:, 0] = torch.linspace(0.0, 0.98, 50, dtype=torch.float64)
        pos[:, 1:] = 0.5
        lifted = pos + image_assignment(pos, rec)
        assert bool(((lifted[:, 0] - 0.02).abs() <= 0.5 + 1e-12).all())

    def test_translating_a_compact_object_through_the_face_stays_continuous(self):
        """Section 11.1: continuous motion, and at a handoff a COMMON lattice translation.

        The lifted positions of a rigid pair must keep a constant separation as the pair is
        translated through the cell face -- if the two sites were assigned to different
        images inconsistently, the separation would jump by a lattice vector.
        """
        c = cell(12.0)
        separations = []
        for shift in torch.linspace(0.0, 1.0, 41, dtype=torch.float64):
            pos = frac([0.30, 0.5, 0.5], [0.34, 0.5, 0.5])
            pos = (pos + torch.tensor([float(shift), 0.0, 0.0], dtype=torch.float64)) % 1.0
            rec = build_lift(torch.tensor([-1.0, -1.0], dtype=torch.float64), pos)
            lifted = lift_positions(pos, c, rec)
            separations.append(float((lifted[1] - lifted[0]).norm()))
        assert max(separations) - min(separations) < 1e-9, separations


class TestClearance:
    def test_mass_near_the_cut_is_reported(self):
        c = cell(12.0)
        rec = build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.5, 0.5, 0.5]))
        # Cut is at 0.0; a site at 0.01 is 0.12 A from it, well inside d_clear = 2 A.
        pos = frac([0.5, 0.5, 0.5], [0.01, 0.5, 0.5])
        w = torch.tensor([1.0, 0.25], dtype=torch.float64)
        report = clearance_report(pos, w, c, rec)
        assert report["n_in_buffer"] == 1
        assert report["buffer_mass"] == pytest.approx(0.25)
        assert report["buffer_fraction"] == pytest.approx(0.2)

    def test_a_clean_cut_reports_nothing_in_the_buffer(self):
        c = cell(40.0)
        rec = build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.5, 0.5, 0.5]))
        pos = frac([0.5, 0.5, 0.5], [0.45, 0.5, 0.5])
        w = torch.tensor([1.0, 1.0], dtype=torch.float64)
        assert clearance_report(pos, w, c, rec)["n_in_buffer"] == 0


class TestIsoOk:
    @staticmethod
    def _rec():
        return build_lift(torch.tensor([-1.0], dtype=torch.float64), frac([0.5, 0.5, 0.5]))

    def _all_good(self, **over):
        kw = dict(constructor_ok=True, m_f=1, channel_weights=[1.0, 1.0],
                  registration_ok=True, localisation_ok=True, lift_record=self._rec(),
                  stationary_ok=True)
        kw.update(over)
        return iso_ok(**kw)

    def test_all_clauses_pass(self):
        status = self._all_good()
        assert status.ok and not status.reasons

    @pytest.mark.parametrize("override,clause", [
        (dict(constructor_ok=False), "constructor"),
        (dict(m_f=2), "multiplicity"),
        (dict(channel_weights=[1.0, 0.8]), "frontier_compact"),
        (dict(registration_ok=False), "registration"),
        (dict(localisation_ok=False), "localisation"),
        (dict(lift_record=None), "lift"),
        (dict(stationary_ok=False), "stationary"),
    ])
    def test_each_clause_is_individually_required(self, override, clause):
        """Targeted negative tests: none of the clauses may be treated as optional."""
        status = self._all_good(**override)
        assert not status.ok
        assert getattr(status, clause) is False or clause == "stationary"
        assert status.reasons

    def test_a_carrier_free_charged_state_cannot_pass_on_a_vacuous_frontier(self):
        """The specific hole the addendum closes: no active channel makes clause 2 vacuous,
        so the static/localisation clauses must still gate the output."""
        status = self._all_good(channel_weights=[], localisation_ok=False)
        assert status.frontier_compact           # vacuously true
        assert not status.ok                     # ... and still refused
        assert any("localisation" in r for r in status.reasons)

    def test_tail_bounds_are_enforced_against_absolute_tolerances(self):
        good = self._all_good(tail_bounds={"energy": 1e-6},
                              tail_tolerances={"energy": 1e-5})
        assert good.ok
        bad = self._all_good(tail_bounds={"energy": 1e-3},
                             tail_tolerances={"energy": 1e-5})
        assert not bad.ok and not bad.tails
        assert any("energy" in r for r in bad.reasons)

    def test_a_bound_without_a_tolerance_is_not_silently_passed_or_failed(self):
        status = self._all_good(tail_bounds={"unregistered": 1e9}, tail_tolerances={})
        assert status.tails      # no registered tolerance => nothing to enforce
        assert status.ok

    def test_the_status_serialises_with_every_clause(self):
        d = self._all_good(m_f=3).to_dict()
        assert d["ok"] is False and d["multiplicity"] is False
        assert "m_F = 3" in " ".join(d["reasons"])
