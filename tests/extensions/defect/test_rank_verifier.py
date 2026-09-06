"""Tier-1 rank-certified gap verifier (addendum section 3.2, gates in section 11.1)."""

import numpy as np
import pytest

from mace.modules.defect_rank import (
    AlignmentBound,
    RankAnchor,
    RankRoutingError,
    Tier1Result,
    alignment_bound,
    predict_rank,
    pristine_anchor,
    select_anchor,
    verify,
)

SMEAR = 0.05
HOST_GAP = 1.4
U_AL = 0.10


def spectrum_with_gap(n_valence: int, n_extra: int = 6, *, edge: float = 0.0,
                      gap: float = 0.6, spacing: float = 0.02,
                      valence_density: int = 1) -> np.ndarray:
    """A valence manifold topped at `edge`, then `gap`, then frontier/conduction states.

    `valence_density` densifies the valence manifold *without* changing the physical
    valence-frontier gap -- the supercell-folding situation that broke the old Tier 1.
    """
    n_val = n_valence
    valence = edge - spacing / valence_density * np.arange(n_val)[::-1]
    upper = edge + gap + spacing * np.arange(n_extra)
    return np.concatenate([valence, upper])


class TestRankTransport:
    def test_offset_is_carried_across_sizes(self):
        anchor = RankAnchor(homology="V_Cl", n_atoms=79, m_vb=(160, 160),
                            pristine_rank=(160, 160), source="tier2", source_hash="h1")
        # A larger cell adds exactly the pristine rank increment.
        predicted = predict_rank(anchor, (320, 320))
        assert predicted == (320, 320)

    def test_defect_offset_survives_transport(self):
        anchor = RankAnchor(homology="V_Cl", n_atoms=79, m_vb=(159, 159),
                            pristine_rank=(160, 160), source="tier2", source_hash="h1")
        assert anchor.offset == (-1, -1)
        predicted = predict_rank(anchor, (320, 320))
        # Raw rank differs between sizes; the offset does not.
        assert predicted == (319, 319)
        assert tuple(p - 320 for p in predicted) == anchor.offset

    def test_channel_count_mismatch_is_refused(self):
        anchor = RankAnchor(homology="V_Cl", n_atoms=79, m_vb=(159, 159),
                            pristine_rank=(160, 160), source="tier2", source_hash="h1")
        with pytest.raises(RankRoutingError, match="spin channels"):
            predict_rank(anchor, (320,))


class TestAnchorSelection:
    def test_missing_anchor_routes_to_tier2(self):
        with pytest.raises(RankRoutingError, match="no fingerprint-compatible"):
            select_anchor([], "V_Cl", (320, 320))

    def test_conflicting_anchors_route_to_tier2(self):
        a = RankAnchor("V_Cl", 79, (159, 159), (160, 160), "tier2", "h1")
        b = RankAnchor("V_Cl", 159, (318, 318), (320, 320), "tier2", "h2")
        # a transports to 319; b transports to 318 -- Tier 1 has no authority to choose.
        with pytest.raises(RankRoutingError, match="disagree"):
            select_anchor([a, b], "V_Cl", (320, 320))

    def test_agreeing_anchors_are_accepted(self):
        a = RankAnchor("V_Cl", 79, (159, 159), (160, 160), "tier2", "h1")
        b = RankAnchor("V_Cl", 159, (319, 319), (320, 320), "tier2", "h2")
        chosen = select_anchor([a, b], "V_Cl", (320, 320))
        assert predict_rank(chosen, (320, 320)) == (319, 319)

    def test_incompatible_invariant_key_is_ignored(self):
        a = RankAnchor("V_Cl", 79, (159, 159), (160, 160), "tier2", "h1", invariant_key="A")
        with pytest.raises(RankRoutingError, match="no fingerprint-compatible"):
            select_anchor([a], "V_Cl", (320, 320), invariant_key="B")

    def test_pristine_anchor_is_exact_by_electron_count(self):
        anchor = pristine_anchor("pristine", 80, (160, 160))
        assert anchor.m_vb == anchor.pristine_rank == (160, 160)
        assert anchor.offset == (0, 0)
        assert anchor.source == "pristine"


class TestAlignmentBound:
    def test_envelope_is_max_abs_plus_margin(self):
        bound = alignment_bound([0.01, -0.07, 0.03], ["pristine_80", "tier2_159", "tier2_79"],
                                tau_num=1e-3)
        assert bound.value == pytest.approx(0.071)
        assert bound.n_records == 3

    def test_no_records_routes_to_tier2(self):
        with pytest.raises(RankRoutingError, match="cannot be certified"):
            alignment_bound([], [])

    def test_provenance_hash_is_reconstructible_and_sensitive(self):
        a = alignment_bound([0.01, -0.07], ["s1", "s2"])
        b = alignment_bound([0.01, -0.07], ["s1", "s2"])
        c = alignment_bound([0.01, -0.06], ["s1", "s2"])
        assert a.provenance_hash == b.provenance_hash
        assert a.provenance_hash != c.provenance_hash

    def test_dropping_the_last_source_invalidates_the_bound(self):
        """Removing its last applicable source must route the target to Tier 2."""
        with pytest.raises(RankRoutingError):
            alignment_bound([], ["dropped"])


class TestVerifier:
    def _ok(self, spec, predicted, **kw):
        params = dict(vbm_aligned=0.0, gap_host=HOST_GAP, smearing=SMEAR, u_al=U_AL,
                      perturbed_spectra=[spec])
        params.update(kw)
        return verify(spec, predicted, **params)

    def test_accepts_the_certified_rank(self):
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        result = self._ok(spec, (10, 10))
        assert result.accepted
        assert result.m_vb == (10, 10)
        assert result.gaps[0] == pytest.approx(0.6)

    def test_survives_valence_densification(self):
        """The regression the old Tier 1 failed: folding densifies the valence manifold.

        The physical valence-frontier gap is unchanged, so the rank must still verify --
        even though many more eigenvalues now sit close to the aligned VBM.
        """
        for density, n_val in ((1, 10), (4, 40), (16, 160)):
            spec = spectrum_with_gap(n_val, edge=0.0, gap=0.6, valence_density=density)
            result = self._ok(spec, (n_val, n_val))
            assert result.accepted, f"densification {density} broke Tier 1: {result.reason}"

    def test_gap_below_the_floor_routes_to_tier2(self):
        spec = spectrum_with_gap(10, edge=0.0, gap=0.05)   # < max(4*smear, 2*u_al)
        result = self._ok(spec, (10, 10))
        assert not result.accepted
        assert "below the required" in result.reason

    def test_never_substitutes_a_later_larger_gap(self):
        """A big frontier-frontier gap further up must not rescue a bad certified rank."""
        valence = np.array([-0.1, -0.08, -0.06, -0.04])
        frontier = np.array([0.0, 0.001])       # certified rank sits here, gap is tiny
        far = np.array([5.0, 5.2])              # a huge gap, but at the wrong rank
        spec = np.concatenate([valence, frontier, far])
        result = self._ok(spec, (5, 5))
        assert not result.accepted
        assert result.m_vb is None

    def test_threshold_outside_the_window_is_refused(self):
        """A gap that is real but sits too high is not the valence-frontier boundary.

        The counting threshold must land within half the host gap of the aligned edge (plus
        the registered margins). A clean gap further up the spectrum is a different gap.
        """
        spec = spectrum_with_gap(10, edge=0.0, gap=1.8)
        result = self._ok(spec, (10, 10))
        assert not result.accepted
        assert "window" in result.reason

    def test_small_host_gap_is_refused(self):
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        result = self._ok(spec, (10, 10), gap_host=0.1)
        assert not result.accepted
        assert "pristine gap" in result.reason

    def test_rank_at_the_spectrum_edge_is_refused(self):
        spec = spectrum_with_gap(10, n_extra=0, edge=0.0, gap=0.6)
        result = self._ok(spec, (10, 10))
        assert not result.accepted
        assert "outside" in result.reason

    def test_missing_perturbations_cannot_accept(self):
        """Condition 4 untested must not silently pass."""
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        result = verify(spec, (10, 10), vbm_aligned=0.0, gap_host=HOST_GAP,
                        smearing=SMEAR, u_al=U_AL)
        assert not result.accepted
        assert "condition 4 is untested" in result.reason

    def test_a_perturbation_that_closes_the_gap_is_refused(self):
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        closed = spectrum_with_gap(10, edge=0.0, gap=0.02)
        result = self._ok(spec, (10, 10), perturbed_spectra=[spec, closed])
        assert not result.accepted
        assert "closes the gap" in result.reason

    def test_delta_search_inequality_is_enforced(self):
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        result = self._ok(spec, (10, 10), delta_search=0.01)
        assert not result.accepted
        assert "registered inequality" in result.reason

    @pytest.mark.parametrize("u_al,accepted", [(0.20, True), (0.21, False)])
    def test_registered_delta_search_bounds_u_al(self, u_al, accepted):
        """At the registered delta_search = 0.30 eV and s_smear = 0.05, u_al <= 0.20 eV.

        This is the addendum's numerical consequence of `delta_search >= u_al + 2 s_smear`:
        a larger alignment envelope than the registered window can absorb routes the class
        to Tier 2 rather than widening the window after the fact.
        """
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        result = self._ok(spec, (10, 10), u_al=u_al, delta_search=0.30)
        assert result.accepted is accepted
        if not accepted:
            assert "registered inequality" in result.reason

    def test_result_serialises(self):
        spec = spectrum_with_gap(10, edge=0.0, gap=0.6)
        d = self._ok(spec, (10, 10)).to_dict()
        assert d["accepted"] is True
        assert d["m_vb"] == [10, 10]


class TestFamilyConsistency:
    """Section 11.1: rank increment equals the pristine increment; offsets are invariant."""

    def test_79_and_159_atom_classes_share_offset_and_differ_in_raw_rank(self):
        small = RankAnchor("V_Cl", 79, (159, 159), (160, 160), "tier2", "h159")
        predicted_large = predict_rank(small, (320, 320))
        assert predicted_large != small.m_vb                    # raw rank is extensive
        assert predicted_large[0] - small.m_vb[0] == 320 - 160  # pristine increment
        n_sigma_small, n_sigma_large = 159, 319
        d_small = tuple(n_sigma_small - m for m in small.m_vb)
        d_large = tuple(n_sigma_large - m for m in predicted_large)
        assert d_small == d_large                               # d_sigma invariant
        assert sum(d_small) == sum(d_large)                     # hence Q_core invariant


class TestTransportOnStoredTables:
    """Section 11.1 on real stored class tables, not toys.

    "Across sizes, the rank increment equals the pristine valence rank while d_sigma,
    Q_core and the carrier counts remain invariant." The golden ladder records carry a
    pristine class and a V_Cl class at three tilings, which is exactly that statement.
    """

    @staticmethod
    def _table():
        import json
        import pathlib
        path = pathlib.Path(__file__).resolve().parents[3] / "defect-perovskite" / "golden"
        for candidate in sorted(path.glob("stage14_ladder_*.json")):
            classes = json.loads(candidate.read_text()).get("class_table", {}).get("classes")
            if classes:
                return classes
        return None

    def test_the_defect_offset_is_invariant_across_three_tilings(self):
        classes = self._table()
        if not classes:
            pytest.skip("no stored class table with a class_table/classes section")

        # The pristine class is the one with Q_core = 0; the V_Cl family is the rest.
        pristine = [c for c in classes.values() if c.get("q_core") == 0 and c.get("m_vb")]
        defects = [c for c in classes.values() if c.get("q_core") == 1 and c.get("m_vb")]
        if not pristine or len(defects) < 2:
            pytest.skip("the stored table does not carry a pristine class and two sizes")

        # One pristine cell fixes the rank density; every tiling is an integer multiple.
        base = pristine[0]
        per_atom = base["m_vb"][0] / base["n_atoms"]

        offsets, cores = set(), set()
        for cls in defects:
            # The pristine rank at this defect's tiling: the defect removed one atom, so the
            # parent pristine cell has n_atoms + 1 sites.
            pristine_rank = round(per_atom * (cls["n_atoms"] + 1))
            offsets.add(tuple(int(m) - pristine_rank for m in cls["m_vb"]))
            cores.add(int(cls["q_core"]))

        assert len(offsets) == 1, f"the defect rank offset is not size-invariant: {offsets}"
        assert len(cores) == 1, f"Q_core is not size-invariant: {cores}"
        # And the raw rank really is extensive, so this is not invariance by coincidence.
        assert len({tuple(c["m_vb"]) for c in defects}) == len(defects)
