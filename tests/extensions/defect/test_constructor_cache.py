"""Split constructor cache keys (addendum section 3.5, gates in section 11.1)."""

import dataclasses

import pytest

from mace.modules.defect_constructor_cache import (
    InvariantFields,
    NumericalRegime,
    TargetFields,
    Tier1VerifierKey,
    Tier2AnchorKey,
    compatible_across_sizes,
    invalidates_tier1,
    invalidates_tier2,
    pristine_rank_increment,
    route,
    tiling_volume,
)


def invariant(**over) -> InvariantFields:
    base = dict(rank_core_schema="rank-core/1", constructor_checkpoint="ckpt-abc",
                parameter_hash="p-123", host_reference="CsPbCl3/pristine-40",
                basis_convention="sp-orthogonal/1", correspondence_algorithm="mincost/1",
                homology_algorithm="compdiff/1", homology_signature="V_Cl",
                smearing_convention="gaussian/0.05")
    base.update(over)
    return InvariantFields(**base)


def target(n_cells: int = 1, **over) -> TargetFields:
    base = dict(composition_hash=f"c-{n_cells}", tiling=(n_cells, 1, 1),
                pristine_rank=(160 * n_cells, 160 * n_cells),
                geometry_hash=f"g-{n_cells}", cell_hash=f"h-{n_cells}",
                n_atoms=80 * n_cells - 1, composition_difference=((17, -1),))
    base.update(over)
    return TargetFields(**base)


def regime(**over) -> NumericalRegime:
    base = dict(eigensolver="eigh", backend="cpu", dtype="float64", precision="double",
                tolerances={"scf": 1e-8})
    base.update(over)
    return NumericalRegime.of(**base)


def anchor_key(n_cells: int = 1, **over) -> Tier2AnchorKey:
    base = dict(invariant=invariant(), target=target(n_cells), tier2_schema="tier2/1",
                continuation_paths=("A", "B"), step_sizes=(0.05, 0.025), e_sink=50.0,
                eta=1e-3, endpoint_contiguity=True, regime=regime())
    base.update(over)
    return Tier2AnchorKey(**base)


def verifier_key(n_cells: int = 2, **over) -> Tier1VerifierKey:
    base = dict(verifier_schema="tier1/1", invariant=invariant(), target=target(n_cells),
                source_anchor_hash=anchor_key(1).digest,
                pristine_rank_increment=(160, 160), u_al=0.10,
                u_al_provenance="prov-abc", delta_search=0.30,
                perturbations=("precision", "geometry"), g_num=1e-4, regime=regime())
    base.update(over)
    return Tier1VerifierKey(**base)


class TestPartition:
    def test_verifier_schema_is_not_an_invariant_field(self):
        """A Tier-1 schema bump must not be able to reach the anchor key."""
        assert "verifier" not in " ".join(f.name for f in dataclasses.fields(InvariantFields))

    def test_invariant_digest_is_stable_and_sensitive(self):
        assert invariant().digest == invariant().digest
        assert invariant().digest != invariant(parameter_hash="p-999").digest

    def test_target_digests_differ_between_sizes_by_design(self):
        assert target(1).digest != target(2).digest


class TestInvalidation:
    def test_a_tier1_only_change_leaves_the_anchor_alone(self):
        """The whole point of the split: retuning the verifier keeps the expensive record."""
        before, after = verifier_key(), verifier_key(u_al=0.15, delta_search=0.35)
        assert invalidates_tier1(before, after)
        assert not invalidates_tier2(anchor_key(), anchor_key())

    def test_a_verifier_schema_bump_leaves_the_anchor_alone(self):
        assert invalidates_tier1(verifier_key(), verifier_key(verifier_schema="tier1/2"))
        assert not invalidates_tier2(anchor_key(), anchor_key())

    def test_a_common_field_change_invalidates_the_anchor(self):
        assert invalidates_tier2(anchor_key(), anchor_key(invariant=invariant(
            parameter_hash="p-999")))

    def test_a_tier2_setting_change_invalidates_the_anchor(self):
        assert invalidates_tier2(anchor_key(), anchor_key(step_sizes=(0.1, 0.05)))
        assert invalidates_tier2(anchor_key(), anchor_key(e_sink=200.0))
        assert invalidates_tier2(anchor_key(), anchor_key(endpoint_contiguity=False))

    def test_a_numerical_regime_change_invalidates_both(self):
        """Section 11.1: eigensolver/backend/dtype/precision changes invalidate the record."""
        other = regime(dtype="float32")
        assert invalidates_tier2(anchor_key(), anchor_key(regime=other))
        assert invalidates_tier1(verifier_key(), verifier_key(regime=other))

    def test_target_geometry_change_invalidates_the_anchor(self):
        assert invalidates_tier2(anchor_key(),
                                 anchor_key(target=target(1, geometry_hash="g-moved")))


class TestCrossSizeCompatibility:
    def test_reuse_succeeds_through_the_homology_predicate(self):
        ok, reason = compatible_across_sizes(anchor_key(1), target(2))
        assert ok, reason

    def test_exact_target_hashes_are_never_compared(self):
        """They are expected to differ; equality would make every larger cell a miss."""
        src, tgt = anchor_key(1), target(2)
        assert src.target.geometry_hash != tgt.geometry_hash
        assert src.target.cell_hash != tgt.cell_hash
        assert compatible_across_sizes(src, tgt)[0]

    def test_a_different_composition_difference_is_incompatible(self):
        ok, reason = compatible_across_sizes(
            anchor_key(1), target(2, composition_difference=((17, -1), (55, -1))))
        assert not ok
        assert "homologous" in reason

    def test_a_pristine_rank_that_does_not_scale_is_incompatible(self):
        ok, reason = compatible_across_sizes(anchor_key(1), target(2, pristine_rank=(300, 300)))
        assert not ok
        assert "scale with the tiling volume" in reason

    def test_an_invariant_mismatch_is_incompatible(self):
        ok, reason = compatible_across_sizes(anchor_key(1), target(2),
                                             invariant(basis_convention="spd/1"))
        assert not ok
        assert "invariant" in reason

    def test_the_rank_increment_is_the_pristine_increment(self):
        assert pristine_rank_increment(target(1), target(2)) == (160, 160)

    def test_tiling_volume(self):
        assert tiling_volume((2, 2, 1)) == 4


class TestRouting:
    def test_a_cached_verifier_record_short_circuits(self):
        key = verifier_key()
        where, _ = route({}, key, {key.digest: "record"})
        assert where == "tier1_cached"

    def test_a_compatible_anchor_runs_the_cheap_verifier_not_tier2(self):
        """Reading an anchor fingerprint must not execute a continuation."""
        key = verifier_key()
        where, reason = route({key.source_anchor_hash: "anchor"}, key, {})
        assert where == "tier1"
        assert "cheap verifier" in reason

    def test_no_anchor_routes_to_tier2(self):
        key = verifier_key(source_anchor_hash="")
        where, _ = route({}, key, {})
        assert where == "tier2"

    def test_an_unknown_anchor_hash_routes_to_tier2(self):
        key = verifier_key(source_anchor_hash="not-present")
        where, _ = route({"other": "anchor"}, key, {})
        assert where == "tier2"
