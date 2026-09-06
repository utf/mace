"""Plan v8 section 2.1, Stage 0 task 0.7: the composition-class constructor, Tier 1.

Two layers, tested separately. The INTEGER layer (`align_edges`, `class_integers`,
`frame_counts`) is exercised on toy spectra whose expected integers are defined by the toy
alone. The v8.1 rank-certified Tier-1 verifier itself is tested in `test_rank_verifier.py`;
the VBM-proximity `tier1` it replaced has been deleted. The SPECTRUM layer is exercised on a Harrison-initialised counting head over cubic
CsPbCl3 supercells -- the pristine class, V_Cl at 39 and 79 atoms -- where the plan's
benchmark (V_Cl^0: Q_core = +1, n_e,maj = 1, q_F = -1; V_Cl^+: n_e = 0, q_F = 0) is asserted.
An AST test fences the Tier 2 machinery inside the constructor module.
"""

from __future__ import annotations

import ast
import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from mace import data, tools
from mace.modules import defect_composition as dc
from mace.modules.defect_cache import attach_frame_keys
from mace.modules.defect_models import MACEDefect
from mace.modules.defect_protocol import apply_harrison
from mace.modules.defect_state import StateBatch
from tests.extensions.defect.test_neutral_reference_skip import _perovskite
from tests.unit.test_base_cache_precision import _model

Z_TABLE = tools.AtomicNumberTable([17, 55, 82])
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module", autouse=True)
def _f64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


# ------------------------------------------------------------------ the integer layer


class TestIntegers:
    def test_the_key_is_the_sorted_multiset(self):
        assert dc.composition_key([82, 17, 55, 17, 17]) == "17x3,55x1,82x1"
        assert dc.composition_key([17] * 47 + [55] * 16 + [82] * 16) == "17x47,55x16,82x16"

    def test_stoichiometry_is_a_composition_test_not_an_atom_count(self):
        formula = {17: 3.0, 55: 1.0, 82: 1.0}
        assert dc.is_stoichiometric([17] * 48 + [55] * 16 + [82] * 16, formula)
        assert dc.is_stoichiometric([17] * 3 + [55] + [82], formula)
        assert not dc.is_stoichiometric([17] * 47 + [55] * 16 + [82] * 16, formula)
        assert not dc.is_stoichiometric([17] * 48 + [55] * 16, formula)

    def test_the_reference_fill_is_the_count_fill_of_the_valence_sum(self):
        assert dc.reference_fill(409) == (205, 204)
        assert dc.reference_fill(416) == (208, 208)

    def test_class_integers_from_either_tier(self):
        assert dc.class_integers((204, 204), (205, 204)) == ((1, 0), (0, 0), 1)
        assert dc.class_integers((208, 208), (208, 208)) == ((0, 0), (0, 0), 0)
        assert dc.class_integers((205, 205), (204, 204)) == ((0, 0), (1, 1), -2)

    def test_the_alignment_recovers_a_rigid_shift_and_ignores_the_frontier(self):
        rng = np.random.default_rng(0)
        pristine = np.sort(np.concatenate([rng.uniform(-12.0, -7.0, 200),
                                           rng.uniform(-3.0, 0.0, 100)]))
        n_pri = 200
        a = dc.align_edges(pristine + 0.37, n_pri, pristine, n_pri)
        assert a.shift == pytest.approx(0.37, abs=1e-12)
        assert a.spread == pytest.approx(0.0, abs=1e-12)
        assert a.vbm_al == pytest.approx(pristine[199] + 0.37)
        assert a.cbm_al == pytest.approx(pristine[200] + 0.37)
        assert a.gap_pristine == pytest.approx(pristine[200] - pristine[199])
        # A frontier level inside the occupied count sits above the quantile window; it
        # changes the alignment only through the O(1/N) rescaling of the quantile positions.
        shifted = np.sort(np.concatenate([pristine[:200] + 0.37, [-6.0], pristine[200:] + 0.37]))
        b = dc.align_edges(shifted, 201, pristine, n_pri)
        assert abs(b.shift - 0.37) < 0.01

    @staticmethod
    def record(n_e, n_h, key="17x47,55x16,82x16"):
        q_core = sum(n_e) - sum(n_h)
        return dc.ClassRecord(key=key, n_atoms=79, reference_frame_key=1, n_total=409,
                              n_sigma=(205, 204), tier=1, ambiguous=False, reason="",
                              m_vb=(204, 204), n_e=n_e, n_h=n_h, q_core=q_core, vbm_al=0.0,
                              cbm_al=1.0, shift=0.0, spread=0.0, gap_pristine=1.0,
                              delta=0.1, nearest=-0.1)

    def test_the_benchmark_per_frame_counts(self):
        """V_Cl^0: Q_core = +1, n_e,maj = 1, q_F = -1; V_Cl^+: n_e = 0, q_F = 0."""
        vcl = self.record((1, 0), (0, 0))
        assert vcl.q_core == 1
        states = StateBatch.from_counts(torch.tensor([[0.0, 0, 0, 0], [0, 0, 1.0, 0]]))
        assert dc.frame_counts(vcl, states, 0) == ((1, 0), (0, 0), -1)
        # The frontier electron REMOVED: no electron and no hole, not one of each.
        assert dc.frame_counts(vcl, states, 1) == ((0, 0), (0, 0), 0)

    def test_pristine_frames_carry_only_the_counter(self):
        pristine = self.record((0, 0), (0, 0), key="17x48,55x16,82x16")
        states = StateBatch.from_counts(torch.tensor([[0.0, 0, 1.0, 0], [1.0, 0, 0, 0]]))
        assert dc.frame_counts(pristine, states, 0) == ((0, 0), (1, 0), 1)
        assert dc.frame_counts(pristine, states, 1) == ((1, 0), (0, 0), -1)

    def test_the_identity_q_f_equals_q_formal_minus_q_core_is_asserted(self):
        rec = self.record((1, 0), (0, 0))
        states = StateBatch.from_counts(torch.tensor([[0.0, 1.0, 0, 0], [0, 0, 0, 2.0]]))
        for g in range(2):
            n_e, n_h, q_f = dc.frame_counts(rec, states, g)
            assert q_f == int(states.q_formal[g]) - rec.q_core
            assert q_f == sum(n_h) - sum(n_e)

    def test_an_uncounted_class_has_no_per_frame_counts(self):
        rec = dc.ClassRecord(**{**self.record((0, 0), (0, 0)).__dict__, "tier": None,
                                "ambiguous": True, "reason": "synthetic"})
        states = StateBatch.from_counts(torch.zeros(1, 4))
        with pytest.raises(ValueError, match="no core/frontier decomposition"):
            dc.frame_counts(rec, states, 0)

    @pytest.mark.parametrize("m", [2, 3, 5])
    def test_a_synthetic_class_with_m_occupied_frontier_levels(self, m):
        """`n_e = m`, `Q_core = m` from the toy spectrum and electron count alone; with a
        Q_formal = 0 synthetic counter state, `q_F = -m`. No physical class is implied."""
        vbm, delta = -5.0, 0.1
        pristine = np.concatenate([np.linspace(-9.0, vbm, 40), np.linspace(vbm + 3, vbm + 5, 10)])
        spectrum = np.sort(np.concatenate([pristine, np.full(m, vbm + 1.5)]))
        n_pri = 40
        n_sigma = (n_pri + m, n_pri + m)   # both spins fill m frontier levels
        a = dc.align_edges(spectrum, n_sigma[0], pristine, n_pri)
        # The toy's valence rank is exact by construction (40 pristine valence levels),
        # which is the only kind of rank the v8.1 verifier accepts without an anchor.
        m_vb = n_pri
        n_e, n_h, q_core = dc.class_integers((m_vb, m_vb), n_sigma)
        assert n_e == (m, m) and n_h == (0, 0) and q_core == 2 * m
        rec = dc.ClassRecord(key="synthetic", n_atoms=0, reference_frame_key=0,
                             n_total=2 * n_sigma[0], n_sigma=n_sigma, tier=1, ambiguous=False,
                             reason="", m_vb=(m_vb, m_vb), n_e=n_e, n_h=n_h, q_core=q_core,
                             vbm_al=a.vbm_al, cbm_al=a.cbm_al, shift=a.shift, spread=a.spread,
                             gap_pristine=a.gap_pristine, delta=delta, nearest=0.0)
        neutral = StateBatch.from_counts(torch.zeros(1, 4))
        assert dc.frame_counts(rec, neutral, 0) == ((m, m), (0, 0), -2 * m)


# ------------------------------------------------------------------ the spectrum layer


def _frame(atoms, counts):
    config = data.Configuration(
        atomic_numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
        cell=np.array(atoms.get_cell()), pbc=(True, True, True),
        properties={"carrier_counts": counts}, property_weights={})
    return data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=6.0)


def _remove_cl(atoms, which):
    out = atoms.copy()
    cl = [i for i, z in enumerate(out.get_atomic_numbers()) if z == 17]
    del out[cl[which]]
    return out


@pytest.fixture(scope="module")
def harrison_model():
    model = _model()
    apply_harrison(model, model.atomic_numbers)
    return model


@pytest.fixture(scope="module")
def frames():
    pristine = _perovskite(reps=(2, 2, 2), rattle=0.02, seed=1)
    big = _perovskite(reps=(2, 2, 4), rattle=0.02, seed=2)
    out = {
        "pristine": _frame(pristine, [0.0] * 4),
        "pristine_plus": _frame(pristine, [0.0, 0.0, 1.0, 0.0]),
        "vcl_39": _frame(_remove_cl(pristine, 0), [0.0] * 4),
        "vcl_39_plus": _frame(_remove_cl(pristine, 0), [0.0, 0.0, 1.0, 0.0]),
        "pristine_80": _frame(big, [0.0] * 4),
        "vcl_79": _frame(_remove_cl(big, 3), [0.0] * 4),
    }
    attach_frame_keys(list(out.values()), z_table=Z_TABLE)
    return out


@pytest.fixture(scope="module")
def table(harrison_model, frames):
    return dc.build_class_table(harrison_model, list(frames.values()), log=False)


class TestOnTheHarrisonHead:
    def test_the_pristine_class_counts_nothing(self, table):
        rec = dc.lookup_class(table, [17] * 24 + [55] * 8 + [82] * 8)
        assert rec.counted and rec.tier == 1
        assert rec.n_e == (0, 0) and rec.n_h == (0, 0) and rec.q_core == 0
        assert rec.m_vb == rec.n_sigma == (104, 104)
        assert rec.shift == 0.0 and rec.spread == 0.0
        assert rec.gap_pristine > 4 * table["smearing_width"]

    def test_the_pristine_gap_and_the_margin_are_recorded(self, table, harrison_model):
        width = float(harrison_model.spectral.t_el)
        assert table["delta"] == pytest.approx(2 * width)
        assert table["window"] == pytest.approx(width)
        assert table["reference_geometry"] == "first_frame"
        assert table["pristine_key"] == "17x24,55x8,82x8"

    def test_v_cl_at_39_atoms_is_the_benchmark(self, table, frames):
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        assert rec.counted, rec.reason
        assert rec.n_sigma == (101, 100) and rec.m_vb == (100, 100)
        assert rec.n_e == (1, 0) and rec.n_h == (0, 0) and rec.q_core == 1
        assert rec.reference_frame_key == int(frames["vcl_39"].frame_key)
        states = StateBatch.from_counts(torch.tensor([[0.0, 0, 0, 0], [0, 0, 1.0, 0]]))
        assert dc.frame_counts(rec, states, 0) == ((1, 0), (0, 0), -1)
        assert dc.frame_counts(rec, states, 1) == ((0, 0), (0, 0), 0)

    def test_v_cl_at_79_atoms_gives_the_same_integers(self, table):
        """The larger cell yields the same integers (plan section 7.1).

        Under the v8.1 verifier this is now reached by TRANSPORT, not by a second
        continuation: the 39-atom class of the same homologous family has no anchor and so
        runs Tier 2, and the 79-atom class then verifies the transported rank at Tier 1.
        That is the design -- the expensive continuation runs once per family, and other
        sizes are checked by the cheap verifier. The integers are what must not move.
        """
        small = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        rec = dc.lookup_class(table, [17] * 47 + [55] * 16 + [82] * 16)
        assert rec.tiling == (1, 1, 2)
        assert rec.counted, rec.reason
        assert rec.n_e == (1, 0) and rec.n_h == (0, 0) and rec.q_core == 1

        assert small.tier == 2, "the first size of a family has no anchor and must continue"
        assert rec.tier == 1, rec.tier1_reason
        assert "anchor" in rec.tier1_reason

        # The raw rank is extensive and MUST differ between the two sizes, while the
        # offset -- the part with physical content -- is invariant.
        assert rec.m_vb != small.m_vb
        assert rec.d_sigma == small.d_sigma
        assert rec.q_core == small.q_core

    def test_the_larger_pristine_cell_is_aligned_against_the_tiled_reference(self, table):
        rec = dc.lookup_class(table, [17] * 48 + [55] * 16 + [82] * 16)
        assert rec.tiling == (1, 1, 2)
        assert rec.counted, rec.reason
        assert rec.q_core == 0 and rec.n_e == (0, 0) and rec.n_h == (0, 0)
        assert abs(rec.shift) < 0.05 and rec.spread < 0.05

    def test_the_integers_do_not_depend_on_the_frame_charge_state(self, harrison_model,
                                                                  frames):
        """The constructor evaluates at S_ref whatever counters the frame carries."""
        charged = dc.build_class_table(
            harrison_model, [frames["pristine_plus"], frames["vcl_39_plus"]], log=False)
        neutral = dc.build_class_table(
            harrison_model, [frames["pristine"], frames["vcl_39"]], log=False)
        for key in charged["classes"]:
            for f in ("m_vb", "n_e", "n_h", "q_core", "vbm_al", "tier"):
                assert charged["classes"][key][f] == neutral["classes"][key][f]

    def test_the_table_records_the_constructor_parameters(self, table):
        for key in dc.DEFAULT_CONSTRUCTOR:
            assert key in table
        assert table["eta"] == 1.0e-3 and table["dlambda"] == 0.02 and table["r_match"] == 2.0
        # The sink was resolved for the class Tier 2 counted: 50 eV above the pristine CBM.
        pristine = dc.lookup_class(table, [17] * 24 + [55] * 8 + [82] * 8)
        assert table["e_sink"] == pytest.approx(pristine.cbm_al + 50.0)

    def test_a_composition_outside_the_table_is_refused(self, table):
        with pytest.raises(KeyError, match="no class record"):
            dc.lookup_class(table, [17] * 22 + [55] * 8 + [82] * 8)

    def test_verification_reproduces_the_table(self, harrison_model, frames, table):
        harrison_model.composition_classes = table
        assert dc.verify_class_table(harrison_model, list(frames.values())) == []
        # A moved integer is reported, not silently re-fitted.
        broken = copy.deepcopy(table)
        broken["classes"]["17x23,55x8,82x8"]["q_core"] = 3
        diffs = dc.verify_class_table(harrison_model, list(frames.values()), table=broken)
        assert diffs == ["17x23,55x8,82x8: q_core 3 -> 1"]

    def test_without_a_pristine_frame_every_class_is_uncounted(self, harrison_model, frames):
        t = dc.build_class_table(harrison_model, [frames["vcl_39"]], log=False)
        assert t["pristine_key"] is None
        rec = dc.lookup_class(t, [17] * 23 + [55] * 8 + [82] * 8)
        assert not rec.counted and "no pristine reference" in rec.reason

    def test_the_table_is_config_and_survives_extraction_and_pickling(self, harrison_model,
                                                                      table):
        from mace.tools.scripts_utils import extract_config_mace_model

        harrison_model.composition_classes = table
        config = extract_config_mace_model(harrison_model)
        assert config["composition_classes"] == table
        # The None defaults were resolved at construction (every non-parameter float is a
        # number in the config): delta = 2 x smearing, window = smearing.
        width = float(harrison_model.spectral.t_el)
        expected = dc.constructor_config(None)
        expected.update(delta=2 * width, window=width)
        assert config["class_constructor"] == expected
        assert config["functional"]["delta"] == 2 * width
        assert config["functional"]["delta_s"] == width
        assert config["functional"]["r_split"] == 4.0
        assert config["functional"]["r_orb"] == {17: 1.02, 55: 2.44, 82: 1.46}
        rebuilt = MACEDefect(**config)
        assert rebuilt.composition_classes == table
        assert pickle.loads(pickle.dumps(harrison_model)).composition_classes == table
        assert dc.ClassRecord.from_dict(table["classes"]["17x23,55x8,82x8"]).q_core == 1


class TestTiling:
    def test_integer_factors_are_recovered_from_thermal_cells(self):
        ref = np.diag([15.8, 16.1, 11.4])
        assert dc.tiling_factors(np.diag([31.9, 16.25, 11.3]), ref, 159, 80) == (2, 1, 1)
        assert dc.tiling_factors(ref * 1.01, ref, 80, 80) == (1, 1, 1)
        assert dc.tiling_factors(np.diag([15.8, 16.1, 5.7]), ref, 40, 80) is None
        assert dc.tiling_factors(np.diag([23.7, 16.1, 11.4]), ref, 120, 80) is None

    def test_the_159_atom_cell_is_the_80_atom_cell_doubled_along_c_and_relabelled(self):
        """The golden frames: pristine (15.80, 16.08, 11.38), V_Cl 159 (16.09, 22.53, 15.72).
        Factors come back in the REFERENCE's axis order."""
        ref = np.array([[15.79, 0.09, 0.18], [0.0, 16.08, -0.26], [0.0, 0.0, 11.38]])
        cell = np.diag([16.092, 22.528, 15.717])
        assert dc.tiling_factors(cell, ref, 159, 80) == (1, 1, 2)
        # Two thermal pristine snapshots with a and b interchanged are the same cell.
        other = np.array([[16.17, 0.14, 0.09], [0.0, 15.88, -0.12], [0.0, 0.0, 11.4]])
        assert dc.tiling_factors(other, ref, 80, 80) == (1, 1, 1)
        # A wrong shape at the right lengths is refused: the angles must match too.
        sheared = ref.copy()
        sheared[0, 1] = 6.0
        assert dc.tiling_factors(sheared, ref, 80, 80) is None

    def test_tiling_a_frame_replicates_positions_and_scales_the_cell(self):
        nums, pos, cell = dc.tile_frame([17, 55], [[0.0, 0, 0], [1.0, 1, 1]],
                                        np.diag([2.0, 3.0, 4.0]), (2, 1, 3))
        assert nums.tolist() == [17, 55] * 6
        assert np.allclose(cell, np.diag([4.0, 3.0, 12.0]))
        assert pos.shape == (12, 3) and np.allclose(pos[2], [0.0, 0.0, 4.0])


# ------------------------------------------------------------------ Tier 2


def _toy(n_sites, rng, n_deep=None, valence=-9.0, conduction=-1.0, spread=0.5, hop=0.3):
    """A random tight-binding toy with a GAP: the first `n_deep` sites (default half) carry
    ORB orbitals each around `valence`, the rest around `conduction`, with hoppings of scale
    `hop` on a ring. Its occupied manifold at rank `ORB * n_deep` is well separated, which
    is what makes a valence subspace a definite thing to transport; a gapless toy is a
    metal, and a metal's "valence subspace" is genuinely path-dependent."""
    n_deep = n_sites // 2 if n_deep is None else n_deep
    dim = dc.ORB * n_sites
    H = np.zeros((dim, dim))
    for i in range(n_sites):
        centre = valence if i < n_deep else conduction
        block = rng.normal(0.0, 0.2, (dc.ORB, dc.ORB))
        block = 0.5 * (block + block.T) + np.eye(dc.ORB) * (centre + spread * rng.uniform(-1, 1))
        H[dc.ORB * i: dc.ORB * (i + 1), dc.ORB * i: dc.ORB * (i + 1)] = block
        j = (i + 1) % n_sites
        t = rng.normal(0.0, hop, (dc.ORB, dc.ORB))
        H[dc.ORB * i: dc.ORB * (i + 1), dc.ORB * j: dc.ORB * (j + 1)] += t
        H[dc.ORB * j: dc.ORB * (j + 1), dc.ORB * i: dc.ORB * (i + 1)] += t.T
    return H


def _groups(dim, ghost=(), added=(), substituted=()):
    orb = lambda sites: np.array([dc.ORB * s + o for s in sites for o in range(dc.ORB)],
                                 dtype=np.int64)
    return {"ghost": orb(ghost), "added": orb(added), "substituted": orb(substituted)}


def _embed_removal(H0, site, deep=-14.0):
    """`H1` for the removal of `site` from the toy `H0`, already in the union basis (the
    site's rows and columns zeroed). The removed site's levels are first made DEEP in `H0`
    so all four of its orbitals are occupied at the toy's fill, as an anion's are."""
    idx = np.arange(dc.ORB * site, dc.ORB * (site + 1))
    H0[np.ix_(idx, idx)] = np.eye(dc.ORB) * deep + 0.1 * (H0[np.ix_(idx, idx)]
                                                          - np.diag(np.diag(H0[np.ix_(idx, idx)])))
    H1 = H0.copy()
    idx = np.arange(dc.ORB * site, dc.ORB * (site + 1))
    H1[idx, :] = 0.0
    H1[:, idx] = 0.0
    return H1


class TestTier2OnTheHarrisonHead:
    def test_the_family_is_counted_at_tier_2_with_one_ghost_cl(self, table):
        """Tier 2's continuation on the class that actually runs it.

        Under the v8.1 verifier the FIRST size of a homologous family has no anchor and so
        runs the continuation; later sizes are verified by transport. The assertions are
        written size-independently -- one ghost Cl contributes four ghost orbitals at any
        cell size, and the physical part of the transported subspace has exactly M_VB
        eigenvalues -- so they keep testing the machinery rather than a routing outcome.
        """
        rec = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        assert rec.counted and rec.tier == 2, rec.reason
        assert rec.n_e == (1, 0) and rec.n_h == (0, 0) and rec.q_core == 1
        assert rec.path_agreement and rec.schedule_agreement
        assert rec.correspondence["n_ghost"] == 1 and rec.correspondence["ghost_species"] == [17]
        assert rec.correspondence["n_added"] == 0 and rec.correspondence["n_substituted"] == 0
        gamma = np.asarray(rec.gamma)
        assert (gamma > 1 - table["eta"]).sum() == 4          # one Cl: s + p
        assert (gamma < table["eta"]).sum() == rec.m_vb[0]    # the physical valence part

    def test_the_transported_size_agrees_with_the_continued_one(self, table):
        """The integers Tier 2 established must survive transport to the larger cell."""
        small = dc.lookup_class(table, [17] * 23 + [55] * 8 + [82] * 8)
        large = dc.lookup_class(table, [17] * 47 + [55] * 16 + [82] * 16)
        assert large.counted, large.reason
        assert large.n_e == small.n_e and large.n_h == small.n_h
        assert large.q_core == small.q_core and large.d_sigma == small.d_sigma
        assert large.m_vb == (204, 204) and large.n_sigma == (205, 204)

    def test_tier_2_reproduces_tier_1_on_the_39_atom_class_on_both_paths_and_schedules(
            self, harrison_model, frames):
        spec0 = dc.head_spectra(harrison_model, dc._single_frame_dict(frames["pristine"], "cpu"))[0]
        spec1 = dc.head_spectra(harrison_model, dc._single_frame_dict(frames["vcl_39"], "cpu"))[0]
        out = dc.tier2(harrison_model, frames["vcl_39"], frames["pristine"], (1, 1, 1),
                       (0, 1, 2), 104, spec0["H"], spec1["H"], dc.constructor_config(None))
        assert out["accepted"] and out["m_vb"] == 100
        for run in out["runs"].values():
            assert run["m_vb"] == 100 and run["n_ghost"] == 4 and run["n_closure"] == 0
            assert run["contiguous"] and run["overlap"] > 0.999
            assert run["min_separation"] > 0.9

    def test_the_site_correspondence_finds_the_vacancy_through_thermal_noise(self, harrison_model,
                                                                             frames):
        n1, p1, c1 = dc._frame_geometry(frames["vcl_79"], harrison_model)
        n0, p0, c0 = dc._frame_geometry(frames["pristine"], harrison_model)
        n0, p0, c0 = dc.tile_frame(n0, p0, c0, (1, 1, 2))
        corr = dc.site_correspondence(n1, p1, c1, n0, p0, c0, perm=(0, 1, 2), r_match=2.0)
        assert corr["n_ghost"] == 1 and corr["n_added"] == 0 and corr["n_substituted"] == 0
        assert corr["n_matched"] == 79 and corr["max_displacement"] < 0.2
        assert int(n0[corr["ghosts"][0]]) == 17
        # The correspondence is a bijection between matched atoms, species-consistent.
        assert len(set(corr["match"].values())) == 79
        assert all(n1[i] == n0[j] for i, j in corr["match"].items())


class TestTier2Synthetic:
    """Toy Hamiltonians whose expected integers are defined by the toy alone."""

    def test_a_removal_gives_one_ghost_site_worth_of_gamma_on_both_paths(self):
        rng = np.random.default_rng(0)
        H0 = _toy(12, rng, n_deep=8)
        H1 = _embed_removal(H0, 5)
        groups = _groups(H0.shape[0], ghost=[5])
        rank = 32
        out = dc.tier2_continuation(H0, H1, groups, rank)
        assert out["accepted"] and out["m_vb"] == rank - dc.ORB
        gamma = np.asarray(out["gamma"])
        assert (gamma > 0.95).sum() == dc.ORB and (gamma < 0.05).sum() == rank - dc.ORB

    def test_charge_character_exchange_between_two_sites_leaves_the_count_unchanged(self):
        """(i) Two same-species sites swap their on-site levels along lambda: the occupied
        manifold is the same subspace at both ends, whichever site carries the charge."""
        rng = np.random.default_rng(1)
        H0 = _toy(10, rng, n_deep=5)
        H1 = H0.copy()
        a, b = np.arange(0, 4), np.arange(12, 16)
        H1[np.ix_(a, a)], H1[np.ix_(b, b)] = H0[np.ix_(b, b)], H0[np.ix_(a, a)]
        H1[np.ix_(a, a)] -= 1.5 * np.eye(4)
        H1[np.ix_(b, b)] += 1.5 * np.eye(4)
        groups = _groups(H0.shape[0])
        rank = 20
        out = dc.tier2_continuation(H0, H1, groups, rank)
        assert out["accepted"] and out["m_vb"] == rank
        assert all(r["min_separation"] > 0.5 for r in out["runs"].values())

    def test_levels_crossing_inside_the_valence_manifold_cost_nothing(self):
        """(ii) Two occupied levels exchange order along the path -- a crossing INSIDE the
        transported subspace -- and the count is unchanged with unit separation."""
        dim = 24
        levels0 = np.linspace(-10.0, -6.0, 12).tolist() + np.linspace(-2.0, 2.0, 12).tolist()
        H0 = np.diag(levels0)
        levels1 = list(levels0)
        levels1[2], levels1[9] = levels0[9], levels0[2]    # both occupied; they cross
        H1 = np.diag(levels1)
        # a small coupling so the crossing is avoided, not exact
        H0[2, 9] = H0[9, 2] = 0.05
        H1[2, 9] = H1[9, 2] = 0.05
        groups = _groups(dim)
        out = dc.tier2_continuation(H0, H1, groups, rank=12)
        assert out["accepted"] and out["m_vb"] == 12
        assert all(r["min_separation"] > 0.99 for r in out["runs"].values())

    def test_a_synthetic_interstitial_with_its_level_in_the_gap(self):
        """An added site whose occupied level lands in the gap: no ghost, the transported
        rank is the pristine one, so every electron the addition brings is frontier."""
        rng = np.random.default_rng(2)
        H0 = _toy(8, rng, n_deep=5)
        dim0 = H0.shape[0]
        # union: 8 pristine sites + 1 added site whose on-site block sits in the gap
        dim = dim0 + dc.ORB
        H0_u = np.zeros((dim, dim))
        H0_u[:dim0, :dim0] = H0
        H1_u = H0_u.copy()
        add = np.arange(dim0, dim)
        H1_u[np.ix_(add, add)] = np.eye(dc.ORB) * (-5.0)
        H1_u[np.ix_(add, np.arange(0, 4))] = 0.2
        H1_u[np.ix_(np.arange(0, 4), add)] = 0.2
        groups = _groups(dim, added=[8])
        rank = 20
        out = dc.tier2_continuation(H0_u, H1_u, groups, rank)
        assert out["accepted"] and out["m_vb"] == rank
        # The class's fill then decides the integers: e.g. two more electrons per spin.
        assert dc.class_integers((rank, rank), (rank + 2, rank + 2)) == ((2, 2), (0, 0), 4)

    def test_a_synthetic_substitution_keeps_the_site_physical_on_both_paths(self):
        rng = np.random.default_rng(3)
        H0 = _toy(10, rng, n_deep=6)
        H1 = H0.copy()
        s = np.arange(8, 12)
        H1[np.ix_(s, s)] = H0[np.ix_(s, s)] - 2.0 * np.eye(4)   # a deeper species
        H1[np.ix_(s, np.arange(4, 8))] *= 1.3                     # different hoppings
        H1[np.ix_(np.arange(4, 8), s)] *= 1.3
        groups = _groups(H0.shape[0], substituted=[2])
        rank = 24
        out = dc.tier2_continuation(H0, H1, groups, rank)
        assert out["accepted"] and out["m_vb"] == rank
        assert all(r["n_ghost"] == 0 for r in out["runs"].values())

    def test_a_genuine_valence_frontier_closure_raises_the_flag(self):
        """A substituted site's level rises from inside the valence manifold to above the
        frontier level while the two are coupled: on Path A (coupled while moving) the two
        step schedules or the two paths disagree, or a closure eigenvalue appears; either
        way the class carries no decomposition."""
        dim = 8
        H0 = np.diag([-9.0, -8.0, -7.0, -6.0, -1.0, 0.0, 1.0, 2.0])
        H1 = H0.copy()
        H1[0, 0] = 2.5          # site 0's s level (occupied) rises through the frontier
        H0[0, 4] = H0[4, 0] = 0.02
        H1[0, 4] = H1[4, 0] = 0.02
        H0[0, 5] = H0[5, 0] = 0.02
        H1[0, 5] = H1[5, 0] = 0.02
        groups = _groups(dim, substituted=[0])
        out = dc.tier2_continuation(H0, H1, groups, rank=4, dlambda=0.05)
        assert not out["accepted"]
        assert (not out["path_agreement"] or not out["schedule_agreement"]
                or out["closure"] or not out["contiguous"]), out

    def test_the_endpoint_classification_is_invariant_under_rotations_in_the_subspace(self):
        rng = np.random.default_rng(4)
        H0 = _toy(9, rng, n_deep=5)
        H1 = _embed_removal(H0, 4)
        groups = _groups(H0.shape[0], ghost=[4])
        psi, _ = dc.transport_valence_subspace(H0, H1, groups, 20, "A", 0.02, 100.0)
        base = dc.endpoint_classification(psi, groups["ghost"], 0.05)
        q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
        rotated = dc.endpoint_classification(psi @ q, groups["ghost"], 0.05)
        assert np.allclose(base["gamma"], rotated["gamma"], atol=1e-10)
        assert base["n_physical"] == rotated["n_physical"] == 16
        assert base["n_ghost"] == rotated["n_ghost"] == 4

    @pytest.mark.parametrize("scale", [4.0, 16.0])
    def test_the_integers_and_gamma_are_unchanged_by_the_sink_energy(self, scale):
        rng = np.random.default_rng(5)
        H0 = _toy(10, rng, n_deep=6)
        H1 = _embed_removal(H0, 3)
        groups = _groups(H0.shape[0], ghost=[3])
        a = dc.tier2_continuation(H0, H1, groups, 24, e_sink=100.0)
        b = dc.tier2_continuation(H0, H1, groups, 24, e_sink=100.0 * scale)
        assert a["accepted"] and b["accepted"] and a["m_vb"] == b["m_vb"] == 20
        assert np.allclose(a["gamma"], b["gamma"], atol=1e-8)

    def test_halving_dlambda_changes_nothing(self):
        rng = np.random.default_rng(6)
        H0 = _toy(10, rng, n_deep=6)
        H1 = _embed_removal(H0, 1)
        groups = _groups(H0.shape[0], ghost=[1])
        a = dc.tier2_continuation(H0, H1, groups, 24, dlambda=0.02)
        b = dc.tier2_continuation(H0, H1, groups, 24, dlambda=0.01)
        assert a["m_vb"] == b["m_vb"] == 20
        assert np.allclose(a["gamma"], b["gamma"], atol=1e-8)

    def test_both_paths_share_their_endpoints_exactly(self):
        rng = np.random.default_rng(7)
        H0 = _toy(6, rng, n_deep=3)
        H1 = _embed_removal(H0, 2)
        H1[np.ix_(np.arange(0, 4), np.arange(0, 4))] -= 0.7 * np.eye(4)   # a substitution too
        groups = _groups(H0.shape[0], ghost=[2], substituted=[0])
        for lam in (0.0, 1.0):
            a = dc.interpolated_hamiltonian(H0, H1, lam, "A", groups, 100.0)
            b = dc.interpolated_hamiltonian(H0, H1, lam, "B", groups, 100.0)
            assert np.array_equal(a, b)
        end = dc.interpolated_hamiltonian(H0, H1, 1.0, "A", groups, 100.0)
        ghost = groups["ghost"]
        assert np.allclose(end[ghost][:, ghost], 100.0 * np.eye(4))
        keep = np.setdiff1d(np.arange(H0.shape[0]), ghost)
        assert np.array_equal(end[np.ix_(keep, keep)], H1[np.ix_(keep, keep)])
        # ... and the paths differ in between.
        mid_a = dc.interpolated_hamiltonian(H0, H1, 0.5, "A", groups, 100.0)
        mid_b = dc.interpolated_hamiltonian(H0, H1, 0.5, "B", groups, 100.0)
        assert not np.allclose(mid_a, mid_b)


# ------------------------------------------------------------------ the fence


def _names(path: Path) -> set:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add(alias.name.split(".")[-1])
    return out


PRODUCTION_SAFE = {"frame_counts", "lookup_class", "ClassRecord", "composition_key",
                   "constructor_config", "DEFAULT_CONSTRUCTOR",
                   "defect_composition"}


class TestTheFence:
    def test_tier_2_names_appear_in_no_production_module(self):
        offenders = {}
        for path in (REPO / "mace").rglob("*.py"):
            if path.name == "defect_composition.py":
                continue
            hit = sorted(n for n in _names(path) if n in dc.TIER2_NAMES)
            if hit:
                offenders[str(path.relative_to(REPO))] = hit
        assert not offenders, offenders

    def test_production_modules_reach_only_the_per_frame_layer(self):
        """A forward path may look up a class and net the counters; it may not construct."""
        constructor_only = set(dc.__all__) - PRODUCTION_SAFE
        offenders = {}
        for name in ("defect_models.py", "defect_counting.py", "defect_terms.py",
                     "latent_ewald.py", "defect_madelung.py", "defect_cache.py"):
            path = REPO / "mace" / "modules" / name
            hit = sorted(n for n in _names(path) if n in constructor_only)
            if hit:
                offenders[name] = hit
        assert not offenders, offenders

    def test_the_trainer_builds_and_the_extractor_carries_the_table(self):
        assert "build_class_table" in _names(REPO / "mace" / "cli" / "run_train.py")
        # The extractor writes the table under a string key, so a text check is the right one.
        assert '"composition_classes"' in (REPO / "mace" / "tools" / "scripts_utils.py").read_text()
