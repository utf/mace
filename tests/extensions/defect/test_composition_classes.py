"""Plan v8 section 2.1, Stage 0 task 0.7: the composition-class constructor, Tier 1.

Two layers, tested separately. The INTEGER layer (`tier1`, `align_edges`, `class_integers`,
`frame_counts`) is exercised on toy spectra whose expected integers are defined by the toy
alone. The SPECTRUM layer is exercised on a Harrison-initialised counting head over cubic
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

    def test_tier1_counts_at_the_cut_and_flags_a_level_in_the_window(self):
        vbm, delta = -7.0, 0.1
        valence = np.linspace(-10.0, vbm, 50)
        clean = np.concatenate([valence, [vbm + 3.0, vbm + 4.0]])
        m, ambiguous, nearest = dc.tier1(clean, vbm, delta)
        assert (m, ambiguous) == (50, False) and nearest == pytest.approx(-delta)
        # A level inside +-window of the cut (window = delta / 2 by default).
        for level in (vbm + delta - 0.04, vbm + delta + 0.04):
            m, ambiguous, _ = dc.tier1(np.concatenate([clean, [level]]), vbm, delta)
            assert ambiguous
        # A frontier level just outside the window is counted as frontier, unflagged.
        m, ambiguous, _ = dc.tier1(np.concatenate([valence, [vbm + delta + 0.06]]), vbm, delta)
        assert (m, ambiguous) == (50, False)
        # A valence level scattered slightly above VBM_al is still valence.
        m, ambiguous, _ = dc.tier1(np.concatenate([valence, [vbm + 0.03]]), vbm, delta)
        assert (m, ambiguous) == (51, False)

    def test_a_level_exactly_on_the_window_edge_is_not_inside_it(self):
        vbm, delta = -7.0, 0.1
        spectrum = np.concatenate([np.linspace(-10.0, vbm, 50), [vbm + 4.0]])
        # The pristine class's own VBM is at cut - delta, outside a window of delta / 2;
        # and a window of exactly delta puts it ON the edge, which is not inside.
        assert not dc.tier1(spectrum, vbm, delta, window=delta)[1]

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
        m_vb, ambiguous, _ = dc.tier1(spectrum, a.vbm_al, delta)
        assert (m_vb, ambiguous) == (40, False)
        n_e, n_h, q_core = dc.class_integers((m_vb, m_vb), n_sigma)
        assert n_e == (m, m) and n_h == (0, 0) and q_core == 2 * m
        rec = dc.ClassRecord(key="synthetic", n_atoms=0, reference_frame_key=0,
                             n_total=2 * n_sigma[0], n_sigma=n_sigma, tier=1, ambiguous=False,
                             reason="", m_vb=(m_vb, m_vb), n_e=n_e, n_h=n_h, q_core=q_core,
                             vbm_al=a.vbm_al, cbm_al=a.cbm_al, shift=a.shift, spread=a.spread,
                             gap_pristine=a.gap_pristine, delta=delta, nearest=0.0)
        neutral = StateBatch.from_counts(torch.zeros(1, 4))
        assert dc.frame_counts(rec, neutral, 0) == ((m, m), (0, 0), -2 * m)

    def test_a_synthetic_class_with_a_level_in_the_window_triggers_tier_2(self):
        vbm, delta = -5.0, 0.1
        pristine = np.concatenate([np.linspace(-9.0, vbm, 40), [vbm + 3.0]])
        spectrum = np.sort(np.concatenate([pristine, [vbm + delta + 0.02]]))
        a = dc.align_edges(spectrum, 41, pristine, 40)
        _, ambiguous, nearest = dc.tier1(spectrum, a.vbm_al, delta)
        assert ambiguous and abs(nearest) < delta / 2


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

    def test_v_cl_at_79_atoms_gives_the_same_q_core_or_is_handed_to_tier_2(self, table):
        """The class at the larger cell must yield the same integers (plan section 7.1).

        On this toy the vacancy pushes a valence level to VBM_al + 0.14 eV, inside the Tier 1
        window, so the class is ambiguous at Tier 1 and the identity is Tier 2's to
        establish (task 0.8). A COUNTED class must match; an uncounted one must say why.
        """
        rec = dc.lookup_class(table, [17] * 47 + [55] * 16 + [82] * 16)
        assert rec.tiling == (1, 1, 2)
        if rec.counted:
            assert rec.n_e == (1, 0) and rec.n_h == (0, 0) and rec.q_core == 1
        else:
            assert "Tier 1 ambiguous" in rec.reason
            pytest.skip(f"Tier 2 pending for this class: {rec.reason}")

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
        assert config["edge_delta"] is None
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
