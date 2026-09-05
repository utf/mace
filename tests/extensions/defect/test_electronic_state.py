"""Plan v8 section 2.1 and section 3: the electronic state interface and its dispatch.

What these pin, in order: the spec's identity is its PHYSICAL key and nothing else; the
adapter from the loader's counters forms the integers the count fill always formed; the null
is equality to `S_ref`'s key rather than a zero counter; `count_fill` is dispatched to the
legacy implementation and reports the fills it used; and nothing but `count_fill` can be
configured into a model, so a test-only policy registered here cannot reach a forward.
"""

from __future__ import annotations

import ast
import io
import pickle
from pathlib import Path

import pytest
import torch

from mace.modules import defect_state as ds
from mace.modules.defect_counting import (ORBITALS_PER_ATOM, CountingHead,
                                          density_matrix, head_energy_hf,
                                          batched_head_energy_hf)
from mace.modules.defect_models import MACEDefect
from tests.extensions.defect.test_neutral_reference_skip import (_batch, _gapped, _model,
                                                                  _perovskite)

torch.set_default_dtype(torch.float64)


class TestSpec:
    def test_the_physical_key_excludes_the_state_id(self):
        a = ds.ElectronicStateSpec(1, (-1, 0), state_id="frame 12")
        b = ds.ElectronicStateSpec(1, (-1, 0), state_id="something else")
        assert a.physical_key() == b.physical_key()
        assert a.key_digest() == b.key_digest()
        assert a != b, "the dataclass still distinguishes them; only the KEY does not"

    def test_a_different_policy_is_a_different_state_at_the_same_counts(self):
        a = ds.ElectronicStateSpec(0, (0, 0))
        b = ds.ElectronicStateSpec(0, (0, 0), occupation_policy="mock",
                                   occupation_payload=(("f", (1.0, 0.0)),))
        assert a.physical_key() != b.physical_key()
        assert not b.is_reference(ds.reference_state())

    def test_integers_are_enforced_and_count_fill_takes_no_payload(self):
        with pytest.raises(ValueError, match="exact integers"):
            ds.ElectronicStateSpec(0.5, (0, 0))
        with pytest.raises(ValueError, match="no occupation payload"):
            ds.ElectronicStateSpec(0, (0, 0), occupation_payload={"f": [1.0]})
        with pytest.raises(ValueError, match="delta_n must be"):
            ds.ElectronicStateSpec(0, (0,))

    def test_the_charge_count_identity_is_checked_against_the_reference(self):
        ref = ds.reference_state()
        ds.ElectronicStateSpec(1, (-1, 0)).check_against(ref)       # V_Cl+
        ds.ElectronicStateSpec(-1, (1, 0)).check_against(ref)       # an added electron
        with pytest.raises(ValueError, match="Q_formal - Q_ref"):
            ds.ElectronicStateSpec(1, (0, 0)).check_against(ref)
        with pytest.raises(ValueError, match="delta_n = \\(0, 0\\)"):
            ds.ElectronicStateSpec(1, (-1, 0)).check_against(ds.ElectronicStateSpec(1, (-1, 0)))

    def test_it_round_trips_through_a_dict_and_json(self):
        s = ds.ElectronicStateSpec(1, (-1, 0), state_id="x")
        back = ds.ElectronicStateSpec.from_dict(s.to_dict())
        assert back == s
        assert "state_id" in s.to_json()

    def test_a_foreign_schema_is_refused(self):
        with pytest.raises(ValueError, match="schema"):
            ds.ElectronicStateSpec(0, (0, 0), schema_version=ds.STATE_SCHEMA_VERSION + 1)


class TestAdapter:
    def test_the_benchmark_states(self):
        """V_Cl0 is S_ref with q_F = -1 in section 2.1's language; V_Cl+ is Q = +1 with the
        majority channel one electron short. The adapter must land both."""
        counts = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        sb = ds.StateBatch.from_counts(counts)
        assert sb.q_formal.tolist() == [0, 1]
        assert sb.delta_n.tolist() == [[0, 0], [-1, 0]]
        assert sb.is_reference(ds.reference_state()).tolist() == [True, False]
        sb.check_against(ds.reference_state())

    def test_the_null_is_the_physical_key_not_a_zero_counter(self):
        """(1, 0, 1, 0) is an electron and a hole in the same channel: the same fill as the
        reference, so the same state, though its counter is not zero."""
        sb = ds.StateBatch.from_counts(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
        assert sb.spec(0).physical_key() == ds.reference_state().physical_key()
        assert sb.all_reference(ds.reference_state())

    def test_a_batch_under_another_policy_is_never_at_the_reference(self):
        sb = ds.StateBatch.from_counts(torch.zeros(3, 4), policy="mock")
        assert not sb.is_reference(ds.reference_state()).any()

    def test_the_counter_shape_is_checked(self):
        with pytest.raises(ValueError, match=r"\[B, 4\]"):
            ds.StateBatch.from_counts(torch.zeros(2, 3))


class _Mock(ds.OccupationPolicy):
    key = "mock_for_registry_test"
    version = 7


class TestDispatch:
    def test_count_fill_dispatches_to_the_legacy_implementation(self):
        p = ds.dispatch(ds.COUNT_FILL)
        assert isinstance(p, ds.CountFillPolicy)
        assert ds.policy_version(ds.COUNT_FILL) == 1

    def test_an_unknown_key_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError, match="no occupation policy"):
            ds.dispatch("aufbau_by_label")

    def test_a_test_policy_is_dispatchable_but_not_configurable(self):
        ds.register_test_policy(_Mock())
        assert isinstance(ds.dispatch(_Mock.key), _Mock)
        with pytest.raises(ValueError, match="not a production policy"):
            _model(occupation_policy=_Mock.key)
        with pytest.raises(ValueError, match="not a production policy"):
            CountingHead(num_elements=3, feature_dim=4, atomic_numbers=[17, 55, 82],
                         occupation_policy=_Mock.key)

    def test_the_production_key_cannot_be_replaced(self):
        class Impostor(ds.OccupationPolicy):
            key = ds.COUNT_FILL

        with pytest.raises(ValueError, match="production policy"):
            ds.register_test_policy(Impostor())
        assert isinstance(ds.dispatch(ds.COUNT_FILL), ds.CountFillPolicy)

    def test_no_production_module_registers_a_policy(self):
        """The registry's test entry point appears as code in no module under `mace/`
        except the one that defines it. An AST walk, as for the protocol module."""
        root = Path(__file__).resolve().parents[3] / "mace"
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "defect_state.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                name = (node.id if isinstance(node, ast.Name)
                        else node.attr if isinstance(node, ast.Attribute) else None)
                if name == "register_test_policy":
                    offenders.append(str(path.relative_to(root)))
                    break
        assert not offenders, offenders


class TestTheFillsAreReported:
    @staticmethod
    def h(seed, n_sites=5):
        g = torch.Generator().manual_seed(seed)
        dim = n_sites * ORBITALS_PER_ATOM
        m = torch.randn(dim, dim, generator=g)
        return 0.5 * (m + m.T)

    def test_the_loop_solver_reports_the_mu_and_occupations_it_used(self):
        H = self.h(0)
        got = {}
        e, lam, psi, p_now, p_ref = head_energy_hf(H, 20, (0, 0, 1, 0), 0.05, internals=got)
        assert got["mu"].shape == (4,) and got["occupations"].shape == (4, 20)
        # 20 electrons: 10/10 at the reference; a majority hole takes one from the first.
        assert got["fills"] == (9.0, 10.0, 10.0, 10.0)
        f = got["occupations"]
        # P is exactly what those occupations build on the reported spectrum.
        rebuilt = density_matrix(psi, f[0]) + density_matrix(psi, f[1])
        assert torch.equal(rebuilt, p_now)
        assert torch.equal(density_matrix(psi, f[2]) + density_matrix(psi, f[3]), p_ref)

    def test_the_batched_solver_reports_them_too(self):
        H = torch.stack([self.h(1), self.h(2)])
        got = {}
        counts = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        e, lam, D, refs = batched_head_energy_hf(H, torch.tensor([20.0, 20.0]), counts,
                                                 0.05, internals=got)
        assert got["mu"].shape == (2, 4) and got["occupations"].shape == (2, 4, 20)
        assert torch.equal(got["fills"], torch.tensor([[9.0, 10.0, 10.0, 10.0],
                                                       [10.0, 10.0, 10.0, 10.0]]))
        # The neutral graph's D is exactly zero: the same fills subtract to nothing.
        assert torch.equal(D[1], torch.zeros_like(D[1]))

    def test_the_head_collects_them_under_internals(self):
        frames = [_perovskite(seed=1), _perovskite(seed=2)]
        model = _gapped(frames[:1])       # a charged graph needs the class table (1.2)
        batch = _batch(frames, [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        grabbed = {}
        head = model.spectral
        original = head.forward

        def wrapped(*a, **k):
            k["internals"] = grabbed
            return original(*a, **k)

        head.forward = wrapped
        try:
            with torch.no_grad():
                model(batch.to_dict(), training=False, compute_force=False)
        finally:
            head.forward = original
        fills = grabbed["fills"]
        assert len(fills) == 1 and fills[0]["mu"].shape == (2, 4)


class TestTheOverrideIsGone:
    def test_the_head_refuses_a_stated_fill(self):
        head = CountingHead(num_elements=3, feature_dim=4, atomic_numbers=[17, 55, 82])
        with pytest.raises(ValueError, match="alternate occupation policy"):
            head(node_feats=torch.zeros(2, 4), counter_emb=None, counts=torch.zeros(1, 4),
                 batch=torch.zeros(2, dtype=torch.long), num_graphs=1,
                 edge_index=torch.zeros(2, 0, dtype=torch.long), edge_length=None,
                 node_species=torch.zeros(2, dtype=torch.long),
                 edge_vector=torch.zeros(0, 3), occupations=torch.tensor([[10.0, 10.0]]))

    def test_the_model_refuses_it_in_the_data_dict(self):
        model = _model()
        batch = _batch([_perovskite(seed=3)], [[0.0, 0.0, 1.0, 0.0]])
        d = batch.to_dict()
        d["occupations"] = torch.tensor([[200.0, 200.0]])
        with pytest.raises(ValueError, match="occupation override"):
            model(d, training=False, compute_force=False)


class TestModelSemantics:
    def test_the_reference_is_config_and_survives_extraction_and_pickling(self):
        from mace.tools.scripts_utils import extract_config_mace_model

        model = _model()
        assert model.occupation_policy == ds.COUNT_FILL
        assert ds.ElectronicStateSpec.from_dict(model.reference_state) == ds.reference_state()
        config = extract_config_mace_model(model)
        assert config["occupation_policy"] == ds.COUNT_FILL
        assert config["reference_state"] == model.reference_state
        rebuilt = MACEDefect(**config)
        assert rebuilt.reference_state == model.reference_state
        assert rebuilt.spectral.reference_state == model.reference_state

    def test_a_legacy_pickle_without_the_attributes_loads_with_the_defaults(self):
        model = _model()
        del model.__dict__["occupation_policy"]
        del model.__dict__["reference_state"]
        del model.spectral.__dict__["occupation_policy"]
        del model.spectral.__dict__["reference_state"]
        buf = io.BytesIO()
        pickle.dump(model, buf)
        back = pickle.loads(buf.getvalue())
        assert back.occupation_policy == ds.COUNT_FILL
        assert ds.ElectronicStateSpec.from_dict(back.reference_state) == ds.reference_state()
        assert back.spectral.occupation_policy == ds.COUNT_FILL   # the class attribute
        assert back.spectral.s_ref() == ds.reference_state()

    def test_a_reference_state_that_is_not_a_reference_is_refused(self):
        bad = ds.ElectronicStateSpec(1, (-1, 0)).to_dict()
        with pytest.raises(ValueError, match="not a valid S_ref"):
            _model(reference_state=bad)

    def test_the_head_is_null_on_the_physical_key_not_the_counter(self):
        """(1, 0, 1, 0) is S_ref by key. The correction is exactly zero there, and the
        reference skip fires for it, so the forward equals the zero-counter forward in
        every head quantity."""
        model = _model()
        frames = [_perovskite(seed=5)]
        d0 = _batch(frames, [[0.0, 0.0, 0.0, 0.0]]).to_dict()
        d1 = _batch(frames, [[1.0, 0.0, 1.0, 0.0]]).to_dict()
        out0 = model(d0, training=False, compute_force=True)
        out1 = model(d1, training=False, compute_force=True)
        assert torch.equal(out1["delta_sr_energy"], torch.zeros(1))
        for key in ("energy", "forces", "correction_energy", "carrier_alpha"):
            assert torch.equal(out0[key], out1[key]), key

    def test_the_skip_is_decided_on_the_reference_state_key(self):
        """A supplied reference counter that IS S_ref by key -- (1, 0, 1, 0) -- still lets
        the skip fire; one that is not -- (1, 0, 0, 0) -- does not."""
        frames = [_perovskite(seed=7)]
        model = _gapped(frames)
        base = _batch(frames, [[0.0, 0.0, 1.0, 0.0]])
        d_same = base.to_dict()
        d_same["carrier_counts_ref"] = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        d_other = base.to_dict()
        d_other["carrier_counts_ref"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            same = model(d_same, training=False, compute_force=False)
            other = model(d_other, training=False, compute_force=False)
        # Under the skip the reference correction is an exact zero, so the paired
        # difference IS the correction at this frame's own state.
        assert torch.equal(same["delta_energy"], same["correction_energy"])
        assert float((other["delta_energy"] - other["correction_energy"]).abs().max()) > 0.0
