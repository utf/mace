"""Plan v8 section 3, Stage 0 task 0.10: config round-trip of every key the section lists.

    "Config round-trip covers r_split, r_orb, r_res, ewald_sigma, delta, Delta_s, r_match,
    E_sink, eta, dlambda, w parameters, a/b bounds, gauge flag, SCF tolerances, C_Q mode,
    ElectronicStateSpec schema version, occupation-policy key and payload, S_ref, and the
    cached class integers with their tier, path/schedule agreement and gamma spectrum; every
    non-parameter float serialised."

Each key is named here verbatim, located in the extracted config, checked to be a NUMBER
(or the structured object the plan names, with numbers inside), and checked to survive
`extract_config_mace_model -> MACEDefect(**config)` and pickling unchanged.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import torch

from mace.modules import defect_composition as dc
from mace.modules.defect_models import MACEDefect
from mace.modules.defect_state import STATE_SCHEMA_VERSION
from mace.tools.scripts_utils import extract_config_mace_model
from tests.extensions.defect.test_composition_classes import (frames, harrison_model,
                                                              table)  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _f64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def configured(harrison_model, table):
    """The model as the trainer leaves it: class table built, sink resolved."""
    model = copy.deepcopy(harrison_model)
    model.composition_classes = copy.deepcopy(table)
    model.class_constructor["e_sink"] = table["e_sink"]
    return model


# (plan name, where it lives in the extracted config, the kind of value it must be)
SECTION_3_KEYS = [
    ("r_split", ("functional", "r_split"), float),
    ("r_orb", ("functional", "r_orb"), dict),
    ("r_res", ("functional", "r_res"), float),
    ("ewald_sigma", ("les_arguments", "sigma"), float),
    ("delta", ("functional", "delta"), float),
    ("Delta_s", ("functional", "delta_s"), float),
    ("r_match", ("class_constructor", "r_match"), float),
    ("E_sink", ("class_constructor", "e_sink"), float),
    ("eta", ("class_constructor", "eta"), float),
    ("dlambda", ("class_constructor", "dlambda"), float),
    ("w parameters: p*", ("functional", "p_star"), float),
    ("w parameters: Delta_p", ("functional", "delta_p"), float),
    ("a bounds", ("functional", "a_bounds"), list),
    ("b bounds", ("functional", "b_bounds"), list),
    ("gauge flag", ("gauge",), str),
    ("SCF tolerance eps_P", ("functional", "eps_p"), float),
    ("SCF tolerance eps_H", ("functional", "eps_h"), float),
    ("C_Q mode", ("functional", "c_q_mode"), str),
    ("ElectronicStateSpec schema version", ("reference_state", "schema_version"), int),
    ("occupation-policy key", ("occupation_policy",), str),
    ("occupation-policy payload", ("reference_state", "occupation_payload"), (dict, type(None))),
    ("S_ref", ("reference_state",), dict),
    ("cached class integers", ("composition_classes", "classes"), dict),
]


def _dig(config, path):
    value = config
    for key in path:
        assert key in value, f"{path}: {key!r} absent from {sorted(value)}"
        value = value[key]
    return value


def _no_none_floats(obj, where):
    """Every leaf is a number, string, bool or None-free container: 'every non-parameter
    float serialised' means no placeholder survives extraction."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _no_none_floats(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _no_none_floats(v, f"{where}[{i}]")
    else:
        assert obj is not None, f"{where} is None: a float left unresolved"


class TestSection3RoundTrip:
    @pytest.mark.parametrize("name,path,kind", SECTION_3_KEYS, ids=[k[0] for k in SECTION_3_KEYS])
    def test_the_key_is_present_typed_and_survives_the_round_trip(self, configured, name, path,
                                                                  kind):
        config = extract_config_mace_model(configured)
        value = _dig(config, path)
        assert isinstance(value, kind), f"{name}: {type(value).__name__}"
        rebuilt = MACEDefect(**config)
        again = extract_config_mace_model(rebuilt)
        assert _dig(again, path) == value, name
        pickled = extract_config_mace_model(pickle.loads(pickle.dumps(configured)))
        assert _dig(pickled, path) == value, name

    def test_every_non_parameter_float_is_serialised_as_a_number(self, configured):
        config = extract_config_mace_model(configured)
        for section in ("functional", "class_constructor"):
            _no_none_floats(config[section], section)
        assert config["class_constructor"]["e_sink"] == pytest.approx(
            dc.lookup_class(config["composition_classes"],
                            [17] * 24 + [55] * 8 + [82] * 8).cbm_al + 50.0)

    def test_the_schema_version_and_policy_are_the_programme_s(self, configured):
        config = extract_config_mace_model(configured)
        assert config["reference_state"]["schema_version"] == STATE_SCHEMA_VERSION
        assert config["occupation_policy"] == "count_fill"
        assert config["reference_state"]["occupation_policy"] == "count_fill"
        assert not config["reference_state"]["occupation_payload"]
        assert config["reference_state"]["delta_n"] == [0, 0]
        assert config["reference_state"]["q_formal"] == 0

    def test_the_class_integers_carry_tier_agreement_and_gamma(self, configured):
        config = extract_config_mace_model(configured)
        classes = config["composition_classes"]["classes"]
        by_tier = {r["tier"] for r in classes.values()}
        assert by_tier == {1, 2}, by_tier
        for key, r in classes.items():
            for f in ("tier", "m_vb", "n_e", "n_h", "q_core", "path_agreement",
                      "schedule_agreement", "gamma", "placement", "tiling", "perm"):
                assert f in r, (key, f)
            if r["tier"] == 2:
                assert r["path_agreement"] is True and r["schedule_agreement"] is True
                assert len(r["gamma"]) == r["m_vb"][0] + 4
        rebuilt = MACEDefect(**config)
        assert rebuilt.composition_classes == config["composition_classes"]

    def test_an_unknown_functional_or_constructor_key_is_refused(self):
        from mace.modules.defect_density import functional_config

        with pytest.raises(ValueError, match="unknown functional parameter"):
            functional_config({"r_splat": 1.0})
        with pytest.raises(ValueError, match="unknown class-constructor parameter"):
            dc.constructor_config({"E_sink": 1.0})

    def test_explicit_values_override_the_resolved_defaults(self, harrison_model):
        config = extract_config_mace_model(harrison_model)
        config["functional"] = dict(config["functional"], r_res=0.7, delta=0.2, p_star=0.4)
        config["class_constructor"] = dict(config["class_constructor"], r_match=1.4, eta=0.01)
        model = MACEDefect(**config)
        assert model.functional["r_res"] == 0.7 and model.functional["delta"] == 0.2
        assert model.functional["p_star"] == 0.4
        assert model.class_constructor["r_match"] == 1.4 and model.class_constructor["eta"] == 0.01
        again = extract_config_mace_model(model)
        assert again["functional"] == model.functional
        assert again["class_constructor"] == model.class_constructor
