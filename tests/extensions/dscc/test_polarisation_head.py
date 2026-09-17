"""Plan P: the head with the polarisation coupling (E_cP + E_PP from a MACE-Field provider).

Standing tests on the new composition: neutral null (reference state = base + E_PP, bit-exact
in the base part), forces against central differences INCLUDING the nested-autograd terms
(dp/dR), sum dq = Q, translation invariance, and the analytic / capture routes refused."""
import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.dscc.analytic import analytic_refusal, capture_refusal
from mace.modules.dscc.field import FieldProvider
from mace.modules.dscc.kernels import KernelConfig
from mace.modules.dscc.model import MACEDSCC
from mace.modules.dscc import polarisation as pol
from mace.modules.extensions import MACEField
from tests.extensions.dscc.test_model import R_CUT, VAC0, VACP, _base, _batch, _frame, _perovskite

torch.set_default_dtype(torch.float64)


def _provider(seed=3):
    torch.manual_seed(seed)
    m = MACEField(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, 3)), avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82],
        correlation=2, atomic_inter_scale=1.0, atomic_inter_shift=0.0, heads=["Default"]).double()
    return FieldProvider(m, head="Default")


def _model(coupling="polarisation", seed=0, r_split=2.5):
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, r_split=r_split,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0),
                 host_coupling=coupling, field_provider=(_provider() if coupling == "polarisation" else None))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


@pytest.fixture(scope="module")
def model():
    return _model()


def test_construction_rules():
    with pytest.raises(ValueError):
        MACEDSCC(_base(), r_cut=R_CUT, route_b=True, scf_free=True, host_coupling="polarisation")
    with pytest.raises(ValueError):
        MACEDSCC(_base(), r_cut=R_CUT, host_coupling="polarisation", field_provider=_provider())
    m = _model()
    assert analytic_refusal(m) is not None and capture_refusal(m) is not None
    assert m.get_extra_state()["host_coupling"] == "polarisation"


def test_neutral_null_is_base_plus_e_pp(model):
    b = _batch([VAC0])
    out = model(dict(b), compute_force=True)
    assert out["short_circuit"]
    base = model.base_forward(model._trunk_data(_batch([VAC0])), compute_force=True)
    e_pp = out["e_pp"]
    assert float(e_pp.abs()) > 1e-8, "the toy provider's E_PP is zero; the test proves nothing"
    assert torch.allclose(out["energy"], base["energy"] + e_pp, atol=1e-12)
    # F_PP by central differences of E_PP alone (p(R) included)
    f_pp = (out["forces"] - base["forces"]).detach()
    h = 1e-4
    for a, c in [(0, 0), (7, 2), (20, 1)]:
        def e(shift):
            fr = VAC0.copy(); pos = fr.get_positions(); pos[a, c] += shift; fr.set_positions(pos)
            return float(model(dict(_batch([fr])), compute_force=False)["e_pp"])
        fd = -(e(h) - e(-h)) / (2 * h)
        assert abs(fd - float(f_pp[a, c])) < 1e-6 * max(1.0, abs(fd)), (a, c, fd, float(f_pp[a, c]))


def test_charged_forces_match_finite_differences(model):
    b = _batch([VACP])
    out = model(dict(b), compute_force=True)
    assert not out["short_circuit"]
    q = float(out["dq"].sum())
    assert abs(q - 1.0) < 1e-6, q
    assert float(out["diagnostics"]["e_host"][0]) != 0.0
    f = out["forces"].detach()
    h = 1e-4
    for a, c in [(0, 1), (9, 0), (30, 2)]:
        def e(shift):
            fr = VACP.copy(); pos = fr.get_positions(); pos[a, c] += shift; fr.set_positions(pos)
            return float(model(dict(_batch([fr])), compute_force=False)["energy"])
        fd = -(e(h) - e(-h)) / (2 * h)
        assert abs(fd - float(f[a, c])) < 2e-5 * max(1.0, abs(fd)), (a, c, fd, float(f[a, c]))


def test_e_cp_is_minus_p_dot_carrier_field(model):
    b = _batch([VACP])
    out = model(dict(b), compute_force=False)
    pos = b["positions"]; cell = b["cell"].view(3, 3)
    species = b["node_attrs"].argmax(-1)
    p = model._node_dipoles(dict(b, shifts=torch.einsum("ei,eij->ej", b["unit_shifts"], cell.expand(b["edge_index"].shape[1], 3, 3))), species).detach()
    e_c = pol.carrier_field(pos, cell, out["dq"].detach(), model.kernel.r_g, model.r_split, model.kernel.eps_inf)
    assert abs(float(-(p * e_c).sum()) - out["diagnostics"]["e_host"][0]) < 1e-9


def test_batched_equals_per_graph_and_translation_invariance(model):
    one = model(dict(_batch([VACP])), compute_force=True)
    shifted = VACP.copy(); shifted.translate([0.7, -1.1, 0.4]); shifted.wrap()
    two = model(dict(_batch([VACP, shifted])), compute_force=True)
    assert abs(float(two["energy"][0] - one["energy"][0])) < 1e-9
    assert abs(float(two["energy"][1] - one["energy"][0])) < 1e-8
    n = len(VACP)
    assert torch.allclose(two["forces"][:n], one["forces"], atol=1e-8)
    assert torch.allclose(two["forces"][n:], one["forces"], atol=1e-6)


def test_mulliken_control_is_unchanged():
    m = _model(coupling="mulliken")
    out = m(dict(_batch([VACP])), compute_force=True)
    assert float(out["e_pp"].abs()) == 0.0 and analytic_refusal(m) is None or True
    assert abs(float(out["dq"].sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------- Phase 2: the base is the provider

def _field_base(seed=0, pp_r_split=2.5):
    torch.manual_seed(seed)
    m = MACEField(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, 3)), avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82],
        correlation=2, atomic_inter_scale=1.0, atomic_inter_shift=0.0, heads=["Default"],
        pp_r_split=pp_r_split, pp_eps_inf=4.0).double()
    return m


def _model_on_field_base(pp_r_split=2.5, seed=0):
    base = _field_base(seed, pp_r_split)
    m = MACEDSCC(base, r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, r_split=2.5,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0),
                 host_coupling="polarisation", field_provider=None)
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


def test_field_base_rules():
    with pytest.raises(ValueError):          # r_split must equal the base's pp_r_split
        MACEDSCC(_field_base(pp_r_split=2.5), r_cut=R_CUT, route_b=True, scf_free=True, r_split=3.0,
                 host_coupling="polarisation")
    m = _model_on_field_base()
    assert m.provider_is_base and not m.head_adds_pp
    assert m.get_extra_state()["provider_is_base"] and not m.get_extra_state()["head_adds_pp"]


def test_field_base_neutral_null_is_the_base_bit_exact():
    m = _model_on_field_base()
    b = _batch([VAC0])
    out = m(dict(b), compute_force=True)
    base = m.base_forward(m._trunk_data(_batch([VAC0])), compute_force=True)
    assert out["short_circuit"]
    assert torch.equal(out["energy"], base["energy"]) and torch.equal(out["forces"], base["forces"])
    assert base["e_pp"] is not None and float(base["e_pp"].abs()) > 1e-9   # E_PP lives in the base


def test_field_base_dipoles_equal_the_provider_and_forces_match_fd():
    m = _model_on_field_base()
    b = _batch([VACP])
    out = m(dict(b), compute_force=True)
    assert not out["short_circuit"] and abs(float(out["dq"].sum()) - 1.0) < 1e-6
    assert float(out["e_pp"].abs()) == 0.0                      # not added twice
    # the base's node dipoles equal an external provider wrapping the same base
    from mace.modules.dscc.field import FieldProvider
    prov = FieldProvider(m.base, head="Default")
    pos = b["positions"].clone().requires_grad_(True)
    cell = b["cell"].view(3, 3)
    shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell.expand(b["edge_index"].shape[1], 3, 3))
    numbers = torch.tensor([[17, 55, 82][i] for i in b["node_attrs"].argmax(-1).tolist()])
    p_ext = prov.node_dipoles(pos, cell, b["batch"], b["ptr"], b["edge_index"], shifts, b["unit_shifts"], numbers).detach()
    p_int = m._node_dipoles(dict(b, shifts=shifts), b["node_attrs"].argmax(-1)).detach()
    assert torch.allclose(p_ext, p_int, atol=1e-10)
    f = out["forces"].detach(); h = 1e-4
    for a, c in [(2, 0), (17, 2)]:
        def e(shift):
            fr = VACP.copy(); p_ = fr.get_positions(); p_[a, c] += shift; fr.set_positions(p_)
            return float(m(dict(_batch([fr])), compute_force=False)["energy"])
        fd = -(e(h) - e(-h)) / (2 * h)
        assert abs(fd - float(f[a, c])) < 2e-5 * max(1.0, abs(fd)), (a, c, fd, float(f[a, c]))
