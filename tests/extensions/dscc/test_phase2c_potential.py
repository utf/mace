"""Plan P Phase 2c, arm B_w: the near-field host potential on the diagonal of H0.

Plan §1 tests before training: neutral null bit-exact; s = 0 reproduces B_0 to 1e-9; the identity
d(ΔF_band)/ds = sum_c dq_c W_c / s to 1e-6 relative; central-difference forces to 1e-4 eV/A on a
charged frame; sum dq = Q; the gauge shift (a constant a added to the potential moves ΔF_band by
exactly a Q and nothing else); batched = per-graph; and the readout instrument
(`filled_hamiltonians`) returns the Hamiltonian the forward filled."""
import copy
import importlib.util, math, os
import pytest
import torch

from mace.modules.dscc import polarisation as pol
from mace.modules.dscc.precondition import PreconditionConfig, bound_state_precondition, hole_distribution
from mace.modules.dscc.scf import two_fillings
from tests.extensions.dscc.test_model import VAC0, VACP, _batch

_spec = importlib.util.spec_from_file_location("_pol_helpers", os.path.join(os.path.dirname(__file__), "test_polarisation_head.py"))
_pol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pol)

torch.set_default_dtype(torch.float64)
R_C, DELTA = 3.0, 1.0            # the toy cell's head cutoff is small: the ramp must sit inside it
S_INIT = 0.4                     # B_s's trained s is 0.33-0.56 on w01


def _arm(model, near_field, s=S_INIT, r_c=R_C, delta=DELTA):
    m = copy.deepcopy(model)
    m.near_field, m.r_c, m.delta = near_field, float(r_c), float(delta)
    with torch.no_grad():
        m.s_near_raw.fill_(math.atanh(s / m.s_max))
    return m


def _band(out, g=0):
    """The two-fillings band energy of graph g: head_energy minus E_host (= E_far here) and E_M."""
    return float(out["head_energy"][g]) - out["diagnostics"]["e_host"][g] - out["diagnostics"]["e_madelung"][g]


@pytest.fixture(scope="module")
def arm_b():
    return _pol._model()                      # arm B: polarisation coupling, near_field "off"


def test_construction_and_state(arm_b):
    m = _arm(arm_b, "potential")
    assert abs(float(m.near_scale()) - S_INIT) < 1e-12
    st = m.get_extra_state()
    assert st["near_field"] == "potential"
    m2 = copy.deepcopy(arm_b); m2.set_extra_state(st)
    assert m2.near_field == "potential"
    with pytest.raises(ValueError):
        _pol.MACEDSCC(_pol._base(), r_cut=_pol.R_CUT, directional=True, hidden=16, coupling=False, route_b=True, scf_free=True,
                      kernel=_pol.KernelConfig(regime="B", r_g=1.0, r_s=5.0), host_coupling="polarisation",
                      field_provider=_pol._provider(), near_field="potential", s_init=2.5)


def test_potential_matches_the_near_field_energy():
    """sum_c dq_c W_c is switched_terms' e_near on the same edges, kernel and switch (the identity
    the plan's 'nothing counted twice' rests on), for an arbitrary dq and s q0."""
    torch.manual_seed(3)
    n = 12
    pos = torch.rand(n, 3) * 6.0
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    ei = torch.tensor(pairs).T
    ev = pos[ei[1]] - pos[ei[0]]
    dq, sq0, p = torch.randn(n), torch.randn(n), torch.zeros(n, 3)
    _, e_near = pol.switched_terms(dq, p, ei, ev, 1.0, 2.5, 4.0, R_C, DELTA, sq0=sq0)
    W = pol.near_field_potential(sq0, ei, ev, 1.0, 4.0, R_C, DELTA)
    assert abs(float((dq * W).sum()) - float(e_near)) < 1e-12


def test_neutral_null_is_base_plus_e_pp(arm_b):
    m = _arm(arm_b, "potential")
    out = m(dict(_batch([VAC0])), compute_force=True)
    assert out["short_circuit"]
    base = m.base_forward(m._trunk_data(_batch([VAC0])), compute_force=True)
    assert torch.allclose(out["energy"], base["energy"] + out["e_pp"], atol=1e-12)
    assert torch.allclose(out["forces"], base["forces"] + (out["forces"] - base["forces"]), atol=0)   # no NaN


def test_s_equal_zero_reproduces_b0(arm_b):
    ref = _arm(arm_b, "none")(dict(_batch([VACP])), compute_force=True)
    out = _arm(arm_b, "potential", s=0.0)(dict(_batch([VACP])), compute_force=True)
    assert abs(float(out["energy"] - ref["energy"])) < 1e-9
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-9)
    assert torch.allclose(out["dq"], ref["dq"], atol=1e-9)
    assert abs(out["diagnostics"]["e_near_first_order"][0]) < 1e-12


def test_band_response_identity(arm_b):
    """d ΔF_band / d s (central difference in s) = sum_c dq_c W_c / s with dq from H0 - diag(W)."""
    m = _arm(arm_b, "potential")
    b = _batch([VACP])
    out = m(dict(b), compute_force=False)
    first_order = out["diagnostics"]["e_near_first_order"][0] / S_INIT
    assert abs(first_order) > 1e-4, "the toy potential does nothing; the test proves nothing"
    h = 1e-4
    e = []
    for s in (S_INIT + h, S_INIT - h):
        ms = _arm(arm_b, "potential", s=s)
        e.append(_band(ms(dict(b), compute_force=False)))
    fd = (e[0] - e[1]) / (2 * h)
    assert abs(fd - first_order) < 1e-6 * abs(first_order), (fd, first_order)


def test_charged_forces_match_finite_differences(arm_b):
    m = _arm(arm_b, "potential")
    b = _batch([VACP])
    out = m(dict(b), compute_force=True)
    assert not out["short_circuit"]
    assert abs(float(out["dq"].sum()) - 1.0) < 1e-6                     # sum dq = Q
    f = out["forces"].detach()
    h = 1e-4
    for a, c in [(0, 1), (9, 0), (30, 2)]:
        def e(shift):
            fr = VACP.copy(); pos = fr.get_positions(); pos[a, c] += shift; fr.set_positions(pos)
            return float(m(dict(_batch([fr])), compute_force=False)["energy"])
        fd = -(e(h) - e(-h)) / (2 * h)
        assert abs(fd - float(f[a, c])) < 1e-4 * max(1.0, abs(fd)), (a, c, fd, float(f[a, c]))


def test_gauge_shift(arm_b):
    """H0' = H0 - (W + a): every level moves by -a, dq is unchanged, E_far is unchanged, and
    ΔF_band = F(N_S) - F(N_ref) moves by -a (N_S - N_ref) = +a Q (the plan's '-a Q' for the
    opposite sign convention of the on-site shift). The total energy moves by exactly a Q."""
    m0 = _arm(arm_b, "potential")
    m1 = _arm(arm_b, "potential"); m1._w_gauge = 0.37
    b = _batch([VACP])
    o0 = m0(dict(b), compute_force=True); o1 = m1(dict(b), compute_force=True)
    assert torch.allclose(o0["dq"], o1["dq"], atol=1e-9)
    assert abs(o0["diagnostics"]["e_far"][0] - o1["diagnostics"]["e_far"][0]) < 1e-9
    assert abs((_band(o1) - _band(o0)) - 0.37 * 1.0) < 1e-9
    assert abs(float(o1["energy"] - o0["energy"]) - 0.37 * 1.0) < 1e-9
    assert torch.allclose(o0["forces"], o1["forces"], atol=1e-8)


def test_batched_equals_per_graph_and_translation_invariance(arm_b):
    m = _arm(arm_b, "potential")
    one = m(dict(_batch([VACP])), compute_force=True)
    shifted = VACP.copy(); shifted.translate([0.7, -1.1, 0.4]); shifted.wrap()
    two = m(dict(_batch([VACP, shifted])), compute_force=True)
    assert two["diagnostics"].get("batched", False)
    assert abs(float(two["energy"][0] - one["energy"][0])) < 1e-9
    assert abs(float(two["energy"][1] - one["energy"][0])) < 1e-8
    n = len(VACP)
    assert torch.allclose(two["forces"][:n], one["forces"], atol=1e-8)
    assert torch.allclose(two["forces"][n:], one["forces"], atol=1e-6)
    # the per-graph path (a mixed batch forces it): same numbers
    mixed = m(dict(_batch([VACP, VAC0])), compute_force=True)
    assert not mixed["diagnostics"].get("batched", False)
    assert abs(float(mixed["energy"][0] - one["energy"][0])) < 1e-9
    assert torch.allclose(mixed["forces"][:n], one["forces"], atol=1e-8)


def test_filled_hamiltonian_is_the_instrument(arm_b):
    """`filled_hamiltonians` returns H0 for B_s (bit-exact against h0) and H0 - diag(W) for B_w,
    whose one-hole fill is the forward's dq; the precondition reads the same object."""
    b = _batch([VACP])
    ms = _arm(arm_b, "mulliken")
    (H_s, numbers), = ms.filled_hamiltonians(dict(b))
    with torch.no_grad():
        out_s = ms.base_forward(ms._trunk_data(dict(b)), training=False, compute_force=False)
        sc, ve = ms.features(out_s["node_feats"])
        H_ref = ms.h0(sc, ve, b["node_attrs"].argmax(dim=-1), b["edge_index"], ms.head_edge_vectors(b))
    assert torch.equal(H_s, H_ref)
    mw = _arm(arm_b, "potential")
    (H_w, numbers), = mw.filled_hamiltonians(dict(b))
    assert not torch.allclose(H_w, H_ref)
    assert torch.allclose(H_w - H_ref, torch.diag(torch.diag(H_w - H_ref)), atol=1e-12)   # diagonal shift only
    fwd = mw(dict(b), compute_force=False)
    _, dq = hole_distribution(H_w, numbers, mw.sigma_s)
    assert torch.allclose(dq, fwd["dq"].detach(), atol=1e-9)
    rec_w = bound_state_precondition(mw, [dict(_batch([VAC0]))], PreconditionConfig())["records"][0]
    rec_s = bound_state_precondition(ms, [dict(_batch([VAC0]))], PreconditionConfig())["records"][0]
    assert rec_w["separation"] != rec_s["separation"]
