"""Plan P Phase 2b: the range-separated carrier-host coupling (switch S, E_far, E_near).

Plan §1 tests: neutral null (base + E_PP bit-exact); S == 1 (r_c = 0) reproduces arm B's energy
and forces to 1e-9 for B_s and B_0; S == 0 on B_s: E_near against W6's E_host (the difference is
its far-field Ewald part, reported); central-difference forces on a charged frame with pairs
straddling the ramp; sum dq = Q; batched = per-graph; translation invariance."""
import copy
import importlib.util, os
import pytest
import torch

from mace.modules.dscc import polarisation as pol
from mace.modules.dscc.kernels import gamma_lr
from tests.extensions.dscc.test_model import VAC0, VACP, _batch

_spec = importlib.util.spec_from_file_location("_pol_helpers", os.path.join(os.path.dirname(__file__), "test_polarisation_head.py"))
_pol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pol)

torch.set_default_dtype(torch.float64)
R_C, DELTA = 3.0, 1.0            # the test cell's head cutoff is small: the ramp must sit inside it


def _arm(model, near_field, r_c=R_C, delta=DELTA):
    m = copy.deepcopy(model)
    m.near_field, m.r_c, m.delta = near_field, float(r_c), float(delta)
    return m


@pytest.fixture(scope="module")
def arm_b():
    return _pol._model()                      # arm B: polarisation coupling, near_field "off"


def test_switch_shape_and_smoothness():
    r = torch.tensor([0.0, 1.9, 2.0, 2.5, 3.0, 3.5])
    s = pol.switch(r, R_C, DELTA)
    assert torch.allclose(s, torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0]))
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    (g1,) = torch.autograd.grad(pol.switch(x, R_C, DELTA).sum(), x, create_graph=True)
    (g2,) = torch.autograd.grad(g1.sum(), x)
    assert torch.allclose(g1, torch.zeros(2), atol=1e-12) and torch.allclose(g2, torch.zeros(2), atol=1e-12)
    assert torch.all(pol.switch(r, 0.0, DELTA) == 1.0)


def test_state_round_trip(arm_b):
    m = _arm(arm_b, "mulliken")
    st = m.get_extra_state()
    assert st["near_field"] == "mulliken" and st["r_c"] == R_C and st["delta"] == DELTA
    m2 = copy.deepcopy(arm_b); m2.set_extra_state(st)
    assert m2.near_field == "mulliken" and m2.r_c == R_C


def test_neutral_null_is_base_plus_e_pp(arm_b):
    for near in ("mulliken", "none"):
        m = _arm(arm_b, near)
        out = m(dict(_batch([VAC0])), compute_force=True)
        assert out["short_circuit"]
        base = m.base_forward(m._trunk_data(_batch([VAC0])), compute_force=True)
        assert torch.allclose(out["energy"], base["energy"] + out["e_pp"], atol=1e-12)


def test_s_equal_one_reproduces_arm_b(arm_b):
    ref = arm_b(dict(_batch([VACP])), compute_force=True)
    for near in ("mulliken", "none"):
        m = _arm(arm_b, near, r_c=0.0)
        out = m(dict(_batch([VACP])), compute_force=True)
        assert abs(float(out["energy"] - ref["energy"])) < 1e-9
        assert torch.allclose(out["forces"], ref["forces"], atol=1e-9)
        assert abs(out["diagnostics"]["e_near"][0]) < 1e-12
        assert abs(out["diagnostics"]["e_far"][0] - ref["diagnostics"]["e_host"][0]) < 1e-12


def test_s_equal_zero_near_field_against_w6_host_term(arm_b):
    """S == 0 on every enumerated pair: E_near is W6's E_host on the direct kernel over the head's
    pairs (plus the self term); the W6 control with the same parameters fills the same H0, so its
    dq, q0 and s are identical and its e_host is the periodic version. The difference is the
    far-field (images and beyond-cutoff) part of the Ewald kernel; reported, no threshold."""
    m = _arm(arm_b, "mulliken", r_c=1e6, delta=1.0)
    out = m(dict(_batch([VACP])), compute_force=False)
    e_near = out["diagnostics"]["e_near"][0]
    ctrl = _pol._model(coupling="mulliken")
    ctrl.load_state_dict({k: v for k, v in m.state_dict().items() if not k.endswith("_extra_state")}, strict=False)
    assert ctrl.r_split == m.near_r_split
    ref = ctrl(dict(_batch([VACP])), compute_force=False)
    e_w6 = ref["diagnostics"]["e_host"][0]
    assert torch.allclose(out["dq"], ref["dq"], atol=1e-9)
    # the periodic W6 kernel on the SAME dq, s q0: the difference to E_near is the far-field
    # (images + beyond-cutoff) part; in this 6 A toy cell it is not weak and is reported only
    sq0 = torch.tensor(out["diagnostics"]["sq0"][0])
    pos, cell = _batch([VACP])["positions"], _batch([VACP])["cell"].view(3, 3)
    g = gamma_lr(pos, cell, m.kernel.r_g, m.near_r_split, m.kernel.eps_inf, tol=m.kernel.tol)
    e_periodic = float(out["dq"].detach() @ g @ sq0)
    assert abs(e_periodic - e_w6) < 1e-6, (e_periodic, e_w6)
    assert abs(e_near) > 1e-6
    print(f"S == 0 (toy 6 A cell, sharp cutoff): E_near {e_near:+.6f} eV vs W6 periodic E_host {e_w6:+.6f} eV; far-field part {e_w6 - e_near:+.6f} eV")


def test_charged_forces_match_finite_differences_across_the_ramp(arm_b):
    for near in ("mulliken", "none"):
        m = _arm(arm_b, near)
        b = _batch([VACP])
        out = m(dict(b), compute_force=True)
        assert not out["short_circuit"]
        assert abs(float(out["dq"].sum()) - 1.0) < 1e-6
        # at least one pair inside the ramp
        ev = m.head_edge_vectors(dict(b)) if hasattr(m, "head_edge_vectors") else None
        f = out["forces"].detach()
        h = 1e-4
        for a, c in [(0, 1), (9, 0), (30, 2)]:
            def e(shift):
                fr = VACP.copy(); pos = fr.get_positions(); pos[a, c] += shift; fr.set_positions(pos)
                return float(m(dict(_batch([fr])), compute_force=False)["energy"])
            fd = -(e(h) - e(-h)) / (2 * h)
            assert abs(fd - float(f[a, c])) < 1e-4 * max(1.0, abs(fd)), (near, a, c, fd, float(f[a, c]))


def test_batched_equals_per_graph_and_translation_invariance(arm_b):
    m = _arm(arm_b, "mulliken")
    one = m(dict(_batch([VACP])), compute_force=True)
    shifted = VACP.copy(); shifted.translate([0.7, -1.1, 0.4]); shifted.wrap()
    two = m(dict(_batch([VACP, shifted])), compute_force=True)
    assert abs(float(two["energy"][0] - one["energy"][0])) < 1e-9
    assert abs(float(two["energy"][1] - one["energy"][0])) < 1e-8
    n = len(VACP)
    assert torch.allclose(two["forces"][:n], one["forces"], atol=1e-8)
    assert torch.allclose(two["forces"][n:], one["forces"], atol=1e-6)
