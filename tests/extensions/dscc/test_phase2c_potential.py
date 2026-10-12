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


def test_gap_regulariser_reads_the_filled_hamiltonian(arm_b):
    """The Γ-gap term acts on H0 - diag(W) for B_w (the Route B / L3 rule), on H0 for B_s; both equal
    the gap of `filled_hamiltonians` on the same cell."""
    from tests.extensions.dscc.test_model import PRISTINE
    from mace.modules.dscc.species import S_REF, neutral_count
    b = _batch([PRISTINE])
    for near in ("mulliken", "potential"):
        m = _arm(arm_b, near)
        gap = float(m.pristine_gap(dict(b)))
        (H, numbers), = m.filled_hamiltonians(dict(b))
        n_up, _ = S_REF.counts(neutral_count(numbers)); eps = torch.linalg.eigvalsh(H)
        assert abs(gap - float(eps[n_up] - eps[n_up - 1])) < 1e-9, near
    g_s = float(_arm(arm_b, "mulliken").pristine_gap(dict(b))); g_w = float(_arm(arm_b, "potential").pristine_gap(dict(b)))
    assert abs(g_s - g_w) > 1e-6, "W changes the filled gap; the term must see it"
    g_w0 = float(_arm(arm_b, "potential", s=0.0).pristine_gap(dict(b)))
    assert abs(g_w0 - g_s) < 1e-9


def test_on_site_half_width_rules():
    from mace.modules.dscc.hamiltonian import delta_from_baselines, delta_from_sp_splitting
    eps0 = torch.tensor([[-24.616, -11.747], [-3.404, -1.796], [-15.199, -8.029]])
    d_old = delta_from_baselines(eps0, 0.4); d_new = delta_from_sp_splitting(eps0, 0.1)
    assert torch.allclose(d_old, torch.full((3, 2), float(0.4 * eps0.reshape(-1).std(unbiased=False))))
    assert torch.allclose(d_new[:, 0], torch.tensor([1.2869, 0.1608, 0.7170]), atol=1e-3) and torch.equal(d_new[:, 0], d_new[:, 1])
    m = _pol.MACEDSCC(_pol._base(), r_cut=_pol.R_CUT, directional=True, hidden=16, coupling=False, route_b=True, scf_free=True,
                      kernel=_pol.KernelConfig(regime="B", r_g=1.0, r_s=5.0), host_coupling="polarisation",
                      field_provider=_pol._provider(), delta_rule="sp_splitting", delta_frac=0.1)
    e = m.h0.sk.eps0.detach()
    assert torch.allclose(m.h0.sk.delta, (0.1 * (e[:, 0] - e[:, 1]).abs()).unsqueeze(-1).expand_as(e))
    m0 = _pol._model()
    assert torch.allclose(m0.h0.sk.delta, delta_from_baselines(m0.h0.sk.eps0, 0.4))      # the default is unchanged


def test_gap_loss_tolerance_band():
    """Spike investigation II: band 0 is the plain square; no gradient inside the band; the gradient is
    continuous at the edge and matches a central difference outside it."""
    from mace.modules.dscc.train import gap_loss
    e_gap, w, b = 2.40, 1.0, 0.0259
    for d in (-0.1, -0.01, 0.0, 0.004, 0.3):
        g = torch.tensor([e_gap + d], dtype=torch.float64)
        assert torch.allclose(gap_loss(g, e_gap, w, 0.0), w * torch.tensor([d * d], dtype=torch.float64))
    for d in (-0.02, -0.001, 0.0, 0.01, 0.0258):
        g = torch.tensor([e_gap + d], dtype=torch.float64, requires_grad=True)
        gap_loss(g, e_gap, w, b).backward()
        assert float(gap_loss(g, e_gap, w, b)) == 0.0 and float(g.grad.abs()) == 0.0
    # continuity at the edge: the gradient just outside is O(epsilon)
    g = torch.tensor([e_gap + b + 1e-6], dtype=torch.float64, requires_grad=True); gap_loss(g, e_gap, w, b).backward()
    assert float(g.grad) == pytest.approx(2 * w * 1e-6, rel=1e-6)
    # outside: analytic 2 w (|d| - b) sign(d) against a central difference
    for d in (1.5 * b, -1.5 * b, 0.3):
        g = torch.tensor([e_gap + d], dtype=torch.float64, requires_grad=True); gap_loss(g, e_gap, w, b).backward()
        h = 1e-6; num = (float(gap_loss(torch.tensor([e_gap + d + h]), e_gap, w, b)) - float(gap_loss(torch.tensor([e_gap + d - h]), e_gap, w, b))) / (2 * h)
        assert float(g.grad) == pytest.approx(2 * w * (abs(d) - b) * math.copysign(1, d), rel=1e-9)
        assert float(g.grad) == pytest.approx(num, rel=1e-5)


def test_smooth_gap_continuous_across_a_crossing():
    """Spike investigation II: sigma = 0 is the hard gap; with sigma > 0 the gap's derivative is continuous
    where a single level crosses a degenerate pair at the band edge (the hard gap's is not), and the
    smooth gap is within sigma ln 3 of the hard gap at the degenerate point."""
    from mace.modules.dscc.model import smooth_gap
    sigma = 0.0259
    def levels(a):
        return torch.sort(torch.stack([torch.tensor(-10.0, dtype=torch.float64), torch.tensor(-9.9, dtype=torch.float64),
                                       torch.tensor(-9.5, dtype=torch.float64), a, torch.tensor(-7.0, dtype=torch.float64),
                                       torch.tensor(-7.0, dtype=torch.float64)]))[0]
    for a0 in (-7.3, -7.02, -6.9):
        a = torch.tensor(a0, dtype=torch.float64)
        assert float(smooth_gap(levels(a), 3, 0.0)) == pytest.approx(min(a0, -7.0) + 9.5)
    def dgap(a0, sig):
        a = torch.tensor(a0, dtype=torch.float64, requires_grad=True); g = smooth_gap(levels(a), 3, sig); g.backward(); return float(a.grad)
    # hard gap: derivative jumps from 1 (a is the edge) to 0 (the pair is the edge)
    assert dgap(-7.001, 0.0) == pytest.approx(1.0) and dgap(-6.999, 0.0) == pytest.approx(0.0)
    # smooth gap: the derivative changes by O(1e-3 / sigma) across the same step, and equals 1/3 at the degenerate point
    d_minus, d_plus = dgap(-7.001, sigma), dgap(-6.999, sigma)
    assert abs(d_minus - d_plus) < 0.1 and abs(d_minus - 1 / 3) < 0.05
    a = torch.tensor(-7.0, dtype=torch.float64)
    assert 0.0 <= float(smooth_gap(levels(a), 3, sigma)) - float(smooth_gap(levels(a), 3, 0.0)) <= sigma * math.log(3) + 1e-9
    # an isolated edge far from other levels: smooth equals hard to exp(-d/sigma)
    far = torch.tensor([-10.0, -9.9, -9.0, -6.0, -5.0, -4.0], dtype=torch.float64)
    assert float(smooth_gap(far, 3, sigma)) == pytest.approx(3.0, abs=1e-9)


def test_plain_step_outside_adam_is_proportional_and_capped():
    """Second lever (spike investigation II): the plain step's response to a stiff scalar constraint is
    proportional with gain lr * 2w * |dgap/dtheta|^2, independent of Adam's state; lr = 0 is a no-op; the
    cap bounds the step's norm."""
    from mace.modules.dscc.train import apply_plain_step, gap_loss
    torch.manual_seed(0)
    a = torch.randn(1000, dtype=torch.float64) * 0.3                      # dgap/dtheta, |a|^2 ~ 90: a stiff mode
    theta = torch.zeros(1000, dtype=torch.float64, requires_grad=True)
    w, target = 1.0, 2.40
    def gap(th): return 2.0 + (a * th).sum()
    # lr = 0: nothing moves
    g0 = torch.autograd.grad(gap_loss(gap(theta).unsqueeze(0), target, w), [theta])
    assert apply_plain_step([theta], g0, 0.0, 0.01) == 0.0 and float(theta.abs().max()) == 0.0
    # proportional response: one plain step changes the gap by -gain * (gap - target), gain = lr * 2w * |a|^2
    lr = 0.5 / (2 * w * float((a * a).sum()))                               # gain 0.5
    e0 = float(gap(theta)) - target
    apply_plain_step([theta], g0, lr, 0.0)
    e1 = float(gap(theta)) - target
    assert e1 == pytest.approx(0.5 * e0, rel=1e-9)
    # ten steps converge monotonically, no sign change (gain < 1)
    errs = [e1]
    for _ in range(10):
        g = torch.autograd.grad(gap_loss(gap(theta).unsqueeze(0), target, w), [theta]); apply_plain_step([theta], g, lr, 0.0); errs.append(float(gap(theta)) - target)
    assert all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1)) and all(e * errs[0] > 0 for e in errs)
    # the cap bounds the step's norm
    theta2 = torch.zeros(1000, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(gap_loss(gap(theta2).unsqueeze(0), target, w), [theta2])
    n = apply_plain_step([theta2], g, 1.0, 0.01)
    assert n == pytest.approx(0.01) and float(theta2.norm()) == pytest.approx(0.01)
