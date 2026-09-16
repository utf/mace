"""LES head line, L0 (2026-09-15): the head on a MACELES base. `base_forward` was pinned to
`ScaleShiftMACE.forward`, which silently drops a MACELES base's latent Ewald term; it now
dispatches on the base's class, the neutral null is the LES base bit-identically, charged
forces stay exact against central differences, the latent charges are surfaced as a
differentiable tensor (L2's `q_LES`), and capture refuses the base (this line runs eager)."""
import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.dscc.analytic import capture_refusal
from mace.modules.dscc.model import MACEDSCC
from tests.extensions.dscc.test_model import (PRISTINE, R_CUT, VAC0, VACP, _base, _batch, _frame,
                                              _perovskite)

pytest.importorskip("les")
torch.set_default_dtype(torch.float64)


def _les_base(seed=0):
    from mace.modules.extensions import MACELES
    torch.manual_seed(seed)
    base = MACELES(
        les_arguments={"use_atomwise": False},
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu,
        atomic_energies=np.zeros((1, 3)), avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82],
        correlation=2, atomic_inter_scale=1.0, atomic_inter_shift=0.0).double()
    return base


def _wrap(base):
    m = MACEDSCC(base, r_cut=R_CUT, directional=True, hidden=16)
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
    return _wrap(_les_base())


class TestTheBaseIsCalledThroughItsOwnForward:
    def test_base_forward_carries_the_ewald_term(self, model):
        trunk = model._trunk_data(_batch([VAC0]))
        own = model.base(dict(trunk), compute_force=True)
        via = model.base_forward(dict(trunk), compute_force=True)
        pinned = modules.ScaleShiftMACE.forward(model.base, dict(trunk), compute_force=True)
        assert torch.equal(via["energy"], own["energy"]) and torch.equal(via["forces"], own["forces"])
        assert float(own["les_energy"].abs()) > 1e-8, "the toy base's Ewald term is zero; the test proves nothing"
        assert float((pinned["energy"] + own["les_energy"] - own["energy"]).abs()) < 1e-10
        assert float((pinned["energy"] - own["energy"]).abs()) > 1e-8

    def test_neutral_null_is_the_les_base_bit_identically(self, model):
        batch = _batch([PRISTINE, VAC0])
        out = model(dict(batch), compute_force=True, compute_stress=True)
        ref = model.base(model._trunk_data(dict(_batch([PRISTINE, VAC0]))), compute_force=True,
                         compute_stress=True)
        assert out["short_circuit"] and torch.equal(out["energy"], ref["energy"])
        assert torch.equal(out["forces"], ref["forces"]) and torch.equal(out["stress"], ref["stress"])

    def test_charged_forces_against_central_differences(self, model):
        out = model(_batch([VACP]), compute_force=True)
        assert not out["short_circuit"]
        h = 1e-4
        for atom, comp in ((1, 0), (20, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(model(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(model(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=2e-6)

    def test_charged_energy_contains_the_bases_ewald_term(self, model):
        out = model(_batch([VACP]), compute_force=False)
        own = model.base(model._trunk_data(_batch([VACP])), compute_force=False)
        assert float((out["base_energy"] - own["energy"]).abs()) < 1e-10
        assert float((out["base_les_energy"] - own["les_energy"]).abs()) < 1e-10


class TestLatentChargesAreExposed:
    def test_surfaced_and_differentiable_in_the_positions(self, model):
        batch = _batch([VACP])
        out = model(batch, compute_force=True, training=True)
        q = out["base_latent_charges"]
        assert torch.is_tensor(q) and q.shape == (len(VACP),) and q.requires_grad and q.grad_fn is not None
        own = model.base(model._trunk_data(_batch([VACP])), compute_force=False)
        assert float((q - own["latent_charges"]).abs().max()) < 1e-10
        # d(sum q)/dR by autograd through the head's positions leaf, against central differences
        atom, comp = 3, 1
        (g,) = torch.autograd.grad(q.sum(), batch["positions"], retain_graph=True)
        h = 1e-4
        plus, minus = VACP.copy(), VACP.copy()
        plus.positions[atom, comp] += h
        minus.positions[atom, comp] -= h
        fd = (float(model(_batch([plus]), compute_force=False)["base_latent_charges"].sum())
              - float(model(_batch([minus]), compute_force=False)["base_latent_charges"].sum())) / (2 * h)
        assert abs(fd) > 1e-6, "the toy charges do not move with the positions; the test proves nothing"
        assert float(g[atom, comp]) == pytest.approx(fd, rel=1e-4, abs=1e-8)

    def test_a_plain_base_reports_none(self):
        m = _wrap(_base())
        out = m(_batch([VACP]), compute_force=False)
        assert out["base_latent_charges"] is None and out["base_les_energy"] is None
        assert m.base_forward(m._trunk_data(_batch([VAC0])))["energy"].shape == (1,)


def test_capture_is_refused_on_a_les_base(model):
    r = capture_refusal(model)
    assert r is not None and "MACELES" in r


# ----------------------------------------------------------------------------- L2: E_cross
from mace.modules.dscc.kernels import KernelConfig, gamma_lr   # noqa: E402


def _l2(seed=0, les_sign=1.0, les_cross=True):
    m = MACEDSCC(_les_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, les_cross=les_cross, les_sign=les_sign,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


def _cached_batch(model, frame):
    """The trainer's cached-base batch for one frame: E_base, F_base, q_LES and its Jacobian."""
    batch = _batch([frame])
    own = model.base(model._trunk_data(dict(batch)), compute_force=True)
    q, J = model.latent_charge_jacobian(dict(batch))
    b = dict(_batch([frame]))
    b["dscc_base_energy"], b["dscc_base_forces"] = own["energy"].detach(), own["forces"].detach()
    b["dscc_base_q"], b["dscc_base_J"] = q, J
    return b


@pytest.fixture(scope="module")
def l2():
    return _l2()


class TestL2Construction:
    def test_guards(self):
        with pytest.raises(ValueError):
            MACEDSCC(_les_base(), r_cut=R_CUT, route_b=True, scf_free=False, les_cross=True)
        with pytest.raises(ValueError):
            MACEDSCC(_base(), r_cut=R_CUT, route_b=True, scf_free=True, les_cross=True)
        with pytest.raises(ValueError):
            MACEDSCC(_les_base(), r_cut=R_CUT, route_b=True, scf_free=True, les_cross=True, les_sign=0.5)

    def test_scale_initialised_at_one_over_sqrt_eps_inf_and_bounded(self, l2):
        assert float(l2.cross_scale()) == pytest.approx(1.0 / 4.0 ** 0.5, abs=1e-12)
        with torch.no_grad():
            l2.s_cross_raw.fill_(50.0)
            assert float(l2.cross_scale()) == pytest.approx(l2.s_max, abs=1e-9)
            l2.s_cross_raw.fill_(-50.0)
            assert float(l2.cross_scale()) == pytest.approx(-l2.s_max, abs=1e-9)
            l2.s_cross_raw.copy_(torch.atanh(torch.tensor(0.5 / l2.s_max)))

    def test_round_trip_keeps_the_flag_and_sign(self):
        m = _l2(les_sign=-1.0)
        clone = _l2(les_sign=1.0)
        clone.load_state_dict(m.state_dict())
        assert clone.les_cross and clone.les_sign == -1.0
        assert float(clone.cross_scale()) == float(m.cross_scale())


class TestL2Energy:
    def test_neutral_null_is_the_les_base(self, l2):
        out = l2(_batch([VAC0]), compute_force=True)
        ref = l2.base(l2._trunk_data(_batch([VAC0])), compute_force=True)
        assert out["short_circuit"] and torch.equal(out["energy"], ref["energy"]) and torch.equal(out["forces"], ref["forces"])

    def test_e_cross_is_s_dq_K_q_les_and_no_reference_fill_enters(self, l2):
        out = l2(_batch([VACP]), compute_force=False)
        d = out["diagnostics"]
        assert "q0_sum" not in d and "e_cross" in d and "s_cross" in d
        own = l2.base(l2._trunk_data(_batch([VACP])), compute_force=False)
        q = own["latent_charges"].reshape(-1) * l2.les_sign
        pos = torch.tensor(VACP.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(VACP.get_cell()), dtype=torch.float64)
        K = gamma_lr(pos, cell, l2.kernel.r_g, l2.r_split, 1.0, tol=l2.kernel.tol)     # BARE
        e_cross = float(l2.cross_scale()) * float(out["dq"] @ K @ q)
        assert d["e_cross"][0] == pytest.approx(e_cross, abs=1e-9)
        assert abs(e_cross) > 1e-6, "the toy E_cross vanishes; the test proves nothing"

    def test_sign_fix_flips_e_cross(self):
        plus, minus = _l2(les_sign=1.0), _l2(les_sign=-1.0)
        ep = plus(_batch([VACP]), compute_force=False)["diagnostics"]["e_cross"][0]
        em = minus(_batch([VACP]), compute_force=False)["diagnostics"]["e_cross"][0]
        assert ep == pytest.approx(-em, abs=1e-9) and abs(ep) > 1e-6


class TestL2Forces:
    def test_attached_route_against_central_differences(self, l2):
        out = l2(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (20, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_p = float(l2(_batch([plus]), compute_force=False)["energy"])
            e_m = float(l2(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=2e-6)

    def test_cached_jacobian_route_equals_the_attached_route(self, l2):
        attached = l2(_batch([VACP]), compute_force=True)
        cached = l2(_cached_batch(l2, VACP), compute_force=True)
        assert float((attached["energy"] - cached["energy"]).abs()) < 1e-10
        assert float((attached["forces"] - cached["forces"]).abs().max()) < 1e-8
        assert cached["base_latent_charges"] is None and attached["base_latent_charges"] is not None

    def test_cached_route_in_training_mode_against_central_differences(self, l2):
        out = l2(_cached_batch(l2, VACP), compute_force=True, training=True)
        assert out["forces"].requires_grad          # the force loss needs the parameter graph
        h = 1e-4
        for atom, comp in ((7, 2),):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_p = float(l2(_cached_batch(l2, plus), compute_force=False)["energy"])
            e_m = float(l2(_cached_batch(l2, minus), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=2e-6)

    def test_the_jacobian_term_is_not_negligible(self, l2):
        """Dropping J (a zero Jacobian) must move a force by more than the FD tolerance,
        otherwise the cached route would pass its gates while carrying nothing."""
        b = _cached_batch(l2, VACP)
        full = l2(dict(b), compute_force=True)["forces"]
        b["dscc_base_J"] = torch.zeros_like(b["dscc_base_J"])
        without = l2(b, compute_force=True)["forces"]
        assert float((full - without).abs().max()) > 1e-5


# ----------------------------------------------------------------------------- L3: the well in H0
from mace.modules.dscc.model import S_WELL_INIT   # noqa: E402


def _l3(seed=0, les_sign=1.0, s_well=None):
    m = MACEDSCC(_les_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, les_cross=True, les_sign=les_sign, les_well=True,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
        if s_well is not None:
            m.s_well_raw.copy_(torch.logit(torch.tensor(s_well / m.s_max, dtype=torch.float64)))
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


@pytest.fixture(scope="module")
def l3():
    return _l3(s_well=0.5)          # a well large enough to be measurable


class TestL3Construction:
    def test_needs_les_cross(self):
        with pytest.raises(ValueError):
            MACEDSCC(_les_base(), r_cut=R_CUT, route_b=True, scf_free=True, les_cross=False, les_well=True)

    def test_starts_at_l2_within_the_registered_init(self):
        l3_init, l2 = _l3(), _l2()
        assert float(l3_init.well_scale()) == pytest.approx(S_WELL_INIT, rel=1e-9)
        with torch.no_grad():
            l3_init.s_well_raw.fill_(-60.0)             # s_W -> 0: L2 exactly
        e3 = l3_init(_batch([VACP]), compute_force=True)
        e2 = l2(_batch([VACP]), compute_force=True)
        assert float((e3["energy"] - e2["energy"]).abs()) < 1e-9
        assert float((e3["forces"] - e2["forces"]).abs().max()) < 1e-8
        with torch.no_grad():
            l3_init.s_well_raw.fill_(60.0)
        assert float(l3_init.well_scale()) == pytest.approx(l3_init.s_max, abs=1e-9)

    def test_round_trip(self, l3):
        clone = _l3()
        clone.load_state_dict(l3.state_dict())
        assert clone.les_well and float(clone.well_scale()) == pytest.approx(0.5, abs=1e-12)


class TestL3Physics:
    def test_neutral_null_is_the_les_base(self, l3):
        out = l3(_batch([VAC0]), compute_force=True)
        ref = l3.base(l3._trunk_data(_batch([VAC0])), compute_force=True)
        assert out["short_circuit"] and torch.equal(out["energy"], ref["energy"])

    def test_the_well_moves_the_charged_energy(self, l3):
        e3 = float(l3(_batch([VACP]), compute_force=False)["energy"])
        e2 = float(_l2()(_batch([VACP]), compute_force=False)["energy"])
        assert abs(e3 - e2) > 1e-5

    def test_gap_regulariser_acts_on_h0_minus_w(self, l3):
        from mace.modules.dscc.model import S_REF, neutral_count
        batch = _batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])
        gap = float(l3.pristine_gap(dict(batch)))
        # by hand: H0 of the pristine cell, minus the centred bare-kernel well of s_W q_LES
        feats = l3.features(l3.first_block(l3._trunk_data(dict(batch))))
        species = batch["node_attrs"].argmax(dim=-1)
        pos, cell = batch["positions"], batch["cell"].view(-1, 3, 3)
        sender, receiver = batch["edge_index"]
        shifts = torch.einsum("ei,eij->ej", batch["unit_shifts"].to(pos.dtype), cell[batch["batch"][sender]])
        H0 = l3.h0(feats[0], feats[1], species, batch["edge_index"], pos[receiver] - pos[sender] + shifts)
        q = l3.base(l3._trunk_data(dict(batch)), compute_force=False)["latent_charges"].reshape(-1) * l3.les_sign
        K = gamma_lr(pos, cell[0], l3.kernel.r_g, l3.r_split, 1.0, tol=l3.kernel.tol)
        w = K @ q
        W = float(l3.well_scale()) * (w - w.mean())
        H = H0 - torch.diag(W.repeat_interleave(4))
        numbers = [l3.atomic_numbers[int(s_)] for s_ in species.tolist()]
        n_up, _ = S_REF.counts(neutral_count(numbers))
        eps = torch.linalg.eigvalsh(H.detach())
        assert gap == pytest.approx(float(eps[n_up] - eps[n_up - 1]), abs=1e-9)
        eps0 = torch.linalg.eigvalsh(H0.detach())
        assert abs(gap - float(eps0[n_up] - eps0[n_up - 1])) > 1e-6, "the well does not move the gap; the test proves nothing"


class TestL3Forces:
    def test_attached_route_against_central_differences(self, l3):
        out = l3(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (20, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_p = float(l3(_batch([plus]), compute_force=False)["energy"])
            e_m = float(l3(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=2e-6)

    def test_cached_jacobian_route_equals_the_attached_route(self, l3):
        attached = l3(_batch([VACP]), compute_force=True)
        cached = l3(_cached_batch(l3, VACP), compute_force=True)
        assert float((attached["energy"] - cached["energy"]).abs()) < 1e-10
        assert float((attached["forces"] - cached["forces"]).abs().max()) < 1e-8

    def test_cached_route_in_training_mode_against_central_differences(self, l3):
        out = l3(_cached_batch(l3, VACP), compute_force=True, training=True)
        h = 1e-4
        atom, comp = 7, 2
        plus, minus = VACP.copy(), VACP.copy()
        plus.positions[atom, comp] += h
        minus.positions[atom, comp] -= h
        e_p = float(l3(_cached_batch(l3, plus), compute_force=False)["energy"])
        e_m = float(l3(_cached_batch(l3, minus), compute_force=False)["energy"])
        assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=2e-6)

    def test_stress_route_matches_the_pair_route(self, l3):
        """compute_stress switches the kernel to autograd (no pair derivatives): same forces."""
        pairs = l3(_batch([VACP]), compute_force=True)
        auto = l3(_batch([VACP]), compute_force=True, compute_stress=True)
        assert float((pairs["forces"] - auto["forces"]).abs().max()) < 1e-8


class TestLaddersRunWithoutGradients:
    """The embedding and tiling ladders call the head under `torch.no_grad()` with
    `compute_force=False`: the L2/L3 paths must run there (no cache, base attached to nothing)
    and give the same energy as the gradient-enabled forward."""
    @pytest.mark.parametrize("build", [_l2, lambda: _l3(s_well=0.5)])
    def test_no_grad_energy_equals_the_attached_energy(self, build):
        m = build()
        e_grad = float(m(_batch([VACP]), compute_force=True)["energy"])
        with torch.no_grad():
            out = m(_batch([VACP]), compute_force=False)
        assert float(out["energy"]) == pytest.approx(e_grad, abs=1e-9)
        assert "e_cross" in out["diagnostics"]
