"""W6 gates (plan section W6): the SCF-free model `E = E_base + dF_band(H0) + E_M(Q; h) +
E_host + C_Q`, with `E_host = dq^T Gamma_LR (s q0)` non-self-consistent and `E_M` the
point-charge Madelung energy of the cell.

The Phi = 0 gates are re-run here because W6 adds two terms to that path: the charges must
still be the fill's, the two force routes must still agree, and the derivatives must still
match finite differences -- now with `E_host`'s d dq/dR and d q0/dR, which are NOT
Hellmann-Feynman terms (the energy is not stationary in a non-self-consistent charge) and
the cell-only `E_M`, whose derivative is a stress and no force.
"""
import numpy as np
import pytest
import torch

from mace.modules.dscc.ewald import COULOMB, madelung_constant_cubic, madelung_self
from mace.modules.dscc.kernels import KernelConfig
from mace.modules.dscc.model import MACEDSCC
from mace.modules.dscc.scf import two_fillings
from tests.extensions.dscc.test_model import (R_CUT, VAC0, VACP, _base, _batch, _frame,
                                              _perovskite)

torch.set_default_dtype(torch.float64)

# Q = 0 but not the reference: one electron up, one hole down. E_M must vanish exactly.
VACX = _frame(_perovskite(remove_cl=0), [1, 0, 0, 1], 0)


def _w6(seed=0, scf_free=True, route_b=True):
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=route_b, scf_free=scf_free,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


@pytest.fixture(scope="module")
def w6():
    return _w6()


class TestMadelung:
    def test_matches_the_cubic_constant(self):
        for L in (5.6, 11.2, 16.8):
            cell = torch.eye(3, dtype=torch.float64) * L
            assert float(madelung_self(cell)) == pytest.approx(
                -madelung_constant_cubic() * COULOMB / L, rel=1e-9)

    def test_is_independent_of_the_splitting_width(self):
        cell = torch.tensor([[11.2, 0.0, 0.0], [0.3, 11.0, 0.0], [0.1, 0.2, 11.6]], dtype=torch.float64)
        values = [float(madelung_self(cell, eta=e)) for e in (1.4, 1.8, 2.2, 2.6)]
        assert max(values) - min(values) < 1e-10


class TestScfFreeConstruction:
    def test_scf_free_requires_route_b_and_forbids_coupling(self):
        with pytest.raises(ValueError):
            MACEDSCC(_base(), r_cut=R_CUT, route_b=False, scf_free=True)
        with pytest.raises(ValueError):
            MACEDSCC(_base(), r_cut=R_CUT, route_b=True, coupling=True, scf_free=True)

    def test_round_trip_keeps_the_flag(self, w6):
        import copy
        clone = copy.deepcopy(w6)
        clone.scf_free = False
        clone.load_state_dict(w6.state_dict())
        assert clone.scf_free is True

    def test_neutral_null_is_still_the_base(self, w6):
        out = w6(_batch([VAC0]), compute_force=True)
        assert out["short_circuit"] and float(out["head_energy"].abs().max()) == 0.0


class TestTerms:
    def test_charges_are_the_fill_and_q0_is_neutral(self, w6):
        out = w6(_batch([VACP]), compute_force=True)
        assert abs(out["diagnostics"]["q0_sum"][0]) < 1e-9     # the reference fill is neutral
        assert float(out["dq"].sum()) == pytest.approx(1.0, abs=1e-12)

    def test_the_occupation_difference_is_the_fills_dq(self, w6):
        """`dq = occ_ref - occ_S` (the plan's first Frechet contraction, sharing `occ_ref`
        with `q0`) is the same charge `two_fillings` reports -- on an arbitrary `H` at this
        geometry, so the identity is tested and not the model."""
        from mace.modules.dscc.fill import eigh_for
        n = len(VACP)
        g = torch.Generator().manual_seed(5)
        A = torch.randn(1, 4 * n, 4 * n, generator=g, dtype=torch.float64)
        H = 0.5 * (A + A.transpose(-1, -2))
        t = lambda v: torch.tensor([float(v)], dtype=torch.float64)          # noqa: E731
        counts_s, counts_r = (t(2 * n - 1), t(2 * n)), (t(2 * n), t(2 * n))
        pos = torch.tensor(np.array(VACP.positions), dtype=torch.float64).reshape(1, n, 3)
        cell = torch.tensor(np.array(VACP.get_cell()), dtype=torch.float64).reshape(1, 3, 3)
        with torch.no_grad():
            sol = two_fillings(H, counts_s, counts_r, w6.sigma_s)
            _, _, _, dq, _, _ = w6.scf_free_terms(H, eigh_for(H, "auto"), counts_s, counts_r,
                                               torch.zeros(1, n, dtype=torch.float64),
                                               pos, cell, torch.ones(1, dtype=torch.float64), False)
        assert float((dq - sol.dq).abs().max()) < 1e-10

    def test_energy_is_the_phi0_band_plus_the_two_terms(self, w6):
        import copy
        out = w6(_batch([VACP]), compute_force=False)
        phi0 = copy.deepcopy(w6)
        phi0.scf_free = False                                  # the same H0, band term alone
        band = phi0(_batch([VACP]), compute_force=False)
        d = out["diagnostics"]
        assert float(out["head_energy"][0]) == pytest.approx(
            float(band["head_energy"][0]) + d["e_host"][0] + d["e_madelung"][0], abs=1e-9)
        # Both terms are resolvable, so the derivative gates below genuinely cover them.
        assert abs(d["e_host"][0]) > 1e-3 and abs(d["e_madelung"][0]) > 1e-3

    def test_madelung_vanishes_at_zero_charge(self, w6):
        out = w6(_batch([VACX]), compute_force=False)
        assert not out["short_circuit"]
        assert float(out["diagnostics"]["e_madelung"][0]) == 0.0
        assert abs(float(out["diagnostics"]["e_host"][0])) > 0.0      # dq != 0 at Q = 0

    def test_madelung_is_the_registered_point_charge_energy(self, w6):
        """C13: the point-charge Madelung constant of the cell PLUS the model density's
        second-moment term, over eps_inf -- equivalently the Gaussian-cloud periodic self
        energy with the (size-independent, C_Q-absorbed) own-cloud term removed."""
        from mace.modules.dscc.ewald import ewald_matrix
        out = w6(_batch([VACP]), compute_force=False)
        cell = torch.tensor(np.array(VACP.get_cell()), dtype=torch.float64)
        r_g, volume = w6.kernel.r_g, float(np.linalg.det(np.array(VACP.get_cell())))
        expected = 0.5 * (float(madelung_self(cell))
                          + 4.0 * np.pi * r_g ** 2 * COULOMB / volume) / w6.kernel.eps_inf
        assert float(out["diagnostics"]["e_madelung"][0]) == pytest.approx(expected, abs=1e-10)
        diag = float(ewald_matrix(torch.zeros(1, 3, dtype=torch.float64), cell, r_g,
                                  background="density")[0, 0])
        assert expected == pytest.approx(
            0.5 * (diag - COULOMB / np.sqrt(np.pi) / r_g) / w6.kernel.eps_inf, abs=1e-10)


class TestRegulariser:
    def test_the_gap_regulariser_acts_on_h0_not_h0_minus_w(self, w6):
        """W6's host term is an energy, never a potential: the Hamiltonian it fills is `H0`,
        so the gap regulariser (and C5's reading) must be `H0`'s gap, not route B's
        `H0 - W`."""
        import copy
        batch = _batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])
        phi0 = copy.deepcopy(w6)
        phi0.scf_free, phi0.route_b = False, False               # H0 alone
        route_b = copy.deepcopy(w6)
        route_b.scf_free = False                                 # H0 - W
        assert float(w6.pristine_gap(batch)) == pytest.approx(float(phi0.pristine_gap(batch)), abs=1e-12)
        assert abs(float(w6.pristine_gap(batch)) - float(route_b.pristine_gap(batch))) > 1e-4


class TestDerivatives:
    def test_forces_against_central_differences(self, w6):
        out = w6(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (7, 2), (20, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(w6(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(w6(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=3e-6)

    def test_stress_against_strain_differences(self, w6):
        out = w6(_batch([VACP]), compute_force=False, compute_stress=True)
        volume = VACP.get_volume()
        h = 1e-5                                  # see test_model: the FD side is the coarse one
        for i, j in ((0, 0), (0, 2)):
            e = []
            for sign in (1.0, -1.0):
                strained = VACP.copy()
                eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = sign * h
                strained.set_cell(np.array(VACP.get_cell()) @ (np.eye(3) + eps), scale_atoms=True)
                e.append(float(w6(_batch([strained]), compute_force=False)["energy"]))
            fd = (e[0] - e[1]) / (2 * h) / volume
            expected = float(out["stress"][0, i, j] + (out["stress"][0, j, i] if i != j else 0.0))
            assert expected == pytest.approx(fd, abs=1e-7)

    def test_the_pair_route_matches_the_autograd_route(self, w6):
        out = w6(_batch([VACP, VACP]), compute_force=True)         # pairs (the default)
        w6.gamma_force_mode = "autograd"
        try:
            ref = w6(_batch([VACP, VACP]), compute_force=True)
        finally:
            w6.gamma_force_mode = "pairs"
        assert float((out["energy"] - ref["energy"]).abs().max()) < 1e-12
        assert float((out["forces"] - ref["forces"]).abs().max()) < 1e-9

    def test_training_forces_and_gradients_match_the_cotangent_route(self):
        """The pair route must give the same forces AND the same parameter gradients under
        `create_graph` as the attached-Gamma_LR route -- the inference comparison above
        cannot see a detached contraction, which costs `s_raw` its force gradient."""
        m = _w6(seed=1)
        batch = _batch([VACP])
        m.gamma_force_mode = "autograd"
        ref = m(dict(batch), compute_force=True)
        params = [p for p in m.parameters() if p.requires_grad]
        torch.manual_seed(3)
        target = ref["forces"].detach() + 0.01 * torch.randn_like(ref["forces"])
        grads = {}
        for mode in ("pairs", "autograd"):
            m.gamma_force_mode = mode
            out = m(dict(batch), training=True, compute_force=True)
            assert float((out["forces"] - ref["forces"]).abs().max()) < 1e-9, mode
            assert out["forces"].requires_grad
            loss = ((out["forces"] - target) ** 2).sum()
            grads[mode] = torch.autograd.grad(loss, params, allow_unused=True)
        m.gamma_force_mode = "pairs"
        n_compared = 0
        for gp, ga, p in zip(grads["pairs"], grads["autograd"], params):
            if gp is None and ga is None:
                continue
            assert gp is not None and ga is not None, p.shape
            scale = max(float(ga.abs().max()), 1e-6)
            assert float((gp - ga).abs().max()) <= 1e-8 * scale + 1e-12, (p.shape, scale)
            n_compared += 1
        assert n_compared > 0
        s_index = next(i for i, p in enumerate(params) if p is m.s_raw)   # `index` compares tensors
        s_grad = grads["pairs"][s_index]
        assert s_grad is not None and float(s_grad.abs()) > 0.0     # E_host's scale is trained

    @pytest.mark.parametrize("kind", ["w6", "route_b"])
    def test_fused_and_looped_hellmann_feynman_agree(self, kind):
        """`MACEDSCC` takes ONE `autograd.grad` call for all the cotangent terms (they share
        the base's block-0 graph and H0's). That is the sum of the per-term calls by
        linearity; this holds the two forms to round-off on forces, energy and every
        parameter gradient, so the optimisation can never become a behaviour change."""
        from mace.modules.dscc import model as mdl
        from tests.extensions.dscc.test_model import _coupled
        m = _w6(seed=2) if kind == "w6" else _coupled(regime="B", route_b=True)
        batch = _batch([VACP, VACP])
        params = [p for p in m.parameters() if p.requires_grad]
        got = {}
        try:
            for fused in (True, False):
                mdl.FUSED_HF_BACKWARD = fused
                out = m(dict(batch), training=True, compute_force=True)
                loss = (out["forces"] ** 2).sum() + (out["energy"] ** 2).sum()
                got[fused] = (out["forces"].detach().clone(), out["energy"].detach().clone(),
                              torch.autograd.grad(loss, params, allow_unused=True))
        finally:
            mdl.FUSED_HF_BACKWARD = True
        f1, e1, g1 = got[True]; f0, e0, g0 = got[False]
        assert float((f1 - f0).abs().max()) < 1e-10 * max(float(f0.abs().max()), 1e-3)
        assert float((e1 - e0).abs().max()) < 1e-10
        for a, b in zip(g1, g0):
            if a is None or b is None:
                assert a is None and b is None
                continue
            assert float((a - b).abs().max()) <= 1e-10 * max(float(b.abs().max()), 1e-6)

    def test_batched_equals_per_graph(self, w6):
        """A mixed batch takes the per-graph path; a uniform one the batched path."""
        single = w6(_batch([VACP]), compute_force=True)
        mixed = w6(_batch([VACP, VAC0]), compute_force=True)       # VAC0 is a reference graph
        n = len(VACP)
        assert float((mixed["energy"][0] - single["energy"][0]).abs()) < 1e-12
        assert float((mixed["forces"][:n] - single["forces"]).abs().max()) < 1e-10
        assert mixed["diagnostics"]["e_host"][0] == pytest.approx(single["diagnostics"]["e_host"][0], abs=1e-12)

    def test_gauge(self, w6):
        """`H0 -> H0 + a I`: dq, q0, E_host and the forces are unchanged; `E` shifts by `-a Q`."""
        ref = w6(_batch([VACP]), compute_force=True)
        w6.h0.gauge_shift = 0.37
        try:
            out = w6(_batch([VACP]), compute_force=True)
        finally:
            w6.h0.gauge_shift = 0.0
        assert float((out["dq"] - ref["dq"]).abs().max()) < 1e-10
        assert float((out["forces"] - ref["forces"]).abs().max()) < 1e-9
        assert out["diagnostics"]["e_host"][0] == pytest.approx(ref["diagnostics"]["e_host"][0], abs=1e-9)
        assert float(out["energy"] - ref["energy"]) == pytest.approx(-0.37, abs=1e-9)

    def test_the_charge_derivatives_are_not_negligible(self, w6):
        """Dropping d dq/dR and d q0/dR (the two Frechet contractions) moves a force by more
        than the FD tolerance above -- so that gate genuinely covers them."""
        out = w6(_batch([VACP]), compute_force=True)
        original = w6.scf_free_terms
        w6.scf_free_terms = lambda *a, **k: tuple(
            x.detach() if torch.is_tensor(x) else x for x in original(*a, **k))
        try:
            detached = w6(_batch([VACP]), compute_force=True)
        finally:
            w6.scf_free_terms = original
        assert float((out["energy"] - detached["energy"]).abs()) < 1e-12
        assert float((out["forces"] - detached["forces"]).abs().max()) > 1e-4


class TestO1FusedAssembly:
    """O1: the hopping MLP deduplicated by unordered node pair. Its descriptor is symmetric
    in (i, j) and carries no `r`, so it is a function of the node pair alone while the edge
    list holds both directions and every periodic image -- 9122 edges collapse to 2872 pairs
    on a 79-atom frame, and the float64 GEMM over a 2 x 512-wide input was the whole assembly
    cost."""

    def test_integrals_fast_matches_the_reference(self, w6):
        batch = _batch([VACP])
        out = w6.base_forward(w6._trunk_data(dict(batch)), compute_force=False)
        scalars, _ = w6.features(out["node_feats"])
        scalars = scalars.to(torch.float64)
        species = batch["node_attrs"].argmax(dim=-1)
        src, dst = batch["edge_index"][0], batch["edge_index"][1]
        ev = batch["positions"][dst] - batch["positions"][src] + batch["shifts"]
        r = ev.norm(dim=-1).clamp_min(1e-9)
        sk = w6.h0.sk
        ref = sk.integrals(scalars[src], scalars[dst], r, species[src], species[dst])
        fast = sk.integrals_fast(scalars, species, src, dst, r)
        assert float((fast - ref).abs().max()) < 1e-12 * max(float(ref.abs().max()), 1.0)

    def test_the_deduplication_is_real(self, w6):
        """If the edge list were already unique per pair the optimisation would be a no-op."""
        batch = _batch([VACP])
        src, dst = batch["edge_index"][0], batch["edge_index"][1]
        n = int(batch["positions"].shape[0])
        pairs = torch.unique(torch.minimum(src, dst) * n + torch.maximum(src, dst)).numel()
        assert pairs < 0.6 * src.numel()


# --------------------------------------------------------------------------- O3


def _o3(seed=0, n_radial=4):
    """O3: the decoupled head. Nothing from the base enters `H0` -- the descriptor is the
    head's own (Bessel x taper, species-resolved `l = 0`), the hopping modulation is gone.
    Built with the same off-zero `alpha`/`beta`/`vector_mix` as `_w6`, so the rank-1 and
    rank-2 on-site blocks are live and their derivatives are exercised."""
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, decoupled=True, n_radial=n_radial,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
        m.h0.descriptor.c1.normal_(0.0, 0.5)
        m.h0.descriptor.c2.normal_(0.0, 0.5)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    return m


@pytest.fixture(scope="module")
def o3():
    return _o3()


class TestO3Decoupled:
    def test_the_head_reads_nothing_from_the_base(self):
        """Perturbing every base parameter must leave the HEAD energy bit-identical (the
        base energy of course moves); the head is decoupled by construction, not by
        a small number."""
        m = _o3(seed=2)
        batch = _batch([VACP])
        before = float(m(dict(batch), compute_force=False)["head_energy"])
        with torch.no_grad():
            for p in m.base.parameters():
                p.add_(0.05 * torch.randn_like(p))
        after = float(m(dict(batch), compute_force=False)["head_energy"])
        assert after == before

    def test_no_feature_statistics_are_required(self, o3):
        assert o3.require_feature_stats is False
        assert not bool(o3.h0.sk.feat_stats_set)
        assert float(o3(_batch([VACP]), compute_force=False)["energy"]) == pytest.approx(
            float(o3(_batch([VACP]), compute_force=False)["energy"]), abs=0.0)

    def test_neutral_null_is_still_the_base(self, o3):
        out = o3(_batch([VAC0]), compute_force=True)
        assert out["short_circuit"] and float(out["head_energy"].abs().max()) == 0.0

    def test_batched_equals_per_graph(self, o3):
        one = o3(_batch([VACP]), compute_force=True)
        two = o3(_batch([VACP, VACX]), compute_force=True)
        assert float(two["energy"][0] - one["energy"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float((two["forces"][:len(VACP)] - one["forces"]).abs().max()) < 1e-9

    def test_forces_against_central_differences(self, o3):
        out = o3(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (7, 2), (20, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(o3(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(o3(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=3e-6)

    def test_stress_against_strain_differences(self, o3):
        out = o3(_batch([VACP]), compute_force=False, compute_stress=True)
        volume = VACP.get_volume()
        h = 1e-5
        for i, j in ((0, 0), (0, 2)):
            e = []
            for sign in (1.0, -1.0):
                strained = VACP.copy()
                eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = sign * h
                strained.set_cell(np.array(VACP.get_cell()) @ (np.eye(3) + eps), scale_atoms=True)
                e.append(float(o3(_batch([strained]), compute_force=False)["energy"]))
            fd = (e[0] - e[1]) / (2 * h) / volume
            expected = float(out["stress"][0, i, j] + (out["stress"][0, j, i] if i != j else 0.0))
            assert expected == pytest.approx(fd, abs=1e-7)

    def test_training_and_inference_forces_agree(self, o3):
        batch = _batch([VACP])
        a = o3(dict(batch), training=False, compute_force=True)["forces"]
        b = o3(dict(batch), training=True, compute_force=True)["forces"]
        assert float((a - b.detach()).abs().max()) < 1e-9

    def test_the_analytic_force_route_matches_the_autograd_route(self, o3):
        """The closed-form Hellmann-Feynman contraction against the cotangent route it
        replaces -- the same model, the same frame, two derivative routes. FD (above) bounds
        both at 3e-6; this one has to be exact."""
        assert o3.analytic_head_forces is True
        batch = _batch([VACP, VACX])
        fast = o3(dict(batch), compute_force=True)
        o3.analytic_head_forces = False
        try:
            ref = o3(dict(batch), compute_force=True)
        finally:
            o3.analytic_head_forces = True
        assert float((fast["energy"] - ref["energy"]).abs().max()) < 1e-12
        assert float((fast["forces"] - ref["forces"]).abs().max()) < 1e-9
        assert float(ref["forces"].abs().max()) > 1e-3        # not a comparison of zeros

    def test_the_analytic_route_forms_no_density_matrix(self, o3):
        """The point of running the head under `no_grad`: `two_fillings` takes its frontier
        path, so the [4n, 4n] density is never built."""
        seen = []
        import mace.modules.dscc.scf as _scf
        real = _scf.two_fillings

        def spy(*a, **k):
            out = real(*a, **k)
            seen.append(out._dP is not None)      # the property BUILDS it; ask the field
            return out
        _scf.two_fillings, mdl = spy, __import__("mace.modules.dscc.model", fromlist=["x"])
        mdl_two = mdl.two_fillings
        mdl.two_fillings = spy
        try:
            o3(_batch([VACP]), compute_force=True)
        finally:
            _scf.two_fillings, mdl.two_fillings = real, mdl_two
        assert seen and not any(seen)

    def test_the_per_graph_path_matches_the_batched_one(self, o3):
        """A batch that mixes a charged frame with a REFERENCE one leaves the uniform path
        and takes the per-graph one, which slices the features `H0` is handed. The decoupled
        head has none, and `None` is not sliceable -- this is the crash that took down the
        first O3 launch at its first evaluation batch."""
        one = o3(_batch([VACP]), compute_force=True)
        mixed = o3(_batch([VACP, VAC0]), compute_force=True)
        assert not mixed["short_circuit"]
        assert float(mixed["energy"][0] - one["energy"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float((mixed["forces"][:len(VACP)] - one["forces"]).abs().max()) < 1e-9
        assert float(mixed["head_energy"][1]) == 0.0        # the reference graph adds nothing


# ------------------------------------------------------------------------- O3.1


def _o31(seed=0, n_radial=4):
    """O3.1: O3 with A1.1's per-species standardisation of its OWN `l = 0` descriptor, and
    the readout final layer at A1.1's registered 0.1. One change of substance from O3 --
    what the readouts are fed -- plus the init scale that registration pairs with it."""
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, decoupled=True, n_radial=n_radial,
                 standardise_inputs=True, readout_init_scale=0.1,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
        m.h0.descriptor.c1.normal_(0.0, 0.5)
        m.h0.descriptor.c2.normal_(0.0, 0.5)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


class TestO31Standardised:
    def test_the_statistics_are_the_heads_own_descriptor(self):
        """Not the base's block-0 invariants: the buffers are `n_radial x n_species` wide and
        what goes into them is what the readouts actually see."""
        m = _o31()
        assert bool(m.h0.sk.feat_stats_set)
        assert m.h0.sk.feat_mean.shape[1] == 4 * 3          # n_radial x n_species
        assert m.readout_scalars(_batch([VACP])).shape[1] == m.h0.sk.feat_mean.shape[1]
        # On the ensemble the statistics were frozen from, the standardised descriptor has
        # zero mean and unit sd per species on every live channel -- which is the whole of
        # what A1.1 does, and what O3's raw input does not have.
        rows, sps = [], []
        for s_ in (11, 12, 13):
            b = _batch([_frame(_perovskite(rattle=0.05, seed=s_), [0, 0, 0, 0], 0)])
            sp = b["node_attrs"].argmax(-1)
            rows.append(m.h0.sk.standardise(m.readout_scalars(b).to(torch.float64), sp))
            sps.append(sp)
        std, sp = torch.cat(rows), torch.cat(sps)
        for z in range(3):
            sel = std[sp == z]
            live = m.h0.sk.feat_scale[z] > 0
            assert float(sel[:, live].mean(0).abs().max()) < 1e-8
            assert float((sel[:, live].std(0, unbiased=False) - 1.0).abs().max()) < 1e-6

    def test_standardisation_is_off_for_o3_and_refuses_to_be_frozen(self, o3):
        assert o3.standardise_inputs is False and o3.require_feature_stats is False
        with pytest.raises(RuntimeError):
            o3.set_feature_stats([_batch([VACP])])

    def test_a_head_without_its_statistics_refuses_to_run(self):
        m = MACEDSCC(_base(0), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                     route_b=True, scf_free=True, decoupled=True, n_radial=4,
                     standardise_inputs=True, kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
        with pytest.raises(RuntimeError):
            m(_batch([VACP]), compute_force=False)

    def test_forces_against_central_differences(self):
        m = _o31()
        out = m(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (7, 2)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(m(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(m(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=3e-6)

    def test_the_analytic_and_autograd_routes_still_agree(self):
        m = _o31()
        batch = _batch([VACP, VACX])
        fast = m(dict(batch), compute_force=True)
        m.analytic_head_forces = False
        try:
            ref = m(dict(batch), compute_force=True)
        finally:
            m.analytic_head_forces = True
        assert float((fast["forces"] - ref["forces"]).abs().max()) < 1e-9

    def test_the_readout_init_scale_is_recorded_and_applied(self):
        assert _o31().h0.sk.readout_init_scale == 0.1
        assert _o3().h0.sk.readout_init_scale == 0.05        # what O3 and W6 shipped with
