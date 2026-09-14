"""O3.2 gates: the three-body bond screening.

    S_ij^{(Z',l)} = (1/N_ref) sum_{k in Z'} g(r_ik) g(r_jk) P_l(cos theta_ikj)

registered in the tracker before the arm ran, with `theta_ikj` the angle AT k, `g` H0's own
taper at `r_s3 = 4.5`, `l = 0, 1, 2` species-resolved in `k`, and a zero-initialised linear
map into the head's EXISTING bounded modulation.

The entry gate is the first test: with `A = 0` the screened head is its unscreened parent BIT
FOR BIT, so any difference the arm produces is attributable to the one change.
"""
import numpy as np
import pytest
import torch

from mace.modules.dscc.kernels import KernelConfig
from mace.modules.dscc.legacy import SCREEN_NORM, BondScreening
from mace.modules.dscc.model import MACEDSCC
from tests.extensions.dscc.test_model import (R_CUT, VAC0, VACP, _base, _batch, _frame,
                                              _perovskite)
from tests.extensions.dscc.test_w6 import _o31

torch.set_default_dtype(torch.float64)

R_S3 = 4.5


def _o32(seed=0, n_radial=4, a_scale=0.0):
    """O3.1 plus the screening. `a_scale = 0` is the shipped initialisation (the arm starts at
    its parent); the gates that need a live modulation pass a nonzero one."""
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, decoupled=True, n_radial=n_radial,
                 standardise_inputs=True, readout_init_scale=0.1,
                 screened=True, r_s3=R_S3,
                 kernel=KernelConfig(regime="B", r_g=1.0, r_s=5.0))
    with torch.no_grad():
        m.h0.vector_mix.normal_(0.0, 0.3)
        m.h0.alpha.fill_(0.5)
        m.h0.beta.fill_(0.5)
        m.h0.descriptor.c1.normal_(0.0, 0.5)
        m.h0.descriptor.c2.normal_(0.0, 0.5)
        if a_scale:
            m.h0.sk.screen.A.normal_(0.0, a_scale)
    m.set_pristine_reference([_batch([_frame(_perovskite(rattle=0.0), [0, 0, 0, 0], 0)])])
    m.set_feature_stats([_batch([_frame(_perovskite(rattle=0.05, seed=s), [0, 0, 0, 0], 0)])
                         for s in (11, 12, 13)])
    return m


def _triangle(d_ij=3.0, k=(1.4, 1.9, 0.3), species=(0, 0, 1)):
    """Three atoms, all six directed edges, no periodic images: `(edge_index, vectors, sp)`."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [d_ij, 0.0, 0.0], list(k)], dtype=torch.float64)
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    ei = torch.tensor(pairs, dtype=torch.long).T
    return ei, pos[ei[1]] - pos[ei[0]], torch.tensor(species, dtype=torch.long)


class TestTheForm:
    """`S` itself, against the closed form written in the tracker."""

    def test_a_triangle_matches_the_closed_form(self):
        screen = BondScreening(3, R_S3)
        ei, vec, sp = _triangle()
        S = screen.invariants(ei, vec, sp)
        # Edge 0 -> 1; the only screener is atom 2 (species 1 -> channels 3, 4, 5).
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.4, 1.9, 0.3]], dtype=torch.float64)
        a, b = pos[0] - pos[2], pos[1] - pos[2]          # k -> i and k -> j
        r_ik, r_jk = float(a.norm()), float(b.norm())
        cos = float((a * b).sum() / (r_ik * r_jk))
        g = float(screen.taper(torch.tensor(r_ik)) * screen.taper(torch.tensor(r_jk))) / SCREEN_NORM
        expect = [g, g * cos, g * (1.5 * cos ** 2 - 0.5)]
        row = S[pairs_index(ei, 0, 1)]
        assert [float(x) for x in row[3:6]] == pytest.approx(expect, abs=1e-14)
        assert float(row[:3].abs().max()) == 0.0          # no species-0 screener of this bond
        assert float(row[6:].abs().max()) == 0.0

    def test_the_angle_is_the_one_at_the_apex(self):
        """The sign trap: a flipped vector leaves `l = 0` and `l = 2` looking right and
        silently negates `l = 1`. A collinear i-k-j (angle at k = 180 deg) must give
        `P_1 = -1`, and a k far off the line must not."""
        screen = BondScreening(3, R_S3)
        ei, vec, sp = _triangle(d_ij=3.0, k=(1.5, 0.0, 0.0))     # k exactly between i and j
        S = screen.invariants(ei, vec, sp)
        row = S[pairs_index(ei, 0, 1)]
        assert float(row[4] / row[3]) == pytest.approx(-1.0, abs=1e-12)      # cos(180)
        assert float(row[5] / row[3]) == pytest.approx(1.0, abs=1e-12)       # P_2(-1) = 1

    def test_a_bond_longer_than_twice_the_cutoff_is_exactly_unscreened(self):
        """Not small: zero. No `k` can lie inside `r_s3` of both ends, by the triangle
        inequality, and the enumeration's `r_ij < 2 r_s3` filter is that statement."""
        screen = BondScreening(3, R_S3)
        ei, vec, sp = _triangle(d_ij=9.5, k=(1.0, 1.0, 0.0))
        S = screen.invariants(ei, vec, sp)
        assert float(S[pairs_index(ei, 0, 1)].abs().max()) == 0.0
        assert float(S[pairs_index(ei, 0, 2)].abs().max()) == 0.0    # i-k has no third atom
        assert float(S.abs().max()) == 0.0

    def test_the_third_atom_being_the_far_end_is_not_a_triplet(self):
        """`f == e` would be `k = j`: no triangle, `r_jk = 0`, and a 1/0 in the cosine."""
        screen = BondScreening(3, R_S3)
        ei, vec, sp = _triangle()
        S = screen.invariants(ei, vec, sp)
        assert bool(torch.isfinite(S).all())
        # Two atoms alone have no triplet at all.
        ei2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        v2 = torch.tensor([[2.9, 0.0, 0.0], [-2.9, 0.0, 0.0]], dtype=torch.float64)
        S2 = screen.invariants(ei2, v2, torch.tensor([0, 1]))
        assert float(S2.abs().max()) == 0.0

    def test_the_species_channel_is_the_third_atom(self):
        screen = BondScreening(3, R_S3)
        for z_k, lo in ((0, 0), (1, 3), (2, 6)):
            ei, vec, sp = _triangle(species=(0, 0, z_k))
            row = screen.invariants(ei, vec, sp)[pairs_index(ei, 0, 1)]
            live = [i for i in range(9) if abs(float(row[i])) > 0.0]
            assert all(lo <= i < lo + 3 for i in live) and live


def pairs_index(edge_index, i, j):
    pairs = [(int(a), int(b)) for a, b in zip(edge_index[0], edge_index[1])]
    return pairs.index((i, j))


class TestHermiticity:
    """`S_ij = S_ji` and `A` symmetric in the species pair: the two things that keep `H`
    Hermitian BY CONSTRUCTION, which is the property `sk_block` exists to preserve."""

    def test_S_is_invariant_under_reversing_the_edge(self):
        m = _o32()
        batch = _batch([VACP])
        vec = m.head_edge_vectors(batch)
        sp = batch["node_attrs"].argmax(dim=-1)
        S = m.h0.sk.screen.invariants(batch["edge_index"], vec, sp)
        rev = _reverse_edges(batch)
        assert float((S - S[rev]).abs().max()) < 1e-12
        assert float(S.abs().max()) > 1e-3                  # not a comparison of zeros

    def test_the_pre_activation_is_invariant_under_reversing_the_edge(self):
        """`S` symmetric is not enough: `A[Z_i, Z_j]` must equal `A[Z_j, Z_i]`, and it does
        because the forward symmetrises it."""
        m = _o32(a_scale=0.4)
        batch = _batch([VACP])
        sp = batch["node_attrs"].argmax(dim=-1)
        pre = m.h0.sk.screen(batch["edge_index"], m.head_edge_vectors(batch), sp)
        assert float((pre - pre[_reverse_edges(batch)]).abs().max()) < 1e-12
        assert float(pre.abs().max()) > 1e-3
        # And the raw parameter is NOT symmetric -- the symmetry is imposed, not assumed.
        A = m.h0.sk.screen.A
        assert float((A - A.transpose(0, 1)).abs().max()) > 1e-3


def _reverse_edges(batch):
    """Index of the reverse of every edge: `(j, i, -shift)` for `(i, j, shift)`."""
    src, dst = batch["edge_index"][0].tolist(), batch["edge_index"][1].tolist()
    us = batch["unit_shifts"].round().to(torch.long).tolist()
    where = {(i, j, tuple(s)): e for e, (i, j, s) in enumerate(zip(src, dst, us))}
    out = [where[(j, i, tuple(-x for x in s))] for i, j, s in zip(src, dst, us)]
    return torch.tensor(out, dtype=torch.long)


class TestTheArmStartsAtItsParent:
    def test_zero_A_is_the_unscreened_head_bit_for_bit(self):
        """The entry gate. `pre = 0` -> `exp(eta tanh 0) = 1` exactly, so the screened head
        multiplies every hopping by a floating-point 1.0 and must reproduce O3.1 EXACTLY --
        not to a tolerance."""
        parent, child = _o31(seed=3), _o32(seed=3)
        missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
        assert [k for k in missing if "screen" not in k] == [] and list(unexpected) == []
        child.screened, child.r_s3 = True, R_S3          # the parent's extra state said False
        assert float(child.h0.sk.screen.A.abs().max()) == 0.0
        batch = _batch([VACP])
        assert float(parent(dict(batch), compute_force=False)["energy"]) == \
            float(child(dict(batch), compute_force=False)["energy"])
        # Forces on the SAME route: the parent's analytic kernel takes the frontier path and
        # never forms the density, which moves the energy in the last two digits -- a
        # difference of route, not of model, and not what this gate is about.
        parent.analytic_head_forces = False
        assert torch.equal(parent(dict(batch), compute_force=True)["forces"],
                           child(dict(batch), compute_force=True)["forces"])

    def test_removing_the_module_changes_nothing_at_zero_A(self):
        """Deleting the module is what the analytic gate reads, so hold both runs on the
        autograd route -- otherwise this compares two force ROUTES, not two models."""
        m = _o32(seed=4)
        m.analytic_head_forces = False
        batch = _batch([VACP])
        with_screen = m(dict(batch), compute_force=True)
        m.h0.sk.screen = None
        without = m(dict(batch), compute_force=True)
        assert float(with_screen["energy"]) == float(without["energy"])
        assert torch.equal(with_screen["forces"], without["forces"])

    def test_a_live_A_actually_moves_the_energy(self):
        """The complement: the gates above would also pass on a screening wired to nothing."""
        batch = _batch([VACP])
        e0 = float(_o32(seed=5)(dict(batch), compute_force=False)["head_energy"])
        e1 = float(_o32(seed=5, a_scale=0.4)(dict(batch), compute_force=False)["head_energy"])
        assert abs(e1 - e0) > 1e-3


class TestDerivatives:
    def test_forces_match_finite_differences(self):
        m = _o32(seed=6, a_scale=0.4)
        out = m(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (7, 2), (20, 1)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(m(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(m(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(
                -(e_plus - e_minus) / (2 * h), abs=3e-6)

    def test_the_analytic_route_is_refused(self):
        """Registered, not hidden: `h0_position_gradient` writes `dH0/dR` assuming `t_ij`
        depends on `r_ij` alone, which the screening breaks. A screened head must take the
        autograd cotangent route -- and its unscreened parent must still take the fast one,
        or this test would pass on a model that never had the route to begin with."""
        import mace.modules.dscc.model as mdl
        real = mdl.h0_position_gradient
        calls = []

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)
        mdl.h0_position_gradient = spy
        try:
            _o32(seed=7, a_scale=0.4)(_batch([VACP]), compute_force=True)
            assert calls == []
            _o31(seed=7)(_batch([VACP]), compute_force=True)
            assert calls == [1]
        finally:
            mdl.h0_position_gradient = real

    def test_the_refusal_survives_loading_an_unscreened_state_dict(self):
        """`screened` travels in the extra state, so loading O3.1's `state_dict` into a
        screened model CLEARS the flag while `sk.screen` stays live. If the analytic gate
        asked the flag, that model would take the closed-form route on a screened
        Hamiltonian: wrong forces, no error. It asks the module."""
        import mace.modules.dscc.model as mdl
        m = _o32(seed=17, a_scale=0.4)
        m.load_state_dict(_o31(seed=17).state_dict(), strict=False)
        assert m.screened is False and m.h0.sk.screen is not None    # the inconsistent state
        real, calls = mdl.h0_position_gradient, []
        mdl.h0_position_gradient = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            out = m(_batch([VACP]), compute_force=True)
        finally:
            mdl.h0_position_gradient = real
        assert calls == []
        h = 1e-4
        plus, minus = VACP.copy(), VACP.copy()
        plus.positions[1, 0] += h
        minus.positions[1, 0] -= h
        fd = -(float(m(_batch([plus]), compute_force=False)["energy"])
               - float(m(_batch([minus]), compute_force=False)["energy"])) / (2 * h)
        assert float(out["forces"][1, 0]) == pytest.approx(fd, abs=3e-6)

    def test_training_and_inference_forces_agree(self):
        m = _o32(seed=8, a_scale=0.4)
        batch = _batch([VACP])
        a = m(dict(batch), training=False, compute_force=True)["forces"]
        b = m(dict(batch), training=True, compute_force=True)["forces"]
        assert float((a - b.detach()).abs().max()) < 1e-9

    def test_the_parameter_gets_a_gradient(self):
        m = _o32(seed=9, a_scale=0.4)
        out = m(_batch([VACP]), training=True, compute_force=True)
        (out["energy"].sum() + out["forces"].sum()).backward()
        g = m.h0.sk.screen.A.grad
        assert g is not None and float(g.abs().max()) > 0.0


class TestBatching:
    def test_the_batched_path_matches_the_per_graph_one(self):
        """A triplet never crosses a graph because no edge does -- but the screening is the
        first thing in the head to index a THIRD node, so the batched path gets its own gate."""
        m = _o32(seed=10, a_scale=0.4)
        other = _frame(_perovskite(remove_cl=1, rattle=0.05, seed=4), [0, 0, 1, 0], 1)
        one = m(_batch([VACP]), compute_force=True)
        two = m(_batch([other]), compute_force=True)
        both = m(_batch([VACP, other]), compute_force=True)
        assert bool(both["diagnostics"].get("batched"))
        assert float(both["energy"][0] - one["energy"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float(both["energy"][1] - two["energy"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float((both["forces"][:len(VACP)] - one["forces"]).abs().max()) < 1e-9

    def test_a_mixed_batch_takes_the_per_graph_path(self):
        """The crash that took down the first O3 launch, re-run for this arm."""
        m = _o32(seed=11, a_scale=0.4)
        one = m(_batch([VACP]), compute_force=True)
        mixed = m(_batch([VACP, VAC0]), compute_force=True)
        assert not mixed["short_circuit"]
        assert float(mixed["energy"][0] - one["energy"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float((mixed["forces"][:len(VACP)] - one["forces"]).abs().max()) < 1e-9

    def test_different_sized_frames_in_one_batch_screen_independently(self):
        """Sizes differ, so the head takes the per-graph path; the point is that neither
        graph's `S` is contaminated by the other's atoms."""
        m = _o32(seed=12, a_scale=0.4)
        small = _frame(_perovskite(remove_cl=0, rattle=0.05, seed=5), [0, 0, 1, 0], 1)
        big = _frame(_perovskite(rattle=0.05, seed=6), [0, 0, 1, 0], 1)
        alone = [float(m(_batch([f]), compute_force=False)["energy"]) for f in (small, big)]
        together = m(_batch([small, big]), compute_force=False)["energy"]
        assert [float(x) for x in together] == pytest.approx(alone, abs=1e-9)


class TestItStaysDecoupled:
    def test_the_head_still_reads_nothing_from_the_base(self):
        """O3's defining property, re-gated: the screening is built from GEOMETRY, so
        perturbing every base parameter must leave the head energy bit-identical."""
        m = _o32(seed=13, a_scale=0.4)
        batch = _batch([VACP])
        before = float(m(dict(batch), compute_force=False)["head_energy"])
        with torch.no_grad():
            for p in m.base.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        assert float(m(dict(batch), compute_force=False)["head_energy"]) == before

    def test_the_saturation_report_reads_the_screening(self):
        """The registered diagnostic for this arm: the modulation must not be reported as
        learned while sitting at its stop -- the mistake O3 made with the on-site term."""
        m = _o32(seed=14, a_scale=0.4)
        batch = _batch([VACP])
        rep = m.h0.saturation(m.readout_scalars(batch), batch["node_attrs"].argmax(dim=-1),
                              batch["edge_index"], None, edge_vector=m.head_edge_vectors(batch))
        assert "screen" in rep and "screen_by_species" in rep
        assert 0.0 <= rep["screen"] <= 1.0
        assert rep["screen_pre_abs_max"] >= rep["screen_pre_abs_p95"] > 0.0
        assert rep["hop_not_applicable"] == 1.0        # `sk.hop` still does not run

    def test_capture_refuses_a_screened_head(self):
        """Its triplet list is data dependent (`nonzero`), so a replay would reuse the
        captured frame's triplets. Registered as a follow-up, refused in the meantime."""
        from mace.modules.dscc.capture import GraphedStep
        with pytest.raises(ValueError, match="screened"):
            GraphedStep(_o32(seed=15), _batch([VACP]))

    def test_the_sparse_assembly_refuses_a_screened_head(self):
        """It evaluates the two-body `integrals` and would silently drop the screening."""
        from mace.modules.dscc import sparse
        m = _o32(seed=16, a_scale=0.4)
        batch = _batch([VACP])
        vec = m.head_edge_vectors(batch)
        species = batch["node_attrs"].argmax(dim=-1)
        scalars, vectors, _ = m.h0.geometric_inputs(species, batch["edge_index"], vec,
                                                    int(species.shape[0]))
        with pytest.raises(NotImplementedError, match="screening"):
            sparse.csr_hamiltonian(m.h0, scalars, vectors, species, batch["edge_index"], vec)


# ------------------------------------------------------------------- O3.3


def _o33(seed=0, n_radial=4):
    """O3.3: O3.1 plus the correlation-2 invariants of the head's OWN A-tensors. One change
    from O3.1 -- what the on-site readout is fed -- and nothing from the base."""
    m = MACEDSCC(_base(seed), r_cut=R_CUT, directional=True, hidden=16, coupling=False,
                 route_b=True, scf_free=True, decoupled=True, n_radial=n_radial,
                 standardise_inputs=True, readout_init_scale=0.1, correlation2=True,
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


class TestO33Correlation2:
    def test_the_width_is_the_registered_count(self):
        """n0 linear + n0(n0+1)/2 pairwise l=0 + nr(nr+1)/2 from A1.A1 + 1 from ||A2||^2."""
        m = _o33(n_radial=4)
        d = m.h0.descriptor
        n0, nr = 4 * 3, 4
        assert d.scalar_width() == n0 + n0 * (n0 + 1) // 2 + nr * (nr + 1) // 2 + 1
        assert m.readout_scalars(_batch([VACP])).shape[1] == d.scalar_width()
        assert m.h0.sk.feat_mean.shape[1] == d.scalar_width()

    def test_the_invariants_are_the_stated_contractions(self):
        """Checked against an explicit loop, not against another vectorised expression."""
        m = _o33(seed=3)
        d = m.h0.descriptor
        batch = _batch([VACP])
        species = batch["node_attrs"].argmax(dim=-1)
        vec = m.head_edge_vectors(batch)
        A0, v, Q = d(species, batch["edge_index"], vec, int(species.shape[0]))
        # rebuild the raw A0 by turning the flag off on a copy of the same module
        d.correlation2 = False
        try:
            raw, v2, Q2 = d(species, batch["edge_index"], vec, int(species.shape[0]))
        finally:
            d.correlation2 = True
        assert torch.equal(v, v2) and torch.equal(Q, Q2)
        n0 = raw.shape[1]
        assert torch.allclose(A0[:, :n0], raw, atol=0)          # linear block unchanged
        k = n0
        for a in range(n0):
            for b in range(a, n0):
                assert torch.allclose(A0[:, k], raw[:, a] * raw[:, b], atol=1e-12)
                k += 1
        for a in range(v.shape[1]):
            for b in range(a, v.shape[1]):
                assert torch.allclose(A0[:, k], (v[:, a] * v[:, b]).sum(-1), atol=1e-12)
                k += 1
        assert torch.allclose(A0[:, k], (Q * Q).sum(dim=(-1, -2)), atol=1e-12)
        assert k + 1 == A0.shape[1]

    def test_it_still_reads_nothing_from_the_base(self):
        m = _o33(seed=5)
        batch = _batch([VACP])
        before = float(m(dict(batch), compute_force=False)["head_energy"])
        with torch.no_grad():
            for p in m.base.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        assert float(m(dict(batch), compute_force=False)["head_energy"]) == before

    def test_forces_match_finite_differences(self):
        m = _o33(seed=6)
        out = m(_batch([VACP]), compute_force=True)
        h = 1e-4
        for atom, comp in ((1, 0), (7, 2)):
            plus, minus = VACP.copy(), VACP.copy()
            plus.positions[atom, comp] += h
            minus.positions[atom, comp] -= h
            e_plus = float(m(_batch([plus]), compute_force=False)["energy"])
            e_minus = float(m(_batch([minus]), compute_force=False)["energy"])
            assert float(out["forces"][atom, comp]) == pytest.approx(
                -(e_plus - e_minus) / (2 * h), abs=3e-6)

    def test_the_new_channels_are_alive_after_standardisation(self):
        """A quadratic channel that is constant across the ensemble would be standardised to
        nothing and would add no capacity. Count the dead ones."""
        m = _o33(seed=7)
        batch = _batch([VACP])
        raw = m.readout_scalars(batch)
        species = batch["node_attrs"].argmax(dim=-1)
        std = m.h0.sk.standardise(raw, species)
        assert bool(torch.isfinite(std).all())
        n0 = m.h0.descriptor.n_radial * m.h0.descriptor.num_elements
        quad = std[:, n0:]
        dead = int((quad.std(dim=0) < 1e-10).sum())
        assert dead == 0, f"{dead} of {quad.shape[1]} correlation-2 channels are dead"

    def test_a_mixed_batch_matches_per_graph(self):
        m = _o33(seed=8)
        one = m(_batch([VACP]), compute_force=True)
        mixed = m(_batch([VACP, VAC0]), compute_force=True)
        assert not mixed["short_circuit"]
        assert float(mixed["energy"][0] - one["energy"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float((mixed["forces"][:len(VACP)] - one["forces"]).abs().max()) < 1e-9

    def test_the_analytic_route_still_applies(self):
        """Unlike O3.2, this arm does not touch the hoppings, so the closed-form force kernel
        is still valid and must still be taken."""
        import mace.modules.dscc.model as mdl
        real, calls = mdl.h0_position_gradient, []
        mdl.h0_position_gradient = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            fast = _o33(seed=9)(_batch([VACP]), compute_force=True)
        finally:
            mdl.h0_position_gradient = real
        assert calls == [1]
        m = _o33(seed=9)
        m.analytic_head_forces = False
        ref = m(_batch([VACP]), compute_force=True)
        assert float((fast["forces"] - ref["forces"]).abs().max()) < 1e-9
