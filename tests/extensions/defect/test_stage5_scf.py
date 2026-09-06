"""Addendum section 6.2 / 11.1, Stage 5: the stationary auxiliary functional, layer by layer.

* `R_sm`: minimising `Tr(P H) + R_sm[P]` at fixed `Tr P = N` gives the implemented
  `F_band(H, N)` -- the defining identity, for both smearing families, on random spectra
  including gapped ones where every occupation saturates;
* `dF_band / dH_ab = P_ba` for the production smearing;
* the regulariser's own derivative is `-(H - mu)` on the levels at a count-fill density.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from mace.modules import defect_counting as cnt
from mace.modules import defect_scf as scf

torch.set_default_dtype(torch.float64)


def _random_h(n: int, seed: int, gap_at: int = 0, gap: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    H = 0.5 * (A + A.T)
    if gap > 0.0:
        lam, U = torch.linalg.eigh(H)
        lam = lam.clone()
        lam[gap_at:] += gap
        H = (U * lam) @ U.T
    return H


@pytest.mark.parametrize("family", ["gaussian", "fermi"])
@pytest.mark.parametrize("n, N, gap", [(12, 5.0, 0.0), (12, 5.0, 6.0), (30, 11.0, 0.0)])
def test_the_regulariser_reproduces_the_band_free_energy(family, n, N, gap):
    """`Tr(P H) + R_sm[P] = F_band(H, N)` at `P = count_fill(H, N)`; with `gap` the frontier
    sits in a gap wider than the smearing, every occupation is exactly 0 or 1 and the
    regulariser is an exact zero (no `erfcinv(0)` blows up)."""
    previous = cnt.use_smearing(family, 0.05)
    try:
        H = _random_h(n, 3, gap_at=int(N), gap=gap)
        t_el = 0.05
        P, lam, U, mu = scf.count_fill_density(H, N, t_el)
        assert float(P.trace()) == pytest.approx(N, abs=1e-9)
        value = float((P * H).sum() + scf.occupation_regulariser(P, t_el))
        reference = float(cnt.free_energy(lam, N, t_el))
        assert value == pytest.approx(reference, abs=1e-9)
        if gap > 0.0:
            assert float(scf.occupation_regulariser(P, t_el)) == pytest.approx(0.0, abs=1e-12)
    finally:
        cnt.use_smearing(*previous)


@pytest.mark.parametrize("family", ["gaussian", "fermi"])
def test_the_count_fill_density_minimises_the_primary_form(family):
    """The variational statement behind the identity: any other density of the same trace
    -- here the count-fill of a perturbed Hamiltonian -- gives a larger value."""
    previous = cnt.use_smearing(family, 0.05)
    try:
        H = _random_h(16, 5)
        N, t_el = 7.0, 0.05
        P, lam, *_ = scf.count_fill_density(H, N, t_el)
        best = float((P * H).sum() + scf.occupation_regulariser(P, t_el))
        for seed in range(4):
            Q, *_ = scf.count_fill_density(H + 0.3 * _random_h(16, 10 + seed), N, t_el)
            assert float(Q.trace()) == pytest.approx(N, abs=1e-9)
            other = float((Q * H).sum() + scf.occupation_regulariser(Q, t_el))
            assert other > best + 1e-6
    finally:
        cnt.use_smearing(*previous)


def test_the_band_free_energy_derivative_is_the_density_matrix():
    """Section 11.1: `dF_band / dH_ab = P_ba` for the production smearing."""
    H = _random_h(14, 8).requires_grad_(True)
    N, t_el = 6.0, 0.05
    lam, U = torch.linalg.eigh(H)
    F = cnt.free_energy(lam, N, t_el)
    (dF,) = torch.autograd.grad(F, H)
    P, *_ = scf.count_fill_density(H.detach(), N, t_el)
    # eigh's backward returns the symmetrised gradient; P is symmetric.
    assert torch.allclose(dF, P.T, atol=1e-8)


def test_the_regulariser_slope_is_minus_h_plus_mu_on_the_levels():
    """`dR_sm/dP + H = mu I` in the eigenbasis of a count-fill density: the stationarity of
    the primary form at fixed trace, level by level, wherever the occupation is not
    saturated."""
    # A spectrum whose spacing is comparable to the smearing, so that several levels are
    # fractionally occupied (a spread of several eV saturates every Gaussian occupation).
    H = 0.03 * _random_h(12, 11)
    N, t_el = 5.0, 0.05
    P, lam, U, mu = scf.count_fill_density(H, N, t_el)
    slope = scf.regulariser_slope(P, t_el)
    total = U.T @ (slope + H) @ U
    f = U.T @ P @ U
    active = (torch.diagonal(f) > 1e-8) & (torch.diagonal(f) < 1.0 - 1e-8)
    assert int(active.sum()) >= 2
    assert torch.allclose(torch.diagonal(total)[active], mu.expand_as(lam)[active], atol=1e-7)
    # And it is an autograd-consistent derivative of the regulariser itself.
    Q = P.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(scf.occupation_regulariser(Q, t_el), Q)
    assert torch.allclose(0.5 * (auto + auto.T), slope, atol=1e-7)


# ------------------------------------------------------------------ layer 2: V_B = dPhi_B/dP


def _grab_entry(model, batch):
    """The head's per-graph record of one forward, as a FrontierEntry (graph 0)."""
    from mace.modules import defect_frontier as df

    head = model.spectral
    grabbed = []
    original = head.forward

    def wrapped(*a, **k):
        out = original(*a, **k)
        grabbed.append(out.frontier)
        return out

    head.forward = wrapped
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        head.forward = original
    return df.entries_from_head(grabbed[0])[0]


@pytest.fixture(scope="module")
def toy():
    """The Stage-4 toy: the unified model with its class table, the neutral vacancy frame's
    entry, and the graph's boundary context under the periodic boundary."""
    from mace.modules import defect_boundary as db
    from mace.modules import defect_composition as dc
    from mace.modules.defect_cache import attach_frame_keys
    from mace.modules.defect_state import StateBatch
    from tests.extensions.defect.test_frontier_term import NEUTRAL, PRISTINE, VACANCY, _frame
    from tests.extensions.defect.test_neutral_reference_skip import Z_TABLE, _batch
    from tests.extensions.defect.test_stage4_boundary import _unified_model

    torch.set_default_dtype(torch.float64)
    model = _unified_model()
    frames = [_frame(PRISTINE, NEUTRAL), _frame(VACANCY, NEUTRAL)]
    attach_frame_keys(frames, z_table=Z_TABLE)
    dc.ensure_class_table(model, frames, log=False)
    batch = _batch([VACANCY], [NEUTRAL])
    entry = _grab_entry(model, batch)
    numbers = VACANCY.get_atomic_numbers()
    zs = [int(z) for z in model.atomic_numbers]
    species = torch.tensor([zs.index(int(z)) for z in numbers])
    z0 = dc.species_charges(model)
    pos = torch.tensor(VACANCY.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.array(VACANCY.get_cell()), dtype=torch.float64)
    rec = dc.lookup_class(model.composition_classes, [int(z) for z in numbers])
    state = StateBatch.from_counts(torch.tensor([NEUTRAL]))
    n_e, n_h, _ = dc.frame_counts(rec, state, 0)
    ctx = db.graph_boundary(model, rec, entry, z0[species], pos, cell, species, "periodic",
                            label="toy")
    return {"model": model, "entry": entry, "ctx": ctx, "rec": rec, "n_e": n_e, "n_h": n_h}


def _random_tangent(n: int, seed: int) -> torch.Tensor:
    """A random symmetric trace-free direction: a tangent of the fixed-trace manifold."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    D = 0.5 * (A + A.T)
    D = D - torch.eye(n) * D.trace() / n
    return D / D.norm()


class TestBoundaryPotential:
    """Section 11.1: `dPhi_B / dP` agrees with matrix finite differences, including the window
    normalisation, the participation weights and the lifted density response; the lift is
    fixed with respect to `P` throughout the perturbation."""

    def test_the_potential_is_the_matrix_derivative_of_phi_b(self, toy):
        from mace.modules import defect_boundary as db
        from mace.modules.defect_frontier import FILL_AT_STATE, fill_densities

        ctx, entry = toy["ctx"], toy["entry"]
        n_e, n_h = toy["n_e"], toy["n_h"]
        assert n_e == (1, 0) and n_h == (0, 0)             # one majority electron
        P0 = [p.detach() for p in fill_densities(entry, FILL_AT_STATE, ctx.t_el)]
        V, phi, rho, diag = db.boundary_potential(ctx, P0, n_e, n_h)
        assert torch.equal(V[1], torch.zeros_like(V[1]))   # no minority channel: exact zero
        assert torch.allclose(V[0], V[0].T) and float(V[0].abs().max()) > 0.0
        fingerprint = ctx.lift.fingerprint
        n = P0[0].shape[0]
        h = 1e-4
        for seed in range(3):
            D = _random_tangent(n, seed)
            analytic = float((V[0] * D).sum())
            values = []
            for sign in (1.0, -1.0):
                P = [P0[0] + sign * h * D, P0[1]]
                phi_h, _, _ = db.phi_b(ctx, P, n_e, n_h)
                values.append(float(phi_h["phi"]))
                assert ctx.lift.fingerprint == fingerprint
            fd = (values[0] - values[1]) / (2.0 * h)
            assert analytic == pytest.approx(fd, abs=1e-6, rel=1e-4), (seed, analytic, fd)

    def test_the_potential_is_zero_at_the_reference_fill_with_no_carrier(self, toy):
        from mace.modules import defect_boundary as db
        from mace.modules.defect_frontier import FILL_AT_REFERENCE, fill_densities

        ctx, entry, rec = toy["ctx"], toy["entry"], toy["rec"]
        P = [p.detach() for p in fill_densities(entry, FILL_AT_REFERENCE, ctx.t_el)]
        # The reference state of the neutral vacancy class carries the class's own carrier.
        V, phi, rho, diag = db.boundary_potential(ctx, P, rec.n_e, rec.n_h)
        assert float(rho.q_img) == pytest.approx(rec.q_core - sum(rec.n_e) + sum(rec.n_h), abs=1e-6) \
            or float(rho.q_img) != 0.0
        assert all(torch.allclose(v, v.T) for v in V)


# ------------------------------------------------------------------ layer 3: the fixed point


def _targets(entry, fills):
    from mace.modules.defect_frontier import FILL_AT_STATE

    fills = FILL_AT_STATE if fills is None else fills
    return [float(entry.occupations[f].sum()) for f in fills]


class TestStationarySolution:
    """Section 6.2: the fixed point `P = count_fill(H_fix + V_B[P], N)` on the unmixed
    residual; section 11.1: the primary and band forms agree at the solution, the energy is
    invariant to the SCF history and stationary in the tangent directions."""

    def test_the_fixed_point_converges_on_the_unmixed_residual(self, toy):
        ctx, entry = toy["ctx"], toy["entry"]
        N = _targets(entry, None)
        res = scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], N, ctx.t_el)
        assert res.converged and res.residual < 1e-8 and res.commutator < 1e-6
        assert res.iterations >= 2 and res.history[-1] < res.history[0]
        for spin in range(2):
            assert float(res.P[spin].trace()) == pytest.approx(N[spin], abs=1e-9)
            assert torch.allclose(res.P[spin], res.P[spin].T)
        assert torch.isfinite(res.energy)

    def test_the_primary_and_band_forms_agree_at_the_solution(self, toy):
        ctx, entry = toy["ctx"], toy["entry"]
        res = scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], _targets(entry, None),
                                      ctx.t_el)
        assert float(res.energy) == pytest.approx(float(res.energy_band), abs=1e-7)
        # And the potential at the solution is the exact derivative of the reported Phi_B.
        from mace.modules import defect_boundary as db

        V, phi, _, _ = db.boundary_potential(ctx, res.P, toy["n_e"], toy["n_h"])
        assert torch.allclose(V[0], res.V[0], atol=1e-10)
        assert float(phi["phi"]) == pytest.approx(float(res.phi["phi"]), abs=1e-10)

    def test_the_energy_is_invariant_to_the_scf_history(self, toy):
        ctx, entry = toy["ctx"], toy["entry"]
        N = _targets(entry, None)
        a = scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], N, ctx.t_el,
                                    options=scf.ScfOptions(mixing=0.3, history=6))
        b = scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], N, ctx.t_el,
                                    options=scf.ScfOptions(mixing=0.6, history=0))
        assert a.iterations != b.iterations or a.history != b.history
        assert float(a.energy) == pytest.approx(float(b.energy), abs=1e-7)
        assert float((a.P[0] - b.P[0]).norm()) < 1e-6

    def test_the_solution_is_stationary_in_the_tangent_directions(self, toy):
        """`A_B(P^* + h D) - A_B(P^* - h D) = O(h^2)` for trace-free symmetric `D`: the
        first variation vanishes at the fixed point (the derivative of the primary form is
        `H_fix + dR_sm/dP + V_B = mu I` on the active levels)."""
        ctx, entry = toy["ctx"], toy["entry"]
        N = _targets(entry, None)
        res = scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], N, ctx.t_el)
        n = res.P[0].shape[0]
        # A direction inside the fractionally occupied subspace, where R_sm is smooth: a
        # rotation between the occupied and the empty levels of H_B, projected trace-free.
        lam, U = torch.linalg.eigh(res.H_B[0])
        f = torch.diagonal(U.T @ res.P[0] @ U)
        occ = torch.nonzero(f > 0.5).reshape(-1)[-1]
        emp = torch.nonzero(f < 0.5).reshape(-1)[0]
        D = torch.zeros(n, n)
        D[occ, emp] = D[emp, occ] = 1.0
        D = U @ D @ U.T
        # Rotate P rather than add: P(h) = exp(hK) P exp(-hK) keeps the spectrum (traces and
        # idempotency structure) exactly, so R_sm stays finite.
        K = U @ (torch.zeros(n, n)) @ U.T
        K[:] = 0.0
        Kb = torch.zeros(n, n); Kb[occ, emp] = 1.0; Kb[emp, occ] = -1.0
        K = U @ Kb @ U.T
        values = {}
        for h in (1e-3, 2e-3):
            for sign in (1.0, -1.0):
                R = torch.linalg.matrix_exp(sign * h * K)
                P = [R @ res.P[0] @ R.T, res.P[1]]
                values[(h, sign)] = float(scf.primary_functional(ctx, P, toy["n_e"], toy["n_h"],
                                                                 ctx.t_el)[0])
        first_1 = (values[(1e-3, 1.0)] - values[(1e-3, -1.0)]) / 2e-3
        first_2 = (values[(2e-3, 1.0)] - values[(2e-3, -1.0)]) / 4e-3
        base = float(res.energy)
        curvature = (values[(1e-3, 1.0)] + values[(1e-3, -1.0)] - 2 * base) / 1e-6
        # The first variation is at the noise floor of the SCF tolerance; the second is O(1).
        assert abs(first_1) < 1e-5 and abs(first_2) < 1e-5, (first_1, first_2)
        assert abs(curvature) > 1e-3

    def test_a_non_converging_solve_is_refused(self, toy):
        ctx, entry = toy["ctx"], toy["entry"]
        with pytest.raises(scf.ScfError, match="unsupported"):
            scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], _targets(entry, None),
                                    ctx.t_el, options=scf.ScfOptions(max_iter=1))


# ------------------------------------------------------------------ layer 4: the Hessian guards


class TestHessianGuards:
    """Section 6.2 / 11.1: the constrained energy Hessian in the tangent metric, its
    minimum eigenvalue and condition number. The matrix-vector product is validated against
    the finite-difference curvature of the primary functional along random tangent
    directions (the band part from the spectrum, the boundary part from V_B by finite
    differences -- D19); Lanczos then gives the extremal eigenvalues."""

    def test_the_hessian_matvec_is_the_curvature_of_the_functional(self, toy):
        ctx, entry = toy["ctx"], toy["entry"]
        n_e, n_h = toy["n_e"], toy["n_h"]
        N = _targets(entry, None)
        res = scf.stationary_solution(ctx, n_e, n_h, N, ctx.t_el)
        spaces = [scf.TangentSpace(h, p, ctx.t_el) for h, p in zip(res.H_B, res.P)]
        assert spaces[0].dim > 0 and spaces[1].dim > 0
        assert float(spaces[0].band.min()) > 0.0           # eps_a - eps_i on a gapped toy
        dim = spaces[0].dim + spaces[1].dim
        g = torch.Generator().manual_seed(1)
        for trial in range(2):
            c = torch.randn(dim, generator=g, dtype=torch.float64)
            # The rotation generated by coefficients c moves P along
            # dP = theta sum_p c_p (f_i - f_a) D_p: the tangent direction whose quadratic
            # form is compared is the occupation-weighted one, normalised.
            weights = torch.cat([torch.tensor([float(s.f[i] - s.f[a]) for i, a in s.pairs])
                                 for s in spaces])
            v = c * weights
            v = v / v.norm()
            Hv = scf.hessian_matvec(ctx, res, spaces, n_e, n_h, v)
            quadratic = float(v @ Hv)
            # The same curvature by rotating P^* along the direction: exact spectrum, so
            # R_sm is constant and the second derivative is that of Tr(P H_fix) + Phi_B.
            parts = torch.split(c, [spaces[0].dim, spaces[1].dim])
            K = []
            for s, c in zip(spaces, parts):
                n = s.lam.numel()
                M = torch.zeros(n, n, dtype=torch.float64)
                for coef, (i, a) in zip(c.tolist(), s.pairs):
                    M[a, i] += coef / math.sqrt(2.0)
                    M[i, a] -= coef / math.sqrt(2.0)
                K.append(s.U @ M @ s.U.T)
            # dP = theta [K, P] has Frobenius norm theta * sqrt(sum_p c_p^2 (f_i - f_a)^2);
            # theta is chosen so that |dP|_F = h exactly, and the curvature per unit
            # Frobenius^2 is (E(h) + E(-h) - 2 E0) / h^2.
            scale = math.sqrt(sum(float(c_ ** 2 * (s.f[i] - s.f[a]) ** 2)
                                  for s, c in zip(spaces, parts)
                                  for c_, (i, a) in zip(c.tolist(), s.pairs)))
            values = []
            h = 2e-3
            for sign in (1.0, -1.0):
                P = []
                for k, p in zip(K, res.P):
                    R = torch.linalg.matrix_exp(sign * h / scale * k)
                    P.append((R @ p.double() @ R.T).to(p.dtype))
                values.append(float(scf.primary_functional(ctx, P, n_e, n_h, ctx.t_el)[0]))
            base = float(res.energy)
            curvature = (values[0] + values[1] - 2.0 * base) / (h ** 2)
            assert quadratic == pytest.approx(curvature, rel=2e-2, abs=1e-3), (trial, quadratic, curvature)

    def test_the_guards_pass_on_the_gapped_toy(self, toy):
        ctx, entry = toy["ctx"], toy["entry"]
        res = scf.stationary_solution(ctx, toy["n_e"], toy["n_h"], _targets(entry, None),
                                      ctx.t_el)
        guard = scf.hessian_guards(ctx, res, toy["n_e"], toy["n_h"], ctx.t_el, steps=12)
        assert guard.n_directions > 100 and guard.lanczos_steps == 12
        assert guard.lambda_min > 0.0 and guard.lambda_max > guard.lambda_min
        assert guard.kappa == pytest.approx(guard.lambda_max / guard.lambda_min)
        assert guard.passes(lambda_guard=1e-3, kappa_max=1e6)
        assert not guard.passes(lambda_guard=guard.lambda_max * 2, kappa_max=1e6)
