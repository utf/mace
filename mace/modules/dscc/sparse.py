"""Plan section 8: the sparse path for the size ladder.

At the training sizes the dense `eigh` is exact and cheap; beyond them the two fillings
differ only in the FRONTIER states -- `dP = sum_a [f_S(eps_a) - f_ref(eps_a)] psi_a psi_a^T`
is nonzero only for levels within the smearing window of either chemical potential -- so
the head needs: (i) the count of levels below the gap (certified), (ii) the eigenpairs in
a window around the two chemical potentials, (iii) certified bounds on what the window
omits. Here:

* `H0` in CSR from the edge blocks (never a dense `[4N, 4N]` at scale);
* the below-slice count by the inertia of `H0 - sigma I` (sparse LDL^T through an
  unpivoted LU, Sylvester's law; the factorisation is verified by its residual);
* the frontier eigenpairs by shift-invert ARPACK (`eigsh`) around mid-gap, `k = |Q| +
  k_buffer`, re-using the same factorisation;
* Fermi-tail bounds: the omitted levels lie beyond the window edges, so their
  `|f_S - f_ref|` is at most `erfc(x_edge) / 2`; the omitted charge and energy are bounded by
  that times the number of omitted levels (from the count) and the window edge energies.
The dense path is the reference; the ladder gate asserts agreement at 159 atoms and on
every ladder cell. Electrostatics stay the dense Ewald matrix within the dense regime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import torch

from mace.modules.dscc.fill import SIGMA_S, chemical_potential, occupations
from mace.modules.dscc.legacy import ORBITALS_PER_ATOM, sk_block

K_BUFFER_DEFAULT = 8          # registered: frontier window = |Q| + k_buffer states around mid-gap
TAIL_TOL_DEFAULT = 1e-8       # registered: certified bound on the omitted charge (e) and energy (eV)


def csr_hamiltonian(h0_module, scalars: torch.Tensor, vectors: Optional[torch.Tensor], species: torch.Tensor,
                    edge_index: torch.Tensor, edge_vector: torch.Tensor) -> sps.csr_matrix:
    """`H0` of one graph as a SciPy CSR matrix from the Slater-Koster edge blocks, the
    on-site levels and the directional site blocks (the same terms as `H0.forward`)."""
    sk = h0_module.sk
    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1).clamp_min(1e-9)
        direction = edge_vector / r.unsqueeze(-1)
        v = sk.integrals(scalars[src], scalars[dst], r, species[src], species[dst])
        blocks = sk_block(direction, v).cpu().numpy()                     # [E, 4, 4]
        n = int(species.shape[0])
        o = np.arange(ORBITALS_PER_ATOM)
        rows = (src.cpu().numpy()[:, None, None] * ORBITALS_PER_ATOM + o[None, :, None]).repeat(ORBITALS_PER_ATOM, axis=2).reshape(-1)
        cols = (dst.cpu().numpy()[:, None, None] * ORBITALS_PER_ATOM + o[None, None, :]).repeat(ORBITALS_PER_ATOM, axis=1).reshape(-1)
        H = sps.coo_matrix((blocks.reshape(-1), (rows, cols)), shape=(4 * n, 4 * n)).tocsr()
        levels = sk.on_site(scalars, species, None, centre=h0_module.centre)
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1).cpu().numpy()
        H = H + sps.diags(diag)
        if h0_module.directional and vectors is not None:
            site = h0_module.directional_site_blocks(vectors, species, edge_index, edge_vector).cpu().numpy()
            idx = np.arange(n)[:, None, None] * ORBITALS_PER_ATOM
            r2 = (idx + o[None, :, None]).repeat(ORBITALS_PER_ATOM, axis=2).reshape(-1)
            c2 = (idx + o[None, None, :]).repeat(ORBITALS_PER_ATOM, axis=1).reshape(-1)
            H = H + sps.coo_matrix((site.reshape(-1), (r2, c2)), shape=H.shape).tocsr()
        H = 0.5 * (H + H.T)
        if h0_module.site_shift is not None:
            H = H + sps.diags(np.repeat(h0_module.site_shift.cpu().numpy(), ORBITALS_PER_ATOM))
        if h0_module.gauge_shift:
            H = H + float(h0_module.gauge_shift) * sps.eye(H.shape[0])
    return H.tocsr()


@dataclass
class Inertia:
    below: int                 # eigenvalues below sigma
    sigma: float
    residual: float            # |LU - A| relative, the factorisation check


def inertia_below(H: sps.csr_matrix, sigma: float) -> Tuple[Inertia, spla.SuperLU]:
    """Number of eigenvalues of `H` below `sigma` by Sylvester's law on the unpivoted LU of
    `H - sigma I` (symmetric: LU = L D L^T, the signs of U's diagonal are D's). Returns the
    factorisation for the shift-invert solves."""
    A = (H - sigma * sps.eye(H.shape[0], format="csr")).tocsc()
    lu = spla.splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    d = lu.U.diagonal()
    below = int((d < 0).sum())
    # Verify the factorisation on a random vector (unpivoted LU can be unstable in principle).
    x = np.random.default_rng(0).normal(size=H.shape[0])
    y = lu.solve(A @ x)
    residual = float(np.linalg.norm(y - x) / np.linalg.norm(x))
    return Inertia(below=below, sigma=float(sigma), residual=residual), lu


def mid_gap_sigma(H: sps.csr_matrix, n_occupied: int, tol: float = 1e-6) -> float:
    """A shift inside the gap above the `n_occupied`-th level, by bisection on the inertia
    (each step one sparse LDL^T): the certified way to place the window without any
    eigenvalue in hand. Returns the midpoint of the bracket once the counts straddle."""
    d = H.diagonal()
    lo, hi = float(d.min()) - 50.0, float(d.max()) + 50.0
    # Bracket: below(lo) = 0 < n_occupied <= below(hi) = n_orb.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = inertia_below(H, mid)[0].below
        if below < n_occupied:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    # lo has < n_occupied levels below, hi has >= n_occupied: the n_occupied-th level sits in
    # [lo, hi]; the gap above it is found by bracketing the (n_occupied + 1)-th level too.
    lo2, hi2 = hi, float(d.max()) + 50.0
    for _ in range(80):
        mid = 0.5 * (lo2 + hi2)
        below = inertia_below(H, mid)[0].below
        if below < n_occupied + 1:
            lo2 = mid
        else:
            hi2 = mid
        if hi2 - lo2 < tol:
            break
    return 0.5 * (hi + lo2)          # between the HOMO (<= hi) and the LUMO (>= lo2)


@dataclass
class FrontierWindow:
    eps: np.ndarray            # eigenvalues in the window, ascending
    psi: np.ndarray            # [4N, k] eigenvectors
    n_below_window: int        # levels below the lowest found (from the inertia)
    sigma: float
    tail_bound_charge: float   # certified bound on the omitted |dq| (e)
    tail_bound_energy: float   # certified bound on the omitted |dJ| (eV)


def frontier_window(H: sps.csr_matrix, sigma: float, k: int, lu: Optional[spla.SuperLU] = None,
                    tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """The `k` eigenpairs nearest `sigma` by shift-invert Lanczos (ARPACK), re-using `lu`."""
    n = H.shape[0]
    if lu is None:
        _, lu = inertia_below(H, sigma)
    op = spla.LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)
    v0 = np.cos(np.arange(n) * 0.37) + 0.5                                   # deterministic start
    vals, vecs = spla.eigsh(op, k=min(k, n - 2), which="LM", tol=0.0, v0=v0, maxiter=20000)   # nearest sigma, to machine precision
    eps = sigma + 1.0 / vals
    # Rayleigh-quotient refinement and an explicit residual check of every pair.
    Hv = H @ vecs
    eps = np.einsum("ia,ia->a", vecs, Hv)
    resid = np.linalg.norm(Hv - vecs * eps[None, :], axis=0)
    if resid.max() > 1e-8:
        raise RuntimeError(f"frontier eigenpairs not converged: max residual {resid.max():.1e}")
    order = np.argsort(eps)
    return eps[order], vecs[:, order]


def two_fillings_sparse(H: sps.csr_matrix, n_s: Tuple[int, int], n_ref: Tuple[int, int],
                        sigma: Optional[float] = None, k_buffer: int = K_BUFFER_DEFAULT,
                        sigma_s: float = SIGMA_S, tail_tol: float = TAIL_TOL_DEFAULT):
    """Frontier-only two fillings: `(J, dP_window (as (psi, weights)), dq, window)`; the
    omitted levels are certified by the Fermi-tail bounds (raises when the bound exceeds
    `tail_tol`, the registered tolerance). `sigma` defaults to the gap estimate from a small
    first window; the count below the window comes from the inertia."""
    n_orb = H.shape[0]
    n_atoms = n_orb // ORBITALS_PER_ATOM
    q = abs(sum(n_s) - sum(n_ref))
    k = int(q) + int(k_buffer)
    if sigma is None:
        sigma = mid_gap_sigma(H, n_ref[0])
    inertia, lu = inertia_below(H, sigma)
    # The window grows until the certified tail bound is below `tail_tol` (the potential
    # of a converging SCF moves levels through the window edges).
    while True:
        eps, psi = frontier_window(H, sigma, k, lu)
        below = inertia.below - int((eps < sigma).sum())      # levels below the lowest found
        eps_t = torch.tensor(eps)
        mus_probe = [float(chemical_potential(eps_t, float(n_el - below), sigma_s)) for n_el in (n_s[0], n_s[1], n_ref[0], n_ref[1])]
        x_edge = min(abs(eps.min() - min(mus_probe)), abs(eps.max() - max(mus_probe))) / sigma_s
        if 2.0 * (n_orb - len(eps)) * 0.5 * math.erfc(x_edge) <= tail_tol or k >= n_orb - 2:
            break
        k = min(2 * k, n_orb - 2)
    # Each filling's mu from the WINDOW plus the count below it: the levels below are fully
    # occupied for both fillings (certified by the tail bound), so sum f over the window
    # must equal N - below.
    weights = []
    mus = []
    for n_el in (n_s[0], n_s[1], n_ref[0], n_ref[1]):
        target = float(n_el - below)
        mu = chemical_potential(eps_t, target, sigma_s)
        mus.append(float(mu))
        weights.append(occupations(eps_t, mu, sigma_s))
    f_s = weights[0] + weights[1]; f_r = weights[2] + weights[3]
    w = (f_s - f_r)                                                    # [k]
    # Certified tails: beyond the window edges every |f_S - f_ref| <= erfc(x_edge)/2 with
    # x_edge the distance of the edge to the nearest mu, in smearing widths.
    edges = np.array([eps.min(), eps.max()])
    x_edge = min(abs(edges[0] - min(mus)), abs(edges[1] - max(mus))) / sigma_s
    tail = 0.5 * math.erfc(x_edge)
    n_omitted = n_orb - len(eps)
    bound_charge = 2.0 * n_omitted * tail                              # two spins
    bound_energy = bound_charge * float(np.abs(edges).max())
    if bound_charge > tail_tol:
        raise RuntimeError(f"frontier window too narrow: omitted-charge bound {bound_charge:.2e} > {tail_tol}")
    psi_t = torch.tensor(psi)
    dP = (psi_t * w.unsqueeze(0)) @ psi_t.T
    dq = -torch.diagonal(dP).reshape(n_atoms, ORBITALS_PER_ATOM).sum(-1)
    # Band-form energy difference (frontier-only): sum_a w_a eps_a + R_S - R_ref over the window
    x_s = [(eps_t - m) / sigma_s for m in mus[:2]]; x_r = [(eps_t - m) / sigma_s for m in mus[2:]]
    R = lambda xs: sum(-sigma_s * torch.exp(-x * x).sum() / (2.0 * math.sqrt(math.pi)) for x in xs)  # noqa: E731
    J = (w * eps_t).sum() + R(x_s) - R(x_r)
    window = FrontierWindow(eps=eps, psi=psi, n_below_window=below, sigma=float(sigma),
                            tail_bound_charge=bound_charge, tail_bound_energy=bound_energy)
    return J, (psi_t, w), dq, window


# ------------------------------------------------------------------ sparse D-SCC solve and forces

def _window_response(eps: np.ndarray, psi: np.ndarray, mus, n_atoms: int, sigma_s: float,
                     counts) -> torch.Tensor:
    """Approximate `d dq_new / dV` from the window's eigenpairs alone (pairs with both
    levels inside the window): a quasi-Newton Jacobian -- the fixed point is exact because
    the residual is, only the step is approximate."""
    eps_t = torch.tensor(eps); psi_t = torch.tensor(psi)
    k = eps_t.numel()
    A = (psi_t.reshape(n_atoms, ORBITALS_PER_ATOM, k).unsqueeze(-1) * psi_t.reshape(n_atoms, ORBITALS_PER_ATOM, k).unsqueeze(-2)).sum(1)  # [N, k, k]
    M = torch.zeros(n_atoms, n_atoms, dtype=torch.float64)
    for (n_el, sign) in counts:
        mu = chemical_potential(eps_t, float(n_el), sigma_s)
        x = (eps_t - mu) / sigma_s
        f = 0.5 * torch.erfc(x)
        fp = -torch.exp(-x * x) / (sigma_s * math.sqrt(math.pi))
        d_eps = eps_t.unsqueeze(-1) - eps_t.unsqueeze(0)
        d_f = f.unsqueeze(-1) - f.unsqueeze(0)
        near = d_eps.abs() <= 1e-7
        mid = 0.5 * (eps_t.unsqueeze(-1) + eps_t.unsqueeze(0))
        L = torch.where(near, -torch.exp(-((mid - mu) / sigma_s) ** 2) / (sigma_s * math.sqrt(math.pi)),
                        d_f / torch.where(near, torch.ones_like(d_eps), d_eps))
        contrib = torch.einsum("iab,ab,jab->ij", A, L, A)
        s_fp = fp.sum()
        if float(s_fp.abs()) > 1e-300:
            g = torch.einsum("iaa,a->i", A, fp)
            contrib = contrib - torch.outer(g, g) / s_fp
        M = M + sign * contrib
    return M


def solve_dscc_sparse(H0: sps.csr_matrix, gamma: torch.Tensor, n_s: Tuple[int, int], n_ref: Tuple[int, int],
                      sigma: float, k_buffer: int = K_BUFFER_DEFAULT, sigma_s: float = SIGMA_S,
                      W: Optional[torch.Tensor] = None, tol_q: float = 1e-8, n_max: int = 100,
                      mixing: float = 0.3, tail_tol: float = TAIL_TOL_DEFAULT):
    """The D-SCC fixed point on the frontier window: `dq -> dq(H0 - diag(Gamma dq + W))`
    with the window re-solved (shift-invert refactorised) every iteration, a quasi-Newton
    step from the window response, and the same unmixed-residual criterion."""
    from mace.modules.dscc.scf import _anderson
    n_atoms = gamma.shape[0]
    dq = torch.zeros(n_atoms, dtype=torch.float64)
    history = []
    dq_hist, res_hist = [], []
    converged = False
    last = None
    for it in range(n_max):
        V = gamma @ dq + (W if W is not None else 0.0)
        H = H0 + sps.diags(np.repeat(-V.numpy(), ORBITALS_PER_ATOM))
        J, (psi, w), dq_new, window = two_fillings_sparse(H, n_s, n_ref, sigma=sigma, k_buffer=k_buffer,
                                                          sigma_s=sigma_s, tail_tol=tail_tol)
        res = dq_new - dq
        history.append(float(res.abs().max()))
        last = (J, psi, w, window, V)
        if history[-1] < tol_q:
            converged = True
            break
        below = window.n_below_window
        counts = ((n_s[0] - below, 1.0), (n_s[1] - below, 1.0), (n_ref[0] - below, -1.0), (n_ref[1] - below, -1.0))
        M = _window_response(window.eps, window.psi, None, n_atoms, sigma_s, counts)
        jac = M @ gamma
        step = torch.linalg.solve(torch.eye(n_atoms, dtype=torch.float64) - jac, res)
        # damped: accept the Newton step if it reduces the residual, else a mixing step
        trial = dq + step
        V_t = gamma @ trial + (W if W is not None else 0.0)
        H_t = H0 + sps.diags(np.repeat(-V_t.numpy(), ORBITALS_PER_ATOM))
        _, _, dq_t, _ = two_fillings_sparse(H_t, n_s, n_ref, sigma=sigma, k_buffer=k_buffer, sigma_s=sigma_s, tail_tol=tail_tol)
        dq_hist.append(dq); res_hist.append(res)
        if len(dq_hist) > 6:
            dq_hist.pop(0); res_hist.pop(0)
        # The window Jacobian is approximate (pairs with a level outside the window are
        # missing), so its step converges only linearly near the fixed point: take it while
        # it at least halves the residual, otherwise Anderson on the (exact) residual history.
        dq = trial if float((dq_t - trial).abs().max()) < 0.5 * history[-1] else _anderson(dq_hist, res_hist, mixing)
    J, psi, w, window, V = last
    energy = J - 0.5 * dq @ gamma @ dq
    return {"energy": energy, "dq": dq, "psi": psi, "w": w, "V": V, "window": window,
            "iterations": len(history), "converged": converged, "history": history}


def frontier_forces(h0_module, scalars: torch.Tensor, vectors: Optional[torch.Tensor], species: torch.Tensor,
                    edge_index: torch.Tensor, edge_vector: torch.Tensor, positions: torch.Tensor,
                    psi: torch.Tensor, w: torch.Tensor, gamma: Optional[torch.Tensor] = None,
                    dq: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`-Tr(dP dH0/dR) - 1/2 dq^T dGamma/dR dq` with `dP = psi diag(w) psi^T` of rank k:
    the cotangent on every 4x4 edge block and site block is gathered from `psi` (never a
    dense `[4N, 4N]`), and `Gamma`'s geometry derivative (dense within the dense regime)
    is contracted with the fixed charges."""
    sk = h0_module.sk
    n = int(species.shape[0])
    k = psi.shape[1]
    P_site = psi.reshape(n, ORBITALS_PER_ATOM, k)                          # [N, 4, k]
    src, dst = edge_index[0], edge_index[1]
    r = edge_vector.norm(dim=-1).clamp_min(1e-9)
    direction = edge_vector / r.unsqueeze(-1)
    v = sk.integrals(scalars[src], scalars[dst], r, species[src], species[dst])
    blocks = sk_block(direction, v)                                        # [E, 4, 4], attached
    cot_edges = torch.einsum("eak,k,ebk->eab", P_site[src], w, P_site[dst])   # dP restricted to (src, dst) blocks
    levels = sk.on_site(scalars, species, None, centre=h0_module.centre)   # [N, 2], attached
    cot_levels = torch.einsum("nak,k,nak->na", P_site, w, P_site)          # diagonal of dP per orbital, [N, 4]
    cot_levels2 = torch.stack([cot_levels[:, 0], cot_levels[:, 1:].sum(-1)], dim=-1)                              # [N, 2]
    tensors, cots = [blocks, levels], [cot_edges, cot_levels2]
    if h0_module.directional and vectors is not None:
        site = h0_module.directional_site_blocks(vectors, species, edge_index, edge_vector)     # [N, 4, 4]
        cot_site = torch.einsum("nak,k,nbk->nab", P_site, w, P_site)
        tensors.append(site); cots.append(cot_site)
    if gamma is not None and dq is not None:
        tensors.append(gamma); cots.append(0.5 * dq.unsqueeze(-1) * dq.unsqueeze(0))
    grads = torch.autograd.grad(tensors, positions, grad_outputs=cots, allow_unused=True)
    total = torch.zeros_like(positions)
    for g in grads:
        if g is not None:
            total = total + g
    return -total


# ------------------------------------------------------------------ model-level sparse inference

def model_forward_sparse(model, data, k_buffer: int = K_BUFFER_DEFAULT, tol_q: float = 1e-8,
                         n_max: int = 100, compute_force: bool = True):
    """Inference (no training graph) of one charged graph through the sparse path: block-0
    features, CSR `H0`, the dense Ewald kernels (the dense regime), the frontier-window
    D-SCC solve, rank-k frontier forces plus the base's own forces. Route A or Route B'
    (with `q0` from a full sparse... not available: Route B' needs the full reference
    density, so it uses the dense fill up to `max_dense_atoms` and is refused beyond).
    Returns `{"energy", "forces", "head_energy", "dq", "diagnostics"}`."""
    from mace.modules.models import ScaleShiftMACE
    from mace.modules.dscc.kernels import gamma_matrix, gamma_lr, host_potential, kernel_components
    from mace.modules.dscc.species import S_REF, State, neutral_count, states_from_batch, N0

    num_graphs = int(data["ptr"].numel() - 1)
    assert num_graphs == 1, "one graph at a time on the sparse path"
    state = states_from_batch(data["carrier_counts"].view(1, -1))[0]
    positions = data["positions"].requires_grad_(True)
    cell = data["cell"].view(3, 3)
    base_out = ScaleShiftMACE.forward(model.base, model._trunk_data(dict(data)), compute_force=False)
    node_feats = base_out["node_feats"]
    base_forces = None
    if compute_force:
        # The base's own forces, keeping the graph alive for the frontier contraction below.
        (g,) = torch.autograd.grad(base_out["energy"].sum(), positions, retain_graph=True)
        base_forces = -g
    scalars, vectors = model.features(node_feats)
    species = data["node_attrs"].argmax(dim=-1)
    ei = data["edge_index"]
    ev = positions[ei[1]] - positions[ei[0]] + data["unit_shifts"].to(positions.dtype) @ cell
    numbers = [model.atomic_numbers[int(x)] for x in species.tolist()]
    n_ref = neutral_count(numbers)
    n_s, n_r = state.counts(n_ref), S_REF.counts(n_ref)
    H_csr = csr_hamiltonian(model.h0, scalars.detach(), vectors.detach() if vectors is not None else None, species, ei, ev.detach())
    k_sr, k_lr = kernel_components(positions, cell, model.kernel)
    gamma = gamma_matrix(k_sr, k_lr, model.lambda_dir(), model.u_eff()[species], model.kernel.eps_inf)
    W = None
    if model.route_b:
        raise NotImplementedError("Route B' on the sparse path needs the full reference density (dense fill); "
                                  "use the dense forward below the dense-regime size")
    # sigma: mid-gap of the reference count, from a first window around the diagonal median
    # (two_fillings_sparse does this when sigma is None).
    if not model.coupling:
        J, (psi, w), dq, window = two_fillings_sparse(H_csr, n_s, n_r, sigma=None, k_buffer=k_buffer, sigma_s=model.sigma_s)
        head_energy = J
        diagnostics = {"iterations": 1, "converged": True, "window": len(window.eps), "tail_bound_charge": window.tail_bound_charge}
    else:
        _, _, _, window0 = two_fillings_sparse(H_csr, n_s, n_r, sigma=None, k_buffer=k_buffer, sigma_s=model.sigma_s)
        sol = solve_dscc_sparse(H_csr, gamma.detach(), n_s, n_r, window0.sigma, k_buffer, model.sigma_s, W, tol_q, n_max)
        head_energy, dq, psi, w = sol["energy"], sol["dq"], sol["psi"], sol["w"]
        diagnostics = {"iterations": sol["iterations"], "converged": sol["converged"], "window": len(sol["window"].eps),
                       "tail_bound_charge": sol["window"].tail_bound_charge, "residual": sol["history"][-1]}
    forces = None
    if compute_force:
        head_forces = frontier_forces(model.h0, scalars, vectors, species, ei, ev, positions, psi, w,
                                      gamma if model.coupling else None, dq if model.coupling else None)
        forces = base_forces.detach() + head_forces.detach()
    energy = base_out["energy"].detach().reshape(()) + head_energy.detach() + model.c_q(state.Q)
    return {"energy": energy, "forces": forces, "head_energy": head_energy.detach(), "dq": dq.detach(), "diagnostics": diagnostics}
