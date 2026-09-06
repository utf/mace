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
    vals, vecs = spla.eigsh(op, k=min(k, n - 2), which="LM", tol=tol)      # largest of (H - sigma)^-1: nearest sigma
    eps = sigma + 1.0 / vals
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
        # A first window around the median of the diagonal, then re-centre on the gap between
        # the reference count and the next level.
        sigma0 = float(np.median(H.diagonal()))
        eps0, _ = frontier_window(H, sigma0, k)
        inertia0, _ = inertia_below(H, sigma0)
        sigma = sigma0
        for target in (n_ref[0],):
            # levels below sigma0 = inertia0.below; index of the level 'target' relative to it
            j = target - inertia0.below
            if 0 <= j < len(eps0):
                sigma = 0.5 * (eps0[j - 1] + eps0[j]) if j > 0 else float(eps0[0]) - 1.0
    inertia, lu = inertia_below(H, sigma)
    eps, psi = frontier_window(H, sigma, k, lu)
    below = inertia.below - int((eps < sigma).sum())          # levels below the lowest found
    eps_t = torch.tensor(eps)
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
