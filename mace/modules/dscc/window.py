"""v5 W2 closing item: the active-window partial solve inside the Newton loop (dense reference;
the Phase-4 sparse solver in embryo).

From a warm start only the levels within a few `sigma_s` of the two chemical potentials change
occupation between the state's and the reference's fill. The window is those levels plus a
buffer on each side, carried as an orthonormal basis `S` from the last full (or partial) solve
together with the certified count `n_below` of eigenvalues under it. A pass at a new `H`:
Rayleigh-Ritz on `S`, refinement by dense inverse iteration with two shifts (the two chemical
potentials), Rayleigh-Ritz again, residual check on the active levels, and the certification
that exactly `m` eigenvalues of `H` lie in the window's range (two LDL^T inertia counts, so
that a level entering from outside cannot be missed). The levels below the window are fully
occupied and those above empty in BOTH fills (the window is required to extend `WINDOW_MARGIN`
smearing widths beyond both potentials), so `dq`, the band-energy difference and the entropy
difference are exact sums over the window, and `dP` is the window's low-rank matrix. On any
failed check the caller falls back to a full diagonalisation, which also rebuilds the window."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from mace.modules.dscc.fill import SIGMA_S, FillResult, chemical_potential, generalised_entropy, occupations

ORBITALS = 4
WINDOW_MARGIN = 6.0        # smearing widths the window must extend beyond both chemical potentials
RESIDUAL_TOL = 1e-8        # ||H y - theta y|| on the active levels (registered)
N_BUFFER = 8               # levels kept on each side of the active set (registered)
COUNT_MARGIN = 1e-8        # eV: the inertia counts are taken just outside the Ritz range


@dataclass
class Window:
    S: torch.Tensor            # [n, m] orthonormal basis
    theta: torch.Tensor        # [m] Ritz values (ascending)
    n_below: int               # certified count of eigenvalues below the window
    eps_full: torch.Tensor     # the last full spectrum (Jacobian; stale between full solves)
    U_full: torch.Tensor
    mu_s: float                # the potentials at the last solve (the refinement shifts)
    mu_r: float


def inertia_below(H: torch.Tensor, e: float) -> int:
    """The number of eigenvalues of the symmetric `H` below `e` (Sylvester's law): the negative
    inertia of the Bunch-Kaufman LDL^T factorisation of `H - e I`, taken on the host."""
    A = (H.detach().cpu().double() - e * torch.eye(H.shape[-1], dtype=torch.float64))
    LD, piv = torch.linalg.ldl_factor(A, hermitian=True)
    n = A.shape[-1]; count = 0; i = 0
    piv = piv.tolist()
    while i < n:
        if piv[i] > 0:                              # 1x1 block
            count += int(LD[i, i] < 0); i += 1
        else:                                       # 2x2 block at (i, i+1)
            a, b, c = float(LD[i, i]), float(LD[i + 1, i]), float(LD[i + 1, i + 1])
            det = a * c - b * b
            if det < 0:
                count += 1
            elif a + c < 0:
                count += 2
            i += 2
    return count


def build_window(eps: torch.Tensor, U: torch.Tensor, mu_s: float, mu_r: float, sigma_s: float = SIGMA_S,
                 tol_f: float = 1e-10, n_buffer: int = N_BUFFER) -> Window:
    """The window from a full spectrum: the levels whose occupation differs between the two
    fills or that lie within `WINDOW_MARGIN` widths of either potential, plus `n_buffer` on each
    side."""
    df = occupations(eps, torch.as_tensor(mu_s, dtype=eps.dtype, device=eps.device), sigma_s) - \
        occupations(eps, torch.as_tensor(mu_r, dtype=eps.dtype, device=eps.device), sigma_s)
    lo_e, hi_e = min(mu_s, mu_r) - WINDOW_MARGIN * sigma_s, max(mu_s, mu_r) + WINDOW_MARGIN * sigma_s
    inside = (df.abs() > tol_f) | ((eps >= lo_e) & (eps <= hi_e))
    idx = torch.nonzero(inside).reshape(-1)
    n = eps.shape[-1]
    lo = max(int(idx.min()) - n_buffer, 0) if idx.numel() else max(int((eps < lo_e).sum()) - n_buffer, 0)
    hi = min(int(idx.max()) + n_buffer, n - 1) if idx.numel() else min(int((eps <= hi_e).sum()) + n_buffer, n - 1)
    return Window(S=U[:, lo:hi + 1].detach().clone(), theta=eps[lo:hi + 1].detach().clone(), n_below=lo,
                  eps_full=eps.detach(), U_full=U.detach(), mu_s=float(mu_s), mu_r=float(mu_r))


def _rayleigh_ritz(H: torch.Tensor, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T = S.transpose(-1, -2) @ H @ S
    theta, Z = torch.linalg.eigh(0.5 * (T + T.transpose(-1, -2)))
    return theta, S @ Z


MAX_ROUNDS = 8             # refinement rounds before the pass gives up (each: two shifted solves)


def partial_fillings(H: torch.Tensor, window: Window, n_s: Tuple[float, float], n_ref: Tuple[float, float],
                     sigma_s: float = SIGMA_S, tol_f: float = 1e-10, rounds: int = MAX_ROUNDS):
    """One partial pass at `H` (float64, `[n, n]`, any device; the work runs on the host).
    Returns `(energy, dq, fills, frontier, new_window)` with the same meaning as the frontier
    path of `scf.two_fillings`, or `None` when a check fails (the caller then diagonalises)."""
    dev = H.device
    Hc = H.detach().cpu().double()
    n = Hc.shape[-1]; n_atoms = n // ORBITALS
    S = window.S.detach().cpu().double()
    theta, Y = _rayleigh_ritz(Hc, S)
    shifts = (window.mu_s, window.mu_r)
    eye = torch.eye(n, dtype=torch.float64)
    lo_w, hi_w = min(shifts) - WINDOW_MARGIN * sigma_s, max(shifts) + WINDOW_MARGIN * sigma_s
    used = 0
    for used in range(1, rounds + 1):
        # inverse iteration with two shifts: each Ritz vector is refined with the nearer shift
        near = torch.stack([(theta - s).abs() for s in shifts]).argmin(0)
        X = torch.empty_like(Y)
        for k, s in enumerate(shifts):
            cols = torch.nonzero(near == k).reshape(-1)
            if cols.numel() == 0:
                continue
            X[:, cols] = torch.linalg.solve(Hc - s * eye, Y[:, cols])
        Q, _ = torch.linalg.qr(X)
        theta, Y = _rayleigh_ritz(Hc, Q)
        # converged when the levels that can carry occupation (inside the tails' range at the
        # last potentials) meet the residual tolerance; the buffer levels need not
        R = Hc @ Y - Y * theta.unsqueeze(0)
        inner = (theta >= lo_w) & (theta <= hi_w)
        if not bool(inner.any()) or float(R[:, inner].norm(dim=0).max()) <= RESIDUAL_TOL:
            break
    m = theta.numel()
    # potentials from the window alone: the levels below are full, those above empty
    counts = torch.tensor([float(n_s[0]) - window.n_below, float(n_s[1]) - window.n_below,
                           float(n_ref[0]) - window.n_below, float(n_ref[1]) - window.n_below], dtype=torch.float64)
    if bool((counts <= 0).any()) or bool((counts >= m).any()):
        return None
    mus = chemical_potential(theta.unsqueeze(0).expand(4, -1), counts, sigma_s)                  # [4]
    lo_e, hi_e = float(mus.min()) - WINDOW_MARGIN * sigma_s, float(mus.max()) + WINDOW_MARGIN * sigma_s
    if float(theta[0]) > lo_e or float(theta[-1]) < hi_e:
        return None                                       # the window no longer covers both tails
    # residuals on the active levels
    f = [occupations(theta, mus[k], sigma_s) for k in range(4)]
    df = (f[0] - f[2]) + (f[1] - f[3])
    active = df.abs() > tol_f
    R = Hc @ Y - Y * theta.unsqueeze(0)
    if active.any() and float(R[:, active].norm(dim=0).max()) > RESIDUAL_TOL:
        return None
    # certification: exactly m eigenvalues of H in the window's range
    if inertia_below(Hc, float(theta[0]) - COUNT_MARGIN) != window.n_below:
        return None
    if inertia_below(Hc, float(theta[-1]) + COUNT_MARGIN) != window.n_below + m:
        return None
    # quantities (exact sums over the window)
    w = (Y * Y).reshape(n_atoms, ORBITALS, m).sum(1)                                       # |Pi_i y_k|^2
    energy = torch.zeros((), dtype=torch.float64); dq = torch.zeros(n_atoms, dtype=torch.float64)
    frontier = []; fills: List[Optional[FillResult]] = [None] * 4
    for sigma in (0, 1):
        f_s, f_r = f[sigma], f[sigma + 2]
        R_s, R_r = generalised_entropy(theta, mus[sigma], sigma_s), generalised_entropy(theta, mus[sigma + 2], sigma_s)
        for k, (ff, RR) in ((sigma, (f_s, R_s)), (sigma + 2, (f_r, R_r))):
            fills[k] = FillResult(P=None, mu=mus[k].to(dev), eps=window.eps_full, U=window.U_full, f=None,
                                  F_band=((ff * theta).sum() + RR).to(dev), entropy=RR.to(dev))
        d = f_s - f_r
        energy = energy + (d * theta).sum() + (R_s - R_r)
        dq = dq - w @ d
        idx = torch.nonzero(d.abs() > tol_f).reshape(-1)
        frontier.append((Y[:, idx].to(dev), d[idx].to(dev)))
    new_window = Window(S=Y, theta=theta, n_below=window.n_below, eps_full=window.eps_full, U_full=window.U_full,
                        mu_s=float(mus[0]), mu_r=float(mus[2]))
    new_window.rounds = used
    return energy.to(dev), dq.to(dev), tuple(fills), frontier, new_window
