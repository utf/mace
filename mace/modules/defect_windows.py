"""Addendum section 5.1: the explicit frontier functional's building blocks.

THE CIRCULARITY THIS REMOVES. The legacy channel objects were `f_k s^e(eps_k) |psi_k|^2` with
`s` a strictly positive sigmoid of the level's energy -- which (a) leaks a nonzero tail over
every one of the `O(N_orb)` bulk states, enough to overwhelm an `O(1)` carrier as the cell
grows, and (b) would, once a potential `V[P]` enters `H`, rebuild the windows from `H_B[P]`
while also treating them as fixed inside `dPhi/dP`. Here the windows are functions of the
occupation-independent `H_fix` only, have exact-zero outer plateaus, and the carrier is
extracted from an INDEPENDENT `P` as a positive excess over a continued valence reference.

THE OBJECTS.

    b_c(eps)        compact-support window: exact-zero outer plateaus, a unit inner plateau,
                    C^2 quintic transitions; support fixed relative to the aligned edges
    M_c = b_c(H_fix)^2,  B_c = M_c^{1/2} = b_c(H_fix)         no singular square root at zero
    P_V^fix         = Pi_{M_VB}(H_fix)     the rank-M_VB continued-valence projector (its
                                           separating gap must stay above the floor)
    Delta P_sigma   = P_sigma - P_V^fix
    r_+(x)          exactly 0 for x <= eta_0, quintic for eta_0 < x < eta_1, x for x >= eta_1
    D_e = B_e r_+(Delta P) B_e,   D_h = B_h r_+(-Delta P) B_h                    (PSD)
    rho^_c = D[D_c] / Tr D_c      with D the site-diagonal map (trace-preserving)
    l_i = Tr(Pi_i D_c) / Tr D_c,  N_eff = 1 / sum_i l_i^2,  R_eff^2 = 1/2 sum_ij l_i l_j d_h^2
    w_c = W(N_eff; N_loc, N_ext) W(R_eff; R_loc, R_ext),   W an exact-plateau C^2 switch

THE GATES (all absolute, none per atom): `alpha_min n_c <= Tr D_c <= alpha_max n_c`; a
carrier-free channel's background trace and an active channel's excess outside its window
both below `leakage_tol`; the valence projector's gap above `gap_floor`. A failure is an
`UnsupportedStateError`, never a denominator offset.

DERIVATIVES. Every matrix function here is `U diag(g(lam)) U^T` with the backward taken by
the Daleckii-Krein map of `defect_counting` (the divided difference `(g_k - g_l) / (lam_k -
lam_l)`, its limit `g'` at coincidence) -- never through eigenvectors, so a degenerate
spectrum (and `Delta P` has an `O(N)`-fold zero eigenvalue for one carrier) is finite. The
windows depend on geometry through `H_fix` and their full matrix-function derivative
reaches the forces; they are fixed only with respect to the variational derivative in `P`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mace.modules.defect_carriers import UnsupportedStateError
from mace.modules.defect_counting import ORBITALS_PER_ATOM, _dk_eigenbasis_general

__all__ = ["s5", "s5_prime", "plateau_window", "r_plus", "exact_switch",
           "electron_window", "hole_window", "spectral_function", "valence_projector",
           "positive_excess", "positive_excess_pair", "background_gates",
           "channel_density", "localisation", "WindowConfig", "ChannelDiagnostics"]

#: The eigenvalue coincidence tolerance of the Daleckii-Krein maps.
DEGENERACY_TOL = 1e-7


# ------------------------------------------------------------------ scalar shapes


def s5(t: torch.Tensor) -> torch.Tensor:
    """`6 t^5 - 15 t^4 + 10 t^3` on [0, 1]: 0 -> 1 with zero first and second derivatives at
    both ends, clamped outside."""
    t = t.clamp(0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def s5_prime(t: torch.Tensor) -> torch.Tensor:
    inside = (t > 0.0) & (t < 1.0)
    tc = t.clamp(0.0, 1.0)
    return torch.where(inside, 30.0 * tc * tc * (1.0 - tc) ** 2, torch.zeros_like(t))


def plateau_window(x: torch.Tensor, x0: float, x1: float, x2: float, x3: float
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(b(x), b'(x))`: exactly 0 for `x <= x0` and `x >= x3`, exactly 1 on `[x1, x2]`,
    quintic in between. Requires `x0 < x1 <= x2 < x3`."""
    if not (x0 < x1 <= x2 < x3):
        raise ValueError(f"a window needs x0 < x1 <= x2 < x3, got {(x0, x1, x2, x3)}")
    up = (x - x0) / (x1 - x0)
    down = (x - x2) / (x3 - x2)
    value = torch.where(x < x1, s5(up), torch.where(x <= x2, torch.ones_like(x),
                                                    1.0 - s5(down)))
    slope = torch.where(x < x1, s5_prime(up) / (x1 - x0),
                        torch.where(x <= x2, torch.zeros_like(x), -s5_prime(down) / (x3 - x2)))
    return value, slope


def r_plus(x: torch.Tensor, eta0: float, eta1: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(r_+(x), r_+'(x))`: exactly 0 for `x <= eta0`, `x s5((x - eta0) / (eta1 - eta0))` in
    between, exactly `x` for `x >= eta1`; nonnegative and C^2 (the quintic's first and second
    derivatives vanish at both knots, so `x s5` joins `0` and `x` with matching curvature)."""
    if not (0.0 <= eta0 < eta1 <= 1.0):
        raise ValueError(f"r_+ needs 0 <= eta0 < eta1 <= 1, got {(eta0, eta1)}")
    t = (x - eta0) / (eta1 - eta0)
    mid = x * s5(t)
    mid_slope = s5(t) + x * s5_prime(t) / (eta1 - eta0)
    value = torch.where(x <= eta0, torch.zeros_like(x), torch.where(x >= eta1, x, mid))
    slope = torch.where(x <= eta0, torch.zeros_like(x),
                        torch.where(x >= eta1, torch.ones_like(x), mid_slope))
    return value, slope


def exact_switch(x: torch.Tensor, x1: float, x2: float) -> torch.Tensor:
    """`W(x; x1, x2)`: exactly 1 for `x <= x1`, exactly 0 for `x >= x2`, `1 - s5` between."""
    if not (x1 < x2):
        raise ValueError(f"W needs x1 < x2, got {(x1, x2)}")
    return 1.0 - s5((x - x1) / (x2 - x1))


# ------------------------------------------------------------------ the windows


@dataclass(frozen=True)
class WindowConfig:
    """The registered numbers of section 5.1, all absolute (none scales with N_at or L)."""

    delta: float          # eV, edge margin: the electron window opens at VBM_al + delta
    delta_s: float        # eV, half-width of each quintic transition
    extent: float         # eV, how far past the FAR edge the unit plateau reaches
    eta: Tuple[float, float]         # (eta_0, eta_1) of r_+
    alpha: Tuple[float, float]       # (alpha_min, alpha_max) trace bounds per carrier
    leakage_tol: float               # absolute occupation-tail bound
    gap_floor: float                 # eV, the valence projector's separating gap
    n_loc: float                     # site counts of W(N_eff)
    n_ext: float
    r_loc: float                     # A, of W(R_eff)
    r_ext: float

    @classmethod
    def from_functional(cls, f: Dict[str, Any]) -> "WindowConfig":
        """From the model's functional dict; a key a pre-Stage-5 model's dict lacks takes
        the registered default of `defect_density.DEFAULT_FUNCTIONAL`."""
        from mace.modules.defect_density import DEFAULT_FUNCTIONAL

        def get(key):
            return f[key] if key in f else DEFAULT_FUNCTIONAL[key]

        eta = tuple(float(x) for x in get("eta_plus"))
        alpha = tuple(float(x) for x in get("alpha_bounds"))
        cfg = cls(delta=float(f["delta"]), delta_s=float(f["delta_s"]),
                  extent=float(get("window_extent")), eta=eta, alpha=alpha,
                  leakage_tol=float(get("leakage_tol")), gap_floor=float(get("gap_floor")),
                  n_loc=float(get("n_loc")), n_ext=float(get("n_ext")),
                  r_loc=float(get("r_loc")), r_ext=float(get("r_ext")))
        if not (1.0 <= cfg.n_loc < cfg.n_ext):
            raise ValueError(f"need 1 <= N_loc < N_ext, got {(cfg.n_loc, cfg.n_ext)}")
        if not (0.0 < cfg.r_loc < cfg.r_ext):
            raise ValueError(f"need 0 < R_loc < R_ext, got {(cfg.r_loc, cfg.r_ext)}")
        if not (0.0 < cfg.alpha[0] < 1.0 < cfg.alpha[1]):
            raise ValueError(f"need 0 < alpha_min < 1 < alpha_max, got {cfg.alpha}")
        return cfg


def electron_window(eps: torch.Tensor, vbm_al: float, cbm_al: float, cfg: WindowConfig
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`b_e`: opens across `VBM_al + delta` (the legacy sigmoid's centre), unit up to
    `CBM_al + extent`, closes over one more transition. Fixed relative to the edges."""
    lo = vbm_al + cfg.delta
    hi = cbm_al + cfg.extent
    return plateau_window(eps, lo - cfg.delta_s, lo + cfg.delta_s, hi - cfg.delta_s,
                          hi + cfg.delta_s)


def hole_window(eps: torch.Tensor, vbm_al: float, cbm_al: float, cfg: WindowConfig
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`b_h`: the mirror image, closing across `CBM_al - delta`."""
    hi = cbm_al - cfg.delta
    lo = vbm_al - cfg.extent
    return plateau_window(eps, lo - cfg.delta_s, lo + cfg.delta_s, hi - cfg.delta_s,
                          hi + cfg.delta_s)


# ------------------------------------------------------------------ matrix functions


class _SpectralFunction(torch.autograd.Function):
    """`g(A) = U diag(g) U^T` of a symmetric `A`, differentiable in `A` by Daleckii-Krein.

    `lam`, `U` are `A`'s detached eigenpairs; `g`, `g_slope` are `g(lam)` and `g'(lam)`. No
    fixed-N correction: these functions carry no chemical potential."""

    @staticmethod
    def forward(ctx, A, lam, U, g, g_slope, tol: float):
        out = (U * g.unsqueeze(-2)) @ U.transpose(-1, -2)
        ctx.save_for_backward(lam, U, g, g_slope)
        ctx.tol = float(tol)
        ctx.in_dtype = A.dtype
        return out.to(A.dtype)

    @staticmethod
    def backward(ctx, grad):
        lam, U, g, g_slope = ctx.saved_tensors
        G = grad.double()
        G = 0.5 * (G + G.transpose(-1, -2))
        Ghat = U.transpose(-1, -2) @ G @ U
        zeros = torch.zeros_like(lam)
        M = _dk_eigenbasis_general(lam, g, g_slope, zeros, zeros, ctx.tol, Ghat)
        dA = U @ M @ U.transpose(-1, -2)
        return dA.to(ctx.in_dtype), None, None, None, None, None


def spectral_function(A: torch.Tensor, g: torch.Tensor, g_slope: torch.Tensor,
                      spectrum: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                      tol: float = DEGENERACY_TOL) -> torch.Tensor:
    """`U diag(g) U^T`, differentiable in `A`; `spectrum` is `(lam, U)` in float64 when the
    caller has it (the head's), else it is computed here, detached."""
    if spectrum is None:
        lam, U = torch.linalg.eigh(A.detach().double())
    else:
        lam, U = spectrum
    return _SpectralFunction.apply(A, lam.double(), U.double(), g.double(), g_slope.double(),
                                   float(tol))


def valence_projector(H_fix: torch.Tensor, lam: torch.Tensor, U: torch.Tensor, m_vb: int,
                      gap_floor: float, label: str = "") -> torch.Tensor:
    """`P_V^fix = Pi_{M_VB}(H_fix)`: the rank-`m_vb` projector on the lowest levels, refused
    when its separating gap `lam_M - lam_{M-1}` is below the registered floor (the divided
    difference across it is `1 / gap`, bounded only by that floor)."""
    n = lam.numel()
    if not (0 < int(m_vb) < n):
        raise UnsupportedStateError(f"valence rank {m_vb} outside 1..{n - 1}{label}")
    gap = float(lam[m_vb] - lam[m_vb - 1])
    if gap < float(gap_floor):
        raise UnsupportedStateError(
            f"the continued-valence projector's separating gap {gap:.4f} eV at rank {m_vb} "
            f"is below the registered floor {gap_floor} eV{label}: the projector and the "
            "state are unsupported (addendum 5.1)")
    g = torch.zeros_like(lam)
    g[:m_vb] = 1.0
    return spectral_function(H_fix, g, torch.zeros_like(lam), spectrum=(lam, U))


def positive_excess(delta_p: torch.Tensor, eta: Tuple[float, float], sign: float = 1.0
                    ) -> torch.Tensor:
    """`r_+(sign * Delta P)`, differentiable in `Delta P`."""
    nu, V = torch.linalg.eigh((float(sign) * delta_p).detach().double())
    value, slope = r_plus(nu, float(eta[0]), float(eta[1]))
    return spectral_function(float(sign) * delta_p, value, slope, spectrum=(nu, V))


def positive_excess_pair(delta_p: torch.Tensor, eta: Tuple[float, float]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(r_+(Delta P), r_+(-Delta P))` from ONE eigendecomposition of `Delta P`."""
    nu, V = torch.linalg.eigh(delta_p.detach().double())
    v_e, s_e = r_plus(nu, float(eta[0]), float(eta[1]))
    v_h, s_h = r_plus(-nu, float(eta[0]), float(eta[1]))
    R_e = spectral_function(delta_p, v_e, s_e, spectrum=(nu, V))
    # r_+(-Delta P) = U diag(r_+(-nu)) U^T; as a function of Delta P its slope is -r_+'(-nu).
    R_h = spectral_function(delta_p, v_h, -s_h, spectrum=(nu, V))
    return R_e, R_h


def background_gates(R: torch.Tensor, D: torch.Tensor, count: int, tol: float,
                     label: str = "") -> float:
    """The two absolute leakage gates of section 5.1 / 6.2 for one channel; returns the
    excess the window did not capture (the channel's leakage diagnostic).

    * carrier-free background: a channel with zero exact count must show no excess inside
      its window -- `Tr D_c <= tol` -- so a smearing tail over the bulk states is annihilated
      by the exact-zero plateau of `r_+`, not accumulated (the `O(N_orb)` failure);
    * captured excess: for an active channel, the positive excess OUTSIDE the window,
      `Tr r_+ - Tr D_c`, must be below `tol` -- the counted carrier lives where the window
      says it does, and the window's compact support does not cut it.
    """
    trace_r = float(torch.diagonal(R).sum().detach())
    trace_d = float(torch.diagonal(D).sum().detach())
    if int(count) == 0:
        if trace_d > tol:
            raise UnsupportedStateError(
                f"carrier-free background trace {trace_d:.3e} inside the window exceeds the "
                f"registered leakage bound {tol}{label}: the smearing background is not "
                "below the exact-zero plateau (addendum 5.1, 6.2)")
        return trace_d
    outside = trace_r - trace_d
    if outside > tol:
        raise UnsupportedStateError(
            f"positive excess {outside:.3e} lies outside the compact window (captured "
            f"{trace_d:.4f} of {trace_r:.4f}) against the registered bound {tol}{label}: the "
            "counted carrier is not where the window is (addendum 5.1)")
    return outside


# ------------------------------------------------------------------ channels


@dataclass
class ChannelDiagnostics:
    trace: float
    n_eff: float
    r_eff: float
    leakage: float


def channel_density(D: torch.Tensor, n_sites: int, count: int, cfg: WindowConfig,
                    label: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """`(rho^_c [n_sites], Tr D_c)`: the site-diagonal map of `D_c`, normalised to one, with
    the trace bounds `alpha_min n <= Tr D <= alpha_max n` enforced (absolute in the
    carrier count, not per atom)."""
    trace = torch.diagonal(D).sum()
    lo, hi = cfg.alpha[0] * count, cfg.alpha[1] * count
    if not (lo <= float(trace) <= hi):
        raise UnsupportedStateError(
            f"channel trace {float(trace):.4f} outside [{lo:.3f}, {hi:.3f}] for {count} "
            f"carrier(s){label}: the extracted excess does not account for the counted "
            "carriers (addendum 5.1); an unsupported state, not a denominator offset")
    site = torch.diagonal(D).reshape(n_sites, ORBITALS_PER_ATOM).sum(dim=-1)
    return site / trace, trace


def _pair_distances(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """`d_h(R_i, R_j)`: the minimum-image metric, which agrees with the Cartesian distance
    inside the compact-support range (half a cell) -- the registered periodic metric."""
    d = positions[:, None, :] - positions[None, :, :]
    frac = d @ torch.linalg.inv(cell)
    frac = frac - torch.round(frac)
    d = frac @ cell
    return torch.sqrt((d * d).sum(dim=-1) + 1e-24)


def localisation(rho_hat: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
                 cfg: WindowConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(w, N_eff, R_eff)` of one channel: the exact-plateau switch on the effective site
    count and the centre-free spread, with fixed absolute thresholds."""
    l = rho_hat
    n_eff = 1.0 / (l * l).sum().clamp_min(1e-300)
    d2 = _pair_distances(positions, cell) ** 2
    r_eff = torch.sqrt((0.5 * (l[:, None] * l[None, :] * d2).sum()).clamp_min(0.0))
    w = exact_switch(n_eff, cfg.n_loc, cfg.n_ext) * exact_switch(r_eff, cfg.r_loc, cfg.r_ext)
    return w, n_eff, r_eff
