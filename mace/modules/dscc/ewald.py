"""Plan section 2.4: the Ewald matrix of Gaussian charges under PBC.

`E_PBC` is the `[N, N]` matrix with `0.5 q^T E_PBC q` the electrostatic energy of Gaussian
charges `q_i` (width `s_i`, density `exp(-r^2 / 2 s_i^2)`) in a periodic cell with a
neutralising background (VASP / label-code convention, registered). Two Gaussians of widths
`s_i`, `s_j` interact through the bare kernel `C erf(r / w_ij) / r`, `w_ij = sqrt(2 (s_i^2 +
s_j^2))` -- equal widths `r_g` give the plan's `erf(r / (2 r_g)) / r` and the self term
`C / (sqrt(pi) r_g)`, which the diagonal INCLUDES (so that `K_LR_ii = E_PBC_ii - K_SR_ii`
tends to `-alpha_M C / L` on a tiling ladder). The convention was pinned against the LES
Ewald sum (`latent_ewald.LatentEwald`, LES `sigma = sqrt(2) r_g`, self term excluded there):
`tests/extensions/dscc/test_ewald.py`.

The split: a broad Gaussian of width `eta` per charge carries the long range in reciprocal
space, `(4 pi / V) sum_{k != 0} exp(-eta^2 k^2) / k^2 cos(k . r_ij)`; the remainder
`[erf(r / w_ij) - erf(r / (2 eta))] / r` is summed over images in real space; the k = 0
term under the background is the constant `-4 pi eta^2 / V` on every pair. `eta` is a
convergence parameter, not a model quantity: every prediction is independent of it to the
registered tolerance (gate), and the cutoffs are derived from that tolerance rather than
fixed. Everything is torch, so positions and cell carry gradients (forces, stress).

Also here: the cross-width matrix for `Gamma_LR` (plan section 2.5: width `r_g` on one
side, `r_split` on the other -- the pair kernel depends only on `s_i^2 + s_j^2`, so the
matrix is symmetric) and the direct lattice sum of a short-ranged pair function for kernel
regime B (`K_SR_ij = sum_L s(|r_ij + L|)`).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.special import erfcinv

COULOMB = 14.399645          # eV A (e^2 / 4 pi eps_0); `defect_madelung.COULOMB_CONSTANT`
_SQRT_PI = math.sqrt(math.pi)
# Per-term tolerance of the real- and reciprocal-space sums, in eV per unit charge pair.
# The 1e-10 eV gate of plan section 4 is on the ENERGY; the neglected tails are sums of
# O(N^2 x shell) terms, so the per-term tolerance sits six decades below the gate. It costs
# ~20 % in cutoff radius over 1e-13 (erfcinv(1e-16) = 5.9 against 5.0).
EWALD_TOL = 1e-16


def _as_widths(width: Union[float, torch.Tensor], n: int, like: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(width, dtype=like.dtype, device=like.device)
    return w.expand(n) if w.dim() == 0 else w


def pair_width(s_i: torch.Tensor, s_j: torch.Tensor) -> torch.Tensor:
    """`w_ij = sqrt(2 (s_i^2 + s_j^2))`: the bare kernel is `erf(r / w_ij) / r`."""
    return torch.sqrt(2.0 * (s_i.unsqueeze(-1) ** 2 + s_j.unsqueeze(-2) ** 2))


def default_eta(cell: torch.Tensor, widths: Sequence[float]) -> float:
    """A splitting width balancing the two sums: a fifth of the shortest cell height, never
    below the broadest charge. Any value gives the same matrix to `EWALD_TOL`."""
    heights = _cell_heights(cell.detach())
    return float(max(max(float(w) for w in widths), 0.2 * float(heights.min())))


def _cell_heights(cell: torch.Tensor) -> torch.Tensor:
    """Perpendicular widths of the cell along its three axes: `V / |b x c|` etc."""
    volume = torch.det(cell).abs()
    cross = torch.stack([torch.cross(cell[1], cell[2], dim=0),
                         torch.cross(cell[2], cell[0], dim=0),
                         torch.cross(cell[0], cell[1], dim=0)])
    return volume / cross.norm(dim=-1)


def real_space_cutoff(eta: float, tol: float = EWALD_TOL) -> float:
    """`r_c` with `erfc(r_c / (2 eta)) / r_c <= tol` (the real-space remainder decays as the
    broad Gaussian's complement)."""
    r_c = 2.0 * eta * float(erfcinv(tol))
    for _ in range(4):
        r_c = 2.0 * eta * float(erfcinv(tol * max(r_c, 1.0)))
    return r_c


def reciprocal_cutoff(eta: float, volume: float, tol: float = EWALD_TOL) -> float:
    """`k_c` with `(4 pi / V) exp(-eta^2 k_c^2) / k_c^2 <= tol`."""
    k_c = math.sqrt(-math.log(tol)) / eta
    for _ in range(4):
        k_c = math.sqrt(max(-math.log(tol * volume * k_c ** 2 / (4.0 * math.pi)), 1.0)) / eta
    return k_c


def image_vectors(cell: torch.Tensor, r_c: float, margin: float = 0.0) -> torch.Tensor:
    """All lattice vectors `n @ cell` with `|n_a| <= ceil((r_c + margin) / h_a)`, `[n_L, 3]`,
    differentiable in the cell (the integer set is fixed by the detached cell). `margin` is
    the largest in-cell pair separation, so that every image pair within `r_c` is in the
    box."""
    heights = _cell_heights(cell.detach())
    n_max = [int(math.ceil((r_c + margin) / float(h))) for h in heights]
    grids = [torch.arange(-m, m + 1, dtype=cell.dtype, device=cell.device) for m in n_max]
    n = torch.cartesian_prod(*grids).reshape(-1, 3)
    return n @ cell


def reciprocal_vectors(cell: torch.Tensor, k_c: float) -> torch.Tensor:
    """Reciprocal vectors `2 pi m @ inv(cell)^T` with `|k| <= k_c`, `k != 0`, `[n_k, 3]`;
    the integer set is fixed by the detached cell so the result is differentiable in it."""
    rec = 2.0 * math.pi * torch.linalg.inv(cell).transpose(0, 1)          # rows: b_1, b_2, b_3
    rec_heights = _cell_heights(rec.detach())
    m_max = [int(math.ceil(k_c / float(h))) for h in rec_heights]
    grids = [torch.arange(-m, m + 1, dtype=cell.dtype, device=cell.device) for m in m_max]
    m = torch.cartesian_prod(*grids).reshape(-1, 3)
    m = m[(m != 0).any(dim=-1)]
    k = m @ rec
    keep = (k.detach().norm(dim=-1) <= k_c)
    return k[keep]


def pair_vectors_of(positions: torch.Tensor) -> torch.Tensor:
    """`d0_ij = r_i - r_j`, `[N, N, 3]`: the pair leaf every lattice sum is a function of."""
    return positions.unsqueeze(1) - positions.unsqueeze(0)


def lattice_sum(positions: torch.Tensor, cell: torch.Tensor,
                pair_function: Callable[[torch.Tensor], torch.Tensor],
                self_value: torch.Tensor, r_c: float, chunk: int = 32,
                pair_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`M_ij = sum_L phi(|r_i - r_j + L|)` over images within `r_c` (all `L` of the
    enclosing box), the `i = j, L = 0` term replaced by `self_value` (`[N]` or scalar): the
    analytic `r -> 0` limit of `phi`. `pair_function` maps distances `[N, N, c]` to values.
    `pair_vectors` (`d0_ij = r_i - r_j`, `[N, N, 3]`) may be given as the leaf instead of
    `positions`: then `M_ij` is a function of `d0_ij` alone, which is what `pair_gradient`
    differentiates."""
    n = positions.shape[0]
    d0 = pair_vectors_of(positions) if pair_vectors is None else pair_vectors   # [N, N, 3]
    # Minimum-image pair separations: the integer shift is a constant of the local
    # geometry (gradients flow through `d0` and `cell`), and it bounds the separations by
    # the cell so the image box stays small and complete.
    shift = torch.round((d0.detach() @ torch.linalg.inv(cell.detach())))
    d0 = d0 - shift @ cell
    images = image_vectors(cell, r_c, margin=float(d0.detach().norm(dim=-1).max()))
    zero = int(torch.nonzero(images.detach().abs().sum(dim=-1) < 1e-12).reshape(-1)[0])
    total = torch.zeros(n, n, dtype=positions.dtype, device=positions.device)
    eye = torch.eye(n, dtype=torch.bool, device=positions.device)
    limit = (self_value if torch.is_tensor(self_value)
             else torch.tensor(float(self_value), dtype=positions.dtype, device=positions.device))
    limit = limit.expand(n).unsqueeze(-1)                                  # [N, 1]
    for start in range(0, images.shape[0], chunk):
        block = images[start:start + chunk]                                 # [c, 3]
        d = d0.unsqueeze(2) + block.reshape(1, 1, -1, 3)                    # [N, N, c, 3]
        r = d.norm(dim=-1)
        if start <= zero < start + chunk:
            # The self pair at zero separation is a 0/0 in `phi`: evaluate it at a safe
            # distance (no gradient reaches it) and overwrite with the analytic limit.
            at_zero = torch.zeros(r.shape[-1], dtype=torch.bool, device=r.device)
            at_zero[zero - start] = True
            self_mask = eye.unsqueeze(-1) & at_zero.reshape(1, 1, -1)
            values = pair_function(torch.where(self_mask, torch.ones_like(r), r))
            values = torch.where(self_mask, limit.unsqueeze(-1).expand_as(values), values)
        else:
            values = pair_function(r)
        total = total + values.sum(dim=-1)
    return total


def ewald_matrix(positions: torch.Tensor, cell: torch.Tensor,
                 width_i: Union[float, torch.Tensor], width_j: Optional[Union[float, torch.Tensor]] = None,
                 eta: Optional[float] = None, tol: float = EWALD_TOL,
                 pair_vectors: Optional[torch.Tensor] = None, k_chunk: int = 256) -> torch.Tensor:
    """`E_PBC` in eV per unit charge pair (`COULOMB` included): `[N, N]`, symmetric, the
    diagonal including the Gaussian self term. `width_i` (`width_j`) are the Gaussian widths
    of the charges on the row (column) side -- a scalar or `[N]`; `width_j` defaults to
    `width_i`. Gradients flow to `positions` and `cell`. With `pair_vectors` (`d0_ij =
    r_i - r_j`) the matrix is evaluated as a function of that leaf instead -- the real-space
    sum through `lattice_sum`, the reciprocal sum per pair, `sum_k w_k cos(k . d0_ij)` --
    so that each entry depends on its own pair vector only (`pair_gradient`)."""
    if positions.dtype != torch.float64 or cell.dtype != torch.float64:
        raise TypeError("the Ewald matrix is evaluated in float64 (plan section 1)")
    n = positions.shape[0]
    s_i = _as_widths(width_i, n, positions)
    s_j = s_i if width_j is None else _as_widths(width_j, n, positions)
    w = pair_width(s_i, s_j)                                               # [N, N]
    if eta is None:
        eta = default_eta(cell, [float(s_i.max()), float(s_j.max())])
    if eta < float(max(s_i.max(), s_j.max())) - 1e-12:
        raise ValueError(f"the splitting width eta={eta} must not be below the broadest "
                         f"charge ({float(max(s_i.max(), s_j.max()))})")
    volume = torch.det(cell).abs()
    r_c = real_space_cutoff(eta, tol)
    k_c = reciprocal_cutoff(eta, float(volume.detach()), tol)
    two_eta = 2.0 * eta

    def remainder(r: torch.Tensor) -> torch.Tensor:
        return (torch.erf(r / w.unsqueeze(-1)) - torch.erf(r / two_eta)) / r

    # r -> 0 of the remainder on the self pair: 2/(sqrt(pi) w_ii) - 1/(sqrt(pi) eta).
    self_value = 2.0 / (_SQRT_PI * torch.diagonal(w)) - 1.0 / (_SQRT_PI * eta)
    real = lattice_sum(positions, cell, remainder, self_value, r_c, pair_vectors=pair_vectors)
    # Reciprocal space: (4 pi / V) sum_k exp(-eta^2 k^2) / k^2 cos(k . (r_i - r_j)).
    k = reciprocal_vectors(cell, k_c)                                      # [n_k, 3]
    k2 = (k * k).sum(dim=-1)
    weight = torch.exp(-(eta ** 2) * k2) / k2                              # [n_k]
    if pair_vectors is None:
        phase = positions @ k.transpose(0, 1)                              # [N, n_k]
        c = torch.cos(phase) * weight.sqrt().unsqueeze(0)
        s = torch.sin(phase) * weight.sqrt().unsqueeze(0)
        recip = (4.0 * math.pi / volume) * (c @ c.transpose(0, 1) + s @ s.transpose(0, 1))
    else:
        # Per pair (k is a reciprocal-lattice vector, so the minimum-image shift is immaterial).
        recip = torch.zeros(n, n, dtype=positions.dtype, device=positions.device)
        for start in range(0, k.shape[0], k_chunk):
            kk = k[start:start + k_chunk]                                     # [c, 3]
            recip = recip + (torch.cos(pair_vectors @ kk.transpose(0, 1)) * weight[start:start + k_chunk]).sum(-1)
        recip = (4.0 * math.pi / volume) * recip
    background = -4.0 * math.pi * eta ** 2 / volume
    return COULOMB * (real + recip + background)


def self_term(width: Union[float, torch.Tensor]) -> torch.Tensor:
    """`C / (sqrt(pi) r_g)`: the Gaussian self-energy kernel on the diagonal."""
    return COULOMB / (_SQRT_PI * torch.as_tensor(width, dtype=torch.float64))


def short_range_lattice_sum(positions: torch.Tensor, cell: torch.Tensor,
                            pair_function: Callable[[torch.Tensor], torch.Tensor],
                            self_value: Union[float, torch.Tensor], r_c: float,
                            pair_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Regime B: `K_ij = sum_L s(|r_ij + L|)` for a short-ranged `s`, images within `r_c`
    (the caller converges `r_c` against the tolerance); `K_ii = s(0) + sum_{L != 0} s(|L|)`
    with `s(0) = self_value`. Rewrapping-invariant by construction."""
    return lattice_sum(positions, cell, pair_function, torch.as_tensor(self_value, dtype=positions.dtype, device=positions.device), r_c,
                       pair_vectors=pair_vectors)


def pair_gradient(matrix_of_pairs: Callable[[torch.Tensor], Sequence[torch.Tensor]],
                  positions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """`D_ij = d M_ij / d d0_ij` for each matrix `M` returned by `matrix_of_pairs(d0)`, a
    function of the pair leaf `d0_ij = r_i - r_j` alone entry by entry (a lattice sum or the
    per-pair Ewald matrix): one first-order backward per matrix, no graph kept. `[N, N, 3]`
    each, detached. The geometry derivative of any contraction `sum_ij A_ij M_ij(R)` is then
    `d/dR_k = sum_j A_kj D_kj - sum_i A_ik D_ik` (`gradient_of_contraction`), which is how
    the training path takes the force of the kernel terms without a second-order graph
    through the lattice sums."""
    with torch.enable_grad():                       # callers may sit inside no_grad: the leaf needs a graph
        d0 = pair_vectors_of(positions.detach()).requires_grad_(True)
        matrices = matrix_of_pairs(d0)
        out = []
        for m, M in enumerate(matrices):
            (D,) = torch.autograd.grad(M.sum(), d0, retain_graph=m + 1 < len(matrices))
            out.append(D.detach())
    return tuple(out)


def gradient_of_contraction(A: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """`d/dR_k sum_ij A_ij M_ij` from the pair derivatives `D_ij = dM_ij/d(r_i - r_j)`:
    `sum_j A_kj D_kj - sum_i A_ik D_ik`. `A [..., N, N]` (attached), `D [..., N, N, 3]`
    (constant); returns `[..., N, 3]`."""
    return torch.einsum("...kj,...kjc->...kc", A, D) - torch.einsum("...ik,...ikc->...kc", A, D)


def madelung_constant_cubic() -> float:
    """`alpha_M` of a point charge in a cubic lattice with neutralising background (the
    Makov-Payne / Leslie-Gillan constant), for the tiling-ladder gate."""
    return 2.837297479


def energy(E: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """`0.5 q^T E q`."""
    return 0.5 * q @ E @ q
