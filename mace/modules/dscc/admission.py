"""Plan section 6: energy admission -- null-gated, numeric, out-of-fold, coverage-gated,
per fold and per size.

Inside each outer training fold, for each size class `L`, within the charged window of the
collective coordinate `d` (the Pb-Pb distance across the vacancy, identified label-free as
the two Pb whose first shell holds five Cl):

    coverage(L): every bin of the charged window (registered width) holds at least n_min
                 out-of-fold neutral frames of this fold; any empty bin -> s0(L)
                 "unmeasurable", charged ENERGIES at L not admitted.
    s0(L) +- SE: slope of E_label(V0) - E_base against d, base residuals OUT-OF-FOLD only.
    sQ(L):       slope of E_label(V+) - E_base against d, before any head.
    admit(L)  <=> coverage(L) and |s0(L)| + z SE(s0) <= s_tol.

Forces are admitted at every size. The table is stored before the fold's results are
opened. Nothing here reaches the model: it is loss metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.modules.dscc.kernels import minimum_image_distances


def collective_coordinate(positions: torch.Tensor, cell: torch.Tensor, numbers: Sequence[int],
                          cation: int = 82, anion: int = 17, shell_gap: float = 4.0
                          ) -> Optional[float]:
    """`d`: the minimum-image distance between the two flanking Pb (first shell of five
    Cl); None when the frame has no vacancy (0 such Pb) or an ambiguous one (not 2)."""
    z = torch.as_tensor([int(x) for x in numbers])
    r = minimum_image_distances(positions.detach(), cell.detach())
    pb = torch.nonzero(z == cation).reshape(-1)
    cl = torch.nonzero(z == anion).reshape(-1)
    if pb.numel() < 2 or cl.numel() < 6:
        return None
    d = torch.sort(r[pb][:, cl], dim=1).values
    flank = pb[d[:, 5] > shell_gap]
    if flank.numel() != 2:
        return None
    return float(r[flank[0], flank[1]])


def slope_with_se(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """`(slope, standard error, intercept)` of the least-squares line `y = a + s x`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan"), float("inf"), float("nan")
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(x.size - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[1]), float(np.sqrt(cov[1, 1])), float(coef[0])


@dataclass(frozen=True)
class AdmissionConfig:
    """Registered (plan section 6 / 11; defaults to confirm before Phase 2 results)."""
    bin_width: float = 0.2      # A, bins of the charged d window
    n_min: int = 3              # out-of-fold neutral frames per bin
    s_tol: float = 0.05         # eV/A on |s0| + z SE
    z: float = 2.0
    base_protocol: str = "cf_base_4fold_seed0"


@dataclass
class SizeAdmission:
    size_class: int
    n_charged: int
    n_neutral_oof: int
    window: Tuple[float, float]
    bins: List[Tuple[float, float, int]]      # (lo, hi, out-of-fold neutral count)
    coverage: bool
    s0: float
    s0_se: float
    sQ: float
    sQ_se: float
    admitted: bool
    reason: str


def admission_table(charged_d: Dict[int, Sequence[float]], charged_resid: Dict[int, Sequence[float]],
                    neutral_d: Dict[int, Sequence[float]], neutral_resid_oof: Dict[int, Sequence[float]],
                    cfg: AdmissionConfig = AdmissionConfig()) -> Dict[int, SizeAdmission]:
    """Per size class: the coverage table, `s0 +- SE` (out-of-fold neutral residuals),
    `sQ`, and the admission decision. Inputs are keyed by size class."""
    out: Dict[int, SizeAdmission] = {}
    for size in sorted(charged_d):
        dq_ = np.asarray(charged_d[size], dtype=np.float64)
        rq = np.asarray(charged_resid[size], dtype=np.float64)
        d0 = np.asarray(neutral_d.get(size, []), dtype=np.float64)
        r0 = np.asarray(neutral_resid_oof.get(size, []), dtype=np.float64)
        if dq_.size == 0:
            continue
        lo, hi = float(dq_.min()), float(dq_.max())
        edges = np.arange(lo, hi + cfg.bin_width, cfg.bin_width)
        if edges[-1] < hi:
            edges = np.append(edges, hi)
        bins = []
        coverage = True
        for a, b in zip(edges[:-1], edges[1:]):
            count = int(((d0 >= a) & (d0 < b)).sum()) if d0.size else 0
            bins.append((float(a), float(b), count))
            if count < cfg.n_min:
                coverage = False
        s0, s0_se, _ = slope_with_se(d0, r0) if d0.size else (float("nan"), float("inf"), float("nan"))
        sq, sq_se, _ = slope_with_se(dq_, rq)
        if not coverage:
            admitted, reason = False, "coverage failed: s0 unmeasurable in the charged window"
        elif not np.isfinite(s0) or abs(s0) + cfg.z * s0_se > cfg.s_tol:
            admitted, reason = False, f"|s0| + z SE = {abs(s0) + cfg.z * s0_se:.4f} > s_tol {cfg.s_tol}"
        else:
            admitted, reason = True, "admitted"
        out[size] = SizeAdmission(size_class=size, n_charged=int(dq_.size), n_neutral_oof=int(d0.size),
                                  window=(lo, hi), bins=bins, coverage=coverage, s0=s0, s0_se=s0_se,
                                  sQ=sq, sQ_se=sq_se, admitted=admitted, reason=reason)
    return out


def table_record(table: Dict[int, SizeAdmission], cfg: AdmissionConfig, fold: int) -> Dict[str, object]:
    """The stored record (plan: stored before the fold's results are opened)."""
    return {"fold": int(fold), "config": asdict(cfg), "sizes": {str(k): asdict(v) for k, v in table.items()}}
