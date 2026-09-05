"""Plan v8 section 7.2: the finite-difference harness, per term and assembled.

WHAT IT MEASURES. For each registered term (section 2.5) and for the assembled energy,
central differences of the RECOMPUTED energy against the analytic force, at
`h in {1e-2, 3e-3, 1e-3, 3e-4, 1e-4}` A, on a sample of (atom, component) pairs. The
error of a central difference against an exact derivative is `A h^2 + epsilon / h`:
truncation falling with `h`, and the evaluation's own noise floor `epsilon` (a solver
tolerance, a float32 trunk, a bisection) divided by `h`. The two are fitted from the five
points, and the pattern is the verdict:

  * a clean minimum inside the range, with the minimum error below tolerance: PASS --
    the analytic derivative is the derivative of the energy;
  * errors that do not fall with `h` at all -- flat across a factor of a hundred in `h`
    at a level far above the fitted `epsilon / h` -- is an h-INDEPENDENT FLOOR: a piece of
    the derivative is MISSING, because the analytic and numerical derivatives disagree by
    a fixed amount however carefully the numerical one is taken. That is the signature the
    plan asks for, and it is reported as `missing_derivative`, not as a tolerance failure;
  * anything else is a noisy failure, reported with the fitted floor.

STRAIN. The stress is checked against a HOMOGENEOUS deformation applied to the geometry
itself -- positions, cell and the edge shifts rebuilt from the strained cell -- and never
through the model's own displacement machinery. The distinction is the point: a term that
reads the undisplaced cell agrees with its own displacement-based finite difference
perfectly, because both sides make the same omission, and only a strain of the geometry
exposes the missing cell derivative. `stress_ab = (1/V) dE/dS_ab` in the code's convention
(`get_outputs` returns `-dE/dD` as the virial and `+dE/dD / V` as the stress).

FRAMES. Section 7.2 asks for an ordinary frame, one near a level crossing (selected by the
model's own spectrum), one inside a projector window (after the frontier projectors of
section 2.1 exist), in both gauges. `select_frames` picks the first two from a spectrum
readout; the third is recorded as not-yet-applicable until Stage 0.9.

The harness is a MEASUREMENT: it records a status per term, and Stage 0's acceptance is
that the status is recorded. Stage 1 is where per-term PASS is required.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.modules.defect_terms import registry, term_energies

__all__ = ["H_VALUES", "STRAINS", "FDPoint", "TermReport", "force_check", "strain_check",
           "fit_floor", "classify", "select_frames", "as_records"]

H_VALUES: Tuple[float, ...] = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
STRAINS: Tuple[float, ...] = (1e-3, -1e-3, 3e-4, -3e-4)


# ------------------------------------------------------------------------------ fitting

def fit_floor(h: Sequence[float], err: Sequence[float]) -> Tuple[float, float, float]:
    """Least-squares fit of `err(h) = c + A h^2 + eps / h`, returning `(A, eps, c)`.

    `c` is the h-INDEPENDENT part -- the amount by which the analytic and numerical
    derivatives disagree however small or large `h` is made -- and it is what a missing
    derivative looks like. Non-negative least squares by clipping: the three shapes are
    far from collinear over a factor of a hundred in `h`, so the clipped fit is stable.
    """
    h = np.asarray(h, dtype=np.float64)
    e = np.asarray(err, dtype=np.float64)
    X = np.stack([np.ones_like(h), h ** 2, 1.0 / h], axis=1)
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    c, A, eps = (float(max(x, 0.0)) for x in coef)
    return A, eps, c


def classify(h: Sequence[float], err: Sequence[float], tol: float
             ) -> Tuple[str, Dict[str, float]]:
    """The verdict on one derivative from its error curve. See the module docstring.

    Pass when the best error is within tolerance. Otherwise, a fitted floor `c` that is
    above tolerance and is the larger part of the best error is a missing derivative --
    the curve is not falling with `h` and cannot be made to; anything else is a noisy
    failure, reported with the fitted `epsilon` so the floor's origin can be traced.
    """
    h = list(h)
    err = [float(x) for x in err]
    A, eps, c = fit_floor(h, err)
    i_min = int(np.argmin(err))
    best, worst = err[i_min], max(err)
    info = dict(A=A, epsilon=eps, floor=c, min_error=best, h_at_min=h[i_min],
                max_error=worst)
    if best <= tol:
        return "pass", info
    if c > tol and c >= 0.5 * best:
        return "missing_derivative", info
    return "fail", info


# ------------------------------------------------------------------------------ records

@dataclass
class FDPoint:
    atom: int
    component: int
    analytic: float
    numerical: Dict[float, float] = field(default_factory=dict)      # h -> value
    errors: Dict[float, float] = field(default_factory=dict)         # h -> |num - ana|


@dataclass
class TermReport:
    term: str
    kind: str                       # "force" | "stress"
    status: str                     # pass | missing_derivative | fail | not_applicable
    tol: float
    fit: Dict[str, float]
    points: List[FDPoint]
    note: str = ""


def as_records(reports: Sequence[TermReport]) -> List[dict]:
    return [asdict(r) for r in reports]


# ------------------------------------------------------------------------------ forwards

def _energies(model, data: dict, **forward_kwargs) -> Dict[str, torch.Tensor]:
    out = model(data, training=False, compute_force=False, **forward_kwargs)
    return {k: v.detach() for k, v in term_energies(model, out).items()}


def _analytic_forces(model, data: dict) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """`-dE_term/dR` by autograd for every term, and the model's own `forces` output."""
    d = dict(data)
    d["positions"] = d["positions"].detach().clone().requires_grad_(True)
    # `training=True` keeps the graph the model's own force pass would otherwise free, so
    # the per-term gradients below can be taken from the same forward.
    out = model(d, training=True, compute_force=True)
    energies = term_energies(model, out)
    forces = {}
    for name, e in energies.items():
        # A term that is a CONSTANT in this forward (a cached base energy) has no graph to
        # differentiate: its analytic per-term force is zero here, and what the model
        # actually returns for it is in `assembled_model` (the cached forces).
        if not e.requires_grad:
            forces[name] = torch.zeros_like(d["positions"])
            continue
        g = torch.autograd.grad(e.sum(), d["positions"], retain_graph=True,
                                allow_unused=True)[0]
        forces[name] = (torch.zeros_like(d["positions"]) if g is None else -g).detach()
    return forces, out["forces"].detach()


def _displaced(data: dict, atom: int, component: int, h: float) -> dict:
    d = dict(data)
    pos = d["positions"].detach().clone()
    pos[atom, component] += h
    d["positions"] = pos
    return d


def sample_components(n_atoms: int, n_sample: int, seed: int = 0) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    atoms = rng.choice(n_atoms, size=min(n_sample, n_atoms), replace=False)
    return [(int(a), int(rng.integers(3))) for a in atoms]


def force_check(model, data: dict, components: Sequence[Tuple[int, int]],
                h_values: Sequence[float] = H_VALUES, tol: float = 1e-4,
                terms: Optional[Sequence[str]] = None,
                numerical_transform=None) -> List[TermReport]:
    """Per-term and assembled force finite differences on `components` of one batch.

    `tol` is in eV/A. The assembled term is compared BOTH as the autograd of the reported
    total energy and as the model's own `forces` output (`assembled_model`), so a force
    the model assembles differently from the energy it reports is caught.

    `numerical_transform`, when given, is applied to every displaced batch before its
    energy is evaluated and NOT to the analytic pass: with a base cache attached, the
    analytic forces come from the cached forward (the one training uses) while the numerical
    derivative is taken of the full, uncached energy (the transform strips `frame_key`), so
    the check asks whether the cached training force is the derivative of the physical
    energy -- which a cache of later-block features cannot supply.
    """
    names = [t.name for t in registry(model)] + ["assembled"]
    if terms is not None:
        names = [n for n in names if n in set(terms)]
    analytic, model_forces = _analytic_forces(model, data)
    points: Dict[str, List[FDPoint]] = {n: [] for n in names + ["assembled_model"]}
    xf = (lambda d: d) if numerical_transform is None else numerical_transform
    with torch.no_grad():
        for atom, comp in components:
            plus = {h: _energies(model, xf(_displaced(data, atom, comp, h))) for h in h_values}
            minus = {h: _energies(model, xf(_displaced(data, atom, comp, -h)))
                     for h in h_values}
            for name in names:
                ana = float(analytic[name][atom, comp])
                p = FDPoint(atom=atom, component=comp, analytic=ana)
                for h in h_values:
                    num = -float((plus[h][name].sum() - minus[h][name].sum()) / (2 * h))
                    p.numerical[h] = num
                    p.errors[h] = abs(num - ana)
                points[name].append(p)
            # The model's own force against the assembled numerical derivative.
            ana = float(model_forces[atom, comp])
            p = FDPoint(atom=atom, component=comp, analytic=ana)
            for h in h_values:
                num = -float((plus[h]["assembled"].sum() - minus[h]["assembled"].sum())
                             / (2 * h))
                p.numerical[h] = num
                p.errors[h] = abs(num - ana)
            points["assembled_model"].append(p)
    reports = []
    for name, pts in points.items():
        if not pts:
            continue
        # The worst component at each h, so one missing derivative on one atom is enough.
        err = [max(pt.errors[h] for pt in pts) for h in h_values]
        status, fit = classify(h_values, err, tol)
        reports.append(TermReport(term=name, kind="force", status=status, tol=tol,
                                  fit=fit, points=pts))
    return reports


# ------------------------------------------------------------------------------ strain

def _strained(data: dict, strain: torch.Tensor) -> dict:
    """The geometry under `x -> x + x . S`: positions, cell and the edge shifts, and no
    displacement tensor -- see the module docstring for why."""
    d = dict(data)
    pos = d["positions"].detach()
    cell = d["cell"].detach().reshape(-1, 3, 3)
    batch = d["batch"]
    S = strain.to(pos.dtype)
    d["positions"] = pos + pos @ S
    new_cell = cell + cell @ S
    d["cell"] = new_cell.reshape(-1, 3)
    sender = d["edge_index"][0]
    d["shifts"] = torch.einsum("be,bec->bc", d["unit_shifts"].to(pos.dtype),
                               new_cell[batch[sender]])
    return d


def strain_check(model, data: dict, strains: Sequence[float] = STRAINS,
                 tol: float = 1e-4, terms: Optional[Sequence[str]] = None,
                 numerical_transform=None) -> List[TermReport]:
    """The analytic stress against a homogeneous strain of the geometry, per term.

    Only the assembled stress is a model output; per-term stresses are taken here as the
    autograd of each term's energy with respect to the displacement, exactly as the model
    forms its own, so a term's cell dependence is checked on the same footing as the
    total's. `tol` is in eV/A^3. Both magnitudes of section 7.2 are used, and the two
    signs of each give the central difference.
    """
    names = [t.name for t in registry(model)] + ["assembled"]
    if terms is not None:
        names = [n for n in names if n in set(terms)]
    # Analytic: dE/dD per term through the model's displacement, in stress units.
    d = dict(data)
    num_graphs = int(d["ptr"].numel() - 1)
    d["positions"] = d["positions"].detach().clone().requires_grad_(True)
    out = model(d, training=True, compute_force=True, compute_stress=True)
    energies = term_energies(model, out)
    disp = out["displacement"]
    volume = torch.linalg.det(d["cell"].reshape(-1, 3, 3)).abs()
    analytic = {}
    for name, e in energies.items():
        if not e.requires_grad:      # a cached constant; see _analytic_forces
            analytic[name] = torch.zeros_like(disp)
            continue
        g = torch.autograd.grad(e.sum(), disp, retain_graph=True, allow_unused=True)[0]
        analytic[name] = (torch.zeros_like(disp) if g is None else g).detach() / volume.reshape(-1, 1, 1)
    analytic["assembled_model"] = out["stress"].detach()
    magnitudes = sorted({abs(s) for s in strains})
    pairs = [(a, b) for a in range(3) for b in range(a, 3)]
    points: Dict[str, List[FDPoint]] = {n: [] for n in names + ["assembled_model"]}
    with torch.no_grad():
        for a, b in pairs:
            S = torch.zeros(3, 3, dtype=torch.float64)
            S[a, b] = S[b, a] = 1.0
            evals = {}
            xf = (lambda x: x) if numerical_transform is None else numerical_transform
            for eps in magnitudes:
                evals[eps] = (_energies(model, xf(_strained(data, eps * S))),
                              _energies(model, xf(_strained(data, -eps * S))))
            for name in names + ["assembled_model"]:
                key = "assembled" if name == "assembled_model" else name
                p = FDPoint(atom=-1, component=3 * a + b,
                            analytic=float(analytic[name][:, a, b].sum()))
                for eps in magnitudes:
                    plus, minus = evals[eps]
                    dE = float(plus[key].sum() - minus[key].sum()) / (2 * eps)
                    # dE/dS_ab + dE/dS_ba for a != b, dE/dS_aa on the diagonal; the code's
                    # stress is (1/V) dE/dD_ab with D symmetrised, i.e. W_ab itself.
                    w = dE / (2.0 if a != b else 1.0)
                    num = float(sum(w / float(v) for v in volume) / max(num_graphs, 1)) \
                        if num_graphs == 1 else w / float(volume[0])
                    p.numerical[eps] = num
                    p.errors[eps] = abs(num - p.analytic)
                points[name].append(p)
    reports = []
    for name, pts in points.items():
        err = [max(pt.errors[e] for pt in pts) for e in magnitudes]
        # Two magnitudes cannot support the two-parameter fit; the verdict is on the
        # smaller strain's error and whether halving the strain moved it.
        best = min(err)
        fit = dict(min_error=best, max_error=max(err), strains=magnitudes)
        if best <= tol:
            status = "pass"
        elif max(err) / max(best, 1e-300) < 4.0:
            status = "missing_derivative"
        else:
            status = "fail"
        reports.append(TermReport(term=name, kind="stress", status=status, tol=tol,
                                  fit=fit, points=pts))
    return reports


# ------------------------------------------------------------------------------ frames

def select_frames(gaps: Sequence[float], n_ordinary: int = 1, n_crossing: int = 1
                  ) -> Dict[str, List[int]]:
    """Frame indices by the model's own frontier gap: `ordinary` are the median-gap
    frames, `near_crossing` the smallest-gap ones. `in_projector_window` is empty until
    the projectors of section 2.1 exist (Stage 0.9), and is reported as such."""
    order = list(np.argsort(np.asarray(gaps, dtype=np.float64)))
    n = len(order)
    mid = order[n // 2: n // 2 + n_ordinary]
    return dict(ordinary=[int(i) for i in mid],
                near_crossing=[int(i) for i in order[:n_crossing]],
                in_projector_window=[])
