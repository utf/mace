"""Composition-class constructor (plan v8 section 2.1): the integers of a charge state.

A COMPOSITION CLASS is the multiset of species in a cell together with its pristine
reference. For each class this module establishes, once and cached, the integers that every
per-frame carrier count is then built from:

    M_VB,sigma       occupied levels of H_0 on the class reference geometry that belong to
                     the valence manifold, counted against the ALIGNED pristine valence edge
    n_e,sigma^class  = max(N_sigma - M_VB,sigma, 0)     electrons above the valence manifold
    n_h,sigma^class  = max(M_VB,sigma - N_sigma, 0)     holes in it
    Q_core           = sum_sigma (n_e - n_h)^class

where `N_sigma` is the per-spin electron count of the reference state `S_ref` (the count
fill of the composition's valence electrons, `resolve_fills` with zero counters). Nothing
here forms a site-resolved oxidation state or orbital count, and nothing here reads a defect
label, a position of "the defect", or a charge counter: the integers depend only on the
composition, the pristine reference and the head's Hamiltonian at S_ref.

TIER 1 (this module's `tier1`) is the direct edge count. The pristine valence edge is
aligned to the class reference cell by the CONTINUUM manifold -- the median offset between
matched quantiles of the two occupied manifolds, over a quantile window that stops well
below the frontier (`QUANTILES`, the alignment first written for the depth scorer
`defect-perovskite/b6_depth_edges.py::align`) -- and the class is accepted at Tier 1 only
when no eigenvalue lies within +-delta of the aligned edge and the pristine gap is at least
four smearing widths wide. Otherwise the class is AMBIGUOUS at Tier 1 and is handed to
Tier 2, the valence-subspace continuation, which lives in the fenced section at the end of
this module and NOWHERE ELSE (`TIER2_NAMES`; the AST test in
`tests/extensions/defect/test_composition_classes.py` asserts their absence from every
production module).

THE CLASS REFERENCE GEOMETRY is the class's FIRST FRAME, always. The plan allows "the ideal
defect geometry built from the pristine cell when constructible"; on the data in hand it is
not: the stoichiometric training frames are thermal snapshots (cells 15.80 vs 16.25 A for
the same 80-atom composition), so there is no ideal cell to build from, and building V_Cl
from one would require choosing WHICH Cl to remove -- orthorhombic CsPbCl3 has inequivalent
Cl sites -- which is a defect position, and principle 9 forbids the model any such input.
The cache records the frame key of the geometry used.

PER-FRAME COUNTS (`frame_counts`) net the class integers against the exact counters:

    net_sigma = n_e,sigma^class - n_h,sigma^class + DeltaN_sigma
    n_e,sigma = max(net, 0),  n_h,sigma = max(-net, 0),  q_F = sum_sigma (n_h - n_e)

so that V_Cl+ (the V_Cl class, one electron removed from the majority channel) has
n_e = 0 and q_F = 0 -- the plan's benchmark -- rather than one electron AND one hole. The
identity q_F = Q_formal - Q_core holds by construction and is asserted anyway.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.modules.defect_carriers import (
    assert_charge_consistency,
    carrier_multiplicity,
    class_counts,
    counts_from_excess,
    signed_excess,
)
from mace.modules.defect_rank import (
    DEFAULT_G_NUM,
    RankAnchor,
    RankRoutingError,
    alignment_bound,
    predict_rank,
    select_anchor,
    verify as verify_rank,
)
from mace.modules.defect_state import COUNT_FILL, StateBatch

__all__ = ["CLASS_TABLE_VERSION", "QUANTILES", "TIER2_NAMES", "EdgeAlignment",
           "ClassRecord", "composition_key", "is_stoichiometric", "reference_fill",
           "align_edges", "tier1", "class_integers", "frame_counts", "head_spectra",
           "tiling_factors", "tiling_map", "tile_frame", "build_class_table",
           "lookup_class", "verify_class_table", "describe", "DEFAULT_CONSTRUCTOR",
           "constructor_config", "frame_counts_batch", "frame_static_densities",
           "pristine_placement", "species_charges", "tiled_pristine_scaled"]

CLASS_TABLE_VERSION = 1
BOUNDARY_TOL = 1e-9   # eV; a level exactly on the window's edge is on the edge, not inside

# The occupied-manifold window for the edge alignment: quantiles of the occupied spectrum
# from 5 % to 60 %, i.e. the continuum well below the frontier, so a frontier level inside
# the gap (V_Cl^0's) or a resonance at the edge cannot move the alignment. From the depth
# scorer (b6_depth_edges.QUANTILES), where it was chosen for the same reason.
QUANTILES = np.linspace(0.05, 0.60, 23)

# Names of the Tier 2 machinery. The site correspondence and the union basis are used ONLY
# inside the constructor; a production module that mentions any of these has crossed the
# fence, and the AST test fails it.
TIER2_NAMES = ("site_correspondence", "union_basis", "transport_valence_subspace",
               "endpoint_classification", "tier2", "linear_sum_assignment", "ghost_orbitals",
               "interpolated_hamiltonian", "tier2_continuation")

# The constructor's parameters (plan section 3: every one of them is config and round-trips).
# `delta` and `window` default to 2x and 1x the head's smearing width when None.
DEFAULT_CONSTRUCTOR: Dict[str, Any] = {
    "delta": None,        # eV, the counting margin above VBM_al (None: 2 x smearing width)
    "window": None,       # eV, half-width of the Tier 1 ambiguity window about the cut
                          # (None: one smearing width)
    "r_match": 2.0,       # A, the largest displacement a site correspondence may carry.
                          # Plan section 2.7 says "half the pristine nearest-neighbour
                          # distance" (1.4 A for Pb-Cl); on the data in hand thermal Cl swing
                          # up to 1.6 A between snapshots, and 1.4 A turned a swung Cl into
                          # a spurious ghost + addition. Half the Cl-Cl distance instead.
    "e_sink": None,       # eV, where decoupled orbitals are parked (None: 50 eV above the
                          # pristine conduction edge, section 2.7); the transport never
                          # depends on it (see `interpolated_hamiltonian`)
    "eta": 1.0e-3,        # endpoint classification threshold on gamma (section 2.7)
    "dlambda": 0.02,      # transport step; the second schedule uses dlambda / 2
}


def constructor_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The full constructor config with `overrides` applied; unknown keys are refused."""
    config = dict(DEFAULT_CONSTRUCTOR)
    for key, value in (overrides or {}).items():
        if key not in config:
            raise ValueError(f"unknown class-constructor parameter {key!r}; "
                             f"expected one of {sorted(config)}")
        config[key] = None if value is None else float(value)
    return config


# ------------------------------------------------------------------ the pure integer layer


def composition_key(atomic_numbers: Sequence[int]) -> str:
    """`"17x47,55x16,82x16"`: the species multiset, sorted by Z. The class's name."""
    zs, counts = np.unique(np.asarray([int(z) for z in atomic_numbers]), return_counts=True)
    return ",".join(f"{int(z)}x{int(c)}" for z, c in zip(zs, counts))


def is_stoichiometric(atomic_numbers: Sequence[int], formula: Dict[int, float]) -> bool:
    """Whether the multiset is an integer multiple of `formula` (Z -> count per unit).

    A composition test, never an atom-count one: the largest cells in the perovskite set are
    159-atom DEFECT supercells.
    """
    zs, counts = np.unique(np.asarray([int(z) for z in atomic_numbers]), return_counts=True)
    present = {int(z): int(c) for z, c in zip(zs, counts)}
    if set(present) != {int(z) for z, n in formula.items() if n > 0}:
        return False
    ratios = [present[z] / float(formula[z]) for z in present]
    return bool(np.allclose(ratios, ratios[0], atol=1e-9)
                and abs(ratios[0] - round(ratios[0])) < 1e-9)


def reference_fill(n_total: int) -> Tuple[int, int]:
    """`(N_maj, N_min)` of `S_ref`: the count fill of `n_total` valence electrons, exactly as
    `resolve_fills` forms the neutral origin."""
    from mace.modules.defect_counting import resolve_fills

    (_, _), (n_maj, n_min) = resolve_fills(int(n_total), (0, 0, 0, 0))
    return int(round(n_maj)), int(round(n_min))


@dataclass(frozen=True)
class EdgeAlignment:
    """The pristine edges carried into the class cell by the continuum manifold."""
    shift: float          # median quantile offset, class minus pristine (eV)
    spread: float         # interquartile range of the per-quantile offsets (eV)
    vbm_al: float         # pristine VBM + shift
    cbm_al: float         # pristine CBM + shift
    gap_pristine: float   # pristine CBM - VBM


def align_edges(spectrum: np.ndarray, n_occ: int, pristine: np.ndarray, n_pristine: int,
                quantiles: np.ndarray = QUANTILES) -> EdgeAlignment:
    """Align the pristine valence edge to `spectrum` by the occupied continuum.

    `n_occ` and `n_pristine` are the S_ref occupied counts of the two spectra (NOT anything
    this module outputs -- the alignment must not depend on the integers it is used to
    establish). The window `quantiles` is applied to the occupied part of each spectrum.
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=np.float64))
    pristine = np.sort(np.asarray(pristine, dtype=np.float64))
    if n_pristine < 1 or n_pristine >= pristine.size:
        raise ValueError(f"pristine occupied count {n_pristine} outside 1..{pristine.size - 1}")
    if n_occ < 2 or n_occ > spectrum.size:
        raise ValueError(f"class occupied count {n_occ} outside 2..{spectrum.size}")
    a = spectrum[:n_occ]
    b = pristine[:n_pristine]
    off = np.quantile(a, quantiles) - np.quantile(b, quantiles)
    shift = float(np.median(off))
    spread = float(np.subtract(*np.percentile(off, [75, 25])))
    vbm, cbm = float(pristine[n_pristine - 1]), float(pristine[n_pristine])
    return EdgeAlignment(shift=shift, spread=spread, vbm_al=vbm + shift, cbm_al=cbm + shift,
                         gap_pristine=cbm - vbm)


def _serialisable(value: Any) -> Any:
    """Replace non-finite floats with `None`, recursively.

    The class table is compared for equality after a pickle round-trip (and after config
    extraction and rebuild). Those produce fresh objects, and `NaN != NaN`, so a single
    non-finite number anywhere in a record makes the table unequal to itself -- with a
    diff that shows two apparently identical dicts. `None` means "not measured", which is
    what a NaN was being used to say anyway.
    """
    if isinstance(value, dict):
        return {k: _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def composition_counts(key: str) -> Dict[int, int]:
    """`{Z: count}` from a class key like `17x47,55x16,82x16`."""
    counts: Dict[int, int] = {}
    for piece in str(key).split(","):
        z, _, n = piece.partition("x")
        counts[int(z)] = int(n)
    return counts


def homology_signature(key: str, pristine_key: str, tiling: Sequence[int] = (1, 1, 1)) -> str:
    """The composition difference against the pristine reference TILED to this cell.

    Label-free: arithmetic on two multisets of species counts, with no notion of "vacancy"
    or "defect" anywhere.

    The tiling factor is essential and not a detail. Differencing against the *untiled*
    pristine key makes the signature grow with the cell -- a single Cl vacancy in a 2x cell
    reads as `17+23,55+8,82+8` rather than `17-1` -- so no two sizes of one family would
    ever match and the transport equation could never fire. The signature must be exactly
    the size-invariant part of the composition.
    """
    volume = 1
    for factor in tiling:
        volume *= int(factor)
    present, pristine = composition_counts(key), composition_counts(pristine_key)
    diff = {z: present.get(z, 0) - pristine.get(z, 0) * volume
            for z in sorted(set(present) | set(pristine))}
    return ",".join(f"{z}{d:+d}" for z, d in diff.items() if d != 0) or "pristine"


def precision_replicas(hamiltonian) -> List[np.ndarray]:
    """Registered precision perturbation: the same H diagonalised at reduced precision.

    Condition 4 of the verifier asks that the certified rank stay separated under the
    registered perturbations. This is the cheap half of that -- it reuses an H that has
    already been built, so it costs one extra eigensolve and no model forward. A
    small-geometry replica is the other half and is opt-in through the constructor config,
    because it needs a fresh forward per replica; whichever were applied is recorded on the
    class so a record cannot claim more stability than it was tested for.
    """
    h = hamiltonian.detach().double()
    return [np.sort(np.linalg.eigvalsh(h.cpu().numpy())),
            np.sort(np.linalg.eigvalsh(h.float().cpu().numpy()).astype(np.float64))]


def class_integers(m_vb: Sequence[int], n_sigma: Sequence[int]
                   ) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """`((n_e,maj, n_e,min), (n_h,maj, n_h,min), Q_core)` from either tier's `M_VB,sigma`.

    Delegates to the signed-excess algebra of addendum section 3.3, which is the same
    arithmetic written around one signed `d_sigma = N_sigma - M_VB,sigma` per channel. The
    production reference is neutral, so `Q_core = -q_F(S_ref) = sum_sigma d_sigma`.
    """
    counts = class_counts(m_vb, n_sigma, q_formal_ref=0)
    return counts["n_e"], counts["n_h"], counts["q_core"]


@dataclass(frozen=True)
class ClassRecord:
    """One composition class's cached integers, with how they were established."""
    key: str
    n_atoms: int
    reference_frame_key: int
    n_total: int
    n_sigma: Tuple[int, int]
    tier: Optional[int]                    # 1, 2, or None when the class is ambiguous
    ambiguous: bool
    reason: str                            # "" when accepted; why not otherwise
    m_vb: Tuple[int, int]
    n_e: Tuple[int, int]
    n_h: Tuple[int, int]
    q_core: int
    vbm_al: float
    cbm_al: float
    shift: float
    spread: float
    gap_pristine: float
    delta: float
    nearest: float                         # closest level to the cut (signed, eV)
    tiling: Tuple[int, int, int] = (1, 1, 1)   # pristine reference tiling aligned against
    path_agreement: Optional[bool] = None  # Tier 2 only
    schedule_agreement: Optional[bool] = None
    gamma: Tuple[float, ...] = field(default_factory=tuple)   # Tier 2 endpoint spectrum
    correspondence: Optional[Dict[str, Any]] = None           # Tier 2 site correspondence
    tier1_reason: str = ""                                    # why Tier 1 handed it on
    perm: Tuple[int, int, int] = (0, 1, 2)                    # frame axis i <- tiled axis perm[i]
    placement: Optional[Dict[str, Any]] = None                # density-level pristine placement
    # Addendum 3.3: the signed excess the counts are read from, and the absolute frontier
    # multiplicity of the REFERENCE state. Defaulted so records written before v8.1 still
    # load; `refreshed` recomputes them from m_vb and n_sigma, which is exact.
    d_sigma: Tuple[int, int] = (0, 0)
    m_f: int = 0
    # The Tier-1 verifier's own record (addendum 3.2/3.5): predicted and accepted rank,
    # thresholds, gaps, window, u_al and its provenance, and every gate result. Stored so a
    # class cannot claim more stability than it was actually tested for.
    tier1_record: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # d_sigma = n_e - n_h is an identity: the two counts are the positive and negative
        # parts of the same signed excess. Derived from the record's own counts rather than
        # re-derived from (m_vb, n_sigma), so that the record has a single source of truth
        # and legacy records restore to the same integers they were written with.
        excess = tuple(int(e) - int(h) for e, h in zip(self.n_e, self.n_h))
        object.__setattr__(self, "d_sigma", excess)
        object.__setattr__(self, "m_f", carrier_multiplicity(self.n_e, self.n_h))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassRecord":
        d = dict(d)
        for k in ("n_sigma", "m_vb", "n_e", "n_h", "gamma", "tiling", "perm", "d_sigma"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**d)

    @property
    def counted(self) -> bool:
        return self.tier is not None and not self.ambiguous


def frame_counts(record: ClassRecord, state: StateBatch, g: int
                 ) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """Per-frame `(n_e,sigma, n_h,sigma, q_F)` from the class integers and the exact state.

    Netted per spin (module docstring); `q_F = Q_formal - Q_core` is asserted, not assumed.
    A class without integers (ambiguous) has no per-frame counts, and this refuses rather
    than returning zeros that would look like a pristine frame.
    """
    if not record.counted:
        raise ValueError(
            f"composition class {record.key} carries no core/frontier decomposition "
            f"({record.reason or 'ambiguous'}); no per-frame carrier counts exist for it")
    dn = (int(state.delta_n[g, 0]), int(state.delta_n[g, 1]))
    # d_sigma(S) = N_sigma(S) - M_VB,sigma = d_sigma(S_ref) + Delta N_sigma(S). Recomputed
    # from the signed excess rather than by updating separate electron and hole counters,
    # which would go negative wherever a charge sequence crosses the valence rank.
    excess = tuple(record.d_sigma[s] + dn[s] for s in range(2))
    n_e, n_h, q_f = counts_from_excess(excess)
    assert_charge_consistency(int(state.q_formal[g]), record.q_core, q_f,
                              context=f"composition class {record.key}")
    return n_e, n_h, q_f


def frame_multiplicity(record: ClassRecord, state: StateBatch, g: int) -> int:
    """`m_F` for one frame: every frontier carrier present, both signs, both spins.

    The Stage 4-6 support guard is applied to this, and to the reference state's value,
    before any energy, force or stress is built (addendum section 3.4).
    """
    n_e, n_h, _ = frame_counts(record, state, g)
    return carrier_multiplicity(n_e, n_h)


# ------------------------------------------------------------------ the spectrum layer


def head_spectra(model, batch_dict: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """The counting head's Hamiltonian, spectrum and valence count per graph, AT S_ref.

    The counters in `batch_dict` are overridden with zeros: the class integers are a property
    of the composition at its reference state, whatever charge state the frame was recorded
    in. Runs the full model forward once (no gradients) and reads the head's `internals`,
    the same hook the scorers use, so the Hamiltonian is the one the model actually builds --
    including its Madelung on-site shift -- rather than a re-assembly written here.
    """
    head = getattr(model, "spectral", None)
    if head is None or not hasattr(head, "valence"):
        raise ValueError("the composition-class constructor needs a counting head")
    d = dict(batch_dict)
    counts = d["carrier_counts"]
    d["carrier_counts"] = torch.zeros_like(counts)
    grabbed: Dict[str, Any] = {}
    original = head.forward

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(d, training=False, compute_force=False)
    finally:
        head.forward = original
        model.train(was_training)
    if "H_orbital" not in grabbed:
        raise RuntimeError("the counting head reported no Hamiltonian in its internals")
    hams = grabbed["H_orbital"]
    batch = grabbed["batch"].cpu()
    species = d["node_attrs"].argmax(dim=-1).cpu()
    valence = head.valence.to(torch.float64).cpu()
    out = []
    for g, H in enumerate(hams):
        sel = batch == g
        n_total = int(round(float(valence[species[sel].long()].sum())))
        H64 = H.detach().to(torch.float64).cpu()
        lam = torch.linalg.eigvalsh(H64).numpy()
        out.append({"H": H64, "spectrum": np.sort(lam), "n_total": n_total,
                    "n_atoms": int(sel.sum())})
    return out


def tiling_map(cell: np.ndarray, reference_cell: np.ndarray, n_atoms: int,
               n_reference: int, tol: float = 0.05
               ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """`(factors, perm)`: `cell` is the reference tiled by `factors` (in the REFERENCE's
    axis order) with its lattice vectors relabelled so that `cell[i]` corresponds to the
    tiled reference's vector `perm[i]`; None if no such integer tiling exists within `tol`
    (relative, per lattice vector).

    The relabelling matters on the data in hand: the 80-atom pristine cell is 2 x 2 x 1 of
    the 20-atom orthorhombic cell and the 159-atom V_Cl cell is 2 x 2 x 2 of it with the
    axes in a different order, so the latter is the former tiled (1, 1, 2) and rotated. The
    head's spectrum is invariant under the rotation (Slater-Koster hoppings and the trunk's
    invariants see distances and angles only), so the spectrum needs only the factors; the
    site correspondence needs the relabelling too. Thermal cells differ from the reference
    by ~2 %, so ratios are rounded per axis, the cell ANGLES are required to match under the
    same relabelling, and the rounded product is checked against the atom-count ratio
    (tolerating the defect's missing or extra atoms).
    """
    from itertools import permutations

    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    ref = np.asarray(reference_cell, dtype=np.float64).reshape(3, 3)
    lengths, ref_lengths = np.linalg.norm(cell, axis=1), np.linalg.norm(ref, axis=1)

    def cosines(c):
        u = c / np.linalg.norm(c, axis=1, keepdims=True)
        return np.array([abs(u[0] @ u[1]), abs(u[0] @ u[2]), abs(u[1] @ u[2])])

    for perm in permutations(range(3)):
        ratio = lengths / ref_lengths[list(perm)]
        n = np.round(ratio).astype(int)
        if (n < 1).any() or (np.abs(ratio - n) > tol * n).any():
            continue
        permuted = ref[list(perm)]
        if (np.abs(cosines(cell) - cosines(permuted)) > tol).any():
            continue
        if abs(n_atoms / float(n_reference) - int(np.prod(n))) > 0.25 * int(np.prod(n)):
            continue
        factors = [0, 0, 0]
        for i, axis in enumerate(perm):
            factors[axis] = int(n[i])
        return (factors[0], factors[1], factors[2]), (int(perm[0]), int(perm[1]), int(perm[2]))
    return None


def tiling_factors(cell: np.ndarray, reference_cell: np.ndarray, n_atoms: int,
                   n_reference: int, tol: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """The factors of `tiling_map`, or None."""
    found = tiling_map(cell, reference_cell, n_atoms, n_reference, tol)
    return None if found is None else found[0]


def tile_frame(numbers: Sequence[int], positions: np.ndarray, cell: np.ndarray,
               factors: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate a periodic frame `factors` times along its lattice vectors."""
    numbers = np.asarray(numbers, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    shifts = np.array([[i, j, k] for i in range(factors[0]) for j in range(factors[1])
                       for k in range(factors[2])], dtype=np.float64) @ cell
    pos = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    nums = np.tile(numbers, len(shifts))
    return nums, pos, cell * np.asarray(factors, dtype=np.float64).reshape(3, 1)


def _frame_geometry(fr, model) -> Tuple[List[int], np.ndarray, np.ndarray]:
    return (_numbers_of(fr, model), fr.positions.detach().cpu().numpy(),
            fr.cell.detach().cpu().numpy().reshape(3, 3))


def _graph_cutoff(fr) -> float:
    """The neighbour cutoff the frame was built with, recovered from its longest edge and
    rounded up: a tiled copy must carry the same graph as the frame it is compared to."""
    pos = fr.positions.detach().cpu().numpy()
    src, dst = fr.edge_index.detach().cpu().numpy()
    shifts = fr.shifts.detach().cpu().numpy()
    d = np.linalg.norm(pos[dst] - pos[src] + shifts, axis=1)
    return float(np.ceil(d.max() * 100.0) / 100.0)


def tiled_pristine_dict(pristine_frame, model, factors: Sequence[int], device="cpu"
                        ) -> Dict[str, torch.Tensor]:
    """The pristine reference tiled to `factors`, as a one-graph batch dict at S_ref."""
    from mace import data
    from mace.tools import AtomicNumberTable, torch_geometric

    numbers, pos, cell = _frame_geometry(pristine_frame, model)
    nums, pos, cell = tile_frame(numbers, pos, cell, factors)
    config = data.Configuration(
        atomic_numbers=nums, positions=pos, cell=cell, pbc=(True, True, True),
        properties={"carrier_counts": np.zeros(4)}, property_weights={})
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    frame = data.AtomicData.from_config(config, z_table=z_table,
                                        cutoff=_graph_cutoff(pristine_frame))
    loader = torch_geometric.dataloader.DataLoader([frame], batch_size=1)
    return next(iter(loader)).to(device).to_dict()


def _formula_of(model, formula: Optional[Dict[int, float]]) -> Dict[int, float]:
    if formula is not None:
        return {int(z): float(n) for z, n in formula.items()}
    madelung = getattr(model, "madelung", None)
    comp = getattr(madelung, "composition", None) if madelung is not None else None
    if comp is None:
        raise ValueError(
            "the pristine formula is an input: pass `formula` (Z -> count per unit) or give "
            "the model a Madelung composition")
    zs = [int(z) for z in model.atomic_numbers]
    return {z: float(c) for z, c in zip(zs, comp)}


def _single_frame_dict(frame, device) -> Dict[str, torch.Tensor]:
    from mace.tools import torch_geometric

    loader = torch_geometric.dataloader.DataLoader([frame], batch_size=1)
    batch = next(iter(loader)).to(device)
    return batch.to_dict()


def build_class_table(model, frames: Sequence, device="cpu", formula=None,
                      config: Optional[Dict[str, Any]] = None,
                      quantiles: np.ndarray = QUANTILES, log: bool = True) -> Dict[str, Any]:
    """Establish the integers of every composition class present in `frames`.

    `frames` are `AtomicData` (with `frame_key` attached, `defect_cache.attach_frame_keys`).
    The FIRST frame of each composition, in the order given, is its reference geometry. The
    pristine class is the stoichiometric one under `formula`; its first frame's spectrum
    defines the edges every other class is aligned to. A class ambiguous at Tier 1 is handed
    to Tier 2. `config` overrides the model's `class_constructor`. Returns the JSON-able
    table that `model.composition_classes` holds.
    """
    formula = _formula_of(model, formula)
    head = model.spectral
    width = float(head.t_el)
    cfg = constructor_config(getattr(model, "class_constructor", None))
    if config is not None:
        cfg.update(constructor_config(config) if not set(config) <= set(cfg) else config)
    delta = 2.0 * width if cfg["delta"] is None else float(cfg["delta"])
    window = width if cfg["window"] is None else float(cfg["window"])
    firsts: Dict[str, Any] = {}
    for fr in frames:
        key = composition_key(_numbers_of(fr, model))
        firsts.setdefault(key, fr)
    pristine_key = next((k for k, fr in firsts.items()
                         if is_stoichiometric(_numbers_of(fr, model), formula)), None)
    table: Dict[str, Any] = {
        "version": CLASS_TABLE_VERSION, "formula": {str(z): n for z, n in formula.items()},
        "delta": delta, "window": window, "smearing_width": width,
        "r_match": float(cfg["r_match"]), "e_sink": cfg["e_sink"],
        "eta": float(cfg["eta"]), "dlambda": float(cfg["dlambda"]),
        "quantiles": [float(q) for q in quantiles],
        "reference_geometry": "first_frame", "pristine_key": pristine_key, "classes": {}}
    if pristine_key is None:
        for key, fr in firsts.items():
            table["classes"][key] = _uncounted(key, fr, model, delta,
                                               "no stoichiometric frame: no pristine "
                                               "reference to align to").to_dict()
        if log:
            logging.warning("Composition classes: NO stoichiometric frame among %d classes; "
                            "every class is uncounted and the isolated gauge is unavailable",
                            len(firsts))
        return table
    pristine_frame = firsts[pristine_key]
    _, _, pristine_cell = _frame_geometry(pristine_frame, model)
    tiled: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    # The sink is a number in the table whether or not any class needs Tier 2: section 2.7's
    # default is 50 eV above the pristine conduction edge, read off the reference frame.
    tiled[(1, 1, 1)] = head_spectra(
        model, tiled_pristine_dict(pristine_frame, model, (1, 1, 1), device))[0]
    if cfg["e_sink"] is None:
        n_ref = max(reference_fill(tiled[(1, 1, 1)]["n_total"]))
        cfg["e_sink"] = float(tiled[(1, 1, 1)]["spectrum"][n_ref]) + 50.0
    table["e_sink"] = float(cfg["e_sink"])
    # Accepted ranks accumulate as anchors, so Tier 1 can verify a later size by transport.
    # Order matters: the pristine class is exact by electron count and must be ranked first,
    # then defect classes smallest-first so a cheap small cell can anchor a large one. A
    # family whose first size has no anchor routes to Tier 2, which is the intended cost.
    anchors: List[RankAnchor] = []
    calibration: List[Tuple[float, str]] = []
    ordered = sorted(firsts.items(),
                     key=lambda kv: (kv[0] != pristine_key, kv[1].num_nodes, kv[0]))
    for key, fr in ordered:
        numbers, _, cell = _frame_geometry(fr, model)
        mapping = tiling_map(cell, pristine_cell, len(numbers), pristine_frame.num_nodes)
        if mapping is None:
            table["classes"][key] = _uncounted(
                key, fr, model, delta, "the class cell is not an integer tiling of the "
                "pristine reference cell; no aligned edge exists").to_dict()
            continue
        factors, perm = mapping
        # The pristine spectrum the class is aligned to is the reference TILED to the class
        # cell, so both sample the same k-points: a 2x1x1 cell folds more of the zone and
        # its valence top is higher than the primitive reference's by up to ~0.1 eV, which
        # is not a frontier level and must not read as one.
        if factors not in tiled:
            tiled[factors] = head_spectra(
                model, tiled_pristine_dict(pristine_frame, model, factors, device))[0]
        pri = tiled[factors]
        spec = head_spectra(model, _single_frame_dict(fr, device))[0]
        n_sig = reference_fill(spec["n_total"])
        # ONE alignment and ONE count per class: the head is spin-restricted, so both spins
        # share the spectrum and M_VB,sigma is the same integer for both; the occupied
        # manifold of S_ref is the lowest max(N_sigma) levels. Aligning per spin would give
        # two edges differing by the quantile shift of one level, which is noise.
        n_pri = max(reference_fill(pri["n_total"]))
        aligned = align_edges(spec["spectrum"], max(n_sig), pri["spectrum"], n_pri, quantiles)
        # ---------------------------------------------------------------- Tier 1 (v8.1)
        # A rank-CERTIFIED gap verifier. The rank is never inferred from the spectrum: it
        # is exact by electron count on the pristine class, or transported from an accepted
        # anchor of the same homologous family by the pristine-rank increment. The old
        # VBM-proximity test measured the density of the folded valence manifold, so it grew
        # more likely to fail as cells grew with nothing physical changing.
        homology = homology_signature(key, pristine_key, factors)
        nearest = float("nan")
        m_vb_one = 0
        t1 = None
        tier1_reason = ""
        if key == pristine_key:
            predicted = (max(n_sig), max(n_sig))
        else:
            try:
                anchor = select_anchor(anchors, homology, (n_pri, n_pri))
                predicted = predict_rank(anchor, (n_pri, n_pri))
                tier1_reason = f"anchor {anchor.source} at {anchor.n_atoms} atoms"
            except RankRoutingError as exc:
                predicted = ()
                tier1_reason = str(exc)

        if predicted:
            # u_al must come from records ranked INDEPENDENTLY of this bound. The pristine
            # class is exact by electron count, so it seeds the envelope; a target may only
            # contribute after its own independent acceptance, never to certify itself.
            try:
                bound = alignment_bound([r for r, _ in calibration] or [0.0],
                                        [s for _, s in calibration] or ["pristine_exact"])
                u_al = bound.value
                provenance = bound.provenance_hash
            except RankRoutingError as exc:
                u_al, provenance = float("nan"), ""
                tier1_reason = str(exc)
            if np.isfinite(u_al):
                t1 = verify_rank(
                    spec["spectrum"], predicted, vbm_aligned=aligned.vbm_al,
                    gap_host=aligned.gap_pristine, smearing=width, u_al=u_al,
                    delta_search=cfg.get("delta_search"), g_num=cfg.get("g_num", DEFAULT_G_NUM),
                    perturbed_spectra=precision_replicas(spec["H"]))
                nearest = float(t1.gaps[0]) if t1.gaps else float("nan")

        gap_ok = aligned.gap_pristine >= 4.0 * width
        tier: Optional[int] = 1
        reason = ""
        extra: Dict[str, Any] = {}
        if t1 is not None and t1.accepted:
            m_vb_one = int(t1.m_vb[0])
        if not gap_ok:
            tier, reason = None, (f"pristine gap {aligned.gap_pristine:.3f} eV < 4 x smearing "
                                  f"{4 * width:.3f} eV")
        elif t1 is None or not t1.accepted:
            tier1_reason = (t1.reason if t1 is not None else tier1_reason) or "no anchor"
            # Tier 2: the valence-subspace continuation from the tiled pristine.
            t2 = tier2(model, fr, pristine_frame, factors, perm, n_pri, pri["H"], spec["H"],
                       cfg, device=device, conduction_cut=aligned.cbm_al - delta)
            extra = dict(path_agreement=t2["path_agreement"],
                         schedule_agreement=t2["schedule_agreement"],
                         gamma=tuple(t2["gamma"]), correspondence=t2["correspondence"],
                         tier1_reason=tier1_reason)
            if t2["accepted"]:
                tier, m_vb_one = 2, int(t2["m_vb"])
            else:
                tier, reason = None, "Tier 1 declined; Tier 2: " + t2["reason"]
        if tier == 1:
            # Record the Tier-1 provenance on ACCEPTANCE too, not only when it declines:
            # the anchor it transported from and the gates it passed are what make the
            # rank auditable later.
            extra.setdefault("tier1_reason", tier1_reason)
        if t1 is not None:
            extra["tier1_record"] = _serialisable(
                dict(t1.to_dict(), u_al_provenance=provenance, homology=homology,
                     # A list, not a tuple: the table round-trips through JSON and pickle,
                     # and a tuple would come back as a list and break table equality.
                     perturbations=["precision_float32"]))
        if m_vb_one:
            # The spectral diagnostic at the accepted counting boundary, for either tier.
            # Kept finite: see _serialisable on why a NaN here breaks table equality.
            ordered_spec = np.sort(np.asarray(spec["spectrum"], dtype=np.float64))
            if 0 < m_vb_one < ordered_spec.size:
                nearest = float(ordered_spec[m_vb_one] - ordered_spec[m_vb_one - 1])
        m_vb = (m_vb_one, m_vb_one)
        accepted = tier is not None
        n_e, n_h, q_core = class_integers(m_vb, n_sig) if accepted else ((0, 0), (0, 0), 0)
        placement = pristine_placement(model, fr, pristine_frame, factors, perm)
        record = ClassRecord(
            key=key, n_atoms=spec["n_atoms"], reference_frame_key=_frame_key_of(fr),
            n_total=spec["n_total"], n_sigma=n_sig, tier=tier, ambiguous=not accepted,
            reason=reason, m_vb=m_vb, n_e=n_e, n_h=n_h, q_core=q_core, vbm_al=aligned.vbm_al,
            cbm_al=aligned.cbm_al, shift=aligned.shift, spread=aligned.spread,
            gap_pristine=aligned.gap_pristine, delta=delta, nearest=nearest, tiling=factors,
            perm=perm, placement=placement, **extra)
        table["classes"][key] = record.to_dict()

        if accepted:
            # An accepted rank becomes an anchor for later sizes of the same family, and
            # its top occupied level feeds the alignment envelope. Only ranks established
            # independently of u_al may do so: exact-by-electron-count (pristine) or
            # Tier-2-accepted. A Tier-1 acceptance was itself certified BY u_al, so it is
            # deliberately not added to the calibration -- that would let the envelope
            # grow to justify the acceptances it produced.
            anchors.append(RankAnchor(
                homology=homology, n_atoms=int(spec["n_atoms"]), m_vb=m_vb,
                pristine_rank=(n_pri, n_pri),
                source="pristine" if key == pristine_key else (
                    "tier2" if tier == 2 else "certified"),
                source_hash=f"{key}@{_frame_key_of(fr)}"))
            if tier == 2 or key == pristine_key:
                calibration.append((
                    float(np.sort(np.asarray(spec["spectrum"]))[m_vb_one - 1] - aligned.vbm_al),
                    f"{'pristine' if key == pristine_key else 'tier2'}_{spec['n_atoms']}"))
        if log:
            logging.info("Composition class %s (%d atoms, ref frame %d): %s", key,
                         record.n_atoms, record.reference_frame_key, describe(record))
    return table


def ensure_class_table(model, frames: Sequence, device=None, formula=None,
                       config: Optional[Dict[str, Any]] = None, log: bool = True
                       ) -> Dict[str, Any]:
    """The model's class table, built over `frames` if it carries none (Stage 1.2).

    ONE helper for every driver -- the trainer, the FD driver, the golden capture, the
    scorers -- so they all choose the reference geometry the same way (the first frame of
    each composition IN THE ORDER GIVEN, principle 9) and log which frame it was. Two drivers
    that build tables from different pristine snapshots get different `vbm_al` and
    `placement`, and their frontier energies would not agree. A table already on the model
    is returned untouched.
    """
    table = getattr(model, "composition_classes", None)
    if table and table.get("classes"):
        return table
    if device is None:
        # The model's own device: a driver that moved the model to a GPU and builds the
        # table afterwards (the b3 smoke did) must not hand it CPU batches.
        device = next(model.parameters()).device
    table = build_class_table(model, frames, device=device, formula=formula, config=config,
                              log=log)
    model.composition_classes = table
    # Every non-parameter float is a number in the saved config: the resolved sink too.
    model.class_constructor["e_sink"] = table["e_sink"]
    if log:
        refs = {k: r["reference_frame_key"] for k, r in table["classes"].items()}
        logging.info("Composition classes built over %d frames; reference frame keys %s",
                     len(frames), refs)
    return table


# The record fields that are the head's ALIGNMENT and PLACEMENT (refreshed as the head
# trains) as against the INTEGERS (established once, decision 21).
ALIGNMENT_FIELDS = ("vbm_al", "cbm_al", "shift", "spread", "gap_pristine", "nearest",
                    "placement", "reference_frame_key")
INTEGER_FIELDS = ("n_total", "n_sigma", "tier", "ambiguous", "reason", "m_vb", "n_e", "n_h",
                  "q_core", "path_agreement", "schedule_agreement", "gamma", "correspondence",
                  "tier1_reason")


def refresh_class_table(model, frames: Sequence, device=None, log: bool = True
                        ) -> Dict[str, Any]:
    """Re-align the model's class table to the head AS IT IS NOW (decision 21).

    The integers of a class are established once -- the first time the constructor counts
    it -- and kept; what the frontier term reads every step, the aligned edges `VBM_al`,
    `CBM_al` (the projector windows) and the density placement, is a property of the head
    and moves as the head trains (the Stage B recipe opens the pristine gap from 0.2 eV at
    the Harrison initialisation to 2.4 eV). So the trainer refreshes the alignment every
    epoch from a fresh construction over the same frames: alignment fields are taken from
    the fresh table for every class; a class the old table left uncounted adopts the fresh
    count when there is one; a class already counted keeps its integers, and a fresh count
    that disagrees is logged as a failed invariance (never adopted). Returns a summary.
    """
    old = getattr(model, "composition_classes", None)
    if not old or not old.get("classes"):
        return {"built": True, "table": ensure_class_table(model, frames, device=device, log=log)}
    if device is None:
        device = next(model.parameters()).device
    fresh = build_class_table(model, frames, device=device, log=False)
    summary: Dict[str, Any] = {"adopted": [], "disagree": [], "still_uncounted": [],
                               "refreshed": []}
    for key, new in fresh["classes"].items():
        rec = old["classes"].get(key)
        if rec is None:
            old["classes"][key] = new
            summary["adopted"].append(key)
            continue
        fresh_aligned = new.get("vbm_al") is not None and np.isfinite(float(new["vbm_al"]))
        if fresh_aligned:
            for f in ALIGNMENT_FIELDS:
                rec[f] = new.get(f)
            rec["tiling"], rec["perm"], rec["delta"] = (new.get("tiling"), new.get("perm"),
                                                        new.get("delta"))
        else:
            # the fresh construction could not align this class (no tiling, no pristine
            # gap): its NaN edges must not replace a counted class's
            summary.setdefault("not_realigned", []).append(key)
        old_counted = rec.get("tier") is not None and not rec.get("ambiguous", True)
        new_counted = new.get("tier") is not None and not new.get("ambiguous", True)
        if not old_counted and new_counted:
            for f in INTEGER_FIELDS:
                rec[f] = new.get(f)
            summary["adopted"].append(key)
        elif not old_counted:
            rec["reason"] = new.get("reason", rec.get("reason"))
            summary["still_uncounted"].append(key)
        elif new_counted and (tuple(new["n_e"]), tuple(new["n_h"]), int(new["q_core"])) != (
                tuple(rec["n_e"]), tuple(rec["n_h"]), int(rec["q_core"])):
            summary["disagree"].append((key, (rec["n_e"], rec["n_h"], rec["q_core"]),
                                        (new["n_e"], new["n_h"], new["q_core"])))
        else:
            summary["refreshed"].append(key)
    for k in ("delta", "window", "smearing_width", "pristine_key"):
        old[k] = fresh[k]
    if log:
        edges = {k: (round(r["vbm_al"], 3), round(r["cbm_al"], 3))
                 for k, r in old["classes"].items()}
        logging.info("Composition classes refreshed: edges %s; adopted %s; still uncounted %s",
                     edges, summary["adopted"], summary["still_uncounted"])
        for key, was, now in summary["disagree"]:
            logging.warning("Composition class %s: the refreshed head counts %s but the "
                            "cached integers are %s -- a failed invariance; the cached "
                            "integers are kept", key, now, was)
    summary["table"] = old
    return summary


def species_charges(model) -> Optional[torch.Tensor]:
    """`Z0` per species in the model's own order (the Madelung baseline), or None when the
    model carries no static charges -- then no static density exists."""
    madelung = getattr(model, "madelung", None)
    if madelung is None:
        return None
    # On the CPU whatever device the model trains on: the constructor's densities and its
    # Newton refinement of the placement are CPU objects, and the trainer calls this hook
    # after the model has moved to the GPU.
    return madelung.z.detach().cpu()


def tiled_pristine_scaled(model, pristine_frame, factors, perm
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The tiled pristine reference as `(species index [n0], scaled positions [n0, 3])`, the
    scaled coordinates already in the FRAME's axis order (`perm`)."""
    numbers, pos, cell = _frame_geometry(pristine_frame, model)
    numbers, pos, cell = tile_frame(numbers, pos, cell, factors)
    scaled = pos @ np.linalg.inv(cell)
    scaled = scaled[:, list(perm)]
    zs = [int(z) for z in model.atomic_numbers]
    species = torch.tensor([zs.index(int(z)) for z in numbers], dtype=torch.long)
    return species, torch.tensor(scaled, dtype=torch.get_default_dtype())


def pristine_placement(model, class_frame, pristine_frame, factors, perm,
                       r_res: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The fractional shift placing the tiled pristine density in the class reference frame,
    by minimising `||rho_static^raw||` (`defect_density.align_pristine`). None when the model
    carries no static charges."""
    from mace.modules import defect_density as dd

    z0 = species_charges(model)
    if z0 is None:
        return None
    r_res = float(_functional(model)["r_res"]) if r_res is None else float(r_res)
    numbers, pos, cell = _frame_geometry(class_frame, model)
    zs = [int(z) for z in model.atomic_numbers]
    species = torch.tensor([zs.index(int(z)) for z in numbers], dtype=torch.long)
    dtype = torch.get_default_dtype()
    present = dd.static_present(z0.to(dtype)[species], torch.tensor(pos, dtype=dtype),
                                torch.tensor(cell, dtype=dtype), r_res)
    pri_species, scaled = tiled_pristine_scaled(model, pristine_frame, factors, perm)
    shared = sorted(set(species.tolist()) & set(pri_species.tolist()))
    anchor_species = max(shared, key=lambda i: zs[i])
    anchor = int(torch.nonzero(species == anchor_species).reshape(-1)[0])
    shift, residual = dd.align_pristine(present, z0.to(dtype)[pri_species], scaled,
                                        pri_species == anchor_species, anchor)
    return {"shift": [float(x) for x in shift], "residual_norm": residual, "r_res": r_res}


def _functional(model) -> Dict[str, Any]:
    from mace.modules.defect_density import functional_config

    return functional_config(getattr(model, "functional", None))


def frame_static_densities(model, record: ClassRecord, pristine_frame, charges: torch.Tensor,
                           positions: torch.Tensor, cell: torch.Tensor,
                           r_res: Optional[float] = None) -> Dict[str, Any]:
    """`rho_Z^present`, `rho_Z^pristine`, `rho_static^raw`, `g_res` and `rho_static^def` for
    one frame of `record`'s class, reusing the class placement. `charges` are the frame's
    static charges `Z_i` (with any per-site deviation), `positions`/`cell` the frame's.
    Reports `||rho_static^raw||` so a frame from a different origin is visible."""
    from mace.modules import defect_density as dd

    if record.placement is None:
        raise ValueError(f"class {record.key} has no pristine placement (no static charges)")
    r_res = float(record.placement["r_res"]) if r_res is None else float(r_res)
    z0 = species_charges(model).to(dtype=positions.dtype, device=positions.device)
    present = dd.static_present(charges, positions, cell, r_res)
    pri_species, scaled = tiled_pristine_scaled(model, pristine_frame, record.tiling, record.perm)
    shift = torch.tensor(record.placement["shift"], dtype=positions.dtype, device=positions.device)
    pristine = dd.pristine_placed(z0[pri_species.to(z0.device)],
                                  scaled.to(dtype=positions.dtype, device=positions.device),
                                  cell, shift, r_res)
    raw = dd.static_raw(present, pristine)
    g_res = dd.residual_shape(raw, positions, pristine.centres)
    if not record.counted:
        raise ValueError(f"class {record.key} carries no Q_core; rho_static^def is undefined")
    static = dd.static_def(raw, g_res, record.q_core)
    return {"present": present, "pristine": pristine, "raw": raw, "g_res": g_res,
            "static": static, "q_raw": raw.integral(), "raw_norm": raw.norm()}


def frame_counts_batch(table: Dict[str, Any], atomic_numbers: Sequence[Sequence[int]],
                       state: StateBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`frame_counts` over a batch: `(n_e [B, 2], n_h [B, 2], q_F [B])` as long tensors, from
    each graph's composition and the exact state. Refuses a composition outside the table."""
    n_e, n_h, q_f = [], [], []
    for g, numbers in enumerate(atomic_numbers):
        record = lookup_class(table, numbers)
        e, h, q = frame_counts(record, state, g)
        n_e.append(e)
        n_h.append(h)
        q_f.append(q)
    return (torch.tensor(n_e, dtype=torch.long), torch.tensor(n_h, dtype=torch.long),
            torch.tensor(q_f, dtype=torch.long))


def _uncounted(key, fr, model, delta, reason) -> ClassRecord:
    numbers = _numbers_of(fr, model)
    return ClassRecord(key=key, n_atoms=len(numbers), reference_frame_key=_frame_key_of(fr),
                       n_total=0, n_sigma=(0, 0), tier=None, ambiguous=True, reason=reason,
                       m_vb=(0, 0), n_e=(0, 0), n_h=(0, 0), q_core=0, vbm_al=float("nan"),
                       cbm_al=float("nan"), shift=float("nan"), spread=float("nan"),
                       gap_pristine=float("nan"), delta=delta, nearest=float("nan"))


def _numbers_of(fr, model) -> List[int]:
    zs = [int(z) for z in model.atomic_numbers]
    return [zs[int(i)] for i in fr.node_attrs.argmax(dim=-1).tolist()]


def _frame_key_of(fr) -> int:
    key = getattr(fr, "frame_key", None)
    return -1 if key is None else int(key.reshape(-1)[0])


def describe(record: ClassRecord) -> str:
    if not record.counted:
        return f"UNCOUNTED ({record.reason})"
    tier2_note = ""
    if record.tier == 2:
        c = record.correspondence or {}
        tier2_note = (f" [Tier 2: {c.get('n_ghost', 0)} ghost, {c.get('n_added', 0)} added, "
                      f"{c.get('n_substituted', 0)} substituted; gamma physical "
                      f"{sum(1 for g in record.gamma if g < 0.5)}, ghost "
                      f"{sum(1 for g in record.gamma if g >= 0.5)}; {record.tier1_reason}]")
    return (f"tier {record.tier}, M_VB={list(record.m_vb)}, N={list(record.n_sigma)}, "
            f"n_e={list(record.n_e)}, n_h={list(record.n_h)}, Q_core={record.q_core:+d}, "
            f"VBM_al={record.vbm_al:.3f} eV (shift {record.shift:+.3f}, spread "
            f"{record.spread:.3f}), nearest level to the cut {record.nearest:+.3f} eV, "
            f"pristine tiled {'x'.join(str(t) for t in record.tiling)}" + tier2_note)


def lookup_class(table: Dict[str, Any], atomic_numbers: Sequence[int]) -> ClassRecord:
    """The record for a composition; a composition absent from the table is refused."""
    key = composition_key(atomic_numbers)
    classes = (table or {}).get("classes", {})
    if key not in classes:
        raise KeyError(
            f"composition {key} has no class record; the class table was built over "
            f"{sorted(classes)} and a frame outside it cannot be defaulted")
    return ClassRecord.from_dict(classes[key])


def verify_class_table(model, frames: Sequence, device="cpu", table=None) -> List[str]:
    """Recompute every class from the SAME reference frames and list the integers that moved.

    Stage 4's class-count invariance test calls this after training: the integers must be
    unchanged by parameter motion, R, Q and cell tiling. Returns the differences, empty when
    the table is reproduced.
    """
    table = model.composition_classes if table is None else table
    fresh = build_class_table(model, frames, device=device,
                              formula={int(z): n for z, n in table["formula"].items()},
                              config={k: table[k] for k in DEFAULT_CONSTRUCTOR},
                              quantiles=np.asarray(table["quantiles"]), log=False)
    diffs = []
    for key, old in table["classes"].items():
        new = fresh["classes"].get(key)
        if new is None:
            diffs.append(f"{key}: absent from the frames")
            continue
        for f in ("tier", "m_vb", "n_e", "n_h", "q_core"):
            if old[f] != new[f]:
                diffs.append(f"{key}: {f} {old[f]} -> {new[f]}")
    return diffs


# ======================================================================= Tier 2 (fenced)
#
# The valence-subspace continuation (plan section 2.1, Tier 2). Everything below this line
# is the constructor's own machinery: the site correspondence, the union basis, the two
# interpolation paths, the transported projector and the endpoint classification. None of
# it is reachable from a production forward, and the AST test in
# tests/extensions/defect/test_composition_classes.py holds that line (`TIER2_NAMES`).
#
# THE IDEA. A class that Tier 1 cannot count (a level near the counting cut) is counted by
# CONTINUITY instead: the pristine occupied valence projector P_V^(0), of rank M_V^(0), is
# carried from the tiled pristine Hamiltonian to the class Hamiltonian along a path of
# Hamiltonians in a common ("union") orbital basis, tracking the SUBSPACE by maximum overlap
# with the eigenvectors at each step -- never individual eigenstates, so crossings inside the
# manifold cost nothing. At the endpoint the transported subspace is split, basis-invariantly,
# into its physical and ghost parts by the eigenvalues gamma of P_V P_ghost P_V; the physical
# rank is M_VB^class. Two geometrically distinct paths and two step schedules must agree, no
# gamma may sit in the closure band, and the physical part must coincide (overlap > 1 - eta)
# with a spectrally contiguous set of eigenvectors of the class Hamiltonian. Otherwise the
# class is genuinely ambiguous: it carries no core/frontier decomposition, and no override
# input exists.

ORB = 4   # orbitals per site (s, px, py, pz), defect_counting.ORBITALS_PER_ATOM


def _minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def site_correspondence(class_numbers, class_positions, class_cell, pristine_numbers,
                        pristine_positions, pristine_cell, perm=(0, 1, 2),
                        r_match: float = 2.0, n_candidates: int = 6,
                        refinements: int = 2) -> Dict[str, Any]:
    """Minimum-cost assignment on POSITIONS between the class reference geometry and the
    tiled pristine cell, placed into the class cell through scaled coordinates (with the
    lattice vectors relabelled by `perm`) and the best rigid translation.

    Unmatched pristine sites are ghosts (vacancies), unmatched class atoms are additions
    (interstitials), matched pairs of different species are substitutions. A pair further
    apart than `r_match` is never matched: the assignment is augmented with "unmatched" at
    a cost of r_match^2 per unmatched pair, so a displaced atom is an addition next to a
    ghost rather than a match beyond r_match. The correspondence is used only here.

    The translation is searched over the placements of the class's first atom OF THE
    HEAVIEST SPECIES (the one that moves least thermally -- Pb in the perovskite, whose Cl
    swing by up to ~1.4 A between snapshots) onto every pristine site of that species, the
    best few candidates going through the full assignment; the winner is then refined by
    re-centring on the mean matched displacement and re-assigning.
    """
    from scipy.optimize import linear_sum_assignment

    class_numbers = np.asarray(class_numbers, dtype=np.int64)
    pristine_numbers = np.asarray(pristine_numbers, dtype=np.int64)
    class_positions = np.asarray(class_positions, dtype=np.float64)
    class_cell = np.asarray(class_cell, dtype=np.float64).reshape(3, 3)
    pristine_cell = np.asarray(pristine_cell, dtype=np.float64).reshape(3, 3)
    n1, n0 = len(class_numbers), len(pristine_numbers)
    scaled = np.asarray(pristine_positions, dtype=np.float64) @ np.linalg.inv(pristine_cell)
    mapped = scaled[:, list(perm)] @ class_cell
    unmatched_cost = 0.5 * r_match ** 2
    big = 1e6

    def distances(t):
        d = _minimum_image(class_positions[:, None, :] - (mapped + t)[None, :, :], class_cell)
        return np.linalg.norm(d, axis=-1), d

    def assign(dist):
        cost = np.full((n1 + n0, n0 + n1), big)
        cost[:n1, :n0] = dist ** 2
        cost[:n1, n0:] = np.where(np.eye(n1, dtype=bool), unmatched_cost, big)
        cost[n1:, :n0] = np.where(np.eye(n0, dtype=bool), unmatched_cost, big)
        cost[n1:, n0:] = 0.0
        rows, cols = linear_sum_assignment(cost)
        pairs = {int(r): int(c) for r, c in zip(rows, cols) if r < n1 and c < n0}
        return float(cost[rows, cols].sum()), pairs

    shared = sorted(set(class_numbers.tolist()) & set(pristine_numbers.tolist()))
    if not shared:
        raise ValueError("the class and its pristine reference share no species")
    anchor_species = max(shared)
    anchor = int(np.nonzero(class_numbers == anchor_species)[0][0])
    candidates = []
    for j in np.nonzero(pristine_numbers == anchor_species)[0]:
        t = class_positions[anchor] - mapped[j]
        dist, _ = distances(t)
        candidates.append((-int((dist.min(axis=1) < r_match).sum()), int(j), t))
    candidates.sort(key=lambda c: c[0])
    best = None
    for _, _, t in candidates[:n_candidates]:
        dist, vec = distances(t)
        total, pairs = assign(dist)
        for _ in range(refinements):
            if not pairs:
                break
            # Re-centre: the mean displacement of the matched pairs is the residual
            # translation; a better centring can only lower the cost, so it is kept if it does.
            shift = np.mean([vec[i, j] for i, j in pairs.items()], axis=0)
            dist2, vec2 = distances(t + shift)
            total2, pairs2 = assign(dist2)
            if total2 < total - 1e-12:
                t, dist, vec, total, pairs = t + shift, dist2, vec2, total2, pairs2
            else:
                break
        if best is None or total < best[0]:
            best = (total, t, dist, pairs)
    total, t, dist, match = best
    ghosts = sorted(set(range(n0)) - set(match.values()))
    added = sorted(set(range(n1)) - set(match))
    substituted = sorted((i, j) for i, j in match.items()
                         if class_numbers[i] != pristine_numbers[j])
    displacements = np.array([dist[i, j] for i, j in match.items()]) if match else np.zeros(0)
    return dict(match=match, ghosts=ghosts, added=added, substituted=substituted,
                translation=t, max_displacement=float(displacements.max()) if match else 0.0,
                mean_displacement=float(displacements.mean()) if match else 0.0,
                n_ghost=len(ghosts), n_added=len(added), n_substituted=len(substituted),
                n_matched=len(match), total_cost=total)


def union_basis(H0: np.ndarray, H1: np.ndarray, correspondence: Dict[str, Any]
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Embed the pristine `H0` (n0 sites) and the class `H1` (n1 sites) in the union basis:
    the pristine sites in their order, then the added atoms. Returns `(H0_u, H1_u, groups)`
    with `groups` the orbital index arrays `ghost`, `added`, `substituted`."""
    n0 = H0.shape[0] // ORB
    n1 = H1.shape[0] // ORB
    match, ghosts, added = correspondence["match"], correspondence["ghosts"], correspondence["added"]
    n_union = n0 + len(added)
    site_of_class = np.zeros(n1, dtype=np.int64)
    for i in range(n1):
        site_of_class[i] = match[i] if i in match else n0 + added.index(i)
    dim = ORB * n_union
    H0_u = np.zeros((dim, dim))
    H0_u[:ORB * n0, :ORB * n0] = H0
    idx1 = (ORB * site_of_class[:, None] + np.arange(ORB)[None, :]).reshape(-1)
    H1_u = np.zeros((dim, dim))
    H1_u[np.ix_(idx1, idx1)] = H1
    orbitals = lambda sites: np.array([ORB * s + o for s in sites for o in range(ORB)],
                                      dtype=np.int64)
    groups = {"ghost": orbitals(ghosts), "added": orbitals(n0 + k for k in range(len(added))),
              "substituted": orbitals(j for _, j in correspondence["substituted"])}
    return H0_u, H1_u, groups


def interpolated_hamiltonian(H0_u: np.ndarray, H1_u: np.ndarray, lam: float, path: str,
                             groups: Dict[str, np.ndarray], e_sink: float) -> np.ndarray:
    """H(lambda) on Path A or Path B between the two union Hamiltonians.

    Path A: every matrix element interpolates linearly at once -- hoppings to a ghost fade as
    (1 - lambda) because H1_u is zero there, hoppings to an addition rise as lambda, a
    substitution switches alchemically while coupled.
    Path B, three phases: (1) the hoppings of ghost and substituted sites fade to zero while
    the unaffected block interpolates and additions stay decoupled; (2) with the affected
    sites decoupled, substituted on-site levels switch species; (3) the hoppings of additions
    and substitutions rise to their final values.

    THE SINK IS NEVER CROSSED WHILE COUPLED, on either path. A ghost's on-site level stays
    where the pristine put it until its hoppings are exactly zero (lambda = 1) and only then
    is parked at +E_sink; an addition's level leaves the sink at lambda = 0+, when it is still
    exactly decoupled. Moving a decoupled block is a relabelling of exact eigenstates, so the
    step is exact and E_sink cannot enter the transport at all. A level that instead rode to
    the sink while coupled would sweep through every state above it with a residual hopping;
    the fine-step limit of that sweep is ADIABATIC following, which swaps the ghost for a
    physical state at each crossing and converges to the wrong answer (the lowest-M subspace
    of H^(1), which counts the vacancy level as valence). The ghost levels sit inside the
    occupied manifold throughout, so the crossings they do have are within the transported
    subspace and cost nothing.
    """
    dim = H0_u.shape[0]
    ghost = np.zeros(dim, dtype=bool)
    ghost[groups["ghost"]] = True
    added = np.zeros(dim, dtype=bool)
    added[groups["added"]] = True
    subst = np.zeros(dim, dtype=bool)
    subst[groups["substituted"]] = True
    affected = ghost | added | subst
    site = np.arange(dim) // ORB
    same = site[:, None] == site[None, :]
    eye = np.eye(dim)
    sink_ghost = eye * ghost[:, None] * e_sink
    sink_added = eye * added[:, None] * e_sink
    on_ghost = same & ghost[:, None]
    on_added = same & added[:, None]
    on_subst = same & subst[:, None]
    ghost_block = sink_ghost if lam >= 1.0 else H0_u
    added_block = sink_added if lam <= 0.0 else H1_u
    if path == "A":
        H = (1.0 - lam) * H0_u + lam * H1_u
        H[on_ghost] = ghost_block[on_ghost]
        H[on_added] = added_block[on_added]
        return H
    if path != "B":
        raise ValueError(f"unknown path {path!r}; expected 'A' or 'B'")
    un = ~affected
    UU = un[:, None] & un[None, :]
    G = ghost[:, None] | ghost[None, :]
    A = added[:, None] | added[None, :]
    S = subst[:, None] | subst[None, :]
    off = ~same
    tau = 3.0 * lam
    H = np.zeros((dim, dim))
    if tau <= 1.0:
        u = tau
        H[UU] = ((1.0 - u) * H0_u + u * H1_u)[UU]
        fade = off & (G | S) & ~A
        H[fade] = ((1.0 - u) * H0_u)[fade]
        H[on_subst] = H0_u[on_subst]
    elif tau <= 2.0:
        v = tau - 1.0
        H[UU] = H1_u[UU]
        H[on_subst] = ((1.0 - v) * H0_u + v * H1_u)[on_subst]
    else:
        w = tau - 2.0
        H[UU] = H1_u[UU]
        rise = off & (A | S) & ~G
        H[rise] = (w * H1_u)[rise]
        H[on_subst] = H1_u[on_subst]
    H[on_ghost] = ghost_block[on_ghost]
    H[on_added] = added_block[on_added]
    return H


def transport_valence_subspace(H0_u: np.ndarray, H1_u: np.ndarray, groups: Dict[str, np.ndarray],
                               rank: int, path: str, dlambda: float, e_sink: float
                               ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Carry the lowest-`rank` subspace of H(0) to lambda = 1 by maximum subspace overlap.

    At each step the eigenvectors of H(lambda) are ranked by their weight in the current
    subspace, `w_k = sum_a |<u_k|psi_a>|^2`, and the `rank` heaviest are kept: the subspace,
    never an eigenstate. The smallest gap between the kept and dropped weights along the
    path is returned as `min_separation` (1 for a perfectly adiabatic step, ~0 at an
    unresolved crossing between the manifold and its complement).
    """
    n_steps = max(int(round(1.0 / dlambda)), 1)
    _, U = np.linalg.eigh(interpolated_hamiltonian(H0_u, H1_u, 0.0, path, groups, e_sink))
    psi = U[:, :rank]
    min_sep, min_weight = 1.0, 1.0
    for k in range(1, n_steps + 1):
        lam = k / n_steps
        _, U = np.linalg.eigh(interpolated_hamiltonian(H0_u, H1_u, lam, path, groups, e_sink))
        weight = ((U.T @ psi) ** 2).sum(axis=1)
        order = np.argsort(-weight, kind="stable")
        kept = np.sort(order[:rank])
        sep = float(weight[order[rank - 1]] - weight[order[rank]]) if rank < len(weight) else 1.0
        min_sep = min(min_sep, sep)
        min_weight = min(min_weight, float(weight[order[rank - 1]]))
        psi = U[:, kept]
    return psi, {"min_separation": min_sep, "min_weight": min_weight, "steps": n_steps}


def endpoint_classification(psi: np.ndarray, ghost_orbitals: np.ndarray, eta: float
                            ) -> Dict[str, Any]:
    """`gamma = eig(P_V P_ghost P_V)` on ran P_V, basis-invariantly: the eigenvalues of the
    rank x rank matrix `psi^T P_ghost psi`. `gamma < eta` physical, `> 1 - eta` ghost,
    between: closure. Returns the spectrum, the counts and the physical part of `psi`."""
    g = psi[np.asarray(ghost_orbitals, dtype=np.int64), :] if len(ghost_orbitals) else \
        np.zeros((0, psi.shape[1]))
    gamma, V = np.linalg.eigh(g.T @ g)
    gamma = np.clip(gamma, 0.0, 1.0)
    physical = gamma < eta
    ghost = gamma > 1.0 - eta
    closure = ~physical & ~ghost
    return dict(gamma=gamma, n_physical=int(physical.sum()), n_ghost=int(ghost.sum()),
                n_closure=int(closure.sum()), psi_physical=psi @ V[:, physical])


def _contiguity(psi_physical: np.ndarray, H1_final: np.ndarray, ghost_orbitals, eta: float
                ) -> Tuple[bool, float, Tuple[int, int]]:
    """Does the physical part coincide with a spectrally contiguous set of eigenvectors of
    the class Hamiltonian? Returns `(ok, overlap, (first, last))` over the physical
    eigenvectors of H^(1)_union (those not parked at the sink)."""
    m = psi_physical.shape[1]
    if m == 0:
        return True, 1.0, (0, -1), -np.inf
    lam, U = np.linalg.eigh(H1_final)
    ghost_weight = (U[np.asarray(ghost_orbitals, dtype=np.int64), :] ** 2).sum(axis=0) \
        if len(ghost_orbitals) else np.zeros(U.shape[1])
    physical_cols = np.nonzero(ghost_weight < 0.5)[0]
    weight = ((U[:, physical_cols].T @ psi_physical) ** 2).sum(axis=1)
    order = np.argsort(-weight, kind="stable")[:m]
    chosen = np.sort(physical_cols[order])
    overlap = float(weight[order].sum() / m)
    contiguous = bool(chosen[-1] - chosen[0] + 1 == m)
    top = float(lam[chosen].max())
    return contiguous, overlap, (int(chosen[0]), int(chosen[-1])), top


def tier2_continuation(H0_u: np.ndarray, H1_u: np.ndarray, groups: Dict[str, np.ndarray],
                       rank: int, e_sink: float = 100.0, eta: float = 1.0e-3,
                       dlambda: float = 0.02, conduction_cut: Optional[float] = None
                       ) -> Dict[str, Any]:
    """Both paths, both schedules, the endpoint classification and the acceptance rule, on
    Hamiltonians already in the union basis. `rank` is M_V^(0). Returns the decision with
    every quantity the cache records.

    The acceptance rule is the plan's: path and schedule agreement, no closure eigenvalue,
    and the physical part coinciding with a SPECTRALLY CONTIGUOUS set of class eigenvectors
    (overlap > 1 - eta). `identified` (the overlap alone), `top_identified_level` and
    `conduction_cut` (`CBM_al - delta`, when given) are recorded as diagnostics: a relaxation
    accepting an identified but non-contiguous manifold below the conduction cut was
    considered for the fresh head's 159-atom V_Cl class and NOT adopted (decision 19) -- on
    that head the gap was 0.2 eV, the cut coincided with the counting cut, and the reference
    fill had a hole in a valence-derived level, which is the pattern the clause flags.
    """
    H1_final = interpolated_hamiltonian(H0_u, H1_u, 1.0, "A", groups, e_sink)
    runs: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in ("A", "B"):
        for schedule, dl in ((1, dlambda), (2, dlambda / 2.0)):
            psi, stats = transport_valence_subspace(H0_u, H1_u, groups, rank, path, dl, e_sink)
            cls = endpoint_classification(psi, groups["ghost"], eta)
            ok, overlap, span, top = _contiguity(cls["psi_physical"], H1_final,
                                                 groups["ghost"], eta)
            runs[(path, schedule)] = dict(m_vb=cls["n_physical"], n_ghost=cls["n_ghost"],
                                          n_closure=cls["n_closure"], gamma=cls["gamma"],
                                          contiguous=ok, overlap=overlap, span=span,
                                          top_identified_level=top, **stats)
    m_values = {r["m_vb"] for r in runs.values()}
    path_agreement = runs[("A", 1)]["m_vb"] == runs[("B", 1)]["m_vb"]
    schedule_agreement = all(
        runs[(p, 1)]["m_vb"] == runs[(p, 2)]["m_vb"]
        and float(np.abs(runs[(p, 1)]["gamma"] - runs[(p, 2)]["gamma"]).max()) < eta
        for p in ("A", "B"))
    closure = any(r["n_closure"] > 0 for r in runs.values())
    contiguous = all(r["contiguous"] for r in runs.values())
    # Decision 19 (considered, not adopted): the contiguity clause is the plan's rule and
    # stays; `identified` and the conduction-cut comparison are diagnostics.
    identified = all(r["overlap"] > 1.0 - eta for r in runs.values())
    top = max(r["top_identified_level"] for r in runs.values())
    below_cut = conduction_cut is not None and top <= float(conduction_cut)
    reasons = []
    if not path_agreement:
        reasons.append(f"paths disagree (A: {runs[('A', 1)]['m_vb']}, B: {runs[('B', 1)]['m_vb']})")
    if not schedule_agreement:
        reasons.append("the two step schedules disagree")
    if closure:
        reasons.append("a closure eigenvalue eta <= gamma <= 1 - eta at the endpoint")
    if not identified:
        worst = min(runs.values(), key=lambda r: r["overlap"])
        reasons.append(f"the physical part is not identified with class eigenvectors "
                       f"(overlap {worst['overlap']:.3f}, span {worst['span']})")
    note = ""
    if not contiguous:
        worst = min(runs.values(), key=lambda r: r["overlap"])
        reasons.append(f"the physical part is not a contiguous set of class eigenvectors "
                       f"(overlap {worst['overlap']:.3f}, span {worst['span']}; top identified "
                       f"level {top:+.3f} eV"
                       + (f", conduction cut {conduction_cut:+.3f} eV" if conduction_cut
                          is not None else "") + ")")
    accepted = path_agreement and schedule_agreement and not closure and contiguous \
        and len(m_values) == 1
    return dict(accepted=accepted, m_vb=int(runs[("A", 1)]["m_vb"]) if accepted else None,
                gamma=[float(g) for g in runs[("A", 1)]["gamma"]],
                path_agreement=bool(path_agreement), schedule_agreement=bool(schedule_agreement),
                closure=bool(closure), contiguous=bool(contiguous), identified=bool(identified),
                below_conduction_cut=bool(below_cut), top_identified_level=float(top),
                reason="; ".join(reasons), note=note,
                runs={f"{p}{s}": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in r.items()} for (p, s), r in runs.items()})


def tier2(model, class_frame, pristine_frame, factors, perm, rank: int, H0, H1,
          cfg: Dict[str, Any], device="cpu", conduction_cut: Optional[float] = None
          ) -> Dict[str, Any]:
    """Tier 2 for one class: geometry correspondence, union basis, the continuation."""
    numbers1, pos1, cell1 = _frame_geometry(class_frame, model)
    numbers0, pos0, cell0 = _frame_geometry(pristine_frame, model)
    numbers0, pos0, cell0 = tile_frame(numbers0, pos0, cell0, factors)
    corr = site_correspondence(numbers1, pos1, cell1, numbers0, pos0, cell0, perm=perm,
                               r_match=float(cfg["r_match"]))
    H0_u, H1_u, groups = union_basis(_numpy(H0), _numpy(H1), corr)
    e_sink = cfg["e_sink"]
    if e_sink is None:
        # Section 2.7's default: 50 eV above the pristine conduction edge.
        e_sink = float(np.linalg.eigvalsh(_numpy(H0))[rank]) + 50.0
    out = tier2_continuation(H0_u, H1_u, groups, rank, e_sink=float(e_sink),
                             eta=float(cfg["eta"]), dlambda=float(cfg["dlambda"]),
                             conduction_cut=conduction_cut)
    out["e_sink"] = float(e_sink)
    out["correspondence"] = {k: corr[k] for k in ("n_ghost", "n_added", "n_substituted",
                                                    "n_matched", "max_displacement",
                                                    "mean_displacement")}
    out["correspondence"]["ghost_species"] = [int(numbers0[j]) for j in corr["ghosts"]]
    out["correspondence"]["added_species"] = [int(numbers1[i]) for i in corr["added"]]
    return out


def _numpy(H) -> np.ndarray:
    return H.detach().cpu().numpy() if isinstance(H, torch.Tensor) else np.asarray(H, dtype=np.float64)
