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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.modules.defect_state import COUNT_FILL, StateBatch

__all__ = ["CLASS_TABLE_VERSION", "QUANTILES", "TIER2_NAMES", "EdgeAlignment",
           "ClassRecord", "composition_key", "is_stoichiometric", "reference_fill",
           "align_edges", "tier1", "class_integers", "frame_counts", "head_spectra",
           "tiling_factors", "tile_frame", "build_class_table", "lookup_class",
           "verify_class_table", "describe"]

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
               "endpoint_classification", "tier2", "linear_sum_assignment", "ghost_orbitals")


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


def tier1(spectrum: np.ndarray, vbm_al: float, delta: float, window: Optional[float] = None
          ) -> Tuple[int, bool, float]:
    """`(M_VB, ambiguous, nearest)`: levels at or below the cut `vbm_al + delta`, whether
    any level lies strictly inside `+-window` of THE CUT, and the closest level's signed
    distance from the cut. `window` defaults to `delta / 2`, the smearing width.

    INTERPRETATION OF RECORD. The plan's acceptance clause, "no eigenvalue lies within
    +-delta of VBM_al", read literally excludes every class including the pristine one: the
    valence edge IS at VBM_al by definition of the alignment, and every class cell's own
    valence top scatters about it by a few meV (thermal frames) to a few tens of meV
    (vacancy-perturbed cells). The operative condition is therefore that the COUNTING CUT
    at VBM_al + delta falls in a spectral gap: levels at or below VBM_al + delta - window
    are valence, levels at or above VBM_al + delta + window are frontier, and a level in
    between makes the class ambiguous at Tier 1. With delta = 2 x smearing and window =
    smearing, the ambiguous band is (VBM_al + Delta_s, VBM_al + 3 Delta_s): the margin
    delta absorbs edge scatter, the window is one smearing width either side of the cut.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    window = 0.5 * float(delta) if window is None else float(window)
    cut = float(vbm_al) + float(delta)
    dist = spectrum - cut
    m_vb = int((dist <= 0.0).sum())
    nearest = float(dist[np.argmin(np.abs(dist))]) if spectrum.size else float("nan")
    # Strictly inside the window, with a tolerance for the exact boundary.
    ambiguous = bool((np.abs(dist) < window - BOUNDARY_TOL).any())
    return m_vb, ambiguous, nearest


def class_integers(m_vb: Sequence[int], n_sigma: Sequence[int]
                   ) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """`((n_e,maj, n_e,min), (n_h,maj, n_h,min), Q_core)` from either tier's `M_VB,sigma`."""
    n_e = tuple(max(int(n) - int(m), 0) for m, n in zip(m_vb, n_sigma))
    n_h = tuple(max(int(m) - int(n), 0) for m, n in zip(m_vb, n_sigma))
    q_core = int(sum(n_e) - sum(n_h))
    return n_e, n_h, q_core


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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassRecord":
        d = dict(d)
        for k in ("n_sigma", "m_vb", "n_e", "n_h", "gamma", "tiling"):
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
    net = [record.n_e[s] - record.n_h[s] + dn[s] for s in range(2)]
    n_e = tuple(max(x, 0) for x in net)
    n_h = tuple(max(-x, 0) for x in net)
    q_f = int(sum(n_h) - sum(n_e))
    q_formal = int(state.q_formal[g])
    if q_f != q_formal - record.q_core:
        raise AssertionError(
            f"q_F = {q_f} but Q_formal - Q_core = {q_formal} - {record.q_core}: the counters "
            f"and the class integers of {record.key} disagree")
    return n_e, n_h, q_f


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
    batch = grabbed["batch"]
    species = d["node_attrs"].argmax(dim=-1)
    valence = head.valence.to(torch.float64)
    out = []
    for g, H in enumerate(hams):
        sel = batch == g
        n_total = int(round(float(valence[species[sel].long()].sum())))
        H64 = H.detach().to(torch.float64)
        lam = torch.linalg.eigvalsh(H64).cpu().numpy()
        out.append({"H": H64, "spectrum": np.sort(lam), "n_total": n_total,
                    "n_atoms": int(sel.sum())})
    return out


def tiling_factors(cell: np.ndarray, reference_cell: np.ndarray, n_atoms: int,
                   n_reference: int, tol: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """`(n_a, n_b, n_c)`, in the REFERENCE's axis order, such that `cell` is the reference
    tiled by those factors up to a relabelling of the lattice vectors and a rigid motion;
    None if no such integer tiling exists within `tol` (relative, per lattice vector).

    The relabelling matters on the data in hand: the 80-atom pristine cell is 2 x 2 x 1 of
    the 20-atom orthorhombic cell and the 159-atom V_Cl cell is 2 x 2 x 2 of it with the
    axes in a different order, so the latter is the former tiled (1, 1, 2) and rotated. The
    head's spectrum is invariant under the rotation (Slater-Koster hoppings and the trunk's
    invariants see distances and angles only), so only the factors are needed. Thermal
    cells differ from the reference by ~2 %, so ratios are rounded per axis, the cell
    ANGLES are required to match under the same relabelling, and the rounded product is
    checked against the atom-count ratio (tolerating the defect's missing or extra atoms).
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
        return factors[0], factors[1], factors[2]
    return None


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
                      delta: Optional[float] = None, window: Optional[float] = None,
                      quantiles: np.ndarray = QUANTILES, log: bool = True) -> Dict[str, Any]:
    """Establish the integers of every composition class present in `frames`.

    `frames` are `AtomicData` (with `frame_key` attached, `defect_cache.attach_frame_keys`).
    The FIRST frame of each composition, in the order given, is its reference geometry. The
    pristine class is the stoichiometric one under `formula`; its first frame's spectrum
    defines the edges every other class is aligned to. Returns the JSON-able table that
    `model.composition_classes` holds.
    """
    formula = _formula_of(model, formula)
    head = model.spectral
    width = float(head.t_el)
    delta = 2.0 * width if delta is None else float(delta)
    window = width if window is None else float(window)
    firsts: Dict[str, Any] = {}
    for fr in frames:
        key = composition_key(_numbers_of(fr, model))
        firsts.setdefault(key, fr)
    pristine_key = next((k for k, fr in firsts.items()
                         if is_stoichiometric(_numbers_of(fr, model), formula)), None)
    table: Dict[str, Any] = {
        "version": CLASS_TABLE_VERSION, "formula": {str(z): n for z, n in formula.items()},
        "delta": delta, "window": window, "smearing_width": width,
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
    for key, fr in firsts.items():
        numbers, _, cell = _frame_geometry(fr, model)
        factors = tiling_factors(cell, pristine_cell, len(numbers), pristine_frame.num_nodes)
        if factors is None:
            table["classes"][key] = _uncounted(
                key, fr, model, delta, "the class cell is not an integer tiling of the "
                "pristine reference cell; no aligned edge exists").to_dict()
            continue
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
        aligned = align_edges(spec["spectrum"], max(n_sig), pri["spectrum"],
                              max(reference_fill(pri["n_total"])), quantiles)
        m_vb_one, ambiguous, nearest = tier1(spec["spectrum"], aligned.vbm_al, delta, window)
        m_vb = (m_vb_one, m_vb_one)
        gap_ok = aligned.gap_pristine >= 4.0 * width
        reason = ""
        if not gap_ok:
            reason = (f"pristine gap {aligned.gap_pristine:.3f} eV < 4 x smearing "
                      f"{4 * width:.3f} eV")
        elif ambiguous:
            reason = (f"a level within +-{window:.3f} eV of the cut VBM_al + delta (nearest "
                      f"{nearest:+.3f} eV): Tier 1 ambiguous")
        accepted = gap_ok and not ambiguous
        n_e, n_h, q_core = class_integers(m_vb, n_sig) if accepted else ((0, 0), (0, 0), 0)
        record = ClassRecord(
            key=key, n_atoms=spec["n_atoms"], reference_frame_key=_frame_key_of(fr),
            n_total=spec["n_total"], n_sigma=n_sig, tier=1 if accepted else None,
            ambiguous=not accepted, reason=reason, m_vb=m_vb, n_e=n_e, n_h=n_h,
            q_core=q_core, vbm_al=aligned.vbm_al, cbm_al=aligned.cbm_al,
            shift=aligned.shift, spread=aligned.spread,
            gap_pristine=aligned.gap_pristine, delta=delta, nearest=nearest,
            tiling=factors)
        table["classes"][key] = record.to_dict()
        if log:
            logging.info("Composition class %s (%d atoms, ref frame %d): %s", key,
                         record.n_atoms, record.reference_frame_key, describe(record))
    return table


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
    return (f"tier {record.tier}, M_VB={list(record.m_vb)}, N={list(record.n_sigma)}, "
            f"n_e={list(record.n_e)}, n_h={list(record.n_h)}, Q_core={record.q_core:+d}, "
            f"VBM_al={record.vbm_al:.3f} eV (shift {record.shift:+.3f}, spread "
            f"{record.spread:.3f}), nearest level to the cut {record.nearest:+.3f} eV, "
            f"pristine tiled {'x'.join(str(t) for t in record.tiling)}")


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
                              delta=table["delta"], window=table.get("window"),
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
# The valence-subspace continuation: site correspondence, union basis, two paths, two
# schedules, transported projector, endpoint classification. Task 0.8 of the plan. Every
# name in TIER2_NAMES is defined below this line and used above it only through `tier2`.
