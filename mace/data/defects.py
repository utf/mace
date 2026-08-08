###########################################################################################
# Carrier counters, band-edge referencing and target construction for defect models
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Data-side machinery for charge-aware defect models.

Three responsibilities, all applied before any tensor exists:

* the carrier counter algebra of the plan's section 2.1 (canonicalisation is mandatory,
  not cosmetic: without it time-reversed labellings of the same physical state reach the
  network as different inputs);
* band-edge referencing of the energy labels (section 2.3), so that what the model learns
  is the carrier binding energy relative to the band edge;
* construction of the base-branch and delta targets the loss consumes (section 4).
"""

from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .utils import Configuration, Configurations

# Channel order is fixed everywhere: (electron majority, electron minority,
# hole majority, hole minority).
CARRIER_CHANNELS = ("e_maj", "e_min", "h_maj", "h_min")
NUM_CARRIER_CHANNELS = len(CARRIER_CHANNELS)

# s_c of the plan: -1 for electron-like channels, +1 for hole-like ones.
CARRIER_SIGNS = np.array([-1, -1, 1, 1], dtype=np.int64)

# Members of a pair must be the same geometry, not merely a similar one.
POSITION_TOLERANCE = 1e-8  # Angstrom
BAND_EDGE_TOLERANCE = 1e-6  # eV

_SWAP_INDEX = np.array([1, 0, 3, 2], dtype=np.int64)


def as_counts(value: Any, context: str = "") -> np.ndarray:
    """Coerce a carrier counter vector to four non-negative integers."""
    where = f" ({context})" if context else ""
    counts = np.asarray(value).reshape(-1)
    if counts.size != NUM_CARRIER_CHANNELS:
        raise ValueError(
            f"carrier_counts must have {NUM_CARRIER_CHANNELS} entries "
            f"{CARRIER_CHANNELS}, got {counts.size}{where}"
        )
    as_float = counts.astype(np.float64)
    rounded = np.rint(as_float)
    if not np.allclose(as_float, rounded, atol=1e-9):
        raise ValueError(
            f"carrier_counts must be integers, got {counts.tolist()}{where}"
        )
    counts = rounded.astype(np.int64)
    if (counts < 0).any():
        raise ValueError(
            f"carrier_counts must be non-negative, got {counts.tolist()}{where}"
        )
    return counts


def swap_spin_channels(counts: np.ndarray) -> np.ndarray:
    """Time reversal: exchange the majority and minority channel of each carrier type."""
    return np.asarray(counts)[_SWAP_INDEX]


def net_charge(counts: np.ndarray) -> int:
    """q = (holes) - (electrons)."""
    counts = np.asarray(counts)
    return int(counts[2] + counts[3] - counts[0] - counts[1])


def spin_magnetisation(counts: np.ndarray) -> int:
    """M_s = 2 S_z, the number of unpaired electrons."""
    counts = np.asarray(counts)
    return int((counts[0] - counts[2]) - (counts[1] - counts[3]))


def canonicalise_counts(counts: Any, context: str = "") -> np.ndarray:
    """Map a counter vector to the unique representative of its time-reversal pair.

    ``M_s > 0`` fixes the gauge on its own; at ``M_s = 0`` both labellings satisfy the
    sign convention, so a lexicographic tie-break picks one. The ``M_s`` clause must come
    first: a bare lexicographic rule would return ``M_s = -1`` for a lone hole.
    """
    counts = as_counts(counts, context=context)
    magnetisation = spin_magnetisation(counts)
    if magnetisation > 0:
        return counts
    if magnetisation < 0:
        return swap_spin_channels(counts)
    swapped = swap_spin_channels(counts)
    return counts if tuple(counts.tolist()) >= tuple(swapped.tolist()) else swapped


def validate_counts(
    counts: Any, multiplicity: Optional[float] = None, context: str = ""
) -> np.ndarray:
    """Assert a counter vector is well formed, canonical and consistent with the SCF."""
    where = f" ({context})" if context else ""
    counts = as_counts(counts, context=context)
    if not np.array_equal(counts, canonicalise_counts(counts)):
        raise ValueError(
            f"carrier_counts {counts.tolist()} is not canonical; expected "
            f"{canonicalise_counts(counts).tolist()}{where}"
        )
    if multiplicity is not None:
        expected = spin_magnetisation(counts) + 1
        if int(round(float(multiplicity))) != expected:
            raise ValueError(
                f"carrier_counts {counts.tolist()} implies multiplicity {expected}, "
                f"but {int(round(float(multiplicity)))} was recorded{where}"
            )
    return counts


def enumerate_counts(
    charge: int, max_carriers: Optional[int] = None
) -> List[np.ndarray]:
    """Canonical counter vectors at fixed net charge, bounded as in the plan's section 7.3.

    The unbounded set is infinite (electron-hole pairs can be added indefinitely); the
    default bound admits the minimal assignment plus at most one extra pair.
    """
    if max_carriers is None:
        max_carriers = abs(int(charge)) + 2
    seen = set()
    candidates: List[np.ndarray] = []
    for raw in itertools.product(range(max_carriers + 1), repeat=NUM_CARRIER_CHANNELS):
        counts = np.array(raw, dtype=np.int64)
        if counts.sum() > max_carriers or net_charge(counts) != int(charge):
            continue
        canonical = canonicalise_counts(counts)
        key = tuple(canonical.tolist())
        if key not in seen:
            seen.add(key)
            candidates.append(canonical)
    return sorted(candidates, key=lambda c: (c.sum(), tuple(c.tolist())))


@dataclass(frozen=True)
class BandEdges:
    """Supercell band edges, as total-energy differences (plan section 2.5).

    These are the ``cell`` edges used to reference training targets. The size-converged
    ``infinity`` edges are deliberately absent: nothing in the loss may see them.
    """

    e_cbm_cell: float
    e_vbm_cell: float


def load_band_edges(path: Union[str, Path]) -> Dict[str, BandEdges]:
    """Read the per-host band-edge table used to reference energy labels."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"band edge file '{path}' must map host names to edges")
    table: Dict[str, BandEdges] = {}
    for host, entry in raw.items():
        try:
            table[host] = BandEdges(
                e_cbm_cell=float(entry["e_cbm_cell"]),
                e_vbm_cell=float(entry["e_vbm_cell"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"host '{host}' in '{path}' must provide float 'e_cbm_cell' and "
                f"'e_vbm_cell' entries"
            ) from exc
    return table


def referencing_constant(counts: np.ndarray, edges: BandEdges) -> float:
    """The term subtracted from a raw energy to reference it to the band edges.

    Removed electrons go to the VBM, added electrons come from the CBM, so this is
    linear in the counters over the whole domain with no derivative discontinuity.
    """
    counts = np.asarray(counts)
    electrons = int(counts[0] + counts[1])
    holes = int(counts[2] + counts[3])
    return electrons * edges.e_cbm_cell - holes * edges.e_vbm_cell


def _pair_id(config: Configuration) -> Optional[str]:
    value = config.properties.get("pair_id")
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None"):
        return None
    return text


def _has_counts(config: Configuration) -> bool:
    return config.properties.get("carrier_counts") is not None


def _canonicalise_all(configs: Configurations) -> None:
    missing = 0
    for index, config in enumerate(configs):
        context = f"config {index}"
        raw = config.properties.get("carrier_counts")
        if raw is None:
            # A frame that carries charge-state metadata but no counters is a labelling
            # error, not an implicit neutral: refuse it rather than mislabel the physics.
            suspicious = [
                key
                for key in ("e_cbm_cell", "e_vbm_cell")
                if config.properties.get(key) is not None
            ]
            total_charge = config.properties.get("total_charge")
            if total_charge is not None and abs(float(total_charge)) > 0:
                suspicious.append("total_charge")
            if suspicious:
                raise ValueError(
                    f"{context} has {sorted(suspicious)} but no carrier_counts; the "
                    "carrier configuration cannot be inferred from them (plan 2.1)"
                )
            missing += 1
            config.properties["carrier_counts"] = np.zeros(
                NUM_CARRIER_CHANNELS, dtype=np.int64
            )
            continue
        counts = canonicalise_counts(raw, context=context)
        multiplicity = config.properties.get("multiplicity")
        validate_counts(counts, multiplicity=multiplicity, context=context)
        config.properties["carrier_counts"] = counts
    if missing:
        logging.warning(
            f"{missing} configurations have no carrier_counts and are treated as the "
            "reference state n = 0 (neutral, closed shell)"
        )


def _assert_same_geometry(group: Sequence[Configuration], pair_id: str) -> None:
    first = group[0]
    for other in group[1:]:
        same = (
            np.array_equal(first.atomic_numbers, other.atomic_numbers)
            and first.positions.shape == other.positions.shape
            and np.allclose(first.positions, other.positions, atol=POSITION_TOLERANCE)
        )
        if same and first.cell is not None and other.cell is not None:
            same = np.allclose(first.cell, other.cell, atol=POSITION_TOLERANCE)
        if not same:
            raise ValueError(
                f"configurations sharing pair_id '{pair_id}' are not the same geometry; "
                "a paired charge-state difference is only defined at fixed R"
            )


def _group_indices(configs: Configurations) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, config in enumerate(configs):
        pair_id = _pair_id(config)
        if pair_id is not None:
            groups[pair_id].append(index)
    return dict(groups)


def _join_pairs(configs: Configurations, groups: Dict[str, List[int]]) -> None:
    for pair_id, indices in groups.items():
        members = [configs[i] for i in indices]
        neutral = [
            config
            for config in members
            if int(np.asarray(config.properties["carrier_counts"]).sum()) == 0
        ]
        if len(neutral) != 1:
            raise ValueError(
                f"pair_id '{pair_id}' has {len(neutral)} reference-state (n = 0) members; "
                "exactly one is required to define the paired difference"
            )
        _assert_same_geometry(members, pair_id)
        reference = neutral[0]
        for config in members:
            for target, source in (
                ("base_energy", "energy"),
                ("base_forces", "forces"),
                ("base_stress", "stress"),
            ):
                if config.properties.get(target) is None:
                    config.properties[target] = reference.properties.get(source)


def _resolve_band_edges(
    config: Configuration,
    index: int,
    table: Dict[str, BandEdges],
    seen: Dict[Any, BandEdges],
) -> Optional[BandEdges]:
    e_cbm = config.properties.get("e_cbm_cell")
    e_vbm = config.properties.get("e_vbm_cell")
    if (e_cbm is None) != (e_vbm is None):
        raise ValueError(
            f"config {index} provides only one of e_cbm_cell / e_vbm_cell; both or "
            "neither are required"
        )
    host = config.properties.get("host")
    if e_cbm is None:
        return table.get(str(host)) if host is not None else None

    edges = BandEdges(e_cbm_cell=float(e_cbm), e_vbm_cell=float(e_vbm))
    key = (str(host), len(config.atomic_numbers))
    previous = seen.get(key)
    if previous is None:
        seen[key] = edges
    elif (
        abs(previous.e_cbm_cell - edges.e_cbm_cell) > BAND_EDGE_TOLERANCE
        or abs(previous.e_vbm_cell - edges.e_vbm_cell) > BAND_EDGE_TOLERANCE
    ):
        # Inconsistent edges shift every transition level of that host uniformly, which
        # no downstream metric can see.
        logging.warning(
            f"config {index}: band edges for host {key} disagree with an earlier "
            f"configuration ({edges} vs {previous})"
        )
    return edges


def _apply_band_edge_referencing(
    configs: Configurations, table: Dict[str, BandEdges]
) -> None:
    seen: Dict[Any, BandEdges] = {}
    for index, config in enumerate(configs):
        counts = np.asarray(config.properties["carrier_counts"])
        raw_energy = config.properties.get("energy")
        config.properties["raw_energy"] = raw_energy
        edges = _resolve_band_edges(config, index, table, seen)
        if edges is not None:
            # Write the edges actually used back onto the configuration, so the registry
            # that ships with the model reflects the table as well as per-frame keys.
            config.properties["e_cbm_cell"] = edges.e_cbm_cell
            config.properties["e_vbm_cell"] = edges.e_vbm_cell
        if int(counts.sum()) == 0:
            # The referencing constant is identically zero, so no edges are needed.
            continue
        if edges is None:
            raise ValueError(
                f"config {index} has carrier_counts {counts.tolist()} but no band edges; "
                "supply e_cbm_cell / e_vbm_cell or a band edge file entry for host "
                f"'{config.properties.get('host')}'"
            )
        if raw_energy is None:
            continue
        config.properties["energy"] = float(raw_energy) - referencing_constant(
            counts, edges
        )


def _derive_targets(configs: Configurations) -> None:
    for config in configs:
        counts = np.asarray(config.properties["carrier_counts"])
        is_reference = int(counts.sum()) == 0

        # Base-branch labels: the n = 0 energy, forces and stress at this geometry.
        for target, source in (
            ("base_energy", "raw_energy"),
            ("base_forces", "forces"),
            ("base_stress", "stress"),
        ):
            if config.properties.get(target) is None and is_reference:
                config.properties[target] = config.properties.get(source)

        # Delta labels. The energy is already referenced, so subtracting the raw n = 0
        # energy gives the plan's Delta E_target directly.
        delta_energy = None
        delta_forces = None
        if not is_reference:
            energy = config.properties.get("energy")
            base_energy = config.properties.get("base_energy")
            if energy is not None and base_energy is not None:
                delta_energy = float(energy) - float(base_energy)
            forces = config.properties.get("forces")
            base_forces = config.properties.get("base_forces")
            if forces is not None and base_forces is not None:
                delta_forces = np.asarray(forces) - np.asarray(base_forces)
        config.properties["delta_energy"] = delta_energy
        config.properties["delta_forces"] = delta_forces
        config.property_weights["delta_energy"] = (
            1.0 if delta_energy is not None else 0.0
        )
        config.property_weights["delta_forces"] = (
            1.0 if delta_forces is not None else 0.0
        )


def _assign_base_weights(configs: Configurations, groups: Dict[str, List[int]]) -> None:
    """Exactly one configuration per geometry supervises the base branch.

    ``base_energy`` is a model output on every frame, so a geometry with several charge
    states would otherwise contribute the same base-branch target once per state.
    """
    owner = set()
    grouped = set()
    for indices in groups.values():
        grouped.update(indices)
        reference = [
            i
            for i in indices
            if int(np.asarray(configs[i].properties["carrier_counts"]).sum()) == 0
        ]
        owner.add(reference[0] if reference else min(indices))
    for index in range(len(configs)):
        if index not in grouped:
            owner.add(index)

    for index, config in enumerate(configs):
        carries = index in owner
        for name in ("base_energy", "base_forces", "base_stress"):
            defined = config.properties.get(name) is not None
            config.property_weights[name] = 1.0 if (carries and defined) else 0.0


def prepare_defect_configurations(
    configs: Configurations,
    band_edges: Optional[Dict[str, BandEdges]] = None,
) -> Configurations:
    """Canonicalise counters, reference energies and build the defect training targets.

    Mutates ``configs`` in place and returns it. Applied at ``Configuration`` level so
    that the HDF5 preprocessing path carries the derived targets without further work.
    """
    table = dict(band_edges) if band_edges else {}
    _canonicalise_all(configs)
    groups = _group_indices(configs)
    _join_pairs(configs, groups)
    _apply_band_edge_referencing(configs, table)
    _derive_targets(configs)
    _assign_base_weights(configs, groups)
    return configs


def has_carrier_data(configs: Configurations) -> bool:
    """Whether any configuration carries carrier counters."""
    return any(_has_counts(config) for config in configs)


def canonicalise_config_counters(config: Configuration) -> np.ndarray:
    """Canonicalise (and validate) the counters of a single configuration, in place.

    Inference entry points call this rather than the full transform: there are no
    labels to reference or join, but the counter vector reaching the network must still
    be canonical, or time-reversed labellings of the same physical state get different
    energies (plan section 8.1).
    """
    raw = config.properties.get("carrier_counts")
    if raw is None:
        counts = np.zeros(NUM_CARRIER_CHANNELS, dtype=np.int64)
    else:
        counts = canonicalise_counts(raw)
        validate_counts(counts, multiplicity=config.properties.get("multiplicity"))
    config.properties["carrier_counts"] = counts
    return counts


def registry_key(host: Optional[Any], num_atoms: int) -> str:
    """Key of the band-edge registry: the edges are a property of (host, supercell)."""
    return f"{host}|{int(num_atoms)}"


def collect_band_edge_registry(
    configs: Configurations,
) -> Dict[str, Dict[str, float]]:
    """Gather the band edges actually used, so inference can reproduce them exactly.

    Saved with the model. Without it the affine map back to raw energies depends on the
    user re-deriving constants that training already resolved, which is the section 7.2
    failure mode: a uniform shift of every transition level that no metric can see.
    """
    registry: Dict[str, Dict[str, float]] = {}
    for config in configs:
        e_cbm = config.properties.get("e_cbm_cell")
        e_vbm = config.properties.get("e_vbm_cell")
        if e_cbm is None or e_vbm is None:
            continue
        key = registry_key(config.properties.get("host"), len(config.atomic_numbers))
        registry.setdefault(
            key, {"e_cbm_cell": float(e_cbm), "e_vbm_cell": float(e_vbm)}
        )
    return registry


def lookup_band_edges(
    registry: Dict[str, Any], host: Optional[Any], num_atoms: int
) -> BandEdges:
    """Resolve the edges for a configuration, erring rather than defaulting silently."""
    entry = registry.get(registry_key(host, num_atoms))
    if entry is None:
        raise KeyError(
            f"no band edges recorded for host '{host}' with {num_atoms} atoms; "
            f"known entries: {sorted(registry)}"
        )
    if isinstance(entry, BandEdges):
        return entry
    return BandEdges(
        e_cbm_cell=float(entry["e_cbm_cell"]), e_vbm_cell=float(entry["e_vbm_cell"])
    )
