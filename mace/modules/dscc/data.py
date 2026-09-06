"""Plan section 4 / 6: the data pipeline. Frames become `AtomicData` at the HEAD's cutoff
(D6: the trunk is fed the `<= r_max` subset at forward time); each carries its state (D2),
its formal charge, a geometry key, and loss metadata that is statically unreachable from
the model: the stratum key `(source, host, Q, composition hash, cell convention, size
class)` and the geometry-state group. Groups are formed BEFORE the split so that no
geometry is seen in two folds under different states. Batches are grouped by atom count
(D4) so the head's dense `[B, 4n, 4n]` path applies.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace import data as mace_data
from mace.modules.dscc.legacy import frame_key
from mace.modules.dscc.species import State, neutral_count, state_from_carrier_counts

CELL_CONVENTION = "vasp_background"      # registered (plan section 11)


def composition_hash(numbers: Sequence[int]) -> str:
    counts = {}
    for z in numbers:
        counts[int(z)] = counts.get(int(z), 0) + 1
    text = ",".join(f"{z}x{n}" for z, n in sorted(counts.items()))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class FrameMeta:
    """Loss metadata of one frame; never handed to the model."""
    index: int
    state: State
    n_ref: int
    stratum: Tuple[str, str, int, str, str, int]
    group: int                 # geometry-state group id (the geometry key)
    n_atoms: int
    config_type: str
    source: str


def frame_meta(index: int, atoms, host: str, pristine_atoms: int) -> FrameMeta:
    numbers = [int(z) for z in atoms.get_atomic_numbers()]
    state = state_from_carrier_counts(atoms.info["carrier_counts"])
    if "cell_charge" in atoms.info and int(atoms.info["cell_charge"]) != state.Q:
        raise ValueError(f"frame {index}: counters give Q={state.Q} but cell_charge="
                         f"{int(atoms.info['cell_charge'])}")
    n_atoms = len(numbers)
    # Size class: the pristine cell multiple this frame is nearest to (79 -> 1, 159 -> 2).
    size_class = max(1, int(round(n_atoms / pristine_atoms)))
    source = str(atoms.info.get("source_dir", ""))
    stratum = (source, host, state.Q, composition_hash(numbers), CELL_CONVENTION, size_class)
    group = frame_key(numbers, atoms.get_positions(), np.array(atoms.get_cell()))
    return FrameMeta(index=index, state=state, n_ref=neutral_count(numbers), stratum=stratum,
                     group=int(group), n_atoms=n_atoms,
                     config_type=str(atoms.info.get("config_type", "")), source=source)


def atomic_data(atoms_list: Sequence, z_table, r_cut: float, energy_key: str = "REF_energy",
                forces_key: str = "REF_forces", stress_key: str = "REF_stress"
                ) -> List[mace_data.AtomicData]:
    """`AtomicData` at the head's cutoff with the labels and the per-frame fields the
    model reads: `carrier_counts` (the state) and `frame_key` (the geometry key)."""
    out = []
    for i, atoms in enumerate(atoms_list):
        numbers = atoms.get_atomic_numbers()
        props = {"carrier_counts": np.asarray(atoms.info["carrier_counts"], dtype=np.float64)}
        if energy_key in atoms.info:
            props["energy"] = float(atoms.info[energy_key])
        if forces_key in atoms.arrays:
            props["forces"] = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
        if stress_key in atoms.info:
            props["stress"] = np.asarray(atoms.info[stress_key], dtype=np.float64)
        cfg = mace_data.Configuration(atomic_numbers=numbers, positions=atoms.get_positions(),
                                      cell=np.array(atoms.get_cell()), pbc=(True, True, True),
                                      properties=props, property_weights={})
        d = mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(r_cut))
        d.frame_key = torch.tensor([int(frame_key(numbers, atoms.get_positions(),
                                                  np.array(atoms.get_cell())))],
                                   dtype=torch.long)
        d.frame_index = torch.tensor([i], dtype=torch.long)
        out.append(d)
    return out


def split_by_group(metas: Sequence[FrameMeta], fractions: Sequence[float], seed: int
                   ) -> List[List[int]]:
    """Frame indices per fold; every geometry-state group lands whole in one fold."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to one")
    groups: Dict[int, List[int]] = {}
    for m in metas:
        groups.setdefault(m.group, []).append(m.index)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    bounds = np.cumsum([0.0] + list(fractions)) * len(keys)
    folds: List[List[int]] = []
    for k in range(len(fractions)):
        lo, hi = int(round(bounds[k])), int(round(bounds[k + 1]))
        folds.append(sorted(i for key in keys[lo:hi] for i in groups[key]))
    return folds


def assert_no_group_split(metas: Sequence[FrameMeta], folds: Sequence[Sequence[int]]) -> None:
    by_index = {m.index: m for m in metas}
    seen: Dict[int, int] = {}
    for f, fold in enumerate(folds):
        for i in fold:
            g = by_index[i].group
            if seen.setdefault(g, f) != f:
                raise ValueError(f"geometry group {g} appears in folds {seen[g]} and {f}")


class SizeGroupedSampler(torch.utils.data.Sampler):
    """Batches of equal atom count (D4), shuffled within and across groups each epoch."""

    def __init__(self, sizes: Sequence[int], batch_size: int, seed: int = 0,
                 drop_last: bool = False) -> None:
        self.sizes = [int(s) for s in sizes]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        by_size: Dict[int, List[int]] = {}
        for i, s in enumerate(self.sizes):
            by_size.setdefault(s, []).append(i)
        batches = []
        for s in sorted(by_size):
            idx = np.array(by_size[s])
            rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                chunk = idx[start:start + self.batch_size].tolist()
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        order = rng.permutation(len(batches))
        return [batches[k] for k in order]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
