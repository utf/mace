###########################################################################################
# The Stage 1-4 charged-energy objective (transition plan v8.1 addendum, section 8)
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""Total-cell-eV charged-energy residuals, strata, the within-stratum shape loss, and the
analytic intercept profilers.

WHAT THE ADDENDUM FIXES. The previous objective squared a charged frame's energy residual
divided by its atom count, so an O(1) defect-energy error gave the energy constant a gradient
suppressed as 1/N_at^2 -- and it admitted a charged energy only when a same-size neutral null
happened to exist. Here every valid charged energy enters, in total-cell eV, and the
population balance between strata is an explicit frozen weight, never a per-atom scaling or
an ad-hoc size multiplier.

THE RESIDUAL PATHS (section 8). For a charged observation i, before any energy constant,

    r_i        = E_base(R_i) + Delta E_head(R_i, S_i) - E_i^label                (unpaired)
    r_i^Delta  = Delta E_head(R_i, S_i) - [E_i^label(S_i) - E_i^label(S_ref)]    (paired)

and xi_i is exactly one of them, by the observation's registered label provenance: the paired
path when a same-geometry requested/reference pair exists (the data pipeline's `pair_id`
groups, formed before the split), the unpaired path otherwise. A charged label never enters
both. `Delta E_head` is the complete training-boundary head correction with every c_g or C_Q
omitted; the model must carry no energy constant of its own (asserted at wiring time).

STRATA. A stratum g is keyed ONLY by (label provenance, host, formal charge, composition
key, cell convention, size class); it is loss metadata, never a model input. Every
represented stratum has a frozen total weight W_g, and the frame weights within it are
normalised to one, so a 928-frame stratum and a 16-frame stratum cannot dominate one
another through their counts.

THE SHAPE LOSS (Stages 1-4).

    L_E^shape = sum_g W_g  sum_{i in g} w_i [xi_i - xibar_g]^2 / sum_{i in g} w_i,
    xibar_g   = sum_{i in g} w_i xi_i / sum_{i in g} w_i,

which equals the weighted pair form sum_{i,j in g} w_i w_j [xi_i - xi_j]^2 / (2 (sum w)^2)
exactly (`shape_loss_centred` and `shape_loss_pairs` agree in value and gradient; the tests
pin it). Training uses either the exact full-stratum statistics or the registered unbiased
pair sampler of `WithinStratumPairSampler`: a stratum is drawn with probability W_g / sum W,
then two members independently with probability w_i, and the batch term is the mean over
its pairs of (xi_i - xi_j)^2 / 2, whose expectation is L_E^shape / sum_g W_g. A minibatch
mean that depends on accidental batch composition is not the objective and is not offered.

INTERCEPTS. The analytically profiled nuisance intercept c_g* = -xibar_g is a diagnostic of
Stages 1-4: it exposes the energy shape to the architecture tests and is never a
checkpoint parameter or a production output. From Stage 5 the intercepts are absent and the
single production constant per non-reference charge, C_Q*, is the weighted profile over the
permitted TRAINING observations of every size (`profile_charge_constant`), with
C_{Q_ref} = 0. The two conventions are stage alternatives (`ObjectiveStage`) and are never
added together; a profiler is computed from training observations only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "ObjectiveStage", "PROVENANCE_UNPAIRED", "PROVENANCE_PAIRED", "STRATUM_FIELDS",
    "stratum_key", "Stratum", "StrataTable", "assign_strata", "residual_paths",
    "shape_loss_centred", "shape_loss_pairs", "profile_intercepts",
    "profile_charge_constant", "WithinStratumPairSampler", "assert_no_split_groups",
    "PairBatchCollater", "pair_loader", "FORCE_WEIGHT_COLUMNS", "manifest",
]

PROVENANCE_UNPAIRED = 0
PROVENANCE_PAIRED = 1
STRATUM_FIELDS = ("provenance", "host", "q_formal", "composition", "cell", "size_class")


class ObjectiveStage(str, Enum):
    """Which energy-constant convention the objective is in: the two never coexist."""
    NUISANCE = "nuisance_intercepts"    # Stages 1-4: analytic c_g*, diagnostic only
    PRODUCTION = "production_cq"        # Stages 5-6: one profiled C_Q per charge


def stratum_key(provenance: int, host: str, q_formal: int, composition: str,
                cell: str, size_class: int) -> str:
    prov = "paired" if int(provenance) == PROVENANCE_PAIRED else "unpaired"
    return f"{prov}|{host}|Q{int(q_formal):+d}|{composition}|{cell}|{int(size_class)}x"


@dataclass
class Stratum:
    key: str
    index: int
    weight: float                     # W_g, frozen
    members: List[int] = field(default_factory=list)      # dataset indices
    member_weights: List[float] = field(default_factory=list)   # w_i, normalised to one

    @property
    def informative(self) -> bool:
        """A stratum with fewer than two distinct energy observations supplies no shape
        information (section 8); it is reported, never silently weighted."""
        return len(self.members) >= 2


@dataclass
class StrataTable:
    strata: Dict[str, Stratum]
    host: str
    pristine_atoms: int
    cell_convention: str
    energy_scale: float = 1.0          # total-cell eV; frozen

    def index_of(self, key: str) -> int:
        return self.strata[key].index

    @property
    def informative(self) -> List[Stratum]:
        return [s for s in self.strata.values() if s.informative]

    def summary(self) -> Dict[str, Any]:
        return {k: dict(index=s.index, weight=s.weight, n=len(s.members),
                        informative=s.informative) for k, s in self.strata.items()}


def _composition_key(numbers: Sequence[int]) -> str:
    from collections import Counter
    c = Counter(int(z) for z in numbers)
    return ",".join(f"{z}x{c[z]}" for z in sorted(c))


def _numbers_of(d, z_table) -> List[int]:
    idx = d.node_attrs.argmax(dim=-1).tolist()
    return [int(z_table.zs[i]) for i in idx]


def _is_charged(d) -> bool:
    return int(d.carrier_counts.reshape(-1).sum()) != 0


def _q_formal(d) -> int:
    c = d.carrier_counts.reshape(-1).tolist()
    return int(round(c[2] + c[3] - c[0] - c[1]))


def _is_paired(d) -> bool:
    w = getattr(d, "delta_energy_weight", None)
    v = getattr(d, "delta_energy", None)
    return (w is not None and float(w) > 0.0 and v is not None
            and not bool(torch.isnan(v).any()))


def assign_strata(dataset: Sequence, z_table, host: str, pristine_atoms: int,
                  weights: Optional[Mapping[str, float]] = None, cell_convention: str = "pbc",
                  frame_weight_attr: str = "weight", log: bool = True) -> StrataTable:
    """Build the strata of the charged frames of `dataset` and stamp each frame.

    Every charged `AtomicData` receives `stratum_id` (long, the stratum's index, -1 on a
    neutral frame), `provenance` (long) and `shape_weight` (float, its normalised w_i within
    the stratum). `weights` maps stratum keys to W_g; strata absent from it get 1.0 and a
    log line -- the intent is that the manifest freezes every W_g before results are read.
    """
    strata: Dict[str, Stratum] = {}
    for i, d in enumerate(dataset):
        if not _is_charged(d):
            d.stratum_id = torch.tensor([-1], dtype=torch.long)
            d.provenance = torch.tensor([PROVENANCE_UNPAIRED], dtype=torch.long)
            d.shape_weight = torch.tensor([0.0], dtype=torch.get_default_dtype())
            continue
        prov = PROVENANCE_PAIRED if _is_paired(d) else PROVENANCE_UNPAIRED
        n = int(d.positions.shape[0])
        key = stratum_key(prov, host, _q_formal(d), _composition_key(_numbers_of(d, z_table)),
                          cell_convention, int(round(n / pristine_atoms)))
        if key not in strata:
            strata[key] = Stratum(key=key, index=len(strata),
                                  weight=float((weights or {}).get(key, 1.0)))
        s = strata[key]
        s.members.append(i)
        s.member_weights.append(float(getattr(d, frame_weight_attr, 1.0)))
        d.provenance = torch.tensor([prov], dtype=torch.long)
    for s in strata.values():
        tot = float(sum(s.member_weights)) or 1.0
        s.member_weights = [w / tot for w in s.member_weights]
        for i, w in zip(s.members, s.member_weights):
            dataset[i].stratum_id = torch.tensor([s.index], dtype=torch.long)
            dataset[i].shape_weight = torch.tensor([w], dtype=torch.get_default_dtype())
        if log:
            logging.info("Energy stratum %s: %d frames, W_g = %.3f%s", s.key, len(s.members),
                         s.weight, "" if s.informative else " (UNINFORMATIVE: < 2 frames)")
            if weights is not None and s.key not in weights:
                logging.warning("Energy stratum %s has no registered weight; using 1.0", s.key)
    return StrataTable(strata=strata, host=host, pristine_atoms=int(pristine_atoms),
                       cell_convention=cell_convention)


def assert_no_split_groups(splits: Mapping[str, Sequence]) -> None:
    """Geometry-paired groups (`pair_id`) never span train/validation/test."""
    seen: Dict[str, str] = {}
    for name, ds in splits.items():
        for d in ds:
            pid = getattr(d, "pair_id", None)
            if pid is None:
                continue
            pid = str(pid)
            if pid in seen and seen[pid] != name:
                raise ValueError(f"pair group {pid!r} appears in both {seen[pid]} and {name}: "
                                 "groups are formed before the split and never divided")
            seen[pid] = name


# ------------------------------------------------------------------ residuals


def residual_paths(pred: Mapping[str, torch.Tensor], batch, energy_scale: float = 1.0
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(xi [B], path [B])` in total-cell eV divided by the frozen `energy_scale`.

    path = 1 (paired): xi = pred["delta_energy"] - batch.delta_energy;
    path = 0 (unpaired): xi = pred["energy"] - batch.energy;
    path = -1 on a neutral frame (no charged-energy residual; xi = 0).
    """
    prov = batch["provenance"].reshape(-1)
    charged = batch["carrier_counts"].reshape(int(batch.num_graphs), -1).sum(dim=-1) != 0
    unpaired = pred["energy"] - batch["energy"]
    if "delta_energy" in pred and getattr(batch, "delta_energy", None) is not None:
        paired = pred["delta_energy"] - batch["delta_energy"]
        paired = torch.where(torch.isnan(paired), torch.zeros_like(paired), paired)
    else:
        paired = torch.zeros_like(unpaired)
    xi = torch.where(prov == PROVENANCE_PAIRED, paired, unpaired) / float(energy_scale)
    path = torch.where(charged, prov, torch.full_like(prov, -1))
    xi = torch.where(charged, xi, torch.zeros_like(xi))
    return xi, path


# ------------------------------------------------------------------ the shape loss


def shape_loss_centred(xi: torch.Tensor, stratum_id: torch.Tensor, w: torch.Tensor,
                       W: torch.Tensor) -> torch.Tensor:
    """Exact full-stratum form: sum_g W_g * weighted variance of xi within g. `xi`, `w`
    are per-observation (`w` normalised within each stratum or not; the ratio is formed
    here), `W` is indexed by stratum id; observations with id < 0 are ignored."""
    total = xi.new_zeros(())
    for g in torch.unique(stratum_id[stratum_id >= 0]).tolist():
        m = stratum_id == g
        wg, xg = w[m], xi[m]
        if int(m.sum()) < 2:
            continue                      # uninformative: reported by the table, no term
        denom = wg.sum()
        mean = (wg * xg).sum() / denom
        total = total + W[g] * (wg * (xg - mean) ** 2).sum() / denom
    return total


def shape_loss_pairs(xi: torch.Tensor, stratum_id: torch.Tensor, w: torch.Tensor,
                     W: torch.Tensor) -> torch.Tensor:
    """The exactly equivalent weighted pair form, sum_{i,j in g} w_i w_j (xi_i - xi_j)^2 /
    (2 (sum w)^2), summed over strata with W_g."""
    total = xi.new_zeros(())
    for g in torch.unique(stratum_id[stratum_id >= 0]).tolist():
        m = stratum_id == g
        wg, xg = w[m], xi[m]
        if int(m.sum()) < 2:
            continue
        diff = xg.unsqueeze(0) - xg.unsqueeze(1)
        total = total + W[g] * (wg.unsqueeze(0) * wg.unsqueeze(1) * diff ** 2).sum() \
            / (2.0 * wg.sum() ** 2)
    return total


def sampled_pair_term(xi_pairs: torch.Tensor) -> torch.Tensor:
    """The batch term of the registered pair sampler: mean over pairs of (xi_i - xi_j)^2/2.
    `xi_pairs` is `[P, 2]`. Its expectation over the sampler is L_E^shape / sum_g W_g."""
    if xi_pairs.numel() == 0:
        return xi_pairs.new_zeros(())
    return 0.5 * ((xi_pairs[:, 0] - xi_pairs[:, 1]) ** 2).mean()


# ------------------------------------------------------------------ profilers


def profile_intercepts(xi: np.ndarray, stratum_id: np.ndarray, w: np.ndarray
                       ) -> Dict[int, float]:
    """c_g* = -weighted mean of xi in g (Stages 1-4; diagnostic only)."""
    out = {}
    for g in np.unique(stratum_id[stratum_id >= 0]):
        m = stratum_id == g
        out[int(g)] = float(-(w[m] * xi[m]).sum() / w[m].sum())
    return out


def profile_charge_constant(xi: np.ndarray, q_formal: np.ndarray, stratum_id: np.ndarray,
                            W_by_stratum: Mapping[int, float], w: np.ndarray,
                            q_ref: int = 0) -> Dict[int, Dict[str, float]]:
    """C_Q* for squared loss (Stage 5): the W_g w~_i weighted profile over the TRAINING
    observations of charge Q across every size stratum, C_{Q_ref} = 0. Returns per charge
    the optimum, its optimality residual (the weighted mean of xi + C, zero to the floor),
    and the residual between-size-stratum means AFTER the one constant (a diagnostic)."""
    out: Dict[int, Dict[str, float]] = {}
    for q in np.unique(q_formal):
        q = int(q)
        if q == q_ref:
            out[q] = dict(C=0.0, optimality=0.0, n=0)
            continue
        sel = (q_formal == q) & (stratum_id >= 0)
        if not sel.any():
            continue
        wt = np.array([W_by_stratum[int(g)] for g in stratum_id[sel]]) * w[sel]
        C = float(-(wt * xi[sel]).sum() / wt.sum())
        per_stratum = {int(g): float((xi[sel][stratum_id[sel] == g] + C).mean())
                       for g in np.unique(stratum_id[sel])}
        out[q] = dict(C=C, optimality=float((wt * (xi[sel] + C)).sum() / wt.sum()),
                      n=int(sel.sum()), per_stratum_mean_after=per_stratum,
                      between_stratum_spread=float(max(per_stratum.values())
                                                   - min(per_stratum.values())))
    return out


# ------------------------------------------------------------------ the pair sampler


class WithinStratumPairSampler(torch.utils.data.Sampler):
    """Ordinary shuffled batches with `n_pair_slots` registered pairs appended to each.

    Every batch is `base[:B] + [i1, j1, i2, j2, ...]`: the first B indices are the ordinary
    shuffled sweep of the dataset (forces, neutral base terms), the last 2 * n_pair_slots
    are pairs drawn by the registered rule -- a stratum with probability W_g / sum W among
    the informative strata, then two members independently with probability w_i (with
    replacement: an equal pair contributes zero and is what the exact pair form contains).
    The loss reads the pairs by position; `pair_slots` records the layout.
    """

    def __init__(self, n_items: int, batch_size: int, table: StrataTable,
                 n_pair_slots: int = 2, generator: Optional[torch.Generator] = None,
                 drop_last: bool = True) -> None:
        self.n_items, self.batch_size = int(n_items), int(batch_size)
        self.n_pair_slots = int(n_pair_slots)
        self.generator = generator
        self.drop_last = bool(drop_last)
        strata = table.informative
        if not strata:
            raise ValueError("no informative energy stratum: nothing to pair")
        self._members = [torch.tensor(s.members, dtype=torch.long) for s in strata]
        self._probs = [torch.tensor(s.member_weights, dtype=torch.float64) for s in strata]
        self._stratum_probs = torch.tensor([s.weight for s in strata], dtype=torch.float64)
        self._stratum_probs = self._stratum_probs / self._stratum_probs.sum()
        self.pair_slots = list(range(self.batch_size, self.batch_size + 2 * self.n_pair_slots))

    def __len__(self) -> int:
        return (self.n_items // self.batch_size if self.drop_last
                else math.ceil(self.n_items / self.batch_size))

    def draw_pair(self) -> Tuple[int, int]:
        g = int(torch.multinomial(self._stratum_probs, 1, generator=self.generator))
        ij = torch.multinomial(self._probs[g], 2, replacement=True, generator=self.generator)
        return int(self._members[g][ij[0]]), int(self._members[g][ij[1]])

    def __iter__(self) -> Iterator[List[int]]:
        order = torch.randperm(self.n_items, generator=self.generator).tolist()
        for k in range(len(self)):
            base = order[k * self.batch_size:(k + 1) * self.batch_size]
            pairs: List[int] = []
            for _ in range(self.n_pair_slots):
                pairs.extend(self.draw_pair())
            yield base + pairs


# ------------------------------------------------------------------ the pair batches

# The per-graph columns the Stage B terms multiply. `weight` enters every term; the force
# columns enter the atom-mean terms. Both are rescaled on the base graphs so that the
# non-shape terms on a 12-graph pair batch equal, exactly, what Stage B computes on the
# 8-graph batch the base graphs form on their own.
FORCE_WEIGHT_COLUMNS = ("forces_weight", "base_forces_weight", "delta_forces_weight")


class PairBatchCollater:
    """Collate a pair-sampler batch and mark it structurally.

    The first `num_graphs - 2 n_pair_slots` graphs are the ordinary sweep, the rest are the
    registered pairs. The batch is stamped with `pair_slots` (the number of pairs) so the
    loss finds the pairs by a property of the batch, not by trusting its length or the
    module's mode: a validation batch carries no stamp and scores no pair term.

    The pair graphs enter ONLY the shape term. Their `weight` is set to zero, which removes
    them from every other term (each of those multiplies `weight`), and the base graphs'
    columns are rescaled by the graph ratio (graph-mean terms) and the atom ratio (atom-mean
    force terms) so those terms' values are what the same base graphs give on their own.
    Without this the pair draws -- half of them from the 16-frame stratum under equal W_g --
    would enter the force terms some forty times per epoch each, at the two-size force
    upweight, and the realised force shares would no longer be the recipe's. The shape term
    does not read `weight`: the sampler already draws members with probability w_i.
    """

    def __init__(self, inner, n_pair_slots: int) -> None:
        self.inner = inner
        self.n_pair_slots = int(n_pair_slots)

    def __call__(self, data_list):
        batch = self.inner(data_list)
        n_pair = 2 * self.n_pair_slots
        n_graphs = int(batch.num_graphs)
        if n_graphs <= n_pair:
            raise ValueError(f"a pair batch of {n_graphs} graphs has no base graphs")
        n_base = n_graphs - n_pair
        counts = batch.ptr[1:] - batch.ptr[:-1]
        graph_ratio = n_graphs / n_base
        atom_ratio = float(counts.sum()) / float(counts[:n_base].sum())
        weight = batch.weight.view(-1)
        weight[:n_base] = weight[:n_base] * graph_ratio
        weight[n_base:] = 0.0
        for name in FORCE_WEIGHT_COLUMNS:
            col = getattr(batch, name, None)
            if col is None:
                continue
            col = col.view(-1)
            col[:n_base] = col[:n_base] * (atom_ratio / graph_ratio)
        batch.pair_slots = torch.tensor(self.n_pair_slots, dtype=torch.long)
        return batch


def pair_loader(dataset, sampler: WithinStratumPairSampler, **kwargs):
    """A DataLoader over `sampler`'s batches whose collater is `PairBatchCollater`."""
    from mace.tools import torch_geometric

    loader = torch_geometric.dataloader.DataLoader(dataset=dataset, batch_sampler=sampler,
                                                   **kwargs)
    loader.collate_fn = PairBatchCollater(loader.collate_fn, sampler.n_pair_slots)
    return loader


# ------------------------------------------------------------------ manifest


def manifest(table: StrataTable, stage: ObjectiveStage, energy_shape_weight: float,
             forces_weight: float, n_pair_slots: int, tolerances: Mapping[str, float],
             extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The frozen record of the objective (section 8: energy scale, strata and weights,
    force/energy balance, pair rule, tolerances), with a content hash. Written BEFORE the
    corrected results are opened."""
    body = dict(objective_version=1, stage=stage.value, energy_scale_eV=table.energy_scale,
                host=table.host, pristine_atoms=table.pristine_atoms,
                cell_convention=table.cell_convention, strata=table.summary(),
                energy_shape_weight=float(energy_shape_weight),
                forces_weight=float(forces_weight),
                pair_rule=dict(sampler="WithinStratumPairSampler", n_pair_slots=int(n_pair_slots),
                               stratum_probability="W_g / sum W over informative strata",
                               member_probability="w_i, with replacement"),
                tolerances=dict(tolerances), extra=dict(extra or {}))
    body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return body
