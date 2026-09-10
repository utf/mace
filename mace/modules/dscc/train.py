"""Plan section 6 / 7 (v4.2): the training protocol for one arm configuration.

Training is on FORCES at every size (C7): the head's parameters see only the charged frames
(at `S_ref` the model returns the base, which is frozen), weighted by stratum -- frozen total
weight per stratum, frame weights normalised within it -- plus the mandatory gap regulariser
on the static pristine cell (D8) and, when the coupling is on, a per-epoch single-valuedness
check on a registered subsample (C5). `C_Q` is profiled post hoc (`calibration.py`), never in
the loss. Every registered number lives in `TrainConfig` and is written to the run record
before training starts; the record travels with the checkpoint.

The outer folds are the cross-fit folds of `dataset_cf` for the neutral frames (so the
out-of-fold null of the admission table aligns with them) and a seeded geometry-group split
for the charged frames. Pristine frames enter every fold's training set (they carry no head
signal; they set the pristine centre and the thermal-gap diagnostic).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace import tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.admission import collective_coordinate
from mace.modules.dscc.model import MACEDSCC
from mace.modules.dscc.scf import root_rule
from mace.modules.dscc.species import state_from_carrier_counts
from mace.tools import torch_geometric


@dataclass
class TrainConfig:
    """Registered training numbers (to confirm before Arm-1 results are opened)."""
    name: str = "arm1"
    seed: int = 0
    fold: int = 0                     # outer fold held out (0..3)
    n_folds: int = 4
    epochs: int = 60
    lr: float = 2e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    r_cut: float = 10.0
    force_weight: float = 1.0
    gap_weight: float = 1.0           # L_gap = gap_weight (E_gap_model - E_gap)^2, eV^-2
    # v5 amendment A1: the weak L2 on the tanh outputs that replaces the pristine centre as
    # the thing keeping corrections near the element default. `L_l2 = l2_weight * sum_terms
    # mean(tanh^2)`, the same weight on every term. Registered 1.8e-6 = 0.05 * L_conv / 0.25,
    # L_conv = 8.94e-6 the median converged train force loss of the six archived pre-A1
    # Phi = 0 seeds: the penalty is 5 % of the converged force loss at |tanh| = 0.5.
    l2_weight: float = 1.8e-6
    e_gap: float = 2.40               # registered host gap (static lattice, C2)
    coupling: bool = False            # Arm 1: Phi = 0
    coupling_mode: str = "full"       # Arm 2+3: lr_only | lr_u | full | lambda1
    route_b: bool = False
    directional: bool = True          # False: the scalar-only control
    regime: str = "B"
    stratum_weights: Dict[str, float] = field(default_factory=dict)   # frozen per stratum key
    single_valued_subsample: int = 16     # pre-v4.5: frames per epoch (coupling on); superseded by the fraction
    single_valued_ceiling: float = 0.10   # failing fraction that fails the arm (registered ceiling)
    warm_check_fraction: float = 0.05     # v4.5: per-epoch subsample re-run by continuation against the warm start
    eval_every: int = 5
    avg_window: int = 10              # W0.4 (v5): epochs whose parameters are averaged for the evaluation model
    # Setup-pass batching. These passes are geometry-only and run once before training; their
    # batch size changes no reported number, only the peak memory of `prepare`, which on the
    # 512-wide base v2 features is the high-water mark of the whole run. Defaults are the
    # literals the campaign ran with, so nothing moves unless a run sets them.
    setup_batch_size: int = 16        # pristine-reference pass
    base_cache_batch_size: int = 8    # E_base / F_base cache
    grad_clip: float = 10.0
    cache_base: bool = True           # E_base/F_base cached per frame; block 0 recomputed
    device: str = "cuda"
    static_cell_path: str = "defect-perovskite/static_pristine_cell.json"


def stratum_id(meta: dd.FrameMeta) -> str:
    return "|".join(str(x) for x in meta.stratum)


def load_frames(train_xyz: str, valid_xyz: str):
    import ase.io
    return ase.io.read(train_xyz, ":") + ase.io.read(valid_xyz, ":")


def outer_folds(frames, metas: Sequence[dd.FrameMeta], cf_dir: str, n_folds: int, seed: int
                ) -> Dict[int, int]:
    """Fold of every frame index: neutral defective frames from the cross-fit `null_oof`
    files, charged frames by a seeded geometry-group split, pristine frames in no fold
    (they train everywhere)."""
    import ase.io
    from mace.modules.dscc.legacy import frame_key
    fold_of: Dict[int, int] = {}
    keys = {int(frame_key(a.get_atomic_numbers(), a.get_positions(), np.array(a.get_cell()))): i
            for i, a in enumerate(frames)}
    for k in range(n_folds):
        for a in ase.io.read(f"{cf_dir}/fold{k}/null_oof.xyz", ":"):
            key = int(frame_key(a.get_atomic_numbers(), a.get_positions(), np.array(a.get_cell())))
            if key in keys:
                fold_of[keys[key]] = k
    charged = [m for m in metas if m.state.Q != 0]
    parts = dd.split_by_group(charged, [1.0 / n_folds] * n_folds, seed)
    for k, part in enumerate(parts):
        for i in part:
            fold_of[i] = k
    return fold_of


def static_cell_atoms(path: str):
    from ase import Atoms
    rec = json.load(open(path))
    atoms = Atoms(numbers=rec["numbers"], positions=rec["positions"], cell=rec["cell"], pbc=True)
    atoms.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "static"})
    return atoms


def vacancy_centre_full(pos: torch.Tensor, cell: torch.Tensor, numbers):
    """Distance of every atom from the vacancy centre: the midpoint of the flanking Pb pair
    (the two Pb whose sixth-nearest Cl is beyond 4 A), label-free. The pair can be neighbours
    through MORE THAN ONE image of a short cell axis (the 79-atom cell is two octahedra thick
    along z, 11.1 A): one shared site holds the vacancy, the other a normal bridging Cl, and in
    the +1 state the occupied path is often the shorter one. Among the pair-vector images
    shorter than 8 A the vacancy-side midpoint is the one whose nearest non-flanking atom is
    farthest away (D15, 2026-09-09; the minimum-image midpoint before that). None when no
    flanking pair is found."""
    from mace.modules.dscc.kernels import minimum_image_distances
    z = torch.as_tensor(numbers, device=pos.device)
    r = minimum_image_distances(pos, cell)
    pb = torch.nonzero(z == 82).reshape(-1); cl = torch.nonzero(z == 17).reshape(-1)
    if pb.numel() == 0 or cl.numel() < 6:
        return None
    dist = torch.sort(r[pb][:, cl], dim=1).values
    flank = pb[dist[:, 5] > 4.0]
    if flank.numel() != 2:
        return None
    inv = torch.linalg.inv(cell)
    raw = pos[flank[1]] - pos[flank[0]]
    raw = raw - torch.round(raw @ inv) @ cell
    best = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                v = raw + torch.tensor([i, j, k], dtype=pos.dtype, device=pos.device) @ cell
                length = float(v.norm())
                if length >= 8.0:
                    continue
                dv = pos - (pos[flank[0]] + 0.5 * v)
                dv = dv - torch.round(dv @ inv) @ cell
                rad = dv.norm(dim=-1)
                other = rad.clone(); other[flank] = float("inf")
                score = (float(other.min()), -length)
                if best is None or score > best[0]:
                    best = (score, rad)
    if best is None:
        return None
    return best[1], flank


def vacancy_centre(pos: torch.Tensor, cell: torch.Tensor, numbers) -> Optional[torch.Tensor]:
    """The radii alone (the signature every pre-W0 caller uses)."""
    out = vacancy_centre_full(pos, cell, numbers)
    return None if out is None else out[0]


FIRST_SHELL_CL = 3.5      # A: a Cl is "first shell" if it is within this of a flanking Pb (W0.2)


def near_field_categories(pos: torch.Tensor, cell: torch.Tensor, numbers,
                          flank: torch.Tensor) -> Dict[str, torch.Tensor]:
    """W0.2 near-field split of the 2-4 A shell: the flanking Pb pair, the first-shell Cl
    (Cl within FIRST_SHELL_CL of a flanking Pb) and the remainder. Boolean masks over the
    frame's atoms, before any radius window is applied -- `evaluate` intersects them with
    the 2-4 A shell and reports the unrestricted sets' counts as information."""
    from mace.modules.dscc.kernels import minimum_image_distances
    z = torch.as_tensor(numbers, device=pos.device)
    r = minimum_image_distances(pos, cell)
    pb_flank = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    pb_flank[flank] = True
    cl = z == 17
    cl_first = cl & (r[:, flank].min(dim=1).values <= FIRST_SHELL_CL)
    return {"pb_flank": pb_flank, "cl_first": cl_first, "other": ~(pb_flank | cl_first)}


def average_into(avg_sum: Optional[Dict[str, torch.Tensor]], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """W0.4: accumulate the TRAINABLE parameters of `model` into a float64 running sum."""
    with torch.no_grad():
        named = {k: v for k, v in model.named_parameters() if v.requires_grad}
        if avg_sum is None:
            return {k: v.detach().to(torch.float64).clone() for k, v in named.items()}
        for k, v in named.items():
            avg_sum[k] += v.detach().to(torch.float64)
    return avg_sum


def load_average(model: torch.nn.Module, avg_sum: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    """W0.4: write the uniform average into `model`, returning the parameters it replaced (so
    the caller can restore the last-epoch model afterwards)."""
    saved = {}
    with torch.no_grad():
        for k, v in model.named_parameters():
            if v.requires_grad:
                saved[k] = v.detach().clone()
                v.copy_((avg_sum[k] / n).to(v.dtype))
    return saved


def restore_parameters(model: torch.nn.Module, saved: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in saved:
                v.copy_(saved[k])


def to_device(batch, device):
    return batch.to(device).to_dict()


def force_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
               weights: torch.Tensor) -> torch.Tensor:
    """Sum over graphs of `w_g mean_atoms |F - F_label|^2` (eV/A)^2; `weights` per graph."""
    diff = (out["forces"] - batch["forces"]) ** 2
    per_atom = diff.sum(-1)
    num_graphs = int(batch["ptr"].numel() - 1)
    per_graph = torch.zeros(num_graphs, dtype=per_atom.dtype, device=per_atom.device)
    per_graph = per_graph.index_add(0, batch["batch"], per_atom)
    counts = torch.bincount(batch["batch"], minlength=num_graphs).to(per_atom.dtype)
    return (weights * per_graph / counts).sum()


class Trainer:
    def __init__(self, model: MACEDSCC, cfg: TrainConfig, frames, metas: Sequence[dd.FrameMeta],
                 fold_of: Dict[int, int], run_dir: str) -> None:
        self.model, self.cfg = model, cfg
        self.frames, self.metas, self.fold_of = frames, list(metas), fold_of
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.z_table = tools.AtomicNumberTable(model.atomic_numbers)
        self.device = cfg.device
        charged = [m for m in self.metas if m.state.Q != 0]
        self.train_idx = [m.index for m in charged if fold_of.get(m.index) != cfg.fold]
        self.held_idx = [m.index for m in charged if fold_of.get(m.index) == cfg.fold]
        # Stratum weights: frozen totals (default 1 per stratum), frames normalised within.
        strata: Dict[str, List[int]] = {}
        for i in self.train_idx:
            strata.setdefault(stratum_id(self.metas[i]), []).append(i)
        self.frame_weight: Dict[int, float] = {}
        for key, members in strata.items():
            total = float(cfg.stratum_weights.get(key, 1.0))
            for i in members:
                self.frame_weight[i] = total / len(members)
        self.strata_record = {key: {"n": len(v), "total_weight": float(cfg.stratum_weights.get(key, 1.0))}
                              for key, v in strata.items()}
        self.static_batch = None
        self.log: List[Dict[str, object]] = []
        # v4.5: the converged `dq` of every frame visited in this arm (CPU), the warm start
        # of its later visits; a frame the check fails, or an unconverged solve, is dropped
        # from the store and starts again from the continuation.
        self.dq_store: Dict[int, torch.Tensor] = {}
        self.check_rng = np.random.default_rng(cfg.seed + 7)
        self.sv_ceiling_exceeded_epochs: List[int] = []

    # ------------------------------------------------------------ data

    def _dataset(self, indices: Sequence[int]):
        return dd.atomic_data([self.frames[i] for i in indices], self.z_table, self.cfg.r_cut)

    def prepare(self, pristine_indices: Sequence[int]) -> None:
        """Pristine composition and q0 reference from the pristine frames; the static cell
        batch. A1 removed the feature-mean pass this used to open with."""
        ds = dd.atomic_data([self.frames[i] for i in pristine_indices], self.z_table, self.cfg.r_cut)
        loader = torch_geometric.dataloader.DataLoader(ds, batch_size=self.cfg.setup_batch_size)
        self.model.set_pristine_reference([to_device(b, self.device) for b in loader])
        static = dd.atomic_data([static_cell_atoms(self.cfg.static_cell_path)], self.z_table, self.cfg.r_cut)
        self.static_batch = to_device(next(iter(torch_geometric.dataloader.DataLoader(static, batch_size=1))), self.device)
        # The static cell never moves: its base features are geometry-only, cached once.
        with torch.no_grad():
            feats = self.model.first_block(self.model._trunk_data(dict(self.static_batch)))
            self.static_features = tuple(t.detach() for t in self.model.features(feats))
        # E_base and F_base of every charged frame (frozen base, geometry-only), cached once.
        self.base_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if self.cfg.cache_base:
            self._fill_base_cache(self.train_idx + self.held_idx)
        self.write_record()

    def _fill_base_cache(self, indices: Sequence[int]) -> None:
        ds = self._dataset(indices)
        k = 0
        for b in torch_geometric.dataloader.DataLoader(ds, batch_size=self.cfg.base_cache_batch_size):
            batch = to_device(b, self.device)
            # `base_forward` runs the base in its own precision and returns float64 (the base
            # may be float32; the cache and every head quantity stay float64).
            out = self.model.base_forward(self.model._trunk_data(dict(batch)), training=False,
                                          compute_force=True)
            ptr = batch["ptr"]
            for g in range(int(ptr.numel() - 1)):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                self.base_cache[indices[k]] = (out["energy"][g].detach().cpu(), out["forces"][lo:hi].detach().cpu())
                k += 1

    def _attach_base(self, batch: Dict[str, torch.Tensor], frame_indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        if not self.base_cache or any(i not in self.base_cache for i in frame_indices):
            return batch
        batch = dict(batch)
        batch["dscc_base_energy"] = torch.stack([self.base_cache[i][0] for i in frame_indices]).to(self.device)
        batch["dscc_base_forces"] = torch.cat([self.base_cache[i][1] for i in frame_indices]).to(self.device)
        return batch

    def write_record(self) -> None:
        """The run record (every registered number, strata, folds), written before training."""
        record = {"config": asdict(self.cfg), "strata": self.strata_record, "n_train": len(self.train_idx),
                  "n_held": len(self.held_idx), "fold_of": {str(k): v for k, v in self.fold_of.items()},
                  "model_extra_state": self.model.get_extra_state()}
        json.dump(record, open(self.run_dir / "run_record.json", "w"), indent=1, default=str)

    # ------------------------------------------------------------ v4.5 warm starts

    def _warm_starts(self, frame_indices: Sequence[int]) -> Optional[List[Optional[torch.Tensor]]]:
        """Per-graph stored `dq` (None on a first visit); None altogether without coupling."""
        if not self.cfg.coupling:
            return None
        starts = [self.dq_store.get(int(i)) for i in frame_indices]
        if all(w is None for w in starts):
            return None
        return [None if w is None else w.to(self.device) for w in starts]

    def _store(self, out: Dict[str, object], batch: Dict[str, torch.Tensor], frame_indices: Sequence[int]) -> None:
        """Keep the converged solutions of this pass; forget the unconverged ones."""
        if not self.cfg.coupling or out.get("dq") is None:
            return
        ptr = batch["ptr"]
        conv = out.get("diagnostics", {}).get("converged", [])
        dq = out["dq"].detach().cpu()
        for g, i in enumerate(frame_indices):
            lo, hi = int(ptr[g]), int(ptr[g + 1])
            if g < len(conv) and conv[g] is True:
                self.dq_store[int(i)] = dq[lo:hi].clone()
            else:
                self.dq_store.pop(int(i), None)

    # ------------------------------------------------------------ steps

    def step(self, batch, frame_indices: Sequence[int], optimizer) -> Dict[str, float]:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        self.model.h0.sk.reset_regularisation()      # A1: the L2 pools over every H0 of the step
        out = self.model(self._attach_base(batch, frame_indices), training=True, compute_force=True,
                         warm_start=self._warm_starts(frame_indices))
        self._store(out, batch, frame_indices)
        weights = torch.tensor([self.frame_weight[i] for i in frame_indices], dtype=torch.float64,
                               device=self.device)
        l_force = force_loss(out, batch, weights)
        gap = self.model.pristine_gap(self.static_batch, getattr(self, "static_features", None))
        l_gap = self.cfg.gap_weight * ((gap - self.cfg.e_gap) ** 2).sum()
        reg = self.model.h0.sk.regularisation()
        l_l2 = self.cfg.l2_weight * sum(reg.values()) if reg else l_force.new_zeros(())
        loss = self.cfg.force_weight * l_force + l_gap + l_l2
        loss.backward()
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        optimizer.step()
        diag = out.get("diagnostics", {})
        conv = diag.get("converged", [])
        return {"loss": float(loss), "force": float(l_force), "gap": float(gap[0]),
                "l2": float(l_l2),
                "unconverged": int(sum(1 for c in conv if c is False)),
                "scf_iterations": int(sum(diag.get("iterations", []))),
                "scf_fills": int(sum(diag.get("fills", []))),
                "warm": int(bool(diag.get("warm_started", False)))}

    @torch.no_grad()
    def saturation_report(self, indices: Sequence[int], n_frames: int = 8) -> Dict[str, object]:
        """A1's bounds report: the fraction of `|tanh| > 0.95` per term and species over a
        sample of held-out frames, plus the section 2.10 readout of `tanh(e_Z(h_i))` (and
        `tanh(g_Z(h_i))` when the rank-2 bound is on) AT THE FLANKING Pb -- the two sites the
        `vacancy_centre_full` rule already identifies. Sampled rather than exhaustive: it is a
        diagnostic on a saturating nonlinearity, not a mean anyone tests against."""
        self.model.eval()
        sample = list(indices)[:max(1, int(n_frames))]
        ds = self._dataset(sample)
        loader = torch_geometric.dataloader.DataLoader(ds, batch_size=1)
        pooled: Dict[str, list] = {}
        flank_site: Dict[str, list] = {}
        for j, b in enumerate(loader):
            batch = to_device(b, self.device)
            feats = self.model.first_block(self.model._trunk_data(dict(batch)))
            scalars, _ = self.model.features(feats)
            species = batch["node_attrs"].argmax(dim=-1)
            a = self.frames[sample[j]]
            found = vacancy_centre_full(batch["positions"], batch["cell"].view(-1, 3, 3)[0],
                                        a.get_atomic_numbers())
            sites = None
            if found is not None:
                sites = torch.zeros(int(species.shape[0]), dtype=torch.bool, device=species.device)
                sites[torch.as_tensor(found[1], device=species.device)] = True
            rep_ = self.model.h0.saturation(scalars, species, batch["edge_index"], sites)
            for key, value in rep_.items():
                (flank_site if key.startswith("site_") else pooled).setdefault(key, []).append(value)
        out: Dict[str, object] = {}
        for key, values in pooled.items():
            if isinstance(values[0], dict):
                out[key] = {z: float(np.mean([v[z] for v in values if z in v]))
                            for z in sorted({z for v in values for z in v})}
            else:
                out[key] = float(np.mean(values))
        for key, values in flank_site.items():
            flat = [x for v in values for x in v]
            out[key] = {"mean_abs": float(np.mean(np.abs(flat))), "max_abs": float(np.max(np.abs(flat))),
                        "n": len(flat)}
        out["n_frames"] = len(sample)
        return out

    def evaluate(self, indices: Sequence[int], tag: str) -> Dict[str, object]:
        """Force RMSE (eV/A) overall and by distance shell from the vacancy (label-free:
        the flanking-Pb midpoint), and the uncalibrated energy residual per frame."""
        self.model.eval()
        ds = self._dataset(indices)
        loader = torch_geometric.dataloader.DataLoader(ds, batch_size=self.cfg.batch_size)
        sq, n_atoms = 0.0, 0
        # Convention: `force_rmse` and the shell values are the RMS of the per-atom force-error
        # VECTOR norm, sqrt(sum |dF|^2 / N_atoms) -- sqrt(3) times the per-component RMS that
        # MACE's own logs report. The > 8 A shells are resolved since C10 (2026-09-08) and the
        # atom counts kept so shells can be pooled.
        # W0.2 (v5): the 4-8 A pooled shell is a first-class key (the far-field gate reads it);
        # the 2-4 A shell is additionally split into the flanking Pb pair, the first-shell Cl
        # and the remainder. The 0-2 A shell is kept (empty under a correct centre for charged
        # frames) so the neutral dimerised pair stays visible as a diagnostic.
        shells = {(0, 2): [0.0, 0], (2, 4): [0.0, 0], (4, 6): [0.0, 0], (6, 8): [0.0, 0], (4, 8): [0.0, 0],
                  (8, 99): [0.0, 0], (8, 10): [0.0, 0], (10, 12): [0.0, 0], (12, 99): [0.0, 0]}
        near = {"2-4:pb_flank": [0.0, 0], "2-4:cl_first": [0.0, 0], "2-4:other": [0.0, 0]}
        near_all = {"pb_flank": [0.0, 0], "cl_first": [0.0, 0]}     # unrestricted, information only
        energies = []
        k = 0
        for b in loader:
            batch = to_device(b, self.device)
            n_graphs = int(batch["ptr"].numel() - 1)
            batch_frames = [indices[k + g] for g in range(n_graphs)]
            with torch.no_grad():
                out = self.model(batch, compute_force=True, warm_start=self._warm_starts(batch_frames))
            self._store(out, batch, batch_frames)
            out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
            diff = ((out["forces"] - batch["forces"]) ** 2).sum(-1)
            sq += float(diff.sum()); n_atoms += int(diff.numel())
            ptr = batch["ptr"]
            for g in range(int(ptr.numel() - 1)):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                a = self.frames[indices[k]]; k += 1
                pos, cell = batch["positions"][lo:hi], batch["cell"].view(-1, 3, 3)[g]
                numbers = a.get_atomic_numbers()
                d = collective_coordinate(pos, cell, numbers)
                found = vacancy_centre_full(pos, cell, numbers)   # vacancy-side midpoint (D15)
                rad = None if found is None else found[0]
                if rad is not None:
                    for (a_, b_), acc in shells.items():
                        m = (rad >= a_) & (rad < b_)
                        acc[0] += float(diff[lo:hi][m].sum()); acc[1] += int(m.sum())
                    cats = near_field_categories(pos, cell, numbers, found[1])
                    window = (rad >= 2.0) & (rad < 4.0)
                    for key, mask in cats.items():
                        acc = near[f"2-4:{key}"]
                        m = window & mask
                        acc[0] += float(diff[lo:hi][m].sum()); acc[1] += int(m.sum())
                    for key in ("pb_flank", "cl_first"):
                        acc = near_all[key]
                        m = cats[key]
                        acc[0] += float(diff[lo:hi][m].sum()); acc[1] += int(m.sum())
                energies.append({"index": indices[k - 1], "n": hi - lo, "d": d,
                                 "resid_uncal": float(out["energy_uncalibrated"][g] - batch["energy"][g])})
        rms = lambda v: (float(np.sqrt(v[0] / v[1])) if v[1] else None)
        report = {"tag": tag, "force_rmse": float(np.sqrt(sq / max(n_atoms, 1))),
                  "shell_rmse": {f"{a_}-{b_}": rms(v) for (a_, b_), v in shells.items()},
                  "shell_counts": {f"{a_}-{b_}": v[1] for (a_, b_), v in shells.items()},
                  "near_rmse": {k: rms(v) for k, v in near.items()},
                  "near_counts": {k: v[1] for k, v in near.items()},
                  "near_rmse_all_radii": {k: rms(v) for k, v in near_all.items()},
                  "near_counts_all_radii": {k: v[1] for k, v in near_all.items()},
                  "energies": energies}
        if self.cfg.coupling and indices:
            # v4.5: the held-out warm starts are checked on the registered fraction too.
            size = max(1, int(round(self.cfg.warm_check_fraction * len(indices))))
            sample = self.check_rng.choice(list(indices), size=min(size, len(indices)), replace=False).tolist()
            report["single_valued"] = self.warm_check(sample)
        self._release_pool()
        return report

    def warm_check(self, indices: Sequence[int]) -> Dict[str, object]:
        """v4.5 (C5): on the registered subsample the full continuation is re-run and
        compared with the warm-started solution from the stored `dq`
        (`|dq_warm - dq_cont| < tol_root`, both converged); a frame without a stored `dq`
        (its first epoch, or dropped since) is checked in the v4.2 form, continuation
        against the zero start. Batched by size; failing frames leave the store. The
        failing fraction is logged and compared with the registered ceiling."""
        if not self.cfg.coupling or not indices:
            return {"checked": 0, "failed": 0, "fraction": 0.0, "failed_frames": [], "warm": 0, "zero": 0}
        self.model.eval()
        tol_root = self.model.scf_options.tol_root
        failed, n_warm, n_zero = [], 0, 0
        by_size: Dict[int, List[int]] = {}
        for i in indices:
            by_size.setdefault(len(self.frames[i]), []).append(int(i))
        for group in by_size.values():
            for start in range(0, len(group), self.cfg.batch_size):
                chunk = group[start:start + self.cfg.batch_size]
                batch = to_device(next(iter(torch_geometric.dataloader.DataLoader(self._dataset(chunk), batch_size=len(chunk)))), self.device)
                ptr = batch["ptr"]
                # The two solves run one after the other with the first's output released
                # before the second: holding both batched outputs doubled the process's GPU
                # peak (four 159-atom frames: 3.6 -> 7 GB) and OOM'd four runs per GPU.
                with torch.no_grad():
                    cont = self.model(batch, compute_force=False)                 # the full continuation
                    dq_c = cont["dq"].detach().cpu()
                    conv_c = list(cont["diagnostics"]["converged"])
                    del cont
                    stored = [self.dq_store.get(i) for i in chunk]
                    alt_start = [torch.zeros(int(ptr[g + 1] - ptr[g]), dtype=torch.float64) if w is None else w
                                 for g, w in enumerate(stored)]
                    alt = self.model(batch, compute_force=False, warm_start=[w.to(self.device) for w in alt_start])
                    dq_a = alt["dq"].detach().cpu()
                    conv_a = list(alt["diagnostics"]["converged"])
                    del alt
                for g, i in enumerate(chunk):
                    lo, hi = int(ptr[g]), int(ptr[g + 1])
                    if stored[g] is None:
                        n_zero += 1
                    else:
                        n_warm += 1
                    ok = (conv_c[g] is True and conv_a[g] is True
                          and float((dq_a[lo:hi] - dq_c[lo:hi]).abs().max()) < tol_root)
                    if ok:
                        self.dq_store[i] = dq_c[lo:hi].clone()
                    else:
                        failed.append(i)
                        self.dq_store.pop(i, None)
        self._release_pool()
        return {"checked": len(indices), "failed": len(failed), "fraction": len(failed) / len(indices),
                "failed_frames": failed, "warm": n_warm, "zero": n_zero}

    def _release_pool(self) -> None:
        """Return the inference passes' cached GPU blocks: their shapes differ from the
        training step's, and a pool fragmented by them made the step reserve ~0.5 GB more
        per process (four runs per 24 GB GPU then OOM'd)."""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def single_valuedness(self, indices: Sequence[int]) -> Dict[str, object]:
        """Pre-v4.5 name of the check (kept for callers)."""
        return self.warm_check(indices)

    # ------------------------------------------------------------ loop

    def fit(self) -> Dict[str, object]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # A1: collect the tanh outputs of every bounded term for the weak L2. On for training
        # only -- `evaluate` reads saturation directly and does not need the accumulator.
        self.model.h0.sk.collect_regularisation(bool(cfg.l2_weight))
        ds = self._dataset(self.train_idx)
        sizes = [len(self.frames[i]) for i in self.train_idx]
        sampler = dd.SizeGroupedSampler(sizes, cfg.batch_size, seed=cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        history = []
        # W0.4 (v5): the evaluation model is the uniform parameter average over the last
        # `avg_window` epochs. Every epoch is accepted in this trainer (there is no rollback),
        # so the window is simply the final `avg_window` epochs; the average is accumulated in
        # float64 over the TRAINABLE parameters only, and the last-epoch model is read too.
        avg_window = max(0, min(int(cfg.avg_window), cfg.epochs))
        avg_sum: Optional[Dict[str, torch.Tensor]] = None
        avg_n = 0
        for epoch in range(cfg.epochs):
            sampler.set_epoch(epoch)
            t0 = time.time()
            stats = {"loss": 0.0, "force": 0.0, "gap": 0.0, "l2": 0.0, "unconverged": 0, "n": 0, "scf_iterations": 0, "scf_fills": 0, "warm": 0}
            # C5 (v4.2 / v4.5): the registered subsample at the start of the epoch -- the
            # warm start against the full continuation (zero start against the continuation
            # for frames without a stored dq, i.e. every frame in epoch 0); frames that fail
            # are dropped from this epoch's steps and from the store, the fraction logged.
            dropped = set()
            sv_pre = None
            if cfg.coupling:
                size = max(1, int(round(cfg.warm_check_fraction * len(self.train_idx))))
                sample = rng.choice(self.train_idx, size=min(size, len(self.train_idx)), replace=False).tolist()
                sv_pre = self.warm_check(sample)
                dropped = set(sv_pre.get("failed_frames", []))
            for local_batch in sampler:
                local_batch = [j for j in local_batch if self.train_idx[j] not in dropped]
                if not local_batch:
                    continue
                batch = to_device(torch_geometric.dataloader.Batch.from_data_list([ds[j] for j in local_batch]), self.device)
                frame_indices = [self.train_idx[j] for j in local_batch]
                s = self.step(batch, frame_indices, optimizer)
                for key in ("loss", "force", "l2", "unconverged", "scf_iterations", "scf_fills", "warm"):
                    stats[key] += s[key]
                stats["gap"] = s["gap"]; stats["n"] += 1
            sv = sv_pre
            entry = {"epoch": epoch, "loss": stats["loss"] / max(stats["n"], 1), "force": stats["force"] / max(stats["n"], 1),
                     "l2": stats["l2"] / max(stats["n"], 1), "gap": stats["gap"], "unconverged": stats["unconverged"], "single_valued": sv,
                     "scf_iterations_per_batch": stats["scf_iterations"] / max(stats["n"], 1),
                     "scf_fills_per_batch": stats["scf_fills"] / max(stats["n"], 1),
                     "warm_batches": stats["warm"], "batches": stats["n"],
                     "lambda_dir": float(self.model.lambda_dir()), "u_eff": self.model.u_eff().detach().cpu().tolist(),
                     "s": float(self.model.pattern_scale()), "time": time.time() - t0}
            # The ceiling applies to the trained model (v4.2 section 5: the root rule and
            # convergence gates apply to trained models; the initialised-model outcome is a
            # diagnostic). Epoch 0's check is that diagnostic -- the zero start against the
            # continuation at the initial coupling -- and is recorded, not counted.
            over = bool(sv is not None and sv["fraction"] > cfg.single_valued_ceiling)
            if over and epoch >= 1:
                self.sv_ceiling_exceeded_epochs.append(epoch)
            entry["sv_ceiling_exceeded"] = over and epoch >= 1
            entry["sv_init_diagnostic"] = epoch == 0
            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                entry["held"] = {k: v for k, v in self.evaluate(self.held_idx, "held").items() if k != "energies"}
            history.append(entry)
            logging.info("epoch %d loss %.4e force %.4e l2 %.2e gap %.3f unconv %d sv %s scf/batch %.1f fills/batch %.1f warm %d/%d lambda %.3f s %.3f held %s (%.0fs)",
                         epoch, entry["loss"], entry["force"], entry["l2"], entry["gap"], entry["unconverged"], sv,
                         entry["scf_iterations_per_batch"], entry["scf_fills_per_batch"], entry["warm_batches"], entry["batches"],
                         entry["lambda_dir"], entry["s"], entry.get("held", {}).get("force_rmse"), entry["time"])
            json.dump(history, open(self.run_dir / "history.json", "w"), indent=1, default=str)
            torch.save(self.model, self.run_dir / "model.pt")
            if avg_window and epoch >= cfg.epochs - avg_window:
                avg_sum = average_into(avg_sum, self.model); avg_n += 1
            if sv is not None and sv["fraction"] > cfg.single_valued_ceiling:
                logging.warning("single-valuedness failing fraction %.3f exceeds the ceiling %.3f", sv["fraction"], cfg.single_valued_ceiling)
        final = self.evaluate(self.held_idx, "held_final")
        final["saturation"] = self.saturation_report(self.held_idx)      # A1 bounds report
        # v4.5: "failure fails the arm" -- the run-level flag the report reads.
        final["sv_ceiling_exceeded_epochs"] = list(self.sv_ceiling_exceeded_epochs)
        final["arm_failed_single_valuedness"] = bool(self.sv_ceiling_exceeded_epochs)
        json.dump(final, open(self.run_dir / "held_final.json", "w"), indent=1, default=str)
        torch.save(self.model, self.run_dir / "model.pt")
        out = {"history": history, "held_final": {k: v for k, v in final.items() if k != "energies"}}
        # The averaged model is evaluated AFTER the last-epoch reading, so the warm-start store
        # the last-epoch numbers were produced with is not overwritten before they are taken.
        if avg_sum is not None and avg_n > 0:
            saved = load_average(self.model, avg_sum, avg_n)
            avg = self.evaluate(self.held_idx, "held_final_avg")
            avg["saturation"] = self.saturation_report(self.held_idx)
            avg["avg_epochs"] = avg_n
            json.dump(avg, open(self.run_dir / "held_final_avg.json", "w"), indent=1, default=str)
            torch.save(self.model, self.run_dir / "model_avg.pt")
            out["held_final_avg"] = {k: v for k, v in avg.items() if k != "energies"}
            restore_parameters(self.model, saved)
        return out
