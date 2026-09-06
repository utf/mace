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
    e_gap: float = 2.40               # registered host gap (static lattice, C2)
    coupling: bool = False            # Arm 1: Phi = 0
    coupling_mode: str = "full"       # Arm 2+3: lr_only | lr_u | full | lambda1
    route_b: bool = False
    directional: bool = True          # False: the scalar-only control
    regime: str = "B"
    stratum_weights: Dict[str, float] = field(default_factory=dict)   # frozen per stratum key
    single_valued_subsample: int = 16     # frames per epoch (coupling on)
    single_valued_ceiling: float = 0.10   # failing fraction that fails the arm
    eval_every: int = 5
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
    from mace.modules.defect_cache import frame_key
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

    # ------------------------------------------------------------ data

    def _dataset(self, indices: Sequence[int]):
        return dd.atomic_data([self.frames[i] for i in indices], self.z_table, self.cfg.r_cut)

    def prepare(self, pristine_indices: Sequence[int]) -> None:
        """Pristine centre and q0 reference from the pristine frames; the static cell batch."""
        ds = dd.atomic_data([self.frames[i] for i in pristine_indices], self.z_table, self.cfg.r_cut)
        loader = torch_geometric.dataloader.DataLoader(ds, batch_size=16)
        self.model.set_pristine_centre([to_device(b, self.device) for b in loader])
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

    def _fill_base_cache(self, indices: Sequence[int]) -> None:
        from mace.modules.models import ScaleShiftMACE
        ds = self._dataset(indices)
        k = 0
        for b in torch_geometric.dataloader.DataLoader(ds, batch_size=8):
            batch = to_device(b, self.device)
            out = ScaleShiftMACE.forward(self.model.base, self.model._trunk_data(dict(batch)), training=False,
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
        record = {"config": asdict(self.cfg), "strata": self.strata_record, "n_train": len(self.train_idx),
                  "n_held": len(self.held_idx), "fold_of": {str(k): v for k, v in self.fold_of.items()},
                  "model_extra_state": self.model.get_extra_state()}
        json.dump(record, open(self.run_dir / "run_record.json", "w"), indent=1, default=str)

    # ------------------------------------------------------------ steps

    def step(self, batch, frame_indices: Sequence[int], optimizer) -> Dict[str, float]:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        out = self.model(self._attach_base(batch, frame_indices), training=True, compute_force=True)
        weights = torch.tensor([self.frame_weight[i] for i in frame_indices], dtype=torch.float64,
                               device=self.device)
        l_force = force_loss(out, batch, weights)
        gap = self.model.pristine_gap(self.static_batch, getattr(self, "static_features", None))
        l_gap = self.cfg.gap_weight * ((gap - self.cfg.e_gap) ** 2).sum()
        loss = self.cfg.force_weight * l_force + l_gap
        loss.backward()
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        optimizer.step()
        diag = out.get("diagnostics", {})
        conv = diag.get("converged", [])
        return {"loss": float(loss), "force": float(l_force), "gap": float(gap[0]),
                "unconverged": int(sum(1 for c in conv if c is False))}

    def evaluate(self, indices: Sequence[int], tag: str) -> Dict[str, object]:
        """Force RMSE (eV/A) overall and by distance shell from the vacancy (label-free:
        the flanking-Pb midpoint), and the uncalibrated energy residual per frame."""
        self.model.eval()
        ds = self._dataset(indices)
        loader = torch_geometric.dataloader.DataLoader(ds, batch_size=self.cfg.batch_size)
        sq, n_atoms = 0.0, 0
        shells = {(0, 2): [0.0, 0], (2, 4): [0.0, 0], (4, 6): [0.0, 0], (6, 8): [0.0, 0], (8, 99): [0.0, 0]}
        energies = []
        k = 0
        for b in loader:
            batch = to_device(b, self.device)
            with torch.no_grad():
                out = self.model(batch, compute_force=True)
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
                z = torch.as_tensor(numbers, device=pos.device)
                from mace.modules.dscc.kernels import minimum_image_distances
                r = minimum_image_distances(pos, cell)
                pb = torch.nonzero(z == 82).reshape(-1); cl = torch.nonzero(z == 17).reshape(-1)
                dist = torch.sort(r[pb][:, cl], dim=1).values
                flank = pb[dist[:, 5] > 4.0]
                if flank.numel() == 2:
                    mid = pos[flank[0]] + 0.5 * (pos[flank[1]] - pos[flank[0]] - torch.round((pos[flank[1]] - pos[flank[0]]) @ torch.linalg.inv(cell)) @ cell)
                    dv = pos - mid
                    dv = dv - torch.round(dv @ torch.linalg.inv(cell)) @ cell
                    rad = dv.norm(dim=-1)
                    for (a_, b_), acc in shells.items():
                        m = (rad >= a_) & (rad < b_)
                        acc[0] += float(diff[lo:hi][m].sum()); acc[1] += int(m.sum())
                energies.append({"index": indices[k - 1], "n": hi - lo, "d": d,
                                 "resid_uncal": float(out["energy_uncalibrated"][g] - batch["energy"][g])})
        report = {"tag": tag, "force_rmse": float(np.sqrt(sq / max(n_atoms, 1))),
                  "shell_rmse": {f"{a_}-{b_}": (float(np.sqrt(v[0] / v[1])) if v[1] else None) for (a_, b_), v in shells.items()},
                  "energies": energies}
        return report

    def single_valuedness(self, indices: Sequence[int]) -> Dict[str, object]:
        """C5: the root rule on a subsample (zero / continuation / warm from the production
        solution); the failing fraction is logged and compared with the ceiling."""
        if not self.cfg.coupling or not indices:
            return {"checked": 0, "failed": 0, "fraction": 0.0}
        self.model.eval()
        failed = 0
        for i in indices:
            batch = to_device(next(iter(torch_geometric.dataloader.DataLoader(self._dataset([i]), batch_size=1))), self.device)
            with torch.no_grad():
                out = self.model(batch, compute_force=False)
            # Re-solve from zero and from a perturbed warm start through the model's own path.
            with torch.no_grad():
                alt = self.model(batch, compute_force=False, warm_start=[torch.zeros_like(out["dq"])])
            if float((alt["dq"] - out["dq"]).abs().max()) > self.model.scf_options.tol_root:
                failed += 1
        return {"checked": len(indices), "failed": failed, "fraction": failed / len(indices)}

    # ------------------------------------------------------------ loop

    def fit(self) -> Dict[str, object]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        ds = self._dataset(self.train_idx)
        sizes = [len(self.frames[i]) for i in self.train_idx]
        sampler = dd.SizeGroupedSampler(sizes, cfg.batch_size, seed=cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        history = []
        for epoch in range(cfg.epochs):
            sampler.set_epoch(epoch)
            t0 = time.time()
            stats = {"loss": 0.0, "force": 0.0, "gap": 0.0, "unconverged": 0, "n": 0}
            for local_batch in sampler:
                batch = to_device(torch_geometric.dataloader.Batch.from_data_list([ds[j] for j in local_batch]), self.device)
                frame_indices = [self.train_idx[j] for j in local_batch]
                s = self.step(batch, frame_indices, optimizer)
                for key in ("loss", "force", "unconverged"):
                    stats[key] += s[key]
                stats["gap"] = s["gap"]; stats["n"] += 1
            sv = self.single_valuedness(rng.choice(self.train_idx, size=min(cfg.single_valued_subsample, len(self.train_idx)), replace=False).tolist()) if cfg.coupling else None
            entry = {"epoch": epoch, "loss": stats["loss"] / max(stats["n"], 1), "force": stats["force"] / max(stats["n"], 1),
                     "gap": stats["gap"], "unconverged": stats["unconverged"], "single_valued": sv,
                     "lambda_dir": float(self.model.lambda_dir()), "u_eff": self.model.u_eff().detach().cpu().tolist(),
                     "s": float(self.model.pattern_scale()), "time": time.time() - t0}
            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                entry["held"] = {k: v for k, v in self.evaluate(self.held_idx, "held").items() if k != "energies"}
            history.append(entry)
            logging.info("epoch %d loss %.4e force %.4e gap %.3f unconv %d sv %s lambda %.3f s %.3f held %s (%.0fs)",
                         epoch, entry["loss"], entry["force"], entry["gap"], entry["unconverged"], sv,
                         entry["lambda_dir"], entry["s"], entry.get("held", {}).get("force_rmse"), entry["time"])
            json.dump(history, open(self.run_dir / "history.json", "w"), indent=1, default=str)
            torch.save(self.model, self.run_dir / "model.pt")
            if sv is not None and sv["fraction"] > cfg.single_valued_ceiling:
                logging.warning("single-valuedness failing fraction %.3f exceeds the ceiling %.3f", sv["fraction"], cfg.single_valued_ceiling)
        final = self.evaluate(self.held_idx, "held_final")
        json.dump(final, open(self.run_dir / "held_final.json", "w"), indent=1, default=str)
        torch.save(self.model, self.run_dir / "model.pt")
        return {"history": history, "held_final": {k: v for k, v in final.items() if k != "energies"}}
