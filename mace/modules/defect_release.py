"""Two-timescale base release, with a guard that reverts it if localisation degrades.

E1 measured both ends of this trade. A hard-frozen base costs 20-33 meV/A on forces, because
it cannot adapt to charged geometries at all. A jointly trained base launders the site signal
away from the correction, which is the mechanism (M1) that made the free-attention arms
uninformative in the first place.

So the base is held fixed while attention settles -- passing seeds settle by epoch ~30 -- and
then released at a hundredth of the head's learning rate. That is slow enough to adapt and,
in principle, too slow to re-launder.

"In principle" is not evidence, so the release is guarded: the hub mass is recorded at the
release epoch and checked afterwards, and if it falls by more than a tolerance the base is
rolled back to its release-epoch weights and frozen for the rest of the run. That converts a
silent failure -- localisation quietly undone after the screen's decisive epochs -- into a
logged, reversible event.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

__all__ = ["BaseRelease"]


class BaseRelease:
    """Holds the base frozen, releases it slowly, and reverts it if the carrier drifts."""

    def __init__(self, model, optimizer, release_epoch: int, release_factor: float,
                 tolerance: float = 0.05, base_groups=None):
        self.model = model
        self.optimizer = optimizer
        self.release_epoch = int(release_epoch)
        self.release_factor = float(release_factor)
        self.tolerance = float(tolerance)
        self.base_groups = base_groups or {
            "embedding", "interactions_decay", "interactions_no_decay",
            "products", "readouts",
        }
        self.reference_mass: Optional[float] = None
        self.snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.released = False
        self.reverted = False

    def _base_state(self):
        from mace.modules.defect_stage import is_correction_param, is_bookkeeping
        return {n: p for n, p in self.model.named_parameters()
                if not is_correction_param(n) and not is_bookkeeping(n)}

    def _set_base_lr(self, factor: float, base_lr: float) -> None:
        for group in self.optimizer.param_groups:
            if group.get("name") in self.base_groups:
                group["lr"] = base_lr * factor

    def on_epoch_start(self, epoch: int, base_lr: float, hub_mass: Optional[float]) -> None:
        """Release at the scheduled epoch, recording the localisation to protect."""
        if self.released or epoch < self.release_epoch:
            return
        self.released = True
        self.reference_mass = hub_mass
        self.snapshot = {n: p.detach().clone() for n, p in self._base_state().items()}
        self._set_base_lr(self.release_factor, base_lr)
        logging.info(
            f"Base released at epoch {epoch} with lr factor {self.release_factor}; "
            f"hub mass at release {hub_mass if hub_mass is None else round(hub_mass, 4)}")

    def on_epoch_end(self, epoch: int, hub_mass: Optional[float]) -> None:
        """Roll the base back if releasing it cost more localisation than allowed."""
        if (not self.released or self.reverted or self.snapshot is None
                or hub_mass is None or self.reference_mass is None):
            return
        drop = self.reference_mass - hub_mass
        if drop <= self.tolerance:
            return
        with torch.no_grad():
            for name, param in self._base_state().items():
                if name in self.snapshot:
                    param.copy_(self.snapshot[name])
        self._set_base_lr(0.0, 1.0)
        self.reverted = True
        logging.warning(
            f"Base reverted at epoch {epoch}: hub mass fell {drop:.4f} "
            f"({self.reference_mass:.4f} -> {hub_mass:.4f}), beyond the {self.tolerance} "
            "tolerance. Base is frozen at its release-epoch weights for the rest of the run.")
