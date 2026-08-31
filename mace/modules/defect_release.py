"""Two-timescale base release, with a guard that reverts it if localisation degrades.

E1 measured both ends of this trade. A hard-frozen base costs 20-33 meV/A on forces, because
it cannot adapt to charged geometries at all. A jointly trained base launders the site signal
away from the correction, which is the mechanism (M1) that made the free-attention arms
uninformative in the first place.

So the base is held fixed while attention settles -- passing seeds settle by epoch ~30 -- and
then released at a hundredth of the head's learning rate. That is slow enough to adapt and,
in principle, too slow to re-launder.

"In principle" is not evidence, so the release is guarded: the carrier's state is recorded at
the release epoch and checked afterwards, and if it moves or spreads the base is rolled back to
its release-epoch weights and frozen for the rest of the run. That converts a silent failure --
localisation quietly undone after the screen's decisive epochs -- into a logged, reversible
event.

The trigger is deliberately LABEL-FREE: attention overlap against the release epoch, N_eff,
and the active/null channel ratio. Keying it on hub mass would put the vacancy assignment
into a training decision.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

__all__ = ["BaseRelease"]


class BaseRelease:
    """Holds the base frozen, releases it slowly, and reverts it if the carrier drifts."""

    def __init__(self, model, optimizer, release_epoch: int, release_factor: float,
                 overlap_min: float = 0.90, n_eff_rise: float = 0.50,
                 null_ratio_max: float = 0.80, base_groups=None):
        self.model = model
        self.optimizer = optimizer
        self.release_epoch = int(release_epoch)
        self.release_factor = float(release_factor)
        # Thresholds on label-free quantities only. Deliberately NOT hub mass: that needs the
        # vacancy assignment, and a guard that reads it is a defect label reaching a training
        # decision, however indirectly.
        self.overlap_min = float(overlap_min)
        self.n_eff_rise = float(n_eff_rise)
        self.null_ratio_max = float(null_ratio_max)
        self.base_groups = base_groups or {
            "embedding", "interactions_decay", "interactions_no_decay",
            "products", "readouts",
        }
        self.reference: Dict[str, float] = {}
        self.snapshot: Optional[Dict[str, torch.Tensor]] = None
        self.released = False
        self.reverted = False
        self.armed = False

    def _base_state(self):
        from mace.modules.defect_stage import is_correction_param, is_bookkeeping
        return {n: p for n, p in self.model.named_parameters()
                if not is_correction_param(n) and not is_bookkeeping(n)}

    def _set_base_lr(self, factor: float, base_lr: float) -> None:
        for group in self.optimizer.param_groups:
            if group.get("name") in self.base_groups:
                group["lr"] = base_lr * factor

    def on_epoch_start(self, epoch: int, base_lr: float,
                       state: Optional[Dict[str, float]] = None) -> None:
        """Release at the scheduled epoch, recording the state to protect.

        `state` is LABEL-FREE by design: alpha (the attention vector itself), n_eff, and the
        active/null channel ratio. An earlier version keyed the guard on hub mass, which needs
        the vacancy assignment -- weaker than a label in the loss, but still a defect label
        reaching a training decision. These quantities answer "did the attention move, and did
        it spread out?" without anyone telling the model where the defect is.
        """
        if self.released or epoch < self.release_epoch:
            return
        self.released = True
        self.reference = dict(state or {})
        self.snapshot = {n: p.detach().clone() for n, p in self._base_state().items()}
        self._set_base_lr(self.release_factor, base_lr)

        # Arm only if there is localisation to protect. If the carrier is still spread at the
        # release epoch -- null ratio already past its threshold -- then rolling the base back
        # later preserves nothing and merely freezes the base for the rest of the run. Better
        # to let it train and say so.
        ratio = self.reference.get("null_ratio")
        self.armed = not (ratio is not None and ratio > self.null_ratio_max)
        logging.info(
            f"Base released at epoch {epoch} with lr factor {self.release_factor}; "
            f"reference state {self.reference}; guard "
            f"{'armed' if self.armed else 'NOT armed (carrier not localised at release, '
               'so there is nothing for a rollback to protect)'}")

    def _trigger(self, state: Dict[str, float]) -> Optional[str]:
        """Which label-free condition, if any, says the carrier moved."""
        ref = self.reference
        overlap = state.get("alpha_overlap")
        if overlap is not None and overlap < self.overlap_min:
            return f"alpha_overlap {overlap:.3f} < {self.overlap_min}"
        n_eff, n_eff_ref = state.get("n_eff"), ref.get("n_eff")
        if (n_eff is not None and n_eff_ref not in (None, 0)
                and n_eff > n_eff_ref * (1.0 + self.n_eff_rise)):
            return (f"n_eff {n_eff:.2f} rose more than "
                    f"{self.n_eff_rise:.0%} above {n_eff_ref:.2f}")
        ratio = state.get("null_ratio")
        if ratio is not None and ratio > self.null_ratio_max:
            return f"active/null ratio {ratio:.3f} > {self.null_ratio_max}"
        return None

    def on_epoch_end(self, epoch: int,
                     state: Optional[Dict[str, float]] = None) -> None:
        """Roll the base back if releasing it cost the carrier its localisation."""
        # Never on the release epoch itself: start and end are called in the same hook, so
        # checking here would compare the reference against itself with zero epochs of drift
        # -- which fired an immediate rollback the first time this ran.
        if (not self.released or not self.armed or self.reverted
                or self.snapshot is None or not state or epoch <= self.release_epoch):
            return
        reason = self._trigger(state)
        if reason is None:
            return
        with torch.no_grad():
            for name, param in self._base_state().items():
                if name in self.snapshot:
                    param.copy_(self.snapshot[name])
        self._set_base_lr(0.0, 1.0)
        self.reverted = True
        logging.warning(
            f"Base reverted at epoch {epoch}: {reason}. Base is frozen at its "
            "release-epoch weights for the rest of the run.")
