"""One forward context, built once, consumed identically by train / evaluate / capture.

This exists because the same bug has now been found four times, in four disguises:

1. R1, R2 and Test 1 were voided by a graph built at ``r_max`` while the head was configured
   for ``spectral_r_cut`` -- the hub pair had an edge in 0 of 40 frames.
2. The step-2 controls referenced an arch model that existed only on one machine.
3. ``train`` applied the clamp mask; ``evaluate`` and ``capture`` did not, so a two-atom clamp
   reported N_eff = 35. Same model, three answers.
4. Editing a head module while a sweep had it imported gave ``AttributeError`` on pickled
   models -- the run and the analysis disagreed about what the class was.

Every one is the same shape: **two code paths that must agree about the forward pass, each
constructing their own version of it.** The fix is not another assertion. It is to make the
forward pass constructible in exactly one place, so that "the evaluation ran a different model
than the training" stops being expressible.

Use it like this::

    ctx = ForwardContext.production(model)            # refuses masks, see below
    d = ctx.forward_dict(batch, requires_grad=True)
    out = model(d, training=True, compute_force=True)

and identically in ``evaluate`` and ``capture``. The clamp lives on the context, not on the
call site, so a path that forgets it cannot exist -- there is nothing to forget.

PRODUCTION REFUSES MASKS. ``clamp_mask`` and ``probe_loss_mask`` are the two places the
vacancy assignment touches training code (hard rule 1), and production configs must refuse
both. That refusal is enforced at construction here rather than by convention:
``ForwardContext.production`` raises if handed either, and ``ForwardContext.diagnostic`` is
the only constructor that accepts them. A diagnostic context carries ``is_diagnostic = True``
so a result written from one can be labelled as such.

EPSILON-INFINITY IS THREADED, NOT READ FROM THE MODEL. ``model.eps_inf_init`` is 6.5 on the
retained arch and base checkpoints while both launchers default to 4.0 -- inert there because
those models have ``use_long_range = False``, but not inert once the Madelung on-site term
divides by it. One constant, carried here, used by both the Madelung screen and E_LR's
amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import torch

__all__ = ["ForwardContext", "EPS_INF_DEFAULT"]

# High-frequency dielectric constant of CsPbCl3. The ONE place it is written down.
# Not the static constant: `a = 1/sqrt(eps_inf)` screens the carrier's own field, which the
# lattice has not had time to respond to. Nothing in this repository reads a static value.
# STANDING RULE 1 (speed cycle): 4.0 is CsPbCl3's high-frequency dielectric constant, and a
# per-host constant may not live in a default. It survives here as the LAST resort for a
# model that carries no Madelung term to read it off -- `_from_model` prefers the model's own
# `madelung_eps_inf`, which is the value the model was trained with, so a scorer run against
# a different host uses that host's number rather than this one. Passing `eps_inf` explicitly
# still wins over both.
EPS_INF_DEFAULT = 4.0


@dataclass(frozen=True)
class ForwardContext:
    """Graph spec, masks and convention flags for one run.

    Construct with :meth:`production` or :meth:`diagnostic`; the bare constructor is not part
    of the interface.
    """

    # ---- graph spec: what the neighbour list must reach ----------------------------------
    cutoff: float
    r_max: float
    r_couple: float

    # ---- conventions ---------------------------------------------------------------------
    eps_inf: float = EPS_INF_DEFAULT

    # ---- masks: diagnostic only ----------------------------------------------------------
    clamp: Optional[str] = None
    probe_loss_mask: Optional[str] = None
    frame_masks: Optional[Dict[int, Any]] = None
    stack_fn: Optional[Callable[[Sequence[Any], str, Any], torch.Tensor]] = None

    device: Any = "cpu"
    is_diagnostic: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ constructors

    @classmethod
    def production(cls, model, *, device="cpu", cutoff: Optional[float] = None,
                   eps_inf: Optional[float] = None, clamp=None,
                   probe_loss_mask=None,
                   **extra) -> "ForwardContext":
        """The only context a training run may use. Refuses both defect-derived masks."""
        if clamp is not None or probe_loss_mask is not None:
            raise ValueError(
                "production forward context refuses clamp_mask and probe_loss_mask: they are "
                "the two places the vacancy assignment touches training code (hard rule 1). "
                "Use ForwardContext.diagnostic() and label the result as diagnostic.")
        return cls._from_model(model, device=device, cutoff=cutoff, eps_inf=eps_inf,
                               is_diagnostic=False, extra=extra)

    @classmethod
    def diagnostic(cls, model, *, device="cpu", cutoff: Optional[float] = None,
                   eps_inf: Optional[float] = None, clamp: Optional[str] = None,
                   probe_loss_mask: Optional[str] = None,
                   frame_masks: Optional[Dict[int, Any]] = None,
                   stack_fn: Optional[Callable] = None, **extra) -> "ForwardContext":
        """A context that may carry masks. Everything built from it is a diagnostic."""
        if clamp is not None and (frame_masks is None or stack_fn is None):
            raise ValueError(
                "a clamp needs frame_masks and stack_fn to build the per-batch mask; without "
                "them the clamp would silently be dropped, which is exactly the failure this "
                "object exists to prevent")
        return cls._from_model(model, device=device, cutoff=cutoff, eps_inf=eps_inf,
                               is_diagnostic=True, extra=extra, clamp=clamp,
                               probe_loss_mask=probe_loss_mask, frame_masks=frame_masks,
                               stack_fn=stack_fn)

    @classmethod
    def _from_model(cls, model, *, device, cutoff, eps_inf, is_diagnostic, extra,
                    **masks) -> "ForwardContext":
        r_max = float(model.r_max)
        r_couple = float(getattr(model, "spectral_r_cut", 0.0) or 0.0)
        resolved = float(cutoff) if cutoff else max(r_max, r_couple)
        # Standing rule 1. `eps_inf` is a per-host INPUT, so it comes from the model that
        # was trained with it, not from a module constant. An explicit argument still wins
        # -- a sensitivity sweep is entitled to ask what a different value would do -- and
        # the constant is the last resort for a model with no Madelung term at all.
        if eps_inf is None:
            eps_inf = getattr(model, "madelung_eps_inf", None)
        if eps_inf is None or float(eps_inf) <= 0.0:
            eps_inf = EPS_INF_DEFAULT
        return cls(cutoff=resolved, r_max=r_max, r_couple=r_couple or resolved,
                   eps_inf=float(eps_inf), device=device, is_diagnostic=is_diagnostic,
                   extra=dict(extra), **masks)

    # ------------------------------------------------------------------ the forward pass

    def clamp_mask_for(self, frames) -> Optional[torch.Tensor]:
        """The clamp mask for these frames, or None. The single place it is assembled."""
        if self.clamp is None:
            return None
        if frames is None:
            raise ValueError(
                f"context carries clamp={self.clamp!r} but the caller passed no frames, so "
                "the mask cannot be built. A clamped context forwarding unclamped is the "
                "N_eff-35 bug.")
        per_frame = [self.frame_masks[id(f)] for f in frames]
        return self.stack_fn(per_frame, self.clamp, self.device)

    def forward_dict(self, batch, frames=None, *, requires_grad: bool = False) -> dict:
        """The model input dict. Every path builds it here or not at all.

        ``requires_grad`` controls only whether positions carry a graph for forces; it is a
        property of what the caller wants to differentiate, not of the model configuration,
        so it stays an argument rather than moving onto the context.
        """
        d = batch.to_dict() if hasattr(batch, "to_dict") else dict(batch)
        if requires_grad:
            pos = d["positions"]
            d["positions"] = pos.detach().clone().requires_grad_(True)
        d["_clamp_mask"] = self.clamp_mask_for(frames)
        if self.probe_loss_mask is not None:
            d["_probe_loss_mask"] = self.probe_loss_mask
        return d

    # ------------------------------------------------------------------ in-loop assertions

    def assert_reach(self, batch, strict: bool = True) -> Dict[str, float]:
        """The in-loop reach assertion, required on every run."""
        from mace.modules.defect_reach import assert_carrier_reach
        return assert_carrier_reach(batch, self.r_max, self.cutoff, strict=strict)

    def assert_coupling(self, batch, strict: bool = True) -> Dict[str, float]:
        """The label-free flanking-pair coupling test."""
        from mace.modules.defect_reach import assert_coupling_envelope
        return assert_coupling_envelope(batch, self.r_couple, strict=strict)

    # ------------------------------------------------------------------ provenance

    def as_dict(self) -> Dict[str, Any]:
        """Recorded in every result JSON, so a number carries the context that produced it."""
        return dict(cutoff=self.cutoff, r_max=self.r_max, r_couple=self.r_couple,
                    eps_inf=self.eps_inf, clamp=self.clamp,
                    probe_loss_mask=self.probe_loss_mask,
                    is_diagnostic=self.is_diagnostic, **self.extra)
