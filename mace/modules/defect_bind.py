"""Delta_bind: how far the carrier level sits below where it sits with no defect present.

lambda_2 - lambda_1 was used as the binding indicator and is wrong. Measured on trained
heads it reads 2.1-2.7 eV while N_eff stays around 60 -- a large gap coexisting with a state
spread over three quarters of the cell. That gap is a property of a strongly connected graph
with all-negative off-diagonals, whose lowest eigenvector is the in-phase "superatom" mode:
the splitting is between that mode and the rest of the band, not between a bound level and a
band edge. It is large whether or not anything is bound.

The reference has to be the same Hamiltonian without the defect:

    Delta_bind = mean_over_pristine lambda_1(pristine cell, same counters)
                 - lambda_1(defect cell)

A split-off level gives a positive Delta_bind; a band state gives ~0, because its lambda_1 is
the band bottom and the pristine cell has the same band bottom. The superatom gap cancels: it
is present in both terms.

Evaluation only. The pristine frames are scored with the SAME counters as the defect frame,
which is off-distribution for them by construction -- a pristine cell has no carrier to hold.
That is intentional and is why this is a diagnostic and never a loss term: it asks "where
would this Hamiltonian put a carrier if there were no vacancy", which is exactly the reference
a binding energy needs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

__all__ = ["lambda1_of", "delta_bind"]


def lambda1_of(model, batch_dict, channel: Optional[int] = None) -> np.ndarray:
    """Lowest physical eigenvalue per graph, for the supervised channel.

    Padded slots sit at ~1e3 by construction and must be excluded, or a cell with fewer atoms
    than `num_states` reports the padding energy as its ground state.
    """
    grabbed: dict = {}
    head = model.spectral
    original = head.forward

    def wrapped(*args, **kwargs):
        kwargs["internals"] = grabbed
        return original(*args, **kwargs)

    head.forward = wrapped
    try:
        with torch.no_grad():
            model(batch_dict, training=False, compute_force=False)
    finally:
        head.forward = original
    if "lam" not in grabbed:
        raise RuntimeError("no spectral internals captured; not a spectral-head model")

    lam = grabbed["lam"]                                   # [G, C, m]
    if channel is None:
        counts = batch_dict["carrier_counts"].reshape(lam.shape[0], -1)
        channel = int(torch.argmax(counts.sum(dim=0)).item())
    lam_c = lam[:, channel, :]
    physical = lam_c < 500.0
    out = []
    for g in range(lam_c.shape[0]):
        vals = lam_c[g][physical[g]]
        out.append(float(vals.min()) if vals.numel() else float("nan"))
    return np.asarray(out)


def delta_bind(model, defect_batch, pristine_batches: Sequence,
               channel: Optional[int] = None) -> dict:
    """Delta_bind per defect graph, against the mean pristine lambda_1.

    `pristine_batches` are pre-built batches of pristine cells carrying the defect frames'
    counters. They are fixed across epochs so the reference does not wander: a moving
    reference would make Delta_bind track the pristine frames' sampling rather than the
    defect level.
    """
    ref = []
    for pb in pristine_batches:
        ref.append(lambda1_of(model, pb, channel))
    ref_all = np.concatenate(ref) if ref else np.array([np.nan])
    ref_mean = float(np.nanmean(ref_all))

    lam_defect = lambda1_of(model, defect_batch, channel)
    delta = ref_mean - lam_defect
    return dict(delta_bind_mean=float(np.nanmean(delta)),
                delta_bind_median=float(np.nanmedian(delta)),
                lambda1_pristine=ref_mean,
                lambda1_defect=float(np.nanmean(lam_defect)),
                n_pristine=int(np.isfinite(ref_all).sum()),
                n_defect=int(np.isfinite(lam_defect).sum()))
