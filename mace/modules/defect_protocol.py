"""The Stage-3 training protocol, in the package rather than in a harness script.

WHY THIS FILE EXISTS. Every Stage-3 result in this programme was produced by
`defect-perovskite/stage_run.py`, which carries five things the production trainer does not:
the Harrison initialisation, the c-shift calibration, the linear warmup, `loss_gap`, and the
initialisation gate. Section 3 established that the two paths build a bit-identical MODEL --
but a model is not a run, and "the joint run comes from config" is empty while the protocol
that makes a Stage-3 run work lives in a script the trainer cannot see.

Each piece is a function taking a model and returning a number or a tensor, with no harness
state and no file I/O, so `stage_run.py` and `mace.cli.run_train` can call the same code and
cannot drift. What remains for the trainer is the call sites, not the logic.

WHAT IS DELIBERATELY NOT HERE: the data selection, the seed handling and the evaluation.
Those are experiment design, they differ between a stage screen and a joint run, and folding
them in is how a "protocol" becomes a second trainer.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch


def calibrate_c_shift(out, energy_target, counts) -> Optional[float]:
    """The uniform on-site offset `c` that removes the median energy mismatch, or None.

    `E_head` moves by `c * Delta_n` and by nothing else, so ONE scalar solves the median
    mismatch exactly -- this is `mu_c`'s old initialisation role without `mu_c`'s per-channel
    bookkeeping. Solved on the initialisation batch and then trained.

    The MEDIAN, not the mean: a single frame whose base energy is badly extrapolated would
    otherwise set the offset for the whole run.

    Returns None when no frame in the batch carries a net carrier, because `Delta_n = 0`
    makes the ratio undefined -- and silently returning 0.0 there would look like a
    calibration that had happened.
    """
    if energy_target is None:
        return None
    counts = counts.reshape(counts.shape[0], -1)
    delta_n = (counts[:, 0] + counts[:, 1] - counts[:, 2] - counts[:, 3]).to(
        out["delta_sr_energy"].dtype)
    sel = delta_n.abs() > 0
    if not bool(sel.any()):
        return None
    resid = energy_target - out["base_energy"] - out["delta_sr_energy"]
    return float(torch.median(resid[sel] / delta_n[sel]))


def warmup_factor(epoch: int, warmup: int = 5) -> float:
    """Linear warmup, as a multiplier on the target learning rate.

    The counting head's correction is eV-scale at step 0, so the first few steps see gradients
    three orders larger than the converged ones. Going in at full rate is how a seed gets
    thrown somewhere it cannot return from -- observed, not supposed.
    """
    if warmup <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / float(warmup))


def loss_gap(model, forward_dict, e_gap: float, w_gap: float = 1.0):
    """`w * (gap_pristine - E_gap)^2` on one pristine draw. A SPECTRUM constraint.

    Not an energy label: the frontier gap of a defect-free cell must be the host band gap,
    which is a property of the material and carries none of M1b's base-extrapolation slope.
    One pristine draw per step, ensemble mean over its graphs.
    """
    out = model(forward_dict, training=True, compute_force=False)
    residual = out["logit_gap"][:, 0].mean() - float(e_gap)
    return float(w_gap) * residual ** 2


def initialisation_report(lam, n_electrons: float, e_gap: float,
                          t_el: Optional[float] = None) -> Dict[str, float]:
    """The step-0 gate on a PRISTINE spectrum: bands, not atoms. See `initialisation_gate`.

    Re-exported here so the trainer imports the protocol rather than reaching into the head's
    module for one function -- which is how a caller ends up with a different default.
    """
    from mace.modules.defect_counting import initialisation_gate, smearing

    return initialisation_gate(lam, n_electrons, e_gap,
                               t_el=smearing()[1] if t_el is None else t_el)


def trainable_mask(name: str, freeze_z: bool = False) -> bool:
    """Which parameters a head-only stage trains. One predicate, so train and evaluate agree.

    The correction parameters, everything under `.spectral.`, and the Madelung charges --
    with `freeze_z` pinning the last group for the diagnostic arm that asks whether the
    learned scale of Z buys anything.
    """
    from mace.modules.defect_stage import is_correction_param

    if freeze_z and name.startswith("madelung."):
        return False
    return bool(is_correction_param(name) or ".spectral." in name
                or name.startswith("madelung."))


def post_step(model) -> None:
    """The after-every-optimiser-step hook. Currently the Z neutrality projection.

    A function rather than a line in each training loop: `Z` lives on the pristine
    composition hyperplane, and without the projection the species charges drift as a group,
    which is a gauge on `phi` and a slow divergence. A loop that forgot it would train a
    slightly different model with no error anywhere.
    """
    madelung = getattr(model, "madelung", None)
    if madelung is not None:
        madelung.project_()


def apply_harrison(model, atomic_numbers: Sequence[int], bond_length: float) -> None:
    """Harrison term values and universal hoppings, at the MEASURED bond length.

    eps0 = 0 makes every site degenerate -- the atomic limit, where the bond order vanishes
    and, on the frozen-P gradient, nothing could move the hoppings at all. The bond length is
    passed in rather than baked in so the head stays host-agnostic.
    """
    from mace.modules.defect_counting import harrison_initialise

    harrison_initialise(model.spectral, [int(z) for z in atomic_numbers],
                        bond_length=float(bond_length))


def protocol_summary(stage: int, e_gap: float, w_gap: float, warmup: int,
                     clip: float, freeze_z: bool, model=None,
                     c_shift: Optional[float] = None) -> Dict[str, object]:
    """What a run will actually do, as a dict to be logged and saved beside the results.

    A run whose artefact does not record its own protocol is a run whose numbers cannot be
    compared to anything later, which this programme has paid for four times.
    """
    from mace.modules.defect_counting import smearing

    # The HEAD's own setting when a model is given. The module-level default is only what the
    # process starts with, and reporting it while the head runs at a different width is
    # exactly the mismatch that trained six seeds at Gaussian 0.025 under a 0.05 banner.
    family, width = smearing()
    if model is not None and getattr(model, "spectral", None) is not None:
        family = str(getattr(model.spectral, "smearing_family", family))
        width = float(getattr(model.spectral, "t_el", width))
    # `c_shift_calibrated` reports the OUTCOME, not the intent. It used to be `stage >= 3`,
    # so a run whose initialisation batch happened to carry no net carrier -- calibration
    # skipped, warning logged, head starting at c = 0 -- still recorded "calibrated: true" in
    # its own artefact. Caught by the first end-to-end trainer run. Exactly the class of
    # mismatch this summary exists to prevent, one level up from the smearing width.
    return dict(stage=int(stage), e_gap=float(e_gap), w_gap=float(w_gap),
                warmup=int(warmup), grad_clip=float(clip), freeze_z=bool(freeze_z),
                smearing_family=family, smearing_width=float(width),
                harrison_init=stage >= 3,
                c_shift_calibrated=c_shift is not None,
                c_shift=None if c_shift is None else float(c_shift))
