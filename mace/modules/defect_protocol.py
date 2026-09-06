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


def c_shift_terms(out, energy_target, counts):
    """The per-frame ratios `(E_target - E_base - E_head) / Delta_n`, charged frames only.

    Split out from `calibrate_c_shift` so that a calibration over MANY batches takes one
    median over all the terms rather than a median of per-batch medians, which is a different
    and shuffle-dependent statistic.
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
    return (resid[sel] / delta_n[sel]).detach().reshape(-1)


def calibrate_c_shift(out, energy_target, counts) -> Optional[float]:
    """The uniform on-site offset `c` that removes the median energy mismatch, or None.

    `E_head` moves by `c * Delta_n` and by nothing else, so ONE scalar solves the median
    mismatch exactly -- this is `mu_c`'s old initialisation role without `mu_c`'s per-channel
    bookkeeping.

    The MEDIAN, not the mean: a single frame whose base energy is badly extrapolated would
    otherwise set the offset for the whole run.

    Returns None when no frame carries a net carrier, because `Delta_n = 0` makes the ratio
    undefined -- and silently returning 0.0 there would look like a calibration that had
    happened. Prefer `calibrate_c_shift_over_loader` in a real run: a single batch makes the
    initialisation depend on the shuffle.
    """
    terms = c_shift_terms(out, energy_target, counts)
    if terms is None or terms.numel() == 0:
        return None
    return float(torch.median(terms))


def c_shift_table_terms(out, energy_target, counts, graph_sizes, pristine_atoms=None):
    """Stage A' section 3: per-frame ratios keyed by (charge class, size class), with
    E_LR's value INCLUDED in the residual -- `correction_energy` is delta_sr + delta_lr, so
    what remains is what the per-(charge, size) constant has to absorb and nothing the
    long-range branch already accounts for."""
    from mace.modules.defect_counting import c_shift_classes

    if energy_target is None:
        return None
    counts = counts.reshape(counts.shape[0], -1)
    delta_n = (counts[:, 0] + counts[:, 1] - counts[:, 2] - counts[:, 3]).to(
        out["correction_energy"].dtype)
    sel = delta_n.abs() > 0
    if not bool(sel.any()):
        return None
    resid = energy_target.to(delta_n.dtype) - out["base_energy"] - out["correction_energy"]
    ratio = (resid / torch.where(sel, delta_n, torch.ones_like(delta_n))).detach()
    charge_cls, size_cls = c_shift_classes(counts, graph_sizes, pristine_atoms)
    return [(int(c), int(s), float(r)) for c, s, r, keep
            in zip(charge_cls.tolist(), size_cls.tolist(), ratio.tolist(), sel.tolist())
            if keep]


def calibrate_c_shift_table_over_loader(model, loader, device, forward=None):
    """The (charge, size) medians over EVERY charged frame, written into the head's table.

    Returns `{(charge_cls, size_cls): (median, n)}`. The scalar `c_shift` is left at zero
    so the table is the whole calibration and reads directly as c(79), c(159).
    """
    terms: Dict[tuple, list] = {}
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                out = (forward(batch) if forward is not None
                       else model(batch.to_dict(), training=False, compute_force=False))
            sizes = batch.ptr[1:] - batch.ptr[:-1]
            t = c_shift_table_terms(
                out, getattr(batch, "energy", None), batch.carrier_counts, sizes,
                getattr(getattr(model, "spectral", None), "pristine_atoms", None))
            for c, s, r in (t or []):
                terms.setdefault((c, s), []).append(r)
    finally:
        model.train(was_training)
    head = getattr(model, "spectral", None)
    table = getattr(head, "c_shift_table", None)
    result = {}
    with torch.no_grad():
        if table is not None:
            table.zero_()
        for (c, s), vals in terms.items():
            med = float(torch.median(torch.tensor(vals)))
            result[(c, s)] = (med, len(vals))
            if table is not None:
                table[c, s] = med
    return result


def calibrate_c_shift_over_loader(model, loader, device, forward=None,
                                  max_batches: Optional[int] = None):
    """`c` from EVERY charged frame the loader holds, in one pass. Shuffle-independent.

    WHY NOT THE FIRST BATCH. `c` is the head's energy zero and it is set once, at
    initialisation, from data -- so a value that depends on which frames the shuffle happened
    to put first is a run-to-run difference with no physical content. Worse in the degenerate
    case: a first batch that happens to be all neutral skips the calibration entirely, which
    is what the production smoke hit. One pass over the loader costs one forward per batch at
    epoch zero and removes both problems.

    Returns `(c, n_frames)`; `c` is None when no charged frame was found anywhere, which is a
    statement about the dataset and is logged by the caller rather than papered over with 0.0.
    """
    terms = []
    n = 0
    was_training = model.training
    model.eval()
    try:
        for k, batch in enumerate(loader):
            if max_batches is not None and k >= max_batches:
                break
            batch = batch.to(device)
            with torch.no_grad():
                out = (forward(batch) if forward is not None
                       else model(batch.to_dict(), training=False, compute_force=False))
            t = c_shift_terms(out, getattr(batch, "energy", None), batch.carrier_counts)
            if t is not None and t.numel():
                terms.append(t.float().cpu())
                n += int(t.numel())
    finally:
        model.train(was_training)
    if not terms:
        return None, 0
    return float(torch.median(torch.cat(terms))), n


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
    # Plan v8.1: a head pickled before the energy constant left H still carries the old
    # `c_shift` / `c_shift_table` parameters; the forward never reads them and they must
    # not sit in the optimiser (weight decay would move a number nothing consumes).
    if name.split(".")[-1] in ("c_shift", "c_shift_table"):
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
    # Unwrap first. The trainer hands `take_step` whatever it is training, which under
    # DistributedDataParallel is a wrapper whose own attributes do not include `madelung` --
    # so a `getattr` on the wrapper would find nothing, skip the projection silently, and let
    # Z drift off the composition hyperplane for the whole run with no error anywhere.
    target = getattr(model, "module", model)
    madelung = getattr(target, "madelung", None)
    if madelung is not None:
        madelung.project_()


def apply_harrison(model, atomic_numbers: Sequence[int]) -> bool:
    """Harrison term values and universal hoppings, at the head's own covalent anchors.

    eps0 = 0 makes every site degenerate -- the atomic limit, where the bond order vanishes
    and, on the frozen-P gradient, nothing could move the hoppings at all.

    SECTION 2.3 of the speed cycle removed the bond-length argument. It was a measured
    per-host number that also anchored the envelope, so a host entered the head twice and
    the two could be given different values; the scale now comes from the head's
    `d_ref_pair` buffer, which is the same object the envelope reads.

    Returns whether it ran, so `protocol_summary` can report an OUTCOME. A caller that infers
    "initialised" from the stage number is recording its own intent, which is the fault the
    c-shift field was fixed for.
    """
    from mace.modules.defect_counting import harrison_initialise

    head = getattr(model, "spectral", None)
    if head is None:
        return False
    harrison_initialise(head, [int(z) for z in atomic_numbers])
    return True


def protocol_summary(stage: int, e_gap: float, w_gap: float, warmup: int,
                     clip: float, freeze_z: bool, model=None,
                     c_shift: Optional[float] = None,
                     harrison_applied: Optional[bool] = None,
                     c_shift_n_frames: Optional[int] = None) -> Dict[str, object]:
    """What a run ACTUALLY DID, as a dict to be logged and saved beside the results.

    A run whose artefact does not record its own protocol is a run whose numbers cannot be
    compared to anything later, which this programme has paid for four times.

    INTENT VERSUS OUTCOME. Every field here is one of two kinds, and mixing them is the fault
    this function exists to prevent. Settings the caller chose (`e_gap`, `w_gap`, `warmup`,
    `grad_clip`) are intent and are reported as given. Everything that describes what happened
    -- the smearing family and width, whether Harrison initialisation ran, whether the c-shift
    calibrated and to what, whether Z is actually frozen, whether the base is actually frozen
    -- is MEASURED off the model or passed in as an outcome. `c_shift_calibrated` used to be
    `stage >= 3` and `harrison_init` still was; both are outcomes now.
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
    # Measured, not assumed: whether the charges and the base are actually receiving gradient
    # right now. A run configured with --defect_protocol_freeze_z but whose mask never ran
    # would otherwise record the intent and train the opposite.
    z_frozen = bool(freeze_z)
    base_frozen = None
    if model is not None:
        named = dict(model.named_parameters())
        z = [p for n, p in named.items() if n.startswith("madelung.")]
        if z:
            z_frozen = not any(p.requires_grad for p in z)
        base = [p for n, p in named.items()
                if n.startswith(("interactions.", "products.", "readouts.",
                                 "node_embedding."))]
        if base:
            base_frozen = not any(p.requires_grad for p in base)
    return dict(stage=int(stage), e_gap=float(e_gap), w_gap=float(w_gap),
                warmup=int(warmup), grad_clip=float(clip),
                freeze_z=z_frozen, base_frozen=base_frozen,
                smearing_family=family, smearing_width=float(width),
                harrison_init=(stage >= 3 if harrison_applied is None
                               else bool(harrison_applied)),
                c_shift_calibrated=c_shift is not None,
                c_shift=None if c_shift is None else float(c_shift),
                c_shift_n_frames=(None if c_shift_n_frames is None
                                  else int(c_shift_n_frames)))


def zero_on_site_correction(model) -> bool:
    """Start the on-site correction channel at exactly zero output. Returns whether it ran.

    WHAT THIS REMOVES. The measured channel applies a near-uniform +0.27 eV to every atom in
    the cell -- a global gauge, degenerate with `eps0` by species and with the whole-spectrum
    offset F9 measured. It is free to be one because a uniform on-site shift moves every level
    together and contributes exactly zero force, so a forces-only objective never touches it.

    Zeroing the final layer makes the channel start at no shift at all, so whatever it ends up
    carrying was put there by the joint run's energy loss rather than inherited from a random
    initialisation and then frozen in by a flat direction. The centred correction -- which
    would remove the species mean by construction, not merely at step zero -- stays at R3.

    Weights AND bias: zeroing only the weight leaves a constant per-channel offset, which is
    precisely the mode being removed.
    """
    head = getattr(model, "spectral", None)
    site = getattr(getattr(head, "h", None), "site", None)
    if site is None:
        return False
    last = None
    for module in site.modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    if last is None:
        return False
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
    return True
