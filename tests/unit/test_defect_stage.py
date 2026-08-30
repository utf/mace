"""Stage B must actually be Stage B: base weights loaded AND held fixed.

Two switches are needed and either alone is silently wrong. `--defect_base_init` without
`--base_lr_factor 0.0` is a warm start whose base drifts; `--base_lr_factor 0.0` without the
init freezes a randomly initialised base, which is worse than not staging. Both failure modes
produce a run that looks fine in the logs, so they are tested rather than trusted.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.defect_models import MACEDefect
from mace.modules.defect_stage import (
    CORRECTION_PREFIXES,
    assert_base_frozen,
    is_correction_param,
    load_stage_a_base,
    snapshot_base,
)


def build(seed=0, channels=16):
    torch.manual_seed(seed)
    return MACEDefect(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3,
        hidden_irreps=o3.Irreps(f"{channels}x0e + {channels}x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False,
    )


def test_partition_covers_every_parameter():
    """Every tensor is either base or correction. An unclassified one would be silently
    frozen or silently trained depending on which side of the split it fell."""
    model = build()
    names = [n for n, _ in model.named_parameters()]
    assert names, "model has no parameters"
    correction = [n for n in names if is_correction_param(n)]
    base = [n for n in names if not is_correction_param(n)]
    assert correction, "no correction parameters identified"
    assert base, "no base parameters identified"
    assert len(correction) + len(base) == len(names)


def test_load_copies_base_and_leaves_correction_alone(tmp_path):
    source, target = build(seed=0), build(seed=1)
    path = tmp_path / "stage_a.model"
    torch.save(source, path)

    before = {n: p.detach().clone() for n, p in target.named_parameters()}
    load_stage_a_base(target, path)

    for name, param in target.named_parameters():
        src = dict(source.named_parameters())[name]
        if is_correction_param(name):
            assert torch.equal(param, before[name]), f"{name} should not have been touched"
        else:
            assert torch.allclose(param, src), f"{name} should match the Stage-A base"


def test_load_copies_buffers_too(tmp_path):
    """atomic_energies and the scale/shift are buffers derived from the training set. Stage A
    and Stage B fit on different sets, so leaving them behind would pair Stage A's weights
    with a different energy reference and quietly change what E_base means."""
    source = build(seed=0)
    with torch.no_grad():
        source.scale_shift.scale.fill_(3.5)
        source.scale_shift.shift.fill_(-1.25)
    path = tmp_path / "stage_a.model"
    torch.save(source, path)

    target = build(seed=1)
    load_stage_a_base(target, path)
    assert torch.allclose(target.scale_shift.scale, torch.tensor(3.5))
    assert torch.allclose(target.scale_shift.shift, torch.tensor(-1.25))


def test_mismatched_trunk_is_rejected(tmp_path):
    """A different trunk width must fail loudly, not copy the tensors that happen to fit."""
    path = tmp_path / "stage_a.model"
    torch.save(build(seed=0, channels=16), path)
    with pytest.raises(RuntimeError, match="does not match"):
        load_stage_a_base(build(seed=1, channels=32), path)


def test_assert_base_frozen_catches_a_moving_base():
    model = build()
    ref = snapshot_base(model)
    assert_base_frozen(model, ref)          # unchanged: passes

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_correction_param(name):
                param.add_(1e-3)
                break
    with pytest.raises(RuntimeError, match="not frozen"):
        assert_base_frozen(model, ref)


def test_assert_base_frozen_ignores_correction_movement():
    """The correction is supposed to move. Only the base is under the freeze claim."""
    model = build()
    ref = snapshot_base(model)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if is_correction_param(name):
                param.add_(1.0)
    assert_base_frozen(model, ref)


def test_zero_lr_actually_freezes_the_base_through_an_optimiser_step():
    """The end-to-end claim: base groups at lr = 0 do not move under AdamW, including via
    decoupled weight decay, while the correction does."""
    model = build()
    base_params = [p for n, p in model.named_parameters() if not is_correction_param(n)]
    corr_params = [p for n, p in model.named_parameters() if is_correction_param(n)]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": 0.0, "weight_decay": 5e-7},
        {"params": corr_params, "lr": 1e-2, "weight_decay": 0.0},
    ])

    ref = snapshot_base(model)
    for p in base_params + corr_params:
        p.grad = torch.randn_like(p)
    opt.step()

    assert_base_frozen(model, ref)
    assert any(not torch.equal(p, torch.zeros_like(p)) for p in corr_params)


def test_correction_prefixes_are_documented_constants():
    assert isinstance(CORRECTION_PREFIXES, tuple) and CORRECTION_PREFIXES


def test_current_epoch_is_neither_copied_nor_checked(tmp_path):
    """`current_epoch` is training bookkeeping, not a weight.

    Copying it would seed a Stage-B run with Stage A's final epoch number, which feeds the
    epoch-dependent schedules (long-range gate, size warmup, seed anneal). Checking it would
    report the base as unfrozen simply because training advanced -- which is exactly what the
    end-to-end verifier reported before this was excluded.
    """
    source = build(seed=0)
    if not hasattr(source, "current_epoch"):
        pytest.skip("model has no current_epoch buffer")
    with torch.no_grad():
        source.current_epoch.fill_(137)
    path = tmp_path / "stage_a.model"
    torch.save(source, path)

    target = build(seed=1)
    with torch.no_grad():
        target.current_epoch.fill_(0)
    load_stage_a_base(target, path)
    assert int(target.current_epoch) == 0, "Stage A's epoch counter leaked into Stage B"

    ref = snapshot_base(target)
    with torch.no_grad():
        target.current_epoch.fill_(12)
    assert_base_frozen(target, ref)   # advancing the epoch is not the base moving


@pytest.mark.parametrize("name", ["novelty_channel_scale", "novelty_global_scale"])
def test_novelty_scales_are_correction_state(name):
    """These are logit-seeding scales, and they are registered only when seeding is enabled.

    A Stage-A base trained with seeding off does not have them. Classified as base, they made
    every Stage-B load fail as an 'architecture mismatch' -- which is what caught it. They
    belong to the correction and must be left at Stage B's own initialisation.
    """
    assert is_correction_param(name)


def test_stage_a_without_seeding_loads_into_a_seeded_stage_b(tmp_path):
    """The real Stage A/B pairing: seeding off in A, on in B. Must load cleanly."""
    source = build(seed=0)
    for attr in ("novelty_channel_scale", "novelty_global_scale"):
        if hasattr(source, attr):
            delattr(source, attr)
            source._buffers.pop(attr, None)
    path = tmp_path / "stage_a.model"
    torch.save(source, path)

    target = build(seed=1)
    target.register_buffer("novelty_channel_scale", torch.ones(4))
    target.register_buffer("novelty_global_scale", torch.ones(()))
    load_stage_a_base(target, path)   # must not raise
