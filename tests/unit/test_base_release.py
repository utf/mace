"""The two-timescale base must release on schedule, roll back on a LABEL-FREE signal, and
never consult the vacancy assignment to decide.

Both halves of the release matter. Never releasing reproduces E1's hard freeze, which cost
20-33 meV/A on forces. Releasing without a guard risks the base quietly re-absorbing the site
signal after the screen's decisive epochs -- the M1 laundering this cycle exists to avoid --
with nothing in the logs to show it.

The guard was first keyed on hub mass, which needs the vacancy assignment. That is weaker
than a label in the loss but still a defect label reaching a training decision, so it now
keys on alpha overlap, N_eff and the active/null channel ratio: "did the attention move, and
did it spread out?", answerable without telling the model where the defect is.
"""

import inspect

import numpy as np
import torch

from mace.modules import defect_release
from mace.modules.defect_release import BaseRelease


class Toy(torch.nn.Module):
    """Two named parameter groups standing in for base and correction."""

    def __init__(self):
        super().__init__()
        self.readouts = torch.nn.Linear(3, 3)          # base-side name
        self.carrier_head = torch.nn.Linear(3, 3)      # correction-side name


def make(release_epoch=30, factor=0.01):
    model = Toy()
    base = [p for n, p in model.named_parameters() if n.startswith("readouts")]
    corr = [p for n, p in model.named_parameters() if n.startswith("carrier_")]
    opt = torch.optim.AdamW([
        {"name": "readouts", "params": base, "lr": 0.0},
        {"name": "carrier", "params": corr, "lr": 0.01},
    ])
    return model, opt, BaseRelease(model, opt, release_epoch, factor)


def base_lr(opt):
    return [g["lr"] for g in opt.param_groups if g.get("name") == "readouts"][0]


SETTLED = dict(alpha_overlap=1.0, n_eff=2.0, null_ratio=0.03)


def test_base_stays_frozen_before_the_release_epoch():
    model, opt, rel = make()
    for e in range(30):
        rel.on_epoch_start(e, base_lr=0.01, state=SETTLED)
    assert not rel.released
    assert base_lr(opt) == 0.0


def test_release_sets_the_slow_learning_rate():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    assert rel.released
    assert np.isclose(base_lr(opt), 0.01 * 0.01)


def test_no_revert_while_the_carrier_stays_put():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    rel.on_epoch_end(31, state=dict(alpha_overlap=0.97, n_eff=2.2, null_ratio=0.05))
    assert not rel.reverted
    assert base_lr(opt) > 0.0


def test_revert_on_attention_moving():
    """alpha_overlap is the primary trigger: the attention vector itself moved."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    saved = model.readouts.weight.detach().clone()
    with torch.no_grad():
        model.readouts.weight.add_(1.0)
    rel.on_epoch_end(35, state=dict(alpha_overlap=0.80, n_eff=2.1, null_ratio=0.05))
    assert rel.reverted
    assert torch.allclose(model.readouts.weight, saved), "base was not rolled back"
    assert base_lr(opt) == 0.0, "base was not refrozen"


def test_revert_on_delocalisation():
    """N_eff rising by more than half means the state spread out, wherever it sits."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    rel.on_epoch_end(35, state=dict(alpha_overlap=0.99, n_eff=3.5, null_ratio=0.05))
    assert rel.reverted


def test_revert_on_null_ratio():
    """Approaching the unsupervised channels means supervision stopped doing anything."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    rel.on_epoch_end(35, state=dict(alpha_overlap=0.99, n_eff=2.1, null_ratio=0.9))
    assert rel.reverted


def test_revert_leaves_the_correction_alone():
    """Only the base is under the guard; the head keeps whatever it learned."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    with torch.no_grad():
        model.carrier_head.weight.add_(2.0)
    moved = model.carrier_head.weight.detach().clone()
    rel.on_epoch_end(35, state=dict(alpha_overlap=0.10, n_eff=9.0, null_ratio=0.99))
    assert torch.allclose(model.carrier_head.weight, moved)


def test_revert_happens_at_most_once():
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    rel.on_epoch_end(31, state=dict(alpha_overlap=0.1, n_eff=9.0, null_ratio=0.99))
    after = model.readouts.weight.detach().clone()
    with torch.no_grad():
        model.readouts.weight.add_(5.0)
    rel.on_epoch_end(32, state=dict(alpha_overlap=0.1, n_eff=9.0, null_ratio=0.99))
    assert not torch.allclose(model.readouts.weight, after), "reverted a second time"


def test_missing_state_is_not_treated_as_collapse():
    """A batch with no measurable diagnostics must not trigger a spurious rollback."""
    model, opt, rel = make()
    rel.on_epoch_start(30, base_lr=0.01, state=SETTLED)
    rel.on_epoch_end(31, state=None)
    rel.on_epoch_end(32, state={})
    assert not rel.reverted


def test_guard_never_reads_the_vacancy_assignment():
    """The guardrail, asserted on the source rather than trusted.

    The decision path must not mention hub mass, the shell, the cage, or the locator. If a
    future change reintroduces any of them, this fails rather than quietly letting a defect
    label back into a training decision.
    """
    import ast

    # Check the CODE, not the prose. A text scan either misses the identifier written as
    # "hub mass" in a docstring, or -- once tightened -- forbids the module from explaining
    # why it avoids hub mass, which is precisely the context that stops someone putting it
    # back. Walking the AST checks what the guard actually reads.
    tree = ast.parse(inspect.getsource(defect_release))
    forbidden = {"hub_mass", "locate_vacancy", "vacancy_site", "shell", "cage",
                 "reference_mass"}

    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.arg):
            seen.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            seen.add(mod.split(".")[-1])
            for alias in node.names:
                seen.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # String keys are how the state dict is read, so they count as code.
            if node.value in forbidden:
                seen.add(node.value)

    leaked = forbidden & seen
    assert not leaked, f"guard's decision path references {sorted(leaked)}"
