"""Every defect flag must survive the whole chain: launcher -> CLI -> model.

A flag can go missing at three points, and each break is silent in a different way:

  * declared in the arg parser but never passed by the launcher -> the run trains with the
    DEFAULT while the config claims otherwise. This is how R2 came within one smoke test of
    running 32 seeds of plain H0 with no gauge control, reported as H2/H3;
  * passed by the launcher but absent from the parser -> argparse rejects it, which is loud
    but only at launch;
  * declared and passed but never reaching the model constructor -> the model silently
    ignores it.

None of these is caught by any unit test of the head itself, because the head is correct in
all three cases. So the chain is asserted directly.
"""

import re
from pathlib import Path

import pytest

from mace.tools.arg_parser import build_default_arg_parser

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "defect-example" / "train_defect_model.sh"
MODEL_UTILS = REPO / "mace" / "tools" / "model_script_utils.py"


def cli_flags():
    """Every --defect_* / --base_lr_factor option the parser accepts."""
    parser = build_default_arg_parser()
    out = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--defect_") or opt == "--base_lr_factor":
                out.add(opt.lstrip("-"))
    return out


def launcher_text():
    return LAUNCHER.read_text()


def launcher_passes():
    """Flags the launcher actually puts on the command line."""
    return set(re.findall(r"--(\w+)=\"?\$\{", launcher_text()))


def test_every_flag_the_launcher_passes_exists_in_the_parser():
    """The loud failure: argparse rejects an unknown option and the run dies at launch."""
    unknown = {f for f in launcher_passes()
               if f.startswith("defect_") or f == "base_lr_factor"} - cli_flags()
    assert not unknown, (
        f"launcher passes flags the parser does not define: {sorted(unknown)}")


def test_every_defect_flag_is_passed_by_the_launcher():
    """The silent failure, and the one that matters.

    A flag the launcher never passes leaves the run on the default while the experiment
    config says otherwise -- a whole arm quietly reduced to the baseline it was meant to be
    compared against.
    """
    passed = launcher_passes()
    text = launcher_text()
    missing = []
    for flag in sorted(cli_flags()):
        # A flag may be passed unconditionally or guarded by ${VAR:+...}; both count.
        if flag in passed or f"--{flag}=" in text:
            continue
        missing.append(flag)
    assert not missing, (
        f"parser defines {len(missing)} defect flags the launcher never passes: {missing}")


def test_every_spectral_setting_survives_config_extraction():
    """A fourth place a setting can vanish: the config extractor.

    The saved model is rebuilt from `extract_config_mace_model` during the cuEq conversion,
    so any constructor argument the extractor omits reverts to its DEFAULT in the artefact --
    with no shape mismatch to catch it when the option only toggles behaviour.

    This is not hypothetical twice over. `spectral_head` was omitted once, and then
    `spectral_sigma` and `spectral_gauge_penalty` were omitted while being correctly wired
    through the launcher, the parser and both construction sites: training ran with them ON
    and the saved model had them OFF. Every downstream score would have been of a different
    model than the one trained.

    Rather than list the settings I remember, build a model with every one flipped away from
    its default and assert the round trip preserves all of them.
    """
    import numpy as np
    import torch
    from e3nn import o3

    from mace import modules
    from mace.modules.defect_models import MACEDefect
    from mace.tools.scripts_utils import extract_config_mace_model

    flipped = dict(spectral_head=True, spectral_decay=True, spectral_first_shell=True,
                   spectral_sigma=True, spectral_gauge_penalty=True,
                   spectral_num_states=4, spectral_smearing=0.05, spectral_r_cut=7.5)

    torch.manual_seed(0)
    model = MACEDefect(
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu, atomic_energies=np.zeros((1, 3)),
        avg_num_neighbors=8.0, atomic_numbers=[17, 55, 82], correlation=2,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        carrier_feature_dim=16, counter_embedding_dim=8, carrier_mlp_hidden=16,
        use_long_range=False, **flipped)

    config = extract_config_mace_model(model)
    missing = [k for k in flipped if k not in config]
    assert not missing, f"extractor drops {missing}; they revert to defaults on rebuild"

    rebuilt = model.__class__(**config)
    assert rebuilt.spectral_head is True
    assert rebuilt.spectral.use_decay is True
    assert rebuilt.spectral_first_shell is True
    assert rebuilt.spectral.use_sigma is True, "sigma lost in the round trip"
    assert rebuilt.spectral.gauge_penalty is True, "gauge penalty lost in the round trip"
    assert rebuilt.spectral.num_states == 4
    assert rebuilt.spectral.r_cut == pytest.approx(7.5)


@pytest.mark.parametrize("flag,ctor", [
    ("defect_spectral_head", "spectral_head"),
    ("defect_spectral_decay", "spectral_decay"),
    ("defect_spectral_first_shell", "spectral_first_shell"),
    ("defect_spectral_sigma", "spectral_sigma"),
    ("defect_spectral_gauge_penalty", "spectral_gauge_penalty"),
    ("defect_spectral_r_cut", "spectral_r_cut"),
    ("defect_spectral_states", "spectral_num_states"),
    ("defect_spectral_smearing", "spectral_smearing"),
])
def test_spectral_flags_reach_the_model_constructor(flag, ctor):
    """Declared and passed is not enough; it has to arrive at the model."""
    assert flag in cli_flags(), f"--{flag} is not defined in the parser"
    text = MODEL_UTILS.read_text()
    assert f"{ctor}=args.{flag}" in text, (
        f"{ctor} is never set from args.{flag} in model_script_utils")
