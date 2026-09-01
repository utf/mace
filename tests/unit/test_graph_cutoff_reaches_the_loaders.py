"""Every dataset loader must build the neighbour graph at graph_cutoff, not r_max.

This is the bug that voided the R2 screen. `graph_cutoff` exists because the spectral carrier
Hamiltonian is longer-ranged than the trunk: the two vacancy-sharing Pb sit a median 5.4 A
apart, so at r_max = 5.0 they have NO EDGE (measured: 0 of 40 charged frames; at 10.0 A,
40 of 40). Without that edge the two-site state the screen was looking for cannot be
represented at all.

The function was written and used -- but only on the valid-file FALLBACK path, which never
executes when an explicit valid_file is given. The training loader, the valid loader and both
test loaders all passed `r_max=args.r_max`, so all 20 R2 seeds trained on a 5 A graph with the
hub pair uncoupled, and the screen did not test its own hypothesis.

Reading a call-site list is what missed it the first time, so this asserts on the source: no
loader may take r_max=args.r_max. For models without the spectral head graph_cutoff returns
r_max unchanged, so ordinary MACE training is unaffected.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_TRAIN = REPO / "mace" / "cli" / "run_train.py"


def test_no_loader_builds_the_graph_at_r_max():
    text = RUN_TRAIN.read_text()
    offenders = [ln.strip() for ln in text.splitlines()
                 if re.search(r"r_max\s*=\s*args\.r_max", ln)]
    assert not offenders, (
        "these call sites build a neighbour graph at r_max instead of graph_cutoff, which "
        f"drops the hub-pair edge for the spectral head: {offenders}")


def test_graph_cutoff_is_actually_used():
    text = RUN_TRAIN.read_text()
    assert text.count("graph_cutoff(args)") >= 5, (
        "graph_cutoff is barely used; the loaders are probably back on r_max")


def test_graph_cutoff_is_r_max_without_the_spectral_head():
    """Ordinary MACE training must be untouched by this."""
    import sys

    sys.path.insert(0, str(REPO))
    from mace.cli.run_train import graph_cutoff

    class Args:
        r_max = 6.0
        num_interactions = 2
        defect_spectral_head = False
        defect_spectral_r_cut = 0.0

    assert graph_cutoff(Args()) == 6.0


def test_graph_cutoff_covers_the_hamiltonian_range_with_the_head_on():
    import sys

    sys.path.insert(0, str(REPO))
    from mace.cli.run_train import graph_cutoff

    class Args:
        r_max = 5.0
        num_interactions = 2
        defect_spectral_head = True
        defect_spectral_r_cut = 10.0

    assert graph_cutoff(Args()) == 10.0

    class Implicit(Args):
        defect_spectral_r_cut = 0.0

    # Falls back to the receptive field, which is what the hub pair needs.
    assert graph_cutoff(Implicit()) == 10.0
