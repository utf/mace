"""Section 1.3 of the speed cycle: per-component timing of one training step.

WHY A FLAG RATHER THAN ALWAYS-ON MARKERS. `torch.profiler.record_function` builds a python
object and pushes a RecordFunction on every entry. In the head's hot path that is per graph
per fill per step, and the Fermi bisection alone would take 200 of them. The marks are
therefore no-ops until `enable()` is called, so the production step pays one boolean test
per region and the profile pays the real cost only while it is being measured.

WHAT THE NAMES MEAN, and they are the components section 1.3 asks for:

    trunk            the MACE base: embedding, interactions, products, readouts
    head/loop        the whole per-graph python loop in `CountingHead._forward`
    head/assemble    building the dense H for one graph (SK blocks, index_put)
    head/eigh        `torch.linalg.eigh` inside `head_energy_hf`
    head/fills       the four occupations: bisection, density matrix, entropy
    head/bisect      `find_mu` alone, which is where the GPU->CPU syncs live
    head/response    the differentiable density difference for the force response
    head/forces      the `autograd.grad` that contracts the response into the forces
    ewald/madelung   `site_potential`: phi_LR of the learnable species charges
    ewald/lr         the long-range energy branch
    ewald/image      the image-compensation potential (arm B)
    loss/backward    the loss's own backward, timed by the caller

A region that is absent from the table did not run, which is itself a reading -- an E_LR
branch that never appears is an E_LR branch that is switched off.
"""

from __future__ import annotations

import contextlib

__all__ = ["mark", "enable", "enabled"]

_ON = False


def enable(on: bool = True) -> bool:
    """Turn the marks on (or off) process-wide; returns the previous setting."""
    global _ON            # pylint: disable=global-statement
    previous = _ON
    _ON = bool(on)
    return previous


def enabled() -> bool:
    return _ON


@contextlib.contextmanager
def mark(name: str):
    """A profiler region, or nothing at all."""
    if not _ON:
        yield
        return
    from torch.profiler import record_function

    with record_function(name):
        yield
