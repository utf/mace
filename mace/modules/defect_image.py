"""Section 2.3 of the Stage A' spec: the image-compensation potential, in its own module.

Kept OUT of `defect_madelung.py` on purpose: that module carries the recorded prohibition
that the phi_LR path has no isolated-mode branch (test_madelung_convention asserts it on
the AST). This term does use the isolated evaluator -- it is the potential of the carrier's
images, periodic minus isolated, and it is not adopted (probe F15 failed; the tiling test
passed); it lives behind `image_compensation`, off by default.
"""

from __future__ import annotations

import torch

__all__ = ["image_potential"]


def image_potential(ewald, charges: torch.Tensor, positions: torch.Tensor,
                    cell: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """``phi_img_i = [A_per q]_i - [A_iso q]_i``: the potential of the IMAGES of ``q`` only.

    Section 2.3 of the Stage A' spec. ``q`` is the carrier density from a first solve of H,
    detached -- the charge is fixed, the geometry is not, so what comes back carries the
    position dependence at fixed ``q_c`` and nothing else. Same kernel object, smearing and
    ``G = 0`` convention as ``E_LR`` and ``site_potential``: the periodic evaluator minus the
    isolated one, differentiated with respect to the charge exactly as ``site_potential`` is.

    Zero on any cell whose ``q`` is zero, by construction.
    """
    q = charges.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        e_per = ewald.energy(q, positions, cell, batch).sum()
        e_iso = ewald.isolated_energy(q, positions, batch, num_graphs).sum()
        return torch.autograd.grad(e_per - e_iso, q,
                                   create_graph=torch.is_grad_enabled())[0]
