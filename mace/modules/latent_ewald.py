###########################################################################################
# Latent Ewald evaluators for charge-aware defect models
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
"""A thin layer over the LES Ewald sum (plan section 9.1).

Nothing here reimplements the reciprocal-space sum, the smearing convention or the
self-energy treatment: those come from ``les`` and are used as they are. What this module
adds is exactly the three things the defect model needs on top:

* the three-way decomposition of section 7.1, so that each piece can be given its own
  boundary condition. ``E_LR`` is quadratic in the structure factor -- self-energy
  subtraction included -- so the polarisation identity

      ``2 E_cross[A, B] = E[A + B] - E[A] - E[B]``

  is exact, and the decomposition needs no access to LES internals;
* the isolated evaluator, which is the same LES module under a null cell: LES falls back
  to a direct real-space sum of the smeared charges there, sharing the smearing and
  self-energy code paths with the periodic version;
* the ``E_dilute`` assembly, in which only the carrier self-term changes boundary
  condition. The environment and cross terms are real physics in an infinite crystal and
  survive to the dilute limit unchanged.

Cost: training needs two evaluations per step (``E[q_host]`` and ``E[q_total]``), not
three. The full decomposition is an inference and testing tool.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch

from mace.tools.scatter import scatter_sum

LES_INSTALL_HINT = (
    "Cannot import 'les'. Please install the 'les' library from "
    "https://github.com/ChengUCB/les."
)


def _build_ewald(les_arguments: Optional[Dict[str, Any]]) -> torch.nn.Module:
    try:
        from les.module.ewald import (  # pylint: disable=import-outside-toplevel
            Ewald,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without les
        raise ImportError(LES_INSTALL_HINT) from exc

    arguments = les_arguments or {}
    # Same keys and defaults as the LES library itself, so an --les_arguments file
    # written for MACELES is valid here.
    return Ewald(
        dl=arguments.get("dl", 2.0),
        sigma=arguments.get("sigma", 1.0),
        remove_self_interaction=arguments.get("remove_self_interaction", True),
        norm_factor=arguments.get("norm_factor", 90.4756),
    )


class LatentEwald(torch.nn.Module):
    """Periodic and isolated evaluators sharing one LES Ewald module.

    ``sigma`` is fixed, not learned: learning it worsens short-range/long-range
    identifiability for no gain. ``dl`` sets the reciprocal-space cutoff and is a
    convergence parameter to be recorded, not tuned against validation loss.
    """

    def __init__(self, les_arguments: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.les_arguments = dict(les_arguments) if les_arguments else {}
        self.ewald = _build_ewald(les_arguments)

    @property
    def sigma(self) -> float:
        return float(self.ewald.sigma)

    def energy(
        self,
        charges: torch.Tensor,  # [n_nodes]
        positions: torch.Tensor,  # [n_nodes, 3]
        cell: torch.Tensor,  # [n_graphs, 3, 3]
        batch: torch.Tensor,  # [n_nodes]
    ) -> torch.Tensor:
        """Periodic latent-Ewald energy, one value per graph.

        A cell with zero volume selects the isolated evaluator inside LES, which is how
        non-periodic configurations are handled.

        Includes the neutralising-background term that LES omits. LES is not a split Ewald:
        it sums ``exp(-sigma^2 k^2 / 2) / k^2 * |S(k)|^2`` over ``k != 0`` only. For a
        neutral cell that is exact, because ``S(0) = sum_i q_i = 0``. For a net-charged one
        the ``k -> 0`` limit is finite once a uniform compensating background cancels the
        divergent ``1/k^2`` piece, and what remains is

            ``E_bg = -(2 pi / V) (sigma^2 / 2) Q^2 * C = -pi sigma^2 Q^2 C / V``

        (the same term as ``-pi Q^2 / (2 V alpha^2)`` with ``alpha = 1 / (sigma sqrt 2)``).
        It is ``O(1/V)`` -- about -24 meV at the 79-atom perovskite training cells and
        -3 meV at 639 atoms -- so it is a correctness item, not a size-extensivity one: it
        is a pure ``Q^2`` effect and vanishes identically for a neutral distribution. The
        reason to include it is that the DFT labels use the jellium convention, so without
        it the model is fitting to a different electrostatic convention than its own.
        """
        cell = cell.view(-1, 3, 3)
        energy, _, _ = self.ewald(q=charges, r=positions, cell=cell, batch=batch)
        num_graphs = int(energy.shape[0])
        net = scatter_sum(charges, batch, dim=0, dim_size=num_graphs)
        volume = torch.det(cell)
        # A zero-volume cell is the isolated evaluator, which has no background at all.
        background = torch.where(
            volume.abs() > 0,
            -math.pi
            * self.sigma**2
            * net**2
            * self.ewald.norm_factor
            / volume.clamp_min(1e-30),
            torch.zeros_like(energy),
        )
        return energy + background

    def isolated_energy(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Isolated (non-periodic) energy: no images, no compensating background.

        Shares the smearing and self-energy code paths with the periodic evaluator, so
        the difference between the two is purely image-plus-background. It is
        independent of the cell and therefore contributes no stress.
        """
        null_cell = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        energy, _, _ = self.ewald(q=charges, r=positions, cell=null_cell, batch=batch)
        return energy

    def decompose(
        self,
        q_env: torch.Tensor,
        q_carrier: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the periodic energy into (environment, cross, carrier) pieces.

        The factor of two in ``E_LR = E[env] + 2 E_cross + E[carrier]`` is written
        explicitly here and unit-tested against a two-charge analytic case: an absorbed
        or doubled cross term is one of the likelier prefactor bugs in this code.
        """
        energy_env = self.energy(q_env, positions, cell, batch)
        energy_carrier = self.energy(q_carrier, positions, cell, batch)
        energy_total = self.energy(q_env + q_carrier, positions, cell, batch)
        energy_cross = 0.5 * (energy_total - energy_env - energy_carrier)
        return energy_env, energy_cross, energy_carrier

    def dilute_correction(
        self,
        q_carrier: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """``E_dilute - E_total``, which is a change of boundary condition on one piece.

        The environment and cross terms carry the same boundary condition in both
        evaluators and drop out of the difference exactly, leaving only the carrier's
        interaction with its own images and with the compensating background. The
        result scales as ``a^2``, so the accuracy of every dilute-limit number is
        inherited from the accuracy of the amplitude, quadratically.
        """
        periodic = self.energy(q_carrier, positions, cell, batch)
        isolated = self.isolated_energy(q_carrier, positions, batch, num_graphs)
        return isolated - periodic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.ewald})"
