"""Edit 1: the Madelung potential of learnable per-species charges, inside H.

The response channel measured two things before it was retired, and both point here:

* **reach confirmed.** Electrostatics does give a compact state the ~6 A force footprint
  that a short-ranged tight-binding head cannot produce -- its forces come from ``dt/dR``
  and ``deps/dR``, both of which die with the hopping envelope.
* **site selection absent.** It did not make the head prefer the hub.

The second is the reason for the move. ``E_resp`` was a bolt-on energy readout: the carrier's
potential could change the energy but never the Hamiltonian, so it could not change where the
carrier goes. Putting the host's Madelung potential on the on-site energies instead makes it
**variational** -- an anion vacancy raises phi at the neighbouring cations, which lowers their
electron on-site energy, which is a donor well the eigenproblem can actually bind into.

    eps_i = eps_local_i  -  phi_LR_i / eps_inf

THE MINUS SIGN IS THE WHOLE EDIT AND IT SHIPS WITH ITS TOY TABLE. ``phi_LR`` is the
electrostatic *potential* of the charges {Z}; the carrier is an electron, charge -1, so its
energy in that potential is ``-phi``. Get it backwards and the model builds a barrier where
the well should be, fits something anyway, and every downstream number is wrong in a way no
gate catches. `tests/extensions/defect/test_madelung.py` checks the sign three ways on cases
whose answers are known independently of this code.

WHAT IS AND IS NOT SCREENED. ``eps_inf`` is the high-frequency constant: the carrier's own
field, which the lattice has not had time to respond to. It is threaded from
``ForwardContext``, never read from ``model.eps_inf_init`` -- the retained checkpoints carry
6.5 there while both launchers pass 4.0.

CONVENTION, decided here and recorded because Stage 3 depends on it. ``phi_LR`` is the smooth
(reciprocal-space) part of the Ewald potential of {Z}, with the same smearing, the same
``G = 0`` treatment and the same background as ``E_LR``; the self term is excluded and there
is no per-cell alignment. The real-space part is absorbed by the learned local term, which is
what makes ``Z`` identifiable at all. In Stages 1 and 2 the existing difference gauge removes
the per-cell mean of ``eps``, so what acts there is the Madelung **contrast** (test 2). The
absolute offset becomes load-bearing only in Stage 3, where ``eps0`` is ungauged and the
energy labels pin it; test 3 is the assert that the ``G = 0`` convention is a per-(charge,
size) constant so that the two stages agree.

NEUTRALITY. ``Z`` is projected onto the pristine composition hyperplane after each optimiser
step: ``Z <- Z - (n.Z / n.n) n`` with ``n`` the pristine stoichiometry. Without it the species
charges drift as a group, which is a gauge on ``phi`` and a slow divergence in training. The
net charge of a defect cell is not this term's business -- LatentEwald's background handles it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

__all__ = ["site_potential", "self_potential_of", "MadelungOnSite"]


def self_potential_of(ewald, cell: torch.Tensor) -> torch.Tensor:
    """``A_ii``: the smeared potential of a unit charge at its own centre, per cell.

    Taken from the kernel rather than from a formula, so it cannot drift from whatever
    convention LES actually uses. A single unit charge alone in the cell has energy
    ``A_ii / 2``, so ``A_ii`` is twice that. It depends on the cell -- through the images and
    the background -- but not on where the atom sits.
    """
    cell = cell.view(-1, 3, 3)
    out = []
    for g in range(cell.shape[0]):
        q = torch.ones(1, device=cell.device, dtype=cell.dtype)
        r = torch.zeros(1, 3, device=cell.device, dtype=cell.dtype)
        b = torch.zeros(1, dtype=torch.long, device=cell.device)
        out.append(2.0 * ewald.energy(q, r, cell[g: g + 1], b).reshape(()))
    return torch.stack(out)


def site_potential(ewald, charges: torch.Tensor, positions: torch.Tensor,
                   cell: torch.Tensor, batch: torch.Tensor,
                   self_potential: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``V_i = dE/dq_i``, self term removed. The potential at each atom from all the others.

    Differentiating the Ewald energy with respect to the charges is exact and reuses the
    production kernel. ``E = q^T A q / 2``, so ``dE/dq_i = (A q)_i = V_i``, which INCLUDES the
    diagonal ``A_ii q_i``; ``A_ii`` is one constant per cell and is subtracted.

    ``create_graph`` follows the ambient grad mode, NOT ``module.training``. Keying it to
    training mode is a silent force bug: at evaluation with forces requested, the potential
    would detach and the Madelung contribution to the ionic force would vanish -- quietly,
    with forces still returned. The A1 finite-difference check runs in eval mode for exactly
    this reason.
    """
    q = charges.clone().requires_grad_(True)
    with torch.enable_grad():
        energy = ewald.energy(q, positions, cell, batch).sum()
        v = torch.autograd.grad(energy, q, create_graph=torch.is_grad_enabled())[0]
    if self_potential is not None:
        v = v - self_potential[batch] * charges
    return v


def project_neutral_(z: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
    """``Z <- Z - (n.Z / n.n) n``. In place, under no_grad; call after each optimiser step."""
    with torch.no_grad():
        n = composition.to(dtype=z.dtype, device=z.device)
        z -= (torch.dot(n, z) / torch.dot(n, n)) * n
    return z


class MadelungOnSite(nn.Module):
    """Learnable per-species charges {Z} and the on-site shift they induce.

    Holds no length constants and no material knowledge beyond the pristine stoichiometry,
    which is a property of the training set rather than of the defect.
    """

    def __init__(self, num_elements: int, composition: Sequence[float],
                 z_init: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        if len(composition) != num_elements:
            raise ValueError(
                f"composition has {len(composition)} entries for {num_elements} elements; "
                "it is the pristine stoichiometry in the model's own species order")
        init = (torch.zeros(num_elements) if z_init is None
                else torch.as_tensor(list(z_init), dtype=torch.get_default_dtype()))
        self.z = nn.Parameter(init.clone())
        self.register_buffer("composition",
                             torch.as_tensor(list(composition),
                                             dtype=torch.get_default_dtype()))
        project_neutral_(self.z.data, self.composition)

    def charges(self, node_species: torch.Tensor) -> torch.Tensor:
        return self.z[node_species.long()]

    def project_(self) -> None:
        """The post-step hook. Cheap enough to call unconditionally."""
        project_neutral_(self.z.data, self.composition)

    def potential(self, ewald, node_species, positions, cell, batch,
                  self_potential: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``phi_LR`` at every atom, from {Z}."""
        return site_potential(ewald, self.charges(node_species), positions, cell, batch,
                              self_potential=self_potential)

    def on_site_shift(self, ewald, node_species, positions, cell, batch, eps_inf: float,
                      self_potential: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``-phi_LR / eps_inf``, the quantity added to ``eps_local``.

        Returned as the shift rather than the potential so that the sign lives in one place
        and every caller inherits it.
        """
        phi = self.potential(ewald, node_species, positions, cell, batch,
                             self_potential=self_potential)
        return -phi / float(eps_inf)
