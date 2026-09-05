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

CONVENTION, SETTLED: THE FULL SUM. ``phi_LR`` is the smooth (reciprocal-space) part of the
Ewald potential of {Z}, with the same smearing, the same ``G = 0`` treatment and the same
background as ``E_LR``; only the true self term (``j = i``, ``R = 0``) is excluded, and there
is no per-cell alignment and no self-image subtraction. The previous convention subtracted
``A_ii``, which measurement showed to be the Makov-Payne self-image potential ``-alpha_M/L``
-- ion i's own periodic images, which are real atoms of the crystal that a carrier on site i
feels, and which a different supercell merely relabels. Removing them made ``phi_LR`` a
function of the box rather than of the crystal, by ~0.3 eV * Z_i between the 79- and
159-atom training cells. The real-space part is absorbed by the learned local term, which is
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

import math

import torch
from torch import nn

__all__ = ["site_potential", "self_potential_of", "MadelungOnSite"]


def self_potential_of(ewald, cell: torch.Tensor) -> torch.Tensor:
    """``A_ii``, the potential a lone unit charge feels in its own cell. **DIAGNOSTIC ONLY.**

    NO LONGER SUBTRACTED FROM ``phi_LR``, and the reason is what this function turns out to
    measure. LES sums over ``k != 0`` only, which is the jellium neutralisation, and its
    kernel already excludes the true self term. What is left is exactly the Makov-Payne
    self-image potential of a point charge in jellium, ``-alpha_M / L``:

        L (A)   5.6      11.2     16.8     22.4     33.6
        A_ii   -7.320   -3.666   -2.450   -1.843   -1.235
        -a/L   -7.297   -3.648   -2.432   -1.824   -1.216     (alpha_M = 2.8373, cubic)

    Agreement to ~1.5% at every size, and there is no additive Gaussian self-energy present
    (that would be +11.49 eV/e at sigma = 1 and would swamp the table). So subtracting `A_ii`
    was removing the potential of ion i's OWN PERIODIC IMAGES -- real atoms of the crystal
    that a carrier on site i genuinely feels, and which a different supercell choice merely
    relabels as separate atoms. That made ``phi_LR`` depend on the box rather than on the
    crystal.

    Kept, because the identity above is the calibration of the shared kernel's ``G = 0``
    convention and `test_madelung_convention.py` asserts it. It must not re-enter the
    potential: `site_potential` refuses a `self_potential` argument outright.
    """
    cell = cell.view(-1, 3, 3)
    out = []
    for g in range(cell.shape[0]):
        q = torch.ones(1, device=cell.device, dtype=cell.dtype)
        r = torch.zeros(1, 3, device=cell.device, dtype=cell.dtype)
        b = torch.zeros(1, dtype=torch.long, device=cell.device)
        out.append(2.0 * ewald.energy(q, r, cell[g: g + 1], b).reshape(()))
    return torch.stack(out)


# e^2 / (4 pi eps_0) in eV A; LES stores 2 pi times this as its `norm_factor`.
COULOMB_CONSTANT = 14.399645
MADELUNG_RANGES = ("full", "long_range", "off")


def short_range_potential(charges: torch.Tensor, edge_index: torch.Tensor,
                          edge_lengths: torch.Tensor, r_split: float, sigma: float,
                          n_nodes: int, p: int = 6) -> torch.Tensor:
    """`V_SR(r_split)_i`: the part of the smeared Coulomb site potential that comes from the
    ions within `r_split` of site i, switched off smoothly at `r_split` (plan v8 section 7.3,
    `V_full = V_SR(r_split) + V_LR(r_split)`).

    The kernel is LES's own: a Gaussian charge of width `sigma` (the Ewald `sigma`) has the
    potential `C q erf(r / (sigma sqrt 2)) / r`, so `V_SR` is exactly the short-range piece
    of the potential `site_potential` differentiates out of the Ewald energy -- summed over
    the neighbour list handed in (which carries the periodic images), never over a
    minimum-image box. The switch is MACE's polynomial cutoff (p = 6: value, first and second
    derivative zero at `r_split`), so `V_LR = V_full - V_SR` is as smooth as `V_full`.

    Stage 1.3's arm (b) hands the head `V_LR(r_split)` with `r_split` the first-block cutoff:
    the part of the ion-lattice potential the trunk cannot see.
    """
    src, dst = edge_index[0], edge_index[1]
    r = edge_lengths.reshape(-1)
    x = r / float(r_split)
    envelope = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x ** p + p * (p + 2.0) * x ** (p + 1)
                - (p * (p + 1.0) / 2.0) * x ** (p + 2)) * (x < 1.0)
    kernel = COULOMB_CONSTANT * torch.erf(r / (float(sigma) * math.sqrt(2.0))) / r.clamp_min(1e-12)
    contribution = charges[src] * kernel * envelope
    return torch.zeros(n_nodes, dtype=charges.dtype, device=charges.device).index_add_(
        0, dst, contribution)


def site_potential(ewald, charges: torch.Tensor, positions: torch.Tensor,
                   cell: torch.Tensor, batch: torch.Tensor,
                   self_potential: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``V_i = dE/dq_i``: the FULL SUM over the infinite ion lattice, self term excluded.

    STATEMENT OF RECORD, to be asserted and never implemented around:

        Image corrections enter through exactly two places: E_LR's periodic/isolated switch
        and the per-(charge, size) reference constants. H is gauge-invariant -- the same
        periodic ion-lattice potential in training and in isolated evaluation. There is no
        separate carrier-host term; the interaction is `sum_i (P - P_ref)_ii eps_i` through
        `phi_LR`, with forces by Hellmann-Feynman.

    THE CONVENTION, and why it changed. The infinite periodic ion lattice is ONE set of
    charges under any supercell description. The site potential that excludes only the true
    self term (`j = i`, `R = 0`) is description-invariant; only its partition into "in-cell"
    and "image" contributions depends on the box. The reductio that a cell-dependent `A_ii`
    makes `eps_i` description-dependent fails, because the `j != i` sum varies
    compensatingly. Under the old subtraction the potential entering H was not the periodic
    potential the labels saw -- it was that potential minus a supercell-dependent fraction of
    a sublattice.

    Differentiating the Ewald energy with respect to the charges is exact and reuses the
    production kernel. ``E = q^T A q / 2``, so ``dE/dq_i = (A q)_i = V_i``. LES sums over
    ``k != 0`` only and its kernel carries no self term, so ``(A q)_i`` is already the full
    lattice sum with the self term absent -- nothing further is subtracted. See
    `self_potential_of` for the measurement that establishes this.

    ``create_graph`` follows the ambient grad mode, NOT ``module.training``. Keying it to
    training mode is a silent force bug: at evaluation with forces requested, the potential
    would detach and the Madelung contribution to the ionic force would vanish -- quietly,
    with forces still returned. The A1 finite-difference check runs in eval mode for exactly
    this reason.
    """
    if self_potential is not None:
        raise ValueError(
            "site_potential no longer accepts a self-potential: A_ii is the Makov-Payne "
            "self-IMAGE term, not a self-energy, and subtracting it removes real atoms of "
            "the crystal and makes phi_LR depend on the supercell. The argument is refused "
            "rather than ignored so a caller cannot quietly reinstate the old convention.")
    q = charges.clone().requires_grad_(True)
    with torch.enable_grad():
        energy = ewald.energy(q, positions, cell, batch).sum()
        return torch.autograd.grad(energy, q, create_graph=torch.is_grad_enabled())[0]


def project_neutral_(z: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
    """``Z <- Z - (n.Z / n.n) n``. In place, under no_grad; call after each optimiser step."""
    with torch.no_grad():
        n = composition.to(dtype=z.dtype, device=z.device)
        z -= (torch.dot(n, z) / torch.dot(n, n)) * n
    return z


class MadelungOnSite(nn.Module):
    """Learnable per-species charges {Z}, an optional per-SITE deviation, and their shift.

    Holds no length constants and no material knowledge beyond the pristine stoichiometry,
    which is a property of the training set rather than of the defect.

    SECTION 2.2 OF THE SPEED CYCLE'S SPEC adds the per-site channel, in the same corrected
    form as the on-site correction:

        Z_i = Z0[s(i)] + zeta * tanh( z(x_i) - z(xbar_{s(i)}) ),   zeta = 1 e

    Zero on a cell whose atoms all sit at their species' pristine environment, bounded by
    `zeta` per site, and invariant to a constant shift of `z`. At `zeta = 0` the module is
    the per-species one exactly, which is the Stage-B model.

    NEUTRALITY, AND WHY THE PER-SPECIES PROJECTION IS NOT ENOUGH. `project_neutral_` puts
    {Z0} on the pristine composition hyperplane, which fixes the SPECIES charges as a group.
    A per-site deviation is outside that constraint: `sum_i dZ_i` is a learnable per-FRAME
    net charge, and a net charge moves the Ewald `G = 0` constant frame by frame -- which is
    precisely the per-(charge, size) constant the c table is supposed to own. The deviation
    is therefore centred per graph in the forward, so every cell carries exactly the net
    charge the element baseline gives it and the c table's premise survives.
    """

    def __init__(self, num_elements: int, composition: Sequence[float],
                 z_init: Optional[Sequence[float]] = None,
                 site_zeta: float = 0.0, feature_dim: int = 0,
                 hidden: int = 64) -> None:
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
        self.site_zeta = float(site_zeta)
        self.site = None
        if self.site_zeta != 0.0:
            if feature_dim <= 0:
                raise ValueError(
                    "a per-site charge deviation needs the head's feature width; "
                    "madelung_site_zeta was set without one")
            from mace.modules.defect_spectral import _mlp

            self.site = _mlp([int(feature_dim), int(hidden), int(hidden), 1],
                             final_scale=0.05)

    def charges(self, node_species: torch.Tensor, feats: Optional[torch.Tensor] = None,
                centre: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                num_graphs: Optional[int] = None) -> torch.Tensor:
        """`Z_i`. The element baseline alone unless the site channel is on AND the caller
        supplies the features and the pristine centre -- a scorer that has neither gets the
        baseline rather than a silently different model, and `deviation` reports which."""
        base = self.z[node_species.long()]
        dev = self.deviation(node_species, feats, centre, batch, num_graphs)
        return base if dev is None else base + dev

    def deviation(self, node_species: torch.Tensor, feats: Optional[torch.Tensor] = None,
                  centre: Optional[torch.Tensor] = None,
                  batch: Optional[torch.Tensor] = None,
                  num_graphs: Optional[int] = None) -> Optional[torch.Tensor]:
        """`zeta tanh(z(x_i) - z(xbar_s))`, centred per graph. `None` when the channel is off."""
        if getattr(self, "site", None) is None or feats is None or centre is None:
            return None
        from mace.tools.scatter import scatter_mean

        species = node_species.long()
        # STAGE 1 (plan v8 section 4, "no detached quantity anywhere"): the features are not
        # detached. With them detached the per-site charges' dependence on the geometry --
        # through the features -- was cut out of the Madelung potential's analytic force
        # while the numerical derivative kept it: the h-independent floor of 6e-2 eV/A on
        # the band term of charged frames, which vanished when either the shift or the
        # features were held fixed on both sides (the Stage 1.1 decomposition).
        x = feats
        pre = self.site(x).squeeze(-1)
        pre_centre = self.site(centre.to(x.dtype)[species]).squeeze(-1)
        dev = float(self.site_zeta) * torch.tanh(pre - pre_centre)
        if batch is not None and num_graphs is not None:
            dev = dev - scatter_mean(dev, batch, dim=0, dim_size=int(num_graphs))[batch]
        return dev

    def project_(self) -> None:
        """The post-step hook. Cheap enough to call unconditionally."""
        project_neutral_(self.z.data, self.composition)

    def potential(self, ewald, node_species, positions, cell, batch,
                  feats=None, centre=None, num_graphs=None) -> torch.Tensor:
        """``phi_LR`` at every atom, from {Z}. Full lattice sum; see `site_potential`.

        `cell` is ALWAYS the periodic cell, in every mode. There is no isolated branch here
        and there must not be one: H carries the same periodic ion-lattice potential in
        training and in isolated evaluation, and only E_LR switches. The null-cell trick that
        selects LES's isolated evaluator is applied to `cell_les` in the long-range branch
        alone; this function is never handed it.
        """
        return site_potential(
            ewald, self.charges(node_species, feats, centre, batch, num_graphs),
            positions, cell, batch)

    def on_site_shift(self, ewald, node_species, positions, cell, batch, eps_inf: float,
                      self_potential: Optional[torch.Tensor] = None,
                      feats=None, centre=None, num_graphs=None,
                      madelung_range: str = "full", edge_index=None, edge_lengths=None,
                      r_split: Optional[float] = None) -> torch.Tensor:
        """``-phi / eps_inf``, the quantity added to ``eps_local``.

        Returned as the shift rather than the potential so that the sign lives in one place
        and every caller inherits it. `madelung_range` is Stage 1.3's diagnostic switch:
        "full" is the whole lattice potential, "long_range" is `V_full - V_SR(r_split)` over
        the neighbour list given (the part the trunk cannot see), "off" is an exact zero --
        the module stays (its composition is the pristine formula, its charges the static
        ones) and only the term in H is absent.
        """
        if self_potential is not None:
            raise ValueError(
                "on_site_shift no longer accepts a self-potential; see site_potential for "
                "why the subtraction was removed")
        if madelung_range not in MADELUNG_RANGES:
            raise ValueError(f"madelung_range must be one of {MADELUNG_RANGES}, got "
                             f"{madelung_range!r}")
        charges = self.charges(node_species, feats, centre, batch, num_graphs)
        if madelung_range == "off":
            return torch.zeros_like(charges)
        phi = site_potential(ewald, charges, positions, cell, batch)
        if madelung_range == "long_range":
            if edge_index is None or edge_lengths is None or r_split is None:
                raise ValueError("madelung_range='long_range' needs the head's edge list "
                                 "and r_split")
            phi = phi - short_range_potential(charges, edge_index, edge_lengths,
                                              float(r_split), float(ewald.sigma),
                                              int(charges.shape[0]))
        return -phi / float(eps_inf)
