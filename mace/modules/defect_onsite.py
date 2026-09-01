"""V2: a counted on-site term, built so the band-edge escape is not expressible.

T-B imposes Delta_bind = lambda_1(pristine) - lambda_1(defect) >= m. A head whose site
energies are functions of trunk features can satisfy that without binding anything: the trunk's
receptive field tells every atom in a defect cell that it is in a defect cell, so eps can be
lowered across the whole cell, the spectrum drops, lambda_1 drops with it, and the inequality
is met by a state exactly as delocalised as before. The level did not move relative to the
host -- the origin did. That is the escape, and V1 exists to measure whether it gets used.

V2 removes the possibility rather than penalising it:

    eps_i = e(z_i) + sum_j phi(z_i, z_j, r_ij)

Element embeddings and pair distances only. No trunk features enter eps at all. Such a term
can express "this Pb has five neighbours instead of six", which is a genuine local fact about
a vacancy's shell and exactly the physics a bound level needs -- but it cannot express "there
is a vacancy 20 A away", because an atom's coordination in bulk is identical whether or not
a defect sits outside its own neighbour shell. The uniform-lowering escape therefore has no
representation, and any Delta_bind V2 achieves has to come from a real level.

Hopping, the sigma term and the response channel are untouched: this changes what eps may
depend on, not what the Hamiltonian is.

The cutoff spans the first and second shells (Pb-Cl ~2.8 A, Cs-Cl ~3.6 A, Pb-Pb ~5.3 A). A
first-shell-only term would see the vacancy's missing neighbour but not the pair of
under-coordinated Pb that the state is supposed to live on.

Installation is surgery on a built model, matching how the harness already wires the response
channel. `rigid_onsite` defaults to None on the head and the forward branches on it, so a
model that never had it installed runs the production path unchanged -- asserted in
tests/unit/test_rigid_onsite.py rather than assumed.
"""

from __future__ import annotations

import torch
from torch import nn

from mace.tools.scatter import scatter_sum

__all__ = ["RigidOnSite", "install_rigid_onsite"]


class RigidOnSite(nn.Module):
    """eps_i = e(z_i) + sum_j phi(z_i, z_j, r_ij), with no trunk features anywhere."""

    def __init__(self, num_elements: int, num_channels: int = 4, hidden: int = 32,
                 radial_dim: int = 8, r_cut: float = 6.0, embed_dim: int = 8,
                 init_scale: float = 0.01) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.radial_dim = radial_dim
        self.r_cut = float(r_cut)
        self.embed = nn.Embedding(num_elements, embed_dim)
        # The bare on-site level per element. Starts at zero so the installed head begins
        # from a flat eps and has to earn any structure, the same discipline zero_u_init
        # applies to the feature-based term.
        self.e = nn.Embedding(num_elements, num_channels)
        with torch.no_grad():
            self.e.weight.zero_()
        self.phi = nn.Sequential(
            nn.Linear(2 * embed_dim + radial_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, num_channels))
        with torch.no_grad():
            self.phi[-1].weight.mul_(init_scale)
            self.phi[-1].bias.mul_(init_scale)

    def _radial(self, r: torch.Tensor) -> torch.Tensor:
        n = torch.arange(1, self.radial_dim + 1, device=r.device, dtype=r.dtype)
        x = (r / self.r_cut).clamp(max=1.0).unsqueeze(-1)
        return torch.sin(n * torch.pi * x) / x.clamp_min(1e-6)

    def _envelope(self, r: torch.Tensor) -> torch.Tensor:
        """Zero value and zero slope at r_cut, so eps and hence the forces stay continuous."""
        x = (r / self.r_cut).clamp(max=1.0)
        return (1.0 - x) ** 3 * (1.0 + 3.0 * x + 6.0 * x * x)

    def forward(self, node_species: torch.Tensor, edge_index: torch.Tensor,
                edge_length: torch.Tensor, n_nodes: int) -> torch.Tensor:
        if node_species is None:
            raise ValueError("RigidOnSite needs node_species; the head was given none")
        # The head passes lengths as [E, 1] in production and [E] in the unit tests; flatten
        # so the radial basis is [E, radial_dim] either way rather than silently gaining a
        # middle axis and failing at the concatenation.
        r = edge_length.reshape(-1)
        src, dst = edge_index[0], edge_index[1]
        z = self.embed(node_species)
        # phi(z_i, z_j, r_ij) accumulated onto i = dst, matching the head's own convention
        # that edge_index[1] is the receiving node.
        feats = torch.cat([z[dst], z[src], self._radial(r)], dim=-1)
        contrib = self.phi(feats) * self._envelope(r).unsqueeze(-1)
        pair = scatter_sum(contrib, dst, dim=0, dim_size=n_nodes)
        return self.e(node_species) + pair


def install_rigid_onsite(model, r_cut: float = 6.0, hidden: int = 32) -> RigidOnSite:
    """Swap the head's feature-based site energies for the counted form, in place.

    The head keeps `self.site` -- removing it would break state_dict round-trips for a model
    that is otherwise identical -- but stops calling it. Its parameters simply take no
    gradient once the branch is live, which is visible as a zero grad rather than as a
    silently retained second eps.
    """
    head = getattr(model, "spectral", None)
    if head is None:
        raise ValueError("model has no spectral head to install a rigid on-site term into")
    num_elements = int(model.atomic_numbers.numel())
    mod = RigidOnSite(num_elements=num_elements, num_channels=head.num_channels,
                      hidden=hidden, radial_dim=head.radial_dim, r_cut=r_cut)
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    head.rigid_onsite = mod.to(device=device, dtype=dtype)
    return head.rigid_onsite
