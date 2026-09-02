"""Edit 4: the electron-counting head. Occupation replaces every hand-made convention.

Everything the previous head needed a rule for, this one gets from counting electrons:

| previous machinery            | what replaces it                                          |
|-------------------------------|-----------------------------------------------------------|
| `channel_sign`, `s_c`         | a hole is one fewer electron; the fill does the rest       |
| `mu_c`, `init_mu_for_energy`  | the chemical potential, solved for at fixed N              |
| edge floor and cap            | nothing -- there is no level to keep inside a window       |
| the soft gauge anchor         | the energy labels, which pin `eps0` directly               |
| "which channel is the hole"   | `N_sigma = N_sigma,neutral + e_sigma - h_sigma`            |

    F_sigma(N) = sum_k f_k eps_k - T_el S(f),     f Fermi-Dirac at mu solving sum_k f_k = N
    E_head     = sum_sigma [ F_sigma(N_sigma) - F_sigma(N_sigma,neutral) ]

The reference subtraction is what makes the correction vanish identically at n = 0, for any
parameters -- the same property the counter prefactors used to provide by hand, now a
consequence of the definition rather than a term someone remembered to multiply in.

RESONANCE IS SMOOTH HERE, and that is the point. When the defect level is degenerate with the
continuum the occupation simply spreads over the near-degenerate states; nothing has to decide
which level is "the" defect level, and no gate fires on a frame that has no bound state to
find. D-2 measured 18 such frames in the current cohort and found the head doing worse than
its own base on them.

WHY `mu` IS DETACHED. At fixed N the free energy obeys Hellmann-Feynman: `dF/dtheta =
sum_k f_k deps_k/dtheta`, with no `dmu/dtheta` term, because the occupation derivative
contributions cancel against the entropy's at the Fermi-Dirac solution. Detaching `mu` gives
the EXACT gradient, not an approximation -- and backpropagating through the bisection would
give the same answer at several times the cost, with the loop's tolerance as a new source of
noise.

NEVER BACKPROP THROUGH INDIVIDUAL EIGENVECTORS. Degenerate eigenvectors are only defined up to
a rotation within their subspace, so their gradients are unbounded as the gap closes -- and a
defect level near the continuum is exactly the degenerate case. Everything here that needs the
wavefunction goes through the DENSITY MATRIX `P = sum_k f_k psi_k psi_k^T`, which is invariant
under that rotation. The carrier density is `q_i = a sum_{mu in i} (P - P_ref)_{mu mu}`, and
`sum_i q_i = -Delta n` exactly, with the sign carried by P natively -- there is no `sgn(q_c)`
to get wrong.

BASIS. Four orbitals per atom, `[s, px, py, pz]`, uniform across species. Angular factors are
the standard Slater-Koster table; radial parts are Stage 2's `V_b(r)` for
`b` in {ss-sigma, sp-sigma, pp-sigma, pp-pi}. The Madelung shift is identical for both shells
-- it is an electrostatic potential at a site, and does not know about angular momentum -- and
each shell carries its own bounded correction.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = ["VALENCE", "ORBITALS_PER_ATOM", "sk_block", "fermi_fill", "free_energy",
           "neutral_electrons", "spin_targets", "SlaterKosterH"]

ORBITALS_PER_ATOM = 4                       # s, px, py, pz
# Valence electrons per species, by atomic number. Cs 6s^1, Pb 6s^2 6p^2, Cl 3s^2 3p^5.
VALENCE: Dict[int, int] = {55: 1, 82: 4, 17: 7}
T_EL = 0.025                                # eV, electronic temperature for the smearing
BOND_TYPES = ("ss_sigma", "sp_sigma", "pp_sigma", "pp_pi")


# --------------------------------------------------------------------------- Slater-Koster


def sk_block(direction: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The 4x4 Slater-Koster block for one directed edge.

    `direction` is the unit vector from i to j, [n_edges, 3]; `v` holds the four radial
    integrals in BOND_TYPES order, [n_edges, 4]. Returns [n_edges, 4, 4] with the row index
    on atom i and the column index on atom j, orbitals ordered [s, px, py, pz].

    The sign convention is the standard one and it is asymmetric on purpose:
    `E_{s,x} = +l V_sp_sigma` while `E_{x,s} = -l V_sp_sigma`. Evaluating this function on the
    reversed edge (direction -> -direction) therefore returns exactly the transpose, which is
    what makes H Hermitian by construction rather than by a symmetrisation applied afterwards.
    An H that is only symmetric after the fact has silently wrong `eigh` gradients during the
    backward pass -- a failure this project has already paid for once.
    """
    l, m, n = direction[:, 0], direction[:, 1], direction[:, 2]
    ss, sp, pps, ppp = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    zero = torch.zeros_like(l)
    block = torch.stack([
        torch.stack([ss, l * sp, m * sp, n * sp], dim=-1),
        torch.stack([-l * sp, l * l * pps + (1 - l * l) * ppp,
                     l * m * (pps - ppp), l * n * (pps - ppp)], dim=-1),
        torch.stack([-m * sp, m * l * (pps - ppp),
                     m * m * pps + (1 - m * m) * ppp,
                     m * n * (pps - ppp)], dim=-1),
        torch.stack([-n * sp, n * l * (pps - ppp), n * m * (pps - ppp),
                     n * n * pps + (1 - n * n) * ppp], dim=-1),
    ], dim=-2)
    return block + zero.reshape(-1, 1, 1)


# --------------------------------------------------------------------------- occupation


def neutral_electrons(atomic_numbers: Sequence[int]) -> int:
    """Total valence electrons of the neutral cell. A composition sum, not a defect label."""
    return int(sum(VALENCE[int(z)] for z in atomic_numbers))


def spin_targets(n_total: int, counts: Sequence[int]) -> Tuple[float, float]:
    """`(N_maj, N_min)` from the neutral count and the carrier counters.

    Neutral fill is ceil/floor of half, so an odd electron count gives the doublet
    automatically -- V_Cl^0 at 79 atoms has 409 valence electrons and lands at 205/204 with
    nothing asked of the caller. Counters are (e_maj, e_min, h_maj, h_min): an electron adds
    one to its spin channel, a hole removes one.
    """
    maj = (n_total + 1) // 2
    minor = n_total // 2
    e_maj, e_min, h_maj, h_min = (int(c) for c in counts)
    return float(maj + e_maj - h_maj), float(minor + e_min - h_min)


def fermi_fill(eps: torch.Tensor, n_electrons: float, t_el: float = T_EL,
               tol: float = 1e-10, max_iter: int = 200) -> torch.Tensor:
    """Fermi-Dirac occupations at the `mu` that puts exactly `n_electrons` in the spectrum.

    Bisection, under `no_grad`. `mu` is a function of the eigenvalues, but the free energy's
    derivative does not contain it (Hellmann-Feynman at fixed N), so detaching is exact rather
    than approximate -- see the module docstring.
    """
    with torch.no_grad():
        e = eps.detach()
        lo = float(e.min()) - 50.0 * t_el - 1.0
        hi = float(e.max()) + 50.0 * t_el + 1.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            total = torch.sigmoid(-(e - mid) / t_el).sum()
            if float(total) > n_electrons:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        mu = 0.5 * (lo + hi)
    return torch.sigmoid(-(eps - mu) / t_el)


def free_energy(eps: torch.Tensor, n_electrons: float, t_el: float = T_EL) -> torch.Tensor:
    """`F(N) = sum_k f_k eps_k - T_el S(f)`, differentiable in `eps`."""
    f = fermi_fill(eps, n_electrons, t_el)
    fc = f.clamp(1e-12, 1.0 - 1e-12)
    entropy = -(fc * fc.log() + (1.0 - fc) * (1.0 - fc).log()).sum()
    return (f * eps).sum() - t_el * entropy


# --------------------------------------------------------------------------- the H builder


class SlaterKosterH(nn.Module):
    """Assembles the s+p Hamiltonian from Stage 2's radial scales and bounded corrections.

    Deliberately NOT a subclass of the spectral heads: those own a four-channel eigenproblem
    and a `mu_c` per channel, and this one owns a single spin-resolved spectrum with an
    occupation. Inheriting would carry the machinery Edit 4 exists to delete.
    """

    def __init__(self, num_elements: int, feature_dim: int, elem_dim: int = 8,
                 hidden: int = 64, d_ref: float = 2.8, decay_length: float = 1.0,
                 r_cut: float = 10.0, on_site_range: float = 1.0,
                 hop_range: float = 0.5) -> None:
        super().__init__()
        from mace.modules.defect_bounded import ETA_SS_SIGMA, HBAR2_OVER_M
        from mace.modules.defect_spectral import _mlp

        self.d_ref, self.decay_length, self.r_cut = float(d_ref), float(decay_length), \
            float(r_cut)
        self.on_site_range, self.hop_range = float(on_site_range), float(hop_range)
        self.elem = nn.Embedding(num_elements, elem_dim)

        # Harrison's universal coefficients. eta_ss_sigma is shared with Edit 3 so the two
        # stages cannot disagree about the scale they start from.
        eta = torch.tensor([ETA_SS_SIGMA, 1.84, 3.24, -0.81])
        scale = HBAR2_OVER_M / (self.d_ref ** 2)
        self.v0_raw = nn.Parameter(
            (eta * scale).reshape(1, 1, 4).repeat(num_elements, num_elements, 1))
        # Two on-site levels per species, one per shell.
        self.eps0 = nn.Parameter(torch.zeros(num_elements, 2))
        self.hop = _mlp([2 * feature_dim + 2 * elem_dim, hidden, hidden, len(BOND_TYPES)],
                        final_scale=0.05)
        self.site = _mlp([feature_dim + elem_dim, hidden, hidden, 2], final_scale=0.05)

    # ---------------------------------------------------------------- elements

    def v0(self, species_i, species_j):
        """Symmetric in the species pair. sp-sigma's antisymmetry lives in the SK block's
        sign convention, not in this table -- putting it here as well would apply it twice."""
        m = 0.5 * (self.v0_raw + self.v0_raw.transpose(0, 1))
        return m[species_i, species_j]

    def radial(self, r: torch.Tensor) -> torch.Tensor:
        x = (r / self.r_cut).clamp(max=1.0)
        return torch.exp(-(r - self.d_ref) / self.decay_length) * (1.0 - x ** 6) ** 2

    def integrals(self, feats_i, feats_j, r, species_i, species_j) -> torch.Tensor:
        """The four radial integrals per edge, [n_edges, 4]."""
        e_i, e_j = self.elem(species_i), self.elem(species_j)
        sym = torch.cat([feats_i + feats_j, (feats_i - feats_j).abs(),
                         e_i + e_j, (e_i - e_j).abs()], dim=-1)
        correction = 1.0 + self.hop_range * torch.tanh(self.hop(sym))
        return self.v0(species_i, species_j) * self.radial(r).unsqueeze(-1) * correction

    def on_site(self, feats, species, madelung: Optional[torch.Tensor] = None):
        """`[n_nodes, 2]`: the s and p levels.

        The Madelung shift is added to BOTH shells identically. It is the electrostatic
        potential at a site and has no angular-momentum dependence; giving the shells
        different shifts would be inventing a crystal-field term and calling it electrostatics.
        """
        e = self.elem(species)
        levels = self.eps0[species] + self.on_site_range * torch.tanh(
            self.site(torch.cat([feats, e], dim=-1)))
        if madelung is not None:
            levels = levels + madelung.reshape(-1, 1)
        return levels

    # ---------------------------------------------------------------- assembly

    def forward(self, node_feats, node_species, edge_index, edge_vector,
                madelung: Optional[torch.Tensor] = None,
                n_nodes: Optional[int] = None) -> torch.Tensor:
        """Dense `H`, `[4N, 4N]`, for ONE graph.

        Dense on purpose at training sizes: 4N <= 636 here, so `eigh` plus native autograd is
        both exact and cheap. The windowed shift-invert path with custom eigenvalue gradients
        is for the ladder, and it is validated against this before the ladder uses it -- never
        the other way round.
        """
        n = int(n_nodes if n_nodes is not None else node_feats.shape[0])
        dim = n * ORBITALS_PER_ATOM
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1).clamp_min(1e-9)
        direction = edge_vector / r.unsqueeze(-1)

        v = self.integrals(node_feats[src], node_feats[dst], r,
                           node_species[src], node_species[dst])
        blocks = sk_block(direction, v)                       # [n_edges, 4, 4]

        H = torch.zeros(dim, dim, device=node_feats.device, dtype=node_feats.dtype)
        rows = (src.reshape(-1, 1, 1) * ORBITALS_PER_ATOM
                + torch.arange(ORBITALS_PER_ATOM, device=src.device).reshape(1, -1, 1))
        cols = (dst.reshape(-1, 1, 1) * ORBITALS_PER_ATOM
                + torch.arange(ORBITALS_PER_ATOM, device=src.device).reshape(1, 1, -1))
        H = H.index_put((rows.expand(-1, 4, 4).reshape(-1),
                         cols.expand(-1, 4, 4).reshape(-1)),
                        blocks.reshape(-1), accumulate=True)

        levels = self.on_site(node_feats, node_species, madelung)    # [n, 2]
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
        H = H + torch.diag(diag)
        # Both directed edges are present in the neighbour list and the SK convention makes
        # the reversed block the transpose, so H is already symmetric; this makes it exact in
        # floating point without changing the gradient (a linear operation).
        return 0.5 * (H + H.transpose(0, 1))


def head_energy(eps_maj: torch.Tensor, eps_min: torch.Tensor, n_total: int,
                counts: Sequence[int], t_el: float = T_EL) -> torch.Tensor:
    """`E_head = sum_sigma [F_sigma(N_sigma) - F_sigma(N_sigma,neutral)]`.

    Identically zero at `counts = 0` for any spectrum and any parameters -- the reference is
    the same spectrum at the neutral fill, so the two terms cancel exactly rather than nearly.
    """
    n_maj_ref = float((n_total + 1) // 2)
    n_min_ref = float(n_total // 2)
    n_maj, n_min = spin_targets(n_total, counts)
    return ((free_energy(eps_maj, n_maj, t_el) - free_energy(eps_maj, n_maj_ref, t_el))
            + (free_energy(eps_min, n_min, t_el) - free_energy(eps_min, n_min_ref, t_el)))


def density_matrix(psi: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """`P = sum_k f_k psi_k psi_k^T`. Gauge-invariant under degeneracy, which individual
    eigenvectors are not."""
    return (psi * f.unsqueeze(0)) @ psi.transpose(0, 1)


def site_charges(p_now: torch.Tensor, p_ref: torch.Tensor, n_nodes: int,
                 amplitude: float = 1.0) -> torch.Tensor:
    """`q_i = a sum_{mu in i} (P - P_ref)_{mu mu}`, summing to `-Delta n` exactly.

    The sign is carried by P natively: adding an electron raises the diagonal, so `q_i` comes
    out negative for an added electron without anything here choosing a sign.
    """
    diff = torch.diagonal(p_now - p_ref)
    return -amplitude * diff.reshape(n_nodes, ORBITALS_PER_ATOM).sum(dim=-1)
