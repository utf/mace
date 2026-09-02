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

from typing import Dict, List, Optional, Sequence, Tuple

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


def resolve_fills(n_total: int, counts: Sequence[int], occupation=None):
    """`((N_maj, N_min), (N_maj_ref, N_min_ref))` -- the fill, and the neutral origin.

    One function because two places now need the same answer: `head_energy_hf` builds the
    energy as a difference between these two fills, and `CountingHead` builds the force
    response's density difference from the same pair. Duplicating the four lines is exactly
    how the two would come to disagree about what "the reference" is -- and then the response
    would be the gradient of a different quantity than the energy it is supposed to complete.
    """
    ref = (float((n_total + 1) // 2), float(n_total // 2))
    if occupation is None:
        now = spin_targets(n_total, counts)
    else:
        now = (float(occupation[0]), float(occupation[1]))
    return now, ref


def changed_level_index(n_total: int, counts: Sequence[int], occupation=None) -> int:
    """0-based index of the majority level whose OCCUPATION the counters changed.

    Not "the highest occupied level". For an added electron the two coincide; for a REMOVED
    one the highest occupied level sits below the vacated defect state, and reading it
    measures a valence level's properties instead -- which looks like a clean null rather
    than like a bug. `max(n_maj, n_maj_ref) - 1` is the changed level under either
    convention, and the frames in hand carry `(0, 0, 1, 0)`, a hole.
    """
    (n_maj, _), (n_maj_ref, _) = resolve_fills(n_total, counts, occupation)
    return int(round(max(n_maj, n_maj_ref))) - 1


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
    """`F(N) = sum_k f_k eps_k - T_el S(f)`, differentiable in `eps`.

    THE OCCUPATIONS ARE DETACHED, and that is not an approximation -- it is what makes the
    derivative correct. At fixed N the constrained free energy obeys `dF/deps_k = f_k`. Let
    the autograd flow through `f` as well, with `mu` frozen by the bisection, and the chain
    rule adds a spurious `-mu f_k (1 - f_k) / T_el`: the entropy term cancels the `eps df`
    term but leaves the `mu` piece behind, because `f` was varied at fixed `mu` rather than
    at fixed N.

    This was found by the HF-versus-dense validation, not by inspection. It is invisible on a
    spectrum symmetric about `mu = 0` -- which the first version of the gradient test used,
    so that test passed while the identity was wrong for every real spectrum.
    """
    f = fermi_fill(eps, n_electrons, t_el).detach()
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


class CountingHead(nn.Module):
    """The Edit 4 head, wearing the `SpectralOutput` interface.

    Presenting the same NamedTuple as the spectral heads is what lets `_carrier_head`, the
    training diagnostics, `evaluate` and `capture` keep working unchanged. The alternative --
    a second head interface -- is how `train` and `evaluate` came to disagree about the
    forward pass four times already.

    What the fields mean here, which is NOT what they meant before:

    * `delta_sr`   `E_head`, the free-energy difference from the neutral fill. Identically
                   zero at `counts = 0`.
    * `alpha`      the carrier density from the DENSITY-MATRIX difference, normalised per
                   graph so `N_eff = 1/sum alpha^2` still means participation. Broadcast
                   across the four channel slots: the counting head has no channels, and
                   pretending otherwise is what `channel_sign` was.
    * `gap`        `eps_{N+1} - eps_N`, the frontier gap. This is what `loss_gap` consumes.
    * `site_energy` the on-site levels (s shell), for the gauge diagnostics.

    Both spins share one Hamiltonian and differ only in their fill, so `eigh` runs once per
    graph rather than twice.
    """

    def __init__(self, num_elements: int, feature_dim: int, atomic_numbers,
                 elem_dim: int = 8, hidden: int = 64, d_ref: float = 2.8,
                 decay_length: float = 1.0, r_cut: float = 10.0,
                 t_el: float = T_EL, num_channels: int = 4) -> None:
        super().__init__()
        self.h = SlaterKosterH(num_elements=num_elements, feature_dim=feature_dim,
                               elem_dim=elem_dim, hidden=hidden, d_ref=d_ref,
                               decay_length=decay_length, r_cut=r_cut)
        self.t_el = float(t_el)
        self.num_channels = int(num_channels)
        # Valence per SPECIES INDEX, resolved once from the model's own atomic-number table.
        # Looking it up by Z at every forward would put a python dict in the hot path and,
        # worse, would silently accept a species the table does not cover.
        zs = [int(z) for z in atomic_numbers]
        missing = [z for z in zs if z not in VALENCE]
        if missing:
            raise ValueError(
                f"no valence recorded for Z = {missing}; the counting head cannot fill a "
                "band it does not know the electron count of")
        self.register_buffer("valence",
                             torch.tensor([VALENCE[z] for z in zs], dtype=torch.long))
        # Diagnostics read this to size their spectra; the counting head returns the whole
        # spectrum, so it is 4 orbitals per atom rather than a truncation.
        self.num_states = -1
        # One scalar, uniform across sites. It moves E_head by c * Delta_n and nothing else
        # -- the role mu_c used to play, without mu_c's per-channel bookkeeping. Calibrated
        # once on an init batch against the median energy target, then trainable.
        self.c_shift = nn.Parameter(torch.zeros(()))
        # Read by `_carrier_head` to decide whether to hand this head `positions` and a
        # `force_out` dict. A capability flag rather than an isinstance check: the model must
        # not import the head module to know what its own head can do.
        self.wants_positions = True

    def forward(self, node_feats, counter_emb, counts, batch, num_graphs, edge_index,
                edge_length, site_bias=None, node_species=None, clamp_mask=None,
                edge_vector=None, madelung=None, occupations=None, internals=None,
                positions=None, force_out=None):
        """`occupations`, when given, is [n_graphs, 2] holding (N_maj, N_min) directly.

        Stage 4's interface, and it is an INPUT change rather than an architecture one: the
        counters already map to a fill, and this simply lets a caller state the fill instead.
        That is what makes excited configurations and non-Aufbau occupations expressible
        without a second code path -- the thing the four-channel counter scheme could never
        represent. Ground-state fill remains the default.
        """
        from mace.modules.defect_spectral import SpectralOutput

        if node_species is None or edge_vector is None:
            raise ValueError("the counting head needs node_species and edge_vector")
        # Same detach boundary as V3 and Stages 1-2: no head term carries gradient into the
        # trunk under any run configuration. Done once at entry so a term added later cannot
        # reconnect it -- there is no attached descriptor in scope below.
        node_feats = node_feats.detach()
        device, dtype = node_feats.device, node_feats.dtype
        n_nodes = int(node_feats.shape[0])

        levels = self.h.on_site(node_feats, node_species, madelung) + self.c_shift
        if clamp_mask is not None:
            # DIAGNOSTIC ONLY, same contract as the spectral heads: sites outside the mask are
            # pushed far above the frontier so no occupied state can live on them.
            levels = torch.where(clamp_mask.reshape(-1, 1), levels,
                                 levels + 1.0e3)

        alpha = torch.zeros(n_nodes, self.num_channels, device=device, dtype=dtype)
        delta = torch.zeros(num_graphs, device=device, dtype=dtype)
        gaps = torch.zeros(num_graphs, device=device, dtype=dtype)
        spectra, per_graph_nodes = [], []
        # Section 1: the force response. Collected per graph, contracted once after the loop.
        resp_h: List[torch.Tensor] = []
        resp_d: List[torch.Tensor] = []

        src, dst = edge_index[0], edge_index[1]
        for g in range(num_graphs):
            node_sel = (batch == g).nonzero(as_tuple=True)[0]
            n_g = int(node_sel.numel())
            if n_g == 0:
                spectra.append(torch.zeros(1, device=device, dtype=dtype))
                per_graph_nodes.append(node_sel)
                continue
            remap = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
            remap[node_sel] = torch.arange(n_g, device=device)
            edge_sel = (batch[src] == g).nonzero(as_tuple=True)[0]

            H = self.h(node_feats[node_sel], node_species[node_sel],
                       torch.stack([remap[src[edge_sel]], remap[dst[edge_sel]]]),
                       edge_vector[edge_sel],
                       madelung=None, n_nodes=n_g)
            # The on-site term is applied here rather than inside `self.h` so the clamp above
            # -- which is diagnostic and must never reach the Hamiltonian builder -- has a
            # single place to act.
            diag = torch.cat([levels[node_sel][:, :1],
                              levels[node_sel][:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
            H = H - torch.diag(torch.diagonal(H)) + torch.diag(diag)

            n_total = int(self.valence[node_species[node_sel]].sum())
            c = counts[g].tolist() if counts.dim() > 1 else counts.tolist()
            occ = None if occupations is None else occupations[g]
            resp_bucket = {} if force_out is not None else None
            e_head, lam, psi, p_now, p_ref = head_energy_hf(
                H, n_total, c, self.t_el, occupation=occ, response_out=resp_bucket)
            delta[g] = e_head

            # SECTION 1, THE DENSITY RESPONSE IN THE FORCE GRADIENT.
            #
            # The energy above is built on DETACHED occupations, which is exact: at fixed N
            # the free energy is stationary in the fill, so `dE/dR = Tr((P - P_ref) dH/dR)`
            # with P held fixed. What that route drops is `dP/dtheta` in the force loss's
            # PARAMETER gradient -- the optimiser searches as if the density could not respond
            # to a change in the elements. That is what zeroes the hopping gradient at an
            # atomic-limit initialisation, and what would stop E_LR from ever teaching the
            # head where to put the carrier.
            #
            # The force is LINEAR in its cotangent, so
            #
            #     F(D) - F(D.detach()) = F(D - D.detach())
            #
            # and `D - D.detach()` is EXACTLY zero in value while carrying dD/dtheta. Adding
            # its contraction to the forces therefore cannot move a single force component --
            # that is an identity, not a tolerance -- while contributing precisely the missing
            # `-Tr(dD/dtheta . dH/dR)` to the gradient. The already-present autograd route
            # supplies the other half, `-Tr(D d2H/dR dtheta)`.
            #
            # Deleting this as a no-op is the obvious future mistake. The test that it is not
            # one is `test_wiring_changes_gradient_not_forces`.
            if resp_bucket:
                dens = resp_bucket["density_difference"]
                resp_h.append(H)
                resp_d.append(dens - dens.detach())

            # EIGENVALUES CARRY THE GRADIENT, EIGENVECTORS DO NOT. `eigh`'s backward builds
            # the eigenvector term with 1/(lam_i - lam_j) factors, which is NaN at exact
            # degeneracy -- and a 316-state spectrum in a ~17 eV span has degeneracies for
            # certain. It produced NaN on the first real batch.
            #
            # The energy needs only eigenvalues (Hellmann-Feynman), and the density matrix is
            # a DIAGNOSTIC readout here, so detaching psi removes the divergent path entirely
            # rather than regularising it. This is the concrete form of "never backprop
            # through individual eigenvectors".
            #
            # NOTE for the joint run: with E_LR enabled, `alpha` feeds q_carrier, and a
            # detached alpha cuts that gradient. Decide there whether q_carrier needs a
            # differentiable density -- if so it needs a matrix-function route, not eigh.
            spectra.append(lam.detach())
            per_graph_nodes.append(node_sel)
            n_maj_ref = float((n_total + 1) // 2)
            q = site_charges(p_now, p_ref, n_g)   # sums to -Delta n exactly
            mass = q.abs()
            total = mass.sum()
            alpha[node_sel] = (mass / total).unsqueeze(-1).expand(-1, self.num_channels) \
                if float(total) > 0 else 0.0

            k = min(max(int(round(n_maj_ref)) - 1, 0), lam.numel() - 2)
            gaps[g] = lam[k + 1] - lam[k]

        if force_out is not None and resp_h:
            if positions is None or not positions.requires_grad:
                raise ValueError(
                    "the counting head was asked for a force response but positions are "
                    "absent or detached; returning nothing here would silently train the "
                    "frozen-density gradient while the configuration says otherwise")
            grad = torch.autograd.grad(
                resp_h, [positions], grad_outputs=resp_d, create_graph=True,
                retain_graph=True, allow_unused=True)[0]
            if grad is None:
                raise RuntimeError(
                    "the head Hamiltonian does not depend on positions; the force response "
                    "-- and every head force already in the model -- would be identically "
                    "zero, so this fails rather than returning a plausible zero")
            force_out["force_response"] = -grad

        m = max(int(s.numel()) for s in spectra)
        lam_pad = torch.full((num_graphs, self.num_channels, m), 1.0e3,
                             device=device, dtype=dtype)
        for g, s in enumerate(spectra):
            lam_pad[g, :, : s.numel()] = s.unsqueeze(0)

        site = levels[:, :1].expand(-1, self.num_channels)
        counts_per_graph = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
        eps_mean = (torch.zeros(num_graphs, self.num_channels, device=device, dtype=dtype)
                    .index_add_(0, batch, site) / counts_per_graph.unsqueeze(-1))

        if internals is not None:
            internals["lam"] = lam_pad
            internals["eps"] = site
            internals["eps_raw"] = site
            internals["batch"] = batch
            # NOT under the key "H": the spectral heads' H is [G, C, n_sites, n_sites] and
            # this one is [4N, 4N] per graph. A consumer that indexes by site would read
            # orbitals instead and get a plausible wrong answer, so it fails loudly instead.
            internals["H_orbital"] = spectra

        return SpectralOutput(
            delta_sr=delta, alpha=alpha, site_energy=site,
            gap=gaps.unsqueeze(-1).expand(-1, self.num_channels),
            eps_mean=eps_mean, eigenvalues=lam_pad,
            weights=torch.zeros_like(lam_pad))


def head_energy_hf(H: torch.Tensor, n_total: int, counts: Sequence[int],
                   t_el: float = T_EL, occupation=None, response_out=None):
    """`E_head` by the Hellmann-Feynman route. Same value as `head_energy`, usable gradient.

    WHY THIS EXISTS. Fitting FORCES means backpropagating through a quantity that is itself
    `dE/dR`, so the loss needs the SECOND derivative of the eigenvalues. `torch.linalg.eigh`'s
    double backward builds that from eigenvector response with `1/(lam_i - lam_j)` factors,
    and a 316-state spectrum spanning ~17 eV is degenerate to numerical precision in several
    places. It returns NaN on the first real batch -- not occasionally, immediately.

    The fix is the standard band-structure force expression rather than a regulariser:

        F = Tr(P H) - T_el S,     P = sum_k f_k psi_k psi_k^T

    with `P` and `S` evaluated at the current spectrum and held FIXED. Then

        dF/dx = Tr(P dH/dx)

    which is Hellmann-Feynman, exact, and involves no eigenvector response at all -- so the
    second derivative needs only `d^2H/dR dtheta`, which is well conditioned everywhere.

    WHAT IS APPROXIMATED, stated rather than buried: the VALUE and the FORCE are exact. What
    is dropped is `dP/dR` in the loss's *parameter* gradient — the optimiser searches as if
    the occupations were frozen at their current values. That is the same frozen-density
    convention DFTB force training uses, and it is why `head_energy` (the plain autograd
    version) is kept: the two must agree on energies, forces and site charges, which is what
    `test_counting_head.py` asserts.

    THE EIGENSOLVE RUNS IN FLOAT64 UNCONDITIONALLY. In float32 this head returns NaN -- not
    at some size threshold, but on ordinary frames of any size, intermittently. The spectrum
    spans ~17 eV across hundreds of states with `T_el = 25 meV`, so the occupation is a
    sigmoid of `(lam - mu)/T` with arguments of order 700: float32 has neither the range for
    the exponentials nor the precision for a bisection that must place `mu` to 1e-10 of a
    fixed electron count. The parent spectral head already carries a `solver_dtype` for the
    same reason; this is that precedent applied here.
    """
    in_dtype = H.dtype
    h_in = H
    H = H.double()
    lam, psi = torch.linalg.eigh(H)
    lam_d, psi_d = lam.detach(), psi.detach()

    # Stage 4 lets the caller state the fill. The REFERENCE stays the neutral ground state,
    # so E_head is still the difference from the same origin and still vanishes when the
    # override happens to equal it.
    (n_maj, n_min), (n_maj_ref, n_min_ref) = resolve_fills(n_total, counts, occupation)

    def piece(n_electrons):
        f = fermi_fill(lam_d, n_electrons, t_el)
        p = density_matrix(psi_d, f)
        fc = f.clamp(1e-12, 1.0 - 1e-12)
        entropy = -(fc * fc.log() + (1.0 - fc) * (1.0 - fc).log()).sum()
        return (p * H).sum() - t_el * entropy, p, f

    e_maj, p_maj, f_maj = piece(n_maj)
    e_min, p_min, f_min = piece(n_min)
    e_maj_ref, p_maj_ref, f_maj_ref = piece(n_maj_ref)
    e_min_ref, p_min_ref, f_min_ref = piece(n_min_ref)
    energy = ((e_maj - e_maj_ref) + (e_min - e_min_ref)).to(in_dtype)

    # SECTION 1. The same `P - P_ref`, built as a DIFFERENTIABLE function of `H` for the
    # force response. Here rather than in the head because the spectrum and the four fills
    # are already in hand: the alternative diagonalises the same matrix a second time.
    # Absent when the fill equals the reference -- a neutral frame has no response, and the
    # difference would be an identity zero in the gradient as well as the value.
    if response_out is not None and (n_maj, n_min) != (n_maj_ref, n_min_ref):
        response_out["density_difference"] = fermi_density_difference(
            h_in, (n_maj, n_min, n_maj_ref, n_min_ref), (1.0, 1.0, -1.0, -1.0), t_el,
            spectrum=(lam_d, psi_d),
            occupations=torch.stack([f_maj, f_min, f_maj_ref, f_min_ref]))
    # SUM the spin channels, do not average: the monopole identity `sum_i q_i = -Delta n`
    # is over all electrons, and averaging halves it. Caught by the validation test.
    return (energy, lam.to(in_dtype), psi_d.to(in_dtype),
            (p_maj + p_min).to(in_dtype), (p_maj_ref + p_min_ref).to(in_dtype))


def scc_energy(gamma: torch.Tensor, delta_q: torch.Tensor) -> torch.Tensor:
    """Edit 5's second-order term, `0.5 * sum_ij Gamma_ij dq_i dq_j`. INTERFACE ONLY.

    Reserved and not implemented in the sense that matters: nothing calls it, no model builds
    a `Gamma`, and it is not in any energy. What it fixes is the SHAPE of the eventual
    contract -- `Gamma` is [n_sites, n_sites] and `delta_q` is the site charge this module
    already produces via `site_charges`, so a later implementation cannot quietly redefine
    either.

    Self-consistency is the part that is genuinely absent: a real SCC loop would recompute
    `delta_q` from an H that already contains this term, iterating to convergence. That
    changes the forward pass from one eigensolve to several and needs its own gradient
    treatment, which is why it is a separate edit rather than a few lines here.
    """
    return 0.5 * torch.einsum("ij,i,j->", gamma, delta_q, delta_q)


class _FermiDensityMatrix(torch.autograd.Function):
    """`P = f(H)` at fixed electron number, with its exact Frechet derivative.

    THE TERM THIS RESTORES. The force-loss parameter gradient is

        dl/dtheta = Tr(P d2H/dR dtheta) + Tr(dP/dtheta . dH/dR)

    and the second term -- the density response -- is dropped when `P` is detached. Two
    consequences, both measured rather than supposed. It zeroes the hopping gradient wherever
    the bond order is zero, which is exactly an atomic-limit initialisation, so a seed that
    starts there has no gradient with which to leave: the failure-to-start. And it zeroes the
    carrier-density gradient, so once E_LR is enabled the long-range branch cannot teach the
    head where to put the carrier.

    (Stated precisely: relocation still happens by DRIFT, because each forward recomputes P
    from the current H. What is absent is relocation-from-forces in the GRADIENT. That is why
    Stage 3's converged seeds learned at all.)

    WHY THIS IS SAFE WHERE `eigh`'s BACKWARD IS NOT. The eigenvector backward carries
    `1/(lam_i - lam_j)`, unbounded as a gap closes. The Daleckii-Krein derivative of a SMOOTH
    matrix function carries the divided difference

        L_ij = (f_i - f_j) / (lam_i - lam_j)   ->   f'((lam_i + lam_j)/2)  as lam_i -> lam_j

    which is bounded by `1/(4 T_el)` everywhere, including at exact degeneracy. `P` is
    gauge-invariant under rotations within a degenerate subspace; individual eigenvectors are
    not, and nothing here differentiates them.

    THE FIXED-N CORRECTION. `mu` is not a constant: it moves to hold `sum_k f_k = N`. Its
    response contributes

        dmu = (sum_k f'_k dlam_k) / (sum_k f'_k)

    which adds `-f'_i (sum_j f'_j Ghat_jj) / (sum_k f'_k)` to the diagonal of the transformed
    cotangent. When `mu` sits in a gap much wider than `T_el` every `f'_k` vanishes, the
    denominator goes to zero, and the correction is genuinely absent rather than singular --
    the electron count does not constrain `mu` there. Guarded on the denominator, not on a
    gap estimate.
    """

    @staticmethod
    def forward(ctx, H, n_electrons, t_el, degeneracy_tol):
        lam, U = torch.linalg.eigh(H.double())
        f = fermi_fill(lam, float(n_electrons), float(t_el))
        P = (U * f.unsqueeze(0)) @ U.transpose(-1, -2)
        ctx.save_for_backward(lam, U, f)
        ctx.t_el = float(t_el)
        ctx.tol = float(degeneracy_tol)
        ctx.in_dtype = H.dtype
        return P.to(H.dtype)

    @staticmethod
    def backward(ctx, grad_P):
        lam, U, f = ctx.saved_tensors
        dH = _dk_backward(lam, U, f, ctx.t_el, ctx.tol, grad_P.double())
        return dH.to(ctx.in_dtype), None, None, None


def _dk_eigenbasis(lam: torch.Tensor, f: torch.Tensor, t_el: float, tol: float,
                   Ghat: torch.Tensor) -> torch.Tensor:
    """One fill's Daleckii-Krein map, already in the eigenbasis: `Ghat` in, `M` out.

    Split from the basis round trip because the multi-fill Function applies FOUR of these to
    ONE `Ghat` and returns one `U M U^T`. Rotating in and out per fill instead costs three
    extra 636x636 float64 GEMMs each, which measured as a quarter of the wiring's overhead.
    """
    fp = -f * (1.0 - f) / t_el                      # f'(lam), bounded by 1/(4 T)
    dl = lam.unsqueeze(-1) - lam.unsqueeze(-2)
    df = f.unsqueeze(-1) - f.unsqueeze(-2)
    near = dl.abs() <= tol
    # The divided difference away from coincidence, its limit at it. `torch.where` alone
    # would still evaluate the singular branch and poison the gradient with NaN, so the
    # denominator is made safe BEFORE the division.
    safe = torch.where(near, torch.ones_like(dl), dl)
    mid = 0.5 * (lam.unsqueeze(-1) + lam.unsqueeze(-2))
    fmid = torch.sigmoid(-(mid - _mu_from(lam, f, t_el)) / t_el)
    L = torch.where(near, -fmid * (1.0 - fmid) / t_el, df / safe)

    M = L * Ghat
    denom = fp.sum(-1)
    if bool((denom.abs() > 1e-12).all()):
        num = (fp * torch.diagonal(Ghat, dim1=-2, dim2=-1)).sum(-1)
        M = M - torch.diag_embed(fp * (num / denom).unsqueeze(-1))
    return M


def _dk_backward(lam: torch.Tensor, U: torch.Tensor, f: torch.Tensor, t_el: float,
                 tol: float, G: torch.Tensor) -> torch.Tensor:
    """One fill's Daleckii-Krein pullback: a cotangent on `P` becomes one on `H`."""
    # H is symmetric, so only the symmetric part of the cotangent can act on it.
    G = 0.5 * (G + G.transpose(-1, -2))
    Ghat = U.transpose(-1, -2) @ G @ U
    return U @ _dk_eigenbasis(lam, f, t_el, tol, Ghat) @ U.transpose(-1, -2)


class _FermiDensitySum(torch.autograd.Function):
    """`D = sum_a s_a f_{N_a}(H)` -- several fills of ONE Hamiltonian, differentiable in H.

    The force response needs `P - P_ref` summed over both spins: four fills of the same `H`.
    The eigendecomposition and the occupations are taken as ARGUMENTS rather than recomputed,
    because the caller in the hot path -- `head_energy_hf` -- has just built both from this
    exact `H` for the energy. Recomputing them costs a float64 `eigh` of a 636x636 matrix and
    four bisections per graph per step, which measured as a third of the wiring's overhead.

    `lam`, `U` and `occ` are DETACHED, non-differentiable inputs and must belong to `H`. The
    contract is kept by construction, not by trust: the only hot-path caller builds all four
    inside one function from one matrix. `fermi_density_difference` computes its own when the
    caller has none, which is what the equivalence tests exercise.

    Forward collapses to a single GEMM: `sum_a s_a U diag(f_a) U^T = U diag(sum_a s_a f_a) U^T`.
    The backward does not collapse -- each fill has its own divided-difference matrix -- but it
    shares the basis round trip, applying all four maps to one `Ghat` and rotating out once.
    """

    @staticmethod
    def forward(ctx, H, lam, U, occ, signs, t_el, degeneracy_tol):
        sgn = torch.tensor(signs, dtype=occ.dtype, device=occ.device)
        weighted = (occ * sgn.unsqueeze(-1)).sum(0)
        acc = (U * weighted.unsqueeze(-2)) @ U.transpose(-1, -2)
        ctx.save_for_backward(lam, U, occ)
        ctx.signs = tuple(float(s) for s in signs)
        ctx.t_el = float(t_el)
        ctx.tol = float(degeneracy_tol)
        ctx.in_dtype = H.dtype
        return acc.to(H.dtype)

    @staticmethod
    def backward(ctx, grad_D):
        lam, U, occ = ctx.saved_tensors
        G = grad_D.double()
        G = 0.5 * (G + G.transpose(-1, -2))
        Ghat = U.transpose(-1, -2) @ G @ U
        M = None
        for sign, f in zip(ctx.signs, occ):
            term = sign * _dk_eigenbasis(lam, f, ctx.t_el, ctx.tol, Ghat)
            M = term if M is None else M + term
        dH = U @ M @ U.transpose(-1, -2)
        return dH.to(ctx.in_dtype), None, None, None, None, None, None


def _mu_from(lam: torch.Tensor, f: torch.Tensor, t_el: float) -> torch.Tensor:
    """Recover `mu` from the occupations, for the degenerate-limit branch.

    Read back rather than threaded through: it is exact wherever `f` is not saturated, and the
    branch it feeds only fires for eigenvalue pairs that have collided -- which cannot all be
    saturated, or the divided difference would be zero either way.
    """
    interior = (f > 1e-6) & (f < 1.0 - 1e-6)
    if not bool(interior.any()):
        return lam.median()
    x = lam[interior] + t_el * torch.log(f[interior] / (1.0 - f[interior]))
    return x.median()


def fermi_density_matrix(H: torch.Tensor, n_electrons: float, t_el: float = T_EL,
                         degeneracy_tol: float = 1e-7) -> torch.Tensor:
    """`P = sum_k f_k psi_k psi_k^T` at fixed `N`, differentiable in `H`."""
    return _FermiDensityMatrix.apply(H, n_electrons, t_el, degeneracy_tol)


def fermi_density_difference(H: torch.Tensor, fills: Sequence[float],
                             signs: Sequence[float], t_el: float = T_EL,
                             degeneracy_tol: float = 1e-7, spectrum=None,
                             occupations=None) -> torch.Tensor:
    """`sum_a s_a P(N_a)` for several fills of one `H`, differentiable in `H`.

    `spectrum` is `(lam, U)` in float64 and `occupations` is `[n_fills, n]`, both of THIS `H`.
    Supplied by the head, which has them already; computed here when they are not.
    """
    lam, u = torch.linalg.eigh(H.double()) if spectrum is None else spectrum
    if occupations is None:
        occupations = torch.stack([fermi_fill(lam, float(n), float(t_el)) for n in fills])
    return _FermiDensitySum.apply(H, lam, u, occupations, tuple(signs), t_el,
                                  degeneracy_tol)


# Harrison solid-state-table atomic term values, eV, by atomic number: (eps_s, eps_p).
# Tabulated free-atom values -- host- and defect-agnostic, and not fitted here.
#
# What they buy: the anion p level sits BELOW both cation p levels
# (Cl -11.74 < Pb -8.04 < Cs -1.80) without touching Z, so the valence band comes out
# anion-derived at initialisation. Starting instead from eps0 = 0 makes every site degenerate
# -- the atomic limit, where the bond order vanishes and, on the frozen-P gradient, nothing
# could move the hoppings at all.
HARRISON_TERMS: Dict[int, Tuple[float, float]] = {
    17: (-24.63, -11.74),      # Cl 3s, 3p
    55: (-3.36, -1.80),        # Cs 6s, 6p
    82: (-15.19, -8.04),       # Pb 6s, 6p
}
# Harrison universal coefficients, in BOND_TYPES order.
HARRISON_ETA = (-1.40, 1.84, 3.24, -0.81)


def harrison_initialise(head, atomic_numbers: Sequence[int],
                        bond_length: float = 2.8) -> None:
    """Set on-site levels and hopping scales from the Harrison tables, in place.

    On-sites are the tabulated free-atom term values per species and shell. Hoppings are
    `eta_b * hbar^2 / (m d^2)` at the measured bond length, per bond type.

    `bond_length` is the MEASURED median nearest-neighbour distance of the data, not a
    constant: the universal scaling is a function of the actual bond length, and passing a
    material-specific number in here rather than baking one into the module keeps the head
    host-agnostic.
    """
    zs = [int(z) for z in atomic_numbers]
    missing = [z for z in zs if z not in HARRISON_TERMS]
    if missing:
        raise ValueError(
            f"no Harrison term values recorded for Z = {missing}; the counting head would "
            "fall back to a degenerate atomic-limit initialisation, which is the failure "
            "this function exists to prevent")
    with torch.no_grad():
        for i, z in enumerate(zs):
            eps_s, eps_p = HARRISON_TERMS[z]
            head.h.eps0[i, 0] = eps_s
            head.h.eps0[i, 1] = eps_p
        scale = HBAR2_OVER_M_COUNTING / (float(bond_length) ** 2)
        eta = torch.tensor(HARRISON_ETA, dtype=head.h.v0_raw.dtype,
                           device=head.h.v0_raw.device)
        head.h.v0_raw.copy_(
            (eta * scale).reshape(1, 1, 4).expand_as(head.h.v0_raw).clone())


HBAR2_OVER_M_COUNTING = 7.62      # eV A^2; same constant Edit 3 uses, named here to avoid
                                  # a cross-module import in a hot path


def initialisation_gate(lam: torch.Tensor, n_electrons: float, e_gap: float,
                        t_el: float = T_EL) -> Dict[str, float]:
    """The step-0 sanity gate on a PRISTINE spectrum: bands, not atoms.

    Two conditions, both from the plan:

    * edge spacing <= E_gap / 2 at BOTH edges -- the levels either side of the frontier must
      be closer together than the gap they are supposed to bracket. In the atomic limit the
      spacing is of order the term-value differences (tens of eV) and this fails immediately.
    * bandwidth >= 2 E_gap -- there has to be a band at all.

    A gate, not a filter: a trip means re-initialise AND report. A seed that drifts atomic
    during training is a reportable failure mode, not something to discard quietly.
    """
    lam = torch.sort(lam.detach().reshape(-1)).values
    n = int(round(float(n_electrons)))
    n = max(1, min(n, int(lam.numel()) - 1))
    below = float(lam[n - 1] - lam[n - 2]) if n >= 2 else float("inf")
    above = float(lam[n + 1] - lam[n]) if n + 1 < lam.numel() else float("inf")
    bandwidth = float(lam[-1] - lam[0])
    ok = (below <= 0.5 * e_gap) and (above <= 0.5 * e_gap) and (bandwidth >= 2.0 * e_gap)
    return dict(edge_spacing_below=below, edge_spacing_above=above, bandwidth=bandwidth,
                frontier_gap=float(lam[n] - lam[n - 1]), passed=bool(ok))


def head_forces(H: torch.Tensor, positions: torch.Tensor, P: torch.Tensor,
                P_ref: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """`F = -Tr((P - P_ref) dH/dR)`, with the density response live in the theta gradient.

    `grad_outputs` is the whole trick. `autograd.grad(H, R, grad_outputs=D)` contracts to
    `sum_ab D_ab dH_ab/dR` -- exactly the force -- and under `create_graph` the result keeps a
    graph through BOTH factors, so

        dF/dtheta = Tr(dD/dtheta . dH/dR) + Tr(D . d2H/dR dtheta)

    with the first term, the density response, present. Building `dH/dR` explicitly would be a
    [4N, 4N, N, 3] tensor -- 1.9e8 entries at 159 atoms -- and is not the way.

    `D = P - P_ref` rather than `P`: the head's energy is a DIFFERENCE from the neutral fill,
    so its force is the difference of the two Hellmann-Feynman terms. Using `P` alone would
    return the force of the whole valence manifold, which the base potential already carries.
    """
    D = (P - P_ref).to(H.dtype)
    grad = torch.autograd.grad(H, positions, grad_outputs=D, create_graph=create_graph,
                               retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(positions)
    return -grad
