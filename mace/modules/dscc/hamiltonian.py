"""Plan section 2.1: the runtime Hamiltonian `H0(R; theta) = H_SK + H_onsite_scalar +
H_onsite_dir`, real symmetric, `[4N, 4N]` (s, px, py, pz per atom), spin-independent.

`H_SK` and `H_onsite_scalar` are the retained `defect_counting.SlaterKosterH` (plan section
3, "keep"): Slater-Koster `ss_sigma, sp_sigma, pp_sigma, pp_pi` per species pair with the
Harrison initialisation at each pair's reference bond length, four learned positive decay
lengths (`L_b = L0 exp(ln 2 tanh u_b)`), the bounded log-form modulation from the frozen
first-block base scalars (`exp(beta tanh g)`), the smooth cutoff taper; species/shell
baselines plus bounded corrections `delta_Z tanh(e_Z(h_i))` -- v5 amendment A1: no pristine
reference anywhere, a bias in the readout in its place, and a weak L2 on the tanh output
pulling the correction toward the element default. No unbounded constant channel.

`H_onsite_dir = a_i (v_i . T_sp_i) + b_i (Q_i : T_pp_i)` is new: `v_i` is a per-species
linear combination of the base's `l = 1` first-block features (polar vectors, with
position derivatives through the recomputed block), entering the on-site s-p block
symmetrically; `Q_i = sum_j w(r_ij) (rhat rhat^T - I/3)` is the geometric traceless
quadrupole of the neighbourhood with a smooth cutoff `w` (zero at centrosymmetric sites),
entering the on-site p-p block. `a_Z = a_max tanh(alpha[Z])`, `b_Z = b_max tanh(beta[Z])`
are bounded species-level coefficients, and under A1 each carries an optional bounded
environment factor from its own readout,

    b_i = b_max tanh( beta_Z + beta_b tanh(g_Z(h_i)) ),     a_i likewise,

reference-free like the scalar term, with the environment's bounded shift INSIDE the tanh
rather than multiplying the species coefficient (see `site_coefficients` for why: the
multiplicative form's readout gradient is proportional to the coefficient it modulates, and
that coefficient starts at zero). `beta_b = beta_a = 0` recovers the species-level
coefficients exactly and is the registered W3 setting; W4's factorial moves them
independently. With `directional=False` the block is identically absent: the scalar-only
control of Arm 1.

Gauge: `H0 -> H0 + a I` is not a parameter of this module; nothing here can add a constant
to every level except through the bounded corrections, and a constant added to every level
of every atom is a shift the energy expression is invariant to (A1's second argument for
dropping the centre).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Sequence

import torch
from torch import nn

from mace.modules.dscc.legacy import (DELTA_FRACTION_DEFAULT, HOP_LOG_BETA_DEFAULT,
                                      ORBITALS_PER_ATOM, R_S3_DEFAULT, SlaterKosterH, _mlp,
                                      harrison_initialise)

# Registered defaults (plan section 11: to register before use). `r_cut` is the old head's
# 10 A; `q_cut` reaches the first shell and the Cs neighbours (Pb-Cl 2.8, Cs-Cl 3.5-4.1 A).
N_RADIAL_DEFAULT = 6           # O3: the head's own radial basis size
R_CUT_DEFAULT = 10.0
Q_CUT_DEFAULT = 4.5
A_MAX_DEFAULT = 1.0       # eV per unit |v|
B_MAX_DEFAULT = 1.0       # eV per unit |Q|
ON_SITE_RANGE_DEFAULT = 3.0
# A1 (registered 2026-09-10). `eta`: the SK modulation bound, `exp(eta tanh m)`. `beta_b`,
# `beta_a`: the bounds on the rank-2 and rank-1 environment factors, ZERO for W3 (the
# species-level coefficients) and moved by W4's factorial. `delta_frac`: `Delta_Z` as a
# fraction of the spread of the species onsite baselines. `readout_hidden`: one hidden layer
# for `g_Z` and `f_Z`, A1's "one hidden layer of registered width".
#
# `eta` RULED BACK TO ln 3 (user, 2026-09-10 13:40), reverting the amendment's literal
# `eta = 0.5`. A1's annotation on that line is "(unchanged form, reference removed)" and the
# SK modulation never carried a reference, so nothing on it needed changing; 0.5 also lands
# near `HOP_LOG_BETA_RANGE_EQUIVALENT` (ln 1.5), roughly undoing the Stage A' widening that a
# measurement had asked for -- the cohort sat AT the narrower stop on the vacancy-flanking
# Pb-Pb bond, where `sech^2 ~ 0` makes a parameter look like it is learning when it is not.
# A1's other registered values (beta, Delta_Z, the readouts, the L2) stand unchanged.
ETA_DEFAULT = HOP_LOG_BETA_DEFAULT
# A1.1: the readout final layer at 0.1x the default initialisation, so corrections start near
# zero and grow. The campaign before A1.1 used 0.05 on raw (unstandardised) inputs; on
# standardised inputs 0.1 is the registered value and the initial correction is still well
# under 0.1 eV.
READOUT_INIT_SCALE = 0.1
# THE RANK-1 DEADLOCK, found 2026-09-11 and fixed here.
#
# `sp_block = a_Z * (vector_mix[Z] . vectors)`. Both `alpha` (through `a_Z = a_max tanh
# alpha`) and `vector_mix` were zero-initialised, and each one's gradient is proportional to
# the OTHER: d/d alpha goes as `vector_mix`, d/d vector_mix goes as `a_Z`. Zero times zero is
# a saddle the optimiser can never leave, so the rank-1 s-p block was identically zero after
# 60 epochs in EVERY production run of the campaign (measured: |vector_mix| and |alpha| both
# exactly 0.0 in the converged W3 Phi = 0 and B' heads). `beta`/`b_Z` escaped only because it
# multiplies the GEOMETRIC quadrupole `Q_i`, which is nonzero whatever the parameters do.
#
# No test caught it because the fixtures break the deadlock by hand
# (`test_hamiltonian._model` sets `vector_mix.normal_(0, 0.5)` and `alpha.fill_(0.7)`).
#
# The fix initialises `vector_mix` off zero, fan-in scaled, and LEAVES `alpha` AT ZERO. The
# block therefore still starts exactly absent -- `a_Z = 0` makes the s-p block identically
# zero, so the neutral null and every Phase-1 gate are untouched -- but `d/d alpha` is now
# nonzero and the term can be learned, which is what "starts absent and is learned" was
# always meant to mean.
VECTOR_MIX_INIT = 1.0        # in units of 1/sqrt(n_vectors)
BETA_ENV_DEFAULT = 0.5
READOUT_HIDDEN_DEFAULT = 64
ELEM_DIM_ENV = 8
SATURATED = 0.95            # |tanh| above this counts as saturated in the A1 report


def delta_from_baselines(eps0: torch.Tensor, frac: float) -> torch.Tensor:
    """A1's `Delta_Z`: a registered fraction of the spread of the species onsite baselines,
    shared across shells. Host-free -- it reads the Harrison table through `eps0`, which is
    a property of the elements, not of this crystal. Returned `[n_el, 2]`, filled uniformly,
    so a per-species rule can replace this one without an interface change."""
    spread = eps0.detach().reshape(-1).std(unbiased=False)
    return torch.full_like(eps0.detach(), float(frac) * float(spread))


def quadrupole_descriptor(edge_index: torch.Tensor, edge_vector: torch.Tensor, n_nodes: int,
                          q_cut: float) -> torch.Tensor:
    """`Q_i = sum_j w(r_ij) (rhat_ij rhat_ij^T - I/3)`, `[N, 3, 3]`, traceless symmetric,
    `w(r) = (1 - (r/q_cut)^6)^2` inside `q_cut` and zero outside. Vanishes at every
    centrosymmetric site by symmetry."""
    sender = edge_index[0]
    r = edge_vector.norm(dim=-1)
    # MASKED BY MULTIPLICATION, NOT BY INDEXING. `r[inside]` is a value-dependent shape, which
    # CUDA-graph capture forbids outright, and this is the rank-2 input of every head that
    # reads the base (the decoupled heads pass their own `Q` and never reach here). The weight
    # is zero outside `q_cut` by construction, so every excluded edge contributes exactly 0.0
    # and the sum is the same sum -- but the shape no longer depends on the positions.
    inside = (r < q_cut).to(edge_vector.dtype)
    unit = edge_vector / r.clamp_min(1e-9).unsqueeze(-1)
    w = (1.0 - (r / q_cut) ** 6) ** 2 * inside
    outer = unit.unsqueeze(-1) * unit.unsqueeze(-2) - torch.eye(3, dtype=r.dtype, device=r.device) / 3.0
    contrib = w.reshape(-1, 1, 1) * outer
    Q = torch.zeros(n_nodes, 3, 3, dtype=r.dtype, device=r.device)
    return Q.index_add(0, sender, contrib)


class GeometricDescriptor(nn.Module):
    """O3: the head's own atomic basis, so that nothing from the base enters `H0`.

        A_i^(l) = sum_j c_{Z_i Z_j}^{(n,l)} R_n(r_ij) f_cut(r_ij) Y_l(rhat_ij),  l = 0, 1, 2

    `R_n` is the orthogonal Bessel set on `[0, r_cut]` (`n = 1..N_r`, default 6) times H0's
    own taper `(1 - (r/r_cut)^6)^2`, so the head carries ONE cutoff radius and one smoothness
    class (C^1 at `r_cut`) across hoppings and descriptor alike. Harrison powers `(d/r)^n`
    were considered and rejected: over a 2.8-6 A window they are a nearly collinear monomial
    basis in `1/r`, ill conditioned by orders of magnitude and divergent at short range, and
    nothing reads the coefficients (they feed bounded readouts), so the interpretability
    argument that motivated them does not apply to a BASIS -- only to the hopping envelope,
    which is one specific function and is unchanged.

    `l = 0` is species-resolved and carries NO `c`: `A_{i,n,Z'} = sum_{j in Z'} R_n f_cut`,
    `N_r x n_species` numbers per atom (18 for three species). The readout's first layer is
    that projection, so a separate `c` for `l = 0` would be redundant. `c` remains for
    `l = 1, 2`, where it defines the vector and the tensor.
    """

    def __init__(self, num_elements: int, r_cut: float, n_radial: int = N_RADIAL_DEFAULT,
                 correlation2: bool = False) -> None:
        super().__init__()
        self.num_elements = int(num_elements)
        self.r_cut = float(r_cut)
        self.n_radial = int(n_radial)
        # O3.3: the ACE correlation-2 invariants of the A-tensors built below. See
        # `scalar_width` and `invariants`.
        self.correlation2 = bool(correlation2)
        # c_{Z Z'}^{(n,l)} for l = 1, 2; small init so the blocks start near absent, matching
        # the production convention for `vector_mix` / `alpha` / `beta`.
        self.c1 = nn.Parameter(torch.zeros(num_elements, num_elements, self.n_radial))
        self.c2 = nn.Parameter(torch.zeros(num_elements, num_elements, self.n_radial))
        with torch.no_grad():
            self.c1.normal_(0.0, 1.0 / max(self.n_radial, 1) ** 0.5)
            self.c2.normal_(0.0, 1.0 / max(self.n_radial, 1) ** 0.5)

    def scalar_width(self) -> int:
        """Width of the scalar block the readouts are fed."""
        n0 = self.n_radial * self.num_elements
        if not getattr(self, "correlation2", False):
            return n0
        nr = self.n_radial
        return n0 + n0 * (n0 + 1) // 2 + nr * (nr + 1) // 2 + 1

    def invariants(self, A0: torch.Tensor, v: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """O3.3: correlation-2 invariants of the SAME A-tensors, appended to the `l = 0` block.

        WHY. Block 0 of the base is products of an A-basis up to correlation 3; the head's
        on-site readout sees only `A^(0)`, a two-body radial sum. The crystal-field shift of a
        site in a TILTED perovskite is a three-body quantity -- the tilt angle, the Cs
        displacement relative to the cage -- and no amount of radial richness in a two-body sum
        reproduces it. These are the correlation-2 contractions of the tensors the descriptor
        already builds, so they cost nothing at runtime: no new neighbour loop, no new tensor.

            l = 0:  A^(0)_a A^(0)_b   for a <= b
            l = 1:  A^(1)_n . A^(1)_m for n <= m     (the angular content of the vector block)
            l = 2:  ||A^(2)||^2_F

        They are formed from the RAW channels; A1.1 standardises the whole block afterwards,
        which is what keeps the quadratic channels conditioned.
        """
        n = A0.shape[0]
        ia, ib = torch.triu_indices(A0.shape[1], A0.shape[1], device=A0.device)
        p0 = A0[:, ia] * A0[:, ib]                                    # [N, n0 (n0+1) / 2]
        vv = torch.einsum("nia,nja->nij", v, v)                       # [N, n_rad, n_rad]
        ja, jb = torch.triu_indices(v.shape[1], v.shape[1], device=A0.device)
        p1 = vv[:, ja, jb]                                            # [N, nr (nr+1) / 2]
        p2 = (Q * Q).sum(dim=(-1, -2)).reshape(n, 1)                  # [N, 1]
        return torch.cat([A0, p0, p1, p2], dim=-1)

    def radial(self, r: torch.Tensor) -> torch.Tensor:
        """`[n_edges, N_r]`: Bessel `sin(n pi r / r_cut) / r` times H0's taper."""
        x = (r / self.r_cut).clamp(max=1.0)
        taper = (1.0 - x ** 6) ** 2
        n = torch.arange(1, self.n_radial + 1, dtype=r.dtype, device=r.device)
        bessel = torch.sin(n.reshape(1, -1) * math.pi * x.reshape(-1, 1)) / r.clamp_min(1e-9).reshape(-1, 1)
        return bessel * taper.reshape(-1, 1)

    def forward(self, species: torch.Tensor, edge_index: torch.Tensor,
                edge_vector: torch.Tensor, n_nodes: int):
        """`(scalars, v [N, N_r, 3], Q [N, 3, 3])`; `scalars` is the raw `l = 0` block, widened
        by `invariants` when O3.3 is on."""
        A0, v, Q = self.raw(species, edge_index, edge_vector, n_nodes)
        # `getattr`: descriptors pickled before O3.3 have no such attribute, and the whole
        # O3/O3.1/O3.2 cohort must stay scorable.
        if getattr(self, "correlation2", False):
            return self.invariants(A0, v, Q), v, Q
        return A0, v, Q

    def raw(self, species: torch.Tensor, edge_index: torch.Tensor,
            edge_vector: torch.Tensor, n_nodes: int):
        """`(A0 [N, N_r * n_species], v [N, N_r, 3], Q [N, 3, 3])` -- the A-tensors themselves,
        without the correlation-2 block. The analytic force kernel differentiates through
        THESE: they are what the edge contraction knows how to map back to edges, and the
        correlation-2 channels are functions of them, so autograd carries the chain rule."""
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1).clamp_min(1e-9)
        R = self.radial(r)                                              # [E, N_r]
        unit = edge_vector / r.unsqueeze(-1)
        # l = 0, species resolved: sum over neighbours of each species, no c.
        z_j = species[dst]
        idx = src * self.num_elements + z_j
        A0 = torch.zeros(n_nodes * self.num_elements, self.n_radial,
                         dtype=R.dtype, device=R.device).index_add(0, idx, R)
        A0 = A0.reshape(n_nodes, self.num_elements * self.n_radial)
        # l = 1 and l = 2 carry c_{Z_i Z_j}^{(n,l)}.
        c1 = self.c1.to(R.dtype)[species[src], z_j]                     # [E, N_r]
        c2 = self.c2.to(R.dtype)[species[src], z_j]
        w1 = (c1 * R).unsqueeze(-1) * unit.unsqueeze(-2)                # [E, N_r, 3]
        v = torch.zeros(n_nodes, self.n_radial, 3, dtype=R.dtype, device=R.device).index_add(0, src, w1)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        outer = unit.unsqueeze(-1) * unit.unsqueeze(-2) - eye / 3.0     # [E, 3, 3], traceless
        w2 = (c2 * R).sum(-1).reshape(-1, 1, 1) * outer
        Q = torch.zeros(n_nodes, 3, 3, dtype=R.dtype, device=R.device).index_add(0, src, w2)
        return A0, v, Q


class CrystalField(nn.Module):
    """O5: the damped-Coulomb primitive that generates all three on-site channels.

        V_i      = sum_j q_{Z_j} erf(r_ij / a_{Z_i Z_j}) / r_ij  f_cut(r_ij)     (l = 0)
        E_i      = -grad_i V_i                                                    (l = 1)
        grad E_i = traceless Hessian of V_i                                       (l = 2)

    This is what the original specification's section 8.2 carried and v4 simplified away. It
    restores the crystal field in CLOSED FORM: the rank-1 block gets the FIELD, which is what
    drives s-p mixing at a Pb next to a vacancy, and the rank-2 block gets the field GRADIENT,
    which is the p-splitting -- the physics Arm 1 found the directional block to carry, with
    its correct radial form instead of a neighbour-direction sum.

    EVERYTHING IS TWO-BODY. No triplet list, so `capture` still works, and the analytic force
    kernel extends by a closed-form `dV_i/dR_j` rather than the three-body term O3.2 needed.

    Parameters: one signed charge and one damping length per species, nothing else. The charges
    are initialised from the head's own pristine Mulliken means, so nothing external enters and
    the decoupled head stays decoupled.
    """

    def __init__(self, num_elements: int, r_cut: float) -> None:
        super().__init__()
        self.num_elements = int(num_elements)
        self.r_cut = float(r_cut)
        # Signed charge per species. Overwritten from the pristine Mulliken means by
        # `set_crystal_field_charges`; the init here is the formal-valence sign pattern so the
        # module is usable before that runs (and so a unit test does not depend on it).
        self.charge = nn.Parameter(torch.zeros(num_elements))
        # Damping length per species PAIR, positive by construction through softplus.
        self.damp_raw = nn.Parameter(torch.zeros(num_elements, num_elements))

    def damping(self) -> torch.Tensor:
        """`a_{Z Z'}` in Angstrom, symmetric and positive. softplus(0) = 0.693, so the default
        damping is ~0.69 A -- short compared with the 2.8 A first shell, i.e. the primitive is
        close to bare Coulomb at bonding range and regular at the origin."""
        a = torch.nn.functional.softplus(self.damp_raw)
        return 0.5 * (a + a.transpose(0, 1))

    def radial(self, r: torch.Tensor, a: torch.Tensor):
        """`(g, g', g'')` for `g(r) = erf(r/a)/r * f_cut(r)`, all closed form.

        The second derivative is needed for the field GRADIENT, and writing it here rather than
        taking autograd twice is what keeps the analytic force kernel a possibility.
        """
        rr = r.clamp_min(1e-9)
        x = rr / a
        # erf and its derivatives
        E0 = torch.erf(x)
        c = 2.0 / (a * math.sqrt(math.pi))
        E1 = c * torch.exp(-x * x)                                   # d erf(r/a) / dr
        E2 = -2.0 * rr / (a * a) * E1                                # d2
        phi = E0 / rr
        dphi = E1 / rr - E0 / rr ** 2
        d2phi = E2 / rr - 2.0 * E1 / rr ** 2 + 2.0 * E0 / rr ** 3
        # H0's own taper, so the head keeps one smoothness class everywhere
        xc = (rr / self.r_cut).clamp(max=1.0)
        T = (1.0 - xc ** 6) ** 2
        dT = -12.0 * xc ** 5 * (1.0 - xc ** 6) / self.r_cut
        # MASKED. `T` and `dT` vanish of their own accord at the clamp because they carry a
        # factor (1 - x^6); `d2T` does NOT -- it goes to +72/r_cut^2 there, which is the
        # one-sided limit from inside and not the value of a function that is identically zero
        # outside. Without this mask the field gradient is wrong for every pair beyond r_cut,
        # which is most of them.
        inside = (rr < self.r_cut).to(rr.dtype)
        d2T = -12.0 * (5.0 * xc ** 4 - 11.0 * xc ** 10) / self.r_cut ** 2 * inside
        g = phi * T
        dg = dphi * T + phi * dT
        d2g = d2phi * T + 2.0 * dphi * dT + phi * d2T
        return g, dg, d2g

    def forward(self, species: torch.Tensor, edge_index: torch.Tensor,
                edge_vector: torch.Tensor, n_nodes: int):
        """`(V [N], E [N, 3], GE [N, 3, 3] traceless)`."""
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1).clamp_min(1e-9)
        u = edge_vector / r.unsqueeze(-1)                            # i -> j
        z_i, z_j = species[src], species[dst]
        a = self.damping().to(edge_vector.dtype)[z_i, z_j]
        q = self.charge.to(edge_vector.dtype)[z_j]
        g, dg, d2g = self.radial(r, a)
        V = torch.zeros(n_nodes, dtype=edge_vector.dtype, device=edge_vector.device).index_add(0, src, q * g)
        # E = -grad_i V. With d = R_j - R_i, dr/dR_i = -u, so -grad_i = +sum q g' u.
        E = torch.zeros(n_nodes, 3, dtype=edge_vector.dtype, device=edge_vector.device).index_add(
            0, src, (q * dg).unsqueeze(-1) * u)
        # Hessian of V wrt R_i: g'' u u^T + g' (I - u u^T) / r
        eye = torch.eye(3, dtype=edge_vector.dtype, device=edge_vector.device)
        uu = u.unsqueeze(-1) * u.unsqueeze(-2)
        Hs = (q * d2g).reshape(-1, 1, 1) * uu + (q * dg / r).reshape(-1, 1, 1) * (eye - uu)
        GE = torch.zeros(n_nodes, 3, 3, dtype=edge_vector.dtype, device=edge_vector.device).index_add(0, src, Hs)
        GE = GE - torch.diagonal(GE, dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1) * eye / 3.0
        return V, E, GE


class H0(nn.Module):
    """The runtime Hamiltonian for ONE graph (dense; the batched path stacks equal sizes)."""

    def __init__(self, atomic_numbers: Sequence[int], feature_dim: int = 128,
                 n_vectors: int = 128, r_cut: float = R_CUT_DEFAULT,
                 q_cut: float = Q_CUT_DEFAULT, directional: bool = True,
                 a_max: float = A_MAX_DEFAULT, b_max: float = B_MAX_DEFAULT,
                 on_site_range: float = ON_SITE_RANGE_DEFAULT, hidden: int = 64,
                 eta: float = ETA_DEFAULT, beta_b: float = 0.0, beta_a: float = 0.0,
                 delta_frac: float = DELTA_FRACTION_DEFAULT,
                 readout_hidden: int = READOUT_HIDDEN_DEFAULT,
                 decoupled: bool = False, n_radial: int = N_RADIAL_DEFAULT,
                 readout_init_scale: Optional[float] = None,
                 screened: bool = False, r_s3: float = R_S3_DEFAULT,
                 correlation2: bool = False, on_site_from_base: bool = False,
                 crystal_field: bool = False, cf_mlp: bool = False,
                 cf_linear: bool = False) -> None:
        super().__init__()
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        num_elements = len(self.atomic_numbers)
        # O3: the head owns its descriptor and reads nothing from the base. The readout path
        # is production's, unchanged -- only its INPUT changes, so a failed gate is
        # attributable to the descriptor and not to normalisation.
        self.decoupled = bool(decoupled)
        self.correlation2 = bool(correlation2)
        # O3.D (DIAGNOSTIC ONLY, registered as not adoptable): the head keeps its own
        # descriptor for the hoppings and the directional block, but the ON-SITE readout is
        # fed the base's block-0 invariants. It exists to attribute the O3 lineage's residual,
        # not to ship: a head that reads block 0 anywhere must wait for it, needs the cotangent
        # through it, and returns the `create_graph` traversal to the training step.
        self.on_site_from_base = bool(on_site_from_base)
        # O5a: the damped-Coulomb crystal field. Adds V to the scalar block (the B-basis stays
        # as a second l = 0 input) and REPLACES the rank-1/rank-2 inputs with the field and its
        # gradient. Everything two-body, so capture and a closed-form force stay available.
        self.crystal_field = bool(crystal_field)
        if self.crystal_field:
            self.cfield = CrystalField(num_elements, float(r_cut))
        # O5c: a DEDICATED small MLP on the three crystal-field invariants. In O5a they are
        # three inputs among 214 and the readout can barely distinguish them; here they get an
        # undiluted path of their own whose output is ADDED to the on-site pre-activation,
        # inside the same bounded tanh, so A1's bound and barrier still apply unchanged.
        # The final layer is ZERO-INITIALISED, so O5c starts exactly at O5a -- and unlike the
        # LES case this really does start and not pin: the pre-activation is linear in that
        # layer's weights, so its gradient is the hidden activation and is non-zero.
        self.cf_mlp = bool(cf_mlp) and self.crystal_field
        # O5a AS SPECIFIED: `d eps_i = s_Z (V_i - V_bar_Z)` -- LINEAR, no MLP. The scalar block
        # then keeps the B-basis ALONE (the invariants do not enter the readout at all) and the
        # potential acts only through this term. That stiffness is the point of the form: two
        # parameters per species and no capacity to overfit.
        self.cf_linear = bool(cf_linear) and self.crystal_field
        if self.cf_linear:
            self.cf_scale = nn.Parameter(torch.zeros(num_elements, 2))
            # `V_bar_Z`: the per-species mean of the potential over the PRISTINE ensemble, set
            # by `set_pristine_reference` beside the charges. A buffer, so it travels with the
            # checkpoint; with the linear form the invariants never enter the scalar block, so
            # A1.1's stored means cannot supply it.
            self.register_buffer("cf_vbar", torch.zeros(num_elements, dtype=torch.float64))
        if self.decoupled:
            self.descriptor = GeometricDescriptor(num_elements, float(r_cut), n_radial,
                                                  correlation2=self.correlation2)
            if not self.on_site_from_base:
                feature_dim = self.descriptor.scalar_width()   # O3.3 widens this
                if self.crystal_field and not bool(cf_linear):
                    feature_dim += 3        # V, |E|, tr(grad E . grad E)
            # One vector channel when the crystal field supplies it: the rank-1 block is then
            # literally `a_i (E_i . T_sp)` and `vector_mix` is a single species scalar, not a
            # redundant contraction over broadcast copies of the same field.
            n_vectors = 1 if self.crystal_field else n_radial
        self.r_cut, self.q_cut = float(r_cut), float(q_cut)
        self.directional = bool(directional)
        self.a_max, self.b_max = float(a_max), float(b_max)
        self.eta, self.beta_b, self.beta_a = float(eta), float(beta_b), float(beta_a)
        self.delta_frac = float(delta_frac)
        self.screened, self.r_s3 = bool(screened), float(r_s3)
        self.sk = SlaterKosterH(num_elements, feature_dim, atomic_numbers=self.atomic_numbers,
                                r_cut=self.r_cut, envelope="exp", hop_form="log",
                                decay_learned=True, hop_log_beta=self.eta,
                                on_site_range=on_site_range, hidden=hidden,
                                readout_init_scale=readout_init_scale,
                                screened=self.screened, r_s3=self.r_s3,
                                polarity=self.crystal_field)
        harrison_initialise(SimpleNamespace(h=self.sk), self.atomic_numbers)
        # A1: the on-site half-width, AFTER the Harrison baselines are in place (the rule
        # reads them). Registered fraction times their spread.
        with torch.no_grad():
            self.sk.delta.copy_(delta_from_baselines(self.sk.eps0, self.delta_frac))
        # Directional block: a per-species combination of the l = 1 channels and the two
        # bounded coefficients. Zero-initialised: the block starts absent and is learned.
        self.vector_mix = nn.Parameter(torch.zeros(num_elements, n_vectors))
        with torch.no_grad():
            self.vector_mix.normal_(0.0, VECTOR_MIX_INIT / max(n_vectors, 1) ** 0.5)
        self.alpha = nn.Parameter(torch.zeros(num_elements))          # block starts absent
        self.beta = nn.Parameter(torch.zeros(num_elements))
        # A1: the rank-2 and rank-1 environment readouts, `g_Z` and `f_Z`. Their own species
        # embedding, not the SK one: with `beta_b = beta_a = 0` the readouts are not
        # evaluated at all, and sharing an embedding would have made "off" mean "off except
        # for a gradient path through the SK element table". One hidden layer, final layer
        # (weights AND bias) scaled by 0.05, so the head starts at the species-level
        # coefficients -- which is exactly W4's `beta = 0` variant.
        if self.cf_mlp:
            self.cf_site = _mlp([3 + ELEM_DIM_ENV, readout_hidden, 2], final_scale=0.0)
        self.elem_env = nn.Embedding(num_elements, ELEM_DIM_ENV)
        self.g_read = _mlp([feature_dim + ELEM_DIM_ENV, readout_hidden, 1], final_scale=READOUT_INIT_SCALE)
        self.f_read = _mlp([feature_dim + ELEM_DIM_ENV, readout_hidden, 1], final_scale=READOUT_INIT_SCALE)
        # Test knobs (plan section 5 gates): `H0 -> H0 + a I` for the gauge gate, and a
        # per-site level shift `[N]` that binds a carrier on chosen sites for the tiling
        # ladder. Neither is a parameter; both are None/0 in production.
        self.gauge_shift = 0.0
        self.site_shift: Optional[torch.Tensor] = None

    def __setstate__(self, state) -> None:
        """Refuse a pre-A1 pickle rather than score it silently in the wrong form.

        `torch.save(model)` pickles the module, and unpickling restores `_buffers` whether or
        not `__init__` would have created them -- so a checkpoint trained with the pristine
        centre loads here with its `centre` buffer intact and then runs the UNCENTRED
        `on_site` on centred weights, with no error and no warning. The tracker says such a
        checkpoint is scored at the `pre-a1` tag; this makes it so instead of asking."""
        super().__setstate__(state)
        if "centre" in getattr(self, "_buffers", {}):
            raise RuntimeError(
                "this checkpoint was trained before v5 amendment A1 (it carries the removed "
                "reference buffer); score it at the `pre-a1` tag, where its form still exists")

    # ------------------------------------------------------------- pieces

    def coefficients(self):
        """`(a[Z], b[Z])`, bounded, species level."""
        return self.a_max * torch.tanh(self.alpha), self.b_max * torch.tanh(self.beta)

    def env_shift(self, readout: nn.Module, bound: float, scalars: torch.Tensor,
                  species: torch.Tensor, name: str) -> Optional[torch.Tensor]:
        """`bound * tanh(readout(h_i, Z_i))`, `[N]`: the environment's bounded shift of the
        coefficient's PRE-ACTIVATION. None when the bound is zero, so the factorial's
        "species coefficient" variant does not evaluate the readout at all and turning the
        term off removes it from the graph rather than adding a zero."""
        if not bound:
            return None
        pre = readout(torch.cat([self.sk.standardise(scalars, species),
                                 self.elem_env(species).to(scalars.dtype)], dim=-1))
        self.sk._reg_store(name, pre)
        return float(bound) * torch.tanh(pre).squeeze(-1)

    def site_coefficients(self, scalars: torch.Tensor, species: torch.Tensor):
        """`(a_i, b_i)`, `[N]` each: the bounded coefficients with their environment shift
        INSIDE the tanh,

            b_i = b_max tanh( beta_Z + beta_b tanh(g_Z(h_i)) ),   a_i likewise.

        WHY NOT A1's `b_Z (1 + beta tanh g)`. That form's gradient on the readout is
        `beta * b_Z * sech^2`, proportional to the species coefficient it modulates -- and
        `b_Z = b_max tanh(beta_Z)` is initialised at ZERO. The readout therefore starts with
        exactly no gradient and, after 60 epochs of W3, `b_Z` reaches only ~0.14, so it
        trains all run at about a seventh of the signal. The W4 factorial would have measured
        that starvation rather than the capacity. Warm-starting from a trained head would
        hide it rather than fix it, and production does not warm-start.

        Inside the tanh, `d b_i / d g = b_max * beta * sech^2(.) * sech^2(g)` = `b_max*beta`
        at initialisation and never depends on `b_Z`. The inner tanh keeps the environment's
        shift bounded by `beta` -- A1's bounded-modulation intent -- and the outer one keeps
        `|b_i| <= b_max` as before. At `beta = 0` this is `b_max tanh(beta_Z)`, the species
        coefficient EXACTLY, so W3 and the factorial's `spec` cell are unchanged."""
        a_pre = self.alpha[species].to(scalars.dtype)
        b_pre = self.beta[species].to(scalars.dtype)
        sa = self.env_shift(self.f_read, self.beta_a, scalars, species, "rank1")
        sb = self.env_shift(self.g_read, self.beta_b, scalars, species, "rank2")
        if sa is not None:
            a_pre = a_pre + sa
        if sb is not None:
            b_pre = b_pre + sb
        return self.a_max * torch.tanh(a_pre), self.b_max * torch.tanh(b_pre)

    @torch.no_grad()
    def zero_readouts(self) -> None:
        """A1's zero-readout limit: every bounded correction identically zero, so `H0` is the
        species-default Hamiltonian. Final layer WEIGHTS AND BIASES -- zeroing the weights
        alone leaves a constant pre-activation and a nonzero tanh."""
        for m in (self.sk.site, self.sk.hop, self.g_read, self.f_read):
            last = [x for x in m.modules() if isinstance(x, nn.Linear)][-1]
            last.weight.zero_()
            last.bias.zero_()

    @torch.no_grad()
    def saturation(self, scalars: torch.Tensor, species: torch.Tensor,
                   edge_index: Optional[torch.Tensor] = None,
                   sites: Optional[torch.Tensor] = None,
                   edge_vector: Optional[torch.Tensor] = None) -> dict:
        """A1's bounds report: the fraction of `|tanh| > 0.95` per term, per species, plus
        the section 2.10 readout at chosen `sites` (the flanking Pb). Terms `on_site` (2
        channels per atom), `rank2`/`rank1` (1 per atom, only where the bound is nonzero) and
        `hop` (4 per edge, keyed by the sender's species)."""
        scalars = scalars.to(torch.float64)
        out: dict = {}
        e = self.sk.elem(species)
        h = self.sk.standardise(scalars, species)
        t_site = torch.tanh(self.sk.site(torch.cat([h, e], dim=-1)))                # [N, 2]
        terms = {"on_site": (t_site, species)}
        env = torch.cat([h, self.elem_env(species).to(scalars.dtype)], dim=-1)
        if self.beta_b:
            terms["rank2"] = (torch.tanh(self.g_read(env)), species)
        if self.beta_a:
            terms["rank1"] = (torch.tanh(self.f_read(env)), species)
        screen = getattr(self.sk, "screen", None)
        if screen is not None and edge_index is not None and edge_vector is not None:
            # O3.2. The registered reading for this arm: the modulation must not be reported
            # as "learned" while sitting at its stop, which is the mistake O3 made with the
            # on-site term. Keyed by the SENDER's species, as `hop` was.
            pre_s = screen(edge_index, edge_vector.to(torch.float64), species)
            terms["screen"] = (torch.tanh(pre_s), species[edge_index[0]])
            out["screen_pre_abs_p95"] = float(torch.quantile(pre_s.abs().reshape(-1), 0.95))
            out["screen_pre_abs_max"] = float(pre_s.abs().max())
            out["screen_factor_p95"] = float(torch.quantile(
                self.sk.modulation(pre_s).reshape(-1), 0.95))
        if edge_index is not None and not getattr(self.sk, "hop_modulation", True):
            # O3 removed the hopping modulation, so `sk.hop` is never evaluated by the model.
            # Reporting its saturation would put a live-looking number in every epoch log for
            # an MLP that does not run. Marked, not computed.
            # Still "not applicable" under O3.2: the modulation is back but `sk.hop` is not.
            out["hop_not_applicable"] = 1.0
        elif edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            sym = torch.cat([h[src] + h[dst], (h[src] - h[dst]).abs(),
                             self.sk.elem(species[src]) + self.sk.elem(species[dst]),
                             (self.sk.elem(species[src]) - self.sk.elem(species[dst])).abs()], dim=-1)
            terms["hop"] = (torch.tanh(self.sk.hop(sym)), species[src])
        for name, (t, sp) in terms.items():
            sat = (t.abs() > SATURATED).to(torch.float64)
            out[name] = float(sat.mean())
            per = {}
            for z_i, z in enumerate(self.atomic_numbers):
                sel = sp == z_i
                if bool(sel.any()):
                    per[int(z)] = float(sat[sel].mean())
            out[name + "_by_species"] = per
        # A1.1's registered per-epoch diagnostic: the p95 on-site correction per species, in
        # eV. This is the number that stayed under 0.105 eV through the whole A1 run and is
        # expected to reach several hundred meV within the first epochs with standardisation.
        shift = (t_site.abs() * self.sk.delta[species].to(t_site.dtype))
        out["on_site_shift_eV_p95"] = float(torch.quantile(shift.reshape(-1), 0.95))
        out["on_site_shift_eV_max"] = float(shift.max())
        per_z = {}
        for z_i, z in enumerate(self.atomic_numbers):
            sel = species == z_i
            if bool(sel.any()):
                per_z[int(z)] = float(torch.quantile(shift[sel].reshape(-1), 0.95))
        out["on_site_shift_eV_p95_by_species"] = per_z
        # O5a's ONE unbounded channel. `s_Z (V - V_bar_Z)` has no tanh, so it appears in
        # neither the saturation fractions above nor the barrier, and a term that nothing
        # watches is a term that can quietly grow to dominate the level it shifts. Its p95 and
        # max are reported in the same eV units as the bounded shift, beside it, so the two
        # can be read against each other. (O5c's MLP form IS inside the bound and needs none
        # of this.)
        if getattr(self, "cf_linear", False) and edge_index is not None and edge_vector is not None:
            V, _, _ = self.cfield(species, edge_index, edge_vector.to(torch.float64),
                                  int(species.shape[0]))
            cf = self.cf_linear_shift(V, species).abs()
            out["cf_shift_eV_p95"] = float(torch.quantile(cf.reshape(-1), 0.95))
            out["cf_shift_eV_max"] = float(cf.max())
            out["cf_shift_eV_p95_by_species"] = {
                int(z): float(torch.quantile(cf[species == z_i].reshape(-1), 0.95))
                for z_i, z in enumerate(self.atomic_numbers) if bool((species == z_i).any())}
            out["cf_potential_eV_p95"] = float(torch.quantile(V.abs().reshape(-1), 0.95))
        if sites is not None and bool(sites.any()):
            out["site_on_site"] = [float(x) for x in t_site[sites].reshape(-1)]
            if self.beta_b:
                out["site_rank2"] = [float(x) for x in torch.tanh(self.g_read(env[sites])).reshape(-1)]
            if self.beta_a:
                out["site_rank1"] = [float(x) for x in torch.tanh(self.f_read(env[sites])).reshape(-1)]
        return out

    def directional_block(self, scalars: torch.Tensor, vectors: torch.Tensor,
                          species: torch.Tensor, edge_index: torch.Tensor,
                          edge_vector: torch.Tensor, Q: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`H_onsite_dir` as a dense `[4N, 4N]` block-diagonal matrix."""
        return torch.block_diag(*self.directional_site_blocks(scalars, vectors, species,
                                                              edge_index, edge_vector, Q))

    def batched(self, scalars: torch.Tensor, vectors: Optional[torch.Tensor], species: torch.Tensor,
                edge_index: torch.Tensor, edge_vector: torch.Tensor, batch: torch.Tensor,
                num_graphs: int, n_nodes: int) -> torch.Tensor:
        """Dense `H0` for a batch of EQUAL-SIZED graphs at once, `[B, 4n, 4n]`: the same
        matrices `forward` builds one at a time (asserted equal in the tests), without the
        per-graph Python loop and its slicing -- which the training-step profile put at a
        third of the backward. `edge_index` is global (batch-concatenated), `batch` the
        graph of every node."""
        edge_vector = edge_vector.to(torch.float64)
        Q_geo = None
        base_scalars = scalars
        V_cf = None
        if getattr(self, "decoupled", False):                 # O3: nothing from the base enters H0
            scalars, vectors, Q_geo, V_cf = self.geometric_inputs(
                species, edge_index, edge_vector, int(species.shape[0]))
        scalars = scalars.to(torch.float64)
        site_feats = (base_scalars.to(torch.float64)
                      if getattr(self, "on_site_from_base", False) else scalars)      # O3.D
        local = torch.arange(scalars.shape[0], device=scalars.device) - batch * n_nodes
        edge_graph = batch[edge_index[0]]
        if getattr(self, "crystal_field", False) and V_cf is None:
            raise ValueError("the crystal field needs the decoupled path, which computes `V`")
        pot = V_cf
        H = self.sk.batched(site_feats, species, edge_index, edge_vector, local, edge_graph,
                            n_nodes, num_graphs, potential=pot)
        dim = n_nodes * ORBITALS_PER_ATOM
        levels = self.sk.on_site(site_feats, species, None)                           # [N, 2]
        if getattr(self, "cf_mlp", False):
            levels = levels + self.cf_shift(scalars, species)
        if getattr(self, "cf_linear", False):
            levels = levels + self.cf_linear_shift(V_cf, species)
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1)           # [N, 4]
        diag = diag.reshape(num_graphs, dim)
        eye = torch.eye(dim, dtype=H.dtype, device=H.device)
        H = H * (1.0 - eye) + torch.diag_embed(diag)
        if self.directional:
            if vectors is None:
                raise ValueError("the directional block needs the base's l = 1 features")
            blocks = self.directional_site_blocks(site_feats, vectors.to(torch.float64), species,
                                                  edge_index, edge_vector, Q_geo)
            # Scatter the [N, 4, 4] site blocks onto the block diagonals of [B, 4n, 4n].
            o = torch.arange(ORBITALS_PER_ATOM, device=H.device)
            rows = (local * ORBITALS_PER_ATOM).reshape(-1, 1, 1) + o.reshape(1, -1, 1)
            cols = (local * ORBITALS_PER_ATOM).reshape(-1, 1, 1) + o.reshape(1, 1, -1)
            flat = batch.reshape(-1, 1, 1) * (dim * dim) + rows * dim + cols
            add = torch.zeros(num_graphs * dim * dim, dtype=H.dtype, device=H.device)
            add = add.index_put((flat.reshape(-1),), blocks.reshape(-1), accumulate=True)
            H = H + add.reshape(num_graphs, dim, dim)
        H = 0.5 * (H + H.transpose(-1, -2))
        if self.site_shift is not None:
            H = H + torch.diag_embed(self.site_shift.to(H.dtype).to(H.device).repeat_interleave(ORBITALS_PER_ATOM).reshape(num_graphs, dim))
        if self.gauge_shift:
            H = H + float(self.gauge_shift) * eye
        return H

    def bloch(self, scalars: torch.Tensor, vectors: Optional[torch.Tensor], species: torch.Tensor,
              edge_index: torch.Tensor, edge_vector: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        """Plan P Phase 2b §3: `H0(k) = sum_R H0(0, R) e^{i k . R}` of ONE periodic graph, complex
        Hermitian `[n_k, 4N, 4N]`, from the same per-edge Slater-Koster blocks `batched` scatters
        (each directed edge with its image shift phased by `exp(i k . d_e)`, `d_e = R_j - R_i + L`:
        the atomic gauge, whose spectrum is the Bloch spectrum) and the same k-independent
        on-site, directional and shift terms. At `k = 0` it is `batched`'s `H0` exactly."""
        from mace.modules.dscc.legacy import sk_block
        edge_vector = edge_vector.to(torch.float64)
        n = int(species.shape[0])
        Q_geo = None
        base_scalars = scalars
        V_cf = None
        if getattr(self, "decoupled", False):
            scalars, vectors, Q_geo, V_cf = self.geometric_inputs(species, edge_index, edge_vector, n)
        scalars = scalars.to(torch.float64)
        site_feats = (base_scalars.to(torch.float64)
                      if getattr(self, "on_site_from_base", False) else scalars)
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1).clamp_min(1e-9)
        direction = edge_vector / r.unsqueeze(-1)
        screen_pre = (self.sk.screen(edge_index, edge_vector, species)
                      if getattr(self.sk, "screen", None) is not None else None)
        v = self.sk.integrals_fast(site_feats, species, src, dst, r, screen_pre, V_cf)
        blocks = sk_block(direction, v)                                                  # [E, 4, 4]
        dim = n * ORBITALS_PER_ATOM
        o = torch.arange(ORBITALS_PER_ATOM, device=src.device)
        rows = (src.reshape(-1, 1, 1) * ORBITALS_PER_ATOM + o.reshape(1, -1, 1)).expand(-1, ORBITALS_PER_ATOM, ORBITALS_PER_ATOM)
        cols = (dst.reshape(-1, 1, 1) * ORBITALS_PER_ATOM + o.reshape(1, 1, -1)).expand(-1, ORBITALS_PER_ATOM, ORBITALS_PER_ATOM)
        flat = (rows * dim + cols).reshape(-1)
        # k-independent part: on-site levels, directional site blocks, shifts (as `batched`)
        levels = self.sk.on_site(site_feats, species, None)
        if getattr(self, "cf_mlp", False):
            levels = levels + self.cf_shift(scalars, species)
        if getattr(self, "cf_linear", False):
            levels = levels + self.cf_linear_shift(V_cf, species)
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
        D = torch.diag(diag)
        if self.directional:
            if vectors is None:
                raise ValueError("the directional block needs the base's l = 1 features")
            sb = self.directional_site_blocks(site_feats, vectors.to(torch.float64), species,
                                              edge_index, edge_vector, Q_geo)               # [N, 4, 4]
            nodes = torch.arange(n, device=src.device)
            r_s = (nodes.reshape(-1, 1, 1) * ORBITALS_PER_ATOM + o.reshape(1, -1, 1)).expand(-1, ORBITALS_PER_ATOM, ORBITALS_PER_ATOM)
            c_s = (nodes.reshape(-1, 1, 1) * ORBITALS_PER_ATOM + o.reshape(1, 1, -1)).expand(-1, ORBITALS_PER_ATOM, ORBITALS_PER_ATOM)
            D = D + torch.zeros(dim * dim, dtype=D.dtype, device=D.device).index_put(
                ((r_s * dim + c_s).reshape(-1),), sb.reshape(-1), accumulate=True).reshape(dim, dim)
        if self.site_shift is not None:
            D = D + torch.diag(self.site_shift.to(D.dtype).to(D.device).repeat_interleave(ORBITALS_PER_ATOM))
        if self.gauge_shift:
            D = D + float(self.gauge_shift) * torch.eye(dim, dtype=D.dtype, device=D.device)
        D = 0.5 * (D + D.transpose(0, 1))
        # hopping part per k: sum_e B_e exp(i k . d_e) scattered at (src, dst)
        phase = edge_vector @ kpts.to(edge_vector.dtype).transpose(0, 1)                 # [E, n_k]
        ph = torch.complex(torch.cos(phase), torch.sin(phase))                            # [E, n_k]
        bc = blocks.to(torch.complex128).reshape(blocks.shape[0], -1)                     # [E, 16]
        out = []
        for ik in range(kpts.shape[0]):
            vals = (bc * ph[:, ik].unsqueeze(-1)).reshape(-1)
            Hk = torch.zeros(dim * dim, dtype=torch.complex128, device=D.device).index_put((flat,), vals, accumulate=True).reshape(dim, dim)
            Hk = Hk + D.to(torch.complex128)
            out.append(0.5 * (Hk + Hk.conj().transpose(0, 1)))
        return torch.stack(out)

    def cf_linear_shift(self, V: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        '''O5a as specified: `d eps_i = s_Z (V_i - V_bar_Z)`, linear, no MLP, `[N, 2]` in eV.

        `V_bar_Z` is the per-species mean of the potential over the training ensemble -- the
        same quantity A1.1 already stores, so nothing new is fitted for it. `s_Z` is two numbers
        per species (s and p shells), zero-initialised, so the arm starts exactly at its parent.

        NOTE, recorded rather than hidden: this channel is OUTSIDE A1's `Delta_Z tanh` bound by
        construction -- the specified form has no squashing. `V` is bounded by the cutoff so the
        shift cannot run away, but the barrier and the saturation report do not see it, and its
        p95 is reported separately for that reason.
        '''
        vbar = self.cf_vbar.to(V.dtype)[species]
        d = (V - vbar).unsqueeze(-1) * self.cf_scale.to(V.dtype)[species]
        self.sk._audit_store("cf_linear", d)
        return d

    def cf_shift(self, scalars: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        '''O5c: the dedicated crystal-field readout's contribution to the on-site levels.

        Returns `[N, 2]` in eV, already bounded -- `Delta_Z tanh(.)`, the same form and the same
        bound A1 puts on the main readout -- so nothing here escapes the barrier or the
        saturation report. In O5a the three invariants are three inputs among 214 and the
        readout can barely distinguish them; here they get an undiluted path.

        NOT A SECOND TERM INSIDE THE MAIN BOUND: this is its OWN `Delta_Z tanh`, added to the
        LEVEL beside the main readout's. So the arm's extra channel is squashed at `Delta_Z`
        and pays the L2 and the barrier (`_reg_store` below), where O5a's linear shift pays
        neither. The MLP is the more constrained of the two forms, not the freer one.

        Zero at initialisation, so O5c starts exactly at O5a. Unlike the LES latent charges this
        really does START rather than pin: the pre-activation is linear in the final layer's
        weights, so their gradient is the hidden activation and is non-zero there.
        '''
        h = self.sk.standardise(scalars, species)[:, -3:]     # A1.1, as registered
        e = self.elem_env(species).to(h.dtype)
        pre = self.cf_site(torch.cat([h, e], dim=-1))
        self.sk._audit_store("cf_site", pre)
        self.sk._reg_store("cf_site", pre)
        return self.sk.delta[species].to(pre.dtype) * torch.tanh(pre)

    def geometric_inputs(self, species, edge_index, edge_vector, n_nodes):
        """O3: `(A0, v, Q)` from geometry alone -- the head's inputs when `decoupled`.

        O5a: the rank-1 and rank-2 inputs become the crystal FIELD and its GRADIENT, and the
        scalar block gains the potential and two invariants of the other two. The B-basis stays
        -- it is worth 34 % on energies and nothing here displaces it."""
        A0, v, Q = self.descriptor(species, edge_index, edge_vector, n_nodes)
        if not getattr(self, "crystal_field", False):
            return A0, v, Q, None
        V, E, GE = self.cfield(species, edge_index, edge_vector, n_nodes)
        inv = torch.stack([V, E.norm(dim=-1), (GE * GE).sum(dim=(-1, -2))], dim=-1)
        if not getattr(self, "cf_linear", False):
            A0 = torch.cat([A0, inv], dim=-1)
        # The field takes the vector slot. With `crystal_field` the head is built with
        # `n_vectors = 1`, so this is literally the ruling's `a_i (E_i . T_sp)` -- one species
        # scalar times the field -- rather than a broadcast contracted by a redundant mix.
        v = E.unsqueeze(1)
        return A0, v, GE, V

    def directional_site_blocks(self, scalars: torch.Tensor, vectors: torch.Tensor,
                                species: torch.Tensor, edge_index: torch.Tensor,
                                edge_vector: torch.Tensor, Q: Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
        """`H_onsite_dir` as `[N, 4, 4]` site blocks (global node indexing). `scalars` enters
        only through A1's environment factors, and only when their bounds are nonzero."""
        n = int(species.shape[0])
        a_i, b_i = self.site_coefficients(scalars.to(vectors.dtype), species)
        v = torch.einsum("nc,nca->na", self.vector_mix[species].to(vectors.dtype), vectors)
        if Q is None:                     # O3 passes its own l = 2 tensor
            Q = quadrupole_descriptor(edge_index, edge_vector, n, self.q_cut)
        block = torch.zeros(n, ORBITALS_PER_ATOM, ORBITALS_PER_ATOM, dtype=vectors.dtype, device=vectors.device)
        sp = a_i.unsqueeze(-1) * v
        block[:, 0, 1:] = sp
        block[:, 1:, 0] = sp
        block[:, 1:, 1:] = b_i.reshape(-1, 1, 1) * Q
        return block

    def forward(self, scalars: torch.Tensor, vectors: Optional[torch.Tensor],
                species: torch.Tensor, edge_index: torch.Tensor, edge_vector: torch.Tensor
                ) -> torch.Tensor:
        """Dense `H0` `[4N, 4N]` in float64. `scalars` `[N, F]` and `vectors` `[N, n_vec, 3]`
        are the base's first-block features (frozen w.r.t. base parameters, attached to
        positions and cell); `edge_index`, `edge_vector` the head's graph at `r_cut`."""
        n = int(species.shape[0])
        edge_vector = edge_vector.to(torch.float64)
        Q_geo = None
        base_scalars = scalars
        V_cf = None
        if getattr(self, "decoupled", False):                 # O3, as in `batched`
            scalars, vectors, Q_geo, V_cf = self.geometric_inputs(species, edge_index, edge_vector, n)
        scalars = scalars.to(torch.float64)
        # O3.D: the readouts see the BASE's block 0; the descriptor still drives the hoppings
        # (which read nothing, the modulation being off) and the directional block's v and Q.
        site_feats = (base_scalars.to(torch.float64)
                      if getattr(self, "on_site_from_base", False) else scalars)
        # O5: the polarity modulation and the linear on-site shift both need `V`, which
        # `geometric_inputs` RETURNS (it used to be stashed on the module, which worked only
        # because the producer happened to run immediately before the consumer). `V` exists
        # only on the decoupled path, and O5 is decoupled; say so rather than raise a
        # NoneType error three frames down.
        if getattr(self, "crystal_field", False) and V_cf is None:
            raise ValueError("the crystal field needs the decoupled path, which computes `V`")
        pot = V_cf
        H = self.sk(site_feats, species, edge_index, edge_vector, madelung=None, n_nodes=n,
                    potential=pot)
        levels = self.sk.on_site(site_feats, species, None)
        if getattr(self, "cf_mlp", False):
            levels = levels + self.cf_shift(scalars, species)
        if getattr(self, "cf_linear", False):
            levels = levels + self.cf_linear_shift(V_cf, species)
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
        H = H - torch.diag(torch.diagonal(H)) + torch.diag(diag)
        if self.directional:
            if vectors is None:
                raise ValueError("the directional block needs the base's l = 1 features")
            H = H + self.directional_block(site_feats, vectors.to(torch.float64), species,
                                           edge_index, edge_vector, Q_geo)
        H = 0.5 * (H + H.transpose(0, 1))
        if self.site_shift is not None:
            H = H + torch.diag(self.site_shift.to(H.dtype).to(H.device).repeat_interleave(ORBITALS_PER_ATOM))
        if self.gauge_shift:
            H = H + float(self.gauge_shift) * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        return H
