"""The pieces of the v8 programme that plan section 3 keeps, copied verbatim into the
D-SCC package so that the deletion sweep can remove `mace/modules/defect_*`:

* `SlaterKosterH` (H_fix -> H0: Slater-Koster hoppings with the Harrison initialisation at
  each pair's reference bond length, learned decay lengths, bounded log-form modulation,
  centred scalar on-site corrections) with `sk_block`, `bond_reference` and its constants;
* the Gaussian / Fermi smearing functions, `find_mu` (bisection), `fermi_fill`;
* the Daleckii-Krein divided-difference backward (`_dk_eigenbasis`, `_dk_backward`);
* `harrison_initialise` and its tables; `frame_key`.
The docstrings are the originals' (their "Edit"/"Stage"/"section" references are to the
v8 programme's documents). Behaviour and state-dict keys are unchanged, so checkpoints
carry over by `load_state_dict`.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


@contextlib.contextmanager
def mark(name: str):
    """Profiling region marker of the old code: a no-op here."""
    yield


ETA_SS_SIGMA = -1.40         # Harrison's universal ss-sigma coefficient
HBAR2_OVER_M = 7.62          # eV A^2


def _mlp(sizes, activation=nn.SiLU, final_scale=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:-1]):
        layers += [nn.Linear(a, b), activation()]
    last = nn.Linear(sizes[-2], sizes[-1])
    if final_scale is not None:
        with torch.no_grad():
            last.weight.mul_(final_scale)
            last.bias.mul_(final_scale)
    layers.append(last)
    return nn.Sequential(*layers)


ORBITALS_PER_ATOM = 4                       # s, px, py, pz


T_EL = 0.025                                # eV, electronic temperature for the smearing


BOND_TYPES = ("ss_sigma", "sp_sigma", "pp_sigma", "pp_pi")


# Cordero et al., Dalton Trans. 2008, 2832: single-bond covalent radii in Angstrom, from a
# fit over the Cambridge Structural Database. A UNIVERSAL table, keyed by atomic number and
# containing nothing about this host -- which is the point. Section 2.3 of the speed cycle's
# spec removes `d_ref`, the per-host bond length that was a default in the head and a
# command-line number (2.861 A for CsPbCl3) in the launcher; the anchor is now
# `d_ref[s, s'] = r_cov(s) + r_cov(s')`, a property of the two elements.
#
# The choice between the two options the spec allows -- initialising each bond at its own
# r_ij, or anchoring per species pair at a tabulated reference -- is recorded here as the
# second. Anchoring keeps `v0` a per-pair PARAMETER with a fixed meaning ("the integral at
# the reference separation"), which the learned decay lengths then modulate; initialising at
# r_ij would make the baseline a function of the geometry and put the same distance
# dependence in two places.
COVALENT_RADII: Dict[int, float] = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05,
    17: 1.02, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 55: 2.44, 79: 1.36, 81: 1.45, 82: 1.46, 83: 1.48,
}


def bond_reference(atomic_numbers: Sequence[int]) -> torch.Tensor:
    """`d_ref[s, s'] = r_cov(s) + r_cov(s')`, `[n_elements, n_elements]`, in Angstrom.

    Raises on an element the table does not cover, in the same spirit as `VALENCE`: a
    missing radius silently replaced by a default is a per-host constant re-entering
    through the back door.
    """
    zs = [int(z) for z in atomic_numbers]
    missing = sorted({z for z in zs if z not in COVALENT_RADII})
    if missing:
        raise ValueError(
            f"no covalent radius recorded for Z = {missing}; the hopping envelope has no "
            "host-free anchor for those elements and will not invent one")
    r = torch.tensor([COVALENT_RADII[z] for z in zs], dtype=torch.get_default_dtype())
    return r.reshape(-1, 1) + r.reshape(1, -1)


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


# The label pipeline's own convention. The defect calculations were set up through doped,
# whose default electronic smearing is Gaussian with SIGMA = 0.05 eV (ISMEAR = 0). The head
# now matches the labels in FUNCTIONAL FORM and in width; the previous T_el = 25 meV was a
# k_B * 300 K coincidence with no connection to the labels at all.
#
# Both families stay implemented, selected by config: Fermi-Dirac is retained for sensitivity
# work, because "the answer does not depend on the smearing" is a claim that needs a second
# family to test and not an assertion.
SMEARING_FAMILY = "gaussian"


# gamma, the bounded on-site correction half-width. Widened from 1.0 after the
# saturation audit; see CountingHead.__init__ for the measurement.
ON_SITE_RANGE_DEFAULT = 3.0


SMEARING_WIDTH = 0.05            # eV


# The radial envelope on the hopping integrals. "exp" is what Stage 3 ran; "power" is
# Harrison's own d^-2, which is also what `v0` is initialised from. See
# `SlaterKosterH.radial` for the measurement that separates them.
ENVELOPES = ("exp", "power")


ENVELOPE_DEFAULT = "exp"


# The environment modulation on each hopping integral. "linear" is Stage 3's
# `1 + hop_range * tanh(g)`, bounded in [1 - hop_range, 1 + hop_range] and therefore
# ASYMMETRIC in log space: at hop_range = 0.5 it can halve a bond but only add 50%. "log" is
# `exp(beta * tanh(g))`, bounded in [exp(-beta), exp(+beta)] and symmetric, so widening does
# not privilege weakening over strengthening. Both are exactly 1 at g = 0, so bulk-like bonds
# are untouched by the choice and the two forms differ only where the head is straining.
HOP_FORMS = ("linear", "log")


HOP_FORM_DEFAULT = "linear"


HOP_LOG_BETA_DEFAULT = math.log(3.0)      # x[1/3, 3]


# Stage A' spec section 2.4: the range-equivalent log modulation. exp(ln 1.5 * tanh g) spans
# [2/3, 3/2] -- the same upper reach as the linear form's 1 + 0.5 tanh g, without the
# sign-flip hazard, and no capacity change.
HOP_LOG_BETA_RANGE_EQUIVALENT = math.log(1.5)


# Stage A' spec section 2.4: decay lengths as four learned universal scalars, one per
# Slater-Koster integral type. L_b = L0 * exp(beta_L * tanh u_b), beta_L = ln 2, so with
# L0 = 1.0 A every L_b lies in [0.5, 2.0] A. u_b = 0 reproduces the fixed-length head exactly.
DECAY_LOG_BETA_DEFAULT = math.log(2.0)


# v5 amendment A1: THE CENTRE IS GONE. Both retired forms subtracted a pristine species
# feature mean inside or outside the tanh,
#
#   "output"    corr_i = gamma [ tanh h(x_i) - tanh h(xbar_s) ]     (Stage A')
#   "argument"  corr_i = gamma   tanh[ h(x_i) - h(xbar_s) ]         (Stage B .. pre-A1)
#
# and A1 removes both: the on-site correction is `delta_Z tanh(e_Z(h_i))`, with a bias in
# the readout in place of the reference and a weak L2 on the tanh output in place of the
# "zero on the pristine cell" property. The reasons are in the amendment: the neutral null
# is algebraic, so nothing is protected by vanishing in the bulk; a static-cell feature mean
# is ill-defined under thermal noise and does not exist for a foundation model.
#
# The argument for keeping a legacy branch (the codebase idiom: models pickled before an
# attribute existed must keep evaluating) is refused HERE and only here, because A1's static
# check asserts that no runtime module reads pristine feature statistics -- a `getattr`
# fallback would put the symbol back in the assembly path and make the test a lie. Pre-A1
# checkpoints are scored at the `pre-a1` tag instead.
#
# `delta_Z`, the per-species on-site half-width, replaces the single `on_site_range` scalar.
# It is a registered fraction of the spread of the species onsite baselines (A1); the scalar
# survives as the constructor default that fills the buffer uniformly, so a per-species rule
# can be registered later without an interface change.
DELTA_FRACTION_DEFAULT = 0.40


# v5 amendment A1.1: PER-SPECIES INPUT STANDARDISATION, and why the readouts do not work
# without it.
#
# A1 removed the pristine centre from `tanh[h(x_i) - h(xbar_Z)]`. The physics argument was
# right -- the neutral null is algebraic, so nothing was protected by vanishing in the bulk --
# but the subtraction was doing a second, numerical job: it fed the nonlinearity a DEVIATION.
# A linear readout on raw features splits as
#
#     w . x_i + b  =  [w . xbar_Z + b]  +  w . (x_i - xbar_Z),
#
# a species-constant part and an environment part. Growing `w` to capture the environment also
# moves the constant, which is a species-dependent onsite shift -- it moves the relative
# Pb/Cl/Cs levels and the gap, so the gap regulariser and the spectral structure of `H0` resist
# it. The bias can compensate in principle, but the two are coupled through a large,
# ill-conditioned direction, and the optimiser's answer is to keep `w` tiny.
#
# MEASURED on base v2 (120 frames, 9600 atoms, 512 channels): ||xbar_Z|| over the median
# ||x_i - xbar_Z|| is 29.2 (Cl), 45.7 (Cs), 37.1 (Pb), and the median per-channel |mu|/sd is
# 18.6 / 28.1 / 20.8. The A1 run's on-site corrections duly never exceeded 0.105 eV against a
# 3.079 eV bound, where the centred head reached 2.99 eV.
#
# The fix keeps A1's reference-free form: standardise the readout inputs per species with
# statistics of the TRAINING ENSEMBLE (mean and sd per channel over all training atoms of that
# species, thermal frames, defect and pristine), computed once and frozen. This is data
# normalisation, not a host reference: it does not tie the model to a pristine cell, and A1's
# thermal-noise objection does not apply to an ensemble statistic. With standardised inputs
# the constant part IS the bias, so the gap regulariser acts on the bias alone and every
# direction of `w` is free.
# A channel the base does not vary across the ensemble carries no information, so its
# standardised value is ZERO -- not `(h - mu)/tiny`, which amplifies numerical noise without
# bound. The first implementation floored the sd instead and a symmetry-perfect test cell,
# where every atom of a species is equivalent and every sd is ~0, drove `H0` to nonsense and
# the SCF to non-convergence. Below this fraction of the species' median channel sd, the
# channel is dead and is zeroed.
FEATURE_SD_FLOOR = 1e-3


# THE SATURATION TRAP, and why an L2 on the tanh output cannot get you out of it.
#
# Seed 0 of the first A1.1 arm diverged at epoch 37 under constant lr 2e-3: the train force
# loss jumped 180x in two epochs, every bounded correction slammed into its bound (p95 on-site
# shift 0.061 -> 3.076 eV against a 3.079 eV bound; saturation fractions 0.52 on-site, 0.68
# hop), and twenty further epochs never escaped -- the held-out RMSE ended at 46.4 against
# 15.4 at epoch 24.
#
# It cannot escape by construction. The registered L2 penalises `tanh(pre)^2`, whose gradient
# with respect to the pre-activation is `2 tanh(pre) sech^2(pre)`, and `sech^2 -> 0` as
# `|pre| -> inf`. The restoring force vanishes exactly where it is needed, and so does the
# data gradient, for the same reason. A saturated unit is a dead unit with no route back.
#
# The barrier is therefore on the PRE-ACTIVATION, where the gradient `2(|pre| - knee)` grows
# rather than vanishes, and one-sided so that it is identically zero in normal operation:
# the healthy runs sit at `|pre| ~ 0.05`, and the knee at 2.0 corresponds to |tanh| = 0.964.
# It is a guard rail, not a regulariser -- it should never fire in a run that behaves.
SATURATION_KNEE = 2.0


# The LIVE setting, read by every fill, every entropy and the density-response backward.
_FAMILY = SMEARING_FAMILY


_WIDTH = SMEARING_WIDTH


_SQRT_PI = math.sqrt(math.pi)


def occupation_of(x: torch.Tensor, family: str = SMEARING_FAMILY) -> torch.Tensor:
    """`f(x)` for `x = (eps - mu) / width`.

    Gaussian (Methfessel-Paxton order 0): `f = erfc(x) / 2`.
    Fermi-Dirac:                          `f = sigmoid(-x)`.
    """
    if family == "gaussian":
        return 0.5 * torch.erfc(x)
    if family == "fermi":
        return torch.sigmoid(-x)
    raise ValueError(f"unknown smearing family {family!r}; expected gaussian or fermi")


def occupation_slope(x: torch.Tensor, width: float,
                     family: str = SMEARING_FAMILY) -> torch.Tensor:
    """`df/deps`, negative. Bounded by `1/(width sqrt(pi))` Gaussian, `1/(4 width)` FD.

    This is the factor the Daleckii-Krein divided difference reduces to at coincidence, so
    its bound is what keeps the density-response backward finite at exact degeneracy.
    """
    if family == "gaussian":
        return -torch.exp(-x * x) / (width * _SQRT_PI)
    if family == "fermi":
        f = torch.sigmoid(-x)
        return -f * (1.0 - f) / width
    raise ValueError(f"unknown smearing family {family!r}")


def entropy_of(x: torch.Tensor, family: str = SMEARING_FAMILY) -> torch.Tensor:
    """The generalised entropy per state, so that `F = sum_k f_k eps_k - width * sum_k S_k`.

    Gaussian: `S_k = exp(-x^2) / (2 sqrt(pi))`, which is NOT the Shannon entropy of `f` --
    it is the Methfessel-Paxton term that makes `F` variational in the occupations, which is
    what the Hellmann-Feynman force argument needs.
    Fermi-Dirac: the usual `-(f ln f + (1-f) ln(1-f))`.
    """
    if family == "gaussian":
        return torch.exp(-x * x) / (2.0 * _SQRT_PI)
    if family == "fermi":
        f = occupation_of(x, "fermi").clamp(1e-12, 1.0 - 1e-12)
        return -(f * f.log() + (1.0 - f) * (1.0 - f).log())
    raise ValueError(f"unknown smearing family {family!r}")


def _col(mu):
    """`mu` shaped to broadcast against a spectrum's last axis.

    `mu` is one number per spectrum, so it is `[...]` where the spectrum is `[..., n]`.
    `unsqueeze(-1)` is right in both the scalar case (a 0-dim tensor becomes `[1]`, which
    broadcasts against `[n]`) and the batched one (`[B]` becomes `[B, 1]`).
    """
    return mu.unsqueeze(-1) if torch.is_tensor(mu) else mu


def find_mu(eps: torch.Tensor, n_electrons, width: float,
            family: str = SMEARING_FAMILY, tol: float = 1e-10,
            max_iter: int = 200) -> torch.Tensor:
    """The `mu` that puts exactly `n_electrons` in the spectrum, by bisection under no_grad.

    Detaching is exact rather than approximate: at fixed N the free energy is stationary in
    the fill, so `mu` does not appear in `dF/deps`.

    BATCHED, AND WITHOUT A HOST SYNCHRONISATION IN THE LOOP -- which is section 1.3's whole
    finding. The previous version tested `float(total) > n_electrons` and `hi - lo < tol` on
    the host every iteration, so one bisection cost ~40 GPU->CPU synchronisations; the head
    runs ten of them per graph per pass, and the component profile put `head/bisect` at 28%
    of the step's CPU time and ~6900 `_local_scalar_dense` calls -- more than every other
    region together, and none of it arithmetic. Bisection halves its bracket by a fixed
    factor, so the iteration count that reaches `tol` is known in advance from the initial
    bracket alone: it is computed once, from ONE synchronisation, and the loop then runs on
    tensors with no host traffic at all. The result is bit-identical to a run of the old
    loop that exits on the same bracket width.

    `eps` is `[..., n_states]` and `n_electrons` a float or a tensor broadcastable to
    `eps.shape[:-1]`; the return is a tensor of shape `eps.shape[:-1]`, so a 1-D spectrum
    gives a 0-dim tensor that broadcasts exactly where the old float did.
    """
    with torch.no_grad(), mark("head/bisect"):
        e = eps.detach()
        n = torch.as_tensor(n_electrons, dtype=e.dtype, device=e.device)
        if n.dim() == 0:
            n = n.expand(e.shape[:-1])
        lo = e.amin(dim=-1) - 50.0 * width - 1.0
        hi = e.amax(dim=-1) + 50.0 * width + 1.0
        # ONE synchronisation, outside the loop. After k halvings the bracket is
        # `span / 2^k`, so `ceil(log2(span / tol))` iterations reach `tol` -- the same
        # stopping point the old loop's `hi - lo < tol` test found, decided in advance.
        span = float((hi - lo).max())
        iters = min(int(max_iter),
                    int(math.ceil(math.log2(max(span, tol) / tol))) + 1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            total = occupation_of((e - mid.unsqueeze(-1)) / width, family).sum(dim=-1)
            too_many = total > n
            hi = torch.where(too_many, mid, hi)
            lo = torch.where(too_many, lo, mid)
        return 0.5 * (lo + hi)


def use_smearing(family: str = SMEARING_FAMILY, width: float = SMEARING_WIDTH):
    """Select the smearing family process-wide and return the previous setting.

    A module-level switch rather than an argument threaded through eight call sites: the
    family must be the SAME everywhere in one forward -- the fill, the entropy and the
    density-response backward are three views of one convention, and a call site that missed
    the argument would silently mix Gaussian occupations with a Fermi-Dirac backward.
    Restoring the previous value is the caller's job; `smearing()` reads it.
    """
    global _FAMILY, _WIDTH
    previous = (_FAMILY, _WIDTH)
    if family not in ("gaussian", "fermi"):
        raise ValueError(f"unknown smearing family {family!r}")
    _FAMILY, _WIDTH = family, float(width)
    return previous


def smearing():
    """The live `(family, width)`."""
    return _FAMILY, _WIDTH


def fermi_fill(eps: torch.Tensor, n_electrons, t_el: float = T_EL,
               tol: float = 1e-10, max_iter: int = 200, mu=None) -> torch.Tensor:
    """Fermi-Dirac occupations at the `mu` that puts exactly `n_electrons` in the spectrum.

    Bisection, under `no_grad`. `mu` is a function of the eigenvalues, but the free energy's
    derivative does not contain it (Hellmann-Feynman at fixed N), so detaching is exact rather
    than approximate -- see the module docstring.
    """
    family = _FAMILY
    if mu is None:
        mu = find_mu(eps, n_electrons, t_el, family, tol=tol, max_iter=max_iter)
    return occupation_of((eps - _col(mu)) / t_el, family)


class SlaterKosterH(nn.Module):
    """Assembles the s+p Hamiltonian from Stage 2's radial scales and bounded corrections.

    Deliberately NOT a subclass of the spectral heads: those own a four-channel eigenproblem
    and a `mu_c` per channel, and this one owns a single spin-resolved spectrum with an
    occupation. Inheriting would carry the machinery Edit 4 exists to delete.
    """

    def __init__(self, num_elements: int, feature_dim: int, elem_dim: int = 8,
                 hidden: int = 64, atomic_numbers: Optional[Sequence[int]] = None,
                 decay_length: float = 1.0,
                 r_cut: float = 10.0, on_site_range: float = 1.0,
                 hop_range: float = 0.5, envelope: str = ENVELOPE_DEFAULT,
                 hop_form: str = HOP_FORM_DEFAULT,
                 hop_log_beta: float = HOP_LOG_BETA_DEFAULT,
                 decay_learned: bool = False,
                 decay_log_beta: float = DECAY_LOG_BETA_DEFAULT) -> None:
        super().__init__()
                
        if envelope not in ENVELOPES:
            raise ValueError(f"unknown radial envelope {envelope!r}; expected one of "
                             f"{sorted(ENVELOPES)}")
        if hop_form not in HOP_FORMS:
            raise ValueError(f"unknown hopping modulation {hop_form!r}; expected one of "
                             f"{sorted(HOP_FORMS)}")
        self.hop_form = str(hop_form)
        self.hop_log_beta = float(hop_log_beta)
        self.decay_length, self.r_cut = float(decay_length), float(r_cut)
        self.envelope = str(envelope)
        # Section 2.3: the envelope's anchor, per species pair, from the universal covalent
        # radii. A BUFFER, so it travels with the checkpoint and the config round trip does
        # not have to reconstruct it from an element list. `d_ref` the scalar is gone from
        # the constructor and from every launcher; models pickled with one keep it in their
        # `__dict__` and `radial` still honours it, which is what keeps the earlier cohorts
        # scorable.
        if atomic_numbers is None:
            raise ValueError(
                "SlaterKosterH needs atomic_numbers to anchor its hopping envelope; the "
                "per-host `d_ref` it replaces was the constant section 2.3 removes")
        self.register_buffer("d_ref_pair", bond_reference(atomic_numbers))
        # Section 2.4 of the Stage A' spec. One scalar per Slater-Koster integral type,
        # shared across every host; `decay_length` becomes L0. The parameter exists even when
        # it is not learned, at zero, so the fixed-length head is the u_b = 0 point of the
        # same model and the config round trip has one shape to carry.
        self.decay_learned = bool(decay_learned)
        self.decay_log_beta = float(decay_log_beta)
        self.decay_u = nn.Parameter(torch.zeros(len(BOND_TYPES)),
                                    requires_grad=bool(decay_learned))
        self.on_site_range, self.hop_range = float(on_site_range), float(hop_range)
        self.elem = nn.Embedding(num_elements, elem_dim)

        # Harrison's universal coefficients. eta_ss_sigma is shared with Edit 3 so the two
        # stages cannot disagree about the scale they start from.
        eta = torch.tensor([ETA_SS_SIGMA, 1.84, 3.24, -0.81])
        # Harrison's rule AT EACH PAIR'S OWN REFERENCE, not at one host bond length: the
        # universal scaling is `eta hbar^2 / (m d^2)` and `d` is a property of the pair.
        scale = HBAR2_OVER_M / (self.d_ref_pair ** 2)            # [n_el, n_el]
        self.v0_raw = nn.Parameter(eta.reshape(1, 1, 4) * scale.unsqueeze(-1))
        # Two on-site levels per species, one per shell.
        self.eps0 = nn.Parameter(torch.zeros(num_elements, 2))
        # A1: the on-site half-width per species and shell. A BUFFER, not a parameter: it is
        # a registered bound, not something the fit may widen. Filled uniformly from
        # `on_site_range` here; `H0` overwrites it with the registered rule once the Harrison
        # baselines are in place.
        self.register_buffer("delta", torch.full((num_elements, 2), float(on_site_range)))
        # A1.1: per-species input standardisation. Buffers, so they travel with the
        # checkpoint; identity until `set_feature_stats` is called.
        self.register_buffer("feat_mean", torch.zeros(num_elements, feature_dim))
        self.register_buffer("feat_sd", torch.ones(num_elements, feature_dim))
        # `1/sd` on live channels, 0 on dead ones -- a multiply, so a dead channel is exactly
        # zero rather than a division by something near zero.
        self.register_buffer("feat_scale", torch.ones(num_elements, feature_dim))
        self.register_buffer("feat_stats_set", torch.tensor(False))
        self.hop = _mlp([2 * feature_dim + 2 * elem_dim, hidden, hidden, len(BOND_TYPES)],
                        final_scale=0.05)
        self.site = _mlp([feature_dim + elem_dim, hidden, hidden, 2], final_scale=0.05)

    # ---------------------------------------------------------------- elements

    def v0(self, species_i, species_j):
        """Symmetric in the species pair. sp-sigma's antisymmetry lives in the SK block's
        sign convention, not in this table -- putting it here as well would apply it twice."""
        m = 0.5 * (self.v0_raw + self.v0_raw.transpose(0, 1))
        return m[species_i, species_j]

    def radial(self, r: torch.Tensor, species_i=None,
               species_j=None) -> torch.Tensor:
        """The distance dependence of every hopping integral, times the cutoff taper.

        TWO FAMILIES, and the choice is a measurement rather than a preference (see
        `defect-perovskite/b7_envelope_choice.py`).

        `"exp"` is what Stage 3 ran: `exp(-(r - d_ref) / decay_length)`. It equals 1 at
        `d_ref` by construction, which is where `v0` is initialised from Harrison's rule, and
        then falls far faster than Harrison does. At the vacancy-flanking Pb-Pb separation of
        5-7 A the measured `t / t_Harrison` is 0.14 +- 0.07 while the learned pair modulation
        -- the head's only lever on that bond -- sits pinned at its +50% bound on the close
        frames. The head is straining against this function and losing.

        `"power"` is Harrison's own `(d_ref / r)^2`, so the envelope and the initialisation
        stop disagreeing about what the radial dependence is. It is not a free constant being
        retuned: it REMOVES one. An exponential cannot carry a power law's log-slope at two
        separations at once -- matching at `d_ref` needs 1.4 A and matching at the hub bond
        needs about 2.9 A -- so retuning `decay_length` only moves which separation is wrong.

        Both keep the same `(1 - (r/r_cut)^6)^2` taper, so the graph stays finite and the
        two differ in exactly one factor.

        `getattr` rather than `self.envelope`: models pickled before this attribute existed
        must keep evaluating, and they were all exponential.
        """
        x = (r / self.r_cut).clamp(max=1.0)
        taper = (1.0 - x ** 6) ** 2
        d = self._anchor(r, species_i, species_j)
        if getattr(self, "envelope", ENVELOPE_DEFAULT) == "power":
            return (d / r.clamp_min(1e-9)) ** 2 * taper
        if getattr(self, "decay_learned", False):
            # [n_edges, 4]: one length per integral type (spec section 2.4).
            lengths = self.decay_lengths().to(r.dtype)
            return (torch.exp(-(r - d).unsqueeze(-1) / lengths.unsqueeze(0))
                    * taper.unsqueeze(-1))
        return torch.exp(-(r - d) / self.decay_length) * taper

    def _anchor(self, r: torch.Tensor, species_i=None, species_j=None):
        """`d_ref` per edge: the pair's covalent-radius sum, or a pickled model's scalar.

        The legacy branch is what keeps every cohort trained before section 2.3 scorable --
        those models carry a float `d_ref` in their `__dict__` and no `d_ref_pair` buffer,
        and re-anchoring them would silently rescale every hopping they learned."""
        pair = getattr(self, "d_ref_pair", None)
        if pair is None or species_i is None or species_j is None:
            return float(getattr(self, "d_ref", 2.8))
        return pair.to(r.dtype)[species_i.long(), species_j.long()]

    def decay_lengths(self) -> torch.Tensor:
        """`L_b = L0 * exp(beta_L * tanh u_b)` per integral type, `[4]`, in Angstrom.

        Reports the fixed length for every type when the lengths are not learned (or on a
        model pickled before `decay_u` existed), so a diagnostic can always ask.
        """
        u = getattr(self, "decay_u", None)
        if u is None or not getattr(self, "decay_learned", False):
            return torch.full((len(BOND_TYPES),), float(self.decay_length),
                              device=self.v0_raw.device, dtype=self.v0_raw.dtype)
        return float(self.decay_length) * torch.exp(
            float(self.decay_log_beta) * torch.tanh(u))

    def integrals(self, feats_i, feats_j, r, species_i, species_j) -> torch.Tensor:
        """The four radial integrals per edge, [n_edges, 4]. A1.1: each end is standardised
        by ITS OWN species before the symmetric combination, so the pair descriptor is built
        from deviations rather than from two large species means."""
        e_i, e_j = self.elem(species_i), self.elem(species_j)
        feats_i = self.standardise(feats_i, species_i)
        feats_j = self.standardise(feats_j, species_j)
        sym = torch.cat([feats_i + feats_j, (feats_i - feats_j).abs(),
                         e_i + e_j, (e_i - e_j).abs()], dim=-1)
        pre = self.hop(sym)
        self._audit_store("hop", pre)
        if getattr(self, "_reg", False):
            self._reg_store("hop", pre)
        correction = self.modulation(pre)
        radial = self.radial(r, species_i, species_j)
        if radial.dim() == 1:
            radial = radial.unsqueeze(-1)
        return self.v0(species_i, species_j) * radial * correction

    def modulation(self, pre: torch.Tensor) -> torch.Tensor:
        """The environment factor multiplying `v0 * radial`. Exactly 1 at `pre = 0`.

        WHY THE SECOND FORM EXISTS, measured rather than supposed. On the vacancy-flanking
        Pb-Pb bond the trained cohort sits AT the linear form's stop: ss-sigma pinned at the
        lower bound, pp-sigma and pp-pi at the upper one, in every d bin. A parameter at its
        stop has gradient `sech^2 ~ 0` -- it looks like it is learning and it is not -- and a
        scaling what-if on that bond moves F4 from -0.061 to -0.090 while the 79-atom force
        loss falls, so the stop is binding on something the data wants.

        The linear form cannot simply be widened without breaking the bulk: `1 + b*tanh(g)`
        with `b > 1` can drive an integral through zero and out the other side, changing the
        SIGN of a hopping the Harrison initialisation fixed. The log form has no such branch
        -- `exp` is positive everywhere -- and it is symmetric, so `beta = ln 3` gives
        x[1/3, 3] rather than the linear form's lopsided [1/2, 3/2].

        `getattr` on both attributes: models pickled before either existed must keep
        evaluating, and they were all linear.
        """
        if getattr(self, "hop_form", HOP_FORM_DEFAULT) == "log":
            beta = float(getattr(self, "hop_log_beta", HOP_LOG_BETA_DEFAULT))
            return torch.exp(beta * torch.tanh(pre))
        return 1.0 + self.hop_range * torch.tanh(pre)

    _audit = False
    _audit_bin: Dict[str, list] = {}

    def audit(self, on: bool = True) -> Dict[str, torch.Tensor]:
        """Turn on pre-tanh capture and return the dict the head writes into.

        NAME-PROOF BY CONSTRUCTION. The previous audit spied on submodules by guessing their
        attribute names, matched none, and reported nothing -- a silent null. The tensors are
        now stored by the code that computes them, so the audit cannot miss a channel that
        exists or invent one that does not.

        WHICH CLASS THIS IS ON, and it is not the one the plan named. Stage 3 replaced the
        spectral head wholesale, so a Stage-3 model contains NO `BoundedLocalHead`: the
        bounded forms that are live are these two, on `SlaterKosterH`. Auditing
        `BoundedLocalHead` would have audited a module the models do not contain.
        """
        self._audit = bool(on)
        if not on:
            self._audit_bin = {}
        return self._audit_bin

    def _audit_store(self, name: str, value: torch.Tensor) -> None:
        if getattr(self, "_audit", False):
            self._audit_bin.setdefault(name, []).append(value.detach().reshape(-1).cpu())

    # ------------------------------------------------- A1.1: input standardisation

    @torch.no_grad()
    def set_feature_stats(self, mean: torch.Tensor, sd: torch.Tensor) -> Dict[str, int]:
        """Freeze the per-species channel mean and sd of the training ensemble.

        A channel the base does not vary across the ensemble is DEAD: it is zeroed, not
        divided by. The threshold is `FEATURE_SD_FLOOR` times that species' median channel sd.
        The count of dead channels is returned, because a large count means the base is
        handing the head channels it cannot use, and that is worth seeing.

        A species whose every channel is dead -- a symmetry-perfect cell, where all its atoms
        are equivalent -- standardises to all zeros, so its readouts contribute their bias and
        nothing else. Degenerate, but finite and stable, which is the point."""
        dead = {}
        for z in range(self.feat_mean.shape[0]):
            m, s_ = mean[z].to(self.feat_mean.dtype), sd[z].to(self.feat_sd.dtype)
            med = float(s_.median())
            lo = FEATURE_SD_FLOOR * med
            alive = s_ > lo if med > 0 else torch.zeros_like(s_, dtype=torch.bool)
            dead[z] = int((~alive).sum())
            self.feat_mean[z] = m
            self.feat_sd[z] = torch.where(alive, s_, torch.ones_like(s_))
            self.feat_scale[z] = torch.where(alive, 1.0 / s_.clamp_min(1e-30), torch.zeros_like(s_))
        self.feat_stats_set.fill_(True)
        return dead

    def standardise(self, feats: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        """`(h - mu_Z) / sd_Z`. Identity while the stats are unset (a freshly constructed
        module), which is why `MACEDSCC` refuses to run without them."""
        if not bool(self.feat_stats_set):
            return feats
        return (feats - self.feat_mean[species].to(feats.dtype)) * self.feat_scale[species].to(feats.dtype)

    # ------------------------------------------------------- A1: the L2 on tanh outputs

    def collect_regularisation(self, on: bool = True) -> None:
        """Accumulate `sum(tanh^2)` and a count per term, ATTACHED, for A1's weak L2.

        Separate from `audit`, which detaches and moves to the CPU because it is a
        diagnostic. This one is in the loss, so it stays on the graph, and it stores two
        scalars per term rather than every value: a training step evaluates `H0` several
        times (SCF iterations, the gap regulariser) and the penalty is the pooled mean over
        all of them, not a per-call mean averaged again."""
        self._reg = bool(on)
        self._reg_bin = {}

    def reset_regularisation(self) -> None:
        self._reg_bin = {}

    def _reg_store(self, name: str, pre: torch.Tensor) -> None:
        """Takes the PRE-ACTIVATION, not the tanh: the barrier needs the unsquashed value,
        and `tanh` of it is what the L2 wants, so storing `pre` serves both."""
        if not getattr(self, "_reg", False):
            return
        prev = getattr(self, "_reg_bin", None)
        if prev is None:
            prev = self._reg_bin = {}
        knee = float(getattr(self, "saturation_knee", SATURATION_KNEE))
        s_l2 = (torch.tanh(pre) ** 2).sum()
        s_bar = ((pre.abs() - knee).clamp_min(0.0) ** 2).sum()
        n = int(pre.numel())
        old = prev.get(name)
        prev[name] = (s_l2, s_bar, n) if old is None else (old[0] + s_l2, old[1] + s_bar, old[2] + n)

    def regularisation(self) -> Dict[str, torch.Tensor]:
        """`mean(tanh^2)` per term over everything seen since the last reset."""
        return {k: v[0] / max(v[2], 1) for k, v in getattr(self, "_reg_bin", {}).items()}

    def barrier(self) -> Dict[str, torch.Tensor]:
        """`mean(relu(|pre| - knee)^2)` per term: zero in normal operation, and its gradient
        GROWS with the excursion instead of vanishing the way the L2's does."""
        return {k: v[1] / max(v[2], 1) for k, v in getattr(self, "_reg_bin", {}).items()}

    def on_site(self, feats, species, madelung: Optional[torch.Tensor] = None):
        """`[n_nodes, 2]`: the s and p levels, `eps_Z + delta_Z tanh(e_Z(h_i))` (A1).

        The Madelung shift is added to BOTH shells identically. It is the electrostatic
        potential at a site and has no angular-momentum dependence; giving the shells
        different shifts would be inventing a crystal-field term and calling it electrostatics.

        NO REFERENCE. The species-constant mode that the centre used to remove (b4 measured
        +0.27 eV on every atom) is now absorbed where it belongs: a constant per species is
        the readout's own bias against `eps0`, the two are redundant by one constant per
        species, and the bound plus the L2 keep that redundancy harmless. A constant on every
        level of every atom is a gauge shift the energy expression is invariant to; a
        constant on one species is a shift of that species' baseline, which `eps0` already
        parameterises."""
        e = self.elem(species)
        pre_site = self.site(torch.cat([self.standardise(feats, species), e], dim=-1))
        self._audit_store("site", pre_site)
        corr = torch.tanh(pre_site)
        self._reg_store("site", pre_site)
        levels = self.eps0[species] + self.delta[species].to(corr.dtype) * corr
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


    def batched(self, node_feats, node_species, edge_index, edge_vector,
                local: torch.Tensor, edge_graph: torch.Tensor, n_nodes: int,
                num_graphs: int) -> torch.Tensor:
        """Dense `H` for a whole batch of EQUAL-SIZED graphs at once: `[B, 4n, 4n]`.

        Section 1.1. The per-graph `forward` above is the reference and stays the reference:
        this returns the same matrices, stacked, and `test_batched_head.py` asserts it to
        1e-8 on eigenvalues and on `P`. What it removes is the python loop -- one call to the
        hopping MLP for every edge in the batch instead of one per graph, one `index_put`
        instead of B of them -- which the section 1.3 profile identified as the step's cost:
        the head's arithmetic is small and its OPERATION COUNT is not.

        `local` is each node's index within its own graph and `edge_graph` each edge's graph.
        Both are the caller's, because the caller already knows the batch is size-uniform;
        recomputing them here would hide the precondition that makes this legal.
        """
        src, dst = edge_index[0], edge_index[1]
        r = edge_vector.norm(dim=-1).clamp_min(1e-9)
        direction = edge_vector / r.unsqueeze(-1)
        v = self.integrals(node_feats[src], node_feats[dst], r,
                           node_species[src], node_species[dst])
        blocks = sk_block(direction, v)                        # [n_edges, 4, 4]

        dim = int(n_nodes) * ORBITALS_PER_ATOM
        o = torch.arange(ORBITALS_PER_ATOM, device=src.device)
        rows = local[src].reshape(-1, 1, 1) * ORBITALS_PER_ATOM + o.reshape(1, -1, 1)
        cols = local[dst].reshape(-1, 1, 1) * ORBITALS_PER_ATOM + o.reshape(1, 1, -1)
        flat = (edge_graph.reshape(-1, 1, 1) * (dim * dim) + rows * dim + cols)
        H = torch.zeros(num_graphs * dim * dim, device=node_feats.device,
                        dtype=node_feats.dtype)
        H = H.index_put((flat.expand(-1, ORBITALS_PER_ATOM,
                                     ORBITALS_PER_ATOM).reshape(-1),),
                        blocks.reshape(-1), accumulate=True)
        H = H.reshape(num_graphs, dim, dim)
        return 0.5 * (H + H.transpose(-1, -2))


def _dk_eigenbasis(lam: torch.Tensor, f: torch.Tensor, t_el: float, tol: float,
                   Ghat: torch.Tensor, mu=None) -> torch.Tensor:
    """One fill's Daleckii-Krein map, already in the eigenbasis: `Ghat` in, `M` out.

    Split from the basis round trip because the multi-fill Function applies FOUR of these to
    ONE `Ghat` and returns one `U M U^T`. Rotating in and out per fill instead costs three
    extra 636x636 float64 GEMMs each, which measured as a quarter of the wiring's overhead.
    """
    # `mu` COMES FROM THE FORWARD, it is not re-derived here. Recovering it by inverting the
    # occupations needs states with a fractional filling, and under Gaussian smearing there
    # may be none: erfc/2 drops below 1e-6 within 3.5 widths, so a spectrum whose frontier
    # sits in a gap has every f at exactly 0 or 1 and the inversion has nothing to work with.
    # The fallback that used to cover that case returned the spectrum's MEDIAN, which is not
    # mu at all -- it put f' at the wrong energy and the degenerate-limit finite-difference
    # check failed by a factor of 16.
    if mu is None:
        mu = _mu_from(lam, f, t_el)
    fp = occupation_slope((lam - _col(mu)) / t_el, t_el, _FAMILY)   # f'(lam); bounded
    dl = lam.unsqueeze(-1) - lam.unsqueeze(-2)
    df = f.unsqueeze(-1) - f.unsqueeze(-2)
    near = dl.abs() <= tol
    # The divided difference away from coincidence, its limit at it. `torch.where` alone
    # would still evaluate the singular branch and poison the gradient with NaN, so the
    # denominator is made safe BEFORE the division. The limit is the family's own slope --
    # bounded by 1/(sigma sqrt(pi)) Gaussian, 1/(4 T) Fermi-Dirac -- so degeneracy is finite
    # under both.
    safe = torch.where(near, torch.ones_like(dl), dl)
    mid = 0.5 * (lam.unsqueeze(-1) + lam.unsqueeze(-2))
    L = torch.where(near, occupation_slope((mid - _col(_col(mu))) / t_el, t_el, _FAMILY),
                    df / safe)

    M = L * Ghat
    # Fixed-N correction (mu moves with H), PER FRAME. The verbatim single-frame original
    # tested `denom > 1e-12` as one scalar; applied to a batch that test read `.all()`,
    # which switched the correction off for EVERY frame whenever one frame's frontier sat
    # in a gap (all occupations saturated, f' = 0) -- the batched coupled gradient then
    # disagreed with the per-frame one by orders of magnitude on the other frames
    # (found 2026-09-07 by the batched-vs-per-graph gradient test; fixed here, the only
    # deliberate departure from the verbatim copy). Frames with a saturated frontier get
    # no correction (0/0 -> 0), the others their own.
    denom = fp.sum(-1)
    ok = denom.abs() > 1e-12
    if bool(ok.any()):
        num = (fp * torch.diagonal(Ghat, dim1=-2, dim2=-1)).sum(-1)
        safe = torch.where(ok, denom, torch.ones_like(denom))
        ratio = torch.where(ok, num / safe, torch.zeros_like(denom))
        M = M - torch.diag_embed(fp * ratio.unsqueeze(-1))
    return M


def _dk_backward(lam: torch.Tensor, U: torch.Tensor, f: torch.Tensor, t_el: float,
                 tol: float, G: torch.Tensor, mu=None) -> torch.Tensor:
    """One fill's Daleckii-Krein pullback: a cotangent on `P` becomes one on `H`."""
    # H is symmetric, so only the symmetric part of the cotangent can act on it.
    G = 0.5 * (G + G.transpose(-1, -2))
    Ghat = U.transpose(-1, -2) @ G @ U
    return U @ _dk_eigenbasis(lam, f, t_el, tol, Ghat, mu) @ U.transpose(-1, -2)


def _mu_from(lam: torch.Tensor, f: torch.Tensor, t_el: float) -> torch.Tensor:
    """Recover `mu` from the occupations, for the degenerate-limit branch.

    Read back rather than threaded through: it is exact wherever `f` is not saturated, and the
    branch it feeds only fires for eigenvalue pairs that have collided -- which cannot all be
    saturated, or the divided difference would be zero either way.
    """
    interior = (f > 1e-6) & (f < 1.0 - 1e-6)
    if not bool(interior.any()):
        return lam.median()
    if _FAMILY == "gaussian":
        # `f = erfc(x)/2` inverts as `x = erfinv(1 - 2f)`, so `mu = lam - width * x`.
        x = torch.erfinv((1.0 - 2.0 * f[interior]).clamp(-1 + 1e-12, 1 - 1e-12))
        return (lam[interior] - t_el * x).median()
    return (lam[interior] + t_el * torch.log(
        f[interior] / (1.0 - f[interior]))).median()


# Harrison universal coefficients, in BOND_TYPES order.
HARRISON_ETA = (-1.40, 1.84, 3.24, -0.81)


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


def harrison_initialise(head, atomic_numbers: Sequence[int]) -> None:
    """Set on-site levels and hopping scales from the Harrison tables, in place.

    On-sites are the tabulated free-atom term values per species and shell. Hoppings are
    `eta_b * hbar^2 / (m d^2)` at each PAIR's covalent-radius reference, per bond type.

    SECTION 2.3 REMOVED THE BOND LENGTH. It used to be a measured per-host number -- 2.861 A
    for CsPbCl3, passed on the command line -- and the same number anchored the envelope, so
    a host entered the head through two doors. The anchor is now `r_cov(s) + r_cov(s')` from
    a universal table, and this function reads the head's own `d_ref_pair` buffer so the
    initialisation and the envelope cannot disagree about what the reference separation is.
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
        pair = getattr(head.h, "d_ref_pair", None)
        if pair is None:
            raise ValueError(
                "the head has no d_ref_pair buffer; it was built before section 2.3 and "
                "re-initialising it here would anchor its hoppings at a different "
                "separation than its envelope uses")
        scale = HBAR2_OVER_M_COUNTING / (pair.to(head.h.v0_raw.dtype) ** 2)
        eta = torch.tensor(HARRISON_ETA, dtype=head.h.v0_raw.dtype,
                           device=head.h.v0_raw.device)
        head.h.v0_raw.copy_(eta.reshape(1, 1, 4) * scale.unsqueeze(-1))


HBAR2_OVER_M_COUNTING = 7.62      # eV A^2; same constant Edit 3 uses, named here to avoid


def frame_key(atomic_numbers, positions, cell) -> int:
    """A content hash of the geometry, as a signed 64-bit integer.

    Positions are rounded to 1e-6 A and the cell to 1e-6 A before hashing, so a frame
    re-read from a file with a different float formatting still keys to the same entry; two
    frames that differ by more than that are different frames. Label-free by construction:
    only numbers, positions and cell enter.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(np.asarray(atomic_numbers, dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(positions, dtype=np.float64), 6).tobytes())
    h.update(np.round(np.asarray(cell, dtype=np.float64), 6).reshape(-1).tobytes())
    return int(np.frombuffer(h.digest(), dtype=np.int64)[0])

