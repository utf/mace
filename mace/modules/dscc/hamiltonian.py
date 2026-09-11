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

    b_i = b_Z (1 + beta_b tanh(g_Z(h_i))),    a_i = a_Z (1 + beta_a tanh(f_Z(h_i))),

reference-free like the scalar term. `beta_b = beta_a = 0` recovers the species-level
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
                                      ORBITALS_PER_ATOM, SlaterKosterH, _mlp,
                                      harrison_initialise)

# Registered defaults (plan section 11: to register before use). `r_cut` is the old head's
# 10 A; `q_cut` reaches the first shell and the Cs neighbours (Pb-Cl 2.8, Cs-Cl 3.5-4.1 A).
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
    inside = r < q_cut
    r_in, v_in = r[inside], edge_vector[inside]
    unit = v_in / r_in.unsqueeze(-1)
    w = (1.0 - (r_in / q_cut) ** 6) ** 2
    outer = unit.unsqueeze(-1) * unit.unsqueeze(-2) - torch.eye(3, dtype=r.dtype, device=r.device) / 3.0
    contrib = w.reshape(-1, 1, 1) * outer
    Q = torch.zeros(n_nodes, 3, 3, dtype=r.dtype, device=r.device)
    return Q.index_add(0, sender[inside], contrib)


class H0(nn.Module):
    """The runtime Hamiltonian for ONE graph (dense; the batched path stacks equal sizes)."""

    def __init__(self, atomic_numbers: Sequence[int], feature_dim: int = 128,
                 n_vectors: int = 128, r_cut: float = R_CUT_DEFAULT,
                 q_cut: float = Q_CUT_DEFAULT, directional: bool = True,
                 a_max: float = A_MAX_DEFAULT, b_max: float = B_MAX_DEFAULT,
                 on_site_range: float = ON_SITE_RANGE_DEFAULT, hidden: int = 64,
                 eta: float = ETA_DEFAULT, beta_b: float = 0.0, beta_a: float = 0.0,
                 delta_frac: float = DELTA_FRACTION_DEFAULT,
                 readout_hidden: int = READOUT_HIDDEN_DEFAULT) -> None:
        super().__init__()
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        num_elements = len(self.atomic_numbers)
        self.r_cut, self.q_cut = float(r_cut), float(q_cut)
        self.directional = bool(directional)
        self.a_max, self.b_max = float(a_max), float(b_max)
        self.eta, self.beta_b, self.beta_a = float(eta), float(beta_b), float(beta_a)
        self.delta_frac = float(delta_frac)
        self.sk = SlaterKosterH(num_elements, feature_dim, atomic_numbers=self.atomic_numbers,
                                r_cut=self.r_cut, envelope="exp", hop_form="log",
                                decay_learned=True, hop_log_beta=self.eta,
                                on_site_range=on_site_range, hidden=hidden)
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

    def env_factor(self, readout: nn.Module, bound: float, scalars: torch.Tensor,
                   species: torch.Tensor, name: str) -> Optional[torch.Tensor]:
        """A1's `1 + bound * tanh(readout(h_i, Z_i))`, `[N]`, or None when the bound is zero.

        None rather than a tensor of ones: the factorial's "species coefficient" variant
        must not evaluate the readout, so that turning the term off removes it from the
        graph rather than multiplying by one."""
        if not bound:
            return None
        pre = readout(torch.cat([self.sk.standardise(scalars, species),
                                 self.elem_env(species).to(scalars.dtype)], dim=-1))
        t = torch.tanh(pre).squeeze(-1)
        self.sk._reg_store(name, pre)
        return 1.0 + float(bound) * t

    def site_coefficients(self, scalars: torch.Tensor, species: torch.Tensor):
        """`(a_i, b_i)`, `[N]` each: the species coefficients with A1's environment factors."""
        a_z, b_z = self.coefficients()
        a_i, b_i = a_z[species].to(scalars.dtype), b_z[species].to(scalars.dtype)
        fa = self.env_factor(self.f_read, self.beta_a, scalars, species, "rank1")
        fb = self.env_factor(self.g_read, self.beta_b, scalars, species, "rank2")
        return (a_i if fa is None else a_i * fa), (b_i if fb is None else b_i * fb)

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
                   sites: Optional[torch.Tensor] = None) -> dict:
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
        if edge_index is not None:
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
        if sites is not None and bool(sites.any()):
            out["site_on_site"] = [float(x) for x in t_site[sites].reshape(-1)]
            if self.beta_b:
                out["site_rank2"] = [float(x) for x in torch.tanh(self.g_read(env[sites])).reshape(-1)]
            if self.beta_a:
                out["site_rank1"] = [float(x) for x in torch.tanh(self.f_read(env[sites])).reshape(-1)]
        return out

    def directional_block(self, scalars: torch.Tensor, vectors: torch.Tensor,
                          species: torch.Tensor, edge_index: torch.Tensor,
                          edge_vector: torch.Tensor) -> torch.Tensor:
        """`H_onsite_dir` as a dense `[4N, 4N]` block-diagonal matrix."""
        return torch.block_diag(*self.directional_site_blocks(scalars, vectors, species,
                                                              edge_index, edge_vector))

    def batched(self, scalars: torch.Tensor, vectors: Optional[torch.Tensor], species: torch.Tensor,
                edge_index: torch.Tensor, edge_vector: torch.Tensor, batch: torch.Tensor,
                num_graphs: int, n_nodes: int) -> torch.Tensor:
        """Dense `H0` for a batch of EQUAL-SIZED graphs at once, `[B, 4n, 4n]`: the same
        matrices `forward` builds one at a time (asserted equal in the tests), without the
        per-graph Python loop and its slicing -- which the training-step profile put at a
        third of the backward. `edge_index` is global (batch-concatenated), `batch` the
        graph of every node."""
        scalars = scalars.to(torch.float64)
        edge_vector = edge_vector.to(torch.float64)
        local = torch.arange(scalars.shape[0], device=scalars.device) - batch * n_nodes
        edge_graph = batch[edge_index[0]]
        H = self.sk.batched(scalars, species, edge_index, edge_vector, local, edge_graph, n_nodes, num_graphs)
        dim = n_nodes * ORBITALS_PER_ATOM
        levels = self.sk.on_site(scalars, species, None)                              # [N, 2]
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1)           # [N, 4]
        diag = diag.reshape(num_graphs, dim)
        eye = torch.eye(dim, dtype=H.dtype, device=H.device)
        H = H * (1.0 - eye) + torch.diag_embed(diag)
        if self.directional:
            if vectors is None:
                raise ValueError("the directional block needs the base's l = 1 features")
            blocks = self.directional_site_blocks(scalars, vectors.to(torch.float64), species,
                                                  edge_index, edge_vector)
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

    def directional_site_blocks(self, scalars: torch.Tensor, vectors: torch.Tensor,
                                species: torch.Tensor, edge_index: torch.Tensor,
                                edge_vector: torch.Tensor) -> torch.Tensor:
        """`H_onsite_dir` as `[N, 4, 4]` site blocks (global node indexing). `scalars` enters
        only through A1's environment factors, and only when their bounds are nonzero."""
        n = int(species.shape[0])
        a_i, b_i = self.site_coefficients(scalars.to(vectors.dtype), species)
        v = torch.einsum("nc,nca->na", self.vector_mix[species].to(vectors.dtype), vectors)
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
        scalars = scalars.to(torch.float64)
        edge_vector = edge_vector.to(torch.float64)
        H = self.sk(scalars, species, edge_index, edge_vector, madelung=None, n_nodes=n)
        levels = self.sk.on_site(scalars, species, None)
        diag = torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1).reshape(-1)
        H = H - torch.diag(torch.diagonal(H)) + torch.diag(diag)
        if self.directional:
            if vectors is None:
                raise ValueError("the directional block needs the base's l = 1 features")
            H = H + self.directional_block(scalars, vectors.to(torch.float64), species,
                                           edge_index, edge_vector)
        H = 0.5 * (H + H.transpose(0, 1))
        if self.site_shift is not None:
            H = H + torch.diag(self.site_shift.to(H.dtype).to(H.device).repeat_interleave(ORBITALS_PER_ATOM))
        if self.gauge_shift:
            H = H + float(self.gauge_shift) * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        return H
