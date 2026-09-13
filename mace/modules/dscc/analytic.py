"""Analytic position derivatives of the head (O3 speed plan, coadvisor 2026-09-13).

The head's force is a Hellmann-Feynman contraction,

    dS/dR_I,  S = <X, H0(R)>  at FIXED X,

and autograd was computing it by walking the assembly graph backwards: every `index_put`,
every `stack` in `sk_block`, every `index_add` in the descriptor, in float64, as separate
kernels. The contraction itself is O(edges x 16 x 3) arithmetic -- under a millisecond -- so
the traversal was almost all overhead. This module writes it out.

ONE kernel serves every consumer, because they differ only in `X`:

  * the band term `dF_band = Tr(dP H0)`   ->  X = dP,
  * `E_host`'s `d dq/dR` and `d q0/dR`    ->  X = the Frechet (Daleckii-Krein) pullback of
    the cotangents on the site occupations, `U (M_s + M_r) U^T`,

and both share the eigenvectors, so `head_cotangent` builds ONE `X` in the eigenbasis and
rotates out once (four n^3 GEMMs for the pair, against ten through autograd).

SCOPE. Analytic in the edge-scaling pieces (the Slater-Koster block, the descriptor
radials, the unit-vector projection), and a *small* autograd for the per-node readouts:
`A0`, `vectors` and `Q` are made leaves and the PRODUCTION `on_site` /
`directional_site_blocks` are called on them, so the block layout and A1's environment
factors are not transcribed a second time. Those tensors are [N, ...], not [E, ...].

The decoupled (O3) head only: with the base's first-block features in `H0` the position
derivative runs through the base's graph and there is nothing to write out. Forces only --
the stress keeps the autograd route (it needs the strain graph, and it is not on the timed
path).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple

import torch

from mace.modules.dscc.legacy import ORBITALS_PER_ATOM, _dk_eigenbasis


# ------------------------------------------------------------------ radial derivatives

def _taper(x: torch.Tensor, r_cut: float):
    """H0's cutoff taper `(1 - x^6)^2` and `d/dr` of it, with `x = (r/r_cut)` clamped at 1
    (the clamp makes the derivative zero beyond the cutoff, as autograd's does)."""
    inside = x < 1.0
    t = (1.0 - x ** 6) ** 2
    dt = torch.where(inside, -12.0 * x ** 5 * (1.0 - x ** 6) / r_cut, torch.zeros_like(x))
    return t, dt, inside


def sk_radial_and_derivative(sk, r: torch.Tensor, species_i: torch.Tensor,
                             species_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`SlaterKosterH.radial` and its `d/dr`, `[E, 4]` each (broadcast for the scalar-length
    and power envelopes, which return one column)."""
    x = (r / sk.r_cut).clamp(max=1.0)
    t, dt, _ = _taper(x, sk.r_cut)
    d = sk._anchor(r, species_i, species_j)                       # pylint: disable=protected-access
    if getattr(sk, "envelope", "exp") == "power":
        rc = r.clamp_min(1e-9)
        val = (d / rc) ** 2 * t
        der = d ** 2 * (dt / rc ** 2 - 2.0 * t / rc ** 3)
        return val.unsqueeze(-1).expand(-1, 4), der.unsqueeze(-1).expand(-1, 4)
    if getattr(sk, "decay_learned", False):
        L = sk.decay_lengths().to(r.dtype).unsqueeze(0)            # [1, 4]
        e = torch.exp(-(r - d).unsqueeze(-1) / L)
        return e * t.unsqueeze(-1), e * (dt.unsqueeze(-1) - t.unsqueeze(-1) / L)
    e = torch.exp(-(r - d) / sk.decay_length)
    val, der = e * t, e * (dt - t / sk.decay_length)
    return val.unsqueeze(-1).expand(-1, 4), der.unsqueeze(-1).expand(-1, 4)


def descriptor_radial_and_derivative(desc, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`GeometricDescriptor.radial` and its `d/dr`, `[E, N_r]`."""
    rc = r.clamp_min(1e-9)
    x = (r / desc.r_cut).clamp(max=1.0)
    t, dt, inside = _taper(x, desc.r_cut)
    n = torch.arange(1, desc.n_radial + 1, dtype=r.dtype, device=r.device).reshape(1, -1)
    arg = n * math.pi * x.reshape(-1, 1)
    s, c = torch.sin(arg), torch.cos(arg)
    bessel = s / rc.reshape(-1, 1)
    dbessel = (torch.where(inside.reshape(-1, 1), n * math.pi / desc.r_cut * c,
                           torch.zeros_like(c)) / rc.reshape(-1, 1)
               - s / rc.reshape(-1, 1) ** 2)
    return bessel * t.reshape(-1, 1), dbessel * t.reshape(-1, 1) + bessel * dt.reshape(-1, 1)


# ------------------------------------------------------------------ the SK block

def sk_block_cotangents(Xe: torch.Tensor, u: torch.Tensor, v: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(dS/dv [E, 4], dS/du [E, 3])` for `S = <Xe, sk_block(u, v)>`.

    `sk_block` is, with `x00 = Xe[0,0]`, `xs = Xe[0,1:]`, `xp = Xe[1:,0]`, `M = Xe[1:,1:]`:

        B[0,0] = ss,  B[0,1+k] = u_k sp,  B[1+k,0] = -u_k sp,
        B[1+k,1+l] = u_k u_l (pps - ppp) + delta_kl ppp
    """
    x00 = Xe[:, 0, 0]
    xs, xp = Xe[:, 0, 1:], Xe[:, 1:, 0]
    M = Xe[:, 1:, 1:]
    Mu = torch.einsum("ekl,el->ek", M, u)
    Mtu = torch.einsum("elk,el->ek", M, u)
    uMu = (Mu * u).sum(-1)
    trM = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    sp = v[:, 1]
    g_v = torch.stack([x00, ((xs - xp) * u).sum(-1), uMu, trM - uMu], dim=-1)
    g_u = sp.unsqueeze(-1) * (xs - xp) + (v[:, 2] - v[:, 3]).unsqueeze(-1) * (Mu + Mtu)
    return g_v, g_u


# ------------------------------------------------------------------ the head cotangent

def head_cotangent(fills: Sequence[object], g_occ_s, g_occ_r, sigma_s: float, tol: float
                   ) -> torch.Tensor:
    """`head_cotangent_eager` behind the lazy `torch.compile` (the four divided-difference
    maps are a dozen `[n, n]` elementwise kernels each; at 316 orbitals that is launch-bound
    the same way the assembly is)."""
    return _compiled(head_cotangent_eager)(
        fills[0].eps, fills[0].U, tuple(f.f for f in fills), tuple(f.mu for f in fills),
        g_occ_s, g_occ_r, sigma_s, tol)


def head_cotangent_eager(eps: torch.Tensor, U: torch.Tensor, f: Sequence[torch.Tensor],
                         mu: Sequence[torch.Tensor], g_occ_s: Optional[torch.Tensor],
                         g_occ_r: Optional[torch.Tensor], sigma_s: float, tol: float
                         ) -> torch.Tensor:
    """`X = U (D + M_s + M_r) U^T`: the band term's `dP` and the site-occupation cotangents'
    Daleckii-Krein pullbacks, formed together in the eigenbasis and rotated out ONCE.

    `f` and `mu` are `two_fillings`' four fills in order (S up, S down, ref up, ref down);
    `D = diag(sum_sigma (f_S,sigma - f_ref,sigma))` is `dP` in that basis. `g_occ_*` are the
    cotangents on the two site occupations `occ_S`, `occ_R` (`[..., N]`, or None), each
    expanded over its site's four orbitals exactly as `_SiteOccupation.backward` does.
    """
    df = (f[0] - f[2]) + (f[1] - f[3])
    K = torch.diag_embed(df)
    for g, (k_up, k_dn) in ((g_occ_s, (0, 1)), (g_occ_r, (2, 3))):
        if g is None:
            continue
        g_orb = g.repeat_interleave(ORBITALS_PER_ATOM, dim=-1)
        Ghat = U.transpose(-1, -2) @ (g_orb.unsqueeze(-1) * U)
        K = K + _dk_eigenbasis(eps, f[k_up], sigma_s, tol, Ghat, mu[k_up])
        K = K + _dk_eigenbasis(eps, f[k_dn], sigma_s, tol, Ghat, mu[k_dn])
    X = U @ K @ U.transpose(-1, -2)
    return 0.5 * (X + X.transpose(-1, -2))


# ------------------------------------------------------------------ the position gradient

# The kernel's arithmetic is ~2.7 ms of GPU time at 79 atoms, but eager it issues 542
# separate kernels and costs ~7.5 ms: it is launch-bound, not arithmetic-bound. Compiled
# (lazily, once, `dynamic=True` so that a changed edge count -- every relaxation step changes
# it -- reuses the graph instead of recompiling) it is ~2x that. `COMPILE = False` runs the
# eager reference; the two agree to 2e-14, and the gate holds them there.
COMPILE = os.environ.get("MACE_DSCC_NO_COMPILE", "") == ""
_COMPILED: dict = {}


def _compiled(fn):
    if not COMPILE:
        return fn
    got = _COMPILED.get(fn.__name__)
    if got is None:
        got = _COMPILED[fn.__name__] = torch.compile(fn, dynamic=True)
    return got


def assemble(h0, scalars, vectors, species, edge_index, edge_vector, batch, num_graphs, n_nodes):
    """`H0.batched` behind the same lazy `torch.compile`. The assembly is the other
    launch-bound half of the head: ~9000 edges x a 4x4 Slater-Koster block built from a
    dozen `stack`s, then one `index_put`. Inference only -- under gradients the graph has to
    exist for the parameter derivative."""
    return h0.batched(scalars, vectors, species, edge_index, edge_vector, batch, num_graphs, n_nodes)


def _site_blocks(h0, A0: torch.Tensor, vec: torch.Tensor, Qg: torch.Tensor,
                 species: torch.Tensor, edge_index: torch.Tensor,
                 edge_vector: torch.Tensor) -> torch.Tensor:
    """The `[N, 4, 4]` on-site blocks as a function of the descriptor alone, through the
    PRODUCTION readouts. Split out so that the cotangent on `(A0, vec, Q)` is one
    `autograd.grad` on this function -- and so that `torch.compile` owns its backward.
    Transcribing A1's environment factors and the block layout a second time is exactly what
    calling the production code here avoids."""
    levels = h0.sk.on_site(A0, species, None)                               # [N, 2]
    site = torch.diag_embed(torch.cat([levels[:, :1], levels[:, 1:].expand(-1, 3)], dim=-1))
    if h0.directional:
        site = site + h0.directional_site_blocks(A0, vec, species, edge_index, edge_vector, Qg)
    return site


def _edge_gradient(h0, Xe: torch.Tensor, u: torch.Tensor, r: torch.Tensor,
                   z_i: torch.Tensor, z_j: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                   gA0: torch.Tensor, gvec: torch.Tensor, gQ: torch.Tensor,
                   n_total: int) -> torch.Tensor:
    """The edge-scaling half in closed form: the Slater-Koster block and the descriptor's
    three channels, each split into a radial part (along `u`) and a tangential part (the
    projector `(I - u u^T)/r` from `du/dd`), then scattered to the edge's two ends."""
    dtype, device = Xe.dtype, Xe.device
    # --- Slater-Koster hopping
    v_rad, dv_rad = sk_radial_and_derivative(h0.sk, r, z_i, z_j)
    v0 = h0.sk.v0(z_i, z_j).to(dtype)
    g_v, g_u = sk_block_cotangents(Xe, u, v0 * v_rad)
    d_radial = (g_v * v0 * dv_rad).sum(-1)                                  # dS/dr, [E]
    d_tangent = g_u                                                         # dS/du, [E, 3]

    desc = h0.descriptor
    R, dR = descriptor_radial_and_derivative(desc, r)                       # [E, N_r]
    # --- l = 0, species resolved: A0[i, z' * N_r + n] = sum_{j in z'} R_n(r_ij)
    cA = gA0.reshape(n_total, desc.num_elements, desc.n_radial)[src, z_j]   # [E, N_r]
    d_radial = d_radial + (cA * dR).sum(-1)
    # --- l = 1: vectors[i, n, :] = sum_j c1 R_n(r) u
    c1 = desc.c1.to(dtype)[z_i, z_j]                                        # [E, N_r]
    W = gvec[src]                                                           # [E, N_r, 3]
    p = torch.einsum("enm,em->en", W, u)                                    # W_n . u
    d_radial = d_radial + (c1 * dR * p).sum(-1)
    d_tangent = d_tangent + torch.einsum("en,enm->em", c1 * R, W)
    # --- l = 2: Q[i] = sum_j (sum_n c2 R_n) (u u^T - I/3)
    c2 = desc.c2.to(dtype)[z_i, z_j]
    s2, ds2 = (c2 * R).sum(-1), (c2 * dR).sum(-1)
    Qc = gQ[src]                                                            # [E, 3, 3]
    Qu = torch.einsum("ekl,el->ek", Qc, u) + torch.einsum("elk,el->ek", Qc, u)
    trQ = Qc[:, 0, 0] + Qc[:, 1, 1] + Qc[:, 2, 2]
    d_radial = d_radial + ds2 * (0.5 * (Qu * u).sum(-1) - trQ / 3.0)
    d_tangent = d_tangent + s2.unsqueeze(-1) * Qu

    g_d = (d_radial.unsqueeze(-1) * u
           + (d_tangent - (d_tangent * u).sum(-1, keepdim=True) * u) / r.unsqueeze(-1))
    grad = torch.zeros(n_total, 3, dtype=dtype, device=device)
    return grad.index_add(0, dst, g_d).index_add(0, src, -g_d)


def assemble_compiled(h0, *args):
    return _compiled(assemble)(h0, *args)


def h0_position_gradient(h0, X: torch.Tensor, species: torch.Tensor,
                         edge_index: torch.Tensor, edge_vector: torch.Tensor,
                         local: torch.Tensor, batch: torch.Tensor, edge_graph: torch.Tensor,
                         n_nodes: int, num_graphs: int) -> torch.Tensor:
    """`d <X, H0(R)> / dR`, `[N_total, 3]`, at fixed `X` `[B, 4n, 4n]`.

    `local` is each node's index within its own graph, `batch` its graph, `edge_graph` each
    edge's graph -- the same three the batched assembly takes, for the same reason: the
    caller knows the batch is size-uniform and recomputing them here would hide it.
    """
    if not h0.decoupled:
        raise ValueError("the analytic head gradient is the decoupled (O3) head's: with the "
                         "base's features in H0 the derivative runs through the base's graph")
    device, dtype = X.device, X.dtype
    src, dst = edge_index[0], edge_index[1]
    edge_vector = edge_vector.to(dtype)
    r = edge_vector.norm(dim=-1).clamp_min(1e-9)
    u = edge_vector / r.unsqueeze(-1)
    n_total = int(species.shape[0])
    o = torch.arange(ORBITALS_PER_ATOM, device=device)

    def blocks_of(a: torch.Tensor, b: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        rows = (a * ORBITALS_PER_ATOM).reshape(-1, 1, 1) + o.reshape(1, -1, 1)
        cols = (b * ORBITALS_PER_ATOM).reshape(-1, 1, 1) + o.reshape(1, 1, -1)
        return X[graph.reshape(-1, 1, 1), rows, cols]

    Xe = blocks_of(local[src], local[dst], edge_graph)                     # [E, 4, 4]
    # The assembly REPLACES the matrix diagonal with the on-site levels, so a periodic
    # self-image edge (src == dst, present whenever r_cut > L/2) contributes its off-diagonal
    # entries and not its diagonal ones. Masked unconditionally: it costs one `where`.
    eye4 = torch.eye(ORBITALS_PER_ATOM, dtype=torch.bool, device=device)
    Xe = torch.where((src == dst).reshape(-1, 1, 1) & eye4, torch.zeros_like(Xe), Xe)

    # ---- per-node readouts: the production code on leaf descriptors, one small backward.
    # The descriptor runs under `no_grad`: `c1`/`c2` are parameters, so building its graph
    # here only to detach it cost 4 ms of the 8 the whole kernel took.
    with torch.no_grad():
        A0, vec, Qg = h0.descriptor(species, edge_index, edge_vector, n_total)
    with torch.enable_grad():
        A0, vec, Qg = (t.requires_grad_(True) for t in (A0, vec, Qg))
        site = _compiled(_site_blocks)(h0, A0, vec, Qg, species, edge_index, edge_vector)
        Xd = blocks_of(local, local, batch).detach()                        # [N, 4, 4]
        gA0, gvec, gQ = torch.autograd.grad((Xd * site).sum(), (A0, vec, Qg), allow_unused=True)
    fill_zero = lambda t, ref: torch.zeros_like(ref) if t is None else t     # noqa: E731
    gA0, gvec, gQ = fill_zero(gA0, A0), fill_zero(gvec, vec), fill_zero(gQ, Qg)

    with torch.no_grad():
        return _compiled(_edge_gradient)(h0, Xe, u, r, species[src], species[dst], src, dst,
                                         gA0, gvec, gQ, n_total)
