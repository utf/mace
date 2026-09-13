"""Edge-bucket padding: the prerequisite for CUDA-graph capture (coadvisor 2026-09-13).

At 79 atoms the model is launch-bound everywhere -- the base's forward+backward is 41.6 ms of
wall time of which 41.4 ms is CPU issue -- so the only lever that attacks the bottleneck is
removing launches, and that means capturing the work as CUDA graphs. Capture fixes both the
shapes and the addresses of every tensor in the captured region, while a neighbour list
changes length every step. This module makes the length constant.

THE PADDING IS EXACTLY ZERO, not approximately. A dummy edge is placed from node 0 to node 0
with a lattice shift far enough out that its length exceeds every cutoff in the model, and
every consumer of an edge multiplies by a factor that is identically zero there:

  * the base: `PolynomialCutoff.calculate_envelope` returns `envelope * (x < r_max)`, and
    e3nn's `FullyConnectedNet` (the radial weights) has no bias, so zero in is zero out and
    the message scattered from a dummy edge is exactly 0.0;
  * `H0`'s hoppings and the O3 descriptor: both carry H0's taper `(1 - (r/r_cut)^6)^2` with
    `r/r_cut` clamped at 1, which is exactly zero at and beyond `r_cut`;
  * `quadrupole_descriptor`: masked by `r < q_cut`.

The one consumer that is NOT automatically safe is O1's `pair_correction`, which deduplicates
the hopping MLP by unordered node pair: every dummy edge is the pair (0, 0), so it adds ONE
extra row to the unique set and cannot contaminate a real pair's input. Its correction is then
multiplied by the zero radial factor. `test_padding_leaves_every_output_unchanged` covers the
modulated head as well as O3's unmodulated one.

Bucketing rule (registered here): capacity is the count times `1 + HEADROOM`, rounded up to
`GRANULARITY`, and a bucket is only reduced when the count falls below `1 - HYSTERESIS` of it
-- so a relaxation that drifts by a few edges per step keeps one graph, and a trajectory that
genuinely changes size re-captures once rather than oscillating.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

GRANULARITY = 512
HEADROOM = 0.10
HYSTERESIS = 0.25


def edge_bucket(n_edges: int, current: Optional[int] = None) -> int:
    """The capacity to pad `n_edges` to, with hysteresis against a `current` bucket."""
    want = int(n_edges * (1.0 + HEADROOM))
    cap = max(GRANULARITY, -(-want // GRANULARITY) * GRANULARITY)
    if current is not None and current >= n_edges and cap <= current:
        # Only shrink once the count is well clear of the bucket; otherwise keep the graph.
        if n_edges > current * (1.0 - HYSTERESIS):
            return int(current)
    return int(cap)


def dummy_unit_shift(cell: torch.Tensor, r_cut: float) -> torch.Tensor:
    """A lattice translation `[3]` (integer, in `cell` units) whose image of node 0 is further
    than `r_cut` from node 0, for every cell shape. Uses the shortest lattice vector, so the
    same multiple is safe along all three axes."""
    lengths = cell.reshape(3, 3).norm(dim=-1)
    shortest = float(lengths.min())
    n = int(r_cut / max(shortest, 1e-9)) + 2
    return torch.tensor([n, n, n], dtype=torch.long, device=cell.device)


def pad_edges(edge_index: torch.Tensor, unit_shifts: torch.Tensor, capacity: int,
              dummy: torch.Tensor, keep: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad (or select-and-pad) an edge list to exactly `capacity` entries.

    `keep`, when given, marks the edges to retain -- the base's `r <= r_max` subset. The
    retained edges are moved to the front by a stable sort on the mask, which is a device-side
    operation: compacting with `nonzero` would copy a count to the host, and the point of
    capture is that the host does nothing. Returns `(edge_index, unit_shifts, live)`.
    """
    n = int(edge_index.shape[1])
    device = edge_index.device
    if keep is None:
        keep = torch.ones(n, dtype=torch.bool, device=device)
    order = torch.argsort((~keep).to(torch.int8), stable=True)      # kept first, order preserved
    if capacity <= n:
        sel = order[:capacity]
    else:
        pad = order.new_zeros(capacity - n)
        sel = torch.cat([order, pad])
    live = torch.zeros(capacity, dtype=torch.bool, device=device)
    live[:min(n, capacity)] = keep[sel[:min(n, capacity)]]
    ei = torch.where(live.unsqueeze(0), edge_index[:, sel], torch.zeros_like(edge_index[:, sel]))
    us = torch.where(live.unsqueeze(-1), unit_shifts[sel].to(dummy.dtype),
                     dummy.reshape(1, 3).expand(capacity, 3))
    return ei, us, live


def pad_batch(data: Dict[str, torch.Tensor], capacity: int, r_cut: float
              ) -> Dict[str, torch.Tensor]:
    """A batch whose head edge list is exactly `capacity` long, the extras inert.

    Single-graph batches only: a dummy edge belongs to node 0, and in a multi-graph batch that
    is graph 0's node, so the other graphs' edge counts would no longer be theirs. Capture is
    a single-frame path (one MD/relaxation step), so this is the case that matters; a batched
    trainer does not capture.
    """
    if int(data["ptr"].numel() - 1) != 1:
        raise ValueError("edge-bucket padding is for single-graph batches (capture is a "
                         "one-frame path); this batch has "
                         f"{int(data['ptr'].numel() - 1)} graphs")
    cell = data["cell"].view(3, 3)
    dummy = dummy_unit_shift(cell, r_cut)
    ei, us, live = pad_edges(data["edge_index"], data["unit_shifts"], capacity, dummy)
    out = dict(data)
    out["edge_index"] = ei
    out["unit_shifts"] = us.to(data["unit_shifts"].dtype)
    out["shifts"] = us.to(data["positions"].dtype) @ cell.to(data["positions"].dtype)
    out["edge_live"] = live
    return out


# --------------------------------------------------------------------------- capture

def _states_and_counts(model, data):
    """The per-graph electron counts and formal charge, on the host. Python, and it stays
    outside every graph: it reads the batch's integer metadata, which does not move."""
    from mace.modules.dscc.species import N0, State, neutral_count, states_from_batch
    num_graphs = int(data["ptr"].numel() - 1)
    states = states_from_batch(data["carrier_counts"].view(num_graphs, -1))
    species = data["node_attrs"].argmax(dim=-1)
    ptr = data["ptr"]
    n_up, n_dn, r_up, r_dn, n0_rows = [], [], [], [], []
    for g, state in enumerate(states):
        numbers = [model.atomic_numbers[int(x)] for x in species[int(ptr[g]):int(ptr[g + 1])].tolist()]
        n_ref = neutral_count(numbers)
        a, b = state.counts(n_ref)
        c, d = State(0, 0, 0).counts(n_ref)
        n_up.append(a); n_dn.append(b); r_up.append(c); r_dn.append(d)
        n0_rows.append([float(N0[z]) for z in numbers])
    dev = data["positions"].device
    t = lambda v: torch.tensor(v, dtype=torch.float64, device=dev)      # noqa: E731
    return states, (t(n_up), t(n_dn)), (t(r_up), t(r_dn)), t(n0_rows)


def head_tail(model, H, spectrum, fills_s, fills_r, counts_s, counts_r, n0, positions, cell,
              charges, species, edge_index, edge_vector, local, batch, edge_graph,
              n_nodes: int, num_graphs: int, plan, xi):
    """Everything the head does on the GPU once the spectrum is known: the site charges, the
    two SCF-free terms, the one cotangent `X`, and the analytic contraction.

    Written as a function of tensors alone -- no chemical-potential solve (it is a host Newton
    iteration; `fills_*` carry its result), no `nonzero` (the frontier's active set is a
    value-dependent shape, and the analytic route does not need it), no cell-dependent lattice
    sums (`plan` and `xi` are hoisted) -- so the same code runs eagerly and inside a CUDA
    graph. `test_capture` asserts the two agree.

    Returns `(head_energy, dq, head_position_gradient, e_host, e_m)`.
    """
    from mace.modules.dscc.analytic import h0_position_gradient, head_cotangent_eager
    from mace.modules.dscc.fill import DEGENERACY_TOL
    from mace.modules.dscc.scf import ORBITALS
    eps, U = spectrum
    (mu_s_up, mu_s_dn, f_s_up, f_s_dn) = fills_s
    (mu_r_up, mu_r_dn, f_r_up, f_r_dn) = fills_r
    df = (f_s_up - f_r_up) + (f_s_dn - f_r_dn)
    # The band term over ALL levels: with equal counts in a spin channel its contribution is
    # identically zero, so the branch `two_fillings` takes to skip it is an optimisation and
    # not a different formula.
    from mace.modules.dscc.fill import generalised_entropy
    energy = ((df * eps).sum(-1)
              + generalised_entropy(eps, mu_s_up, model.sigma_s) - generalised_entropy(eps, mu_r_up, model.sigma_s)
              + generalised_entropy(eps, mu_s_dn, model.sigma_s) - generalised_entropy(eps, mu_r_dn, model.sigma_s))
    n_orb = eps.shape[-1]
    w_all = (U * U).reshape(*eps.shape[:-1], n_orb // ORBITALS, ORBITALS, n_orb).sum(-2)
    dq = -torch.einsum("...ia,...a->...i", w_all, df)
    e_host, e_m, pair_grad, _, _, occ_cot = model.scf_free_terms(
        H, spectrum, counts_s, counts_r, n0, positions, cell, charges, True,
        need_cotangents=True, plan=plan, xi=xi, fills_s=fills_s, fills_r=fills_r)
    g_s, g_r = occ_cot
    X = head_cotangent_eager(eps, U, (f_s_up, f_s_dn, f_r_up, f_r_dn),
                             (mu_s_up, mu_s_dn, mu_r_up, mu_r_dn), g_s, g_r,
                             model.sigma_s, DEGENERACY_TOL)
    grad = h0_position_gradient(model.h0, X, species, edge_index, edge_vector,
                                local, batch, edge_graph, n_nodes, num_graphs)
    return energy + e_host + e_m, dq, grad + pair_grad.reshape(-1, 3), e_host, e_m


class GraphedStep:
    """One frame shape of the decoupled model, as two CUDA graphs around a CPU segment.

        graph A0 : H0 assembly                         (~1 ms of GPU, issued first)
        D2H      : H to pinned host memory
        graph A1 : base forward + backward             (~30 ms of GPU, ~0 ms of CPU)
        CPU      : eigh, chemical potentials, occupations, entropies -- under A1's window
        graph B  : site charges, E_host, E_M, X, the analytic contraction, accumulation

    WHY THIS ORDER. The coadvisor's sketch puts the assembly in graph A with the base; putting
    it FIRST is what lets the host copy `H` back and run the eigendecomposition while the
    base's 30 ms of GPU work is still going. Recorded after the base, `H` would not be ready
    until that window closed and the CPU would idle through it and then work alone.

    Measured before any of this existed: the base's forward+backward at 79 atoms is 41.6 ms of
    wall time of which 41.4 ms is CPU kernel-issue, and 30.6 ms of GPU. Nothing here is
    compute; it is all launches.

    The graphs share one memory pool and are replayed in capture order, which is the
    documented requirement for sharing one. Every input lives in a static buffer that
    `__call__` overwrites; `edge_index` VALUES may change from step to step (they are read at
    replay), only the LENGTH is fixed, which is what the bucket is for.
    """

    def __init__(self, model, data: Dict[str, torch.Tensor], head_capacity: Optional[int] = None,
                 base_capacity: Optional[int] = None, warmup: int = 3) -> None:
        import math as _math
        from mace.modules.dscc.ewald import madelung_self, reciprocal_plan
        if getattr(model.h0.sk, "screen", None) is not None:   # the module, not the flag
            raise ValueError(
                "capture is not available for the screened head (O3.2): its triplet list is "
                "built with `nonzero`, so the shapes are data dependent and a replay would "
                "silently reuse the captured frame's triplets. Registered as a follow-up "
                "alongside the analytic three-body force; neither is on the training path")
        if not (model.decoupled and model.scf_free and not model.coupling):
            raise ValueError("capture is for the decoupled SCF-free head (O3): the coupled "
                             "head solves, and a head that reads the base cannot be split "
                             "from it by an event")
        if int(data["ptr"].numel() - 1) != 1:
            raise ValueError("capture is a one-frame path")
        self.model = model
        dev = data["positions"].device
        n_edges = int(data["edge_index"].shape[1])
        self.head_capacity = int(head_capacity or edge_bucket(n_edges))
        padded = pad_batch(dict(data), self.head_capacity, model.r_cut)

        # The base's subset, sized once with the same rule.
        send, recv = padded["edge_index"][0], padded["edge_index"][1]
        vec = padded["positions"][recv] - padded["positions"][send] + padded["shifts"]
        n_kept = int((vec.norm(dim=-1) <= model.r_max).sum())
        self.base_capacity = int(base_capacity or edge_bucket(n_kept))

        # Static inputs. `node_attrs`, `ptr`, `batch` and the carrier counts are the size
        # class's and do not move; positions, cell and the edge list are refreshed per step.
        # `.detach()` FIRST: the batch handed in has usually been through an eager forward,
        # which calls `positions.requires_grad_(True)` in place, so a bare `.clone()` is a
        # non-leaf carrying that forward's graph -- and capture refuses a graph whose nodes
        # were recorded on another stream.
        self.static = {k: padded[k].detach().clone() for k in
                       ("positions", "cell", "edge_index", "unit_shifts", "shifts",
                        "node_attrs", "batch", "ptr", "carrier_counts")}
        self.static["positions"].requires_grad_(True)
        self.n_nodes = int(self.static["node_attrs"].shape[0])
        self.species = self.static["node_attrs"].argmax(dim=-1)
        self.local = torch.arange(self.n_nodes, device=dev) - self.static["batch"] * self.n_nodes
        self.n_kept = torch.zeros((), dtype=torch.long, device=dev)

        states, self.counts_s, self.counts_r, self.n0 = _states_and_counts(model, padded)
        self.charges = torch.tensor([float(s.Q) for s in states], dtype=torch.float64, device=dev)
        cell0 = self.static["cell"].view(-1, 3, 3)[0]
        width = _math.sqrt(2.0 * (model.kernel.r_g ** 2 + model.r_split ** 2))
        self.plan = reciprocal_plan(cell0, width, tol=model.kernel.tol)
        # Hoisted: deriving it inside the captured region reads the cell to the host.
        self.dummy = dummy_unit_shift(cell0, float(model.r_max))
        self.xi = torch.stack([madelung_self(cell0)])
        self.sigma_s = float(model.sigma_s)

        self._build(warmup)

    # ------------------------------------------------------------------ capture

    def _edges(self):
        d = dict(self.static)
        cell = d["cell"].view(-1, 3, 3)
        d["shifts"] = torch.einsum("ei,eij->ej", d["unit_shifts"].to(d["positions"].dtype),
                                   cell[d["batch"][d["edge_index"][0]]])
        return d

    def _assembly(self):
        d = self._edges()
        send, recv = d["edge_index"][0], d["edge_index"][1]
        ev = d["positions"][recv] - d["positions"][send] + d["shifts"]
        with torch.no_grad():
            H = self.model.h0.batched(None, None, self.species, d["edge_index"], ev,
                                      d["batch"], 1, self.n_nodes)
        return H, ev, d

    def _base(self, d):
        trunk = self.model._trunk_data(d, capacity=self.base_capacity, n_kept_out=self.n_kept,
                                       dummy=self.dummy)
        out = self.model.base_forward(trunk, training=False, compute_force=True)
        return out["energy"], out["forces"]

    def _tail(self, H, ev, d):
        with torch.no_grad():
            return head_tail(self.model, H, (self.eps_d, self.U_d), self.fills_s_d, self.fills_r_d,
                             self.counts_s, self.counts_r, self.n0,
                             d["positions"].detach().reshape(1, self.n_nodes, 3),
                             d["cell"].view(-1, 3, 3), self.charges, self.species,
                             d["edge_index"], ev.detach(), self.local, d["batch"],
                             d["batch"][d["edge_index"][0]], self.n_nodes, 1, self.plan, self.xi)

    def _build(self, warmup: int) -> None:
        dev = self.static["positions"].device
        n_orb = self.n_nodes * 4
        self.eps_d = torch.zeros(1, n_orb, dtype=torch.float64, device=dev)
        self.U_d = torch.zeros(1, n_orb, n_orb, dtype=torch.float64, device=dev)
        mk = lambda: tuple(torch.zeros(1, n_orb if i >= 2 else 1, dtype=torch.float64, device=dev).squeeze(-1)
                           if i < 2 else torch.zeros(1, n_orb, dtype=torch.float64, device=dev)
                           for i in range(4))                                    # noqa: E731
        self.fills_s_d, self.fills_r_d = mk(), mk()
        self.H_host = torch.zeros(1, n_orb, n_orb, dtype=torch.float64, device="cpu").pin_memory()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                H, ev, d = self._assembly()
                e_b, f_b = self._base(d)
                self._tail(H, ev, d)
                # The loop's bindings OUTLIVE the loop in Python, and with them the warm-up's
                # autograd graph -- whose nodes hold a reference to the warm-up stream. Capture
                # then refuses ("stale reference to the default stream"). Dropped explicitly.
                del H, ev, d, e_b, f_b
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self.pool = torch.cuda.graph_pool_handle()
        self.g_assembly = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_assembly, pool=self.pool):
            self.H, self.ev, self.d = self._assembly()
        self.g_base = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_base, pool=self.pool):
            self.e_base, self.f_base = self._base(self.d)
        self.g_tail = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_tail, pool=self.pool):
            self.head_energy, self.dq, self.head_grad, self.e_host, self.e_m = \
                self._tail(self.H, self.ev, self.d)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ one step

    def fits(self, data: Dict[str, torch.Tensor]) -> bool:
        """Whether this capture is valid for the frame.

        The edge count must fit the bucket, and -- less obviously -- the CELL, the composition
        and the carrier counts must be the ones captured. The reciprocal plan, the Madelung
        constant `xi` and the padding translation are all constants OF THE CELL, held on the
        object; the formal charge enters `E_M` directly. A frame with a different cell replayed
        through this capture gets the captured cell's electrostatics and the new frame's
        geometry, which is wrong by millielectronvolts and raises nothing -- measured at 5.5 meV
        and 9.3 meV/A on the 79-atom training frames, whose cells span 15.76-16.07 A because
        they are NPT snapshots. Capture is therefore a FIXED-CELL path: relaxation at fixed
        cell, NVT/NVE dynamics. NPT re-captures per cell, or does not capture.

        The comparisons read a 3x3 and two short integer tensors back to the host -- about
        50 us, against the alternative of a silently wrong answer."""
        if int(data["edge_index"].shape[1]) > self.head_capacity:
            return False
        for key in ("cell", "carrier_counts", "node_attrs", "ptr"):
            got = data.get(key)
            if got is None or not torch.equal(self.static[key], got.reshape(self.static[key].shape)):
                return False
        return True

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from mace.modules.dscc.fill import chemical_potential, occupations
        if not self.fits(data):
            raise RuntimeError(
                f"this frame is outside the capture: {int(data['edge_index'].shape[1])} edges "
                f"against a bucket of {self.head_capacity}, or a different cell, composition or "
                "charge state (all three are baked into the recorded graphs); re-capture")
        padded = pad_batch(dict(data), self.head_capacity, self.model.r_cut)
        with torch.no_grad():
            for k in ("positions", "cell", "edge_index", "unit_shifts", "shifts"):
                self.static[k].copy_(padded[k])
        self.g_assembly.replay()
        self.H_host.copy_(self.H)                       # waits on the assembly alone
        self.g_base.replay()                            # ~0.02 ms of CPU; 30 ms of GPU follows
        # --- the CPU segment, under the base's window
        eps_c, U_c = torch.linalg.eigh(self.H_host)
        def solve(counts):
            cc = torch.stack([counts[0].cpu().expand(eps_c.shape[:-1]),
                              counts[1].cpu().expand(eps_c.shape[:-1])])
            mus = chemical_potential(eps_c.unsqueeze(0).expand(2, *eps_c.shape), cc, self.sigma_s)
            return (mus[0], mus[1], occupations(eps_c, mus[0], self.sigma_s),
                    occupations(eps_c, mus[1], self.sigma_s))
        fs, fr = solve(self.counts_s), self.counts_r and solve(self.counts_r)
        with torch.no_grad():
            self.eps_d.copy_(eps_c, non_blocking=True)
            self.U_d.copy_(U_c, non_blocking=True)
            for dst, src in zip(self.fills_s_d + self.fills_r_d, fs + fr):
                dst.copy_(src, non_blocking=True)
        self.g_tail.replay()
        # The base's bucket is the one the host cannot check in advance (the kept subset is a
        # property of the positions), so it is checked HERE -- after the step, where the sync
        # is free, and loudly: `pad_edges` truncates in silence and a base missing a few edges
        # is wrong by millielectronvolts with nothing to show for it.
        if int(self.n_kept) > self.base_capacity:
            raise RuntimeError(f"the base's edge bucket overflowed ({int(self.n_kept)} live "
                               f"edges against {self.base_capacity}); this step's result is "
                               "invalid -- re-capture with a larger bucket and repeat it")
        c_q = self.model.c_q(int(round(float(self.charges[0]))))
        uncal = self.e_base.to(torch.float64) + self.head_energy
        return {"energy_uncalibrated": uncal, "energy": uncal + c_q.to(uncal.device),
                "base_energy": self.e_base, "head_energy": self.head_energy,
                "forces": self.f_base.to(torch.float64) - self.head_grad,
                "dq": self.dq.reshape(-1), "e_host": self.e_host, "e_madelung": self.e_m,
                "base_edges_live": self.n_kept, "short_circuit": False}
