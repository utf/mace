"""Plan section 0 / 2: `MACEDSCC` -- the D-SCC head on a frozen neutral MACE base.

    E(R, S) = E_base(R) + J*(R, S) + C_Q          (C_0 = 0)

* `E_base`, its forces and (through the strain below) its stress come from the parent-class
  forward of the base (`ScaleShiftMACE.forward`, D6) on the `<= r_max` subset of the head's
  graph; every base parameter is frozen. The base's first-block features reach the head
  attached to positions and cell.
* At `S = S_ref` the forward never enters the head (D5): the base's own outputs are returned
  bit-identically, forces and stress computed by the base's own path.
* `J*` is the band-form value at the solution (envelope: `P` and `dq` enter detached where
  the value is stationary in them, so autograd returns the exact parameter gradient); the
  head forces are the Hellmann-Feynman contraction `-Tr(dP dH0/dR) - 0.5 dq^T (dGamma/dR)
  dq` with the density and charges as constant cotangents whose own graph is kept for the
  training gradient (plan section 2.7); the stress is the same contraction against a
  symmetric strain applied to positions and cell before the base is evaluated.
* `C_Q` is one constant per formal charge, held with its calibration record; a model
  without a record reports uncalibrated energies and says so.

Phase 0 wires the `Phi = 0` path (two fillings of `H0`); the self-consistent loop of
section 2.6 plugs into `solve` in Phase 1.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from e3nn import o3
from torch import nn

from mace.modules.models import ScaleShiftMACE
from mace.modules.dscc.fill import SIGMA_S
from mace.modules.dscc.hamiltonian import (A_MAX_DEFAULT, B_MAX_DEFAULT, H0, Q_CUT_DEFAULT,
                                           R_CUT_DEFAULT)
from mace.modules.dscc.kernels import (KernelConfig, gamma_lr, gamma_matrix, host_potential,
                                       kernel_components)
from mace.modules.dscc.scf import (ScfOptions, ScfResult, continuation_solve, solve_dscc,
                                   two_fillings)
from mace.modules.dscc.species import (N0, S_REF, State, U_MAX_GFN1, neutral_count,
                                       states_from_batch)
from mace.modules.dscc.fill import fill

# Registered defaults for the bounded learnables (plan section 2.4; to confirm before use).
LAMBDA_0_DEFAULT = 0.05
LAMBDA_MAX_DEFAULT = 2.0
U_INIT_FRACTION = 0.05          # U_eff starts at 5 % of its bound ("initialised small")
R_SPLIT_DEFAULT = 2.5           # Route B', A (> r_g)
S_MAX_DEFAULT = 2.0             # Route B': the single global scale s in [0, s_max], init 1 (v4.2)


def block0_layout(base: nn.Module) -> Tuple[int, int, int]:
    """`(n_scalars, n_vectors, width)` of the base's first block: `mul x 0e` then
    `mul x 1o`, in that order (e3nn lays a `mul x ir` block out as `[mul, 2l+1]`)."""
    irreps = o3.Irreps(str(base.products[0].linear.irreps_out))
    n_s = n_v = 0
    seen_vector = False
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1 and not seen_vector:
            n_s += mul
        elif ir.l == 1 and ir.p == -1:
            n_v += mul
            seen_vector = True
        else:
            raise ValueError(f"unsupported first-block irreps {irreps}: expected scalars "
                             "followed by polar vectors")
    return n_s, n_v, irreps.dim


class MACEDSCC(nn.Module):
    """The D-SCC head on a frozen base. Registered numbers travel in `extra_state`."""

    def __init__(self, base: nn.Module, *, r_cut: float = R_CUT_DEFAULT,
                 kernel: Optional[KernelConfig] = None, sigma_s: float = SIGMA_S,
                 coupling: bool = False, route_b: bool = False, directional: bool = True,
                 lambda_0: float = LAMBDA_0_DEFAULT, lambda_max: float = LAMBDA_MAX_DEFAULT,
                 u_max: Optional[Dict[int, float]] = None, r_split: float = R_SPLIT_DEFAULT,
                 q_cut: float = Q_CUT_DEFAULT, a_max: float = A_MAX_DEFAULT,
                 b_max: float = B_MAX_DEFAULT, hidden: int = 64,
                 scf: Optional[ScfOptions] = None) -> None:
        super().__init__()
        self.scf_options = scf or ScfOptions()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.atomic_numbers = [int(z) for z in base.atomic_numbers]
        self.r_max = float(base.r_max)
        self.r_cut = float(r_cut)
        if self.r_cut < self.r_max:
            raise ValueError(f"the head's r_cut ({r_cut}) must reach at least the base's "
                             f"r_max ({self.r_max}): the trunk is fed the head graph's subset")
        self.kernel = kernel or KernelConfig()
        self.sigma_s = float(sigma_s)
        self.coupling = bool(coupling)       # Phi_cc on (self-consistent) or Phi = 0
        self.route_b = bool(route_b)
        self.r_split = float(r_split)
        self.lambda_max = float(lambda_max)
        n_s, n_v, self.block0_width = block0_layout(base)
        self.n_scalars, self.n_vectors = n_s, n_v
        self.h0 = H0(self.atomic_numbers, feature_dim=n_s, n_vectors=n_v, r_cut=self.r_cut,
                     q_cut=q_cut, directional=directional, a_max=a_max, b_max=b_max,
                     hidden=hidden)
        n_el = len(self.atomic_numbers)
        u_max = dict(u_max or U_MAX_GFN1)
        self.register_buffer("u_max", torch.tensor([float(u_max[z]) for z in self.atomic_numbers],
                                                   dtype=torch.float64))
        logit = lambda x: float(torch.logit(torch.tensor(x, dtype=torch.float64)))  # noqa: E731
        self.lambda_raw = nn.Parameter(torch.tensor(logit(lambda_0 / lambda_max), dtype=torch.float64))
        self.u_raw = nn.Parameter(torch.full((n_el,), logit(U_INIT_FRACTION), dtype=torch.float64))
        # Route B' (v4.2): the static pattern is the reference-fill site charge q0(R) of
        # H0(R) -- locally neutral by construction -- times ONE global scale s in
        # [0, s_max], initialised at 1. Per-species scales are prohibited (they break local
        # neutrality at the vacancy and bring the 1/L error back).
        self.s_max = float(S_MAX_DEFAULT)
        self.s_raw = nn.Parameter(torch.zeros((), dtype=torch.float64))       # s_max sigmoid(0) = 1
        self.register_buffer("q0_pristine", torch.zeros(n_el, dtype=torch.float64))   # species means
        self.register_buffer("pristine_atoms", torch.tensor(0, dtype=torch.long))
        # C_Q: one constant per formal charge, with its record (plan section 6); both in
        # `extra_state`, so a checkpoint carries exactly what it was calibrated with.
        self.c_q_table: Dict[int, float] = {}
        self.calibration_record: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------------- state

    def get_extra_state(self) -> Dict[str, Any]:
        return {"kernel": asdict(self.kernel), "scf": asdict(self.scf_options),
                "r_cut": self.r_cut, "sigma_s": self.sigma_s,
                "coupling": self.coupling, "route_b": self.route_b, "r_split": self.r_split,
                "lambda_max": self.lambda_max, "c_q_table": dict(self.c_q_table),
                "calibration_record": self.calibration_record}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.kernel = KernelConfig(**state["kernel"])
        self.scf_options = ScfOptions(**state["scf"])
        self.r_cut, self.sigma_s = float(state["r_cut"]), float(state["sigma_s"])
        self.coupling, self.route_b = bool(state["coupling"]), bool(state["route_b"])
        self.r_split, self.lambda_max = float(state["r_split"]), float(state["lambda_max"])
        self.c_q_table = {int(k): float(v) for k, v in state.get("c_q_table", {}).items()}
        self.calibration_record = state.get("calibration_record")

    # ----------------------------------------------------------------- learnables

    def lambda_dir(self) -> torch.Tensor:
        return self.lambda_max * torch.sigmoid(self.lambda_raw)

    def u_eff(self) -> torch.Tensor:
        """`U_eff[Z]` in `[0, U_max[Z]]`."""
        return self.u_max * torch.sigmoid(self.u_raw)

    def c_q(self, q: int) -> torch.Tensor:
        """`C_Q`; zero for `Q = 0` and for an uncalibrated model."""
        device = self.u_max.device
        if q == 0 or not self.c_q_table:
            return torch.zeros((), dtype=torch.float64, device=device)
        if int(q) not in self.c_q_table:
            raise KeyError(f"no C_Q calibrated for Q={q}")
        return torch.tensor(self.c_q_table[int(q)], dtype=torch.float64, device=device)

    @property
    def calibrated(self) -> bool:
        return bool(self.c_q_table) and self.calibration_record is not None

    def set_calibration(self, values: Dict[int, float], record: Dict[str, Any]) -> None:
        self.c_q_table = {int(q): float(v) for q, v in values.items() if int(q) != 0}
        self.calibration_record = dict(record)

    def pattern_scale(self) -> torch.Tensor:
        """Route B': the single global scale `s = s_max sigmoid(s_raw)`, 1 at initialisation."""
        return self.s_max * torch.sigmoid(self.s_raw)

    def reference_charges(self, H: torch.Tensor, numbers: Sequence[int]) -> torch.Tensor:
        """Route B' (v4.2): `q0_i = n0[Z_i] - sum_sigma Tr(Pi_i P_ref_sigma(H0))` at `H = H0`
        -- NEVER at `H0 - V` -- attached to `H0` through the divided-difference backward, so
        its geometry derivative (the mandatory `s dq^T Gamma_LR dq0/dR` force term, one
        Frechet contraction) and its parameter derivative are on the graph. Sums to zero
        exactly: the reference is neutral."""
        n_up, n_dn = S_REF.counts(neutral_count(numbers))
        with torch.no_grad():
            spectrum = torch.linalg.eigh(H)
        P = fill(H, float(n_up), self.sigma_s, spectrum).P + fill(H, float(n_dn), self.sigma_s, spectrum).P
        occupied = torch.diagonal(P).reshape(-1, 4).sum(-1)
        n0 = torch.tensor([float(N0[z]) for z in numbers], dtype=H.dtype, device=H.device)
        return n0 - occupied

    def compensation_cloud(self, q0: torch.Tensor, species: torch.Tensor, positions: torch.Tensor,
                           cell: torch.Tensor) -> Dict[str, float]:
        """v4.2 diagnostic: the extent `R_eff` of `q0 - q0_pristine` (the reference fill's
        compensation of the missing ion): radius of gyration of |dq0| about its
        minimum-image centroid, and the total |dq0|."""
        dev = (q0 - self.q0_pristine[species]).detach()
        w = dev.abs()
        if float(w.sum()) < 1e-12:
            return {"r_eff": 0.0, "total_abs": 0.0}
        anchor = positions[int(w.argmax())]
        d = positions.detach() - anchor
        d = d - torch.round(d @ torch.linalg.inv(cell.detach())) @ cell.detach()
        centroid = (w.unsqueeze(-1) * d).sum(0) / w.sum()
        r2 = ((d - centroid) ** 2).sum(-1)
        return {"r_eff": float(torch.sqrt((w * r2).sum() / w.sum())), "total_abs": float(w.sum())}

    # ----------------------------------------------------------------- base

    def _trunk_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The base's own graph: the head graph's edges no longer than `r_max`."""
        sender, receiver = data["edge_index"][0], data["edge_index"][1]
        vectors = (data["positions"][receiver] - data["positions"][sender] + data["shifts"])
        keep = vectors.detach().norm(dim=-1) <= self.r_max
        trunk = dict(data)
        trunk["edge_index"] = data["edge_index"][:, keep]
        trunk["shifts"] = data["shifts"][keep]
        trunk["unit_shifts"] = data["unit_shifts"][keep]
        return trunk

    def first_block(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The base's block-0 node features alone (embedding + first interaction +
        product), attached to positions and cell -- the plan's "recomputation of the
        base's early feature block for derivatives". Used with cached `E_base`/`F_base`
        so the second interaction and the readouts are not evaluated in training."""
        from mace.modules.utils import prepare_graph
        base = self.base
        ctx = prepare_graph(data, compute_virials=False, compute_stress=False,
                            compute_displacement=False, lammps_mliap=False)
        node_feats = base.node_embedding(data["node_attrs"])
        edge_attrs = base.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = base.radial_embedding(ctx.lengths, data["node_attrs"], data["edge_index"],
                                                   base.atomic_numbers)
        ikw = ctx.interaction_kwargs
        node_feats, sc = base.interactions[0](node_attrs=data["node_attrs"], node_feats=node_feats,
                                              edge_attrs=edge_attrs, edge_feats=edge_feats,
                                              edge_index=data["edge_index"], cutoff=cutoff, first_layer=True,
                                              lammps_class=ikw.lammps_class, lammps_natoms=ikw.lammps_natoms)
        return base.products[0](node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])

    def features(self, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block-0 scalars `[N, n_s]` and polar vectors `[N, n_v, 3]`."""
        n_s, n_v = self.n_scalars, self.n_vectors
        scalars = node_feats[:, :n_s]
        vectors = node_feats[:, n_s:n_s + 3 * n_v].reshape(-1, n_v, 3)
        return scalars, vectors

    @torch.no_grad()
    def set_pristine_centre(self, batches: Sequence[Dict[str, torch.Tensor]]) -> int:
        """Plan 2.1: the per-species first-block feature means over the pristine cell(s);
        also the pristine composition (sum rule) and atom count. Returns atoms averaged."""
        feats, species, sizes = [], [], []
        for data in batches:
            out = ScaleShiftMACE.forward(self.base, self._trunk_data(dict(data)), training=False,
                                         compute_force=False)
            scalars, _ = self.features(out["node_feats"])
            feats.append(scalars.detach())
            species.append(data["node_attrs"].argmax(dim=-1))
            ptr = data["ptr"]
            sizes.extend(int(x) for x in (ptr[1:] - ptr[:-1]).tolist())
        if not feats:
            raise ValueError("no pristine frames")
        scalars, species_t = torch.cat(feats), torch.cat(species)
        self.h0.set_centre(scalars, species_t)
        self.pristine_atoms.fill_(min(sizes))
        # Route B' diagnostic reference: the pristine q0 per species (with the centre set).
        q0_all, sp_all = [], []
        for data in batches:
            out = ScaleShiftMACE.forward(self.base, self._trunk_data(dict(data)), training=False,
                                         compute_force=False)
            sc_, vec_ = self.features(out["node_feats"])
            sp_ = data["node_attrs"].argmax(dim=-1)
            positions, cell = data["positions"], data["cell"].view(-1, 3, 3)
            sender, receiver = data["edge_index"][0], data["edge_index"][1]
            edge_graph = data["batch"][sender]
            shifts = torch.einsum("ei,eij->ej", data["unit_shifts"].to(positions.dtype), cell[edge_graph])
            ev = positions[receiver] - positions[sender] + shifts
            ptr = data["ptr"]
            for g in range(int(ptr.numel() - 1)):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                e_mask = edge_graph == g
                H = self.h0(sc_[lo:hi], vec_[lo:hi], sp_[lo:hi], data["edge_index"][:, e_mask] - lo, ev[e_mask])
                numbers = [self.atomic_numbers[int(x)] for x in sp_[lo:hi].tolist()]
                q0_all.append(self.reference_charges(H, numbers).detach())
                sp_all.append(sp_[lo:hi])
        q0, sp = torch.cat(q0_all), torch.cat(sp_all)
        for s_ in range(len(self.atomic_numbers)):
            if bool((sp == s_).any()):
                self.q0_pristine[s_] = q0[sp == s_].mean()
        return int(scalars.shape[0])

    # ----------------------------------------------------------------- gap (plan 6)

    def pristine_gap(self, data: Dict[str, torch.Tensor],
                     cached_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """`E_gap_model = LUMO - HOMO` at the exact valence count of the Hamiltonian
        actually filled at `dq = 0` -- `H0` in Route A, `H0 - W` in Route B -- for each
        graph of a pristine batch (the static cell for the regulariser, C2 ruled
        static-lattice; thermal frames for the ensemble-mean diagnostic). Differentiable
        in the head parameters through the eigenvalues (not eigenvectors)."""
        num_graphs = int(data["ptr"].numel() - 1)
        if cached_features is None:
            scalars, vectors = self.features(self.first_block(self._trunk_data(dict(data))))
        else:
            # A fixed cell (the static pristine cell): its geometry-only base features are
            # cached by the caller; the gap's parameter gradient flows through H0 alone.
            scalars, vectors = cached_features
        species = data["node_attrs"].argmax(dim=-1)
        positions, cell = data["positions"], data["cell"].view(-1, 3, 3)
        sender, receiver = data["edge_index"][0], data["edge_index"][1]
        edge_graph = data["batch"][sender]
        shifts = torch.einsum("ei,eij->ej", data["unit_shifts"].to(positions.dtype), cell[edge_graph])
        edge_vector = positions[receiver] - positions[sender] + shifts
        ptr = data["ptr"]
        gaps = []
        for g in range(num_graphs):
            lo, hi = int(ptr[g]), int(ptr[g + 1])
            nodes = slice(lo, hi)
            e_mask = edge_graph == g
            H = self.h0(scalars[nodes], vectors[nodes], species[nodes],
                        data["edge_index"][:, e_mask] - lo, edge_vector[e_mask])
            sp_g = species[nodes]
            numbers = [self.atomic_numbers[int(s_)] for s_ in sp_g.tolist()]
            if self.route_b:
                # v4.2: the gap regulariser acts on H0 - W with W from the pristine q0.
                g_lr = gamma_lr(positions[nodes], cell[g], self.kernel.r_g, self.r_split,
                                self.kernel.eps_inf, tol=self.kernel.tol)
                W = host_potential(g_lr, self.pattern_scale() * self.reference_charges(H, numbers))
                H = H - torch.diag(W.repeat_interleave(4))
            n_up, n_dn = S_REF.counts(neutral_count(numbers))
            eps = torch.linalg.eigvalsh(H)
            # Spin-independent H: the gap at the majority count (the larger fill).
            gaps.append(eps[n_up] - eps[n_up - 1])
        return torch.stack(gaps)

    # ----------------------------------------------------------------- forward

    def forward(self, data: Dict[str, torch.Tensor], training: bool = False,
                compute_force: bool = True, compute_stress: bool = False,
                warm_start: Optional[Sequence[Optional[torch.Tensor]]] = None,
                **_: Any) -> Dict[str, Optional[torch.Tensor]]:
        """`warm_start`: per-graph `dq` to start the solve from (root-rule initialisation
        iii, along a trajectory; also what a derivative check on a multi-branch landscape
        needs to stay on one branch). Production uses the continuation (D11)."""
        num_graphs = int(data["ptr"].numel() - 1)
        states = states_from_batch(data["carrier_counts"].view(num_graphs, -1))
        # Forces and stress are derivatives: they need autograd even inside a no_grad
        # evaluation (as the base's own forward does).
        with torch.set_grad_enabled(torch.is_grad_enabled() or compute_force or compute_stress):
            return self._dispatch(data, states, training, compute_force, compute_stress, warm_start)

    def _dispatch(self, data, states, training, compute_force, compute_stress, warm_start):
        num_graphs = len(states)
        if all(s.is_reference for s in states):
            # D5: never enter the head at S_ref -- the base's outputs, bit-identically.
            out = ScaleShiftMACE.forward(self.base, self._trunk_data(dict(data)), training=training,
                                         compute_force=compute_force, compute_stress=compute_stress)
            out["head_energy"] = torch.zeros(num_graphs, dtype=torch.float64, device=out["energy"].device)
            out["energy_uncalibrated"] = out["energy"]
            out["short_circuit"] = True
            return out
        return self._charged_forward(data, states, training, compute_force, compute_stress, warm_start)

    def _charged_forward(self, data, states: Sequence[State], training: bool,
                         compute_force: bool, compute_stress: bool,
                         warm_start: Optional[Sequence[Optional[torch.Tensor]]] = None):
        num_graphs = len(states)
        device = data["positions"].device
        batch = data["batch"]
        positions = data["positions"]
        cell = data["cell"].view(-1, 3, 3)
        strain = None
        if compute_stress:
            # A symmetric strain on positions and cell, applied BEFORE the base sees the
            # frame, so base and head stresses are one derivative of one geometry.
            strain = torch.zeros(num_graphs, 3, 3, dtype=positions.dtype, device=device,
                                 requires_grad=True)
            sym = 0.5 * (strain + strain.transpose(-1, -2))
            positions = positions + torch.einsum("ni,nij->nj", positions, sym[batch])
            cell = cell + cell @ sym
        else:
            positions.requires_grad_(True)
        sender, receiver = data["edge_index"][0], data["edge_index"][1]
        edge_graph = batch[sender]
        shifts = torch.einsum("ei,eij->ej", data["unit_shifts"].to(positions.dtype), cell[edge_graph])
        strained = dict(data)
        strained["positions"], strained["cell"], strained["shifts"] = positions, cell.reshape(-1, 3), shifts
        # The frozen base: its energy and forces are geometry-only. When the batch carries
        # them (`base_energy`, `base_forces`, cached per frame by the trainer) only block 0
        # is recomputed for the head's features; otherwise the full base forward runs.
        # (`dscc_base_*`: the MACE data pipeline carries its own `base_energy` /
        # `base_forces` keys, filled with zeros when absent -- a different thing.)
        cached_base = (torch.is_tensor(data.get("dscc_base_energy")) and torch.is_tensor(data.get("dscc_base_forces"))
                       and not compute_stress)
        if cached_base:
            node_feats = self.first_block(self._trunk_data(strained))
            e_base = data["dscc_base_energy"].reshape(num_graphs).to(torch.float64)
            f_base = data["dscc_base_forces"].to(torch.float64)
        else:
            out = ScaleShiftMACE.forward(self.base, self._trunk_data(strained), training=training,
                                         compute_force=False)
            node_feats, e_base, f_base = out["node_feats"], out["energy"], None
        scalars, vectors = self.features(node_feats)
        species = data["node_attrs"].argmax(dim=-1)
        edge_vector = positions[receiver] - positions[sender] + shifts
        ptr = data["ptr"]
        create = bool(training)
        head_energy = torch.zeros(num_graphs, dtype=torch.float64, device=device)
        cotangent_terms: List[Tuple[torch.Tensor, torch.Tensor]] = []   # (tensor, cotangent)
        dq_all = torch.zeros(positions.shape[0], dtype=torch.float64, device=device)
        diagnostics: Dict[str, List[Any]] = {"dq_sum": [], "n_atoms": []}
        sizes = (ptr[1:] - ptr[:-1])
        uniform = (not self.coupling and all(not s_.is_reference for s_ in states)
                   and bool((sizes == sizes[0]).all()))
        if uniform:
            # Phi = 0, equal sizes: one [B, 4n, 4n] Hamiltonian, one batched fill, one
            # cotangent -- the per-graph path below is the reference it is tested against.
            n_nodes = int(sizes[0])
            H = self.h0.batched(scalars, vectors, species, data["edge_index"], edge_vector, batch,
                                num_graphs, n_nodes)
            n_up, n_dn, r_up, r_dn = [], [], [], []
            for g, state in enumerate(states):
                numbers = [self.atomic_numbers[int(x)] for x in species[int(ptr[g]):int(ptr[g + 1])].tolist()]
                n_ref = neutral_count(numbers)
                a, b = state.counts(n_ref); c, d = State(0, 0, 0).counts(n_ref)
                n_up.append(a); n_dn.append(b); r_up.append(c); r_dn.append(d)
            t = lambda v: torch.tensor(v, dtype=torch.float64, device=device)  # noqa: E731
            sol = two_fillings(H, (t(n_up), t(n_dn)), (t(r_up), t(r_dn)), self.sigma_s)
            head_energy = sol.energy
            cotangent_terms.append((H, 0.5 * (sol.dP + sol.dP.transpose(-1, -2))))
            dq_all = sol.dq.reshape(-1)
            diagnostics["dq_sum"] = sol.dq.sum(-1).detach().cpu().tolist()
            diagnostics["n_atoms"] = [n_nodes] * num_graphs
            diagnostics["batched"] = True
        for g, state in enumerate(states):
            if uniform:
                break
            lo, hi = int(ptr[g]), int(ptr[g + 1])
            if state.is_reference:
                continue
            nodes = slice(lo, hi)
            e_mask = edge_graph == g
            ei = data["edge_index"][:, e_mask] - lo
            ev = edge_vector[e_mask]
            H = self.h0(scalars[nodes], vectors[nodes], species[nodes], ei, ev)
            numbers = [self.atomic_numbers[int(s)] for s in species[nodes].tolist()]
            n_ref = neutral_count(numbers)
            n_s = state.counts(n_ref)
            n_r = State(0, 0, 0).counts(n_ref)
            if self.coupling:
                pos_g, cell_g, sp_g = positions[nodes], cell[g], species[nodes]
                k_sr, k_lr = kernel_components(pos_g, cell_g, self.kernel)
                gamma = gamma_matrix(k_sr, k_lr, self.lambda_dir(), self.u_eff()[sp_g],
                                     self.kernel.eps_inf)
                W = None
                if self.route_b:
                    # Route B' (v4.2): W = Gamma_LR (s q0), q0 the reference fill of H0
                    # itself, attached (its geometry derivative is in the force through the
                    # (H, dP) cotangent below; a detached q0 would be non-conservative).
                    g_lr = gamma_lr(pos_g, cell_g, self.kernel.r_g, self.r_split, self.kernel.eps_inf,
                                    tol=self.kernel.tol)
                    q0 = self.reference_charges(H, numbers)
                    W = host_potential(g_lr, self.pattern_scale() * q0)
                    diagnostics.setdefault("q0_sum", []).append(float(q0.detach().sum()))
                    diagnostics.setdefault("compensation_cloud", []).append(
                        self.compensation_cloud(q0, sp_g, pos_g, cell_g))
                # Training gradient through the fixed point: unrolled for Anderson, the
                # implicit-function derivative for Newton (plan section 6). D11: the
                # production solve is the continuation from Phi = 0.
                newton = self.scf_options.method == "newton"
                start = None if warm_start is None else warm_start[g]
                if start is not None:
                    res: ScfResult = solve_dscc(H, gamma, n_s, n_r, self.sigma_s, W, start.detach(),
                                                self.scf_options, unroll=training and not newton,
                                                implicit=training and newton)
                else:
                    res = continuation_solve(H, gamma, n_s, n_r, self.sigma_s, W,
                                             self.scf_options, unroll=training and not newton,
                                             implicit=training and newton)
                head_energy[g] = res.energy
                dP_sym = 0.5 * (res.dP + res.dP.transpose(0, 1))
                # Hellmann-Feynman (plan 2.7): -Tr(dP dH/dR) - 0.5 dq^T dGamma/dR dq, with
                # V(dq) inside H so the potential's geometry dependence (Gamma, W) is in the
                # first term. The CHARGES are held fixed (detached): the functional is
                # stationary in them, and differentiating their own geometry dependence
                # would add a term the envelope theorem says is zero -- and is not, if the
                # attached dq is used, because the attached dq is the fill's charge at
                # FIXED potential, not the self-consistent one.
                dq_fixed = res.dq.detach()
                H_sc = H - torch.diag((gamma @ dq_fixed + (W if W is not None else 0.0)).repeat_interleave(4))
                cotangent_terms.append((H_sc, dP_sym))
                cotangent_terms.append((gamma, -0.5 * res.dq.unsqueeze(-1) * res.dq.unsqueeze(0)))
                # (the cotangent keeps dq's graph for the training gradient; autograd does
                # not differentiate through grad_outputs)
                dq_all[nodes] = res.dq
                for key, value in (("iterations", res.iterations), ("converged", res.converged),
                                   ("residual", res.residual), ("commutator", res.commutator),
                                   ("rho", res.rho), ("delta_energy", res.delta_energy),
                                   ("band_minus_primary", float(res.energy.detach() - res.energy_primary))):
                    diagnostics.setdefault(key, []).append(value)
                if not res.converged:
                    import logging
                    logging.warning("D-SCC: graph %d hit n_max=%d (residual %.2e); flagged, not accepted",
                                    g, self.scf_options.n_max, res.residual)
                diagnostics["dq_sum"].append(float(res.dq.sum()))
            else:
                sol = two_fillings(H, n_s, n_r, self.sigma_s)
                head_energy[g] = sol.energy
                cotangent_terms.append((H, 0.5 * (sol.dP + sol.dP.transpose(0, 1))))
                dq_all[nodes] = sol.dq
                diagnostics["dq_sum"].append(float(sol.dq.sum()))
            diagnostics["n_atoms"].append(hi - lo)
        c_q = torch.stack([self.c_q(s.Q) for s in states]).to(device)
        energy_uncal = e_base + head_energy
        result: Dict[str, Optional[torch.Tensor]] = {
            "energy_uncalibrated": energy_uncal, "energy": energy_uncal + c_q,
            "base_energy": e_base, "head_energy": head_energy, "c_q": c_q,
            "dq": dq_all, "node_feats": node_feats, "short_circuit": False,
            "calibrated": self.calibrated, "diagnostics": diagnostics,
        }
        if compute_force or compute_stress:
            inputs = [positions] + ([strain] if compute_stress else [])
            grads = [torch.zeros_like(t) for t in inputs]
            # Base: the full derivative of E_base (or its cached forces).
            if cached_base:
                grads[0] = grads[0] - f_base
            else:
                g_base = torch.autograd.grad(e_base.sum(), inputs, create_graph=create,
                                             retain_graph=True, allow_unused=True)
                for k, g in enumerate(g_base):
                    if g is not None:
                        grads[k] = grads[k] + g
            # Head: Hellmann-Feynman -- the density (and charges) as constant cotangents.
            for tensor, cot in cotangent_terms:
                g_head = torch.autograd.grad(tensor, inputs, grad_outputs=cot, create_graph=create,
                                             retain_graph=True, allow_unused=True)
                for k, g in enumerate(g_head):
                    if g is not None:
                        grads[k] = grads[k] + g
            result["forces"] = -grads[0]
            if compute_stress:
                volume = torch.det(data["cell"].view(-1, 3, 3)).abs()
                result["stress"] = grads[1] / volume.reshape(-1, 1, 1)
        return result
