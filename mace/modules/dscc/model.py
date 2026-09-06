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
from mace.modules.dscc.kernels import (KernelConfig, centred_pattern, gamma_lr, gamma_matrix,
                                       host_potential, kernel_components, project_sum_rule)
from mace.modules.dscc.scf import ScfOptions, ScfResult, solve_dscc, two_fillings
from mace.modules.dscc.species import (N0, S_REF, State, U_MAX_GFN1, neutral_count,
                                       states_from_batch)
from mace.modules.dscc.fill import fill

# Registered defaults for the bounded learnables (plan section 2.4; to confirm before use).
LAMBDA_0_DEFAULT = 0.05
LAMBDA_MAX_DEFAULT = 2.0
U_INIT_FRACTION = 0.05          # U_eff starts at 5 % of its bound ("initialised small")
R_SPLIT_DEFAULT = 2.5           # Route B, A (> r_g)
Z_MAX_FACTOR = 2.0              # Route B: |Zstar_s| <= Z_MAX_FACTOR x max |init| (plan 2.5)


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
        # Route B pattern: bounded deviation from a model-derived init (set in Phase 1).
        self.zstar_raw = nn.Parameter(torch.zeros(n_el, dtype=torch.float64))
        self.register_buffer("zstar_init", torch.zeros(n_el, dtype=torch.float64))
        self.register_buffer("z_max", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("pristine_composition", torch.zeros(n_el, dtype=torch.float64))
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

    def zstar(self) -> torch.Tensor:
        """Route B: the per-species pattern, bounded by `z_max` and on the sum rule."""
        return project_sum_rule(self.z_max * torch.tanh(self.zstar_raw), self.pristine_composition)

    @torch.no_grad()
    def initialise_route_b(self, batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Plan 2.5: `Zstar_s <- species mean over the pristine cell of q0_i = n0[Z_i] -
        sum_sigma Tr(Pi_i P_ref_sigma)` at `H = H0` with the current checkpoint, projected
        onto the sum rule; bounds from the init magnitudes; the per-site `q0` reported."""
        q0_all, species_all = [], []
        for data in batches:
            out = ScaleShiftMACE.forward(self.base, self._trunk_data(dict(data)), training=False,
                                         compute_force=False)
            scalars, vectors = self.features(out["node_feats"])
            species = data["node_attrs"].argmax(dim=-1)
            positions, cell = data["positions"], data["cell"].view(-1, 3, 3)
            sender, receiver = data["edge_index"][0], data["edge_index"][1]
            edge_graph = data["batch"][sender]
            shifts = torch.einsum("ei,eij->ej", data["unit_shifts"].to(positions.dtype), cell[edge_graph])
            edge_vector = positions[receiver] - positions[sender] + shifts
            ptr = data["ptr"]
            for g in range(int(ptr.numel() - 1)):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                nodes = slice(lo, hi)
                e_mask = edge_graph == g
                H = self.h0(scalars[nodes], vectors[nodes], species[nodes],
                            data["edge_index"][:, e_mask] - lo, edge_vector[e_mask])
                numbers = [self.atomic_numbers[int(s)] for s in species[nodes].tolist()]
                n_up, n_dn = S_REF.counts(neutral_count(numbers))
                eps, U = torch.linalg.eigh(H)
                P = fill(H, float(n_up), self.sigma_s, (eps, U)).P + fill(H, float(n_dn), self.sigma_s, (eps, U)).P
                occupied = torch.diagonal(P).reshape(-1, 4).sum(-1)
                n0 = torch.tensor([float(N0[z]) for z in numbers], dtype=torch.float64, device=P.device)
                q0_all.append(n0 - occupied)
                species_all.append(species[nodes])
        q0, sp = torch.cat(q0_all), torch.cat(species_all)
        n_el = len(self.atomic_numbers)
        mean = torch.stack([q0[sp == s].mean() if bool((sp == s).any()) else q0.new_zeros(()) for s in range(n_el)])
        init = project_sum_rule(mean, self.pristine_composition)
        z_max = Z_MAX_FACTOR * float(init.abs().max()) + 1e-6
        self.z_max.fill_(z_max)
        self.zstar_init.copy_(init)
        self.zstar_raw.copy_(torch.atanh((init / z_max).clamp(-0.999, 0.999)))
        return {"q0_sites": q0.cpu(), "species": sp.cpu(), "zstar_init": init.cpu(), "z_max": z_max}

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
        counts = torch.bincount(species_t, minlength=len(self.atomic_numbers)).to(torch.float64)
        self.pristine_composition.copy_(counts / counts.sum() * min(sizes))
        return int(scalars.shape[0])

    # ----------------------------------------------------------------- forward

    def forward(self, data: Dict[str, torch.Tensor], training: bool = False,
                compute_force: bool = True, compute_stress: bool = False,
                **_: Any) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = int(data["ptr"].numel() - 1)
        states = states_from_batch(data["carrier_counts"].view(num_graphs, -1))
        if all(s.is_reference for s in states):
            # D5: never enter the head at S_ref -- the base's outputs, bit-identically.
            out = ScaleShiftMACE.forward(self.base, self._trunk_data(dict(data)), training=training,
                                         compute_force=compute_force, compute_stress=compute_stress)
            out["head_energy"] = torch.zeros(num_graphs, dtype=torch.float64, device=out["energy"].device)
            out["energy_uncalibrated"] = out["energy"]
            out["short_circuit"] = True
            return out
        return self._charged_forward(data, states, training, compute_force, compute_stress)

    def _charged_forward(self, data, states: Sequence[State], training: bool,
                         compute_force: bool, compute_stress: bool):
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
        out = ScaleShiftMACE.forward(self.base, self._trunk_data(strained), training=training,
                                     compute_force=False)
        e_base = out["energy"]
        scalars, vectors = self.features(out["node_feats"])
        species = data["node_attrs"].argmax(dim=-1)
        edge_vector = positions[receiver] - positions[sender] + shifts
        ptr = data["ptr"]
        create = bool(training)
        head_energy = torch.zeros(num_graphs, dtype=torch.float64, device=device)
        cotangent_terms: List[Tuple[torch.Tensor, torch.Tensor]] = []   # (tensor, cotangent)
        dq_all = torch.zeros(positions.shape[0], dtype=torch.float64, device=device)
        diagnostics: Dict[str, List[Any]] = {"dq_sum": [], "n_atoms": []}
        for g, state in enumerate(states):
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
                    g_lr = gamma_lr(pos_g, cell_g, self.kernel.r_g, self.r_split, self.kernel.eps_inf,
                                    tol=self.kernel.tol)
                    pattern = self.zstar()[sp_g]
                    # `_uncentred_test` exists only for the negative tiling-ladder test of
                    # plan section 5; an uncentred pattern in production is a bug (2.5).
                    zbar = pattern if getattr(self, "_uncentred_test", False) else centred_pattern(pattern)
                    W = host_potential(g_lr, zbar)
                # Training gradient through the fixed point: unrolled for Anderson, the
                # implicit-function derivative for Newton (plan section 6).
                newton = self.scf_options.method == "newton"
                res: ScfResult = solve_dscc(H, gamma, n_s, n_r, self.sigma_s, W, None,
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
            "dq": dq_all, "node_feats": out["node_feats"], "short_circuit": False,
            "calibrated": self.calibrated, "diagnostics": diagnostics,
        }
        if compute_force or compute_stress:
            inputs = [positions] + ([strain] if compute_stress else [])
            grads = [torch.zeros_like(t) for t in inputs]
            # Base: the full derivative of E_base.
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
