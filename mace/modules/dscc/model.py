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

import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from e3nn import o3
from torch import nn

from mace.modules.models import ScaleShiftMACE
from mace.modules.dscc.fill import SIGMA_S
from mace.modules.dscc.hamiltonian import (A_MAX_DEFAULT, B_MAX_DEFAULT, DELTA_FRACTION_DEFAULT,
                                           ETA_DEFAULT, H0, Q_CUT_DEFAULT, R_CUT_DEFAULT)
from mace.modules.dscc.ewald import COULOMB, gradient_of_contraction, madelung_self
from mace.modules.dscc.kernels import (KernelConfig, gamma_lr, gamma_matrix, gamma_pair_derivative, host_potential,
                                       kernel_components, kernel_pair_gradients, gamma_lr_pair_gradient)
from mace.modules.dscc.scf import (ScfOptions, ScfResult, continuation_solve, continuation_solve_batched,
                                   solve_dscc, solve_dscc_batched, two_fillings)
from mace.modules.dscc.species import (N0, S_REF, State, U_MAX_GFN1, neutral_count,
                                       states_from_batch)
from mace.modules.dscc.fill import eigh_for, fill, site_occupation
from mace.modules.dscc.fscc import fscc_head, excess_trace_norm

# Registered defaults for the bounded learnables (plan section 2.4; to confirm before use).
LAMBDA_0_DEFAULT = 0.05
LAMBDA_MAX_DEFAULT = 2.0
U_INIT_FRACTION = 0.05          # U_eff starts at 5 % of its bound ("initialised small")
R_SPLIT_DEFAULT = 2.5           # Route B', A (> r_g)
# One `autograd.grad` call for every Hellmann-Feynman cotangent term instead of one per term.
# The terms share the base's block-0 graph and H0's, so the loop re-walked the expensive part
# once per term; `grad([t1, t2], inputs, [c1, c2])` is their sum by linearity, in one traversal.
# Set False to recover the per-term loop -- `test_fused_and_looped_hellmann_feynman_agree`
# holds the two to 1e-10, and the difference is float reassociation only.
FUSED_HF_BACKWARD = True
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

    # Class-level default so that models pickled before W6 (whose `__dict__` has no
    # `scf_free`) keep loading: `torch.load` of a whole module restores `__dict__` and never
    # calls `set_extra_state`, so an instance attribute alone would break every earlier
    # checkpoint -- which it did, on the first benchmark of the Phi = 0 arms.
    scf_free: bool = False

    def __init__(self, base: nn.Module, *, r_cut: float = R_CUT_DEFAULT,
                 kernel: Optional[KernelConfig] = None, sigma_s: float = SIGMA_S,
                 coupling: bool = False, route_b: bool = False, scf_free: bool = False,
                 directional: bool = True,
                 lambda_0: float = LAMBDA_0_DEFAULT, lambda_max: float = LAMBDA_MAX_DEFAULT,
                 u_max: Optional[Dict[int, float]] = None, r_split: float = R_SPLIT_DEFAULT,
                 q_cut: float = Q_CUT_DEFAULT, a_max: float = A_MAX_DEFAULT,
                 b_max: float = B_MAX_DEFAULT, hidden: int = 64,
                 eta: float = ETA_DEFAULT, beta_b: float = 0.0, beta_a: float = 0.0,
                 delta_frac: float = DELTA_FRACTION_DEFAULT,
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
        # W6: SCF-free. One fill of H0 as at Phi = 0, plus the NON-self-consistent host term
        # E_host = dq^T Gamma_LR (s q0) and the analytic point-charge Madelung E_M(Q; h) in
        # place of the self-consistent 1/2 dq^T Gamma dq. It is a route-B' object (the
        # pattern is s q0) and it never solves, so `coupling` must be off.
        self.scf_free = bool(scf_free)
        if self.scf_free and (self.coupling or not self.route_b):
            raise ValueError("scf_free is the W6 model: route_b on, coupling off "
                             f"(got route_b={route_b}, coupling={coupling})")
        self.r_split = float(r_split)
        self.lambda_max = float(lambda_max)
        n_s, n_v, self.block0_width = block0_layout(base)
        self.n_scalars, self.n_vectors = n_s, n_v
        self.h0 = H0(self.atomic_numbers, feature_dim=n_s, n_vectors=n_v, r_cut=self.r_cut,
                     q_cut=q_cut, directional=directional, a_max=a_max, b_max=b_max,
                     hidden=hidden, eta=eta, beta_b=beta_b, beta_a=beta_a,
                     delta_frac=delta_frac)
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
        self.lambda_fixed: Optional[float] = None
        self.u_zero = False
        self.coupling_mode = "full"
        # Arm 4 (plan 2.9): "" (D-SCC), "matched" (F-SCC with the D-SCC kernel and bounds)
        # or "full" (F-SCC with Gamma_F = E_PBC / eps_inf + diag(U_eff)). Offline comparators.
        self.fscc = ""
        self.init_from: Optional[str] = None
        self.s_max = float(S_MAX_DEFAULT)
        self.s_raw = nn.Parameter(torch.zeros((), dtype=torch.float64))       # s_max sigmoid(0) = 1
        # A1.1: the readouts must see standardised features. A model that runs without the
        # statistics reproduces the A1 failure silently (corrections that never leave 0.1 eV),
        # so this is a hard guard rather than a default. Set False only for unit tests that
        # build an `H0` directly and do not care about conditioning.
        self.require_feature_stats = True
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
                "scf_free": getattr(self, "scf_free", False),
                "lambda_max": self.lambda_max, "c_q_table": dict(self.c_q_table),
                "calibration_record": self.calibration_record,
                "coupling_mode": getattr(self, "coupling_mode", "full"), "fscc": getattr(self, "fscc", ""),
                "lambda_fixed": getattr(self, "lambda_fixed", None),
                "u_zero": getattr(self, "u_zero", False), "init_from": getattr(self, "init_from", None)}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self.kernel = KernelConfig(**state["kernel"])
        self.scf_options = ScfOptions(**state["scf"])
        self.r_cut, self.sigma_s = float(state["r_cut"]), float(state["sigma_s"])
        self.coupling, self.route_b = bool(state["coupling"]), bool(state["route_b"])
        self.scf_free = bool(state.get("scf_free", False))      # W6; absent in pre-W6 checkpoints
        self.r_split, self.lambda_max = float(state["r_split"]), float(state["lambda_max"])
        self.c_q_table = {int(k): float(v) for k, v in state.get("c_q_table", {}).items()}
        self.calibration_record = state.get("calibration_record")
        self.coupling_mode = state.get("coupling_mode", "full")
        self.fscc = state.get("fscc", "")
        self.lambda_fixed = state.get("lambda_fixed")
        self.u_zero = bool(state.get("u_zero", False))
        self.init_from = state.get("init_from")

    # ----------------------------------------------------------------- learnables

    def _inference_options(self, training: bool):
        """v5 W2 closing item: the registered inference tolerance `tol_q_inference` (1e-6)
        replaces the gate tolerance (1e-8) when the model is called with `training=False`;
        every other solver number is unchanged."""
        opt = self.scf_options
        tol_inf = getattr(opt, "tol_q_inference", None)
        if training or tol_inf is None or tol_inf == opt.tol_q:
            return opt
        import dataclasses
        return dataclasses.replace(opt, tol_q=tol_inf)

    def _need_sr(self) -> bool:
        """v5 W2: `K_SR` enters `Gamma` only through `lambda_dir`; with `lambda_dir` held at
        zero (LR-only, LR + U) its lattice sum and pair derivative are skipped."""
        return getattr(self, "lambda_fixed", None) != 0.0

    def _lr_route(self) -> str:
        return getattr(self.kernel, "lr_route", "reciprocal")

    def _background(self) -> str:
        """C13: the background convention (density for v5; a pickled config may say point)."""
        return getattr(self.kernel, "background", "density")

    def lambda_dir(self) -> torch.Tensor:
        """`lambda_dir`: learned in `[0, lambda_max]`, or held at `lambda_fixed` (Arm 2+3
        couplings "LR-only" (0), "LR + U" (0) and "lambda_dir = 1 fixed")."""
        fixed = getattr(self, "lambda_fixed", None)          # getattr: models pickled before Arm 2+3
        if fixed is not None:
            return torch.tensor(float(fixed), dtype=torch.float64, device=self.u_max.device)
        return self.lambda_max * torch.sigmoid(self.lambda_raw)

    def u_eff(self) -> torch.Tensor:
        """`U_eff[Z]` in `[0, U_max[Z]]`, or zero when `u_zero` (Arm 2+3 "LR-only")."""
        if getattr(self, "u_zero", False):
            return torch.zeros_like(self.u_max)
        return self.u_max * torch.sigmoid(self.u_raw)

    def set_coupling_mode(self, mode: str) -> None:
        """Arm 2+3 (v4.1 section 7): `lr_only` (lambda = 0, U = 0), `lr_u` (lambda = 0, U
        learned), `full` (both learned), `lambda1` (lambda = 1 fixed, U learned)."""
        modes = {"lr_only": (0.0, True), "lr_u": (0.0, False), "full": (None, False), "lambda1": (1.0, False)}
        if mode not in modes:
            raise ValueError(f"unknown coupling mode {mode!r}; expected one of {sorted(modes)}")
        self.lambda_fixed, self.u_zero = modes[mode]
        self.coupling_mode = mode
        self.lambda_raw.requires_grad_(self.lambda_fixed is None)
        self.u_raw.requires_grad_(not self.u_zero)

    def load_h0_from(self, path: str) -> None:
        """Arm 2+3 start from the Arm-1 winner: the trained `H0` from a saved `MACEDSCC` or
        from its `h0_state.pt` (a plain state dict, the form that survives the deletion
        sweep); the coupling learnables keep their inits. Pre-A1 checkpoints carry a `centre`
        buffer this `H0` no longer has and load only at the `pre-a1` tag."""
        obj = torch.load(path, weights_only=False, map_location=self.u_max.device)
        if isinstance(obj, dict) and "h0" in obj:
            self.h0.load_state_dict(obj["h0"])
            self.q0_pristine.copy_(obj["q0_pristine"].to(self.q0_pristine.device))
            self.pristine_atoms.copy_(obj["pristine_atoms"].to(self.pristine_atoms.device))
        else:
            self.h0.load_state_dict(obj.h0.state_dict())
            self.q0_pristine.copy_(obj.q0_pristine)
            self.pristine_atoms.copy_(obj.pristine_atoms)
        self.init_from = str(path)

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

    def reference_charges_batched(self, H: torch.Tensor, n0: torch.Tensor, n_up: torch.Tensor,
                                  n_dn: torch.Tensor) -> torch.Tensor:
        """`reference_charges` for `H [B, 4n, 4n]` with per-graph `n0 [B, N]` and counts."""
        with torch.no_grad():
            spectrum = eigh_for(H, getattr(self.scf_options, 'eigh_device', 'auto'))
        if not torch.is_grad_enabled():
            return n0 - self._occupied_from_spectrum(spectrum, n_up, n_dn)     # v5 W2: no density matrix
        return n0 - site_occupation(H, spectrum, n_up, n_dn, self.sigma_s)      # v5 W2 item 6: one contraction

    def _occupied_from_spectrum(self, spectrum, n_up, n_dn) -> torch.Tensor:
        """`sum_sigma sum_a f_sigma,a |Pi_i a|^2` per site from the spectrum alone (v5 W2
        closing item: the reference fill without forming `P`; inference only, detached)."""
        eps, U = spectrum
        counts = torch.stack([torch.as_tensor(n_up, dtype=eps.dtype, device=eps.device).expand(eps.shape[:-1]),
                              torch.as_tensor(n_dn, dtype=eps.dtype, device=eps.device).expand(eps.shape[:-1])])
        from mace.modules.dscc.fill import chemical_potential, occupations
        mus = chemical_potential(eps.unsqueeze(0).expand(2, *eps.shape), counts, self.sigma_s)
        f = occupations(eps, mus[0], self.sigma_s) + occupations(eps, mus[1], self.sigma_s)      # [..., n]
        w = (U * U).reshape(*eps.shape[:-1], -1, 4, eps.shape[-1]).sum(-2)                        # |Pi_i a|^2, [..., N, n]
        return torch.einsum("...ia,...a->...i", w, f)

    def reference_charges(self, H: torch.Tensor, numbers: Sequence[int]) -> torch.Tensor:
        """Route B' (v4.2): `q0_i = n0[Z_i] - sum_sigma Tr(Pi_i P_ref_sigma(H0))` at `H = H0`
        -- NEVER at `H0 - V` -- attached to `H0` through the divided-difference backward, so
        its geometry derivative (the mandatory `s dq^T Gamma_LR dq0/dR` force term, one
        Frechet contraction) and its parameter derivative are on the graph. Sums to zero
        exactly: the reference is neutral."""
        n_up, n_dn = S_REF.counts(neutral_count(numbers))
        with torch.no_grad():
            spectrum = eigh_for(H, getattr(self.scf_options, 'eigh_device', 'auto'))
        n0 = torch.tensor([float(N0[z]) for z in numbers], dtype=H.dtype, device=H.device)
        if not torch.is_grad_enabled():
            return n0 - self._occupied_from_spectrum(spectrum, float(n_up), float(n_dn))
        return n0 - site_occupation(H, spectrum, float(n_up), float(n_dn), self.sigma_s)   # v5 W2 item 6

    def scf_free_terms(self, H: torch.Tensor, spectrum, counts_s, counts_r, n0: torch.Tensor,
                       positions: torch.Tensor, cell: torch.Tensor, charges: torch.Tensor,
                       use_pairs: bool):
        """W6: the two SCF-free terms that sit on top of the `Phi = 0` band energy, from the
        SAME eigendecomposition the fill used.

        `E_host = dq^T Gamma_LR (s q0)` -- non-self-consistent: `dq` and `q0` are both fills
        of `H0` itself, no potential ever enters `H`. `E_M = 1/2 Q^2 xi(h) / eps_inf` is the
        point-charge Madelung energy of the actual cell (`madelung_self`), which takes the
        place of the self-consistent `1/2 dq^T Gamma dq`; it depends on the cell alone, so it
        carries a stress and no force.

        `occ_S` and `occ_R` are the plan's two Frechet contractions: `dq = occ_R - occ_S` and
        `q0 = n0 - occ_R` SHARE the reference occupation, so the backward costs two
        Daleckii-Krein contractions and not three. `E_host` is not a Hellmann-Feynman term --
        the energy is not stationary in a non-self-consistent charge -- so its force keeps the
        full `d dq/dR` and `d q0/dR`, which is what the `(e_host, 1)` cotangent the caller
        pushes delivers. Under the pair route `Gamma_LR` is detached and its own geometry
        derivative comes back in `pair_grad` instead (sign +: an explicit energy, not `-Tr(P dH)`).

        Batched throughout (`H [B, 4n, 4n]`, `positions [B, N, 3]`, `cell [B, 3, 3]`); the
        per-graph branch calls it at `B = 1`, so the two paths agree by construction.
        Returns `(e_host [B], e_M [B], pair_grad [B, N, 3] or None, dq [B, N], q0 [B, N])`.
        """
        n_graphs = H.shape[0]
        occ_s = site_occupation(H, spectrum, counts_s[0], counts_s[1], self.sigma_s)
        occ_r = site_occupation(H, spectrum, counts_r[0], counts_r[1], self.sigma_s)
        dq, q0 = occ_r - occ_s, n0 - occ_r
        sq0 = self.pattern_scale() * q0
        with torch.set_grad_enabled(torch.is_grad_enabled() and not use_pairs):
            g_lr = torch.stack([gamma_lr(positions[g], cell[g], self.kernel.r_g, self.r_split,
                                         self.kernel.eps_inf, tol=self.kernel.tol,
                                         route=self._lr_route(), background=self._background())
                                for g in range(n_graphs)])
        e_host = (dq * torch.einsum("bij,bj->bi", g_lr, sq0)).sum(-1)
        pair_grad = None
        if use_pairs:
            with torch.no_grad():
                d_w = torch.stack([gamma_lr_pair_gradient(positions[g], cell[g], self.kernel.r_g,
                                                          self.r_split, self.kernel.eps_inf,
                                                          tol=self.kernel.tol, route=self._lr_route())
                                   for g in range(n_graphs)])
            # ATTACHED, as route B's own `A_w`: `d_w` is the constant of the geometry, but
            # the contraction's parameter gradient (`s` above all, which enters the force only
            # here and through the cotangent) has to survive `create_graph` in training.
            pair_grad = gradient_of_contraction(dq.unsqueeze(-1) * sq0.unsqueeze(-2), d_w)
        # C13 (registered): E_M carries the model density's second-moment term
        # `4 pi r_g^2 C / Omega` alongside the point-charge Madelung constant, so that W6 and
        # the SCF models agree at fixed cell -- `xi + 4 pi r_g^2 C / Omega` is exactly
        # `E_PBC_ii(density)` minus the own-cloud self term `C / (sqrt(pi) r_g)`, which is
        # size-independent and absorbed by `C_Q` exactly (a point charge has no self term).
        xi = torch.stack([madelung_self(cell[g]) for g in range(n_graphs)])
        volume = torch.det(cell).abs()
        second_moment = 4.0 * math.pi * self.kernel.r_g ** 2 * COULOMB / volume
        e_m = 0.5 * charges ** 2 * (xi + second_moment) / self.kernel.eps_inf
        return e_host, e_m, pair_grad, dq, q0

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

    def base_dtype(self) -> torch.dtype:
        """The frozen base's own precision. The head is always float64; the base may be
        float32 (it was fine-tuned in float32, so float64 adds arithmetic precision, not
        fidelity), which halves the activation memory of its forward and backward."""
        for p_ in self.base.parameters():
            return p_.dtype
        return torch.float64

    @staticmethod
    def _cast_floats(d: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Every floating tensor to `dtype`; integer tensors and non-tensors untouched. The
        cast is differentiable, so gradients reach the float64 positions through it."""
        return {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in d.items()}

    def base_forward(self, data: Dict[str, torch.Tensor], **kwargs):
        """`ScaleShiftMACE.forward` on the frozen base in the BASE's precision, with every
        floating output returned in the head's float64. The single entry point to the base, so
        the precision split lives in one place."""
        from mace.modules.models import ScaleShiftMACE
        dtype = self.base_dtype()
        if dtype == torch.float64:
            return ScaleShiftMACE.forward(self.base, data, **kwargs)
        out = ScaleShiftMACE.forward(self.base, self._cast_floats(data, dtype), **kwargs)
        return {k: (v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in out.items()}

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
        dtype = self.base_dtype()
        if dtype != torch.float64:
            data = self._cast_floats(data, dtype)
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
        out = base.products[0](node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
        return out.to(torch.float64) if dtype != torch.float64 else out

    def features(self, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block-0 scalars `[N, n_s]` and polar vectors `[N, n_v, 3]`."""
        n_s, n_v = self.n_scalars, self.n_vectors
        scalars = node_feats[:, :n_s]
        vectors = node_feats[:, n_s:n_s + 3 * n_v].reshape(-1, n_v, 3)
        return scalars, vectors

    @torch.no_grad()
    def set_pristine_reference(self, batches: Sequence[Dict[str, torch.Tensor]]) -> int:
        """The pristine composition (sum rule), the atom count, and Route B''s `q0` reference.

        v5 amendment A1 REMOVED THE OTHER JOB THIS DID. It used to also average the
        first-block features per species and store them in `H0` as the centre of every
        bounded correction; that reference is gone from the runtime path, and with it the
        setup pass that computed it. What is left is host geometry and a charge reference,
        not feature statistics: `pristine_atoms` for the size classes, and `q0_pristine`, the
        reference-fill site charge of the pristine cell, which is Route B''s `R_eff`
        diagnostic and not read by `H0` at all. The name changed with the job.

        Returns the number of pristine atoms seen."""
        sizes = []
        for data in batches:
            ptr = data["ptr"]
            sizes.extend(int(x) for x in (ptr[1:] - ptr[:-1]).tolist())
        if not sizes:
            raise ValueError("no pristine frames")
        self.pristine_atoms.fill_(min(sizes))
        # Route B' diagnostic reference: the pristine q0 per species.
        q0_all, sp_all = [], []
        for data in batches:
            out = self.base_forward(self._trunk_data(dict(data)), training=False,
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
        return int(sp.shape[0])

    @torch.no_grad()
    def set_feature_stats(self, batches: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, object]:
        """A1.1: per-species channel mean and sd of the base's block-0 invariants over the
        TRAINING ensemble, computed once and frozen into the checkpoint.

        Training frames only -- the caller passes them. The features are label-free, so a
        held-out frame would not leak a label, but ensemble statistics fitted on the
        evaluation set are the kind of thing that is indefensible later for no gain now.

        Accumulated as sums and sums of squares in float64 over species, which is exact
        enough here (the channel means are O(10) and the deviations O(0.3), so the
        catastrophic-cancellation regime of the naive formula is far away, and the
        alternative would be two passes over the base)."""
        n_el, n_s = len(self.atomic_numbers), self.n_scalars
        dev = self.q0_pristine.device
        total = torch.zeros(n_el, n_s, dtype=torch.float64, device=dev)
        total_sq = torch.zeros(n_el, n_s, dtype=torch.float64, device=dev)
        count = torch.zeros(n_el, dtype=torch.float64, device=dev)
        for data in batches:
            out = self.base_forward(self._trunk_data(dict(data)), training=False, compute_force=False)
            scalars, _ = self.features(out["node_feats"])
            scalars = scalars.detach().to(torch.float64)
            sp = data["node_attrs"].argmax(dim=-1)
            total = total.index_add(0, sp, scalars)
            total_sq = total_sq.index_add(0, sp, scalars ** 2)
            count = count.index_add(0, sp, torch.ones_like(sp, dtype=torch.float64))
        seen = count > 0
        if not bool(seen.all()):
            missing = [self.atomic_numbers[i] for i in range(n_el) if not bool(seen[i])]
            raise ValueError(f"no training atoms of Z = {missing}; their readouts would be "
                             "standardised by statistics that do not exist")
        mean = total / count.unsqueeze(-1)
        var = (total_sq / count.unsqueeze(-1) - mean ** 2).clamp_min(0.0)
        dead = self.h0.sk.set_feature_stats(mean, var.sqrt())
        report = {"atoms_per_species": {int(self.atomic_numbers[i]): int(count[i]) for i in range(n_el)},
                  "dead_channels": {int(self.atomic_numbers[i]): int(dead[i]) for i in range(n_el)},
                  "mean_norm": {int(self.atomic_numbers[i]): float(mean[i].norm()) for i in range(n_el)},
                  "median_abs_mean_over_sd": {
                      int(self.atomic_numbers[i]):
                      float((mean[i].abs() / self.h0.sk.feat_sd[i].to(torch.float64)).median())
                      for i in range(n_el)}}
        return report

    # ----------------------------------------------------------------- gap (plan 6)

    def pristine_gap(self, data: Dict[str, torch.Tensor],
                     cached_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """`E_gap_model = LUMO - HOMO` at the exact valence count of the Hamiltonian
        actually filled at `dq = 0` -- `H0` in Route A and in W6 (`scf_free`: the host term
        is an energy, not a potential), `H0 - W` in Route B -- for each
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
            if self.route_b and not getattr(self, "scf_free", False):
                # v4.2: the gap regulariser acts on H0 - W with W from the pristine q0.
                # NOT in W6: there the host term is an ENERGY, never a potential -- the
                # Hamiltonian W6 fills is H0 itself, so its gap regulariser must act on H0
                # (the same Hamiltonian as the Phi = 0 arm, which is also what C5's
                # localisation gate reads).
                g_lr = gamma_lr(positions[nodes], cell[g], self.kernel.r_g, self.r_split,
                                self.kernel.eps_inf, tol=self.kernel.tol, route=self._lr_route(), background=self._background())
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
        if self.require_feature_stats and not bool(self.h0.sk.feat_stats_set):
            raise RuntimeError(
                "the readout input statistics are not set (v5 amendment A1.1): call "
                "set_feature_stats on the training frames first. Running without them is the "
                "A1 failure -- the species mean is ~30x the thermal deviation on base v2, the "
                "readout weights stay tiny and the on-site corrections never leave 0.1 eV")
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
            out = self.base_forward(self._trunk_data(dict(data)), training=training,
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
            out = self.base_forward(self._trunk_data(strained), training=training,
                                         compute_force=False)
            node_feats, e_base, f_base = out["node_feats"], out["energy"], None
        scalars, vectors = self.features(node_feats)
        species = data["node_attrs"].argmax(dim=-1)
        edge_vector = positions[receiver] - positions[sender] + shifts
        ptr = data["ptr"]
        create = bool(training)
        head_energy = torch.zeros(num_graphs, dtype=torch.float64, device=device)
        cotangent_terms: List[Tuple[torch.Tensor, torch.Tensor]] = []   # (tensor, cotangent)
        # Forces of the kernel terms (`-Tr(dP dV/dR) - 1/2 dq^T dGamma/dR dq`) from the
        # kernels' PAIR derivatives (constants of the geometry) contracted with the attached
        # charges: the same numbers as the cotangent route, without the autograd graph through
        # the lattice sums -- second-order under `create_graph` (~20 GB per four 79-atom
        # frames) in training, first-order (9.4 GB per four 159-atom frames) at inference.
        # Route B' takes it too (its Gamma_LR geometry term from the Gamma_LR pair
        # derivatives, its q0(H0(R)) term through a (W, -n_site) cotangent); the stress keeps
        # the cotangent route; `gamma_force_mode = "autograd"` selects it everywhere (the
        # tests' reference).
        use_pairs = (compute_force and not compute_stress
                     and (self.coupling or getattr(self, "scf_free", False))
                     and getattr(self, "gamma_force_mode", "pairs") == "pairs")
        pair_grad = torch.zeros_like(positions) if use_pairs else None
        dq_all = torch.zeros(positions.shape[0], dtype=torch.float64, device=device)
        diagnostics: Dict[str, List[Any]] = {"dq_sum": [], "n_atoms": []}
        sizes = (ptr[1:] - ptr[:-1])
        # v4.5 warm starts: a uniform batch stays on the batched path whatever its mixture
        # of stored and first-visit graphs -- the first visits get their continuation on the
        # sub-batch (detached) and every graph then takes the batched warm-started solve
        # (the per-graph path on a mixed 159-atom batch cost ~0.4 GB more per process).
        if warm_start is not None and all(w is None for w in warm_start):
            warm_start = None
        uniform = (all(not s_.is_reference for s_ in states) and bool((sizes == sizes[0]).all())
                   and not getattr(self, "fscc", ""))
        if uniform:
            # Equal sizes, no reference graph: one [B, 4n, 4n] Hamiltonian, batched fills
            # (and the batched solver when the coupling is on), block-diagonal cotangents --
            # the per-graph path below is the reference it is tested against.
            n_nodes = int(sizes[0])
            H = self.h0.batched(scalars, vectors, species, data["edge_index"], edge_vector, batch,
                                num_graphs, n_nodes)
            n_up, n_dn, r_up, r_dn, n0_rows = [], [], [], [], []
            for g, state in enumerate(states):
                numbers = [self.atomic_numbers[int(x)] for x in species[int(ptr[g]):int(ptr[g + 1])].tolist()]
                n_ref = neutral_count(numbers)
                a, b = state.counts(n_ref); c, d = State(0, 0, 0).counts(n_ref)
                n_up.append(a); n_dn.append(b); r_up.append(c); r_dn.append(d)
                n0_rows.append([float(N0[z]) for z in numbers])
            t = lambda v: torch.tensor(v, dtype=torch.float64, device=device)  # noqa: E731
            counts_s, counts_r = (t(n_up), t(n_dn)), (t(r_up), t(r_dn))
            if not self.coupling:
                sol = two_fillings(H, counts_s, counts_r, self.sigma_s)
                head_energy = sol.energy
                cotangent_terms.append((H, 0.5 * (sol.dP + sol.dP.transpose(-1, -2))))
                dq_all = sol.dq.reshape(-1)
                if self.scf_free:                                   # W6
                    pos_b = positions.reshape(num_graphs, n_nodes, 3)
                    q_form = t([float(s_.Q) for s_ in states])
                    e_host, e_m, pg, _, q0 = self.scf_free_terms(
                        H, (sol.fills[0].eps, sol.fills[0].U), counts_s, counts_r, t(n0_rows),
                        pos_b, cell, q_form, use_pairs)
                    head_energy = head_energy + e_host + e_m
                    cotangent_terms.append((e_host, torch.ones_like(e_host)))
                    if compute_stress:      # E_M is a function of the cell alone: no force
                        cotangent_terms.append((e_m, torch.ones_like(e_m)))
                    if pg is not None:
                        pair_grad = pair_grad + pg.reshape(-1, 3)
                    diagnostics["q0_sum"] = q0.detach().sum(-1).cpu().tolist()
                    diagnostics["e_host"] = e_host.detach().cpu().tolist()
                    diagnostics["e_madelung"] = e_m.detach().cpu().tolist()
            else:
                gammas, Ws = [], []
                pos_b = positions.reshape(num_graphs, n_nodes, 3)
                sp_b = species.reshape(num_graphs, n_nodes)
                for g in range(num_graphs):
                    # Under the pair route the kernels' geometry graph is not needed (the
                    # force comes from the pair derivatives, the energy's parameter gradient
                    # from lambda and U in gamma_matrix): detached, which halves the memory
                    # again at 159 atoms.
                    with torch.set_grad_enabled(not use_pairs):
                        k_sr, k_lr = kernel_components(pos_b[g], cell[g], self.kernel, need_sr=self._need_sr())
                        if self.route_b:            # Gamma_LR likewise: detached under the pair route
                            Ws.append(gamma_lr(pos_b[g], cell[g], self.kernel.r_g, self.r_split, self.kernel.eps_inf, tol=self.kernel.tol, route=self._lr_route(), background=self._background()))
                    gammas.append(gamma_matrix(k_sr, k_lr, self.lambda_dir(), self.u_eff()[sp_b[g]], self.kernel.eps_inf))
                gamma = torch.stack(gammas)
                W = None
                if self.route_b:
                    q0 = self.reference_charges_batched(H, t(n0_rows), counts_r[0], counts_r[1])
                    sq0 = self.pattern_scale() * q0
                    W = torch.einsum("bij,bj->bi", torch.stack(Ws), sq0)
                    diagnostics["q0_sum"] = q0.detach().sum(-1).cpu().tolist()
                    diagnostics["compensation_cloud"] = [self.compensation_cloud(q0[g], sp_b[g], pos_b[g], cell[g]) for g in range(num_graphs)]
                implicit = training and self.scf_options.method == "newton"
                scf_opts = self._inference_options(training)
                first_visit_fills = [0] * num_graphs
                if warm_start is None:
                    res = continuation_solve_batched(H, gamma, counts_s, counts_r, self.sigma_s, W, scf_opts,
                                                     implicit=implicit, mixed=not training)
                else:
                    starts = torch.zeros(num_graphs, n_nodes, dtype=torch.float64, device=device)
                    missing = [g for g, w in enumerate(warm_start) if w is None]
                    for g, w in enumerate(warm_start):
                        if w is not None:
                            starts[g] = w.detach().to(device)
                    if missing:
                        # First visits inside a warm batch: their continuation on the
                        # sub-batch, detached; the batched warm solve below re-converges
                        # from its fixed point in a step or two (the same fixed point).
                        m = torch.tensor(missing, device=device)
                        with torch.no_grad():
                            sub = continuation_solve_batched(
                                H[m].detach(), gamma[m].detach(), (counts_s[0][m], counts_s[1][m]),
                                (counts_r[0][m], counts_r[1][m]), self.sigma_s,
                                None if W is None else W[m].detach(), scf_opts, implicit=False, mixed=not training)
                        starts[m] = sub.dq.detach()
                        for j, g in enumerate(missing):
                            first_visit_fills[g] = int(sub.n_fills[j])
                    res = solve_dscc_batched(H, gamma, counts_s, counts_r, self.sigma_s, W, starts, scf_opts,
                                             implicit=implicit)
                    res.n_fills = [a + b for a, b in zip(res.n_fills, first_visit_fills)]
                    res.iterations = [a + b for a, b in zip(res.iterations, first_visit_fills)]
                diagnostics["warm_started"] = warm_start is not None
                diagnostics["first_visits"] = int(sum(1 for f in first_visit_fills if f)) if warm_start is not None else num_graphs
                head_energy = res.energy
                dq_fixed = res.dq.detach()
                dP_sym = 0.5 * (res.dP + res.dP.transpose(-1, -2))
                if use_pairs:
                    with torch.no_grad():
                        pairs = [kernel_pair_gradients(pos_b[g], cell[g], self.kernel, need_sr=self._need_sr()) for g in range(num_graphs)]
                    gamma_p = gamma_pair_derivative(torch.stack([d[0] for d in pairs]), torch.stack([d[1] for d in pairs]),
                                                    self.lambda_dir(), self.kernel.eps_inf)          # [B, n, n, 3]
                    n_site = torch.diagonal(dP_sym, dim1=-2, dim2=-1).reshape(num_graphs, n_nodes, 4).sum(-1)
                    A = (-0.5 * res.dq.unsqueeze(-1) * res.dq.unsqueeze(-2)
                         - n_site.unsqueeze(-1) * dq_fixed.unsqueeze(-2))                             # [B, n, n]
                    pair_grad = pair_grad + gradient_of_contraction(A, gamma_p).reshape(-1, 3)
                    cotangent_terms.append((H, dP_sym))
                    if self.route_b:
                        # Route B': the -sum_i n_site_i W_i piece of the Hellmann-Feynman
                        # term, W = Gamma_LR (s q0) -- Gamma_LR's geometry from its pair
                        # derivatives, q0(H0(R))'s through the cotangent (W, -n_site)
                        # (Gamma_LR detached above, s and q0 attached as on the cotangent route).
                        with torch.no_grad():
                            d_w = torch.stack([gamma_lr_pair_gradient(pos_b[g], cell[g], self.kernel.r_g, self.r_split,
                                                                      self.kernel.eps_inf, tol=self.kernel.tol, route=self._lr_route())
                                               for g in range(num_graphs)])
                        A_w = -n_site.unsqueeze(-1) * sq0.unsqueeze(-2)
                        pair_grad = pair_grad + gradient_of_contraction(A_w, d_w).reshape(-1, 3)
                        cotangent_terms.append((W, -n_site))
                else:
                    V_fixed = torch.einsum("bij,bj->bi", gamma, dq_fixed) + (W if W is not None else 0.0)
                    H_sc = H - torch.diag_embed(V_fixed.repeat_interleave(4, dim=-1))
                    cotangent_terms.append((H_sc, dP_sym))
                    cotangent_terms.append((gamma, -0.5 * res.dq.unsqueeze(-1) * res.dq.unsqueeze(-2)))
                dq_all = res.dq.reshape(-1)
                for key, values in (("iterations", res.iterations), ("fills", res.n_fills), ("converged", res.converged),
                                    ("residual", res.residual), ("commutator", res.commutator), ("rho", res.rho),
                                    ("band_minus_primary", (res.energy.detach() - res.energy_primary).cpu().tolist())):
                    diagnostics[key] = list(values)
                if not all(res.converged):
                    import logging
                    logging.warning("D-SCC: %d of %d graphs hit n_max=%d; flagged, not accepted",
                                    sum(1 for c in res.converged if not c), num_graphs, self.scf_options.n_max)
            diagnostics["dq_sum"] = dq_all.reshape(num_graphs, n_nodes).sum(-1).detach().cpu().tolist()
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
            if self.coupling and getattr(self, "fscc", ""):
                # Arm 4: two independent full-SCC solves with absolute charges (plan 2.9).
                pos_g, cell_g, sp_g = positions[nodes], cell[g], species[nodes]
                use_pairs_f = (compute_force and not compute_stress
                               and getattr(self, "gamma_force_mode", "pairs") == "pairs")
                with torch.set_grad_enabled(not use_pairs_f):        # as the D-SCC branches
                    k_sr, k_lr = kernel_components(pos_g, cell_g, self.kernel, need_sr=self._need_sr())
                if self.fscc == "matched":
                    gamma_f = gamma_matrix(k_sr, k_lr, self.lambda_dir(), self.u_eff()[sp_g], self.kernel.eps_inf)
                else:                                              # full kernel: E_PBC / eps + diag(U)
                    gamma_f = (k_sr + k_lr) / self.kernel.eps_inf + torch.diag(self.u_eff()[sp_g])
                n0 = torch.tensor([float(N0[z]) for z in numbers], dtype=H.dtype, device=device)
                head, st, rf = fscc_head(H, gamma_f, n0, n_s, n_r, self.sigma_s, self.scf_options)
                head_energy[g] = head
                # HF at fixed P_X and Dq_X per state: -Tr(P dH0/dR) - 1/2 Dq^T dGamma/dR Dq, S minus ref.
                dP = 0.5 * ((st.P - rf.P) + (st.P - rf.P).transpose(0, 1))
                cotangent_terms.append((H, dP))
                A_f = 0.5 * (st.dq.unsqueeze(-1) * st.dq.unsqueeze(0) - rf.dq.unsqueeze(-1) * rf.dq.unsqueeze(0))
                if use_pairs_f:
                    # The kernel term's force from the pair derivatives (see the D-SCC branch):
                    # Gamma_F is linear in the components, with lambda_dir (matched) or 1 (full).
                    d_sr, d_lr = kernel_pair_gradients(pos_g, cell_g, self.kernel, need_sr=self._need_sr())
                    lam_f = self.lambda_dir() if self.fscc == "matched" else torch.ones((), dtype=H.dtype, device=device)
                    gamma_p = gamma_pair_derivative(d_sr, d_lr, lam_f, self.kernel.eps_inf)
                    if pair_grad is None:
                        pair_grad = torch.zeros_like(positions)
                    pair_grad = pair_grad.index_add(0, torch.arange(lo, hi, device=device), gradient_of_contraction(A_f, gamma_p))
                else:
                    cotangent_terms.append((gamma_f, A_f))
                dq_all[nodes] = st.dq - rf.dq
                for key, value in (("iterations", st.iterations + rf.iterations), ("converged", st.converged and rf.converged),
                                   ("residual", max(st.residual, rf.residual)),
                                   ("excess_trace_norm", excess_trace_norm(st.P, rf.P, state.Q))):
                    diagnostics.setdefault(key, []).append(value)
                diagnostics["dq_sum"].append(float((st.dq - rf.dq).detach().sum()))
                diagnostics["n_atoms"].append(hi - lo)
                continue
            if self.coupling:
                pos_g, cell_g, sp_g = positions[nodes], cell[g], species[nodes]
                with torch.set_grad_enabled(not use_pairs):            # see the batched branch
                    k_sr, k_lr = kernel_components(pos_g, cell_g, self.kernel, need_sr=self._need_sr())
                gamma = gamma_matrix(k_sr, k_lr, self.lambda_dir(), self.u_eff()[sp_g],
                                     self.kernel.eps_inf)
                W = None
                if self.route_b:
                    # Route B' (v4.2): W = Gamma_LR (s q0), q0 the reference fill of H0
                    # itself, attached (its geometry derivative is in the force through the
                    # (H, dP) cotangent below; a detached q0 would be non-conservative).
                    with torch.set_grad_enabled(not use_pairs):      # see the batched branch
                        g_lr = gamma_lr(pos_g, cell_g, self.kernel.r_g, self.r_split, self.kernel.eps_inf,
                                        tol=self.kernel.tol, route=self._lr_route(), background=self._background())
                    q0 = self.reference_charges(H, numbers)
                    sq0 = self.pattern_scale() * q0
                    W = host_potential(g_lr, sq0)
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
                if use_pairs:
                    with torch.no_grad():
                        d_sr, d_lr = kernel_pair_gradients(pos_g, cell_g, self.kernel, need_sr=self._need_sr())
                    gamma_p = gamma_pair_derivative(d_sr, d_lr, self.lambda_dir(), self.kernel.eps_inf)
                    n_site = torch.diagonal(dP_sym).reshape(-1, 4).sum(-1)
                    A = -0.5 * res.dq.unsqueeze(-1) * res.dq.unsqueeze(0) - n_site.unsqueeze(-1) * dq_fixed.unsqueeze(0)
                    pair_grad = pair_grad.index_add(0, torch.arange(lo, hi, device=device), gradient_of_contraction(A, gamma_p))
                    cotangent_terms.append((H, dP_sym))
                    if self.route_b:                                   # see the batched branch
                        with torch.no_grad():
                            d_w = gamma_lr_pair_gradient(pos_g, cell_g, self.kernel.r_g, self.r_split,
                                                         self.kernel.eps_inf, tol=self.kernel.tol, route=self._lr_route())
                        A_w = -n_site.unsqueeze(-1) * sq0.unsqueeze(0)
                        pair_grad = pair_grad.index_add(0, torch.arange(lo, hi, device=device), gradient_of_contraction(A_w, d_w))
                        cotangent_terms.append((W, -n_site))
                else:
                    H_sc = H - torch.diag((gamma @ dq_fixed + (W if W is not None else 0.0)).repeat_interleave(4))
                    cotangent_terms.append((H_sc, dP_sym))
                    cotangent_terms.append((gamma, -0.5 * res.dq.unsqueeze(-1) * res.dq.unsqueeze(0)))
                    # (the cotangent keeps dq's graph for the training gradient; the
                    # gradient is differentiable in grad_outputs under create_graph)
                dq_all[nodes] = res.dq
                for key, value in (("iterations", res.iterations), ("fills", res.n_fills), ("converged", res.converged),
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
                e_g = sol.energy
                cotangent_terms.append((H, 0.5 * (sol.dP + sol.dP.transpose(0, 1))))
                if self.scf_free:                                   # W6, at B = 1
                    one = lambda v: torch.tensor([float(v)], dtype=torch.float64, device=device)  # noqa: E731
                    n0_g = torch.tensor([[float(N0[z]) for z in numbers]], dtype=torch.float64, device=device)
                    e_host, e_m, pg, _, q0 = self.scf_free_terms(
                        H.unsqueeze(0), (sol.fills[0].eps.unsqueeze(0), sol.fills[0].U.unsqueeze(0)),
                        (one(n_s[0]), one(n_s[1])), (one(n_r[0]), one(n_r[1])), n0_g,
                        positions[nodes].unsqueeze(0), cell[g].unsqueeze(0), one(state.Q), use_pairs)
                    e_g = e_g + e_host[0] + e_m[0]
                    cotangent_terms.append((e_host, torch.ones_like(e_host)))
                    if compute_stress:
                        cotangent_terms.append((e_m, torch.ones_like(e_m)))
                    if pg is not None:
                        pair_grad = pair_grad.index_add(0, torch.arange(lo, hi, device=device), pg[0])
                    diagnostics.setdefault("q0_sum", []).append(float(q0.detach().sum()))
                    diagnostics.setdefault("e_host", []).append(float(e_host.detach()))
                    diagnostics.setdefault("e_madelung", []).append(float(e_m.detach()))
                head_energy[g] = e_g
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
            if not create and not cached_base:
                # Inference: ONE backward for base and head together. The Hellmann-Feynman
                # contraction is the derivative of sum_terms <cot, tensor> at fixed cotangents,
                # and E_base adds its own derivative -- the same numbers as the split path
                # (tested), at the cost of a single pass through the base's graph.
                total = e_base.sum() + sum((cot.detach() * tensor).sum() for tensor, cot in cotangent_terms)
                g_all = torch.autograd.grad(total, inputs, allow_unused=True)
                for k, g in enumerate(g_all):
                    if g is not None:
                        grads[k] = grads[k] + g
                if pair_grad is not None:
                    grads[0] = grads[0] + pair_grad.detach()             # no graph at inference
            else:
                # Base: the full derivative of E_base (or its cached forces).
                if cached_base:
                    grads[0] = grads[0] - f_base
                else:
                    g_base = torch.autograd.grad(e_base.sum(), inputs, create_graph=create,
                                                 retain_graph=True, allow_unused=True)
                    for k, g in enumerate(g_base):
                        if g is not None:
                            grads[k] = grads[k] + g
                # Head: Hellmann-Feynman -- the density (and charges) as constant cotangents,
                # their graphs kept for the training gradient.
                # ONE traversal for all of them: `grad([t1, t2], inputs, [c1, c2])` is the sum
                # of the separate calls by linearity, and the terms share the base's block-0
                # graph and H0's, which the per-term loop re-walked once each. (NOT the same as
                # differentiating `sum (cot * tensor)`, which would add a `d cot/dR` term the
                # Hellmann-Feynman form must not have -- the cotangents stay multipliers here.)
                # Measured on one training step, batch 4 x 79 atoms: W6 500 -> 362 ms, B' 1070
                # -> 928 ms; outputs and parameter gradients unchanged (`ref_before.json`).
                terms = ([(([t for t, _ in cotangent_terms]), [c for _, c in cotangent_terms])]
                         if FUSED_HF_BACKWARD else [([t], [c]) for t, c in cotangent_terms])
                for tensors, cots in terms:
                    g_head = torch.autograd.grad(tensors, inputs, grad_outputs=cots,
                                                 create_graph=create, retain_graph=True,
                                                 allow_unused=True)
                    for k, g in enumerate(g_head):
                        if g is not None:
                            grads[k] = grads[k] + g
                if pair_grad is not None:
                    grads[0] = grads[0] + pair_grad
            result["forces"] = -grads[0]
            if compute_stress:
                volume = torch.det(data["cell"].view(-1, 3, 3)).abs()
                result["stress"] = grads[1] / volume.reshape(-1, 1, 1)
        return result
