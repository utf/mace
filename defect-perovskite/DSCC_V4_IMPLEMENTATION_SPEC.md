# D-SCC v4 — implementation tracker

Normative documents: `defect-perovskite/DSCC_PLAN_V4.md` (the user's plan v4, verbatim,
2026-09-06), `DSCC_PLAN_V4_1_AMENDMENT.md` (kernel-regime ruling and confirmations,
2026-09-06 evening) and `DSCC_PLAN_V4_2_AMENDMENT.md` (C5–C7 rulings: Route B′,
bound-state precondition, forces-only training with post-hoc `C_Q`; 2026-09-07). Later
amendments take precedence. It **overrules** `TRANSITION_PLAN_V8_SPEC.md`, `TRANSITION_PLAN_V8_1_ADDENDUM.md`
and `TRANSITION_PLAN_V8_1_IMPLEMENTATION_SPEC.md`, which are history only. This file is the
task list and the registers the plan's §10 rules require. Paths relative to the worktree
`/home/alex/src/mace/.claude/worktrees/size-extensivity` (branch `defect`).

## 0. Operating rules

- **Task tracking.** The §3 checklist below is the task list; the status column is updated
  in the same commit as the work.
- **Phase gates.** A phase's code may be written early, but no later phase's *results* are
  opened until the previous phase's gates pass (plan preamble).
- **Thresholds first.** Every threshold, default and convention is entered in the register
  (§2) *before* the result it governs is opened; anything entered later is labelled
  `retrospective`.
- **GPUs.** b3 GPUs 4–7 only, up to two runs per GPU. **GPU 6 (PCI 0000:8B:00.0) faulted
  ("Unknown Error") at ~16:50 on 2026-09-06** and is unavailable until b3 is cold
  power-cycled — not done without a fresh instruction. Usable now: 4, 5, 7 (six runs). The
  local A4000 (GPU 0 on val-perovskite) is allowed for tests and small probes.
- **Git.** Commit only own paths, never `git add -A`; b3 is synced from committed content.
- **Analysis.** No analysis of any run beyond the plan's registered reports.

## 1. Open confirmations and decisions for the user

| # | Item | Status |
|---|---|---|
| C1 | Smearing of the labels: the old code records "doped's default ISMEAR = 0, SIGMA = 0.05 eV" as the labels' smearing (`defect_counting.py`). Plan §11 registers Gaussian 0.05 eV and asks for confirmation from the label paper's methods (Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025)). | **confirmed** (v4.1) |
| C2 | Whether `E_gap = 2.40 eV` is the static-lattice or the thermal-average PBE gap (decides which cell the gap regulariser acts on, §6). `band_edges.json` only carries the symmetric ±1.2 eV placement. | **ruled: static lattice** (v4.1) → the regulariser acts on the static pristine cell (D8); thermal-mean gap reported |
| C3 | b3 GPU 6 fault: cold power cycle now, or leave until GPU waves are needed (Phase 3)? | **ruled: leave** while GPUs 4, 5, 7 are accessible (checked 2026-09-06 evening: yes); cold power cycle only if not |
| C7 | **Energy admission (plan §6) admits no charged energies at either size under the registered defaults** (`dscc_admission_tables.json`, 4 outer folds, out-of-fold null from `cf_base_f0..f3`, bin 0.2 Å, `n_min` 3, `s_tol` 0.05 eV/Å, z = 2). 79 atoms: charged window d = 4.93–6.80 Å (1019 frames), 865 out-of-fold neutral frames per outer fold but none beyond 6.1 Å → coverage fails (as the plan expected); where measurable the null slope is `s0 = +0.13 ± 0.01 eV/Å` (> s_tol on its own); `sQ = +0.371 ± 0.022 eV/Å`. 159 atoms: 17 charged frames over 4.84–7.12 Å, only 12 out-of-fold neutral frames → every 0.2 Å bin is below `n_min` → coverage fails; `s0 = +0.08 ± 0.09`, `sQ = −0.095 ± 0.006 eV/Å`. Consequence: with nothing admitted, `C_Q` cannot be profiled and the head would train on forces alone. Options: (a) register a coarser 159-atom coverage rule (e.g. bin 0.5 Å, `n_min` 2 — still fails at the window ends), (b) admit 159-atom energies on the s0 criterion alone with coverage reported as failed (a documented deviation), (c) accept forces-only training and report `E(+1)−E(0)` as uncalibrated. Ruling needed before Phase 2 opens. | **ruled (v4.2)**: the registered outcome stands — forces at every size, no energy term in the loss; `C_Q` profiled post hoc (labelled retrospective) on 159-atom charged frames with `d` inside the out-of-fold neutral range; `s0(159) ± SE` reported from the 12 frames (currently +0.08 ± 0.09: SE alone fails `s_tol` → flag "base support at 159 unverified"); the 159-atom shape residual becomes the held-out prediction that replaces "energy shape on admitted sizes" in arm selection |
| C6 | **Route B centring does not remove the `1/L` cross term** (kernel-level tiling ladder with a fixed localised carrier, `test_kernels.TestRouteBLadder`): a per-species pattern on the present ions leaves the missing ion as a LOCAL charge `z_d = -Zstar_Cl` at the vacancy; centring spreads its compensation uniformly over the cell, an O(1/L³) change. The carrier's images interact with the local charge, so `Phi_cc + E_SF` carries `-alpha_M C Q z_d / (eps_inf L)` on top of the physical `-alpha_M C Q² / (2 eps_inf L)`: measured slope −8.59 eV·Å (centred), −9.79 (uncentred; the `(1 + 2 z_d)` prediction is −10.2) vs −5.11 (Madelung) with `Zstar_Cl = -0.5`. A pattern that is neutral locally (the missing ion's charge put back at the vacancy — what the model's own reference-fill charges `q0_i` do by construction, since `sum_i q0_i = 0` on the reference) keeps the Madelung slope to 5 %. So plan §2.5's positive/negative ladder test cannot distinguish centred from uncentred, and Route B as written adds a size-dependent term no `C_Q` absorbs (0.13 eV × 2 z_d between 79 and 159 atoms). Proposed Route B′: the pattern is the reference-fill site charge `q0_i` (model-derived, label-free, exactly neutral per cell, local compensation included), optionally scaled by one learnable per species; ruling needed. | **ruled (v4.2): Route B′** — pattern = `q0(R)` from `fill(H0(R), N_ref)` at `H = H0` (never `H0 − V`), one optional global scale `s ∈ [0, s_max]` (init 1; per-species scales prohibited), the geometry derivative `s dqᵀ Γ_LR ∂q0/∂R` mandatory in the force; local-neutrality ladder gate (5 % of Madelung) with the species pattern as the negative test, re-run on the trained Arm-1 `H0`; the centred/uncentred test withdrawn |
| C5 | Single-valuedness gate (plan §5, root rule): at initialisation the real frames have no bound vacancy state and the SCF has distinct fixed points on some frames (D10 finding). Read the gate on the trained model (the arms report the root rule anyway), or require an initialisation with a bound state first? | **ruled (v4.2)**: gate on trained models; bound-state precondition for Arm 2+3 (`Delta_c` = 10 σ_s, `N_eff ≤ N_loc`, ≥ 95 % of neutral-vacancy frames); per-frame SCF init = continuation from Φ = 0; single-valuedness monitored on a per-epoch subsample with a registered ceiling; 8/20 and 900/1030 recorded as the initialised-model diagnostic |
| C4 | **Regime-A placement check fails on the training set at the plan's defaults** (`r_d1 = 3.2`, `r_d2 = 3.6` Å, floor 1e-3). Identified first-shell Pb–Cl bonds beyond `r_d1`: 3.1 % on the 544 pristine frames, 5–7 % on the 79-atom vacancy frames (the six-nearest rule counts a Cl across the vacancy for the flanking Pb), 1.2–1.6 % at 159 atoms; intra-octahedron Cl–Cl edges below `r_d2`: 2.0–2.6 % (0 at 159 atoms). The first-shell Pb–Cl distribution is thermal and wide: p50 2.88 Å, fraction > 3.0 / 3.1 / 3.2 / 3.3 Å = 22.5 / 10.8 / 5.6 / 3.4 %. No window in the 3.2–3.6 Å gap clears a 1e-3 floor on these frames. Options for ruling: (a) register a higher floor (≈ 5e-2) with the physical consequence that a few % of first-shell pairs sit inside the switch; (b) move the window up (e.g. 3.5–4.0 Å, still below the 4.8 Å vacancy-spanning minimum, but then the Cs–Cl shell at 3.5–4.1 Å is inside it); (c) make regime B (no switch) the primary regime and report regime A as failing its placement gate. Also: exclude the vacancy-flanking Pb's sixth neighbour from the identified set, or run the check on pristine frames only. | **ruled (v4.1): regime B primary; regime A failed its gate, retained as a reduced ablation only** (retrospective floor 5e-2 for the ablation; flanking-Pb bond fraction and `m_sw` reported; excluded from selection) |

## 2. Registers

### 2.1 Registered values (plan §11)

| Quantity | Value | Source / status | Registered |
|---|---|---|---|
| `sigma_s` | 0.05 eV, Gaussian (`f = erfc(x)/2`, `R = -sigma_s sum exp(-x^2)/(2 sqrt(pi))`) | plan §1/§11; label-code convention | 2026-09-06 (plan) |
| `n0` | Cs 1, Pb 4, Cl 7 | plan §1 | 2026-09-06 (plan) |
| `eps_inf` | 4 | plan §1 (host input) | 2026-09-06 (plan) |
| `E_gap` | 2.40 eV | plan §1 (host input); convention C2 open | 2026-09-06 (plan) |
| Coulomb constant `C` | 14.399645 eV·Å | `defect_madelung.COULOMB_CONSTANT` | 2026-09-06 |
| Gaussian width convention | a charge of width `r_g` has density `∝ exp(-r²/(2 r_g²))`; two of widths `s_i`, `s_j` interact as `C erf(r / sqrt(2 (s_i² + s_j²))) / r` (equal widths: `erf(r/(2 r_g))/r`, self term `C/(sqrt(pi) r_g)`); the LES `sigma` is `sqrt(2) r_g` (measured: cross term of two unit charges, Richardson-extrapolated in 1/L, matches `r_g = sigma/sqrt(2)` to 4e-5 at r = 1–3 Å); LES with `remove_self_interaction` excludes the Gaussian self term (a lone unit charge gives `-alpha_M C/(2L)` exactly: `E·L = -20.428` at L = 20/40/80 Å) | `scratchpad/les_convention.py`, 2026-09-06 | 2026-09-06 |
| `U_max[Z]` | GFN1-xTB hardness `GAM` (Hartree): Cl 0.519712, Cs 0.085110, Pb 1.000000 → 14.142, 2.316, 27.211 eV; bounds only | `param_gfn1-xtb.txt` (grimme-lab/xtb, doi 10.1021/acs.jctc.7b00118) | 2026-09-06 |
| Background | VASP neutralising background; `E_bg = -pi C sigma^2 Q^2 / V` for Gaussian width `sigma` | plan §11; `latent_ewald.LatentEwald.energy` | 2026-09-06 (plan) |
| Spin split | majority-channel: `N_ref_up = ceil(N_ref/2)`, `N_ref_dn = floor(N_ref/2)` | plan §1 | 2026-09-06 (plan) |
| Base features | `l = 1` only, position derivatives through the recomputed first block | plan §11 | 2026-09-06 (plan) |
| Regime A `r_d1`, `r_d2` | 3.2 Å, 3.6 Å (defaults) | plan §2.4 | to register before Arm 2+3 |
| Placement-check floor | 1e-3 (default) | plan §2.4 | to register before Phase 0 gate |
| `m_sw` floor, `f_SR` floor | — | plan §2.4 | to register before Arm 2+3 |
| Regime B `r_s` | 6.5 Å (default); `K_SR` image range: converged to 1e-10 eV automatically | plan §2.4 | to register before Phase 0 gate |
| `r_g`, `r_split` | — | plan §2.4/§2.5 | to register before Phase 0 gate |
| `lambda_0`, `lambda_max`, `U_max[Z]` (GFN1-xTB hardness, bounds only) | — | plan §2.4 | to register before Phase 1 |
| `n_max`, `tol_q`, `tol_E`, `tol_c`, `tol_root`, `rho` ceiling, continuation schedule, mixing | — | plan §2.6 | to register before Phase 1 gate |
| `Z_max`, Route B L2 weight | `Z_max = 2 × max_s |Zstar_init,s|` (`Z_MAX_FACTOR`); L2 weight: — | plan §2.5; `model.py` | default 2026-09-06, to confirm |
| `lambda_0`, `lambda_max`, `U_eff` init | 0.05, 2.0, `0.05 U_max[Z]` (`LAMBDA_0_DEFAULT`, `LAMBDA_MAX_DEFAULT`, `U_INIT_FRACTION`); bounded by `x_max sigmoid(raw)` | `model.py` | defaults 2026-09-06, to confirm before Phase 1 gate |
| `r_g`, `r_split`, `q_cut`, `a_max`, `b_max`, `r_cut` | 1.0 Å, 2.5 Å, 4.5 Å, 1.0 eV, 1.0 eV, 10.0 Å | `kernels.KernelConfig`, `model.py`, `hamiltonian.py` | defaults 2026-09-06, to confirm |
| Solver: `tol_q`, `tol_E`, `tol_c`, `n_max`, mixing, history, `tol_root`, `rho` ceiling, continuation steps | 1e-8 e, 1e-10 eV, 1e-7, 100, 0.3 (Anderson), 6, 1e-6, 0.9, 4 | `scf.ScfOptions` | defaults 2026-09-06, to confirm before Phase 1 gate |
| `s_tol`, `z`, `n_min`, coverage bin width, out-of-fold base protocol id | 0.05 eV/Å, 2.0, 3, 0.2 Å, `cf_base_4fold_seed0` (`admission.AdmissionConfig`) | plan §6 | defaults 2026-09-06, to confirm before Phase 2 results are opened |
| Gap-regulariser convention | static-lattice or thermal-mean (C2) | plan §6 | before Phase 2 |
| Benchmark hardware / batch / trajectory | — | plan §8 | before Phase 4 |
| Arm thresholds | — | plan §7 | before each arm's results are opened |

### 2.2 Decisions

**D1 — build strategy (2026-09-06).** The head is built in a new namespace —
`mace/modules/dscc/` with a new model class `MACEDSCC(ScaleShiftMACE)` and tests in
`tests/extensions/dscc/` — and the old `defect_*` modules are left untouched until the Phase 1
gates pass; the §3 deletion is then one sweep commit ticked against the plan's delete list
(§3.6 below). Reuse is by import: `SlaterKosterH`, `harrison_initialise`, `find_mu`,
`occupation_of`/`entropy_of`/`occupation_slope` (Gaussian default), `_FermiDensityMatrix` /
`_dk_backward`, `site_charges`, `head_forces` (`defect_counting`); `ElectronicStateSpec`,
`reference_state` (`defect_state`); `profile_charge_constant`, strata (`defect_objective`);
`LatentEwald` (`latent_ewald`) as the Ewald *oracle*; the base feature cache
(`defect_cache.BaseOutputCache`). Not imported: gauge, windows, lift, boundary, composition,
`profile_intercepts` (all on the delete list). Every commit stays green; old checkpoints stay
loadable for reference until the sweep.

**D2 — state adapter (2026-09-06).** From the dataset's `carrier_counts = (e_maj, e_min,
h_maj, h_min)`: `dN_up = e_maj - h_maj`, `dN_dn = e_min - h_min`, `Q = -(dN_up + dN_dn)`;
`Q == cell_charge` asserted on every frame at load. Reference split per §1 (majority channel);
`m_s_ref_doubled` of the dataset is the same convention (79-atom V_Cl: N_ref = 409 → (205,
204); `vcl_qp1`: (204, 204), Q = +1).

**D3 — Ewald matrix (2026-09-06).** `E_PBC` is built as an explicit real + reciprocal Ewald
matrix of unit Gaussian charges in float64 with autograd through positions and cell and
cutoffs derived from the tolerance; `0.5 q^T E_PBC q` must equal `LatentEwald.energy(q)`
(LES + background) for random neutral and net-charged `q` — that is how "keep the verified
full-sum Ewald" is honoured. The LES `sigma` ↔ `r_g` relation and the diagonal convention
(`E_PBC_ii` includes the Gaussian self term, so `K_LR_ii = E_PBC_ii - 1/(sqrt(pi) r_g)`) are
verified numerically, not assumed (Phase 0 tests).

**D4 — batching by orbital count (2026-09-06).** The SCF's `eigh` is `[B, n_orb, n_orb]`;
316- and 636-orbital frames do not batch. Batches are grouped by atom count in the loader
(the old `--defect_size_grouped_batches` mechanism), registered now because the loader is
Phase 0.

**D5 — the neutral null is a control-flow short-circuit (2026-09-06).** At `S = S_ref`
`forward` never enters the head branch, so the base outputs are returned bit-identically
(adding an exact zero can change the last bit).

**D6 — base wiring (2026-09-06).** The frozen neutral base is
`/home/alex/runs/aprime_prod/aprime_prod.model` (the `--defect_base_init` of the last v8
wave): a `MACEDefect` object with `spectral_head = False`, `r_max = 5.0 Å`, two interactions,
block-0 irreps `128x0e+128x1o`, species [Cl, Cs, Pb]. `E_base`, `F_base` and the stress are
taken from `ScaleShiftMACE.forward(base, data)` (the parent class's forward: measured equal
to `base(data)` at `S_ref` to 1e-16), with every base parameter frozen; `out["node_feats"]`
`[N, 640]` is block 0 (`[:, :128]` scalars, `[:, 128:512]` the 128 `l = 1` vectors as
`[N, 128, 3]`) followed by block 1, attached to positions and cell. The head's neighbour
list is built in the data pipeline at the head's `r_cut` (registered; the old head used 10 Å
and the vacancy-spanning Pb–Pb pair is 4.8–7.1 Å, so the base's 5 Å list cannot serve) and
the trunk is fed the `≤ r_max` subset, as the old model did. `cf_base_f0..f2` are the
k-fold bases for the out-of-fold `s0` of §6 (Phase 2).

**D7 — the rank-2 descriptor (2026-09-06).** `Q_i = sum_j w(r_ij)(rhat rhat^T - I/3)` with
`w = (1 - (r/q_cut)^6)^2`, `q_cut = 4.5 Å` (registered default). A traceless rank-2 tensor is
even under inversion, so `Q_i` vanishes at sites of cubic symmetry (ideal Pb, Cs), not at
every centrosymmetric site as the plan's remark says: on the ideal Cl site (D4h) it is the
uniaxial quadrupole along the Pb–Cl–Pb axis, `2w(2.8) diag(-1/3, -1/3, 2/3)`, which is the
crystal field that splits Cl `p_sigma` from `p_pi` — the sign Arm 1 (iii) tests. The formula
is what is implemented; the remark is corrected here. The rank-1 term is the one that
vanishes at centrosymmetric sites.

**P0.4 at production size (2026-09-06).** On a 159-atom training frame with a random `dq`
of net +1 (energy 496 eV): `eta` = 1.5 / 2.5 / 4.0 Å against the default 3.14 Å gives
|ΔE| ≤ 2.6e-11, |ΔF| ≤ 2e-12, |Δstress| ≤ 5.2e-10 / 6.9e-11 / 2.3e-11 eV. The stress
figure at `eta = 1.5` is float64 rounding at the 500-eV test scale (1e-12 relative); at
physical `dq` (|Q| = 1, energies of a few eV) the absolute figure is 1e-11. Cost 0.1–0.4 s
per matrix on CPU at 160 atoms, 0.34 s on the local GPU.

**D8 — the static pristine cell for the gap regulariser (2026-09-06, after C2).** The
dataset has no relaxed pristine frame (every `ideal` frame is a thermal snapshot) and no new
DFT is allowed. The static pristine cell is therefore the label-free time-average of the
544 pristine frames — the registered site-median lattice symmetrised over the group it
generates (Pnma × the 2×2×1 supercell translations, 32 operations; built in the v8 programme
as `defect_composition.mean_pristine_lattice`, cell 16.035 × 16.076 × 11.395 Å, Cl rms
symmetrisation shift 0.10 Å). Its construction is ported into `dscc/` before the deletion
sweep (Phase 2 item P2.2a). The thermal ensemble-mean gap over pristine frames is reported
as a diagnostic. If the user can supply the paper's relaxed bulk structure, it replaces this
without other change.

**Flanking-Pb placement fraction (v4.1 pre-condition for any regime-A run; computed
2026-09-06 evening, `scratchpad/flanking_pb.py`).** Flanking Pb identified label-free as the Pb
whose sixth-nearest Cl is beyond 4.0 Å (a first shell of five): exactly 2 per frame on 1957 of
1985 79-atom vacancy frames (1 on 27, 3 on 1). Their real first-shell bonds beyond `r_d1 =
3.2 Å` against the other Pb's: **V_Cl⁰ 79 at.: 15.4 % vs 4.3 % (×3.6); 159 at.: 5.3 % vs 0.3 %
(×21)** — the expected enhancement, on the neutral frames (where `dq = 0`); **V_Cl⁺ 79 at.:
2.9 % vs 3.2 % (×0.9); 159 at.: 0 % vs 0.2 %** — no enhancement on the charged frames, where
`dq` sits (the flanking Pb of V_Cl⁺ move away from the vacancy and their remaining bonds are
not stretched). Either way the fraction is 30× the 1e-3 floor; the ruling stands.

**D9 — kernel regime (v4.1).** `KernelConfig.regime` defaults to `"B"`; regime A is
constructed only explicitly for the ablation arm. The flanking-Pb restricted placement
fraction is computed before any regime-A run (see the note under P0.5).

**D10 — the SCF solver is a damped Newton on the exact response Jacobian (2026-09-06
evening).** On the real 79-atom V_Cl⁺ frames at the initialised head, Anderson mixing
(0.3 / 6) converged on some frames in 30–50 iterations and not at all on others within
100 (residual oscillating at 0.05–0.17 e): the map `dq -> dq_new` is strongly non-linear
there (see the finding below). The plan's "broyden_or_anderson" is replaced by its exact
limit: `hole_response` gives `M = d dq_new / dV` in closed form from the eigenbasis
(Daleckii-Krein divided differences with the fixed-N correction, over the active levels
only — `O(N_act n_orb N²)`, FD-verified to 1e-6), the fixed-point Jacobian is `J = M Gamma`,
and each step solves `(I - J) delta = r` with backtracking on the unmixed residual. On the
same two frames: 6 and 8 iterations (Anderson 34/44 or failing). Anderson stays available
(`ScfOptions.method`). The Anderson least squares is now the Tikhonov-regularised normal
equations (a full-rank QR driver on CUDA returned garbage on nearly collinear residual
differences — the solve must not depend on the device). Convergence criteria unchanged
(unmixed residual `tol_q` and `|ΔJ| < tol_E`).

**Finding (2026-09-06 evening): the initialised `H0` has no bound vacancy state.** On the
real frames, Harrison-initialised `H0` gives the pristine cell a 2.400 eV gap (−9.93 →
−7.53 eV). On the V_Cl frames the reference majority electron (205th) sits in a
near-degenerate manifold at the conduction-band edge (−7.532, −7.529, −7.502, −7.45 eV;
gap(up) 0.003–0.045 eV): the vacancy-spanning Pb–Pb `pp` hopping at 5–7 Å does not split a
dimer state off the CBM at initialisation (the Arm-1 question). Consequences: the
`Phi = 0` hole is delocalised over the Pb sublattice (N_eff 17–18); the localising
feedback on a near-degenerate manifold produces **distinct fixed points** on some frames
(frame 2: max|dq| 0.287 by Newton vs 0.402 by Anderson on CPU, 0.290 on CUDA — different
converged solutions), i.e. the single-valuedness gate of plan section 5 fails at
initialisation on real frames for a physical reason. A full-set run with Newton is in
progress (`scratchpad/stability_probe.py cpu`); ruling needed on whether the Phase-1
single-valuedness gate is read at initialisation or on the trained model (C5).

**D11 — continuation from Φ = 0 (v4.2, C5).** With Φ on, every per-frame solve starts from
the Φ = 0 two-fillings solution and ramps `Gamma` and `W` together from zero to their full
values in `continuation_steps` (registered, 4) warm-started Newton solves, the last of which
is the production solution (implicit differentiation applies to it alone). The root-rule
harness keeps its zero-coupling variant (`lambda_dir = U_eff = W = 0`, `K_LR` on) as a
validation initialisation.

**Registered for the precondition (v4.2, before Arm-1 results are opened):** `Delta_c = 10
sigma_s = 0.5 eV`; `N_loc = 4` (default: a Pb-dimer state has `N_eff` 2–3; to confirm);
fraction 0.95; `s_max = 2` for the Route B′ scale; training-time single-valuedness subsample
and ceiling: to register before Arm 2+3.

## 3. Task list

Status: `todo` / `wip` / `done` / `blocked`.

### 3.0 Bookkeeping
| # | Task | Status |
|---|---|---|
| 0.1 | Cancel the v8 s14a wave; leave the v8 branch clean (`06c04c8`, 689 tests green) | done |
| 0.2 | Plan v4 saved verbatim; this tracker; memory updated | done |
| 0.3 | Report to the user: cancellation outcome, GPU 6 fault, C1–C4 | done 2026-09-06 (evening) |

### 3.1 Phase 0 — scaffold (plan §4)
| # | Task | Gate | Status |
|---|---|---|---|
| P0.1 | Species table `n0`, state adapter (D2), `N_ref`, spin split; dataset assertion `Q == cell_charge` | counts match on every frame | done — `dscc/species.py`; all 2877 train+valid frames: `Q == cell_charge`, (205,204)/(204,204)/(208,208)/(413,412)/(412,412) |
| P0.2 | Data pipeline: geometry-state groups formed before the split; strata keys; size-grouped batches (D4) | no group split across folds | done — `dscc/data.py` (`frame_meta`, `atomic_data` at `r_cut`, `split_by_group`, `assert_no_group_split`, `SizeGroupedSampler`) |
| P0.3 | `fill(H, N)`: Gaussian smearing, bisection `mu`, `P`, `F_band`, generalised entropy; matrix-function backward (batched Function over `_dk_backward`) | `dF_band/dH_ab = P_ba` to 1e-10 | done — `dscc/fill.py`; FD 1e-8, autograd exact; density response vs FD 1e-6; batched == per-frame |
| P0.4 | `E_PBC` Ewald matrix of Gaussians with derivatives (D3); LES oracle test; tiling-ladder `K_LR_ii` → `-alpha_M/L` | independent of the splitting parameter to 1e-10 eV (E, F, stress); oracle agreement | done at toy size — `dscc/ewald.py`; η-independence 1e-10 (E, F, stress), LES oracle 1e-6, ladder 1e-8, pair width convention; 159-atom gate: see log |
| P0.5 | Regime A: `K_SR` (erf, `w_dir` C2 switch), `K_LR`; placement check over identified first-shell bonds; `m_sw` | placement floors satisfied on the training set | **gate FAILED (C4, ruled v4.1)**: regime A is a reduced ablation only; flanking-Pb restricted fraction: `scratchpad/flanking_pb.py` (running) |
| P0.6 | Regime B: `K_SR` lattice sum with automatic image range; `K_LR`; `f_SR` | converged to 1e-10 eV; rewrapping-invariant | done — `K_SR + K_LR = E_PBC` to 1e-10, range-converged 1e-12, rewrapping 1e-12 |
| P0.7 | `Gamma` (both regimes, `lambda_dir`, `U_eff` bounded), `Gamma_LR` (`r_g`/`r_split` cross Ewald), `Zbar` centring, `W` | splitting-parameter independence of `Gamma_LR` | matrices done (`gamma_matrix`, `gamma_lr`, `centred_pattern`, `project_sum_rule`, `host_potential`); the bounded learnables live in the head module (P1) |
| P0.8 | `H0`: reuse `SlaterKosterH` (SK, Harrison init, decay lengths, log modulation, centred scalar onsite); rank-1 `l=1` descriptor; geometric rank-2 `Q_i`; head graph at its own `r_cut` (D6) | invariance under translation / rotation / permutation / rewrapping with covariant derivatives; `Q_i = 0` at cubic sites (D7) | done — `dscc/hamiltonian.py`, `dscc/graph.py`; rotation covariance `H' = D H D^T` 1e-10, permutation 1e-12, FD of the spectrum 1e-7; scalar-only control = block off |
| P0.9 | Serialisation and checkpoint round trip | loaded checkpoint reproduces an uncached forward bit-for-bit | done — `dscc/model.py` (`MACEDSCC`; registered numbers, `C_Q` table and record in `extra_state`); whole-object and state-dict round trips bit-identical |

### 3.2 Phase 1 — D-SCC forward and derivatives (plan §5)
| # | Task | Status |
|---|---|---|
| P1.1 | Batched SCF loop (Anderson/Broyden, unmixed residual, `tol_q`/`tol_E`/`tol_c`, `n_max` cap + flag, `rho`) | per-graph loop done — `dscc/scf.py` (`solve_dscc`: damped Newton on the exact Jacobian (D10) with Anderson fallback, unmixed residual, cap flagged, `rho`, commutator, primary vs band; implicit differentiation for training); cost at 79 atoms on 8 CPU threads ≈ 4.8 s/frame at the initialised head (many frames at the cap); batching over equal-size frames: deferred to the training-throughput pass (Phase 2), the per-graph loop is the reference |
| P1.2 | Energy (band form vs primary functional to 1e-9 eV), HF forces, autodiff stress; neutral short-circuit (D5) | done — model `_charged_forward`: band form, HF cotangents `(H, dP)`, `(Gamma, -½ dq dqᵀ)`, strain-based stress; short-circuit bit-identical |
| P1.3 | Route B′ switch (off by default); both regimes selectable | done (v4.2) — `route_b`: `W = Γ_LR (s q0)`, `q0 = reference_charges(H0)` attached (its geometry derivative reaches the force through the `(H, dP)` cotangent = the Fréchet contraction; a detached `q0` measurably changes the force, test), one global `s = s_max σ(s_raw)` (init 1); `compensation_cloud` R_eff diagnostic; `pristine_gap` on `H0 − W` |
| P1.4 | Root-rule harness: initialisations (i)–(iii), `tol_root`, symmetry-equivalent collapse | `scf.root_rule` (zero / continuation / warm, spread, `passed`); production per-frame init = `continuation_solve` from Φ = 0 (D11); `forward(warm_start=...)` for initialisation iii; symmetry-equivalent collapse by identical observables: todo |
| P1.5 | Gates: neutral null; `sum dq = Q` to 1e-12; gauge shift; FD forces {1e-2,1e-3,1e-4} Å to 1e-4 eV/Å and six strains; invariances to 1e-10 eV; stability / single-valuedness on every frame of both charge states; Route B centred/uncentred ladder test | toy-size gates pass (`test_model.py`, `test_scf.py`): neutral null bit-identical; `sum dq = Q` 1e-12 (Newton-polished `mu`); gauge `-aQ` 1e-10; FD forces 3e-6 and strains 1e-7 (both regimes, Route B); invariances 1e-9; band vs primary 1e-9; root rule; unrolled vs envelope gradient. FD step ladder {1e-2,1e-3,1e-4} (< 1e-4 eV/Å, not worse at tighter SCF tolerance) and all six strains: pass (`test_gates.py`). **Real-data run at the initialised head** (regime B, Route A, Newton without fallback, every charged frame of train+valid, CPU): 79 at.: 900/1030 converged within n_max = 100 (iterations p50 7), 159 at.: 16/17; `sum dq = Q` to 6e-14 on every frame; |band − primary| ≤ 9e-3 (non-converged frames only); 0.95 s/frame (79), 1.8 s (159); J* 7.55 ± 0.05 eV; max|dq_i| p50 0.31. **Root rule on 20 frames: 8/20 pass** (spreads up to 0.40 e between the zero-start and continuation fixed points) — the single-valuedness gate FAILS at initialisation (no bound state, D10 finding; C5 ruled: read on trained models). With the Anderson fallback in Newton (final solver): 79 at. **1018/1030** converged, 159 at. 16/17, root rule 10/20, 0.98 s/frame — the initialised-model diagnostic of record. Route B model-level tiling ladder: needs a bound carrier (the toy cannot bind one; C6 records the kernel-level result) |

### 3.2b v4.2 items (C5–C7)
| # | Task | Status |
|---|---|---|
| V2.1 | Bound-state precondition for Arm 2+3 (`Delta_c`, `N_loc`, fraction; label-free on neutral-vacancy frames) | done — `dscc/precondition.py` (`bound_state_precondition`, per-frame separation and `N_eff`); to be run on the Arm-1 `H0` |
| V2.2 | Local-neutrality ladder gate with the species-pattern negative test | done — `ladder.local_neutrality_gate` (fixed carrier on a flanking Pb, `q0` vs species pattern centred/uncentred); kernel-level mechanism test in `test_kernels`; to be run on the trained Arm-1 `H0` |
| V2.3 | Post-hoc `C_Q` calibration (interpolation-only 159-atom frames, SE, "base support at 159 unverified" flag) | done — `dscc/calibration.py` (`profile_c_q`); the current `s0(159) = +0.08 ± 0.09` already raises the flag |
| V2.4 | Training-time single-valuedness monitoring (per-epoch subsample, registered ceiling, failing frames dropped from the step with the fraction logged) | todo — trainer |

### 3.3 Phase 2 — training protocol (plan §6)
| # | Task | Status |
|---|---|---|
| P2.1 | Energy admission per fold and size: coverage table, `s0 ± SE` (out-of-fold base), `sQ`; stored before results | module done — `dscc/admission.py` (label-free `d` = flanking-Pb distance, coverage bins, `slope_with_se`, decision, record); tables built (`defect-perovskite/dscc_admission_tables.json`, `dscc_base_residuals.json`; 4 outer folds; out-of-fold null from `cf_base_f0..f3` on their `null_oof` frames, 298/298/298/297): **no size admitted under the registered defaults — C7** |
| P2.2 | Objective (v4.2): forces at every size, no energy term; strata weights on the force loss; `L_gap` on the Hamiltonian actually filled at `dq = 0` on the static pristine cell (D8); `C_Q` outside the loss (V2.3); Route B′ has no pattern regulariser (no learnable pattern) | todo |
| P2.2a | Static pristine cell: port the symmetrised site-median lattice construction into `dscc/` (D8); thermal-mean gap diagnostic | cell done — `dscc/static_cell.py` (global anchor-site registration + species-wise assignment + iterated median + group symmetrisation); on the 544 training pristine frames: 32 operations, cell [16.035, 16.076, 11.395] Å, thermal rms Cl 0.52 / Cs 0.45 / Pb 0.22 Å, symmetrisation shift 0.19 Å, last median pass 0.022 Å; saved `defect-perovskite/static_pristine_cell.json` (the dataset directory is a symlink outside the repository) (fingerprint cc78a367e345352d). Thermal-mean gap diagnostic: todo |
| P2.3 | Loss-path audit (±1 eV injection, masks, units, restore; no validation/test/tiling frame in `C_Q`) | todo |
| P2.4 | Fixed protocol frozen: splits, strata, weights, balance, six seeds, metrics, all §6/§7 thresholds | todo |
| P2.5 | Leak readout `d(J* + C_Q)/dd` per size | todo |

### 3.4 Phase 3 — arms (plan §7)
| # | Task | Status |
|---|---|---|
| P3.1 | Arm 1: full `H0` vs scalar-only control, `Phi = 0`, Route A; thresholds (i)–(iv) registered before opening | todo |
| P3.2 | Arm 2+3 (v4.1 layout): regime B — route {A, B} × coupling {LR-only; LR+U; full; λ=1 fixed} + Φ=0 per route = 10 configurations × 6 seeds; regime-A ablation: full coupling × 2 routes × 6 seeds, excluded from selection; thresholds registered; forecasts §2.10 checked | todo |
| P3.3 | Arm 4: matched-kernel and full-kernel F-SCC comparators; decision (1)–(6) | todo |

### 3.5 Phase 4 — ladder, sparse solver, benchmark (plan §8)
| # | Task | Status |
|---|---|---|
| P4.1 | Tiling-ladder generator and report (`E(+1) - E(0)` vs `1/L`, `K_LR_ii`, active states, `dq` spread, zero total force) | todo |
| P4.2 | Sparse path (CSR `H0`, SP2/LDL inertia, Chebyshev/LOBPCG, tail bounds, PME/FMM) | todo |
| P4.3 | 2x benchmark per the registered definition | todo |

### 3.6 Deletion sweep (plan §3, after the Phase 1 gates)
- [ ] Tier-1/Tier-2 constructor and `u_al` (`defect_composition`, `defect_rank`, `defect_constructor_cache`, `defect_seed`)
- [ ] static density, residual monopole, covariant registration (`defect_density`, registration in `defect_composition`)
- [ ] spectral windows, `r_+` matrix functions, channel normalisation, localisation switches, `rho_img` (`defect_windows`, `defect_frontier`, `defect_image`, `defect_spectral*`)
- [ ] canonical lift, branch/cut/tail certificates, `IsoOK` (`defect_lift`)
- [ ] separate `Phi_SF`/`Phi_img` objects (`defect_boundary`, `defect_image`)
- [ ] spectral-gauge record (`defect_gauge`)
- [ ] nuisance intercepts (`defect_objective.profile_intercepts`)
- [ ] energy-term registry (`defect_terms`)
- [ ] induced-polarisation stage; isolated-boundary outputs
- [ ] their tests, launcher flags and docs

## 4. History note (v8 programme, closed 2026-09-06)

Last v8 commit `06c04c8`; full old defect suite 689 passed. The s14a wave: s1/s3/s4/s6
finished before cancellation, s2 stopped on the §3.1 gauge guard (gap floor), s5 was killed
on cancellation. No analysis of these runs.
