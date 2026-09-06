# D-SCC v4 — implementation tracker

Normative document: `defect-perovskite/DSCC_PLAN_V4.md` (the user's plan v4, verbatim,
2026-09-06). It **overrules** `TRANSITION_PLAN_V8_SPEC.md`, `TRANSITION_PLAN_V8_1_ADDENDUM.md`
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
| C1 | Smearing of the labels: the old code records "doped's default ISMEAR = 0, SIGMA = 0.05 eV" as the labels' smearing (`defect_counting.py`). Plan §11 registers Gaussian 0.05 eV and asks for confirmation from the label paper's methods (Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025)). | **open** — proceeding with Gaussian 0.05 eV |
| C2 | Whether `E_gap = 2.40 eV` is the static-lattice or the thermal-average PBE gap (decides which cell the gap regulariser acts on, §6). `band_edges.json` only carries the symmetric ±1.2 eV placement. | **open** — Phase 0 does not need it; needed before Phase 2 |
| C3 | b3 GPU 6 fault: cold power cycle now, or leave until GPU waves are needed (Phase 3)? | **open** |

## 2. Registers

### 2.1 Registered values (plan §11)

| Quantity | Value | Source / status | Registered |
|---|---|---|---|
| `sigma_s` | 0.05 eV, Gaussian (`f = erfc(x)/2`, `R = -sigma_s sum exp(-x^2)/(2 sqrt(pi))`) | plan §1/§11; label-code convention | 2026-09-06 (plan) |
| `n0` | Cs 1, Pb 4, Cl 7 | plan §1 | 2026-09-06 (plan) |
| `eps_inf` | 4 | plan §1 (host input) | 2026-09-06 (plan) |
| `E_gap` | 2.40 eV | plan §1 (host input); convention C2 open | 2026-09-06 (plan) |
| Coulomb constant `C` | 14.399645 eV·Å | `defect_madelung.COULOMB_CONSTANT` | 2026-09-06 |
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
| `Z_max`, Route B L2 weight | from init magnitudes | plan §2.5 | to register before Phase 1 (Route B) |
| `s_tol`, `z`, `n_min`, coverage bin width, out-of-fold base protocol id | — | plan §6 | to register before Phase 2 |
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

## 3. Task list

Status: `todo` / `wip` / `done` / `blocked`.

### 3.0 Bookkeeping
| # | Task | Status |
|---|---|---|
| 0.1 | Cancel the v8 s14a wave; leave the v8 branch clean (`06c04c8`, 689 tests green) | done |
| 0.2 | Plan v4 saved verbatim; this tracker; memory updated | done |
| 0.3 | Report to the user: cancellation outcome, GPU 6 fault, C1–C3 | todo |

### 3.1 Phase 0 — scaffold (plan §4)
| # | Task | Gate | Status |
|---|---|---|---|
| P0.1 | Species table `n0`, state adapter (D2), `N_ref`, spin split; dataset assertion `Q == cell_charge` | counts match on every frame | todo |
| P0.2 | Data pipeline: geometry-state groups formed before the split; strata keys; size-grouped batches (D4) | no group split across folds | todo |
| P0.3 | `fill(H, N)`: Gaussian smearing, bisection `mu`, `P`, `F_band`, generalised entropy; matrix-function backward (reuse `_FermiDensityMatrix`) | `dF_band/dH_ab = P_ba` to 1e-10 | todo |
| P0.4 | `E_PBC` Ewald matrix of Gaussians with derivatives (D3); LES oracle test; tiling-ladder `K_LR_ii` → `-alpha_M/L` | independent of the splitting parameter to 1e-10 eV (E, F, stress); oracle agreement | todo |
| P0.5 | Regime A: `K_SR` (erf, `w_dir` C2 switch), `K_LR`; placement check over identified first-shell bonds; `m_sw` | placement floors satisfied on the training set | todo |
| P0.6 | Regime B: `K_SR` lattice sum with automatic image range; `K_LR`; `f_SR` | converged to 1e-10 eV; rewrapping-invariant | todo |
| P0.7 | `Gamma` (both regimes, `lambda_dir`, `U_eff` bounded), `Gamma_LR` (`r_g`/`r_split` cross Ewald), `Zbar` centring, `W` | splitting-parameter independence of `Gamma_LR` | todo |
| P0.8 | `H0`: reuse `SlaterKosterH` (SK, Harrison init, decay lengths, log modulation, centred scalar onsite); rank-1 `l=1` descriptor; geometric rank-2 `Q_i`; `r_cut ≤` base `r_max` asserted | invariance under translation / rotation / permutation / rewrapping with covariant derivatives; `Q_i = 0` at centrosymmetric sites | todo |
| P0.9 | Serialisation and checkpoint round trip | loaded checkpoint reproduces an uncached forward bit-for-bit | todo |

### 3.2 Phase 1 — D-SCC forward and derivatives (plan §5)
| # | Task | Status |
|---|---|---|
| P1.1 | Batched SCF loop (Anderson/Broyden, unmixed residual, `tol_q`/`tol_E`/`tol_c`, `n_max` cap + flag, `rho`) | todo |
| P1.2 | Energy (band form vs primary functional to 1e-9 eV), HF forces, autodiff stress; neutral short-circuit (D5) | todo |
| P1.3 | Route B switch (off by default); both regimes selectable | todo |
| P1.4 | Root-rule harness: initialisations (i)–(iii), `tol_root`, symmetry-equivalent collapse | todo |
| P1.5 | Gates: neutral null; `sum dq = Q` to 1e-12; gauge shift; FD forces {1e-2,1e-3,1e-4} Å to 1e-4 eV/Å and six strains; invariances to 1e-10 eV; stability / single-valuedness on every frame of both charge states; Route B centred/uncentred ladder test | todo |

### 3.3 Phase 2 — training protocol (plan §6)
| # | Task | Status |
|---|---|---|
| P2.1 | Energy admission per fold and size: coverage table, `s0 ± SE` (out-of-fold base), `sQ`; stored before results | todo |
| P2.2 | Objective: total-cell residuals, strata, `C_Q` profiling on admitted frames (closed form, quadratic loss), `L_gap` on the Hamiltonian actually filled at `dq = 0`, Route B L2 | todo |
| P2.3 | Loss-path audit (±1 eV injection, masks, units, restore; no validation/test/tiling frame in `C_Q`) | todo |
| P2.4 | Fixed protocol frozen: splits, strata, weights, balance, six seeds, metrics, all §6/§7 thresholds | todo |
| P2.5 | Leak readout `d(J* + C_Q)/dd` per size | todo |

### 3.4 Phase 3 — arms (plan §7)
| # | Task | Status |
|---|---|---|
| P3.1 | Arm 1: full `H0` vs scalar-only control, `Phi = 0`, Route A; thresholds (i)–(iv) registered before opening | todo |
| P3.2 | Arm 2+3: 18 configurations × 6 seeds; thresholds registered; forecasts §2.10 checked | todo |
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
