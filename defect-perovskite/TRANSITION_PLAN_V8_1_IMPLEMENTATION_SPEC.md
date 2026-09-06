# v8.1 implementation spec and work tracker

Derived working document. **It is not a specification of record.** The two normative files are:

1. `TRANSITION_PLAN_V8_SPEC.md` — the v8 plan of record (received 4 Sep 2026);
2. `TRANSITION_PLAN_V8_1_ADDENDUM.md` — the v8.1 correction (received 6 Sep 2026), which
   takes precedence wherever the two conflict.

This file exists to translate those two into ordered, checkable work items against the actual
code, and to track progress. Where this file and either normative file disagree, the
normative files win and this file is wrong and must be corrected.

Created 2026-09-06. Stage order is unchanged: no stage starts until the previous stage's
acceptance tests pass.

---

## 0. Operating rules for this programme

- **Task tracking.** The harness task-list tools are unavailable in this session, so the
  checklist in §3 is the task list. Update the status column in the same commit as the work.
- **GPU waves.** b3 only, GPUs **4–7 only**, **up to two runs per GPU**. As of 2026-09-06
  GPU 6 reports `Unknown Error` from `nvidia-smi` and is unavailable, so today's usable
  capacity is GPUs 4, 5, 7 → **6 concurrent runs**. Every wave launcher must query
  availability at launch and pack to the live set rather than assuming four GPUs.
- **No analysis of the cancelled Stage 1.3 runs** until §3 WP2 (loss-path audit and
  saved-checkpoint recalibration) is complete. Their arm (a) and (b) seeds finished and arm
  (c) had started; all were cancelled on receipt of the addendum.
- **Regime tags.** Every trained number carries its regime tag. A number produced before a
  gauge, objective or functional change is not comparable with one produced after it.

---

## 1. What the addendum changes, in one table

| # | Change | Normative source | Blocking for |
|---|---|---|---|
| 1 | `H_0` retired; three named Hamiltonians `H_class`, `H_fix(R)`, `H_{B,S}[P,u]` | add §3.1 | Stage 0 |
| 2 | Frozen-pristine, rank-normalised **spectral gauge** `μ_g(θ)`; orthonormal representation mandatory | add §3.1 | Stage 0 |
| 3 | Tier 1 replaced by a **rank-certified gap verifier**; VBM-proximity test deleted | add §3.2 | Stage 0 |
| 4 | `u_al` built only from independently ranked records, with provenance; else route to Tier 2 | add §3.2 | Stage 0 |
| 5 | Counts from **signed excess** `d_σ = N_σ − M_VB,σ`; additive carrier updates deleted | add §3.3 | Stage 0 |
| 6 | `m_F ≤ 1` hard support guard on every nontrivial charge-head/boundary path | add §3.4 | Stage 4 |
| 7 | Constructor cache **split** into Tier-1 verifier key and Tier-2 anchor key, with invariant/target field partitions | add §3.5 | Stage 0 |
| 8 | `q_raw` defined as present-minus-pristine integral | add §4.1 | Stage 0 |
| 9 | Smooth, everywhere-defined `g_res`; size-vanishing uniform fallback; support gate | add §4.1 | Stage 4 |
| 10 | **Canonical isolated-space lift** `𝒰_h` and the conjunctive **`IsoOK`** predicate | add §4.2 | Stage 4 |
| 11 | Frontier density an **explicit functional of `P`**; compact-support spectral windows; exact-zero plateaus | add §5.1 | Stage 5 |
| 12 | Unified signed **image-active density** `ρ_img`, `q_img`, exact-plateau localisation switch `W` | add §5.2 | Stage 5 |
| 13 | Single image functional `Φ_img` containing S–S, S–F, F–F exactly once; no separate image potential | add §6.1 | Stage 5 |
| 14 | **Fixed-training-boundary reference anchor** total-energy formula | add §6.3 | Stage 5 |
| 15 | Five distinct **output fields** + structured support record | add §6.4 | Stage 5 |
| 16 | Stage 1 **energy-zero correction**: total-cell-eV residuals, stratum weights, within-stratum shape loss | add §8 | Stage 1 |
| 17 | Loss-path **audit** and saved-checkpoint recalibration before any Stage-1 interpretation | add §8 | Stage 1 |
| 18 | Stage 2 routing made **exhaustive** (outcome D) | add §8 | Stage 2 |
| 19 | Stage 6 screening **replaced** term by term, not augmented | add §7 | Stage 6 |

---

## 2. Code map — where each change lands

Paths relative to the worktree root `/home/alex/src/mace/.claude/worktrees/size-extensivity`.

| Area | File | Current state |
|---|---|---|
| State spec, policy dispatch | `mace/modules/defect_state.py` | `ElectronicStateSpec`, `CountFillPolicy`, `dispatch` exist |
| Composition constructor, tiers | `mace/modules/defect_composition.py` | `tier1()` is the **VBM-proximity** test to be replaced; `ClassRecord`, `class_integers`, `frame_counts` |
| Occupations, smearing, band free energy | `mace/modules/defect_counting.py` | `find_mu`, `fermi_fill`, `occupation_of`, `entropy_of` |
| Frontier channels and energy | `mace/modules/defect_frontier.py` | `channel_site_density`, `frontier_energy` |
| Static/frontier densities | `mace/modules/defect_density.py` | `static_raw`, `residual_shape`, `static_def`, `edge_projectors`, `channel_densities`, `participation_fraction` |
| Electrostatics | `mace/modules/defect_madelung.py` | `MadelungOnSite`, `site_potential` |
| Term registry, kernels | `mace/modules/defect_terms.py` | `Kernel`, `TermSpec`, `registry` |
| Caching | `mace/modules/defect_cache.py` | cache keys to be extended |
| FD harness | `mace/modules/defect_fd.py` | reused for §11.3 |

New modules required:

| New file | Purpose | Source |
|---|---|---|
| `mace/modules/defect_gauge.py` | `μ_g(θ)`, orthonormalisation map, gauge record | add §3.1 |
| `mace/modules/defect_lift.py` | canonical lift `𝒰_h`, branch anchor, clearance/tail bounds, `IsoOK` | add §4.2 |
| `mace/modules/defect_image.py` | `ρ_img`, `q_img`, `Φ_img`, `Φ_SF^{∞,LR}`, boundary kernels | add §5.2, §6.1 |
| `mace/modules/defect_boundary.py` | fixed-training-boundary anchor, five output fields, support record | add §6.3–6.4 |
| `mace/modules/defect_objective.py` | total-eV residuals, strata, shape loss, `c_g*`, `C_Q*` | add §8 |

---

## 3. Task list

Status: `todo` / `wip` / `done` / `blocked`. Keep this column current.

### WP0 — spec bookkeeping

| id | item | status |
|---|---|---|
| 0.1 | Record v8.1 addendum as a file | done |
| 0.2 | Cancel running Stage 1.3 jobs and pollers | done |
| 0.3 | Write this implementation spec / tracker | done |

### WP1 — Stage 0 corrections (adopt immediately; blocking for everything downstream)

| id | item | source | status |
|---|---|---|---|
| 1.1 | Rename to `H_class` / `H_fix` / `H_{B,S}`; no `state_id`, policy name or formal charge reaches `H_fix` | add §3.1 | todo |
| 1.2 | `defect_gauge.py`: `μ_g(θ)` from frozen pristine projector, rank-normalised, spin-summed | add §3.1 | done |
| 1.3 | Enforce registered orthonormal representation; `H − μ_g S` path for generalised eigenproblem; forbid `μ_g I` in nonorthogonal basis | add §3.1 | done |
| 1.4 | Subtract same `μ_g` from all aligned spectral edges; assert pristine projector keeps its separating gap | add §3.1 | done |
| 1.5 | Replace `tier1()` with the rank-certified gap verifier (5 acceptance conditions) | add §3.2 | done |
| 1.6 | Pristine-rank transport `M_pred(L2) = M_acc(L1) + [M_pris(L2) − M_pris(L1)]` | add §3.2 | done |
| 1.7 | `u_al` construction from independently ranked records + provenance hash; route to Tier 2 when uncertifiable | add §3.2 | done |
| 1.8 | Signed-excess counts `d_σ`, `n_e,σ`, `n_h,σ`, `q_F`, `Q_core`; delete additive updates | add §3.3 | done |
| 1.9 | `m_F(S)` computed and stored in the class/state record | add §3.3 | done |
| 1.10 | Require `Q_formal(S_ref) = 0` on the production path; nonzero reference retained as algebra-only test | add §3.3 | done (module; production wiring in 1.1) |
| 1.11 | Split constructor cache: Tier-1 verifier key vs Tier-2 anchor key; invariant vs target field partitions; field-wise cross-size predicate | add §3.5 | todo |
| 1.12 | Migrate the existing 159-atom Tier-2 record into the anchor cache **only** if it reconstructs every key field and passes every Tier-2 gate | add §3.5 | todo |
| 1.13 | Fix `q_raw` definition and its tests | add §4.1 | todo |
| 1.14 | Extend every result cache key with geometry/cell, canonical state, checkpoint, constructor, boundary+potential-zero, occupation/smearing, solver-regime and (when `G_∞` is called) lift/support fingerprints | add §3.5, §6.3 | todo |
| 1.15 | Tests: electron↔hole crossings; gauge shift invariance (`H → H + aI`); Tier-1 routing test asserting no Tier-2 eigensolve after Tier-1 passes | add §11.1 | wip (gauge shift + crossings done; routing test owed) |

### WP2 — Stage 1 energy-zero correction (blocking before any Stage-1 result is interpreted)

| id | item | source | status |
|---|---|---|---|
| 2.1 | Loss-path audit item 1: perturb the implemented 159-atom constant by ±1 eV; compare autodiff `∂L/∂c` with the analytic derivative | add §8 | todo |
| 2.2 | Audit item 2: trace all 16 retained charged energies through masks, indexing, units, reduction, weighting, optimiser groups, clipping, dtype, restore | add §8 | todo |
| 2.3 | Audit item 3: deterministic optimiser replay accounting for the observed 0.03 eV motion, or identify the detach/overwrite/frozen-parameter event | add §8 | todo |
| 2.4 | Audit item 4: analytic profiler recovers an injected constant offset to the numerical floor | add §8 | todo |
| 2.5 | Audit item 5: recompute diagnostic `c_g*` for every saved seed from training frames only; non-energy predictions must stay bit-identical | add §8 | todo |
| 2.6 | `defect_objective.py`: total-cell-eV residual `r_i`, paired `r_i^Δ`, single registered path `ξ_i` per observation | add §8 | todo |
| 2.7 | Stratum keys (label provenance, host, formal charge, composition hash, cell convention, size/shape class) with frozen weights `W_g` | add §8 | todo |
| 2.8 | Within-stratum shape loss + exactly equivalent weighted pair form; unbiased pair sampler | add §8 | todo |
| 2.9 | Group geometry-paired states **before** the split; assert no group spans splits | add §8, §11.1 | todo |
| 2.10 | Retire the same-size-neutral-null admission rule; keep neutral-null availability as a diagnostic | add §8 | todo |
| 2.11 | Stage separation test: Stages 1–4 contain no production `C_Q`; Stages 5–6 contain no nuisance `c_g*` | add §11.1 | todo |
| 2.12 | Freeze energy scale, strata, weights, force/energy balance, pair rule and tolerances **before** opening corrected retraining results | add §8 | todo |
| 2.13 | Retrain the Stage-1 reference under the corrected gauge and objective (wave-packed) | add §8 | todo |

### WP3 — Stage 2

| id | item | source | status |
|---|---|---|---|
| 3.1 | Exhaustive routing A / B / C / D, including the criterion-1-only failure (outcome D) | add §8 | todo |
| 3.2 | Inherit gauge-fixed Hamiltonian and total-eV objective; forbid per-size constants | add §8 | todo |

### WP4 — Stage 4

| id | item | source | status |
|---|---|---|---|
| 4.1 | Smooth `g_res`: `a_i`, `ω_i` with `ε_Z`, `ε_ω`, `λ_d d_i`; `O(1/N_at)` fallback; finite derivatives as `δZ_i → 0` | add §4.1 | todo |
| 4.2 | Support gate: nonzero `|Q_core − q_raw|` with sub-threshold departure signal ⇒ unsupported, not a silent monopole | add §4.1 | todo |
| 4.3 | Covariant registration (co-translation, co-rotation, wrap, permutation, affine strain); discrete correspondence fixed per class | add §4.1 | todo |
| 4.4 | `defect_lift.py`: constructor-topology branch envelope `ζ_lift`, circular moment, cut placement, integer image assignment fixed w.r.t. `P` | add §4.2 | todo |
| 4.5 | Boundary-clearance and tail contract with certified `ε_ρ`, `ε_E`, `ε_F`, `ε_σ` bounds | add §4.2 | todo |
| 4.6 | `IsoOK` conjunctive predicate (6 clauses) with per-clause negative tests | add §4.2 | todo |
| 4.7 | Forward-only two-boundary diagnostic; retain `dP⁽⁰⁾/d(R,h)`; do **not** feed `δΦ/δP` back into `H` | add §8 Stage 4 | todo |
| 4.8 | Remove any independent static image potential or pairwise image patch | add §8 Stage 4 | todo |

### WP5 — Stage 5

| id | item | source | status |
|---|---|---|---|
| 5.1 | Compact-support spectral windows `b_e`, `b_h` with exact-zero plateaus and `C²` quintic transitions | add §5.1 | todo |
| 5.2 | `P_V^fix` projector, `ΔP_σ`, `r_+`, `D_{e/h,σ}[P]`, `ρ̂_{c,σ}[P]` with trace bounds and leakage gate | add §5.1 | todo |
| 5.3 | Exact-plateau localisation switch `W`; absolute `N_loc/N_ext`, `R_loc/R_ext` thresholds | add §5.1 | todo |
| 5.4 | `ρ_img`, `q_img`; `Φ_img^PBC` single quadratic form; `Φ_SF^{∞,LR}` | add §5.2, §6.1 | todo |
| 5.5 | Stationarity on the **unmixed** fixed-point residual; multistart; competing-solution gap guard | add §6.2 | todo |
| 5.6 | Constrained energy-Hessian guards (`λ_min`, `κ`) in the registered tangent metric | add §6.2 | todo |
| 5.7 | Fixed-training-boundary anchor `E_B` with exact algebraic nulls in energy, force and stress | add §6.3 | todo |
| 5.8 | Five output fields + structured support record | add §6.4 | todo |
| 5.9 | Single profiled `C_Q*`; retire per-size `c`; exact reprofile before every gradient evaluation | add §8, §10 | todo |
| 5.10 | Ladder and FD gates in both boundaries | add §11.3, §11.4 | todo |

### WP6 — Stage 6 (conditional, last)

| id | item | source | status |
|---|---|---|---|
| 6.1 | Replace screened kernels by bare + induced term by term; no screened copy survives | add §7 | todo |
| 6.2 | Anisotropic `ε_∞` material frame, polar decomposition, clamped strain law | add §6.1 | todo |
| 6.3 | Polarisation eigenvalue / condition-number / scaled-residual guards | add §7 | todo |
| 6.4 | Recompute `C_Q*` after the functional change | add §7 | todo |

### WP7 — verification gates (§11)

| id | item | status |
|---|---|---|
| 7.1 | Algebraic and variational tests (§11.1) | todo |
| 7.2 | Boundary-reference identities (§11.2) | todo |
| 7.3 | Finite differences, all used functional values (§11.3) | todo |
| 7.4 | Size and support ladder (§11.4) | todo |
| 7.5 | Partition sensitivity (§11.5) | todo |

---

## 4. Sequencing

1. **WP1** in full. It is interface, identity and routing work; it does not need the GPU.
2. **WP2.1–2.5** (the audit) — also CPU-only, and blocking before any Stage-1 number is read.
3. **WP2.6–2.12** (objective rebuild), then **WP2.13** retrain as the first GPU wave.
4. WP3, then WP4, WP5, WP6, each behind its own gates.

WP1 and WP2.1–2.5 can proceed in parallel with no GPU contention.

---

## 5. GPU wave policy

```
allowed GPUs      : 4,5,6,7   (b3 only)
runs per GPU      : 2
live check        : nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
                    GPUs that error out of nvidia-smi are dropped from the wave
2026-09-06 status : GPU6 = Unknown Error -> capacity 3 GPUs x 2 = 6 concurrent runs
```

A wave launcher must: enumerate live allowed GPUs, assign at most two runs per GPU, pin each
run with `CUDA_VISIBLE_DEVICES`, and refuse to start if the requested run count exceeds
capacity rather than oversubscribing.

---

## 6. Blockers

### B1 — b3 has no usable CUDA (opened 2026-09-06)

`torch.cuda.is_available()` is **False** on b3 and `torch.cuda.device_count()` is **0**, for
every process, with `UserWarning: Can't initialize NVML`. `nvidia-smi` still lists GPUs
0-5 and 7 and only GPU 6 reports `Unknown Error`, so the fault is visible as one dead card
but its effect is node-wide: **no GPU training is possible at all**, and the packed-wave
policy of section 5 cannot be exercised until NVML recovers. Verified independently of the
peer session's report.

Consequence for the plan: every GPU-bound item (WP2.13 retrain, WP3, WP5 ladders) is
blocked. Every CPU-bound item -- all of WP1, and the WP2.1-2.5 loss-path audit -- proceeds
unaffected, which is the current critical path anyway.

---

## 7. Decisions recorded during implementation

### D1 — the Tier-1 `delta_search` inequality needs a boundary tolerance

Addendum section 3.2 requires `delta_search >= u_al + 2 s_smear`, and separately registers
`delta_search = 0.30 eV` with `s_smear = 0.05 eV`, "which requires `u_al <= 0.20 eV`".
Those numbers meet the inequality with **exact equality**:

```
u_al + 2 s_smear = 0.20 + 2 x 0.05 = 0.30 = delta_search
```

In binary floating point `0.20 + 2 * 0.05 = 0.30000000000000004 > 0.3`, so a naive `<`
comparison rejects the registered configuration itself. `defect_rank.verify` therefore
compares against `required_search - 1e-12 * max(1, |required_search|)`. The tolerance is a
numerical artefact allowance, not a loosening of the gate: `u_al = 0.21 eV` is still
refused (test `test_registered_delta_search_bounds_u_al`).

### D2 — "u_al too large" surfaces only as a `delta_search` violation

With the inequality enforced, the Tier-1 window

```
[E_v_al + u_al - delta_search,  E_v_al - u_al + Eg_host/2 + delta_search]
```

can never invert: its width is `Eg_host/2 + 2(delta_search - u_al) >= Eg_host/2 + 4 s_smear
> 0`. A defensive "the alignment interval swallows the window" branch was therefore
unreachable and has been removed rather than left as dead code. The operative form of the
addendum's `u_al <= 0.20 eV` rule is the `delta_search` inequality, which routes to Tier 2.

### D3 — `d_sigma` is derived from the record's own counts

`ClassRecord.d_sigma` is set from `n_e - n_h`, which is an identity (the two counts are the
positive and negative parts of one signed excess), rather than re-derived from
`(m_vb, n_sigma)`. Production builds `n_e/n_h` from `class_integers(m_vb, n_sigma)`, so the
chain is consistent end to end; deriving from `(m_vb, n_sigma)` instead would let a record
whose counts were set independently disagree with itself, and legacy records would restore
to different integers than they were written with.
