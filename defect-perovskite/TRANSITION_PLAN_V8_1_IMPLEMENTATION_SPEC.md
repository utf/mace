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
- **One session.** From 2026-09-06 this programme is worked by a single session. A second
  session previously owned WP2, the launchers and b3; it was stopped on consolidation and
  its work (all committed) is now owned here. See §8 for what came across and what is
  still open from it.
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
| 1.1 | Rename to `H_class` / `H_fix` / `H_{B,S}`; no `state_id`, policy name or formal charge reaches `H_fix` | add §3.1 | done — delivered by the stopped session; owned here |
| 1.2 | `defect_gauge.py`: `μ_g(θ)` from frozen pristine projector, rank-normalised, spin-summed | add §3.1 | done |
| 1.3 | Enforce registered orthonormal representation; `H − μ_g S` path for generalised eigenproblem; forbid `μ_g I` in nonorthogonal basis | add §3.1 | done |
| 1.4 | Subtract same `μ_g` from all aligned spectral edges; assert pristine projector keeps its separating gap | add §3.1 | done |
| 1.5 | Replace `tier1()` with the rank-certified gap verifier (5 acceptance conditions) | add §3.2 | done — wired into `build_class_table`; old `tier1()` deleted |
| 1.6 | Pristine-rank transport `M_pred(L2) = M_acc(L1) + [M_pris(L2) − M_pris(L1)]` | add §3.2 | done |
| 1.7 | `u_al` construction from independently ranked records + provenance hash; route to Tier 2 when uncertifiable | add §3.2 | done |
| 1.8 | Signed-excess counts `d_σ`, `n_e,σ`, `n_h,σ`, `q_F`, `Q_core`; delete additive updates | add §3.3 | done |
| 1.9 | `m_F(S)` computed and stored in the class/state record | add §3.3 | done |
| 1.10 | Require `Q_formal(S_ref) = 0` on the production path; nonzero reference retained as algebra-only test | add §3.3 | done (module; production wiring in 1.1) |
| 1.11 | Split constructor cache: Tier-1 verifier key vs Tier-2 anchor key; invariant vs target field partitions; field-wise cross-size predicate | add §3.5 | done — keys in `defect_constructor_cache.py`; anchors persisted and re-seeded in `build_class_table` (D12) |
| 1.12 | Migrate the existing 159-atom Tier-2 record into the anchor cache **only** if it reconstructs every key field and passes every Tier-2 gate | add §3.5 | done — migration implemented and gated on the §3.2 conditions (`anchors_from_table`); supersedes D9 |
| 1.13 | Fix `q_raw` definition and its tests | add §4.1 | done — shorthand shown FALSE here, see D4 |
| 1.14 | Extend every result cache key with geometry/cell, canonical state, checkpoint, constructor, boundary+potential-zero, occupation/smearing, solver-regime and (when `G_∞` is called) lift/support fingerprints | add §3.5, §6.3 | done — delivered by the stopped session; owned here |
| 1.15 | Tests: electron↔hole crossings; gauge shift invariance (`H → H + aI`); Tier-1 routing test asserting no Tier-2 eigensolve after Tier-1 passes | add §11.1 | done — routing test `TestAnchorReuseAndRouting` (tier2 monkeypatched to raise) |

### WP2 — Stage 1 energy-zero correction (blocking before any Stage-1 result is interpreted)

| id | item | source | status |
|---|---|---|---|
| 2.1 | Loss-path audit item 1: perturb the implemented 159-atom constant by ±1 eV; compare autodiff `∂L/∂c` with the analytic derivative | add §8 | done — stage1_audit.py / STAGE1_V8_1_AUDIT.md |
| 2.2 | Audit item 2: trace all 16 retained charged energies through masks, indexing, units, reduction, weighting, optimiser groups, clipping, dtype, restore | add §8 | done — STAGE1_V8_1_AUDIT.md |
| 2.3 | Audit item 3: deterministic optimiser replay accounting for the observed 0.03 eV motion, or identify the detach/overwrite/frozen-parameter event | add §8 | done — optimiser replay recorded in the audit |
| 2.4 | Audit item 4: analytic profiler recovers an injected constant offset to the numerical floor | add §8 | done — profiler recovers an injected offset |
| 2.5 | Audit item 5: recompute diagnostic `c_g*` for every saved seed from training frames only; non-energy predictions must stay bit-identical | add §8 | done — stage1_recalibrate.py over the saved seeds |
| 2.6 | `defect_objective.py`: total-cell-eV residual `r_i`, paired `r_i^Δ`, single registered path `ξ_i` per observation | add §8 | done — defect_objective.py |
| 2.7 | Stratum keys (label provenance, host, formal charge, composition hash, cell convention, size/shape class) with frozen weights `W_g` | add §8 | done — defect_objective.py |
| 2.8 | Within-stratum shape loss + exactly equivalent weighted pair form; unbiased pair sampler | add §8 | done — centred and pair forms, registered sampler |
| 2.9 | Group geometry-paired states **before** the split; assert no group spans splits | add §8, §11.1 | done — collater stamps pair_slots; groups precede the split |
| 2.10 | Retire the same-size-neutral-null admission rule; keep neutral-null availability as a diagnostic | add §8 | done |
| 2.11 | Stage separation test: Stages 1–4 contain no production `C_Q`; Stages 5–6 contain no nuisance `c_g*` | add §11.1 | done — stage-separation test |
| 2.12 | Freeze energy scale, strata, weights, force/energy balance, pair rule and tolerances **before** opening corrected retraining results | add §8 | done — pre-registered manifest (8eb00f0) |
| 2.13 | Retrain the Stage-1 reference under the corrected gauge and objective (wave-packed) | add §8 | RUNNING — 6 seeds, b3 GPUs 4-7, 2/GPU (s14a_s1..s6); all six past epoch 4 at 11:21, ~25 min/epoch, 24 epochs |

### WP3 — Stage 2

| id | item | source | status |
|---|---|---|---|
| 3.1 | Exhaustive routing A / B / C / D, including the criterion-1-only failure (outcome D) | add §8 | done — `defect_routing.route` (total on the 16 verdict vectors; enumerated by test), `Stage2Router` (outcome C's single registered bound release, then stop), driver `stage2_route.py` |
| 3.2 | Inherit gauge-fixed Hamiltonian and total-eV objective; forbid per-size constants | add §8 | done — `assert_inherited_contract` checked by `run_train` whenever the v8.1 objective is on (gauge on, total-eV shape term, no per-atom total term, no per-class c, no null file, the Stage-1 scalar-range arm); recipe `stage23_v81_run` refuses the two retired knobs |

### WP4 — Stage 4

| id | item | source | status |
|---|---|---|---|
| 4.1 | Smooth `g_res`: `a_i`, `ω_i` with `ε_Z`, `ε_ω`, `λ_d d_i`; `O(1/N_at)` fallback; finite derivatives as `δZ_i → 0` | add §4.1 | done |
| 4.2 | Support gate: nonzero `|Q_core − q_raw|` with sub-threshold departure signal ⇒ unsupported, not a silent monopole | add §4.1 | done — enforced in `frame_static_densities(lift=True)`, the forward's path (raises `UnsupportedStateError`) |
| 4.3 | Covariant registration (co-translation, co-rotation, wrap, permutation, affine strain); discrete correspondence fixed per class | add §4.1 | done — five covariances pass; the alignment is differentiated (implicit step) and the correspondence is frozen per class, stored by site and transported by lattice translation; see D11, D16 |
| 4.4 | `defect_lift.py`: constructor-topology branch envelope `ζ_lift`, circular moment, cut placement, integer image assignment fixed w.r.t. `P` | add §4.2 | done — `defect_lift.py`, constructor-topology envelope + circular moment |
| 4.5 | Boundary-clearance and tail contract with certified `ε_ρ`, `ε_E`, `ε_F`, `ε_σ` bounds | add §4.2 | done — `clearance_report`, absolute buffer mass |
| 4.6 | `IsoOK` conjunctive predicate (6 clauses) with per-clause negative tests | add §4.2 | done — `iso_ok`, 8 clauses with per-clause negative tests |
| 4.7 | Forward-only two-boundary diagnostic; retain `dP⁽⁰⁾/d(R,h)`; do **not** feed `δΦ/δP` back into `H` | add §8 Stage 4 | done — `image_functional="unified"`: `defect_image.py` (Φ_SF^{∞,LR}, Φ_img^B on the lift) + `defect_boundary.stage4_terms`; force and strain FD pass for `static_frontier`, `image`, `assembled` under both boundaries; see D13–D15 |
| 4.8 | Remove any independent static image potential or pairwise image patch | add §8 Stage 4 | done — the unified regime REPLACES `Φ_FF` (registry: base/band/static_frontier/image) and refuses the Madelung shift inside `H` (D13); no static image potential existed to remove (v8's `G_img ⋆ ρ_static^def` was never built) |
| 4.9 | `m_F ≤ 1` guard wired on every nontrivial charge-head path, requested and reference state, before any energy | add §3.4, §8 Stage 4 | done — `stage4_terms` (`assert_supported_multiplicity` on both states); test `TestGates` |
| 4.10 | Affine-reference strain and thermal-background tests | add §8 Stage 4 | done — `strain_check` on the unified terms (both boundaries); thermal pristine background reported in the cut buffers and refused by the lift (`test_stage4_boundary.py`) |

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
2026-09-06 10:00  : after the cold power cycle all of 4-7 are back (GPU 6 included);
                    capacity 4 GPUs x 2 = 8. s14a_s1..s6 occupy 4,5 (two each) and 6,7.
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

### D4 — the v8 `q_raw = sum_i Z_i` shorthand is false in this codebase

The addendum asked only that the pristine integral be *declared* rather than assumed.
Declaring it showed it is not zero: the tiled pristine baselines integrate to about
**+0.65 e** on the V_Cl frames. So `sum_i Z_i` and `q_raw` differ by of order one electron,
and any code reaching for the present-atom sum alone is wrong by that much. The
implementation already used the difference of integrals, so nothing numerical changed; the
test now asserts the pristine integral is **non**-zero so the shorthand cannot quietly
become true and then quietly break again.

### D5 — cross-size cache reuse is field-wise, never a single hash

A single fingerprint over the whole constructor record makes every larger cell a cache
miss, which defeats the transport equation. `compatible_across_sizes` therefore requires
the invariant partition to match exactly while the target partition need only satisfy the
registered *relations* (same composition difference, pristine rank scaling with the tiling
volume). Exact geometry/cell/composition hashes are expected to differ between sizes and
are never compared for equality.

### D6 — the homology signature must be taken against the TILED pristine

`homology_signature` first compared a class's composition against the *untiled* pristine
key, which made the signature grow with the cell: a single Cl vacancy in a 2x cell read as
`17+23,55+8,82+8` instead of `17-1`. No two sizes of one family would ever have matched, so
the transport equation could never have fired and every class would have run Tier 2 forever
— the failure would have looked like "the verifier never accepts" rather than like a bug in
the signature. It now scales the pristine counts by the tiling volume.

### D7 — records must be free of non-finite floats

The class table is compared for equality after a pickle round-trip and after config
extraction and rebuild. Those build fresh objects, and `NaN != NaN`, so one non-finite
number anywhere in a record makes the table unequal to itself — presenting as a diff
between two dicts that print identically. `_serialisable` maps non-finite floats to `None`
("not measured"), and `nearest` is now always the finite consecutive gap at the accepted
rank, for either tier.

### D8 — the first size of a family pays for the continuation, later sizes do not

A consequence of the verifier worth stating, because it changes observed routing: with no
prior anchor, the smallest class of a homologous family runs Tier 2, and every other size
is then verified at Tier 1 by transport. Previously all sizes ran Tier 1's VBM-proximity
test. Several tests asserted the old outcome and have been retargeted to assert the
integers (which must not move) and the routing (which now must differ between the first
size and the rest).

### D9 — the 159-atom Tier-2 migration is declined

The record exists, but not as a raw log: it is the `ClassRecord` inside a saved model's
class table (`composition_classes['classes'][...]`, with table-level `e_sink`, `eta`,
`dlambda`, `r_match`, `delta`, `window`, `quantiles`), and the numerical regime was numpy
float64 `eigh` on CPU at code 7e155c9.

The addendum permits migration without rerunning "only if its stored raw log reconstructs
every invariant and target-specific key field, **including the exact geometry/cell** and
numerical regime". It does not: `reference_frame_key` is a hash of the class's reference
frame, and a geometry is not reconstructible from a hash. Identifying the frame by
description ("the first 159-atom frame of `dataset_pbe/train.xyz` in file order") and then
verifying it against the hash would be defensible, but it buys nothing here: the Tier-2
continuation costs about 30 s of CPU, so the anchor cache saves no meaningful work, and
enshrining an anchor whose provenance does not fully reconstruct is the larger risk. The
cache exists to avoid expensive recomputation; this recomputation is not expensive.

Decision, agreed with the parallel session: **declined**. The class takes the ordinary
Tier-2 route, which is the addendum's own stated default when a record fails any acceptance
condition. No anchor is enshrined and no provenance is invented.

**Stage 0 is complete** with 1.1 and 1.14 delivered by the parallel session.

---

## 8. Consolidation (2026-09-06)

A second session implemented WP2, the launchers and everything running on b3 under an agreed
split. It has been stopped and this session now owns all of it. Its worktree was clean and
the shared stash empty at handover, so nothing was lost; every item below is committed.

**Came across, and now owned here:**

| Area | Files |
|---|---|
| Stage 1 objective | `mace/modules/defect_objective.py`, `tests/extensions/defect/test_energy_objective.py` |
| Loss / trainer wiring | the objective's hooks in the loss and `mace/cli/run_train.py` |
| Gauge wiring into the head | `defect_counting.py`, `defect_models.py`, `defect_protocol.py`, `tests/extensions/defect/test_gauge_wiring.py` |
| Cache keys (1.14) | `mace/modules/defect_cache.py` — `model_fingerprint` / `result_key` |
| Audit and recalibration | `defect-perovskite/stage1_audit.py`, `stage1_recalibrate.py`, `STAGE1_V8_1_AUDIT.md` |
| Launchers | `defect-perovskite/queue_packed.sh`, `stage_b_recipe.sh`, `queue_stage1_audit.sh` |

**Open threads inherited, not yet closed:**

1. **The v6 golden compare under the gauge was never reported.** The stopped session said it
   was re-running it (`--stage12` allow-list) and would report whether the Tier-1 routing
   change altered the golden integers on the golden frames. It went quiet after `c7c0e34`
   without reporting. **Treat this as unrun**, not as passed: it must be run here before any
   Stage-1 number is read.
2. Three of its bash pollers had been looping `until ! ssh b3 pgrep -f queue_protocol_smoke`
   since 2026-09-02, waiting on a finished job. Killed on consolidation.
3. Its WP2 items keep their tracker status; ownership is now here.

---

## 9. The golden routing thread: CLOSED (2026-09-06)

The inherited open thread -- does the v8.1 Tier-1 routing change (`06f4fd1`) alter the class
integers? -- is now answered directly rather than by inference.

`stage0_routing_ab.py` builds the class table on the golden frames and records the integers.
The post-change capture is at HEAD; the pre-change side is the committed acceptance record
`golden/stage0_acceptance.json` (`d0561b0`, 2026-09-05 00:48), which predates the routing
change (2026-09-06 02:17). Both are stored as `golden/stage0_routing_{pre,post}.json`.

```
routing moved (expected): 17x47,55x16,82x16  tier 1 -> 2
INTEGERS UNCHANGED across the routing change: 3 classes,
fields compared: m_vb, n_e, n_h, n_sigma, q_core
```

Every integer is identical on all three classes: the 79-atom V_Cl (`m_vb [204,204]`,
`n_e [1,0]`, `q_core +1`), the pristine 80-atom cell (`[208,208]`, `q_core 0`) and the
159-atom V_Cl (`[412,412]`, `q_core +1`). Only `tier` moved, and only on the 79-atom class,
which is the designed behaviour: it is the first size of its homologous family, so it now
runs the continuation while the 159-atom class is verified by transport --
`412 = 204 + (416 - 208)`, the pristine-rank increment.

Two notes on method, because the first two attempts were wrong:

* Re-running the *pre-change tree* was abandoned: at `06f4fd1^` the mixed-precision dtype
  bug (fixed later in `0f86246`) makes `arma_s1` fail before the table is built. The
  committed acceptance record is the better instrument anyway -- it is the artefact of
  record, not a reconstruction.
* The comparator initially reported nine "changed" integers. All nine were fields absent
  from the pre record (`d_sigma`, `m_f`, `ambiguous`) -- schema additions from this
  programme, not value changes. It now separates "new since the pre record" from "changed",
  because conflating them would hide a real difference among the additions.

### D10 — the unwrap is a minimum image about the support centre, not an offset from the cut

Written first as "shift everything whose fractional offset from the cut exceeds the
centre's", which splits a compact object across two images: a site just past the centre is
more than half a cell from the cut and was shifted while its neighbour was not. The
translation-continuity test caught it -- a rigid pair 0.48 A apart reported a separation of
11.52 A, exactly one cell minus 0.48. The rule is `shift = -round(s - centre)`, which is the
same statement as "cut half a cell away" but cannot separate neighbours.

The lesson generalises: a branch rule should be expressed relative to the object being kept
together, not relative to the boundary being avoided.

### D11 — rigid-translation covariance: RESOLVED by re-fitting the placement per frame

Addendum 4.1: "Under a rigid translation or rotation, both densities co-transform
identically." The implementation does **not** satisfy this for translation, and it is not an
oversight on either side:

* the pristine reference is placed by a fractional shift found **once per composition class**
  by minimising `||rho_static^raw||^2` on the class reference frame, then reused by every
  frame of the class;
* that is deliberate. `defect_density`'s own docstring says the per-frame norm is reported
  "so a frame recorded from a different origin is visible rather than silently a dipole
  array". A cached placement is what makes a wrong-origin frame detectable.

Measured: translating every atom of a V_Cl frame by a constant takes `||rho_static^raw||`
from 0.144 to 1.406 -- a 10x jump for a transformation that is a symmetry of a periodic
system. Rotation, periodic wrap, atom permutation and homogeneous strain all pass, because
each carries the cell with it and the placement is fractional.

**Resolution (user decision): do what the addendum asks.** `frame_static_densities` now
re-fits the alignment per frame, seeded by the cached class shift, and all five covariances
pass. Two things were tried and rejected on measurement first:

1. **An origin proxy** -- correct the cached shift by the circular centroid of the present
   atoms. It fails on bulk-like cells: the atoms tile the cell, so the circular mean has
   near-zero magnitude and its angle is noise. It broke two pristine tests, and it is the
   same degeneracy the isolated lift guards against with `z_min`.
2. **A local re-minimisation** from the cached shift (damped Newton, then L-BFGS with a
   strong-Wolfe line search). Better but still wrong: a translation of a fraction of a
   lattice spacing crosses into a neighbouring basin, and the local search converged to
   0.72 against the correct 0.14.

What works is the **global** candidate search that `align_pristine` already implements,
anchored on present atom 0 against every pristine site -- no species map needed, since
wrong-species candidates simply rank out. The minimiser of `||rho_static^raw||^2`
co-translates exactly with the atoms, so this is covariant by construction rather than by
approximation.

The cost is a per-frame alignment rather than a cached constant, which the addendum
anticipates ("any continuous geometry-dependent alignment is fully differentiated"). The
wrong-origin diagnostic is not lost: `||rho_static^raw||` is still reported per frame, and
now measures a genuine mismatch rather than a bookkeeping offset.

### D12 — the anchor registry must be persisted, or the continuation is not a one-off

Found by a question that was exactly right: *why did any CsPbCl3 class enter Tier 2 at all,
when it is not mixed valence?* Tier 2 exists for genuinely ambiguous classes, and a Cl
vacancy is not one. Seeing it there meant the routing had failed to find an anchor, not that
the class was ambiguous.

Three defects, all in what this session wrote:

1. **Anchors were never persisted.** The registry was a local list rebuilt inside
   `build_class_table`, so every fresh build started with no anchor and the first class of
   each family re-ran the continuation. §3.5's whole purpose is that the expensive Tier-2
   record is cached and the cheap verifier reuses it.
2. **`defect_constructor_cache.py` was imported nowhere.** The keys and the field-wise
   cross-size predicate were written and tested, and never wired in -- the same error as
   marking Tier 1 done while `build_class_table` still called the old VBM test. Writing a
   module and its tests is not the same as the system using it.
3. **Ordering forced the smallest class to continue.** Classes were sorted by size, so the
   79-atom class always ran the continuation even when a larger accepted anchor existed.
   The addendum expects transport to run *backward* from the 159-atom anchor.

Fixed: `anchors_from_table` seeds the registry from a previous build's stored anchors and
from accepted Tier-2 class records (the §3.5 migration, gated on §3.2's conditions -- both
paths agreed, both schedules agreed, no closure eigenvalue; `Q_core` alone is explicitly
insufficient because compensating per-spin rank errors give the same total). Accepted
anchors are written to `table["anchors"]`. Ordering is by anchor availability, not size.
Only ranks established independently of a verifier -- exact-by-count or Tier-2-accepted --
may seed, so a family cannot bootstrap from its own verifier.

`verify_class_table` no longer compares `tier`. It is a routing diagnostic: after the anchor
exists a rebuild verifies by transport where the first build continued, so `tier` moves
2 -> 1 while every integer is unchanged. Comparing it reported the cache *working* as a
failure.

**Measured effect on a cold build** (79/159-atom classes, `arma_s1`):

```
cold (no anchor): 9.93 s   tiers [1, 1, 2]
warm (anchored) : 4.61 s   tiers [1, 1, 1]        2.2x, integers identical
```

The instrumented test the addendum names is now present: `tier2` is monkeypatched to raise,
and a rebuild with a seeded anchor completes without calling it.

### D13 — under the unified regime `H_fix = H_local`: the Madelung shift leaves `H`

Addendum §8 Stage 4 says "keep `H_fix = H_local`", §3.1 defines `H_fix` as the local
Slater–Koster plus accepted on-site and edge terms, and §6.1 assigns the long-range
static–frontier interaction to the *explicit* functional `Φ_SF^{∞,LR}[P] = B_{K_∞^LR}[ρ_S,
ρ_F[P]]`, whose Hamiltonian contribution `V_SF = δΦ_SF/δP` is activated only at Stage 5 ("no
unsigned scalar field is inserted directly into an electron Hamiltonian"; §13: "Stage 4 is
forward-only"). The legacy Madelung on-site shift is that same interaction as a potential
inside `H`. Keeping it while also evaluating `Φ_SF[P⁽⁰⁾]` counts the defect-induced
long-range S–F piece twice, which §6.1 forbids ("no separate image potential or image energy
may duplicate it").

Decision: `image_functional="unified"` **requires** `madelung_range="off"`; the combination
is refused at construction and on checkpoint load, with the double-count reason. The default
regime (`"frontier_ff"`) is untouched, so the Stage-1 reference and the runs in flight are
unaffected. The consequence for the Stage-4 arm is real and stated: `P⁽⁰⁾` is filled from a
Hamiltonian with no electrostatics in it, so the electrostatic site selection Edit 1 gave the
head is absent until Stage 5's stationary solve puts `V_B` into `H`. That is what the
addendum prescribes for Stage 4; it is a diagnostic stage, not a production model.

### D14 — the legacy `Φ_FF` is half of v8 §2.8

`LatentEwald.energy` is the energy `E = ½ qᵀAq` (`test_madelung_convention` pins
`E = ½ Σ q V`). v8 §2.8 defines `Φ_FF = ½ B_img[wρ, wρ] = E_PBC − E_∞`. The legacy code
writes `0.5 * (E_PBC − E_∞)` — a quarter of `B_img`, half the specified term. Left as it is:
the Stage-1 reference and s14a depend on its numerics, and changing it mid-flight would make
the corrected retrain incomparable with its pre-registration. Recorded here, in the
`defect_image` module docstring, and to be raised with the user; the unified term uses the
correct `E_PBC − E_∞`. Also: the legacy term calls `isolated_energy` on **wrapped**
positions, which §6.1 prohibits; the unified term lifts first.

### D15 — the periodic evaluator's `dl` is a numerical floor that the image term inherits

LES sums reciprocal space to `k_max = 2π/dl`. At its default `dl = 2` with `σ = r_res = 1`
the Gaussian weight at the cutoff is still `7e-3`, and because the isolated half of `K_img`
is exact the remainder lands in the image term undiminished: measured 0.5 % on a lone
charge's Makov–Payne energy, 2.4 % on a 64-site lattice (`TestConvergence`). With
`k_max σ ≥ 2π` the lone charge matches Makov–Payne to `1e-6`. The unified regime's
evaluator (`image_ewald`) uses `dl = min(dl_config, r_res)`; the legacy `frontier_ewald`
keeps LES's default so Stage-1 numerics are untouched. The 1.5 % "agreement" the Madelung
convention test tolerates against Makov–Payne is this truncation.

### D16 — the registration derivative, and the correspondence by site

`align_pristine` returned a detached minimiser, so every functional of the placed pristine
density had no registration response in its force (the FD harness would have reported it
as `missing_derivative`). One Newton step at the converged shift with the graph attached
gives the implicit-function derivative without moving the value; the refinement builds its
own graph under `enable_grad` because the harness's energy pass runs under `no_grad`.

The discrete correspondence is established once per class (`pristine_placement`) and stored
**by pristine site**, not by atom index, with the departure signal per site: the global
re-fit on a thermal frame may land on an equivalent representative (a pristine lattice
translation — measured on the 2×2×4 class, where the re-fit differs from the class shift by
(−½, −½, +¾) of the supercell, one 5.6 Å lattice vector), and an atom permutation relabels
the present atoms. `transport_correspondence` finds the one lattice translation carrying the
frozen site sets onto the frame's (tolerance `MERGE·r_res`, a physical distance, because the
class reference is itself thermal) and refuses anything else as a topology event. The
per-frame proximity rule is used only to find the representative and to verify — never to
re-establish — the correspondence.

### Cost note (Stage 4 arm)

`stage4_terms` re-fits the pristine placement (global candidate search, 8 candidates × 12
Newton steps on an `n × m` Gaussian kernel) for every off-reference graph on every forward.
Fine on the toy and for the diagnostic arm; measure the per-step cost on the 159-atom cells
before committing six seeds to it.
