# Speed / corrected-forms / two-arms cycle — running results

Plan of record: `CYCLE_SPEED_ARMS_SPEC.md`, received 4 Sep 2026, executed in the order of
its section 8. Every number below carries its regime tag and appears in the order it landed,
including the ones that do not reach the report.

Base throughout: `aprime_prod`, frozen. Folds `aprime_f0..3`. No charged label touches the
base.

---

## §5.1 — the existing Stage B cohort's 79-atom head slope (F25, gate 9)

Regime: the six Stage B seeds (`stageb_s1..6`; head only on the frozen A′ production base,
γ = 3, Gaussian 0.05, exp envelope L0 = 1.0 with four learned decay lengths, log modulation
β = ln 1.5, centred on-site correction, E_LR from epoch 0 detached and frozen, charged
79-atom energies at full weight, 24 epochs, trunk f32 / head f64). Scored forward-only with
`b2_size_slopes.py`: 120 stratified 79-atom charged frames and all 16 large ones, matched
window, `d(delta_sr)/dd` and `dlambda_frontier/dd`.

| observable | 79 atoms | 159 atoms |
|---|---|---|
| `d(delta_sr)/dd`, Stage B (this cycle) | **−0.1804 ± 0.0244** | −0.1296 ± 0.0224 |
| `d(delta_sr)/dd`, s7 forces-only cohort (BATTERY_REPORT b2) | −0.1675 ± 0.0231 | −0.0629 ± 0.0087 |
| `dlambda_frontier/dd`, Stage B | +0.1885 ± 0.0240 | +0.1409 ± 0.0242 |
| `dlambda_frontier/dd`, s7 | +0.1846 ± 0.0223 | +0.0736 ± 0.0103 |

Per seed, 79 atoms: s1 −0.1611, s2 −0.1810, s3 −0.1901, s4 −0.1817, s5 −0.1926, s6 −0.1757
(`~/runs/stageb_b2.json` on b3).

**F25 clause 1 fails.** The forecast was that giving the head full-weight 79-atom charged
energies moved its small-cell slope *toward* the base's +0.37 artefact. It did not: the
slope moved from −0.1675 to −0.1804, i.e. slightly further from +0.37, and the seed spread
is unchanged (0.023 → 0.024). Gate 9's condition — negative, within 2× of −0.17, nowhere
near +0.37 — is already met by the existing cohort. Removing the 79-atom charged energies
(§0 rule 2) is therefore a precaution against a harm that is not measured here, not a repair
of one; it stays in the spec as written, and arm A's gate 9 is a regression test rather than
a remedy.

Read the two 159-atom numbers as the different statistics they are: **−0.1296 is the matched
window** on the 16 large frames, and F4's **−0.1004 is the full range** on the same frames
(STAGE_APRIME_REPORT §6.2). They are not comparable to each other, only each to itself
across arms.

## §5.2 — kNN distance in the cached first-block feature space

Deferred to after the arms are launched; it has no training role and the spec marks it
optional. Recorded here so its absence is a decision rather than an omission.

---

## §1.3 — where one training step's time goes (before)

Regime: the Stage-B configuration on `stagebsmoke_s1`, batch 8 of charged 79-atom frames,
local A4000, trunk f32 / head f64 with the base cache attached, `torch.profiler` over three
steps after a warm-up. Marks from `mace/modules/defect_profile.py`; script
`defect-perovskite/p1_profile_step.py`; raw `~/runs/p1_before.json`.

Seconds per step, CPU total (a region's total includes its children):

| region | before | after §1.1 |
|---|---|---|
| wall per step | 3.63 | **1.51** |
| step/forward | 1.742 | 0.793 |
| &nbsp;&nbsp;trunk (cached) | 0.038 | 0.040 |
| &nbsp;&nbsp;head/loop | 0.957 | **0.111** |
| &nbsp;&nbsp;&nbsp;&nbsp;head/assemble | 0.061 | 0.011 |
| &nbsp;&nbsp;&nbsp;&nbsp;head/eigh | 0.173 | 0.066 |
| &nbsp;&nbsp;&nbsp;&nbsp;head/fills | 0.567 | 0.016 |
| &nbsp;&nbsp;&nbsp;&nbsp;head/bisect | 0.670 | **0.015** |
| &nbsp;&nbsp;&nbsp;&nbsp;head/response | 0.134 | 0.002 |
| &nbsp;&nbsp;head/forces | 0.032 | 0.025 |
| &nbsp;&nbsp;ewald/madelung | 0.092 | 0.095 |
| &nbsp;&nbsp;ewald/lr | 0.082 | 0.080 |
| model/grad_corr | — | 0.046 |
| model/grad_corr_ref | — | 0.140 |
| model/outputs | — | 0.093 |
| step/backward | 0.147 | 0.077 |

Kernel launches per step 51 712 → 19 697; `_local_scalar_dense` 6 899 → ~0.

**The finding.** The step was not eigensolve-bound and not arithmetic-bound. `head/bisect`
— the Fermi bisection's host synchronisations, `float(total) > n` once per iteration per
fill per graph per pass — was 28% of the step's CPU time and the largest single region,
against 5% for `eigh`. The Stage A′ report's reading ("CPU-bound in the head's per-graph
eigensolve and Ewald loop") was right about the loop and wrong about which part of it.

---

## §1.1 — eigensolve batching by size group

Implemented as `SlaterKosterH.batched` (one `[B, 4n, 4n]` assembly), `find_mu` batched and
free of host traffic, and `CountingHead._batched_solve`. The four per-fill density matrices
collapse to one, `D = U diag(Σ_a s_a f_a) U^T`, which is an identity rather than an
approximation because the trace is linear in P and `D`'s diagonal is exactly what
`site_charges` wanted. The per-graph loop is kept as `_loop_solve` and still runs for any
batch the sampler did not group.

**Bit-identity (`tests/extensions/defect/test_batched_head.py`, 8 tests).** Batched against
per-graph, on random matrices and on the full model: energy, eigenvalues, P, the density
response, and the model's energy, forces, `delta_sr`, `alpha` and gap all agree to < 1e-8.
The response still carries a gradient (a detached one would train the wrong thing silently).

**Speed.** 3.63 → 1.51 s per step, **2.4×**, on the batch above.

---

## §1.2 — Ewald geometry precompute: a measured negative

Regime: `LatentEwald(sigma = 1.0)`, 8 graphs × 79 charges, float64, local A4000, mean of ten
calls after a warm-up (`$CLAUDE_JOB_DIR/tmp/ewald_split.py`).

| what | ms |
|---|---|
| energy value, no grad | 12.84 |
| dE/dq (φ), no `create_graph` | 33.39 |
| dE/dq with `create_graph` | 36.49 |
| dE/dq + d/dR of `q·φ` — what `site_potential` costs today | **52.02** |
| three energies + one d/dR — the polarisation-identity alternative | 73.57 |

A per-frame `A_per` of 79 atoms is 49 kB in float64 and 24 kB in float32; 2877 frames is
144 MB / 72 MB, so storage was never the obstacle.

**Why the store was not built.** A precomputed `A_per` removes the *value* of φ, which is
12.8 of the 52.0 ms the Madelung term costs. Every Ewald quantity in the step — φ in H,
E_LR, the arm-B image potential — needs a live position derivative, and a stored matrix
cannot supply one: reconstructing the force from the stored kernel means the cross energy
`q̃ᵀ A(R) Z`, which is the 73.6 ms route, *slower* than what is there. The spec's own
force clause ("the carrier-density field at the ions by one batched reciprocal-space pass")
is the part that survives, and it is a fusion of calls rather than a cache of kernels: at
12.8 ms for 632 charges the sum is launch-bound, not FLOP-bound.

Recorded as a scope decision from evidence: **the `A_per`/`A_iso` per-frame store and its
1e-8 drift guard were not built**, and the reason is the four numbers above.
