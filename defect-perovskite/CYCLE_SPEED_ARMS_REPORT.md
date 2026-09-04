# Speed, corrected head forms, a null-gated objective, and two arms — report

V_Cl⁺ in orthorhombic CsPbCl₃. Labels: Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025);
PBE scalar-relativistic, defect calculations set up through `doped`. Plan of record:
`CYCLE_SPEED_ARMS_SPEC.md`, received 4 Sep 2026, executed in the order of its section 8.
Running numbers in the order they landed: `CYCLE_SPEED_ARMS_RESULTS.md`.

**Every trained number carries its regime tag.** Base throughout: `aprime_prod`, frozen;
folds `aprime_f0..3` for the nulls; no charged label touches the base. Three regimes appear:
the **s7 head-only cohort** (frozen Stage-A base, γ = 3, Gaussian 0.05, forces + `loss_gap`,
60 epochs, float32), the **Stage B cohort** (head only on the A′ production base, log
modulation at ln 1.5, learned decay lengths, output-centred on-site correction, charged
79-atom energies at full weight, 24 epochs, trunk f32 / head f64), and **this cycle's arms**
(Stage B plus the section 2 forms, the null-gated objective and the size-grouped sampler).

---

## 1. Section 5 — the two readouts taken before anything changed

**§5.1, the Stage B cohort's own 79-atom head slope.** `d(δ_sr)/dd` on 120 stratified
charged 79-atom frames, matched window, six seeds: **−0.1804 ± 0.0244 eV/Å**, against the
s7 forces-only cohort's **−0.1675 ± 0.0231**. **F25's first clause fails**: giving the head
full-weight 79-atom charged energies did *not* move its small-cell slope toward the base's
+0.37 eV/Å artefact — it moved 0.013 eV/Å further away, with an unchanged seed spread. Gate
9's condition is therefore already met by the cohort the gate was written to catch, and
standing rule 2 is a precaution rather than a repair. That is worth stating plainly: the
rule stays, because a residual that means the base's extrapolation should not train the
head whether or not this particular cohort was damaged by it, but the damage was not
measured.

**§5.2, the feature-space kNN diagnostic (report only).** Per atom, the distance to the
nearest atom of 350 neutral training frames in the head's own 32-dimensional feature space;
per frame, the maximum over atoms; 100 **disjoint** neutral frames held out as the floor.

| population | n | median | p95 |
|---|---|---|---|
| neutral, held out | 100 | 0.1768 | 0.2921 |
| charged, 79 atoms | 400 | 0.2311 | 0.3500 |
| charged, 159 atoms | 16 | 0.0920 | 0.2207 |

The charged 79-atom frames sit 1.31× the neutral floor with overlapping p95s and a +0.27
correlation with d. **Two independent label-free detectors have now failed to see the
+0.37 eV/Å slope** — the fold-ensemble indicator `w_E` (F16, twice) and this distance. It is
not an out-of-distribution signature, and that is the argument for standing rule 2 being
framed as a statement about what a residual *means* rather than as a detector.

---

## 2. Section 1 — speed

### 2.1 The profile, and what it overturned

The Stage A′ report concluded from an operator table that the step was "CPU-bound in the
head's per-graph eigensolve and Ewald loop", and the spec proposed two remedies on that
reading. Instrumenting by REGION (`mace/modules/defect_profile.py`,
`defect-perovskite/p1_profile_step.py`) gave a different answer. Seconds of CPU total per
step, Stage-B configuration, batch 8 of charged 79-atom frames, local A4000:

| region | before | + batched solve | + reference skip |
|---|---|---|---|
| **wall per step** | **3.63** | **1.51** | **1.07** |
| step/forward | 1.742 | 0.793 | 0.611 |
| head/loop | 0.957 | 0.111 | 0.058 |
| head/**bisect** | **0.670** | 0.015 | 0.007 |
| head/eigh | 0.173 | 0.066 | 0.034 |
| ewald/madelung | 0.092 | 0.095 | 0.049 |
| ewald/lr | 0.082 | 0.080 | 0.055 |
| model/grad_corr_ref | — | 0.131 | **0.000** |
| step/backward | 0.147 | 0.077 | 0.084 |

`eigh` was **5%** of the step. The largest region, at **28%**, was the Fermi bisection — not
its arithmetic but its host synchronisations, `float(total) > n` once per iteration per fill
per graph per pass, about 6900 device-to-host round trips a step.

### 2.2 The two changes, and that both are identities

**The batched solve (§1.1).** `find_mu` runs a fixed iteration count computed once from the
initial bracket — the same stopping point the early exit found, decided in advance — and the
whole size-uniform batch is one `[B, 4n, 4n]` eigenproblem, one bisection over `[B, 4]`
fills, and one density matrix: the four per-fill matrices collapse to
`D = U diag(Σ_a s_a f_a) Uᵀ`, which is an identity because the trace is linear in P and
`D`'s diagonal is exactly what `site_charges` wanted. The per-graph loop is kept, is the
reference, and still runs for any batch the sampler did not group.
`test_batched_head.py` asserts the two agree to **1e-8** on eigenvalues, on P, on the density
response, and on the model's energy, forces, `δ_sr`, `α` and gap — in float64 and on a
float32 batch under the mixed-precision policy.

**The size-grouped sampler.** Grouping is by exact atom count; frames are permuted within a
group each epoch and the batches are then permuted **across** groups, so a large-cell step is
as likely early in the epoch as late; the short tail batch of each group is kept, because
`drop_last` would discard up to seven of the 34 large frames every epoch.

**The neutral reference branch (§1.1, not in the spec).** The model evaluates the correction
at the frame's counter and again at a reference counter and subtracts. When the reference is
neutral — every frame here — every term of that second branch is **exactly** zero: the
head's energy is a difference from that same fill so the occupations subtract exactly;
`q^pol` carries a counter factor and `q^carrier` carries signed counts, so the long-range
reference charge is `q_host` exactly and its energy difference is `E[0]`, with zero position
gradient because E is quadratic in q. A whole head pass, a Madelung potential, an Ewald
evaluation and a force gradient, computed every step to produce a zero.
`test_neutral_reference_skip.py` measures the claim in five long-range configurations on the
model's outputs **and on every trainable parameter's loss gradient**: agreement to 1e-12.

### 2.3 F24

| | Stage A′ | this cycle | gain |
|---|---|---|---|
| uncached step | 4.239 s | **1.143 s** | 3.71× |
| cached step | 3.451 s | **0.783 s** | **4.41×** |
| base cache build (2797 frames) | 338 s | **81 s** | 4.2× |
| cached epoch 0, wall | 8.4 min | **2 min 28 s** | **3.4×** |

**F24 (≥ 4× per epoch) fails on its own terms at 3.4×, while the step it named is 4.41×
faster.** The gap is the part of the epoch section 1 did not touch — the trainer's own train
and validation evaluation passes, which are forward-only and were never the loop's problem.
Scoring F24 on the step alone would be scoring it on the half that was optimised.

### 2.4 Section 1.2, a measured negative

`LatentEwald(σ = 1.0)`, 8 graphs × 79 charges, float64, mean of ten calls:

| what | ms |
|---|---|
| energy value, no grad | 12.84 |
| dE/dq (φ) with `create_graph` | 36.49 |
| dE/dq + d/dR of `q·φ` — what the Madelung term costs today | **52.02** |
| three energies + one d/dR — the polarisation-identity alternative | 73.57 |

A stored `A_per` removes the **value** of φ, 12.8 of those 52.0 ms. Every Ewald quantity in
the step needs a live position derivative, which a stored matrix cannot supply, and
reconstructing it from one is the 73.6 ms route — slower than what is there. **The per-frame
`A_per`/`A_iso` store and its 1e-8 drift guard were not built, and these four numbers are the
reason.** What survives of §1.2's wording is its force clause: at 12.8 ms for 632 charges the
sum is launch-bound, so fusing the surviving calls into one reciprocal-space pass is the
lever — worth about 8%, measured and not done.

### 2.5 One regression, caught by the smoke

The initialisation gate is scored on a pristine spectrum, and the trainer took it from the
first training batch. Under the size-grouped sampler the first batch is one size group, and
1985 of 2560 training frames are charged 79-atom cells — so the gate logged UNSCORED. **A
speed change had silently removed a gate.** The batch is now built from stoichiometric
frames; the surviving warning says "no batch in the training loader carries a stoichiometric
cell", which is a different and much louder condition.

---

## 3. Section 2 and the standing rules — the model as it now stands

| item | what it became |
|---|---|
| **2.1** on-site correction | `γ tanh(h(x_i) − h(x̄_s))`, the subtraction inside the tanh. The Stage A′ output-centred form is kept behind `counting_centre_form` so that cohort stays scorable. The test that separates them: push `h` past saturation with a constant, and the output form's channel is dead (\|∂corr/∂x\| < 1e-12) while the argument form's is untouched — which is what F10's 0-of-6 measured. |
| **2.2** per-site charges | `Z_i = Z0[s] + ζ tanh(z(x_i) − z(x̄_s))`, ζ = 1 e, with the deviation **centred per graph**. Without that centring `Σ_i δZ_i` is a learnable per-frame net charge, which moves the Ewald G = 0 constant frame by frame and breaks the c table's per-(charge, size) premise. Recorded as an interpretation of "neutrality projection on the per-cell sum". |
| **2.3** envelope anchor | `d_ref` removed from the constructor, the parser and the launcher. The envelope and the Harrison initialisation share one anchor, `r_cov(s) + r_cov(s′)` from Cordero's universal table, so a host cannot enter the head through two doors with two values. Models pickled with a scalar `d_ref` keep it — re-anchoring them would rescale every hopping they learned. |
| **2.4** E_LR | unchanged: density detached, branch frozen (host charges zero, polarisation off, amplitude 1/√ε∞), the density-sum invariant checked in the forward. |
| **2.5** image compensation | gains the validity switch `s = sigmoid((depth/δ_L − 2)/0.5)`, the differentiable form of the bound condition the dilution scorer has used since Stage 3, with δ_L collected from the pristine pass and carried in a buffer. Arm B only. |
| **rule 1** no per-host default | the c table's size class is `round(N/N_pristine)` and the large-cell threshold `1.5 N_pristine`, both reading the pristine count off the training set by composition; `--defect_madelung_eps_inf` has no default and the term refuses to run without it; scorers read ε∞ off the model rather than from a module constant. |
| **rule 2** null-gated energies | charged energies enter only for a size class whose out-of-fold neutral null brackets zero — 159 here (+0.024 [−0.070, +0.118] eV/Å), not 79 (+0.115 [+0.095, +0.135]). Forces untouched. `w_E` deleted: the code path, the flag, the launcher variable and the extractor key. |
| **rule 3** formation reference | `nulled_sizes_from` is the one place that decides what "has a neutral null" means, so the trainer and the scorers cannot disagree, and the largest nulled class is read from the file rather than written as 159. |

### 3.1 The interpretations recorded, each because the spec did not settle it

**The centre is the pristine ensemble, because there is no single pristine geometry.** All
544 stoichiometric training frames are 80-atom cells labelled `config_type=ideal`, and they
are thermal: the Pb–Cl first-shell spread runs 0.082–0.213 Å. There is no relaxed pristine
cell to centre on, so the centre stays the per-species mean over all of them.

**The per-species centre is not an atom-by-atom identity on this host.** Pnma CsPbCl₃ has
more than one Wyckoff site per species and the frames are thermal, so `corr_i` does not
vanish atom by atom on a real pristine cell. The identity is asserted where it holds (a cell
with one environment per species: residual 5.2e-18 eV) and the residual on the real host is
reported.

**Both forms of §2.5's constancy identity fail on this host, structurally.** On a simple
cubic Bravais lattice the periodic potential of equal charges is constant to 2.2e-16 eV — the
kernel is right. On the perovskite it spreads by 8.98e-02 eV, because equal charges on five
inequivalent sublattices are **not a uniform charge density**: the structure factor is
non-zero away from k = 0. The A′ spec asserted constancy of the difference; the speed cycle's
moved it to the periodic part; neither holds here. The term's adoption test remains the
tiling drift, which is a statement about size dependence and does not rest on this.

**The §3 energy share is not a target any more.** With the 79-atom charged energies zeroed,
the 159-atom share of the charged *energy* loss is 1.000 by construction. It is logged every
epoch and reads 1.000 for that stated reason; the number to compare across arms is the
charged-force share, which stays at its 0.25 target. `apply_size_upweight` refuses the
degenerate case the gate creates — with the small cells' energy mass at zero the old formula
returns a factor of **zero**, which would have deleted every charged energy from the loss
without a word.

### 3.2 The init gate under the new anchor

| | Stage B (2.861 Å) | arm A (covalent anchors) |
|---|---|---|
| edge spacing below / above | 0.249 / 0.018 eV | **0.233 / 0.009 eV** |
| bandwidth | 32.84 eV | **33.73 eV** |
| verdict | PASS | **PASS** |

The anchors are Cl–Cl 2.04, Cl–Pb 2.48, Cl–Cs 3.46, Pb–Pb 2.92, Cs–Pb 3.90, Cs–Cs 4.88 Å, so
Harrison's 1/d² rescales the pair initialisations by 1.97, 1.33, 0.68, 0.96, 0.54 and 0.34.
§2.3's "init gate unchanged" holds as measured.

---

## 4. The arms

### 4.1 The regime, which is one string for both arms but one difference

Six seeds each, launched 4 Sep 2026 on b3 GPUs 4–7, four at a time, two waves of four and
two. Head only on the frozen A′ production base (`aprime_prod`); no charged label touches the
base in either arm.

| | value |
|---|---|
| trunk | frozen `aprime_prod`, `BASE_LR_FACTOR=0`, base cache on, trunk f32 / head f64 |
| head | counting head, γ = 3.0, hop range 0.5, Gaussian smearing at t_el = 0.05 |
| centring | `DEFECT_ON_SITE_CENTRED=True`, **`CENTRE_FORM=argument`** — §2.1's corrected form |
| envelope | exp, L0 = 1.0 Å, four learned decay lengths at β_L = ln 2, anchored at r_cov(s) + r_cov(s′); **no `d_ref`** |
| hopping | log modulation, β = ln 1.5 (arms A and B); **ln 2 in arm C** and nothing else changes |
| charges | Madelung on-site, ε∞ = 4.0 passed explicitly, composition 3,1,1, Z init (−1, 1, 2), **ζ = 1.0 e per-site channel** |
| E_LR | on from epoch 0, density detached, branch frozen |
| objective | forces on every charged frame, large-cell share 0.25; **charged energies only for size classes with a neutral null**, share 0.25; no `w_E` |
| batching | **size-grouped**, batch 8, 24 epochs, lr 5e-3, eval every 4 |
| protocol | head-only, warm-up 5, c per (charge, size) |
| **the arm difference** | `DEFECT_IMAGE_COMPENSATION` — **False in A, True in B** |

The null file was rebuilt from `aprime_reference.json` at launch and read:

    79 atoms   +0.1147 [+0.0947, +0.1348] eV/Å   resolved, not nulled
    159 atoms  +0.0243 [−0.0697, +0.1184] eV/Å   NULLED
    charged energies will enter the loss for sizes [159]

so §0 rule 2 bites exactly where it was designed to: 159 keeps its charged energies, 79 keeps
only its forces.

### 4.2 What each arm cost

*(pending — wall clock per wave, epochs, and the realised shares)*

### 4.3 Arm A

*(pending)*

### 4.4 Arm B

*(pending)*

### 4.5 Arm C

*(pending — runs only if gate 7 fires on arm A, under the trigger pre-registered in
`CYCLE_SPEED_ARMS_RESULTS.md` before arm A's numbers existed: any integral type at its stop,
|tanh g| > 0.98, in more than 1/6 seeds)*

---

## 5. The gates

*(pending)*

---

## 6. The forecasts, scored

| | forecast | outcome |
|---|---|---|
| **F21** | arm A: F10 passes ≥ 4/6 | *pending* |
| **F22** | arm A: seeds at a pp stop fall by half relative to Stage B | *pending* |
| **F23** | arm B: Δc falls 0.2–0.4 eV; R inside [0.66, 1.34]; thermal tiling drift ≤ 0.3·D0; F4 holds | *pending* |
| **F24** | ≥ 4× per epoch from section 1 | **fails at 3.4× per epoch; the step it named is 4.41× faster** |
| **F25** | the Stage B cohort's 79-atom head slope has moved toward +0.37; arm A restores it | **clause 1 fails**: −0.1804 ± 0.0244 against s7's −0.1675 ± 0.0231, i.e. further from +0.37. Clause 2 is scored on arm A. |

---

## 7. What this leaves

*(pending — including the Δc origin decision)*

---

## 8. Operational record

### 8.1 What the smokes and tests caught that the loss would not have

Eight defects reached a running configuration and were caught before they reached a number.
All eight are silent failures: none raises, and each would have produced a plausible loss
curve.

| # | defect | how it would have shown up | caught by |
|---|---|---|---|
| 1 | `apply_size_upweight` returns a factor of **zero** once the null gate zeroes the small cells' energy mass | every charged energy silently deleted from the loss; training proceeds on forces alone and reports a fine RMSE_F | reading the formula for the case the new gate creates; now guarded, logged and covered by a test |
| 2 | `reference_state_data_loader` copies `train_loader.batch_size`, which is `None` under a `batch_sampler` | `batch_size=None` disables automatic batching; the collater is handed one `Data` object and raises `attribute name must be string, not 'int'` deep inside the mean/rms pass | the local one-epoch smoke |
| 3 | `zero_at_neutral_counts` and `batch_by_size` were instance attributes set in `__init__` | absent from every pickled checkpoint, so the reference skip and the batched solve silently never fire on a loaded model | the skip's own test, run against a saved-and-reloaded model |
| 4 | the reference-skip gate tested for the *key* `carrier_counts_ref` | `prepare_defect_configurations` writes that key onto every paired frame, so the gate never fired on real data | instrumenting the gate on a real batch rather than a constructed one; it now tests the *value* |
| 5 | the init gate reported UNSCORED | the size-grouped sampler makes the first batch a single size group, which need not contain a stoichiometric cell | the local smoke's log; the gate now searches the loader for a stoichiometric batch |
| 6 | `marks_table` reported all-zero CPU columns | one `record_function` key yields two `key_averages()` rows, a host annotation and a device row; the dict assignment overwrote the first | the numbers not summing to the wall time |
| 7 | `torch.utils.data.Sampler.__init__(None)` raises | `Sampler.__init__` takes no argument in this version | the sampler's unit tests |
| 8 | `c10_feature_knn`'s neutral floor was ~0 by construction | the neutral queries were inside the reference cloud, so every charged number looked enormous against a floor of zero | reading the number and disbelieving it; 100 neutral frames are now held out |

### 8.2 The regression check, and the transcription it caught

Scoring `stageb_s1.model` under this cycle's code reproduces its pre-cycle 79-atom head
slope to **1.1e-08**, and the batched solve, the per-graph loop and the reference-skip-off
path agree with each other to the same tolerance. The legacy pickle path — a head with no
`d_ref_pair` buffer and no `centre_form` attribute — is exact, so the Stage A′ and Stage B
cohorts remain directly comparable to this cycle's.

The check also caught a transcription error: the per-seed list under §5.1 of the results did
not match `stageb_b2.json`. It was replaced with the JSON's own numbers. Every aggregate
quoted against a gate had been read from the JSON and did not change. The lesson is narrow
and worth stating: *a per-seed list retyped from a log is not evidence; the JSON is.*

### 8.3 Machine discipline

- b3 GPUs **4–7 only**, at most four in use at once. Every stage of the chain launches four
  seeds and waits; the two-seed waves leave 6 and 7 idle rather than starting a fifth.
  GPU 3 has a recurring fault and 0–3 are off limits.
- The chain waits on **conditions** (the six model files exist) rather than on processes,
  which is the rule LEDGER entries 10 and 11 were written for.
- Code reaches b3 by `rsync`, never by git; `ssh b3 '...'` starts in `$HOME`, so every
  remote command goes through the `b3_run.sh` launcher with the checkout on `PYTHONPATH`.
- The local box has one A4000 and runs the smokes, the profiles and the forward-only
  regression scoring. `local_run.sh` is `b3_run.sh`'s mirror: without it a bare
  `python defect-perovskite/x.py` imports the *other* checkout's `mace`.

### 8.4 One thing the chain cannot do for itself

The running chain parsed its `gates()` body at launch, before the tiling-drift block was
added to the file. `gates armb` will therefore not run **gate 10**, and
`c3_tiling_drift.py --thermal 3` has to be launched by hand once arm B lands. Recorded here
rather than in a comment because it is the kind of thing that is discovered by its absence
from a table three days later.
