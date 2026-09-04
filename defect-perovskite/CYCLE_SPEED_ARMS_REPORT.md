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

### 4.2 What arm A cost

| | |
|---|---|
| wave 1 (seeds 1–4, GPUs 4–7) | 10:31:49 → 11:39:18, **67 min** |
| wave 2 (seeds 5–6, GPUs 4–5) | 11:39:18 → 12:45:45, **66 min** |
| gates (six scorers, four GPUs) | 12:45:45 → 13:05:15, **20 min** |
| per epoch | **2.2 min** at 24 epochs |

Wave 2 took as long as wave 1 with a third of the seeds because two of the four GPUs were
running wave-1 scoring in parallel — four GPUs in use throughout, never five.

Validation, per seed (meV/atom, meV/Å): 5.3/16.8, 6.9/16.8, 5.5/16.6, 5.6/16.5, 5.3/16.5,
5.5/16.4. Seed 2 is the outlier on energy and it is the same seed that is the outlier on
depth (0.817 eV from the CBM against 0.15–0.29 for the rest) and on Δc.

### 4.3 Arm A

The gate table is §5. The three results that matter:

**The size slopes converged.** `d(δ_sr)/dd` is −0.1212 ± 0.0268 at 79 atoms and
−0.1083 ± 0.0174 at 159 — a **0.013 eV/Å** gap, against 0.051 in Stage B and 0.105 in the
forces-only cohort. This is the closest the programme has come to a head whose size
dependence does not itself depend on size, and it is what §0's rule 2 and §1.1's grouped
batches were for.

**The on-site channel came alive and took the level with it.** `tanh h` is saturated on 100%
of Cl atoms, exactly as in Stage B, and the corrections are nonetheless ±0.13–0.51 eV because
the corrected form reads the difference of the arguments. Knockouts at fixed weights put the
level at 0.41 eV below the CBM as trained, 0.12 with the correction disabled, and **0.046
with both learned on-site channels disabled — Stage B's own value.** Stage B's shallow donor
was a dead channel, not good physics.

**The hub coupling got worse, not better.** Gate 7 fires on all four integral types
(5/6, 3/6, 2/6, 4/6 against Stage B's 3/6, 1/6, 2/6, 5/6). F22 forecast the pp stops would
halve; pp-σ's rate was unchanged and ss-σ's nearly doubled.

### 4.4 Arm B

Arm A plus §2.5's image compensation, and nothing else. Two waves, relaunched at 13:13 after
the ordering defect in §8.1b; wave 1 13:13 → 15:09, wave 2 → 17:07, gates → 18:06.
**3.89 min per epoch against arm A's 2.21 — the term costs +76%**, from a second head solve
and an Ewald image potential with a live position derivative, neither touched by §1.1's
batching.

Validation per seed (meV/atom, meV/Å): 5.8/16.6, 5.7/16.6, **8.8**/17.1, 5.7/16.6, 5.6/16.8,
5.6/16.7. Seed 3 is the outlier, as seed 2 was in arm A.

**What the term does, in one table.** Against arm A, six seeds each:

| | arm A | arm B | |
|---|---|---|---|
| gate 10, tiling drift | — (term off) | **0.135–0.173 of D0, 24/24** | the term's purpose, achieved |
| **gate 2, F4** | −0.0874 ± 0.0142, **5/6** | **−0.3112 ± 0.0151, 0/6** | **the term's cost** |
| gate 9, 79-atom slope | −0.1212 ± 0.0268, **PASS** | **−0.3422 ± 0.0249, FAIL** | same cause |
| gate 7, stops | 5/6, 3/6, 2/6, 4/6 | **6/6 on all four** | worse |
| gate 4, F10 | 1/6 | 0/6 | worse |
| gate 5, gap | 6/6 | 6/6 | unchanged |
| gate 6, R | 0.925 ± 0.144, 6/6 | 0.958 ± 0.090, 6/6 | unchanged, tighter |
| depth from the CBM | 0.334, shallow 5/6 | **0.234 ± 0.049, shallow 6/6** | **better** |
| `N_eff` | 18.79 ± **3.62** | 19.21 ± **0.83** | same mean, **4.4× tighter** |
| §2.2 max \|δZ\| | 0.585 ± 0.068 e | 0.663 ± 0.075 e | unchanged |

Three of those deserve naming.

**It removes the drift it was built to remove.** 24 of 24 model–tile pairs inside the gate,
at 0.135 of D0 on the ideal tile and 0.133–0.173 on the three thermal ones — the thermal
clause this cycle added, passed as comfortably as the ideal one.

**It triples both size slopes.** F4 goes −0.087 → −0.311 and the 79-atom matched slope
−0.121 → −0.342, so gate 9 fails too. The knockout in §7.4b attributes −0.2439 eV/Å of that
to the term itself: patch `image_potential` to zero on the same trained models and F4 returns
to −0.0761 ± 0.0029, inside the band on every seed. **The head is not damaged; the term's own
contribution at evaluation is too steep in `d`.**

**It stabilises what arm A destabilised.** `N_eff` spread falls from 3.62 to 0.83 at the same
mean, and the level comes back from 0.334 to 0.234 eV below the CBM with all six seeds
classified shallow. §2.1's revived on-site channel made the carrier's localisation
seed-dependent and the level deep; the image term undoes a good part of both. That is not
enough to adopt it — gate 2 is a gate — but it is the strongest argument in the cycle for
the prefactor experiment that §7.4b leaves open.

### 4.5 Arm C — triggered, and deliberately not run

Gate 7 fired on arm A's six seeds under the trigger pre-registered before those numbers
existed — any integral type at its stop (|tanh g| > 0.98) in more than 1/6 seeds — at ss-σ
5/6, sp-σ 3/6, pp-σ 2/6 and pp-π 4/6. By §4 that calls for arm C: arm A with the log
modulation widened to β = ln 2.

**It was not run.** The decision was taken after arm B's numbers were in and is recorded here
as a decision rather than an omission. The reasoning:

- The stop counts got monotonically **worse** across three regimes — Stage B (3/6, 1/6, 2/6,
  5/6) → arm A (5/6, 3/6, 2/6, 4/6) → arm B (4/4 on all four types). Every change this cycle
  made — the corrected on-site form, the per-site charges, the covalent anchors, the image
  term — pushed the head *harder* against the coupling bound, not less.
- Widening the bound from ln 1.5 to ln 2 is a 1.33× change in the modulation's range against
  a trend that four independent model edits have failed to reverse. It would most likely
  return "at the stop again, at a higher stop".
- §7.3's conclusion does not depend on it: the head asks for more hub coupling whatever it is
  given, and superexchange — a three-centre term, excluded by §9 — is the standing hypothesis
  for what it is asking for. Arm C would be the fourth cycle to answer that question with a
  bound rather than a term.

**What this costs the cycle, stated plainly.** Gate 7 is left unresolved for the third cycle,
and there is no measurement here of whether a wider bound helps. If the next cycle wants that
answer it is `ARM=c` in `queue_arms.sh` and two waves, roughly 2 h 20 on four GPUs; the
launcher is `queue_arms_chain_c.sh`, written and committed. What the cycle has instead is the
two hours spent on arm B's diagnosis — the F4 knockout that attributes −0.244 eV/Å to the
image term — which is a target, where arm C would most likely have produced another
observation of the same trend.

---

## 5. The gates

Assembled by `c12_gate_table.py` from the scorers' JSON rather than retyped from logs;
the script was validated by reproducing Stage A′'s published table line for line.

### Gates — arma

| gate | condition | measured | verdict |
|---|---|---|---|
| 1 regression | no criterion that held in Stage B now fails | c1_leakage 6/6; c1_not_shrunk 6/6; c2_neutral_toward_zero 0/6; c3_not_degraded 0/6; c4_f4_in_band 5/6; c6_shallow 5/6; c5_gap 6/6 | (read) |
| 2 F4 | slope in [-0.142, -0.063] in ≥ 4/6 | -0.0874 ± 0.0142; 5/6 in band (-0.0595, -0.1014, -0.0974, -0.0863, -0.0980, -0.0818) | **PASS** |
| 3 c consistency | report only | c(79) +9.7464 ± 0.1971, c(159) +10.7886 ± 0.3940, **Δc +1.0422 ± 0.1968**, predicted +0.0457 ± 0.0059 | (report) |
| 4 F10 | ligand-Cl − bulk-Cl > 50 meV and > 2σ, ≥ 4/6 | 1/6 seeds; difference +0.6514 ± 0.2660 eV | **FAIL** |
| 5 gap | pristine gap 2.4 ± 0.1 eV | +2.3788 ± 0.0190; 6/6 in window | **PASS** (pinned continuum and bandwidth: see the init-gate line) |
| 6 dilution | R inside [0.66, 1.34] | R +0.9249 ± 0.1442, 6/6 inside; bound fraction +0.6354 ± 0.0233; depth +0.0667 ± 0.0094 eV; δ_L +0.0172 ± 0.0010 eV | **PASS** |
| 7 stops / L_b | no type at its stop in more than 1/6 of seeds (1 of these 6) | ss_sigma 5/6; sp_sigma 3/6; pp_sigma 2/6; pp_pi 4/6; L_b ss_sigma 1.129±0.091, sp_sigma 1.272±0.028, pp_sigma 1.365±0.077, pp_pi 1.299±0.088 | **FAIL — arm C fires** (ss_sigma, sp_sigma, pp_sigma, pp_pi) |
| 8 participation | report; spread against the comparison cohorts | this arm +0.7538 ± 0.1874; head-only +0.4915 ± 0.0275; E_LR-off +0.4777 ± 0.0711 | (read) |
| 9 head slope | 79-atom d(δ_sr)/dd < 0, within 2× of -0.17, never near +0.37 | -0.1212 ± 0.0268; 5/6 seeds individually in [-0.340, -0.085] (-0.0792, -0.1553, -0.0961, -0.1319, -0.1183, -0.1466) | **PASS** (on the cohort mean) |
| 10 tiling (arm B) | ideal and thermal drift ≤ 0.3·D0; log the s distribution | **arma_tiling.json did not run** — `c3_tiling_drift.py --thermal 3` is launched by hand, the chain parsed the older gates() body | — |

**§2.2 per-site charge deviation** (report): max +0.5853 ± 0.0678 e, rms +0.0515 ± 0.0081 e, sites at the ζ stop +0.0000 ± 0.0000, worst |per-graph sum| 1.8e-15 e.

**Gate 1** is a read, not a boolean: criteria 2 and 3 are properties of the frozen base — the
neutral 159 slope and the neutral-79 window are the same numbers Stage B reported and fail
for the same reason — so they are not evidence about the arm. Of the criteria that are,
`c4_f4_in_band` fell from 6/6 to 5/6 and `c6_shallow` from 6/6 to 5/6.

**Gate 5**'s pinned continuum is a unit test rather than a run output, and the bandwidth comes
from each run's init-gate line: 0.233/0.009 eV edges and 33.73 eV on every seed, under the
covalent anchors that replaced `d_ref`.

### Gates — armb

| gate | condition | measured | verdict |
|---|---|---|---|
| 1 regression | no criterion that held in Stage B now fails | c1_leakage 6/6; c1_not_shrunk 6/6; c2_neutral_toward_zero 0/6; c3_not_degraded 0/6; c4_f4_in_band 0/6; c6_shallow 6/6; c5_gap 6/6 | (read) |
| 2 F4 | slope in [-0.142, -0.063] in ≥ 4/6 | -0.3112 ± 0.0151; 0/6 in band (-0.3330, -0.3203, -0.3058, -0.3210, -0.2993, -0.2880) | **FAIL** |
| 3 c consistency | report only | c(79) +10.3348 ± 0.3142, c(159) +11.2109 ± 0.6282, **Δc +0.8761 ± 0.3140**, predicted +0.0446 ± 0.0025 | (report) |
| 4 F10 | ligand-Cl − bulk-Cl > 50 meV and > 2σ, ≥ 4/6 | 0/6 seeds; difference +0.3541 ± 0.3518 eV | **FAIL** |
| 5 gap | pristine gap 2.4 ± 0.1 eV | +2.3915 ± 0.0125; 6/6 in window | **PASS** (pinned continuum and bandwidth: see the init-gate line) |
| 6 dilution | R inside [0.66, 1.34] | R +0.9579 ± 0.0899, 6/6 inside; bound fraction +0.6146 ± 0.0233; depth +0.0605 ± 0.0084 eV; δ_L +0.0179 ± 0.0019 eV | **PASS** |
| 7 stops / L_b | no type at its stop in more than 1/6 of seeds (1 of these 6) | ss_sigma 6/6; sp_sigma 6/6; pp_sigma 6/6; pp_pi 6/6; L_b ss_sigma 1.102±0.093, sp_sigma 1.232±0.022, pp_sigma 1.472±0.044, pp_pi 1.280±0.081 | **FAIL — arm C fires** (ss_sigma, sp_sigma, pp_sigma, pp_pi) |
| 8 participation | report; spread against the comparison cohorts | this arm +0.7818 ± 0.0975; head-only +0.4915 ± 0.0275; E_LR-off +0.4777 ± 0.0711 | (read) |
| 9 head slope | 79-atom d(δ_sr)/dd < 0, within 2× of -0.17, never near +0.37 | -0.3422 ± 0.0249; 3/6 seeds individually in [-0.340, -0.085] (-0.3191, -0.3221, -0.3926, -0.3454, -0.3461, -0.3278) | **FAIL** (on the cohort mean) |
| 10 tiling (arm B) | ideal and thermal drift ≤ 0.3·D0 | ideal 6/6 (ratio 0.135); thermal0 6/6 (ratio 0.133); thermal1 6/6 (ratio 0.173); thermal2 6/6 (ratio 0.165); s median 1.000, min 1.000, > 0.5 on 100% | **PASS** (24/24 model-tile pairs) |

**§2.2 per-site charge deviation** (report): max +0.6626 ± 0.0753 e, rms +0.0572 ± 0.0054 e, sites at the ζ stop +0.0000 ± 0.0000, worst |per-graph sum| 2.0e-15 e.

**Arm B fails four gates and passes three.** Gates 2 and 9 fail for one measured reason —
the image term contributes −0.2439 eV/Å to the size slopes (§7.4b) — and gate 7 continues
the trend of the whole cycle. **Gate 10 passes 24/24**, which is the term's own adoption
test and the only gate arm A could not be scored on.

`s` = 1.000 at every quantile across all 72 tiled frames, exactly as pre-registered before
arm B ran: the switch never engaged, so gate 10 measured the compensation and not the gating.

**No arm C table.** It was triggered by gate 7 and deliberately not run — see §4.5.

---

## 6. The forecasts, scored

Every pass condition below was written down before its numbers existed; where the forecast
itself left a threshold or a baseline unstated, the reading was pre-registered in
`CYCLE_SPEED_ARMS_RESULTS.md` and is quoted here.

| | forecast | outcome |
|---|---|---|
| **F21** | arm A: F10 passes ≥ 4/6 | **fails**: 1/6. Better than Stage B's 0/6 and for a different reason — the channel is alive, and the ligand-Cl − bulk-Cl separation is +0.651 ± 0.266 eV against Stage B's +0.188 ± 0.156 — but its 2σ is larger still, so it does not resolve. |
| **F22** | arm A: seeds at a pp stop fall by half relative to Stage B | **fails**: pre-registered as pp-σ ≤ 1/6 and pp-π ≤ 2/6 against Stage B's 2/6 and 5/6. Measured 2/6 and 4/6 — pp-σ unchanged, pp-π down by one seed. Gate 7 fires on all four types, and ss-σ nearly doubled (3/6 → 5/6). |
| **F23** | arm B: Δc falls 0.2–0.4 eV; R inside [0.66, 1.34]; thermal tiling drift ≤ 0.3·D0; F4 holds | **fails, 2 of 4 clauses.** Δc +0.876 ± 0.314 against arm A's +1.042 ± 0.197 — a fall of **0.166 eV** against a pre-registered [+0.64, +0.84]: **fails**. R 0.958 ± 0.090, 6/6 inside: **passes**. Thermal tiling drift 0.133–0.173 of D0, 18/18 thermal pairs: **passes**. F4 −0.3112 ± 0.0151, 0/6 in band: **fails**. All four were required. The Δc clause is not evidence about the term (§7.1b); the F4 clause is, and it is the one that matters. |
| **F24** | ≥ 4× per epoch from section 1 | **fails at 3.4× per epoch**; the step it named is 4.41× faster. The gap is the uncached epoch's fixed costs, which section 1 did not touch. |
| **F25** | the Stage B cohort's 79-atom head slope has moved toward +0.37; arm A restores it | **clause 1 fails**: −0.1804 ± 0.0244 against the forces-only cohort's −0.1675 ± 0.0231 — *further* from +0.37, not nearer. Clause 2 is therefore moot and holds trivially: arm A is −0.1212 ± 0.0268, negative on every seed and nowhere near the artefact. The forecast's premise — that full-weight 79-atom charged energies drag the head toward the base's small-cell artefact — was not observed, so the null gate that removes them is a precaution against an unmeasured harm rather than a repair of a measured one. |

**All five scored. Four failed and one is moot because its premise did not hold.** F21 and F22 were the two forecasts that said this cycle's model edits would fix
something, and neither did. What the edits demonstrably did do — revive a dead channel,
converge the size slopes, remove 87% of the tiling drift — was forecast by no one, and F23
asked about the one term that worked using an instrument that could not see it.

---

## 7. What this leaves

### 7.1 The Δc origin decision (spec §8.5) — settled

`c` is calibrated **once, before training**, as the median over every charged frame of
`(E_label − E_base − Δ_SR − Δ_LR)/Δn`. At that point, before a single gradient step, arm A's
six seeds already carry Δc = **+0.766 ± 0.000 eV**.

Neutral frames never enter that calibration, so they are an independent sample, and the
frozen base's residual steps by −0.745 eV between the 79- and 159-atom cells on neutral
frames against −0.793 on charged ones. In c units (Δn = −1, one hole):

| | value, 95% |
|---|---|
| Δc as calibrated | **+0.7657 ± 0.0004** eV |
| carrier-**independent** part, measured on frames with no carrier | **+0.7456 [+0.7199, +0.7790]** eV |
| carrier-**dependent** remainder | **+0.0440 [−0.0419, +0.1149]** eV |
| the electrostatic prediction | **+0.0491 ± 0.0026** eV |

**97 % of Δc was never about the carrier.** `c` is calibrated on charged frames with no
same-size neutral reference, so it absorbs the base's size-dependent total-energy offset
against these labels and reports it as if it were carrier physics. Stage A′'s "Δc = +0.77
against +0.05 predicted" was not a fifteen-fold failure of the electrostatics; it was one
number measuring two things, and the prediction was inside the interval all along.

**The remedy, and it is one function.** In `c_shift_table_terms`, subtract the median neutral
residual at the same cell size before dividing by Δn. It changes no model term. It is *not*
in this cycle — §9 excludes label-side work and the arms are trained on the current
definition — and it is the first item for the next, with +0.044 [−0.042, +0.115] eV as the
number to reproduce.

**Two consequences for the rules as written.** Standing rule 3 (reference formation energies
to the largest nulled size class, 159) turns out to be correct *for the right reason*: it
avoids the 79-atom column, which is the one carrying an uncorrected 0.75 eV offset. And gate
3 could not have been promoted from a report to a gate this cycle, because the quantity it
reports is dominated by something that is not the quantity it names.

**Caveat.** The large-cell populations are 15 neutral and 16 charged frames; the
carrier-dependent remainder's interval is ±0.08 eV, so the agreement with +0.049 is
"consistent with" rather than "confirms". The carrier-independent claim is the robust half.

### 7.1b Δc should not be used to score a model term again

Two independent reasons, both found inside this cycle:

1. **It is 97% carrier-independent** (§7.1). Of the +0.766 eV present at initialisation,
   +0.746 is a size offset of the frozen base against these labels, measurable on frames with
   no carrier in them. A model term that changed the carrier physics perfectly would move the
   remaining 4%.
2. **The null gate freezes the column that would have moved.** Under §0's rule 2, c(79) gets
   no gradient and sits at its epoch-0 calibration; c(159) is the only energy-referenced
   constant left, and it drifts to wherever the energy loss needs it. So trained Δc measures
   that drift, not the term.

The second is visible in the arm B numbers and is the sharpest evidence this cycle produced
that a diagnostic can be confidently wrong. **Before training**, the image compensation cut
Δc from +0.766 to +0.417 — a 0.349 eV fall, inside F23's band, and a correct signal that the
term does something size-dependent and large. **After 24 epochs** the fall is 0.048 eV.
Training erased 86% of a real effect, because the constant that absorbs it was free to move
and the constant that would have opposed it was frozen.

F23 asked whether the image term reduces Δc. The term reduces the tiling drift by 87%
(gate 10, 16/16, thermal tiles included) and Δc by nothing, and both statements are true at
once. **The forecast was aimed at the right term through the wrong instrument.**

Recommendation for the next cycle, and it is cheap: score model terms on gate 10's tiling
drift, keep Δc as a *label-quality* diagnostic where it belongs, and fix its referencing
(§7.1's one-function remedy) before quoting it again.

### 7.2 The on-site channel is alive, and that is now the problem

§2.1 was introduced because Stage A′'s output-centred form `γ[tanh h(x) − tanh h(x̄)]` goes
flat once `h` saturates, and F10 measured 0/6 with the channel dead. The corrected form
works exactly as designed: `tanh h` is still saturated on **100 % of Cl atoms**, and the
corrections are nonetheless ±0.13–0.51 eV, because `γ tanh[h(x) − h(x̄)]` reads the
difference of the arguments.

Three of this cycle's results are consequences of that revival, and none of them is good yet:

- **The level moved deep.** Knockouts at fixed weights put arm A at 0.41 eV below the CBM,
  0.12 with the on-site correction disabled and 0.046 with both on-site channels disabled —
  which is exactly Stage B's 0.05. **Stage B's shallow donor was the absence of a
  correction, not the presence of good physics.** The revived channel's first act is to push
  the level 0.29 eV away from the shallow donor V_Cl⁺ is expected to be.
- **F10 still fails, for the opposite reason.** The ligand-Cl − bulk-Cl separation is
  +0.49 eV — ten times the 50 meV threshold and 2.6× Stage B's — with a 2σ of 1.43 eV. The
  channel is loud and inconsistent rather than quiet and dead.
- **It is seed-unstable**: 0.26 to 0.80 eV of level shift across four seeds, and the
  alignment IQR roughly doubles when it is switched on.

A fourth result explains the other three. Weighted by atom count the channel spends
**−374 eV·atom on the 680 bulk Cl** against −3.7 on the 16 hub Pb: it is a near-uniform shift
of the Cl sublattice, not a defect correction. A uniform shift moves the frontier eigenvalue
down, builds no potential well, and does not separate ligand from bulk — the deep level, the
delocalisation and the unresolved F10 are one thing seen three ways.

**And the shift is a size artefact of the correction's own centre**, measured on populations
chosen so that one of them contains no carrier at all (`c14_centre_offset.py`, three seeds,
|bulk-Cl correction| in eV):

| population | mean | × the centre's own population |
|---|---|---|
| neutral **80** — the cells `x̄_s` is built from | **0.091** | 1.0 |
| neutral **79** — small cell, a vacancy, no carrier | 0.092 | ≈ 1 |
| neutral **159** — big cell, a vacancy, **no carrier** | **0.404** | **3.2, 4.4, 7.2** |
| charged **159** | 0.635 | 12.5, 1.2, 29.8 |

The centring works where it is defined and fails where it is applied. `x̄_s` is the
per-species mean first-block feature over 80-atom stoichiometric thermal cells; bulk Cl in a
159-atom cell does not sit at it, so `h(x_i) − h(x̄_s)` carries a non-zero mean over 680
atoms and γ tanh of it is a sublattice-wide constant. Nothing in training can tell that from
a genuine on-site shift, and it is present on frames with no carrier in them.

Two levers follow, and neither is among §9's exclusions: a **per-size-class centre** (or one
taken from the neutral frames of each size), and a prior or penalty on the correction's
shell-to-shell spread, which nothing in this cycle's spec constrains.

### 7.2b One error, in two places

§7.1 and §7.2 are the same mistake:

| # | the reference | built on | applied to | what leaks in |
|---|---|---|---|---|
| 1 | `c_shift_table` | charged frames, no same-size neutral | both size classes | **+0.75 eV of Δc** that is not carrier physics |
| 2 | on-site centre `x̄_s` | 80-atom stoichiometric cells | 79- and 159-atom defective cells | a **0.4–0.6 eV bulk Cl shift** that is not defect physics |
| 3 | participation reference | 80-atom pristine cells | 159-atom charged cells | **every ratio inflated 1.98×** — the diagnostic used to judge 1 and 2 |
| 4 | `δ_L` | 80-atom stoichiometric cells | 159-atom charged cells | 0.72× error; **nil**, while `s` saturates at 1.000 |

The audit was run rather than filed, and items 3 and 4 came out of it. Item 3 is the
uncomfortable one: the participation ratio is the statistic this cycle used to say how the
carrier localises, and its denominator was measured on a cell half the size of its numerator.
Item 4 is the reassuring one: the pattern is present, the mechanism is confirmed, and the
consequence is nothing at all on this host — an audit that only ever finds disasters is not
being run honestly.

**A reference quantity computed on one cell population and applied to another leaks the
difference between the populations into the physics.** Both instances were invisible to the
loss — the fit absorbs them and reports a good RMSE either way. Both were found the same way,
by scoring a control population that carries no carrier and asking whether the quantity
survives. Both have the same shape of remedy: build the reference inside the population it is
subtracted from.

That is a stronger statement than either finding alone, and it is the one to carry into the
next cycle: **every reference in this model should name the population it was computed on,
and be refused when applied to another.** The two known instances are the first two items;
the audit is the third.

### 7.3 The hub coupling, for the third cycle running

Gate 7 fires on every integral type. F22 forecast that the corrected on-site form and the
per-site charges would relieve the pressure by half; instead pp-σ's stop rate rose. Arm C
widens the log modulation to ln 2 as the spec's escape, but the pattern across Stage 3,
Stage A′ and now this cycle is consistent: **whatever freedom the head is given, it asks for
more hub coupling.** Superexchange — a three-centre term, explicitly excluded by §9 — remains
the standing hypothesis and has now been deferred three times.

### 7.4 The image compensation's switch is inert on this host

`s = sigmoid((depth/δ_L − 2)/0.5)` with the measured δ_L = 0.0166 eV is half-open at 33 meV
of depth and fully closed above 91 meV. Every model this programme has produced sits at
0.1–0.8 eV, so `s = 1.000` to six decimal places. The switch is not wrong; it cannot engage
on a level this deep. Gate 10 therefore tests the image term itself, and the A-to-B
difference is attributable to the compensation rather than to the gating.

### 7.4b The image compensation: do not adopt, and here is the target

The term's evidence is split, and every part of it is measured:

| | arm B |
|---|---|
| **gate 10, tiling drift** | **PASS** — **24/24 pairs**, 0.135 of D0 on the ideal tile and 0.133–0.173 on the three thermal ones |
| **gate 2, F4** | **FAIL — 0/6**, −0.3112 ± 0.0151 against a band of [−0.142, −0.063] |
| gate 7, hub stops | worse: **6/6 on every integral type** |
| gates 5, 6 | pass |
| gate 9, 79-atom slope | **FAIL** — −0.3422 ± 0.0249, same cause as gate 2 |
| depth from the CBM | **0.234 ± 0.049, shallow 6/6** — better than arm A's 0.334 (5/6) |
| `N_eff` spread | **0.83 against arm A's 3.62** — the term stabilises localisation |
| Δc | unchanged (and Δc cannot see the term — §7.1b) |
| cost | **+76% per epoch** |

**It fixes one size dependence and breaks another.** Gate 10 measures how the frontier
potential changes with **cell size L** under tiling; F4 measures how δ_SR changes with **hub
separation d** inside a fixed 159-atom cell. They are different geometry dependences and the
term moves them in opposite directions.

**The knockout localises the failure exactly.** Scoring the same trained models with
`image_potential` patched to zero — no weight touched — gives F4 = −0.0761 ± 0.0029, inside
the band on every seed and with a *tighter* spread than arm A's −0.0874 ± 0.0142. The term
contributes **−0.2439 eV/Å**, which is 2.4× the entire label slope, and it is the whole of
the excess with no residual.

So the head arm B learned is not damaged — on this measure it is the best-behaved head in the
cycle. What fails is the term's own additive contribution at evaluation.

**Verdict: do not adopt.** Gate 2 is a gate, not a report; arm B is the only cohort in this
programme to fail it, and it fails gate 9 with it. A term that removes 87% of the cell-size drift while tripling the
hub-separation slope has moved the error from one axis to another rather than removed it.

**And the failure is a magnitude, which §2.5 does not expose.** `comp = −s · φ_img / ε∞`
carries no amplitude. If the response is linear, a prefactor near λ ≈ 0.27 would put F4 at
the band's edge and λ ≈ 0.10 near its middle; what is not known is how much of gate 10's
86.8% survives that scaling. `c20_image_scaling.py` is written and committed to answer it
cheaply — it wraps `image_potential` with a scale and re-measures the drift — and it was
**deliberately not run in this cycle**: the amplitude is a design decision, not a
measurement to be slipped in at the end of a run.

That is what this cycle hands the next one about the image term: not a verdict to trust, but
a number (−0.244 eV/Å), a mechanism (the term rides on `d` far harder than the labels do),
and a knob that does not exist yet.

### 7.5 What the cycle bought

Speed, at 3.4× per epoch against a 4× forecast, from a finding that was not the one being
looked for: the step was never eigensolve-bound. And the closest this programme has come to a
size-extensive head — arm A's 79- and 159-atom slopes agree to 0.007 eV/Å against 0.051 in
Stage B and 0.105 in the forces-only cohort — bought at the cost of F4 margin, one seed
crossing the band's lower edge, and a level 0.29 eV too deep.

---

## 8. Operational record

### 8.1 What the smokes and tests caught that the loss would not have

Ten defects reached a running configuration this cycle. Eight were caught before they reached
a number and are silent failures — none raises, and each would have produced a plausible
loss curve. The last two are listed separately because they are a different kind: one was
caught by arithmetic before it fired, and one was not caught at all until it killed an arm.

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

### 8.1b The two that were not like the others

**9. The chain script was edited while a copy of it was running.** `bash` reads a script by
byte offset and seeks back to the end of the last parsed command before running it, so an
insertion *above* the driver's resume point shifts every offset after it.
`queue_arms_chain.sh` was 2308 bytes at launch (10:31:49) and 2635 after gate 10's block went
into `gates()` at 10:45:13. When wave 1 finished, the driver would have resumed 327 bytes
early — inside `gates()`, on a continuation line with `${tag}` unset — and `set -u` would
have aborted it: four models, no error, nobody watching. Caught by arithmetic on the two file
sizes rather than by anything failing, and fixed at 11:11 by restoring the launch bytes
without killing a process. The chain resumed correctly into wave 2 at 11:39:18.

*Statement of record: never edit a shell script while a copy of it is running.* The file on
disk is not the program that is running. Every subsequent change went into a new file
(`post_gates.sh`, `queue_arms_chain_c.sh`, `queue_arms_chain_bc.sh`).

**10. The one that got through.** Arm B failed on every seed within a minute of launch:
`collect_pristine_centre` — the pristine pass that *measures* δ_L — ran the image term's
bound switch, which raises when δ_L is unset. It raised from inside itself.

It got through because **every test in `test_image_compensation.py` set δ_L by hand** through
a `_with_spacing` helper before touching the model. The helper existed precisely because the
switch needs δ_L, and in supplying it the tests removed the ordering the trainer actually
uses. A fixture that makes the code under test convenient to call can delete the failure mode
it was written to expose. Arm A never touched the path because the term is off there, and the
local smoke was run with `IMAGE=False`.

The fix is one clause; the two regression tests were each checked to fail without it, and the
second asserts the guard does not *latch* — because a latched guard would let arm B train,
report no error, and silently be arm A, which is worse than the crash.

**11. And one that cost ten minutes for the oldest reason there is.** The arm B gate stage
was launched over `ssh` with a *relative* path to its script. `ssh host '...'` starts in
`$HOME`, so it failed instantly with "No such file or directory" and sat there while a
waiter blocked on a condition that would never come true. `b3_run.sh` exists precisely to
make this impossible and had been used for every scorer in the cycle; it was bypassed for
the one command whose failure mattered most, at the point where the chain driver had been
killed and nothing else would have retried it.

The lesson is not "be careful". It is that a convenience wrapper only helps while it is the
*only* way in — the moment a command is typed around it, the wrapper's guarantees are gone,
and the commands most likely to be typed around it are the unusual ones, which are also the
ones with no second chance.

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
