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

Per seed, 79 atoms: s1 −0.2156, s2 −0.1697, s3 −0.1922, s4 −0.1367, s5 −0.1926, s6 −0.1757
(`~/runs/stageb_b2.json` on b3; mean −0.1804, sd 0.0244). Per seed, 159 atoms (matched):
−0.1050, −0.1578, −0.1055, −0.1475, −0.1498, −0.1119.

*(An earlier revision of this line listed a different six numbers; they were mis-transcribed
from the JSON. The aggregates quoted in the table above were always the JSON's own and are
unchanged.)*

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

## §1.1 continued — the sampler, and the reference branch that was never there

**The batched solver needs size-uniform batches to fire.** A shuffled loader over 2377
frames at 79/80 atoms and 34 at 159/160 produces one by chance about a batch in six.
`mace/data/size_sampler.py` groups by EXACT atom count (79, 80, 159 and 160 are four
groups), permutes within each group every epoch, then permutes the batches across groups so
a large-cell step is as likely early in the epoch as late — grouping without that second
shuffle is a systematic change to the optimiser's trajectory dressed as a speed fix. The
short tail batch of each group is kept: `drop_last` would discard up to seven of the 34
large frames every epoch, which is the population the two-size upweight exists to protect.

**The neutral reference branch is exactly zero, and was being computed anyway.** The model
evaluates the correction twice per forward, once at the frame's counter and once at the
reference counter, and subtracts. When the reference is the neutral counter — which it is
unless a caller supplies `carrier_counts_ref` — every term of that second branch is
identically zero:

| term | why |
|---|---|
| `delta_sr_ref` | the head's energy is a difference from the neutral fill; at `counts_ref = 0` the four fills are two identical pairs, the occupations subtract to exactly zero, and `D = 0` |
| `q^pol` | zero in all three of its branches (the ungated one carries a `total_carriers` factor, the gated one a gate built from the same counts, the disabled one by construction) |
| `q^carrier` | carries `carrier_signs * counts` |
| `delta_lr_ref` | therefore `E[q_host] - E[q_host]` coupled, `E[0]` uncoupled; both zero, with zero position gradient since `E` is quadratic in `q` |
| `_isolated_carrier_self` | the same signed counts |

Skipping it removes a whole head pass, a Madelung potential, an Ewald evaluation and the
`correction_energy_ref` force gradient. `test_neutral_reference_skip.py` measures the claim
rather than trusting it: on a real charged batch, in five long-range configurations (frozen,
live host charges, polarisation on, gated polarisation with the isolated carrier self-term,
and host-carrier coupling), the energy, forces, `delta_forces`, `base_forces` and **every
trainable parameter's loss gradient** agree to < 1e-12 with and without the skip. A caller
that supplies its own reference counter gets the full branch.

Both flags are CLASS attributes on `CountingHead`, not instance ones: every trained
checkpoint is a pickled module, so an attribute set in `__init__` is absent from every model
written before it existed and `getattr` silently returns the fallback. That is why the first
profile after the skip landed showed no change at all.

### The step, by component, after §1.1

Seconds per step, CPU total, same configuration and batch as the table above:

| region | before | +batched solve | +reference skip |
|---|---|---|---|
| **wall per step** | **3.63** | **1.51** | **1.07** |
| step/forward | 1.742 | 0.793 | 0.611 |
| head/loop | 0.957 | 0.111 | 0.058 |
| head/bisect | 0.670 | 0.015 | 0.007 |
| head/eigh | 0.173 | 0.066 | 0.034 |
| ewald/madelung | 0.092 | 0.095 | 0.049 |
| ewald/lr | 0.082 | 0.080 | 0.055 |
| model/grad_corr_ref | — | 0.131 | **0.000** |
| model/outputs | — | 0.084 | 0.179 |
| step/backward | 0.147 | 0.077 | 0.084 |

**3.4× on the step.** F24 asks for ≥ 4× and is scored on the epoch wall, not the profiler's
per-step number; the epoch measurement follows below. What is left is the trunk's own
overhead (spherical harmonics, the jit-scripted e3nn graphs), the three force gradients, and
the Ewald branch — of which the fusion of the surviving calls into one reciprocal-space pass
is worth about 8% and was **measured and not done**, recorded here so the omission is a
decision.

---

## §2 — decisions recorded before the arms

**The centre is the pristine ENSEMBLE, because there is no single pristine geometry.** All
544 stoichiometric training frames are 80-atom cells labelled `config_type=ideal`, and they
are thermal snapshots: the Pb–Cl first-shell spread runs from 0.082 Å at the tightest to
0.213 Å at the loosest (median 0.144). There is no relaxed pristine cell in the dataset to
centre on, so the centre stays the per-species mean of the first block's features over all
of them — one snapshot's means would carry its own distortion into every `corr_i`. Recorded
as an interpretation of standing rule 1's "pristine geometry".

**The per-species centre is not an atom-by-atom identity on this host.** Pnma CsPbCl₃ has
more than one Wyckoff site per species and the frames are thermal, so `corr_i` does not
vanish atom by atom on a real pristine cell. The identity is asserted where it holds (a cell
with one environment per species: residual 5.2e-18 eV) and the per-species residual on the
real host is a measurement to be reported with the arms, not a claim.

**`delta_L` comes from the 80-atom pristine cells**, the only pristine size the dataset has,
and is collected on the same pass as the centre. The common-δ_L convention (one size for
every frame the bound flag touches) is kept; it is recorded here because the arm-B tiling
test applies the flag to 639- and 2159-atom cells.

**Both forms of §2.5's constancy identity fail on this host, structurally.** On a simple
cubic Bravais lattice the periodic potential of equal charges is constant to 2.2e-16 eV — the
kernel is right. On the 40-atom perovskite it spreads by 8.98e-02 eV, and the
periodic-minus-isolated difference by 6.06e-02 eV² in variance, because **equal charges on
five inequivalent sublattices are not a uniform charge density**: the structure factor is
non-zero away from k = 0. The A′ spec asserted constancy of the difference and the speed
cycle's moved it to the periodic part; neither holds here. The term's adoption test remains
the tiling drift, which is a statement about size dependence and does not rest on this.

**The covalent-radius anchor moves the initial hoppings.** Cordero radii give Cl–Cl 2.04,
Cl–Pb 2.48, Cl–Cs 3.46, Pb–Pb 2.92, Cs–Pb 3.90, Cs–Cs 4.88 Å against the retired single
2.861 Å, so Harrison's 1/d² scales the pair initialisations by 1.97, 1.33, 0.68, 0.96, 0.54
and 0.34 respectively. The init gate is reported from the arms' own logs.

**The §3 share is ambiguous once the null gate is in force, and is read as follows.** With
the 79-atom charged energies zeroed, the 159-atom share of the charged ENERGY loss is 1.000
by construction, so "realised energy share 0.25–0.5 of the charged energy loss" cannot be a
target any more. It is logged every epoch and expected to read 1.000 for that stated reason;
the number to compare across arms is the charged-force share, which stays at its 0.25
target. `apply_size_upweight` refuses the degenerate case rather than solving it: with the
small cells' energy mass at zero the old formula returns a factor of **zero**, which would
have multiplied the surviving large-cell energy weights by nothing and deleted every charged
energy from the loss without a word.

### F24, on the trainer's own profile

The trainer profiles one training step before and after attaching the base cache, and logs
both. Same machine (local A4000), same configuration, same measurement as the Stage A′
report's table — which is what makes the comparison a comparison:

| | Stage A′ | this cycle | gain |
|---|---|---|---|
| uncached step | 4.239 s | **1.143 s** | 3.71× |
| cached step | 3.451 s | **0.783 s** | 4.41× |
| base cache build (2797 frames) | 338 s | **81 s** | 4.2× |

**F24, scored both ways.** On the cached step, **4.41×** — holds. On the epoch wall,
which is what "per epoch" means: the Stage A′ report's cached epoch 0 (c-shift calibration
through evaluation) was **8.4 min** on this machine; the same measurement here is
**2 min 28 s**, i.e. **3.4×** — falls short. The gap between the two is the part of the
epoch section 1 did not touch: the trainer's own train- and validation-set evaluation
passes, which are forward-only and were never the eigensolve loop's problem. Recorded as
**F24 fails on its own terms (3.4× against 4×) while the step it named is 4.4× faster**,
because scoring it on the step alone would be scoring it on the half that was optimised.

Both come from the two changes of section 1.1 — the batched size-uniform solve and the neutral reference
branch that was being computed and thrown away. Neither is an approximation; both are
asserted equal to the paths they replace, one to 1e-8 and one to 1e-12 including every
parameter gradient. The epoch wall follows the step.

The 79/80/159 groups give 321 batches per epoch against 360 unsorted, so the epoch is
slightly shorter for a second, uninteresting reason (the tail batches of three groups
instead of one).

**One regression the smoke caught, and it is section 1.1's doing.** The initialisation gate
is scored on a PRISTINE spectrum, and the trainer took its spectrum from the first training
batch. Under the size-grouped sampler the first batch is one size group, and 1985 of 2560
training frames are charged 79-atom cells — so the gate logged UNSCORED. A speed change had
silently removed a gate. The batch is now built from stoichiometric frames rather than hoped
for; the warning that survives says "no batch in the training loader carries a
stoichiometric cell", which is a different and much louder condition.

---

## §5.2 — the feature-space kNN diagnostic (report only)

Regime: `stageb_s1` (Stage B seed 1, head only on the frozen A′ production base), its own
first-block feature space (32 dimensions, the head's slice of `defect_feature_readouts[0]`).
350 neutral training frames build the reference cloud (28 159 atoms); **100 disjoint neutral
frames are held out as the floor** — scoring a frame that is itself in the cloud gives zero
by construction, and a floor of zero makes every charged number look enormous. Per atom, the
distance to the nearest reference atom; per frame, the maximum over its atoms.

| population | n | median | p95 |
|---|---|---|---|
| neutral, held out | 100 | 0.1768 | 0.2921 |
| charged, 79 atoms | 400 | **0.2311** | 0.3500 |
| charged, 159 atoms | 16 | 0.0920 | 0.2207 |

Against d, on the charged 79-atom frames: 0.269 (4.5–5.0 Å), 0.217, 0.242, 0.247, 0.274
(6.5–7.5 Å); correlation with d **+0.267**.

**This agrees with F16, by a different route, and that is the reading.** `w_E` asked four
fold bases whether they disagreed and they did not. This asks whether a charged frame's
atoms have neighbours in the base's own training distribution, with no ensemble in it at
all — and they do: the charged 79-atom frames sit 1.31× the held-out neutral floor, their
p95 overlaps the neutral p95, and the trend with d is weak. The 159-atom charged frames are
*closer* to the cloud than held-out neutral frames, which is what a larger cell of mostly
bulk-like atoms should give.

So **two independent label-free detectors have now failed to see the +0.37 eV/Å residual
slope at 79 atoms**: it is not an out-of-distribution signature. That is the argument for
standing rule 2 being framed as it is — the rule is not a statistic about the input
distribution but a statement about what a residual MEANS, and it needs a measured neutral
null rather than a detector. `defect-perovskite/c10_feature_knn.py`, `~/runs/c10_knn.json`.

---

## §2.3 — the init gate under the covalent-radius anchor

Regime: arm A seed 1 at initialisation, on a stoichiometric 80-atom training cell,
`E_gap = 2.4 eV`.

| | Stage B (anchor 2.861 Å) | arm A (covalent anchors) |
|---|---|---|
| edge spacing below / above | 0.249 / 0.018 eV | **0.233 / 0.009 eV** |
| bandwidth | 32.84 eV | **33.73 eV** |
| verdict (edges ≤ 1.20, bandwidth ≥ 4.80) | PASS | **PASS** |

The anchors are Cl–Cl 2.04, Cl–Pb 2.48, Cl–Cs 3.46, Pb–Pb 2.92, Cs–Pb 3.90, Cs–Cs 4.88 Å
against the retired single 2.861, so Harrison's 1/d² rescales the pair initialisations by
1.97, 1.33, 0.68, 0.96, 0.54 and 0.34. The gate is unmoved: the initialisation is still
band-like, with a slightly wider band and slightly tighter frontier edges. §2.3's
"init gate unchanged" holds as measured rather than as assumed.

---

## Pre-registration, written 2026-09-04 while arm A is still training

Everything below is fixed **before** arm A's numbers exist. It is here so that the arm-C
decision, the F22 verdict and the Δc reading are not choices made after seeing the answer.

### The arm-C trigger (spec §4, gate 7)

Gate 7 is a *report* in §6, so the spec does not itself define the firing condition; the
condition is carried over verbatim from Stage A′, where it was pre-registered and failed:

> **fires if any integral type is at its stop in more than 1/6 seeds**, "at the stop" meaning
> the type's log modulation satisfies `|tanh g| > 0.98`.

Stage B's counts under exactly this rule: ss-σ 3/6, sp-σ 1/6, pp-σ 2/6, pp-π 5/6 — it fired
on three of four types. Arm A is expected to fire too; the arm-C launch is therefore planned,
not contingent on a surprise. If it fires, arm C is `ARM=c` in `queue_arms.sh` (β widened to
ln 2 = 0.693), six seeds in two waves of four and two, ≈ 3.3 h, scored on gates 2, 5 and 7 at
minimum and on the full ten if the wall clock allows.

### F22's pass condition

F22 says "the pp stop falls by half". Against Stage B's baseline that is

| type | Stage B at the stop | F22 passes if |
|---|---|---|
| pp-σ | 2/6 | ≤ 1/6 |
| pp-π | 5/6 | ≤ 2/6 |

Both must hold. ss-σ and sp-σ are reported but do not enter F22.

### How Δc will be read (spec §8.5, the Δc-origin decision)

Three measurements, of which two now exist and the third arrives with arm B:

1. **Stage B**: Δc = +0.772 ± 0.029 eV, predicted +0.049 ± 0.003. 79-atom charged *energies*
   were in the objective.
2. **Arm A**: the null gate removes 79-atom charged energies from the loss entirely.

**What c actually is, checked in the code rather than assumed.**
`calibrate_c_shift_table_over_loader` runs once at the end of training, under `no_grad`: it
**zeroes** `c_shift_table` and writes, into each (charge class, size class) cell, the
*median over every charged frame in the loader* of

    [ E_label − E_base − (Δ_SR + Δ_LR) ] / Δn ,

i.e. the residual per carrier. Neither c(79) nor c(159) is a trained parameter, and neither
is weighted by the loss: the null gate cannot touch them directly. Δc is therefore a
statement about **how much residual per carrier the model leaves at each cell size**, and it
changes across arms only through what the model learned, never through which frames were
scored. This was verified on the parameter itself — arm A's one-epoch smoke carries
`c_shift_table[0] = (9.4655, 10.2530)` and Stage B `(9.6883, 10.4130)`, with the scalar
`c_shift` added to both columns by the scorer and therefore cancelling in Δc.

An earlier draft of this block said arm A's Δc would be "the calibration offset alone, with
no fitted component". That reading was wrong in mechanism — nothing about c is fitted in
either arm — and it is corrected here, before the numbers exist.
3. **Arm B**: arm A plus the image term. The image term is the only electrostatic change.

The decision rule, fixed now:

- If **Δc(A) ≈ Δc(B)** to within the seed spread, the image interaction is not its origin.
- If **Δc(A) ≈ Δc(Stage B)**, then removing the 79-atom charged energies from the objective
  did not change how much residual per carrier the model leaves at either size. The
  difference is in the labels' size referencing — a quantity no head trained on these labels
  can absorb — and the remedy is a label-side re-referencing, not a model term.
- If **Δc(A) falls towards the predicted +0.05** while Stage B's was +0.77, then fitting the
  79-atom charged energies was *creating* the residual difference it was meant to remove,
  and the null gate has already fixed it.
- If **Δc(B) < Δc(A)** by more than the spread, the image term is carrying part of it and the
  compensation is doing real work on the referencing.

Any other pattern is reported as unresolved rather than forced into one of these.

### Gate 10 must be run by hand

The chain now running on b3 parsed its `gates()` body at launch, before the
`c3_tiling_drift.py --thermal 3` block was added to the file. `gates armb` will therefore
**not** run gate 10. After arm B lands:

    CUDA_VISIBLE_DEVICES=6 python -u c3_tiling_drift.py \
        --models ~/runs/armb_models/armb_s*.model --thermal 3 \
        --device cuda --out ~/runs/armb_tiling.json

and the `s` distribution per frame goes in gate 10's row alongside the drift verdicts.

---

## Regression check: the Stage B cohort is still scorable under the new code

Run 2026-09-04 on the local A4000, forward-only, `stageb_s1.model` (md5
`00ffb8ad…`, byte-identical to b3's `stageb_models/stageb_s1.model`), against the value
recorded in `~/runs/stageb_b2.json` from before any of this cycle's model edits.

| path | `d(delta_sr)/dd`, 79 atoms | difference from the recorded value |
|---|---|---|
| recorded, pre-cycle (b3) | −0.215623631885 | — |
| new code, batched solve | −0.215623620610 | +1.1e-08 |
| new code, `batch_by_size` forced off (loop solve) | −0.215623622067 | +9.8e-09 |
| new code, `zero_at_neutral_counts` forced off (reference computed, not skipped) | −0.215623623284 | +8.6e-09 |

Three claims are checked at once and all three hold:

1. **The legacy pickle path is exact.** A Stage B model has no `d_ref_pair` buffer and no
   `centre_form` attribute; the scalar-`d_ref` fallback in `radial`/`_anchor` and the
   `"output"` default in `__setstate__` reproduce that model's own arithmetic to 1e-8 across
   a different GPU. Old cohorts stay comparable to new ones.
2. **The batched solve is the loop solve.** Not a synthetic bit-identity test this time but a
   fitted slope over 120 frames, which would amplify any per-frame discrepancy.
3. **The neutral-reference skip is exactly the branch it replaces**, on a real trained model
   rather than on the test's constructed one — the branch really was computing zero.

The residual 1e-8 is the two GPUs' reduction order, not a code difference.

**One correction fell out of this.** The per-seed list under §5.1 did not match
`stageb_b2.json`; it has been replaced with the JSON's own numbers. The cohort aggregates
quoted against the gates were read from the JSON and never changed.

---

## §2.2 — the per-site charge deviation, and an early Δc signal

`c11_site_deviation.py` is new; nothing in the suite reported §2.2's required number.
Smoke regime: the **one-epoch** local arm A smoke model (`armsmoke_a`), ten charged
159-atom frames, ζ = 1.0 e. Not a trained result — it is here to say the channel is wired
and alive, and the 24-epoch numbers replace it.

| statistic | value |
|---|---|
| max \|Z_i − Z0\| | 0.085 e |
| rms \|Z_i − Z0\| | 0.025 e |
| per-species max (Cl / Cs / Pb) | 0.057 / 0.056 / 0.085 e |
| sites at the ζ stop (\|dev\| > 0.98 ζ) | 0.0% |
| \|per-graph sum after centring\| | ≤ 8.9e-16 e |

The channel is neither dead nor at its bound after one epoch, and the centring is exact to
machine precision, which is the check that the neutrality projection is not being undone by
the site term.

**An early reading on Δc, from the same one-epoch model.** `c4` on `armsmoke_a` gives
c(79) +9.487, c(159) +10.274, **Δc +0.787**, predicted +0.055. Stage B's was **+0.772**
against +0.049 predicted. The null gate removes 79-atom charged energies from the loss
entirely, so under the pre-registered rule above this is the third branch's *opposite*: Δc
did not fall towards the prediction when the fit lost those energies. On one epoch that is
weak evidence — the model has barely moved off its initialisation and c is a residual mean
dominated by the labels either way — but it is the branch to expect, and it says Δc is a
property of the labels' size referencing rather than of what the head was trained on.
Settled on arm A's six 24-epoch seeds, not here.

---

## Two operational findings during the wait, one of which would have killed the chain

### The chain script was edited 13 minutes after the chain started

`bash` reads a script by **byte offset** and seeks back to the end of the last parsed command
before running it, so editing a running script shifts every offset after the edit point.
`queue_arms_chain.sh` was 2308 bytes at launch (10:31:49) and 2635 bytes after the gate-10
block was inserted into `gates()` at 10:45:13 — a 327-byte insertion **above** the driver's
resume point. When wave 1 finished, the driver would have resumed at the byte offset of
`stage a "5 6"` in the *old* file, which in the new file is 327 bytes earlier: inside
`gates()`'s body, on a continuation line with `${tag}` unset. Under `set -u` that aborts the
shell. The chain would have stopped after wave 1 with four models and no error anyone was
watching for.

Fixed at 11:11 by restoring the b3 copy to the launch bytes (`git show 89d17a3:` …, md5
`2dd4d5f8…`, 2308 bytes) — no process killed, the driver's offsets are valid again, and
gate 10 stays a manual step as already recorded. `queue_arms.sh` (mtime 10:24) and
`queue_stage_b_gates.sh` (mtime 3 Sep) were both older than the launch and are invoked as
fresh processes per stage, so neither is affected.

**Statement of record: never edit a shell script while a copy of it is running.** Write the
change to a differently-named file, or wait. This is the same class of failure as LEDGER
entry 10's "wait on the condition, not the process": the file on disk is not the program
that is running.

### A per-host constant still lives in a launcher default (deferred, not fixed)

`defect-example/train_defect_model.sh:302` reads

    DEFECT_MADELUNG_EPS_INF="${DEFECT_MADELUNG_EPS_INF:-4.0}"

which is exactly what standing rule 1 forbids: the trainer's raise can never fire because
the launcher fills in CsPbCl₃'s number first. `queue_arms.sh` sets it explicitly to 4.0, so
no arm is affected and no number in this cycle changes.

It is **not** fixed now for the reason immediately above — four copies of that script are
mid-execution. It is fixed after `=== chain complete ===`, and it is written here so the
delay is a decision rather than an oversight.

### The arm-C launch, ready to paste

Arm C runs only **after** `=== chain complete ===`. GPUs 4–7 are fully committed until then
and the gate stages use all four, so overlapping arm C on 6–7 during arm B's two-seed wave
would break the four-GPU rule for the duration of a gate stage. On b3:

    bash ~/src/mace/.claude/worktrees/size-extensivity/defect-perovskite/queue_arms_chain_c.sh

where that file is a two-stage chain (`ARM=c SEEDS="1 2 3 4"`, then `"5 6"`, then
`TAG=armc` gates), written as a **new file** rather than by editing the chain script — see
the byte-offset finding above. `ARM=c` differs from arm A in one constant: `HOP_BETA`
0.4054651 (ln 1.5) → 0.6931472 (ln 2). Every guard is unchanged.

### F23's baseline, fixed before arm B runs

F23 reads "arm B: Δc falls 0.2–0.4 eV; R inside [0.66, 1.34]; thermal tiling drift ≤ 0.3·D0;
F4 holds". "Falls" needs a baseline and the forecast does not name one. Fixed now: **the
comparison is arm A, not Stage B**, because arm B is arm A plus the image term and nothing
else, so anything measured against Stage B would confound the image term with the null gate.
The clause passes if

    Δc(A) − 0.40 ≤ Δc(B) ≤ Δc(A) − 0.20   (eV, on the six-seed means)

and the other three clauses are the gate-6, gate-10 and gate-2 conditions unchanged. All
four must hold for F23 to pass; each is scored separately in the table so a partial result
reads as one.

F21 ("arm A: F10 passes ≥ 4/6") is already gate 4's condition and needs no interpretation.

---

## §3 as actually run — the objective, read off arm A's own log

Every clause of §3 that could fail silently is logged by the trainer, and all of it reads
correctly on the live run (`~/runs/arma_s1.log`, 4 Sep 10:32–10:36):

    Null-gated charged energies: sizes with a neutral null [159]; kept 16 charged frames'
      energies, dropped 928. Per size: {"79": {dropped 928, kept 0},
                                        "159": {dropped 0, kept 16}}
    Size upweight (charged, energy): the small cells carry no mass (the null gate zeroed
      them); the large cells are 100% of this channel and no factor is applied.
    Two-size upweight: factor 9.61, realised charged-force-loss share 25.0% (target 25%)
    Charged energy share: factor 1.00 -> realised 100.0% (target 25%)
    Realised large-cell shares: {"charged_large": 0.25, "charged_large_E": 1.0}
    Stage-3 protocol: the first batch carries no stoichiometric cell (size-grouped sampler);
      the gate is scored on the first batch that does.
    Stage-3 protocol: init gate edges 0.233/0.009 eV (need <= 1.20), bandwidth 33.73 eV
      (need >= 4.80) -> PASS

Five things are confirmed by those seven lines, none of which the loss curve would have shown:

1. **The null gate bites in the right direction and only there** — 928 small-cell charged
   energies dropped, all 16 large-cell ones kept, forces untouched.
2. **The degenerate upweight is caught.** With the small cells' energy mass at zero the old
   formula returns a factor of *zero*, which deletes every charged energy. The guard fires,
   says why, and applies no factor.
3. **The realised energy share reads 1.000 against a 0.25 target**, which is correct rather
   than a miss: with one size class contributing, its share is one by construction. §3's
   0.25–0.5 clause has no content under rule 2, and this is the line that says so.
4. **The charged-force share is on target at 0.250**, which is the number that *is*
   comparable across arms.
5. **The init gate found a stoichiometric batch.** The size-grouped sampler makes the first
   batch a single size group, which need not contain an 80-atom cell; the search fires,
   announces itself, and the gate passes on the covalent anchors (0.233/0.009 eV, 33.73 eV).

---

## §8.5 SETTLED — where Δc comes from, and why the prediction was never wrong

Three measurements, in order. Regime: arm A wave 1 (four seeds, 24 epochs, the §4.1 regime),
plus a forward-only pass of the frozen base `aprime_prod` over the training set
(`c13_c_origin.py`, `~/runs/c13_c_origin.json`).

### 1. Δc is there before training starts

`calibrate_c_shift_table_over_loader` runs **once, before the training loop** — `run_train.py`
calls it around line 1779 and `tools.train` at 1897 — and writes, into each (charge, size)
cell, the median over every charged frame of `(E_label − E_base − Δ_SR − Δ_LR)/Δn`. Arm A's
four wave-1 seeds logged:

| seed | c(79) | c(159) | Δc |
|---|---|---|---|
| 1 | +9.4655 | +10.2314 | +0.7659 |
| 2 | +9.4692 | +10.2349 | +0.7657 |
| 3 | +9.4678 | +10.2340 | +0.7662 |
| 4 | +9.4738 | +10.2389 | +0.7651 |
| | | | **+0.7657 ± 0.0004** |

against Stage B's *trained* +0.772 ± 0.029. **Δc is present at initialisation**, to four
decimal places, and it is the floor both arms start from: the seed-to-seed scatter at epoch
0 is 0.4 meV against 766 meV of offset.

What training then does depends on the arm, and arm A's four wave-1 seeds say so plainly:

| | Δc at epoch 0 | Δc after 24 epochs |
|---|---|---|
| Stage B (both columns trained) | +0.766 | **+0.772 ± 0.029** |
| arm A (79 column frozen by the null gate) | +0.766 | **+1.096 ± 0.222** |

Stage B barely moves it. Arm A moves it up by a third of an eV and multiplies the seed
spread by eight, which follows from rule 2 rather than contradicting it: with the 79 column
receiving no gradient, the 159 column is the *only* energy-referenced constant in the model,
so everything the energy loss cannot express elsewhere lands there. In Stage B the two
columns moved together and the difference stayed put.

*(An earlier revision of this paragraph said "24 epochs of training move it by less than the
seed spread", which was written from Stage B's trained value before arm A's existed. It is
true of Stage B and false of arm A. The decomposition below is unaffected — it explains the
+0.766 both arms begin with, not what either does to it afterwards.)*

The Gauge log confirms the mechanism directly: `c_table (0,0)` reads **+9.4655 at every one
of arm A's 24 epochs**, because the null gate removed all 928 small-cell charged energies
and no gradient reaches that column. `(0,1)` drifts +10.2314 → +10.2832, so the large-cell
column *is* trained. That asymmetry is a consequence of rule 2, not a defect.

### 2. The offset is carrier-independent — it is on the neutral frames too

Neutral frames never enter the c calibration, so they are an independent sample. Median
`E_label − E_base` from the frozen base alone:

| population | frames | median residual | per atom |
|---|---|---|---|
| neutral 79 | 300 | −0.5300 eV | −6.709 meV |
| neutral 80 | 300 | −0.2455 eV | −3.069 meV |
| neutral 159 | 15 | −1.2752 eV | −8.020 meV |
| charged 79 | 300 | −3.8874 eV | −49.208 meV |
| charged 159 | 16 | −4.6805 eV | −29.437 meV |

The first hypothesis — that the base carries a constant per-atom offset `b`, making
Δc = b × (159 − 79) — is **falsified**: the per-atom residual is −6.7, −3.1 and −8.0
meV/atom at the three sizes, sharing no common value and nowhere near the +9.57 meV/atom the
two c-table points imply.

What is true instead is the **total** step. Going 79 → 159 atoms the base's residual moves by
−0.7452 eV on *neutral* frames and −0.7931 eV on *charged* ones. Since `Δn = −1` on every
charged frame here (`carrier_counts = [0, 0, 1, 0]`, one hole) and `c = residual/Δn`, those
are c-space steps of **+0.745** and **+0.791** eV.

### 3. The decomposition, with a bootstrap

20 000 resamples of each population (the large-cell samples are 15 and 16 frames, so a
difference of medians needs an interval, not a point):

| quantity | value, 95% |
|---|---|
| Δc as calibrated, charged frames only | **+0.7657 ± 0.0004** eV |
| carrier-**independent** size step, from the neutral frames | **+0.7456 [+0.7199, +0.7790]** eV |
| carrier-**dependent** remainder | **+0.0440 [−0.0419, +0.1149]** eV |
| the cycle's electrostatic prediction | **+0.0491 ± 0.0026** eV |

**97% of Δc is a carrier-independent size offset of the frozen base against these labels.**
The remaining 4 % — the part that is actually about the carrier — is +0.044 eV with the
prediction of +0.049 eV comfortably inside its interval.

### The decision

The prediction was never wrong; the measurement it was compared against was contaminated.
`c` is calibrated on **charged frames only**, with no neutral frame of the same size to
reference against, so it absorbs the base's size-dependent total-energy offset wholesale and
reports it as if it were carrier physics. Stage B's "Δc = +0.77 against +0.05 predicted"
was never a fifteen-fold discrepancy in the electrostatics; it was one number measuring two
things.

Under the pre-registered rule this is the second branch — *"it is in the labels' size
referencing … the remedy is a label-side re-referencing, not a model term"* — reached by a
stronger route than the rule anticipated: not "the arms agree" but "it is there before
training, and it is there on frames with no carrier in them."

**The remedy, stated precisely.** In `c_shift_table_terms`, subtract the median neutral
residual at the same cell size before dividing by Δn:

    resid_charged(N) → resid_charged(N) − median[ resid_neutral(N) ]

Then c is the carrier's own cost and Δc is +0.044 [−0.042, +0.115] eV. This is one function,
it changes no model term, and it is **not** in this cycle — §9 excludes label-side work and
the arms are running on the current definition. It is the first item for the next one, and
the number to beat is above.

**What stays true meanwhile.** Standing rule 3 — reference formation energies to the largest
size class with a neutral null (159) — is *correct as written and for the right reason*: it
avoids the 79-atom column entirely, which is the one carrying an uncorrected 0.75 eV offset.
Gate 3 remains a report, not a gate, and the interval above says why it could not have been
promoted this cycle: the quantity it reports is dominated by something that is not the
quantity it names.

**Caveat, stated rather than buried.** The large-cell populations are 15 neutral and 16
charged frames. The carrier-dependent remainder's interval is ±0.08 eV, so the agreement
with +0.049 is "consistent with" and not "confirms". The carrier-*independent* claim is the
robust one: +0.746 [+0.720, +0.779] eV excludes zero by twenty-eight sigma-equivalents of
its own interval.

---

## Arm A, wave 1 — an early read on four of six seeds

Scored at 11:46 on GPUs 6 and 7 while wave 2 trains on 4 and 5 (four GPUs in use, the rule
holds). Four seeds, not six; the six-seed numbers replace these and any of them may move.

### Gate 7 fires. Arm C runs.

| type | Stage B (6 seeds) | arm A wave 1 (4 seeds) |
|---|---|---|
| ss-σ | 3/6 | **3/4** |
| sp-σ | 1/6 | **2/4** |
| pp-σ | 2/6 | **2/4** |
| pp-π | 5/6 | **3/4** |

L_b: ss-σ 1.144 ± 0.107, sp-σ 1.275 ± 0.029, pp-σ 1.360 ± 0.093, pp-π 1.316 ± 0.104 Å.

Under the trigger pre-registered before these numbers existed — any type at its stop
(|tanh g| > 0.98) in more than 1/6 seeds — **gate 7 fires on all four integral types**, more
broadly than Stage B did. **Arm C is confirmed**, at β = ln 2, after the A/B chain completes.

**F22 fails on this evidence.** Its condition was pp-σ ≤ 1/6 and pp-π ≤ 2/6. Wave 1 gives
2/4 and 3/4: pp-σ's rate rose (0.33 → 0.50) and pp-π's barely moved (0.83 → 0.75). The
corrected on-site form and the per-site charges did not relieve the hub coupling; if
anything the head presses harder against the stop. This is the third cycle in which the
head's answer to more freedom is more hub coupling, and it is why arm C exists.

### Gate 9 passes, and the size dependence has converged

| | 79 atoms | 159 atoms |
|---|---|---|
| arm A wave 1 | **−0.1156 ± 0.0298** | **−0.1085 ± 0.0193** |
| Stage B | −0.1804 ± 0.0244 | −0.1296 ± 0.0224 |
| s7 forces-only | −0.1675 ± 0.0231 | −0.0629 ± 0.0087 |

Per seed at 79 atoms: −0.0792, −0.1553, −0.0961, −0.1218. Negative on every seed and nowhere
near the base's +0.37 artefact, so gate 9's substantive clause holds; seed 1 at −0.079 sits
just outside a literal reading of "within 2× of −0.17" on the *shallow* side, which is the
harmless direction.

The result worth naming is the second column. **The head's small- and large-cell slopes now
agree to 0.007 eV/Å**, against a 0.051 eV/Å gap in Stage B and 0.105 in the forces-only
cohort. Size-extensivity of the head's own size dependence is what this programme is for, and
this is the closest it has come.

### Gate 3, with the §8.5 result in hand

Δc = **+1.096 ± 0.222** (Stage B +0.772 ± 0.029), predicted +0.047 ± 0.007. Read through
§8.5 this is two things: a **+0.766 eV floor** that is in the labels and the base before
training, plus **+0.33 ± 0.22 eV** that arm A's own objective added by freezing the 79 column
and leaving the 159 one to absorb whatever else the energy loss could not place. Neither part
is electrostatic. The remedy for the first is the same-size neutral reference; the second is
an argument for referencing c to the neutral frames *per size class* rather than leaving one
column untrained, and it belongs to the next cycle with the rest of the label-side work.

### Gate 4 / F10 on wave 1 — fails, but not the way Stage B failed

**0 of 4 seeds**, so F21 fails on this evidence (Stage B: 0 of 6). The interesting part is
the mechanism, which has changed completely:

| shell | Stage B, corr (eV) | arm A wave 1, corr (eV) | raw `tanh h` saturated, arm A |
|---|---|---|---|
| hub Pb | −0.307 ± 0.255 | −0.313 ± 0.134 | 0% |
| ligand Cl | +0.336 ± 0.767 | +0.361 ± 0.475 | **100%** |
| bulk Cl | −0.103 ± 0.331 | −0.127 ± 0.506 | **100%** |
| bulk Pb | −0.122 ± … | −0.079 ± 0.054 | 0% |
| Cs | — | +0.368 ± 1.061 | 50% |
| **ligand − bulk Cl** | **+0.188 ± 0.156** | **+0.494** | |

Stage B failed because the channel was **dead**: `tanh h` saturated and the output-centred
form `γ[tanh h(x) − tanh h(x̄)]` is identically flat once both terms sit at ±1. Arm A's
`γ tanh[h(x) − h(x̄)]` is **saturated in exactly the same place and alive anyway** — 100% of
Cl atoms have |tanh h| > 0.98, and the corrections are nonetheless ±0.13–0.51 eV, because
the difference of the arguments is what the form reads. **§2.1's correction does the thing it
was introduced to do.**

What it now fails on is *resolution*, not magnitude. The ligand-Cl − bulk-Cl separation is
+0.494 eV, ten times the 50 meV threshold and 2.6× Stage B's, but its 2σ is 1.43 eV: the
channel is loud and inconsistent rather than quiet and dead. That is a different problem with
a different remedy, and it is the first time in this programme that F10 has failed on the
spread rather than on the signal.

### §2.2 — the per-site charge deviation, arm A wave 1

| statistic | value over four seeds |
|---|---|
| max \|Z_i − Z0\| | **0.627 ± 0.041 e** (largest single site 0.670 e, always a Pb) |
| rms \|Z_i − Z0\| | 0.055 ± 0.007 e |
| per-species max, seed 4 (Cl / Cs / Pb) | 0.118 / 0.077 / 0.656 e |
| sites at the ζ stop (\|dev\| > 0.98 ζ) | **0.0%**, 0 of 4 seeds |
| dead channels (max \|dev\| < 1e-6 e) | 0 of 4 seeds |
| \|per-graph sum after centring\| | ≤ 1.1e-15 e |

The channel is working hard and is not at its bound: excursions reach two thirds of an
electron on Pb sites while the typical site moves 0.055 e, and no site is at ζ. ζ = 1.0 e is
therefore the right order — at ζ = 0.1 this channel would be pinned at its stop on the hub —
and the neutrality centring is exact to machine precision, which is the check that the site
term is not quietly reintroducing a per-cell net charge.
