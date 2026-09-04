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
calls after a warm-up (`c18_ewald_split_timing.py`).

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

### Gates 1, 2 and 5 on wave 1 — F4 survives, the level moves deep

`b10_adoption.py` against the A′ references and the four cross-fit folds.

| criterion | Stage B (6 seeds) | arm A wave 1 (4 seeds) |
|---|---|---|
| 1 no carrier leakage into the base | 6/6 | **4/4** |
| 1b base not shrunk | 6/6 | **4/4** |
| 2 neutral 159 slope toward zero | 0/6 | 0/4 (base-level, frozen: identical every seed) |
| 3 neutral-79 window not degraded | 0/6 | 0/4 (likewise) |
| **4 F4 in [−0.142, −0.063]** | **6/6** | **3/4** |
| 5 pristine gap 2.4 ± 0.1 | 6/6 | **4/4** |
| 6 shallow | 6/6 | **3/4** |

**Gate 2 (F4).** −0.0862 ± 0.0163 against Stage B's −0.1004 ± 0.0166; per seed −0.0595
(out, low), −0.1014, −0.0974, −0.0863. The slope has moved **shallower by 0.014 eV/Å** and
one seed has crossed the band's lower edge. On four seeds this is 3/4 and the gate needs
≥ 4/6; it survives on this evidence but with much less margin than Stage B had, and the
direction of travel is the same one gate 9 reports — arm A's head is uniformly *less*
size-dependent than Stage B's.

**Gate 5 (gap).** 2.391, 2.368, 2.397, 2.396 eV, all inside 2.4 ± 0.1, with the init gate
passed on the covalent anchors (edges 0.233/0.009 eV, bandwidth 33.73 eV). Passes.

**The level moved deep, and this is new.** Depth from the CBM: **+0.289, +0.817, +0.228,
+0.281 eV**, against Stage B's +0.014 to +0.093 eV. One seed is no longer classifiable as a
shallow donor at all. V_Cl⁺ in CsPbCl₃ is expected to be a shallow donor, so this is a move
*away* from the expected physics, and it is the first arm-A number that is worse rather than
merely different. It is not any single gate's business — criterion 6 catches it at 3/4 — but
it is the thing to watch when arm B's image term and arm C's wider modulation land, and it
belongs in the report whatever the six-seed numbers say.

Criteria 2 and 3 read identically on every seed because they are properties of the **frozen
base**, not of the head: the neutral 159 slope and the neutral-79 window are the same
numbers Stage B reported, and they fail for the same reason they failed then. They are not
evidence about the arms.

### Why arm A's level is deep — knocked out term by term

`b6_depth_edges.py` on the same four wave-1 models, with terms disabled at **fixed weights**
(`c17_depth_knockouts.py`; b3 `~/runs/arma_depth_probe*.log`).

| configuration | depth from CBM, seeds 1–4 (eV) | mean |
|---|---|---|
| **as trained** | 0.287, 0.804, 0.263, 0.298 | **0.413** |
| §2.2 per-site charges off | 0.212, 0.720, 0.194, 0.203 | 0.332 |
| §2.1 on-site correction off (`on_site_range → 0`) | 0.130, 0.111, 0.116, 0.127 | 0.121 |
| **both off** | 0.063, 0.036, 0.043, 0.042 | **0.046** |
| Stage B, as trained | 0.014 – 0.093 | ≈ 0.05 |

**With both learned on-site channels removed, arm A's level lands exactly on Stage B's.**
The whole 0.37 eV of extra depth is those two channels: **0.29 eV from the on-site
correction and 0.08 eV from the per-site charges.** Seed 2's outlier is the same story —
0.804 eV as trained, 0.111 eV with the correction off, in line with its siblings.

**This reframes Stage B's shallow level.** Stage B's on-site channel was *dead* — `tanh h`
saturated and the output-centred form is flat there — so it moved the level by nothing, and
the shallow donor Stage B reported was the *absence of a correction*, not the presence of
good physics. §2.1 revived the channel, exactly as intended and as F10's saturation numbers
confirm, and the first thing the revived channel did was push the level 0.29 eV deeper,
away from the shallow donor V_Cl⁺ is expected to be.

So the honest summary of arm A's on-site channel is: **alive, large, and unregularised.**
It is unresolved between shells (F10's separation is +0.494 eV with a 2σ of 1.43), it is
seed-unstable (0.26 to 0.80 eV of level shift), and it moves the one observable with an
external expectation in the wrong direction. That is a better problem than a dead channel —
it is at least a channel — but it is not yet a working one, and nothing in this cycle's spec
constrains it. A prior or a penalty on the correction's shell-to-shell spread is the obvious
next lever, and it is not among §9's exclusions.

*Caveat, stated: these are knockouts at fixed weights. They measure how much of the level's
present position each term contributes, not where the level would have settled had the model
been trained without that term. The second question needs an arm, not a probe.*

### A pre-registered prediction for gate 10, written before arm B runs

§2.5's validity switch is `s = sigmoid((depth/δ_L − 2)/0.5)` with `depth = probe.gap`, the
frontier level's separation from the continuum in the head's own spectrum, and δ_L the
pristine level spacing. Arm A's measured δ_L is **0.0166 ± 0.0004 eV**. That fixes where the
switch actually switches:

| depth (eV) | depth/δ_L | s |
|---|---|---|
| 0.020 | 1.20 | 0.169 |
| **0.033** | 2.00 | **0.500** |
| 0.051 | 3.10 | 0.900 |
| 0.091 | 5.45 | 0.999 |
| 0.27 | 16.3 | 1.000000 |
| 0.41 | 24.7 | 1.000000 |
| 0.80 | 48.2 | 1.000000 |

The switch is half-open at a depth of **33 meV** and fully closed above **91 meV**. Arm A's
frontier gaps run 0.27–0.80 eV and even Stage B's ran ~0.1, so on every model this programme
has produced, **s = 1.000 to six decimal places**. The `c3_tiling_drift` smoke already
measured exactly that: min, median and max all 1.000 over six tiled frames.

**Prediction, recorded before arm B exists: gate 10's `s` distribution will be a constant
1.000, and §2.5's switch will have changed nothing.** The image term in arm B is
unconditionally on. If that is what lands, the switch is not wrong — it is *inert on this
host*, because δ_L is 17 meV and the levels this head produces are ten to fifty times
deeper than the threshold. It would only engage on a genuinely continuum-resonant state,
which is not what V_Cl⁺ gives here.

The useful consequence is negative and worth stating: **gate 10 is a test of the image term,
not of the switch**, and any difference between arms A and B is attributable to the
compensation itself. If instead some frames come back with s < 1, that is news and the
distribution says which frames.

---

## Arm B failed on every seed at launch, and what fixing it revealed

### The defect

At 13:09 the chain started arm B. Every seed died within a minute:

    RuntimeError: the image compensation's bound switch needs delta_L; call
    collect_pristine_centre (which records it) before the first forward

raised from **inside `collect_pristine_centre` itself** (`defect_models.py:667` → `forward`
→ `_carrier_head` → `bound_switch`). The pristine pass that *measures* δ_L was running the
switch that *consumes* it. Arm A never touched this path because the image term is off
there, and every test in `test_image_compensation.py` called a `_with_spacing` helper that
sets δ_L by hand — so nothing in the suite exercised the order the trainer actually uses.

**The fix** is one clause: skip the image block while `_collecting_centre` is set. It costs
nothing physically — the collection pass is a stoichiometric, carrier-free cell where `alpha`
is zero, so the compensation is the image potential of a carrier that is not there.

**Two regression tests**, and both were checked to fail without the guard:

- `test_the_pristine_pass_that_records_delta_L_does_not_also_consume_it` — reproduces the
  exact failure and asserts the pass records δ_L and the term works straight afterwards.
- `test_the_guard_is_scoped_to_the_collection_and_released_afterwards` — the guard must not
  latch. If it did, arm B would train, report no error, and silently *be arm A*, which is a
  worse outcome than the crash.

**The chain was stopped rather than left running.** `queue_stage_b_gates.sh` waits for six
model files in a `while true` loop, so `gates armb` would have hung for ever on an arm that
produced none. Arm A was complete and scored, so nothing was lost. The remaining schedule is
`queue_arms_chain_bc.sh` — a **new file**, per this morning's byte-offset lesson — which also
refuses to score an arm with fewer than six models instead of waiting.

### What the relaunch immediately showed: the image term is large, and size-dependent

Same seed, same data, same frozen base; the only difference is the term. At the c
calibration, **before any training**:

| | c(79) | c(159) | Δc₀ |
|---|---|---|---|
| arm A | +9.4655 | +10.2314 | **+0.7659** |
| arm B | +9.8731 | +10.2904 | **+0.4173** |
| difference | **+0.4076** | **+0.0590** | **−0.3486** |

The compensation moves the small cell nearly seven times as far as the large one and cuts Δc
by 0.349 eV — the right sign, the right size scaling, and inside F23's forecast band of a
0.2–0.4 eV fall. (F23 is scored on the *trained* six-seed numbers, not on this; recorded here
because it is the first direct evidence the term does anything.)

**But it raises a question §8.5 cannot answer from Δc alone.** §8.5 measured a +0.745 eV
carrier-*independent* size step in the frozen base's residual, on neutral frames with no
carrier in them. The image term cannot have changed that — it enters the head's correction,
not the base. So the term is removing 0.35 eV of a 0.77 eV gap of which 0.75 eV is a
base/label artefact. An image interaction scales as 1/L and so does a great deal else,
including whatever produces that artefact, and **Δc cannot tell them apart.** If the
compensation is being rewarded for cancelling a labelling offset, it will look right on Δc
and wrong on everything referenced to a fixed size.

This is a second, independent argument for the same-size neutral reference in
`c_shift_table_terms`: with the base's offset cancelled, Δc would isolate whatever image
physics is actually there, and the term could be scored on it rather than credited for it.
Gate 10's tiling drift stays the term's real adoption test for this cycle, exactly as §2.5
intended, and now for a sharper reason than when that was written.

### The test suite after the guard

`tests/unit` + `tests/extensions/defect`: **953 passed, 1 skipped, 16 errors**. All sixteen
errors are `fixture 'benchmark' not found` in `tests/unit/test_compile.py` — `pytest-benchmark`
is not installed in this environment. Pre-existing, unrelated to any change in this cycle, and
confined to that one file. Recorded rather than rounded away.

---

## How arm A's models localise, and why the on-site correction is a bulk shift

### The localisation numbers (six seeds, 159-atom charged frames)

| measure | arm A | head-only (s7) | E_LR-off baseline | Stage B |
|---|---|---|---|---|
| participation `n_eff` | **18.79 ± 3.62** | 13.17 ± 0.94 | 14.98 ± 1.88 | — |
| ratio to the pristine frontier state | **0.754 ± 0.187** | 0.492 ± 0.028 | 0.478 ± 0.071 | 0.619 ± 0.089 |
| dilution R (gate 6) | 0.925 ± 0.144, 6/6 in band | — | — | 0.868 ± 0.076 |
| bound fraction | 0.635 ± 0.023 | — | — | 0.615 ± 0.067 |
| depth (dilution's measure) | 0.0667 ± 0.0094 eV | — | — | 0.0516 ± 0.0082 |
| depth from the CBM | **0.334** (0.150–0.817) | — | — | 0.014–0.093 |

**They localise worse.** `n_eff` rose from 13 to 19 atoms and the ratio to the pristine band
state from 0.49 to 0.75 — seed 1's 25.5 is indistinguishable from the pristine 25.2. The seed
spread rose **6.7×**.

**And that is backwards.** The level went 0.28 eV *deeper* while the state got *more*
delocalised. Energy and extent should move together; a deeper level binds more tightly.

### The correction is spent on the bulk, not on the defect

Per-shell mean correction, six seeds:

| shell | mean correction | atoms | correction × atoms |
|---|---|---|---|
| hub Pb | −0.232 ± 0.161 | 16 | −3.7 |
| ligand Cl | +0.101 ± 0.590 | 80 | +8.1 |
| **bulk Cl** | **−0.551 ± 0.789** | **680** | **−374** |
| bulk Pb | −0.080 ± 0.046 | 240 | −19 |
| Cs | +0.034 ± 1.004 | 256 | +8.6 |

One mechanism explains all three symptoms: a large, near-uniform shift of the Cl sublattice
moves the frontier eigenvalue down (deeper level), builds no potential well (no extra
localisation), and does not separate ligand from bulk (F10 1/6, unresolved).

### Where the bulk shift comes from — measured, not inferred

`c14_centre_offset.py` on three arm A seeds, |bulk-Cl correction| per population. Magnitudes,
because the *sign* differs by seed and a signed mean across seeds is therefore meaningless:

| population | s1 | s3 | s5 | mean | × the centre's own |
|---|---|---|---|---|---|
| neutral **80** — the centre's own population | 0.065 | 0.178 | 0.029 | **0.091** | 1.0 |
| neutral **79** — small cell, a vacancy, no carrier | 0.156 | 0.091 | 0.029 | 0.092 | 2.4, 0.5, 1.0 |
| neutral **159** — big cell, a vacancy, **no carrier** | 0.208 | 0.791 | 0.213 | **0.404** | **3.2, 4.4, 7.2** |
| charged **159** — big cell, vacancy and carrier | 0.818 | 0.211 | 0.876 | 0.635 | 12.5, 1.2, 29.8 |

**The centring works where it is defined and fails where it is applied.** On the 80-atom
cells the centre is built from, the correction is 0.09 eV — near zero, by construction, as
§2.1 intended. On 79-atom cells, also near zero. On **159-atom cells with no carrier in them
at all** it is 3.2×, 4.4× and 7.2× larger, in every seed. The charged frames add more on two
seeds of three, inconsistently.

So the bulk shift is **primarily a size artefact of the centre**: `x̄_s` is the per-species
mean first-block feature over 80-atom stoichiometric thermal cells, and bulk Cl in a
159-atom cell does not sit at it. The residual `h(x_i) − h(x̄_s)` then has a non-zero mean
over 680 atoms, and γ tanh of that is a sublattice-wide constant that no amount of training
can distinguish from a genuine on-site shift.

### The unifying statement

This is the **same error as §8.5's**, in a second place:

| | the reference | where it is applied | the consequence |
|---|---|---|---|
| `c_shift_table` | charged frames only, no same-size neutral | both size classes | +0.75 eV of Δc that is not carrier physics |
| on-site centre `x̄_s` | 80-atom stoichiometric cells | 79- and 159-atom defective cells | a 0.4–0.6 eV bulk Cl shift that is not defect physics |

**A reference quantity computed on one cell population and applied to another leaks the
difference between the populations into the physics.** Both instances were invisible to the
loss, both were found by scoring a control population that carries no carrier, and both have
the same shape of remedy: build the reference within the population it is subtracted from.
For c that is the same-size neutral frames; for the centre it is a per-size-class `x̄_s`, or
a centre taken from the neutral frames of each size.

Neither is in this cycle — §9 excludes label-side work and the arms are trained on the
current definitions — and together they are the first two items for the next.

**Caveat.** Three seeds, eight frames per population. The direction is unambiguous
(|80-atom| < |159-atom| in every seed) but the magnitudes carry the seed spread of a channel
whose sign is not even consistent across seeds, which is itself the §7.2 finding.

---

## The audit found a third instance, in the participation ratio itself

§7.2b said the next cycle should audit every reference for the population it was computed on.
Doing that immediately turned up a third case, and it is in the diagnostic this cycle used to
make its localisation claim.

`b13_participation.py` reports `pristine_ratio = N_eff(charged) / N_eff(pristine)` and its
docstring says the ratio "does not depend on the cell size". It cannot: the numerator is
measured on **159-atom** charged cells and the denominator on **80-atom** pristine ones,
because the dataset holds no defect-free cell larger than 80 and the script took the biggest
it had. `N_eff` counts atoms, so the reference is extensive and the ratio is inflated.

`c15_participation_reference.py` measures the size dependence rather than assuming it, by
tiling a pristine 80-atom cell to 160 (defect-free, right size), three arm A seeds:

| | s1 | s3 | s5 | mean |
|---|---|---|---|---|
| N_eff, charged 159 | 20.85 | 13.07 | 14.34 | 16.09 ± 3.41 |
| N_eff, pristine **80** (the old denominator) | 22.00 | 25.82 | 26.64 | 24.82 ± 2.02 |
| N_eff, pristine **160** (the right one) | 42.28 | 51.59 | 53.28 | **49.05 ± 4.84** |
| ratio as reported | 0.948 | 0.506 | 0.538 | 0.664 |
| **ratio, size-matched** | 0.493 | 0.253 | 0.269 | **0.339** |

**The reference grows 1.98× from 80 to 160 atoms**, against 2.00× for a perfectly
delocalised state — so the pristine frontier state is extensive to within 1%, and the
80-atom denominator inflates every published ratio by almost exactly a factor of two.

Rescaling the six-seed numbers by the measured 1.98:

| cohort | as reported | **size-matched** |
|---|---|---|
| arm A | 0.754 | **0.381** |
| Stage B | 0.619 | **0.312** |
| head-only (s7) | 0.491 | **0.248** |
| E_LR-off baseline | 0.478 | **0.241** |

**What changes and what does not.** Every cohort was divided by the same wrong denominator,
so the *comparisons* are untouched: arm A's carriers are still 1.53× more spread out than the
head-only cohort's, and the raw `N_eff` on 159-atom charged frames was size-matched all
along. The claim "arm A localises worse" stands unaltered.

What changes is the absolute reading. **Arm A's carrier occupies about a third of what a
delocalised state would, not three quarters**, and an earlier note in this file that seed 1's
`N_eff` of 25.5 was "indistinguishable from the pristine 25.2" compared a 159-atom cell
against an 80-atom reference — the right comparison is ≈ 49, and seed 1 sits at about half of
it. The carriers are localised; they are simply less localised than the cohorts before them.

That is three instances of one error — `c_shift_table`, the on-site centre, and now the
participation reference — and the third was found inside the diagnostic used to judge the
first two. The audit is not a tidying task for the next cycle; it is the next cycle's first
result.

### Audit item four: δ_L — the same error, and this time it is harmless

`bound_switch(depth) = sigmoid((depth/δ_L − 2)/0.5)` divides by the pristine frontier level
spacing, which `collect_pristine_centre` records from the **80-atom** stoichiometric cells and
which is then applied to 159-atom frames. Same pattern as the other three.

I expected a band's level spacing to go as 1/N, making the 80-atom value roughly twice the
160-atom one. **Measured, it does the opposite**, on the same tiled cells as above:

| | s1 | s3 | s5 | mean |
|---|---|---|---|---|
| δ_L, 80 atoms | 0.02077 | 0.01795 | 0.01759 | **0.01877 ± 0.00142** eV |
| δ_L, 160 atoms (tiled) | 0.02658 | 0.02518 | 0.02694 | **0.02623 ± 0.00076** eV |
| ratio 80/160 | 0.78 | 0.71 | 0.65 | **0.72×** |

So the 80-atom spacing is 0.72× the 160-atom one, not 2×; the mean spacing of a folded band
is not what `_frontier_spacings` measures. The mismatch is therefore mild and it makes
`depth/δ_L` **larger** than it should be, pushing `s` toward 1.

**And it changes nothing here.** `s` is already 1.000 to six decimal places on every model
this programme has produced, because the levels sit 10–50× above the switch's 33 meV
threshold. A 1.4× error in the denominator of a saturated sigmoid is invisible.

That is worth recording as the audit's first *benign* finding: the pattern is present, the
mechanism is confirmed, and the consequence is nil on this host. An audit that only ever
finds disasters is not being run honestly. The entry that matters is that δ_L would become
harmful the moment a level came within ~50 meV of the continuum — which is exactly the
regime the switch exists for.

**The audit so far: four instances of one pattern, three that changed a number and one that
did not.**

| # | reference | built on | applied to | consequence |
|---|---|---|---|---|
| 1 | `c_shift_table` | charged frames, no same-size neutral | both size classes | +0.75 eV of Δc |
| 2 | on-site centre `x̄_s` | 80-atom stoichiometric | 79- and 159-atom defective | 0.4–0.6 eV bulk Cl shift |
| 3 | participation reference | 80-atom pristine | 159-atom charged | every ratio inflated 1.98× |
| 4 | `δ_L` | 80-atom stoichiometric | 159-atom charged | 0.72× error, invisible while `s` saturates |

### What the image term costs, measured

Seed 1 of each arm, same four-GPU wave, epochs 0 → 20 from the run logs:

| arm | wall per epoch |
|---|---|
| A (image compensation off) | **2.21 min** |
| B (image compensation on) | **3.89 min** |

**§2.5 costs +76% per epoch.** The term adds a second head solve — the `probe` forward that
produces `alpha` and `gap` before the compensation can be built — plus an Ewald image
potential carrying a live position derivative. Both are per step, both are on the critical
path, and neither is affected by §1.1's batching, which is why the cycle's 3.4× speed-up
does not show here.

Recorded now because it is an **adoption input, not a footnote**: gate 10 asks whether the
term reduces tiling drift, and the answer has to be worth 76% of the training budget. A term
that halves the drift is worth it; one that shaves 10% is not, and the number to weigh that
against is above rather than discovered afterwards.

It also revises this cycle's schedule: arm B's two waves cost about 1.8× what arm A's did,
which is most of the slip in the completion estimate.

---

## Arm B, wave 1 — F23's Δc clause fails, and the null gate is why

Four of six seeds, scored on GPUs 6–7 while wave 2 trains on 4–5.

### Gate 7 got worse again

| type | Stage B | arm A (6 seeds) | **arm B (4 seeds)** |
|---|---|---|---|
| ss-σ | 3/6 | 5/6 | **4/4** |
| sp-σ | 1/6 | 3/6 | **4/4** |
| pp-σ | 2/6 | 2/6 | **4/4** |
| pp-π | 5/6 | 4/6 | **4/4** |

Every seed is at the stop on every integral type. L_b: 1.132 ± 0.090, 1.223 ± 0.021,
1.459 ± 0.047, 1.229 ± 0.018 Å. The image term did not relieve the hub coupling; it tightened
it. Arm C's wider bound is now the only lever left in this cycle, and the trend across three
regimes (Stage B → arm A → arm B) is monotone in the wrong direction.

### Δc barely moved, and F23's first clause fails

| | c(79) | c(159) | **Δc** |
|---|---|---|---|
| arm A (6 seeds) | +9.746 ± 0.197 | +10.789 ± 0.394 | **+1.042 ± 0.197** |
| **arm B (4 seeds)** | +10.455 ± 0.319 | +11.449 ± 0.638 | **+0.994 ± 0.320** |
| fall | | | **0.048 eV** |

F23's pre-registered clause was Δc(A) − 0.40 ≤ Δc(B) ≤ Δc(A) − 0.20, i.e. **[+0.64, +0.84]**.
Measured **+0.994** — above the band. **The clause fails**, and by a wide margin: the fall is
0.048 eV against a forecast 0.2–0.4.

**And that contradicts the calibration-stage evidence, which is the interesting part.** At
epoch 0, before any training, the image term cut Δc₀ from +0.766 to +0.417 — a fall of
**0.349 eV**, inside the forecast band, and the first direct proof the term does anything.
After 24 epochs that fall is 0.048 eV. Training erased 86% of it.

The mechanism is §0's own rule 2. Under the null gate c(79) receives no gradient and is
frozen at its epoch-0 calibration; c(159) is the only energy-referenced constant left and is
trained. So the *trained* Δc is dominated by wherever the 159 column drifts to — and it
drifts to roughly the same place in both arms, because it is absorbing whatever the energy
loss cannot place elsewhere, which the image term does not change.

**The null gate made Δc insensitive to the thing F23 asked it to measure.** F23 was written
before the null gate's consequence for the c table was understood, and it is measuring the
159 column's drift rather than the image compensation. That is a defect in the *forecast*,
not in the term: the honest reading is that Δc cannot score the image term while one column
of the c table is frozen, and the term's evidence has to come from gate 10's tiling drift —
which is what §2.5 said in the first place, and now for a third reason.

*(F23's other three clauses — R in band, thermal drift ≤ 0.3·D0, F4 holds — are scored on the
six-seed set.)*

### Gate 10 — the image term's real adoption test, and it passes decisively

Four arm B seeds × four tiles (one ideal, three thermal) × three tilings, `c3_tiling_drift`:

| | value |
|---|---|
| drift **without** the term, D0 | 0.4179 ± 0.0462 eV (0.341–0.481) |
| drift **with** the term | **0.0560 ± 0.0213 eV** (0.007–0.085) |
| ratio | **0.132 ± 0.048** — the gate allows ≤ 0.30 |
| drift removed | **86.8%** |
| pairs inside the gate | **16 / 16** |
| ideal tiles / thermal tiles | 0.122 / 0.135 |

**PASS**, at less than half the allowance, and the thermal tiles are as good as the ideal one
— which is the clause the spec added this cycle precisely because an ideal-tile-only result
would have been easy.

**The pre-registered prediction held exactly.** `s` = 1.000 at min, p25, median, p75 and max
over all 48 tiled frames. The switch never engaged, so gate 10 measured the compensation and
not the gating, as recorded before arm B existed.

### So the image term's verdict is split, and the split is informative

| evidence | result |
|---|---|
| **gate 10, tiling drift** | **removes 86.8% of a 0.42 eV size error, 16/16, ideal and thermal** |
| F23's Δc clause | fails — but Δc is blind to the term while c(79) is frozen |
| gate 7, hub stops | worse: 4/4 on every type against arm A's 5/6, 3/6, 2/6, 4/6 |
| cost | **+76% per epoch** |

The term does the one thing it was introduced to do, and does it well: tiling drift is the
size-extensivity error this whole programme exists to remove, and 0.42 → 0.056 eV on
thermal tiles is the largest single reduction in it that any cycle has produced. Against
that, it costs 76% more per epoch and it tightened the hub coupling.

**The recommendation this supports** — for the report, not a decision taken here: adopt the
term on gate 10's evidence, and stop using Δc to score it. Δc was the wrong instrument for
two independent reasons now, §8.5's (it is 97% carrier-independent) and this one (the null
gate freezes the column that would have moved). Both were discovered inside this cycle,
which is an argument for gate 10 being promoted from "the term's adoption test" to the
*only* test the term is scored on.

### Arm B, the rest of wave 1 — F4 fails outright, and it changes the recommendation

| | Stage B | arm A | **arm B (4 seeds)** |
|---|---|---|---|
| **F4, `d(δ_sr)/dd` at 159, full range** | −0.1004 ± 0.0166, **6/6 in band** | −0.0874 ± 0.0142, **5/6** | **−0.3200 ± 0.0096, 0/4** |
| pristine gap | 2.392 ± 0.016 | 2.379 ± 0.019 | 2.403 ± 0.005, 4/4 in window |
| depth from the CBM | 0.014–0.093 | 0.334 | **0.235** (0.200–0.298) |
| dilution R | 0.868 ± 0.076 | 0.925 ± 0.144 | 0.955 ± 0.000, in band |
| bound fraction | 0.615 ± 0.067 | 0.635 ± 0.023 | 0.625 ± 0.000 |

**Gate 2 fails on arm B, 0 of 4**, at −0.320 against a band of [−0.142, −0.063]. That is
3.7× steeper than arm A and 3.2× steeper than Stage B, on every seed, with a seed spread of
0.0096 — the tightest number in the cycle and the furthest outside its gate.

**The image term fixes one size dependence and breaks another.** Gate 10 measures how the
frontier potential changes with **cell size L** under tiling: the term removes 86.8% of it.
F4 measures how δ_SR changes with **hub separation d** inside a fixed 159-atom cell: the term
triples it. Both are geometry dependences and they are not the same one. The image potential
of the carrier's own images depends on the carrier's extent, and the extent changes with d —
so within a fixed cell the term rides on d far more strongly than the labels do.

**This revokes the recommendation recorded an hour ago.** On gate 10 alone the term looked
adoptable. It is not, on this evidence:

| | arm B |
|---|---|
| gate 10, tiling drift | **PASS** — 86.8% removed, 16/16, thermal included |
| **gate 2, F4** | **FAIL — 0/4, −0.320 against [−0.142, −0.063]** |
| gate 7, hub stops | worse: 4/4 on every type |
| gate 5, gap | PASS, 4/4 |
| gate 6, dilution R | PASS, in band |
| depth from CBM | 0.235, better than arm A's 0.334, worse than Stage B's ~0.05 |
| cost | +76% per epoch |

Gate 2 is a **gate**, not a report, and it fails outright — the only cohort in the programme
to do so. A term that removes 87% of the cell-size drift while tripling the hub-separation
slope has not been shown to be an improvement; it has been shown to move the error from one
axis to another. **The honest recommendation is: do not adopt, and find out why the two
dependences disagree.**

That is a more useful outcome than adoption would have been. The cycle now has a term with a
large, measured, *specific* defect — F4 at −0.320 with a 0.0096 spread is about as clean a
target as a diagnosis ever gets — rather than a term that passes everything asked of it and
is carried forward on faith.

**Six seeds will confirm or revise this.** The wave-1 spread is 0.0096, so a reversal is
unlikely, but the verdict is written against four seeds and says so.

### Why F4 fails — measured, and it is entirely the term

`c19_image_slope.py`: the same four trained arm B models, scored twice, with
`image_potential` patched to return zeros in the second pass. No weight is touched.

| | F4, `d(δ_sr)/dd` at 159 | in band [−0.142, −0.063] |
|---|---|---|
| arm B, **term on** (as trained) | **−0.3200 ± 0.0096** | **0 / 4** |
| arm B, **term knocked out** | **−0.0761 ± 0.0029** | **4 / 4** |
| **the term's contribution** | **−0.2439 eV/Å** | |
| arm A, for comparison | −0.0874 ± 0.0142 | 5 / 6 |

**The image compensation contributes −0.244 eV/Å to the hub-separation slope — 2.4× the
entire label slope the gate is measured against.** It is the whole of the excess, exactly and
with no residual: knock it out and arm B's head lands at −0.0761, inside the band, on every
seed, with a tighter spread (0.0029) than arm A's (0.0142).

**Two things follow, and the second is the useful one.**

First, the head arm B learned is not damaged — it is arguably the best-behaved head in the
cycle on this measure. The failure is not "training with the term produced a bad head"; it is
"the term's own additive contribution at evaluation is far too steep in `d`".

Second, **that makes the defect a magnitude problem, and §2.5 has no magnitude to adjust.**
The term is applied at full strength, `comp = −s · φ_img / ε∞`, with no amplitude anywhere.
Scaled by about 0.1 it would contribute −0.024 eV/Å, F4 would land near −0.10 — inside the
band — and the question would become how much of gate 10's 86.8% drift reduction survives
that scaling. A prefactor on the image compensation, fixed or learned, is the obvious next
experiment, it is **not** among §9's exclusions, and this measurement gives it a starting
value rather than a search.

That is the difference between "the term failed a gate" and a target: the excess is
−0.244 eV/Å, it is all of it, and it has a knob that does not exist yet.

**The prefactor is not explored in this cycle.** `c20_image_scaling.py` is written and
committed — it wraps `image_potential` with a scale and re-measures the tiling drift, so the
drift-against-λ curve can be had cheaply — but it was stopped before producing numbers and
the amplitude question is deferred as a decision, not answered here. What this cycle records
is the measurement that makes the question askable: the term contributes −0.2439 eV/Å, the
head alone is at −0.0761, and if the response is linear the band's edge sits near λ ≈ 0.27.

---

## Arm B, six seeds — the four-seed reading holds

Gate stage 17:18 → 17:39, then F10, gate 9 and `post_gates armb` to 18:06.
Full table in `~/runs/armb_gates.md` and in the report's §5.

| gate | arm A (6) | **arm B (6)** |
|---|---|---|
| 2 F4 | −0.0874 ± 0.0142, 5/6 **PASS** | **−0.3112 ± 0.0151, 0/6 FAIL** |
| 3 Δc | +1.042 ± 0.197 | +0.876 ± 0.314 |
| 4 F10 | 1/6 FAIL | 0/6 FAIL |
| 5 gap | 6/6 PASS | 6/6 PASS |
| 6 R | 0.925 ± 0.144 PASS | 0.958 ± 0.090 PASS |
| 7 stops | 5/6, 3/6, 2/6, 4/6 FAIL | **6/6 on all four FAIL** |
| 8 `N_eff` | 18.79 ± **3.62** | 19.21 ± **0.83** |
| 9 79-atom slope | −0.1212 ± 0.0268 **PASS** | **−0.3422 ± 0.0249 FAIL** |
| 10 tiling | not applicable | **24/24 PASS**, 0.133–0.173 of D0 |
| depth from CBM | 0.334, shallow 5/6 | **0.234 ± 0.049, shallow 6/6** |
| §2.2 max \|δZ\| | 0.585 ± 0.068 e | 0.663 ± 0.075 e, none at the stop |

Everything the four seeds showed survives, and two things sharpen.

**Gate 9 fails too**, which the four-seed set had not shown: the 79-atom matched slope is
−0.3422 ± 0.0249 against arm A's −0.1212 ± 0.0268. Gates 2 and 9 fail for the same measured
reason, and the knockout has already attributed −0.2439 eV/Å of it to the term.

**The term stabilises what §2.1 destabilised.** `N_eff` spread 3.62 → 0.83 at the same mean,
and the level 0.334 → 0.234 eV with all six seeds shallow rather than five. The revived
on-site channel made localisation seed-dependent and the level deep; the image term undoes
much of both. It does not rescue the verdict — gate 2 and gate 9 are gates — but it is the
strongest reason in the cycle to think the prefactor experiment is worth running.

**F23 final: fails, 2 of 4 clauses.** Δc fell 0.166 eV against a required 0.20–0.40 (fails);
R 0.958, 6/6 inside (passes); thermal drift 0.133–0.173 of D0, 18/18 (passes); F4 0/6
(fails). All four were required. The Δc clause was never evidence about the term (§7.1b);
the F4 clause was, and it is the one that decided it.

`s` = 1.000 at every quantile over all 72 tiled frames — the pre-registered prediction, held
exactly, across three times the frames it was written against.
