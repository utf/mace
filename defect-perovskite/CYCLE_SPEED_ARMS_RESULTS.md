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
