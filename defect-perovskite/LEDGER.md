# Ledger — closures and open items at the Madelung-in-H direction change

Opened 2 Sep 2026. §9 of the build plan requires these entries **verbatim** in the final
report, so this file is their canonical home. Append, never rewrite.

---

## Closures

**M2 cancelled** (direction change). The A/B ON-arm fit gain is closed **unattributed**.

What that gain was, for the record: axial_red +0.348 → +0.588, no negative cell where OFF had
one at −0.464, rmse_nbhd 47.7 → 43.5, across six seeds per arm. M3 then showed it was not
bought through the hub coupling (|t′| indistinguishable between arms, both far below target;
corr(N_eff, d) flat in both). M2 would have said *where* it was bought. It will not be run, so
the answer is not known and must not be asserted.

**One-manifold programme archived.** Superseded in full: C1, the M3 extensions, the
consistency triangle, the sign A/B observables, `channel_sign` / `s_c`, `μ_c` and
`init_mu_for_gap`, and the soft gauge anchor.

The anchor's retirement has a reason worth keeping: the counting head's
`E_head = F(N) − F(N_neutral)` is **not** invariant under a uniform ε shift, so the energy
labels themselves pin the absolute edge into `eps0`. An anchor would be a second, weaker
statement of something the loss now enforces exactly.

Nothing was in flight at the change (both machines idle, verified). Outputs archived to
`~/runs/archive_one_manifold/` on b3 and locally; the model checkpoints in
`~/runs/tbv3_models/` and `~/runs/ab_models/` stay in place as D-1/D-2 inputs and are
archived programme members regardless — no result of theirs is evidence for the new
architecture.

**Response channel retired**, with its two measured legacies recorded:

* **clause 1 — reach confirmed.** The mechanism does give a compact state the long force
  footprint; electrostatics has the reach a short-ranged tight-binding head cannot.
* **clause 2 — site selection absent.** It did not make the head prefer the hub.

Both motivate the variational move of Edit 1: the same physics, but inside `H` where it can
select a site, rather than in a bolt-on energy readout where it demonstrably could not.
Retirement lands **in the same commit as Edit 1** (`85d195b`) — never both active, which
would double count the carrier's electrostatics.

*Consequence, verified rather than assumed:* every archived checkpoint in `~/runs/ab_models/`
and `~/runs/tbv3_models/` pickles a `CarrierResponse` instance, so on post-`85d195b` code
`torch.load` raises `ModuleNotFoundError: mace.modules.defect_response`. D-1 and D-2 ran
before the deletion and are unaffected; the arch and base checkpoints never had the channel
and still load. A future reader needing one of these should write a throwaway stub, not
restore the module.

**Cancelled with the programme, not to be run:** M2 attribution, C1 post-μ-fix, the F2
closure-ratio update.

---

## Open items

**The λ–d anomaly.** `corr(λ, d) < 0` in 10 of 12 A/B cells (−0.31 to −0.57): the defect level
falls as the vacancy-flanking pair separates, so `Δ_bind` *rises* with d. That is the reverse
of the physical expectation. It is equally present in both arms, so it is not a sign effect.

Carried forward as **expected to be probed by D-2**: if the anomaly localises to the resonant
bin (`d/δ_L < 1`), it is recorded as explained-by-resonance pending Edit 4. §7 makes it a stop
condition — if `corr(level, d)` is still anti-physical **on bound frames** under the counting
head, stop and investigate before R3.

**D-2 has now run, and the resonance explanation is refuted (2 Sep 2026).** Measured on the
frames each model's own spectrum marks bound (`depth/δ_L > 2`), `corr(λ, d_hub)` is negative
in **17 of 18** models, mean −0.422; and **13 of the 18 models have no resonant frame at
all** while still showing it. The anomaly is not a near-degeneracy artefact.

**Stage 1 then changed its sign (2 Sep 2026, same day).** Three cohorts, same measurement:

| cohort | mean `corr(λ, d_hub)` |
|---|---|
| archived (T-B edge+gap loss, no Madelung) | −0.422, negative in 17/18 |
| Stage-1 **OFF** (force-only loss, no Madelung) | +0.020 |
| Stage-1 **ON** (force-only loss, Madelung) | +0.422, positive in 5/6 |

Two statements, deliberately kept apart. The **loss change alone** removed the negative, so
the anomaly was never purely architectural — it was at least partly a property of training
against the band-edge constraint. The **Madelung term** then drives the correlation positive,
which is the physical direction: the level is deepest when the pair dimerises. That
comparison is one edit apart and is clean.

**CLOSED 2 Sep 2026, on the coadvisor's call.** Recorded verbatim as the closure:

> Removed by Edit 1 + the loss change; archived-cohort mechanism unattributed; `Δ_bind`
> usable as depth in the counting head, revisit only if the sign recurs there.

The reasoning: the anomaly failed to survive either change, is positive in **10/10** under
Stage 2 — exactly the conditions where the superatom escape is shut — and strengthens as the
model gets more physical. That is the profile of an artefact of the retired training setup,
not of the architecture going forward. Explaining a dead configuration's pathology is the
side-quest class to avoid.

§7's stop condition still stands for the counting head: if `corr(level, d)` comes back
anti-physical on bound frames there, stop and investigate before R3.

---

## New open item: `Z`'s scale is not identified (2 Sep 2026)

Stage 1's ON arm is bimodal and splits by `|Z|`. Two of six seeds inflated the species
charges to ~2.5× nominal — (−2.6, +2.3, +5.4) against (−1, +1, +2) — and are the two worst
fits by a wide margin (axial_red +0.17, +0.18 against the other four's +0.63 to +0.72).
Neutrality holds to 1 part in 10⁷ on every seed, so the projection is exact; what is missing
is any constraint on the **scale**.

Cause: the smooth part of `phi_LR` is dominated by a per-species constant that the learned
on-site term can also produce, so only the small defect-induced deviation constrains `|Z|`.
The plan's claim that the real-space part's absorption "is what makes Z identifiable" is
right in principle and too weak in practice.

**Ledger note on ε∞, from the §0 verification.** Nothing ties the screening amplitude `a` to
the static dielectric constant — there is no static-dielectric literal anywhere in the
repository, and `a` is initialised to `1/√ε∞` from `eps_inf_init` alone. But the retained
arch and base checkpoints (`r2_h3_anneal_s5`, `e0_base_s1`) carry **`eps_inf_init = 6.5`**,
not the plan's 4.0, with `freeze_amplitude = False`. It is inert in those models —
`use_long_range = False`, so `a` never acted — and the launchers (`run_arm.sh`,
`run_perovskite.sh`) both default to 4.0, so the 6.5 entered by environment override on a run
whose provenance is not recorded.

Consequence for the build: **Stage 1 must not read `model.eps_inf_init`**, or the Madelung
screen silently picks up 6.5 while E_LR's amplitude uses 4.0. ε∞ is threaded as one explicit
constant through the forward context.

---

## Retained measurements (not superseded)

Test 2 (R_DFT ≈ 0.95); the E0 footprint; the D1 decomposition; M1 and M1b (clean subset =
159-atom, slope −0.134 eV/Å); the F1 and F2 measurements. Per-fold bases remain available as
optional low-priority evaluation nulls.

---

## Superseded decisions and qualifications, 2 Sep 2026 (recorded verbatim)

**Withdrawn (ours):** the T_el warm start as the failure-to-start fix. At an atomic-limit init
the bond order P_ij = 0 at *any* smearing, so smearing does not create the missing gradient.
Wrong mechanism; sixth entry in the forecast-phenomenology class.

**Superseded (ours):** Q2 answer (a) accept-the-detach. With the P-backward available at
O(n³), (b)-now is strictly better and is also what E_LR needs.

**Confirmed cause of the lr anomaly:** biased gradient (frozen P) + pathological random init
(atomic limit), not a property of the counting head. lr 0.01 is the target again.

**Qualification 1:** "frozen-P training cannot relocate the carrier" is overstated — each
forward recomputes P, so relocation happens by drift, which is why Stage 3's converged seeds
learned at all. Accurate form: relocation-from-forces is absent from the *gradient*.
Conclusion unchanged: build the exact backward before the joint run.

**Qualification 2:** in the tiling test, learned matrix elements are bit-identical; Madelung
phi matches only to numerical tolerance across G-grids. Test accordingly or the assertion
fails for the wrong reason.

**Forecasts on record, before results.** F1: the NaN trips assertion 3 or 4 of the triage
ladder. F2: with P-backward + Harrison init, 6/6 seeds train at lr 0.01, init gate rarely
fires. F3: N_eff <= control at matched force fit in >= 5/6 seeds, read at matched pristine
bandwidth. F4: counting-head dE_head slope on the 159-atom subset right sign and within 3x of
-0.134 eV/A. F5: corr(lambda, d_hub) stays positive.

---

## Record, 2 Sep 2026 (verbatim)

**+0.987 correlation, final form:** state it from what the control demonstrates directly --
the bounded s-only head fits *only* by delocalising (its two best seeds are its two most
delocalised, N_eff 64-69) -- not from any inferred correlation. The double retraction is
logged; the claim no longer rests on a regime-dependent number.

**New standing rule (regime tagging):** any claim measured in a single optimiser regime is
tagged with that regime in the report and does not transfer without a cross-regime check.

**Superseded numbers:** the Stage-1 spectral-head lambda-d turnover (+0.42/+0.27, lr-0.01
regime) is superseded by F5 on the counting-head rerun models once measured; ledger item stays
closed either way. The Stage-2 "s-only slope -0.0035, right sign 2/2" is marked **unreliable**
(superatom-contaminated regime); nothing downstream uses it.

**Retired diagnostic:** pristine split fraction (reference 1/(n_states-1) is not comparable
across heads). d/delta_L is the quantity of record.

**Float32 lesson:** the head runs float64 permanently; construction-time assert on dtype
(occupation arguments ~700 and mu-bisection to 1e-10 are outside float32 by construction, not
by accident).

## Entry 7 (replaces the previous #7) — the full-sum Madelung convention

Recorded verbatim from the plan of 2026-09-02.

**Convention settled: full-sum kernel.** The infinite periodic ion lattice is one set of
charges under any supercell description; the site potential excluding only the true
self-term (j = i, R = 0) is description-invariant, and only its partition into
"in-cell" vs "image" depends on the box. The reductio "A_ii varies with L ⇒ ε_i is
description-dependent" fails because the j ≠ i sum varies compensatingly. The images
of ion i are real atoms; a carrier sitting on atom i feels them; A_ii·Z_i belongs in
H. Under the subtraction convention the host φ entering H is *not* the periodic
potential the labels saw — it is that potential minus a supercell-dependent fraction
of a sublattice, which was the defect.

**Ledger (analyst's, rewritten):** the structural error of the last cycle was the
*capitulation* — endorsing the description-dependence reductio, with "verified point
by point", without running the compensation check — not the original two-density
resolution, which stands. Entry replaces the previous #7.

**Statement of record (assert, never implement around):**
"Image corrections enter through exactly two places: E_LR's periodic/isolated switch
and the per-(charge, size) reference constants. H is gauge-invariant — the same
periodic ion-lattice potential in training and in isolated evaluation. There is no
separate carrier–host term; the interaction is Σ_i (P − P_ref)_ii ε_i through φ_LR,
with forces by Hellmann–Feynman."

### What this invalidates, and what it does not

Every number produced with `phi_LR` under the subtraction convention was computed with a
supercell-dependent host potential. That is Stages 1-3 and every gate scored on their
models. It does NOT invalidate the section-1 wiring evidence (the density response is a
statement about the gradient, independent of the kernel), the tiling test's element and
spectrum clauses, or the F4 head-versus-base discrimination (both slopes were measured
through the same kernel). It DOES mean the six wired models must be retrained before their
gate numbers stand — which is section 5.

### The measurement that settled it

`self_potential_of` returns, to ~1.5% at every size, the Makov-Payne potential of a point
charge in jellium:

| L (A) | 5.6 | 11.2 | 16.8 | 22.4 | 33.6 |
|---|---|---|---|---|---|
| kernel A_ii | -7.320 | -3.666 | -2.450 | -1.843 | -1.235 |
| -alpha_M C / L | -7.297 | -3.648 | -2.432 | -1.824 | -1.216 |

A Gaussian self-energy would add a constant +11.49 eV/e at sigma = 1 and is absent. So what
was being subtracted was entirely ion i's own periodic images, and the plan's instruction to
"keep the Gaussian self-energy removal" has nothing left to remove: LES's k != 0 sum carries
no self term. `test_madelung_convention.py::TestKernelCalibration` pins this identity so the
claim is checked rather than remembered.

---

## Entry 8 — the forward-only battery: F4 decomposed, no rerun spent

**Closure.** F4's "unexplained amplitude" is closed as a mystery and replaced by a
decomposition in which every factor is measured. The three that matter:

1. **Label availability.** 98.4% of the charged frames are 79-atom. Their energy residual's
   d-slope is +0.3641 over the full range and +0.0806 [−0.1108, +0.2720] inside the window
   where the base's own training set is dense — consistent with zero. The carrier-free neutral
   null at the same size gives +0.1317 [+0.1131, +0.1503]: the small-cell "trend" is base
   error, and it points the opposite way to the physics.
2. **Loss coverage.** The Stage-3 objective is forces plus `loss_gap`. `delta_sr` carries no
   energy term, so F4 has never been a fitted quantity in any Stage-3 run. In the fitted
   channel the head tracks the labels at both sizes, sign flip included: axial force slope
   +0.2641 ± 0.0220 at 79 atoms against a label +0.4202, and −0.3635 ± 0.0495 at 159 against
   −0.1901.
3. **Coupling.** The direct hub Pb–Pb bond carries 70% of the head's d-response (deleting it
   leaves 30%), and the envelope supplies t/t_Harrison = 0.183 ± 0.112 there, with the learned
   pair modulation pinned at its bound on 33% of hub bonds.

**The reference survives its own null.** `d(E_label − E_base)/dd = −0.1338 [−0.1446, −0.1230]`
on the 17 charged 159-atom frames; the neutral cells at the same size, same base, same
geometries give +0.0800 [−0.0503, +0.2102]. The gate rests on carrier physics.

**Closure, negative and load-bearing.** F7 fires as a diagnosis and its remedy is
contraindicated. At Harrison initialisation, every longer-ranged envelope LOWERS the level's
d-slope monotonically — exp L = 1.0 gives dλ/dd = +0.3211, L = 1.4 gives +0.2268, L = 2.0
+0.1172, the d^-2 power law +0.1242, L = 3.0 +0.0575 — because dt/dd = t · dln t/dd and the
exponential's log-slope is three times steeper, while the denser spectrum a longer envelope
produces costs more than the coupling buys. The current setting is already the maximum.
**No six-seed rerun was spent.**

**Closure.** The +2.1 eV move in λ_frontier between the γ = 1 and γ = 3 cohorts is a rigid
gauge shift: depth below the conduction manifold changed by 25 meV, from +0.1277 ± 0.0235 to
+0.1025 ± 0.0259 eV. The earlier reading — "the widened bound letting the on-site correction go
where it was pinned" — is retracted.

**Open, and now precisely stated.** The on-site correction channel is INERT, not merely
unsaturated: within-shell spread of order 1 meV, every species within 0.01 pre-tanh of every
other, a near-uniform +0.27 eV on every atom. A uniform on-site shift contributes exactly zero
force, and the Stage-3 loss is forces plus a pristine-gap term, so the constant mode is an
unidentified gauge direction by construction. Deferred to R3 per the decision tree. Under the
joint run's energy loss it becomes degenerate with `c_shift` rather than with nothing, so it
partially self-resolves there.

**Open.** Beyond-two-centre / superexchange. The hub ablation leaves 30% of the d-response in
the indirect channel, which is where this would live. It is the last item on the
close-or-explain page.

**Statement of record (assert, never implement around):**
"A decision tree's branch is a hypothesis about the remedy, not an instruction. F7's branch
presumed the envelope was the lever for the d-slope; b7 is the measurement the branch asked
for and it falsified the presumption. Executing the change anyway would have been following the
letter of the tree against the evidence the tree called for."

### The bug the end-to-end run found that no unit test could

`--defect_base_init` and `--defect_madelung_on_site` could not be combined at all — the joint
run's own configuration. `CORRECTION_PREFIXES` omitted `madelung`, so `_base_state` classified
the learned species charges as base weights and `load_stage_a_base` demanded a Stage-A
checkpoint contain `madelung.z`, which no Stage-A base will ever have. Every unit test passed
throughout, because each exercised one flag. `defect_protocol.trainable_mask` already worked
around the same omission with an explicit `startswith("madelung.")`, so the codebase knew the
answer in one place and tripped over it in another.

Separately: `tests/unit/test_spectral_v3.py` imported a sibling by bare name while
`tests/unit/__init__.py` makes the directory a package, so it raised at collection and its
twelve tests had been silently skipped whenever the suite was collected by directory.

---

## Entry 9 — the modulation ceiling, and the joint run's leakage detector

**Closure (F11).** The hub-bond modulation's stop binds in **two of six seeds**, not in a third
of the bonds: in those two it is 100% of hub bonds in every distance bin, and in the other four
nothing is within 0.79 of a stop. That reconciles b3's pooled 33% with the per-seed picture and
falsifies "concentrated in short-d bins" -- it is a per-seed phenomenon, not a per-distance one.

**And both stops are in use on the same bond.** ss-sigma sits at the LOWER bound while pp-sigma
and pp-pi sit at the UPPER one: the head wants less s-s overlap and more p-p overlap than the
bound allows, simultaneously. Any uniform scale factor on that bond therefore helps one channel
and hurts another, which is a structural reason the what-if is a blunt instrument and the
per-channel bound is the finer one.

**Closure (F12), on the registered sign-and-direction rule.** Scaling the hub integrals by
x1.25 lowers the 79-atom force loss (-0.8%, in 4/6 seeds) but moves F4 only to -0.0675 against
the -0.08 the rule required. First clause met, second not: **F12 FAILS**, the stop is not the
lever, and the joint run proceeds on the current bound. No head-only rerun was spent.

Two facts survive the failed forecast. F4 moves monotonically toward the reference under
scaling in EVERY seed, so the coupling is the channel -- it is simply not reachable by widening
a bound that only two seeds are pressed against. And the 79-atom force loss has a shallow
minimum near x1.25 and rises 5.5% by x1.5, so the small cells actively prefer a hub coupling
close to what the head already has.

**Open, and now first on the R3 page:** beyond-two-centre / superexchange. The hub ablation
leaves 30% of the d-response in the indirect channel and both coupling levers -- global
envelope (b7) and local modulation ceiling (b9) -- are now closed as remedies.

**Statement of record (assert, never implement around):**
"With the base unfrozen, M1b's +0.36 eV/A small-cell artefact can be removed two ways: the base
learns the long-d region it used to extrapolate into, which is the point, or it absorbs the
carrier, which improves every aggregate number and destroys the decomposition. Both look like
success in the loss. They differ only in whether the null-cleared -0.134 stays put when measured
against the trained model's own base branch. That slope is the detector, and shrinkage toward
-0.06 is not adopted regardless of total fit."

### What the joint run was given, and what it was not

Launched with both size upweights at 25% of their own populations' force loss -- charged large
cells because they carry the only bound-versus-band measurement, neutral large cells because
they are the base's only direct constraint at large d. Raising the charged seventeen alone would
have asked the correction to absorb a base error the base was never given the chance to fix,
which is the leakage the detector tests for.

Recorded so the numbers are readable later: the TRAINING set holds 16 charged and 15 neutral
159-atom frames; the remaining one of each is in the validation file. The seventeen-frame
references (b1, and the -0.134 itself) are measured on train + valid.

### The bit-identity result, stronger than asked

Two identical invocations of the production trainer at the final config gave losses agreeing to
**0.000e+00** across both epochs -- not "within the scatter-atomics floor" but exactly equal. In
float64 with cuEq off this trainer is deterministic, so any later difference between runs is a
change rather than noise. That is worth more than the tolerance the section-3 plan asked for,
and it was established before eight seeds were spent.

---

## Entry 10 — Stage B was normalising its trunk wrongly, and the c-shift found it

**Closure, and it is a real bug rather than a convention.** `avg_num_neighbors` divides every
message in the MACE trunk and is a plain float on each interaction block -- not a parameter,
not a buffer, absent from `state_dict`. So `load_stage_a_base`'s name-and-shape check could not
see it and its copy loop could not carry it. Stage A trained at r_max = 5.0 without a carrier
head and got 14.08; any run with the head builds its graph at the CARRIER cutoff of 10 A and
computed 112.5 on the same data. Eight times the divisor on every message, in the branch whose
whole purpose is to be the reference the correction is defined against, and the loader reported
success.

**How it was found, because nothing else was going to.** The deterministic c-shift was
introduced so the head's energy zero would not depend on the shuffle. It did something more
useful: it disagreed with the harness by a factor of four on identical data. `c_shift` is the
median of `(E_label - E_base - E_head)/Delta_n`, so it is a direct read on E_base. Every other
candidate was measured and eliminated -- frame selection (raw ratio +3.756 first-48 against
+3.587 across the file), the forward path (ForwardContext against batch.to_dict(), agreeing to
0.000000 eV), the referencing (-1.200 eV exactly), the head initialisation (0.012 eV), the
atomic energies (identical, and cancelling to -0.002 eV on this composition). The trunk
normalisation was what was left.

**Two repairs, and they cross-validate.** A Stage-B run inherits the checkpoint's value, warns,
and refuses outright if the block counts differ. A from-scratch run -- the joint run's own arm
B -- has nothing to inherit, so the count is rescaled by the measured edge-count ratio at r_max,
averaged over batches, rather than by a (10/5)^3 volume argument the periodic per-frame graph
does not obey. The rescaling returns 14.095 against Stage A's 14.083, 0.09% apart, by a route
that shares no code with it; 1/ratio = 8.01 against the 8x the two cutoffs predict.

**And the quantity that exposed it certifies the repair.** The trainer's c-shift fell from
+36.5421 to +8.9600 over the same 944 frames, against b12's independent +8.94 on a stride sample
and the harness's +8.70 on its first forty-eight. Predicted before it was measured.

**Scope, checked rather than assumed.** The arch the harness builds from carries 14.14,
essentially Stage A's 14.08, so the Stage-1..3 lineage and every number in the battery stand.
The fault is confined to the production trainer, which had never been run end to end with a
carrier head and a Stage-A base before this cycle -- which is exactly what section 3 was for.

**Statement of record (assert, never implement around):**
"A value that changes what a module computes and does not appear in `state_dict` is invisible to
every check built on `state_dict`. `avg_num_neighbors` was one. Any future constant of that kind
-- a divisor, a cutoff, a normalisation held as a plain attribute -- must be carried explicitly
across a stage boundary or recomputed from the same definition on both sides, and a test must
assert the premise that it is invisible, so that if it later becomes a buffer the special case
is known to be redundant rather than quietly wrong."

### The operational lesson, three times over in one night

A liveness check is not the condition you want. Three faults of one shape: two `pgrep -f`
patterns that matched the shell carrying them, and a post-run chain that waited on a PROCESS and
so read a deliberate relaunch as the end of the run -- aborting the unattended scoring that was
its entire purpose. All three now test the condition itself: bracket-quoted patterns, and a
completion marker in the log rather than a process in the table.

---

## Entry 11 — the joint run leaked, F14 confirmed on 8 of 8, nothing adopted

**Closure.** The joint run was the end of the plan and it ran to completion: six Stage-A seeds
and two from-scratch seeds, 20 epochs, float64, both size upweights at 0.25, E_LR from epoch
12, protocol on, on-site channel zero-initialised, scored by code committed before any joint
model existed. Regime tag: joint `DefectLoss` (energy + force + gap), lr 0.005, base at 0.1x,
γ = 3, Gaussian 0.05, exp envelope L = 1.0, linear hop form. The charged 159-atom residual
energy slope, measured against each trained model's own base branch, is −0.0713 ± 0.0076 on
the Stage-A arm and −0.0843 ± 0.0016 from scratch, against the null-cleared reference
−0.1338 [−0.1447, −0.1232]. Every seed's interval lies above the −0.10 floor. **F14 confirmed,
8 of 8; adopted 0 of 8.**

**What that is, said plainly.** The base network was allowed to train alongside the correction
head with energy terms in the loss, learned the distance-dependent energy itself, and the
carrier's energy signature largely left the residual. The fit is better by every aggregate
(arm A 4.5 ± 0.4 meV/atom, 12.4 ± 0.2 meV/Å on validation; forces halved from the start of
training; gaps 2.39–2.43 eV; the level still a shallow donor). That is the reason rule 2 exists
and the reason criterion 1 was written on the residual slope: a base that removes M1b's
+0.36 eV/Å artefact by absorbing the carrier passes every other test.

**Forces did not leak; energies did.** The charged 159-atom force slope held (−0.220 against
−0.190), and the neutral 159-atom force slope, +0.064 of base error at the pre-joint null, fell
to +0.006…+0.015 — the base learned the long-d region it used to extrapolate into, which is
what the run was for. The force channel had the large cells upweighted; the energy channel had
no equivalent protection and is the channel that lost the signal. The next attempt is a
decision about the objective, not the head: energy-channel large-cell weight, or a frozen base
while the head fits energies, or both.

**Both arms leak, so this is not a staging result.** The pre-registered reading for a leak in
one arm and not the other was "a property of initialisation". From scratch leaks slightly less.
The Stage-A start is not what lets the base take the carrier; the objective is.

**Criterion 2 is left unresolved, as the reading rule required.** Against the registered
out-of-fold +0.0800 the neutral slope grew on every seed; against the like-for-like production
base +0.0968 the pooled +0.1101 ± 0.0122 with per-seed intervals ±0.06 wide is not resolved.
The composite `adopted` boolean is a summary of the six-row table and not the verdict, and the
scorer's `c1_leakage` flag is named backwards (true = inside the reference interval = no leak);
recorded, not patched after the fact.

**Zero-init did not hold.** Every seed started with the on-site channel at exactly zero and
ended with +0.09…+0.40 eV species constants of within-shell spread ≤ 0.026 — b4's gauge, grown
back under a loss that contains energies. F10 restated on that channel fails by 4.8 meV. The
centred correction goes to R3 carrying evidence from both objectives.

**The participation split is three and three, and E_LR is part of it.** a1–a3 end near 11
and a4–a6 near 5.5, and the three delocalised seeds are the three shallow ones by s3_dilution's
depth (0.011–0.015 against 0.059–0.063 eV). Seed 1 with E_LR off, run locally, is identical to
joint_a1 to every printed digit through epoch 12 — the same seed, data order and intermediate
state, so the divergence after that is E_LR alone: 7.80 → 7.75 without it, 9.24 → 11.14 with
it. The E_LR-on arm may not be converged at 20 epochs. LEDGER_PARTICIPATION_SLOT

**The modulation ceiling on the joint models.** No bond of any type is at its stop in any seed;
the ×1.25 what-if raises the 79-atom force loss by 24% in 6/6 and leaves F4 at −0.067. F12
fails both clauses here, more cleanly than on the pre-joint cohort. Superexchange stays first
on the R3 list.

**Statement of record (assert, never implement around):**
"A joint objective that carries energy terms and protects only the force channel at large cell
size will hand the carrier's energy signature to the base. The detector is the null-cleared
charged 159-atom residual slope measured against the trained model's own base branch, and it
is scored before the fit is looked at. Any future joint run registers that slope, its floor,
and the large-cell weight of *every* channel in the loss, before launch."

### The operational lesson, a fourth time

The E_LR-off control queue waited on `pgrep` for the post-run chain. The chain was killed and
relaunched for the dtype fix, and in that gap the control saw no process and started, concurrent
with the scoring it was meant to follow. Inside the four-GPU cap by timing, not by design, in a
script written after entry 10's lesson. The condition it should have tested was the `post-run
complete` marker. Written into the script header, and the rule is unchanged: test the
condition, not the liveness of the thing that produces it.
