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
