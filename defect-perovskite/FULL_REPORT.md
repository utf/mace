# Full-sum Madelung kernel: sections 0–6 — report

V_Cl⁺, orthorhombic CsPbCl₃ (Mosquera-Lois & Walsh, PRX Energy 4, 043008, 2025; PBE
scalar-relativistic labels). **Regime tag on every trained number: lr 0.01, AdamW, 60 epochs,
frozen Stage-A base, forces-only + loss_gap.** Nothing here transfers to another optimiser
regime without a cross-regime check.

---

## §0 — record

Entry 7 of `LEDGER.md`, replacing the previous #7, carries the convention statement, the
analyst's rewritten entry and the statement of record verbatim. What it invalidates: every
number computed with `phi_LR` under the subtraction convention — Stages 1–3 and every gate
scored on those models. What it does **not** invalidate: the §1 wiring evidence (a statement
about the gradient, independent of the kernel), the tiling test's element and spectrum
clauses, and the F4 head-vs-base discrimination (both slopes measured through the same
kernel).

---

## §1 — the kernel

`phi_LR` is now the full lattice sum with only the true self term excluded.
`site_potential` **refuses** a `self_potential` argument rather than ignoring it.

**One deviation from the plan's letter, in service of its content.** The plan says to keep
the Gaussian self-energy removal (+11.49 eV/e). There is nothing left to remove.
`self_potential_of` returns the Makov–Payne potential of a point charge in jellium:

| L (Å) | 5.6 | 11.2 | 16.8 | 22.4 | 33.6 |
|---|---|---|---|---|---|
| kernel `A_ii` | −7.320 | −3.666 | −2.450 | −1.843 | −1.235 |
| `−α_M C / L` | −7.297 | −3.648 | −2.432 | −1.824 | −1.216 |

~1.5% at every size. A Gaussian self-energy would add a constant +11.49 eV/e and swamp the
table, so LES's `k ≠ 0` sum carries no self term and what was being subtracted was entirely
ion i's own periodic images. Now a standing test rather than a remembered claim.

---

## §2 — test table

| | test | status |
|---|---|---|
| (a) | `(AZ)_i` invariant under three tilings, 1e−8 | **pass** |
| (a′) | counterfactual: retired convention still drifts ~1.3 eV | **pass** (so (a) is not vacuous) |
| (b) | kernel vs `−α_M/L` | **pass**, ≤3% at five sizes |
| (b′) | φ shares E_LR's object *by identity* (`E = qᵀAq/2`) | **pass**, rel 1e−10 |
| (c) | prohibition on a constant-difference test across **defect** sizes | recorded in-suite with the `O(Q/εL)` reason |
| (d) | 79↔159 pristine on-site equality | **pass**, strict |
| (e) | `changed_level_index` property test | **pass**, 5 cases |
| — | no isolated-mode branch in the φ path | **pass**, enforced against the **AST**, not a text grep |

The old `xfail` is now a strict pass for the right reason: not a loosened tolerance, but the
description-invariant object being computed.

**126 → 142 tests pass** across the defect suite and flag plumbing.

---

## §3 — plumbing and bit-identity

Five flags through `arg_parser`, both `MACEDefect` construction sites and the launcher, via
one shared `_defect_madelung_kwargs` that **refuses** the counting head without the spectral
branch and the Madelung term without a composition. No P-backward flag exists to plumb: the
density response is requested structurally (training ∧ compute_force), so no configuration
can silently train the frozen-P gradient.

**The gate found a defect on its first run.** `extract_config_mace_model` read
`model.spectral.smearing` and `.t_min` by plain access; the counting head has neither, so a
Stage-3 model **could not be round-tripped through the production path at all** — could not
be saved and reloaded, and the cuEq conversion performs exactly that round trip at the end of
every run.

| | cross-path | same-model repeat floor |
|---|---|---|
| eps (site) | **0.000e+00** | 0.000e+00 |
| delta_sr | **0.000e+00** | 0.000e+00 |
| energy | 3.05e−05 | 9.16e−05 |
| forces | 1.01e−06 | 8.05e−07 |

The head's own outputs are exactly identical. Energy and forces sit inside the floor from
running the *same* model twice (MACE's scatter reductions use atomics); the floor is measured
in the same script, so a real 1e−6 difference cannot be dismissed as noise and noise cannot be
reported as a difference. **`SECTION 3 BIT-IDENTITY: PASS`.**

---

## §4 — forward-only, on the six pre-correction models

Read-only. These are *what the old weights predict through the corrected kernel* — the
weights were fitted under the subtraction, so `eps0`, `Z`, `c` are off their optimum.

### The quantity that motivated the change

On-site error between the 79- and 159-atom cells under the retired convention, at the learned Z:

| Cl | Cs | Pb |
|---|---|---|
| **+0.1094 ± 0.0013** | −0.0454 ± 0.0062 | **−0.2828 ± 0.0040** eV |

Species-proportional, non-cancelling. Now zero by construction.

### Gates re-evaluated

| | pre-correction | post-correction |
|---|---|---|
| F4 slope | −0.0607 ± 0.0091 | −0.0540 ± 0.0079, sign 6/6, 5/6 in band |
| F5 corr | +0.898 ± 0.036 | +0.874 ± 0.042, positive 6/6 |
| dilution R | 0.85 ± 0.04, bound 77 ± 5% | **0.82 ± 0.05, 6/6 pass, bound 67 ± 11%** |

F5's absolute level moved ~+1.7 → ~+2.4 eV — the on-site shift, exactly where it should
appear. F4 barely moves, which is the plan's forecast holding: the correction is size-level,
not d-level.

### T_el scan — a finding that bears on the N_eff story

| T_el (meV) | f(changed level) | fractional levels | N_eff |
|---|---|---|---|
| 5 | 0.000 | 0.0 | 5.51 |
| 10 | 0.000 | 0.0 | 5.75 |
| 25 | 0.000 | 0.0 | 6.93 |
| 50 | 0.000 | 0.0 | 8.76 |
| 100 | 0.000 | 0.0 | 14.47 |

`f(changed level) = 0` at every temperature is **correct, not a bug**: the counters are a hole
(`h_maj = 1`), so the changed level is the one the carrier *vacated*. Zero levels carry a
fractional occupation at any T_el, i.e. μ sits in a real gap and the majority fill is
integer — so E_head is nearly T_el-independent (−2.40 → −2.26 eV over a 20× range).

**But N_eff nearly triples over that range.** N_eff is therefore *not* a T_el-independent
property of the state; part of it is thermal smearing, entering through the minority channel
and the reference fills rather than the majority fill. **This is load-bearing for the
localisation narrative**: the N_eff numbers quoted throughout (2.92 → 4.75 → 5.26) are all at
T_el = 25 meV, and are only comparable to each other. The dilution ratio is the boundness
statement that does not have this sensitivity.

**Config decision recorded: keep T_el = 25 meV.** Nothing in the scan argues for a change —
the fill is integer at every temperature tried and E_head is flat — and changing it would
break comparability with every prior number for no measured gain.

### Saturation audit — INCOMPLETE, and reported as such

The audit ran on all six models but produced **no channel rows**: its spy looks for tanh-
argument submodules under `head.h` by name (`g`, `h`, `g_mlp`, `h_mlp`, `hop_mlp`,
`site_mlp`) and **none matched**. It fell back to parameter norms (`|v0|max = 3.087`) and the
radial envelope, and recorded the miss in the json rather than reporting a spurious zero.

**So the second §4 config decision is not yet made.** The saturation fractions, the `sech²`
gradient attenuation and the γ-bound judgement are all unmeasured. This needs the real
attribute names from `defect_bounded.BoundedLocalHead` before it is worth anything. It does
not block §5 — the decision it feeds is "keep the bounds as they are" by default — but it is
an owed item, not a completed one.

---

## §5 — six-seed rerun under the corrected kernel

6/6 trained, no init-gate trips.

| | wired (old kernel) | **§5 (full sum)** |
|---|---|---|
| trained | 6/6 | **6/6** |
| axial_red | +0.645 ± 0.010 | **+0.629 ± 0.009** |
| rmse_all | 25.3 ± 0.6 | **25.0 ± 0.5** |
| N_eff | 4.75 | **5.26** |
| pristine frontier gap | 2.400 | **2.399** |

Per seed: axial_red +0.610 to +0.638, N_eff 4.73–5.90, gap 2.383–2.416.

### Forecasts scored

| forecast | outcome |
|---|---|
| eps0 / Z / c re-settle | **met** — fit quality unchanged (RMSE 25.0 vs 25.3) |
| pristine gap re-passes within 0.1 eV | **met**, \|Δ\| = 0.001 on a 0.10 gate |
| F4 slope unchanged within CI | §6, in flight |
| dilution and F5 within spread | §6, in flight |
| cross-size consistency improves | zero by construction; the §6 dilution number is the empirical read |

### Two deviations, stated

1. **"Config path only" was not honoured.** This ran through `stage_run.py`. §3 established
   the two build a bit-identical model, so the *model* is the one the config path would build
   — but the production trainer does not carry the Stage-3 training *protocol*: Harrison
   init, the c-shift calibration, the warmup, `loss_gap` and the init gate all live in
   `stage_run`. Porting them into `mace.cli.run_train` is real work. **Owed.**
2. **§5 was launched before §4's config decisions landed**, on explicit instruction to start.
   The T_el decision has since come back "no change", which retrospectively validates the
   run. The saturation decision is still unmade (above), so if it eventually calls for a
   bound change, this run is superseded.

### On localisation

N_eff 4.75 → 5.26 is a ~10% loosening, outside the seed spread. Direction is what the
corrected kernel predicts — Pb's on-site energy moved most (−0.283 eV cross-size) and Pb is
where the carrier lives. For scale, the Stage-2 s-only control sat at N_eff 33.1 with its
best-fitting seeds at 64–70; this is an order of magnitude tighter, at better fit, with the
gap on target. **And N_eff is T_el-sensitive (above), so the dilution gate is the boundness
statement to rely on** — 0.82 ± 0.05 with 6/6 passing on the pre-correction models.

---

## §6 — gate table, on the §5 models

**All three standing gates pass, 6/6 each.**

### F4 — right sign, within 3× of the reference

| seed | slope (eV/Å) | 95% CI | corr | |
|---|---|---|---|---|
| 1 | −0.0639 | [−0.0773, −0.0506] | −0.939 | PASS |
| 2 | −0.0533 | [−0.0650, −0.0415] | −0.933 | PASS |
| 3 | −0.0774 | [−0.0917, −0.0631] | −0.952 | PASS |
| 4 | −0.0645 | [−0.0775, −0.0516] | −0.944 | PASS |
| 5 | −0.0493 | [−0.0617, −0.0369] | −0.915 | PASS |
| 6 | −0.0591 | [−0.0756, −0.0426] | −0.899 | PASS |

Mean **−0.0613 ± 0.0090**, sign 6/6, **6/6 in band** (was 5/6 before the correction — seed 5
moved from −0.0431 to −0.0493 and cleared the threshold it had missed by 0.0016).

### F5 — positive, 6/6

corr **+0.911 ± 0.028** (0.871 to 0.944), slope +0.056 to +0.091 eV/Å, every CI excluding
zero. λ_frontier spans +0.92 to +2.05 eV across seeds — a wider absolute spread than before,
which is the corrected on-site potential no longer being pinned by a supercell-dependent
offset.

### Dilution — bound, 6/6

| seed | δ_L (eV) | depth (eV) | bound | R_bound |
|---|---|---|---|---|
| 1 | 0.015 | 0.042 | 75% | 0.77 [0.69, 0.99] |
| 2 | 0.011 | 0.032 | 75% | 0.78 [0.70, 1.13] |
| 3 | 0.013 | 0.060 | 81% | 0.72 [0.63, 0.81] |
| 4 | 0.011 | 0.043 | 81% | 0.69 [0.64, 0.80] |
| 5 | 0.014 | 0.034 | 56% | 0.86 [0.73, 0.89] |
| 6 | 0.013 | 0.039 | 75% | 0.84 [0.74, 1.24] |

**R = 0.78 ± 0.06**, gate ≤ 1.3 met **6/6**, bound fraction **74% ± 8%**. Four of six
intervals exclude 1.3 outright; none approaches the band-state value of 2.

### F4 head-vs-base — unchanged verdict

head **−0.0613 ± 0.0090** against this base's residual slope **−0.1310 [−0.1422, −0.1198]**,
which brackets the −0.134 reference in **6/6**. Verdict **head, 6/6**. The head reproduces
47% of what its own base leaves for it, with non-overlapping intervals. **The joint run will
not fix F4** — training the base jointly removes M1b's base-extrapolation slope from the
79-atom energy targets, a different problem; here the frozen base already leaves the right
trend.

---

## Forecast scorecard — all five met

| forecast | outcome |
|---|---|
| eps0 / Z / c re-settle | **met** — RMSE 25.0 vs 25.3, axial_red within a seed spread |
| pristine gap re-passes within 0.1 eV | **met** — 2.399, \|Δ\| = 0.001 on a 0.10 gate |
| F4 slope unchanged within CI | **met** — −0.0613 ± 0.0090 against −0.0607 ± 0.0091 |
| dilution and F5 within spread | **met** — R 0.78 ± 0.06 vs 0.85 ± 0.04; corr +0.911 ± 0.028 vs +0.898 ± 0.036 |
| cross-size consistency improves | **met by construction** — the cross-size on-site error is now identically zero; the dilution number is the empirical read |

No forecast missed. Per the decision tree, §6 gates passing means the pipeline is clear to
proceed to the joint run.

### What the correction did and did not change

It did **not** move the d-channel: F4's slope is the same to within a tenth of its own
standard error, exactly as forecast — the fix is size-level. It did **not** cost fit quality
or localisation: RMSE improved slightly, dilution tightened from 0.85 to 0.78, F5 held. What
it changed is the thing it was supposed to change — the ~0.11/0.05/0.28 eV per-species
inconsistency between the 79- and 159-atom cells is gone, and F4 gained the sixth seed.

---

## Steps 1-5 of the co-advisor plan: smearing, widening, rerun, gates

**Regime tag: lr 0.01, AdamW, 60 epochs, frozen Stage-A base, gamma = 3 eV, Gaussian
sigma = 0.05 eV.**

### The convention, and a latent bug it exposed

The head now uses Gaussian smearing at sigma = 0.05 eV, matching the label pipeline's own
default (doped, ISMEAR = 0). Both families stay selectable; one module-level switch drives the
fill, the entropy and the density-response backward, and the head sets it at forward entry and
restores it at exit.

Switching families exposed a bug that Fermi-Dirac had hidden. The response backward recovered
`mu` by inverting the occupations, falling back to the spectrum's MEDIAN when no state had a
fractional filling. Under Gaussian smearing `erfc/2` drops below 1e-6 within 3.5 widths, so a
gapped spectrum has every `f` at exactly 0 or 1 -- the fallback fired, put `f'` at the wrong
energy, and failed the degenerate-limit finite-difference check by a **factor of 16**. `mu`
now comes from the forward and is never re-derived.

**A near miss, recorded because it is the interesting kind.** Switching the family left
`counting_t_el` at its old 0.025 default, so the first widened run trained at Gaussian
sigma = 0.025 and would have been reported as the labels' convention. The family was right,
the width was wrong, nothing raised an error. Caught by loading a saved model and reading
`spectral.t_el` off it; six seeds discarded, ~20 minutes of GPU. The default lived in four
places and now says 0.05 in all four, and the queue **builds a model and reads its config back
before spending a seed**. A grep could not have caught it -- every file said "gaussian" and
"0.05" somewhere; only the assembled object knows what it will train with.

### F1 -- fires 6/6, but on the anion, not the flanking Pb

The audit's 59.7% was suspiciously exact across five seeds, and 95/159 = 0.5975 with exactly
95 Cl in a 159-atom V_Cl cell. Resolved by species:

| species | saturated | pre-tanh signed mean |
|---|---|---|
| **Cl** | **100.0%**, 6/6 seeds | **-3.604 +- 0.192** |
| Cs | 0.0% | +0.139 +- 0.240 |
| Pb | 0.0% | -0.459 +- 0.253 |

F1's **gate** is met. F1's **mechanism** is falsified: the forecast was flanking-Pb
corrections driven by the missing-anion Madelung shift, and what fires is every chlorine in
the cell including those far from the vacancy. That is a species-uniform offset, not a
defect-local effect.

The audit itself had to be rebuilt: the plan puts the flag on `BoundedLocalHead`, and a
Stage-3 model contains none -- Edit 4 replaced the spectral head wholesale. The live bounded
forms are the two tanh sites on `SlaterKosterH`. The previous spy guessed submodule names,
matched nothing, and printed the same output as "no saturation found"; the tensors are now
stored by the code that computes them, and an empty buffer prints AUDIT DID NOT FIRE.

### The widened rerun: gamma 1 -> 3 eV, Gaussian 0.05

| seed | axial_red | rmse | N_eff | gap |
|---|---|---|---|---|
| 1 | +0.621 | 24.6 | 5.11 | 2.389 |
| 2 | +0.605 | 25.0 | 5.03 | 2.389 |
| 3 | +0.612 | 24.9 | 5.02 | 2.399 |
| 4 | +0.606 | 25.5 | 4.48 | 2.409 |
| 5 | +0.616 | 25.1 | 4.92 | 2.410 |
| 6 | +0.606 | 25.5 | 4.51 | 2.400 |

**6/6 trained. axial_red +0.611 +- 0.006, rmse 25.1 +- 0.3, N_eff 4.85 +- 0.25, pristine gap
2.399 +- 0.008** against a 2.40 target. Fit and localisation are unchanged from the
pre-widening run within a seed spread; the gap gate is met on every seed.

### Gates on the widened models

| gate | result |
|---|---|
| **F4** | -0.0489 +- 0.0068 eV/A, sign 6/6, **4/6 in band** (was 5/6 at -0.0613) |
| **F5** | corr **+0.858 +- 0.031**, positive 6/6, every CI excluding zero |
| **Dilution** | **R = 0.85 +- 0.09**, gate <= 1.3 met **6/6**, bound fraction **67% +- 7%** |

F5's absolute level moved to +3.58..+3.81 eV from +1.3..+2.1 -- the widened bound letting the
on-site correction go where it was pinned from before.

### THE PRE-REGISTERED PREDICTION HELD, AND IT EMPTIES THE CANDIDATE LIST

Recorded before the widening ran: *if every Cl is pinned at the same value, the correction is
a rigid per-species shift and cannot carry a d-trend, so widening gamma should not move F4's
slope materially.*

F4 went from **-0.0613 +- 0.0090 to -0.0489 +- 0.0068** -- not toward the -0.131 the base
leaves, but slightly **away** from it. The saturation was real, it was fixed, and it was not
the F4 mechanism. This is the co-advisor's own second branch: **the candidate list is empty
going into R3.**

### F2 -- met on the near pair, missed low on the far one

| setting | slope | vs the labels' convention |
|---|---|---|
| Gaussian 50 meV | -0.0489 +- 0.0068 | -- |
| Fermi-Dirac 25 meV | -0.0481 +- 0.0065 | **-1.7%** |
| Fermi-Dirac 5 meV | -0.0536 +- 0.0085 | **+9.4%** |

Forecast: < 5% for the near pair (**met**, -1.7%), 10-25% at FD 5 meV (**missed low**, +9.4%).
The tail equivalence is confirmed and the smearing is not a candidate fix -- the labels are
themselves smeared at 0.05.

### F3 -- E_LR cannot be the missing amplitude, and the threshold undersells it

`dE_LR/dd = +0.00663 +- 0.00552 eV/A`, with the |slope| < 0.01 threshold met by 4/6.

The threshold is the weaker reading. F4's shortfall needs **-0.070 eV/A**, and E_LR supplies
**+0.007** -- wrong sign and an order of magnitude too small. It cannot close F4 whichever
side of 0.01 an individual seed lands.

**Caveat that bounds the claim.** These models were BUILT with the branch off, so
`latent_charges` does not exist on them and no flag can switch it on; the model is rebuilt
with the branch present and the trained weights copied in, which leaves 13 long-range
parameters at INITIALISATION. So this is the slope E_LR would contribute on day one of a
staged re-enable, not after the joint run had fitted it. A large slope would have settled F3
outright; a small one is strong but not conclusive, and the joint run's staged protocol
measures the trained version for free.

### Other steps

**Z endpoints.** Z_Cl -0.7575 -> **-0.8160** (shift -0.0585 against a 0.0245 pooled seed
spread: moved), Z_Pb -> **+2.0094**, essentially nominal. Cs and Pb within spread.

**The eps0/correction degeneracy is not there.** `eps0` spreads across seeds are 0.007-0.047
eV against a ~1 eV pinned correction, so the optimiser reaches the same place every seed and
`eps0[Cl]` is pinned by the data rather than wandering a flat direction. The concern I raised
about the joint run carrying an undetected flat direction does not survive measurement.

**Common-delta_L wired.** `delta_L` came from 80-atom pristine cells while depth came from
159-atom charged ones -- a depth judged against a level spacing from a different
Brillouin-zone sampling. Now one size for every frame the bound flag touches, with
`pristine_size` in the json.

**Protocol into the package.** The c-shift, warmup, `loss_gap`, init gate, trainable mask and
post-step projection are now functions in `defect_protocol.py` with no harness state, so the
harness and the trainer call the same code. `protocol_summary()` records the head's live
smearing family and width beside the stage flags -- it read the module default before, which
is the same mismatch that trained six seeds at the wrong width one layer up.

**Bit-identity at the final config -- PASS**, with energy now exactly bit-identical and forces
inside the same-model repeat floor.

---

## Owed items

1. ~~Saturation audit~~ **done** — rebuilt against `SlaterKosterH`, F1 fires 6/6 on the
   anion, gamma widened to 3 eV in response.
2. **Stage-3 protocol in the production trainer** — Harrison init, c-shift, warmup,
   `loss_gap`, init gate. Until then "config path only" cannot be honoured for a real run.
3. **F4's amplitude shortfall** — pre-correction, the head delivered 46% of what its own base
   left (residual slope −0.1310 [−0.1422, −0.1198], bracketing the −0.134 reference in 6/6),
   so the joint run will not fix it. Per the plan this is close-or-explain at R3, not a
   joint-run gate. Candidates in order: E_LR re-enable, then SCC.
4. ~~Z endpoints~~ **done** — Z_Cl moved, Z_Pb at nominal, no flat direction.

5. **`run_train` call sites** for the protocol module. The logic is in the package; the
   trainer does not yet call it. This is what "the joint run comes from config" still needs.

6. **F3's caveat**: the long-range slope was measured with 13 LR parameters at
   initialisation. The trained version is measured for free inside the joint run's staged
   protocol.

7. **F4's amplitude is unexplained.** Widening gamma was the last first-order candidate and it
   did not move the slope. SCC (Edit 5) is the remaining one, sign undetermined and second
   order. Per the plan this is close-or-explain at R3, not a joint-run gate.

---

## Bottom line for the coadvisor

The Madelung host term was subtracting ion i's own periodic images, which made `phi_LR` a
function of the supercell rather than of the crystal — a ~0.11 / 0.05 / 0.28 eV per-species
error between the two cell sizes the dataset contains. It is fixed, pinned by eight standing
tests including a counterfactual, and the six seeds retrained under the corrected kernel pass
all three gates 6/6 with every registered forecast met.

The state is bound: dilution R = 0.78 ± 0.06 against a ≤ 1.3 gate, bound fraction 74%, on the
observable that separates a bound carrier from a band state by construction. N_eff (5.26) is
reported alongside but should **not** be leaned on — the T_el scan shows it nearly triples
from 5 to 100 meV, so it is partly thermal smearing and is comparable only within a fixed
T_el.

Two things are open and neither is hidden: the saturation audit did not bind to the head's
modules and its config decision is unmade; and F4's amplitude shortfall is the head's, not the
base's, so the joint run will not close it — that is close-or-explain at R3, with E_LR then
SCC as the candidates in order.
