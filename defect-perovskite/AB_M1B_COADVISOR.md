# M1b and the A/B — overnight results, 2 Sep 2026

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** M1b clears §2: the base's own error rises with d and steepens exactly where
predicted, so the 159-atom subset speaks for the labels and the small-cell inversion is
contamination. The A/B then ran to completion, 12/12 cells. It reads as the **named failure**:
the sign flip buys a materially better force fit and spends it *without* moving the carrier
onto the hub — axial_red +0.348 → +0.588 while region mass falls 0.570 → 0.429.

Two caveats on that reading, both ours: the A/B was run **ungated**, ahead of C1 and M1b, on
an explicit instruction to have results by morning; and C1 and M3 were never run, so we have
no direct measurement of whether ON actually moved t(d).

## M1b — which subset speaks for the labels

| cut | n | corr | slope (eV/Å) |
|---|---|---|---|
| null, 79-atom neutral | 1174 | +0.387 | **+0.128** |
| null, 79-atom **d > 5.5** | 56 | +0.280 | **+0.294** |
| charged 79, all d | 1030 | +0.447 | +0.369 |
| charged 79, **d ≤ 5.5** (neutral-dense window) | 536 | **+0.062** | +0.128 |
| charged 79, axis ~ c | 503 | +0.192 | +0.282 |
| charged 79, axis ~ a/b | 527 | +0.296 | +0.199 |
| charged 79, null-corrected | 1030 | +0.310 | +0.241 |
| **charged 159 (reference)** | 17 | **−0.989** | **−0.134** |

Cuts 1 and 2 behave as §2 predicts. The frozen base's error grows with d and **steepens to
+0.294 eV/Å in the long-d region** — the extrapolation regime — and restricting the charged
79-atom frames to the neutral-dense window collapses their correlation from +0.447 to +0.062.
A positive base slope of that size is more than enough to swamp a true −0.134.

**Verdict: §2 is not falsified; the 159-atom subset speaks for the labels.** The chain is
interpretable.

Three things that stop this being clean, and we would not want them lost:

* The null is **in-sample** — no per-fold bases exist, so the base was trained on these neutral
  frames. +0.128 is a *lower bound* on the contamination, and this cut cannot prove absence.
* The **null-corrected trend stays positive** (+0.241, corr +0.310) rather than flipping toward
  the large-cell sign. A single linear correction does not fully account for the small-cell
  behaviour, so the reconciliation is partial.
* The **orientation cut did not separate** (+0.282 vs +0.199). The predicted
  orientation-selection of the long-d population is not visible.

The `dE swing` on the clean subset remains 0.28 eV, so §1's rescaling of the |t′| target to
0.10–0.20 eV/Å stands and the eV-scale forecast remains withdrawn.

## A/B — 12 cells, ON vs OFF, differing only in `channel_sign`

| arm | n | axial_red | N_eff | null ratio | region mass | Delta_bind | rmse_nbhd |
|---|---|---|---|---|---|---|---|
| **ON** | 6 | **+0.588** | 28.67 | 0.743 | 0.429 | +0.546 | 43.5 |
| OFF | 6 | +0.348 | **13.87** | 0.806 | **0.570** | +0.445 | 47.7 |

Per seed:

| arm | seed | axial_red | N_eff | region mass |
|---|---|---|---|---|
| OFF | 1–6 | +0.669, +0.632, +0.343, +0.499, +0.405, **−0.464** | 22.9, 11.3, 5.8, 1.3, 40.7, 1.4 | 0.41, 0.67, 0.24, 0.97, 0.33, 0.80 |
| ON | 1–6 | +0.552, **+0.785**, +0.649, +0.316, +0.679, +0.550 | 40.7, 30.0, 8.2, 39.6, 3.8, 49.7 | 0.36, 0.32, 0.44, 0.33, 0.76, 0.36 |

Gates — ON: N_eff 2/6, ratio 0/6, Delta_bind 4/6, retained 0/6. OFF: 4/6, 0/6, 3/6, 0/6.

**Observable 1 moves the right way and consistently.** axial_red +0.348 → +0.588; ON has no
negative cell where OFF has one at −0.464; rmse_nbhd improves 47.7 → 43.5. The energy channel
does something.

**Observable 3 moves the wrong way.** Region mass *falls* 0.570 → 0.429 and N_eff *rises*
13.87 → 28.67. The prediction was hub mass rising from 0.13; ON is more delocalised than OFF,
not less, and it is the arm with the *worse* localisation gate count (2/6 vs 4/6).

Seed spread is large in both arms (ON N_eff 3.8–49.7, OFF 1.3–40.7), so at n = 6 the
localisation difference is not statistically clean. But the direction is consistent against the
prediction while the fit gain is consistent for it, and that combination is precisely §5's
**named failure reading**: ON fits energies better while the carrier stays off the hub — the
model found a route for the d-trend that does not require hub occupancy. On the pre-registered
tree that routes to s+p / orbital character.

## What we did not run, and the override

* The A/B ran **ungated** — ahead of M1b's verdict and without C1 — on an explicit instruction
  to have results by morning. We recorded that override in the queue script's own header at
  launch rather than afterwards. M1b having since cleared §2 makes the results interpretable,
  but the order was not the pre-registered one.
* **C1 never ran.** It was launched, then killed for the μ-init fix, and the A/B was run
  directly instead. So the clamped isolation of the energy channel from placement — the test
  designed precisely for the question the A/B has now raised — has not been done.
* **M3 is still unbuilt.** No autograd `dH_ab/dd`, so: no |t′| measurement on the A/B cells, no
  consistency triangle (observable 10), and F2's `t′` is still the profile slope.

The consequence is worth stating plainly: **we have a clean read on placement and no direct
measurement of coupling.** The named failure reading rests on region mass and N_eff, not on
whether ON moved t(d) at all. C1 plus M3 would close that, and they are hours rather than days.

## One correction to our overnight handover

We flagged two "ON-arm watch items" — force loss starting ~4700× baseline, and Δ_bind driven to
≈ −3 eV — as possible sign-convention artefacts. Both were misattributions. Running OFF with
the same clamp gives force **14.92** against ON's **14.91**: they are hub2-clamp effects
common to both arms. The A/B compares like with like, which is the more important conclusion.

The sign-aware μ initialiser (`s_c (Λ + μ_c) = median dE target`) was built anyway and is
correct — it just was not the cause.

## What we are asking

1. Whether the named failure reading should be taken now, or held until C1 and M3 supply the
   coupling measurement. Our inclination is to hold: the A/B says the carrier did not move,
   but not why, and "the head found a cage route for the d-trend" and "the sign flip never
   reached t(d)" have different consequences.
2. Whether the partial M1b reconciliation (null-corrected trend still +0.241, orientation cut
   flat) is acceptable, or whether per-fold bases should be trained for a clean out-of-fold
   null before §2 is relied on further.
