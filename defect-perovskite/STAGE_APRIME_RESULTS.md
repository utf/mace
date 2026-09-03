# Stage A′ cycle — running results (regime-tagged)

Plan of record: `STAGE_APRIME_SPEC.md`. Entries are appended as they land; the report is
assembled from here at the end.

## 5.2 — precision and caching (F20)

Regime: Stage-B configuration on the production model (Stage-A base `e0_base_s1` frozen,
head only, `spectral_first_shell=True`, counting head, γ = 3, Gaussian 0.05, exp envelope
L = 1.0, log modulation β = ln 1.5, learned decay lengths on, E_LR on from epoch 0 with the
density detached and the branch frozen, both size upweights 0.25, batch 8, float64 data),
local A4000, 2877 frames (2797 distinct geometries).

| run | trunk | cache | profiler step (mean of 3) | epoch 0 wall (c-shift → eval) | valid E / F |
|---|---|---|---|---|---|
| uniform (all float64, uncached) | f64 | off | — | 12.7 min | — |
| mixed (trunk f32, head f64) | f32 | off | 4.239 s | — | — |
| mixed + cache | f32 | on | 3.451 s (1.23×) | 8.4 min (1.5×) | 3.6 meV/atom, 19.2 meV/Å after 2 epochs |

Where the time goes (profiler, CUDA self time 1.74 s of a 4.24 s step): `linalg_eigh`
0.50 s, `mm` 0.21 s, `erfc` 0.14 s, tridiagonalisation kernels 0.29 s; CPU self time
5.7 s per step — the step is CPU-bound in the head's per-graph eigensolve and Ewald loop,
not in the trunk the cache removes. Both of the excluded remedies (eigensolve batching,
Ewald geometry precompute) are the ones that would move it.

- Identity half of F20: head outputs (eps, δ_sr, E_head, F_head) within 1e-6 eV / 1e-6 eV/Å
  of the all-float64 model on a fixed frame, and E_total within 1e-3 eV — **holds** (unit
  test `test_f20_head_outputs_match_the_all_float64_model`).
- Speed half of F20 (≥ 3× per epoch): **fails**, 1.5× measured.
- Drift guard, per epoch on a random training frame: |ΔE| 6.7e-7 and 8.3e-7 eV, max|ΔF|
  1.3e-6 and 2.0e-6 eV/Å (float32 trunk, GPU non-determinism).
- Cache: 2797 distinct geometries from 2877 frames (80 pristine frames are exact repeats),
  built in 338 s, 1 file keyed by the base's SHA-256.

## 5.3 — image compensation (§2.3), forward-only on the s7 cohort

Regime: s7 head-only cohort (γ = 3, Gaussian 0.05, frozen Stage-A base, forces + gap, 60
epochs, lr 0.01, float32 trunk), the term switched on and off on the same trained weights.

**Identities.** Zero at zero charge: holds. Constant across atoms for a uniform carrier on
a pristine cell: **does not hold as stated** — on a 40-atom cubic CsPbCl₃ cell the image
potential of a uniform q = −1/N has mean +2.52 eV, variance 6.1e-2 eV², range +2.07 to
+3.03 eV. The isolated evaluator's potential of a finite cluster is not constant across
the cluster, and the identity as written assumes it is. Recorded; the term was not tuned
to pass it.

**Tiling drift (adoption test, ideal geometries, one Cl vacancy in 1×/2×/3× tilings of
the 80-atom pristine cell: 79, 639, 2159 atoms; L = 14.2, 28.5, 42.7 Å):**

| model | D0 = D(1×) − D(3×), no term | with term | ratio |
|---|---|---|---|
| s1 | +0.432 | +0.027 | 0.062 |
| s2 | +0.439 | +0.035 | 0.079 |
| s3 | +0.410 | −0.003 | 0.008 |
| s4 | +0.408 | −0.007 | 0.017 |
| s5 | +0.429 | +0.019 | 0.045 |
| s6 | +0.407 | −0.008 | 0.020 |

6/6 within 0.3·D0 — **passes**. The frontier-weighted compensation is −0.61 eV at 1×,
−0.32 at 2×, −0.23 at 3× on every model, i.e. it cancels the ~1/L drift of the ion
Madelung term (2.78 → 2.49 → 2.34 eV).

**Probe (F15), s7 cohort, both sizes, on minus off, six models:**

| reading | off | on | Δ (mean ± sd over models) | sign per model |
|---|---|---|---|---|
| depth from CBM, 79 | +0.2125 | +0.2125 | −0.0000 ± 0.0000 | mixed |
| depth from CBM, 159 | +0.0947 | +0.0947 | −0.0000 ± 0.0000 | mixed |
| participation ratio, 79 | 0.157 | 0.141 | −0.016 ± 0.003 | 6/6 − |
| participation ratio, 159 | 0.492 | 0.401 | −0.091 ± 0.017 | 6/6 − |
| R_bound | 0.738 | 0.590 | −0.148 ± 0.039 | 6/6 − |
| force loss, 79 | 0.00066 | 0.00075 | +0.00009 | 6/6 + |

F15 clauses: depth rises more at 79 than 159 — **unmeasurable under the aligned depth**,
not "no effect": for a localised carrier (N_eff ≈ 3) the potential of its periodic images is
nearly constant over the home cell, so the term is a near-uniform on-site shift that the
quantile alignment of the occupied manifold removes by construction. Measured on the
tiling frames: the per-atom compensation has mean −0.53 eV and spread 0.062 eV at 1×,
−0.29 / 0.024 at 2×, −0.20 / 0.016 at 3× (against the 0.06 eV² variance of the uniform-
carrier identity, which is the opposite regime). The ratio falls — true; R → 0.9–1.0 —
false (0.59, further from 1); the 79-atom force loss does not rise — false (+13%).
**F15 fails, 1 of 4** (one clause not scorable by this probe).

**Adoption:** the tiling test passes and the probe fails, so under the registered rule
("adopt as config default only if both pass") the term is **not adopted** as a default. It
stays implemented behind `--defect_image_compensation` (off).

## Correction to the joint-run report (found while building Stage A′, 3 Sep 2026)

The neutral two-size upweight of the joint run **never reached the loss**. `DefectLoss`
scores an n = 0 frame through its base terms, which read `base_energy_weight` and
`base_forces_weight`; the upweight scaled the generic `forces_weight`, which no term reads
for a neutral frame. Verified by printing the columns on a 159-atom neutral frame from
`dataset_cf/fold0`: after the upweight `forces_weight` = 0.497, `base_forces_weight` = 1.0.
The joint run's logged "neutral large-cell share 25.0%" was the function's own arithmetic
on the column it had written — intent, not outcome, the fault class ledger entry 10 names.
Consequences: (i) the joint run's neutral 159-atom frames trained at natural weight, so
the neutral 159-atom force-slope collapse (+0.064 → +0.01) happened without any upweight;
(ii) the first Stage A′ fold launch (16:36, b3) and production launch trained with the
same inert weight for ~40 minutes and were killed and restarted with the base columns
scaled; (iii) `weight_column(population, channel)` now names the column per population,
`realised_shares` reads the same column, and a test asserts the `DefectLoss` VALUE moves.
The charged two-size upweight was never affected: charged frames are scored by the totals
terms, which read the generic columns it scaled.

## 5.3 — the extrapolation indicator on the existing Stage-A folds (§1, F16)

Regime: the four cross-fit fold bases `cf_base_f0..3` (Stage-A recipe, neutral quarters)
and the production base `e0_base_s1`; 2877 frames of train + valid; `c1_ood_indicator.py`.

| population | n | ood_E median (eV/atom) | p95 | ood_F median (eV/Å) | w_E median | w_E > 0.5 |
|---|---|---|---|---|---|---|
| neutral 79/80 | 1813 | 0.00094 | 0.00103 | 0.0040 | 1.000 | 100% |
| neutral 159 | 17 | 0.00097 | 0.00102 | 0.0034 | 1.000 | 100% |
| charged 79 | 1030 | 0.00125 | 0.00147 | 0.0069 | 0.683 | 93.7% |
| charged 159 | 17 | 0.00101 | 0.00102 | 0.0030 | 1.000 | 100% |

s_E (95th percentile of ood_E over neutral frames) = 0.00103 eV/atom. w_E against d on the
charged 79-atom frames is flat: median 0.65–0.76 in every bin from 4.5 to 7.0 Å.

Charged 79-atom residual slope d(E_label − E_base)/dd against the production base:
full range +0.3692 [+0.3241, +0.4143] (n = 1030); **w_E > 0.5: +0.3593 [+0.3144, +0.4042]**
(n = 965); w_E ≤ 0.5: +0.0622 [−0.4057, +0.5302] (n = 65). Charged 159: −0.1338
[−0.1445, −0.1230] (b1's reference reproduced).

**F16 fails.** The +0.36 slope is not carried by the low-w_E frames; the frames the fold
bases agree on carry it unchanged. The four fold bases disagree by about 1 meV/atom on
charged and neutral geometries alike, so the disagreement between bases trained on
quarters of the neutral set does not mark the charged 79-atom frames as extrapolation —
the base's long-d error (b1's +0.132 eV/Å carrier-free slope) is shared by all four. As a
weight, w_E is near-uniform (0.68 median) on the population it was meant to discount.
Stage B applies it as specified; its effect on the charged 79-atom energy channel is a
~0.7× scale, not a selection.

## 5.4 — Stage A′ bases (§1)

Regime: e0 recipe (140 epochs, float32, cuEq, energy 10 / total-energy 10, no long-range
branch, every carrier regulariser zero, lr default, batch 8) plus the neutral 159-atom
frames raised to a 0.25 share of BOTH the base energy loss and the base force loss, on the
columns the base terms read, logged every epoch. No charged frame in any training or
validation set.

| base | data | 159-atom frames | energy factor | force factor | final train E / F | final valid E / F |
|---|---|---|---|---|---|---|
| aprime_f0 | dataset_cf/fold0 (1379 train) | 12 | 92.0× | 46.1× | 2.5 / 11.5 | 1.0 / 6.1 |
| aprime_f1 | fold1 | 12 | 92.0× | 46.1× | 2.5 / 11.8 | 1.2 / 6.1 |
| aprime_f2 | fold2 | 12 | 92.0× | 46.1× | 3.1 / 11.8 | 1.8 / 6.1 |
| aprime_f3 | fold3 (1380 train) | 12 | 73.5× | 36.9× | 1.8 / 11.3 | 1.4 / 6.0 |
| aprime_prod | dataset_e0 (1616 train) | 15 | 83.9× | 42.1× | 6.1 / 10.9 | 4.9 / 11.8 |

Errors in meV/atom and meV/Å. The fold validation sets are pristine cells only (153
'ideal' frames each); the production validation set holds 95 pristine, 57 neutral 79-atom
and 2 neutral 159-atom frames, which is why its numbers are not comparable to the folds'.
Realised shares 0.2500 / 0.2500 on every base at every epoch. Fold bases trained on b3
GPUs 4–7 (one shared with the OOD indicator) at 0.8 min/epoch; the production base on the
local A4000 at 0.85 min/epoch. All five load with `trunk_avg_num_neighbors` equal to the
blocks' float (14.05 on fold 0), through the cuEq conversion.

Operational note: the post-A′ chain (`queue_after_aprime.sh`, waiting on the five model
files at a 120 s poll) did not wake for over ten minutes after the production model landed
on b3 at 18:33 (b3 clock), while its process was alive and sleeping. Cause not established;
a fresh launch of the same script saw the files at once. A first attempt to stop it with
`pkill -f` matched the ssh shell carrying the launch command and killed that instead — the
bracket trick protects the pattern, not the other text on the same command line. Recorded
as the fifth liveness-shaped fault of the programme, and the first on the file side.

## 5.4 — re-derived references against Stage A′ (§1, F17)

Regime: A′ production base `aprime_prod` and the four A′ fold bases; b1's machinery with
`--fold-prefix aprime_f`, 95% intervals from the per-frame residuals; c5 for the
production-base neutral slope.

| observable | A′ | pre-joint (e0_base_s1 / cf folds) |
|---|---|---|
| charged 159 energy slope, production base | **−0.0948 [−0.1084, −0.0812]**, corr −0.968 | −0.1338 [−0.1447, −0.1232] |
| charged 159 force slope, production base | **−0.2665 [−0.2938, −0.2393]** | −0.1901 [−0.2103, −0.1700] |
| neutral 159 null, out-of-fold (energy) | **+0.0243 [−0.0697, +0.1184]** | +0.0800 [−0.0503, +0.2102] |
| neutral 159 null, out-of-fold (force) | +0.0167 [−0.0163, +0.0497] | +0.0643 |
| neutral 159, production base (energy) | +0.0920 [+0.0492, +0.1347] | +0.0968 |
| charged 79, full range (energy) | +0.3724 [+0.3265, +0.4183] | +0.3641 |
| charged 79, neutral-dense window | +0.0793 [−0.0976, +0.2562] | +0.0806 |
| neutral 79 window null (energy) | +0.1147 [+0.0947, +0.1348] | +0.1317 |

The reference itself moved: with the neutral 159-atom frames finally weighted in the base
loss, the out-of-fold null at 159 atoms fell from +0.080 to +0.024 and the charged residual
from −0.134 to −0.095 — a third of the old reference was the base's own extrapolation at
large d, removed by giving the base the large neutral cells. The charged residual is still
carrier physics (its interval and the null's are disjoint). The 79-atom picture is
unchanged: +0.37 full range, +0.08 in the window, +0.11 of carrier-free base error.

**F17:** 159 null within ±0.05 of zero — holds (+0.024); charged 159 residual inside the old
interval — no (−0.095 against [−0.145, −0.123]), shifted by 0.039 < the old null's width
0.26 — holds; 79-atom charged residual magnitude < 0.2 — fails (+0.37). **F17 fails, 2 of 3.**
The new F4 reference for gate 2 is −0.0948, band [−0.142, −0.063] at 1.5×.
