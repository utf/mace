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

## 5.4 — w_E from the Stage A′ folds (§1), the weight Stage B uses

Regime: the four A′ fold bases and `aprime_prod`; 2877 frames.

| population | ood_E median | p95 | w_E median | w_E > 0.5 |
|---|---|---|---|---|
| neutral 79/80 | 0.00121 | 0.00139 | 1.000 | 100% |
| neutral 159 | 0.00105 | 0.00106 | 1.000 | 100% |
| charged 79 | 0.00108 | 0.00125 | 1.000 | 100% |
| charged 159 | 0.00097 | 0.00105 | 1.000 | 100% |

s_E = 0.00139 eV/atom. With the neutral 159-atom cells in the base loss, the four A′ fold
bases disagree on the charged 79-atom geometries by LESS than on the neutral ones (median
1.08 against 1.21 meV/atom), so w_E = 1.000 on every charged frame and the w_E > 0.5 slope
is the full-range slope, +0.3679 [+0.3244, +0.4113]. F16 fails here too, and more plainly:
the indicator does not distinguish the charged 79-atom population at all. Stage B applies
it as specified (944/944 frames matched, mean w_E 1.000), which is to say it applies no
weight; the charged 79-atom energy channel enters at its natural per-frame weight and the
charged 159-atom frames at a 25% share of the charged energy loss (factor 19.3).

## 5.5 — Stage B, in flight (wave 1: seeds 1–4, launched 19:14 b3 clock)

Regime: head only on `aprime_prod` (frozen, cached), first-shell features, counting head,
γ = 3, Gaussian 0.05, exp envelope L0 = 1.0 with learned decay lengths (β_L = ln 2), log
modulation β = ln 1.5, centred on-site correction (centre from 544 pristine frames), E_LR
from epoch 0 with the density detached and the branch frozen at physical values (host
charges zero, polarisation off, amplitude 1/√4), forces on every charged frame at a 0.25
large-cell share, charged 159-atom energies at a 0.25 share (factor 19.3), charged 79-atom
energies at w_E = 1.000, `loss_gap` w = 1 at 2.4 eV, c per (charge, size), 24 epochs, lr
0.005, warmup 5, batch 8, trunk float32 / head float64, image compensation off.

- Setup identical on all four seeds: 944/944 charged frames matched; charged energy share
  25.0%; centre norms per species (5.50, 8.59, 6.42); base cache 2797 frames.
- c table at initialisation (E_LR in the residual): c(79) = +9.164 ± 0.002, c(159) = +9.920
  ± 0.002 eV over the four seeds (Δc = +0.756 eV); the scalar `c_shift` starts at 0 and
  drifts by 0.01–0.10 eV in the first epoch, sharing the gauge with the table.
- The spec's neutrality residual `|Σq + Δn|` logs as exactly 0.500 on every seed: with
  the amplitude frozen at 1/√ε∞ = 0.5 the detached carrier charge sums to −0.5·Δn. The
  density-sum invariant (Σ q/a + Δn = 0) is checked in the forward and holds.
- Epoch 0: 10.1 min per epoch on b3 with the cache (the joint run trained the full base at
  10.0); validation after epoch 0: 5.4 meV/atom, 20–21 meV/Å on all four.

## 5.5 — Stage B wave 1 (seeds 1–4), finals and the first gates

Regime as above (Stage B, 24 epochs). Wave 1 ran 19:14–23:40 (b3 clock), 9.7 min/epoch.

| seed | valid E / F (meV/atom, meV/Å) | c_shift | c(79) | c(159) | Δc | |W_site| | Z | partic (epoch 20) |
|---|---|---|---|---|---|---|---|---|
| s1 | 5.6 / 16.6 | +0.640 | +10.321 | +11.046 | +0.725 | 0.083 | −0.889 +0.824 +1.843 | 8.08 |
| s2 | 5.4 / 16.8 | +0.358 | +9.932 | +10.717 | +0.786 | 0.063 | −0.887 +0.795 +1.866 | 6.96 |
| s3 | 5.5 / 17.1 | +0.548 | +10.186 | +10.966 | +0.780 | 0.066 | −0.908 +0.851 +1.873 | 7.90 |
| s4 | 5.5 / 17.1 | +0.408 | +9.890 | +10.641 | +0.751 | 0.061 | −0.851 +0.707 +1.847 | 6.14 |

c(79), c(159) are the table entries plus the scalar (both trainable, one gauge). The
neutrality projection held (3Z_Cs + Z_Pb + Z_Cl = 0.000 on every seed). The centred on-site
channel carries |W_site| 0.06–0.08 against 0.013 in the joint run.

**Gate 3 (report), c-consistency.** c(79) +10.08 ± 0.18, c(159) +10.84 ± 0.17, Δc = +0.760
± 0.024 eV over four seeds. Predicted size difference = E_LR difference + Ewald G = 0
difference = +0.048 ± 0.003 (mean E_LR per carrier −0.066 at 79, −0.019 at 159; background
−0.0039 / −0.0020). The measured Δc is sixteen times the electrostatic prediction: the
per-size constant is absorbing something that is not the image interaction — the 79-atom
base's +0.37 eV/Å residual slope, or the labels' own referencing between the two cells.
Report, not gate, as registered; the tolerance for a future gate is set by this number.

**Gate 7, stops and decay lengths (four seeds; the rule is ≤ 1 seed in six per type).**

| type | L_b (Å), mean ± sd | seeds with hub bonds at the stop | mean tanh g |
|---|---|---|---|
| ss-σ | 1.071 ± 0.012 | 2/4 (98%, 54%) | +0.99, +0.97 there |
| sp-σ | 1.273 ± 0.073 | 1/4 (25%) | −0.95 |
| pp-σ | 1.332 ± 0.033 | 2/4 (98%, 98%) | −0.99 |
| pp-π | 1.359 ± 0.036 | 4/4 (4%, 98%, 88%, 98%) | +0.81 … +0.99 |

**Gate 7 fails on wave 1**: the log modulation at β = ln 1.5 is range-equivalent to the
linear form at its upper end, and the head presses against it on the hub bond in pp-π on
every seed and in ss-σ / pp-σ on half of them — the same pattern b9 found on the s7 cohort
(two seeds of six at the stop) but on more seeds. The learned decay lengths all moved UP
from 1.0 Å (1.07–1.36), toward longer-ranged hopping, and none is near its own bound
(0.5–2.0 Å).

**Scoring incident, recorded.** The first b10 run on these four models reported 249 meV/Å
on the neutral 79-atom window (criterion 3) — impossible for a head that is zero at n = 0.
Cause: under the mixed policy the forward rebound `positions` to its head-dtype cast, and
on the scorers' float32 batches the force derivative was then taken with respect to the
copy, which the trunk's energy does not depend on; the base forces lost the trunk (RMS
0.30 against the A′ base's 0.010 eV/A on a neutral frame, base energies identical). Float64
training never took the cast, so the trained models and every training-time number
(drift guard, validation) are unaffected. Fixed (the gradient leaf never moves; a test
scores a mixed model on a float32 batch), and the wave-1 adoption scoring was rerun.

**Wave 1 by the adoption rule (b10 against the A′ references, leak floor 0.75 × ref):**

| seed | c1 charged 159 E / F | c2 neutral 159 E | c3 window E / F (meV/atom, meV/Å; oof base 2.0 / 12.8) | F4 δ_sr [95%] | gap | depth from CBM |
|---|---|---|---|---|---|---|
| s1 | −0.0948 / −0.2665 (= ref) | +0.0919 | 7.1 / 11.2 | −0.0812 [−0.0909, −0.0716] | 2.381 | 0.014 |
| s2 | = ref | +0.0919 | 7.1 / 11.2 | −0.1244 [−0.1398, −0.1091] | 2.384 | 0.042 |
| s3 | = ref | +0.0920 | 7.1 / 11.2 | −0.0877 [−0.0963, −0.0791] | 2.384 | 0.040 |
| s4 | = ref | +0.0920 | 7.1 / 11.2 | −0.1111 [−0.1279, −0.0942] | 2.426 | 0.111 |

- Gate 1 (regression): the charged 159-atom residual slopes against the frozen A′ base
  equal the §1 reference on every seed, to four decimals, in both channels — an identity by
  construction now that the base does not move, and the test that would have caught a base
  that did.
- Gate 2 (F4): correct sign and within 1.5× of −0.0948 on **4 of 4** (pooled −0.1011 ±
  0.0174). The gate asks ≥ 4/6, already met before wave 2.
- Criterion 2 (neutral 159): +0.0920 = the production base's own value, unchanged by the
  head (as it must be at n = 0); against the out-of-fold +0.0243 it reads "grew", which is a
  statement about the production base, not the head.
- Criterion 3 (neutral-79 window): E 7.1 meV/atom against the out-of-fold cross-fit base's
  2.0; F 11.2 against 12.8. The energy figure is the production A′ base's own error on
  those frames (a frozen base, so the head cannot change it); the production base's final
  train energy error (6.1 meV/atom) is worse than the fold bases' (2.5) and the old
  e0 base's (3.3), which is a finding about the A′ production run rather than Stage B.
- Gate 5: pristine gap 2.381–2.426 eV, all within 2.4 ± 0.1.
- Depth from the CBM 0.014–0.111 eV (shallow donor), seed spread larger than the joint
  cohort's.

## 5.5 — Stage B wave 2 (seeds 5–6) finals, and participation over all six

Wave 2 ran 23:40–04:02 (b3 clock). Finals: s5 5.6 / 16.6, s6 5.5 / 16.8 meV/atom, meV/Å;
c(79) / c(159) +9.709 / +10.452 (s5), +9.535 / +10.292 (s6); |W_site| 0.059 / 0.084; Z
projection held.

Participation (1/Σα² on carrier-bearing validation frames) at epochs 0 / 4 / 8 / 12 / 16
/ 20:

| seed | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| s1 | 10.17 | 9.05 | 7.88 | 7.49 | 6.70 | 8.08 |
| s2 | 9.82 | 8.36 | 8.23 | 6.06 | 6.64 | 6.96 |
| s3 | 9.57 | 8.41 | 7.46 | 7.85 | 8.21 | 7.90 |
| s4 | 11.36 | 8.67 | 7.70 | 7.56 | 6.09 | 6.14 |
| s5 | 9.71 | 8.46 | 6.41 | 7.53 | 7.61 | 6.75 |
| s6 | 9.60 | 6.88 | 6.49 | 6.23 | 6.96 | 6.27 |

Six seeds, one direction: 9.6–11.4 at epoch 0 to 6.1–8.1 at epoch 20, spread 2.0 at the
end against the joint cohort's 5.7 (11.2 vs 5.4). No reversal at any epoch and no
three-and-three split: with E_LR on from epoch 0, detached and frozen, the branch that
turned the joint cohort's participation around at epoch 12 does not act. F18's first
clause (spread halves relative to the joint cohort) is met on this measure: 2.0 against
5.7; the registered measure is the charged/pristine ratio from b13, scored with the gates.
