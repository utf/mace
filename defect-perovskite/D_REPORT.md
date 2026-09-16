# Distilled LES base — overnight report (2026-09-16, 08:30)

Units: forces per-component meV/Å; energies meV/atom. Plan: MH-1 → cutoff-limited LES student on
MH-1's predictions (D0/D1) → DFT fine-tune with distillation replay (D2) → head line on the base (D3).
Every threshold below was registered before the corresponding result was opened (tracker,
`DSCC_V5_IMPLEMENTATION_SPEC.md`, "Distilled LES base").

## 1. Headline

1. **D1 — the identification argument did not hold on this set.** All four from-scratch students
   (narrow/wide × 6 Å/12 Å reach) carry large ionic-signed charges with |Pb/Cs| in the registered
   window, and all four violate the acoustic sum rule by 0.4–6 e per cell; the LES term is
   structural in every one of them (E_LES −100 to −330 meV/atom), not a long-range correction.
   Reach and width both change the *fit* to the teacher, neither turns the charges on or off.
   None meets G-D1.1 on forces. The mechanism control (MH-1 weights) reproduced L0's collapse exactly
   as predicted.
2. **D2 — every fine-tuned student is a worse base than base v2**, on neutral validation (14.6–37.5
   vs 10.23), on the charged shell table (worse at every shell in three of four; worse at the far
   field in the fourth) and on the Born sum rule. The DFT fine-tune *grows* the charges (up to 5×)
   rather than pinning them.
3. **D3 on the benchmark base** (the full-shape student's fine-tune, used as a benchmark, not an
   adoption): the L2 pre-test finds a +0.40 e defect charge in the LES pattern → a −116 meV 1/L
   residual at 79 atoms, so E_cross is forces-only on this base. **L1** (plain W6, six seeds) is
   inferior to W6-on-v2 (21.1 vs 11.6 meV/Å; C5 localisation 1/6). L2 is launching this morning
   per the registered rule (L1 is the baseline, not a gate); L3/L4 follow the chain.

## 2. D0 (as built)

30 318 structures, none above 160 atoms (the tilings family was removed by the 13:40 amendment):
charged geometries relabelled neutral 1 047, neutral 1 770, perturbed 25 353 (3 copies each at
σ = 0.05 / 0.10 / 0.20 Å), strained 1 980 (iso ±2/±4 %, xy shear ±3 %; stress weight only here),
polar sublattice displacements 168; teacher MH-1 `omat_pbe` (sha256 `a522eb7f…`), float32 + cueq,
338 s; teacher-predicted E0s Cl −2.252 / Cs −0.970 / Pb −1.037 eV (stored row −0.257 / −0.130 /
−0.774); split 90/10 by parent geometry (281 held parents, 2 994 held structures).

## 3. D1 — five students (stage-two EMA models; `les_fit` recipe unless stated)

| arm | width × reach | epochs / batch | teacher fit E / F (held) | per-family F: charged / neutral / perturbed / strained / polar | G-D1.1 |
|---|---|---|---|---|---|
| light | 96 ch, 1 layer (6 Å) | 80 / 32 | 1.09 / **38.5** | 25 / 25 / 40 / 27 / 6.3 | FAIL |
| light2 | 96 ch, 2 layers (12 Å) | 80 / 32 | 0.44 / **17.1** | 9.8 / 9.5 / 18.1 / 12.0 / 1.2 | FAIL |
| full-scratch | 512 ch, 2 layers (12 Å) | 40 / 32 | 0.37 / **14.1** | 7.5 / 7.2 / 14.9 / 10.5 / 0.8 | FAIL (polar passes) |
| full1 | 512 ch, 1 layer (6 Å) | 40 / 32 | 0.88 / **32.3** | 20 / 20 / 34 / 23 / 3.9 | FAIL |
| full-init | MH-1 shape + weights | 30 / 8 | 0.39 / 2.11 | 0.8 / 0.9 / 2.3 / 1.2 / 0.2 | PASS |

Energies pass ≤ 1 meV/atom in every arm (light misses on `perturbed`, 1.14). Forces fail by 3–8× in
the from-scratch arms; the σ = 0.2 Å perturbed copies dominate every miss. Reach helps the fit (12 Å
arms 17 / 14 against 6 Å arms 38 / 32) and width helps at fixed reach (14 vs 17; 32 vs 38).

**Charges and Born check** (8 neutral 80-atom held frames, EMA weights, one global sign fix — it was
−1 in every arm — so that Cs > 0):

| arm | \|q\| (e) | E_LES (meV/atom) | Z* Cl / Cs / Pb | \|Pb/Cs\| (2–3.5) | ΣZ* per cell (≤ 0.3) | G-D1.2 | LES ablation (F, all / polar) |
|---|---:|---:|---|---:|---:|---|---|
| light | 0.168 | −114 | −0.73 / +0.57 / +1.37 | 2.42 ✓ | −4.00 | FAIL | 229 / 66 (from 38 / 6.3) |
| light2 | 0.246 | −334 | −0.58 / +0.39 / +1.00 | 2.56 ✓ | −5.39 | FAIL | 132 / 27 (from 17 / 1.2) |
| full-scratch | 0.148 | −98 | −0.49 / +0.44 / +1.06 | 2.43 ✓ | +0.64 | FAIL | 39 / 4.8 (from 14 / 0.8) |
| full1 | 0.177 | −148 | −0.76 / +0.52 / +1.39 | 2.65 ✓ | −6.10 | FAIL | 268 / 81 (from 32 / 3.9) |
| full-init | 0.027 | −3.7 | −0.11 / +0.06 / +0.12 | 2.04 ✓ | −2.45 | FAIL | 2.95 / 0.4 (from 2.1 / 0.2) |
| LES-FT2 (reference) | 0.031 | −4.4 | −0.47 / +0.38 / +1.03 | 2.69 | +0.18 | (pass) | — |

Reading against the pre-registered predictions:
- *Light passes and the full arms do not → identification confirmed*: **not observed.** Every
  from-scratch arm shows the same halide-perovskite pattern (Cl −, Cs +, Pb + at 2.4–2.7× Cs) and
  every one fails the sum rule; the 12 Å arms are not weaker on Born than the 6 Å arms (light2's
  |q| is the largest). Reach identifies nothing here — the charges are a fitting device the students
  adopt regardless of receptive field, with E_LES two orders of magnitude above LES-FT2's.
- *Full-init collapses (E_LES → 0, |q| falls)*: **observed** in the sense registered — E_LES stays at
  the init scale (−3.7), the raw species means carry no ionic pattern (all one sign), the readout
  norms barely move. L0's mechanism reproduced.
- G-D1.3 (ablation raises the long-range-family error) passes trivially in every from-scratch arm
  and is not discriminating: zeroing the charges removes ~100 meV/atom of the model's energy on every
  family. The reading rule's "light fails and full-scratch passes" branch was not reached because no
  arm passes G-D1.2; by G-D1.1 the best student is full-scratch (14.1), which the user designated the
  benchmark base.

## 4. D2 — DFT fine-tune with distillation replay (30 epochs each, multi-head, shared LES readout)

Recipe: `Default` = DFT head (base v2's `train_iso.xyz`), `pt_head` = teacher replay (a fixed
stratified 1 616-frame D0 subset + teacher isolated atoms; replay 1:1 by count), the student's LES
readout transferred unscaled into the shared readout, base v2's optimiser settings, float32 + cueq.
Two loader defects were found by the one-epoch smoke and fixed before launch (the multi-head loader
re-initialised a MACELES foundation's LES readouts; the cueq⇄e3nn converters dropped
`les_shared_readout`); after the fixes the replay head starts at the student's own teacher error.

| D2 arm (from) | DFT head, epoch 29 E / F | G-D2.1 (≤ 1.4 / 12.3) | charged 79 shells total / 2–4 / 4–8 / 8–10 / 10–12 / >12 | G-D2.2 (≤ v2 at every shell) | Born Cl / Cs / Pb, ratio, ΣZ* | \|q\|, E_LES |
|---|---|---|---|---|---|---|
| base v2 (reference) | 1.16 / 10.23 | — | 33.2 / 74.9 / 30.4 / 15.3 / 8.9 / 8.0 | — | — | — |
| d2_full_scratch | 2.28 / 14.6 | FAIL | 35.9 / 82.0 / 32.3 / 16.4 / 10.7 / 9.5 | FAIL (all) | −0.52 / +0.44 / +1.14, 2.57, **+0.52** | 0.332, −480 |
| d2_light2 | 2.48 / 21.5 | FAIL | 38.5 / 83.5 / 35.1 / 21.2 / 15.6 / 15.1 | FAIL (all) | −0.45 / +0.38 / +0.99, 2.61, +0.39 | 0.273, −342 |
| d2_full1 | 1.96 / 27.8 | FAIL | 30.5 / 59.7 / 26.9 / 24.1 / 21.7 / 19.5 | FAIL (8–10, 10–12, >12) | −0.76 / +0.53 / +1.43, 2.71, −5.15 | 0.199, −182 |
| d2_light | 4.23 / 37.5 | FAIL | 36.6 / 60.1 / 34.4 / 31.9 / 28.5 / 25.4 | FAIL (all) | −0.73 / +0.53 / +1.40, 2.65, −4.16 | 0.168, −131 |

Neutral 79-atom (in-sample) shells for the four: 17.8 / 26.3 / 33.0 / 46.3 total against base v2's
10.96. Two observations: the fine-tune barely moves the forces from the student's teacher error
(full-scratch 15.4 replay → 14.6 DFT; base v2 went 43.9 → 10.2 from MH-1), so the students are
weaker starting points than MH-1 itself; and the replay does not hold the charges — |q| grows 2–5×
under the DFT fine-tune (full-scratch 0.148 → 0.332 e, E_LES −98 → −480 meV/atom). The 1-layer
wide student is the only one that beats base v2 near the vacancy (59.7 vs 74.9 at 2–4 Å) and it is
the worst in the far field (19.5 vs 8.0 at >12 Å).

G-D2.4 (cross-fit bases) was not run: no arm passes.

## 5. D3 on the benchmark base (`d2_full_scratch_base.pt`, sign fix −1)

**L2 pre-test** (frozen-core ladder, 639–2 879 atoms, bare `gamma_lr(r_g 1.0, r_split 2.5)`):
monopole exact (residual 0.9 % of the span). The LES pattern's defect charge is **Q_defect = +0.40 e**
(the removed Cl site carried −0.25 e; the base's response to the vacancy adds +0.15 e, all of it
within 4 Å: −0.045 e inside 4 Å, −0.004 e inside 8 Å) → E_cross has a 1/L slope of −1.22 eV·Å
(expected −1.43 from Q_defect) and a **−116 meV residual at 79 atoms** against the ≤ 10 meV rule.
**On this base L2's cross-size energy claim cannot stand; E_cross is a forces-only term.**

**L1 — plain W6 on the benchmark base** (six seeds, 60 epochs, ≈ 72 min each; W6-era estimator
since the W6-on-v2 reference has epoch-averaged files for two seeds only):

| | W6 on base v2 (reference) | L1 on the benchmark base |
|---|---:|---:|
| held force RMSE, median | 11.58 | **21.06** |
| per seed s0 / s1 / s2 / s3 / s4 / s6 | 11.2 / 11.5 / 11.7 / 11.1 / 12.2 / 11.9 | 26.6 / 13.6 / 15.6 / 13.4 / 30.8 / 27.3 |
| flanking-Pb force | 27.4 | 54.2 |
| far field 8–12 Å | 9.67 | 15.38 |
| energy 79 after one C_Q | 0.515 | **0.421** |
| energy 159 (pred.) | 0.041 | 0.105 |
| between-size after one C_Q | +5.0 | +12.5 |
| C5 localisation (pass / 6) | 1 / 6 | 1 / 6 (s1 0.977; s3 0.906; s0, s4, s6 delocalised, N_eff 14–34) |
| paired TOST on forces, τ = 1.7 | — | d = −9.6 [−15.9, −3.4] → **inferior** |

The seeds are bimodal: three (s1, s2, s3) at 13–16 meV/Å with localised carriers (N_eff 1.5–2.8),
three (s0, s4, s6) at 27–31 with the carrier spread over 14–34 sites and the on-site correction
saturating (p95 1.3–2.8 eV). The energy at 79 atoms is *better* than W6-on-v2 (0.42 vs 0.52) while
the between-size term is worse (12.5 vs 5.0 meV, fails the 10 meV rule). Hop-modulation knee
fraction 0–2e-3 (W6-on-v2: 1.4e-5).

**Chain status.** The automated verdict stopped the chain at 06:05 with "INCOMPLETE" — a bug in my
verdict script (it chose the epoch-averaged block, on which only two W6-on-v2 seeds pair, and then
required six); fixed at 08:35 and the L1 stage re-run: L2 (`--les_cross 1 --les_sign -1`, cached
Jacobians, six seeds) is launching now (local 0 1 2, b3 3 4 6, ≈ 1.2 h each), `lx_watch.sh L2`
then reads it against L1 by the registered rule (not inferior on forces AND far-field 8–12 Å
superior) and launches L3 from L2's models if adopted. L4 is by hand.

## 6. Record and retractions

- The plan's principle — "the student must be cutoff- and capacity-limited so the Coulomb term is
  *needed*" — was tested on the 2 × 2 (width × reach) and did not identify the charges: every
  from-scratch student, including the 12 Å ones, builds ~100–300 meV/atom of its energy on the
  Ewald term with a physical sign pattern and an unphysical net charge. What the students lack is
  not a reason to use LES but the constraint that would make the charges neutral.
- The mechanism control did confirm the L0 diagnosis: a student that *is* the teacher gives the
  charges nothing to explain.
- Two bugs of the readers were found and fixed during the run and are recorded with their effect:
  raw checkpoints had been loaded into e3nn templates without the cueq product weights (in-flight L0
  trajectory numbers approximate; every gate reading used saved models), and the stage-one model
  was read first for full-scratch (superseded).
- Base v2 (MH-1 multi-head fine-tune, no LES) remains the best base on every DFT reading in this
  record: 10.23 meV/Å neutral, 33.2 charged-frame total.

## 7. Suggested next decision (not taken)

The line's remaining automated steps (L2, L3 on the benchmark base) will complete today, but the
base is not adoptable and L1 on it is inferior to W6-on-v2. If the LES charges are to carry the
head's host term, the identifying constraint has to come from the labels or the loss, not from the
architecture: (a) a charge-neutrality / sum-rule penalty in the student's loss; (b) an LES fine-tune
of *base v2 itself* under that penalty (the base that already fits, with the charges added as a
constrained term); or (c) M1 (the head-learned antisymmetric bond-charge pattern), the plan's
registered fallback, on base v2.

## Addendum (2026-09-16 09:50) — corrections of record
The reading in §§1, 3 and 6 that the students' charges are "a fitting device" and their E_LES "not a
long-range correction" is withdrawn: the Born pattern of every from-scratch student (Cl −0.5 to −0.8,
Cs +0.4 to +0.6, Pb +1.0 to +1.4, |Pb/Cs| 2.4–2.7) is the halide-perovskite signature at ≈ Z*/√ε∞,
which is what a latent charge in an ε∞ medium should be, and E_LES of −100 to −300 meV/atom is LES's
Madelung partition of an ionic crystal's cohesion; LES-FT2's 0.03 e was the anomaly. The head's 2×
deficit on the benchmark base is the base's teacher fit (14 meV/Å where a distillation should reach a
few) anchored by the 1:1 replay, not LES. The sum-rule failure and the net charge are one finding
(Σ Z* = Σ q for environment-dependent charges); the +0.40 e compact defect charge has the wrong sign
(a neutral vacancy is locally neutral) and costs the base ≈ 1 meV/atom of spurious size dependence.
The "cutoff-limited student" hypothesis is withdrawn for ionic solids. The plan continues with
bond-transfer LES charges (q_i = Σ_j antisymmetric edge transfers × w(r_ij): Σq = 0 exactly, no
compact defect charge, the vacancy response with the right sign), the D0/D1/D2 fixes, and M1 on
base v2 in parallel — see the tracker amendment of 09:40. The overnight D3 numbers above stand as
what they are: a head on a failing base.
