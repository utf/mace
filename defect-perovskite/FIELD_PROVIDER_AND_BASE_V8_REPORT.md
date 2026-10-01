# Improving the MACE-Field provider, and what it did to the D-SCC base and head

*CsPbCl3 chlorine-vacancy project. Work of 2026-09-20 to 2026-09-30. This document is self-contained: every model, dataset, metric and threshold it uses is defined in §0, and every number quoted is in the tables here.*

## 0. Scope, conventions and definitions

**Scope.** Three pieces of work, in order: (1) an attempt to improve the MACE-Field "provider", the model that supplies polarisation response (Born effective charges Z\*, polarisability α, polarisation P) to the defect model; (2) a new neutral-data base model ("base v8") fine-tuned from the improved provider, with a readout-only refit, a learning-rate variant, a loss-weight variant ("w01"), and a diagnosis of what limits the head on each base; (3) the charged-carrier head trained on two of those bases (Phase 2b), and the large-cell localisation gate for Phase 3. Nothing is running at the time of writing.

**Conventions.** Forces are per-component RMSE in meV/Å. Energies are meV/atom. Z\* in e, α as ε∞ − I (dimensionless), P in e/Å² after folding by the polarisation quantum. Head results are read from epoch-averaged checkpoints unless stated. "Seed s" of the head means seed s trained on fold f with the pairing s → f = 0→0, 1→1, 2→2, 3→3, 4→0, 6→2; the paired comparisons pair arms by seed. Every threshold and expectation quoted below was written down before the corresponding result was read; where a prediction failed it is reported as failed.

**Definitions.**

- **MH-1.** MACE-MH-1, the multi-head foundation model: six heads, 512 channels (512×0e + 512×1o), two interaction layers, cutoff 6 Å, 89 elements, 6 439 878 parameters. Its `omat_pbe` head is the PBE energy/force model used throughout.
- **MACE-Field.** MACE with an electric-field coupling layer (one injection after layer 0: a tensor product of the node features with the field, a linear map, residual subtracted), whose energy derivatives give P (first derivative with respect to field), Z\* (mixed derivative, positions × field) and α (second derivative). **1.0.2** is the published `MACEField-MH-0-omat-dielectric` checkpoint: a 128-channel model trained from `mace-mh-0`, three heads (`pt_head`, `mp-dielectric`, `mp-ferroelectric`). It was the project's provider before this work ("MH-0 provider").
- **Provider head.** The provider's `mp-dielectric` head is the one carrying Z\* and α; `omat_pbe` (in MH-1-derived models) carries the best energies and forces but no useful response.
- **Base.** A MACEField fine-tuned on the project's neutral CsPbCl3 data (pristine 80-atom cells, neutral Cl-vacancy cells of 79 and 159 atoms; DFT energies, forces and stresses), then pruned to the elements Cl, Cs, Pb and a single head. The base supplies the forces of the neutral lattice and the atomic dipoles used by the head's coupling terms. **The base sees neutral frames only.** Base v2 (2026-09-09) is MH-1's `omat_pbe` head fine-tuned without any response; base v7 (2026-09-18) and base v8 (this work) are MACEField bases with response.
- **Base gates** (three, on the pruned base alone). Gate 1: on the 154 neutral validation frames, forces ≤ 11.2 meV/Å (base v2's 10.2 + 1.0) and energies ≤ 1.5 meV/atom. Gate 2: on the relaxed pristine cell, polycrystalline ε∞ and the per-species Frobenius norm of Z\* within 10 % of the provider head's own values on that cell. Gate 3: on the charged (+1) training frames, the base-alone force error by shell of distance from the vacancy (total, 2–4, 4–8, 8–10, 10–12, > 12 Å) not worse than base v2's by more than 3 meV/Å in any shell. Base v2's gate-3 reference: total 33.2, 2–4 Å 74.9, 4–8 30.4, 8–10 15.3, 10–12 8.9, 12+ 8.0.
- **E_PP.** The dipole–dipole energy among the provider's atomic dipoles, smoothed below r_split = 4.5 Å and screened by ε∞ = 4.0 (both fixed by the project's Phase 0). **E_cP** is the coupling of the carrier's charge to those dipoles with the same smoothing. **E_PP mismatch:** the RMS force that E_PP, evaluated with the trained base's own dipoles, exerts on the 154 neutral validation frames; threshold τ = 1.7 meV/Å; if exceeded, the registered remedy is a readout-only refit of the base with E_PP switched on inside it.
- **Head (D-SCC head).** The carrier model stacked on a base: a one-particle Hamiltonian H0 built from the base's first-layer features (invariants and l = 1 vectors), filled with Gaussian smearing, giving per-atom charge changes dq (Σdq = Q, Q = +1 here), an energy correction with a per-charge-state calibration constant C_Q, and forces. It is exactly zero on neutral frames by construction. Trained on the charged frames (785 training, 262 held-out per fold, four folds) with the "W6 schedule": 60 epochs, learning rate 2e-3 with cosine decay, batch 4, energy weight 0.05. The readouts are bounded; the trainer logs the saturation penalty on their pre-activations (`sat`) and the 95th percentile of the head's on-site energy shift in eV (`dEp95`).
- **Arms.** **A** (the control): the head as originally shipped on base v2 ("W6"), with a Mulliken-type carrier–host energy at all ranges, scaled by a learned factor s. **B_s:** the same Mulliken near field below about 8 Å (switch of width 2 Å), the provider's dipoles (E_cP) as the far field beyond, coefficient fixed at 1, plus E_PP. **B_0:** no near-field term (H0 alone near the carrier), far field as in B_s, plus E_PP. In the B arms the head adds E_PP unless the base already carries it.
- **Head metrics.** Per-component force RMSE on the held-out charged frames: total, the 4–12 Å band, and the shells above. **Paired TOST** across the six seeds at τ = 1.7 meV/Å: "equivalent" if the 90 % confidence interval of the paired mean difference lies within ±τ, "superior"/"inferior" if it lies wholly on one side of zero, "inconclusive" otherwise. Adoption rule: an arm is adopted over the control on superiority or equivalence on the total or the 4–12 Å band. **Energy gate:** C_Q profiled at 79 atoms transfers to 159 atoms within 10 meV, and the 79-atom energy after C_Q is ≤ 0.6 meV/atom.
- **C5 localisation.** On 1191 neutral-vacancy training frames, the carrier level's separation from the band edge (eV) and the hole's participation number N_eff; a frame passes at separation ≥ 0.2 eV and N_eff ≤ 4; reported as pass fraction and medians. **Large-N gate (Phase 3 entry):** on embedded 2×2×2 (639 atoms) and 3×3×3 (2159 atoms) structures, separation ≥ 0.15 eV, N_eff ≤ 4, Σdq within 5 Å of the vacancy ≥ 0.8.
- **Replay and replay drift.** Fine-tuning a foundation model uses a "replay" set of its own training-distribution structures labelled by the model itself, so the loss on it is zero at step 0; the base's final error on those frames ("replay drift", E / F) measures how far the backbone moved.
- **Hardware.** "Local card": NVIDIA A4000, 16 GB. "b3": two usable Quadro RTX 6000 cards, 24 GB each.

## 1. The Field provider: MACE-MH-1-Field (2026-09-20 to 2026-09-27)

### 1.1 Background

The first attempt (2026-09-19/20) put MACE-Field's coupling layers on a frozen MH-1 and trained only the coupling on the dielectric data. The zero-field model was MH-1 to round-off, but on that attempt's own validation split Z\* stayed near 0.45 against about 0.23 for the published provider, and the plan was replaced by the two-model plan below.

### 1.2 What the published provider is

- **1.0.2** was trained from `mace-mh-0` on the authors' own files: 5501 dielectric frames (Z\*, α), 2500 ferroelectric frames (P only; energy and force weights 0), and a 372 355-frame OMAT-PBE replay set, with MACE's random 10 % validation split at seed 123, batch 1 on four GPUs, learning rate 1e-4, EMA 0.99999, float32, for 360 epochs; 1.0.2 is the epoch-260 checkpoint. The authors supplied the archive with these files and the split is regenerated exactly.
- On the exact validation sets (550 dielectric frames, 250 ferroelectric), in float64: **Z\* 0.2294 e, α 0.990, P 0.00328 e/Å²**. These are the registered targets ("beat 1.0.2").
- The replay labels are MH-1's own predictions, not DFT, so the plan's criterion "energies and forces within 5 % of MH-1's error against DFT" is not evaluable; the replay drift is reported in its place. 1.0.2's own replay drift from its labels is 54 meV/atom / 194 meV/Å (41 / 135 from its own base by its log).
- The polarisability in the authors' files is ε∞ − I, which is exactly MACE-Field's output; no conversion.

### 1.3 Construction

An eight-head MACEField was built from MH-1: MH-1's six heads keep their tensors; two new heads (`mp-dielectric`, `mp-ferroelectric`) are copies of `omat_pbe`'s readout entries; one field-coupling injection after layer 0 (1 048 576 new parameters, the linear map initialised to zero so the field residual is exactly 0 at zero field); 7 506 310 parameters in float64. Checks: all six MH-1 heads reproduce MH-1 at zero field to ≤ 9.1e-10 meV/atom and ≤ 7.2e-11 meV/Å on a fixed 2000-structure set (1400 MH-1 replay frames, 600 CsPbCl3 defect frames); the cueq (accelerated tensor product) path matches the e3nn path in float64 to a relative loss difference of 2.6e-10 and gradient differences of 3e-8, so cueq was used (2.2× faster).

Training: random batches drawn across all pools each step under an atom budget (never size-sorted), a two-pass batch (the nested derivatives for Z\*/α on dielectric frames, up to 96 atoms, in one pass; energies/forces/P in the other); loss weights E 1 / F 100 / σ 1 / P 100 / Z\* 100 / α 100 (the paper's table), Huber δ 0.01, cell-aware folded polarisation metric; Adam with AMSGrad, EMA 0.995, weight decay 1e-8, gradient clip 10; learning rate reduced on plateau of validation Z\* (factor 0.8, patience 5); early stop at patience 20 on validation Z\*; cap 150 epochs. Labels as the published run used them: dielectric frames Z\* and α; ferroelectric frames P only; replay E, F, σ.

### 1.4 Runs and results

| run | trainable parameters | learning rate | epochs (best) | validation Z\* | α | P | replay drift E / F | step cost |
|---|---|---|---|---|---|---|---|---|
| **A2** ("Model A": MH-1 energies untouched) | field coupling + the two new heads' readouts (MH-1 entries masked; asserted bit-identical at the end) | 1e-3 | 150 = cap (149) | 0.3471 | 0.952 | 0.00391 | 0 | 0.68 s, 9 GB |
| **B2** ("Model B": the published recipe on MH-1) | all | 1e-4, reduced by the scheduler to 3.3e-5 | 72, early stop (best 51) | 0.2428 | 0.995 | 0.00359 | 30 / 98 | 0.97 s, 12.8 GB |
| **A2B** (user's addition: A2's epoch-95 checkpoint, all weights unfrozen) | all | 3e-5 | 150 = cap (148) | **0.2387** | **0.889** | 0.00361 | 11.7 / 41.5 | 0.97 s, 14.1 GB |
| 1.0.2 (reference) | | | | 0.2294 | 0.990 | 0.00328 | 54 / 194 | |

Observations: A2 improved monotonically to the cap and was still gaining about 0.0005 in Z\* per epoch; α reached the target early. B2 at 1e-4 plateaued at validation Z\* 0.25–0.28 from epoch 20 while its training Z\* kept falling (overfitting, not convergence); its replay drift was flat at about 30 / 95 from epoch 3. A2B descended steadily and without instability from A2's state and was still improving (about 0.0003 per epoch) at the cap; its drift grew roughly in proportion to its Z\* gain and ended at a third of B2's.

**Against the registered targets:** no model beats 1.0.2 on Z\* (A2B misses by 0.009) or on P (every run, including the published one, sits at 0.0033–0.0040); A2 and A2B beat it on α. For scale, 1.0.2 reached its Z\* after 360 epochs of batch-1 training of a 128-channel model; A2B was still improving at 150 epochs.

**Checks on the final models (float64).** Identities: ΣZ\* per frame ≤ 9e-15 e and α asymmetry ≤ 1e-15 on 100 validation frames (all three). CsPbCl3 on the base-v2-relaxed cell, |Z\*| Cl / Cs / Pb against MP DFPT 3.52 / 2.24 / 6.21 (±10 %): A2 3.37 / **1.75 (−22 %)** / 5.65; B2 3.47 / 2.09 / 5.77; A2B 3.45 / 2.17 / 5.89; ε∞ 3.55 / 3.54 / 3.66 against 3.55. Zero-field exactness of A2 after training (the MH-1 heads): 2.6e-6 to 4.9e-6 meV/atom and ≤ 5.3e-5 meV/Å over the six heads on the 2000-structure set (plan: ≤ 1e-4 and ≤ 1e-3), after restoring MH-1's own symmetric-contraction constant tables, which MACE's format converters regenerate with ~1e-8 differences.

**Selection.** Where MH-1's energies must be untouched: A2. Otherwise **A2B**: the lowest Z\* of the unfrozen models, α 0.889, the CsPbCl3 check passed, and a drift of 11.7 meV/atom / 41.5 meV/Å from MH-1 against B2's 30 / 98. The user chose A2B as the provider for the base work below.

## 2. Base v8: the perovskite base fine-tuned from A2B (2026-09-27 to 2026-09-28)

### 2.1 Setting: the bases and head results that existed before

| | base v2 (MH-1 `omat_pbe`, no response) | base v7 (1.0.2 provider, `mp-dielectric` head) |
|---|---|---|
| neutral validation F / E | 10.23 / 1.16 | 13.23 / 0.88 (gate 1 fail) |
| gate 2 (response) | none | pass (within 0.5 %) |
| gate 3: charged shells total / 2–4 / 4–8 / 8–10 Å | 33.2 / 74.9 / 30.4 / 15.3 (reference) | 46.8 / 104.9 / 43.9 / 20.5 (+13.6 / +30.0 / +13.5 / +5.2; fail) |
| E_PP mismatch | — | 1.49 (pass) |
| replay drift E / F | 4.1 / 17 | 17.3 / 72 |

The head on base v2 ("W6", six seeds, medians): total 11.6, 4–12 Å 10.5, shells 19.3 / 11.0 / 10.0 / 9.0 / 7.1, 79-atom energy 0.52, C_Q transfer 4.3 meV; C5 pass 0.71–0.98, separation 230–422 meV, N_eff 1.26–1.35. Its weakness, found in the project's Phase 0, is that it does not reproduce the dielectric screening of the host at large cell size; the plan ("Plan P") couples the carrier to the host through the provider's dipoles instead, which is why a MACEField base is needed. On base v7 (opened by user override although gates 1 and 3 failed), six seeds each: A 15.42 / 13.79 (total / 4–12 Å), B_s 15.96 / 14.42 (equivalent to A, adopted), B_0 21.25 / 18.71 (inferior); C5 medians A 0.98 / 457 meV / 1.56, B_s 0.89 / 346 / 1.67, B_0 0.07 / 115 / 4.6; the large-N gate failed for B_s (separation 0.023 eV at 639 atoms, 0.021 at 2159; N_eff 20.5 and 38.2).

### 2.2 Head choice, recipe and labels

**Head choice** (A2B, float64, the rule: the head with the lower initial neutral force error unless another head's response is closer to the reference by more than the force difference warrants):

| A2B head | initial neutral validation F | energy offset | relaxed-cell ε∞ (reference 3.547) | Z\* Cl / Cs / Pb (reference 3.52 / 2.24 / 6.21) |
|---|---|---|---|---|
| `omat_pbe` | 40.9 | +30 meV/atom | 1.69 | 65–70 % low |
| `mp-dielectric` | 83.6 | −624 meV/atom | 3.66 (+3 %) | 3.45 / 2.17 / 5.89 (−2 / −3 / −5 %) |
| `mp_pbe_refit_add` | 239 | +2912 | no response | no response |

Only `mp-dielectric` carries the response the base exists to provide; the energy offset is absorbed by the estimated E0s. Chosen: **mp-dielectric**, as for base v7.

**Recipe** (base v7's, which is base v2's with the response terms added):

| | base v2 | base v8 |
|---|---|---|
| foundation, head | MH-1, `omat_pbe` | A2B, `mp-dielectric` |
| loss | energies 10 / forces 10 | energies 10 / forces 10 / stress 1 / P 10 / Z\* 1 / α 1 |
| replay | MH-1's 10 000 FPS-selected OMAT frames, MH-1 labels | the same 9360 frames as base v2's replay, relabelled E/F/S by the provider head, no response labels |
| learning rate, EMA, clip, weight decay | 1e-4, 0.9999, 1.0, 0 | 3e-4 (base v7's) and 1e-4 (base v2's), same otherwise |
| epochs, batch, precision | 30, 8, float32 + cueq | 30 (extended to 60), 4, float32 + cueq |
| E0s | MACE's estimate + isolated atoms at weight 1000 | the same (Cl −0.475, Cs +1.747, Pb +1.103 eV) |
| E_PP during training | — | off (pp_r_split 0) |

**Labels.** The neutral training and validation frames keep their DFT E/F/S and receive P, Z\* and α from A2B's `mp-dielectric` head. Checks: no outliers (largest |Z\*| component 4.52 e, acoustic-sum residual 1e-7, α ≤ 3.0); the Z\* labels differ from the MH-0 provider's by 0.11 e RMS (α by +0.25). For comparison, A2B's `omat_pbe` head would have labelled the same frames with Z\*(Cl) 1.22 e against 3.41 e and ε∞ 1.71 against 3.72.

### 2.3 Results

Launched 2026-09-27; the 60-epoch rows are resumptions of the 30-epoch runs from their last checkpoint (optimiser and EMA state continued).

| base | neutral F (≤ 11.2) | neutral E (≤ 1.5) | gate 2 retention | charged total | 2–4 Å (74.9 + 3) | 4–8 | 8–10 | gates 1 / 2 / 3 | E_PP mismatch (τ 1.7) | replay drift E / F |
|---|---|---|---|---|---|---|---|---|---|---|
| v8, lr 3e-4, 30 ep | 12.79 | 0.54 | 0.1–0.2 % | 34.6 (+1.4) | 76.8 (+1.9) | 32.3 (+1.9) | 16.3 (+1.0) | FAIL / pass / pass | 2.09 exceeds | 14.9 / 143 |
| v8, lr 1e-4, 30 ep | 14.01 | 0.78 | 0.1 % | 33.3 (+0.1) | 77.0 (+2.1) | 29.8 (−0.6) | 15.8 (+0.5) | FAIL / pass / pass | 2.23 exceeds | 5.3 / 38 |
| v8, lr 3e-4, 60 ep | 11.56 | 0.84 | 0.1 % | 35.5 (+2.3) | 78.9 (+4.0) | 33.4 (+3.0) | 16.3 (+1.0) | FAIL / pass / FAIL | 2.07 exceeds | 15.4 / 158 |
| v8, lr 1e-4, 60 ep | 12.39 | 0.48 | 0.1 % | 34.1 (+0.9) | 78.7 (+3.8) | — | — | FAIL / pass / FAIL | 2.23 exceeds | 6.0 / 49 |
| v7 (reference) | 13.23 | 0.88 | 0.5 % | 46.8 (+13.6) | 104.9 (+30.0) | 43.9 (+13.5) | 20.5 (+5.2) | FAIL / pass / FAIL | 1.49 pass | 17.3 / 72 |
| v2 (reference) | 10.23 | 1.16 | — | 33.2 | 74.9 | 30.4 | 15.3 | | | 4.1 / 17 |

Readings at the time: the A2B-derived base transfers to the charged cells at base v2's level where the MH-0-derived base was 14 meV/Å worse; the response is retained to 0.1 %; no run passes gate 1; the extra 30 epochs bought 1.2–1.6 meV/Å on the neutral validation and cost about 2 meV/Å in the charged 2–4 Å shell for both rates. Both runs were still improving at 30 epochs (0.08–0.1 meV/Å per epoch). The user chose the **30-epoch lr 1e-4 base** (passes gates 2 and 3) as the base of record.

### 2.4 The E_PP mismatch, the refit, and the decision

The mismatch exceeds τ on every base v8 run, so the registered remedy was run on the chosen base: a readout-only refit with E_PP switched on inside the base (r_split 4.5, ε∞ 4.0), on base v8's own labelled training file (8720 trainable parameters, 5 epochs, lr 1e-4, batch 2, best-force epoch kept). Acceptance, written with the script: validation forces ≤ base-alone + 0.2 and energies ≤ base-alone + 0.2. The refit was accepted, and its base was then re-gated with the rule, registered the same morning, that it would be adopted only if gates 2 and 3 still passed. It lost gate 3. A diagnostic (no training) attributed the loss:

| base of record (30 ep, lr 1e-4) | neutral F | neutral E | charged total | 2–4 Å (+3 allowed) | 4–8 | gate 3 |
|---|---|---|---|---|---|---|
| alone | 14.01 | 0.78 | 33.3 (+0.1) | 77.0 (+2.1) | 29.8 | pass |
| + E_PP, readouts untouched | 13.92 | 0.90 | 34.0 (+0.8) | 77.7 (+2.8) | 30.8 | pass |
| + E_PP, readouts refit (accepted: 13.62 ≤ 14.21, 0.51 ≤ 0.98) | 13.62 | 0.51 | 34.5 (+1.3) | 78.8 (+3.9) | 31.1 | FAIL |

E_PP's own force on the charged frames costs 0.7 meV/Å in the 2–4 Å shell and the readout change a further 1.1. The refit buys 0.3 meV/Å on the neutral set for 1.1 in the charged near shell, the same trade as the 30-to-60-epoch extension.

**Decision (user, 2026-09-28):** proceed with the base alone, the head adding E_PP in the B arms and the control carrying none. This is the plan's own composition and the base-v7 configuration, so the control is like-for-like with the base-v7 one. The re-gating rule and the clause forbidding a head-added E_PP on an unmatched base were the assistant's, not the plan's, and were waived.

## 3. Phase 2b on the 30-epoch base v8 (2026-09-28 11:45 to 2026-09-29 01:46)

Arms A, B_s, B_0, six seeds each, W6 schedule, no Brillouin-zone regulariser in the loss (its readout kept per checkpoint), identical settings to the base-v7 arms except the base. Registered expectations: B_s equivalent to A; B_0 inferior to both; 79-atom bound state present for A and B_s; large-N undecided.

| arm | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E (≤ 0.6) | C_Q transfer (≤ 10 meV) | energy gate seeds | C5 pass / separation / N_eff (medians) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 15.80 | 14.45 | 26.0 | 15.0 | 13.7 | 12.6 | 9.1 | 0.87 | 8.6 | 0 / 6 | 0.58 / 222 meV / 1.34 |
| B_s | 15.56 | 13.95 | 27.0 | 14.7 | 13.2 | 11.5 | 8.4 | 0.44 | 5.1 | 6 / 6 | 0.03 / 73 meV / 2.5 |
| B_0 | 15.64 | 14.25 | 25.9 | 15.3 | 13.2 | 10.8 | 8.7 | 0.86 | 7.8 | 0 / 6 | 0.09 / 108 meV / 1.9 |

Per seed (seeds 0 1 2 3 4 6): A 15.19 / 16.63 / 15.31 / **25.99** / 15.75 / 15.86; B_s 15.13 / 16.94 / 15.41 / 17.74 / 15.71 / 15.11. The control's seed 3 is a head with no bound state (C5 N_eff 16), the same failure the base-v7 control showed on its seed 1.

Paired TOST: B_s against A, total +1.45 with 90 % CI [−1.31, +4.20], 4–12 Å +1.59 [−0.93, +4.12], both inconclusive (the width comes from the one unbound control pair), B_s superior in the 10–12 and 12+ shells; B_0 against A inconclusive throughout; B_0 against B_s inconclusive throughout (total −0.67 [−2.63, +1.28]). **Verdict by the rule: no arm adopted** (neither superior nor equivalent; none inferior, so the r_c = 10 Å fallback is not triggered). Expectations: "B_s equivalent to A" not resolved; "B_0 inferior" wrong, it was level; "bound state for A and B_s" wrong for B_s, whose carrier is shallow on every seed (60–108 meV against 346 on base v7) though its forces are unchanged.

Against the base-v7 arms on the same folds, B_s differs by at most 0.9 meV/Å on four seeds and is 1.7 better on seed 6: the better base did not move the head's charged-frame forces. The three arms sit at 15.6–15.8, which is 1.6 above the base's neutral error of 14.0. That observation led to §4.

## 4. What sets the head model's error: the floor

The base's error on a charged frame has two parts: the ordinary error it makes on any perovskite cell (measured by the neutral validation: v2 10.2, v7 13.2, v8 14.0), and the error from having no carrier, which is large and concentrated near the vacancy (about 75 meV/Å in the 2–4 Å shell for v2 and v8 alike). On the charged frames the second part dominates, so v2 and v8 look the same (33.2 against 33.3). The head removes only the second part, being exactly zero on neutral frames; what remains after the head is the first.

| base | neutral validation F | head model, charged frames (median over seeds) |
|---|---|---|
| v2 | 10.2 | 11.6 |
| v7 | 13.2 | 15.4 (A) |
| v8, 30 ep | 14.0 | 15.6–15.8 (all arms) |

Each head model sits about 1.5 above its base's neutral error; the v8-to-v2 gap after the head (4.0) equals the gap in neutral error (3.8); v7 and v8 heads coincide because their neutral errors are close, although v7 is far worse on charged frames before the head. The base-v8 gate-1 failure is therefore the head's floor, and gate 3 (the base's own charged error) is not what the head's forces respond to.

**Test** (registered prediction: control arm, seed 0, on the 60-epoch lr 3e-4 base v8, neutral 11.56, gate 3 lost: held-out total 13.0 ± 0.7 against 15.19 on the 30-epoch base). Result: **11.90**, 4–12 Å 10.97, shells 18.9 / 11.3 / 10.5 / 10.0 / 6.9, 79-atom energy 0.66, C_Q transfer 1.2 meV; C5 pass 0.968, separation 357 meV, N_eff 1.36. Direction confirmed and the effect larger than predicted: the head lands 0.3 above the base's neutral error, at the base-v2 head's level (W6 seed 0: 11.21, 2–4 Å 18.3). The reading holds when the head binds the carrier; §5–6 show its limit when it does not.

## 5. Base v8 w01: the response weights divided by ten (2026-09-28 to 2026-09-29)

With the floor identified, the question became why base v8's neutral error (14.0 at 30 epochs, 11.6 at 60) is above base v2's 10.2 when the recipes share weights (E 10 / F 10), learning rate, EMA, precision and length. The differences that remain: the starting head (MH-1's `omat_pbe` at about 13 meV/Å after one epoch, against A2B's `mp-dielectric` at 83.6 at epoch 0, a readout never trained on forces); the response terms in the target-frame loss, which a gradient-share measurement on the record (2026-09-17, same recipe) found to carry 40 times the force gradient on the target frames (P 0.21, α 0.14, Z\* 0.12 against F 0.011); batch 8 against 4; and the replay set. The response was being held to 0.1 % against gate 2's 10 % allowance in all four base-v8 runs.

**One change** (user's choice, 2026-09-28): `base_v8_w01` = base_v8 at lr 3e-4 for 60 epochs with P 1 / Z\* 0.1 / α 0.1 (was 10 / 1 / 1); registered prediction: gate 1 passes with gate 2 within 10 %; gate 3 as it falls.

| epoch | 0 | 1 | 9 | 19 | 29 | 39 | 59 |
|---|---|---|---|---|---|---|---|
| w01 neutral validation F | 22.6 | 19.2 | 12.9 | 12.2 | 11.7 | 11.0 | 10.99 |
| original weights, same run | 27.5 | | | | 12.8 | | 11.6 |

Final gates of w01: **gate 1 PASS** (F 10.99, E 1.13), **gate 2 PASS** (ε∞ 3.667 against 3.661; Z\* within 0.2 %), gate 3 FAIL (total 36.3, +3.1; 2–4 Å 79.5, +4.6; 4–8 34.7, +4.3; 8–10 +1.4; 10–12 +0.4; 12+ −1.0); E_PP mismatch 2.11; replay drift 16.7 / 145. The prediction held; gate 3 moved a further 0.6–1.3 meV/Å in the near shells, consistent with the neutral fit and the base-alone charged transfer trading against each other. Training time 11.6 h for 60 epochs on one b3 card.

**Floor check on w01** (same diagnostic as §4: control arm, seed 0; registered prediction about 11.3). Result: **23.71** (2–4 Å 45.8, 79-atom energy 1.25, C_Q transfer 26 meV); C5 pass 0.078, separation 367 meV, N_eff 11.6: the level is separated but the hole is spread over about twelve atoms. The prediction is recorded as missed; it did not allow for a control seed that fails to bind, a failure already seen on base v7 (seed 1) and the 30-epoch base v8 (seed 3), and now on seed 0.

## 6. Phase 2b on base_v8_w01 (2026-09-29 11:45 to 2026-09-30 04:52)

Same arms, seeds and settings as §3, on w01 (base alone, head adds E_PP in the B arms). Registered expectations: control about 11.3 by the floor reading; B_s equivalent to A and passing the energy gate where A does not; B_s bound state and B_0 undetermined.

### 6.1 Forces, energies, localisation

| arm | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E | C_Q transfer | energy gate seeds | C5 pass / separation / N_eff (medians) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 19.95 | 17.11 | 37.8 | 19.0 | 13.5 | 11.4 | 7.5 | 0.85 | 23.2 | 1 / 6 | 0.41 / 264 meV / 3.1 |
| **B_s** | **13.89** | **12.42** | 24.0 | 13.3 | 11.2 | 9.8 | 7.9 | 0.52 | 5.6 | 5 / 6 | 0.07 / 130 meV / 1.47 |
| B_0 | 19.56 | 16.96 | 36.4 | 19.2 | 13.0 | 10.0 | 7.1 | 0.93 | 22.1 | 0 / 6 | 0.29 / 135 meV / 1.63 |

Per seed (0 1 2 3 4 6):

| arm | s0 | s1 | s2 | s3 | s4 | s6 |
|---|---|---|---|---|---|---|
| A total | 23.71 | 16.19 | 35.72 | 24.13 | 13.40 | 11.18 |
| A C5 pass / sep / N_eff | 0.08 / 367 / 11.6 | 0.39 / 320 / 3.8 | 0.04 / 191 / 13.9 | 0.58 / 222 / 1.6 | 0.42 / 191 / 2.4 | 0.92 / 305 / 1.4 |
| B_s total | 13.72 | 13.90 | 13.88 | 13.03 | 13.95 | 14.37 |
| B_s C5 | 0.28 / 165 / 1.45 | 0.00 / 98 / 1.44 | 0.15 / 145 / 1.41 | 0.08 / 148 / 1.60 | 0.07 / 115 / 1.49 | 0.02 / 73 / 1.85 |
| B_0 total | 15.72 | 21.18 | 20.06 | 19.07 | 20.14 | 17.71 |
| B_0 C5 | 0.40 / 172 / 1.3 | 0.17 / 97 / 1.4 | 0.00 / 28 / 38.6 | 0.02 / 84 / 5.7 | 0.51 / 204 / 1.6 | 0.84 / 286 / 1.7 |

B_s's one energy-gate failure (seed 6) is a 13.4 meV C_Q transfer with a 79-atom energy of 0.46.

### 6.2 The control's instability

The control's spread is a late-training instability, visible in the logs. Held-out error during training (meV/Å) and the 95th-percentile on-site shift `dEp95` (eV) at the end:

| run | epoch 19 | 29 | 39 | 59 | dEp95 at 59 | readout saturation at 59 |
|---|---|---|---|---|---|---|
| A s0 (w01) | 14.0 | 13.6 | 32.3 | 23.2 | 1.02 | 7e-7 |
| A s1 | 13.3 | 14.6 | 17.3 | 15.5 | 1.22 | 4e-8 |
| A s2 | 13.9 | 57.5 | 33.0 | 41.5 | 1.15 | 6e-7 |
| A s3 | 13.4 | 65.4 | 29.3 | 23.3 | 1.52 | |
| A s4 | 16.5 | 14.3 | 13.2 | 13.6 | 0.56 | |
| A s6 | 16.0 | 12.3 | 11.8 | 11.1 | 0.22 | |
| A s0 on the 60-epoch original base v8 (§4) | 14.0 | 12.8 | 12.4 | 12.0 | 0.20 | 2e-9 |
| B_s s0 (w01) | 17.8 | 25.4 | 15.0 | 13.6 | 0.15 | 4e-10 |
| B_s s1 | 18.3 | 15.9 | 14.8 | 14.0 | 0.29 | 1e-9 |
| B_s s2 | 21.0 | 19.5 | 17.1 | 13.9 | 0.14 | |

On w01 the Mulliken control reaches 13.3–14.0 by epoch 20–30 and then, on four of six seeds, the head's on-site shift jumps to about 1 eV and the bounded readouts saturate a hundred to a thousand times more; the held-out error doubles or worse and the epoch-averaged checkpoint ends delocalised. The two seeds that stay stable end at the floor (13.4, 11.2) with a bound carrier. The polarisation arm on the same base and folds never shows it. The control on the 60-epoch original base v8 (one seed) was stable. The feature standardisation statistics are the same on both bases (|μ|/σ medians 14.5–14.8 / 22.4 / 13.0–14.5 for Cl / Cs / Pb, no dead channels), so the base's feature scale is not the difference. Why w01 provokes it is not known.

### 6.3 Paired comparisons and verdict

| pair | total: mean d, 90 % CI, reading | 4–12 Å |
|---|---|---|
| B_s against A | +6.91 [−0.71, +14.53], inconclusive | +5.05 [−0.41, +10.50], inconclusive |
| B_0 against A | +1.74 [−5.88, +9.37], inconclusive | +0.98 [−4.36, +6.32], inconclusive |
| B_0 against B_s | −5.17 [−6.84, −3.50], **inferior** | −4.07 [−5.41, −2.72], **inferior** (and inferior at 2–4, 4–8, 8–10; equivalent at 10–12 and 12+) |

By the rule: no arm adopted over the control (the control is not a stable reference on this base, so neither superiority nor equivalence can be shown; none is inferior, so no fallback); between the B arms, B_s. Expectations: "control about 11.3" wrong on four of six seeds, right on the two stable ones; "B_s passes the energy gate where A does not" held (5/6 against 1/6); "B_0 undetermined" resolved to inferior, as on base v7.

### 6.4 Large-N localisation (Phase 3 entry gate)

B_s's seed-median checkpoint (seed 2, 13.88, averaged weights), registered fill, registered expectation: fails on separation (already 145 meV in the training cell).

| structure | separation (≥ 0.15 eV) | N_eff (≤ 4) | Σdq within 5 Å (≥ 0.8) |
|---|---|---|---|
| training 79-atom cell (+1) | 0.126 | 1.4 | 0.47 |
| embedded 2×2×2, 639 atoms | 0.015 | 23.3 | 0.29 |
| embedded 3×3×3, 2159 atoms | 0.010 | 57.3 | not computed (out of memory on 24 GB with this base) |
| base-v7 B_s, same read | 0.308 / 0.023 / 0.021 | 1.6 / 20.5 / 38.2 | |

**FAIL**, as expected: the level is compact but shallower than on base v7 already in the training cell, and it collapses with cell size in the same way.

### 6.5 Cost

| | base v7 arms (base 13.5 MB) | base v8 arms (base 65 MB, 512 channels) |
|---|---|---|
| A, s/epoch (local card) | 40 | 67 (peak 6.4 GB) |
| B_s, s/epoch (b3) | 90 | 127–132 (peak 14.6 GB) |
| B_0, s/epoch (b3) | 84 | 126 |
| twelve B runs on two b3 cards | ≈ 9 h | ≈ 13 h |

## 7. Where it stands

**What the evidence supports.**

- The improved provider (A2B) is a better response model than the published one on α and close on Z\* (0.239 against 0.229), with a third of Model B's drift from MH-1; neither it nor any run here beats 1.0.2 on Z\* or P.
- A base fine-tuned from A2B transfers to the charged cells at base v2's level (the MH-0-derived base was 14 meV/Å worse there) and keeps the response to 0.1–0.2 %. With the response weights divided by ten it passes gate 1 (10.99) and gate 2; gate 3 then fails by 1.6 meV/Å in the 2–4 Å shell.
- The head model's charged-frame error is set by the base's ordinary (neutral) force error when the head binds the carrier: 11.9 on a base with 11.56, 11.2–13.4 on w01's stable control seeds, 15.6–15.8 on the 14.0 base. The base's own charged-frame error (gate 3) does not govern it.
- On w01, B_s is the only arm that trains stably on every seed; its forces (13.0–14.4, median 13.89) are 1.2–1.7 meV/Å better than B_s on either earlier base and 2.3 above the base-v2 head; it passes the energy gate on five of six seeds; B_0 is inferior to it by 5.2 meV/Å with a tight interval, so the near-field term stays.
- B_s's carrier is compact on every seed (N_eff 1.4–1.9) but shallow (73–165 meV in the training cell), and the bound state collapses with cell size exactly as on base v7. The Phase 3 entry gate fails.

**What it does not settle.**

- Adoption of B_s over the Mulliken control, by the plan's paired test: the control destabilises on four of six seeds on w01, for a reason not identified, so the test is inconclusive on both metrics although the mean favours B_s by 6.9 meV/Å.
- Whether the shallowness of B_s's level is a property of the A2B-derived bases or of the weaker provider response in the loss: on the 30-epoch base (response weights 10 / 1 / 1) it was 73 meV, on w01 (1 / 0.1 / 0.1) 130 meV, on base v7 346 meV.

**Open decisions.**

1. Adopt B_s on w01 as the carried model despite the unresolved control, or rerun the control with whatever stabilises it (not identified).
2. The registered localisation remedy: one retrain of B_s with the gap-regulariser weight tripled. The expectation on record is null, with the reason: on every checkpoint the regularised quantity (the pristine cell's gap at Γ) is already direct with zero residual, so a tripled weight adds no gradient; the failure is the depth and size dependence of the vacancy level in H0, which the pristine gap does not constrain.
3. Phase 3 entry, or the plan's own next line after a failed remedy, "stop and report".

**Assistant's recommendation (mine, not a result).** Carry B_s on w01 as the model of record for forces and energies, with the control's instability recorded as a limitation of the adoption test rather than of B_s. Do not run the tripled-gap retrain; the record predicts it changes nothing, and the GPU time is better spent on the open question of the level's depth. The localisation failure is the same shallow-carrier finding the project met on base v2 and base v7, now with a compact but shallow level; it is a property of the head's Hamiltonian, not of the base, and lies outside what Plan P's coupling terms can change.

## Appendix A. Run names, settings and locations

Run directories: field provider `~/runs/mh1f2/` (deliverables `deliver/MACE-MH-1-Field-{A2,B2,A2B}.model`, exact splits `data_mh0/`); bases `~/runs/dscc/p1v8/{base_v8, base_v8_lr1e4, base_v8_w01}/` with the 30-epoch artefacts under `ep30/`, refit records `base_v8_lr1e4/ep30/{P2_REFIT.json, P1_GATES_epp.json, P1_GATES_pp_norefit.json}`; head arms `~/runs/dscc/dscc_p2v8{a,bs,b0}_s{0,1,2,3,4,6}` (30-epoch base), `dscc_p2v8w{a,bs,b0}_s*` (w01), floor tests `dscc_p2v8x_a60_s0` and `dscc_p2v8x_aw01_s0`; summaries `~/runs/dscc/p2v8/` and `~/runs/dscc/p2v8w/` (paired-test JSON, C5 JSON, large-N JSON). Base-v7 arms `dscc_p2a_s*`, `dscc_p2bs_s*`, `dscc_p2b0_s*`; base-v2 head `dscc_w6_s*`.

Head training command (arm B_s; A replaces the last line with `--host_coupling mulliken --cache_base 1`; B_0 uses `--near_field none`):

```
dscc_train.py --seed S --fold F --epochs 60 --directional 1 --regime B --base BASE.pt --coupling 0 --scf_free 1
  --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4
  --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 0 --bz_readout 1
  --host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0
  --r_c 8.0 --delta 2.0 --near_field mulliken
```

Base fine-tune command (base v8; w01 appends `--max_num_epochs 60 --polarization_weight 1 --becs_weight 0.1 --polarizability_weight 0.1`):

```
mace run_train --name NAME --heads "{'Default': {'train_file': train_v8_iso.xyz, 'valid_file': valid_v8.xyz}}"
  --foundation_model MACE-MH-1-Field-A2B.model --foundation_head mp-dielectric --pt_train_file replay_v8.xyz
  --multiheads_finetuning True --pseudolabel_replay False --keep_isolated_atoms True
  --model MACEField --loss universal_field --error_table PerAtomFieldRMSE
  --energy_weight 10 --forces_weight 10 --stress_weight 1 --polarization_weight 10 --becs_weight 1 --polarizability_weight 1
  --compute_forces True --compute_stress True --compute_polarization True --compute_becs True --compute_polarizability True
  --force_mh_ft_lr True --lr LR --ema --ema_decay 0.9999 --weight_decay 0.0 --clip_grad 1.0
  --max_num_epochs 30 --batch_size 4 --valid_batch_size 4 --default_dtype float32 --enable_cueq True
  --pp_r_split 0 --pp_eps_inf 4.0 --seed 1 --eval_interval 1 --head_grouped_batches True
```

## Appendix B. Reference tables in full

Base-v2 head (W6), six seeds, epoch-averaged where available: totals 11.21 / 11.46 / 11.71 / 11.08 / 12.15 / 11.94 (seeds 0 1 2 3 4 6); medians total 11.6, 4–12 Å 10.5, shells 19.3 / 11.0 / 10.0 / 9.0 / 7.1; 79-atom energy 0.52; C_Q transfer 4.3 meV; C5 (seeds 0–3) pass 0.71–0.98, separation 230–422 meV, N_eff 1.26–1.35.

Base-v7 arms, per seed (0 1 2 3 4 6): A 15.31 / 18.82 / 15.34 / 16.54 / 14.40 / 15.50; B_s 15.13 / 16.03 / 15.89 / 17.31 / 15.70 / 16.81; medians A 15.42 / 13.79 / 27.1 (total / 4–12 / 2–4), B_s 15.96 / 14.42 / 26.7, B_0 21.25 / 18.71 / 38.4; 79-atom energy A 0.61, B_s 0.45, B_0 0.74; TOST B_s against A equivalent on both metrics, B_0 inferior on both; C5 medians A 0.98 / 457 meV / 1.56, B_s 0.89 / 346 / 1.67, B_0 0.07 / 115 / 4.6.

Provider validation (exact seed-123 sets, float64): 1.0.2 Z\* 0.2294 / α 0.990 / P 0.00328; A2 0.3471 / 0.952 / 0.00391; B2 0.2428 / 0.995 / 0.00359; A2B 0.2387 / 0.889 / 0.00361. Replay hold-out (1000 frames, MH-1 labels) E / F / σ: 1.0.2 53.8 / 193.7 / 9.4; A2 0 / 0 / 0; B2 29.8 / 97.5 / 4.8; A2B 11.7 / 41.5 / 2.1.
