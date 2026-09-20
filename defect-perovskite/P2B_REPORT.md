# Plan P Phase 2 report (2026-09-20 00:40)

Runs read: {'a': 6, 'bs': 6, 'b0': 6}; estimator per run in the JSON. Base: base_v7_lr3e4 (user override of the Phase 1 gates). Units: forces per component meV/Å, energies meV/atom; τ = 1.7.

## Medians over seeds

| arm | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E | C_Q transfer meV |
|---|---|---|---|---|---|---|---|---|---|
| A (Mulliken) | 15.42 | 13.79 | 27.06 | 14.70 | 12.92 | 11.60 | 7.60 | 0.61 | 9.37 |
| B_s (near Mulliken + far dipoles) | 15.96 | 14.42 | 26.75 | 15.38 | 13.10 | 11.60 | 8.82 | 0.45 | 4.46 |
| B_0 (H0 near + far dipoles) | 21.25 | 18.71 | 38.36 | 20.86 | 15.71 | 11.57 | 8.03 | 0.74 | 14.44 |

## Paired TOST (positive mean favours the second arm)

| pair | metric | n | mean d | 90 % CI | reading |
|---|---|---|---|---|---|
| a_vs_bs | force_rmse | 6 | -0.16 | [-1.43, +1.11] | equivalent |
| a_vs_bs | 4-12 | 6 | -0.27 | [-1.29, +0.75] | equivalent |
| a_vs_bs | 2-4 | 6 | +0.40 | [-2.54, +3.33] | inconclusive |
| a_vs_bs | 4-8 | 6 | -0.39 | [-1.60, +0.82] | equivalent |
| a_vs_bs | 8-10 | 6 | -0.02 | [-0.77, +0.74] | equivalent |
| a_vs_bs | 10-12 | 6 | -0.15 | [-0.79, +0.49] | equivalent |
| a_vs_bs | 12-99 | 6 | -1.12 | [-1.50, -0.74] | equivalent (inferior within tau) |
| a_vs_b0 | force_rmse | 6 | -7.19 | [-11.77, -2.61] | inferior |
| a_vs_b0 | 4-12 | 6 | -6.04 | [-9.72, -2.37] | inferior |
| a_vs_b0 | 2-4 | 6 | -14.50 | [-24.33, -4.68] | inferior |
| a_vs_b0 | 4-8 | 6 | -7.52 | [-11.80, -3.24] | inferior |
| a_vs_b0 | 8-10 | 6 | -3.87 | [-6.77, -0.97] | inferior |
| a_vs_b0 | 10-12 | 6 | -0.72 | [-2.60, +1.16] | inconclusive |
| a_vs_b0 | 12-99 | 6 | -0.49 | [-1.54, +0.55] | equivalent |
| bs_vs_b0 | force_rmse | 6 | -7.03 | [-11.29, -2.77] | inferior |
| bs_vs_b0 | 4-12 | 6 | -5.77 | [-9.14, -2.41] | inferior |
| bs_vs_b0 | 2-4 | 6 | -14.90 | [-24.39, -5.41] | inferior |
| bs_vs_b0 | 4-8 | 6 | -7.13 | [-11.14, -3.11] | inferior |
| bs_vs_b0 | 8-10 | 6 | -3.85 | [-6.27, -1.44] | inferior |
| bs_vs_b0 | 10-12 | 6 | -0.57 | [-2.33, +1.19] | inconclusive |
| bs_vs_b0 | 12-99 | 6 | +0.63 | [-0.27, +1.52] | equivalent |

## Gates

Gate 1 (adoption over A, total and 4–12 Å): {'bs': {'total': 'equivalent', '4-12': 'equivalent', 'adopted': True, 'inferior_both': False, 'n_pairs': 6}, 'b0': {'total': 'inferior', '4-12': 'inferior', 'adopted': False, 'inferior_both': True, 'n_pairs': 6}}

Gate 2 per arm (C_Q 79→159 within 10 meV and 79-atom E ≤ 0.6): {'a': {'pass_seeds': [4], 'fail_seeds': [0, 1, 2, 3, 6]}, 'bs': {'pass_seeds': [0, 1, 2, 4, 6], 'fail_seeds': [3]}, 'b0': {'pass_seeds': [6], 'fail_seeds': [0, 1, 2, 3, 4]}}

Cost, median s/epoch: {'a': 41.0, 'bs': 82.5, 'b0': 84.0}

**Verdict: B_s adopted over A: True (equivalent, equivalent); B_0 adopted over A: False (inferior, inferior); B_0 within tau of B_s on both: False -> chosen arm B_s**

## §2 Kernel validation (no training)

Embedded 3×3×4 structure (2879 atoms), switched far-field force against Z*·E_c by shell (ratio, cosine):

| shell Å | unit charge: ratio | cos | head dq (σ_s × 0.25): ratio | cos |
|---|---|---|---|---|
| 0–4 | 0.14 | +0.96 | 0.29 | +0.99 |
| 4–6 | 0.23 | +0.96 | 0.38 | +0.99 |
| 6–8 | 0.43 | +0.99 | 0.50 | +0.99 |
| 8–10 | 0.61 | +1.00 | 0.62 | +1.00 |
| 10–12 | 0.76 | +0.99 | 0.73 | +0.99 |
| 12–14 | 0.81 | +1.00 | 0.81 | +1.00 |
| 14–17 | 0.91 | +1.00 | 0.88 | +1.00 |
| 17–20 | 0.98 | +1.00 | 0.95 | +1.00 |
| 20–24 | 0.99 | +0.99 | 0.98 | +0.99 |

Head dq at the sharpened fill in the 2879-atom cell: Σdq 1.000, within 5 Å 0.854 (A s0 checkpoint). Beyond 10 Å the switched far field is Z*·E_c to 0.73–0.98 with cosine ≥ 0.99 (the P0 ratios); inside 6 Å the residual 0.14–0.38 is the nonlocal Born response of atoms in the ramp region, not the kernel.

Training frames (A s0's dq and q0 at s = 1; per-component RMS force, meV/Å):

| family | n | F(E_far) | F(E_near) | F(sum) | E_far meV | E_near meV |
|---|---|---|---|---|---|---|
| charged_159 | 17 | 9.00 | 20.66 | 23.63 | -12.3 | -538.2 |
| charged_79 | 1030 | 3.74 | 37.54 | 37.15 | +11.5 | -787.3 |
| neutral_159 | 17 | 0.00 | 0.00 | 0.00 | +0.0 | +0.0 |
| neutral_79 | 1174 | 0.00 | 0.00 | 0.00 | +0.0 | +0.0 |
| pristine_80 | 639 | 0.00 | 0.00 | 0.00 | +0.0 | +0.0 |

## E_near / E_far decomposition on the held-out charged frames (B_s, trained weights)

| seed | learned s | F(E_far) meV/Å | F(E_near) | F(sum) | E_far meV | E_near meV |
|---|---|---|---|---|---|---|
| 0 | 0.323 | 2.62 | 7.37 | 7.45 | +7.5 | -175.4 |
| 1 | 0.388 | 2.45 | 7.38 | 7.51 | +6.4 | -188.2 |
| 2 | 0.318 | 2.85 | 7.55 | 7.68 | +7.5 | -153.8 |
| 3 | 0.198 | 3.71 | 6.96 | 7.45 | +10.0 | -190.6 |
| 4 | 0.351 | 2.74 | 7.68 | 7.88 | +6.3 | -183.4 |
| 6 | 0.631 | 1.56 | 9.22 | 9.22 | +4.6 | -345.1 |

The far field contributes 1.6–3.7 meV/Å on the 79/159-atom held-out cells (periodicity-suppressed, as expected); the near-field Mulliken term 7–9 meV/Å at the learned s of 0.20–0.63.

## Localisation

79-atom C5 precondition (1191 neutral-vacancy frames, epoch-averaged checkpoints):

| arm | seed | pass fraction | separation p50 meV | N_eff p50 |
|---|---|---|---|---|
| A | 0 | 0.98 | 439 | 1.54 |
| A | 1 | 0.03 | 340 | 14.47 |
| A | 2 | 0.98 | 520 | 1.53 |
| A | 3 | 0.98 | 664 | 1.96 |
| A | 4 | 0.96 | 411 | 1.57 |
| A | 6 | 0.99 | 474 | 1.32 |
| B_0 | 0 | 0.06 | 124 | 3.30 |
| B_0 | 1 | 0.07 | 119 | 1.52 |
| B_0 | 2 | 0.00 | 33 | 12.06 |
| B_0 | 3 | 0.14 | 111 | 3.34 |
| B_0 | 4 | 0.10 | 122 | 5.90 |
| B_0 | 6 | 0.00 | 32 | 35.48 |
| B_s | 0 | 0.86 | 364 | 1.87 |
| B_s | 1 | 0.78 | 279 | 1.57 |
| B_s | 2 | 0.92 | 403 | 1.56 |
| B_s | 3 | 0.51 | 204 | 2.16 |
| B_s | 4 | 0.94 | 357 | 1.78 |
| B_s | 6 | 0.93 | 335 | 1.46 |

Large-N readout on the embedded structures, B_s seed-median checkpoint (seed 1, averaged weights), registered fill (gate for Phase 3 entry: separation ≥ 0.15 eV, N_eff ≤ 4, Σdq within 5 Å ≥ 0.8):

| structure | separation eV | N_eff | Σdq within 3/5/8/12/16 Å | gate |
|---|---|---|---|---|
| train 79 (+1) (N = 79, σ_s 0.05) | see below | 5.8 | 0.54 / 0.57 / 0.84 / 1.00 / 1.00 | |
| embedded 2,2,2 (N = 639, σ_s 0.05) | see below | 20.5 | 0.30 / 0.32 / 0.42 / 0.60 / 0.82 | |
| embedded 3,3,3 (N = 2159, σ_s 0.05) | see below | 38.2 | 0.22 / 0.23 / 0.30 / 0.39 / 0.52 | |
| embedded 3,3,3 sigma x1.0 (N = 2159, σ_s 0.05) | see below | 38.2 | 0.22 / 0.23 / 0.30 / 0.39 / 0.52 | |
| embedded 3,3,3 sigma x0.25 (N = 2159, σ_s 0.0125) | see below | 41.6 | 0.21 / 0.22 / 0.28 / 0.37 / 0.49 | |

Bound state (p0_boundstate.py): training 79-atom frame separation 0.308 eV, N_eff 1.6 (pass); embedded 2×2×2 (639 atoms) 0.023 eV, N_eff 20.5 (FAIL); embedded 3×3×3 (2159 atoms) 0.021 eV, N_eff 38.2 (FAIL). Σdq within 5 Å: 0.565 (79 atoms), 0.316 (639), 0.234 (2159); the sharpened fill (σ_s × 0.25) does not recover it (0.222). **The Phase 3 entry gate fails**: the bound state B_s holds at 79 atoms collapses with cell size, as W6's did in Phase 0. Registered remedy applied once: B_s retrained with w_gap × 3 (`dscc_p2bsg_s*`, six seeds, running); this report is updated with its result.

## BZ readouts (per checkpoint, averaged weights; min_k [E_CBM(k) − E_CBM(Γ)], max_k [E_VBM(k) − E_VBM(Γ)], eV)

- A: s3: +0.012 / -0.024; s4: +0.008 / -0.025; s6: +0.005 / -0.011
- B_s: s0: +0.062 / -0.011; s1: +0.070 / -0.007; s2: +0.068 / -0.014; s3: +0.042 / -0.015; s4: +0.067 / -0.011; s6: +0.079 / -0.012
- B0: s0: +0.023 / -0.024; s1: +0.033 / -0.023; s2: +0.012 / -0.017; s3: +0.022 / -0.014; s4: +0.006 / -0.097; s6: +0.127 / -0.031

All gaps are direct at Γ (CBM minimum at Γ within 0.13 eV, VBM maximum at Γ); the regulariser was taken out of the loss for cost (1.1 s/step) and kept as a readout.

## Decisions (plan §5)

- **Forces.** B_s vs A: equivalent (total), equivalent (4–12 Å) → **B_s adopted over A**. B_0 vs A: inferior / inferior → not adopted. B_0 vs B_s: inferior on both → the Mulliken near-field term stays; **chosen arm B_s**. No r_c fallback is triggered (B_s is not inferior).
- **Energies.** Gate 2 (C_Q 79→159 within 10 meV, 79-atom E ≤ 0.6): B_s passes on seeds [0, 1, 2, 4, 6] (fails [3]); A on [4]; B_0 on [6].
- **Localisation.** 79-atom C5: B_s median pass 0.89 / 346 meV / N_eff 1.67 (A 0.98 / 457 / 1.56; B_0 0.07 / 115 / 4.6). Large-N: FAIL → w_gap × 3 retrain (pending).
- **B_x(M).** Three seeds of B_s with ε∞ = 3.548 follow the large-N decision.

## Cost

Median s/epoch: A 41 (local A4000), B_s 82.5, B_0 84 (b3). Arm B_s carries the base's nested autograd every step; nothing cached.
