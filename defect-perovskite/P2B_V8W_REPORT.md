# Plan P Phase 2 report (2026-09-30 03:54)

Runs read: {'a': 6, 'bs': 6, 'b0': 6}; estimator per run in the JSON. Base: base_v8_w01 (A2B provider, lr 3e-4, 60 epochs, response weights /10; gates 1 and 2 pass). Units: forces per component meV/Å, energies meV/atom; τ = 1.7.

## Medians over seeds

| arm | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E | C_Q transfer meV |
|---|---|---|---|---|---|---|---|---|---|
| A (Mulliken) | 19.95 | 17.11 | 37.82 | 19.04 | 13.51 | 11.42 | 7.46 | 0.85 | 23.17 |
| B_s (near Mulliken + far dipoles) | 13.89 | 12.42 | 23.97 | 13.34 | 11.17 | 9.83 | 7.89 | 0.52 | 5.59 |
| B_0 (H0 near + far dipoles) | 19.56 | 16.96 | 36.43 | 19.21 | 12.98 | 9.99 | 7.07 | 0.93 | 22.14 |

## Paired TOST (positive mean favours the second arm)

| pair | metric | n | mean d | 90 % CI | reading |
|---|---|---|---|---|---|
| a_vs_bs | force_rmse | 6 | +6.91 | [-0.71, +14.53] | inconclusive |
| a_vs_bs | 4-12 | 6 | +5.05 | [-0.41, +10.50] | inconclusive |
| a_vs_bs | 2-4 | 6 | +16.68 | [-2.30, +35.66] | inconclusive |
| a_vs_bs | 4-8 | 6 | +6.46 | [-0.51, +13.44] | inconclusive |
| a_vs_bs | 8-10 | 6 | +2.12 | [-0.38, +4.62] | inconclusive |
| a_vs_bs | 10-12 | 6 | +1.60 | [-0.21, +3.41] | inconclusive |
| a_vs_bs | 12-99 | 6 | -0.10 | [-1.05, +0.85] | equivalent |
| a_vs_b0 | force_rmse | 6 | +1.74 | [-5.88, +9.37] | inconclusive |
| a_vs_b0 | 4-12 | 6 | +0.98 | [-4.36, +6.32] | inconclusive |
| a_vs_b0 | 2-4 | 6 | +5.13 | [-14.30, +24.56] | inconclusive |
| a_vs_b0 | 4-8 | 6 | +1.12 | [-5.81, +8.05] | inconclusive |
| a_vs_b0 | 8-10 | 6 | +0.19 | [-1.99, +2.38] | inconclusive |
| a_vs_b0 | 10-12 | 6 | +1.52 | [-0.24, +3.27] | inconclusive |
| a_vs_b0 | 12-99 | 6 | +0.49 | [-0.36, +1.33] | equivalent |
| bs_vs_b0 | force_rmse | 6 | -5.17 | [-6.84, -3.50] | inferior |
| bs_vs_b0 | 4-12 | 6 | -4.07 | [-5.41, -2.72] | inferior |
| bs_vs_b0 | 2-4 | 6 | -11.55 | [-15.48, -7.63] | inferior |
| bs_vs_b0 | 4-8 | 6 | -5.34 | [-6.98, -3.70] | inferior |
| bs_vs_b0 | 8-10 | 6 | -1.93 | [-2.85, -1.01] | inferior |
| bs_vs_b0 | 10-12 | 6 | -0.09 | [-0.85, +0.67] | equivalent |
| bs_vs_b0 | 12-99 | 6 | +0.59 | [+0.15, +1.04] | equivalent (superior within tau) |

## Gates

Gate 1 (adoption over A, total and 4–12 Å): {'bs': {'total': 'inconclusive', '4-12': 'inconclusive', 'adopted': False, 'inferior_both': False, 'n_pairs': 6}, 'b0': {'total': 'inconclusive', '4-12': 'inconclusive', 'adopted': False, 'inferior_both': False, 'n_pairs': 6}}

Gate 2 per arm (C_Q 79→159 within 10 meV and 79-atom E ≤ 0.6): {'a': {'pass_seeds': [6], 'fail_seeds': [0, 1, 2, 3, 4]}, 'bs': {'pass_seeds': [0, 1, 2, 3, 4], 'fail_seeds': [6]}, 'b0': {'pass_seeds': [], 'fail_seeds': [0, 1, 2, 3, 4, 6]}}

Cost, median s/epoch: {'a': 67.0, 'bs': 127.0, 'b0': 126.5}

**Verdict: B_s adopted over A: False (inconclusive, inconclusive); B_0 adopted over A: False (inconclusive, inconclusive); B_0 within tau of B_s on both: False -> chosen arm None**
