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
