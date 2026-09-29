# Plan P Phase 2 report (2026-09-29 01:25)

Runs read: {'a': 6, 'bs': 6, 'b0': 6}; estimator per run in the JSON. Base: base v8 (A2B provider, lr 1e-4, 30 epochs; gate 1 under override). Units: forces per component meV/Å, energies meV/atom; τ = 1.7.

## Medians over seeds

| arm | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E | C_Q transfer meV |
|---|---|---|---|---|---|---|---|---|---|
| A (Mulliken) | 15.80 | 14.45 | 25.99 | 15.04 | 13.73 | 12.65 | 9.11 | 0.87 | 8.63 |
| B_s (near Mulliken + far dipoles) | 15.56 | 13.95 | 26.96 | 14.70 | 13.15 | 11.50 | 8.43 | 0.44 | 5.05 |
| B_0 (H0 near + far dipoles) | 15.64 | 14.25 | 25.93 | 15.29 | 13.20 | 10.81 | 8.68 | 0.86 | 7.78 |

## Paired TOST (positive mean favours the second arm)

| pair | metric | n | mean d | 90 % CI | reading |
|---|---|---|---|---|---|
| a_vs_bs | force_rmse | 6 | +1.45 | [-1.31, +4.20] | inconclusive |
| a_vs_bs | 4-12 | 6 | +1.59 | [-0.93, +4.12] | inconclusive |
| a_vs_bs | 2-4 | 6 | +0.96 | [-3.62, +5.54] | inconclusive |
| a_vs_bs | 4-8 | 6 | +1.60 | [-1.24, +4.45] | inconclusive |
| a_vs_bs | 8-10 | 6 | +1.45 | [-0.61, +3.51] | inconclusive |
| a_vs_bs | 10-12 | 6 | +2.03 | [+0.35, +3.71] | superior |
| a_vs_bs | 12-99 | 6 | +1.52 | [+0.23, +2.81] | superior |
| a_vs_b0 | force_rmse | 6 | +0.77 | [-3.43, +4.98] | inconclusive |
| a_vs_b0 | 4-12 | 6 | +0.76 | [-3.14, +4.66] | inconclusive |
| a_vs_b0 | 2-4 | 6 | +0.99 | [-5.70, +7.67] | inconclusive |
| a_vs_b0 | 4-8 | 6 | +0.44 | [-3.73, +4.61] | inconclusive |
| a_vs_b0 | 8-10 | 6 | +1.05 | [-2.57, +4.68] | inconclusive |
| a_vs_b0 | 10-12 | 6 | +2.30 | [-0.63, +5.24] | inconclusive |
| a_vs_b0 | 12-99 | 6 | +1.26 | [-1.23, +3.75] | inconclusive |
| bs_vs_b0 | force_rmse | 6 | -0.67 | [-2.63, +1.28] | inconclusive |
| bs_vs_b0 | 4-12 | 6 | -0.84 | [-2.69, +1.02] | inconclusive |
| bs_vs_b0 | 2-4 | 6 | +0.03 | [-2.86, +2.92] | inconclusive |
| bs_vs_b0 | 4-8 | 6 | -1.17 | [-2.97, +0.64] | inconclusive |
| bs_vs_b0 | 8-10 | 6 | -0.40 | [-2.44, +1.64] | inconclusive |
| bs_vs_b0 | 10-12 | 6 | +0.28 | [-1.42, +1.97] | inconclusive |
| bs_vs_b0 | 12-99 | 6 | -0.27 | [-1.88, +1.35] | inconclusive |

## Gates

Gate 1 (adoption over A, total and 4–12 Å): {'bs': {'total': 'inconclusive', '4-12': 'inconclusive', 'adopted': False, 'inferior_both': False, 'n_pairs': 6}, 'b0': {'total': 'inconclusive', '4-12': 'inconclusive', 'adopted': False, 'inferior_both': False, 'n_pairs': 6}}

Gate 2 per arm (C_Q 79→159 within 10 meV and 79-atom E ≤ 0.6): {'a': {'pass_seeds': [], 'fail_seeds': [0, 1, 2, 3, 4, 6]}, 'bs': {'pass_seeds': [0, 1, 2, 3, 4, 6], 'fail_seeds': []}, 'b0': {'pass_seeds': [], 'fail_seeds': [0, 1, 2, 3, 4, 6]}}

Cost, median s/epoch: {'a': 67.0, 'bs': 128.5, 'b0': 126.0}

**Verdict: B_s adopted over A: False (inconclusive, inconclusive); B_0 adopted over A: False (inconclusive, inconclusive); B_0 within tau of B_s on both: False -> chosen arm None**
