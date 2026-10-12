# Plan P Phase 3a report: bw6 and the far-field test on base v8 w01

Plan: `PLAN_P_PHASE3A.md` (the user's, 2026-10-12). Model of record bw5 (seed 0 `model_avg.pt`, 11.79 meV/Å). Run directory `~/runs/dscc/p3a/`.
Units: forces per-component meV/Å, energies eV unless stated; τ = 1.7 meV/Å. Registrations: tracker entries of 2026-10-12 (§0 of the plan, and the four
pre-launch registrations: §1's Σdq criterion read on the label tiling; bw6 paired against bw5; the local card is a 16 GB A4000; flank |tanh| per epoch).

## 1. bw6: bw5 at fraction 0.2

Runs `dscc_p2cwbw6_s{0,2,6}` on b3 GPUs 4/5/7, started 12:36–12:44 on 2026-10-12; identical to bw5 except `--delta_frac 0.2` (bounds Pb 1.43, Cs 0.32, Cl 2.58 eV).
Per-epoch flank p-channel |tanh| p50 on the held-out +1 frames added to the level readout (`flank_tanh_p`, readout only; on bw5 seed 0's final checkpoint it reads 0.95, the value of the saturation diagnostic).

*(per-seed table, replacement decision and the flank |tanh| curves: pending the runs, ~3 h, and the finisher)*

## 2. The far-field test

### 2.0 The relaxed pristine cell

Static 80-atom cell relaxed with base v8 w01 + E_PP (the bw5 checkpoint at zero carriers), fixed cell, BFGS to fmax 0.002 eV/Å. Wall clock 16 s (GPU).

| quantity | static | relaxed (base v8 + E_PP) | relaxed (base v2, P0) |
|---|---|---|---|
| E (eV) | -281.3212 | -282.0694 | — |
| fmax (eV/Å) | 0.0572 | 0.0014 (base alone 0.0013) | 0.0019 |
| rms / max displacement from static (Å) | — | 0.261 / 0.322 | 0.253 / 0.312 |
| Pb–Cl bond (Å) | 2.860 | 2.907 | 2.905 |
| Pb–Cl–Pb angle, mean (min–max) (deg) | 167.1 (165.7–169.8) | 155.7 (154.9–157.2) | 156.1 (155.3–157.6) |
| tilt, 180 − mean angle (deg) | 12.9 | 24.3 | 23.9 |
| rms F_PP at the geometry (meV/Å) | — | 0.09 | — |

The tilt pattern is the same as the base-v2 relaxation's (same bond length and angle range within 0.01 Å and 0.5 deg). E_PP is negligible in the pristine cell
(the base-alone residual at the relaxed geometry is 0.0013 eV/Å, below the fmax target), so "base alone" and "base + E_PP" share this minimum.

### 2.2 Base v8 w01's lattice response

**(a) Γ Hessian and ε₀_LST** (h = 0.01 Å, ASR; Z* and ε∞ by autograd on the base, which equal the finite difference of its summed node dipoles to 1e-4 e; wall clock 66 s, GPU).

| quantity | base v8 w01 (this cell) | base v2 (P0, its own relaxed cell) |
|---|---|---|
| imaginary optical modes | 0 | 0 |
| optical range (meV) | 2.0–28.1 | 1.9–27.9 |
| Z* trace/3: Cl / Cs / Pb (e) | -1.544 / 1.231 / 3.401 | -1.618 / 1.277 / 3.577 |
| ε∞ (own), polycrystalline (diag) | 3.658 (3.641, 3.662, 3.670) | 3.547 (MACEField) |
| ε_ion polycrystalline (xx, yy, zz) | 19.20 (19.07, 18.02, 20.52) | 20.63 |
| ε₀_LST at ε∞ = 4.0 | **23.20** | 24.63 |
| ε₀_LST at the base's own ε∞ | 22.86 | 24.18 |
| ε₀_LST at 3.667 (P1 gate-2 value) | 22.87 | — |

Top modes by ε contribution (meV: ε): 10.1: 2.99, 10.5: 2.31, 6.8: 1.72, 9.4: 1.69, 8.7: 1.53, 7.1: 1.41. The base's own ε∞ on this cell is 3.658
(the P1 value 3.667 was read on the base-v2 cell). The registered ε∞ of E_PP stays 4.0.

*(b) ε(q), (c) form factor, (d) dispersion: pending the 3×3×3 finite-displacement set (480 base calls, CPU) and the analysis.*

### 2.3 LO–TO in the 3×3×3 cell
*pending*

### 2.1, 2.4, 2.5, 2.6
*start when the checkpoint is fixed (§1)*

## Wall clock
| item | time | device |
|---|---|---|
| pristine relaxation | 16 s | GPU |
| §2.2a Hessian + Z* | 66 s | GPU |
