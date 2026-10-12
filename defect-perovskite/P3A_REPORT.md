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


**(b) ε(q) from the 3×3×3 finite-displacement set** (480 base-alone calls at 2160 atoms, each returning forces and node dipoles; h = 0.01 Å; wall clock 2.9 h, CPU 48 threads, shared). D(q) and the q-dependent Born charges Z*_κ(q) come from the same set (conventions in `p3a_epsq.py`); ε_ion,L(q) = q̂·ε_ion(q)·q̂ from the bare D(q); ε_L = 4.0 + ε_ion,L.

Checks: D(q = 0) reproduces the §2.2a Γ Hessian to 0.4 µeV in every mode; Σ_b′ Λ at q = 0 reproduces the autograd Z* to 1e-4 e; force constants beyond 12 Å are below 0.021 eV/Å² (largest element 3.85), so the 3×3×3 box (half-lengths 24.1 / 24.1 / 17.1 Å) holds the base's reach without wrap-around; the Fourier-route form factor at the commensurate points equals the frozen-pattern measurement of §2.2c (x n = 1: 0.956 vs 0.956; y n = 1: 0.952 vs 0.952; y n = 2: 0.833 vs 0.833; z n = 1: 0.862 vs 0.862); no soft modes at any q on the paths.

| q | q (Å⁻¹) | ε_ion,L / ε_L, q ∥ x | q ∥ y | q ∥ z | f(q), strongest mode, x / y / z |
|---|---|---|---|---|---|
| 0 (Γ) | 0.000 | 19.1 / **23.1** | 18.0 / **22.0** | 20.5 / **24.5** | 1.000 / 1.000 / 1.000 |
| 2π/47.3 (2879 atoms) | 0.133 | 7.5 / **11.5** | 7.7 / **11.7** | 7.3 / **11.3** | 0.954 / 0.950 / 0.923 |
| 2π/43.0 (2159) | 0.146 | 6.7 / **10.7** | 6.9 / **10.9** | 6.4 / **10.4** | 0.945 / 0.940 / 0.909 |
| 2π/32.8 (959) | 0.192 | 4.6 / **8.6** | 5.0 / **9.0** | 4.3 / **8.3** | 0.910 / 0.902 / 0.852 |
| 2π/28.6 (639) | 0.220 | 3.7 / **7.7** | 4.2 / **8.2** | 3.4 / **7.4** | 0.886 / 0.875 / 0.814 |
| 0.26 | 0.260 | 2.9 / **6.9** | 3.4 / **7.4** | 2.6 / **6.6** | 0.849 / 0.833 / 0.758 |

Comparison with base v2 (P0, q ∥ y, frozen phonons of the top 8 y-polarised modes): ε_ion 19.4 / 11.4 / 4.8 at 0 / 0.13 / 0.26 Å⁻¹; base v8 w01 along y: 18.0 / 7.9 / 3.4. **Base v8 w01's polar dispersion is as strong as base v2's, slightly stronger**: the ionic screening falls to 42 % of its q = 0 value by 0.13 Å⁻¹ (base v2: 59 %) and to 19 % by 0.26 Å⁻¹ (base v2: 25 %). The polycrystalline trace falls more slowly (19.2 → 14.3 → 11.0 along x) because the transverse components survive; the longitudinal one is what a point charge sees.

**Registered expectation for §2.5 (the §0 reading rule):** ε₀_ladder should lie between ε_L at the largest and the smallest cells' wavevectors, i.e. between **7.4 and 11.7** (axis range 7.5–8.2 at 2π/28.6 Å, 11.3–11.7 at 2π/47.3 Å), and above 2ε∞ = 8.

**(d) Dispersion along Γ–X/Y/Z** (`p3a_dispersion.png`): 240 branches of the 80-atom cell, bare and with the nonanalytic term at ε∞ 4.0 built from Z*(q). Optical range 2–28 meV, no imaginary modes along any path; the NA term lifts a handful of branches to 22–27 meV at q → 0 (the LO modes), the rest is unchanged. A published PBE phonon dispersion of Pnma CsPbCl3 with digitised branches was not found in the time available (the closest is the combined experimental/DFT study of CsPbX3 single crystals in Mater. Today Commun. 2025, abstract only); recorded as a gap. Qualitatively the base's 2–28 meV range and the 20–27 meV LO group agree with the published spectra of the Cs–Pb–Cl family (Raman and IR bands up to ~27 meV).

**(c) Form factor of the base's dipoles** (P0 A1 procedure with the base as its own provider: the strongest IR mode, 10.13 meV, Z̃ = (0, -1.889, 0) e/√amu, y-polarised; modulated cos(q·R) in the 3×3×3 cell, amplitude 0.05; wall clock 4 min, CPU). The longitudinal line is q ∥ y:

| n | λ (Å) | q (Å⁻¹) | f, base v8 w01 | f, base v2 (P0) |
|---|---|---|---|---|
| 1 | 48.2 | 0.130 | 0.952 | 0.939 |
| 2 | 24.1 | 0.261 | 0.833 | 0.787 |
| 3 | 16.1 | 0.391 | 0.694 | 0.607 |

Gaussian width from −4 ln f = w²q² along y: **3.13 Å** (base v2: 3.47 Å); below the registered r_split 4.5 Å, as for base v2. The transverse lines (q ∥ x with the y-polarised pattern: f 0.956 / 0.848 / 0.018; q ∥ z: 0.862 / 0.626 / 2.85 at λ = 34.2 / 17.1 / 11.4 Å, the last being the unit-cell period and not a plane wave) are recorded in `p3a_formfactor.json` and not used.


### 2.3 LO–TO in the 3×3×3 cell (base v8 w01 + E_PP from its own dipoles)

P0's reference computation (`p0_loto.py`) with the base as its own provider: the strongest IR mode (10.13 meV, Z̃ = (0, -1.889, 0) e/√amu) modulated cos(q·R) at q = 2π/L along y (LO, longitudinal) and along z (TO, transverse), amplitude 0.05; curvatures k = (E(+A) + E(−A) − 2E(0))/A² for the base alone, the full neutral model (base + E_PP through the bw5 checkpoint at zero carriers; r_split 4.5, ε∞ 4.0), and E_PP alone. Expectation: the smoothed nonanalytic term 2·(4πC/Ωε∞)|S_q|² · g(q) · f², g = exp(−w²q²/4), w = 2 r_split (as P0), f the measured form factor at q. Wall clock 5 min (CPU).

| branch | k_base | k_full | k_E_PP | NAC raw | NAC × g | NAC × g f² | g | f |
|---|---|---|---|---|---|---|---|---|
| LO (q ∥ y, 0.130 Å⁻¹) | 0.5650 | 1.0343 | 0.4693 | 0.7419 | 0.5261 | 0.4768 | 0.709 | 0.952 |
| TO (q ∥ z, 0.184 Å⁻¹) | 0.3345 | 0.3263 | -0.0081 | 0 | 0 | 0 | 0.505 | 0.862 |

(k in eV per unit pattern amplitude squared, A in Å·√amu units as P0.)

| LO − TO | base v8 w01 + E_PP | base v2 + E_PP (P0, r_split 9.645) |
|---|---|---|
| full model | 0.7080 | 0.3858 |
| base alone | 0.2306 | 0.2500 |
| E_PP alone | **0.4774** | 0.1358 |
| NAC × g | 0.5261 | 0.1538 |
| NAC × g f² | **0.4768** | — |

E_PP's LO − TO against the smoothed expectation with the form factor: ratio **1.001**; against the smoothed expectation without it: 0.907. Both within 10 %: **§2.3 passes**. E_PP on this provider delivers the nonanalytic splitting the switched kernel is meant to deliver at this q; the base alone already carries 31 % of the raw NAC splitting at q = 0.13 Å⁻¹ within its 12 Å reach (base v2: 34 %). On the TO branch E_PP changes the curvature by -2.4 %.

### 2.1, 2.4, 2.5, 2.6
*start when the checkpoint is fixed (§1)*

## Wall clock
| item | time | device |
|---|---|---|
| pristine relaxation | 16 s | GPU |
| §2.2a Hessian + Z* | 66 s | GPU |
| §2.2b 3×3×3 finite-displacement set (480 calls at 2160 atoms) + analysis | 2.9 h + 3 min | CPU, 48 threads (shared) |
| §2.3 LO–TO (12 energies + 4 dipole calls at 2160 atoms) | 5 min | CPU, 40 threads |
| §2.2c form factor (18 dipole calls at 2160 atoms) | 4 min | CPU, 40 threads |
