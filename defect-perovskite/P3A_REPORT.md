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

### 2.1 Born-force table (switched kernel): the direct measurement of the +1 far field

Checkpoint bw5 seed 0 (fixed by §1). Structure: the +1 core relaxed at 639 atoms with the full model (fmax 0.002; 101 BFGS steps, 9.5 min GPU; ΔE_relax −0.453 eV, max displacement 0.40 Å), embedded within R_core = 8 Å in the 2879-atom (3×3×4) and 639-atom (2×2×2) tilings of the base-v8-relaxed cell. E_far is the model's switched far-field kernel (the periodic smoothed field minus the (1 − S) near-field part over the head's edges, r_c 8 Å, δ 2 Å); the reconstruction equals the head's own e_host to the printed digit (-29.08 meV at 2879 atoms, -23.71 at 639). F_born = −∂/∂R[−Σ p·E_far^detached] through the base's dipoles; the reference Z*·E_c uses the base's pristine Born tensors with (i) a unit Gaussian charge at the vacancy site and (ii) the actual dq, both switched the same way. Wall clock 175 s (2879, CPU) and 47 s (639, CPU; does not fit the 16 GB card).

**2879 atoms** (Σdq within 3 / 5 / 8 / 12 Å: 0.35 / 0.68 / 0.76 / **0.861**; N_eff 4.78; forces per-component meV/Å)

| shell (Å) | n | rms F_born | rms Z*·E_c (actual dq) | ratio | cos | rms Z*·E_c (unit) | ratio | cos | ratio Cs / Pb / Cl (actual) |
|---|---|---|---|---|---|---|---|---|---|
| 4–6 | 14 | 22.7 | 25.9 | **0.876** | 0.993 | 1.8 | 12.361 | -0.983 | 3.39 / 0.91 / 0.79 |
| 6–8 | 34 | 26.3 | 29.3 | **0.900** | 0.990 | 37.8 | 0.698 | 0.928 | 0.98 / 0.84 / 1.06 |
| 8–10 | 50 | 36.2 | 45.1 | **0.802** | 0.997 | 73.6 | 0.492 | 0.995 | 0.75 / 0.81 / 0.80 |
| 10–12 | 84 | 34.0 | 39.9 | **0.853** | 0.996 | 54.1 | 0.629 | 0.991 | 0.81 / 0.86 / 0.86 |
| 12–14 | 120 | 30.4 | 33.8 | **0.899** | 0.997 | 42.4 | 0.716 | 0.994 | 0.89 / 0.92 / 0.88 |
| 14–17 | 242 | 22.7 | 24.2 | **0.939** | 0.996 | 28.7 | 0.792 | 0.993 | 0.93 / 0.94 / 0.94 |
| 17–20 | 354 | 15.2 | 15.7 | **0.974** | 0.994 | 17.5 | 0.870 | 0.991 | 0.96 / 0.98 / 0.97 |
| 20–24 | 666 | 9.6 | 9.7 | **0.984** | 0.989 | 10.5 | 0.913 | 0.981 | 0.99 / 0.98 / 0.99 |

**639 atoms** (Σdq within 3 / 5 / 8 / 12 Å: 0.34 / 0.66 / 0.79 / **0.911**; N_eff 5.08)

| shell (Å) | n | rms F_born | rms Z*·E_c (actual dq) | ratio | cos | rms Z*·E_c (unit) | ratio | cos | ratio Cs / Pb / Cl (actual) |
|---|---|---|---|---|---|---|---|---|---|
| 4–6 | 14 | 19.3 | 23.7 | **0.814** | 0.635 | 9.9 | 1.953 | -0.744 | 2.05 / 0.86 / 0.74 |
| 6–8 | 34 | 21.2 | 24.6 | **0.864** | 0.974 | 33.4 | 0.636 | 0.766 | 0.95 / 0.79 / 1.06 |
| 8–10 | 50 | 29.3 | 37.3 | **0.787** | 0.980 | 64.7 | 0.454 | 0.985 | 0.72 / 0.78 / 0.80 |
| 10–12 | 83 | 26.5 | 32.4 | **0.819** | 0.986 | 45.1 | 0.588 | 0.972 | 0.78 / 0.82 / 0.82 |
| 12–14 | 102 | 23.7 | 27.3 | **0.867** | 0.989 | 35.3 | 0.671 | 0.983 | 0.86 / 0.88 / 0.85 |
| 14–17 | 179 | 16.3 | 17.5 | **0.930** | 0.943 | 21.5 | 0.759 | 0.952 | 0.91 / 0.93 / 0.94 |
| 17–20 | 123 | 10.7 | 10.8 | **0.993** | 0.946 | 12.8 | 0.838 | 0.946 | 0.94 / 1.01 / 0.99 |
| 20–24 | 45 | 7.9 | 7.7 | **1.026** | 0.964 | 9.3 | 0.847 | 0.915 | 0.99 / 1.03 / 1.02 |

Reading: **§2.1 passes** (actual-dq ratio ≥ 0.85 in every shell beyond 12 Å at both sizes: 0.90–0.98 at 2879, 0.87–1.03 at 639), with cosines 0.99 at 2879 atoms and the three species within 0.03 of each other. The unit-charge reference is lower (0.72–0.91 beyond 12 Å) because the actual hole is not a point charge: the far ions see the charge that is inside their radius. Below 8 Å the switch removes the far field by construction and the unit-charge column has no meaning. Against P0 (base v2 + MACEField, actual dq: 0.88 / 0.92 / 0.96 / 0.97 beyond 12 Å) the coupling is unchanged to slightly better.

**The hole's shell profile** (`dq_profile_bw5_2879.png`, 2879 atoms): the two flanking Pb carry 0.35 and 0.29; 0.09 sits on the Cl and Pb neighbours within 8 Å; the remainder lies on paired Pb along the vacancy's own Pb–Cl–Pb axis at 8.5 / 14.2 / 19.9 / 25.6 Å (shell sums 0.070 / 0.033 / 0.016 / 0.008, a factor 0.45 per 5.7 Å, decay length ≈ 7 Å) with nothing on the atoms between; no atom carries negative dq. Fill-smearing scan at the same structure:

| σ_s (eV) | N_eff | Σdq < 8 Å | Σdq < 12 Å | pair at 8–9 Å | 14–15 Å | 19–20 Å |
|---|---|---|---|---|---|---|
| 0.0125 | 4.10 | 0.777 | 0.888 | 0.089 | 0.042 | 0.020 |
| 0.025 | 4.25 | 0.774 | 0.882 | 0.085 | 0.040 | 0.019 |
| 0.05 (the model) | 4.78 | 0.759 | 0.861 | 0.070 | 0.033 | 0.016 |
| 0.1 | 6.42 | 0.677 | 0.767 | 0.053 | 0.026 | 0.016 |

The smearing accounts for about 0.03 of diffuse spread; the chain tail grows slightly as the fill sharpens, so it belongs to the defect level itself (0.028 eV below the next state at this geometry) hybridised along the σ chain: the tight-binding physics of a shallow vertical level, not a fill artefact and, being on the vacancy's axis with the two sides equal within 10 %, not an equivariance fault. Even at the sharpest fill the charge inside 12 Å is 0.888.

**§0 compactness gate (639 and 2159 atoms):** Σdq(< 12 Å) = 0.911 at 639 (pass) and **0.869 at 2159 (fail, < 0.9)**; N_eff(2159)/N_eff(639) = 0.96 (pass, ≤ 1.2); Born-force ratio beyond 12 Å ≥ 0.85 (pass). **The gate fails on the compactness term.** The 639-atom value is inflated by the cell: its 12 Å sphere wraps onto the images (the cell is 22.8 Å along z), so the chain tail is counted as inside.

### 2.4 Frozen-core ladder (vertical; energies only)

The §2.1 core embedded within 8 Å in the 639 / 959 / 2159 / 2879 tilings; E(+1) − E(0) at fixed geometry, E(0) = base + E_PP. The model's charged energy already carries the exact ε∞ point-charge monopole E_M = Q²ξ(h)/(2ε∞) (its `e_madelung` diagnostic), so the remainder after the monopole is dE itself. Wall clock: 5 + 6 s (GPU) and 65 + 110 s (CPU).

| atoms | L = Ω^⅓ (Å) | dE = E(+1) − E(0) (eV) | E_M (eV) | dE − E_M (eV) | N_eff | Σdq < 3 / 5 / 8 / 12 Å |
|---|---|---|---|---|---|---|
| 639 | 28.64 | 8.1579 | -0.1701 | 8.3281 | 5.08 | 0.34 / 0.66 / 0.79 / 0.911 |
| 959 | 32.79 | 8.1674 | -0.1549 | 8.3224 | 4.99 | 0.34 / 0.67 / 0.75 / 0.863 |
| 2159 | 42.96 | 8.1936 | -0.1138 | 8.3074 | 4.90 | 0.35 / 0.68 / 0.76 / 0.869 |
| 2879 | 47.29 | 8.1971 | -0.1077 | 8.3048 | 4.78 | 0.35 / 0.68 / 0.76 / 0.861 |

Fits: E∞ + c/L³ on 959–2879: E∞(frozen) = **8.2129 eV**, c = -1597 eV Å³, residuals -0.1, +0.8, -0.7 meV; the held-out 639 point misses it by +13.0 meV. Over all four sizes: E∞ + c/L³ gives E∞ 8.2078 with residuals +2.9, -5.2, +1.4, +1.0 meV; adding a 1/L term gives b = -4.29 eV Å (a span of 59 meV over the range) with residuals +0.5, -1.3, +2.3, -1.6 meV.

Reading: **§2.4 fails the flatness criterion.** dE rises by 29.6 meV from 959 to 2879 atoms against the 5 meV allowed. The model's energy without E_M falls by 23 meV from 639 to 2879 while the point-charge term it adds back rises by 62 meV: the charged cell's own image interaction is weaker than that of a point charge in ε∞ = 4, i.e. the size dependence is not the fixed-charge monopole law. The charge is not fixed: Σdq(< 12 Å) goes 0.911 → 0.861 across the sizes as the chain tail (§2.1) finds more room, and the tail's self-interaction and images change with L. Whether the residual is read as a 1/L term (23–59 meV, depending on the fit) or as a large 1/L³ term (c ≈ −1600 eV Å³, which fits 959–2879 to < 1 meV but misses 639 by 13 meV) is not decidable from four points; either way it is above 5 meV and comes from the same cause as the compactness failure.

### 2.5 Relaxation ladder and the linear-response check
*pending (2879-atom relaxation running; linear-response check queued)*

### 2.6 Decision
*pending §2.5*

## Wall clock
| item | time | device |
|---|---|---|
| pristine relaxation | 16 s | GPU |
| §2.2a Hessian + Z* | 66 s | GPU |
| §2.1 core relaxation (639, GPU) / Born force 639 (CPU) / 2879 (CPU) | 9.5 min / 47 s / 175 s | GPU / CPU |
| §2.4 frozen ladder (4 sizes) | 3 min | GPU 639/959, CPU 2159/2879 |
| §2.2b 3×3×3 finite-displacement set (480 calls at 2160 atoms) + analysis | 2.9 h + 3 min | CPU, 48 threads (shared) |
| §2.3 LO–TO (12 energies + 4 dipole calls at 2160 atoms) | 5 min | CPU, 40 threads |
| §2.2c form factor (18 dipole calls at 2160 atoms) | 4 min | CPU, 40 threads |
