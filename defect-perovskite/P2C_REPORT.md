# Plan P Phase 2c report: the carrier level's depth at large N

Units: energies in eV unless stated; separation = ε(N_up) − ε(N_up − 1) of the head Hamiltonian at
the reference fill (the gap between the highest occupied and lowest empty majority-spin level of the
neutral-labelled structure); N_eff = 1 / Σ dq² of the one-hole charge distribution; Σdq(< 5 Å) =
the part of that hole within 5 Å of the vacancy site. Smearing σ_s = 0.05 eV. No DFT was run.
Thresholds were registered in the tracker before each result was read.

## 0. Geometry diagnostics (no training)

Checkpoints: the decisive one, arm B_s on `base_v8_w01`, seed 2 (the seed-median checkpoint of
Phase 2b on w01), and the corroborating one, arm B_s on base v7, seed 1 (seed median of the v7
Phase 2b arms). "Label cell" = the static pristine cell (`static_pristine_cell.json`, tilt 167°);
"relaxed cell" = the base-relaxed pristine cell the ladders were tiled from (tilt 156°, 0.25 Å rms
from the label cell).

### 0.1 Band edges of the pristine tilings (dense eigenvalues of H0)

Identical at 1×1×1, 2×2×2 and 3×3×3 (80 / 640 / 2160 atoms; dimension 320 / 2560 / 8640): the
head is a Γ-point Hamiltonian of a cell whose bands are direct at Γ, so tiling folds nothing onto
the edges.

| checkpoint | cell | VBM | CBM | gap |
|---|---|---|---|---|
| w01 B_s s2 | label | −9.928 | −7.528 | 2.400 |
| w01 B_s s2 | relaxed | −10.107 | −7.468 | 2.638 |
| v7 B_s s1 | label | −10.046 | −7.646 | 2.400 |
| v7 B_s s1 | relaxed | −10.308 | −7.633 | 2.675 |

Criterion (relaxed-tiling CBM ≥ 0.1 eV below the label tiling's): **not met** on either checkpoint;
the relaxed cell's CBM is 0.06 eV (w01) and 0.01 eV (v7) *above* the label cell's. The gap at the
relaxed tilt is 0.24–0.28 eV wider than the registered 2.40 eV of the label cell (recorded, not
registered as a target, per the plan's "not permitted" list).

### 0.2 The +1 core embedded at 639 atoms

Rows: (a) the relaxed tiling with the existing core (the structure the large-N gate was read on);
(b) the label tiling with the core re-relaxed in it (far field fixed beyond R_core = 8 Å, BFGS,
fmax 0.002 eV/Å, ≤ 400 steps; the frozen-core-ladder protocol); (a′) the relaxed tiling with the core
re-relaxed by the same checkpoint, so that (a′) vs (b) isolates the tiling from the relaxing model.

| checkpoint | row | separation | N_eff | Σdq(< 5 Å) | Pb–Pb across vacancy (Å) | relax steps / ΔE |
|---|---|---|---|---|---|---|
| w01 B_s s2 | (a) relaxed tiling, existing core | 0.015 | 23.3 | 0.294 | 5.76 | — |
| w01 B_s s2 | (b) label tiling, core re-relaxed | 0.016 | 87.5 | 0.112 | 5.84 | 109 / −0.371 |
| w01 B_s s2 | (a′) relaxed tiling, core re-relaxed | 0.009 | 40.4 | 0.212 | 5.91 | 48 / −0.036 |
| v7 B_s s1 | (a) relaxed tiling, existing core | 0.022 | 21.7 | 0.221 | 5.76 | — |
| v7 B_s s1 | (b) label tiling, core re-relaxed | 0.009 | 51.2 | 0.178 | 5.84 | 114 |
| v7 B_s s1 | (a′) relaxed tiling, core re-relaxed | 0.020 | 36.3 | 0.226 | 5.89 | — |

Criterion (label-tiling separation ≥ 3× the relaxed tiling's): **not met**; the ratio is 1.1 (w01)
and 0.4 (v7). On neither checkpoint does the label tiling bind the hole more tightly; on both it
spreads it further (N_eff 51–88 against 22–40).

### Decision 0

Candidate (1), the tiling geometry, is **not confirmed** on either checkpoint. Candidate (2), the
level's depth as one scalar the data do not identify, stands alone. Step 1 (arm B_w) runs and blocks
Phase 3; the large-N gate for B_w is read in the label tiling as the plan directs, with B_s re-read
there alongside.

### 0.3 Separation and N_eff on the 79-atom +1 training frames against Pb–Pb across the vacancy

928 charged frames (10 of 938 have no flanking Pb pair by the vacancy-side rule and are skipped).
Pb–Pb across the vacancy spans 4.93–6.80 Å; the embedded cores sit at 5.76 (a), 5.84 (b), 5.91 (a′).

| checkpoint | separation p10 / p50 / p90 / max | N_eff p10 / p50 / p90 | pass (≥ 0.2 eV and N_eff ≤ 4) | corr(separation, Pb–Pb) |
|---|---|---|---|---|
| w01 B_s s2 | 0.011 / 0.026 / 0.052 / 0.122 | 6.2 / 9.2 / 13.1 | 0 / 928 | −0.17 |
| v7 B_s s1 | 0.011 / 0.032 / 0.060 / 0.188 | 4.9 / 7.2 / 11.3 | 0 / 928 | −0.20 |

By Pb–Pb bin (w01 B_s s2; v7 in parentheses), separation p50 / N_eff p50:

| Pb–Pb bin (Å) | n | separation p50 | N_eff p50 |
|---|---|---|---|
| 4.75–5.00 | 4 | 0.065 (0.087) | 7.8 (6.2) |
| 5.00–5.25 | 20 | 0.030 (0.046) | 11.0 (7.1) |
| 5.25–5.50 | 143 | 0.029 (0.038) | 9.7 (7.6) |
| 5.50–5.75 | 215 | 0.027 (0.034) | 9.1 (7.2) |
| 5.75–6.00 | 357 | 0.026 (0.029) | 8.4 (6.8) |
| 6.00–6.25 | 146 | 0.025 (0.029) | 9.5 (7.6) |
| 6.25–6.50 | 22 | 0.024 (0.028) | 9.9 (8.2) |
| 6.50–7.00 | 11 | 0.026 (0.031) | 12.5 (10.6) |

The pass fraction is 0.00 in every bin on both checkpoints.

**The same readout on the 1057 neutral-vacancy 79-atom frames** (w01 B_s s2; the frames the C5
precondition is read on; Pb–Pb 3.68–5.98 Å): separation p50 0.146 (max 0.400), N_eff p50 1.40,
pass 150 / 1057, corr(separation, Pb–Pb) +0.66. By bin the pass fraction rises from 0.00 below
4.25 Å to 0.33–0.44 above 5.0 Å. At the same Pb–Pb (5.25–5.75 Å) the neutral frames give
separation 0.18 eV with N_eff 1.5 and the charged frames 0.028 eV with N_eff 9–10.

**Reading.** The head's bound state exists at the neutral-vacancy geometries and is absent at the
+1 geometries. The +1 frames are the frames the forces are fitted on, and every embedded core is a
+1 geometry, so the large-N failure is the same fact read at larger N. The Pb–Pb distance across
the vacancy is not the coordinate that separates the two populations (their ranges overlap and give
opposite answers); the rest of the +1 relaxation is. The C5 precondition, read on neutral-vacancy
frames, has been measuring a geometry the carrier does not occupy. The plan leaves the +1-geometry
separation "reported, not gated" in Step 1; after 0.3 it is the quantity the large-N gate depends
on, and it is reported per epoch on the held-out +1 frames as the plan requires. Whether to
re-rule the C5 instrument is the user's decision; nothing in Step 1 depends on it.

### 0.4 Sign-resolved dq by shell on one 79-atom +1 frame

First +1 training frame, Pb–Pb 5.80 Å. The reference-fill hole (the C5 instrument) and the +1
forward's dq (the force model's charge) are the same distribution to 1e-12, as they must be in the
SCF-free model at Φ = 0; both are entirely positive.

| checkpoint | 0–3 Å | 3–5 Å | 5–8 Å | 8–12 Å | Σ | N_eff |
|---|---|---|---|---|---|---|
| w01 B_s s2 | +0.455 | +0.018 | +0.305 | +0.222 | 1.000 | 8.25 |
| v7 B_s s1 | +0.536 | +0.026 | +0.273 | +0.165 | 1.000 | 6.00 |

The plan's "N_eff 1.4 with Σdq(< 5 Å) 0.47" came from two different frames (N_eff 1.4 from the
first neutral-vacancy frame, Σdq 0.47 from the first +1 frame). On one frame the two instruments
agree: there is no compensating negative part; the hole is simply spread, 45–54 % on the two
flanking Pb and the rest in the 5–12 Å shells.

### Cost

Step 0 on one checkpoint: 9 minutes on one GPU (the 639-atom core relaxations 4.6 and 2 minutes;
the 8640-dimensional dense eigenproblems seconds each; 0.3 on 928 frames 65 s).

## 1. Arm B_w

### 1.1 First run (on-site half-width 3.08 eV for every species; gap term on H0)

### Step 1: arm B_w against B_s (same seeds; held-out forces, per-component meV/Å; `model_avg.pt`)

| arm | seed | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E (meV/atom) | C_Q transfer (meV) | energy gate | s (B_s) / s_near (B_w) final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_s | 0 | 13.72 | 12.24 | 23.8 | 13.2 | 10.9 | 9.9 | 8.1 | 0.54 | 7.0 | pass | 0.376 |
| B_s | 2 | 13.88 | 12.40 | 24.1 | 13.3 | 11.2 | 9.8 | 7.8 | 0.50 | 4.2 | pass | 0.411 |
| B_s | 6 | 14.37 | 12.94 | 24.8 | 13.8 | 11.9 | 9.9 | 6.5 | 0.46 | 13.4 | FAIL | 0.555 |
| B_w | 0 | 15.18 | 13.60 | 26.1 | 14.9 | 11.8 | 9.9 | 6.2 | 0.60 | 6.9 | FAIL | 0.040 |
| B_w | 2 | 16.00 | 13.76 | 29.9 | 15.2 | 11.4 | 10.5 | 8.0 | 0.84 | 13.1 | FAIL | 0.083 |
| B_w | 6 | 15.75 | 13.61 | 29.6 | 14.9 | 11.8 | 10.1 | 6.1 | 0.78 | 20.4 | FAIL | 0.064 |

Paired mean difference B_w − B_s (n = 3, point estimate): total 1.65 (per seed [1.46, 2.12, 1.38]), 4–12 Å 1.13 (per seed [1.36, 1.37, 0.67]); τ = 1.7. **B_w kept (within +tau or better on total, 4-12 A).**

### Level readouts (B_w; separation eV / N_eff / pass on the held fold's 79-atom +1 frames; separation / N_eff / Σdq(< 5 Å) on the 639-atom label-tiled structure (b))

| seed | epoch | held F | s_near | dEp95 | +1 frames sep p50 / N_eff p50 / pass | embed sep / N_eff / Σdq(<5 Å) |
|---|---|---|---|---|---|---|
| 0 | 0 | — | 0.149 | 0.155 | 0.043 / 12.3 / 0.00 | 0.027 / 134.2 / 0.03 |
| 0 | 5 | — | 0.166 | 0.157 | 0.060 / 3.9 / 0.00 | 0.021 / 6.9 / 0.56 |
| 0 | 10 | — | 0.129 | 0.139 | 0.065 / 3.2 / 0.00 | 0.023 / 5.1 / 0.66 |
| 0 | 20 | — | 0.108 | 0.144 | 0.112 / 2.3 / 0.44 | 0.028 / 2.9 / 0.87 |
| 0 | 30 | — | 0.066 | 0.295 | 0.073 / 4.2 / 0.00 | 0.036 / 4.4 / 0.83 |
| 0 | 40 | — | 0.046 | 0.424 | 0.093 / 4.3 / 0.00 | 0.076 / 4.8 / 0.81 |
| 0 | 50 | — | 0.044 | 0.567 | 0.034 / 4.6 / 0.00 | 0.031 / 5.4 / 0.78 |
| 0 | 59 | 15.17 | 0.037 | 0.652 | 0.031 / 4.5 / 0.00 | 0.033 / 5.5 / 0.78 |
| 0 | avg | 15.18 | 0.040 | — | 0.030 / 4.5 / 0.00 | 0.018 / 5.6 / 0.78 |
| 2 | 0 | — | 0.131 | 0.269 | 0.068 / 7.7 / 0.07 | 0.010 / 105.0 / 0.07 |
| 2 | 5 | — | 0.133 | 0.259 | 0.068 / 3.3 / 0.00 | 0.023 / 5.4 / 0.64 |
| 2 | 10 | — | 0.105 | 0.254 | 0.072 / 2.7 / 0.00 | 0.022 / 3.8 / 0.75 |
| 2 | 20 | — | 0.154 | 0.481 | 0.087 / 5.1 / 0.07 | 0.008 / 4.4 / 0.80 |
| 2 | 30 | — | 0.119 | 0.599 | 0.060 / 5.1 / 0.00 | 0.037 / 5.5 / 0.78 |
| 2 | 40 | — | 0.096 | 0.779 | 0.081 / 3.6 / 0.01 | 0.027 / 4.0 / 0.86 |
| 2 | 50 | — | 0.080 | 0.885 | 0.102 / 3.4 / 0.02 | 0.052 / 3.8 / 0.84 |
| 2 | 59 | 15.95 | 0.083 | 0.926 | 0.100 / 3.4 / 0.01 | 0.059 / 3.8 / 0.84 |
| 2 | avg | 16.00 | 0.083 | — | 0.096 / 3.4 / 0.01 | 0.052 / 3.8 / 0.85 |
| 6 | 0 | — | 0.392 | 0.104 | 0.044 / 11.6 / 0.00 | 0.008 / 44.7 / 0.13 |
| 6 | 5 | — | 0.198 | 0.102 | 0.060 / 3.9 / 0.00 | 0.020 / 6.6 / 0.57 |
| 6 | 10 | — | 0.147 | 0.096 | 0.065 / 3.2 / 0.00 | 0.022 / 5.3 / 0.64 |
| 6 | 20 | — | 0.115 | 0.575 | 0.176 / 2.3 / 0.45 | 0.081 / 2.9 / 0.84 |
| 6 | 30 | — | 0.106 | 0.725 | 0.187 / 2.3 / 0.48 | 0.104 / 2.8 / 0.87 |
| 6 | 40 | — | 0.076 | 1.727 | 0.161 / 2.7 / 0.38 | 0.141 / 3.3 / 0.83 |
| 6 | 50 | — | 0.057 | 1.763 | 0.134 / 2.5 / 0.21 | 0.119 / 3.0 / 0.85 |
| 6 | 59 | 15.29 | 0.070 | 1.710 | 0.141 / 2.6 / 0.25 | 0.144 / 2.9 / 0.87 |
| 6 | avg | 15.75 | 0.064 | — | 0.135 / 2.6 / 0.21 | 0.145 / 2.9 / 0.86 |

Cost, median s/epoch: B_w {0: 138.01593232154846, 2: 139.72544836997986, 6: 134.01075637340546}, B_s {}.

**Large-N on the seed-median checkpoint (seed 6), label tiling, core re-relaxed by that checkpoint:** separation 0.114 eV, N_eff 3.1,
Σdq(< 5 Å) 0.84 — fails the separation threshold (0.15), meets the other two. Relaxed tiling: 0.146 / 2.9 / 0.87. B_s in the label tiling: 0.016 / 87.5 / 0.11.

**How this arm binds the hole (investigation of 2026-10-11, `p2c_spike_diag.py`).** On every +1 frame the on-site readout at the two flanking Pb sits at its rails
(p channel −2.97 eV against a bound of 3.0, s channel +2.58 eV; |tanh| 0.97); B_s has −0.04 / −0.10 eV there. s_near decayed to 0.04–0.08, so the compaction is the
readout's, not W's. The loss spikes at epochs 17–22 mark the readout running to its bound; the force deficit is the basin it entered (the Pb on-site term as a whole
became load-bearing: removing it costs 23.5 meV/Å against 3.6 for B_s). The project's Stage-2 rule (a bound binding on the sites: not an optimum) applies; the arm is
recorded, not adopted. Energy gate fails on every seed.

### 1.2 Rerun (half-width 0.1 × the element's own s–p splitting: Pb 0.72, Cs 0.16, Cl 1.29 eV; gap term on H0 − W)

Same seeds, folds, schedule and s_init as the first run; the two changes are the on-site half-width rule (`delta_rule="sp_splitting"`, frac 0.1) and the gap
regulariser reading the Hamiltonian the model fills (H0 − W). Runs `dscc_p2cwbw2_s{0,2,6}` on b3 GPUs 4/5/7, 2026-10-11 13:14–22:00. B_s rows are the Phase 2b w01 models.

#### Forces and energies: B_w against B_s (same seeds; held-out forces, per-component meV/Å; `model_avg.pt`)

| arm | seed | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E (meV/atom) | C_Q transfer (meV) | energy gate | s (B_s) / s_near (B_w) final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_s | 0 | 13.72 | 12.24 | 23.8 | 13.2 | 10.9 | 9.9 | 8.1 | 0.54 | 7.0 | pass | 0.376 |
| B_s | 2 | 13.88 | 12.40 | 24.1 | 13.3 | 11.2 | 9.8 | 7.8 | 0.50 | 4.2 | pass | 0.411 |
| B_s | 6 | 14.37 | 12.94 | 24.8 | 13.8 | 11.9 | 9.9 | 6.5 | 0.46 | 13.4 | FAIL | 0.555 |
| B_w | 0 | 13.54 | 12.23 | 22.7 | 12.8 | 11.4 | 10.8 | 8.1 | 0.61 | 6.5 | FAIL | 0.263 |
| B_w | 2 | 13.12 | 11.81 | 22.3 | 12.5 | 10.8 | 10.1 | 7.7 | 0.49 | 6.9 | pass | 0.292 |
| B_w | 6 | 16.90 | 15.11 | 29.6 | 16.2 | 13.7 | 12.3 | 8.7 | 0.72 | 3.4 | FAIL | 0.417 |

Paired mean difference B_w − B_s (n = 3, point estimate): total 0.53 (per seed [-0.18, -0.76, 2.52]), 4–12 Å 0.52 (per seed [-0.01, -0.59, 2.17]); τ = 1.7. **B_w kept (within +tau or better on total, 4-12 A).**

#### Level readouts (B_w; separation eV / N_eff / pass on the held fold's 79-atom +1 frames; separation / N_eff / Σdq(< 5 Å) on the 639-atom label-tiled structure (b))

| seed | epoch | held F | s_near | dEp95 | +1 frames sep p50 / N_eff p50 / pass | embed sep / N_eff / Σdq(<5 Å) |
|---|---|---|---|---|---|---|
| 0 | 0 | — | 0.362 | 0.028 | 0.033 / 13.2 / 0.00 | 0.008 / 97.7 / 0.05 |
| 0 | 5 | — | 0.353 | 0.135 | 0.060 / 5.8 / 0.00 | 0.024 / 10.6 / 0.44 |
| 0 | 10 | — | 0.343 | 0.123 | 0.061 / 5.4 / 0.00 | 0.028 / 9.1 / 0.49 |
| 0 | 20 | — | 0.319 | 0.178 | 0.055 / 5.1 / 0.00 | 0.023 / 7.5 / 0.55 |
| 0 | 30 | — | 0.293 | 0.171 | 0.060 / 4.9 / 0.00 | 0.023 / 7.0 / 0.57 |
| 0 | 40 | — | 0.272 | 0.223 | 0.056 / 4.9 / 0.00 | 0.027 / 6.7 / 0.58 |
| 0 | 50 | — | 0.266 | 0.229 | 0.051 / 4.7 / 0.00 | 0.023 / 6.3 / 0.60 |
| 0 | 59 | 13.58 | 0.261 | 0.232 | 0.051 / 4.7 / 0.00 | 0.022 / 5.8 / 0.63 |
| 0 | avg | 13.54 | 0.263 | — | 0.052 / 4.7 / 0.00 | 0.022 / 6.1 / 0.61 |
| 2 | 0 | — | 0.391 | 0.026 | 0.035 / 13.0 / 0.00 | 0.013 / 98.1 / 0.05 |
| 2 | 5 | — | 0.383 | 0.231 | 0.083 / 5.5 / 0.05 | 0.020 / 13.0 / 0.38 |
| 2 | 10 | — | 0.375 | 0.138 | 0.068 / 5.5 / 0.00 | 0.027 / 9.9 / 0.46 |
| 2 | 20 | — | 0.353 | 0.191 | 0.064 / 5.3 / 0.00 | 0.031 / 8.3 / 0.52 |
| 2 | 30 | — | 0.330 | 0.169 | 0.059 / 5.5 / 0.00 | 0.029 / 8.2 / 0.52 |
| 2 | 40 | — | 0.310 | 0.178 | 0.053 / 5.5 / 0.00 | 0.029 / 7.6 / 0.55 |
| 2 | 50 | — | 0.295 | 0.178 | 0.051 / 5.6 / 0.00 | 0.021 / 7.5 / 0.55 |
| 2 | 59 | 13.14 | 0.288 | 0.183 | 0.050 / 5.6 / 0.00 | 0.019 / 7.3 / 0.56 |
| 2 | avg | 13.12 | 0.292 | — | 0.050 / 5.6 / 0.00 | 0.020 / 7.4 / 0.55 |
| 6 | 0 | — | 0.533 | 0.035 | 0.037 / 11.9 / 0.00 | 0.004 / 69.7 / 0.06 |
| 6 | 5 | — | 0.522 | 0.123 | 0.069 / 5.9 / 0.02 | 0.036 / 9.5 / 0.46 |
| 6 | 10 | — | 0.511 | 0.161 | 0.074 / 5.4 / 0.01 | 0.042 / 8.3 / 0.51 |
| 6 | 20 | — | 0.481 | 0.204 | 0.064 / 5.7 / 0.00 | 0.039 / 8.4 / 0.51 |
| 6 | 30 | — | 0.428 | 0.166 | 0.058 / 6.7 / 0.00 | 0.019 / 12.4 / 0.38 |
| 6 | 40 | — | 0.423 | 0.181 | 0.048 / 6.4 / 0.00 | 0.022 / 9.9 / 0.47 |
| 6 | 50 | — | 0.418 | 0.179 | 0.047 / 7.2 / 0.00 | 0.023 / 10.1 / 0.46 |
| 6 | 59 | 16.85 | 0.415 | 0.187 | 0.049 / 6.6 / 0.00 | 0.023 / 9.7 / 0.48 |
| 6 | avg | 16.90 | 0.417 | — | 0.048 / 6.8 / 0.00 | 0.024 / 9.7 / 0.47 |

Cost, median s/epoch: B_w {0: 158.40092384815216, 2: 159.88571393489838, 6: 152.64598274230957}, B_s {}.

**Force gate: kept.** Paired mean B_w − B_s is +0.53 meV/Å total and +0.52 on 4–12 Å (τ = 1.7). Seeds 0 and 2 are better than B_s on every shell except 10–12 Å;
seed 6 is worse by 2.5 (its loss spiked at epoch 30, see below, and its held force error went from 16.2 to 19.9 in that epoch and did not fully recover). The 2–4 Å shell,
where the first run lost most (26–30 vs 24), is now 22.3–22.7 against B_s's 23.8–24.8.

**Energy gate (C_Q transfer ≤ 10 meV and 79-atom energy RMSE after C_Q ≤ 0.6 meV/atom):** seed 2 passes; seed 0 fails by 0.01 meV/atom (0.61); seed 6 fails (0.72).
B_s itself fails on seed 6 (C_Q transfer 13.4). The transfer term is better than B_s on every seed (3.4–6.9 vs 4.2–13.4).

**Large-N on the seed-median checkpoint (seed 0; `p2c_step0.py --stages 0.2 0.3`, local GPU, 22:16–22:31):**

| structure (639 atoms) | separation (eV) | N_eff | Σdq(< 3 / 5 / 8 / 12 Å) | Pb–Pb across the vacancy (Å) |
|---|---|---|---|---|
| (b) label tiling, core re-relaxed by this checkpoint | 0.020 | 6.5 | 0.54 / 0.59 / 0.76 / 0.91 | 5.84 |
| (a) relaxed tiling, existing core | 0.025 | 5.9 | 0.57 / 0.63 / 0.78 / 0.92 | 5.76 |
| (a′) relaxed tiling, core re-relaxed | 0.027 | 6.6 | — / 0.59 / 0.75 / 0.90 | 5.90 |

Fails all three thresholds (0.15 eV, N_eff ≤ 4, Σdq(< 5 Å) ≥ 0.8). The per-epoch readout on the same label-tiled structure without re-relaxation gave 0.022 / 6.1 / 0.61,
so the relaxation changes little. On the 79-atom +1 training frames (0.3): separation p50 0.052 eV, N_eff p50 4.7 (seed 0), 5.6 (seed 2), 6.6–6.8 (seed 6); no
correlation with the Pb–Pb distance across the vacancy (r = −0.03 / +0.02 over 928 frames). The hole this arm produces is the same size at 79 and 639 atoms
(4.7 → 6.1, 5.6 → 7.4, 6.8 → 9.7 from the 79-atom frames to the 639-atom cell); B_s's goes 9 → 88.

**Saturation (Stage-2 rule): none.** On the +1 frames the Pb on-site shift p95 is 0.20–0.30 eV against the 0.72 eV bound; at the two flanking Pb the mean |tanh| is
0.45–0.59 (max 0.85–0.95 on single frames). The first run sat at |tanh| 0.97 on every frame. s_near ends at 0.263 / 0.292 / 0.417 (from 0.376 / 0.411 / 0.555); the
first run's collapsed to 0.04–0.08. The compaction here is W's, not the readout's.

**Spikes.** Seed 6's loss at epoch 30 was 190× its median; 99 % of that step's loss was the Γ-gap term (force part 1.7e-5, ordinary). Seed 0 had a 7× gap-term spike at
epoch 36. The same signature (gap term, on-site shift p95 jumping in the same epoch) is in every arm trained on the w01 base: B_0 seeds 1/4/6, B_s seed 6, the first B_w
seed 2. The replay of seed 6 with the per-step record (`dscc_p2cx_bw2_s6_replay`) is under way; its reading goes in §1.4.

**Reading against the plan's outcomes.** The force and transfer numbers favour the rerun over B_s on two seeds of three. The large-N gate as written fails: the
separation is 0.02 eV against 0.15, and N_eff is 6.5 against 4. §1.3 gives the evidence that bears on whether those thresholds describe this defect: the rerun's N_eff
converges with cell size (unlike B_s), the 159-atom training frames have longer and more spread Pb–Pb bridges than the 79-atom frames, and the rerun's N_eff follows
the bridge length in those frames (r ≈ +0.9), so a looser hole at 159–639 atoms is consistent with the geometry rather than with a failure to bind. The thresholds
were written before the data; changing them is the user's decision, and it is to be made from §1.3, not from this arm's numbers.

### 1.3 What the hole's size depends on: cell size and the Pb–Pb bridge (readouts of 2026-10-11)

These readouts bear directly on how the large-N thresholds should be read. All are from the same instrument (the Hamiltonian the model fills);
N_eff = 1/Σdq², the number of atoms the hole is spread over.

**The same 0 K core in the perfect label lattice at 79 and 639 atoms, and the 17 charged 159-atom training frames (thermal DFT geometries).**
The 159-atom data cell (16.0 × 22.5 × 15.7 Å) is not a tiling of the 80-atom label cell, so there is no 0 K point at that size.

| model | N_eff: 79 atoms, 0 K | 79 atoms, thermal frames (median) | 159 atoms, training frames (median of 17) | 639 atoms, 0 K |
|---|---|---|---|---|
| B_s seed 2 | 9.2 | 8.3 | 10.3 (p10–p90 7.3–14.9) | 87.5 |
| first B_w seed 2 | 3.5 | 3.5 | 3.8 | 3.8 |
| first B_w seed 6 | 2.8 | 2.6 | 2.8 | 2.9 |
| rerun B_w seed 0 | 4.7 | 4.2 | 6.0 (5.2–8.5) | 6.1 |
| rerun B_w seed 2 | 5.4 | 5.1 | 6.9 (5.8–9.9) | 7.4 |

Thermal disorder is not what localises the hole in the training cells: the 0 K 79-atom value equals the thermal median for every model. B_s's hole grows
with the cell (a band state capped by the cell: two thirds of the Pb present at both sizes). The rerun's N_eff rises from 79 to 159 atoms and then stops
(6.0 → 6.1; 6.9 → 7.4): a bound state of six to seven atoms. The first arm's N_eff is 2.5–2.9 everywhere.

**The hole's charge by shell** (median; the 0–3 Å shell is empty in the 159-atom frames because their flanking Pb sit just beyond 3 Å):

| model, structure | 0–3 Å | 3–5 Å | 5–8 Å | 8–12 Å | > 12 Å | N_eff |
|---|---|---|---|---|---|---|
| rerun B_w s2, 79 atoms 0 K | 0.58 | 0.07 | 0.22 | 0.13 | 0.00 | 5.4 |
| rerun B_w s2, 159-atom frames | 0.00 | 0.45 | 0.17 | 0.25 | 0.01 | 6.9 |
| rerun B_w s2, 639 atoms 0 K | 0.50 | 0.05 | 0.18 | 0.17 | 0.10 | 7.4 |
| first B_w s6, 79 atoms 0 K | 0.85 | 0.05 | 0.06 | 0.04 | 0.00 | 2.8 |
| first B_w s6, 639 atoms 0 K | 0.82 | 0.04 | 0.08 | 0.05 | 0.01 | 2.9 |
| B_s s2, 79 atoms 0 K | 0.42 | 0.02 | 0.24 | 0.32 | 0.00 | 9.2 |
| B_s s2, 639 atoms 0 K | 0.11 | 0.00 | 0.06 | 0.23 | 0.60 | 87.5 |

Atoms available per shell (0–3 / 3–5 / 5–8 / 8–12 / > 12 Å): 79-atom cell 2 / 12 / 37 / 27 / 1; 159-atom cell 0 / 14 / 44 / 81 / 21; 639-atom cell 2 / 12 / 44 / 139 / 442.
The rerun's hole keeps about a quarter of its charge beyond 8 Å; the 79-atom cell has almost no lattice there, so it truncates the tail and reads the hole
smaller than it is.

**The Pb–Pb bridge across the vacancy.** 79-atom +1 frames: median 5.83 Å (10–90 %: 5.45–6.13). 159-atom +1 frames: median 6.11 Å (5.21–6.98), a longer and
much more varied bridge. The embedded 0 K cores sit at 5.76–5.91 Å. Within the 17 charged 159-atom frames the rerun's N_eff follows the bridge length almost
exactly: r = +0.94 (seed 2), +0.91 (seed 0); in the 79-atom frames the correlation is +0.1 because the range is narrow. At matched bridge length (5.4–6.2 Å)
the 159-atom frames still read N_eff 5.4–6.2 against 4.0–4.8 at 79 atoms, which is the truncated tail. The first arm shows no dependence on the bridge length
at all (N_eff 2.5–2.9 and 85–89 % of the hole on the two flanking Pb in every cell and frame), as expected of a readout at its rail: it has no gradient left
and cannot respond to the geometry.

**Where the data can judge, they prefer the rerun's six-atom hole:** the 2–4 Å forces, the most direct probe of the hole's local structure, are 22.3–22.7 meV/Å
for the rerun against 29.6–29.9 for the first arm and 23.8–24.1 for B_s, and the energy's size consistency from 79 to 159 atoms (C_Q transfer) is 6.5–6.9 meV
against 13–20 for the first arm. Whether the physical hole is three atoms or six cannot be read from these labels without DFT levels on a larger cell. The
registered thresholds (N_eff ≤ 4, Σdq(< 5 Å) ≥ 0.8) were written before any of this was known; the first arm meets the compactness part of them only through the
saturated readout, and the rerun's bound state sits at about twice their extent. The decision on the thresholds is the user's.

### 1.4 Why seed 6 spiked at epoch 30, and the fix (spike investigation II, 2026-10-12)

**Method.** Seed 6 of the rerun was replayed with the same seed, fold and command, recording every optimiser step (loss by term, the static-cell gap and the
three levels either side of it, the pre-clip gradient norm by parameter group, the frames in the batch and their force errors) and saving the model at the end of
epochs 28–31 (`dscc_p2cx_bw2_s6_replay`; reader `p2c_spike_replay.py`; record `~/runs/dscc/p2c/replay_s6/`). CUDA scatter-adds are not bit-deterministic, so the
replay drifts from the original after epoch 26, but the spike reproduced at epoch 30 (loss 3.9e-3 against 1.6e-3 originally; 99 % the Γ-gap term in both).

**What the record shows.**

1. *The gap term oscillates with period 2 throughout training.* From epoch 3 to epoch 29 the sign of (gap − 2.40) changes on 80–92 % of consecutive steps, with a
   lag-1 autocorrelation of −0.9 to −0.98 and a step-to-step amplitude ratio of 1.0 ± 0.1. The amplitude is 1–2 × 10⁻⁴ eV, far below anything the epoch logs show.
   A ratio of 1.0 held for 27 epochs is the signature of a mode sitting at the optimiser's stability limit (loop gain ≈ 2). My reading of why it stays there, offered
   as interpretation: when the oscillation grows, its own gradient enters Adam's second moment and shrinks the step; when it decays, the second moment relaxes and the
   step grows back. The term is therefore permanently one perturbation away from diverging.
2. *The trigger was a level crossing at the static cell's conduction-band edge.* In this model the CBM is a single level 0.003 eV below an exactly degenerate pair.
   Over steps 61–65 of epoch 30 the split closed 0.0025 → 0.0015 → 0.0009 → 0.0001 → 0 and the pair became the band edge. The gap's gradient then belongs to a
   different eigenvector, and in Adam's per-parameter frame that direction had a smaller second moment, so the same gradient produced a larger step: the amplitude
   ratio went from 1.1 to 3.4, and the gap error ran −0.006, +0.019, −0.037, +0.089, −0.107, +0.52 eV in six steps (pre-clip gradient norm 0.08 → 24; clipping at 10
   first engaged at the last of these).
3. *The hopping readout carried it.* It held 92–99 % of the gradient norm at every step of the event and its parameters moved by 93 % of their norm within the epoch
   (2.6 % in a normal epoch); the on-site readout moved 60 % (2.4 % normally). The species baselines eps0 moved 0.02 eV and s_near 0.454 → 0.390. The whole
   static-cell spectrum shifted rigidly by −0.45 eV with the gap back at 2.400 and the degeneracy gone.
4. *The damage is what the rerun's seed 6 numbers show.* During the 46 excursion steps the per-graph force error was 48 meV/Å against 16 before; 30 for the rest of
   the epoch and 32 in the next. On the +1 frames the Cl s on-site shift moved coherently by +0.2 eV and N_eff went 6.0 → 11.3 → 7.5: the hole partly delocalised and
   did not fully return (the readout table's jump from 5.7 to 6.7 at epoch 30 is this event). After the event the period-2 statistic drops to 0.2: the optimiser had
   stabilised the mode itself by inflating the hopping readout's second moment, at the cost of the force fit.
5. The same gap-term signature with an on-site jump in the same epoch is in the B_0 and B_s spikes and in the first B_w arm's run-away (seed 2, epoch 20). In those
   the unbounded or wide on-site readout was the amplifier; with the sp-splitting bound in place the hopping readout took the role.

**What was tried, in order, and what each showed.**

1. *A tolerance band on the gap term* (third run, bw3; no gradient within ±k_BT of the target). Inside the band it behaved as designed, but when the force term
   carried the gap to the band's edge the first restoring step was a 0.13 eV kick for a 0.02 eV excursion, then a ±0.1 eV oscillation across the band and a doubled
   force error for 25 steps (seeds 6 and 2 at epochs 5 and 8). Withdrawn after nine epochs; step records kept.
2. *Smooth band edges* (fourth run, bw4; each edge the Boltzmann-weighted mean of the levels nearest it at σ = k_B · 300 K = 0.0259 eV, plain square, no band).
   This removes the crossing trigger: the gap and its gradient are continuous when edge levels cross or are degenerate. It is kept (flag `--gap_sigma`, helper
   `smooth_gap`, test `test_smooth_gap_continuous_across_a_crossing`; σ = 0 is the hard gap exactly). It was not sufficient: seed 0 diverged at epoch 19 with no
   crossing and no band (CBM and VBM splits unchanged through the event), the oscillation growing from its resting ±0.0003 eV at a ratio of 3 per step to 0.38 eV
   in eleven steps, 32 excursion steps, force error 16 → 34 meV/Å, epoch loss 300× the median. Seed 2 had a mild excursion in the same epoch; seed 6 none.
   A registered consequence: the regulariser holds the smoothed gap at 2.400 and the hard gap rests at 2.37–2.38, because both edges of the static cell have
   several levels within σ. Which gap carries the 2.40 label is the user's decision.

**The mechanism, as the three records together show it.** The Γ-gap term is a stiff scalar constraint (its sensitivity to the ~71k hopping-readout parameters is
8–17 eV per unit parameter) applied through Adam, whose step for each parameter is normalised by that parameter's second-moment estimate. The second moment of
the hopping readout is set by whichever gradient is present. While the gap oscillation is active it feeds the second moment and the loop gain self-tunes to the
stability limit (the period-2 state at amplitude 1e-4 eV seen from epoch 3 in every run with the plain square). When the oscillation is quiet, the second moment
is the force gradient's alone, and the gain is then proportional to the gap weight and inversely to the force gradient's scale: in bw4 the force gradient on the
hopping readout fell 3× over epochs 15–18 of seeds 0 and 2, and seed 0 diverged in epoch 19. A level crossing (the replay) and a band edge (bw3) are two further
ways of presenting the gap gradient to a second moment that does not reflect it. One mechanism, three routes. The force fit takes the damage every time.

**The two levers, and what is running.** (a) The gap weight: at w = 1 the quiet gain ran between 1.5 and 4; at w = 0.1 it is 0.15–0.4, below the limit with margin
against the measured dip, and the self-tuned oscillation cannot start; the cost is that the gap lags the target by about 1e-3 eV at rest. A fifth run (bw5 = bw4
plus `--gap_weight 0.1`) is queued behind bw4 as a registered diagnostic; its discriminating test is the alternation fraction by epoch 5 (≤ 0.6 against bw4's
0.8–0.9). (b) The gap term's gradient applied as a plain step outside Adam, with its own fixed step size, so its gain is deterministic and independent of the
force gradient's scale; prepared, not run. Neither is a new loss term. The choice is the user's, with bw5's reading in front of them.

**Pass criteria written before each run** are in the tracker (entries of 00:25, 01:50 and 03:55); bw4's are: no epoch above 10× the median after epoch 2; after epoch
2 no step with a gap-term loss above 1e-3 and a maximum per-step |gap − 2.40| below 0.01 eV; the hard gap at every epoch end within 2.40 ± 0.01 (fails by
construction under σ, see above); forces, energy gate, large-N and saturation as for the rerun.

### 1.5 Fourth run (the rerun's settings plus the smooth band edges)

(to follow: three-way per-seed table B_s / rerun / fourth run, the stability table per seed, the gap's resting values, energy gate, large-N on the seed median,
saturation, N_eff)

