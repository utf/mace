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

(to follow)
