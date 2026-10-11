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

(to follow)
