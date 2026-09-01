# Step 2 result — carrier-field response channel, H3

Run: `step2_hublig.json`, 20 cells, complete 2026-09-01 15:26.
Arch `r2_h3_anneal_s5`, frozen base `e0_base_s1`, 48 charged frames, 40 epochs/cell.
Graph 10 A (spectral_r_cut = r_max x num_interactions). hub2 and lig2 both 100% coupled,
distance-matched at 5.57 A.

## Gate (handover §4) — BOTH CLAUSES FAIL

| cell | axial_red mean | sd | median | rmse_nbhd med | lig_shell med | cs_shell med | slope_vs_d med |
|---|---|---|---|---|---|---|---|
| hub2 full | +0.306 | 0.324 | +0.392 | 84.5 | 72.0 | 35.9 | +0.036 |
| hub2 nbhd | +0.332 | 0.309 | +0.394 | 65.5 | 55.5 | 34.3 | +0.000 |
| lig2 full | -0.272 | 1.368 | +0.679 | 46.1 | 43.2 | 30.4 | +0.088 |
| lig2 nbhd | +0.236 | 0.824 | +0.678 | 45.8 | 43.9 | 38.4 | +0.082 |

R1 reference (no channel): hub2/full +0.059 (max +0.264); free +0.807, rmse_nbhd 34.0.

- **Clause 1 FAILS on the threshold, but the mechanism moved.** hub2/full mean +0.306,
  median +0.392, against a gate of 0.4. R1 was +0.059 with a best seed of +0.264 — so the
  step-2 *mean* now exceeds R1's best single seed, and rmse_nbhd fell 147 -> 84.5 toward
  free's 34.0. The channel does give compact states the footprint. It did not clear 0.4.
- **Clause 2 FAILS decisively.** hub2 - lig2 = +0.577 (full) / +0.096 (nbhd) against seed
  sd of 1.368 / 0.824.

## Fine structure (read per §4 when clause 2 fails) — points AWAY from the hub

On medians, robust to lig2's diverged seeds, **lig2 beats hub2 on every metric**:
axial_red +0.679 vs +0.392, rmse_nbhd 46.1 vs 84.5, lig_shell 43.2 vs 72.0,
cs_shell 30.4 vs 35.9. Converged seeds only (axial_red > -0.5): lig2 +0.693 (n=3) vs
hub2 +0.392 (n=5).

This is §4's **"hub2 no better = genuine indifference"** category. The point estimates lean
further, toward the ligand actually fitting better, but that rests on a median over 3
converged lig2 seeds against 5 hub2 seeds -- report it as a lean, not a verdict. The
operative conclusion is: **no hub preference.**

The one asymmetry favouring hub2 is optimisation stability: hub2 converged 5/5, lig2 3/5
(full) and 4/5 (nbhd). At n=5 that is not significant (Fisher p ~ 0.44) and it is a
stability difference, not a fit-quality preference. Not a site preference.

`slope_vs_d` does separate the two (hub2 ~ +0.00..+0.04, lig2 ~ +0.08..+0.09), but which
slope is correct is unknown without the DFT value, so it does not adjudicate.

## Status against the decision tree

Not a listed branch. The tree's clause-1-fail branch is worded "hub2 stays ~ 0", which did
not happen. Nearest reading: mechanism works, site-non-specifically, with the ligand
favoured. Per §4 the prescribed next measurement is the **SCF diagnostic**
(E_resp(hub2) vs E_resp(cage10) at fixed alpha) — still not implemented.
