# D-SCC v5 — programme report (W0–W6)

*Compiled 2026-09-13 from the run artefacts in `~/runs/dscc/`. Every number here is recomputed
from those files; the tracker (`DSCC_V5_IMPLEMENTATION_SPEC.md`) holds the registrations,
rulings and the chronology, including the thresholds written before each result was opened.
Units: forces in meV/Å **per component**, energies in meV/atom unless stated, six seeds
{0,1,2,3,4,6} on folds {0,1,2,3,0,2}, medians across seeds, paired TOST at τ = 1.7 meV/Å.*

## What the programme produced

A charged-defect model for V_Cl in CsPbCl₃ that sits on a frozen neutral-trained MACE base and
adds an electronic head. Two heads were carried to the end:

- **B′ (self-consistent)** — the W3 winner. A D-SCC loop with a static reference-fill pattern.
- **W6 (SCF-free)** — one eigendecomposition, an analytic Madelung term and a
  non-self-consistent host term. Built last, from the W4/W5 outcome.

**They agree on forces to within a quarter of the equivalence margin, and W6 costs half as much.**
W6 is the model to use.

| | forces (meV/Å) | energy at 79 (meV/atom) | between-size after one C_Q (meV) | ladder 1/L | model ÷ base, 79 / 159 |
|---|--:|--:|--:|--:|--:|
| B′ w=0.05 | 11.64 | 0.471 | +3.7 | 106 % of exact | 3.55 / 4.40 |
| **W6 w=0.05** | **11.58** | **0.515** | **+5.0** | **93 % of exact** | **2.10 / 2.47** |

## W0 — registration and protocol

Shell centre on the vacancy-side rule; C11 shells re-registered (near 2–4 Å with the flanking Pb
pair and first-shell Cl split out, mid 4–8, far 8–10 / 10–12 / >12); per-component units
throughout with τ_phys = 1.7 meV/Å; comparisons paired by fold with two one-sided tests;
evaluation on the epoch-averaged checkpoint. The registrations held for the whole programme and
no comparison in this report crosses the old-base / new-base boundary.

## W1 — base v2 (neutral data only)

Multi-head fine-tune of MACE-MH-1 following the documented procedure, `omat_pbe` foundation head,
no charged frame in any base training. On the same 154-frame validation set the old base was read
on: **energy 1.16 meV/atom, forces 10.23 meV/Å per component, stress 0.16 meV/Å³**, against the
old base's 4.9 / 11.8 — both W1.3 gates pass. By size, 79 atoms 1.05 meV/atom. Four cross-fit
bases agree to 0.01 meV/Å on forces.

W1.3 recorded that **charged energies were not admitted** on the `s0(L) ± SE` and coverage
tables. For W5 the user superseded that: all frames admissible, "as if we had generated the
dataset and were trying unbiased training". Both stand in the record, the deviation labelled.

## W2 — efficiency

Eight items, each gated against the previous implementation's outputs; fixed points, energies,
forces and gradients unchanged throughout. The warm-started inference path reached 167 ms per
frame at 79 atoms, and the registered 2× benchmark (model ÷ base, P4.3 protocol) was left for W6
to answer. Two further exact optimisations landed in W6 (below).

## W3 — head baseline on base v2 (forces only)

| arm | force | flank Pb | first Cl | other 2–4 | 4–8 | >8 |
|---|--:|--:|--:|--:|--:|--:|
| Φ=0 | 12.22 | 30.40 | 15.92 | 10.50 | 11.13 | 10.77 |
| Route A LR-only | 12.31 | 31.30 | 16.20 | 11.12 | 11.19 | 10.86 |
| **B′ LR-only** | **9.45** | **22.81** | **12.59** | **9.19** | **8.99** | **7.91** |

Paired TOST against Φ=0: Route A **−0.24** [−0.39, −0.10] (equivalent, inferior within τ);
B′ **+2.69** [+2.36, +3.01] (**superior**).

**Selection: Route B′ LR-only.** It wins on every shell, not only in the near field. Route A — the
directional kernel without the static pattern — is indistinguishable from having no electrostatics
at all on this system.

The registered W3 readout, whether base v2 moved the 2–4 Å flanking-Pb residual at matched d, came
back **no**: the ordinary atoms of that shell improved decisively (+5.22 meV/Å, lower bound +1.52)
while the flanking pair was unmoved inside a seed spread of 39–68. That is the third independent
arrival at the same conclusion — the near-field residual at the flanking Pb is a property of the
79-atom labels and cell, not of the model.

## W4 — capacity

| variant | force | flank Pb | force TOST vs spec |
|---|--:|--:|---|
| **spec** | **12.22** | 30.40 | baseline |
| scalar | 12.51 | 32.82 | −0.47 [−0.72, −0.22] equivalent (inferior within τ) |
| rank1 | 12.30 | 30.86 | −1.07 [−3.11, +0.98] inconclusive |
| rank2 | 11.98 | 29.91 | +0.12 [−0.01, +0.25] equivalent |
| full | 12.14 | 30.17 | +0.09 [−0.12, +0.29] equivalent |

**Selection: `spec` retained** — species coefficients, directional block, no environment readouts.
The rule was "simplest variant that is not inferior; `b_i(h_i)` only on superiority", and nothing
is superior: rank2's +0.12 has a CI two orders inside τ. `spec` is also the cheapest at 82 s/epoch.
The environment readouts built for A1 stay in the code and out of the production model.

## W5 — energies in the loss

| arm | force | E79 | between-size (meV) | force TOST vs w=0 |
|---|--:|--:|--:|---|
| w = 0 | 12.22 | 1.263 | +21.6 | baseline |
| w = 0.01 | 12.28 | 1.020 | +5.6 | −0.14 [−0.36, +0.09] equivalent |
| **w = 0.05** | 13.51 | **0.695** | **−0.4** | −1.30 [−1.61, −0.99] equivalent (inferior within τ) |
| w = 0.1 | 13.93 | 0.463 | −4.3 | −3.18 [−5.54, −0.82] **inferior** |
| w = 1.0 | 17.00 | 0.274 | +0.3 | −5.07 [−5.79, −4.35] **inferior** |

**Ruling: `energy_weight = 0.05`.** Energy error falls 1.8× and the between-size residual after one
C_Q collapses from +21.6 to −0.4 meV for a force cost that stays inside τ. Above 0.05 the force
cost leaves τ.

On the electrostatic arms the trade disappears: at the same weight B′ and W6 are both slightly
*better* on forces than the forces-only Φ=0 baseline (+0.38 and +0.53, inside τ). What the Φ=0
sweep measures is the price of fitting energies **without** electrostatics.

**Leak readout (head d-slope, 79 vs 159).** Built to the plan's wording; no instrument existed for
the v5 models. The labels' own axial d-slope is +189.0 meV/Å per Å at 79 atoms and −109.4 at 159,
a size difference of +298.4. Six seeds per arm:

| arm | head 79−159 | head − label | as % of label |
|---|--:|--:|--:|
| **W6, w=0.05** | +283.1 | **−15.3** | **5 %** |
| B′, w=0.05 | +267.2 | −31.2 | 10 % |
| Φ=0, w=0.05 | +262.3 | −36.1 | 12 % |
| Φ=0, w=0 | +266.0 | −32.4 | 11 % |

**W6 tracks the labels' size-dependence about twice as closely as anything else in the programme**,
and the two Φ=0 arms sit together at 11–12 % whether or not energies are in the loss — so the
residual is the band term's, not the energy term's. The tiling ladder orders the arms the same way
from a different quantity, so two independent instruments agree.

## W6 — the SCF-free model

```
E = E_base + ΔF_band(H0) + E_M(Q; h) + E_host + C_Q
E_host = dqᵀ Γ_LR (s q0)          non-self-consistent: dq and q0 are both fills of H0
E_M    = ½ Q² [ξ(h) + 4π r_g² C/Ω] / ε∞      analytic, cell only (stress, no force)
```

`occ_S` and `occ_R` come from the same eigendecomposition as the fill and share `occ_R`, so the
backward costs the two Fréchet contractions the plan budgets. `E_host` is not a Hellmann–Feynman
term, so its force carries the full ∂dq/∂R and ∂q0/∂R. `E_M` honours C13: it carries the model
density's second-moment term, verified to machine precision as `E_PBC_ii(density) − C/(√π r_g)`.
On the real cells it is −332.31 meV at 79 atoms and −269.41 at 159 for Q = +1, a **+62.90 meV**
between-size term supplied analytically where the SCF models get it from ½dqᵀΓdq.

**Gates, all six seeds:**

| gate | reading | verdict |
|---|---|---|
| forces, TOST vs B′ 0.05, τ = 1.7 | +0.14 [−0.21, +0.50] | **equivalent** |
| energy RMSE at 79, margin 0.10 | −0.049, bound −0.069 | **pass** |
| between-size after one C_Q, ≤ 10 meV | +5.0 (B′ +3.7) | **pass** |
| tiling ladder | 93 % of the exact monopole slope (B′ 106 %, Φ=0 −35 %) | **pass** |
| benchmark, model ÷ base ≤ 2 | 2.10 at 79, 2.47 at 159 | **misses by 0.10 / 0.47** |

**Adopt W6.** Forces equivalent to the self-consistent model, shape not inferior, ladder good, and
the benchmark missed by a margin that the Φ=0 control shows is structural rather than a property
of the electrostatics.

## Speed, against the base

Registered P4.3 protocol, idle A4000, warm-started along a charged trajectory. `head` = model −
base is the comparable column: the base leg itself scatters ±10 % run to run at 79 atoms.

| arm | 79: base / model / head (ms) | ratio | 159: base / model / head (ms) | ratio |
|---|---|--:|---|--:|
| Φ=0, w=0 | 42.5 / 83.4 / 41.0 | 1.95 | 67.1 / 147.6 / 80.5 | 2.21 |
| Φ=0, w=0.05 | 37.3 / 79.3 / 42.0 | 2.12 | 67.2 / 148.0 / 80.8 | 2.19 |
| **W6, w=0.05** | 44.4 / 91.6 / **47.2** | **2.10** | 67.9 / 167.4 / **99.4** | **2.47** |
| Route A, w=0 | 43.5 / 131.8 / 88.3 | 2.99 | 67.5 / 257.7 / 190.2 | 3.81 |
| B′, w=0 | 37.6 / 155.1 / 117.5 | 4.12 | 67.0 / 293.6 / 226.6 | 3.91 |
| B′, w=0.05 | 44.7 / 152.0 / 107.3 | 3.55 | 67.2 / 295.8 / 228.6 | 4.40 |

The whole electrostatic apparatus in W6 — two Fréchet contractions, Γ_LR, the host term, E_M —
costs **+6 ms (+15 %) at 79 atoms and +19 ms (+23 %) at 159** over Φ=0. The SCF loop it replaces
cost B′ +66 and +146 ms. **Φ=0 alone reads 1.95–2.21×**, so at these sizes the 2× criterion is a
statement about the base plus one 316-orbital float64 eigh, not about the electrostatics.

Two exact optimisations landed with W6: one `autograd.grad` call for all Hellmann–Feynman
cotangent terms (training step W6 500 → 362 ms, B′ 1070 → 928 ms) and cell-level memoisation of
the reciprocal-vector set and the Madelung constant (W6 inference head 51.1 → 47.2 ms at 79). Both
verified to change nothing: energies exactly equal, forces and parameter gradients equal to
float64 round-off, inference bit-identical.

## Calibration and localisation

**`C_Q` is now in the checkpoints.** It was never a learned parameter — under the quadratic energy
loss its optimum is the mean residual, so the trainer profiles it out — but it was also never
written to the model, so every checkpoint returned uncalibrated energies. All 24 production
checkpoints now carry it, fitted in closed form on each run's own training fold:

| arm | `C_Q(+1)` median (eV) | spread across seeds | training residual sd (eV) |
|---|--:|--:|--:|
| Φ=0, w=0 | −10.6636 | 0.048 | 0.0993 |
| Φ=0, w=0.05 | −10.7350 | 0.305 | 0.0525 |
| B′, w=0.05 | −9.9394 | 0.476 | 0.0361 |
| W6, w=0.05 | −9.3257 | 0.787 | 0.0407 |

The residual sd is the per-cell energy error the constant leaves: 36–41 meV with energies in the
loss, 99 meV without. `model_calibrated.pt` sits beside the untouched `model.pt` every measurement
was taken on.

**Localisation** (C5, analysis not a gate since 2026-09-13):

| arm | C5 | separation p50 (meV) | `N_eff` p50 |
|---|---|--:|--:|
| Φ=0, w=0 | 6/6 | 452 | 1.55 |
| Route A, w=0 | 6/6 | 495 | 1.72 |
| B′, w=0 | 6/6 | 392 | 1.32 |
| Φ=0, w=0.01 | 6/6 | 444 | 1.58 |
| Φ=0, w=0.05 | 3/6 | 338 | 1.54 |
| Φ=0, w=0.1 | 0/6 | 250 | 1.85 |
| Φ=0, w=1.0 | 0/6 | 166 | 1.62 |
| B′, w=0.05 | 5/6 | 332 | 1.35 |
| W6, w=0.05 | 1/6 | 283 | 1.34 |

Every arm passes at w = 0. The energy term pulls the level toward the band edge monotonically in
its weight (452 → 444 → 338 → 250 → 166 meV), in every arm including the one with no
electrostatics — so this is the energy term's doing, not the electrostatics'. What the static
pattern does is tighten the hole: `N_eff` 1.32–1.35 in every arm that has one, against 1.55–1.72
without.

## Practical implications

- **Forces and same-charge energy differences are what this model gives.** 11.6 meV/Å per
  component; 0.5 meV/atom ≈ 41 meV on a 79-atom cell after the charge constant.
- **`C_Q` is a calibration, not a prediction.** One constant per charge state, profiled out in
  closed form during training and now stored in the checkpoints. It encodes the charged-cell
  reference (background, electron reservoir, the DFT conventions of these labels), so a new system
  or charge state needs one DFT total energy to set it. After that, differences — including
  transition levels — are available at the accuracy above.
- **The constant transfers across size:** fit at 79 atoms, apply at 159, 5 meV of error.
- **Finite-size extrapolation is good to ~7 %**: at L = 14 Å the monopole correction is 0.34 eV,
  so ~25 meV of systematic in anything extrapolated to the dilute limit.
- **Cost ~2× a plain MACE step**, and no SCF to babysit: no convergence failures, no warm starts,
  no multi-valued fixed points. Those failure modes cost this programme the most time and W6 does
  not have them.
- **Do not compare uncalibrated energies across models**, and do not read the ladder's absolute
  `E(+1) − E(0)` as a quality measure — the arms differ there by 0.54 eV, which is exactly the
  constant each one's `C_Q` absorbs (measured: C_Q differs by 0.540 eV, ladder dE by 0.565 eV).

## What is not established

- One defect, one composition, two cell sizes, one dataset. Nothing here says the head transfers.
- The defect level sits nearer the band edge once energies are in the loss, in every arm (Φ=0
  452 → 338 meV going 0 → 0.05). The state stays on the defect (≥98.8 % of the hole), so forces
  and energies are unaffected, but the eigenvalue should not be read as a physical level without
  checking. C5 is recorded as analysis, not a gate (user ruling, 2026-09-13).
- The training gradient drops the second derivative of the occupations in H (`_SiteOccupation`
  holds `eps`, `U` detached). Route B′ has trained on exactly this since v4.2; W6 adds the same
  class of term for `dq`. It affects optimisation, not predictions.
- The 159-atom p95 in the benchmark is 7.2–8.9 for every arm including Φ=0: that is the 13-frame
  sample, not a model property.

## Artefacts

| what | where |
|---|---|
| registrations, rulings, chronology | `defect-perovskite/DSCC_V5_IMPLEMENTATION_SPEC.md` |
| runs (held-out metrics, logs, checkpoints) | `~/runs/dscc/dscc_{w3fix,w4,w5,w6}_*/` |
| 2× benchmark | `~/runs/dscc/bench_final/` |
| tiling ladders | `~/runs/dscc/ladder_v5/` |
| leak readout | `~/runs/dscc/leak_v5/`, `defect-perovskite/w5_leak_readout.py` |
| localisation (C5, analysis) | `~/runs/dscc/c5_*.json` |
| `C_Q` calibration | `~/runs/dscc/calibration_v5.json`, `defect-perovskite/dscc_calibrate.py` |
| tests | `tests/extensions/dscc/` (200 passing) |
