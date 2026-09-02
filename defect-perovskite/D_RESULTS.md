# D-series — diagnostics on existing checkpoints, before any Stage-1 training

Cohort: the 12 A/B cells (delocalised-but-fitting) plus F2's six localised cells, 18 models.
No retraining. Predictions were recorded in the scripts' own docstrings before the runs.

---

## D-1 — sensitivity audit: **null, as pre-registered**

`||dH/dR_j||_F` per atom by Hutchinson probe, 16 charged frames × 8 probes, split
defect-near / bulk-like by block-1 descriptor distance to the pristine per-species median.

| cohort | n models | ratio near/bulk | rank ratio |
|---|---|---|---|
| localised (N_eff ≤ 8) | 5 | 1.01 | 0.89 |
| delocalised | 13 | 1.01 | 0.84 |

Per-model ratios span **0.98–1.05** — no model separates. The within-frame rank split (top
15 % most descriptor-anomalous against the bottom half) gives 0.73–0.97, i.e. the most
anomalous atoms are if anything *slightly less* sensitive than the median.

**H's geometric sensitivity is uniform over the cell.** This holds at N_eff = 1.29 as firmly
as at N_eff = 50.85: even when the eigenstate is compact, the matrix it comes from is not
concentrated anywhere. That is consistent with D1's 1–2 % on-site leak and with the r = 0.995
fit running through missing-neighbour hoppings at physical scales.

Per the plan, **this null does not gate Edit 3** — bounded elements are adopted on structural
grounds (they close the V1 spectrum-shift and superatom escape classes), and the prediction of
a possible null was recorded in advance.

One caveat that belongs with the number: the feature-based classifier only partly tracks the
geometric defect region (agreement 0.63–0.68 against a 0.5 chance baseline, near-fraction
0.28–0.35 where the geometric region is ~0.18). The absolute tolerance is calibrated on
80-atom stoichiometric frames and applied to 79-atom charged ones, which is why the rank split
is reported next to it. Both give the same verdict.

---

## D-2 — error binning: **one prediction weakly met, one refuted**

432 frames (18 models × 24). `depth` is the frontier level's distance to the next state;
`δ_L` is the pristine supercell's mean adjacent spacing, per model.

### Resonance

| bin | n frames | head RMSE (meV/Å) | base RMSE | head − base |
|---|---|---|---|---|
| resonant, `depth/δ_L < 1` | 18 | 36.4 | 33.8 | **+2.6** |
| marginal, 1–2 | 39 | 32.8 | 36.9 | −4.1 |
| bound, > 2 | 375 | 28.8 | 34.2 | **−5.4** |

Head error does fall monotonically from the resonant to the bound bin, and the base error is
flat across bins (33.8 / 36.9 / 34.2) — so the trend is the head's, not the data's. Read
against the base, the head **improves on bound frames and is slightly worse than doing
nothing on resonant ones**, which is the predicted direction.

But `n = 18` resonant frames, from 5 of 18 models. This is suggestive and no more; it is not a
result to build on, and the Stage-1 re-run is where it either firms up or does not.

### Electrostatic depth

| bin | head RMSE (meV/Å) |
|---|---|
| low `|phi_LR|` | 28.4 |
| high `|phi_LR|` | 30.6 |

Weak, in the predicted direction. 2.2 meV/Å across the median split, 18 models. Stage 1's gate
asks for this to shrink; it starts small, so the gate should be read as "does not grow".

### The λ–d anomaly: **not explained by resonance**

This is the substantive finding, and it goes against the prediction.

| where measured | result |
|---|---|
| per model, **bound-bin frames only** (`depth/δ_L > 2`) | mean **−0.422**, negative in **17 / 18** models |
| models with any resonant frame at all | 5 / 18 |

The anomaly is fully present on the frames the model itself marks bound, in almost every
model. It cannot be recorded as explained-by-resonance: 13 of 18 models have **no resonant
frames whatsoever** and still show it.

(The pooled correlation over bound frames is −0.032, but pooling is the wrong statistic here —
each model has its own λ origin, so pooling averages 18 different offsets together. The
per-model figure is the one that means something.)

**Consequence.** The ledger's open item stays open and unexplained, and §7's stop condition
stays live: if `corr(level, d)` is still anti-physical on bound frames under the counting
head, that is a stop-and-investigate, not a footnote. `Delta_bind` remains unusable as a depth
until this is resolved.

---

## What the D-series changes about the plan

Nothing structural. Both edits keep their justifications: Edit 1 on the response channel's
measured clause-1/clause-2 split, Edit 3 on the escape classes it closes. What the D-series
adds is one live risk — the λ–d anomaly is now known **not** to be a resonance artefact,
which was the cheapest available explanation and the one the plan expected.
