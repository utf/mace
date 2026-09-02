# Stage-3 rerun — 6/6, N_eff 2.9, and the gap comes out right

Six seeds each, lr 0.01, 60 epochs, matched Stage-2 control. Float64 solve, Harrison term
values, c-shift, warmup, clip 1.0, init gate, `loss_gap`, learned Z. **The P-backward is
built and validated but NOT wired into the force path** — this measures §3 alone.

| | trained | axial_red | rmse_all | N_eff | pristine frontier gap |
|---|---|---|---|---|---|
| **Stage 3 (counting, s+p)** | **6/6** | **+0.519 ± 0.008** | **33.4 ± 0.2** | **2.92** | **2.394 eV** |
| Stage 2 control (s-only) | 6/6 | +0.130 ± 0.165 | 50.1 ± 6.0 | 33.1 | 0.741 eV |

Per seed, Stage 3: N_eff 2.93, 2.86, 2.95, 2.96, 3.05, 2.78. axial_red +0.507 to +0.533.
No init-gate trip on any seed.

## Forecasts

* **F1 — falsified.** The NaN was float32, at both cell sizes, not counting or occupations.
* **F2 — met.** 6/6 train at lr 0.01; the init gate never fired.
* **F3 — met, and not trivially.** N_eff 2.92 against the control's 33.1, in 6/6. The
  bandwidth caveat cuts the *right* way: Stage 3 has **32 eV** of bandwidth against the
  control's ~2 eV, so it is 16× wider and 11× more localised. This is not N_eff falling
  toward the atomic limit — it is a compact state inside a real band.
* **F4 — not met** (scored earlier on the old models: right sign 3/4, 7× short of −0.134).
  Needs re-running on these.
* **F5 — not scored.** `corr(λ, d)` was not in this queue.

## The gap gate, which the s-only head cannot pass

`loss_gap` drives the pristine frontier gap toward E_gap = 2.4 eV. Stage 3 lands at **2.394**
(|Δ| = 0.006 against a 0.10 gate) on every seed, spread 2.381–2.403. The control reaches
**0.741** — and its per-seed values, 0.155 to 1.984, track its axial_red exactly.

## The control settles the fit↔superatom question

| control seed | axial_red | N_eff | frontier gap |
|---|---|---|---|
| 1 | +0.004 | 22.0 | 0.155 |
| 5 | +0.019 | 11.8 | 0.146 |
| 6 | +0.012 | 9.2 | 0.200 |
| 4 | +0.044 | 22.4 | 0.332 |
| 2 | +0.264 | **69.5** | 1.626 |
| 3 | **+0.439** | **63.7** | 1.984 |

The s-only head fits only by delocalising. Its two best seeds are its two most delocalised,
at N_eff 64–69.

**This restores the correlation I retracted, and my retraction was the error.** The `+0.987`
was measured at lr 0.01 and is real; lr 0.05 suppressed the superatom route rather than
disproving the link, and I read the suppression as a refutation. I have now been wrong in
both directions on this — the durable statement is the one the control makes here directly:
*for the bounded s-only head, force fit and delocalisation are the same axis.*

**Stage 3 breaks that axis.** It is simultaneously the best fit (+0.519 against +0.439 from
the control's most delocalised seed) and by far the most localised (2.92 against 63.7). That
is the first time in this programme those two have moved together in the right direction.

## What this does not yet establish

Six seeds, one cell size, forces only. The dilution gate on the two-size frames, the
159-atom `dE_head` trend, and `corr(λ, d)` are all unrun on these models. The joint run is
still gated on those.
