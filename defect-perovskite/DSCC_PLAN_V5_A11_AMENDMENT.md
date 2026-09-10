# A1.1 — input standardisation for the reference-free readouts (corrects A1)
- Inputs to every readout (e_Z, g_Z, f_Z, m_ZZ'): per-species standardisation
  ĥ = (h − μ_Z)/σ_Z with μ_Z, σ_Z per channel over all training atoms of species Z
  (all frames, thermal ensemble), computed once and frozen; recorded in the checkpoint.
  Foundation form: per-species running statistics over the corpus.
- Readout weights initialised at 0.1× default (corrections start near zero); bias free.
- Bounds, L2, forms, η = ln 3 unchanged.
- Diagnostic during training: p95 onsite correction per species per epoch; expected to
  reach O(0.5–1 eV) within the first epochs on the Φ = 0 arm.
- Record: A1 as run removed input normalisation together with the reference; the Φ = 0
  comparison (A1 vs centred, base v2) is retained as the evidence.

## Why (user, 2026-09-10, recorded verbatim in substance)
The physics argument for dropping the reference was right; the reference was doing a second
job, and that job is numerical. `tanh[h(x_i) − h(x̄_Z)]` feeds the nonlinearity a deviation
from the species mean — that is input normalisation. A linear readout on raw features splits
as `w·x̄_Z + b` plus `w·(x_i − x̄_Z)`, and every step that grows `w` to capture the environment
also moves the species-constant part unless `w` is orthogonal to `x̄_Z`. That constant is a
species-dependent onsite shift: it moves the relative Pb/Cl/Cs levels and the gap, so the gap
regulariser and the spectral structure of `H0` resist it. The bias can compensate in
principle, but the two are coupled through a large, ill-conditioned direction, and the
optimiser's answer is to keep `w` tiny. The centred form never had this problem because its
constant part was structurally zero, so every direction of `w` was free. The uniform
degradation across shells follows: a level that is off reshapes the carrier, and that reaches
every force. With standardised inputs the constant part is exactly the bias, so the gap
regulariser acts on the bias alone and `w` is free.

Rejected alternative, recorded: a per-atom LayerNorm over channels is statistics-free but
removes only the scale of the species-mean direction, not the direction itself.

## The conditioning measurement (base v2, 120 frames, 9600 atoms, 512 channels)
| species | ‖μ_Z‖ | median ‖x_i − μ_Z‖ | ratio | median channel \|μ\|/σ | p95 channel \|μ\|/σ |
|---|---|---|---|---|---|
| Cl | 10.27 | 0.351 | **29.2** | 18.6 | 70.7 |
| Cs | 9.56 | 0.209 | **45.7** | 28.1 | 129.5 |
| Pb | 10.31 | 0.278 | **37.1** | 20.8 | 98.6 |

The predicted threshold was "several or more". It is 29–46.

## Implementation
- `SlaterKosterH`: buffers `feat_mean`, `feat_sd`, `feat_stats_set`; `set_feature_stats`
  (floors σ at `FEATURE_SD_FLOOR` = 1e-4 × that species' median channel σ and returns the
  count of floored channels, so dead base channels are visible rather than silently divided);
  `standardise(h, Z)`. Applied in `on_site`, in `integrals` (each end by its own species,
  before the symmetric combination), and in `H0.env_factor` for `g_Z`/`f_Z`.
- `MACEDSCC.set_feature_stats(batches)`: one pass, sums and sums of squares in float64.
  **Training frames plus the pristine cells; the held-out fold is excluded** — the features are
  label-free so a held-out frame leaks no label, but statistics fitted on the evaluation set
  are indefensible later for no gain now.
- **Hard guard**: `MACEDSCC.forward` raises if the statistics are unset. Running without them
  reproduces the A1 failure silently, which is exactly what must not happen twice.
- `READOUT_INIT_SCALE = 0.1` on the final layer of every readout (the campaign used 0.05 on
  raw inputs; 0.1 is A1.1's registered value).
- Per-epoch `on_site_shift_eV_p95` and `..._by_species` in `history.json`, and in the log line.

## Verification before the retrain is committed
Registered in advance: with the fix in, the p95 on-site correction should reach the
several-hundred-meV range within a few epochs on the Φ = 0 arm. The A1 run held it under
**0.105 eV** for all 60 epochs against a 3.079 eV bound, where the centred head reached
**2.99 eV** (|tanh| = 0.996, at the stop).

## Verification result (2026-09-10 18:20, seed 0, fold 0, 8 epochs, Φ = 0, constant lr)

**The registered criterion is NOT met, and the fix nonetheless works — better than the
criterion was designed to detect.** Both halves are recorded because the second does not
excuse the first.

| epoch | p95 on-site correction (eV) | Cl / Cs / Pb | train force | held RMSE (meV/Å per comp.) |
|---|---|---|---|---|
| 0 | 0.163 | 0.095 / 0.115 / 0.223 | 2.59e−5 | — |
| 1 | 0.161 | 0.094 / 0.113 / 0.221 | 1.81e−5 | 24.30 |
| 3 | 0.158 | 0.092 / 0.112 / 0.225 | 9.55e−6 | 19.12 |
| 5 | 0.155 | 0.088 / 0.110 / 0.219 | 7.99e−6 | 17.41 |
| 7 | 0.154 | 0.087 / 0.105 / 0.207 | 6.68e−6 | **16.53** |

- **Criterion (p95 → O(0.5–1 eV) within the first epochs): FAILS.** The correction sits at its
  initialisation scale (0.1 × default on unit-variance inputs gives ≈ 0.15 eV) and drifts
  slightly DOWN, 0.163 → 0.154. Saturation is 0.0000 on both terms; the max correction is
  0.322 eV.
- **The objective the criterion stood proxy for: EXCEEDED.** Same seed and fold, held-out
  force RMSE per component: A1.1 **16.53 at epoch 7**, against the centred head's **17.99 at
  epoch 59** and the unstandardised A1 head's **23.06 at epoch 59**. Train force loss 6.68e−6
  at epoch 7 against the centred head's converged 8.45e−6. Eight epochs beat sixty.

**Reading, ours.** With well-conditioned inputs the head reaches a better fit with SMALLER
corrections, so "the corrections must grow" was the wrong proxy for "the readouts can learn".
The centred head's 1.85 eV p95 and 2.99 eV max were reached at `|tanh| = 0.996` — a channel
at its stop, where `sech² ≈ 0` and the parameter has no gradient. That is the same pathology
Stage A′ diagnosed on the hopping bound, and it is not obviously a state to aim for. What A1.1
restored is the ability to *place* the correction, not the size of it.

**Caveat kept in view**: one seed, eight epochs, and the two comparators are 60-epoch numbers.
The six-seed arm settles it; nothing above is a registered reading.

## η, closed
`η = 0.5` in A1 was a transcription error against an already registered value; `η = ln 3`
stands. Since nothing approaches either bound on base v2, the choice is moot. **Corollary,
registered on its own line: the Stage A′ hopping-stop cohort does not reproduce on base v2** —
the largest fitted `|tanh m|` is 0.172 (A1) and 0.164 (pre-A1), factors of 1.21 and 1.20,
where Stage A′ found the cohort pinned at the stop in every d bin on the old base.
