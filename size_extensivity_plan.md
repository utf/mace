# MACEDefect — size extensivity of the carrier correction

Standalone work plan. Self-contained; no other document needed.

---

## 1. The problem

The carrier correction is

```
ΔE_SR = Σ_c n_c Σ_i α_i^c u_i^c ,    α^c = softmax(ℓ^c) over all atoms in the cell
```

Measured on the 4H-SiC divacancy, `E_ZPL` falls from 0.903 eV at 256 atoms to 0.715 eV at
2400 atoms and is still moving. The base branch is exactly size-consistent (0.000 meV/atom
drift at all sizes), so the whole effect is in the correction.

Cause, measured: attention on the defect shell decays with cell size. Implied logit gaps are
constant in `N` to ±0.02, so the softmax picture is the right description:

| channel | implied gap | crossover `N*` | α on shell, N = 70 → 2398 |
|---|---|---|---|
| `e_maj` | 2.25 | 57 | 0.483 → 0.023 |
| `e_min` | 2.39 | 66 | 0.496 → 0.027 |
| `h_min` | 5.23 | 1121 | 0.943 → 0.326 |

Three of four channels are already diluted at training sizes (286–398 atoms).

## 2. Why it happens — first principles

Charge conservation requires `Σ_i α_i = 1`, so `α_i = g_i / Σ_j g_j` for some non-negative
gate `g`. The denominator is

```
Σ_j g_j  =  Σ_shell g  +  (N − k)·g_bulk
```

If `g` is a **strictly positive local function of descriptors**, then `g_bulk > 0` is a
constant and the denominator grows without bound. The softmax is not the culprit —
`g = e^ℓ` is one instance, and every strictly-positive local gate fails identically.

The deeper reason: beyond `r_max × n_layers` every atom has an identical descriptor and so
an identical logit. The logit field is **two-level** — shell plus a flat plateau — where the
physics requires an exponentially decaying envelope, `|ψ(r)|² ~ e^{−2κr}`. A real bound
state's normalised weight saturates because each shell is exponentially suppressed. A
two-level field has no such mechanism, so weight leaks to bulk in proportion to `N`.

Consequence: for a strictly positive gate, `α_shell` is `N`-independent only in the limit
`gap → ∞`. The usable condition is

```
gap  ≫  ln(N / k)
```

which is **logarithmically cheap**: gap 2.4 covers ~70 atoms, gap 12 covers ~10⁶. Getting
this right costs almost nothing — but nothing in the current objective asks for it, and the
39% span of training cell sizes leaves the gap unidentified, which is why it is small and
seed-dependent.

## 3. Why the objective is a ratio, not a target cell size

Three candidate objectives, and why two fail:

- **Monotone localisation pressure** (entropy, participation, bulk-weight penalties) forces
  every carrier onto the defect, including band-edge and shallow states that are correctly
  delocalised. Rejected.
- **Exact size-invariance** (`N′ → ∞`) is satisfiable only by `g_bulk = 0` or by destroying
  defect contrast entirely, so with a positive gate its true minimum is *full dilution* —
  the wrong direction. Rejected.
- **Invariance over a finite size range** is correct, and is what a large gap achieves.

So a size scale is structurally required. Express it as a **ratio** `R = N′/N` rather than
an absolute `N_max`: it is dimensionless, needs no knowledge of production cell sizes,
survives retraining on different cells, and states the transferability requirement directly
— *the correction must be stable over R-fold growth beyond whatever it is trained on*.
Absolute sizes appear only in the acceptance test (§7), never in the objective.

## 4. The term to implement

Padding a cell with ideal bulk has a closed form, so **no new structures are needed**.
Adding `m_Z = (R−1)·N_Z` atoms of each species preserves stoichiometry, and each added atom
contributes the bulk gate and bulk site energy.

Per channel `c`, with robust per-species bulk references (medians, since a few shell atoms
with `e^ℓ ~ e^{12}` would otherwise dominate any mean):

```
ℓ̄_Z = median_{i: Z_i = Z} ℓ_i        ū_Z = median_{i: Z_i = Z} u_i      # detached
ln A = logsumexp_i(ℓ_i)                                                  # current cell
ln B = logsumexp_Z(ln m_Z + ℓ̄_Z)                                         # padding
f    = sigmoid(ln B − ln A)                                              # weight captured by padding
ū_w  = Σ_Z m_Z e^{ℓ̄_Z} ū_Z / B                                          # padding-weighted bulk u
```

Then the correction at the padded size is `(1−f)·⟨u⟩_α + f·ū_w`, so the drift is
`f·(ū_w − ⟨u⟩_α)` and the loss is

```
L_size = Σ_c n_c² · max( 0,  | f^c · (ū_w^c − ⟨u⟩_α^c) |  −  tol )²
```

**Defaults:** `R = 1e4`, `tol = 1e-3` eV, `λ_size` calibrated so the term is ~5–10% of the
delta-energy loss when active (log the realised ratio). CLI: `--defect_size_ratio`,
`--defect_size_tol`, `--defect_size_weight` (default 0 → term off).

### Why this term and not another

- **Penalises dilution, not delocalisation.** If `u` has no defect contrast (a genuine
  band-edge or shallow carrier) then `ū_w = ⟨u⟩_α` and the penalty is *exactly* zero for any
  `f` — the state stays fully representable. Only the intermediate regime, weight pinned on
  a shell while the remainder spreads, is penalised. No other localisation penalty makes
  this distinction.
- **Hinge ⇒ it stops pushing once satisfied.** No permanent fight with the energy objective;
  it re-activates only if the fit drifts back into violation.
- **Silent during early training.** Before `u` develops contrast the drift is ~0, so the term
  is inactive through the plateau and escape phase and cannot interfere with the seeding and
  annealing machinery.
- **No labels, no structures, no inference-time change.** `k` never appears; no defect
  positions; nothing is added to the energy function, so forces, Hessians and smoothness are
  untouched and no downstream user inherits a new input.

### Implementation notes

- Work in log space throughout — `e^ℓ` overflows at the gaps we are targeting. Use
  `logsumexp` and the `sigmoid(ln B − ln A)` form above; never form `A` or `B` directly.
- Detach `ℓ̄_Z` and `ū_Z`. Gradients then push the shell rather than dragging the bulk
  reference, and the sparse gradient of a median selection is avoided.
- Median, not mean, for both references. This is load-bearing, not cosmetic.
- `n_c²` weighting zeroes dead channels automatically and matches each channel's actual
  contribution to the energy.
- Activate after the escape phase (`--defect_size_warmup_epochs`, default ≈ 20) so nothing
  competes with the attention search.
- Log `f^c` and the raw drift `f^c(ū_w^c − ⟨u⟩_α^c)` per channel per epoch whether or not
  the term is on. This is a free, label-free extensivity monitor that currently does not
  exist.

## 5. Prerequisites — run these first, or the result is uninterpretable

Both change the quantity being fitted, so enabling them afterwards would invalidate any
size result.

1. **Long-range branch on, `a` frozen at `1/√ε_∞` (DFPT).** With it off, `ΔE_SR` is absorbing
   the electron–hole interaction, which is genuinely long-ranged and genuinely
   size-dependent. Part of the measured drift may be that physics in the wrong term.
   The LR branch cannot fix dilution — screened electrostatics is power-law, and
   `q_i^carrier` is built from the same diluted `α` — but it must be separated out before
   drift is attributed to attention.
2. **Level-mode gauge penalty on.** The current asymptote implies `Σ_c u_bulk^c ≈ −0.3 eV`
   where band-edge referencing requires 0. Pinning it does not fix the drift but makes the
   diluted limit physically interpretable (a band-to-band transition) instead of arbitrary.

## 6. Free re-analysis of existing data — do this before any retraining

The drift is **not logarithmic**; it is logistic in `ln N`, which is what the softmax model
predicts. Fit

```
E(N) = C + Σ_c A_c · α_d^c(N) ,    α_d^c(N) = 1 / (1 + (N/k)·e^{−gap_c})
```

with `gap_c` fixed at the already-measured values (no free shape parameters), fitting only
the linear coefficients. On the existing five-point ladder this reproduces held-out points
to a few meV and yields, at no cost:

- the **localised limit** (`α_d → 1`) — the model's answer with a bound carrier, and the
  honest number to compare against the published 0.91 eV;
- per-channel `Δu`, currently unmeasured;
- the diluted limit `C`, which cross-checks the gauge offset in §5.2.

The existing `1/N` extrapolation fits the wrong functional form and should be withdrawn.
Also state plainly that agreement at 256 atoms is partly compensating error — the model was
trained at 286–398 atoms with `α` already partly diluted — so it is not validation.

## 7. Acceptance

- `alpha_dilution.py`: implied gap and `α` on shell **flat in `N`** across the full ladder.
  This is the gate; ~10 min on CPU, so it can screen every retrain.
- Size-convergence ladder: `dE_ZPL/d(ln N)` below ~1 meV per e-fold across the ladder, and
  the logistic fit's `α_d` flat rather than sigmoidal.
- Held-out `RMSE_dE`/`RMSE_dF` no worse than the current best, on ≥3 seeds. If the gap rises
  and accuracy degrades, report the trade rather than tuning `λ_size` until it hides.
- Per-channel implied gaps ≳ `ln(N_target/k)` for the sizes of interest, reported explicitly
  rather than assumed.
- Re-run seed spread: earlier stages showed large seed-to-seed variation in logit gaps, so
  `N*` per seed must be reported, not a single-model number.

## 8. Fallback — structural gate, only if the gap will not rise

If `L_size` cannot raise the gap without unacceptable accuracy loss, the failure is
structural and the gate must change. Replace the softmax with a compact-support gate on the
deviation from the cell's own bulk level:

```
d_i = ℓ_i − median_{j: Z_j = Z_i} ℓ_j
α_i = [ φ(d_i) + δ/N ] / [ Σ_j φ(d_j) + δ ]
```

`φ` non-negative, C², exactly zero below a margin `w` — the same polynomial cutoff already
used for `r_max`. Then bulk contributes exactly nothing to the denominator and size
consistency is structural, not bounded: no ratio, no tolerance. `Σα = 1` holds exactly;
on a pristine cell `d_i = 0` gives `α_i = 1/N` exactly, so pristine exactness is by
construction; the `δ/N` floor keeps the delocalised limit smooth instead of `0/0`.

**Do not implement without an anneal on `w`.** Atoms below threshold receive exactly zero
gradient, and at initialisation all `d_i ≈ 0`, so a hard cutoff is a permanent plateau.
Ramp `w` from 0 to its final value using the same schedule machinery as the existing `β` and
`γ` anneals, and verify a non-zero fraction of atoms is active throughout.

Residual `N`-dependence from the floor is `O(δ k Δu / N)` — convergent, unlike the logistic
dilution it replaces.

## 9. Rejected, with reasons — do not reopen without new evidence

| Option | Why not |
|---|---|
| Entropy / participation-ratio penalty | Forces localisation on every state, including band-edge and shallow carriers that must stay delocalised; breaks pristine exactness if applied to pristine cells |
| Top-k or sparsemax pooling | Discrete support ⇒ non-smooth PES where the support changes; fatal for phonons, Hessians, configuration-coordinate diagrams and NEB |
| Dropping normalisation for a bounded gate `Σ_i g_i u_i` | Breaks `Σ_i α_i = 1` and hence `Σ_i q_i = a·q`, which the long-range branch and the identification of `a` both depend on; the bulk gate must then vanish *exactly* or the error is `N·ε·u`, unbounded |
| Distance-decay envelope on the logits (`−κ|r_i − r̄|`) | Requires an attention centroid; under PBC the circular mean collapses exactly when `α` is diluted, degrades under thermal disorder, and is undefined for multiple defects |
| Padding with real structures | Requires a bulk-like buffer between defect and new boundary, which cells of this size do not have; would assert invariance between genuinely different configurations. §4 obtains the same constraint analytically |
| Training across many cell sizes | Needs new DFT; §4 gets the constraint from existing frames |

## 10. Recording

Record with the model: `R`, `tol`, `λ_size`, warmup epochs, whether the gate of §8 was used
and with what `w` and anneal schedule, the frozen `ε_∞` and `a`, and the gauge-penalty
weight. Record per-channel implied gaps and `N*` at convergence, per seed — these define the
cell-size range the model is valid over and should be stated wherever it is used.

