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



## !! Update after initial implementation

# L_size — amendment: reformulate in log-ratio space

Supersedes the `f`-space hinge in §4 of the size-extensivity plan. The zero-set is
unchanged; the gradient is not. Everything below is training-time only — no architecture
change, no inference-time change, `Σ_i α_i = 1` untouched.

## 1. Why the shipped form must change

With `D = f·c`, `f = σ(ln B − ln A)`, `c = ū_w − ⟨u⟩_α`:

```
∂D/∂(ln A) = −f(1−f)·c        # raises the gap — what we want
∂D/∂c      = f                 # flattens defect contrast — what we don't
```

The ratio of the two is exactly `(1−f)`. At the plan's `R = 1e4`, `f` pins at 1.000 on
286-atom cells, so the useful path is suppressed by ~10⁻³ and the term descends by
destroying defect contrast — the failure it exists to prevent.

At `f = 1` the feasible set `|f·c| ≤ tol` degenerates to `|c| ≤ tol`, i.e. contrast below
5 meV. The term's unique optimum is the pathology.

`R` is not at fault and must not be reduced to work around this: `R` states the validity
range, and lowering it to restore trainability trades the specification for the optimiser's
convenience. The reformulation decouples the two.

*(Note: the implementation write-up quotes "3–4 orders of magnitude" where the table shows
~48×. Fix the docstring to match the table before this is cited anywhere.)*

## 2. The reformulated term

`σ` is monotone, so `σ(x)·|c| ≤ tol ⟺ x ≤ ln(t/(1−t))` with `t = tol/|c|`. Penalise in
`x`-space:

```
x_c   = ln B_c − ln A_c                       # log-ratio, as before (logsumexp)
t_c   = tol_c / max(|c_c|_EMA, eps_c)         # c detached and EMA-smoothed
x*_c  = ln( t_c / (1 − t_c) )                 # threshold; +inf when t_c >= 1
L_size = Σ_{c : n_c > 0}  max(0, x_c − x*_c)²
```

`∂/∂(ln A) = −2·max(0, x − x*)` — linear in the gap, never vanishing while violated.

### Properties verified against the original spec

| Property | Status |
|---|---|
| Exempt at zero contrast | Holds. `|c| → 0` ⇒ `t → ∞` ⇒ `x* → +∞`, penalty inactive for any `x`. Continuous, not a branch |
| Hinge stops when satisfied | Holds |
| Silent before contrast develops | Holds |
| Badly-violated behaviour | `(ln|c|)²` rather than `|c|²` — gentler, better conditioned |
| Convergence | `f ∝ e^{−gap}` falls exponentially while `|c|` saturates; constraint closes at `gap ≈ ln(RN/k) + ln(Δu/tol)` ≈ 18 on current numbers |
| Labels / structures / inference inputs | None, as before |

## 3. Required alongside — gradient structure of the bulk reference

Two properties are needed, and they are not the same:

- **Shift-invariance:** `Σ_i ∂x/∂ℓ_i = 0`, so uniform logit inflation is not a descent
  direction.
- **Uniform-state null:** `∂x/∂ℓ_i = 0` *pointwise* on a uniform state, so `L_size` cannot
  create structure where none exists.

With `w_Z = m_Z e^{ℓ̄_Z}/B`, the gradient is `∂x/∂ℓ_i = w_{Z_i}·(∂ℓ̄_{Z_i}/∂ℓ_i) − α_i`.

**Use the plain arithmetic mean of `ℓ` in log space, undetached:**

```
ℓ̄_Z = (1/N_Z) Σ_{i : Z_i = Z} ℓ_i          # NOT detached, NOT trimmed, NOT median
ū_Z = median_{i : Z_i = Z} u_i               # detached — robustness is free here
```

Then `∂ℓ̄_Z/∂ℓ_i = 1/N_Z` for every atom of that species, giving
`∂x/∂ℓ_i = w_Z/N_Z − α_i`, which:

- sums to `Σ_Z w_Z − 1 = 0` exactly, always — shift-invariance holds unconditionally;
- equals `1/N − 1/N = 0` pointwise on a uniform state with matched composition, since
  `w_Z = N_Z/N` there — uniform-state null holds;
- in the localised regime gives defect atoms `1/N − α_i` (strongly negative ⇒ logits rise)
  and bulk atoms a small positive value (⇒ logits fall). Correct direction on both.

**Do not detach, trim, or use a median on the `ℓ` side.** Detaching gives
`∂x/∂ℓ_i = −α_i`, summing to `−1`: uniform inflation becomes a false descent direction.
Trimming gives `∂ℓ̄_Z/∂ℓ_i = (1/N_Z^kept)·1[i kept]`, which sums to zero but is **nonzero
pointwise on a uniform state** (`−1/N` on trimmed atoms, measured). Since two-sided trimming
selects by extremal `ℓ`, which is arbitrary under ties, that would promote seed-determined
atoms — the premature-commitment failure mode, reintroduced.

**Why the mean is safe in log space.** The median was originally motivated by shell atoms at
`e^ℓ ~ e^{12}` dominating the average. That applies to averaging in *linear* space —
`ln(mean_i e^{ℓ_i})` overestimates `ℓ_bulk` by `gap − ln(N_Z/k_Z)` ≈ 8 at gap 12. In log
space the shift is only `k_Z·gap/N_Z` ≈ 0.25 at N = 286, gap 12, and 0.38 at gap 18:
bounded, shrinking as `1/N`, and biased so that `ln B` is overestimated — the constraint is
marginally tightened, never loosened.

**Fallback if that bias is ever measured to matter** (much smaller cells, or many defect
atoms per species): straight-through median,
`ℓ̄ = median.detach() + mean − mean.detach()` — robust value, mean gradient, both properties
preserved, ~0.6% gradient bias at current localisation. Do not adopt pre-emptively.

**Consequence for warmup.** With the uniform-state null restored, `L_size` cannot create a
gap on a uniform state, only deepen an existing one. Keep the post-escape warmup as good
practice, but it is no longer the sole guard against structure creation.

## 4. Other changes

- **Move `n_c` into the tolerance:** `tol_c = tol / n_c`, outer weight an indicator on
  `n_c > 0`. The `n_c²` weighting was justified by `D` being an energy; `x` is a log-ratio,
  so that justification does not transfer. Per-channel drift budget does.
- **Recalibrate `λ_size` from scratch.** The loss is now dimensionless; the previous value
  carries no meaning. Target 5–10% of `L_Δ` when active and log the realised ratio.
- **EMA-smooth `|c|`** per channel before forming `x*`, so a detached threshold does not
  chase per-batch noise.
- **Guard `t ≥ 1`:** when `|c| ≤ tol` the constraint is satisfied for any `x` — return zero.
  Use an `eps_c` floor on `|c|` and **log how often this branch fires**; frequent firing
  means contrast has collapsed and is a finding, not a nuisance.
- **Keep `R = 1e4`.** The reason for lowering it has been removed.

## 5. Logging (extend the existing per-epoch set)

Per channel, every epoch, regardless of whether the term is active:

- `x_c`, `x*_c`, and the violation `max(0, x_c − x*_c)`
- `size_f` (retain — it is the direct saturation indicator)
- `|c|_EMA` and the `t ≥ 1` branch-fire count
- realised `L_size / L_Δ` ratio

## 6. Tests

Keep the existing saturation test as a **regression guard** — it encodes a failure invisible
in the loss value, and would silently reappear if anyone reverts to the `f`-space form.
Add:

1. **Zero-set equivalence.** Random `(x, c, tol)`: `σ(x)|c| ≤ tol` iff `x ≤ x*`.
2. **Gradient direction.** At `f > 0.99`, the gradient w.r.t. `ln A` is non-negligible and
   the gradient w.r.t. `c` is zero (detached).
3. **Shift-invariance.** Uniform `ℓ → ℓ + s` leaves `x` and `L_size` unchanged, and
   `Σ_i ∂x/∂ℓ_i = 0` to numerical precision.
4. **Uniform-state null.** On a pristine cell with matched composition, `∂L_size/∂ℓ ≡ 0`.
5. **Delocalised exemption.** `|c| < tol` ⇒ `L_size = 0` for arbitrarily large `x`.
6. **Continuity across `t = 1`**, approached from below.

## 7. Interpretation note for the plan

A required gap of ~18 does **not** mean the carrier must collapse onto `k` atoms. The
constraint is on shell-versus-**bulk-plateau** weight, not on the size of the support: the
model may spread weight over 20–50 atoms following the physical envelope and still satisfy
it. What it cannot retain is the flat, non-decaying plateau beyond the receptive field,
which was never physical. Over-localisation risk is governed by `tol`, not by the mechanism.

Acceptance (`alpha_dilution.py`: implied gap and `α` on shell flat in `N`) is unchanged.
