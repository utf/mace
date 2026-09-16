# O5 — interim report for the coadvisor

**Date:** 2026-09-14, 23:45 (b3 clock). **Status: 4 of 12 runs complete** (two seeds of each arm).
Numbers below are real but thin; the six-seed TOSTs the ruling asked for are not yet possible.
This report is being sent now rather than at completion because it contains **three corrections
of record** and **one registration error of mine that voids a reading already in the tracker**,
and you should have those before the remaining runs land.

All forces are **per-component meV/Å**. τ = 1.7.

---

## 1. Corrections of record

### 1.1 The first "O5a" runs were not O5a. Cancelled.

Your ruling says, twice: `δε_i = s_Z (V_i − V̄_Z)`, "on-site shift, **linear, no MLP**", and
"**linear readouts**". What I built and launched appended the three invariants
`(V_i, |E_i|, tr(∇E·∇E))` to the 211-channel scalar block and fed the 214-wide result to the
existing on-site MLP. That is a strictly larger hypothesis class, it confounds the crystal field
with an MLP you declined, and it would not have answered the question the arm was set to ask.
Those runs were cancelled before completion and **none of their numbers appear anywhere**. The
arms reported below are the corrected ones.

### 1.2 O3.3's invariants were the full pairwise Gram, not the norms.

You specified the correlation-2 invariants `|A⁽¹⁾_n|²` — 6 channels at `N_r = 6`. What was built
is `A⁽¹⁾_n · A⁽¹⁾_m`, the full symmetric set — **21 channels**, containing the specified 6 as its
diagonal. Undisclosed when O3.3 was registered. It propagates into **O3.3b and O5b**, so every
number reported for those arms is a number for the 21-channel form. Alex's decision is not to
re-run; §8 of the O3 report stands as a statement about what was built.

### 1.3 A gate you named found a live bug.

`capture.fits()` had not been run. Running it showed `GraphedStep` kept its own guard list,
separate from the forward's analytic-route gate, and did not exclude the crystal field. A
captured O5-linear model therefore replayed forces **19 meV/Å wrong** — the crystal-field terms
simply absent from the closed-form gradient, widths matching by coincidence so nothing raised.
(The MLP form crashed instead, 101 channels against 104, which is how it surfaced.) Both callers
now ask one predicate; capture refuses O5 with a reason. The primitive itself **is** two-body and
pads exactly — `V` and `E` bit-identical under 216 to 3288 dummy edges — so what it lacks is only
`dV/dR` in the analytic kernel. Neither path is on the training path, so this costs the arms
nothing, but your two-body argument does have that hole.

---

## 2. The arms as built

**O5a (`--cf_linear`)** — your spec, term for term:

| term | form |
|---|---|
| potential | `V_i = Σ_j q_{Z_j} erf(r_ij/a_{Z_iZ_j})/r_ij · f_cut(r_ij)` |
| on-site | `δε_i = s_Z (V_i − V̄_Z)`, **linear, no MLP**, invariants kept OUT of the readout |
| rank-1 | `a_i (E_i · T_sp)` — one vector channel, so this is literally the form, not a broadcast |
| rank-2 | `b_i (∇E_i : T_pp)`, traceless |
| hopping | `t_ij → t_ij[1 + β_{ZZ'}(V_i − V_j)²]`, β zero-init, symmetrised |

`q_Z` from pristine Mulliken means (**Cl −0.338, Cs +0.511, Pb +0.502** — hard-fails if zero);
`a_Z` per species; `V̄_Z` the pristine per-species mean; `s_Z` zero-init, so the arm starts exactly
at O3.3b. A1.1 standardisation on the scalar block; B-basis kept as the second `l = 0` input.
`beta_a = beta_b = 0` means `a_i`, `b_i` are pure species scalars, as you specified.

**O5c (`--cf_mlp`)** — your fallback, run *alongside* rather than after, because the pair is the
measurement: same primitive, same everything else, linear against MLP.

**Three departures from the letter, stated:**

1. `s_Z` is `[n_el, 2]` — an independent shift for the s and p levels, not one scalar per species.
2. `δε_i` sits **outside** A1's `Δ_Z tanh` bound by construction; the specified form has no
   squashing. It is therefore invisible to the barrier and the saturation fractions, so its p95
   and max are reported separately (§4).
3. O5c keeps the B-basis readout beside its new MLP, so its on-site channel carries two MLPs.
   Deliberate — it makes the crystal-field *form* the only difference between the arms — but it
   is not the letter of "an MLP on those three inputs".

**One unbounded term, and what watches it.** `t[1 + βΔV²]` is the only term no tanh bounds, and
`ΔV² ≥ 0`, so a negative β drives the factor through **zero** and flips a hopping's sign. Your
ruling does not bound it, so it is watched: `min(1 + βΔV²)` over every edge, per epoch, with a
**hard stop** if it is not positive. Floors so far **0.964–0.981** — live, nowhere near the stop.

---

## 3. Current errors (4 of 12 runs)

Held-out force RMSE, per-component meV/Å, per seed:

| arm | s0 | s1 | s2 | s3 | s4 | s6 | mean | n |
|---|---|---|---|---|---|---|---|---|
| W6 | 11.2 | — | — | 11.1 | 12.2 | 11.9 | **11.6** | 4 |
| O3.1 | — | 14.2 | 15.1 | 14.8 | 14.3 | 14.2 | 14.5 | 5 |
| O3.3 | 13.2 | 13.9 | 12.0 | 15.6 | 12.2 | 15.3 | 13.7 | 6 |
| O3.3b (parent) | 13.6 | 14.1 | 12.8 | 13.1 | 15.2 | 13.1 | **13.6** | 6 |
| O5b (`N_r` 16) | 13.2 | 15.8 | 12.7 | 14.7 | 15.8 | 14.6 | 14.5 | 6 |
| **O5a-lin** | **13.7** | **16.5** | — | — | — | — | **15.1** | 2 |
| **O5c-mlp** | **14.3** | **15.0** | — | — | — | — | **14.6** | 2 |

Paired differences on the seeds both arms have (positive = worse):

| comparison | n | paired mean | per seed |
|---|---|---|---|
| O5a-lin − O3.3b | 2 | **+1.27** | +0.09, +2.45 |
| O5c-mlp − O3.3b | 2 | **+0.80** | +0.70, +0.89 |
| O5a-lin − O5c-mlp | 2 | +0.47 | −0.62, +1.57 |
| O5a-lin − W6 | 1 | +2.45 | +2.45 |

**Both crystal-field arms are currently INFERIOR to their own parent.** With n = 2 and a seed
spread of ±1.5 this is not yet a result, and no TOST is meaningful; but it is not a promising
start, and the by-shell table below says where the loss is.

### By shell (means over completed seeds)

| arm | flanking Pb | first-shell Cl | 2–4 Å | 4–8 Å | >8 Å | overall |
|---|---|---|---|---|---|---|
| W6 | **27.0** | **16.5** | **19.5** | **11.0** | **9.6** | **11.6** |
| O3.3b | 33.5 | 18.8 | 23.3 | 13.1 | 10.8 | 13.6 |
| O5b | 37.2 | 19.3 | 25.0 | 13.8 | 11.6 | 14.5 |
| O5a-lin | 39.2 | 19.8 | 26.1 | 14.3 | 12.0 | 15.1 |
| O5c-mlp | 35.9 | 19.3 | 24.5 | 14.0 | 12.0 | 14.6 |

**The crystal field has made the flanking Pb worse, not better** — 39.2 and 35.9 against O3.3b's
33.5 and W6's 27.0. That is the site the whole arm was aimed at. The deficit is not confined to
the near field either: every shell out to >8 Å is worse, so this reads as a uniformly weaker
model rather than a near-field fix that failed.

### Energies (G2a / G2b) — the one place the O5 arms win

| arm | 79-atom RMSE (meV/atom) | between-size ‖Δc_Q‖ |
|---|---|---|
| W6 s0 | 0.475 | 3.5 meV |
| O3.3b s0 | 0.353 | 4.1 meV |
| O5a-lin s0 / s1 | **0.324** / 0.428 | 0.5 / 4.7 meV |
| O5c-mlp s0 / s1 | 0.327 / 0.341 | 4.2 / 1.2 meV |

G2a passes comfortably (ceiling 0.10 meV/atom *inferiority*, and these are 30 % **better** than
W6); G2b passes (≤ 10 meV). Single-valuedness: no arm failed. **The energy/force decoupling
reported for O3.3 persists in O5**: these heads get the energy right and the forces wrong.

### Saturation, and the unbounded channel

| model | on-site saturated fraction | on-site shift p95 | `cf_shift` p95 (unbounded) | `V` p95 |
|---|---|---|---|---|
| W6 s0 | 0.000 | **0.483 eV** | — | — |
| O3.3b s0 | 0.000 | 1.211 eV | — | — |
| O5b s0 | 0.000 | 1.687 eV | — | — |
| O5a-lin s0 | 0.000 | 2.017 eV | **1.004 eV** | 0.863 |
| O5a-lin s1 | 0.000 | 1.622 eV | 0.237 eV | 0.575 |
| O5c-mlp s0 / s1 | 0.000 / 0.001 | 1.723 / 1.973 eV | — | — |

Nothing is at the A1 bound. But note the trend in the on-site shift: **W6 needs 0.48 eV; the
decoupled lineage needs 1.2 → 1.7 → 2.0 eV, and O5a adds a further 1.0 eV on top, unbounded.**
Seed 0 carries a 3.0 eV total on-site correction against W6's 0.48 eV — six times as much — to
reach a *worse* answer. Whatever the decoupled head is missing, it is compensating for it through
the on-site term, and the crystal field has given it one more way to do that rather than removing
the need.

---

## 4. Architecture audit (requested after the above)

Asked: has anything changed beyond registered scope between W6 and O5a? **Answer: no.** The
energy expression is gated on `scf_free`, not `decoupled` — both sum
`E_base + ΔF_band(H0) + E_host + E_M + C_Q`, same code path. Read from the runs' own logs:
`eta 1.0986`, `beta_a 0.00`, `beta_b 0.00`, `Δ_Z 3.079 eV`, L2 1.8e−6, lr 2e−3 cosine → 1e−4,
knee 2.0, clip 10, 60 epochs, batch 4, all loss weights, folds and seeds — **identical**. The one
loss difference, `sat_weight` 1e−3 → 1.0, is the registered O3.3b change.

Three things the audit surfaced that you should see anyway.

### 4.1 The largest single difference is `sk.hop`, and it is 64 % of W6

A `requires_grad` count is not a capacity count — modules never evaluated still report weights.
Verified by backward (listed parameters receive exactly zero gradient):

| model | total `h0` | **live** | structurally dead | scalar block |
|---|---:|---:|---:|---:|
| W6 | 177,132 | **110,314** | 66,818 | base block-0 invariants |
| O3.1 | 17,642 | 6,244 | 11,398 | 18 |
| O3.3b | 79,402 | 18,596 | 60,806 | 211 |
| O5b | 447,612 | **92,406** | 355,206 | **1,361** |
| O5a-lin | 79,414 | 18,608 | 60,806 | 211 |
| O5c-mlp | 81,266 | 19,692 | 61,574 | 211 |

`f_read`/`g_read` are unreachable wherever `beta_a = beta_b = 0` (every arm); `sk.hop` is
unreachable wherever `hop_modulation = False` (every arm from O3 on — **live in W6**).

**71,044 of W6's 110,314 live parameters — 64 % of the shipped model — is the hopping-modulation
MLP that made `t_ij` depend on both atoms' base features.** O3 deleted it by design. O5a is a
sixth of W6's live size. The programme's premise is exactly whether physics can replace that MLP,
and so far it has not: screening (O3.2) did not, correlation-2 (O3.3) did not, the crystal field
does not.

### 4.2 I mis-registered O5b. It is a capacity arm, not a resolution control.

Registered as "O3.3b with `N_r = 16` against the standing 6. **One flag.**" With correlation-2 on,
`N_r` enters the scalar block **quadratically** — the `l = 0` Gram is `n0(n0+1)/2`,
`n0 = N_r × n_species`:

* `N_r = 6` → `n0 = 18`, block `18 + 171 + 21 + 1 = 211`, **18,596 live parameters**
* `N_r = 16` → `n0 = 48`, block `48 + 1176 + 136 + 1 = 1,361`, **92,406 live parameters**

A 6.4× widening and a 5× capacity increase — within 20 % of W6's live count. I did not notice when
writing the registration. **The reading "O5b closes it and O5a does not → the basis was
under-resolved" is void**; it is marked so in the tracker. The radial-resolution question is open.

### 4.3 Every input channel's REACH changed — and with it its RESOLUTION

Cutoffs read off the checkpoints. The base is `r_max` **6.0 Å** with two interaction blocks, and
the head reads `first_block` **only** — so W6's inputs have a **one-hop 6.0 Å** receptive field.

| H0 input | W6 | O3 → O5b | O5a / O5c |
|---|---|---|---|
| `l = 0` on-site | base block 0, **6.0 Å**, learned many-body | descriptor, **10 Å**, two-body | same |
| `l = 1` rank-1 | base `l = 1`, **6.0 Å** | descriptor, **10 Å** | crystal field `E`, 10 Å |
| `l = 2` rank-2 | fixed quadrupole, **4.5 Å** | descriptor, **10 Å**, learned `c2` | `∇E`, 10 Å |

In scope — it follows from the registered "one cutoff radius, one smoothness class" decision. But
that argument is uniformity, and the tracker separately records a *physics* argument for 4.5 Å on
`l = 2`: "reaches the first coordination shell and the Cs neighbours (Pb–Cl 2.8, Cs–Cl 3.5–4.1 Å)
and **not the second Pb shell**".

**The operative variable may be resolution, not reach.** `N_r` Bessels over `[0, r_cut]` resolve
`r_cut/N_r`: six over 10 Å is **1.67 Å**, over 6.0 Å is 1.00 Å, over 4.5 Å is 0.75 Å — against a
first shell at 2.8 Å. The decoupled head resolves the near field about **twice as coarsely** as
the window W6's inputs covered, and the by-shell table says the near field is exactly where it
loses. O5b is evidence this bites: a train-versus-held probe on the saved checkpoints shows it
fits the **training** set markedly better than O3.3b (12.1 against 13.7) — what buying back
resolution looks like — but it paid 5× the parameters and carries the only real generalisation
gap in the programme:

| model | ep | train | held | gap |
|---|---:|---:|---:|---:|
| W6 s0 | 60 | 11.1 | 11.2 | +0.1 |
| O3.1 s1 | 60 | 13.9 | 14.2 | +0.4 |
| O3.3b s0 | 60 | 13.7 | 13.6 | −0.1 |
| **O5b s0** | 60 | **12.1** | **13.2** | **+1.1** |
| O5a-lin s0 | 60 | 13.7 | 13.8 | +0.1 |
| O5c-mlp s0 | 60 | 14.7 | 14.6 | −0.1 |

Every arm but O5b sits at gap ≈ 0 — **W6 included, at a lower absolute error with six times the
live parameters.** So the decoupled deficit is neither overfitting nor a parameter shortfall.

---

## 5. What I would put to you

1. **The crystal field is not working as hoped.** Two seeds each, both arms inferior to their own
   parent, and worse at the flanking Pb — the site it was aimed at. Not yet a result; the
   remaining eight runs finish in ~2 hours and the six-seed TOSTs will settle it.
2. **The linear/MLP question is unresolved and may not matter.** O5a −O5c is +0.47 over two seeds
   with the sign flipping (−0.62, +1.57). Note the asymmetry when you read it: O5a's shift is
   unbounded and unregularised, O5c's is inside `Δ_Z tanh` and pays the L2 and barrier — the
   linear arm has *more* headroom in that channel, not less.
3. **The untried control is a smaller cutoff at fixed `N_r`** — matched to W6 that is 6.0 Å for
   `l = 0,1` and 4.5 Å for `l = 2`. It buys the resolution O5b bought, at no parameter cost, and
   it is the opposite lever from the one I mistakenly registered. `GeometricDescriptor` carries
   one `r_cut` across all `l`, so per-`l` cutoffs are an implementation and a registration, not a
   rerun. **Not built — your call.**
4. **The `sk.hop` question is the real one.** Two thirds of the shipped model is a learned
   environment dependence on the hoppings, and three successive physics substitutes have failed to
   replace it. That is worth a decision in its own right, separate from O5.

Outstanding from your gate list: the **`q0` disagreement with W6 by shell** is not yet computed
for O5 (it wants all six seeds). Analytic-force FD gate: **not applicable** for O5 and now
enforced as a refusal rather than silently taken — §1.3.

---

# ADDENDUM (2026-09-15) — the two measurements, and the decision they force

The O5 wave was **killed at 4 of 12 runs** on Alex's explicit instruction — a stated deviation
from "let the six-seed runs finish for the record". Two seeds of each arm completed 60 epochs and
stand as the record; the §3 tables above are final for what exists.

## M1 — the hopping factor is NOT radial. Decisively.

First, a structural point that sharpens the question. `pre` is built from `feats_i + feats_j`,
`|feats_i − feats_j|` and the species embeddings — **none of them a function of `r`**, by the
code's own docstring, and two periodic images of a node pair share one `pre`. So M1 does not ask
whether the MLP was handed the distance; it asks whether it *learned* to encode it. A high `R²`
would have been the interesting result.

60 training frames, 547,730 edges, `η = 1.0986` (factor bounded to [0.333, 3.0]). `R²` is the
**ceiling for any function of `r`** — each edge predicted by its `r`-bin mean, not by a chosen
parametric family:

| pair | factor sd | observed range | `R²(r)` all | `R²(r)` r < 4 Å |
|---|---|---|---|---|
| Cl–Cl | 0.065–0.103 | 0.52 – 1.83 | 0.006–0.010 | 0.015–0.049 |
| Cl–Cs | 0.084–0.218 | 0.39 – 2.11 | 0.003–0.019 | 0.009–0.028 |
| Cl–Pb | 0.072–0.458 | 0.36 – 2.89 | 0.004–0.052 | 0.025–**0.122** |
| Cs–Cs | 0.094–0.253 | 0.42 – 2.33 | 0.009–0.013 | no bonded edges |
| Cs–Pb | 0.123–0.482 | 0.35 – 2.91 | 0.003–0.017 | no bonded edges |
| Pb–Pb | 0.058–0.585 | 0.36 – 2.95 | 0.018–0.026 | no bonded edges |

**The modulation does a great deal and none of it is radial.** It spans nearly the full bound —
0.36 to 2.95, a factor of eight — and rides its stops, yet distance explains **0.3–5 %** of its
variance overall and **at most 12 %** in the bonded shell.

## M2 — the on-site term *is* substantially radial, but it is the wrong term

kNN regression on a held-out half, plus a linear fit:

| species | shell | p95 \|correction\| | `R²` kNN | `R²` linear |
|---|---|---|---|---|
| Cl | s / p | 0.144 / 0.445 eV | 0.64 / 0.72 | 0.58 / 0.76 |
| Cs | s / p | 0.232 / 0.475 eV | 0.48 / 0.39 | 0.51 / 0.47 |
| Pb | s / p | 0.197 / 0.550 eV | 0.70 / 0.77 | 0.64 / 0.61 |

Half to three quarters explained — but this is the *small* term (p95 0.14–0.55 eV), it is not what
O3 deleted, and it is the part the geometric heads already reproduce.

## The decision, by your rule

> "`R²` high → O6. `R²` low, or O6 stalls → block-0 copy with cached Jacobians is the model, the
> geometric head is closed, and the record states the finding."

**`R²` is low, so O6 is not indicated and I have not built it.** Its premise — that
`t_Harrison(r)[1 + tanh(Σ c_n B_n(r))]` with ~300 parameters replaces the 71k-parameter MLP — is
refuted by M1: the function being replaced is not a function of `r`. Six seeds would confirm a
negative two regressions already give. **The geometric head is closed**, and the record states
the finding you drafted, with the mechanism now named: *the pretrained block-0 representation
carries ~2 meV/Å of force accuracy (W6 11.6 against the best decoupled arm's 13.6) that geometric
tight-binding parameterisations at this data scale do not recover, and the carrier is a genuinely
non-radial, environment-dependent modulation of the hoppings.*

## Two corrections to the guaranteed route

**Accuracy: it already exists, and needs less work than the plan assumes.** In training with
`cache_base` on, `forward` takes the `cached_base` branch and evaluates **`first_block` alone**
from the frozen base modules, with `E_base`/`F_base` from the cache — the full base is never run.
That *is* "the head carries a frozen copy of block 0, W6's weights load unchanged, bit-for-bit
W6". It is the shipped model at **11.6 meV/Å, four seeds, today**. What is missing is **capture**:
`GraphedStep` refuses any non-`decoupled` head because its captured tail is the closed-form
gradient. Extending it means bringing block 0 inside the captured region — real, bounded work.

**Speed: the cached-Jacobian item does not survive its own arithmetic.** Measured on b3, warmed,
medians (the first call is 1.5 s of CUDA warm-up and nearly went into this report as the cost):

| quantity | measured |
|---|---|
| channels the head reads | `n_scalars` 512 + 3 × `n_vectors` 512 = **F = 2048** |
| base graph mean degree within `r_max` | **22.8** |
| one vector-Jacobian product | **11.6 ms** |
| full `∂features/∂R`, forward mode (3N = 237) | **2.7 s/frame** → 0.6 h for 785 frames |
| storage, sparse at degree 22.8 | **89 MB/frame → 70 GB** for the training set |
| block 0, forward + ONE backward | 78.6 ms / 4-frame batch → 15.4 s/epoch |
| **block 0, forward + `create_graph` + the SECOND traversal** | **115.2 ms / batch → 22.6 s/epoch** |

**Block 0's whole training-step cost is 22.6 s/epoch — 18 % of W6's ~124 s.** Measured in
isolation, with nothing else in the timed region: forward, the `create_graph` backward, and the
second traversal the loss makes through the retained graph. (An earlier draft of this line
reported only the first traversal and called it a ceiling. It is a floor, which is why the
isolated measurement was made. The second traversal is 0.5× the first, not the 2–4× that would
have overturned the reading.)

W6 is ~124 s/epoch against O3's ~32 — a gap of 92 s, of which block 0 accounts for **23**. The
rest is the head: W6's `sk.hop` is live and evaluated per unique node pair, 71,044 of its 110,314
live parameters, where O3's is dead. **Removing block 0's traversal entirely cannot take W6 below
~101 s/epoch**, so the cached Jacobian cannot deliver what the plan asks of it — and it costs
70 GB plus per-step I/O to buy at most 23 s. **I recommend not building it.**

**The version that does survive the arithmetic is not a cache.** The force loss never needs `J`,
only `J·r` for the force residual `r` — the parameter gradient of the force term contracts to
`(∂²E/∂θ∂f)·(J r)`, and `J r` is ONE forward-mode JVP with tangent `r ∈ ℝ^{3N}`: ~11.6 ms/frame,
about 9 s/epoch, **exact, no storage**. That is the speed item worth having. Not built.

## Process fixes applied

* **Launch gate added** — `tests/extensions/dscc/test_analytic_composition.py`, 13 gates: for
  every composition of head modules the closed-form gradient **either equals autograd or is
  refused**, with no third option; capture must decline exactly what the closed form declines;
  and every H0-changing flag must be either in the kernel with a covered case or named in
  `analytic_refusal`. This is the gate the 19 meV/Å bug would have tripped.
* Gram recorded as built (O3.3/O3.3b/O5b are 21-channel); O5b's reading marked void in both
  documents; train/held gap and by-shell tables kept as standard readouts.

## What is open for you

1. **Capture for a block-0-reading head** — the one real piece of engineering left on the
   production route, and the only thing between W6 and a deployable fast model.
2. **Whether to spend anything further on speed.** The cache is dead on arithmetic (70 GB to buy
   ≤ 23 s/epoch of 124). The **JVP formulation is not** — exact, no storage, ~9 s/epoch — but it
   only attacks the 23 s, and the other 69 s of the W6/O3 gap is `sk.hop`. My recommendation is to
   ship W6 and close the speed item unless training throughput becomes the binding constraint.
3. **O6 is declined on the evidence, not deferred.** If you want it run anyway as a documented
   negative, say so and it is one day.

---

# ADDENDUM 2 (2026-09-15) — capture for a block-0-reading head: built, and what it found

You called this "the one real piece of engineering left on the production route". It is built.
Building it found three things, and the third is larger than this route.

## 1. The gradient — solved, and it fixes the O5 hole as a side effect

The captured tail *was* the closed-form gradient and nothing else. That is why capture refused
every head the kernel could not differentiate, and why an O5 head replayed 19 meV/Å wrong before
that refusal existed. **Capture does not need a closed form** — it needs a gradient with a fixed
kernel sequence and fixed shapes, and at fixed cotangent `X` that is one ordinary backward.

`h0_position_gradient_vjp` rebuilds the features and the assembly from `positions` with grad on,
contracts against a detached `X`, and takes one backward — no `create_graph`, no parameter
gradients. **A backward is capturable**: verified before anything was built on it, a capture
containing `torch.autograd.grad` replays **bit-identically** to eager (max |diff| exactly 0.0).

`head_tail` now dispatches on `analytic_refusal` — closed form where valid (~2.7 ms), VJP where
not. Results, all replay-vs-eager < 1e-9 on energy, `dq` and forces:

* **W6 captures.** The production route works.
* **O5 captures correctly** instead of being refused — both forms. That hole is closed.
* `analytic_refusal` gained the term it was missing: a **non-decoupled head**. The kernel's own
  first line refuses one; the predicate did not say so, so W6 routed into a function that raised.
* `capture_refusal` is now a separate and **shorter** list: capture turns away a
  **value-dependent shape**, not a missing closed form.

Two value-dependent shapes were in the way and both are gone, both bit-gated: the O1 pair
dedup's `torch.unique` (capture switches to the per-edge path — energy and `dq` bit-identical,
forces within one ULP at 3.5e-17, because the GEMM's row count changes and its reduction is
tiled differently), and `quadrupole_descriptor`'s boolean indexing (masked by multiplication —
the weight is zero beyond `q_cut` by construction, so every excluded edge contributes exactly
0.0 and it is the same sum with a position-independent shape).

## 2. CORRECTION: production capture has regressed — pre-existing, and not this route

**I sent you a paragraph claiming capture "has never worked at production width, for any head".
That was wrong.** The record refutes it: §"O5 TIMING (idle A4000, fake O3 weights, P4.3 protocol,
`dscc_benchmark.py --graphed`)" is a methodical production-width measurement on that card — base
captured 32.4 ms, **model graphed 39.7 ms** at 79 atoms against a ≤ 53 ms deliverable. Capture
worked. I made the claim from one failed run and should not have.

What is true, and established by reverting the four modified modules to HEAD and re-running the
same benchmark on the same checkpoint: **production capture WAS broken, by a fault that predated this session, and both
halves of it are now fixed.** Two faults, in order:

1. **`torch.compile` recompiled inside the captured region** — dynamo/inductor running during
   `g_tail`'s capture is a hard CUDA failure. **Fixed by construction**: capture now turns
   `analytic.COMPILE` off for its own lifetime and restores it on `release()`. It costs nothing,
   because those kernels are compiled for being launch-bound and capture removes launch overhead
   anyway. Capture now builds at production width with no environment variable.
2. **The replay then raised an illegal memory access**, on the FIRST replay of the frame the
   capture was built from. **FIXED.**

   The localisation, since my wrong guesses were the instructive part: `g_assembly` alone
   replays correctly; still correct after `_base`/`_tail` are *warmed up*; breaks the moment a
   **second graph is captured**, either one. Not the shared pool, not the assembly's kernels
   (clean under `PYTORCH_NO_CUDA_MEMORY_CACHING=1`), not a general torch bug (a minimal
   two-graph reproducer is clean on torch 2.13.0+cu129).

   **Cause: the assembly's *outputs* were allocated inside the captured region.** `H` and `ev`
   lived in a graph pool and are consumed by the two graphs captured after it, whose capture can
   reuse that memory. The inputs were already static buffers for exactly this reason; the
   outputs now are too.

   **It reproduces your recorded numbers.** `dscc_benchmark.py --graphed --fixed_cell`,
   production O3 checkpoint, 79 atoms, idle A4000: base captured **32.27 ms** against the
   recorded 32.4, model graphed **39.67 ms** against the recorded **39.7**. Replay agrees with
   eager to 1.15e-06 eV/Å, in line with the recorded production tolerance.

Both reproduce on the reverted tree, so neither is mine. The environment has moved since the
timing was taken (deprecation warnings for `torch.jit.load` and `torch.jit.script_method`,
NumPy 2.5, and openequivariance reporting its precompiled extension missing and falling back to
JIT) — but that is a *hypothesis*; what is measured is only that the fault predates this session.
**Registered as a live regression against the recorded 39.7 ms**, to be bisected against the
torch/openequivariance versions it was taken on.

Crucially this is **not about the block-0-reading route**: the fault is in the decoupled head's
own captured assembly, the path that has always been captured. The VJP work in §1 stands on its
own gates, which run on the in-process bases and pass.

## 3. What this means for the production decision

Nothing about accuracy changes: W6 is the model at 11.6 meV/Å, and it is trained. The speed story
is mixed. The Jacobian cache is dead on arithmetic (70 GB to buy ≤ 23 s/epoch of 124). Capture
is now *available by design* to the block-0-reading head, which it was not before — but the
production capture path is currently broken for every head by a regression that predates this
session and needs bisecting against the environment the 39.7 ms was measured on. My
recommendation stands: **ship W6**, treat the capture regression as its own bounded bug (the
recorded 39.7 ms says the target is reachable), and leave the Jacobian cache closed.

---

# ADDENDUM 3 — what the capture fix does to W6's speed: nothing

Measured after the regression was fixed, both arms **alternated in one loop** (this card drifts
~15 % between separate runs, so two benchmark invocations cannot resolve a 5 % difference — the
first pair of numbers I took this way were 96.0 against 100.8 and meant nothing):

| model | captured | eager | speedup |
|---|---|---|---|
| **O3 (decoupled)** | **40.3 ms** | 76.2 ms | **1.89×** |
| **W6 (block-0-reading)** | **97.0 ms** | 92.9 ms | **0.96×** |

**Capture is now available to W6, and it is not worth using.** It costs slightly more than it
saves. Two reasons, both consequences of how the VJP tail is built:

1. **The captured step assembles twice.** The decoupled head's tail is the closed form (~2.7 ms)
   operating on the already-captured `H`. W6 has no closed form, so `h0_position_gradient_vjp`
   rebuilds the features and the assembly under grad and backs through them — a second full
   assembly plus a block-0 backward, measured at ~22 ms on its own. Eager W6 assembles once and
   backs through it.
2. **Capture forces the per-edge hop MLP.** `torch.unique` is a value-dependent shape, so the O1
   pair deduplication is off inside a capture — and that dedup was 4.95 ms of the assembly's
   7.7 ms. With the assembly run twice, that penalty is paid twice.

**So the production speed picture for W6 is unchanged: ~93 ms eager at 79 atoms**, against 40 ms
for the decoupled head. Nothing in this session made W6 faster, and the plan's expectation of
~50 ms captured for this route is not met — it was based on the geometric head's 39.7 ms, which
comes with a closed-form gradient W6 cannot use.

**The identified improvement, not built and its saving not measured.** The second assembly is
avoidable: capture the assembly WITH grad enabled and retain it, then capture the backward as the
tail, instead of rebuilding the forward inside the tail. The cotangent `X` is only known after
the CPU eigendecomposition between the two graphs, which is why the tail currently rebuilds — but
a retained autograd graph can be backed through in a second captured region without recomputing
the forward. That removes one assembly and one block-0 forward from every step. Whether it brings
W6 below its eager 92.9 ms is a measurement nobody has made, and I am not going to estimate it.
