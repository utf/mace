# Report: everything since the ruling — 2026-09-15

Covers the work from the ruling ("kill the jobs, two measurements, O6 if radial, block-0 copy in
parallel, process fixes") to now. Forces are **per-component meV/Å** throughout. τ = 1.7.

**Headline:** the ruling's decision rule fired on the low branch — O6 is declined on the
evidence and the geometric head is closed. The production route is W6 as it stands. Separately,
production CUDA-graph capture turned out to be broken by a pre-existing regression; it is fixed
and reproduces the recorded 39.7 ms. It does not make W6 faster.

---

## 1. The O5 wave was killed at 4 of 12 (your deviation, honoured)

`dscc_o5lin_s0/s1` and `dscc_o5mlp_s0/s1` completed 60 epochs and stand as the record. The four
killed mid-run are renamed `KILLED_*_ep{n}` so no future glob reads a partial as a result. Both
cards idle, queues moved to `*_KILLED.txt`. Final O5 numbers, for completeness:

| arm | s0 | s1 | mean | vs O3.3b (paired) |
|---|---|---|---|---|
| O5a linear | 13.7 | 16.5 | 15.1 | **+1.27** |
| O5c MLP | 14.3 | 15.0 | 14.6 | **+0.80** |
| O3.3b (parent, 6 seeds) | | | 13.6 | — |
| W6 (4 seeds) | | | 11.6 | — |

Both crystal-field arms were inferior to their own parent, and **worse at the flanking Pb** —
39.2 and 35.9 against O3.3b's 33.5 and W6's 27.0 — the site the arm was aimed at. n = 2, so not
a result; but the measurements below made it moot.

---

## 2. M1 — the hopping modulation is NOT radial. Decisively.

A structural point sharpens the question first: `pre` is built from `feats_i + feats_j`,
`|feats_i − feats_j|` and the species embeddings — **none of them a function of `r`**, by the
code's own docstring — and two periodic images of a node pair share one `pre`. So M1 does not
ask whether the MLP was handed the distance; it asks whether it **learned** to encode it.

60 training frames, 547,730 edges, `η = 1.0986` (factor bounded to [0.333, 3.0]). `R²` is the
**ceiling for any function of `r`**: each edge predicted by its `r`-bin mean, not by a chosen
parametric family.

| pair | factor sd | observed range | `R²(r)` all | `R²(r)` r < 4 Å |
|---|---|---|---|---|
| Cl–Cl | 0.065–0.103 | 0.52 – 1.83 | 0.006–0.010 | 0.015–0.049 |
| Cl–Cs | 0.084–0.218 | 0.39 – 2.11 | 0.003–0.019 | 0.009–0.028 |
| Cl–Pb | 0.072–0.458 | 0.36 – 2.89 | 0.004–0.052 | 0.025–**0.122** |
| Cs–Cs | 0.094–0.253 | 0.42 – 2.33 | 0.009–0.013 | no bonded edges |
| Cs–Pb | 0.123–0.482 | 0.35 – 2.91 | 0.003–0.017 | no bonded edges |
| Pb–Pb | 0.058–0.585 | 0.36 – 2.95 | 0.018–0.026 | no bonded edges |

**The modulation does a great deal and none of it is radial.** It spans nearly its whole bound —
0.36 to 2.95, a factor of eight — and rides its stops, yet distance explains **0.3–5 %** of its
variance overall and **at most 12 %** in the bonded shell.

## 3. M2 — the on-site term *is* radial, but it is the wrong term

kNN regression on a held-out half (prediction, not memorisation), plus a linear fit, against each
atom's 12 sorted neighbour distances:

| species | shell | p95 \|correction\| | `R²` kNN | `R²` linear |
|---|---|---|---|---|
| Cl | s / p | 0.144 / 0.445 eV | 0.64 / 0.72 | 0.58 / 0.76 |
| Cs | s / p | 0.232 / 0.475 eV | 0.48 / 0.39 | 0.51 / 0.47 |
| Pb | s / p | 0.197 / 0.550 eV | 0.70 / 0.77 | 0.64 / 0.61 |

Half to three quarters explained — but this is the *small* term (p95 0.14–0.55 eV), it is not
what O3 deleted, and it is the part the geometric heads already reproduce.

## 4. The decision, by your rule

> "`R²` high → O6. `R²` low, or O6 stalls → block-0 copy is the model, the geometric head is
> closed, and the record states the finding."

**`R²` is low. O6 is declined on the evidence, not deferred, and is not built.** Its premise —
`t_Harrison(r)[1 + tanh(Σ c_n B_n(r))]`, ~300 parameters replacing the 71k-parameter MLP — is
refuted by M1: the function being replaced is not a function of `r`. Six seeds would confirm a
negative two regressions already give.

**The geometric head is closed.** The finding, with the mechanism now named: *the pretrained
block-0 representation carries ~2 meV/Å of force accuracy (W6 11.6 against the best decoupled
arm's 13.6) that geometric tight-binding parameterisations at this data scale do not recover, and
the carrier is a genuinely non-radial, environment-dependent modulation of the hoppings.*

**One thing the rule does not cover**, flagged as outside it rather than as a proposal: M1 rules
out a function of `r_ij` *alone*. It does not rule out a learned function of both atoms'
**geometric descriptors** on the bond — `f(A0_i + A0_j, |A0_i − A0_j|, Z_i, Z_j)`, i.e. W6's
`sk.hop` with block 0 swapped for the head's own descriptor. No arm has run it.

---

## 5. The guaranteed route: accuracy needs no work, speed needs more than assumed

**Accuracy already exists.** In training with `cache_base` on, `forward` takes the `cached_base`
branch and evaluates **`first_block` alone** from the frozen base modules, `E_base`/`F_base` from
the cache — the full base is never run. That *is* "the head carries a frozen copy of block 0,
W6's weights load unchanged, bit-for-bit W6". It is the shipped model at **11.6 meV/Å**, four
seeds, today.

**The cached Jacobian does not survive its own arithmetic.** Measured on b3, warmed, medians:

| quantity | measured |
|---|---|
| channels the head reads | `n_scalars` 512 + 3 × `n_vectors` 512 = **F = 2048** |
| base graph mean degree within `r_max` | 22.8 |
| one vector-Jacobian product | 11.6 ms |
| full `∂features/∂R`, forward mode (3N = 237) | 2.7 s/frame → 0.6 h for 785 frames |
| storage, sparse at degree 22.8 | **89 MB/frame → 70 GB** |
| **block 0's whole training-step cost** (fwd + `create_graph` + the second traversal) | **22.6 s/epoch** |

Block 0 is **18 %** of W6's ~124 s/epoch, and explains only 23 of the 92 s gap to O3's 32 —
the rest is `sk.hop`. So the cache buys ≤ 23 s for 70 GB. **Recommend not building it.**

**The version that does survive is not a cache.** The force loss never needs `J`, only `J·r` for
the force residual: the parameter gradient contracts to `(∂²E/∂θ∂f)·(J r)`, and `J r` is one
forward-mode JVP with tangent `r ∈ ℝ^{3N}` — ~11.6 ms/frame, about 9 s/epoch, **exact, no
storage**. Registered, not built.

---

## 6. Capture: built for a block-0-reading head, and a pre-existing regression fixed

**The design blocker is removed.** The captured tail *was* the closed-form gradient and nothing
else — which is why capture refused every head the kernel could not differentiate, and why an O5
head replayed **19 meV/Å wrong** before that refusal existed. Capture does not need a closed
form: at fixed cotangent `X` the gradient is one ordinary backward, and **a backward is
capturable** (verified first: a capture containing `torch.autograd.grad` replays bit-identically
to eager). `head_tail` now dispatches on `analytic_refusal` — closed form where valid,
`h0_position_gradient_vjp` where not. Consequences, all replay-vs-eager < 1e-9 on the toy bases:
W6 captures; **O5 captures correctly** instead of being refused; `analytic_refusal` gained the
term it was missing (a non-decoupled head); `capture_refusal` is now its own, shorter list.

Two value-dependent shapes were in the way, both removed and bit-gated: O1's `torch.unique` pair
dedup (per-edge inside a capture — energy and `dq` bit-identical, forces one ULP at 3.5e-17) and
`quadrupole_descriptor`'s boolean indexing (masked by multiplication — the weight is zero beyond
`q_cut`, so it is the same sum with a position-independent shape).

**Then production capture turned out to be broken, and it was not mine.** Established by saving a
patch, reverting the four modified modules to HEAD, and re-running the same benchmark on the same
checkpoint: it failed identically. Two faults:

1. **`torch.compile` recompiling inside the captured region** — dynamo/inductor during `g_tail`'s
   capture is a hard CUDA failure. **Fixed by construction**: capture disables `analytic.COMPILE`
   for its own lifetime and restores it on `release()`. It costs nothing — those kernels are
   compiled *because* they are launch-bound, and capture removes launch overhead anyway.
2. **An illegal memory access on the first replay.** **Fixed.** Cause: the assembly's *outputs*
   were allocated inside the captured region, so `H` and `ev` lived in a graph pool — and the two
   graphs captured after it consume them, so their capture could reuse that memory underneath.
   The inputs were already static buffers for exactly this reason; the outputs now are too.

**It reproduces the recorded measurement.** `dscc_benchmark.py --graphed --fixed_cell`, production
O3 checkpoint, 79 atoms, idle A4000:

| | recorded (P4.3) | now |
|---|---|---|
| base captured | 32.4 ms | **32.27 ms** |
| model graphed | **39.7 ms** | **39.67 ms** |

Four hypotheses were wrong before the right one, and all are written into the record so they are
not retried: the shared memory pool (one pool per graph changes nothing), the assembly's own
kernels (clean eagerly under `PYTORCH_NO_CUDA_MEMORY_CACHING=1`), openequivariance's JIT fallback
(capturing `g_tail`, which has no base kernels, breaks it too), and a general torch regression (a
minimal two-graph reproducer is clean on torch 2.13.0+cu129). What located it was bisecting
`_build` with explicit syncs: correct after its own capture, correct after `_base`/`_tail` are
*warmed up*, broken the moment a second graph is *captured*.

## 7. What capture does to W6's speed: nothing

Both arms **alternated in one loop** (the card drifts ~15 % between separate runs, so two
invocations cannot resolve a 5 % difference — my first pair of numbers did not):

| model | captured | eager | speedup |
|---|---|---|---|
| **O3 (decoupled)** | **40.3 ms** | 76.2 ms | **1.89×** |
| **W6 (block-0-reading)** | **97.0 ms** | 92.9 ms | **0.96×** |

**Capture is now available to W6 and is not worth using on it.** Two reasons: the captured step
**assembles twice** (the closed form works on the already-captured `H`; the VJP rebuilds features
and assembly under grad, ~22 ms), and capture **forces the per-edge hop MLP**, whose dedup was
4.95 ms of the assembly's 7.7 — paid twice. **W6's production speed is unchanged at ~93 ms
eager**, against 40 ms for the decoupled head.

The plan's ~50 ms expectation for this route is not met; it was extrapolated from the geometric
head's 39.7 ms, which comes with a closed-form gradient W6 cannot use. (The same line treats
63.6 ms as "coupled W6"; in the tracker's table 63.6 is the *eager* number for the same decoupled
model whose graphed number is 39.7. W6 eager is ~93.)

**The avoidable half, not built and not estimated:** capture the assembly *with grad* and capture
the backward as the tail, instead of rebuilding the forward inside the tail. That removes one
assembly and one block-0 forward per step.

---

## 8. Supporting measurements that changed the reading

**Train-versus-held probe** (re-evaluating saved checkpoints with the same `evaluate` that
produces the held numbers; `history.json`'s `force` is a stratum-weighted loss term and is not a
substitute):

| model | epoch | train | held | gap |
|---|---:|---:|---:|---:|
| W6 s0 | 60 | 11.1 | 11.2 | +0.1 |
| O3.1 s1 | 60 | 13.9 | 14.2 | +0.4 |
| O3.3b s0 | 60 | 13.7 | 13.6 | −0.1 |
| **O5b s0** | 60 | **12.1** | **13.2** | **+1.1** |
| O5a-lin s0 | 60 | 13.7 | 13.8 | +0.1 |
| O5c-mlp s0 | 60 | 14.7 | 14.6 | −0.1 |

Every arm but O5b sits at gap ≈ 0 — W6 included, at a lower absolute error with six times the
live parameters. The deficit is neither overfitting nor a parameter shortfall.

**Live parameters** (a `requires_grad` count is not a capacity count; `f_read`/`g_read` are
unreachable wherever `beta_a = beta_b = 0`, `sk.hop` wherever `hop_modulation = False` — verified
by backward):

| model | total `h0` | **live** | scalar block |
|---|---:|---:|---:|
| W6 | 177,132 | **110,314** | base block-0 invariants |
| O3.3b | 79,402 | 18,596 | 211 |
| O5b | 447,612 | **92,406** | **1,361** |
| O5a-lin | 79,414 | 18,608 | 211 |

**71,044 of W6's 110,314 live parameters — 64 % of the shipped model — is the hopping-modulation
MLP** that O3 deleted by design. That is the largest single difference between the lineages, and
M1 says it is not replaceable by a radial form.

---

## 9. Corrections of record

* **The first "O5a" runs were not O5a** — they appended the invariants to the 211-wide block and
  fed the existing MLP, against a ruling that said "linear, no MLP", twice. Cancelled; no numbers
  used.
* **O3.3's invariants were the full 21-channel pairwise Gram, not the specified 6 norms.**
  Propagates into O3.3b and O5b.
* **I mis-registered O5b.** "One flag, `N_r` 6 → 16" is quadratic through the correlation-2 Gram:
  block 211 → 1,361, live parameters 18,596 → 92,406. It is a 5× capacity arm, not a resolution
  control; the reading that would have adopted "the basis was under-resolved" is marked void.
* **I claimed capture "has never worked at production width".** Wrong — the record's P4.3 timing
  refutes it, and the fix above reproduces that timing. Retracted in both documents.
* **I claimed O5b fit the training set ~2 % worse** from the weighted loss term. The direct probe
  says it fits **12 % better** (12.1 vs 13.7) and wins seed 0 on held-out too.

---

## 10. Process fixes applied, and what is open

**Applied.** `tests/extensions/dscc/test_analytic_composition.py` (13 gates): for every
composition of head modules the closed-form gradient **either equals autograd or is refused** —
no third option; capture's refusals and the closed form's are separate, honest lists; no
H0-changing flag may exist without being in the kernel or named in `analytic_refusal`. This is the
gate the 19 meV/Å bug would have tripped. Plus: the Gram recorded as built, O5b's reading marked
void, train/held and by-shell tables kept as standard readouts, and `GraphedStep.release()`
restoring the pair-dedup flag (I had written a comment claiming a restore that did not exist).

**Suite: 319 passing.**

**Open, for your decision.**
1. **Ship W6.** 11.6 meV/Å, trained, four seeds. Nothing since the ruling changes that.
2. **Close the speed item**, unless training throughput binds: the cache is dead on arithmetic,
   capture does not help W6, and the JVP formulation attacks only 23 s of the 92 s gap.
3. **O6 is declined on the evidence.** If you want it as a documented negative anyway, one day.
4. **The untried control**, if the geometric head is ever reopened: a *smaller* descriptor cutoff
   at fixed `N_r` (6.0 Å for `l = 0,1`, 4.5 Å for `l = 2`, matching what W6's inputs actually
   covered) — it buys the resolution O5b bought, at no parameter cost, and is the opposite lever
   from the one I mistakenly registered.
