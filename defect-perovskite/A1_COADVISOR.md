# v5 amendment A1 — the reference-free head is worse, and we can say why

*V_Cl+ orthorhombic CsPbCl3, low-fidelity PBE labels (Mosquera-Lois & Walsh, PRX Energy **4**,
043008 (2025)). Plan: D-SCC charge head v5, amendment A1 ("reference-free environment
dependence"), which removes the pristine species feature means from every bounded correction
in `H0`. W3 = head baseline on base v2, three arms (Φ = 0, LR-only, B′ LR-only), six seeds
{0,1,2,3,4,6}, folds by the W0.3 map, 60 epochs, constant lr 2e-3, epoch-averaged and
last-epoch readings, paired TOST by seed with `tau_phys` = 1.7 meV/Å per component. All force
numbers below are per component (trainer values ÷ √3).*

**Status.** The Φ = 0 arm is complete on the A1 form. It is **worse than the same arm on the
same base with the centred head**, by a seed-median of 4.8 meV/Å, though the paired TOST reads
**inconclusive** at n = 6. The coupled arms are running. We are not asking for a ruling yet;
we are reporting an early negative and a mechanism for it, because the mechanism is testable
and would change what A1 should say.

## 1. We did not run the `η` the amendment specifies

A1's bounds line reads "β = 0.5, η = 0.5". `η` is the bound on the Slater-Koster environment
factor, `t_ij = t^SK(r)·exp(η·tanh m)`. **We ran `η = ln 3 = 1.0986`, not 0.5, because 0.5
diverges from the value already registered for that term.** ln 3 has been the registered bound
since Stage A′ (`×[1/3, 3]`); 0.5 gives `×[0.607, 1.649]`. Nothing else in A1 is a change to an
already-registered number, and A1's own annotation on that line is "(unchanged form, reference
removed)" — the SK term never carried a reference to remove, so on our reading the line did not
need a new value. The amendment's literal 0.5 was run first for six Φ = 0 seeds to epochs 4–16
and then stopped on the user's ruling; those runs are archived at `~/runs/dscc/a1_eta05/`.

**Two measurements about that choice, both of which cut against our own reasoning:**

- Over the epochs both were run, `η = 0.5` and `η = ln 3` are **indistinguishable** on real
  data: identical initialisation per seed (the RNG is seeded before model construction), paired
  mean **−0.07 ± 0.56** meV/Å at epoch 4 (n = 5).
- At the end of training, **nothing is anywhere near the hopping bound**. The largest fitted
  `|tanh m|` over 2.17 M edges of 60 charged frames is **0.172** (A1) and **0.164** (pre-A1),
  i.e. modulation factors of 1.21 and 1.20. A channel would need `|tanh m| > 0.455` before
  `η = 0.5` could not represent it, and **no channel reaches that**. So `η = 0.5` would have
  clipped nothing, and the revert bought no capacity.

The corollary is worth registering on its own: **the Stage A′ finding that motivated ln 3 does
not reproduce on base v2.** That finding was a cohort pinned *at* the hopping stop on the
vacancy-flanking Pb–Pb bond, in every d bin. On base v2 neither the centred nor the
reference-free head gets past a factor of 1.21. Whatever was straining against that bound on
the old base is not straining now.

## 2. Φ = 0 on base v2: A1 against the centred head

Held-out force RMSE per component, six seeds, same folds, same data (785 train / 262 held
charged frames, 639 pristine).

| seed | A1 last | A1 avg | pre-A1 last | pre-A1 avg | d(last) | d(avg) |
|---|---|---|---|---|---|---|
| 0 | 23.06 | 23.94 | 17.99 | 18.87 | +5.07 | +5.06 |
| 1 | 27.28 | 29.32 | 16.47 | 19.74 | +10.80 | +9.58 |
| 2 | 24.06 | 25.28 | 19.60 | 20.20 | +4.47 | +5.08 |
| 3 | 24.42 | 25.90 | 15.02 | 15.68 | +9.40 | +10.22 |
| 4 | 22.83 | 22.64 | 25.24 | 27.08 | −2.41 | −4.44 |
| 6 | 17.07 | 18.79 | 22.17 | 22.99 | −5.10 | −4.20 |
| **median** | **23.56** | **24.61** | **18.79** | **19.97** | | |

Paired TOST, last epoch: mean d **+3.70**, 90 % CI **[−1.50, +8.91]**, τ = 1.7 →
**inconclusive**. Epoch-averaged: +3.55, [−1.77, +8.88] → **inconclusive**. Four seeds worse by
4.5–10.8, two better by 2.4–5.1; SE 2.58 swamps the effect. This is the same n = 6 noise floor
the programme has hit at every comparison.

Cross-base (forces only, per C13), against the old-base re-evaluation on the same folds:
A1 Φ = 0 **24.61** vs old base **23.03**, mean d **−0.17**, inconclusive. The centred head on
base v2 had beaten the old base by **+3.53 [+0.66, +6.40], superior**. So the practical
statement is: **the base-v2 gain is not visible in the reference-free head.**

Seed medians by shell (meV/Å per component):

| | overall | 2–4 | 4–8 | 8–10 | 10–12 | >12 | flanking Pb | first-shell Cl | 2–4 other |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 23.56 | 41.99 | 24.39 | 15.49 | 11.27 | 7.62 | 62.78 | 34.13 | 26.66 |
| pre-A1 | 18.79 | 34.29 | 17.94 | 14.44 | 10.09 | 6.86 | 49.59 | 27.81 | 19.20 |

A1 is worse **everywhere**, near field and far field alike, and worst in relative terms in the
4–8 Å shell (+36 %) rather than at the flanking Pb (+27 %). This is not a localised failure at
the defect; it is a uniformly weaker head.

## 3. The mechanism: the reference-free corrections barely turn on

A1's registered saturation report, read on the finals, is the first clue: the on-site `|tanh|`
exceeds 0.95 on **zero** atoms in four of six seeds, and the hopping bound is untouched. We
then measured the correction magnitudes directly (seed 0, 60 charged frames), computing the
pre-A1 formula explicitly rather than through the runtime, which no longer has a centre:

| on-site level shift, eV | median | p95 | max |
|---|---|---|---|
| pre-A1, `3.0 · tanh[h(x_i) − h(x̄_Z)]` | 0.202 | 1.848 | **2.987** |
| A1, `Δ_Z · tanh[e_Z(h_i)]`, Δ_Z = 3.079 | 0.031 | 0.105 | **0.105** |

**The centred head uses its full ±3 eV range — its largest correction sits at `|tanh| = 0.996`,
against the stop. The reference-free head never exceeds 0.105 eV, about 3 % of the same range.**
Six times smaller at the median, eighteen times at p95. The head is not being regularised into
smallness: A1's L2 ran four to five orders below the force loss all through training
(1.5e−8 to 3.7e−7 against 1.3–1.8e−5), so its gradient is negligible.

**Our hypothesis, offered as a hypothesis.** The two forms differ in what the tanh sees. The
centred form feeds it a *deviation*, which is zero-mean across the ensemble by construction, so
the readout's weights can grow to whatever scale the data wants without the common mode
saturating anything. The reference-free form feeds it the raw feature, whose species-dependent
mean is large; growing the weights drives the tanh into saturation on that mean before it
resolves the deviation, so the optimiser keeps the weights small and the correction stays near
zero. A per-species bias inside the readout *can* in principle cancel the mean — that is A1's
own argument for dropping the reference — but nothing in the loss pushes it to do so quickly,
and 60 epochs at constant lr evidently is not enough. **Note this is precisely the failure the
retired "output" centring form suffered** (recorded in `legacy.py`: centring the output left
`h` free to drift until every atom of a species was past `|tanh| = 0.98` and the channel was
dead). A1 removes the subtraction altogether; the symptom is the mirror image — not a dead
saturated channel but a dead unsaturated one.

If the hypothesis is right, the fix is inside A1's own terms and needs no reference: normalise
the readout input, or initialise each species' readout bias at `−e_Z(mean feature of Z)` over
the *training ensemble* (not a pristine cell — a thermal ensemble statistic, which A1 permits),
or simply give the readouts a longer/warmer schedule. All three are testable on one seed.

## 4. What is unaffected, and what is better

- **The gap regulariser holds without the pristine centre**: 2.399–2.402 against the registered
  2.40 on every seed. This was the term most exposed to A1's change.
- **A1 is less volatile.** Largest epoch-to-epoch jump in held RMSE, per seed: A1 2.55 / 6.05 /
  2.28 / 11.43 / 2.18 / 4.33 against pre-A1 7.16 / 16.90 / 2.71 / 5.59 / 6.60 / 8.66 — smoother
  on five of six. The epoch-averaging penalty also shrank (avg−last +0.87…+2.05 against
  +0.60…+3.27), and seed 4 is the first seed in the campaign where averaging *helps*. Both
  forms are still descending at epoch 59, which remains an open item.
- **Cost is unchanged**: warmed-up epochs 121.2 s (A1) against 121.7 s (pre-A1), same packing.
  The §8 2× benchmark on base v2 + A1 head reads median 1.94 / p95 2.21 at 79 atoms (100
  frames) and 2.11 / 8.93 at 159 atoms (16 frames — all that exist), so the criterion still
  fails on p95. The 79-atom median improving on the old base's 2.07 is the denominator getting
  dearer (512-wide, r_max 6.0), not the head getting cheaper.

## 5. What we ask

Nothing yet on A1 itself — the coupled arms are running and a form change could land
differently under coupling. Two things we would like read now:

1. **Whether the correction collapse is accepted as the explanation**, and if so whether the
   remedy should be registered inside A1 (ensemble-statistic bias initialisation, input
   normalisation, or a schedule change) or whether A1's premise — that a learned bias inside
   the bound replaces the reference — should be revisited. We prefer the first and would
   register the bias initialisation before touching any further result: it is reference-free in
   A1's sense, since a thermal-ensemble mean is not a pristine-cell statistic.
2. **Whether `η` should now go back to 0.5.** Our reason for ln 3 was registration continuity,
   and the measurement says the choice is inert: nothing reaches even a third of the way to the
   narrower bound. If the amendment's value is wanted, it costs one flag and one re-run, and on
   this evidence it should change no number.

Also outstanding and unchanged: **C5 on base v2 has never been read** — it is W3's gate and
W4's entry gate, and it is a diagnostic on trained heads, so it can be run on this arm now.

Artefacts: `~/runs/dscc/w3a1_report.json`, `~/runs/dscc/dscc_w3a1_phi0_s*/held_final{,_avg}.json`
(with A1's `saturation` block), `~/runs/dscc/a1_eta05/` (the superseded η = 0.5 runs),
`~/runs/dscc/bench_a1_{79,159}.json`; tracker `defect-perovskite/DSCC_V5_IMPLEMENTATION_SPEC.md`;
tag `pre-a1` for the centred form. Commits `3a98e05` `a879576` `e1b5709` `88bfccd` `19c2c52`
`6a4115b` `9b3aaa5`.
