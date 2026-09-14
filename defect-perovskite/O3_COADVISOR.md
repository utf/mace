# The O3 lineage: decoupling the charge head from the base

**Status as of 2026-09-14 (SUPERSEDED — see §8 onward for O3.3, O3.3b and O3.D).** Three arms run, all six seeds each, all gates registered before the
corresponding result was opened. **None is adoptable.** The lineage has, however, answered the
question it was built to ask, and it has ruled out the explanation we expected to find.

Units throughout: **meV/Å per component**. The trainer logs the per-atom force-error VECTOR RMS;
every number here is that divided by √3. Folds {0,1,2,3,0,2} for seeds {0,1,2,3,4,6}, paired
TOST at the registered **τ = 1.7**.

---

## 1. What the lineage was for

W6 — the shipped head — reads the base's block-0 invariants. That coupling is what makes the
head expensive (the base's block 0 is recomputed under `create_graph`, 11.6 GB and 124 s/epoch)
and what stops the head from being reasoned about independently of the base. **O3 asked whether
the head can own its inputs**: a geometric descriptor of its own, nothing from the base.

The prize was large and was measured, not assumed: a decoupled head trains at **32 s/epoch
against W6's 124** on the same card (3.9×; on b3, 29 against 170, 5.9×) and holds 4.7 GB against
11.6. The cost is whatever accuracy the base's features were carrying.

---

## 2. The three arms, and what changed between them

| arm | change from its parent | parent |
|---|---|---|
| **O3** | head-owned descriptor (Bessel × taper, species-resolved `l = 0`, own `c` at `l = 1, 2`); hopping modulation **deleted** | W6 |
| **O3.1** | A1.1 per-species standardisation of the head's own `l = 0` input; readout final layer at the registered 0.1 | O3 |
| **O3.2** | hopping modulation **restored from geometry**: three-body bond screening | O3.1 |

O3 changed two things at once, which was registered as a limitation at the time; O3.1 and O3.2
are one change each.

---

## 3. G1 — the adoption gate

| seed (fold) | W6 | O3 | O3.1 | O3.2 |
|---|---:|---:|---:|---:|
| 0 (0) | 11.22 | 18.82 | 14.59 | 13.78 |
| 1 (1) | 11.62 | 19.71 | 14.22 | 13.94 |
| 2 (2) | 11.58 | 18.55 | 15.06 | 13.85 |
| 3 (3) | 11.23 | 20.48 | 14.82 | 13.89 |
| 4 (0) | 12.02 | 18.53 | 14.28 | 13.67 |
| 6 (2) | 11.84 | 18.79 | 14.17 | 14.04 |
| **median** | **11.60** | **18.81** | **14.44** | **13.87** |

Paired TOST, τ = 1.7, positive `d` favours the second arm:

| comparison | mean d | se | 90 % CI | reading |
|---|---:|---:|---|---|
| W6 − O3 | −7.563 | 0.407 | [−8.384, −6.743] | **INFERIOR**, 4.4 τ |
| W6 − O3.1 | −2.937 | 0.247 | [−3.436, −2.439] | **INFERIOR**, 1.7 τ |
| O3 − O3.1 | +4.626 | 0.338 | [+3.945, +5.307] | **SUPERIOR**, 2.7 τ |
| W6 − O3.2 | −2.278 | 0.144 | [−2.568, −1.988] | **INFERIOR**, 1.3 τ |
| O3.1 − O3.2 | +0.659 | 0.166 | [+0.326, +0.993] | **EQUIVALENT** (better on every seed, whole interval inside the band) |

Every comparison has the same sign on every seed. **O3.2 is the best decoupled head we have and
it is still 2.28 meV/Å short of the shipped one.**

---

## 4. The other registered gates

**G2a, energy RMSE at 79 atoms** (margin 0.10 meV/atom, one-sided 95 %). O3.1 and O3.2 both
**PASS**, and pass the *other* way — they are better than W6:

| | O3.1 | O3.2 |
|---|---|---|
| mean d vs W6 | −0.058 | −0.0739 |
| one-sided 95 % upper | −0.025 | −0.0314 |
| seeds better than W6 | 6 / 6 | 5 / 6 |

**G2b, between-size after one `C_Q`**, read as the registered cohort median: W6 **+5.0**, O3.1
**−1.8**, O3.2 **−0.3** meV, all inside ±10. **PASS.** (Per-seed spread is wide on all three
cohorts — the held split carries 4 frames at 159 atoms against 258 at 79 — so the per-seed
numbers should not be read individually. The W6 median on this estimator reproduces the W6
adoption table's +5.0 exactly, which is what identifies it as the same reading.)

**Turn-on / saturation.** This is where O3 was disqualified independently of G1: **40 % of atoms
sat AT the A1 barrier knee** on the on-site term, in every seed, at a p95 shift of 2.97 eV
against a half-width of 3.079. A parameter at its stop has `sech² ≈ 0` — it looks like it is
learning and it is not.

| | saturated fraction | on-site p95 shift (eV) |
|---|---:|---:|
| W6 | 0.000 | 0.846 |
| O3 | **0.40** | 2.97 |
| O3.1 | 0.000 | 0.740 |
| O3.2 | 0.000 | 0.722 |

**A1.1 standardisation removed the saturation completely and it has not come back.** The
diagnosis was made from the input statistics before O3.1 ran: the raw species-resolved `l = 0`
descriptor has a species-constant component 4.8–8.0× the ensemble deviation in the median
channel and up to 84× in the worst, with **0 dead channels** — the readout was saturating on a
per-species offset and squashing the environment signal behind it.

---

## 5. The finding: where the remaining error is, and what it is not

**O3.1 answered its registered question in the middle.** The registration said: if
standardisation closes most of O3's 7.56, the descriptor is sound and the saturation was the
cost; if it closes little, the head-owned basis or the deleted modulation is carrying the loss.
**It closed 4.63 of 7.56 — 61 %.** So the saturation was the larger half, and the descriptor is
not obviously the problem.

The residual is **not uniform in space**, which is what pointed at the hopping. O3.1 − W6, by
range from the vacancy, cohort medians:

| shell (Å) | 2–4 | 4–6 | 6–8 | 8–10 | 10–12 | flanking Pb | first-shell Cl |
|---|---:|---:|---:|---:|---:|---:|---:|
| O3.1 − W6 | +5.71 | +3.68 | +2.89 | +1.25 | +1.46 | **+10.02** | +4.02 |

The deficit is concentrated on the flanking Pb pair and decays with distance — exactly where a
bond screened by a third atom would be, and the one capability O3 deleted and O3.1 did not
restore. Hence O3.2.

### 5.1 O3.2, registered in full before it ran

`S_ij = (1/6) Σ_k g(r_ik) g(r_jk) P_l(cos θ_ikj)`, the angle **at k**; `g` is H0's own taper at
`r_s3 = 4.5 Å` (the registered `Q_CUT`); Legendre `l = 0, 1, 2`, species-resolved in `k`, so nine
invariants per bond; contracted by a `[n_el, n_el, 9, 4]` **linear** map — no MLP, nothing from
the base — into the head's **existing** bounded modulation `exp(η tanh pre)`, carrying the same
L2 and the same one-sided barrier W6's `hop` pre-activations carried. The map is
zero-initialised and symmetrised in the species pair, so the arm starts at O3.1 bit-for-bit and
`H` stays Hermitian by construction. Both are gated, along with a hand-computed triangle, the
`l = 1` sign trap, exact zero beyond `2 r_s3`, FD forces, batched == per-graph, and the
mixed-size batch that killed O3's first launch. Cost 35 s/epoch against O3.1's 28.

### 5.2 And the pre-registered expectation FAILED

Written before the wave: *"if the three-body screening is the right diagnosis, its gain should
be concentrated in the same places — the flanking Pb and the 2–4 Å shell — and largely absent
beyond 8 Å. If instead O3.2 gains uniformly across shells, or gains at long range, the reading
is wrong whatever G1 says."*

| shell (Å) | W6 | O3.1 | O3.2 | **O3.2 − O3.1** |
|---|---:|---:|---:|---:|
| 2–4 | 19.83 | 25.54 | 24.99 | −0.55 |
| 4–6 | 13.81 | 17.49 | 17.00 | −0.50 |
| 6–8 | 9.73 | 12.62 | 12.09 | −0.53 |
| 8–10 | 9.86 | 11.11 | 10.25 | **−0.86** |
| 10–12 | 9.02 | 10.48 | 9.38 | **−1.10** |
| **flanking Pb** | 27.40 | 37.41 | 35.55 | **−1.86** |
| first-shell Cl | 16.55 | 20.56 | 20.26 | −0.31 |
| rest of 2–4 | 10.26 | 11.85 | 11.18 | −0.68 |

Half matches — the flanking Pb takes the largest single gain, three times a typical shell. The
other half does not: **the gain does not decay with distance.** It is larger at 8–12 Å (−0.86,
−1.10) than at 4–8 Å (−0.57). A screening with a 4.5 Å cutoff has no business improving forces
at 10–12 Å more than at 5 Å.

**And the screening barely engaged.** The diagnostic was registered so that the modulation could
not be reported as "learned" while sitting at its stop; it shows the opposite failure — it is
sitting at its **start**. Cohort medians: saturated fraction 0.000, `|pre|` p95 **0.038**,
`|pre|` max 0.390, modulation factor `exp(η tanh pre)` at p95 **1.001**. The fit moved the
hoppings by about a tenth of a percent over most bonds. The +0.66 meV/Å did not come from a
strong screening, and the shell table says it did not come from a bond-local one either.

**By the criterion registered before the result, the reading that "the residual is the hopping's
missing environment dependence" is withdrawn.**

---

## 6. Where this leaves the lineage

Three arms agree on the same thing. O3 lost 7.56; standardisation recovered 4.63; restoring the
modulation from geometry recovers 0.66 more and reads EQUIVALENT to its own parent. **The
remaining ~2.3 meV/Å is not the hopping modulation, and it is not saturation.**

What is left is the one thing every arm in this lineage shares and W6 does not: **the head-owned
descriptor**. Two readings are consistent with everything above and they are distinguishable:

1. The geometric basis is too small or wrongly shaped (6 Bessel × 3 species at `l = 0`, `c` at
   `l = 1, 2`) to reproduce what the base's 512 block-0 invariants carry.
2. The base's features carry information no purely geometric basis of any size reproduces —
   because they are the output of two message-passing layers over a 12 Å receptive field, while
   the head's descriptor is one shell of two-body terms at 10 Å.

**Proposed next arm, one change, cheapest decisive form:** give the decoupled head the base's
block-0 features back **for the on-site term alone**, leaving the hoppings and the directional
blocks on the head's own descriptor. If the 2.3 closes, reading (1) is wrong and the on-site
term is where the base's information was being used — which also tells us exactly how much of
W6's coupling is load-bearing. If it does not close, the loss is in the hoppings or the
directional block and the next split follows from that. Either way it is one change, it reuses
the existing W6 plumbing, and it is ~35 s/epoch.

I have **not** opened this arm. Registering it before it runs is the programme's rule and the
form above is a proposal, not a registration.

---

## 7. Two things carried as known holes, not hidden

- **The analytic head force does not cover O3.2.** `h0_position_gradient` writes `dH0/dR`
  assuming `t_ij` depends on `r_ij` alone; the screening makes it depend on every third atom.
  A screened head falls back to the autograd cotangent route at inference, gated against finite
  differences. Not on the training path.
- **CUDA-graph capture refuses a screened head**, because the triplet list is built with
  `nonzero` and is therefore data-dependent in shape; a replay would silently reuse the captured
  frame's triplets. The sparse/windowed assembly refuses it too — it evaluates the two-body
  `integrals` and would have dropped the screening without a word.

Both refusals are gated on the module, not on a config flag, because the flag travels in the
extra state and loading an unscreened parent's `state_dict` would otherwise clear it while the
module stayed live — wrong forces, no error.

---

# 8. Update — the coadvisor's plan, executed in full

Everything proposed in the ruling has been run: the `q0` diagnostic, **O3.3** (the head's own
correlation-2 B-basis) with a follow-up **O3.3b**, and **O3.D** (block 0 for the on-site term).
Six seeds each, folds as registered, every threshold written before the corresponding result.

## 8.1 The `q0` diagnostic — the mechanism is confirmed

30 charged 79-atom frames, `q0` recomputed from each model's own `H0`.

| shell (Å) | mean \|Δq0\| (e) | W6 F err | O3.1 F err | O3.1 − W6 |
|---|---:|---:|---:|---:|
| 2–4 | 0.3418 | 17.09 | 26.92 | +9.82 |
| 4–6 | 0.1549 | 12.37 | 17.51 | +5.14 |
| 6–8 | 0.0728 | 9.31 | 13.66 | +4.35 |
| 8–10 | 0.0834 | 8.31 | 10.23 | +1.92 |
| 10–12 | 0.0790 | 8.37 | 10.74 | +2.36 |
| >12 | 0.0496 | 6.64 | 7.01 | +0.37 |

**The two heads disagree about `q0` by 21 %** (mean \|Δq0\| 0.1135 e against a mean \|q0\| of
0.5396 e; up to 0.94 e on single atoms) — on ordinary bulk atoms, not just at the defect. Across
shells, mean \|Δq0\| correlates **+0.939** with the O3.1 − W6 force deficit and **+0.963** with
O3.1's absolute error. The `E_host = dqᵀ Γ_LR (s q0)` route you identified is real.

*Limit, stated because it matters:* that correlation is between SHELL MEANS. Within-shell
per-atom correlation is strong only at 2–6 Å (+0.45, +0.43) and ~0 or negative beyond
(+0.06, +0.09, −0.22). Consistent with a collective `Γ_LR` channel — it supports the mechanism
rather than proving it atom by atom.

## 8.2 Results

| seed (fold) | W6 | O3.1 | O3.2 | O3.3 | O3.3b | O3.D |
|---|---:|---:|---:|---:|---:|---:|
| 0 (0) | 11.22 | 14.59 | 13.78 | 13.20 | 13.58 | 15.74 |
| 1 (1) | 11.62 | 14.22 | 13.94 | 13.91 | 14.08 | 14.65 |
| 2 (2) | 11.58 | 15.06 | 13.85 | **11.99** | 12.80 | 15.44 |
| 3 (3) | 11.23 | 14.82 | 13.89 | 15.64 | 13.07 | 14.16 |
| 4 (0) | 12.02 | 14.28 | 13.67 | 12.24 | 15.20 | 17.18 |
| 6 (2) | 11.84 | 14.17 | 14.04 | 15.31 | 13.06 | 14.06 |
| **median** | **11.60** | **14.44** | **13.87** | **13.55** | **13.32** | **15.04** |
| spread | 0.80 | 0.89 | 0.37 | **3.65** | 2.39 | 3.12 |

| comparison | mean d | 90 % CI | reading |
|---|---:|---|---|
| W6 − O3.3 | −2.127 | [−3.487, −0.767] | INFERIOR, 1.3 τ |
| W6 − O3.3b | −2.046 | [−2.678, −1.414] | INFERIOR, 1.2 τ |
| **W6 − O3.D** | **−3.620** | [−4.521, −2.718] | **INFERIOR, 2.1 τ** |
| O3.1 − O3.3 | +0.810 | [−0.551, +2.172] | inconclusive |
| O3.1 − O3.3b | +0.891 | [−0.046, +1.829] | inconclusive |
| O3.1 − O3.D | −0.682 | [−1.702, +0.338] | inconclusive (O3.D worse) |

**Your three-way reading resolves to branch THREE: neither recovers the 2.3.**

## 8.3 O3.D is not merely unhelpful — it is actively worse, and it saturates

Median **15.04**, worse than its own parent, 2.1 τ inferior to W6, worse than O3.1 at **every**
shell including the flanking Pb (39.98 vs 37.41). It is the **first arm since O3 to saturate**:
on-site p95 shift **2.767 eV** against `Δ_Z` = 3.079, saturated fraction 0.002. And it is the
**only arm to fail G2a** — E79 0.6621 against W6's 0.5151, one-sided 95 % upper **+0.213**
against a 0.10 margin.

Read plainly: handing the on-site readout 512 base invariants makes it demand a larger
correction than the registered bound allows, and it performs worse for it. That is O3's failure
reached by the opposite route — O3 through a badly conditioned input, O3.D through too rich a
one. **Your judgement that O3.D was self-defeating as a product is confirmed, and it turns out
to be unattractive as a diagnostic result too.** The body-order argument stands as physics but
does not survive as an explanation for THIS deficit.

## 8.4 The finding worth carrying forward: energy and forces have decoupled

| arm | E79 (meV/atom) | vs W6, 95 % upper | on-site p95 shift (eV) | force median |
|---|---:|---|---:|---:|
| W6 | 0.5151 | — | 0.846 | 11.60 |
| O3.1 | 0.4552 | −0.0245 PASS | 0.740 | 14.44 |
| **O3.3** | **0.3387** | **−0.1108 PASS** | 1.526 | 13.55 |
| **O3.3b** | **0.3395** | **−0.1280 PASS** | 1.569 | 13.32 |
| O3.D | 0.6621 | +0.2132 **FAIL** | 2.767 | 15.04 |

O3.3 **doubles** the on-site correction it uses (p95 1.53 eV against O3.1's 0.74) and converts
that almost entirely into ENERGY: **E79 34 % better than the shipped W6 head**, the best energies
anywhere in this programme. The same capacity buys **+0.81 meV/Å** of force, which does not clear
τ. The B-basis is real, is used, and is worth having — it is simply not what the force deficit
is made of.

Shell localisation agrees: O3.3/O3.3b improve the NEAR field (flanking Pb 37.41 → 33.2,
first-shell Cl 20.56 → 18.0) and are flat-to-worse beyond 8 Å — the opposite localisation to
O3.2's, and the right one for an on-site change.

| shell (Å) | W6 | O3.1 | O3.3 | O3.3b | O3.D |
|---|---:|---:|---:|---:|---:|
| 2–4 | 19.83 | 25.54 | 22.82 | 22.63 | 27.30 |
| 4–6 | 13.81 | 17.49 | 15.72 | 15.64 | 17.87 |
| 6–8 | 9.73 | 12.62 | 11.92 | 11.78 | 12.75 |
| 8–10 | 9.86 | 11.11 | 11.22 | 11.15 | 11.60 |
| 10–12 | 9.02 | 10.48 | 10.51 | 10.62 | 10.74 |
| **flanking Pb** | 27.40 | 37.41 | **33.64** | **33.21** | 39.98 |
| first-shell Cl | 16.55 | 20.56 | 17.96 | 18.14 | 21.43 |

## 8.5 The variance story, and what O3.3b settled

O3.3's spread is **3.65 meV/Å** against 0.37–0.89 for every other arm. Diagnosed and measured:
it is **not** input outliers (the standardised 211-wide block tops out at \|z\| 4.1–6.7 over 4800
atoms, and the 171 product channels are no heavier-tailed than the linear ones — that hypothesis
is recorded as wrong). It is the A1 bound. O3.3 reaches the barrier knee in **11–44 of 60
epochs** where O3.1 never reaches it once on any seed, and barrier incidence orders the seeds
exactly as the final error does (s2: 11 epochs → 11.99; s1: 23 → 13.91; s3: 44 → 15.64).

The A1 barrier had never had the weight to act — `sat_weight` 1e-3 × `sat` ≈ 1e-6 is a millionth
of the loss — because no arm before this one went near the knee. **O3.3b** (one change,
`sat_weight` = 1.0) cut barrier incidence to 3–5 epochs and the spread to 2.39, rescued the two
bad seeds (15.64 → 13.07, 15.31 → 13.06) and cost the two best (11.99 → 12.80, 12.24 → 15.20).
Median essentially unmoved, **se halved** (0.675 → 0.314). It is a variance fix, not an accuracy
fix.

## 8.6 What we would run next, for your decision

The registration says branch three points at the **rank-1 block**: W6 builds it from the base's
many-body `l = 1` feature, the head builds it from a two-body `A⁽¹⁾`. The indicated arm is the
**equivariant correlation-2 product for `v_i`**. Not opened; it will be registered before it runs.

Two things we would want your view on first:

1. **Is the rank-1 block worth one more arm, or is 2.0 meV/Å the decoupled head's floor?** Four
   arms have now moved the force median 18.81 → 13.32 and stalled ~2.0 above W6, while the
   ENERGY has gone from worse-than-W6 to 34 % better. If the decoupled head is to be adopted on
   a speed-and-energy argument with a known force penalty, that is a product ruling available
   now, and `Δ_Z` (see 2) may be the cheaper lever.
2. **`Δ_Z` is now implicated as a binding constraint, and it is a registered A1 bound.** O3.3
   wants 1.53 eV of on-site shift against O3.1's 0.74, O3.D wants 2.77 and saturates against
   3.079. An arm that raises `Δ_Z` is an A1 amendment, not a head change, and would tell us
   whether the bound rather than the parameterisation is what limits the on-site term. We have
   not registered it because amending a declared bound is your call.

Also recorded, unchanged by these arms: **the decoupled head with electrostatics (13.32 at best)
is still worse than the coupled head without them (`Phi = 0`, 12.22).**
