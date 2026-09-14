# The O3 lineage: decoupling the charge head from the base

**Status as of 2026-09-14.** Three arms run, all six seeds each, all gates registered before the
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
