# Stage 3 with the density response wired — 6/6, and the forecast was wrong in both directions

Six seeds, 60 epochs, lr 0.01. Identical to the frozen-P rerun in every respect but one: the
force loss's parameter gradient now carries `-Tr(dD/dtheta . dH/dR)`, the density response.
Harrison init, c-shift, 5-epoch warmup, clip 1.0, `loss_gap` at E_gap = 2.4, learned Z, init
gate on, float64 solve, 48 charged frames, seeds 1-6.

**Regime tag on everything below: lr 0.01, AdamW, 60 epochs, frozen Stage-A base.** Per the
standing rule, none of it transfers to another optimiser regime without a cross-regime check.

## §1 — the wiring itself

The force is linear in its cotangent, so `F(D) - F(D.detach()) = F(D - D.detach())`, and that
argument is **exactly** zero. Adding its contraction to the forces therefore cannot move a
single force component — an identity, not a tolerance — while contributing precisely the term
the frozen-P route drops. Chosen over removing `delta_sr` from the `get_outputs` energy and
adding an explicit force back, which would have silently dropped the head's contribution to
virials and stress and made value-preservation a 1e-6 property.

On-batch gate, 159 atoms, run on both machines before any seed was spent:

| | |
|---|---|
| A explicit vs autograd head force | 3.9e-07 against a 7.5e-02 head force |
| B the response tensor | exactly 0; on-vs-off inside the scatter-atomics repeat-run floor |
| C `d(force loss)/d(v0_raw)` | 4.95e-06 frozen-P → 6.33e-04 wired, **128x** |
| D pristine frame | head force and head energy both exactly 0 |
| E per-step cost | 218 → 240 ms on b3 (1.10x) |

C is the headline. The term that was missing is two orders larger than the one that was
there — which is why the frozen route had no gradient with which to leave an atomic-limit
initialisation.

Cost was 1.45x before two exact optimisations: the multi-fill Function now takes the
spectrum and occupations from `head_energy_hf`, which had just built both from the same H
(a duplicated float64 `eigh` of a 636x636 matrix, 26 ms, plus four bisections, 9 ms), and its
backward shares the eigenbasis round trip across all four fills (six GEMMs, 22 ms). The
response term is `6.331e-04` before and after, to every digit printed.

## §2 — the rerun

| | frozen-P | **wired** |
|---|---|---|
| trained | 6/6 | **6/6** |
| axial_red | +0.519 ± 0.008 | **+0.645 ± 0.010** |
| rmse_all | 33.4 ± 0.2 | **25.3 ± 0.6** |
| N_eff | 2.92 | **4.75** |
| pristine frontier gap | 2.394 | **2.400** (target 2.40, gate ≤ 0.10) |
| init-gate trips | none | none |

**The forecast on record was "axial_red and N_eff within spread of the frozen-P rerun".
Neither is.** The fit improved — RMSE down 24%, axial_red up 0.126, twelve frozen-P standard
deviations, in 6/6 seeds — and the state loosened, N_eff 2.92 → 4.75.

Per seed: axial_red +0.633 to +0.658, N_eff 4.17 to 5.27, gap 2.389 to 2.418.

N_eff cannot say which of two readings is right — that the 2.92 was an artefact of a gradient
that could not move the density, or that the wired head has found some of the
fit-versus-delocalisation trade the s-only control lives on. It measures spread over a fixed
cell, not boundness. The dilution gate is what separates them, and it says bound (below).

For scale: the Stage-2 s-only control sat at N_eff 33.1, and its two best-fitting seeds were
its two most delocalised at 64-70. This is an order of magnitude away from that.

### Z re-partitions, it does not run away

Frozen-P ended at Z = (-0.798, +0.690, +1.706); wired at (-0.763, +0.317, +1.970). The Cl
charge — the one that sets the Madelung well depth at the vacancy — lands in the same place.
Pb goes to nominal +2 and Cs absorbs the difference, decelerating throughout (0.466 → 0.412 →
0.389 → 0.317). Neutrality holds to the projection at every epoch.

## §3 — the three gates

### F5 — PASS, 6/6

`corr(lambda_frontier, d_hub) = +0.898 ± 0.036` on the 159-atom subset, slope +0.052 to
+0.081 eV/A, **every seed's CI excluding zero**. The defect level rises as the two Pb dangling
orbitals separate. An energy offset attached to the vacancy cannot do this; nothing in an
offset knows about d. This supersedes the Stage-1 figure, which was a six-state truncation of
a different eigenproblem read in one optimiser regime.

The counters are `(0, 0, 1, 0)` — **a hole**, `h_maj = 1`, so `n_maj = ref - 1`. The estimator
was made carrier-agnostic (`max(n_maj, n_maj_ref) - 1`, the level whose occupation changed)
**before** the gate ran. The naive "highest occupied level" index would have landed one below
the defect state and measured a valence level's d-dependence — a clean-looking null from the
wrong quantity.

### F4 — 5/6 against the threshold, but the pattern is the finding

| seed | slope (eV/A) | 95% CI | corr | |
|---|---|---|---|---|
| 1 | -0.0600 | [-0.0770, -0.0430] | -0.896 | PASS |
| 2 | -0.0579 | [-0.0707, -0.0451] | -0.933 | PASS |
| 3 | -0.0725 | [-0.0866, -0.0583] | -0.947 | PASS |
| 4 | -0.0650 | [-0.0785, -0.0515] | -0.940 | PASS |
| 5 | -0.0431 | [-0.0565, -0.0297] | -0.879 | **FAIL** |
| 6 | -0.0660 | [-0.0798, -0.0522] | -0.940 | PASS |

Right sign 6/6, mean -0.0607 ± 0.0091 against the -0.134 reference. Against the old models
this was "right sign 3/4, 7x short"; it is now unanimous in sign and 2.2x short. Seed 5 misses
the 3x band by 0.0016 eV/A.

Six seeds in the same place with tight intervals that exclude the reference is a **bias, not
scatter**, and "5/6 pass" reports a threshold while saying nothing about the pattern.

### F4 follow-up — the shortfall is the HEAD's, not the base's

Per model, both slopes on the same 16 frames:

| | slope | 95% CI |
|---|---|---|
| head, `d(delta_sr)/dd` | -0.0607 ± 0.0091 | see table above |
| residual, `d(E_label - E_base)/dd` | **-0.1310 ± 0.0000** | [-0.1422, -0.1198] |
| M1b reference | -0.1340 | — |

**This base's residual slope brackets the -0.134 reference in 6/6 models.** So the base is not
the problem: the trend left for the head to reproduce is -0.131, essentially the reference,
and the head delivers 46% of it with non-overlapping intervals. Verdict "head" in 6/6.

Consequence for the plan: **the joint run will not fix F4 by itself.** Training the base
jointly removes M1b's base-extrapolation slope from the 79-atom energy targets, which is a
different problem from this one — here the frozen base is already leaving the right trend.

What the numbers say about the shape of the miss: `corr` is -0.88 to -0.95, so the head
tracks the trend almost perfectly and is short only in AMPLITUDE, by a near-constant factor
across seeds. That is the signature of a missing multiplicative contribution rather than a
missing mechanism. Two candidates are already staged off and neither is tested here — the
second-order SCC term (`scc_energy` is interface-only, Edit 5) and the long-range branch
(off in Stage 3). Stating them as candidates, not as an explanation.

The gauge does not enter this comparison: E_base at defect geometries is latent (the
under-determined-gauge warning is real), but a gauge freedom is a constant offset and this is
a slope against d.

### Dilution — bound-conditioned R_model

The only observable in hand that identifies boundness. A bound carrier's force on its own
atoms is size-invariant, R ~ 1; a band state's hub amplitude halves when the cell doubles and
its hub force with it, R ~ 2. Scores the model's own carrier force through
`test2_size.matched_ratios` verbatim, so R_model and R_DFT differ by the models and not by two
matching schemes.

Bound condition `depth / delta_L > 2`, with depth = `lam[k+1] - lam[k]` on the charged frame
and `delta_L` the median level spacing above the frontier of the **trained** model's pristine
spectrum.

**PASS, 6/6.**

| seed | delta_L (eV) | depth (eV) | bound | R_bound |
|---|---|---|---|---|
| 1 | 0.0100 | 0.0376 | 75% | 0.86 [0.76, 1.16] |
| 2 | 0.0111 | 0.0420 | 81% | 0.85 [0.80, 0.97] |
| 3 | 0.0137 | 0.0411 | 69% | 0.84 [0.68, 1.60] |
| 4 | 0.0135 | 0.0415 | 81% | 0.81 [0.69, 0.86] |
| 5 | 0.0100 | 0.0359 | 75% | 0.93 [0.82, 1.35] |
| 6 | 0.0134 | 0.0547 | 81% | 0.84 [0.72, 0.91] |

**R = 0.85 ± 0.04**, bound fraction **77% ± 5%** of the 16 large frames. Every seed's median
is below 1.0 and far from the band-state value of 2; four of six have intervals excluding the
1.3 gate outright.

**This is the gate that answers what N_eff could not.** The state is bound — its force on its
own atoms does not dilute when the cell doubles — so the rise from N_eff 2.92 to 4.75 is the
bound state relaxing to its own size, not the onset of delocalisation. Of the two readings
offered above, the first is supported and the second is not: 2.92 was an artefact of a
gradient that could not move the density.

## Subset size

Both d-trend gates run on **16** frames, not the 17 the inventory reports: that is what
`train.xyz` holds at 159 atoms with a locatable vacancy. d range 4.84-7.12 A.

## §4 — the tiling test, and the defect it found

`tests/extensions/defect/test_tiling.py`, in the standing suite. A cubic CsPbCl3 cell against
its own 2x tiling — the same crystal, described twice. **7 pass, 1 xfail-strict.**

Passing: on-site energies bit-identical per species; the off-diagonal multiset doubles exactly;
every eigenvalue of the cell appears in the tiling to 1e-8 (band folding); trace doubles and
bandwidth does not; the neutral fill tiles.

### Correction to the plan's §4: `F(2N) = 2F(N)` is not an identity

The plan states exact extensivity of the free energy. It is not exact and asserting it would
be wrong: `F` sums the occupied states of a **Gamma-point** Hamiltonian, and tiling ADDS
Brillouin-zone sampling points. `F(2N)` samples k = 0 and k = pi/L; `2F(N)` samples k = 0
twice. They agree only in the k-converged limit — measured here, the 2x tiling differs by
**0.23%**. The test now asserts what the gate was reaching for: the free energy per atom
CONVERGES under successive tilings rather than drifting. A term that scaled with cell size
would show a growing difference instead.

### KNOWN DEFECT: phi_LR is not supercell-invariant

**The Madelung potential at chemically equivalent sites of the same crystal changes by eV
between supercell choices of that crystal.** Cl sits at +4.542 eV in a 2x2x2 cell, +5.869 in
its 4x2x2 tiling, +5.758 at 3x3x3. Within any one cell all equivalent sites agree to 1e-15, so
this is not noise.

Traced exactly. The whole change is `-delta(A_ii) * q_i`: it comes entirely from the
`self_potential_of` subtraction. `(A q)_i` on its own is supercell-invariant, as it must be —
a supercell adds only k-points at which the structure factor vanishes. `A_ii` scales as 1/L:

| cell | L (A) | V (A^3) | A_ii (eV/e) |
|---|---|---|---|
| 1x1x1 | 5.6 | 176 | -7.320 |
| 2x2x2 | 11.2 | 1405 | -3.666 |
| 4x2x2 | 22.4 | 2810 | -2.340 |
| 3x3x3 | 16.8 | 4742 | -2.450 |
| 4x4x4 | 22.4 | 11239 | -1.843 |
| 6x6x6 | 33.6 | 37933 | -1.235 |

Halving as the cell doubles in linear size: the Makov-Payne self-image signature, not the
Gaussian self-energy (which is +11.49 eV/e at sigma = 1 and is V-independent).

**Why it matters here.** Edit 1 sets `eps_i = eps_local_i - phi_LR_i / eps_inf`. With
`eps_inf = 4`, the same physical site acquires on-site energies differing by roughly
`0.3 eV * Z_i` between the 79- and 159-atom training cells — **a first-order size dependence,
in the term Edit 1 introduced, at the two cell sizes the dataset actually contains.** It is
species-proportional, so it does not cancel between species.

**What is NOT settled, and is a modelling decision rather than a bug to fix quietly.** `A_ii`
is the self-plus-own-images term of the CURRENT lattice. Removing it is defensible for a
charged defect (it is the standard self-image correction) and wrong for a neutral crystal
(it removes real neighbours that a supercell has merely relabelled). The test is
`xfail(strict=True)` with the measurement in its reason, so the suite stays honest, nothing is
silently patched, and it turns green the moment the convention is decided.

**This does not invalidate the Stage-3 results above.** Every gate reported here is either
within one cell size (F4, F5, axial_red, N_eff, gap) or a ratio between two sizes in which the
head's FORCE, not its on-site energy, is compared (dilution). But it is directly in the path of
the §5 joint run, which trains across both sizes with `E_LR` enabled.

## What this does not establish

One cell size for the force fit, forces only, frozen base, one optimiser regime. The
production trainer still cannot build these models — `counting_head` and `madelung_*` are not
plumbed through `arg_parser`/`model_script_utils` — which blocks §5, not §3. And the phi_LR
convention above should be settled before the joint run, not after.
