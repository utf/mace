# Stage A′ base refit, head edits, and the Stage B objective — report

V_Cl+ in orthorhombic CsPbCl3. Labels: Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025);
PBE scalar-relativistic; defect calculations set up through `doped`. Plan of record:
`STAGE_APRIME_SPEC.md`, received 3 Sep 2026, executed in the order of its section 5.

**Every trained number below carries its regime tag.** Three regimes appear: the s7
head-only cohort (frozen Stage-A base `e0_base_s1`, γ = 3 eV, Gaussian 0.05 eV, forces +
`loss_gap`, 60 epochs, lr 0.01, float32), the Stage A′ bases (neutral data only, e0 recipe,
140 epochs, float32, cuEq, 159-atom share 0.25 in energy AND forces), and the Stage B cohort
(head only on the A′ production base, section 3 of the spec, 24 epochs, lr 0.005, trunk
float32 / head float64 with the base cache).

---

## 0. Decisions of record, and what each became

| decision | outcome |
|---|---|
| Joint training of base and head on charged labels retired; base on neutral labels only, frozen during head training; criterion 1 a regression test | Done. Stage A′ trained on n = 0 frames only (`dataset_e0` and the four `dataset_cf` folds); Stage B trains under the head-only mask with `base_lr_factor 0`, the base's outputs cached, and a drift guard on the cache every epoch. Criterion 1 is gate 1 below, scored against the A′ references. |
| E_LR retained unchanged in form; density detached; long-range parameters frozen at physical values | Done, with one interpretation recorded: "physical values" means host charges zero, polarisation off, amplitude 1/√ε∞ = 0.5 at ε∞ = 4 — the ion lattice sits in H through the Madelung term, and a base trained without E_LR cannot absorb a geometry-dependent term from randomly initialised host charges. The previous isolated-gauge convention (`carrier_self_isolated` off) is kept unchanged. The spec's neutrality check `|Σq + Δn| < 1e-8` holds only at amplitude 1; the invariant enforced is the density's (Σ q/a + Δn = 0), and the spec's residual is logged each epoch. |
| Decay lengths as four learned universal scalars | Done: `L_b = L0 exp(β_L tanh u_b)`, β_L = ln 2, L0 = 1.0 Å, u_b = 0 at start, one set shared across hosts. |
| Precision trunk f32, head f64; base outputs cached; only the first interaction block recomputed | Done (section 2.5 below). Identity half of F20 holds; the ≥ 3× speed half does not (1.5× per epoch): the step is CPU-bound in the head's per-graph eigensolve and Ewald loop, which the excluded remedies own. |
| Not in this cycle: eigensolve batching, Ewald geometry precompute, SCC, occupation/SiC | Not done, as instructed; they are where the remaining time is. |

**Standing-rule addition executed (section 5.1).** `avg_num_neighbors` travels in the state
dict through a buffer refreshed at write time and written back at load time; a test asserts
the premise that the plain float is still invisible to the block's own `state_dict`, so the
carry is retired rather than left to disagree if upstream ever makes it a buffer. Every new
knob (learned decay lengths and their β, the modulation form, the long-range detach and
freeze flags, image compensation, the precision policy, the centred on-site channel, the
cache checksum) is a constructor argument, a CLI flag, a launcher variable and a
config-extractor key from the day it exists; the round-trip test flips all of them.

---

## 1. Corrections found on the way, before any result

**The neutral two-size upweight of the joint run never reached the loss** (LEDGER.md entry
12). `DefectLoss` scores an n = 0 frame through its base terms, which read
`base_energy_weight` and `base_forces_weight`; the upweight scaled the generic
`forces_weight`, which no term reads for a neutral frame. Verified by printing the columns
on a 159-atom neutral frame after the upweight: `forces_weight` 0.497, `base_forces_weight`
1.0. The joint run's "both realised 25.0%" was true of the charged population and false of
the neutral one; the neutral 159-atom force-slope collapse in that report happened at
natural weight. The first Stage A′ launches trained on the same inert weight for forty
minutes and were stopped. The column is now named per population, the realised share reads
the same column, and a test moves the `DefectLoss` value with it.

**Three things the smokes found before the chain ran unattended.** The Stage-A loader
refused every historical checkpoint for lacking the new constants buffer (it is bookkeeping,
carried by the explicit float copy, and is excluded from the check now); the profiler probe
ran before the trainer had moved the loss to the device; and the trainer's own probing
forward for the pristine centre tripped the "no centre yet" guard it was about to satisfy
(the model now collects the centre itself with the correction held off). The pristine-centre
buffers are correction state, which a Stage-A checkpoint never has.

---

## 2. Section 5.1 — the standing rule, and the knobs that exist from day one

`avg_num_neighbors` divides every message in the trunk and is a plain float on each
interaction block: not a parameter, not a buffer, absent from `state_dict`. LEDGER.md entry
10 is what that cost. It now travels in a buffer on `MACEDefect`, refreshed from the blocks
by a `state_dict` hook at write time and written back onto them by a load post-hook, so a
stage boundary cannot lose it and the blocks stay the source of truth while the model runs.
A test asserts the premise — that the float is still invisible to the block's own
`state_dict` — so that if upstream ever makes it a buffer the carry is retired rather than
left to disagree. The Stage-A loader treats the buffer as bookkeeping (the explicit float
copy is what carries the value), which a smoke found was needed before any historical
checkpoint would load.

The seven knobs of the spec — learned decay lengths and their β, the modulation form, the
long-range detach and freeze flags, image compensation, the precision policy, the centred
on-site channel — are constructor arguments, CLI flags, launcher variables and
config-extractor keys, and the round-trip test flips every one and reads it back off the
rebuilt model. `u_b` itself is a parameter of the head (four zeros at construction), so it
travels by state dict; the flag that makes it trainable is what the config carries.

## 3. Section 5.2 — precision and the base cache (F20)

Regime for every number here: the Stage-B configuration on the production model (Stage-A
base `e0_base_s1` frozen under the head-only mask, counting head reading the first block,
γ = 3, Gaussian 0.05, exp envelope L0 = 1.0, log modulation β = ln 1.5, learned decay
lengths on, E_LR from epoch 0 with the density detached and the branch frozen, both size
upweights 0.25, batch 8, float64 data), local A4000, 2877 frames.

Under the mixed policy the embeddings, interactions, products and base readouts run in
float32 and the carrier head, its readouts, the Madelung term and the long-range branch in
float64, with two casts in `forward`: trunk inputs down at the embedding, trunk outputs up
at the head boundary. The data and the loss stay float64 (a 500 eV label at float32 loses
5e-5 eV, which is not the loss's precision). The base cache holds, per frame and keyed by a
content hash of the geometry, the trunk's energy and forces and the later blocks'
invariant readouts; a cached forward recomputes only the first interaction block, which is
the one the head differentiates through. The model refuses to cache a configuration under
which something cached would be training — base unfrozen, head reading every block,
long-range branch not detached or not frozen — and a drift guard runs one random frame
through the full forward every epoch.

| run | trunk | cache | profiler step, mean of 3 | epoch 0 wall (c-shift → eval) |
|---|---|---|---|---|
| uniform, all float64, uncached | f64 | off | — | 12.7 min |
| mixed, uncached | f32 | off | 4.239 s | — |
| mixed + cache | f32 | on | 3.451 s (1.23×) | 8.4 min (1.5×) |

Where the time goes (CUDA self time 1.74 s of a 4.24 s step): `linalg_eigh` 0.50 s,
`mm` 0.21 s, `erfc` 0.14 s, tridiagonalisation kernels 0.29 s; CPU self time 5.7 s per
step. The step is CPU-bound in the head's per-graph eigensolve and Ewald loop, not in the
trunk the cache removes; the two remedies the spec excludes this cycle are the ones that
would move it.

**F20.** Identity half: head outputs (eps, δ_sr, E_head, F_head) within 1e-6 eV / 1e-6 eV/Å
of the all-float64 model on a fixed frame and E_total within 1e-3 eV — **holds** (unit
test). Speed half, ≥ 3× per epoch — **fails**, 1.5× measured. Drift guard on a random
training frame each epoch: |ΔE| 6–8e-7 eV, max|ΔF| 1–2e-6 eV/Å. 2797 distinct geometries
from 2877 frames (80 pristine repeats); cache built in 338 s and written once, keyed by the
base's SHA-256.

## 4. Section 5.3 — forward-only on the s7 cohort (F15, F16)

Regime: the s7 head-only cohort (γ = 3, Gaussian 0.05, frozen Stage-A base, forces + gap,
60 epochs, lr 0.01, float32), six seeds; the image-compensation term switched on and off
on the same trained weights.

### 4.1 Image compensation in H (§2.3)

Implemented as specified: a first solve of H without the term gives the carrier density
(detached); `q_c = −ρ`; `φ_img = [A_per q_c] − [A_iso q_c]` from the same kernel, smearing
and gauge as E_LR, differentiated with respect to the charge exactly as the Madelung site
potential is; `ε_i += −φ_img/ε∞`; second solve. The per-atom term travels out of the
forward for the tiling test.

**Identities.** Zero at zero charge — holds. Constant across atoms for a uniform carrier on
a pristine cell — **does not hold as stated**: on a 40-atom cubic CsPbCl₃ cell the image
potential of `q = −1/N` has mean +2.52 eV, variance 6.1e-2 eV², range +2.07 to +3.03 eV.
The isolated evaluator's potential of a finite cluster is not constant across the cluster,
and the identity as written assumes it is. Recorded; the term was not tuned to pass it.

**Tiling drift (adoption test).** One Cl vacancy in the 1×/2×/3× isotropic tilings of the
80-atom pristine cell (79, 639, 2159 atoms; L = 14.2, 28.5, 42.7 Å), ideal geometries, the
counter bookkeeping copied from a real charged frame; D(L) = Σᵢ ρᵢ [madelungᵢ + compᵢ].

| model | D0 = D(1×) − D(3×), no term | with term | ratio |
|---|---|---|---|
| s1 | +0.432 | +0.027 | 0.062 |
| s2 | +0.439 | +0.035 | 0.079 |
| s3 | +0.410 | −0.003 | 0.008 |
| s4 | +0.408 | −0.007 | 0.017 |
| s5 | +0.429 | +0.019 | 0.045 |
| s6 | +0.407 | −0.008 | 0.020 |

6/6 within 0.3·D0 — **passes**. The frontier-weighted compensation is −0.61, −0.32,
−0.23 eV at 1×, 2×, 3× on every model; it cancels the ~1/L drift of the ion Madelung term
(2.78 → 2.49 → 2.34 eV). Its per-atom spread on the tiling frames is 0.062, 0.024,
0.016 eV — for a localised carrier (N_eff ≈ 3) the image potential is nearly uniform over
the home cell, the opposite regime from the uniform-carrier identity.

**Probe (F15), both sizes, on minus off:**

| reading | off | on | Δ (mean ± sd over six) | sign per model |
|---|---|---|---|---|
| depth from CBM, 79 | +0.2125 | +0.2125 | −0.0000 | — |
| depth from CBM, 159 | +0.0947 | +0.0947 | −0.0000 | — |
| participation ratio, 79 | 0.157 | 0.141 | −0.016 ± 0.003 | 6/6 − |
| participation ratio, 159 | 0.492 | 0.401 | −0.091 ± 0.017 | 6/6 − |
| R_bound | 0.738 | 0.590 | −0.148 ± 0.039 | 6/6 − |
| force loss, 79 | 0.00066 | 0.00075 | +0.00009 | 6/6 + |

The depth clause is **unmeasurable** under the quantile-aligned depth: a near-uniform
on-site shift is exactly what the alignment removes. The ratio falls (true); R moves away
from 1 rather than toward 0.9–1.0 (false); the 79-atom force loss rises 13% (false).
**F15 fails, 1 of 4** with one clause not scorable by this probe. The tiling test passes
and the probe fails, so under the registered rule the term is **not adopted** as a default;
it stays behind `--defect_image_compensation` (off in Stage B).

### 4.2 The extrapolation indicator on the existing Stage-A folds (§1, F16)

Regime: `cf_base_f0..3` (Stage-A recipe, neutral quarters) and `e0_base_s1`; 2877 frames.

| population | n | ood_E median (eV/atom) | p95 | ood_F median (eV/Å) | w_E median | w_E > 0.5 |
|---|---|---|---|---|---|---|
| neutral 79/80 | 1813 | 0.00094 | 0.00103 | 0.0040 | 1.000 | 100% |
| neutral 159 | 17 | 0.00097 | 0.00102 | 0.0034 | 1.000 | 100% |
| charged 79 | 1030 | 0.00125 | 0.00147 | 0.0069 | 0.683 | 93.7% |
| charged 159 | 17 | 0.00101 | 0.00102 | 0.0030 | 1.000 | 100% |

s_E = 0.00103 eV/atom; w_E against d on the charged 79-atom frames is flat (0.65–0.76 in
every bin from 4.5 to 7.0 Å). Charged 79-atom residual slope against the production base:
full range +0.3692 [+0.3241, +0.4143]; **w_E > 0.5: +0.3593 [+0.3144, +0.4042]** (n = 965);
w_E ≤ 0.5: +0.0622 [−0.4057, +0.5302] (n = 65). **F16 fails.** The four fold bases disagree
by about 1 meV/atom on charged and neutral geometries alike; the base's long-d error is
shared by all four, so their disagreement does not mark it. On the A′ folds (section 5
below) w_E is 1.000 everywhere.

## 5. Section 5.4 — Stage A′ (§1, F17)

Regime: the e0 recipe (140 epochs, float32, cuEq, energy 10 / total-energy 10, no
long-range branch, every carrier regulariser zero, batch 8) with the neutral 159-atom
frames raised to a 0.25 share of both the base energy loss and the base force loss, on the
columns the base terms read, logged every epoch. No charged frame in any set.

| base | data | 159 frames | E factor | F factor | final train E / F | final valid E / F |
|---|---|---|---|---|---|---|
| aprime_f0 | dataset_cf/fold0 | 12 | 92.0× | 46.1× | 2.5 / 11.5 | 1.0 / 6.1 |
| aprime_f1 | fold1 | 12 | 92.0× | 46.1× | 2.5 / 11.8 | 1.2 / 6.1 |
| aprime_f2 | fold2 | 12 | 92.0× | 46.1× | 3.1 / 11.8 | 1.8 / 6.1 |
| aprime_f3 | fold3 | 12 | 73.5× | 36.9× | 1.8 / 11.3 | 1.4 / 6.0 |
| aprime_prod | dataset_e0 | 15 | 83.9× | 42.1× | 6.1 / 10.9 | 4.9 / 11.8 |

meV/atom and meV/Å; fold validation sets are pristine only, the production one holds 95
pristine, 57 neutral 79-atom and 2 neutral 159-atom frames. The final-table numbers differ
from the last epoch's own evaluation line on the old base too (e0: 1.72/11.67 at epoch 139
against 2.9/10.9 in the table), a pre-existing evaluation-mode difference not chased here.

**The references, re-derived** (b1's machinery on the A′ folds and production base):

| observable | A′ | pre-joint |
|---|---|---|
| charged 159 energy slope, production base | **−0.0948 [−0.1084, −0.0812]** | −0.1338 [−0.1447, −0.1232] |
| charged 159 force slope, production base | **−0.2665 [−0.2938, −0.2393]** | −0.1901 [−0.2103, −0.1700] |
| neutral 159 null, out-of-fold (energy) | **+0.0243 [−0.0697, +0.1184]** | +0.0800 [−0.0503, +0.2102] |
| neutral 159 null, out-of-fold (force) | +0.0167 [−0.0163, +0.0497] | +0.0643 |
| neutral 159, production base (energy) | +0.0920 [+0.0492, +0.1347] | +0.0968 |
| charged 79, full range | +0.3724 [+0.3265, +0.4183] | +0.3641 |
| charged 79, neutral-dense window | +0.0793 [−0.0976, +0.2562] | +0.0806 |
| neutral 79 window null | +0.1147 [+0.0947, +0.1348] | +0.1317 |

The reference itself moved. With the large neutral cells finally weighted in the base
loss, the out-of-fold null at 159 atoms fell from +0.080 to +0.024 and the charged residual
from −0.134 to −0.095: about a third of the old reference was the base's own extrapolation
at large d. The charged residual is still carrier physics — its interval and the null's are
disjoint. The 79-atom picture is unchanged.

**F17:** null within ±0.05 of zero — holds (+0.024); charged 159 inside the old interval —
no, but shifted by 0.039 against the old null's width of 0.26 — holds; 79-atom charged
residual magnitude < 0.2 — fails (+0.37). **2 of 3.** Gate 2's reference is −0.0948 with the
1.5× band [−0.142, −0.063].

**w_E from the A′ folds** (the weight Stage B uses): s_E = 0.00139 eV/atom, and the charged
79-atom frames' fold disagreement (median 1.08 meV/atom) is below the neutral one (1.21),
so w_E = 1.000 on every charged frame. F16 fails here too; the indicator does not
distinguish the population at all. Stage B applies it as specified (944/944 frames matched,
mean 1.000), which is no weight.

STAGE_B_BODY

---

## 7. Operational record

- Local A4000: F20 measurement, the Stage B smoke, the A′ production base. b3 GPUs 4–7:
  the A′ fold bases, the forward-only probes, the Stage B seeds, the gates; never more than
  four GPUs in use, one shared between a fold base and the OOD indicator.
- Every queue script waits on a file, never on a process (entries 10 and 11). One launcher,
  `b3_run.sh`, starts every remote job from the worktree, because `ssh b3 '…'` starts in
  `$HOME` and a relative script path fails silently — it did, twice, before the launcher.
- The Stage B smoke (one epoch, old base, old w_E) is what caught the centre guard and the
  c-table class; the two costed ~1 h and were the difference between a chain that ran and
  one that stopped at 20:00 with nobody watching.
