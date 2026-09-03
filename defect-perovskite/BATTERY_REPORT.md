# Forward-only battery, the config decision, and the trainer wiring

V_Cl+ in orthorhombic CsPbCl3. Labels: Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025);
PBE scalar-relativistic; defect calculations set up through `doped`.

**Everything below is measured on the widened six-seed cohort (`gamma = 3 eV`, Gaussian
`sigma = 0.05 eV`, frozen Stage-A base, forces + `loss_gap`, 60 epochs, lr 0.01) unless a row
says otherwise. Nothing here trained** except the two-epoch smoke in section 4, which exists to
prove the trainer's call sites fire, and section 6, which trained eight joint seeds and their
E_LR-off control; every number there carries its own regime tag.

---

## 0. Record, verbatim

- **Retracted (ours):** "the candidate list is empty going into R3" and "the joint run will
  not fix F4." Both assumed the −0.131 reference is size-transferable; whether the **79-atom
  labels contain the 159-atom slope at all** was never measured. F4's shortfall decomposes
  into three measurable factors — pair weight × coupling derivative × label availability —
  plus one survivor (beyond-two-centre/superexchange). No mystery term.
- **SCC eliminated as an F4 candidate on sign** (reviewer's argument, adopted): removing the
  electron removes ½U_ψ of self-energy; delocalisation with growing d makes −½ dU_ψ/dd
  **positive**, opposing the required negative slope; the H-feedback worsens it. R3 may
  measure the number; it cannot close F4.
- **E_LR closed** (F3 + quadrupole/O(1/L³) argument). Trained version still measured free in
  the staged re-enable.
- **γ-independence of saturation status:** γ scales the tanh *output*, not its argument; a
  pre-tanh value of −3.6 saturates identically at γ = 1 and γ = 3 unless retraining moved it.
- Cl-saturation ⇒ ligand-Cl channel frozen ⇒ artificially low pair binding: the causal link
  between §2 and §1, on record.

**Second retraction, added by this cycle.** "F5's absolute level moved to +3.58..+3.81 eV from
+1.3..+2.1 — the widened bound letting the on-site correction go where it was pinned from
before." That reading is wrong. The move is +2.103 eV of rigid spectrum offset; the depth
below the conduction manifold changed by −0.025 eV. Nothing physical happened. See F9.

### The record carried into this cycle, verbatim

- b7's null is scoped: it varied the global envelope at initialisation, where the environment
  modulation ≡ 1; the defect-local lever was never exercised. The modulation ceiling (within
  2% of bound on 33% of hub bonds, b3) is the one mechanism that raises hub coupling without
  inflating bandwidth. §1 resolves it before the joint run because §2's leakage risk is
  maximal for a head at its stop.
- Adoption-rule rationale, verbatim: "M1b removed" alone is passable by leakage — a base that
  removes the +0.36 artefact by absorbing the carrier passes it. The detector is the
  null-cleared −0.134 staying put.
- Zero-init of the on-site correction output at joint-run start (removes b4's +0.27 eV gauge;
  final layer of h_θ to zero; one test). Centred correction stays at R3.
- Standing-rule addition: any ratio-of-spread criterion registers an absolute floor (F10's
  lesson).

All four were executed. The first is what b9 measures; the second is criterion 1, and it fired
on every joint seed (section 6); the third fired on every joint seed (`|W_site| 0.00000`,
`b_site +0.00000` at epoch 0) and did not hold — the species-constant gauge regrew to
+0.09…+0.40 eV by the end of training, so zero-init removes b4's artefact at initialisation
and not from the objective; the fourth is why F10 carries a 50 meV floor beside its 2σ test.

**Third correction, to §0's own γ-independence note.** It is sound about a fixed pre-tanh
value and does not apply here: the widened cohort are fresh builds trained from scratch, not
the γ = 1 models re-evaluated. Their Cl pre-tanh is +0.001, not −3.6. F8 was near-certain only
under an assumption the experiment did not satisfy.

---

## 1. The battery

Eight measurements. Seven have their own script, `defect-perovskite/b1..b4, b6..b8_*.py`;
the eighth (b5, saturation by species) is `s6_saturation_by_species.py` re-run on the
widened cohort, unchanged, so the two gamma regimes are scored by identical code. Raw
output in `~/runs/b*_*.json` on b3; b1 ran locally, where the cross-fit bases live.

### b1 — the labels' own d-slope, per size, with two controls (F6)

`d(E_label − E_base)/dd` and the axial pair-force residual slope, cross-fit bases, 95%
intervals from the per-frame residuals.

| arm | n | energy slope (eV/Å) | corr | force slope | corr |
|---|---|---|---|---|---|
| 79 atoms, full range | 1030 | **+0.3641** [+0.3161, +0.4120] | +0.421 | **+0.4202** [+0.3621, +0.4782] | +0.405 |
| 79 atoms, neutral-dense window | 533 | **+0.0806** [−0.1108, +0.2720] | +0.036 | **−0.0470** [−0.3079, +0.2139] | −0.015 |
| 159 atoms, cross-fit | 17 | −0.1459 [−0.2698, −0.0219] | −0.544 | −0.2073 [−0.2316, −0.1829] | −0.978 |
| 159 atoms, **production base** | 17 | **−0.1340** [−0.1447, −0.1232] | **−0.989** | −0.1901 [−0.2103, −0.1700] | −0.982 |

Two things to take from this table.

**The machinery reproduces M1b exactly.** −0.1340 against the recorded −0.134, corr −0.989.
The 79-atom full-range slope of +0.3641 is M1b's +0.37 base-extrapolation artefact, measured
again from scratch, and its correlation collapses from +0.421 to +0.036 inside the window
where the base's own training set is dense — M1b saw +0.447 → +0.062.

**The 79-atom labels carry no resolved d-trend at all.** Not a weaker version of the large-cell
trend: nothing. Both observables have intervals spanning zero once the base stops
extrapolating. What the head sees in 98.4% of its charged frames is a base artefact, and the
artefact points the *opposite* way to the physics in the other 1.6%.

The neutral-dense window is 3.96–5.50 Å (5th–95th percentile of 1174 neutral defective
frames). **Only 4 of the 17 large frames sit inside it**, which is the caveat the −0.134
reference has always carried and the reason for the null below.

### b1's null — is the −0.134 reference itself base error?

Both label arms are `E_label − E_base`, so both inherit whatever the base gets wrong at those
geometries, and the 79-atom arm demonstrably does. The same question has to be asked of the
large frames, and it cannot be asked by restricting the range: only 4 of 17 survive. It can be
asked with a null. **The neutral defective cells carry the same vacancy, the same d(Pb–Pb) and
no carrier, so any d-trend in their residual is base error by construction.** Each frame is
scored by the fold base that never saw it.

| arm | n | energy slope (eV/Å) | corr | force slope | corr |
|---|---|---|---|---|---|
| 79 atoms, neutral null | 1174 | **+0.1317** [+0.1131, +0.1503] | +0.376 | +0.0009 [−0.0039, +0.0057] | +0.011 |
| 159 atoms, neutral null | 17 | **+0.0800** [−0.0503, +0.2102] | +0.320 | +0.0643 [+0.0225, +0.1060] | +0.646 |

**The reference survives.** At 159 atoms the carrier-free null gives **+0.08 with an interval
spanning zero**, while the charged frames at the same size, same base, same geometries give
**−0.1338 [−0.1446, −0.1230]**. The charged value sits far outside the null's interval and on
the opposite side of zero. `d(E_label − E_base)/dd = −0.134` is carrier physics, not base
error, and F4's gate rests on something real.

The null also *confirms* the small-cell diagnosis rather than merely being consistent with it:
at 79 atoms the carrier-free base error carries a resolved **+0.1317** d-trend with a tight
interval — the same sign as the charged +0.3641 and none of the physics.

**What the null can and cannot cover, stated precisely, because the two sizes differ here.**
The null is fitted over the *neutral* frames' own d range, and at 79 atoms the two populations
barely overlap above their common median:

| | n | d range (Å) | median | above 5.5 Å | above 6.0 Å |
|---|---|---|---|---|---|
| neutral null | 1174 | 3.68–6.02 | 4.74 | 56 | **1** |
| charged | 1030 | 4.93–6.80 | 5.49 | 494 | **117** |

The charged median is essentially the null's 95th percentile. So at 79 atoms the honest
reading is narrower than "the trend is base error": **inside the region where
a null exists, neither channel carries a resolved carrier trend** — energy +0.0806
[−0.1108, +0.2720], force −0.0470 [−0.3079, +0.2139] — **and the base's own error there is
+0.1317 in energy and flat in force. Outside it, in the long-d tail where the full-range fits
get their slope, no null is measurable and carrier trend cannot be separated from base
extrapolation at all.** Either way the 79-atom labels supply nothing usable in this observable,
which is the conclusion F6 needed; what they cannot supply is a clean attribution of the
+0.3641 to one cause.

**At 159 atoms the control is genuinely matched, and that is the one the gate rests on.**
Seventeen charged and seventeen neutral cells of the same size, scored by the same bases:

| | n | d range (Å) | median | slope |
|---|---|---|---|---|
| neutral null | 17 | 4.18–6.54 | 5.75 | +0.0800 [−0.0503, +0.2102] |
| charged | 17 | 4.84–7.12 | 6.11 | **−0.1338** [−0.1446, −0.1230] |

The two overlap on 4.84–6.54, which contains twelve of the seventeen charged frames — nothing
like the 1-against-117 mismatch at 79 atoms. A carrier-free set covering most of the same
geometries gives a slope of the opposite sign whose interval contains zero, while the charged
set gives −0.134 with corr −0.989. That is the control F4's gate needed and did not have.

The channel asymmetry is worth carrying forward. The 79-atom null's **force** slope is
+0.0009 — the base makes no d-dependent force error where it has data — while its **energy**
slope is +0.1317. Force errors are local; energy errors accumulate over the cell. That is why
the head's force fit (b8) tracks the labels at both sizes while the energy channel cannot.

### b2 — the head's own energy and level slopes, per size

Matched window 5.02–6.52 Å; 120 stratified 79-atom frames, all 16 large ones; six seeds.

| observable | 79 atoms | 159 atoms |
|---|---|---|
| `d(delta_sr)/dd` | **−0.1675 ± 0.0231** | **−0.0629 ± 0.0087** |
| `dlambda_frontier/dd` | +0.1846 ± 0.0223 | +0.0736 ± 0.0103 |

The head is negative at both sizes while the 79-atom labels are positive. It is **not** being
dragged by the contaminated majority — and it cannot be, because **the Stage-3 loss is forces
plus `loss_gap` and contains no energy term at all.** `delta_sr`'s d-slope has never been a
fitted quantity in any Stage-3 run. F4 has always been a readout.

### b8 — the head against the labels in the channel it is actually trained on

The axial hub force, `forces − base_forces`, same statistic as b1's label arm.

| | 79 atoms | 159 atoms |
|---|---|---|
| **labels** (b1) | +0.4202 | −0.1901 |
| **head** (b8, 6 seeds) | **+0.2641 ± 0.0220** | **−0.3635 ± 0.0495** |

In the fitted channel the head reproduces the labels' sign at **both** sizes, including the
sign flip between them, at 63% of the small-cell slope and 191% of the large-cell one. The
head is fitting what it was given, faithfully — base artefact included.

### b3 — the hopping channel (F7)

The head's radial envelope is `exp(−(r − d_ref)/L) · (1 − (r/r_cut)^6)^2` with `d_ref = 2.8 Å`,
`L = 1.0 Å`, `r_cut = 10 Å`, while `v0` is initialised from Harrison's `V = eta ħ²/(m d²)`. The
two agree at `d_ref` by construction and diverge fast.

| r (Å) | exp | taper | model | Harrison | ratio | dln model | dln Harrison |
|---|---|---|---|---|---|---|---|
| 5.00 | 0.1108 | 0.9690 | 0.1074 | 0.3136 | 0.342 | −1.038 | −0.400 |
| 6.00 | 0.0408 | 0.9089 | 0.0370 | 0.2178 | 0.170 | −1.098 | −0.333 |
| 7.00 | 0.0150 | 0.7785 | 0.0117 | 0.1600 | 0.073 | −1.229 | −0.286 |

The taper is not the mechanism — it is still 0.91 at 6 Å. The exponential is.

On the real hub bond, with the **learned** `v0` and the learned pair modulation:

- `t_pp-sigma / t_Harrison` = **0.183 ± 0.112** (0.315, 0.180, 0.081 in d bins 4.5–5.5,
  5.5–6.5, 6.5–8.0 Å). The bounded modulation can supply at most ×1.5. **F7 fires.**
- The learned pair weight sits **within 2% of its bound on 33% of hub bonds**. The head's only
  lever on that bond is already at its stop.

The clean derivative — the two hub Pb displaced ±0.05 Å along their own axis on a fixed frame,
which the across-frame regression cannot give:

| | `d(delta_sr)/dd` | retained |
|---|---|---|
| full | −0.2088 ± 0.2035 | |
| **direct hub edge deleted** | **−0.0632 ± 0.0604** | **30%** |

**The direct Pb–Pb bond carries 70% of the head's d-response.** The channel is named.

### b7 — which envelope, measured at initialisation, before any seed

`b3`'s envelope swap is on trained weights whose `v0` already absorbed part of what the
envelope failed to supply — a sensitivity bound, not a prediction. The rerun would start from
Harrison initialisation, where `v0` is exactly the universal rule and the envelopes differ in
one factor. Two seeds each, six frames, three pristine cells.

| envelope | init gate | bandwidth (eV) | frontier gap (eV) | split | t/Harrison | `d(delta_sr)/dd` | **`dlambda/dd`** |
|---|---|---|---|---|---|---|---|
| **exp L=1.0 (current)** | 2/2 | 32.86 | 4.229 | 0.046 | 0.257 | −0.2726 | **+0.3211** |
| exp L=1.4 | 2/2 | 36.90 | 3.922 | 0.076 | 0.529 | −0.1951 | +0.2268 |
| exp L=2.0 | 2/2 | 43.58 | 3.345 | 0.110 | 0.918 | −0.1032 | +0.1172 |
| exp L=3.0 | 2/2 | 54.55 | **2.397** | 0.140 | 1.414 | −0.0506 | +0.0575 |
| power `(d_ref/r)²` | 2/2 | 44.99 | 3.451 | 0.129 | 0.909 | −0.1125 | +0.1242 |

Monotone in both directions and unambiguous. **Every candidate that raises the coupling lowers
the d-slope.** `dt/dd = t · dln t/dd`: the exponential's log-slope is three times steeper, so
its smaller `t` still gives the larger derivative, and the denser spectrum a longer envelope
produces costs more than the coupling buys. At L = 3.0 the initialisation gap has already
fallen to 2.397 eV — the target — leaving `loss_gap` nothing to work with.

### b4 — where the on-site correction goes (F10)

Per atom, by shell, six seeds, eight large frames.

| shell | pre-tanh (s) | within-shell sd | correction (eV) |
|---|---|---|---|
| hub Pb | +0.0840 ± 0.0248 | 0.0033 | +0.2512 |
| ligand Cl | +0.0901 ± 0.0183 | 0.0020 | +0.2696 |
| bulk Cl | +0.0910 ± 0.0184 | 0.0007 | +0.2723 |
| bulk Pb | +0.0933 ± 0.0203 | 0.0012 | +0.2789 |
| Cs | +0.0865 ± 0.0200 | 0.0009 | +0.2587 |

**The channel is inert, not merely unsaturated.** Every species within 0.01 pre-tanh of
every other; the within-shell spread of 0.0007-0.0033 pre-tanh is 2-10 meV once multiplied
by gamma, and ligand Cl and bulk Cl are 2.7 meV apart. It applies a
near-uniform +0.27 eV to every atom in the cell: a global gauge, degenerate with `eps0` by
species and with F9's offset overall.

Why it is free to be a gauge: a uniform on-site shift moves every level together, so it
contributes **exactly zero force**, and the Stage-3 loss is forces plus a pristine-gap term.
The constant mode is unconstrained by construction. γ was never the binding constraint.

### b5 — saturation by species at γ = 3 (F8)

| species | saturated | pre-tanh signed mean |
|---|---|---|
| Cl | **0.0%**, 6/6 | +0.001 ± 0.094 |
| Cs | 0.0% | +0.051 ± 0.088 |
| Pb | 0.0% | −0.050 ± 0.068 |

Against 100.0% and −3.604 ± 0.192 at γ = 1.

### b6 — depth against both band edges (F9)

Pristine spectrum from 80-atom stoichiometric cells (the common-δ_L size), aligned to the
defect spectrum by matched quantiles of the occupied manifold over the 5th–60th percentiles,
which excludes the frontier region where the defect state lives.

| arm | γ | smearing | λ_frontier | depth from VBM | depth from CBM | pristine gap | align IQR |
|---|---|---|---|---|---|---|---|
| s5 | 1.0 | Fermi–Dirac 0.025 | +1.5992 ± 0.3271 | +2.2696 ± 0.0222 | **+0.1277 ± 0.0235** | 2.3973 ± 0.0105 | 0.132 |
| s7 | 3.0 | Gaussian 0.05 | +3.7020 ± 0.0267 | +2.2949 ± 0.0330 | **+0.1025 ± 0.0259** | 2.3974 ± 0.0086 | 0.126 |

λ moved **+2.103 eV**; the depth from the VBM moved **+0.025 eV** and from the CBM
**−0.025 eV**. A rigid gauge shift in `eps0`, nothing more. The level sits **0.10–0.13 eV below
the conduction manifold in both regimes across twelve models** — a shallow donor, and the most
stable physical number the head has produced.

Two caveats travel with this table. The arms differ in γ, smearing family **and** width at
once, so it is regime-to-regime and not a measurement of γ; and it cannot be disentangled
after the fact, because eigenvalues do not depend on the smearing at fixed parameters — the
smearing entered only through what training put in the weights. The alignment IQR of ~0.13 eV
is the floor on any depth quoted here.

---

## 2. The config decision: no change, no rerun

The decision tree's branches, taken as written:

**F8 fails → no centring now, defer to R3.** Taken. The stronger finding — that the channel is
inert rather than pinned — is an R3 recommendation and not a reason to override the branch.
Note for R3: under the joint run's energy loss the constant mode is no longer free, because it
becomes degenerate with `c_shift` rather than with nothing, so the gauge partially self-resolves
there.

**F7 fires → adjust the global envelope constant.** **Not taken, and the reason is b7.** The
branch presumed the envelope was the lever for the d-slope; b7 is the measurement the branch
asked for, and it falsifies the presumption — every candidate lowers `dlambda/dd`, and the
current setting is already the maximum. Executing the change would be following the letter of
the tree against the measurement the tree called for. F7 is scored **fires on coupling
magnitude, remedy contraindicated**; the coupling result stands, the remedy does not follow.

So §2 resolves through "no rerun → proceed", by a route the tree did not enumerate — the same
shape as F6's third outcome. **No six-seed rerun was spent.** That is the battery doing its
job: a day of forward passes in place of a day of training that the measurement says would
have moved the number the wrong way.

The envelope is now a config knob either way (`--defect_counting_envelope {exp,power}`,
`--defect_counting_decay_length`), plumbed arg_parser → launcher → `MACEDefect` →
`extract_config_mace_model`, so the choice is recorded in every artefact rather than implied
by a default.

---

## 3. Scoring the forecasts

| | forecast | outcome |
|---|---|---|
| **F6** | the 79-atom label slope is materially smaller in magnitude, and the model's dλ/dd is size-dependent the same way | **premise fails.** The signs are opposite (+0.36 vs −0.134), and inside the neutral-dense window the small-cell slope is consistent with zero. The *consequence* F6 was after holds in its strongest form: the small cells do not contain the large-cell trend. The model's dλ/dd is size-dependent as forecast (+0.185 vs +0.074). |
| **F7** | t_PbPb at 6–7 Å sits below Harrison by more than the pair-weight factor, or inside a taper | **fires**, 0.183 ± 0.112 against a ×1.5 headroom, and not a taper (0.91 at 6 Å). Remedy contraindicated by b7. |
| **F8** | Cl remains saturated at γ = 3 (near-certain) | **fails**, 0.0% in 6/6. The γ-independence argument is sound but does not apply to fresh seeds. |
| **F9** | the λ move is a whole-spectrum offset; depths change by < 0.2 eV | **holds**, 25 meV. |
| **F10** | with the centred correction in, ligand Cl separates from bulk Cl | **fails** without it, by 2.7 meV. The channel carries no structure to separate. Restated on the zero-initialised joint channel: fails again, by 4.8 meV. |
| **F11** | the modulation bound binds at the upper stop, concentrated in the short-d bins | **half right** (§4b). Upper stop for the p channels and lower for ss-σ on the same bonds; concentrated in two of six seeds at 100% in every bin, not in d. |
| **F12** | ×1.25 on the hub bond lowers the 79-atom force loss and moves F4 past −0.08 | **fails** (§4b): −0.8% and −0.0675. The two seeds at the bound are the two it makes worse. |
| **F13** | conditional on F12 | **withdrawn** with F12. |
| **F14** | joint run under the current bound: the 159-atom charged residual slope shrinks below 0.10 in magnitude (leakage); under a widened bound it holds at −0.134 | **confirmed, 8/8** (§6): −0.0713 ± 0.0076 (Stage-A arm) and −0.0843 ± 0.0016 (from scratch). Not adopted. |

**Two scoring rules were amended after the smoke run and before the result, and both
amendments are disclosed here rather than absorbed.**

*F6.* The pre-registered rule scored magnitude only — disjoint 95% intervals with the 79-atom
|slope| smaller — because it assumed the small cells carry a weaker version of the same trend.
Opposite signs are neither "holds" nor "79 carries it", so a third outcome was added and the
literal rule is reported unused rather than stretched to fit.

*F10.* The pre-registered rule was "ligand-Cl mean outside bulk-Cl mean by more than two
within-shell standard deviations". On an inert channel the within-shell sd collapses towards
zero, so that rule **passed on a 2 meV difference** — a pass certifying that nothing happened.
A second condition was added, that the two corrections differ by more than 50 meV, and F10
fails it. Stated plainly: **the original rule technically passed, degenerately; the amended
rule fails; the amendment was made because σ → 0 makes a ratio test meaningless.**

---

## 4. The trainer wiring

`mace.cli.run_train` now runs the Stage-3 protocol behind `--defect_protocol`: Harrison
initialisation, c-shift calibration on the first batch, the initialisation gate, the linear
warmup, the head-only trainable mask, and the post-step Z projection. `stage_run.py` calls the
same six functions and its inlined copies are deleted — a test walks the AST of both files and
fails if either grows a private copy.

Four points worth their own line.

**The projection needed a hook that did not exist.** `tools.train`/`take_step` gained
`post_step_hook`, called after `optimizer.step()` and **before** the EMA update: Z lives on the
pristine composition hyperplane, and averaging a shadow copy of unprojected weights would put
them back into the model that gets evaluated. A test asserts that ordering by reading the
source.

**The warmup is an overlay, not a second scheduler.** `epoch_hook` fires before
`lr_scheduler.step()`, so writing an absolute rate there would let `ExponentialLR` compound its
decay on top of the warmup and the next epoch's write discard it — silently costing
`gamma^warmup` of the schedule. The factor is removed before it is re-applied, so the scheduler
owns the trajectory throughout. **Known edge, noted and not engineered around:** resuming
mid-warmup resets the factor to 1.0 while the checkpointed lr still carries the old one.

**`loss_gap` is selected by composition, never by a label.** It applies to the stoichiometric
graphs already in the batch — no extra forward — with `gap_steps_with_term / gap_steps_total`
recorded, because realised coverage is a property of the shuffle rather than a constant. It
differs from the harness, which draws a dedicated pristine batch; that difference is documented
at the call site.

The first selector had a real bug worth recording: it took units-per-cell from the most
populous species, so a 79-atom V_Cl cell gave 47/3 = 15.667 units, expected
(47, 15.667, 15.667), and the real (47, 16, 16) sat a third of an atom away — inside a
half-atom tolerance. **The defect hides in the species you pivot on.** Units now come from the
total atom count and the tolerance is 1e-3, because these are exact integers and anything
looser only admits near-misses, which is what a vacancy is.

### The bug the c-shift caught, and it was the serious one

The deterministic calibration was introduced so the head's energy zero would not depend on the
shuffle. It did something more useful: it disagreed with the harness by a factor of four on
identical data — **+8.70 eV** against **+36.54 eV** — and that disagreement turned out to be a
direct read on a broken base branch.

`c_shift` is the median of `(E_label − E_base − E_head)/Δn`, so it measures `E_base`. Each
candidate was eliminated by measurement:

| candidate | measurement | verdict |
|---|---|---|
| frame selection | raw ratio +3.756 over the first 48 vs +3.587 across the file, 5–95% span 0.6 eV | harmless |
| forward path | `ForwardContext` vs `batch.to_dict()`: `base_energy`, `delta_sr` and the c-shift agree to **0.000000 eV** | not it |
| per-(charge,size) referencing | exactly −1.200 eV, zero spread across frames | not it |
| head initialisation | zero-init moves the c-shift by 0.012 eV | not it |
| atomic energies | identical between source models; their difference from the trainer's regressed values cancels to −0.002 eV on this composition | not it |
| **trunk normalisation** | Stage A **14.08**, the joint run **112.5** | **this** |

The forward-path row is the important negative: train and evaluate do *not* disagree about the
forward pass, which is a failure this project has already paid for four times.

**`avg_num_neighbors` divides every message in the trunk, and it is neither a parameter nor a
buffer** — a plain float on each interaction block. So it is absent from `state_dict`, the
loader's name-and-shape check cannot see it, and its copy loop cannot carry it. Stage A trained
at `r_max = 5.0` without a carrier head and got 14.08; any run with the head builds its graph at
the **carrier** cutoff of 10 Å and computes 112.5 on the same data. Eight times the divisor on
every message, in the branch whose whole purpose is to be the reference the correction is
defined against — and `load_stage_a_base` reported success.

**Two repairs, because they cover different runs.** A Stage-B run now inherits the checkpoint's
value, with a loud warning and an outright refusal if the block counts differ. A *from-scratch*
run has no checkpoint to inherit from — and the joint run's arm B is exactly that — so the count
itself is rescaled by the measured edge-count ratio at `r_max`, averaged over batches, rather
than by a `(10/5)³` volume argument that the periodic per-frame graph does not obey.

**The two repairs cross-validate, which is what makes this a diagnosis rather than a patch:**

```
computed on the 10.0 A carrier graph            111.57
rescaled by the measured edge-count ratio 0.1248  13.93
Stage A, computed at r_max = 5.0 on its own data  14.08
```

1.1% apart, from completely different routes — a ratio taken on the carrier graph of one
training set against a full pass at `r_max` on another — and `1/ratio = 8.01` against the 8×
the two cutoffs predict.

**Scope, checked rather than assumed.** The arch model the harness builds from carries 14.14,
essentially Stage A's 14.08, so the Stage-1..3 lineage and every number in this report are
unaffected. The fault is confined to the production trainer, which had never been run end to
end with a carrier head and a Stage-A base before this cycle — which is what §3 existed to find.

**The prediction this makes, recorded before the number arrived:** with the normalisation
repaired, the trainer's c-shift should fall from +36.54 to near the harness's +8.9, because the
entire 27 eV gap was `E_base` being wrong.

**Confirmed.** On the relaunched joint run the trainer calibrates to **+8.9600 eV** over all 944
charged frames.

| | c-shift | frames |
|---|---|---|
| trainer, before the fix | +36.5421 | 944 |
| **trainer, after the fix** | **+8.9600** | **944** |
| b12, independent, stride sample | +8.94 | 118 |
| harness, first 48 | +8.70 | 48 |

The two drivers now agree on the head's energy zero to 3%, and the residual is the frame set —
b11 measured the raw ratio moving by 0.17 eV between the first forty-eight and the whole file,
with the head term accounting for the rest. The 27.6 eV was `E_base`, exactly as diagnosed, and
the quantity that exposed the bug is the one that now certifies the repair.

**What is not claimed: step-level weight identity between the two drivers.** They batch
differently — a fixed graph list versus a shuffled DataLoader — so their weights after one
epoch differ for reasons that have nothing to do with the protocol, and chasing that number
would be measuring the shuffle. The test file says so at the top.

**839 tests pass**, 28 of them new, 1 skipped. The 16 collection errors in the same run are
`test_eager_benchmark` / `test_compile_benchmark` asking for a `benchmark` fixture that
`pytest-benchmark` would provide; it is not installed here, and that is unrelated and
pre-existing.

### Two bugs the end-to-end run found that no unit test could

**`--defect_base_init` and `--defect_madelung_on_site` could not be combined at all** — the
joint run's own configuration. `CORRECTION_PREFIXES` omitted `madelung`, so `_base_state`
classified the learned species charges as *base* weights and `load_stage_a_base` demanded that
a Stage-A checkpoint contain `madelung.z`, which no Stage-A base will ever have: Stage A has no
Madelung term. Every unit test passed throughout, because each exercised one flag. The
codebase already knew the answer in one place — `defect_protocol.trainable_mask` works around
the same omission with an explicit `startswith("madelung.")` — and tripped over it in another.
This is the first time the production trainer has been run end to end with the joint run's
flag combination, and it is what that was for.

**Twelve tests had been silently skipped.** `tests/unit/test_spectral_v3.py` imports a sibling
by bare name while `tests/unit/__init__.py` makes the directory a package, so pytest puts the
repo root on `sys.path` and not the directory. Collecting the suite by directory raised
`ModuleNotFoundError` at import and took the whole module with it. Fixed to a package-path
import; the twelve now run and pass.

### The two-epoch production-trainer smoke

Two identical invocations of `mace.cli.run_train` through the launcher, protocol on, at the
final config (γ = 3, Gaussian 0.05, exp envelope L = 1.0, `loss_gap` weight 1.0 with
composition 3:1:1, head-only, Stage-A base frozen).

Every call site fires, on real data, and is visible in the log:

```
Stage-3 protocol: Harrison initialisation at bond length 2.861 A
Stage-3 protocol: head-only, 29 parameter tensors frozen
WARNING: Stage-3 protocol: c-shift NOT calibrated -- no frame in the first batch carries
         a net carrier ... the head starts at c = 0, which is a choice this run did not
         make deliberately
Stage-3 protocol: init gate edges 0.258/0.073 eV (need <= 1.20), bandwidth 32.84 eV
         (need >= 4.80) -> PASS
Stage-3 protocol: {"c_shift_calibrated": ..., "e_gap": 2.4, "harrison_init": true,
         "smearing_family": "gaussian", "smearing_width": 0.05, "stage": 3, "warmup": 5}
Stage-3 warmup: epoch 0, lr x0.200
Stage-3 warmup: epoch 1, lr x0.400
```

The c-shift line is the protocol behaving correctly and it is worth reading twice. This
particular fold's first batch is all neutral, so `Delta_n = 0`, the ratio is undefined, and
the function returns `None` rather than a silent `0.0` that would look like a calibration
that had happened. It warned, and the run continued.

**And that is how the third bug surfaced.** The summary on the same line said
`"c_shift_calibrated": true` — because it reported `stage >= 3`, the intent, rather than the
outcome. A run that skipped its calibration was about to record in its own artefact that it
had performed one. Fixed to report what happened and the value; unit-tested both ways.

**The repeat floor and the round trip, both green.**

```
read off the saved model: gamma 3.0  hop_range 0.5  envelope exp  decay_length 1.0
                          family gaussian  width 0.05
SUMMARY MATCHES THE SAVED HEAD
CONFIG ROUND TRIP OK

run a: [3.75392417, 3.75182575]
run b: [3.75392417, 3.75182575]
max |difference| 0.000e+00  -> within the repeat floor
```

Not merely within the floor — **bit-identical**. In float64 with cuEq off, two invocations of
the production trainer reproduce each other exactly across both epochs, so any later
difference between runs is a change and not the scatter-atomics noise. That is a stronger
statement than section 3 asked for and it is worth having before eight seeds are spent.



---

## 4b. The modulation ceiling: F11 and F12

b7's null is scoped, and the scope matters. It varied the **global** radial envelope at
initialisation, where the learned environment modulation is identically 1 by construction —
so it measured what a uniform, host-wide change does, and found every candidate worse. It
never exercised the **defect-local** lever, which is the only one that can raise the hub
coupling without inflating the bandwidth everywhere. b3 had already found that lever at its
stop on a third of hub bonds.

### F11 — the cohort splits, and both stops are in use at once

`t = v0 · radial(r) · (1 + hop_range·tanh g)`, so `tanh g` is recovered exactly as
`(t/(v0·radial) − 1)/hop_range`. A bond is at its stop when `|tanh g| > 0.98`.

| bond type | mean tanh g | at bound | which stop | 4.5–5.5 Å | 5.5–6.5 Å | 6.5–8.0 Å |
|---|---|---|---|---|---|---|
| ss-σ | −0.486 ± 0.480 | 33.3% | **lower**, 100% | 33% | 33% | 33% |
| sp-σ | −0.288 ± 0.496 | 0.0% | — | 0% | 0% | 0% |
| pp-σ | +0.136 ± 0.651 | 33.3% | **upper**, 100% | 33% | 33% | 33% |
| pp-π | +0.601 ± 0.358 | 33.3% | **upper**, 100% | 33% | 33% | 33% |

The forecast was "upper stop, concentrated in short-d bins where the dimer forms". Half right
and half wrong, and the wrong half is the informative one.

**Both stops are in use on the same bond.** ss-σ is pinned at the *lower* bound while pp-σ and
pp-π are pinned at the *upper* one — the head wants less s–s overlap and more p–p overlap than
the bound allows, simultaneously. A uniform scale factor on that bond, which is what the
what-if applies, therefore pushes one channel the right way and another the wrong way.

**It is not concentrated in d; it is concentrated in seeds.** The 33.3% ± 47.1 is exactly two
of six, and in those two it is 100% of hub bonds in *every* distance bin, while the other four
are interior everywhere (|tanh g| ≤ 0.79). That also reconciles b3's pooled "33% at the bound"
with the per-seed picture: it was never a third of the bonds, it was a third of the seeds.

### F12 — the what-if, and the branch it selects

The hub edge's four integrals scaled by ×1.25 and ×1.5 in the trained models, reading the
79-atom force loss and the 159-atom F4 slope.

| scale | force loss (79) | fell in | F4 (159) |
|---|---|---|---|
| ×1.00 | 0.00065 ± 0.00002 | — | −0.0489 ± 0.0068 |
| **×1.25** | 0.00064 (**−0.8%**) | **4/6 seeds** | **−0.0675 ± 0.0127** |
| ×1.50 | 0.00068 (+5.5%) | 4/6 seeds | −0.0887 ± 0.0204 |

Per seed, and this is where the pooled "4/6" turns out to understate what happened:

| seed | at bound (ss/sp/ppσ/ppπ) | force loss ×1.0 | ×1.25 | change | F4 ×1.0 | ×1.25 | ×1.5 |
|---|---|---|---|---|---|---|---|
| 1 | **100/0/100/100** | 0.00062 | 0.00066 | **+6.5%** | −0.0614 | −0.0904 | −0.1233 |
| 2 | **100/0/100/100** | 0.00063 | 0.00065 | **+3.2%** | −0.0494 | −0.0758 | −0.1066 |
| 3 | 0/0/0/0 | 0.00064 | 0.00062 | −3.1% | −0.0526 | −0.0664 | −0.0818 |
| 4 | 0/0/0/0 | 0.00066 | 0.00063 | −4.5% | −0.0429 | −0.0550 | −0.0684 |
| 5 | 0/0/0/0 | 0.00065 | 0.00063 | −3.1% | −0.0461 | −0.0650 | −0.0865 |
| 6 | 0/0/0/0 | 0.00068 | 0.00065 | −4.4% | −0.0411 | −0.0527 | −0.0658 |

**The split is exact.** The two seeds at the bound are the two the scaling makes *worse*; the
four interior seeds are the four it helps. Not a weak 4/6 majority — a clean partition by
whether the head was already pressed against its stop. The seeds that had spent the bound had
also already taken what the bond could give, and pushing further overshoots; the seeds with
room gain a little. That is a sharper statement of "the stop is not the lever" than the
registered rule itself makes, and it comes from the data the rule was scored on.

The registered rule was *force loss falls **and** F4 moves past −0.08*. The first clause is
met on the pooled mean; the second is not — F4 reaches −0.0675. **F12 fails**, and the branch
is the one written in advance: the stop is not the lever, superexchange moves up the R3 list, and the joint run
proceeds on the current bound. No head-only rerun was spent.

Two things worth carrying forward rather than discarding with the forecast. The direction is
right and consistent — F4 moves monotonically toward the reference under scaling, in every
seed — so the coupling *is* the channel, it is simply not accessible by widening a bound that
only two seeds are against. And the force loss has a shallow minimum near ×1.25 and rises by
5.5% at ×1.5, which says the small cells actively prefer a hub coupling close to what the
head already has.

**Scored on sign and direction only, as registered before the run.** This is a fixed-parameter
intervention on a variational quantity: the occupations re-solve but the rest of the
Hamiltonian is frozen at values fitted under the old coupling, so neither magnitude is a
prediction of what a retrained model would give.

The log-bounded modulation `exp(β·tanh g)` is implemented, tested and **unadopted** — one flag
away if R3 wants it. It exists because the linear form cannot simply be widened: with
`hop_range > 1` it drives an integral through zero and flips the sign the Harrison
initialisation fixed, whereas `exp` is positive everywhere and symmetric in log space, so
β = ln 3 gives ×[1/3, 3] rather than the lopsided [1/2, 3/2].

---

## 5. What this leaves for F4 and the joint run

The battery replaces "F4's amplitude is unexplained" with a decomposition in which every factor
is measured:

1. **Label availability.** 98.4% of the charged frames are 79-atom, and they supply nothing
   usable in this observable: no resolved trend in the region where a null is measurable
   (+0.081 [−0.111, +0.272]), a resolved +0.132 of base error there, and an unresolvable
   mixture of the two in the long-d tail where the full-range +0.364 comes from. The
   remaining 1.6% carry −0.134 against a matched null of +0.080. The head is fitting a
   mixture in which the usable signal is 1.6% of the data and what surrounds it points the
   other way.
2. **Loss coverage.** `delta_sr`'s d-slope was never in the Stage-3 objective. In the fitted
   channel — forces — the head reproduces the labels at both sizes, sign flip included (b8).
   **F4 becomes a fitted quantity for the first time in the joint run**, whose `DefectLoss`
   carries energy terms.
3. **Coupling.** The direct hub bond carries 70% of the head's d-response and the envelope
   supplies a fifth of Harrison's coupling there, with the learned modulation pinned at its
   bound on a third of frames. Real, and not the lever: b7 shows every longer-ranged
   alternative lowers the slope.

**One claim explicitly not made.** The clean derivative at initialisation (+0.3211) against the
trained cohort (+0.2601 ± 0.2535) is the same estimator, and those overlap; the +0.0736
across-frame number is a different estimator and cannot be compared to either. "Training
reduces the head's d-sensitivity" is **unresolved**, not shown. What is shown is only that no
envelope candidate raises it at initialisation.

### R3 close-or-explain, updated

| candidate | status |
|---|---|
| Cl saturation / γ | **closed.** Fixed, and it was not the mechanism (F4 moved −0.0613 → −0.0489, away from −0.131). |
| E_LR | **closed** on sign and magnitude (+0.007 against a required −0.070). Trained version now measured in the staged re-enable: it does not close F4 (§6), and it is what turns the carrier's participation from falling to rising at epoch 12 in seed 1 (7.80 → 7.75 without it, 9.24 → 11.14 with it). JOINT_R3_ELR_SLOT |
| SCC | **closed on sign** by the reviewer's argument; R3 may measure the number. |
| Envelope / coupling range | **closed by b7 (global) and b9 (local).** The envelope fires as a diagnosis and is contraindicated as a remedy; the defect-local modulation ceiling binds in only 2 of 6 seeds and scaling past it misses F12's target. |
| Label availability at 79 atoms | **new, measured, and the largest single factor.** The 79-atom energy residual carries +0.132 of base error and no carrier trend; the 159-atom reference is carrier physics (null +0.08 vs charged −0.134). Not a defect of the head. |
| F4 never being a fitted target | **resolved by the joint run, and the answer is leakage.** F4 became a fitted quantity and the base took it: −0.0713 / −0.0843 against −0.134 in both arms. The next attempt is a decision about the objective (energy-channel large-cell weight, or a frozen base while the head fits energies), not the head. |
| Beyond-two-centre / superexchange | **open, and first.** The hub ablation leaves 30% of the d-response in the indirect channel, F12 removed the direct channel's bound as the lever on the pre-joint cohort, and on the joint models no bond of any type is at its stop while the ×1.25 what-if fails both clauses in 0/6. |
| Centred correction | **deferred to R3** per the decision tree, with b4 *and* the joint run as its evidence: the channel is inert under a forces-only loss, and under the joint objective its constant mode regrew from an exact zero to +0.09…+0.40 eV species constants. An unidentified gauge under both objectives. |

## 6. The joint run

Eight seeds: six with the Stage-A base loaded and trained at 0.1x the head's rate, two jointly
from scratch at equal rates as the staging control. Twenty epochs, batch 8, float64, E_LR
staged in at epoch 12, protocol on, on-site correction zero-initialised, four GPUs at a time.

**Both size upweights, not one.** The charged 159-atom frames carry the only measurement that
separates a bound carrier from a band-like one and are 1.6% of the charged force loss at
natural weight. The *neutral* 159-atom frames are the base's only direct constraint at large d.
Raising the charged seventeen alone would ask the correction to absorb a base error the base
was never given the chance to fix, which is precisely the leakage the adoption rule tests for.
Both realised **25.0%** of their own population's force loss. Recorded for later readers: the
training file holds 16 charged and 15 neutral 159-atom cells; the remaining one of each is in
the validation file, and the seventeen-frame references are measured on train + valid.

### The adoption rule, and why criterion 1 cannot be traded

With the base unfrozen, M1b's +0.36 eV/A small-cell artefact can be removed two ways. The base
can learn the long-d region it used to extrapolate into — that is the point of the joint run.
Or it can absorb the carrier, which improves every aggregate number and destroys the
decomposition the whole programme rests on. **Both look like success in the loss.** They differ
in one place: whether the null-cleared `-0.134` stays put when measured against the trained
model's own base branch. Shrinkage toward `-0.06` is not adopted regardless of total fit.

### The scorer's own null control

`b10_adoption.py` was written and committed before any joint model existed, and run first
against the frozen pre-joint cohort — a model whose answer is known in advance, because it has
not been jointly trained and must not be adopted.

| criterion | pre-joint model | reference | reading |
|---|---|---|---|
| 1 charged 159 energy slope | **-0.1308** | b1: -0.1338 | reproduces |
| 1 charged 159 force slope | **-0.1868** | b1: -0.1901 | reproduces |
| 4 F4 `delta_sr` slope | -0.0624 | within 1.5x of -0.134 | **out of band**, correctly |
| 5 pristine gap | 2.385 eV | 2.40 +- 0.1 | ok |
| 6 depth from CBM | +0.132 eV | 0.10-0.13 | shallow donor |
| | | | **NOT ADOPTED** |

So the machinery and the thresholds both behave on a case with a known answer.

**One scoring subtlety recorded rather than quietly resolved.** Criterion 2's registered
reference is b1's out-of-fold cross-fit null, `+0.0800`; the same neutral slope measured
against the *production* base — the one a Stage-A joint model starts from — is `+0.0968`. Those
are different bases and they disagree. The registered threshold stays as written and the
production-base number is printed beside it, because a joint model is scored against its own
base and that is the like-for-like comparison. On a strict reading the pre-joint model already
fails criterion 2, which is better known before the joint numbers arrive than after.

### How the result will be read, decided before it exists

F13 is withdrawn: it was conditional on F12, which failed. That leaves **F14** as the live
forecast, and under the current bound its clause predicts **leakage** — the 159-atom charged
residual slope shrinking below 0.10 in magnitude. So the two outcomes are both clean, and both
are written down here rather than chosen afterwards:

- **Criterion 1 fails** (slope shrinks toward −0.06): **F14 confirmed**, and the run is *not
  adopted* — the base removed M1b's artefact by absorbing the carrier. That is a real result
  about the joint objective, not a failed experiment, and it says the correction and the base
  are competing for the same signal at these weights.
- **Criterion 1 holds** (slope stays within −0.1446…−0.1230): **F14 fails**, adoption is live,
  and the remaining criteria decide it.

**Criterion 2 is scored against both references, and the composite flag is not the verdict.**
The registered threshold is b1's out-of-fold `+0.0800`; the like-for-like number, measured
against the production base a Stage-A model starts from, is `+0.0968`. The pre-joint model
already fails the registered reading. So the per-criterion table is the result and the boolean
`adopted` is a summary of it — a technicality on criterion 2 must not be allowed to masquerade
as a verdict on criterion 1.

**Both arms are reported side by side.** A from-scratch control that leaks while the Stage-A
arm does not — or the reverse — is a result about *staging*, and will be labelled as one rather
than folded into a single adoption number.

### The run as it happened

**Regime tag for every number in this section:** joint objective (`DefectLoss` with energy,
force and gap terms), 20 epochs, batch 8, float64, lr 0.005, `EVAL_INTERVAL=4`, 128 channels,
`MAX_L=1`, `r_max = 5.0`, carrier cutoff 10.0 Å, γ = 3 eV, Gaussian 0.05 eV, exponential
envelope L = 1.0, linear hop form, protocol on (Harrison init at 2.861 Å, warmup 5), on-site
correction zero-initialised, `loss_gap` w = 1.0 at E_gap 2.4 with composition 3,1,1, both
size upweights 0.25 (both realised 0.25), E_LR staged in at epoch 12, cuEq off. Arm A: six
seeds (a1–a6) from the Stage-A base `e0_base_s1` at 0.1× the head's rate, trunk normalisation
inherited (14.08). Arm B: two seeds (b1 = seed 11, b2 = seed 12) from scratch at equal rates,
trunk normalisation rescaled by the measured edge ratio. This is the *only* trained cohort in
this report besides the smoke; nothing above it trained.

Timeline on b3 (GPUs 4–7, four at a time): two false starts, at 23:17 and 23:52, both killed
for the gauge-block bug below; final launch 00:16; preflight (one epoch on `dataset_cf/fold0`,
required to exit 0 *and* print `Gauge: epoch 0`) passed at 00:36; wave 1 (a1–a4) 00:36:45 →
04:53:32; wave 2 (a5, a6, b1, b2) 04:53:32 → 09:13:17. Ten minutes per epoch, measured
(00:42:42 / 00:53:10 / 01:02:43). All eight exit 0. The scorers ran 09:24 → 10:32 by the
post-run chain, which was written and committed before any joint model existed.

**The training set, for later readers.** `train.xyz` holds 16 charged and 15 neutral 159-atom
cells; the remaining one of each sits in `valid.xyz`. The seventeen-frame references below are
measured on train + valid, the same seventeen b1 used.

### F14 confirmed: the base absorbed the carrier, and the run is not adopted

Criterion 1 is the leakage detector, and it fired on every seed of both arms. The charged
159-atom residual energy slope, measured against each trained model's own base branch:

| arm | charged 159 energy | charged 159 force | neutral 159 energy | neutral 159 force | F4 δ_sr | adopted |
|---|---|---|---|---|---|---|
| pre-joint reference (b1) | −0.1338 [−0.1447, −0.1232] | −0.1901 [−0.2103, −0.1700] | +0.0800 (oof) / +0.0968 (prod. base) | +0.0643 | — | — |
| **A**, Stage-A init, 6 seeds | **−0.0713 ± 0.0076** | −0.2204 ± 0.0125 | +0.1101 ± 0.0122 | +0.0098 ± 0.0040 | −0.0573 ± 0.0155 | **0/6** |
| **B**, from scratch, 2 seeds | **−0.0843 ± 0.0016** | −0.1902 ± 0.0063 | +0.1153 ± 0.0040 | +0.0206 ± 0.010 | −0.0494 ± 0.0014 | **0/2** |

Spreads are one standard deviation across seeds. Every seed's 95% interval on the charged
energy slope lies entirely above the reference interval and entirely above the −0.10 floor,
so F14's clause — "under the current bound the 159-atom charged residual slope shrinks below
0.10 in magnitude" — holds on 8 of 8, and the run is not adopted on criterion 1 alone.

Per seed, arm A (c-shift is the deterministic calibration at the end of training):

| seed | c-shift | charged 159 E slope [95%] | charged 159 F | neutral 159 E [95%] | neutral-79 window E / F (meV/atom, meV/Å; base 1.7 / 10.8) | F4 δ_sr [95%] | depth from CBM | pristine gap |
|---|---|---|---|---|---|---|---|---|
| a1 | +10.96 | **−0.0644** [−0.0746, −0.0542] | −0.1985 | +0.1320 [+0.0625, +0.2014] | 2.0 / 8.8 | −0.0403 [−0.0473, −0.0334] | 0.061 | 2.408 |
| a2 | +9.87 | **−0.0814** [−0.0945, −0.0683] | −0.2321 | +0.1006 [+0.0399, +0.1613] | 3.0 / 9.1 | −0.0416 [−0.0485, −0.0348] | 0.071 | 2.419 |
| a3 | +10.14 | **−0.0708** [−0.0826, −0.0590] | −0.2129 | +0.1098 [+0.0489, +0.1707] | 4.8 / 9.0 | −0.0451 [−0.0519, −0.0384] | 0.045 | 2.414 |
| a4 | +10.49 | **−0.0778** [−0.0884, −0.0672] | −0.2317 | +0.0979 [+0.0360, +0.1597] | 1.9 / 9.2 | −0.0795 [−0.0903, −0.0686] | 0.100 | 2.395 |
| a5 | +10.77 | **−0.0592** [−0.0703, −0.0481] | −0.2160 | +0.1194 [+0.0541, +0.1848] | 1.9 / 8.9 | −0.0707 [−0.0810, −0.0604] | 0.067 | 2.426 |
| a6 | +10.49 | **−0.0740** [−0.0859, −0.0621] | −0.2310 | +0.1009 [+0.0366, +0.1653] | 2.0 / 8.7 | −0.0665 [−0.0759, −0.0570] | 0.059 | 2.392 |

Arm B:

| seed | c-shift | charged 159 E slope [95%] | charged 159 F | neutral 159 E [95%] | neutral-79 window E / F | F4 δ_sr [95%] | depth from CBM | pristine gap |
|---|---|---|---|---|---|---|---|---|
| b1 | +10.81 | **−0.0859** [−0.0962, −0.0757] | −0.1965 | +0.1193 [+0.0401, +0.1986] | 1.6 / 14.1 | −0.0508 [−0.0589, −0.0426] | 0.049 | 2.422 |
| b2 | +9.64 | **−0.0827** [−0.0937, −0.0717] | −0.1839 | +0.1112 [+0.0376, +0.1849] | 9.6 / 14.3 | −0.0480 [−0.0549, −0.0411] | 0.044 | 2.389 |

**All six criteria, scored explicitly, per the registered rule** (the composite boolean is a
summary of this table and not the verdict):

| criterion | rule | arm A | arm B |
|---|---|---|---|
| 1 energy | charged 159 energy slope inside [−0.1447, −0.1232]; shrinkage past −0.10 = leakage | **fails 6/6, leakage 6/6** | **fails 2/2, leakage 2/2** |
| 1 force | charged 159 force slope inside [−0.2103, −0.1700] | 5/6 outside on the *steep* side (−0.213 to −0.232); a1 inside | 2/2 inside |
| 2 | neutral 159 energy slope moves toward zero from the reference | against the registered +0.0800: grew, 6/6. Against the like-for-like +0.0968: pooled +0.1101 ± 0.0122, per-seed intervals ±0.06 wide — **not resolved** either way | grew against +0.0800; unresolved against +0.0968 |
| 3 | neutral-79 window error within 1.1× base (E 1.7 meV/atom, F 10.8 meV/Å) | energy degraded 5/6 (1.9–4.8 vs 1.7; a5 passes, narrowly), force *improved* 6/6 (8.7–9.2) | b1 E passes, F fails (14.1); b2 fails both |
| 4 | F4 δ_sr slope negative and within 1.5× of −0.134 | out of band 6/6 (−0.040 to −0.080) | out of band 2/2 |
| 5 | pristine gap 2.4 ± 0.1 eV | ok 6/6 (2.392–2.426) | ok 2/2 |
| 6 | frontier level stays a shallow donor | ok 6/6, 0.045–0.100 eV below the CBM | ok 2/2, 0.044–0.049 |
| | **adopted** | **0/6** | **0/2** |

One naming defect in the scorer, disclosed rather than patched after the fact: the JSON flag
`c1_leakage` is *true* when the slope sits inside the reference interval, i.e. when there is
**no** leakage. The printed verdict reads it correctly (`OUTSIDE THE REFERENCE CI … SHRUNK past
-0.10: F14 leakage`); the flag name is inverted relative to its meaning and should be read as
`c1_in_reference_ci`. The numbers are the result either way.

**Both arms leak, so this is not a staging result.** The pre-registered reading was that a
leak in one arm and not the other would be labelled a property of initialisation. Arm B leaks
slightly *less* (−0.084 against −0.071), from a base that had never seen the data, so the
Stage-A start is not what lets the base absorb the carrier. What the two arms share is the
objective.

### Plain-language statement

The base network was allowed to train alongside the correction head with energy terms in the
loss. The base learned the distance-dependent energy itself, so the carrier's energy signature
largely left the residual. The overall fit looks good — forces halved, energies unchanged, gaps
intact, the level still a shallow donor — which is exactly why the adoption rule was written on
the residual slope rather than on the fit.

**Forces did not leak; energies did.** The charged 159-atom force slope held or steepened
(−0.220 against −0.190), while the neutral 159-atom *force* slope, which was +0.064 of base
error at the pre-joint null, collapsed to +0.006…+0.015 (correlation 0.1–0.3): the base learned
the long-d force region it used to extrapolate into, which is what the joint run was for. The
force channel had the large cells upweighted to a quarter of the population loss. The energy
channel had no equivalent protection, and it is the energy channel that lost the signal.

**The level became shallower**, 0.10–0.13 eV below the CBM before the joint run to
0.045–0.100 after, consistent with binding energy having moved into the base.

### Final validation errors, and why they are not the verdict

| seed | valid RMSE E (meV/atom) | valid RMSE F (meV/Å) |
|---|---|---|
| a1 | 5.1 | 12.6 |
| a2 | 4.0 | 12.4 |
| a3 | 4.6 | 12.3 |
| a4 | 4.5 | 12.6 |
| a5 | 3.9 | 12.5 |
| a6 | 4.7 | 12.2 |
| **arm A** | **4.5 ± 0.4** | **12.4 ± 0.2** |
| b1 | 9.4 | 16.1 |
| b2 | 6.3 | 16.9 |
| **arm B** | **7.9** | **16.5** |

a1 started at 4.12 meV/atom and 26.83 meV/Å on validation: forces more than halved, energy
slightly worse (it dipped to 3.81 at epoch 12 and rose after E_LR came in). Train against valid
for a1 is 3.5/10.0 against 5.1/12.6. Rule 2 of this programme is that nothing is ranked by
total RMSE, and this table is the reason the rule exists: by it, every seed here is an
improvement on the pre-joint cohort.

### Carrier participation, and the E_LR counterfactual

`partic` in the trainer and `N_eff` in the harness are the same quantity, 1/Σᵢαᵢ² per graph,
measured on carrier-bearing validation frames every fourth epoch; the counting head broadcasts
one α across its four slots, so the four printed values are identical and the "null channel"
ratio the spectral-era gates used is identically 1.000 and is not reported.

| seed | initial | epoch 0 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| a1 | 5.23 | 9.25 | 9.22 | 8.54 | 9.24 | **11.14** |
| a2 | 5.24 | 8.75 | 8.45 | 7.65 | 8.87 | **11.19** |
| a3 | 5.23 | 9.55 | 9.22 | 8.87 | 8.95 | **10.94** |
| a4 | 5.26 | 8.39 | 8.10 | 6.48 | 5.23 | **5.43** |
| a5 | 5.23 | 6.91 | 7.25 | 7.13 | 6.48 | **5.51** |
| a6 | 5.23 | 8.53 | 8.83 | 6.69 | 5.65 | **5.73** |
| b1 | 5.27 | 11.08 | 6.84 | 4.92 | 7.56 | 10.53 |
| b2 | 5.24 | 10.60 | 6.67 | 7.23 | 8.23 | 10.41 |

The six-seed cohort splits three and three, not one against five: a1–a3 end near 11, a4–a6
near 5.5. (An earlier note in this cycle said "one seed of four"; it was written from wave 1
alone.) The split is not only in participation. s3_dilution's median depth on the sixteen
charged 159-atom training frames is 0.011 / 0.015 / 0.013 eV for a1–a3 and 0.063 / 0.060 /
0.059 eV for a4–a6, and the bound fraction is 31 / 38 / 31% against 75 / 69 / 62%: the three
delocalised seeds are the three shallow ones.

**Whether E_LR is what separates them was tested, not inferred.** Because E_LR switches on at
epoch 12, epochs 0–11 of a run with it off are identical to the same seed with it on — same
seed, same data order, every intermediate state. Seed 1 was run with `USE_LONG_RANGE=False` on
the local A4000, and its epoch-12 gauge (c-shift +10.2256) matches joint_a1's to four decimals:

| epoch | seed 1, E_LR off | joint_a1, E_LR on at 12 |
|---|---|---|
| 0 | 9.250 | 9.250 |
| 4 | 9.220 | 9.220 |
| 8 | 8.535 | 8.535 |
| 12 | **7.797** | **9.238** |
| 16 | **7.753** | **11.137** |

Without E_LR the carrier keeps localising and then flattens; with it the trend reverses and is
still climbing at epoch 16, 44% more delocalised. The E_LR-on arm may not be converged at 20
epochs, so 11.14 need not be its endpoint. The E_LR-off seed finished at 4.2 meV/atom and
13.8 meV/Å on validation (train 2.7 / 11.0), against a1's 5.1 / 12.6.

JOINT_PARTICIPATION_SLOT

### The gauge, watched every epoch

The on-site correction started at exactly zero on every seed (`|W_site| 0.00000`, `b_site
+0.00000`, Z = (−1, +1, +2) at epoch 0), and the c-shift on every Stage-A seed started within
3.4 meV of +8.96, the value predicted from the trunk-normalisation repair. Then:

| epoch | a1 c-shift | a1 |W_site| | a1 Z | a4 Z | b1 c-shift |
|---|---|---|---|---|---|
| 0 | +8.9600 | 0.00000 | −1.000 +1.000 +2.000 | −1.000 +1.000 +2.000 | +4.9026 |
| 5 | +9.8098 | 0.01064 | −0.879 +0.765 +1.871 | −0.759 +0.435 +1.843 | +8.4164 |
| 10 | +10.0860 | 0.01167 | −0.822 +0.751 +1.716 | −0.688 +0.455 +1.609 | +10.3303 |
| 15 | +10.8081 | 0.01286 | −0.797 +0.783 +1.607 | −0.624 +0.429 +1.444 | +10.7297 |
| 19 | +11.1135 | 0.01304 | −0.780 +0.783 +1.557 | −0.576 +0.404 +1.322 | +10.8492 |

Three things to read off it. The c-shift is still moving at epoch 19 (about 0.06 eV/epoch on
a1, from 0.33 at the start), so the head's energy zero is not at rest. The neutrality
projection held for the whole run on every seed: 3·Z_Cs + Z_Pb + Z_Cl = 0.000 at epoch 19 on
a1 and −0.002 on a4, some 6 400 optimiser steps each. And b1, from scratch, started its c-shift
at +4.90 and arrived at +10.85, inside arm A's +9.87…+10.96 — the head's energy zero is a
property of the data, not of the initialisation.

**Zero-init removed the gauge at initialisation, and the gauge regrew.** b4's diagnosis of the
on-site channel on the pre-joint cohort was a near-uniform +0.27 eV on every atom with
within-shell spreads of 1–3 meV. The same probe on the joint models (`joint_peratom.json`)
finds the channel has grown back a species constant: pooled corrections hub-Pb +0.26,
ligand-Cl +0.39, bulk-Cl +0.40, bulk-Pb +0.18, Cs +0.09 eV, within-shell pre-tanh spreads
0.0015–0.026, and per seed the constants wander (Cs from −0.19 on a6 to +0.26 on a2). The F10
restatement on the zero-initialised channel: ligand-Cl against bulk-Cl differ by 4.8 meV, not
statistically resolved and far below the 50 meV floor — **F10 fails again**. So §0's third item
did fire on every seed, and it did not hold: the constant mode of the on-site channel is an
unidentified gauge under the joint objective too, which is the evidence the centred correction
carries into R3.

### Dilution (criterion 5's companion), on the joint models

`s3_dilution.py` on the sixteen charged 159-atom training frames, δ_L from four pristine
80-atom cells: R_bound 0.62 / 0.71 / 0.71 / 0.83 / 0.80 / 0.79 for a1–a6, pooled
**0.74 ± 0.07**, gate ≤ 1.3 met by 6/6; bound fraction 51 ± 18%. The same script on the
pre-joint s7 cohort gave R_bound 0.85 ± 0.09, bound fraction 67 ± 7% and median depths of
0.030–0.040 eV on every seed; the joint cohort is less bound on the fraction and splits on the
depth, three seeds at 0.011–0.015 and three at 0.059–0.063 eV. R ≈ 1 is a bound carrier, R ≈ 2
a band state whose hub amplitude halved with the cell, and the script's own caution stands:
anything between is partial binding and not a pass on physics, only on the gate.

### The modulation ceiling on the joint models (F11, F12 restated)

`b9_hub_ceiling.py` on a1–a6, 40 hub bonds each. **No bond of any type is at its stop in any
seed** (0.0% at |tanh g| > 0.98, all four integrals, all six seeds): mean tanh g ss-σ +0.361 ±
0.245, sp-σ +0.092 ± 0.051, pp-σ −0.511 ± 0.097, pp-π +0.283 ± 0.228. The sign pattern is also
not the pre-joint one (there ss-σ sat at the lower stop and pp at the upper). This is a
cohort-to-cohort comparison — fresh heads under a different objective, epoch count and learning
rate — and does not say that training moved the pre-joint seeds off the bound.

The what-if, pooled: ×1.00 force loss (79) 0.00011 ± 0.00001, F4 −0.0567 ± 0.0153; ×1.25
force loss **+24.3%**, fell in **0/6**, F4 −0.0668 ± 0.0170; ×1.5 +69.3%, 0/6, −0.0774 ±
0.0183. F12 fails on both clauses here, more cleanly than on the pre-joint cohort, where the
force loss fell in four seeds. The direction of F4 under scaling is again monotonic toward the
reference in every seed.

### Depth against both edges (F9 restated)

JOINT_DEPTH_SLOT

### F11–F14, scored

| | forecast | outcome |
|---|---|---|
| **F11** | the bound binds at the upper stop, concentrated in the short-d bins | **half right.** Upper stop for pp-σ and pp-π, lower for ss-σ, on the same bonds; concentrated in seeds (2 of 6 at 100% in every bin), not in d. |
| **F12** | ×1.25 on the hub bond lowers the 79-atom force loss and moves F4 past −0.08 | **fails.** Pooled force loss −0.8% (4/6 seeds), F4 −0.0675; the two seeds at the bound are the two it makes worse. On the joint models it fails both clauses (0/6, +24%). |
| **F13** | (conditional on F12) | **withdrawn** with F12. |
| **F14** | under the current bound the joint run leaks: the 159-atom charged residual slope shrinks below 0.10 in magnitude | **confirmed, 8/8.** −0.0713 ± 0.0076 (arm A), −0.0843 ± 0.0016 (arm B), every interval above −0.10. |

### What this says about the next attempt

Not a decision about the head. The joint objective at these weights lets the base and the
correction compete for the same energy signal, and the base wins it because the energy channel
has no large-cell protection while the force channel has. The three options are the obvious
ones and they are about the objective: give the energy channel the same upweight the force
channel got; keep the base frozen while the head fits energies; or both. F4 is now a fitted
quantity, so whichever is chosen can be scored by the same rule on the same seventeen frames.

### Bugs found by running it, in the order they cost something

1. **Gauge-block ordering** (cost: all four wave-1 seeds, ~1 h). The per-epoch gauge log was
   inserted above the line assigning `target`, so epoch 0 raised `UnboundLocalError` on every
   seed. Fixed by moving it below, with the patch asserting file order; and the queue now
   flies one epoch on `dataset_cf/fold0` before spending any seed.
2. `CORRECTION_PREFIXES` omitted `madelung`, so `--defect_base_init` and
   `--defect_madelung_on_site` could not be combined; `madelung.z` then sat in no optimiser
   group and the orphan guard refused the run. Both added.
3. `protocol_summary` reported `c_shift_calibrated: true` when calibration was skipped (it
   reported the stage, not the outcome). Fixed and unit-tested both ways.
4. Criterion 3's baseline was partly in-sample (frames partitioned `i % 4` across the whole
   neutral set, against fold bases trained on three quarters of it). Now drawn from each fold's
   `null_oof.xyz`.
5. The trunk normalisation, §4 above — the serious one.
6. The post-run chain waited on a *process*, so the deliberate relaunch disarmed it (`ABORT: no
   arm-A models were produced`). It now waits on the completion marker in the queue log.
7. All four scorers built float32 batches for float64 models and died together on `both
   inputs should have same dtype` (cost: 12 min). `adopt_model_dtype` is now called per model
   at load in b10, b13, b4, b9, b6 and s3_dilution.
8. `tests/unit/test_spectral_v3.py` imported a sibling by bare name inside a package
   directory, so twelve tests had been silently skipped. Package-path import.
9. b13's null-channel ratio was identically 1.000 (one α broadcast across four slots). Replaced
   by the pristine cell scored with the same counters. Its pristine probe then hit α ≡ 0 at
   `counts = 0` and returned NaN; the harness now keeps a separate pristine pool carrying the
   hole counter.
10. **A fourth liveness fault, in a script written after the lesson.** The E_LR-off control
    queue waited on `pgrep` for the post-run chain. Fix 7 killed and relaunched that chain, and
    in the gap the control saw no process and started at 09:23, concurrent with the scoring it
    was meant to follow. It stayed inside the four-GPU cap by timing, not by design. Recorded
    in the script header; the condition it should have tested is the `post-run complete`
    marker.

### Determinism, twice

Two identical two-epoch invocations of the production trainer were bit-identical
(`max |difference| 0.000e+00`, §4). The stronger result came free from the control: seed 1
trained for a full epoch on the local A4000 and on a b3 Quadro RTX 6000 produced identical
gauges (c-shift +9.2302, |W_site| 0.01286, b_site −0.00004, Z −0.963 +0.911 +1.979), and the
local E_LR-off run matched joint_a1 to every printed digit through epoch 12, the last epoch
before the arms diverge by construction. JOINT_BITID_SLOT The FP64-throughput concern (1/64 on
the A4000 against 1/32 on the RTX 6000) was wrong: both run at 10.0 min/epoch, so the run is
not FP64-bound.

---

### The threat that was checked and did not materialise

The −0.134 reference is measured on 17 frames of which only 4 sit inside the window where the
base's training set is dense, and the 79-atom slope at the same base is demonstrably an
extrapolation artefact — so the reference had to be tested for the same disease before F4
could be made the joint run's first gate. The neutral out-of-fold null does that, and clears
it: +0.08 [−0.05, +0.21] with no carrier, against −0.134 [−0.145, −0.123] with one.

That control is now the thing to re-run first if the dataset or the base ever changes. It is
one command and it is the difference between a gate and a coincidence.
