# Madelung-in-H: D-series, Stages 1 and 2, and the fit that was the superatom

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** The direction change is built through Edit 4 and Stages 1 and 2 have run.

Stage 1 passes all five gates, and the λ–d anomaly we carried as an open architectural
question **turns over and becomes physical** once the host Madelung potential is in `H`.

Stage 2 is the important one. Bounded elements pass the superatom gate decisively and fail
force parity badly, and those turn out to be **the same fact**: across 12 seeds,
`corr(axial_red, split_fraction) = +0.987`, strictly bimodal. Where the pristine spectrum
comes out as bands, the head fits forces no better than the base it corrects. **The force fit
this project has been getting was the superatom fitting forces.**

One diagnostic came back null and one prediction was refuted, both pre-registered. One
conclusion we drew after Stage 1 we withdraw after Stage 2.

---

## 1. D-series — diagnostics on existing checkpoints, no retraining

Cohort: the 12 A/B cells (delocalised-but-fitting) plus F2's six localised cells.

### D-1 sensitivity audit — **null, as pre-registered**

`‖∂H/∂R_j‖` per atom by Hutchinson probe, split defect-near from bulk-like by block-1
descriptor distance to the pristine per-species median — features, not coordinates, so the
defect assignment never enters the definition.

| cohort | n | ratio near/bulk | within-frame rank ratio |
|---|---|---|---|
| localised (N_eff ≤ 8) | 5 | 1.01 | 0.89 |
| delocalised | 13 | 1.01 | 0.84 |

Per-model range **0.98–1.05**; no model separates. H's geometric sensitivity is uniform over
the cell, and stays uniform at N_eff = 1.29 as firmly as at N_eff = 50.85 — *the matrix is
not concentrated anywhere even when the eigenstate is*. Per the plan this does not gate
Edit 3, and the null was recorded in the script before the run.

Caveat kept with the number: the feature classifier only partly tracks the geometric defect
region (agreement 0.63–0.68 against a 0.5 baseline), which is why a within-frame rank split
is reported beside the absolute one. Both give the same verdict.

### D-2 error binning — **the resonance explanation is refuted**

432 frames. `depth` is the frontier level's distance to the next state, `δ_L` the pristine
supercell's own mean spacing, both from the model.

| bin | n | head RMSE (meV/Å) | base RMSE | head − base |
|---|---|---|---|---|
| resonant, `depth/δ_L < 1` | 18 | 36.4 | 33.8 | **+2.6** |
| marginal, 1–2 | 39 | 32.8 | 36.9 | −4.1 |
| bound, > 2 | 375 | 28.8 | 34.2 | **−5.4** |

Head error falls from resonant to bound while base error stays flat, so the trend is the
head's: it improves on bound frames and is slightly worse than doing nothing on resonant
ones. Predicted direction — but 18 resonant frames from 5 of 18 models, so suggestive only.

**The λ–d prediction fails.** Measured on frames each model's own spectrum marks bound
(`depth/δ_L > 2`), `corr(λ, d_hub)` is negative in **17 of 18** models, mean −0.422, and
**13 of 18 models have no resonant frame at all** while still showing it. Not a
near-degeneracy artefact.

---

## 2. Stage 1 — Edit 1 (Madelung on-site) + Edit 2 (response channel deleted, same commit)

Head-only retrain, 48 charged frames, 40 epochs, force-only loss, six seeds per arm. The
control is an OFF arm, not the archived cohort: those trained under T-B's edge and gap
losses, so comparing to them would conflate the edit with the loss change.

A third arm — `Z` pinned at the formal charges — was added mid-stage after `Z` ran away in
the ON arm. Diagnostic control, not an architecture change.

| arm | axial_red | rmse_all | N_eff | null ratio | corr(λ,d) | φ-binned gap |
|---|---|---|---|---|---|---|
| OFF | +0.452 ± 0.189 | 35.5 ± 7.0 | **19.3** | **0.265** | +0.020 | +3.8 |
| ON, learned `Z` | +0.514 ± 0.240 | 32.5 ± 7.2 | 43.0 | 0.803 | **+0.422** | +2.0 |
| **ON, nominal `Z`** | **+0.627 ± 0.184** | **30.4 ± 6.6** | 49.8 | 0.790 | +0.270 | **+1.2** |

**Gates, all five pass.** A1's toy tables — rock-salt donor well, pristine perovskite band
order, the G=0 convention checked by Euler's identity against the energy kernel rather than
re-derived, finite-difference forces in eval mode — 13 tests. A2's pinned continuum restated
for Edit 1 — 6 tests. The φ-binned gap closes 3.8 → 1.2 meV/Å. Force parity improves rather
than degrades. `E_resp` deleted in the same commit Edit 1 landed.

### The λ–d anomaly changes sign

| cohort | `corr(λ, d_hub)` |
|---|---|
| archived (T-B loss, no Madelung) | −0.422, negative in 17/18 |
| Stage-1 OFF (force-only, no Madelung) | +0.020 |
| Stage-1 ON, learned `Z` | +0.422, positive in 5/6 |
| Stage-1 ON, **nominal `Z`** | +0.270, positive in 5/6 |

Positive is physical: the level is deepest when the pair dimerises — what R-A predicted and
never found.

Two statements we are keeping apart. The **loss change alone** removed the negative, so the
anomaly was never purely architectural; it was at least partly a property of training against
the band-edge constraint. The **Madelung term** then drives the correlation positive, and
that comparison is one edit apart on identical seeds.

That it survives at *nominal* charges is the stronger form: the trend comes from the Madelung
contrast, not from an optimiser finding the right `Z`.

Not "explained" — neither cohort says why the archived one was negative, and removed is not
understood. But the cheapest explanation is gone and it does not survive either change.

### `Z` looked unlearnable after Stage 1 — Stage 2 withdrew this

The ON arm was bimodal and split by `|Z|`:

| seed | `Z` (Cl, Cs, Pb) | axial_red | rmse_all |
|---|---|---|---|
| 1 | (−2.57, +2.33, +5.38) | +0.170 | 42.8 |
| 2 | (−2.59, +2.35, +5.42) | +0.184 | 41.6 |
| 3–6 | −0.31 to −1.59 | +0.63 to +0.72 | 25.6–31.7 |

Neutrality held to 1 part in 10⁷ on every seed, so the projection is exact; what was missing
was any constraint on the **scale**. The smooth part of `phi_LR` is dominated by a
per-species constant the learned on-site term can also produce, so only the small
defect-induced deviation constrains `|Z|`.

We proposed fixing or bounding `Z` and **did not apply it**. Stage 2 then made the question
moot — see below. The reasoning is kept because the next such runaway will need it.

---

## 3. Stage 2 — Edit 3 (bounded elements, floors and learned decay removed)

Same harness, same six seeds, same data. Both `Z` arms run.

**Superatom gate: passes decisively.** On a *defect-free* cell the split fraction
`(λ₂−λ₁)/(λ_last−λ₁)` falls from Stage 1's **0.76** to **0.10–0.14**, against an even-spacing
reference of 0.20.

**Force parity gate: fails badly.** axial_red +0.63 → +0.03; the head sits at 299.6 meV/Å on
the axial component against the frozen base's 305.2 — barely distinguishable from doing
nothing.

### They are one result, not two

| arm | n | axial_red | rmse_all | N_eff | null ratio | split fraction |
|---|---|---|---|---|---|---|
| learned `Z` | 6 | +0.142 ± 0.159 | 47.2 ± 4.0 | 34.5 | 0.880 | 0.336 |
| nominal `Z` | 6 | +0.170 ± 0.199 | 48.5 ± 4.7 | 36.1 | 0.890 | 0.343 |

**`corr(axial_red, split_fraction) = +0.987`** over the 12 seeds, strictly bimodal — no seed
lands between:

| group | n | split fraction | axial_red | rmse_all |
|---|---|---|---|---|
| bands | 8 | 0.10–0.14 | **+0.030** | 50.8 |
| superatom | 4 | 0.74–0.78 | **+0.408** | 42.0 |

The bounded form does not close the escape route completely — four seeds found it anyway —
and **the seeds that found it are exactly the seeds that fit**. There is no seed with a
physical spectrum and a good fit.

### What Stage 1 was actually doing

Learned per-species decay lengths in the Stage-1 heads, against the plan's own calibration
target of `t(5.6 Å) = 5–20 %` of `t(2.8 Å)`:

| Stage-1 arm | max decay length | `t(5.6)/t(2.8)` |
|---|---|---|
| ON, learned `Z` | 1.97 Å (up to 3.26) | **60 %** (14–120 %) |
| ON, nominal `Z` | 2.11 Å (up to 3.18) | **50 %** (9–140 %) |
| OFF | 1.97 Å (up to 2.51) | 24 % (9–35 %) |

Two cells exceed **100 %** — hopping that *increases* with distance. Every arm is outside the
target; the Madelung arms are 2–3× outside. Stage 1's +0.63 was bought with the near-complete
graph E2 identified, and the pristine split fraction of 0.76 was reporting it the whole time.

**This does not retract Stage 1's contrasts.** The φ gate, the λ–d sign change and the `Z`
comparison are all between arms trained under identical conditions, with the escape route
equally available to each. It retracts reading Stage 1's *absolute* axial_red as evidence
that the head had found physical structure.

### Readouts that survive Stage 2

**The λ–d turnover strengthens.** Under bounded elements `corr(λ, d_hub)` is **+0.439,
positive in 10 of 10** measurable cells (learned `Z`) and +0.313, 8/10 (nominal), against
Stage 1's +0.422 in 5/6. The physical direction is not a by-product of the escape route — it
holds most uniformly precisely where the route is shut.

**D-1 stays null** here too: ratio 1.06, rank ratio 0.89.

**The φ-binned gap grows**, 1.2 → 3.5 meV/Å. That tracks the overall fit getting worse rather
than saying anything separate, and is reported because the gate is defined on it.

**The `Z` amendment is withdrawn.** Learned and nominal `Z` are now indistinguishable (+0.142
vs +0.170) and `Z` shrinks during training instead of running away. The Stage-1 runaway was
an interaction with the *unbounded on-site term*, not a property of `Z`.

### The finding, stated plainly

The head's force fit and its superatom spectrum have been the same object. Confine the
Hamiltonian to physical tight-binding scales and ranges, and what remains fits forces no
better than the base potential it corrects.

This is close to what §8 reserves as "the first genuinely architectural finding this design
admits" — with the difference that §8 anticipated a *bound-versus-dilution* disagreement, and
this is a *fit-versus-spectrum* one. Both are disagreements between two measured quantities
on one model, which is what makes them architectural rather than procedural.

---

## 4. Built and gated, not yet run

* **Edit 4, the counting head.** `F_σ(N) = Σ f_k ε_k − T_el S`, μ by bisection and detached
  (Hellmann-Feynman makes that exact, not approximate), `E_head` the difference from the
  neutral fill so it vanishes bit-for-bit at n = 0. Four orbitals per atom, SK angular
  factors, density-matrix route so nothing backpropagates through individual eigenvectors.
  **19 toy tests pass before any training**, per the standing rule.

  Writing them corrected two of our own statements. The hole does **not** reverse the hub–hub
  force — Perron–Frobenius says inward at any occupancy, so "the hole pushes them apart" can
  only be about the *difference from the filled reference*, which is exactly `E_head`. And
  the sp node check has to be read off the 2×2 block, because with sp coupling alone the
  full 8×8's extremal states are exactly degenerate and `eigh` returns an arbitrary member of
  each pair — the hazard the density-matrix route exists to avoid, met first in the tests.

* **The forward-context object** with its cross-path N_eff test. That bug — two code paths
  disagreeing about the forward pass — has appeared four times in four disguises; it is now
  one object with one constructor that refuses the defect masks in production.

Not started: Stage 3 integration (wiring the head into `MACEDefect`, `loss_gap`, validating
the windowed eigensolver against dense `eigh`), Stage 4, the joint run.

---

## 5. Process, since it cost real time today

* **Three machine-drift incidents.** The worst: a save-tag fix was committed but not rsynced
  to b3, so the diagnostic arm overwrote the Stage-1 ON checkpoints under their own names. A
  pre-emptive backup covered it and the recovery is verified. Anything that changes what is
  trained now goes in the filename, and syncs use `--delete` or a deletion never propagates.
* **The archived checkpoints no longer load.** Everything in `~/runs/ab_models/` and
  `~/runs/tbv3_models/` pickles a `CarrierResponse`, which Edit 2 deleted. D-1 and D-2 ran
  before the deletion so nothing is lost; a throwaway stub opens them if ever needed.
* **A2's clause 1 had to move** from `eps` to a new `eps_raw` internal: the difference gauge
  subtracts a per-cell mean, so gauged `eps` differs between two cells by a constant even
  when every learned value is bit-identical. The original test would have failed for the
  wrong reason.
* Eleven long-range tests had been dead since 9 August under the multiplicity guard —
  unrelated to this work, fixed because §6 re-enables E_LR and that branch had no coverage.

---

## 6. What we are asking

1. **Does Stage 2's result change the plan?** Our reading is that it does not — Stage 3 is
   exactly the response to "a physical s-only Hamiltonian cannot fit these forces", and one
   orbital per atom cannot represent the p-derived valence band this hole lives in. But this
   is the first gate failure pointing at the *basis* rather than at an implementation detail,
   and §8 makes it your call.

   What we are confident is wrong: loosening Edit 3's bounds until the fit returns. That buys
   back the superatom, and we would be measuring it again in a month under another name.

2. **Is the λ–d turnover enough to close the ledger item**, or do you want the archived
   cohort's negative explained before `Δ_bind` is relied on as a depth?

3. The `Z` amendment is **withdrawn** — no decision needed unless you disagree.

Ten further implementation choices, all already applied and each a place the plan admitted
more than one reading, are listed in `BUILD_CHOICES.md` for you to overrule if any is wrong.
