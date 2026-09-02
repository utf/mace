# Madelung-in-H: D-series, Stages 1 and 2, and the fit that was the superatom

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** The direction change is built through Edit 4 and Stages 1 and 2 have run.

Stage 1 passes all five gates, and the λ–d anomaly we carried as an open architectural
question **turns over and becomes physical** once the host Madelung potential is in `H`.

Stage 2 is the important one. Bounded elements pass the superatom gate decisively and fail
force parity badly, and those turn out to be **the same fact**: across 12 seeds,
`corr(axial_red, split_fraction) = +0.987`, strictly bimodal. Where the pristine spectrum
comes out as bands, the head fits forces no better than the base it corrects. The force fit
this project has been getting was the superatom fitting forces.

Read the Stage-1 section first — several of its numbers are contrasts that survive, and one
of its conclusions Stage 2 withdraws.

One diagnostic came back null and one prediction was refuted, both pre-registered.

---

## D-series (existing checkpoints, no retraining)

**D-1 — sensitivity audit: null, as pre-registered.** `‖∂H/∂R_j‖` split defect-near from
bulk-like by block-1 descriptor distance to the pristine per-species median gives a ratio of
**1.01** in both the localised and delocalised cohorts, per-model range 0.98–1.05. A
within-frame rank split gives 0.73–0.97. H's geometric sensitivity is uniform over the cell,
and it stays uniform at N_eff = 1.29 as firmly as at N_eff = 50.85 — *the matrix is not
concentrated anywhere even when the eigenstate is*. Per the plan this does not gate Edit 3.

**D-2 — the resonance explanation is refuted.** We predicted the λ–d anomaly would localise
to resonant frames. Measured on the frames each model's own spectrum marks bound
(`depth/δ_L > 2`), `corr(λ, d_hub)` is negative in **17 of 18** models, mean −0.422 — and
**13 of 18 models have no resonant frame at all** while still showing it. Not a
near-degeneracy artefact.

Error does fall from the resonant to the bound bin (36.4 → 28.8 meV/Å) while the *base* error
stays flat (33.8 → 34.2), so the head improves on bound frames and is slightly worse than
doing nothing on resonant ones — the predicted direction, but on 18 resonant frames from 5
models, so suggestive only.

---

## Stage 1 — Edit 1 (Madelung on-site) + Edit 2 (response channel deleted, same commit)

Head-only retrain, 48 charged frames, 40 epochs, force-only loss, six seeds per arm. The
control is an OFF arm rather than the archived cohort, because those trained under T-B's edge
and gap losses and comparing to them would conflate the edit with the loss change.

| arm | axial_red | rmse_all | N_eff | null ratio | corr(λ,d) | φ-binned gap |
|---|---|---|---|---|---|---|
| OFF | +0.452 ± 0.189 | 35.5 ± 7.0 | **19.3** | **0.265** | +0.020 | +3.8 |
| ON, learned `Z` | +0.514 ± 0.240 | 32.5 ± 7.2 | 43.0 | 0.803 | **+0.422** | +2.0 |
| **ON, nominal `Z`** | **+0.627 ± 0.184** | **30.4 ± 6.6** | 49.8 | 0.790 | +0.270 | **+1.2** |

**Gates.** A1's toy tables (rock-salt donor well, perovskite band order, the G=0 convention
checked by Euler's identity against the energy kernel, eval-mode finite-difference forces) —
13 tests, pass. A2's pinned continuum, restated for Edit 1 — pass. The φ-binned error gap
closes 3.8 → 1.2 meV/Å. Force parity improves rather than degrades. `E_resp` deleted in the
same commit as Edit 1 landed.

### The λ–d anomaly changes sign

| cohort | `corr(λ, d_hub)` |
|---|---|
| archived (T-B loss, no Madelung) | −0.422, negative in 17/18 |
| Stage-1 OFF (force-only, no Madelung) | +0.020 |
| Stage-1 ON, learned `Z` | +0.422, positive in 5/6 |
| Stage-1 ON, **nominal `Z`** | +0.270, positive in 5/6 |

Positive is physical: the level is deepest when the pair dimerises, which is what R-A
predicted and never found.

Two statements we are keeping apart. The **loss change alone** removed the negative, so the
anomaly was never purely architectural — it was at least partly a property of training
against the band-edge constraint. The **Madelung term** then drives the correlation positive,
and that comparison is one edit apart on identical seeds.

That it survives at *nominal* charges is the stronger form: the physical trend comes from the
Madelung contrast, not from an optimiser finding the right `Z`.

We would not yet call this explained. Neither cohort says why the archived one was negative,
and "removed" is not "understood". But the cheapest explanation is gone and the anomaly does
not survive either change, so it is no longer the same open item.

### `Z` looked unlearnable after Stage 1 — Stage 2 withdrew this (see below)

The ON arm is bimodal and splits by `|Z|`:

| seed | `Z` (Cl, Cs, Pb) | axial_red | rmse_all |
|---|---|---|---|
| 1 | (−2.57, +2.33, +5.38) | +0.170 | 42.8 |
| 2 | (−2.59, +2.35, +5.42) | +0.184 | 41.6 |
| 3–6 | −0.31 to −1.59 | +0.63 to +0.72 | 25.6–31.7 |

Neutrality holds to 1 part in 10⁷ on every seed, so the projection is exact. What is missing
is any constraint on the **scale**: the smooth part of `phi_LR` is dominated by a per-species
constant the learned on-site term can also produce, so only the small defect-induced
deviation constrains `|Z|`. The plan's identifiability argument is right in principle and too
weak in practice.

Pinning `Z` at the formal charges improves everything — axial_red +0.514 → +0.627, force
32.5 → 30.4, the φ gate 2.0 → 1.2 — with a *narrower* seed spread.

**We never applied this**, and Stage 2 says not to: under bounded elements the two `Z` arms
are indistinguishable and `Z` stops running away, so the pathology was an interaction with
the *unbounded on-site term* rather than a property of `Z`. Left in place because the
reasoning is what the next such runaway will need.

---

## Stage 2 has since run, and it is the result of the day

Edit 3 on top of Edit 1, same seeds, same data. **Its superatom gate passes decisively and
its force-parity gate fails badly — and those are the same fact.**

| group (12 seeds, both `Z` arms pooled) | n | pristine split fraction | axial_red | rmse_all |
|---|---|---|---|---|
| bands | 8 | 0.10–0.14 | **+0.030** | 50.8 |
| superatom | 4 | 0.74–0.78 | **+0.408** | 42.0 |

`corr(axial_red, split_fraction) = **+0.987**`, strictly bimodal, no seed in between. Where
the pristine spectrum comes out as bands, the head fits forces no better than the frozen base
it corrects (299.6 against 305.2 meV/Å on the axial component). Where it is still a superatom,
it fits.

The Stage-1 heads say why. Their learned decay lengths reach **3.3 Å**, giving
`t(5.6 Å)/t(2.8 Å)` of 60 % (ON) and 50 % (nominal `Z`) against the plan's own 5–20 % target,
with two cells **above 100 %** — hopping that increases with distance. Stage 1's +0.63 was
bought with the near-complete graph E2 identified, and the split fraction of 0.76 was
reporting it the whole time.

This does not retract Stage 1's contrasts — the φ gate, the λ–d sign change and the `Z`
comparison are all between arms trained under identical conditions, with the escape route
equally available to each. It retracts reading Stage 1's absolute axial_red as evidence that
the head had found physical structure.

**The λ–d turnover survives, and strengthens.** Under bounded elements `corr(λ, d_hub)` is
**+0.439, positive in 10 of 10** measurable cells, against Stage 1's +0.422 in 5/6. So the
physical direction is not a by-product of the escape route — it holds most uniformly exactly
where the route is closed. D-1 stays null here too (ratio 1.06).

**It also withdraws our own `Z` amendment.** Under bounded elements, learned and nominal `Z`
are indistinguishable (+0.142 vs +0.170) and `Z` shrinks rather than running away. The
Stage-1 runaway was an interaction with the unbounded on-site term. We would not now press
the change.

**What we have not concluded** is that the s-only basis is the reason. Edit 4 exists because
one orbital per atom cannot represent the p-derived valence band this hole lives in, and its
toy tables already pass. Stage 2 is a reason to run Stage 3, not to stop. What would be wrong
is loosening Edit 3's bounds until the fit returns — that buys back the superatom, and we
would be measuring it again in a month under another name.

## Built and gated, not yet run

* **Edit 3** (bounded elements): `t = V0[s_i,s_j] f(r) (1 + ½tanh g)`, `eps = eps0[s] − φ/ε∞
  + 1 eV·tanh h`, Harrison-scale init, `t_min` and the learned per-species decay both removed
  — the latter because M3 identified a floored learned decay as exactly the mechanism that
  produces a prior-dominated `t'`. Stage 2 is running now. Its superatom gate has a baseline
  to beat: Stage 1's V3 head reads split fraction **0.76** against an even-spacing reference
  of 0.20 on a *defect-free* cell.
* **Edit 4** (counting head): built, with the SK toy tables passing (19 tests) before any
  training. Writing them corrected two of our own statements — the hole does **not** reverse
  the hub–hub force (Perron–Frobenius says inward at any occupancy; what is outward is the
  hole's *difference* from the filled reference, which is exactly `E_head`), and the sp node
  check has to be read off the 2×2 block because the full 8×8's extremal states are exactly
  degenerate.
* **The forward-context object** with its cross-path N_eff test, so train/evaluate/capture
  cannot disagree about the forward pass. That bug has now appeared four times in four
  disguises.

## What we are asking

1. **Whether Stage 2's result changes the plan.** Our reading is that it does not — Stage 3
   is exactly the response to "a physical s-only Hamiltonian cannot fit these forces", and
   the counting head is built and its toy tables pass. But this is the first time a gate
   failure has pointed at the basis rather than at an implementation detail, and it is your
   call whether that warrants a step back.
2. Whether the λ–d turnover is enough to close the ledger item, or whether you want the
   archived cohort's negative explained before we rely on `Δ_bind` as a depth.
3. The `Z` amendment is **withdrawn** by Stage 2 — no decision needed unless you disagree.
