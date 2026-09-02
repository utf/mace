# Madelung-in-H: D-series, Stage 1, and the λ–d anomaly turning over

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** The direction change is built and Stage 1 has run. All five Stage-1 gates pass.
Two results worth your attention: the λ–d anomaly we flagged as an open architectural
question **turns over and becomes physical** once the host Madelung potential is in `H`, and
the learnable per-species charges `Z` turn out to be **worse than useless** — pinning them at
the formal charges improves every headline number and removes a two-seed divergence.

One measurement went against its prediction and one came back null, both pre-registered.

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

### The one thing we want you to decide: `Z` should probably not be learnable

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

**We have not applied this.** The plan says learnable, so learnable is the arm of record, and
Stage 2 is running both arms so the answer is complete whichever way you decide. The two
candidate forms are: fix `Z` at formal, or make it bounded in Edit 3's own idiom,
`Z[s] = Z_nominal[s] + δ·tanh(·)`. The second is more consistent with the spec's philosophy
of bounded corrections over physical scales; the first is what the data supports today.

---

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

1. Confirm or overrule the `Z` amendment, and if `Z` stays learnable, whether the bounded
   `Z_nominal + δ·tanh` form is the one you want.
2. Whether the λ–d turnover is enough to close the ledger item or whether you want the
   archived cohort's negative explained before we rely on `Δ_bind` as a depth.
3. Anything you want measured before Stage 3, given that D-1 came back null and the D-2
   resonance prediction was refuted — the two diagnostics that were meant to inform it.
