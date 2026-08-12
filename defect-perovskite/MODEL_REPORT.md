# CsPbCl3 V_Cl: every model trained in this investigation

Fourteen models. The question throughout: can the long-range branch be enabled without the
carrier attention collapsing onto a sublattice, and with accuracy at least matching the
short-range model?

**Answer: yes, by holding the branch out of the energy and the loss until the short-range
objective has settled the attention.** `perov_D_s3` reaches 3.1 / 11.5 against the
short-range reference's 3.1 / 12.0, with attention indistinguishable from it, after thirty
epochs with the branch fully live.

---

## 1. All models

`partic` is the participation ratio of the live carrier channel (h_maj) on a 79-atom V_Cl+
cell: 2 is the two under-coordinated Pb, 16 is the Cs sublattice, 47 is Cl.
Valid RMSE is meV/atom and meV/A. Sublattice counts: 16 Cs, 16 Pb, 47 Cl.

| model | configuration | epochs | partic | on | valid E / F |
|---|---|---|---|---|---|
| `perov_nolr_s1` | short-range only | 140 | **2.01** | **Pb** | **3.1 / 12.0** |
| `perov_lr_s1` | long-range, original | 140 | 16.07 | Cs | 2.7 / 9.7 † |
| `perov_lr_a4_s1` | Step A + repartition | 38 ‡ | 16.10 | Cs | n/a |
| `perov_A0_s1` | + site-resolved gate | 19 ‡ | 15.22 | Cs | n/a |
| `perov_A1_s1` | + detached `q^host` | 6 ‡ | 13.11 | — | n/a |
| `perov_Z_s1` | control, `a = 0` | 22 | 46.95 | Cl | 3.9 / 17.5 |
| `perov_Z_s2` | control, `a = 0` | 22 | **2.01** | **Pb** | 3.8 / 17.1 |
| `perov_Z_s3` | control, `a = 0` | 22 | **1.99** | **Pb** | 3.8 / 16.8 |
| `perov_F_s1` | finite-size `E_LR` | 22 | 15.97 | Cs | 8.1 / 18.4 |
| `perov_F_s2` | finite-size `E_LR` | 22 | 16.07 | Cs | 3.1 / 18.5 |
| `perov_F_s3` | finite-size `E_LR` | 22 | 16.07 | Cs | 5.8 / 18.8 |
| `perov_D_s1` | + branch held to epoch 30 | 60 | 46.71 | Cl | 3.4 / 11.4 |
| `perov_D_s2` | + branch held to epoch 30 | 60 | **1.99** | **Pb** | 3.6 / 11.5 |
| `perov_D_s3` | + branch held to epoch 30 | 60 | **2.02** | **Pb** | **3.1 / 11.5** |

† `perov_lr_s1`'s better numbers are an artefact: its `q^pol` term was near-constant per atom
at fixed cell size, so the fit absorbed it as an energy offset. Its `eps_opt` grew linearly
to -100 eV. Training accuracy must not be used to justify enabling the branch.
‡ stopped at a kill criterion, not run to completion.

## 2. The result

`perov_D_s3` attention, against the short-range reference, on real V_Cl+ frames:

```
D s3     partic 1.97-2.00   Pb 1.000   top Pb 0.560/0.440, 0.512/0.488, 0.532/0.468
D s2     partic 1.93-2.00   Pb 1.000   top Pb 0.595/0.405, 0.510/0.490, 0.548/0.452
nolr     partic 1.96-2.00   Pb 1.000   top Pb 0.568/0.432, 0.511/0.489
```

The two under-coordinated Pb at 2.6 and 2.9 A splitting the weight ~55/45 -- the physically
correct answer for a Pb-derived V_Cl state, and the one the short-range model finds unaided.

Physics gates on `perov_D_s3`:

```
ss_carrier   0.125131 CONSTANT over 639-2159 atoms   (= a^2 sum alpha^2, two atoms at 0.5)
sum_i q_i    0.500000 exact at every size
q^pol terms  identically zero
carrier^2    E = E_inf + k/L,  k = -4.582 eV.A,  max residual 0.471 meV, L = 28.6-57.2 A
```

The constant `ss_carrier` is the direct contrast with every failed run, where it decayed as
`1/N` because the weight was spread over a sublattice.

Across the switch-on, `perov_D_s3`: 4.83 meV/atom at epoch 29, 6.11 at epoch 30 as the branch
arrives, **3.30 / 11.93 by epoch 59**. It absorbs the branch and ends at reference accuracy.

**2 of 3 seeds.** Seed 1 was already on Cl before the switch (participation 47.3 at epoch 29)
and stayed there, so it tests nothing about survival -- it never had a correct answer to lose.

Seed 1 is worth one further note: it reaches **3.4 / 11.4**, marginally the best forces of the
three and better than the short-range reference, while sitting on the wrong sublattice. That
is the clearest statement in this whole investigation of why aggregate metrics cannot be the
gate. A constraint-satisfying model with wrong physics looks healthy in every number a
training log reports.

## 3. Why the delayed switch-on works, and what it does not show

The short-range model reliably localises within a few epochs (10.1 -> 4.3 -> 2.6 by epoch 4).
Every long-range run had its attention captured before the energy fit could settle it.
Holding the branch out entirely -- not ramping `a`, the branch is absent, and the model is
bit-identical to a short-range model until it fires -- lets that happen first.

So what D establishes is that **the long-range branch does not destroy a settled correct
answer**. It does not establish that the branch can find one: seeds 2 and 3 entered epoch 30
at participation 2.011 and 2.077, and seed 1 entered at 47.3 and stayed wrong.

## 4. What was fixed along the way

Each was found by measurement and is independently justified.

* **`q^pol` had no locality constraint.** Whole-cell mean subtraction gives neutrality, not
  locality: every atom kept a fixed per-species charge (Cs -0.067, Pb +0.109, Cl -0.015 e),
  flat from 3 A to beyond 10 A. Its Madelung energy and its cross term with `q^host` were
  **97 %** of the size growth (exponents +1.002 and +1.022). Ablated.
* **`host.carrier` is structurally mis-signed.** Under hand-set attention it favours the Cs
  sublattice over the vacancy shell by 1.25 eV against 0.18 eV of short-range difference. A
  classical point-charge potential is deepest at *cation* sites, so it drags a hole onto Cs
  on principle; no rescaling changes a sign. Removed, verified by 8 tests and a live cuEq
  round trip.
* **`carrier^2` carries a spurious in-cell term.** Decomposed at the training cell:
  `A(Pb) - A(Cs) = +0.104 eV` pays to delocalise, while the *total* favours the shell only
  through the physical image term. Implemented as `E_LR = E_periodic - sum_c E_isolated[Q^c]`,
  per channel so electron-hole cross terms survive.
* **The seed anneal gated on gap MAGNITUDE**, which cannot distinguish a species gap from a
  site gap. A run read 24.5 at epoch 1 and lost its seed immediately; its short-range sibling
  reached 6.5 by epoch 5 and kept it. Now gated on `gap_site = mean_Z std_{i in Z} l_i`,
  species-blind by construction, two-pass variance.
* **The delocalised-carrier exemption was a trap** -- flat attention drives the contrast to
  zero, the channel is exempted, the hinge switches off. Guarded, with a fixed localisation
  target rather than a `|c|`-derived one, which measured a penalty of exactly 0.0 in float32.
* **`q^host` was pinned at zero** by a zero-init sitting on a stationary point of a quadratic
  term. Never once exercised in any run before today.
* **The `k = 0` background** was missing, then wrong by exactly `2 pi`: LES folds the `2 pi`
  into `norm_factor`, so the coefficient is `1/2`, not `pi`. -0.48 meV at 639 atoms.

## 5. Open

* **`q^host`'s sign is unidentifiable.** With `host_carrier_coupling=False` nothing in the
  objective is linear in `q^host`, and `E_LR` is quadratic, so `q -> -q` leaves the energy
  unchanged. The F arm settled on physically inverted charges (Cs -0.68, Pb -0.32, Cl +0.34).
  Energies are unaffected; `q^host` is not currently a physically meaningful quantity.
* **The F arm is confounded** -- it changed `carrier_self_isolated` *and* `q^host` being alive
  at once, so its 0/3 result is not a verdict on the finite-size redefinition.
* **Runs are not bit-reproducible at fixed seed.** D and Z share the first 30 epochs by
  construction and matched exactly at epoch 0, but seed 3 diverged by epoch 5 (7.79 vs 1.92)
  through GPU nondeterminism amplified by the sensitivity. "2 of 3" is coarser than it looks.
* **`L_consist` is contraindicated**, and the measurement is worth keeping: every model that
  gets the physics right has `softmax(-beta u)` putting 10^3-10^4 times too little mass on
  the shell, and `nolr` would be targeted onto Cl at participation 49.5. `u` is constrained
  only on `alpha`'s support, so it is not a site-energy field defined everywhere.
* **This dataset cannot validate the branch.** No paired frames, so the `alpha`/`u` degeneracy
  is essentially unbroken and `nolr` localising at all is marginal. SiC, with per-atom
  delta-force supervision, is the validation.
* Not yet run on D: the relaxed size ladder, `eps_opt` drift, the transition level, and a
  re-measured `dilute`.
