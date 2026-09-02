# Stage 3 — Edit 4 (electron-counting head, uniform s+p). Not conclusive, and here is why.

Six seeds, matched against a Stage-2 arm re-run at the same learning rate so the only
difference is the edit. 48 charged frames, 60 epochs, lr 0.05, `loss_gap` on.

---

## The headline is an optimisation failure, not an architecture verdict

| arm | n | converged | axial_red (converged) | rmse_all | N_eff | split |
|---|---|---|---|---|---|---|
| Stage 2 control (s-only, bounded) | 6 | **6/6** | **+0.011 ± 0.001** | 50.8 ± 0.2 | 23.5 | 0.14 |
| Stage 3 (counting, s+p) | 6 | **4/6** | +0.044 ± 0.080 | see below | — | 0.07 |

Per seed:

| seed | axial_red | rmse_all | final force loss | split | N_eff |
|---|---|---|---|---|---|
| 1 | **+0.065** | **50.7** | 0.0027 | 0.012 | **13.1** |
| 2 | −0.018 | 55.6 | 0.0032 | 0.004 | 13.2 |
| 3 | −0.688 | 201.1 | **2.28** | 0.059 | — |
| 4 | −0.118 | 134.2 | **3.34** | 0.214 | — |
| 5 | +0.162 | 121.5 | 0.029 | 0.004 | 45.5 |
| 6 | −0.033 | 61.7 | 0.0039 | 0.135 | 10.4 |

**Seeds 3 and 4 never left their initial plateau** — final force loss 2.28 and 3.34 against
the others' 0.003. That is the same failure the lr-0.01 launch showed across every seed, now
appearing in a third of seeds at lr 0.05. It is a failure to start, not a converged bad
answer, and it is why the arm mean (−0.105) is meaningless and is not quoted as a result.

Two thirds of a screen converging is not a basis for an architecture claim in either
direction. **Stage 3 as run does not settle whether s+p fixes the Stage-2 failure.**

## What the converged seeds do say

**At matched force fit, the s+p head is twice as localised.** Seed 1 reaches
`rmse_all` 50.7 — indistinguishable from the Stage-2 control's 50.8 — with `N_eff` **13.1**
against the control's 23.5, and `axial_red` +0.065 against +0.011. Seeds 2 and 6 sit at
`N_eff` 13.2 and 10.4 on comparable fits.

This is the first change in this programme that has **localised** the carrier rather than
delocalised it. Every previous improvement in fit came with a rise in `N_eff`; three of the
four converged Stage-3 seeds move the other way.

**The ceiling is higher and the floor is lower.** Best converged seed +0.162 against the
control's uniform +0.011; worst converged −0.033. The control's spread is ±0.001 across six
seeds — the s-only bounded head converges to essentially one answer, and the s+p head does
not converge to one answer at all.

## An update to the Stage-2 headline, from the matched control

The `corr(axial_red, split_fraction) = +0.987` we reported was measured at **lr 0.01**. At
lr 0.05 the correlation is gone: the control's six seeds sit at +0.010 to +0.013 across split
fractions from 0.13 to 0.61.

Both halves of this matter and they point the same way:

* the **s-only ceiling of ≈ +0.011 is confirmed and tightened** — competent optimisation does
  not rescue the bounded s-only head, it converges to the same poor fit every time;
* the specific **fit ⟺ superatom identity is weaker than stated**. At lr 0.01 the seeds that
  found the superatom were the seeds that fitted; at lr 0.05 the superatom seeds fit no
  better than the rest. The +0.408 those seeds achieved was reachable *only* through poor
  optimisation, which makes it an artefact rather than a capability.

The conclusion that survives is the one that mattered: **a physically-bounded s-only
Hamiltonian cannot fit these forces**, and it now rests on six tightly-agreeing seeds rather
than on a correlation.

## Two things not to read from the table

**`null_ratio` is 1.000 for Stage 3 and means nothing.** The counting head has no channels —
the same carrier density is broadcast to all four slots so the existing diagnostics keep
working — so the "null" channels are identical to the active one by construction. The
null-channel control does not exist for this head and needs replacing with something else.

**`split_fraction` is not comparable across stages.** Its reference is `1/(n_states − 1)`, and
the counting head returns the whole 4N spectrum (320 states, reference 0.003) where the
spectral head returned 6 (reference 0.20). Stage 3's 0.07 is *above* its own reference;
Stage 2's 0.14 is *below* its own. No Stage-3 seed tripped the 0.30 reseed watch.

## The D-2 readout, which landed after the table above

**`corr(λ, d_hub)` is positive in 6 of 6 Stage-3 models** (+0.11 to +0.44). The turnover first
seen at Stage 1 and strengthened at Stage 2 survives the change of basis, on every seed —
including the two that failed to converge. Whatever else is unsettled, the level's response
to the vacancy-flanking separation is now physical in every arm we have run since Edit 1
landed, across three architectures.

**`δ_L` independently exposes the failed seeds.** The pristine level spacing comes out at
**2.03 and 3.75 eV** for seeds 3 and 4, against 0.087–0.57 eV for the four that converged. A
pristine supercell whose levels are spaced by 2–4 eV is not a band structure. This is a
useful property: the failure is visible in a *label-free* quantity computed from the pristine
spectrum alone, so a screen can reject those seeds without reference to their fit.

**Resonance now discriminates, and in the opposite direction.** The counting head's spectrum
is dense (320 states), so many more frames fall in the resonant bin: seed 2 has 17 of 24,
seed 5 has 20 of 24, seed 1 has 7 marginal and 17 bound. For seed 1 the marginal-bin error is
*lower* than the bound-bin error (45.9 against 53.4 meV/Å) — the reverse of the archived
cohort, where the head was worse than its own base on resonant frames. Too few models to read
as a result; recorded because the bin populations are now large enough to make the comparison
possible at all, which they were not before (18 resonant frames across 18 models).

## A blocker found while running the last gate: the counting head is NaN at 159 atoms

`s3_dehead_trend.py` returned no rows for any Stage-3 model. The cause is not the script:
`delta_sr_energy` is **NaN on every 159-atom frame**, on a model that is finite on all 79-atom
frames. Confirmed directly on three frames, forward pass only, no gradients involved.

This matters more than the gate it blocked:

* the **159-atom subset is the cut that speaks for the labels** (M1b), so the
  `corr(dE_head, d)` gate cannot be measured for the counting head at all until it is fixed;
* **§7's dilution gate is defined on the 17 two-size frames**, and size-invariance is the
  central claim of the whole programme. A head that fails at the larger size cannot be taken
  to the joint run.

Not diagnosed. 79 atoms is 316 orbitals and 159 is 636, so the candidates are `eigh` failing
to converge at that size, an ill-conditioned `H`, or something in the build that only appears
with the larger edge count. **This is the first thing to fix, before the reseed work.**

The gate did produce numbers for the two Stage-2 (s-only) models it could run, and they are
worth recording: slope **−0.0035 ± 0.0022 eV/Å** against the reference **−0.134**, correct
sign in 2/2 but roughly **forty times too small**. The bounded s-only head reproduces the
direction of the label trend and almost none of its magnitude — consistent with everything
else Stage 2 says about it.

## What Stage 3 needs before it can be read

1. **Fix the 159-atom NaN.** Blocks two gates and the joint run outright.
2. **Fix the failure-to-start.** Two of six seeds is too many. Candidates, cheapest first: a
   short warmup on the on-site terms before the hoppings move; gradient clipping tuned to the
   eV-scale initial loss rather than the 0.003-scale converged one; or an initialisation that
   puts `E_head` nearer zero at epoch 0.
3. **Re-run with both fixed**, six seeds, same control.
4. **The two gates still unmeasured**: the pristine frontier gap against E_gap ± 0.1 eV (now
   persisted per seed by the harness, but these runs predate that), and
   `corr(dE_head, d)` on the 159-atom subset against −0.134 eV/Å (`s3_dehead_trend.py`,
   written, not yet run).

---

# NaN triage — F1 falsified, and my earlier diagnosis was wrong

**The NaN is not size-dependent. It is a float32 failure.**

The assertion ladder was run in float64 on one 159-atom charged frame, with a 79-atom
reference, stopping at the first failure:

| assertion | 79 atoms | 159 atoms |
|---|---|---|
| 1. `isfinite(H)` after assembly | pass, (316, 316) | pass, (636, 636) |
| 3. `n_states == 4·n_atoms`, `N_target < n_states` | pass, 316 / N 205 | pass, 636 / N 413 |
| 4. `Tr P == N` to 1e-8; entropy finite | pass, err 3.1e-12 | pass, err 1.2e-12 |

**F1 (the NaN trips assertion 3 or 4) is falsified.** Counting and occupations are exact at
both sizes, and the eigendecomposition is clean. Nothing in the ladder tripped.

The direct dtype test then settled it:

| dtype | 79 atoms | 159 atoms |
|---|---|---|
| float32 | **NaN** | **NaN** |
| float64 | −2.16865199 | −2.25260971 |

**This corrects my report to the coadvisor.** I described the failure as appearing at 159
atoms and not at 79. It appears at both; I had only tested 79-atom frames that happened to
survive. The size correlation was an artefact of which frames I sampled.

**Cause.** The spectrum spans ~17 eV across hundreds of states with `T_el = 25 meV`, so the
occupation is a sigmoid of `(λ−μ)/T` with arguments of order 700. float32 has neither the
range for those exponentials nor the precision for a bisection that must place μ to 1e-10 of
a fixed electron count. The parent spectral head already carries a `solver_dtype` for exactly
this reason.

**Fix:** the eigensolve and occupation solve run in float64 unconditionally, returning to the
caller's dtype. float32 and float64 now agree to seven significant figures at both sizes.

## The two assertions that did fail, and why they are inconclusive here

**[2] tiled-pristine invariance** — on-site energies of the original atoms differ by 0.601 eV,
far more than a φ tolerance. **[5] `F(2N) = 2F(N)`** — 5.7 % off.

Both are confounded on this cell and neither is evidence of a bug. The coupling reach is 10 Å
and the pristine cell is smaller than `2 × r_couple` in the tiled direction, so an atom in the
untiled cell sees its own periodic images and in the tiled cell does not. The environment
genuinely changes, so the learned elements are *not* expected to be bit-identical. This is the
"fails for the wrong reason" hazard the qualification warned about, met in a form the
qualification did not name. A clean version needs a cell already larger than `2 × r_couple`
before tiling, or a shorter reach.

## The unblocked gate — F4 not met

With the dtype fix, the 159-atom `dE_head`–d gate runs on all four converged seeds.

| seed | slope (eV/Å) | corr |
|---|---|---|
| 1 | **−0.0189** | **−0.924** |
| 2 | −0.0127 | −0.598 |
| 5 | **+0.0155** | +0.154 |
| 6 | −0.0083 | −0.185 |
| **mean** | **−0.0061 ± 0.0130** | correct sign in 3/4 |
| bounded s-only (Stage 2) | −0.0035 | −0.31 to −0.66 |
| **label reference** | **−0.134** | −0.989 |

**F4 is not met.** The prediction was right sign and within 3× of −0.134. The sign is right in
3 of 4 (seed 5 is inverted), and the magnitude is 7× short on the best seed and 22× short on
the mean. Against the s-only head's 40× that is an improvement, and seed 1's `corr = −0.924`
is a genuinely strong linear trend — but the head is not reproducing the label slope.

*A correction to my own first pass:* I initially reported that three of the four seeds still
failed this gate after the dtype fix, and inferred a second undiagnosed cause. That was wrong.
Those three model files had never been copied from the compute machine to the analysis
machine — the same sync failure that has now caused four incidents in one day. There is no
second cause.
