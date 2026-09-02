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

## What Stage 3 needs before it can be read

1. **Fix the failure-to-start.** Two of six seeds is too many. Candidates, cheapest first: a
   short warmup on the on-site terms before the hoppings move; gradient clipping tuned to the
   eV-scale initial loss rather than the 0.003-scale converged one; or an initialisation that
   puts `E_head` nearer zero at epoch 0.
2. **Re-run with the failures fixed**, six seeds, same control.
3. **The two gates still unmeasured**: the pristine frontier gap against E_gap ± 0.1 eV (now
   persisted per seed by the harness, but these runs predate that), and
   `corr(dE_head, d)` on the 159-atom subset against −0.134 eV/Å (`s3_dehead_trend.py`,
   written, not yet run).
