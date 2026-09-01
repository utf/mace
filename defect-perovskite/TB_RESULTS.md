# T-A / T-B result — band-edge reference, 1 Sep 2026

All runs complete. T-A on 12 R2 seeds + 8 arm 1b finals; T-B 18 cells across two machines.

## T-A — the diagnosis holds

| set | Delta_bind mean | range | >0.5 eV | escape (gauge-matched) |
|---|---|---|---|---|
| arm 1b epoch-50 (n=8) | **+0.195** | +0.102 … +0.401 | 0/8 | Pb +0.026, Cl +0.002 |
| R2 seeds (n=12) | +0.086 | −0.433 … +0.377 | 0/12 | Pb +0.010, Cl +0.006 |

`gauge_penalty=True` in all 20, so every lambda is absolute and every row is a real binding
energy. Guards: module agreement <= 1.1e-7, lambda_1 self-check <= 2.0e-7.

Against a DFT expectation of order eV, Delta_bind is an order of magnitude too small, and the
escape is not in use. In 5 of the 12 R2 seeds it is **negative** — those models put the
carrier level *above* the pristine host edge, which no bound state can do. Nothing anchors
the level to the host continuum. That is the diagnosis the plan proposed, confirmed.

Note it is not literally "~0 +/- thermal spread" on the clean arm 1b models: it is
consistently and significantly **+0.2 eV**, which the plan's reading has no named box for.

Arm 1b's own localisation at epoch 50: **0/8** on both gates (N_eff 23.0-48.8, ratio
0.31-0.67), matching the prediction recorded before the run.

## T-B — the inequality is satisfiable but not sufficient

Free ceiling, **with** the response channel (from the step-2 controls): axial_red +0.855,
rmse_nbhd 20.6. This is the correct reference; R1's +0.807 / 34.0 is the without-channel one.

| cell | n | N_eff (gate <=8) | ratio (gate <=0.15) | Delta_bind | axial_red | rmse_nbhd |
|---|---|---|---|---|---|---|
| V1 m=0.2 | 3 | 58.46 [31.6, 74.2] | 0.774 | **+1.524** [0.60, 2.35] | +0.803 | 23.3 |
| V1 m=0.5 | 3 | 52.67 [34.2, 72.6] | 0.693 | **+5.494** [1.41, **11.93**] | +0.822 | 25.2 |
| V2 m=0.2 | 6 | 48.67 [35.2, 62.8] | 0.648 | +0.367 [0.28, 0.54] | +0.756 | 28.8 |
| V2 m=0.5 | 6 | **32.34** [**12.50**, 55.3] | **0.431** [0.168, 0.730] | +0.720 [0.61, 0.86] | +0.787 | 31.7 |

**0/18 cells pass the localisation gate.** Closest: V2 m=0.5 s2 at N_eff 12.50, ratio 0.168 —
missing 8 and 0.15 narrowly. A second seed (s5) sits at 14.67 / 0.197. The other four m=0.5
seeds are 28.7-55.3, so the distribution is bimodal rather than merely wide.

**V1 escapes, exactly as predicted.** Its Delta_bind overshoots the margin by 3-24x
(+11.93 against m=0.5) while N_eff stays at 52-58. The head has a cheap way to satisfy the
inequality that has nothing to do with binding. V2, which cannot express it, tracks the
margin closely (+0.72 against m=0.5) and never exceeds +0.86. That contrast in the
*overshoot* is the cleanest escape signature in the data.

The per-species eps offsets are NOT clean confirmation either way, and should not be quoted
as such: they are non-zero in both variants and carry the opposite sign to the plan's
prediction (Pb positive, not negative). The 159-atom charged and pristine frames differ in
composition and relaxation as well as in the presence of a vacancy, so a per-species mean
difference is confounded. Use the overshoot, not the offsets.

**The margin does real work, monotonically.** V2 mean N_eff falls 48.67 -> 32.34 and ratio
0.648 -> 0.431 going from m=0.2 to m=0.5. Localisation also costs fit: the near-miss cell has
the worst rmse_nbhd of its group (46.5 against 23.7-26.4).

**The reference is unpinned.** lambda_1(pristine) ranges -5.2 to +6.4 across cells and
Delta_bind tracks m almost exactly for V2. Much of the inequality is met by moving the host
reference rather than pushing a level out of the continuum -- the plan's stated label-free
limitation, operating at full strength rather than as a mild floor.

## Where this lands in the decision rule

**Off-tree, again.** None of the four listed branches fits:

* not *adopt* -- V2 does not localise (0/12);
* not *V1 localises only via the escape* -- V1 does not localise at all, it merely escapes;
* not *V2 cannot reach m without a large force-loss increase* -- V2 reaches m easily, and its
  force loss ends at 2e-4 against 4.4e-3 at epoch 0;
* not *Delta_bind large in a delocalised model* -- T-A gave +0.2, not large.

The result is a fifth outcome: **Delta_bind >= m is necessary but not sufficient.** Against a
floating host reference, satisfying it does not put the state below the *defect cell's own*
continuum, which is what the localisation theorem actually requires.

## Suggested next step (not run)

The m-dependence is monotone and substantial over the one factor of 2.5 tested. Extrapolating
the V2 trend, reaching N_eff <= 8 plausibly needs m ~ 1.5-2 eV. The plan treats m as a
sensitivity check rather than a tuning knob, and rightly so -- but the sensitivity is now
measured, and an m-ladder (0.5, 1.0, 2.0) at 3 seeds on V2 is the cheapest test of whether
the mechanism reaches the gate at all or saturates short of it. That distinction decides
between "deepen the constraint" and "the constraint is the wrong object".

Pinning the reference is the other lever, and it is the one the plan already names: charged
pristine single points on existing geometries replace m with a measured depth.

## Caveats

* `loc_length` in the harness is **unreliable and was not used above**. V1 m=0.5 s2 reports
  1.10 A with N_eff 51.31, which cannot both be true; the metric takes an RMS radius about a
  centroid without a minimum-image convention, so it is wrong under PBC. Needs fixing before
  anyone quotes it.
* V1 has 3 seeds per margin against V2's 6. Seeds were put where the decision rule turns.
* b3's GPU3 failed again (third time) *after* T-B completed -- all 4 jobs exited 0 with 3
  cells each, so nothing here is truncated. It will need the BMC cold cycle before the next
  b3 run.
