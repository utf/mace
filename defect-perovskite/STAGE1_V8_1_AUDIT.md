# Stage 1 under plan v8.1: the loss-path audit and the saved-checkpoint recalibration

Record of addendum section 8, "Required loss-path audit and immediate recovery of the
completed runs", on the completed original-v8 Stage-1 run `s13ra` (arm (a), `full`, six
seeds, code 7e155c9). Every number here is a **regime tag `v8-stage1-old-objective`**: it
describes the retired objective and is not comparable with anything trained under v8.1.

Machine note: on 2026-09-06 b3's CUDA is down node-wide (GPU 6 faulted; `torch.cuda` cannot
initialise for any process; no root to reset), so items 1-5 ran on b3's CPUs in float64.
The audit harness (`stage1_audit.py`) runs INSIDE the trainer with the run's own recipe, so
the data, weights, null gate, size upweights, loss, optimiser groups, clipping and hooks are
the run's, not a reconstruction; only the training loop is replaced.

## Items 5-6: analytic nuisance intercepts of the saved seeds (`stage1_recalibrate.py`)

Strata (addendum section 8): `provenance|host|Q_formal|composition|cell|size_class`. The
dataset carries no `pair_id` groups, so every charged frame is **unpaired** and its residual
is `r_i = E_pred - E_label` in total-cell eV (the saved checkpoint's own c is inside its
head; `c(79)`, `c(159)` are listed). Training frames only enter the intercept; the held-out
column is centred with the TRAINING intercept and is a shape-only diagnostic.

| seed | c(79) | c(159) | stratum | n_train | raw mean (eV) | c_g* (eV) | profiled mean | within RMS train (eV) | held-out raw mean | held-out centred by train c* | held-out RMS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 9.540 | 10.430 | Q+1 79 | 928 | +2.042 | −2.042 | 2e−16 | 0.294 | +1.827 | −0.215 | 0.485 |
| s1 | | | Q+1 159 | 16 | +1.788 | −1.788 | 0 | 0.053 | +1.773 | −0.015 | 0.015 |
| s2 | 9.470 | 10.464 | Q+1 79 | 928 | +2.772 | −2.772 | −7e−16 | 0.280 | +2.559 | −0.213 | 0.478 |
| s2 | | | Q+1 159 | 16 | +2.398 | −2.398 | 0 | 0.043 | +2.390 | −0.008 | 0.008 |
| s3 | 9.526 | 10.570 | Q+1 79 | 928 | +2.480 | −2.480 | 3e−16 | 0.261 | +2.281 | −0.200 | 0.453 |
| s3 | | | Q+1 159 | 16 | +2.092 | −2.092 | 0 | 0.053 | +2.071 | −0.021 | 0.021 |
| s4 | 10.501 | 10.520 | Q+1 79 | 928 | −2.597 | +2.597 | 4e−16 | 0.323 | −2.824 | −0.227 | 0.496 |
| s4 | | | Q+1 159 | 16 | −1.568 | +1.568 | 0 | 0.064 | −1.580 | −0.012 | 0.012 |
| s5 | 9.489 | 10.379 | Q+1 79 | 928 | +2.751 | −2.751 | −3e−17 | 0.271 | +2.544 | −0.207 | 0.470 |
| s5 | | | Q+1 159 | 16 | +2.516 | −2.516 | 0 | 0.060 | +2.492 | −0.024 | 0.024 |
| s6 | 9.454 | 10.330 | Q+1 79 | 928 | +2.647 | −2.647 | −8e−17 | 0.265 | +2.444 | −0.203 | 0.456 |
| s6 | | | Q+1 159 | 16 | +2.411 | −2.411 | 0 | 0.045 | +2.399 | −0.012 | 0.012 |

Between-size intercept difference `c*(79) − c*(159)` per seed: 0.254, 0.374, 0.389, 1.028
(s4), 0.234, 0.236 eV.

- The profiled mean vanishes to the analytic floor in every stratum (item 5 requirement).
- Non-energy predictions: the profile is post hoc on energies and touches no model
  quantity. The re-evaluation check of forces on eight frames was bit-identical on four
  seeds and differed at the CPU multithreaded reduction floor on two (the tool now reports
  the maximum difference instead of a boolean).
- The held-out 79-atom residual, centred by the training intercept, sits at −0.21 eV on
  every seed: a train/validation shift of the frozen trunk's charged-frame error, seed
  independent, not a property of the head.
- Within-stratum RMS after profiling: 0.26-0.32 eV (79, train), 0.45-0.50 eV (79,
  held-out); 0.04-0.06 eV (159, train, 16 frames); 0.01-0.02 (159, held-out, one frame).
  These are the Stage-1 energy-shape readouts of the old regime.

### Item 6: the selection readouts, repeated after recalibration

The section 7.7 gates read no charged energy, so they are unchanged by the profile: arm (a)
retrained passes F4 (5/6), the gap (6/6), dilution (6/6) and F10 (4/6); the forward-only
arms (b) and (c) fail F4, the gap and dilution (the neutral-null-era table in
`STAGE1_V8_RESULTS.md`). The profiled energy-shape readout of arm (a) is the table above;
arms (b) and (c) have no retrained checkpoints (the chain was cancelled during arm (b)'s
save, before arm (c) ran), so no profiled energy-shape comparison exists for them. **The
Stage-1 choice, arm (a) `full`, is retained** on the force result and on the profiled
shape result of the only retrained arm; no physics arm is reopened. The corrected-regime
retrain of arm (a) is the next Stage-1 action (manifest
`golden/stage1_v81_manifest.json`, frozen before its results are opened).

## Items 1-4: the loss path of the 159-atom constant (`stage1_audit.py`)

Run: `queue_stage1_audit.sh` on b3 CPU (float64, 16 threads; the run's base cache loaded
from disk), seed 1, checkpoint `s13ra_s1_run-1_epoch-20.pt` (model + Adam + scheduler state),
the replayed epoch is epoch 21. Raw output `~/runs/audit_s13ra_s1.json` on b3 (the first
write lost its checkpoint/replay-metadata sections to a serialisation error on a buffer;
the items were complete and are what is reported; the tool is fixed).

### Item 2: the 16 retained charged energies, traced

| what | value |
|---|---|
| charged training frames | 944: 928 at 79 atoms, 16 at 159 |
| 79-atom charged `energy_weight` after the null gate | 0.0 on all 928 (the gate); `delta_energy_weight` 0.0 on all (no pairs in the data) |
| 159-atom charged, each of the 16 | `weight` 1.0, `energy_weight` 1.0 (w_E from the OOD table; the "charged energy share" factor was 1.00), `forces_weight` 9.606 (the two-size force upweight), `base_energy_weight` 0, `delta_energy_weight` 0, float64 |
| the column the loss reads | `energy_weight` in the totals term `total_energy_weight * mean_batch(weight * energy_weight * mask * ((E_label - E_pred) / N_at)^2)`, `total_energy_weight = 0.25`, mask = charged; no other energy column is non-zero for a charged frame |
| the constant | `spectral.c_shift_table[0, 1]`, a trainable parameter in group 9 (`spectral`, lr 0.005, weight decay 0, Adam betas (0.9, 0.999), eps 1e-8, **amsgrad on**); restored from the checkpoint bit-exactly (10.43027 at the end of epoch 20, the run log's epoch-21 line) |
| how it reaches the energy | a rigid shift of every level of a charge-class-0 graph: `dE_pred/dc(159)` by central differences is exactly `-1.0` on every 159-atom charged frame and `0.0` on every other graph. Added exactly once |
| gradient clipping | global norm 10; never active in the replayed epoch (0 of 320 steps) |

### Item 1: the implemented derivative, on the six batches of the epoch that carry a 159-atom charged frame

Per batch (8 graphs, one 159-atom charged frame with residual `E_label - E_pred` of about
-2.4 eV):

| quantity | value |
|---|---|
| totals term `L_E` | 4.0e-6 to 4.9e-6 |
| analytic `dL_E/dc` of the implemented per-atom-squared form, `(0.25 / 8) * 2 * resid / N^2 * (-dE/dc)` | -4.25e-6 to -4.68e-6 |
| autodiff `dL_E/dc` (the totals term alone) | -4.37e-6 to -4.89e-6 (within 3-5% of the analytic value; since `dE_pred/dc` is exactly -1.0 by finite differences there is no value dependence to account for the gap -- it is the same phantom gradient of the force path, below, appearing inside `E_pred`'s autograd) |
| **finite differences of the WHOLE loss at c +- 1 eV and +- 0.1 eV** | equal to the totals term's quadratic prediction to five digits on every batch (e.g. batch 105: dL(+1 eV) = -3.2329e-6, predicted -3.2329e-6): the loss depends on c ONLY through the totals term |
| autodiff `dL/dc` of the whole loss | **-3.5e-2 to +2.8e-1**, 4 to 5 orders of magnitude larger than the totals term, and entirely from the FORCE terms (`dL_forces/dc`: -3.09e-2, +2.77e-1, -3.63e-2, +1.37e-1, -2.69e-2, -3.50e-2) |

So the addendum's hypothesis is confirmed exactly for the energy path: the per-atom square
suppresses the constant's gradient as `2 resid / N^2` (about 5e-6 per step for a 2.4 eV
residual at 159 atoms), and it is the only path by which the loss's VALUE depends on c.

But the implemented gradient carried a second, larger component that the loss's value does
not have. The force terms' autodiff derivative with respect to c is non-zero while their
finite-difference derivative is zero: a **phantom gradient**. Its mechanism, by
construction of decision 15 of the v8 Stage 1.2 record (read off the code, not measured
separately): the frontier term's projector windows were shifted, on the value side, by the
c-table level shift as a DETACHED number, so the term's value is invariant to c, while its
divided-difference backward through the attached Hamiltonian still contains the
window-slope response of the spectrum moving relative to fixed edges. The value was
compensated; the gradient was not. That path acts on every charged graph, including the
928 79-atom frames whose energies the null gate removed. Supporting evidence: the 79-atom
constant and the scalar `c_shift` -- with NO energy term at all -- drifted by -0.49 and
-0.46 eV over the run in lock-step, as one gradient under one Adam dynamics would.

### Item 3: the optimiser replay of epoch 21 from the epoch-20 checkpoint

The loader's seeded order was reconstructed by replaying the generator's draws (four before
epoch 0 in the original run, one fewer here because the base cache loaded from disk, then
one per epoch); the historical batch order is not recorded, so this is the addendum's
"equivalent deterministic reproduction" and the exact step-by-step trajectory remains
unproven. 320 steps, lr 0.005, no clipping event.

| | c(79) = `[0,0]` | c(159) = `[0,1]` |
|---|---|---|
| steps with a non-zero pre-clip gradient | 313 of 320 | **16 of 320** (exactly the 16 batches holding a 159-atom charged frame) |
| gradient: energy (totals) part, summed over the epoch | 0 (no energy weight) | -8.1e-5 |
| gradient: force (phantom) part, summed | +1.09 (mean +3.4e-3, sd 2.5e-2) | -6.8e-3 |
| Adam state at the checkpoint | | m ~ 1e-5, v_max ~ 4e-5 (sqrt 6e-3): AMSGrad keeps the running maximum of v |
| per-step normalised update `m/sqrt(v)` on active steps | ~2e-3 | 2e-4 to 1e-3 |
| net change over the replayed epoch | **-0.0130** | **+0.00023** |
| the run log's change over epoch 21 | -0.017 (9.5398 -> 9.5225) | +0.0084 (10.4303 -> 10.4387) |
| the run log's change over the whole run | -0.53 (9.969 -> 9.509 at epoch 23, before the last epochs) | +0.022 (10.423 -> 10.445) |

The replay reproduces the sign and the order of magnitude of both constants' motion and the
mechanism of the observed 0.03 eV: (i) the energy gradient on c(159) is ~5e-6 per active
step (the per-atom-squared suppression, item 1); (ii) it acts on 5% of the steps, while
Adam's first moment decays by 0.9 per step in between and AMSGrad's `v_max` retains the
memory of the largest gradients ever seen, so the normalised step is 1e-3 of the learning
rate; (iii) the direction on those steps is set by the phantom force-path gradient, which is
10-100x the energy gradient and happens to share its sign on the 159 frames. No overwrite,
detach of the parameter, or frozen-parameter event exists: the constant was trainable,
restored, and updated on every step it had a gradient. The epoch-21 magnitude differs from
the log's by a factor of 36 on c(159), which the reconstructed batch order and the
per-epoch class-table refresh (the windows move between epochs) leave unresolved: recorded
as an unproven detail of the trajectory, not of the mechanism.

### Item 4: the analytic profiler

On the replayed epoch's charged residuals (928 at 79 atoms, 16 at 159): an injected offset
of +0.7 eV is recovered as a shift of exactly -0.7 in `c_g*` on both strata
(`recovery_error` 0.0), and the profiled mean is zero to 4e-16 (79) and 0.0 (159). No
learning rate is involved.

### Verdict on items 1-3, and what the v8.1 code path changes

Items 1-3 close on the audited code path: the constant reaches the loss exactly once, the
implemented derivative is the per-atom-squared form and matches its finite difference, and
the replay accounts for the observed motion's sign and magnitude by the suppression, the
sparsity, and the AMSGrad normalisation -- plus a phantom force-path gradient that the loss
value does not contain. That phantom is the finding beyond the addendum's diagnosis; it is
also why the 79-atom constant moved with no energy in its loss. Under v8.1 both parts are
gone by construction: there is no energy constant in H (the c table is retired; the
intercept is profiled analytically), and the frozen-pristine gauge `mu_g` is attached on
both the value and the gradient side, with the class-table edges re-aligned under it.

