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

AUDIT_ITEMS_PENDING
