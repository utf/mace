# MACEDefect fix work — session handoff

Working notes for the MACEDefect work on the GPU node. This file records environment
facts, what has been established, and the traps found along the way that are *not* written
down anywhere else.

**The plan being executed is `../madedefect_forward_plan.md`** (stages A–F; note the
filename typo, "made" not "mace"). It supersedes `../macedefect_correction_fix_plan.md`,
which is kept only for provenance — sections below that cite "fix plan" stage numbers
refer to the superseded document.

Read §12 first for current state; §§7–11 are the investigation that produced it, in
chronological order, and several of their intermediate conclusions were later revised.

Last updated: 2026-08-08.

---

## 1. Environment

| Thing | Value |
|---|---|
| Node | `gpunode` (hostname `val-perovskite`), single NVIDIA RTX A4000, 16 GB |
| Python | micromamba env `py13` — `export PATH=$HOME/micromamba/envs/py13/bin:$PATH` |
| Torch | 2.13.0+cu129, CUDA available |
| `les` | `~/src/les` (source install), imports fine |
| `mace` | `~/src/mace`, dev install; `import mace` resolves to the working tree |
| Repo | branch `defect`, HEAD `af12cc8` |
| Run outputs | `~/runs/<name>/`, logs at `~/runs/<name>.log` |

`~/.local/bin/claude` is v2.1.226 and already authenticated. It is on `PATH` in an
interactive login shell but *not* for `ssh node 'cmd'` (Ubuntu's `.bashrc` returns early
for non-interactive shells) — use `bash -lc` if invoking it non-interactively.

**Training runs must be backgrounded** (`nohup ... &`) and polled via the log; they
outlast any single command timeout.

Stale and safe to ignore/delete: `~/band_edges.json`, `~/dataset_summary.json` (loose
copies in the home dir), and `~/src/mace/defect-example/runs/` (a 5-epoch scratch run
from before the optimizer fix).

---

## 2. Source data

`ideal.db`, `defect_ground_state.db`, `defect_excited_state.db` now live on the node in
`~/src/mace/defect-example/`, so extraction runs here. (They were originally Mac-only;
copied across 2026-08-08.)

Contents worth knowing before designing anything around them:

| Pool | Frames | Cell sizes (atoms) | Character |
|---|---|---|---|
| `ideal.db` | 59 | 8, 288, 320, 384 | 1 true ideal lattice (the 8-atom cell, \|F\|max 0.0017 eV/A); the rest thermal, median \|F\|max 2.27 |
| `defect_ground_state.db` | 641 | 286, 318, 382, 398 | all thermal, median \|F\|max 2.87 |
| `defect_excited_state.db` | 641 | same | 504 of them share a geometry with a ground-state frame |

---

## 3. Stage 0 status

| Control | Status |
|---|---|
| 0.1 oracle-attention ablation | **dropped** — see below |
| 0.2 delta-forces gradient path | **done** — plumbing verified, both tests written and passing |
| 0.3 EMA / epoch-count control | **done** — EMA confounds validation; slow escape *observed* (§7) |
| 0.4 pristine frames carry `n = 0` | **done** — confirmed by inspection |
| 0.5 gradient-ratio N-test | **done** — ratio is N-independent (§10) |

Stage 1 is complete and verified; see §8. **0.1 was never run and is now moot**: it existed
to decide whether the descriptors could support the fit at all, and that question has been
answered directly — a model with no oracle localises `e_maj` onto the six defect
first-shell atoms (§11) and beats the target σ comfortably. The forward plan does not
carry it.

### 0.2 — done

Both training call sites pass `training=True` (`train.py:462`, `train.py:535`), which
reaches `_energy_gradient` in `defect_models.py` as `create_graph=True`, and
`get_outputs(..., training=training)` likewise. The "missing `create_graph`" failure mode
is **not** present. Two tests now pin this in
`tests/unit/test_defect_model.py::TestTrainingStep`:

- `test_delta_forces_alone_reach_the_correction_heads` — a loss built from `delta_forces`
  only produces non-zero gradients on the energy readouts, the logit readouts, the counter
  embedding and the defect feature readouts;
- `test_zero_u_init_would_starve_the_logits` — with `u` zeroed, the logit gradient is
  *exactly* `0.0`, pinning the mechanism behind the plateau in §7 as a test rather than a
  claim.

### 0.4 — done

`extract_defect_dataset.py:572` labels the `ideal` (pristine) frames with
`COUNTS_GROUND = (0,0,0,0)`. Pristine frames do carry `n = 0`. Note this is currently the
*same* constant used for the defect ground state, which is exactly the mislabelling
Stage 1.1 fixes — after the fix, pristine keeps `(0,0,0,0)` and the defect ground state
becomes `(1,0,0,1)`.

### 0.3 — the EMA half is answered, and the answer is yes

`mace/tools/train.py:299` evaluates validation inside
`ema.average_parameters() if ema is not None else nullcontext()`. **With `--ema` on, every
reported validation metric is that of the averaged weights, not the weights being
optimised.** The run script hard-coded `--ema`, so every number in the diagnosis document
was read off EMA weights. At `ema_decay=0.99` and ~90 steps/epoch the averaging time
constant is ~1.1 epochs, so this cannot by itself explain a metric pinned over many
epochs — but it does contaminate short runs, and it had to be removed before the
200-epoch control could mean anything.

`train_defect_model.sh` now takes `USE_EMA` (default `True`, unchanged behaviour).
`USE_EMA=False` drops both `--ema` and `--ema_decay`.

Baseline launched:

```bash
cd ~/src/mace/defect-example
export PATH=$HOME/micromamba/envs/py13/bin:$PATH
nohup env NAME=s03_baseline_noema WORK_DIR=$HOME/runs/s03_baseline_noema \
  MAX_NUM_EPOCHS=200 NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
  BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
  USE_EMA=False PATIENCE=250 ./train_defect_model.sh \
  > $HOME/runs/s03_baseline_noema.log 2>&1 &
```

Initial epoch line, for reference when reading the tail:

```
Initial: loss=4.49881868, RMSE_E_per_atom=91.01 meV, RMSE_F=767.89 meV/A,
         RMSE_dE=109.78 meV, RMSE_dF=46.11 meV/A
```

`RMSE_dE = 109.78 meV` is the validation-split target σ — i.e. the score of a model that
predicts the mean. The question the run answers: does `RMSE_dE` leave 109.78 and does
`RMSE_dF` move off 46.11, given 200 epochs and honest (non-EMA) weights?

**RESULT: slow escape is observed, not ruled out. This revises the diagnosis.** See
§7 below for the full curve and what it means. The early-epoch data that follows is what
the diagnosis document was based on; it is reproduced here because it is real, but it is
a plateau, not a fixed point.

Through epoch 6, with EMA off:

```
Initial   RMSE_E=91.01  RMSE_F=767.89  RMSE_dE=109.78  RMSE_dF=46.11
Epoch 0   RMSE_E=116.87 RMSE_F=188.63  RMSE_dE=108.25  RMSE_dF=46.11
Epoch 1   RMSE_E=23.28  RMSE_F=120.42  RMSE_dE=115.50  RMSE_dF=46.11
Epoch 2   RMSE_E=27.04  RMSE_F= 92.57  RMSE_dE=116.87  RMSE_dF=46.11
Epoch 3   RMSE_E=22.27  RMSE_F= 83.10  RMSE_dE=114.07  RMSE_dF=46.11
Epoch 4   RMSE_E=23.65  RMSE_F= 74.38  RMSE_dE=108.86  RMSE_dF=46.11
Epoch 5   RMSE_E=25.14  RMSE_F= 76.67  RMSE_dE=124.93  RMSE_dF=46.12
Epoch 6   RMSE_E=65.28  RMSE_F= 69.38  RMSE_dE=113.02  RMSE_dF=46.11
```

`RMSE_dF` holds at 46.11–46.12 across all eight lines — moving only in the fourth
significant figure — while the base branch trains normally (`RMSE_F` 768 → 69). So EMA is
**excluded** as the cause: the pinning survives reading the raw optimised weights.
`RMSE_dE` fluctuates around σ = 109.78 without departing from it. This reproduces the
diagnosis-document symptom on honest weights, at ~33 s/epoch, so the remaining question
for the 200-epoch leg is only whether there is a slow escape later.

Note the metric is *not* frozen — it does move at the 0.01 meV/Å level, which rules out a
severed gradient path (a truly detached prediction would repeat exactly) and points
instead at a prediction that is tiny and nearly constant. That is consistent with the
"predict the mean" attractor, but 0.2's unit test is still the thing that settles whether
any gradient of useful magnitude reaches `MLP_u` from a delta-force-only loss.

### 0.2 — plumbing verified, test still owed

Both training call sites pass `training=True` (`train.py:462`, `train.py:535`), which
reaches `_energy_gradient(delta_energy, positions, training)` at `defect_models.py:338`
as `create_graph=True`, and `get_outputs(..., training=training)` likewise. So the
"missing `create_graph`" failure mode is **not** present. Still to write, per the plan:

- unit test: tiny model, loss built from `delta_forces` **only**, backward, assert
  non-zero grads on the energy readouts *and* the counter embedding;
- companion regression test: at the current zero-`u`-init, grads on the **logit**
  networks are exactly zero — this pins the mechanism in the diagnosis document as a
  test rather than a claim.

---

## 4. Traps found for Stage 1 — read before regenerating the dataset

1. **The anchor set of plan §1.4 is two frames, not four.** Under
   `canonicalise_counts`, `(0,1,0,0)` has `M_s = −1` and canonicalises to `(1,0,0,0)`;
   `(0,0,1,0)` likewise canonicalises to `(0,0,0,1)`. Writing all four gives two
   *duplicate pairs* — same geometry, same counters, same `pair_id` — silently doubling
   the anchor weight. Write **one electron anchor `(1,0,0,0)` and one hole anchor
   `(0,0,0,1)`** per pristine cell, each with `multiplicity = 2` (`M_s = 1`).

2. **The relabelled defect states are both `M_s = 2`, so `multiplicity = 3`.** Ground
   `(1,0,0,1)`: `M_s = (1−0) − (0−1) = 2`. Excited `(1,1,0,2)`: `M_s = (1−0) − (1−2) = 2`.
   Both are already canonical. `multiplicity` was deliberately *omitted* from the old
   extraction because writing it would have tripped the `M_s == multiplicity − 1`
   assertion — that assertion was correct and the data was wrong. It must now be written
   for every spin-polarised frame, and made required (plan §1.1.2).

3. **The delta targets do not change numerically under the relabelling.** Referencing
   subtracts `1·E_g` from the ground state and `2·E_g` from the excited state, so the
   difference still carries `−E_g` exactly as the old `n = 0` / `(0,1,0,1)` labelling did.
   Consequences: Stage 0 numbers stay comparable across the fix, and the **48 meV /
   R² = 0.85 linear-probe floor remains the reference target**. What *does* change is that
   defect geometries no longer supply base-branch labels.

4. **`_join_pairs` will raise on the corrected labels.** `mace/data/defects.py:270–292`
   demands exactly one member with `Σn = 0` per `pair_id`; with correct labels a defect
   pair has none. Generalise to "the unique member of minimal `Σ_c n_c`" (error if not
   unique) and emit `base_*` targets **only** when that member is genuinely `n = 0`
   (i.e. pristine groups only). `_assign_base_weights` (line 397) needs the same
   treatment — it currently falls back to `min(indices)` when no `n = 0` member exists,
   which would hand base-branch supervision to an arbitrary charged frame.

5. **`n_ref` is the one genuinely new piece of plumbing.** A per-graph
   `carrier_counts_ref` must travel Configuration → AtomicData → collate → `forward`,
   defaulting to zeros so every existing test and the TorchScript dict access are
   unchanged. Design the key before rewriting the extraction script so both land in one
   pass.

6. **"Record the frontier occupations" (plan §1.1.1) cannot be fully honoured.** The
   source sqlite rows do not carry them. Record what exists — `multiplicity`, the
   provenance quote, and the assignment rationale in the file header — rather than
   fabricating per-frame occupations.

---

## 5. Trap for Stage 0.1 (oracle ablation)

Do not plumb a per-atom oracle mask through `AtomicData`. Compute it online inside
`MACEDefect.forward` from what is already there: degree = scatter of
`(lengths < 2.3 Å)` over `edge_index`; the six defect neighbours are the atoms with
`degree < 3.5`. Si–C bond is 1.89 Å and the second shell ~3.1 Å, so 2.3 Å separates them
cleanly and `r_max = 4.0` guarantees the edges exist. Gate it behind a diagnostic flag
(an env var read in `__init__` is fine — it must never ship).

Two guards: fall back to all-ones when a graph has no deficit atoms (pristine cells,
where `n = 0` makes the branch inert anyway, and where an empty mask would give `0/0` in
the segment softmax); and **verify the mask standalone before trusting any run** — expect
exactly 6 atoms (3 Si + 3 C) on defect frames and 0 on pristine.

The stop-everything verdict of plan §0.1 must be read off the **128-channel / `max_L=1`**
leg. An 8-channel failure to reach 48 meV could be plain capacity, and a false
"features are insufficient" verdict would misroute the whole project. Run 8 channels
first for speed; only the 128-channel result closes the gate.

---

## 6. Correction to the diagnosis document

`defect_correction_branch_diagnosis.md` asserts the long-range branch is inert at `q = 0`.
**That is wrong**, and the fix plan (§Stage 3) states the correct position:
`q_i^carrier = a(α_i^h − α_i^e)` is a compensated but pointwise non-zero charge whose
self-term is the electron–hole interaction, and `q_i^pol` carries a factor `Σ_c n_c`.
Part of the 0.94 eV observable belongs in the long-range branch. Note this is doubly
relevant after relabelling: the old labels gave the ground state `Σn = 0`, killing
`q^pol` there identically; the corrected labels give `Σn = 2` (ground) and `Σn = 4`
(excited), so the term is live on every defect frame.

All runs so far used `USE_LONG_RANGE=False`, so nothing measured to date constrains it.

---

## 7. Stage 0.3 result: the plateau breaks (2026-08-08)

The 200-epoch EMA-off baseline answers the control's question, and the answer overturns
the central claim of `defect_correction_branch_diagnosis.md`.

```
epoch   0  E  116.87  F  188.63  dE  108.25  dF   46.11
epoch   3  E   22.27  F   83.10  dE  114.07  dF   46.11
epoch   6  E   65.28  F   69.38  dE  113.02  dF   46.11
epoch   9  E   17.00  F   58.06  dE  109.45  dF   46.10
epoch  12  E   13.56  F   54.64  dE  124.26  dF   46.10
epoch  15  E   16.52  F   55.01  dE  109.27  dF   46.10
epoch  18  E   12.46  F   50.11  dE  110.39  dF   46.11
epoch  21  E   13.38  F   50.45  dE  109.20  dF   46.07   <-- last plateau epoch
epoch  24  E   47.24  F   48.21  dE  105.04  dF   33.26   <-- escape
epoch  27  E   14.47  F   43.50  dE   62.80  dF   30.67
epoch  30  E   20.03  F   41.00  dE   62.84  dF   30.25
epoch  33  E   11.98  F   42.26  dE   55.83  dF   29.25
epoch  36  E   10.33  F   40.47  dE   51.17  dF   29.77
epoch  39  E   15.95  F   41.17  dE   51.73  dF   28.66
epoch  42  E   18.66  F   38.54  dE   43.67  dF   28.53
```

**What this changes.** The diagnosis document argues the correction branch settles into a
degenerate "predict the mean" attractor and calls it an architectural failure rather than
undertraining. That is wrong as stated. The behaviour is a **long plateau followed by a
sharp escape** at around epoch 22 (~2000 gradient updates), after which both delta metrics
fall steadily. By epoch 42 `RMSE_dE` is 43.67 meV — already **below the 48 meV
linear-probe floor** that the fix plan sets as the Stage 2 gate — and still falling.

**Why the plateau is nonetheless real, and why the Stage 1.5 fix targets it exactly.**
The mechanism in the diagnosis document is right about the *initial condition* even though
it is wrong about the *fixed point*:

- at `u == 0`, `d(Delta E_SR)/d(logit)` is identically zero, so the attention gets no
  gradient at all, and `Delta E_SR == 0` makes the predicted delta forces identically
  zero — which is why `RMSE_dF` sits at exactly the target norm, to four significant
  figures, for twenty epochs;
- `u` itself *does* receive gradient from the delta-energy term regardless of `alpha`, so
  it creeps away from zero;
- once `u` has any spread, the attention gradient switches on (it scales with that
  spread), localisation follows, and both metrics drop together — which is exactly what
  epoch 24 shows, `dF` moving first.

So the zero-init did not create a trap; it created a **delay**, and the delay was long
enough that every short run in the diagnosis document sampled only the plateau. Removing
the zero-init (fix plan 1.5, now done) should shorten or remove it. That is a sharp,
cheap, falsifiable prediction and it is the first thing to test on the new pipeline.

**Caveats.** This run used the *old* (mislabelled) dataset, which is fine — the delta
targets are numerically identical under the relabelling. The escape epoch will not be
directly comparable to a new-pipeline run, because that changes the labels, the init and
`n_ref` at once. To attribute the change to the init specifically, run the new pipeline
twice, with and without a restored zero-init, and compare escape epochs.

---

## 8. Stage 1 status: done, and verified

All of fix plan Stage 1 is implemented and the dataset is regenerated.

| Item | State |
|---|---|
| 1.1.1 relabelling | done — pristine `(0,0,0,0)`, ground `(1,0,0,1)`, excited `(1,1,0,2)` |
| 1.1.2 multiplicity mandatory | done — required for every spin-polarised frame, `M_s == mult - 1` asserted |
| 1.1.3 generalised pair join | done — `_group_references` picks the unique minimal-`sum(n)` member |
| 1.1.4 `n_ref` in the model | done — `carrier_counts_ref` through Configuration -> AtomicData -> forward |
| 1.1.5 `L_tot` -> 0 | done — `TOTAL_ENERGY_WEIGHT` default is now `0.0` |
| 1.3 band-edge referencing | interim gauge retained (`E_g = 0.9430 eV`); real `E_g^cell` still an external DFT input |
| 1.4 pristine zero anchors | done — 2 canonical anchors per pristine cell, 118 total |
| 1.5 init and regularisation | done — zero-init of `MLP_u` removed, `DEFECT_U_L2` default `0.0` |

Regenerated dataset (`--n-structures 800 --frac-paired 0.925 --frac-unpaired 0.0
--frac-ideal 0.075 --gap-margin 0 --seed 42`): 825 train + 92 valid = 917 frames.

```
train by config_type: anchor_ideal 2, anchor_rattled 104, ideal 53, paired_ex 333, paired_gs 333
referenced delta targets: mean +0.0005 eV, std 0.1232, range [-0.3740, +0.4283]
```

Loader verification on the written files:

```
counters:      (0,0,0,0):53  (1,0,0,0):53  (0,0,0,1):53  (1,0,0,1):333  (1,1,0,2):333
ref counters:  (0,0,0,0):159  (1,0,0,1):666
base_energy supervised: 53   delta_energy: 439   delta_forces: 333
anchors: 106, max |delta_energy| = 0.0 exactly
```

and a forward/backward check on one defect pair plus one pristine group:

```
paired_gs       n=[1,0,0,1] ref=[1,0,0,1]  pred=+0.000000  target=None      w=0.0
paired_ex       n=[1,1,0,2] ref=[1,0,0,1]  pred=-0.168067  target=-0.124831 w=1.0
ideal           n=[0,0,0,0] ref=[0,0,0,0]  pred=+0.000000  target=None      w=0.0
anchor_rattled  n=[1,0,0,0] ref=[0,0,0,0]  pred=+0.197837  target=+0.000000 w=1.0
anchor_rattled  n=[0,0,0,1] ref=[0,0,0,1]  pred=-0.007102  target=+0.000000 w=1.0
energy == base_energy at n = 0:  0.0 exactly
grads: energy_readouts 8.7e+00  logits 1.7e-01  counter_embedding 1.3e+00
```

The reference member predicting exactly zero is structural, not fitted: `n_ref == n` makes
the difference vanish for any parameters. The **logit gradient being non-zero at
initialisation** is the Stage 1.5 change working — under the old zero-init it was exactly
`0.000000e+00`.

### Deviations from the fix plan, and why

1. **Two anchors per cell, not four.** `(0,1,0,0)` and `(0,0,1,0)` canonicalise onto
   `(1,0,0,0)` and `(0,0,0,1)`; writing all four would have doubled the anchor weight
   through duplicate frames.
2. **Rattled anchors are not optional here.** Plan 1.4.4 treats them as a later pass, but
   only *one* frame in `ideal.db` is a true ideal lattice (the 8-atom cell, |F|max
   0.0017 eV/A); the other 58 are thermal, median |F|max 2.27 eV/A. Since every defect
   frame is thermal, ideal-lattice-only anchors would pin `u_bulk` at a descriptor point
   the training data never visits — the failure mode plan 2.5 warns about. Anchors are
   therefore split by config type, `anchor_ideal` (2 frames) vs `anchor_rattled` (116),
   using an `--anchor-ideal-fmax` threshold.
3. **Anchor weighting is by `config_type`, not a new `--anchor_weight` flag.** MACE's
   existing `--config_type_weights` already scales `ref.weight`, which multiplies every
   delta term, so no new plumbing was needed. `delta_forces` weight is forced to zero for
   anchors in the loader regardless.
4. **Unpaired defect frames are dropped** (`--frac-unpaired 0.0`). With correct labels they
   carry no base label, and with `L_tot = 0` they contribute nothing to any loss term —
   they would have been pure cost.
5. **`report_e0_fit` was fitting the wrong subset.** `compute_average_E0s` fits over *all*
   training frames using *referenced* energies, not the `n = 0` subset. Corrected;
   it now predicts `-7.906682 eV` for both species (rank 1 of 2, as before).

### Cell sizes do not line up

Pristine cells are 8 / 288 / 320 / 384 atoms; defect cells are 286 / 318 / 382 / 398.
The first three pair up (a divacancy removes two atoms), but **there is no 400-atom
pristine cell**, so the 398-atom defect frames have no pristine partner and therefore no
anchor at their cell size. Worth knowing before reading any size-series result.

---

## 9. Two datasets: `dataset/` and `dataset_beta/`

Full-set epochs cost ~33 s on the A4000, which is too slow to iterate on. The extraction
script therefore gained `--max-natoms`, and there is now a small beta set for debugging.

| | `dataset/` | `dataset_beta/` |
|---|---|---|
| Frames | 825 train + 92 valid | 116 train + 13 valid |
| Cell sizes | 8 / 286 / 288 / 318 / 320 / 382 / 384 / 398 | 8 / 286 / 288 |
| Pairs | 370 | 48 |
| Anchors | 2 ideal + 116 rattled | 2 ideal + 20 rattled |
| Target σ (valid) | 0.1232 eV | 0.0883 eV |
| Purpose | real fits, reported numbers | debugging, ablations |

```bash
# full
python extract_defect_dataset.py --n-structures 800 \
    --frac-paired 0.925 --frac-unpaired 0.0 --frac-ideal 0.075 \
    --gap-margin 0 --seed 42 --out-dir dataset

# beta
python extract_defect_dataset.py --n-structures 120 \
    --frac-paired 0.80 --frac-unpaired 0.0 --frac-ideal 0.20 \
    --band-gap 0.9430 --seed 1 --max-natoms 290 --out-dir dataset_beta
```

Point a run at either with `DATA_DIR=.../dataset_beta`.

Two deliberate choices in the beta recipe:

- **`--band-gap 0.9430` is passed explicitly**, not fitted. Left to itself the script
  fits the gauge from whatever pairs survive the size cap, and the capped subset gives
  0.9461 eV — so beta and full would be referenced against *different* gauges and their
  delta targets would not be comparable. Pinning it to the full-set value keeps them on
  one scale.
- **`--seed 1`**, because with seed 42 the 8-atom cell lands in the validation split.
  That frame is the only relaxed ideal lattice in the whole dataset, hence the only anchor
  whose zero target is exact rather than approximate; it belongs in train.

Note the beta set has a smaller target σ (0.0883 vs 0.1232 eV), because the spread of the
vertical excitation grows with cell size and beta keeps only the smallest cells. Compare
runs against the σ of their own split, never across the two.

`--max-natoms` always keeps the 8-atom cell regardless of the cap, for the same reason.

---

## 10. Stage 0.5 result: the attention gradient does not weaken with cell size

`stage05_gradient_ratio.py` measures `|grad logit_readouts| / |grad energy_readouts|` on
one excited-state frame per cell size, using **one model instance** so only the cell size
varies.

At 16 channels, `max_L=0`:

```
     N    |g_logit|        |g_u|      ratio    partic.  partic./N     std(u)  max alpha
   286   1.7824e-02   5.4069e-01     0.0330      284.4      0.994     0.1078    0.00376
   318   6.6257e-02   2.0109e+00     0.0329      316.2      0.994     0.1077    0.00338
   382   1.4600e-02   4.4287e-01     0.0330      379.8      0.994     0.1078    0.00282
   398   7.3062e-03   2.2187e-01     0.0329      395.8      0.994     0.1077    0.00270
```

At 128 channels, `max_L=1` the ratio is 0.0044 and equally flat (1.002x over a 1.39x
change in `N`).

**This refutes a specific claim in the diagnosis document.** That document argues the
attention gradient is suppressed by roughly a factor `N` because `alpha ~ 1/N`, and treats
that as a reason the softmax-over-all-atoms form is structurally unable to localise in
large cells. The measurement says otherwise: the ratio is constant to 0.1% across a 1.39x
change in cell size, at both widths. The `alpha_j` prefactor in

    d(Delta E_SR)/d(logit_j) = n_c alpha_j (u_j - <u>_alpha)

is cancelled by the sum over sites, exactly as the algebra says it should be. Whatever is
slow about training the attention, **it does not get worse as cells grow**, so plan §2.4's
"reconsider softmax over all atoms" is not motivated by size-scaling.

The ratio is small in absolute terms (0.033), i.e. the logits get ~30x less gradient than
`u` does — but that is a fixed factor set by the spread of `u`, not a growing one.

**Do not compare the ratio across widths.** These are gradient-*norms*, which grow with
parameter count, so the 0.033 -> 0.0044 change between 16 and 128 channels is confounded
and should not be read as "the attention trains 7x worse at production width". The
N-comparison within a fixed width is the clean one.

With `--zero-u-init` the logit gradient is **exactly `0.0000e+00` at every cell size**,
and `std(u)` is exactly zero — the mechanism of §7, measured directly.

---

## 11. The zero-init A/B: my Stage 1.5 prediction was wrong, and the autopsy says why

> **Read §11.5 before using this section.** The single-seed result below is real but it
> over-states the case; three seeds per arm show escape is dominated by seed, not by the
> init, and the headline finding is not "zero-init is better" but "escape is stochastic
> and often fails entirely".

§7 predicted that removing the zero-init of `MLP_u` would shorten or remove the plateau.
On the first seed it did the opposite. Identical runs on `dataset_beta`, same seed,
differing only in `DEFECT_ZERO_U_INIT` (verified from both run logs and by checking
`|W_last|` = 1.31 vs 0.000000):

```
epoch      beta_noinit (zero-init OFF)      beta_zeroinit (zero-init ON)
   0        dE  82.81   dF 49.30             dE 106.14   dF 49.30
  20        dE  88.90   dF 49.29             dE  65.89   dF 42.43
  40        dE  85.12   dF 49.29             dE  47.74   dF 37.54
```

The arm *without* the zero-init is still fully on the plateau at epoch 40 — `dF` unchanged
in the fourth significant figure — while the arm *with* it has escaped and is converging.

### The autopsy (`autopsy_checkpoint.py`), per carrier channel

Escaped run, `beta_zeroinit` @ epoch 169, delta RMSE **39.30 meV**:

```
 channel  occupied  anchored   partic.  partic./N    std(u)    mean(u)  a_defect
   e_maj      True      True       2.5      0.009    0.0775     0.1358    0.9996
   e_min      True     False     171.4      0.599    0.0224     0.1699    0.0205
   h_maj     False     False     285.3      0.998    0.0000     0.0000    0.0208
   h_min      True      True     285.9      1.000    0.0086     0.0014    0.0202
```

Stuck run, `beta_noinit` @ epoch 157, delta RMSE **96.00 meV**:

```
   e_maj      True      True     285.8      0.999    0.0036    -0.0011    0.0210
   e_min      True     False     285.4      0.998    0.0084    -0.0040    0.0228
   h_maj     False     False     285.9      1.000    0.0534    -0.0134    0.0208
   h_min      True      True     285.8      0.999    0.0072    -0.0090    0.0211
```

(6 defect first-shell atoms of 286, so uniform attention puts 0.0210 of `alpha` there.)

**The escaped run has done exactly what the design intends.** `e_maj` has a participation
ratio of **2.5 out of 286** and puts **99.96% of its attention on the six defect
first-shell atoms**. That is the localisation the whole architecture exists to produce,
and it is the Stage 1 gate ("participation moves measurably off N") passed emphatically.

**Why the random init is worse — measured, not guessed.** Look at `h_maj` in the stuck
run: it is *never occupied* in this dataset, so it receives no gradient and still carries
its initialisation, `std(u) = 0.0534`. That is the scale a random init starts at. Now
compare the occupied channels in the same run: `std(u)` = 0.0036–0.0084, i.e. the fit has
**shrunk `u` by roughly an order of magnitude, back toward zero**.

That is the whole story. A random `u` is structureless noise that contributes a spurious
`Delta E`, and the cheapest early loss reduction is to kill it. So the optimiser spends
its first epochs travelling *back* to `u ~ 0` — the exact configuration where
`d(Delta E_SR)/d(logit)` vanishes and the attention has no gradient. It arrives at the
degenerate point with the attention still perfectly uniform (participation/N = 0.999) and
no seed of structure to grow from. The zero-init run starts at that point but grows `u`
*along the gradient*, which is correlated with the target, and localises hard.

So the zero-init is not "unnecessary" as fix plan 1.5 and diagnosis §7.1 both assert. It
is a **conditioning choice**, and removing it is only safe together with plan §2.3's
*structured* init (`u_init ~ -eps * novelty(h_i)`, spread ~ 1/beta). Plan §1.5 (Stage 1)
and §2.3 (Stage 2) are a package, and splitting them across stages — which is what the
plan says to do, and what was done here — is actively harmful. That is a finding about
the plan, not just about this run.

### The anchors do work — and they reveal a real gap

`h_min` is anchored and its `mean(u)` is **0.0014**, i.e. pinned to zero exactly as
plan §1.4 intends. `e_min` is **not** anchored and its `mean(u)` is **0.1699**, over a
hundred times larger — a free level absorbing the fit.

This matters because of the canonicalisation collapse (§4 item 1). The two surviving
canonical anchors pin `e_maj` and `h_min` only. In this dataset:

- `h_maj` is **dead** — zero in every frame, so it is never trained (confirmed above:
  `std(u)` exactly 0 in the escaped run);
- `e_min` is occupied **only in the excited state** and is therefore the channel that
  *carries the excitation*, and it has **no anchor at all**. No frame with `e_min`
  occupied has a zero target, and the energy readouts are not shared across spin, so its
  bulk level is exactly the unconstrained mean-absorbing direction the anchors were
  introduced to close.

Stage 1.4 as implemented therefore closes two of the three live directions and leaves the
most important one open. Fixing it is a design question, not a quick edit — the options
are sharing the energy readout across the spin channels of a carrier type, or accepting
`L_Delta` as the only constraint on `e_min` — and it belongs in the co-design discussion.

### 11.5 Three seeds per arm: the conclusion is much weaker than the above

Escape epoch = first epoch with `RMSE_dF < 45` (the metric is flat at 49.3 on the
plateau, so this is unambiguous). 200 epochs, `dataset_beta`, otherwise identical:

| run | seed | zero-init | escape epoch | final dF | final dE |
|---|---|---|---|---|---|
| `beta_noinit`     | 1 | off | **never** | 49.26 | 84.95 |
| `ab_noinit_s2`    | 2 | off | **never** | 49.17 | 89.85 |
| `ab_noinit_s3`    | 3 | off | 37 | **23.92** | **28.95** |
| `beta_zeroinit`   | 1 | on  | 10 | 27.85 | 37.55 |
| `ab_zeroinit_s2`  | 2 | on  | 78 | 25.44 | 32.07 |
| `ab_zeroinit_s3`  | 3 | on  | 102 | 28.57 | 43.08 |

**What this actually supports, and what it does not.**

- Zero-init escaped 3/3, no-zero-init 1/3. That is *suggestive*, not established: with
  n = 3 per arm, 3/3 vs 1/3 is Fisher-exact p ~ 0.4 two-tailed. It is not enough to flip
  a default on, and it is nowhere near enough to amend the plan on.
- **Escape times are wildly stochastic even within the winning arm** — 10, 78 and 102
  epochs for the same configuration at three seeds. Seed dominates the init.
- **Once a run escapes, the final quality is the same either way**, and the single best
  result of all six is a *no-zero-init* run (dF 23.92, dE 28.95). So the init affects the
  *reliability of escaping*, not the fit that is reachable.

**Therefore the headline finding is not about the zero-init at all.** It is that the
plateau is a genuine optimisation pathology which, at this width, **fails to resolve
within 200 epochs in a third of runs regardless of initialisation**. A fix that makes
escape reliable is worth more than either init choice, and that is precisely what plan
§2.3's *structured* init is for: seed `u` with descriptor novelty at a controlled scale,
so it is neither zero (no attention gradient) nor noise (crushed back toward zero). Both
arms' failure mode motivates it; neither arm is the answer.

**Actions deliberately NOT taken**, because the evidence does not support them:

- `DEFECT_ZERO_U_INIT` default stays `False`. Flipping it would encode a p ~ 0.4 result.
- Plan §1.5 is *not* amended in the repo. §11's claim that §1.5 and §2.3 "are a package"
  remains a plausible reading of the mechanism, but it is not established by n = 3.
- `test_correction_is_live_at_initialisation` is untouched (it pins a mechanism, and the
  default it relies on has not moved).

What would settle it: ~10 seeds per arm at beta cost (~7 min per run, so ~1 GPU-hour for
both arms), reporting escape fraction and median escape epoch. Worth doing *after* Stage 2
lands, as a three-arm comparison — zero / random / structured — rather than now as a
two-arm one, since structured init is the candidate that should beat both.

The per-channel autopsy in §11 stands on its own: it is a direct measurement of one
escaped and one stuck run, and the `e_min` anchor gap it exposes does not depend on the
A/B outcome.

---

## 12. Stage A (forward plan) — done

Working from `../madedefect_forward_plan.md`, which supersedes the correction fix plan.

### A1 — zero-init default flipped to `True`

`--defect_zero_u_init` and `DEFECT_ZERO_U_INIT` now default `True`; `MACEDefect` and
`CarrierAttentionPooling` default `zero_u_init=True`. The code comment records the
*conditioning* rationale and explicitly warns against re-deriving from the `n = 0`
identity, which is what caused the removal in the first place.

Two tests were adjusted so they pin mechanisms rather than the default: both
`test_correction_is_live_without_the_zero_init` and
`test_delta_forces_alone_reach_the_correction_heads` now construct with
`zero_u_init=False` explicitly. The second one matters — under the shipped default its
logit-gradient assertion would be satisfied trivially by the starvation it is not testing.

> **Caveat on the evidence, which the plan's §0 table understates.** §0 cites seed 1 only
> ("no-zero-init still on the plateau at epoch 40 while zero-init escaped"). Seeds 2 and 3
> only *partly* agreed: `ab_noinit_s3` escaped at epoch 37 and finished best of all six
> runs (dF 23.92). The real tally is escape in 3/3 with the zero-init versus 1/3 without —
> directionally supportive, Fisher-exact p ~ 0.4. The flip follows the plan's decision and
> the escape-reliability tally, not a settled result. See §11.5.

### A2 — anchors removed, datasets regenerated at `v3`

Removed: anchor frame generation, `--no-anchors`, `--anchor-ideal-fmax`, the
`anchor_ideal`/`anchor_rattled` config types, `_is_anchor` and the `delta_forces`-weight
path it drove, and the anchor-vs-referencing consistency assertion.

Replaced with the weaker assertion the plan asks for, and **upgraded from a warning to a
hard error**: frames sharing a `(host, cell)` key must resolve to identical edge
constants. Two frames of the same host and supercell refer to the same physical edges by
definition, so a disagreement is a data error, and as a warning it was invisible.

Kept unchanged: labels, mandatory `multiplicity`, the generalised pair join, `n_ref`
plumbing (reused by §D-opt), `L_tot = 0`, pristine frames, `--max-natoms`, `dataset_beta`.

`DATASET_VERSION = "v3"` is written into `dataset_summary.json`. **No metric from v2 is
comparable to v3** — including the 43.67 meV figure quoted in the forward plan's preamble.

| | `dataset/` v3 | `dataset_beta/` v3 |
|---|---|---|
| train | 719 (53 pristine, 333 gs, 333 ex) | 80 (8 pristine, 36 gs, 36 ex) |
| valid | 80 (6 pristine, 37 gs, 37 ex) | 27 (3 pristine, 12 gs, 12 ex) |
| counters | `(0,0,0,0)`, `(1,0,0,1)`, `(1,1,0,2)` | same |

### A3 — referencing state confirmed: still the interim fitted gauge

`E_gap = 0.9430 eV`, exactly the mean vertical excitation — **not** `E_g^cell` per
supercell. This needs charged pristine total energies, which cannot be produced now, so it
stays. Recorded consequences:

- it is a real error source *within* this dataset, not only a transferability issue,
  because `E_g^cell` genuinely differs between the 286- and 398-atom cells, so one
  constant injects a systematic per-cell offset;
- **held-out sets must be split within cell size, and already are**: every defect cell
  size (286/318/382/398) appears in both the train and validation splits of v3, so no
  whole cell size is held out and the offset cannot masquerade as generalisation error.

### A4 — per-channel diagnostics, and a corrected floor

`DefectRMSE` now accumulates, per carrier channel, over carrier-bearing frames:
participation ratio, `mean(u^c)`, `std(u^c)`, and the logit gap. They are logged every
epoch on their own line, because the transition of interest is sharp:

```
Initial: carrier channels (e_maj e_min h_maj h_min): partic=[349.0 348.8 348.9 348.1],
         mean_u=[0.000 0.000 0.000 0.000], std_u=[0.000 0.000 0.000 0.000],
         gap_l=[0.019 0.029 0.025 0.054]
Epoch 0: partic=[168.1 343.0 349.1 323.7], mean_u=[-0.053 0.018 0.000 0.024],
         std_u=[0.000 0.004 0.000 0.004], gap_l=[6.046 0.156 0.009 0.322]
```

`e_maj` halves its participation and opens a logit gap of 6.0 within one epoch, while the
dead `h_maj` channel holds `std_u` at exactly `0.000` — the invariant, visible live.

**The dead-channel invariant is now a test, and it failed on first run.** Not a structural
leak: the counter prefactor is sound (dead-channel gradient is exactly `0.000000e+00` with
`u_l2 = 0`). The leak was the **L2 on `u` itself**, which is a mean over all channels and
so reaches channels no frame occupies. `DefectLoss(u_l2=...)` and `--defect_u_l2` therefore
now default to `0.0`, matching the plan's settled `λ_u = 0`, and the docstring claiming the
penalty is "load-bearing" is corrected — a logit gap of ~11.7 was reached without it, and
being a *mean* it is intensive and cannot penalise a bulk-wide level of `u` anyway.

#### The 48 meV floor was in-sample. The real floor is 55.81 meV.

`linear_probe_floor.py` fits the probe on **train** and scores it on **validation** — the
same split the model is scored on:

| split | target σ | probe RMSE | R² |
|---|---|---|---|
| `dataset` v3 train | 123.01 meV | 44.34 meV | 0.870 |
| **`dataset` v3 valid** | 124.12 meV | **55.81 meV** | **0.798** |
| `dataset_beta` v3 train | 85.33 meV | 30.13 meV | 0.875 |
| `dataset_beta` v3 valid | 95.16 meV | 63.80 meV | 0.550 |

The long-quoted "48 meV / R² = 0.85" was the probe's **in-sample** error, which is exactly
the like-for-like failure the forward plan warns about in A4. **The floor to beat on the
v3 validation split is 55.81 meV.** Earlier claims that a model "beat the 48 meV floor"
compared a held-out model score against an in-sample probe score, on a different dataset
version; the direction of that conclusion survives, the numbers do not.

Note `dataset_beta`'s probe generalises far worse (R² 0.550 on 36 training pairs). Beta is
for iteration and ablations, never for judging accuracy.

### A4 result — the baseline of record

Full `dataset/` v3, 8 channels, `max_L = 0`, `r_max = 4.0`, float32, EMA off, 2 seeds,
stopped at 75 epochs (the escape and the post-escape plateau are both well resolved by
then; the run was a diagnostic, not a tuned schedule).

```
            escape   best dE   post-escape mean (ep >= 43)
 seed 1      ~22      23.39 meV      dE 31.47   dF 26.07
 seed 2      ~45      30.86 meV      dE 44.41   dF 28.53
```

**Against the corrected floor of 55.81 meV, both seeds win**: 31.47 and 44.41 meV
post-escape mean, 23.39 and 30.86 meV at best. The model is doing substantially better
than ridge regression on 15 defect-neighbour distances, on the same held-out split. This
is the number everything downstream competes against.

Escape epoch differs by 2x between two seeds (~22 vs ~45) on identical data, consistent
with the stochasticity seen in §11.5.

#### `RMSE_F` is not a meaningful metric on this dataset — do not read it as a regression

It sits at ~145 meV/A and does not improve. That is expected and is a direct consequence
of `L_tot = 0` under corrected labels: the only force supervision is `base_forces` on the
53 pristine frames and `delta_forces` on the paired excited frames. **Total forces at
defect geometries are not trained by anything**, and `RMSE_F` is dominated by exactly
those frames. The meaningful metrics here are `RMSE_dE`, `RMSE_dF`, and the base metrics
on pristine frames. Restoring `RMSE_F` needs the closed-shell `M_s = 0` singlepoints from
the external-inputs table.

#### The per-channel decomposition is seed-dependent — this is new

Final channel state, both seeds, same data:

```
seed 1  partic=[  2.44  201.91  348.85    3.18]  gap_l=[15.54  1.65  0.03  11.31]
        mean_u=[ 0.531   0.040   0.000   0.311]  std_u=[0.582 0.045 0.000  0.295]
seed 2  partic=[ 88.86    2.80  349.08   81.21]  gap_l=[ 6.47 21.61  0.02   5.27]
        mean_u=[-0.115  -0.056   0.000   0.028]  std_u=[0.456 0.075 0.000  0.060]
```

Both fit well, but they localise **different channels**. Seed 1 puts the localisation in
`e_maj` and `h_min` (participation 2.4 and 3.2 atoms — the ground-state channels); seed 2
puts it in `e_min` (participation 2.8 — the excitation channel) and leaves `e_maj`/`h_min`
only partly localised at ~85.

Only the *total* `Delta E` is constrained by the data, so the split of the correction
across channels is not identified. Consequences:

- **`Delta u` cannot be read as a per-channel binding energy** without further
  constraint — which is exactly what forward plan §D2(2) wants it for, and it is why §D-opt
  says that reading is "only meaningful with §D-opt on". This is now measured, not argued.
- The level modes differ substantially between seeds (`mean_u` for `e_maj`: +0.531 vs
  -0.115), which is the unidentified gauge direction §D-opt targets. Direct evidence that
  the gauge penalty has something real to fix.
- `gap_l` exceeds plan §3.2's requirement of `~11` in whichever channel each seed
  localises (15.5 and 21.6), so the extensivity requirement is met either way.

**The dead-channel invariant holds in production.** `h_maj` shows participation exactly
`N` (348.85 / 349.08) and `mean_u = std_u = 0.000` in both seeds, after 75 epochs. That is
the control which makes the live-channel numbers above interpretable.

---

## 13. Stage C, first attempt: void (my dataset-design error), and what it revealed

The first Stage C runs never left the plateau — `RMSE_dF` bit-identical for all 120
epochs, in both cell sizes and both seeds. **This is not a result about cell size.** It is
an artifact of how I built the datasets, and the runs are discarded.

### What went wrong

To match loss composition across the two cell sizes I generated both subsets with
`--frac-ideal 0`. The 398-atom defect cells have no pristine partner (there is no 400-atom
pristine cell), so dropping pristine frames looked like the way to keep the two sets
comparable. But pristine frames are the **only** `n = 0` frames in the corrected labelling,
so removing them removes every base-branch label:

```
dataset        n=719  base_E=53  base_F=53  delta_E=333
dataset_n286   n=126  base_E= 0  base_F= 0  delta_E= 63   <-- no base supervision at all
dataset_n398   n=126  base_E= 0  base_F= 0  delta_E= 63
```

With `L_tot = 0` as well, the *only* live loss terms were `delta_energy` and
`delta_forces`. The signature is unmistakable in the logs: `RMSE_F` sits at its untrained
value for the whole run (738.99 -> 732.22 over 120 epochs), against 211 -> 145 in A4.

### The finding worth keeping

**The correction branch cannot bootstrap from delta targets alone.** The trunk is shared,
so with no base labels its only gradient comes through the correction — and during the
plateau that gradient is small and structureless, so the features never develop; without
developed features the attention has nothing to localise onto; so the plateau never ends.
It is a deadlock, and 120 epochs of it in four independent runs is a reasonably strong
statement.

Consequences worth carrying forward:

- **Never train this model on a delta-only dataset.** Base-branch supervision is not
  optional scaffolding; it is what makes the trunk features exist for the correction to
  use. This is a property of the shared-trunk design (`correction_trunk="shared"`).
- It sharpens why `L_tot = 0` is a real cost rather than a free simplification: on this
  dataset only 53 pristine frames carry base labels, and they are doing more work than
  their share of the loss suggests.
- It is an argument for the closed-shell `M_s = 0` singlepoints in the external-inputs
  table, independent of the `L_tot` argument already made for them.

### The corrected design

`--defect-natoms` and `--ideal-natoms` were added so the defect cell size can be varied
while the pristine pool is held **fixed and identical**:

```bash
python extract_defect_dataset.py --n-structures 140 \
    --frac-paired 0.93 --frac-unpaired 0 --frac-ideal 0.07 \
    --band-gap 0.9430 --seed 7 --defect-natoms 286 --ideal-natoms 288 --out-dir dataset_n286
# ... and --defect-natoms 398 --ideal-natoms 288 for the other arm
```

Both now read `n=127, base_E=9, base_F=9, delta_E=59` — identical composition, identical
base supervision, identical steps per epoch. The only difference is the defect cell size,
which is what the measurement is supposed to isolate. Using 288-atom pristine cells in the
398-atom arm is deliberate: they are the same host, they are not the object of the
measurement, and holding them fixed is what makes the comparison clean.

The discarded logs are kept as `~/runs/c_n*_s*_nobase.log`.

### Stage C, corrected run — seed 1 (seeds 2–3 pending)

Matched datasets, identical base supervision, 150 epochs:

```
c_n286_s1   ep 0  F 881.0  dE  61.45  dF 52.90
            ep 50 F 165.6  dE  80.09  dF 40.67
            ep149 F 156.9  dE  40.23  dF 39.41     <-- escaped
c_n398_s1   ep 0  F 875.2  dE 163.29  dF 35.24
            ep 50 F 159.3  dE 129.44  dF 35.24
            ep149 F 156.8  dE 123.36  dF 35.22     <-- never escaped
```

The base branch trains **identically** in both arms (`RMSE_F` 881 -> 157 and 875 -> 157),
which is the control that makes this comparison mean something — the two runs differ only
in defect cell size. At 286 atoms the correction escapes and both delta metrics fall; at
398 atoms `RMSE_dF` is bit-identical at 35.24 for all 150 epochs.

If seeds 2 and 3 agree, this is the plan's "escape epoch rising with N" branch, and
strongly so: not a longer plateau at 398, but no escape at all within the budget that
suffices at 286. Under the plan's decision rule that makes the plateau a **production-scale
hazard** and justifies Stage D on dynamics rather than on the `Delta u` diagnostic alone.

**Caveat on comparing `dF` values across cell sizes.** `RMSE_dF` averages over every atom,
while the delta forces are concentrated on the six defect neighbours, so a larger cell
dilutes the metric — hence 35.24 at 398 versus 52.90 at 286 *at initialisation*, before any
training. Only the escape *epoch* (whether the metric moves at all) is comparable across
sizes; the levels are not.

---

## 14. Stage A5 — `L_tot` restored, `RMSE_F` fixed

Implements the forward plan's A5. Dataset bumped to **v4** (unpaired defect frames
restored, since they now carry a live loss term again).

### Code changes

| Change | Where |
|---|---|
| `L_tot` applies at **every** `n != 0` frame, not just unpaired ones | `DefectLoss._totals_mask` |
| `E_base` receives gradient there (stopgrad removed) | `DefectLoss.forward` |
| Uses `pred["energy"]`, **not** `delta_energy` | `DefectLoss.forward` |
| `--defect_totals_detach_base` ablation switch, default off | arg parser, loss |
| `--defect_zn_l2`, L2 on `MLP_u`'s counter-input columns | `CarrierAttentionPooling.counter_input_l2`, loss |
| `--base_lr_factor`, scales the base-branch groups' LR | `get_params_options` |
| Gauge identifiability report, logged at load | `report_gauge_identifiability` in `defects.py` |

The third row is a correctness trap worth stating: `delta_energy` is now the *paired
difference* against `n_ref`, and only equals the correction when `n_ref = 0`. The old
`L_tot` used it, which was fine while the term covered unpaired frames only; extending
the term to paired frames without changing this would have compared the total energy
against `E_base + [corr(n) - corr(n_ref)]`. `pred["energy"]` is the right quantity.

### `base_lr_factor` — the plan's setting is wrong at this stage, measured

A5.3 asks for the base branch on a low learning rate. One knob at a time on
`dataset_beta`, 25 epochs:

| variant | RMSE_E (meV/atom) | RMSE_F (meV/A) |
|---|---|---|
| A5 as specified, `base_lr_factor = 0.25` | 789.17 | 177.33 |
| ... with `L_tot` off | 1545.19 | 179.45 |
| ... with default `config_type_weights` | 1030.63 | 206.87 |
| ... with `defect_zn_l2 = 0` | 982.18 | 184.54 |
| **... with `base_lr_factor = 1.0`** | **18.20** | **88.68** |

The factor is the whole effect — 40x on the energy, 2x on the forces — and every other
knob is noise beside it. The default is therefore **1.0**, with the knob kept.

The "moving target" rationale assumes a base branch that is already trained. This one
starts from random init and has to learn the entire SiC potential; quartering its learning
rate just leaves it undertrained, and A5 makes that visible for the first time because
`L_tot` now supervises total energy at defect geometries. Lower it when warm-starting from
a trained base — the Stage E retrain is the case the rationale actually fits.

### It does what A5 was for

`dataset_beta`, 120 epochs, corrected settings:

```
 ep   0  E 296.31  F 802.71  dE 91.94  dF 52.03
 ep  15  E  15.11  F 116.01  dE 103.60 dF 52.03
 ep  30  E  13.14  F  82.96  dE 92.56  dF 51.96
 ep  45  E  44.55  F  74.06  dE 87.97  dF 48.78
 ep  60  E  16.69  F  67.38  dE 72.39  dF 42.77
```

`RMSE_F` trains — 802 -> 67 and still falling. That is the gap A5 exists to close: it was
pinned at ~145 on the full set and never moved, because total forces at defect geometries
had no supervision at all. The delta metrics still escape the plateau on their own
schedule (~epoch 45 here).

### Gauge status for this dataset, logged every run

```
Carrier counters observed: [1, 0, 0, 1], [1, 1, 0, 2]
E_base gauge is UNDER-DETERMINED: 1 free direction(s) remain. E_base at defect
geometries is latent, not a validated prediction; do not ship transition levels from
this model (plan A5.6). Adding the q = +-1 doublets would close it.
```

Three live channels and two counters leaves exactly one free direction, matching A5.1.
Adding the two `q = +-1` doublets makes the set inconsistent — `(1,0,0,1) = (1,0,0,0) +
(0,0,0,1)` — and pins `f = 0`. The report counts only channels that are occupied
somewhere: `h_maj`'s `s_c` multiplies `n_c = 0` in every frame, so counting it would
inflate the free dimension by one and misreport the plan's arithmetic.

**Restoring `L_tot` does not fix the seed-dependent channel split** (§12), and was never
going to: it adds no new counter, so the one free direction in A5.1 is untouched. That
symptom needs charged data.

### Tests added

- `test_totals_train_the_base_branch_at_defect_geometries` — asserts on `model.readouts`,
  the base-only energy head. Asserting on the trunk would prove nothing, since the
  correction shares it and would supply gradient either way (my first version of this test
  failed for exactly that reason).
- `test_detach_flag_restores_the_old_behaviour` — same reasoning in reverse.
- `TestGaugeIdentifiability` — the two-counter case is open with one free direction, the
  four-counter case is closed, and dead channels do not inflate the count.

92 tests pass.

---

## 15. Model size and speed: where the parameters and the time actually go

### Parameter share of the correction branch

| config | correction | total | share |
|---|---|---|---|
| 8ch `max_L=0` (debug) | 51,976 | 75,568 | **68.8%** |
| 32ch `max_L=0` | 53,512 | 107,896 | 49.6% |
| 128ch `max_L=0` | 59,656 | 421,528 | 14.2% |
| 128ch `max_L=1` (production) | 59,656 | 616,600 | **9.7%** |

The correction is nearly **width-independent** (52k -> 60k), so it dominates only at debug
width. `carrier_pooling` is 96.5% of it: eight MLPs (four `u^c`, four logits), each
`in_dim -> hidden -> 1`, where `in_dim = carrier_feature_dim * n_layers +
counter_embedding_dim`. The first layer is the entire cost.

`counter_embedding_dim` and `carrier_mlp_hidden` are now **halved** (32 -> 16, 64 -> 32) in
both the arg parser and the run script. The counter embedding maps six numbers; the
readouts emit one scalar each. Neither needed the width.

### Parameters are not what makes it slow

| | ms/step |
|---|---|
| base `ScaleShiftMACE`, forward only | 38.97 |
| MACEDefect, forward only | **38.90** |
| base, with forces | 78.09 |
| MACEDefect, with forces | 92.40 |
| MACEDefect, small heads, with forces | 92.52 |

The correction's **forward pass is free** and shrinking the heads changes nothing. The
whole overhead is the backward pass, specifically the extra `autograd.grad` calls added
for `n_ref`:

```
3 grad calls (current: total, corr(n), corr(n_ref))   97.13 ms
2 grad calls (corr_ref skipped)                        83.02 ms
1 grad call  (both correction grads skipped)           70.41 ms
```

~13-14 ms each. FP64 in the segment softmax costs nothing (92.40 vs 92.74) despite the
A4000's 1:32 FP64 ratio, because the tensor is `[n_nodes, 4]`.

A 2-call restructure is available (compute total forces and forces-at-`n_ref`, then
`delta_forces = forces - forces_at_ref`), worth ~14 ms. It would make `base_forces` exact
only where `n_ref = 0` -- true wherever base labels exist today, but A5.5 anticipates base
labels at defect geometries, which would make it silently wrong. Not done; needs an
assertion if it ever is.

### Backends: cuEquivariance is the win, and it is a memory win

Both accelerate the **trunk** only, which is ~85% of step time. Installed
`cuequivariance-torch` 0.11.1 and `openequivariance` 0.6.8.

128 channels, `max_L=1`, 286-atom cells, float32, A4000 16 GB:

| batch | backend | ms/step | peak MiB | ms/frame |
|---|---|---|---|---|
| 2 | e3nn | 343.3 | 8257 | 171.7 |
| 2 | cueq | 159.8 | **1835** | 79.9 |
| 2 | oeq | **126.2** | 4604 | 63.1 |
| 4 | e3nn | OOM | - | - |
| 4 | cueq | 139.5 | 3648 | 34.9 |
| 4 | oeq | 233.4 | 9144 | 58.3 |
| 8 | cueq | 141.1 | 7274 | **17.6** |
| 8 | oeq | OOM | - | - |

**cuEquivariance uses 4.5x less memory than e3nn**, which is what lets the batch grow, and
throughput follows: 17.6 ms/frame at batch 8 against e3nn's 171.7 at batch 2, roughly
**10x**. OpenEquivariance is the faster single step at batch 2 but takes 2.5x cueq's
memory and OOMs at batch 8, so it loses on throughput despite the fused tensor products.

Narrower configs, batch 8: 8ch `max_L=0` **0.85x (slower)**, 32ch 1.10x, 64ch/`max_L=1`
1.34x. Kernel overhead dominates small models -- leave cueq off for debug runs.

`ENABLE_CUEQ` added to the run script (default `False`), `MACEDefect` added to the cueq
allowlist, and `tests/unit/test_defect_cueq.py` pins numerical equivalence and that the
correction survives conversion. oeq remains blocked for MACEDefect (untested, and worse
memory). cueq is blocked with `USE_LONG_RANGE=True` (untested).

### The cueq+oeq hybrid does not work in this tree

`extract_config_mace_model` reads `model.cueq_config`, but the cueq conversion does not
persist that attribute on the model (`hasattr` -> False). Chaining cueq -> oeq therefore
drops the cueq half at config extraction and then fails loading the cueq-layout weights
into an e3nn+oeq model:

```
HYBRID FAILED: RuntimeError Error(s) in loading state_dict for MACEDefect
```

`run_train` independently disables oeq whenever both flags are set. Making the hybrid work
would need cueq_config persisted on the model and that mutual exclusion lifted — both
upstream changes, and the numbers above say cueq alone already wins on memory and
throughput, so it is not the lever.

### Environment note: openequivariance needs a CUDA toolkit

It JIT-compiles kernels, so it needs `nvcc` and CUDA headers, which a pip torch install
does not provide. Installed `cuda-nvcc`/`cuda-cudart-dev` 12.9 into the env, then had to
make `$CUDA_HOME` look canonical, since conda puts headers under
`targets/x86_64-linux/include` and the pip `nvidia/*` wheels put libraries in their own
package directories:

```bash
CH=$HOME/micromamba/envs/py13
ln -s $CH/targets/x86_64-linux/include/*.h        $CH/include/     # + subdirs crt/, cuda/, ...
ln -s $CH/lib/python3.13/site-packages/nvidia/*/include/*  $CH/include/
ln -s .../nvidia/cublas/lib/libcublas.so.12       $CH/lib/libcublas.so
export CUDA_HOME=$CH
```

Also clear `~/.cache/torch_extensions` after any of this: a failed build is cached and
reported as an unrelated `ImportError: Could not import DeviceProp`.

---

## 16. Stage C result: escape epoch rises with cell size

Matched subsets on v4 — 135 configs, 9 base labels, 63 delta targets, 126 `L_tot` frames
in **each** arm, identical 288-atom pristine pool, 8 channels, `max_L=0`, EMA off, cueq
off, 150 epochs, 3 seeds. The only difference is the defect cell size.

Escape = first epoch whose `RMSE_dF` falls 10% below that arm's own plateau value. An
absolute threshold would be wrong here: `RMSE_dF` averages over every atom while the delta
forces sit on six, so the larger cell starts *lower* (37.81 vs 50.66) purely by dilution,
before any training.

| seed | 286 atoms | 398 atoms | ratio |
|---|---|---|---|
| 1 | 22 | 116 | 5.27 |
| 2 | 50 | 59 | 1.18 |
| 3 | 43 | 57 | 1.33 |
| **median** | **43** | **59** | **1.37** |
| mean | 38.3 | 77.3 | 2.02 |

**Escape is later at 398 atoms in 3/3 seeds**, and the median ratio 1.37 sits almost
exactly on the cell-size ratio 398/286 = **1.39**. That is the linear scaling the plan
predicted for the mechanism where the defect-specific fraction of the `u` gradient goes as
`n_d/N`.

### The control that makes this readable

The base branch trains essentially identically in both arms, so the cell size is doing the
work rather than some difference in how well the trunk fit:

```
        RMSE_F  epoch 0 -> epoch 149
286 s1   693.96 -> 42.39      398 s1   698.24 -> 43.16
286 s2   715.04 -> 37.47      398 s2   719.11 -> 40.75
286 s3   715.68 -> 37.81      398 s3   720.28 -> 42.56
```

### Verdict, and how far to trust it

Under the plan's decision rule this is the "escape epoch rising with N" branch: **the
plateau is a production-scale hazard, and Stage D is justified on dynamics** rather than
having to earn its place on the `Delta u` diagnostic alone. Every production cell is
larger than these.

Caveats, none of which change the direction:

- Three seeds. A paired sign test on 3/3 gives p = 0.125 — consistent, not conclusive.
- Variance is large (286: 22-50; 398: 57-116). The mean ratio of 2.02 is inflated by
  seed 1, whose 286 run escaped unusually early at 22; the median is the honest summary.
- The lever arm is narrow, 1.39x, which the plan itself flags. Smaller cells (<=128
  atoms) would widen it, and none exist in this dataset.
- The agreement between the median ratio (1.37) and the size ratio (1.39) is closer than
  three noisy seeds can really support. Read it as "consistent with linear", not as a
  measured exponent.

`c_n398_s1` finishes at `RMSE_dE` 77.65 because it escaped at epoch 116 and had only ~34
epochs left to converge — that is the cost of a long plateau, not a separate failure.

---

## 17. Stage D implemented — the tie as a switchable mode

`alpha = softmax(-beta * u)`, with `beta = 10 eV^-1` a fixed gauge constant recorded with
the model (`--defect_alpha_mode {logits,tied}`, `--defect_beta`, `DEFECT_ALPHA_MODE` in the
run script).

**Implemented as a mode, not a deletion.** Plan D1 says to delete `MLP_l`, but D5 can
*reject* the tie on an accuracy regression, and the A/B that decides it is impossible if
the alternative has been removed. Under `tied` the logit networks are not constructed at
all (`len(logit_readouts) == 0`, zero parameters), so the parameter saving is real rather
than bypassed. Deletion follows a passing gate.

Two details worth recording:

- **The logit clamp is not applied under the tie.** Clamping to +-40 would break
  `alpha == softmax(-beta u)` exactly where `|u| > 4 eV` — that is, at the bound site the
  model is supposed to find. `segment_softmax` subtracts the per-cell maximum anyway, so
  the large-logit case is already numerically safe.
- **`delta_u` is now logged per channel** (mean-minus-min of `u`, computed identically in
  both modes so the arms are comparable). Under the tie `logit_gap == beta * delta_u`
  identically, which the smoke run confirms: `gap_l = 23.345` against
  `delta_u = 2.335`.

### Smoke test: the attention localises far faster

`dataset_beta`, 4 epochs, tied:

```
Initial  partic=[286.0 286.0 286.0 286.0]   delta_u=[0.000 0.000 0.000 0.000]
Epoch 0  partic=[278.4 285.9 286.0 285.9]   delta_u=[0.018 0.002 0.000 0.002]
Epoch 1  partic=[141.0 282.2 286.0 279.1]   delta_u=[0.287 0.012 0.000 0.017]
Epoch 3  partic=[ 64.2 270.8 286.0 277.4]   delta_u=[2.335 0.027 0.000 0.019]
```

`e_maj` goes 286 -> 64 in four epochs. The untied model needs ~20+ epochs to move at all.
That is the D2(1) claim looking promising — but it is four epochs on the small set, and
`RMSE_dF` has not moved yet, so it is a hint, not the result.

**One thing to watch in the A/B:** `mean_u` for `e_maj` ran to -6.75 eV and
`RMSE_E_per_atom` rose to 2542 meV over those four epochs. Under the tie a uniform shift
in `u` does *not* cancel — it moves `Delta E` by `n_c * c` — so the level is observable,
not gauge. But `e_maj` is occupied in *both* the ground and excited states, so its level
cancels in the paired difference and is constrained only by `L_tot`. If that drift
persists in the full A/B it is the level mode that §D-opt exists to close, now visible in
the energy rather than hidden.

### Tests (plan D3)

`TestTiedAttention`, 8 tests: no logit network exists; `alpha` equals `softmax(-beta u)`
against a direct computation; the `n = 0` identity survives; `sum_i alpha_i == 1`; the
correction is live at `u == 0`; `logit_gap == beta * delta_u`; and finite-difference
forces at two counter vectors, which is what checks the new `-beta n_c Cov_alpha(u, grad u)`
term in the forces.

> One of these tests was wrong on the first attempt in a way worth remembering: it probed
> "is the correction live at `u == 0`?" with a *squared* loss. `delta_energy` is exactly
> zero there, so the gradient vanishes by construction and the test would have passed or
> failed for reasons unrelated to the mechanism. It uses a linear functional now.

The carrier-transfer scan of D3 (two competing sites, sweep a coordinate, check `ΔE(Q)`
and `ΔF(Q)` for kinks) is **not** written yet. It is the test that decides whether the D4
residual head is needed, and it wants a purpose-built geometry rather than a dataset frame.

### A/B queued

`run_stage_d.sh` runs `d_tied_s{1,2,3}` against the already-running `a4_v4_s{1,2,3}`
baseline — same dataset, shape, seeds and settings, differing only in attention mode. It
is chained to start when the baseline finishes, and carries the same `flock` guard that
stopped two Stage C drivers racing.

---

## 18. A4 baseline of record on v4 (40 epochs) — A5 made escape earlier and more consistent

Full `dataset/` v4, 8ch `max_L=0`, float32, EMA off, cueq off, 3 seeds, **40 epochs**
(shortened deliberately for iteration speed; both A/B arms share the budget).

| run | escape epoch | final dE | final dF | best dE |
|---|---|---|---|---|
| `a4_v4_s1` | 15 | 37.16 | 26.63 | 36.42 |
| `a4_v4_s2` | 14 | 38.50 | 27.16 | 37.36 |
| `a4_v4_s3` | 16 | 50.56 | 29.64 | 44.70 |

**Probe floor on this split is 53.72 meV** (fitted on train, scored on the same validation
split). All three seeds beat it.

**The 40-epoch worry did not materialise, and the reason is interesting.** I flagged that
40 epochs might cut off before escape, because v3 saw escape at epochs 22 and 45. On v4
with `L_tot` restored, escape is at **14, 15, 16** — earlier and far more consistent
(v3: 22 and 45; the beta A/B ranged 10-102). The most likely reading is the mechanism
Stage C exposed from the other direction: the correction cannot bootstrap without base
supervision, so *more* supervision at defect geometries makes the trunk features develop
sooner and the correction escapes sooner. A5 was aimed at `RMSE_F`; shortening the plateau
looks like a second benefit. Worth stating as a plausible mechanism, not a measured one —
v3 and v4 differ in dataset composition as well as in `L_tot`.

**The seed-dependent channel split persists**, exactly as §A5.1 says it must (restoring
`L_tot` adds no new counter):

```
s1  partic=[  9.99   2.90 346.53  12.87]  mean_u=[ 1.539  0.534  0.000 -0.174]
s2  partic=[131.19   2.85 346.00   9.63]  mean_u=[ 2.638  0.525  0.000  0.248]
s3  partic=[ 89.81 106.81 346.37 285.74]  mean_u=[ 1.272 -0.057  0.000  0.004]
```

`h_maj` stays dead at participation = N with `std_u` exactly 0.000 in all three.

### Two process notes, both mine

- **Seed 3 was rerun.** Its first attempt died of CUDA OOM at epoch 14 because I was
  running the cueq/oeq benchmarks and the tie smoke test on the same GPU. It was rerun
  alone.
- **Seeds 1-2 predate the `delta_u` logging; seed 3 followed it.** The difference is
  diagnostic-only: `delta_u` is computed and printed but never enters the loss, and the
  `alpha_mode='logits'` path constructs exactly the modules it did before in the same
  order, so the initialisation for a given seed is unchanged. The three seeds are
  comparable; only `a4_v4_s3.log` carries the `delta_u` column.

### Stage D launch also needed two attempts

The first Stage D launch died silently mid-startup: the runs were started from a chain
whose parent was a tool-session process, and the process group was cleaned up under them.
The orphaned CUDA processes then kept ~10 GB reserved, so the *second* attempt OOM'd on a
GPU that `nvidia-smi` would have shown as busy. Fixed by killing the orphans and relaunching
under `setsid` with `expandable_segments`. **Long GPU jobs on this node need `setsid`, not
just `nohup`.**

---

## 19. Stage D result: the tie is bimodal. Gate REJECTS it as-is. Do not delete MLP_l.

Same dataset (v4), shape, seeds, settings and 40-epoch budget; only `alpha_mode` differs.

| run | escape | final dE | final dF | best dE |
|---|---|---|---|---|
| `a4_v4_s1` logits | 15 | 37.16 | 26.63 | 36.42 |
| `a4_v4_s2` logits | 14 | 38.50 | 27.16 | 37.36 |
| `a4_v4_s3` logits | 16 | 50.56 | 29.64 | 44.70 |
| `d_tied_s1` | **none** | **130.52** | 45.49 | 121.46 |
| `d_tied_s2` | **1** | 36.30 | 27.88 | **36.30** |
| `d_tied_s3` | 16 | 55.74 | 27.23 | 44.44 |

Not a small effect either way. Seed 2 escapes at **epoch 1** — the plateau essentially
abolished, and the best `dE` of all six runs. Seed 1 never escapes at all: `RMSE_dF` pinned
at 45.6 for the full 40 epochs and `dE` at 121-135, *above* the target σ of 122.

### The failure is lock-in onto the wrong atoms, measured

Autopsy on the final models, validation frames, 6 defect-shell atoms of 382 (uniform
attention would put 0.0157 of `alpha` there):

```
                partic  partic/N  mean_u  std_u   a_defect
d_tied_s2 e_maj    2.5     0.007   2.274  0.325     0.9892
d_tied_s1 e_maj   92.5     0.253   2.181  1.924     0.0000
```

`d_tied_s2` puts **98.9%** of its `e_maj` attention on the six defect neighbours — the
cleanest localisation seen in this project. `d_tied_s1` puts **exactly zero** there, having
committed instead to ~92 atoms that are not the defect. Both have a large `delta_u`
(1.08 vs 2.11) and a wide-open logit gap, so the usual diagnostics look *healthy* in the
failed run; only `a_defect` distinguishes them.

### Why the tie does this, and why §0's dismissal of the anneal missed it

`alpha = softmax(-beta u)` with `beta = 10 eV^-1` is a hard, immediate commitment: a 0.3 eV
accident in `u` is already a factor `e^3 ~ 20` in attention. With the logits produced by a
separate network, an unlucky early `u` is averaged against a slowly-training `MLP_l`; under
the tie there is nothing to damp it, and the correction is a positive feedback loop —
attention concentrates where `u` is low, which lowers `u` there further.

Forward plan §0 rejects a `beta` anneal on the grounds that
`d(Delta E)/du_j = n_c alpha_j [1 - beta(u_j - <u>_alpha)]` is `n_c/N != 0` at `u == 0`, so
"nothing needs seeding". **That reasoning is correct and answers a different question.**
The failure is not a dead start, it is *premature commitment*: the run seeds fine and then
locks onto the wrong region within the first epochs. An anneal from `beta ~ 2` to `10` —
which the superseded fix plan §2.3 proposed for exactly this symptom — targets the actual
failure. §D-opt's gauge penalty would not: it constrains the *level* of `u`, not which
atoms are selected.

### Gate (plan D5)

D5 requires held-out accuracy no worse than baseline within seed noise **and** at least one
of D2(1)/D2(2), and says to reject on any accuracy regression.

- **D2(1) plateau elimination: realised, spectacularly, in 1/3 runs** (escape at epoch 1
  against a baseline of 14-16) and matched in a second (16).
- **Accuracy: regression in 1/3 runs**, and not marginal — 130.52 vs 37-50 meV, worse than
  predicting the mean.

**The gate therefore rejects the tie as it stands, and `MLP_l` must not be deleted.** But
this is a rejection of the *current form*, not of the idea: the mechanism is understood,
the failure has a named cause and a specific candidate fix, and the successful arm is the
best result in the project so far.

Recommended next step, if Stage D is pursued: `beta` annealed 2 -> 10 over the first ~5
epochs, terminating at the recorded production `beta` before any reported number (the
anneal changes the energy function, not just the optimiser). Re-run the same 3-seed A/B.
If lock-in persists at low `beta`, the tie is not salvageable by scheduling and D4's
residual head or a return to independent logits is the answer.

**Caveat:** three seeds, 40 epochs. A 1/3 catastrophic failure rate is estimated from one
occurrence; the true rate could be anywhere from rare to routine. What is *not* in doubt is
that the failure mode exists and that `a_defect = 0.0000` identifies it.

---

## 20. D7.1 seeding: the descriptor works, but it cannot be applied at step 0

### The descriptor is excellent

`node_power_spectrum` in `mace/modules/defect_seed.py`, built from the model's own radial
basis and spherical harmonics with one-hot neighbour species, contracted over `m`. Verified
invariances (float64, random rotation / permutation / translation):

```
rotation     max|Δp| = 6.0e-15      permutation  3.3e-16      translation  1.0e-15
```

And it identifies the defect shell, label-free, on the real dataset (6 shell atoms per
frame, 2164 atoms total):

| descriptor | defect vs bulk | mean rank of shell atoms |
|---|---|---|
| `l = 0` only | 2.97σ | 66 of 2164 |
| all `l` (`max_ell=3`) | **15.02σ** | **18 of 2164** |

**This is D7.1's "free by-product" delivered:** angular terms improve the separation 5×,
which is direct evidence for Stage B's `max_L = 1` sweep, obtained without paying for
equivariant message passing.

### But the target is not expressible at initialisation

D7.1 says to solve `MLP_u`'s last layer against the centred target. Measured on the real
data, that fails in both possible directions:

| ridge | fit residual (eV) | max \|W\| |
|---|---|---|
| 0 (plain lstsq) | 0.025 | **66 899** |
| 1e-4 | 0.097 | 3.5 |
| 1e-2 | 0.098 | 0.02 |

Target std is 0.0976 eV, so a residual of 0.097 means **nothing is explained**. The
unregularised solution does fit — with weights of norm ~7e4, which diverge within one epoch
once training moves the activations (observed: `RMSE_dE` reaching 1.3e6 meV by epoch 0).

Fitting the *whole* readout by gradient descent does not rescue it either. 2000 Adam steps
at lr 0.05, untrained trunk: rmse 0.09757 → 0.09546 against a target std of 0.09758, i.e.
**~4% of variance explained**.

### The cause, and the fix

The same fit against a **trained** trunk (`a4_v4_s3`, 40 epochs), readout reinitialised:

```
step    0  rmse 0.11279
step  500  rmse 0.03088
step 1500  rmse 0.02932     ->  variance explained 0.910
```

So the target *is* expressible from `[h_i, z(n)]` — **once `h_i` has been trained**. At
initialisation the trunk is a random message-passing network whose 32 invariant features
simply do not carry the novelty statistic in any form a small MLP can read.

This is a real constraint on D7.1 as written, and it has a certain irony: the seed is meant
to short-circuit the exploration phase, but it depends on the very trunk features that
develop *during* that phase. The seed can shorten the **attention** search; it cannot
precede the **feature** development that makes the search possible.

**Recommended amendment:** apply the seed after a short warm-up rather than at step 0 — the
trunk is trained by `L_base`/`L_tot` in those epochs regardless, and `a4_v4` shows escape at
14–16, so a warm-up of ~5 epochs is well inside the plateau it is trying to shorten. The
open question is whether a seed applied at epoch ~5 still beats the unseeded escape at
14–16, or whether by then the attention has already begun committing on its own.

**Not yet run: Experiment 1 (seeded baseline).** Running it as specified would measure a
seed that explains 4% of its target, which is not a test of D7.1. Deferred pending the
warm-up decision.

### State of the code

`--defect_seed_contrast <epsilon>` and `--defect_seed_max_batches` exist and are wired
through `run_train` and `train_defect_model.sh` (`DEFECT_SEED_CONTRAST`, default 0). The
seeding is a single pass over the loader — **the training loader shuffles, so a two-pass
implementation silently misaligns targets with features** (caught as a shape error only
because the batch sizes happened to differ). It fits each channel against its **own**
activations: a shared solve applied to all four readouts seeds channel 0 correctly and
sends the other three to |u| ~ 1e3 eV, since each has an independently initialised first
layer.

---

## 21. Logit seeding works: D2(1) realised, the tie is parked

Replaced the u-seed entirely (see §20 for why that route failed, and note the diagnosis
there was wrong — the target *is* in the readout's span, residual 0.025 at ridge 0; it
needs `|W| ~ 7e4`, which is ill-conditioning, not missing information).

`logit_i^c += gamma_c * s_hat_i`, gamma trainable per channel, descriptor rebuilt in-graph
every forward so forces differentiate it (FD-verified, 1.7e-9).

| run | escape | final dE | final dF |
|---|---|---|---|
| baseline s1/s2/s3 | 15 / 14 / 16 | 37.16 / 38.50 / 50.56 | 26.63 / 27.16 / 29.64 |
| **seeded s1/s2/s3** | **4 / 4 / 6** | 46.71 / 44.85 / 43.12 | 27.11 / 25.78 / 26.32 |

Escape falls ~3x in 3/3 seeds, accuracy inside the baseline spread (43-47 vs 37-51),
forces equal or better, and the seeded runs are tighter across seeds. That is D8.1's
condition, so **the tie is parked permanently and `MLP_l` stays**. D8 items 2 and 3 are
skipped by its own stopping rule, D4 is explicitly ruled out, and D-opt lost its
motivation when D6 retired D2(2). **Only D8.4 remains in stage D.**

### Why attention commits earlier than the escape metric suggests

Participation per epoch on the *unseeded* baselines, which is the measurement that placed
the seed:

```
a4_v4_s1   ep0 194.8 -> ep4 43.6 -> ep5 30.7 -> ep9 9.0     (escape 15)
a4_v4_s2   ep0 172.0 -> ep4 152.0 -> ep13 155.4             (escape 14)
a4_v4_s3   ep0 172.4 -> ep4 117.9 -> ep7 157.8              (escape 16)
```

Attention commits well before `RMSE_dF` moves. A warm-up at epoch 5 would already be too
late for seed 1 -- which is why the u-seed's "wait for the trunk" fix was the wrong shape,
independent of its conditioning problem.

### Anneal

`gamma -> 0` on readiness per channel, with a forced ramp reaching zero at an **absolute
epoch** (`--defect_seed_anneal_epochs`, default 30). Absolute rather than fractional
because early stopping makes the run length unknown in advance, and a fractional schedule
can let a model converge and stop with the seed still active -- shipping an inference-time
descriptor, which is the one outcome the anneal exists to prevent.

The schedule is a **ratchet**: monotone non-increasing by construction. Without it the
readiness reference (built from the current seeded gap) shrinks as gamma falls, and gamma
could tick back up -- re-imposing a prior the model was outgrowing.

---

## 22. Stage E preconditions: one passes decisively, one cannot be tested here

`stage_e_alpha_checks.py`. These matter because enabling the long-range branch promotes
`alpha` from an internal weighting to a physical object: `q_i^carrier` is the charge whose
self-term is the electron-hole interaction, and the energy only constrains the *pooled*
`sum_i alpha_i u_i`, so it is blind to how the weight is spread.

**Check 2, size stability — and it separates the two variants sharply.** Summed `alpha`
on the six defect-shell atoms, per channel:

```
baseline a4_v4_s3      N=286 [0.0344 0.0481 0.0210 0.1133]
                       N=398 [0.0176 0.0227 0.0151 0.0797]   monotone decline

seeded   e1b_logit_s1  N=286 [0.7784 1.0000 0.9994 0.9980]
                       N=398 [0.8052 1.0000 0.9962 0.9936]   flat
```

The seeded model puts **78-100% of its attention on the defect shell at every cell size**,
drift 0.027. The baseline puts 2-11% there and declines monotonically with N -- it would
have exported a nearly-uniform `alpha` as the latent charge, giving a badly wrong
electron-hole term while `Delta E` still looked fine. **Stage E should be run on the seeded
variant, not the baseline.**

Participation ~3 on the seeded model is physically sensible rather than suspicious: the
divacancy has three equivalent dangling bonds per sublattice, so a carrier localised on one
sublattice gives exactly that.

**Check 1, symmetry floor — INCONCLUSIVE and cannot be fixed with this dataset.** The floor
only binds on a relaxed, symmetric frame, and the calmest frame available has
|F|max = 1.113 eV/A. Every frame here is thermal, so a participation below 3 is legitimate
and the check has nothing to bite on. It needs a relaxed ground-state geometry, which is a
(cheap) external calculation, not a DFT campaign.

### Also done for Stage E

- `--freeze_amplitude` pins `a` at `1/sqrt(eps_inf)` with no gradient. `a` is not
  identifiable from a dipole term alone -- with q = 0 on every frame there is no monopole
  for it to scale -- so a free `a` drifts to whatever absorbs the electron-hole energy and
  then reads as a fitted screening constant while being nothing of the kind.
- `MACEDefect.__setstate__` fills in attributes added after a checkpoint was written.
  Whole model objects are pickled, so an older checkpoint previously failed on the *first
  forward* with `AttributeError: no attribute 'logit_seed'` -- at evaluation time, not load
  time, which is a confusing place to discover it.

> `eps_inf = 6.5` is a **literature** value for 4H-SiC, not the DFPT calculation the plan
> asks for. Record it as such: `1/a^2` cannot be checked against it as an independent
> result, and the plan 9.7 step-7 gate stays closed until a charged system exists.

---

## 23. Two anneal bugs, both caught by testing rather than by inspection

The first annealed run finished with a **live, sign-inverted seed**:

```
e1c_anneal_s1 final gamma = [0.0, -0.0013, 0.0, -0.092]
e1c_anneal_s3 final gamma = [0.0,  0.0,    0.0, -0.0839]
```

That is precisely the outcome the anneal exists to prevent — a shipped model carrying an
inference-time descriptor — and worse than a residue, because a negative gain does not
weaken the prior, it **inverts** it: attention is pushed away from the novel atoms.

Two independent causes, and neither was visible from reading the schedule:

1. **gamma was still a trainable parameter.** The schedule set it once per epoch and the
   optimizer moved it for the rest of the epoch. Fixed: while annealing, the schedule owns
   gamma outright (`requires_grad_(False)`), and the optimizer group is now conditional on
   `requires_grad` so the coverage guard stays honest.
2. **The ratchet preserved negatives.** `min(0, -0.09)` is `-0.09`, not `0`, so once
   anything drove a channel negative the monotone rule locked it there. Fixed: gamma is
   clamped non-negative, and the terminal zero is applied **last and unconditionally**, so
   at and past `zero_by_epoch` the model is bias-free whatever else has touched it.

`test_anneal_survives_an_optimiser_that_also_moves_gamma` reproduces the failure — it
perturbs gamma between schedule updates, as the optimizer did — and pins the invariant.
It found bug 2 after bug 1 was already fixed, which is the argument for writing it as a
test of the *outcome* rather than of the schedule.

**The first e1c results are void.** They are reported in §21 only as the measurement that
exposed the bug.

---

## 24. Queued pipeline

`run_pipeline.sh` runs the remaining stages in sequence (one GPU) and analyses each as it
lands, appending to `~/runs/pipeline_results.md`. A failing stage is reported and the
pipeline continues, so one broken stage cannot silently block the rest.

| stage | script | what it decides |
|---|---|---|
| e1c re-run | `run_experiment1c.sh` | does the anneal cost accuracy? |
| **D8.4** | `run_d84_stagec.sh` | does the linear N-scaling survive seeding? *The whole case for stage D.* |
| **Stage E** | `run_stage_e.sh` | how much energy does `Delta E_SR` give up to the electron-hole term? |
| **Stage B** | `run_stage_b.sh` | capacity: 128ch `max_L=0`, then `max_L=1` |

`analyse_runs.sh` is shared: escape epoch, final and best held-out metrics, the final
per-channel state, and the final gamma (which must read exactly zero on an annealed run --
that column exists because of §23).

Two things fixed in the run configuration for these stages:

- **Stage B has `ENABLE_CUEQ=True`, and that is not optional.** e3nn OOMs at batch 4 on
  this 16 GB card at 128ch/`max_L=1`; cueq runs batch 8 in 7.3 GB at ~2.4x the speed, and
  is verified numerically identical. Stage B runs one configuration at a time for the same
  memory reason.
- **Stage E cannot use cueq**: the conversion is only verified with `use_long_range=False`,
  and `run_train` refuses the combination rather than silently converting an untested
  branch.

### Stage E is the least-exercised code in the project

The long-range branch has never been trained. The forward pass is verified --
`a` frozen at exactly `1/sqrt(6.5) = 0.392232`, `sum_i q_i^host = 0` per cell,
`sum_i q_i = a*q`, and the `n = 0` identity exact -- and the flags parse, but the training
loop with LES is untried. The pipeline continues past a Stage E failure by design.

Note `sum_i q_i = a*q` is trivially satisfied here because `q = 0` on every frame. The
invariant cannot be tested in its meaningful form until charged data exists, which is the
same gap that keeps `a` frozen.

---

## 25. D8.4: the plateau is no longer a production-scale hazard

The stage C protocol re-run on the adopted variant (logit seed + anneal), matched 286 vs
398 subsets, 3 seeds each.

| | 286 atoms | 398 atoms | median ratio |
|---|---|---|---|
| unseeded (stage C) | 22, 50, 43 | 116, 59, 57 | 1.37 |
| **seeded (D8.4)** | **3, 5, 3** | **5, 9, 1** | 1.67 |

Median escape falls **43 -> 3** at 286 atoms and **59 -> 5** at 398: a 14x and 12x
reduction. Whatever N-dependence remains is now measured in single epochs.

**Read the ratio with care.** At 1-9 epochs with one-epoch granularity, a ratio of 1.67 is
one or two epochs of noise -- it is not evidence that the scaling *worsened*, and the
direction is weaker than before (later at 398 in 2/3 seeds, against 3/3 unseeded). The
defensible claim is not "the scaling is gone" but the one that actually matters:
**the plateau has stopped being a production-scale hazard.** Stage C justified stage D on
the grounds that plateau length grows with N and every production cell is larger than
these; a cost of 3-5 epochs at both sizes removes that argument whether or not a residual
slope survives.

Combined with §21, stage D is closed: the tie is parked, `MLP_l` is kept, and the D2(1)
objective is met by seeding alone.

### The anneal fix works, and the metrics now self-check

All six D8.4 runs and all three re-run e1c runs finish at **`gamma = [0.0, 0.0, 0.0, 0.0]`**
exactly. Two consequences visible in the logs:

- `gap_l == gap_int` identically on every annealed run -- the seeded and seed-free gaps
  coincide once the gain is zero, which is a free self-consistency check that the model
  really is bias-free rather than merely reported as such;
- `h_maj` returns to participation ~346 (uniform). The dead channel is untouched again
  once the seed is withdrawn, restoring the control that makes the live channels readable.

### Does the anneal cost accuracy? No.

| | escape | final dE | best dE |
|---|---|---|---|
| `e1b` gamma free | 4 / 4 / 6 | 46.71 / 44.85 / 43.12 | 42.71 / 40.31 / 37.88 |
| `e1c` annealed | 3 / 3 / 4 | 45.45 / 45.84 / 42.84 | 32.62 / 33.72 / 41.92 |

Escape is the same or slightly better, final `dE` is indistinguishable, and best `dE` is
better. So the bias-free model is not paying for its compliance -- **adopt the annealed
variant**, which is also the only one that can be shipped without carrying the descriptor.

One caveat for stage E: after annealing, localisation partially relaxes on some seeds
(`e1c_anneal_s1` ends at participation 44.8 for `e_maj`, against 5.7 for the un-annealed
`e1b_logit_s1`). Since stage E exports `alpha` as a latent charge, the alpha checks of §22
must be re-run on an **annealed** model, not on `e1b`.

## 26. The parity-plot energy offset is checkpoint selection, not a referencing bug

`plot_parity.py` on `b_128ch_L1_s1` showed every subset sitting ~15 meV/atom above its
parity line. The offset is real, but its cause is mundane and worth recording because it
will recur on every run and is invisible in the per-epoch log.

**It is not the dataset.** There is exactly one 8-atom cell in v5, it lives in `train`
only, and no `valid` frame is smaller than 286 atoms -- so it cannot move a validation
parity plot at all, and its pull on the `E0` regression is ~0.1 meV/atom. The tail out to
-7.65 eV/atom that exceeds the NEP figure's range is the `unpaired` 286-atom frames
(E/atom up to -7.726), not a small cell.

**It is not the referencing.** The bias is nearly identical across the three states
(+18.7 / +14.8 / +14.7), and a band-edge referencing error would differ per state by
construction -- the excited state carries two gaps, the ground state one. The two defect
states agree to 0.1 meV/atom, which is what a correct reference looks like.

**It is checkpoint selection under the loss weights.** MACE saves the checkpoint with the
lowest *total* validation loss. With `energy_weight=1.0` against `forces_weight=100`,
`delta_forces_weight=100` and `delta_energy_weight=10`, the energy term is under 0.5% of
that total, so selection is effectively blind to it:

| run | saved checkpoint | valid RMSE E |
|---|---|---|
| `b_128ch_L1_s1` | epoch 37 (selected) | 18.5 meV/atom |
| `b_128ch_L1_s1` | epoch 39 (logged) | 12.0 meV/atom |
| `b_128ch_L0_s1` | epoch 130 of 140 | 9.7 meV/atom |

Two epochs apart, 6.5 meV/atom in energy, and the selection took the worse one because it
had marginally better forces. A constant per-atom offset is exactly the quantity forces
cannot see -- `dE/dR` annihilates it -- so nothing in the dominant 99.5% of the loss
either penalises it or is perturbed by it.

**Consequences.** `analyse_runs.sh` reports the *last* epoch, which is not the epoch that
was saved; the two can disagree materially on energy. Always cross-check the `Error-table
on TRAIN and VALID` block that `run_train` writes at the end -- it is evaluated on the
saved artefact, and it agreed with `plot_parity.py` to 0.05 meV/atom on both runs checked
(18.5 vs 18.45; 9.7 vs 9.70), which is what validates the plotting script.

Training does remove the offset given enough epochs: `L0_s1` at 140 epochs ends at
+0.4 / +2.9 / +2.8 meV/atom bias, against +18.7 / +14.8 / +14.7 for the 40-epoch `L1_s1`.
If the offset matters for the deliverable, raise `energy_weight` rather than lengthening
training -- it fixes both the fit and the selection criterion at once.

### `energy_bias_diagnosis.py`

Separates the two faults a uniform offset could represent, since forces cannot: fits
`residual_per_cell = a*N + b` per `config_type`, so `a` is the per-atom part (a wrong
energy reference) and `b` the per-cell part (something localised mis-sized). On `L0_s1`
both are ~0 with R^2 < 0.03, i.e. no systematic component survives -- the residual there
is scatter, concentrated on the `unpaired` frames (+10.7 meV/atom against +1.6 for
`paired`), which are the least sampled (47 train frames) over the widest energy range.

### The loss budget, measured (`loss_term_budget.py`)

Picking `energy_weight` by eye fails here because the terms are not commensurate.
`base_energy` and the totals energy are normalised **per atom** and then squared, so they
carry a factor `1/N^2` -- about 1e-5 for a 300-atom cell -- while `delta_energy` is a
per-frame quantity that is not divided at all. Weight 1.0 on a per-atom term is not
"somewhat smaller" than weight 10 on a per-frame term; it is five orders smaller before
the weights apply. Measured on `b_128ch_L0_s1`, valid split:

| term | weight | share of loss | raw MSE | implied RMSE |
|---|---|---|---|---|
| `forces_weight` | 100 | 71.60% | 4.352e-4 | 20.9 meV/A |
| `delta_forces_weight` | 100 | 25.00% | 1.520e-4 | 12.3 meV/A |
| `delta_energy_weight` | 10 | 3.36% | 2.043e-4 | 14.3 meV |
| `total_energy_weight` | 0.25 | 0.04% | 9.320e-5 | 9.65 meV/atom |
| `energy_weight` | 1.0 | 0.0015% | 8.840e-7 | 0.94 meV/atom |

The two terms that see an absolute energy are together **0.04%** of the objective. That is
the quantitative form of the offset finding above.

The implied-RMSE column is **weighted**: the budget runs through the real loss, so it
carries `ref.weight`, the per-field weight columns and `config_type_weights`. Rows whose
mask is effectively uniform reproduce the error table (`total_energy` 9.65 vs MACE's 9.7;
`forces` 20.9 vs 20.7, which is what validates the harness); rows with a selective mask do
not, and are not meant to (`delta_forces` implies 12.3 meV/A against ~19 in the error
table). Read the share column for budgeting and the error table for accuracy.

**Which weight to raise is not obvious from the offset alone.** `base_energy` is already
fit to 0.94 meV/atom, so `energy_weight` has essentially nothing left to correct -- raising
it is near-inert. The lever is `total_energy_weight`, the only term carrying an absolute
energy at `n != 0` frames, where the residual is 9.65 meV/atom; `0.25 -> 10` moves it from
0.04% to 1.5% of the loss. `delta_energy_weight` must **not** be raised for this purpose:
it is a paired difference and is blind to a constant at any weight.

This is also the likely reason our energies sit ~50x above the NEP paper's 0.14-0.18
meV/atom while our forces beat theirs -- we are spending 96.6% of the objective on forces.

Applied in `run_stage_b_full.sh` only (`ENERGY_WEIGHT=10`, `TOTAL_ENERGY_WEIGHT=10`),
verified to reach the command line by a stub-`python` dry run. The defaults in
`train_defect_model.sh` were deliberately left at 1.0 / 0.25 while the stage B
continuation is in flight, because that script is invoked fresh per run and changing it
mid-queue would give `L1_s1` and `L1_s2` different weights from `L0_s1` and `L0_s2`,
silently breaking the max_L comparison. Flip the defaults once the continuation finishes.
