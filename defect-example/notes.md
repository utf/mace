# 4H-SiC divacancy — MACEDefect test case

Working notes for the first real-data exercise of `MACEDefect`. Read section 2 before
drawing any conclusion from a training run: this dataset does **not** test the
charge-aware half of the model, and a green error table is easy to over-read.

- Source data: `ideal.db`, `defect_ground_state.db`, `defect_excited_state.db` (ASE sqlite)
- Extraction: `extract_defect_dataset.py` → `dataset/`
- Training: `train_defect_model.sh`, `submit_slurm.sh`
- Model spec: `../charge_aware_defect_mlip_implementation.md` (§ references below point there)

---

## 1. Provenance

Verbatim from the record accompanying the source databases:

> This record contains data and code that accompany the paper "Optical line shapes of
> color centers in solids from classical autocorrelation functions". Specifically it
> includes databases in ase sqlite format (*.db) with reference data from density
> functional theory calculations.
>
> Note that the atom type information in the databases already includes the labeling of
> the defect environment that is expected by the NEP model, according to the following
> rules
>
> ```
> Si(gs) → P     C(gs) → N     Si(ex) → S     C(ex) → O
> ```
>
> The "bulk" species (Si, C) remain unchanged.
>
> In 4H-SiC Si and C occupy two symmetry inequivalent sites, commonly referred to as h
> and k. There are thus four different divacancy configurations, corresponding to the
> combinations (h, h), (h, k), (k, h), and (k, k). In this study, we consider the
> divacancy in 4H-SiC (Fig. 2), which hosts a bright transition and has been studied both
> experimentally and computationally. This defect features four localized levels in the
> α-spin channel and five localized levels in the β-spin channel that are occupied by two
> and one electron(s), respectively. In the ground state, the lone electron in the
> β-channel occupies the lowermost level, while in the first excited state, it is promoted
> to the next higher orbital, which is very nearly degenerate with the next-next higher
> orbital.
>
> Using DFT calculations and a 286-atom supercell we obtain a vertical excitation energy
> for absorption of (Fig. 2d) and a ZPL energy of E_ZPL = 0.97 eV.
>
> Collinear spin-polarized DFT calculations were performed using the projector augmented
> wave method as implemented in VASP with a plane wave energy cutoff of 520 eV and the
> PBEsol exchange-correlation functional. The Brillouin zone was sampled with
> automatically generated Γ-centered k-point grids with a maximum spacing of 0.25 Å⁻¹.
> The excited state was obtained by enforcing the occupation of the first defect level in
> the β-spin channel to zero (β, n = 1) and the occupation of the second level to one.
>
> We set up an MLP that distinguishes bulk Si and C representing atoms not directly
> involved with the defect as well as Si_gs, C_gs, Si_ex, and C_ex, which represent the
> nearest neighbors of the divacancy in the ground (gs) and excited (ex) states.

### 1.1 Database contents

| File | Frames | Supercells (atoms) | Sampling protocols |
|---|---|---|---|
| `ideal.db` | 59 | 8, 288, 320, 384 | primitive, rattle, npt |
| `defect_ground_state.db` | 641 | 286, 318, 382, 398 | nvt, npt, rattle, nvt_gs, nvt_ex, npt_gs, npt_ex, config_coord |
| `defect_excited_state.db` | 641 | as above | as above |

Every frame carries energy, forces and stress.

**504 of the 641 frames in each defect database are paired.** They carry a `run` key and
are the *same geometry* evaluated in both electronic states — verified here to
bit-identical raw positions across all 504. The remaining 137 per database are
independent trajectories with no partner.

---

## 2. Scope — what this dataset can and cannot establish

The divacancy excitation is an **intra-defect promotion within the minority spin
channel**: one electron, one hole, **net charge zero on every frame**. That single fact
determines what the experiment is worth.

### 2.1 Genuinely tested

- The data layer end to end: counter canonicalisation, the pair join, band-edge
  referencing arithmetic, the base and delta targets, the loss masks, the `DefectRMSE` table.
- **`ΔE_SR`, hard.** After the species remap (§3.1) the trunk sees *identical input* for a
  ground/excited pair — same atomic numbers, same positions, same cell. The entire 0.94 eV
  must come through `CounterEmbedding` → `CarrierAttentionPooling`. This is strictly harder
  than the published NEP setup, where the relabelled species make gs and ex different
  compositions and two independent PESs can be learned. Success is evidence the
  counter-conditioned correction can carry an electronic-state difference the descriptor
  cannot see; failure is evidence the attention pooling cannot localise a neutral exciton.

### 2.2 Not tested — and cannot be

- **The charge machinery.** `net_charge` is 0 everywhere and the counter algebra collapses
  to a binary label (two vectors, ever). `E_dilute`, the isolated evaluator, the three-piece
  decomposition, formation energies and transition levels — the whole §7.1/§7.2 motivation
  for the architecture — never run in anger.
- **Gate: `1/a²` vs DFPT ε∞ (§12.4).** Unclosable in principle. `signed_counts · counts = 0`,
  so `q_carrier` is charge-neutral, there is no monopole, and the screening amplitude barely
  enters the energy. Trap: `a` will sit near its `softplus⁻¹(1/√ε∞)` initialisation and look
  plausible *precisely because nothing fitted it*. Do not report it as a result.
- **Gate: logit-gap stability (§12.4).** Unclosable as the code stands. `logit_gap` and
  `screening_amplitude` are produced by `MACEDefect.forward`
  (`mace/modules/defect_models.py:389,394`) and `screening_amplitude` is consumed by the
  eps_inf prior, but **neither is written to the `DefectRMSE` table or the log**, despite
  §7 of the blueprint stating they are. Closing this gate needs a patch to
  `mace/tools/{train,tables_utils}.py`, or a standalone checkpoint diagnostic.

### 2.3 Known weaknesses of the referencing here

The band edges in `dataset/band_edges.json` are a **fitted gauge, not PBEsol band edges**.
Two independent reasons the referencing carries no physical content in this test case:

1. It is circular — the effective gap is fitted from the mean of the very data it references.
2. The transition is a promotion between deep defect levels that never involves either band
   edge, so "learn the carrier binding energy relative to the band edge" does not apply.

Consequence: the referencing is a constant offset. `--gap-margin 0` is used so the delta
targets are centred at zero and no arbitrary number enters the story. Energies the trained
model returns are on a referenced scale with no absolute meaning; only differences at fixed
carrier counts are interpretable.

### 2.4 Representational ceiling

Any *second* intra-defect excitation would receive the same canonical counter vector
`(1, 0, 1, 0)` and therefore the same prediction. Harmless here (one excited state), but it
is a limit of the counter representation for intra-defect physics — distinct from the two
open gates above.

Not a problem: identifying a β-channel with an α-channel excitation is *correct*. Time
reversal maps the M_s = +2 reference with a β excitation onto M_s = −2 with an α excitation,
so the M_s = 0 lexicographic tie-break in `canonicalise_counts` is doing the right thing.

### 2.5 To test the charge half

Charged frames are required. The blueprint's own GaN V_Ga, or V_Si in 4H-SiC, or NV in
diamond (q = 0 and q = −1 both well characterised).

---

## 3. Extraction

`extract_defect_dataset.py` — standalone, `ase` + `numpy` only, deterministic under `--seed`.

### 3.1 Species remap (mandatory, not cosmetic)

`P → Si`, `N → C`, `S → Si`, `O → C`. The output contains only Si and C, so the `z_table`
is `{6, 14}`.

This is not optional: `_assert_same_geometry` (`mace/data/defects.py:247`) compares
`atomic_numbers` between members of a `pair_id` group, so without the remap every pair join
would raise. The relabelling and the pairing are mutually exclusive.

### 3.2 Counter vectors

| State | Written | Canonical (loader) |
|---|---|---|
| ground | `0 0 0 0` | `0 0 0 0` |
| excited | `0 1 0 1` | `1 0 1 0` |

`multiplicity` is deliberately **not** written. `validate_counts` computes the expected
multiplicity as `M_s + 1`, which assumes a closed-shell reference; the divacancy ground
state is a triplet, so the cross-check would fire spuriously.

### 3.3 Band-edge gauge

Vertical excitation over all 504 pairs: **mean 0.9430 eV**, std 0.1245, range [0.5691, 1.3905].

| Protocol | n | ⟨ΔE⟩ / eV | σ / eV |
|---|---|---|---|
| `nvt_gs` | 160 | 1.0087 | 0.0783 |
| `nvt_ex` | 160 | 0.8656 | 0.0633 |
| `npt_gs` | 72 | 1.0349 | 0.1010 |
| `npt_ex` | 72 | 0.8793 | 0.0869 |
| `config_coord` | 40 | 0.9397 | 0.2430 |

The `_gs`/`_ex` split of ≈0.14 eV is the Stokes shift; the ZPL of 0.97 eV sits between.
`config_coord` is the smallest stratum but has the widest spread — hence stratified
sampling (§3.4).

Gauge written to `band_edges.json`: `e_cbm_cell = +0.4715`, `e_vbm_cell = −0.4715`
(gap = 0.9430 = mean + margin 0). Only the difference enters, since every non-zero counter
vector here has exactly one electron and one hole.

### 3.4 Sampling and splitting

- Stratified-random over (protocol, supercell size) within each of four pools: paired,
  unpaired ground, unpaired excited, ideal. Still random — but plain global sampling would
  leave the `_gs`/`_ex` balance and the `config_coord` share to chance.
- **Whole groups** are split train/valid, so a pair is never split and a geometry never
  leaks across the split.
- `--frac-*` values are renormalised. Passing one without the others rescales the rest —
  the script now prints the resolved split and any pool that capped out.

### 3.5 Current dataset

```
python extract_defect_dataset.py --n-structures 800 \
    --frac-paired 0.85 --frac-unpaired 0.10 --frac-ideal 0.05 \
    --gap-margin 0 --seed 42
```

| | train | valid |
|---|---|---|
| paired (gs + ex) | 306 + 306 | 34 + 34 |
| unpaired ground | 36 | 4 |
| unpaired excited | 36 | 4 |
| ideal | 36 | 4 |
| **total** | **720** | **80** |

Loader confirms: 306 delta-weighted train configs, 378 base-weighted; delta targets
mean +0.003 eV, σ 0.124 (valid: 34 / 42, mean −0.019, σ 0.108).

`--E0s average` fits `E0_C = E0_Si = −7.898369 eV`. **Rank 1 of 2** — Si and C counts are
equal in every structure, so the design matrix is rank deficient and `lstsq` returns the
minimum-norm split. That is a gauge, not two determined atomic energies. Harmless at fixed
stoichiometry, but it will look odd in the training log.

---

## 4. Which loss term each frame trains

Masks derive from the data, not from config type (§6 of the blueprint).

| `config_type` | counters | `pair_id` | trains |
|---|---|---|---|
| `paired_gs` | `0 0 0 0` | yes | `L_base` (owns the group's base weight) |
| `paired_ex` | `1 0 1 0` | yes | `L_Δ` (energy + forces) |
| `unpaired_gs` | `0 0 0 0` | no | `L_base` |
| `unpaired_ex` | `1 0 1 0` | no | `L_tot`, base branch detached |
| `ideal` | `0 0 0 0` | no | `L_base` |

Note on `L_tot` (`mace/modules/loss.py:812`): its **energy** term is down-weighted to
`--total_energy_weight 0.1`, but its **force** term runs at the full `--forces_weight`.
Unpaired excited frames are strong force signal, not a weak term.

Statistics are computed on the `n = 0` subset only — both the E0 fit and the energy scale.
The log reports this as "Using 378 of 720 configurations (n = 0)".

---

## 5. Training

See `train_defect_model.sh` (single node) and `submit_slurm.sh` (cluster). Everything is
overridable from the environment.

> **The correction branch did not train at all until 2026-08-08.** `get_params_options`
> builds the optimizer from an explicit whitelist of submodules — `node_embedding`,
> `interactions`, `products`, `readouts`, plus a few opt-in extras. `MACEDefect`'s
> `counter_embedding`, `carrier_pooling`, `defect_feature_readouts` and `latent_charges`
> matched none of them, so **38 of 64 parameter tensors (52k of 75k parameters, 69% of
> the model) were never passed to the optimizer** and sat at their initialisation for
> every run. Fixed, with a coverage guard that now raises if any trainable parameter is
> in no group.
>
> **Why nothing caught it.** `MLP_u` is zero-initialised, so a frozen correction branch
> predicts *exactly* zero. That makes `E_total(R, 0) == E_base(R)` hold trivially, puts
> `RMSE dE` exactly on the target σ, and leaves the logit gap perfectly constant — the
> §12.4 gate "the logit gap must stay stable during training" would have read as
> **passing**, because nothing was training. Every §8.1 identity test passed throughout.
> The base branch trained normally (E 291 → 22 meV/atom), so the error table looked
> healthy. Treat any result produced before this fix as base-branch-only.
>
> The `les_readouts` entry in `get_params_options` is the same bug, hit once for MACELES
> and patched for that one submodule without fixing the pattern.

### 5.0 Per-epoch metrics

They now appear on stdout and in the log; nothing needs adding to the run script.
`valid_err_log` had no `DefectRMSE` branch and no `else`, so the epoch line was silently
dropped for exactly this error table. Fixed, plus a fallback that prints the loss for any
unmatched table rather than nothing. Format:

```
Epoch 3: head: Default, loss=0.06467135, RMSE_E_per_atom=   20.53 meV,
         RMSE_F=   83.16 meV / A, RMSE_dE=  115.43 meV, RMSE_dF=   46.11 meV / A
```

Watch `RMSE_dE` against §5.1's per-split σ. If it is *bit-identical* across epochs, the
correction branch is frozen again — that signature is what exposed the optimizer bug.

Fixed choices worth knowing:

- **`--use_long_range False` by default.** With q = 0 there is no monopole and `a` is
  unidentifiable, so the first run keeps it out of the way. Turning it on requires `les`
  (not on PyPI, pinned by commit in `requirements/les.txt`); the preflight checks for it.
- **No Stage Two / SWA.** `get_swa` has no branch for `--loss defect` and falls back to a
  plain energy/forces loss, which would train the total energy against the referenced
  labels and destroy the base/delta decomposition.
- **`--default_dtype float64`.** Costly on GPU (see `submit_slurm.sh`) but it is MACE's
  training default and the base-branch labels are ~2600 eV totals with meV-level structure.

### 5.1 Baseline to beat

The delta targets have σ ≈ 0.124 eV and are centred at zero. **A model that has learned
nothing scores `RMSE dE ≈ 124 meV`**, whatever the E and F columns say. Check this column
against 124 before anything else.

The threshold is **per-split σ**, not the global 124: train σ is 0.124 eV and valid σ is
0.108 eV, so a valid `RMSE dE` of 110 is baseline, not "slightly better than baseline".

Confirmed empirically: a 2-epoch run on this dataset reports `RMSE dE` of 124.0 (train)
and 109.8 (valid) meV — the untrained correction predicting zero, reproducing each
split's σ exactly.
On the earlier `--gap-margin 0.2` dataset the same run gave 228.7 meV, which is
`√(204² + 116²)` for a target distribution with mean −204 meV and σ 116 meV. Both match to
four significant figures, which is what confirms the referencing arithmetic, the pair join
and the metric agree with each other.

### 5.2 Local runs on Apple silicon (MPS)

`DEVICE=mps DEFAULT_DTYPE=float32 ./train_defect_model.sh` works. It is for the debug
loop, **not for the production fit** — MPS has no float64 at all, so the real fit belongs
on the cluster.

This needed a fix. `ScaleShiftMACE.forward` upcast the per-node energy decomposition to
float64 unconditionally, which is a hard error on MPS (`Cannot convert a MPS Tensor to
float64`). That is an **upstream** bug, not a `MACEDefect` one — plain `ScaleShiftMACE`
crashed at the identical line, and `--device mps` is an advertised CLI choice with its own
branch in `torch_tools.py`. Now routed through `mace.tools.torch_tools.to_high_precision`,
which upcasts where the device supports it and is a no-op on MPS. Applied to
`ScaleShiftMACE`, `MACELES`, `PolarMACE` and `MACEDefect`.

Two things to know before trusting an MPS number:

- **The §9.6 float64 softmax accumulation does not happen on MPS.** `sum_i alpha_i = 1`
  then holds to ~1e-5 rather than ~1e-12. The extensivity argument rests on the logit gap
  rather than on precision, so this should be harmless, but it is a deviation from the
  spec, not a free pass. `test_softmax_normalisation_without_the_float64_accumulation`
  exercises exactly this path on CPU so the deviation stays measured on every machine.
- **Verified against CPU**, same weights and batch, float32: `energy` and `base_energy`
  bit-identical, `forces` agreeing to 4.7e-6 relative. The `delta_energy` comparison was
  vacuous (zero-initialised heads give 0 on both), so it proves nothing.

Rough cost on the 720-frame set, 2 epochs: CPU float64 / batch 8 took 7m06s; MPS float32 /
batch 4 took 2m48s despite twice the optimiser steps. Confounded by dtype and batch size —
directional only.

Still broken on MPS, untouched: `AtomicDipolesMACE`, `AtomicDielectricMACE` and
`EnergyDipolesMACE` register `r_max` as a float64 *buffer*, so `.to("mps")` fails on those
regardless; `PolarMACE` has further float64 uses beyond its node energy; the `les`
long-range branch is untested on MPS.

### 5.3 Sharper validation, if wanted

Holding out whole `config_coord` runs — rather than random group holdout — tests
extrapolation along the configuration coordinate, which is what a line-shape calculation
actually needs.

---

## 6. Status

| Item | State |
|---|---|
| Extraction reproducible and deterministic | done |
| Dataset loads through `load_from_xyz` with correct targets | verified |
| End-to-end `run_train`, short-range | verified on the 800-frame set, exit 0 |
| End-to-end `run_train`, long-range (`les`) | verified, exit 0; model saves with `latent_ewald` and an 8-entry band-edge registry |
| E0 fit reproduces the extraction script's prediction | verified (−7.8983694830401365) |
| Preflight guards (missing data, no `les`, no CUDA) | verified to fail fast |
| Correction branch reaches the optimizer | **fixed 2026-08-08** — was 0% trained (§5) |
| Per-epoch metrics on stdout | fixed; `DefectRMSE` had no `valid_err_log` branch (§5.0) |
| Correction branch verified learning | `carrier_readouts` 0.0 → 8.1e-3, `RMSE dE` now moves |
| Gate: logit-gap stability | **blocked** — still not logged anywhere (§2.2) |
| Gate: `1/a²` vs ε∞ | **not addressable** with q = 0 data (§2.2) |
| `L_tot` holdout check (§10 step 5) | **untested, not merely unrun** — with a frozen correction branch `base_energy.detach() + delta_energy` reduced to the base energy alone |
| GPU cluster run | ready, not yet run |

### 6.1 Diagnosis: the correction branch trains into a degenerate solution

Fixing the optimizer (§5) was necessary but not sufficient. The branch now receives
gradients and still learns nothing: `RMSE dE` fluctuates around σ without trending, and
`RMSE dF` is constant to two decimals across every epoch. Measured, not inferred:

| quantity | measured | expected if working |
|---|---|---|
| attention participation ratio | 380.7 of 382 atoms (99.7%) | ~6 (the defect neighbours) |
| max α | 0.00279 (uniform = 0.00262) | ≫ 1/N |
| predicted ΔE across 4 frames | 0.0048, 0.0046, 0.0055, 0.0048 | should track targets |
| target ΔE across those frames | −0.183, +0.005, +0.258, −0.139 | — |
| predicted \|ΔF\| max | 9e-5 eV/Å | ~0.6 eV/Å |

**The attention never localises, so `ΔE_SR` is a cell average.** `ΔE_SR = Σ_c n_c Σ_i α_i^c
u_i^c` with `Σ_i α_i = 1` is a convex combination — an *average* over atoms. With α uniform
it is the mean of `u` over ~300 atoms, dominated by bulk atoms whose descriptors are nearly
identical in every frame. The output is therefore a near-constant, which is exactly what is
measured.

**Three compounding mechanisms keep it there.**

1. **At initialisation the attention gradient is exactly zero.** `zero_last_layer` sets
   `u ≡ 0`, and `∂ΔE_SR/∂ℓ_j = n_c α_j (u_j − ⟨u⟩_α)` vanishes identically when `u ≡ 0`.
   Measured `|grad| logit_readouts = 0.000000e+00`. The zeroed last layer also blocks the
   backward path through `u`, so `counter_embedding` and `defect_feature_readouts` get
   exactly zero gradient too — at step 0 **only the final layer of the u-MLP trains**.
2. **Once `u ≠ 0` the attention gradient is ~260× weaker than the readout gradient**
   (measured ratio 3.8e-3). `u` converges to the best uniform-attention solution — the cell
   mean — long before α moves. That solution has `u` nearly uniform, which drives
   `(u_j − ⟨u⟩) → 0` and kills the attention gradient again. Self-reinforcing.
3. **The regulariser that was supposed to buy localisation is seven orders too weak.**
   §3.2 of the blueprint calls the L2 on `u` "load-bearing … what makes the optimiser buy
   localisation". As implemented it is `1e-4 × torch.mean(u²)` = **6.2e-9** against a data
   term of **1.8e-1**. `mean` also makes it intensive, so a localised solution (6 large `u`)
   and a delocalised one (300 small `u`) are penalised almost identically.

**Why `dE` and `dF` behave differently** — the sharper symptom is `dF`. `ΔE` can absorb a
constant, so with uniform α it fits the *mean* of the target distribution and `RMSE dE` sits
at σ, drifting as that constant moves. A force field has no constant component to fit:
`ΔF_j = −∂ΔE_SR/∂r_j ≈ −(1/N) Σ ∂u/∂r_j` is suppressed by ~N with bulk terms cancelling, so
the prediction is four orders low and `RMSE dF` is pinned exactly at the target norm.
**A constant `RMSE dF` means there is no localised structure in the correction at all.**

**The data is not the problem.** A *linear* regression of ΔE_vert on the 15 pairwise
distances among the six defect neighbours gives **R² = 0.85**, residual 48 meV, over all 504
pairs; held-out 318/382/398-atom cells give 0.86/0.86/0.71. (Held-out 286-atom cells give
−4.05 — a distinct cell shape and the smallest subset, a limitation of the linear probe, not
of the signal.) The target is strong, local and nearly linear in the defect-neighbour
geometry, and the MACE trunk has that information.

**Fixes, in confidence order.**

1. **Delete `zero_last_layer` on the energy readouts.** Verified: with `u` randomly
   initialised at O(1), at `n = 0` `max|E_total − E_base| = 0.000e+00`, `max|Δ_energy| = 0`,
   `max|Δ_forces| = 0`. The identity is guaranteed *structurally* by the counter factor in
   `delta_sr = (counts * pooled).sum(-1)` (`defect_blocks.py:184`), **not** by the zero init.
   The zero init buys nothing and costs the whole attention gradient at step 0. Strict
   improvement — but expected to be insufficient alone, since mechanisms 2 and 3 persist.
2. **Make the localisation pressure real**: an extensive (sum) `u` penalty at a weight
   calibrated against the data term, or — more directly targeting the symptom — an entropy
   penalty on α, or a learnable inverse temperature initialised sharp.
3. **Break the α/u scale imbalance**: a separate, higher learning rate for
   `logit_readouts`, or a reparameterisation that is not a pure product of two mutually
   degenerate factors.
4. **Reconsider the softmax-over-all-atoms prior.** The correction is localised on ~6 of
   ~300 atoms, a ~50× upweight (ln 50 ≈ 3.9 in logits). An architecture that *starts*
   localised — logits biased by coordination deficit, or top-k / sparsemax — would begin in
   the right regime rather than having to search for it from a saddle.

None of this is fixed by a wider model, more epochs or more data. **Do not spend cluster
time until at least fixes 1–3 are in.**

### 6.2 Superseded

**Open question for the first real run: `RMSE dF`.** After 6 epochs of a deliberately tiny
model the delta *force* prediction is ~1e-4 eV/Å against targets of ~0.6, so `RMSE dF` sits
unchanged on the target norm. Undertraining is the benign reading and is consistent with
`RMSE dE` and the readouts both moving — but that evidence concerns the *energy* head, and
a prediction four orders of magnitude low is equally consistent with a weak or
misassembled delta-force gradient path (`base_forces` is derived as the difference of two
autograd calls, §4 of the blueprint). Treat it as unresolved. If `RMSE dF` is still pinned
after a full run at production width, that is the next thing to investigate.
