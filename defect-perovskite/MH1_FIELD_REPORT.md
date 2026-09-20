# MACE-MH-1-Field: two models in parallel — report

*Plan: "MACE-MH-1-Field: two models in parallel" (user, 2026-09-20). Units: Z* e, α = ε∞ − I (dimensionless, the released files' convention), P e/Å² (folded; also μC/cm²), E meV/atom, F meV/Å, σ meV/Å³. Status: §1–§3 complete; targets registered; Model A training (local), Model B queued on b3.*

## 1. Inventory (§1)

### 1.1 Sources

| item | source | local copy | note |
|---|---|---|---|
| `mace-field` code | github.com/mdi-group/mace-field, tag 1.0.2 (2026-03-26, commit 136e4ef) | `~/src/mace-field` (HEAD 45d5c5f, 2026-08-27; 1.0.2 is an ancestor) | the project's port in `mace/modules/extensions.py` is of HEAD |
| 1.0.2 checkpoint | release asset `MACEField-MH-0-omat-dielectric.model` | `~/src/macefield/1.0.2/` | md5 1dac2042… = the project's provider `MACEField-omat-dielectric-converged.model`: **no switch needed** |
| 1.0.2 data asset | `MP-dielectric-ferroelectric-replay-dataset.xyz` (8446 frames: 5946 dielectric + 2500 ferroelectric) | `~/src/dielectric_ferroelectric_combined.xyz` (md5 identical) | stores the polarisability as χ/4π (MgO 0.197); NOT the fine-tuning data (see 1.3) |
| paper repo | github.com/mdi-group/2025-04-mace-field (main) + release "resubmitted" (2026-05-06) | `~/src/2025-04-mace-field` with `MACE-Field-omat-pbe-mh-0.zip`, `MACE-Field-MP-Dielectrics.zip`, `MACE-Field-MP-Ferroelectrics.zip` extracted | |
| MP-Dielectric, full | `Dielectrics/MP-Dielectrics-filtered.extxyz`, 5274 frames, 5274 material ids, ≤ 160 atoms, 77 elements | | `REF_polarizability` = `REF_epsilon_static` − I to 4e-15 (ε∞ − I); ΣZ* residual per frame median 4e-5 e, max 0.28 e |
| MP-Ferroelectric, full | `Ferroelectrics/MP-Ferroelectrics.xyz`, 2500 frames (≈250 materials × 10 path frames), ≤ 288 atoms, 61 elements | | E, F, P per frame; stress weight 0 |
| OMAT-PBE replay | `Foundation/data/subselected-replay-data-mh-0-omat-pbe.xyz`, 10 000 frames (FPS subset of the `mace_mh_1` release's `replay-data-mh-1-omat-pbe.xyz`, 500 MB, 372 355 frames) | | **labels are MH-1's own predictions**: MH-1 (omat_pbe, float64) reproduces `REF_energy`/`REF_forces`/`REF_stress` to 1.5e-12 meV/atom, 2.8e-6 meV/Å, 1.5e-13 meV/Å³ — there is no DFT in this asset (see 1.5) |
| Foundation script | `scripts/Foundation/train_foundation_mh.sh` | | verbatim in 1.2 |
| validation-index file | `scripts/Foundation/MACEField-omat-dielectric_valid_indices_123.txt`, 1000 indices in [3, 9993] | | MACE writes one file per run in write mode, once per head; the committed file holds the LAST head's split = pt_head's 10 % of the 10 000-frame replay (the dielectric split would have 550 entries, the ferroelectric 250) |
| MH-1 | `~/.cache/mace/macemh1model` (float64) | | 6 heads, 512×0e+512×1o, T = 2, r_max 6, 89 elements, 6 439 878 parameters |
| mace-mh-0 (the recipe's base) | `mace_mh_1` release asset | `~/src/macefield/mace-mh-0.model` | 7 heads, 128 channels, 9 077 038 parameters |

### 1.2 Foundation script, verbatim (`train_foundation_mh.sh`)

```
torchrun --standalone --nproc-per-node="gpu" -m mace.cli.run_train \
  --name="MACEField-omat-dielectric-2" \
  --heads "{'mp-dielectric': {'train_file': '../Dielectrics/MP-Dielectrics-filtered-train.xyz', 'valid_file': '../Dielectrics/MP-Dielectrics-filtered-valid.xyz', 'test_file': '../Dielectrics/MP-Dielectrics-filtered-test.xyz'}, 'mp-ferroelectric': {'train_file': '../Ferroelectrics/MP-Ferroelectrics-train.xyz', 'valid_file': '../Ferroelectrics/MP-Ferroelectrics-valid.xyz', 'test_file': '../Ferroelectrics/MP-Ferroelectrics-test.xyz'}}" \
  --foundation_model mace-mh-0.model --foundation_head omat_pbe \
  --pt_train_file data/subselected-replay-data-mh-0-omat-pbe.xyz --multiheads_finetuning True --pseudolabel_replay=True \
  --E0s "foundation" --model="MACEField" --loss="universal_field" --error_table="PerAtomFieldRMSE" \
  --energy_weight=1.0 --forces_weight=10.0 --polarization_weight=10.0 --becs_weight=100.0 --polarizability_weight=10.0 \
  --compute_forces=True --compute_stress=True --compute_polarization=True --polarization_loss_mode=normalized_metric --polarization_loss_scale=1.0 \
  --compute_becs=True --compute_polarizability=True --num_workers 8 --distributed --enable_cueq=True --device="cuda" --plot=True \
  --batch_size 1 --valid_batch_size 1 --max_num_epochs 300 --patience 50 --default_dtype "float32" --restart_latest
```

Defaults the script relies on (mace-field 1.0.2 `arg_parser.py` / `run_train.py`): stress_weight 1.0; huber_delta 0.01 (P: normalized_metric, cell-aware metric, no Huber); lr forced to 1e-4 with EMA on, decay 0.99999 (multihead fine-tuning override; `--force_mh_ft_lr` not set); weight_decay 5e-7; amsgrad True; scheduler ReduceLROnPlateau, factor 0.8, patience 50; clip_grad 10.0; E0s = the foundation's `omat_pbe` row for every head; keys REF_energy / REF_forces / REF_stress / REF_polarization / REF_becs / REF_polarizability / REF_electric_field (missing label ⇒ per-config weight 0); field injection = MACEField defaults (one injection after layer 0 for T = 2: FullyConnectedTensorProduct hidden × 1o → hidden, Linear hidden → hidden, residual subtracted); polarisability output = (dP/dE)/(V ε0) = ε∞ − I, compared to `REF_polarizability` without conversion.

Published training log (`MACEField-omat-dielectric_run-123.log`, 2025-12-24, MACE 0.3.15, 4 processes): heads mp-dielectric (`data/dielectric_corrected_asr.xyz`, 5501 frames → 4951/550 random 10 % split, seed 123), mp-ferroelectric (`data/ferroelectric.xyz`, 2500 → 2250/250), pt_head (`data/replay-data-mh-1-omat-pbe.xyz`, 372 355 → 335 120/37 235; fine-tuning heads upsampled ×4); lr 1e-4, EMA 0.99999, weight decay 5e-7, batch 1, clip 10, cueq; 360 epochs; final validation (mp-dielectric head) Z* 0.217, α 1.010, P 0.0033 e/Å².

### 1.3 Discrepancies (recorded)

- **Which data trained 1.0.2.** The 1.0.2 checkpoint sits 1.0 % (relative Frobenius, backbone) from the paper repo's `run-123` checkpoint and 4.1 % from `mace-mh-0`; `run-123` sits 4.5 % from `mace-mh-0`. 1.0.2 is therefore a checkpoint of the run-123 lineage (full 372 355-frame replay, 5501-frame dielectric file, seed-123 random 10 % splits), not of the run in `train_foundation_mh.sh` (10 000-frame replay, from-scratch split files; its training record `MACEField-omat-dielectric-2_run-123_train.txt` stops at epoch 19 with validation Z* 0.3955). The authors (relayed by the user) confirm: full dielectric and ferroelectric files, MACE's random validation split.
- **Dielectric file.** The log's `data/dielectric_corrected_asr.xyz` has 5501 frames; the released full file has 5274 (`filter_source_index` ≤ 5280), so 1.0.2's own dielectric validation frames cannot be regenerated. The ferroelectric file matches (2500 frames) and its seed-123 split is regenerated exactly by MACE's rule.
- **Table I vs script.** Paper Table S1 ("Fine-tuned": E 1 / F 100 / σ 1 / P 100 / Z* 100 / α 100, lr 1e-4, 300 epochs, patience 50, batch 1, seed 23; EMA 0.995, weight decay 1e-8, double precision) vs the script/log (F 10 / P 10 / α 10, seed 123, EMA 0.99999, weight decay 5e-7, float32). This plan follows Table I (user's §4) and float64 (user's §3).
- **Splits.** The from-scratch split files in the Dielectrics/Ferroelectrics zips are material-disjoint for the dielectric set but frame-level for the ferroelectric set (every valid/test material also in train); they are not the foundation's split. The recipe's own split (seed-123 random 10 %) is likewise frame-level for the ferroelectric file (149 of 149 validation materials also in train): the P target is an interpolation test along distortion paths.
- **Replay labels.** The replay asset carries MH-1 pseudolabels, not DFT (1.1). §6's "E/F/σ within 5 % of MH-1's against DFT" is not evaluable as written: MH-1's error against these labels is 0. See 1.5.
- **Paper's committed metrics** (`analysis_outputs/foundation_workflow/parity/tables/foundation_metrics.csv`, pt_head, float32, filtered valid+test): Z* diag RMSE 0.655 e / off-diag 0.324 e; α diag 8.18 / off-diag 0.895; P (unfolded, μC/cm²) 14.85; replay E 65.4 meV/atom, F 192 meV/Å. They refer to the repo checkpoint and to the from-scratch valid+test sets.

### 1.4 Data used here (the recipe, as the authors describe it)

| pool | file | split rule | train | valid | head |
|---|---|---|---|---|---|
| MP-Dielectric | `MP-Dielectrics-filtered.extxyz` (5274) | MACE `random_train_valid_split`, seed 123, 10 % | 4747 | 527 (527 materials, none in train) | mp-dielectric (Z*, α) |
| MP-Ferroelectric | `MP-Ferroelectrics.xyz` (2500) | same | 2250 | 250 (149 materials, all in train) | mp-ferroelectric (P, E, F) |
| OMAT-PBE replay (Model B) | `subselected-replay-data-mh-0-omat-pbe.xyz` (10 000) | the published 1000-index file = hold-out; the other 9000 pseudolabelled by MH-1 (`omat_pbe`, float64) | 9000 | 1000 | omat_pbe (E, F, σ) |

No test set exists under this recipe; the from-scratch test files are reported as an extra (they overlap the training pools).

### 1.5 Pre-registered targets and lines

- **Target (§2):** Model A and Model B validation Z*, P (folded) and α below the 1.0.2 model's on the seed-123 validation sets above (mp-dielectric / mp-ferroelectric heads, float64; numbers registered in the tracker when the reference evaluation lands, before any A/B validation is read).
- **Model B E/F/σ (§6):** evaluated against the replay hold-out's labels, which are MH-1's. Provisional line (to be confirmed by the user): B's RMSE against MH-1 on the hold-out ≤ 5 % (E, F) / 10 % (σ) of MH-1's own validation error on OMAT-PBE DFT, which is not published in the release notes or the preprint text; until it is, the drift is reported as is and the criterion is marked not evaluable.
- **Scheduler:** ReduceLROnPlateau on validation Z*, factor 0.8, patience 5 (the script's patience 50 never fires under early-stop 20). Early stop: patience 20 on validation Z*; cap 150 epochs.

## 2. Reference (§2)

1.0.2 checkpoint, float64, seed-123 validation sets of the full released files (`ref_valid_1.0.2_f64.json`); all-component RMSE, P folded with the loss's `fold_polarization`; diag/off-diag with the paper's `scalar_metrics`.

| set | head | Z* (e) | α (ε∞ − I) | P folded (e/Å²) | P (μC/cm²) |
|---|---|---|---|---|---|
| dielectric valid (527) | mp-dielectric | **0.2524** (diag 0.307, off 0.220) | **2.489** (diag 4.29, off 0.27) | | |
| dielectric valid | pt_head | 0.2863 | 2.660 | | |
| ferroelectric valid (250) | mp-ferroelectric | | | **0.00182** | 2.92 |
| ferroelectric valid | pt_head | | | 0.00227 | 3.64 |
| dielectric train (4747) | mp-dielectric | 0.2630 (diag 0.381, off 0.176) | 2.704 (diag 4.65, off 0.38) | | |
| ferroelectric train (2250) | mp-ferroelectric | | | TRAIN_P | |
| dielectric test, from-scratch split (527) | mp-dielectric | 0.4685 (diag 0.713, off 0.274) | 6.350 (diag 10.96, off 0.67) | | |
| ferroelectric test, from-scratch split (253) | mp-ferroelectric | | | 0.00169 | 2.71 |

**Targets (registered in the tracker before any Model A/B validation was read): validation Z* < 0.2524, α < 2.489, P < 0.00182, dedicated heads, same sets, float64.** The train figures explain the earlier 0.21: 1.0.2's validation on the released frames it trained on is not lower than on the regenerated validation split (0.263 vs 0.252), because the released file is not the run's own file (§1.3) and its errors are outlier-dominated (α diag RMSE 4.3 against a median absolute residual well below 1). The from-scratch test files are worse for 1.0.2 (Z* 0.47, α 6.35) although 1.0.2 trained on most of those frames: that split concentrates the large-|Z*| outliers.

Paper's own workflow (`plot_foundation.py`, pt_head, float32 + cueq, `ref/parity_pt_head/tables/foundation_metrics.csv`), 1.0.2 on the from-scratch valid+test: Z* diag 0.659 / off 0.282; α diag 8.31 / off 0.776; P unfolded 15.1 μC/cm²; replay (10 000-frame subset, MH-1 labels) E 67.3 meV/atom, F 194 meV/Å. Against the repo checkpoint's committed table (0.655 / 0.324; 8.18 / 0.895; 14.85; 65.4 / 192): same model family, slightly different checkpoint (§1.3).

Replay hold-out (1000 frames, MH-1 labels): 1.0.2 (pt_head, 983 frames in its table) E 59.3 meV/atom, F 192.7 meV/Å, σ 13.3 meV/Å³; MH-1 by construction 0. 1.0.2's energies/forces on the ferroelectric frames against the file's labels: 202 meV/atom, 622 meV/Å (validation) — those labels were never trained on by the published run, which is why the ferroelectric head here is trained on P alone.

Polarisability convention (§2 check): in the released full file `REF_polarizability` = `REF_epsilon_static` − I to 4e-15 on every validation frame (MgO mp-1265 2.21 → ε∞ 3.21; Si mp-149 12.0 → 13.0; diamond mp-66 4.83 → 5.83), which is exactly MACE-Field's output (dP/dE)/(V ε0); no conversion. ΣZ* residual per frame: median 4e-5 e, max 0.28 e (valid). The 1.0.2 release asset xyz instead stores χ/4π (MgO 0.197) and is not used.
## 3. Construction and checks (§3)

- **Build** (`mh1f2_build.py`): `MACEField(**extract_config_mace_model(MH-1))` with the head list extended by `mp-dielectric` and `mp-ferroelectric` (indices 6, 7); MH-1's six heads keep their indices and tensors; the new heads are copies of `omat_pbe` (E0 row, scale/shift, linear-readout column, 16-channel block of the non-linear readout; the widened `linear_2` is rescaled by √(8/6) for e3nn's 1/√fan_in path normalisation). Field layers: MACEField defaults = the Foundation script's (one injection after layer 0: FullyConnectedTensorProduct 512×0e+512×1o ⊗ 1o → hidden, Linear hidden → hidden, residual subtracted), `field_linear` zero, `field_feats` e3nn default (seed 0). float64. 7 506 310 parameters (MH-1 6 439 878 + field 1 048 576 + 17 856 new-head readout entries). Build check on 12 frames: heads 0–5 vs MH-1 max 7e-12 meV/atom, heads 6/7 vs omat_pbe 3e-14, field residual at E = 0 exactly 0.
- **Exactness at E = 0** (fixed 2000-structure set: 1400 MH-1 replay + 600 CsPbCl3 defect frames; float64; `exact_init_f64.json`): all six MH-1 heads max |ΔE| ≤ 9.1e-10 meV/atom (spice_wB97M; others ≤ 1.2e-10), max |ΔF| ≤ 7.2e-11 meV/Å, max |Δσ| ≤ 2.0e-12 meV/Å³; field residual 0.0. **Pass** (plan: ≤ 1e-4 meV/atom, ≤ 1e-3 meV/Å).
- **cueq gradient check** (`cueq_check.json`, one 6-frame / 46-atom batch, float64, field_linear perturbed to 1e-2 so the field gradient is non-zero): loss relative difference 2.6e-10; outputs max |Δ| energy 8.7e-9 eV, P 4e-13, Z* 2.8e-10, α 4.2e-11; loss gradient w.r.t. the field maps: relative-norm difference 3.2e-8 (field_feats) and 3.1e-8 (field_linear), max |Δ| 1.4e-11 / 8.8e-10. **Pass at 1e-6** → cueq used. Note: MACE's converter re-instantiates the field layers with a fresh initialisation; the trainer copies their state across (verified: identical losses and validation numbers to the e3nn trainer on the debug run). Speed: 0.74 vs 1.59 s per step on that batch (b3), 0.44 vs 0.75 s/step on the debug epoch (local).
- **Probes** (`probe_b3.json`, b3 Quadro RTX 6000, float64, e3nn): the 160-atom dielectric frame with Z*+α is out of memory at 24 GB alone (A and B); a 90-atom mixed batch with α: 2.8 s / 11.3 GB (A), 3.1 s / 15.4 GB (B); the 288-atom ferroelectric frame with P + F: 1.9 s / 7.4 GB (A), 1.6 s / 9.6 GB (B). Consequences (recorded): each random batch is split into a dielectric sub-pass (nested derivatives; budget 64 atoms local / 96 atoms b3) and an E/F/P sub-pass, backward per pass; dielectric frames above 96 atoms are excluded (5 of 4747 training frames, 0 of 527 validation).
- **Labels used** (mirroring the published run's log: mp-ferroelectric head "energy: 0, forces: 0, polarization: 2500"; mp-dielectric "becs, polarizability", released file's E/F/S weights 0): dielectric frames → Z*, α; ferroelectric frames → P only; replay (Model B) → E, F, σ. Flag `--ferro_efs 1` would add the ferroelectric E and F.
- **Trainer** (`mh1f2_train.py`): random batches across every pool every batch under an atom budget (never size-sorted); universal_field loss (Huber δ 0.01; P normalized_metric, scale 1); weights E 1 / F 100 / σ 1 / P 100 / Z* 100 / α 100 (A: E/F/σ 0); Adam + AMSGrad, EMA 0.995, weight decay 1e-8, clip 10; ReduceLROnPlateau on validation Z* (0.8, patience 5); early stop patience 20 on validation Z*; cap 150 epochs; Model A gradients masked to the field maps and the two new heads' readout slices (masks carried through the cueq converter; weight decay applied by hand to masked-in entries; frozen entries asserted bit-identical at the end — debug run: YES); Model B all weights, DDP by manual gradient all-reduce (torchrun, rank-disjoint shards, validation on rank 0, metric broadcast).

## 4. Training (§4–§6) — in progress
## 5. Deliverables (§7) — pending
