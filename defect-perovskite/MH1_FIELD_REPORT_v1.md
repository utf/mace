# MACE-MH-1-Field: report

*2026-09-20. Plan: MACE-MH-1-Field, energy-exact field upgrade of MACE-MH-1. Units: P e/Å², Z* e, α per unit ε0 (χ); E meV/atom; F meV/Å.*

## 1. Summary

The field-aware MH-1 was built as specified: MACE-Field's field-coupling layers on the MH-1 backbone, every MH-1 tensor frozen, only the coupling trained on the dielectric data. The zero-field model is MH-1 to float32 round-off and the coupling residual at E = 0 is identically zero, before and after training. **The response targets were not met.** The best exact (frozen) model reaches a validation Born-charge RMSE of 0.45 e against a target of 0.23, polarisability 1.36 against 1.21 and polarisation 0.0030 e/Å² against 0.0020. A direct linear solve of the same problem reaches 0.37 e and confirms the limit is the frozen features' reach, not the optimiser or the coupling's parameter count. The one configuration that goes further, all weights free under an MH-1 replay, reaches 0.37 e with the backbone 2 meV/atom and 10 meV/Å from MH-1, ten times the plan's tolerance for that variant. All fallbacks the plan allows were the configuration from the start; the plan's own stopping line applies (§6.3): the shortfall is reported, and any further gain requires a different model.

## 2. Build (§1)

- MH-1: 6,439,878 parameters, 6 heads (matpes_r2scan, mp_pbe_refit_add, spice_wB97M, oc20_usemppbe, omol, omat_pbe), 89 elements, hidden 512x0e+512x1o, T = 2.
- MH-1-Field: `MACEField(**extract_config_mace_model(MH-1))`, MH-1 tensors copied, 7,488,454 parameters in total; trainable field coupling 1,048,576 (field_feats: FullyConnectedTensorProduct hidden × 1o → hidden, 524,288; field_linear: Linear hidden → hidden, 524,288); 1 injection layer (after every layer t < T, the MH-0 configuration). float32.
- Head for the response: mp_pbe_refit_add (MP PBE/PBE+U). The field layers live in the shared backbone; every head stays exact at zero field.
- Initialisation (amended from the plan): the coupling is the composition of two linear maps, so P and Z* are bilinear in the two weight sets and the gradient of both vanishes at the all-zero point. field_feats keeps its e3nn default random init, field_linear starts at zero (initial response exactly zero, E = 0 residual zero for any weights).

## 3. Exactness at E = 0 (§2), before training

2000 structures (1400 MH-1 replay + 600 CsPbCl3 defect frames), all six heads, float32:

| head | max ΔE meV/atom | max ΔF meV/Å | max Δσ meV/Å³ |
|---|---|---|---|
| matpes_r2scan | 1.5e-02 | 2.0e-02 | 1.1e-03 |
| mp_pbe_refit_add | 1.6e-03 | 3.4e-02 | 6.6e-04 |
| spice_wB97M | 3.3e-01 | 1.7e-02 | 6.7e-04 |
| oc20_usemppbe | 2.1e-03 | 2.6e-02 | 6.0e-04 |
| omol | 9.5e+00 | 2.8e-02 | 8.5e-04 |
| omat_pbe | 1.9e-03 | 3.6e-02 | 2.1e-03 |

Field residual at E = 0: 0.0e+00 (exactly zero). float64 control on 300 structures: max ΔE 1.3e-5 … 2.0e-5 meV/atom, max ΔF 0.7e-4 … 2.2e-4 meV/Å on every head; a plain ScaleShiftMACE re-instantiated from MH-1's config equals the pickled MH-1 to 0.0 meV/atom and 5e-12 meV/Å, so the float64 residual is the port's forward arithmetic order, not the re-instantiation. The float32 differences are round-off of the absolute energies (omol and spice totals are 1e4–1e5 eV; 9.5 meV/atom is relative 1e-7); the plan's figures (≤ 1e-5 meV/atom, ≤ 1e-4 meV/Å) are met in float64 and unattainable in float32 for any two evaluations of MH-1 in a different arithmetic order. Deviation 5–6 orders below the base's own force error: same performance.

## 4. Data, split, loss (§3)

- Dataset: the combined MACE-Field set, 8446 frames = 5946 MP-Dielectric (Z*, α) + 2500 MP-Ferroelectric (P), 79 elements; 40 ferroelectric frames above 120 atoms excluded (α's nested derivatives on the 16 GB card).
- Split (recorded deviation): the MH-0 split indices are not published (paper: material-level splits, details in SI); the repository README recipe uses `--valid_fraction 0.2`. Here ferroelectric frames are split by material (43 of 213 held out with all interpolates), dielectric frames at random (1189 of 5946), seed 123: 6727 train / 1679 validation.
- **Units finding.** The dataset stores the polarisability as χ/4π (MgO 0.197 → ε∞ 3.5, Si 0.935 → 12.7, diamond 0.446 → 6.6), while MACE-Field, fork and port alike, predicts χ = (dP/dE)/(V ε0). Labels are converted χ = 4π × REF_polarizability; the MH-0 reference on this split then reads P 0.00177, Z* 0.2101, α 1.101. The plan's targets (1.1 × reference): P 0.00195, Z* 0.231, α 1.21.
- Loss: universal_field with E/F/S weights 0 (the loss skips a term whose global weight is zero — a code fix), Huber on Z* and α (δ 0.01), folded P (δ 0.05); response weights P 1 / Z* 100 / α 100 (README) at first, P 1000 by the user's choice afterwards; missing labels masked by per-config weights.
- Batches under an atom budget (48–80 atoms, ≤ 6–8 frames), size-sorted at first, then mixed (random) so both subsets share every step: size-sorting had segregated the polarisation-only ferroelectric frames (median 28 atoms) from the dielectric frames (median 14), so each step trained one objective and the clipping at 1.0 acted on P alone.

## 5. Runs and validation curves

| run | node | weights P / Z* / α | lr | epochs | best Z* | P | α | replay drift E / F | s/epoch, GB |
|---|---|---|---|---|---|---|---|---|---|
| run1_p1 | local | 1 / 100 / 100 | 1e-3 | 2 | 0.435 | 0.00353 | 1.48 | — | 1748, 10.0 |
| run1_p1000_lr1e-2 | local | 1000 / 100 / 100 | 1e-2 | 2 | 0.598 | 0.00319 | 2.19 | — | 1764, 10.0 |
| run1 (mixed) | local | 1000 / 100 / 100 | 3e-3 → 3.8e-4 plateau | 17 | 0.564 | 0.00293 | 1.44 | — | 1900, 10.0 |
| run_p100 (mixed) | b3 | 100 / 100 / 100 | 3e-3 | 16 | **0.453** | 0.00304 | 1.37 | — | 1518, 10.0 |
| run_b128 (bottleneck 128, 294,912 weights) | local | 1000 / 100 / 100 | 3e-3 | 3 | 0.566 | 0.00309 | 1.62 | — | 2337, 9.9 |
| run_cg (CG linear solve, second map) | b3 | Z* 1 / P 10 | — | 24 iter | **0.368** | 0.00339 | 9.0 (outside objective) | — | 860 per iter |
| run_full_w10 (all weights, replay 10/10/1) | b3 | 1000 / 100 / 100 | 1e-4 | 6 | 0.312 | 0.00332 | 1.20 | 23.6 / 167 | 3546, 14.0 |
| run_full_w1000 (replay 1000/1000/100) | b3 | 1000 / 100 / 100 | 1e-4 | 1 | 0.471 | 0.00339 | 1.78 | 12.6 / 55.7 | 4129, 14.0 |
| run_full (replay 1000/1000/100, field 3e-3 / backbone 1e-5) | b3 | 1000 / 100 / 100 | diff. | 8 | **0.371** | 0.00296 | 1.34 | 1.96 / 10.7 | 3503, 14.0 |
| MH-0 MACE-Field reference | | | | | 0.210 | 0.00177 | 1.10 | 61.6 / 211.8 (pt_head), 85.8 / 244.7 (mp-dielectric) | |
| target (1.1 × reference) | | | | | 0.231 | 0.00195 | 1.21 | | |

Replay drift is the RMSE against MH-1's own labels on a 10 % hold-out of the MH-1 replay set (tolerance for the unfrozen variant, plan §6.3: 0.2 meV/atom, 1.0 meV/Å).

**Benchmark for the drift column.** The MH-0 MACE-Field itself, evaluated on the same hold-out (934 of the 999 frames lie within its 79-element table), sits 61.6 meV/atom and 211.8 meV/Å from MH-1's labels with its pt_head (85.8 / 244.7 with its mp-dielectric head; energies after a per-element offset fit in the per-atom metric, forces direct; `replay_bench_mh0.json`). MH-0 and MH-1 are different models trained on different data, so this is the distance between the two potentials on MH-1's own distribution, not an error of MH-0 against DFT. Against it the unfrozen fine-tune's drift (1.96 / 10.7) is 3 % of the energy gap and 5 % of the force gap: the differential-lr run keeps MH-1's energetics far more closely than swapping to the MH-0 potential would, while its Born charges (0.37) remain well short of MH-0's (0.21). The tolerance in §6.3 (0.2 / 1.0) is a fidelity requirement to MH-1 itself, and is what the run misses.

## 6. What limited the fit

- **Derivatives and gradients are correct.** On an 8-atom dielectric frame in float64: Z* against central differences of Ω·P over positions agrees to 5.6e-8 e; α against central differences of P over the field to 5e-4 on values of order 2; α symmetric and ΣZ* = 0 to 1e-15; both field maps receive gradient; no frozen tensor does; the batch head is mp_pbe_refit_add.
- **The frozen fit is a convex problem** (P and Z* exactly linear in the second map at E = 0, verified: |Z*| scales linearly with the map over four decades), and Adam was leaving something on the table: the conjugate-gradient solve of the normal equations reaches Z* 0.368 where Adam plateaued at 0.45–0.58. But its gain fell to 0.01 per three iterations by iteration 24, converging to a floor near 0.35.
- **The parameter count is not the limit.** The hypothesis that 1,048,576 weights against ≈ 735,000 scalar labels made the problem underdetermined predicted that a 128-channel bottleneck (294,912 weights) would generalise better; it plateaued at the same 0.57 under Adam. Together with the CG floor this puts the limit in the representational reach of MH-1's frozen first-layer features through a single linear injection: the response the model can express is confined to the directions in which MH-1's fixed features and readouts respond to a feature perturbation.
- **α** passes through the frozen readout curvature (second order). It reached 1.34–1.37 under Adam (target 1.21), and was the easier target in every run.
- **Unfreezing** shows what the backbone contributes: with all weights free the Born-charge error drops to 0.31 in six epochs, but at the price of MH-1's energies and forces (24 meV/atom, 167 meV/Å drift at uniform lr 1e-4). Raising the replay weights 100× halved the drift; giving the backbone its own rate (1e-5, field maps 3e-3) cut it to 2 meV/atom / 10 meV/Å, ten times the tolerance, with Z* 0.37 — no better than the linear solve. The response gain and the backbone drift are the same movement.

## 7. Targets (§5)

| check | target | result |
|---|---|---|
| exactness after training | identical to before | **pass** — same 2000 structures, all six heads: float32 mp_pbe_refit_add 2.2e-3 meV/atom, 2.3e-2 meV/Å (omol 8.7 meV/atom, the round-off case); float64 control 1.3e-5 … 2.0e-5 meV/atom, 0.7e-4 … 2.2e-4 meV/Å; field residual at E = 0 exactly 0 |
| validation Z* / P / α | ≤ 0.231 / 0.00195 / 1.21 | best exact model 0.453 / 0.00304 / 1.369 — **miss**; linear solve 0.368 / 0.00339 / 8.97 — **miss**; unfrozen 0.371 / 0.00296 / 1.336 (backbone outside tolerance) — **miss** |
| CsPbCl3 relaxed cell, Z* Frobenius Cl / Cs / Pb and ε∞ | within 10 % of 3.52 / 2.24 / 6.21; 3.55 | Cl 3.42 (−3 %), **Cs 1.58 (−30 %, miss)**, Pb 6.39 (+3 %); ε∞ 3.57 (+1 %), anisotropy 1.4 %; ΣZ* diag = 0 |
| identities on 100 validation structures | ΣZ* = 0, α symmetric to 1e-10 | **pass**: max |ΣZ*| 1.4e-14 e, max α asymmetry 3.7e-15 (100 validation frames ≤ 30 atoms, float64) |

## 8. Fallbacks (§6)

§6.1 (inject at every layer t < T) and §6.2 (full hidden irreps in the tensor product) are the configuration for T = 2 and this class from the start, so neither could be applied. §6.3 applies: the shortfall is reported with the achieved numbers; the unfrozen variant was run as the user's request, with the registered replay and tolerance, and does not meet the tolerance.

## 9. Deliverables (§7)

- `~/runs/mh1field/MACE-MH-1-Field.model`: the best exact model (run_p100 best epoch, EMA weights; all six heads; response trained on mp_pbe_refit_add; `field_config` and `training_record` in the checkpoint). Also `run_cg/MACE-MH-1-Field-cg.model` (linear solve, Z* 0.368, α unusable) and `run_full/MACE-MH-1-Field.model` (unfrozen, not exact).
- Cost per step: frozen 0.50–0.73 s at 10 GB (48–80-atom batches); unfrozen 0.63–0.75 s at 14 GB; CG 860 s per iteration (one training pass).
- For this project: base v2 + E_PP matching with the new model as provider (RMS E_PP force on the neutral validation frames at r_split 4.5): **2.40 meV/Å, above τ = 1.7** (per-frame maximum 4.85; per-species means ≤ 0.1). The project's base v7 (trained on the MH-0 provider's labels) gives 1.49 meV/Å on the same frames (PASS); the excess follows from the weaker Born charges. P0 locality on the 2×2×2 CsPbCl3 cell with the new provider (`cspbcl3_check.json`): r90 of the dipole response 6.75 Å over species (MH-0 provider: 4.5 Å r_split), partition ambiguity RMS(p − p′)/RMS(p) 0.40 (MH-0: 1.11), self share Cl 0.26 / Cs 0.50 / Pb 0.15. The form-factor and switched Born-force diagnostics were not rerun: the delivered response is below the MH-0 model's, so the MH-0 provider stays the project's provider.

## 10. Records

Tracker: `DSCC_V5_IMPLEMENTATION_SPEC.md` (MACE-MH-1-Field section). Scripts: `mh1_field_build.py`, `mh1_field_exact.py`, `mh1_field_train.py` (frozen, bottleneck, unfrozen/replay, differential lr, mixed batches), `mh1_field_cg.py`, `mh1_field_replay_bench.py` (MH-0 benchmark). Checks on the delivered model: `exact_after*.json`, `identities.out`, `cspbcl3_check.json` (b3), `epp_match_mh1field.json`, `replay_bench_mh0.json`. Run records under `~/runs/mh1field/` (local) and `~/runs/mh1field/b3/` (copied from b3).
