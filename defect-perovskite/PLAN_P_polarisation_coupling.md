# Plan P — Polarisation coupling via MACE-Field

Replace the Mulliken host coupling with a coupling to MACE-Field's zero-field polarisation, add the matching long-range dipole term to the neutral PES, and consolidate MACE-Field as the base. Four phases, each with one decision. No new gates beyond those listed; standing tests (neutral null, FD forces, Σδq = Q, gauge, invariances) run as unit tests, not experiments.

## 0. Definitions (implement once, reuse everywhere)

**Per-atom field input.** In MACE-Field, replace the broadcast global field with a per-node tensor `E_node [N,3]` in every field tensor product. With `E_node[a] = E` for all `a`, the model must reproduce the trained global-field `F`, `P`, `Z*`, `alpha` bit-for-bit (test).

**Zero-field dipoles.** `p[a] = -dF/dE_node[a]` at `E_node = 0`, taken from the same backward as the forces (`create_graph=True` when forces of the new terms are needed). Test: `sum_a p[a] = Omega * P` from the global-field derivative.

**Carrier field.** `E_c(R_a) = -grad_a sum_c dq[c] * Gamma_LR(R_a - R_c) / eps_inf`, using the existing smoothed periodic kernel and its analytic gradient; `dq` is the head's carrier density (unchanged).

**Carrier–polarisation coupling.** `E_cP = - sum_a p[a] . E_c(R_a)`. Replaces `E_host`. No learned scale.

**Host–host dipole term.** `E_PP = 1/2 sum_{a != b} p[a]^T T_LR(a,b) p[b]`, with `T_LR` the periodic dipole–dipole Ewald kernel with Gaussian smoothing width `r_split`, images included, divided by `eps_inf`, `a = b` direct self-term excluded. Belongs to the neutral model.

**Forces.** One nested-autograd pass through MACE-Field with cotangent `(T_LR p - E_c)` supplies `dp/dR` contributions for both terms; the kernel Hessian supplies `dE_c/dR_a` and `dT_LR/dR`; the existing Fréchet contraction supplies `d(dq)/dR` (one contraction, the `q0` one is deleted). Never assemble an energy from `Z*(R) u`; never evaluate `F` at the screened carrier field.

**eps_inf variants.** (R) registered scalar 4.0; (M) polycrystalline mean of MACE-Field's `eps_inf` tensor on the relaxed pristine cell, computed from `alpha` of the base in use. Same code path, one scalar swapped in `E_M`, `E_c` and `T_LR`. Tensor kernel only if the anisotropy of (M) exceeds 10 %.

**Unchanged.** `H0`, fillings, `dF_band`, `E_M`, `C_Q`, loss strata, `w_E = 0.05`, gap regulariser, six-seed paired protocol with `tau = 1.7 meV/Å` per component.

## Phase 0 — Diagnostics on the universal MACE-Field checkpoint (no training)

Run directory `~/runs/dscc/p0/`.

**0.1 Locality (decides `r_split`).** Relaxed pristine cell tiled to ≥ 2×2×2. For one atom of each species, set `E_node = 0.01 V/Å` on that atom only, evaluate `dp[b]` and `dF[b]` for all `b`. Report the radius enclosing 90 % of `sum_b |dp[b]|` per species. Set `r_split = max(3.0 Å, 1.2 × that radius)` for both `E_cP` and `E_PP`. Also report the per-atom-energy partition `p'[a] = -dF_a/dE` on the same cell and the RMS of `p - p'` relative to RMS `p`: this is the partition-ambiguity number, recorded, not gated.

**0.2 Response on the pristine cell.** `Z*` tensors, `alpha`, `eps_inf` (tensor and polycrystalline) from the checkpoint as delivered. Record `Z*` against tabulated PBE Born tensors for orthorhombic CsPbCl3 (validation of the pretrained response, not an input). Record variant (M) and its anisotropy.

**0.3 Mechanism test with a frozen provider.** Model: base v2 (PES) + `E_PP` + W6's carrier terms with `E_host` replaced by `E_cP`, `p` from the MACE-Field checkpoint, no retraining. Variant (R) only.

- Matching check: forces of `E_PP` on the neutral and charged training frames with the chosen `r_split`; report the RMS per component. If above 3 meV/Å, raise `r_split` until it is below, and record the value.
- Relaxation ladder, route 1: cells tiled from the base-relaxed pristine cell (639, 959, 2159, 2879 atoms), relaxed 8 Å core embedded, far field relaxed with the full model, `dE_relax(L) = E_relaxed - E_frozen`. Fit `dE_relax(L) = dE_relax(inf) + (A/L)(1/eps_inf - 1/eps0_ladder)` with `A = alpha(h) Q^2 C / 2`.
- `eps0_LST = eps_inf + (4 pi / Omega) sum_m Z~_m Z~_m^T / omega_m^2` from base v2's Γ-point Hessian and MACE-Field's `Z*` (mode effective charges as in the MACE-Field paper), polycrystalline mean.
- Run the same ladder with W6 as shipped for contrast.

**Decision 0.** Mechanism confirmed if `eps0_ladder` is within 30 % of `eps0_LST` and the W6-as-shipped ladder is not. Proceed to Phase 1. If not confirmed: isolate with a frozen-phonon test at the smallest nonzero `q` of the 3×3×3 cell for the strongest IR mode (LO–TO splitting must appear with `E_PP` and match the nonanalytic value from `Z*` and `eps_inf` to 20 %); if the splitting appears but the ladder fails, the deficit is in base v2's restoring forces and Phase 1 proceeds anyway; if the splitting does not appear, stop and report.

Deliverable: `P0_REPORT.md` with the locality radii, `r_split`, `Z*`/`eps_inf` table, matching RMS, both ladders and the two `eps0` values.

## Phase 1 — Base v5: MACE-Field fine-tuned on the neutral defect data

Run directory `~/runs/dscc/p1/`.

- Start from the universal MACE-Field checkpoint. Training set: neutral pristine and vacancy frames (79 and 159 atoms, energies/forces/stress as available), the replay set (energies/forces/stress, response losses masked), and the bulk response labels (`P`, `Z*`, `alpha`) with `lambda_P, lambda_Z, lambda_alpha` at the values used in the original MACE-Field training.
- `E_PP` with the Phase 0 `r_split` is included in the forward during fine-tuning, so the local part is matched to it; no target surgery. Variant (R) for `eps_inf` during training.
- Same hyperparameters as the base v2 fine-tune (learning rate, epochs, replay ratio) unless the response loss fails to hold, in which case halve the learning rate once.

**Gates (three).**
1. Neutral validation forces ≤ 10.2 + 1.0 meV/Å per component; energies ≤ 1.5 meV/atom.
2. Pristine-cell `Z*` (per species, Frobenius) and polycrystalline `eps_inf` within 10 % of the checkpoint's values from 0.2.
3. Relative robustness: base-alone force error on the charged frames, by shell, not worse than base v2 (33 meV/Å total, 75 in 2–4 Å) by more than 3 meV/Å in any shell.

Deliverable: `base_v5.model`, `P1_REPORT.md` with the three tables and the (M) value of `eps_inf` from v5.

## Phase 2 — Head W7 on base v5

Run directory `~/runs/dscc/p2/`.

- Port the head's readouts to v5's first-block features (invariants and `l = 1`); same bounds, same pristine centring recomputed on v5.
- Two arms, six seeds each, identical folds and schedule:
  - **A:** W6 as shipped (Mulliken `E_host`) on v5. Control for the base change.
  - **B:** `E_cP` + `E_PP` on v5, variant (R).
- **B(M):** arm B with variant (M), three seeds. Forces are expected within `tau` of B; this arm exists for Phase 3.
- Training cost: nested autograd through v5 once per step; cache nothing across steps. Report s/epoch.

**Gates (two).**
1. Paired TOST B vs A on charged-frame forces, total and 4–12 Å shell. B adopted on superiority in either; on equivalence B is still adopted (it removes the Mulliken pattern and the `q0` contraction, and it is the term Phase 3 needs).
2. Energies: `C_Q` profiled at 79 atoms transfers to 159 within 10 meV; 79-atom energy ≤ 0.6 meV/atom.

Standing tests must pass on the new composition (neutral null bit-exact, Σδq = Q, FD forces including the nested-autograd terms, six strains, gauge, invariances, analytic-vs-autograd force agreement).

Deliverable: 15 checkpoints with calibration records, `P2_REPORT.md` with the paired table, shell table, energy table, and cost per frame at 79 and 159 atoms (eager; report memory).

## Phase 3 — Large cells and dynamics

Run directory `~/runs/dscc/p3/`. Model: B (and B(M)), seed-median checkpoint.

- **Frozen-core ladder** (vertical): remainder after the exact `eps_inf` monopole flat to 5 meV over 959–2879 atoms for both variants. This is the only check that the dipole coupling adds no `1/L`.
- **Relaxation ladder**, route 1 as in 0.3: `eps0_ladder` for (R) and (M), each against `eps0_LST` computed from v5's Γ Hessian and v5's `Z*` with the matching `eps_inf`.
- **Dynamics:** NVE segments of 20 ps at 300 K in the charged +1 state at 159 and ~1000 atoms, fillings warm-started, dense eigensolver. Report energy drift (meV/atom/ps), the carrier's participation number and level separation along the trajectory, and the far-field ion displacement pattern against the continuum `q/(eps0 r^2)` form. NVT only for equilibration.
- **Cost:** ms/frame and peak memory at 79, 159, 1000 atoms; ratio to v5 alone.

**Decision 3.** Choose between (R) and (M) by `|eps0_ladder - eps0_LST|`; if both pass within 30 %, choose (M) (one fewer external scalar). Report the relaxed dilute-limit `E(+1) - E(0)` as `E_inf(frozen) + dE_relax(inf)` with the ladder fit's uncertainty, labelled as model convergence, not DFT accuracy.

Deliverable: `P3_REPORT.md`; the specification's §7–§9 rewritten to the new architecture and claims.

## What not to do

- No charge equilibration, no evaluation of `F` at the screened carrier field, no `Z*(R) u` energy, no learned scale on `E_cP`, no host potential inside `H0` while `E_cP` is explicit.
- `eps_inf` appears in exactly three places (`E_M`, `E_c`, `T_LR`) with one value per run.
- If the locality radius in 0.1 exceeds 6 Å, accept that the mid-field coupling reverts to `H0` for this iteration and proceed; do not add a second coupling representation.
- Do not add arms, seeds or gates beyond those listed without a written reason in the phase report.
