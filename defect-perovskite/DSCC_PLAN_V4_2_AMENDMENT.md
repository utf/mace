# D-SCC plan: v4.2 amendments (C5–C7 rulings; apply to v4.1)

Ruled by the user, 2026-09-07, on the tracker items C5 (single-valuedness at initialisation),
C6 (Route B centring) and C7 (energy admission). The user's reasoning, verbatim:

> **C5** — read the root rule on trained models, and add a bound-state precondition before Φ is
> switched on. At Harrison initialisation there is no vacancy level, the reference electron sits
> in a near-degenerate CBM manifold, and the localising feedback branches among ~17 Pb: that is
> the expected physics of §2.10 acting on a degenerate manifold, not an implementation fault.
> The gate belongs to trained models. The plan already provides the initialisation you need:
> Arm 2+3 starts from the Arm-1 winner (Φ = 0, no SCF, no branching). Make that explicit as an
> entry condition: Arm 2+3 opens only if the Arm-1 `H0` places the neutral-vacancy reference
> HOMO in the gap with a registered separation from the next level on a registered fraction of
> frames. Per-frame SCF initialisation with Φ on is the continuation from Φ = 0 (your
> initialisation ii); single-valuedness is monitored on a subsample during training with a
> registered ceiling on the failing fraction. Record the 8/20 and 900/1030 numbers as the
> initialised-model diagnostic. If Arm 1 does not produce a bound state, D-SCC with Φ cannot be
> opened; that is Stage-2 outcome B and goes to F-SCC or a pretraining step.
>
> **C6** — your kernel test is right and my centring argument was wrong. Centring removes the
> uniform (G = 0) component; the spurious 1/L cross term comes from the compact unbalanced
> charge at the vacancy interacting with the compact `dq` under the periodic kernel, and a
> uniform background leaves it untouched. That is exactly the FNV structure, and −8.6/−9.8
> versus −5.1 says so. Adopt Route B′: the static pattern is the reference-fill site charge
> `q0(R)` from `fill(H0(R), N_ref)`, which is locally neutral by construction, with two
> conditions. First, `q0` is evaluated at `H = H0`, never at `H0 − V`, so it does not respond
> to the carrier and does not double-count `eps_inf`. Second, it is conservative only with its
> geometry derivative in the force: `dqᵀ Γ_LR ∂q0/∂R`, one Fréchet contraction per frame with
> the machinery you built for D10; that term is the bond-polarisation physics that makes
> Mulliken statics behave like Born charges in the far field, so no learned scale is needed at
> first. If a scale is added, it is one global scalar; per-species scales break local
> neutrality at the vacancy and bring the 1/L error back. The ladder gate becomes a
> local-neutrality gate (slope within 5 % of Madelung), with the species pattern, centred or
> not, as the registered negative test, re-run on the trained Arm-1 `H0` since `q0`'s
> compensation cloud is the bound state.
>
> **C7** — the registered outcome stands: nothing is admitted. Train on forces at every size.
> Forces determine the head's energy up to one constant per charge state through
> conservativeness, so the only thing lost is `C_Q`, and one scalar does not need to be in the
> loss. Profile `C_Q` post hoc on the 159-atom charged frames whose `d` lies inside the
> out-of-fold neutral range (interpolation only; this is a retrospective refinement of the
> coverage rule and is labelled so, with the bin-level result recorded alongside). The
> 159-atom energy shape, `sQ = −0.095 ± 0.006`, then becomes a held-out test of the
> force-trained head rather than a fit, which is more informative than 17 energies in a loss
> dominated by forces; the 79-atom energies stay a prediction read against `s0 = +0.13`.
> Report `s0(159) ± SE` from the 12 frames; if the SE alone fails the slope test, the
> calibration is flagged "base support at 159 unverified". Arm selection replaces "energy
> shape on admitted sizes" with the 159-atom shape prediction error. D7 is right: `Q_i`
> vanishes at cubic sites, not at all centrosymmetric ones; it is nonzero in Pnma bulk and
> pristine-centred coefficients handle that.

## §2.5 — Route B replaced by Route B′ (locally neutral static pattern)
```
q0_i(R) = n0[Z_i] - sum_sigma Tr(Pi_i P_ref_sigma(H0(R)))     # reference fill at H = H0,
                                                            # NEVER at H0 - V
sum_i q0_i = 0 exactly; locally neutral by construction (the missing ion's charge is
compensated by the fill within the bound-state region)
W    = Gamma_LR @ (s * q0)          # s: optional single global scale in [0, s_max],
                                    # init 1; per-species scales are prohibited
E_SF = dq^T W
```
- Forces: `-dq^T (dGamma_LR/dR_J)(s q0) - s dq^T Gamma_LR (dq0/dR_J)`; the second term is
  mandatory (a detached `q0` is non-conservative). Implement as one Fréchet contraction:
  `A = U^T [sum_i (Gamma_LR dq)_i Pi_i] U`, `B = F o A`, `C = U B U^T`, force component
  `= -s * Tr(C dH0/dR_J)` on the H0 sparsity pattern.
- `q0` does not respond to `dq`; no `eps_inf` double counting.
- Correction of record: per-cell centring removes only the uniform (G = 0) component; a
  compact unbalanced charge at the vacancy produces a 1/L cross term with `dq` under any
  background. The centred/uncentred test is withdrawn.
- Gap regulariser in Route B′: on `H0 - W` with `W` from the pristine `q0`.
- Diagnostics: compensation-cloud extent (`R_eff` of `q0 - q0_pristine`) per frame.

## §5 / §8 — local-neutrality ladder gate (replaces the centred/uncentred gate)
Kernel-level tiling ladder with a fixed localised carrier: the 1/L slope of
`Phi_cc + E_SF` must equal the Madelung value (registered convention) within 5 %.
Registered negative test: the per-species pattern, centred or uncentred, must fail.
Re-run on the trained Arm-1 `H0` before any Route B′ arm opens.

## §5 / §7 — single-valuedness (C5)
- The root rule and convergence gates apply to trained models; the initialised-model
  outcome (root rule 8/20, convergence 900/1030 at 79 atoms) is recorded as a diagnostic.
- Bound-state precondition for Arm 2+3 (label-free, thresholds registered before Arm 1
  results are opened): on neutral-vacancy frames, the reference HOMO of the Arm-1 `H0`
  is separated from the next level by >= `Delta_c` (default 10 sigma_s) and the carrier
  has `N_eff <= N_loc`, on >= 95 % of frames. Failure -> Stage-2 outcome B; Φ is not
  switched on.
- Per-frame SCF initialisation with Φ on: continuation from Φ = 0 (initialisation ii).
  During training, single-valuedness is checked on a registered subsample per epoch;
  failing frames are dropped from that step with the fraction logged; a registered
  ceiling on the fraction fails the arm.

## §6 — energy admission outcome and calibration (C7)
- Registered outcome (v4 defaults): no charged energy admitted at either size
  (79: coverage fails, `s0 = +0.13 ± 0.01`, `sQ = +0.37 ± 0.02`; 159: 12 out-of-fold
  neutral frames, every bin below `n_min`, `sQ = -0.095 ± 0.006`). Recorded as is.
- Training: forces at every size; no energy term in the loss.
- Calibration (retrospective refinement, labelled): `C_Q` profiled post hoc, outside the
  loss, on 159-atom charged frames with `d` inside the out-of-fold neutral `d`-range
  (interpolation only, registered margin); reported with its SE. `s0(159) ± SE` reported
  from the 12 frames; if the slope test fails on SE alone, the calibration is flagged
  "base support at 159 unverified".
- Tests: the 159-atom energy-shape residual (after the one-scalar `C_Q`) is a held-out
  prediction of the force-trained head and replaces "energy shape on admitted sizes" in
  arm selection; the 79-atom energies remain a prediction read against `s0(79)`; the
  leak readout (`s_head` at 79 vs 159) is unchanged.

## §2.1 — descriptor note (D7)
`v_i` vanishes at centrosymmetric sites; `Q_i` vanishes only at sites of cubic point
symmetry and is nonzero in Pnma bulk. Coefficients remain bounded and pristine-centred.
