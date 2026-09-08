# D-SCC v4 — implementation tracker

Normative documents: `defect-perovskite/DSCC_PLAN_V4.md` (the user's plan v4, verbatim,
2026-09-06), `DSCC_PLAN_V4_1_AMENDMENT.md` (kernel-regime ruling and confirmations,
2026-09-06 evening) `DSCC_PLAN_V4_3_AMENDMENT.md` (2026-09-07: C8 ruling — precedence of the C5 gate over route C's stop; Arm 2+3 protocol = the Arm-1 protocol; §9 deferred robustness pass; benchmark engineering item), `DSCC_PLAN_V4_5_AMENDMENT.md` (2026-09-07: SCF warm starts during training, C9), `DSCC_C10_RULINGS.md` (2026-09-08: the user's C10 rulings verbatim, including the only text of a "v4.4" rule on file) and `DSCC_PLAN_V4_2_AMENDMENT.md` (C5–C7 rulings: Route B′,
bound-state precondition, forces-only training with post-hoc `C_Q`; 2026-09-07). Later
amendments take precedence. It **overrules** `TRANSITION_PLAN_V8_SPEC.md`, `TRANSITION_PLAN_V8_1_ADDENDUM.md`
and `TRANSITION_PLAN_V8_1_IMPLEMENTATION_SPEC.md`, which are history only. This file is the
task list and the registers the plan's §10 rules require. Paths relative to the worktree
`/home/alex/src/mace/.claude/worktrees/size-extensivity` (branch `defect`).

## 0. Operating rules

- **Task tracking.** The §3 checklist below is the task list; the status column is updated
  in the same commit as the work.
- **Phase gates.** A phase's code may be written early, but no later phase's *results* are
  opened until the previous phase's gates pass (plan preamble).
- **Thresholds first.** Every threshold, default and convention is entered in the register
  (§2) *before* the result it governs is opened; anything entered later is labelled
  `retrospective`.
- **GPUs.** b3 GPUs 4–7 only, up to two runs per GPU. **GPU 6 (PCI 0000:8B:00.0) faulted
  ("Unknown Error") at ~16:50 on 2026-09-06** and is unavailable until b3 is cold
  power-cycled — not done without a fresh instruction. Usable now: 4, 5, 7 (six runs). The
  local A4000 (GPU 0 on val-perovskite) is allowed for tests and small probes.
- **Git.** Commit only own paths, never `git add -A`; b3 is synced from committed content.
- **Analysis.** No analysis of any run beyond the plan's registered reports.

## 1. Open confirmations and decisions for the user

| # | Item | Status |
|---|---|---|
| C1 | Smearing of the labels: the old code records "doped's default ISMEAR = 0, SIGMA = 0.05 eV" as the labels' smearing (`defect_counting.py`). Plan §11 registers Gaussian 0.05 eV and asks for confirmation from the label paper's methods (Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025)). | **confirmed** (v4.1) |
| C2 | Whether `E_gap = 2.40 eV` is the static-lattice or the thermal-average PBE gap (decides which cell the gap regulariser acts on, §6). `band_edges.json` only carries the symmetric ±1.2 eV placement. | **ruled: static lattice** (v4.1) → the regulariser acts on the static pristine cell (D8); thermal-mean gap reported |
| C3 | b3 GPU 6 fault: cold power cycle now, or leave until GPU waves are needed (Phase 3)? | **ruled: leave** while GPUs 4, 5, 7 are accessible (checked 2026-09-06 evening: yes); cold power cycle only if not |
| C7 | **Energy admission (plan §6) admits no charged energies at either size under the registered defaults** (`dscc_admission_tables.json`, 4 outer folds, out-of-fold null from `cf_base_f0..f3`, bin 0.2 Å, `n_min` 3, `s_tol` 0.05 eV/Å, z = 2). 79 atoms: charged window d = 4.93–6.80 Å (1019 frames), 865 out-of-fold neutral frames per outer fold but none beyond 6.1 Å → coverage fails (as the plan expected); where measurable the null slope is `s0 = +0.13 ± 0.01 eV/Å` (> s_tol on its own); `sQ = +0.371 ± 0.022 eV/Å`. 159 atoms: 17 charged frames over 4.84–7.12 Å, only 12 out-of-fold neutral frames → every 0.2 Å bin is below `n_min` → coverage fails; `s0 = +0.08 ± 0.09`, `sQ = −0.095 ± 0.006 eV/Å`. Consequence: with nothing admitted, `C_Q` cannot be profiled and the head would train on forces alone. Options: (a) register a coarser 159-atom coverage rule (e.g. bin 0.5 Å, `n_min` 2 — still fails at the window ends), (b) admit 159-atom energies on the s0 criterion alone with coverage reported as failed (a documented deviation), (c) accept forces-only training and report `E(+1)−E(0)` as uncalibrated. Ruling needed before Phase 2 opens. | **ruled (v4.2)**: the registered outcome stands — forces at every size, no energy term in the loss; `C_Q` profiled post hoc (labelled retrospective) on 159-atom charged frames with `d` inside the out-of-fold neutral range; `s0(159) ± SE` reported from the 12 frames (currently +0.08 ± 0.09: SE alone fails `s_tol` → flag "base support at 159 unverified"); the 159-atom shape residual becomes the held-out prediction that replaces "energy shape on admitted sizes" in arm selection |
| C6 | **Route B centring does not remove the `1/L` cross term** (kernel-level tiling ladder with a fixed localised carrier, `test_kernels.TestRouteBLadder`): a per-species pattern on the present ions leaves the missing ion as a LOCAL charge `z_d = -Zstar_Cl` at the vacancy; centring spreads its compensation uniformly over the cell, an O(1/L³) change. The carrier's images interact with the local charge, so `Phi_cc + E_SF` carries `-alpha_M C Q z_d / (eps_inf L)` on top of the physical `-alpha_M C Q² / (2 eps_inf L)`: measured slope −8.59 eV·Å (centred), −9.79 (uncentred; the `(1 + 2 z_d)` prediction is −10.2) vs −5.11 (Madelung) with `Zstar_Cl = -0.5`. A pattern that is neutral locally (the missing ion's charge put back at the vacancy — what the model's own reference-fill charges `q0_i` do by construction, since `sum_i q0_i = 0` on the reference) keeps the Madelung slope to 5 %. So plan §2.5's positive/negative ladder test cannot distinguish centred from uncentred, and Route B as written adds a size-dependent term no `C_Q` absorbs (0.13 eV × 2 z_d between 79 and 159 atoms). Proposed Route B′: the pattern is the reference-fill site charge `q0_i` (model-derived, label-free, exactly neutral per cell, local compensation included), optionally scaled by one learnable per species; ruling needed. | **ruled (v4.2): Route B′** — pattern = `q0(R)` from `fill(H0(R), N_ref)` at `H = H0` (never `H0 − V`), one optional global scale `s ∈ [0, s_max]` (init 1; per-species scales prohibited), the geometry derivative `s dqᵀ Γ_LR ∂q0/∂R` mandatory in the force; local-neutrality ladder gate (5 % of Madelung) with the species pattern as the negative test, re-run on the trained Arm-1 `H0`; the centred/uncentred test withdrawn |
| C5 | Single-valuedness gate (plan §5, root rule): at initialisation the real frames have no bound vacancy state and the SCF has distinct fixed points on some frames (D10 finding). Read the gate on the trained model (the arms report the root rule anyway), or require an initialisation with a bound state first? | **ruled (v4.2)**: gate on trained models; bound-state precondition for Arm 2+3 (`Delta_c` = 10 σ_s, `N_eff ≤ N_loc`, ≥ 95 % of neutral-vacancy frames); per-frame SCF init = continuation from Φ = 0; single-valuedness monitored on a per-epoch subsample with a registered ceiling; 8/20 and 900/1030 recorded as the initialised-model diagnostic |
| C8 | **Arm 1 routed C twice and the registered rule stopped the programme before Arm 2+3 (2026-09-07 05:30).** Criterion (ii) fails on both rounds: the scalar-only control loses only 5 % / 8 % of the level-vs-bond slope (threshold 30 %), so the directional block does not carry the modelled level-vs-bond response; (i) and (iii) pass both rounds; (iv) fails on the `N_eff` spread (round one) and on force quality (repeat). The block's measurable contribution is the bound state (separated vacancy level on every full seed; the control loses it on most frames and diverged on two repeat seeds). Route C was applied in the v4 wording ("repeat covariance/sign tests once"); the v8.1 table's extra clause, "and the single registered bound release", was not carried into v4 and was not done. Nothing opens without a ruling. Options, none taken: (a) accept the stop — the attribution experiment is negative as registered; (b) re-register, post hoc (two rounds already opened, so this changes the attribution test after the fact), what Arm 1 is meant to attribute (a localisation criterion — bound-state precondition / HOMO separation — in place of the slope test (ii)) and read the existing twelve full seeds against it, no retraining; (c) a third round under a registered LR decay (the final-checkpoint noise is ±10 meV/Å) — outside the rule; (d) open Arm 2+3 on the full `H0` anyway as an exploratory, non-registered arm. | **ruled (v4.3, 2026-09-07):** "stop" in route C stops capacity growth (no edge residual, no bound release, no further onsite capacity); the C5 bound-state precondition, registered before Arm 1 opened, is an earlier gate for Arm 2+3 and is preserved — the scalar-only `H0` fails it on every seed, the full `H0` passes on 9/12, so the block is retained on the C5 gate, not on (ii). Winner: full `H0`, seeds {0, 1, 2, 3, 4, 6, 7, 9, 10}; excluded {5, 8, 11} (92.9 / 94.1 / 92.9 %). Criterion defects recorded, not re-registered (tensor ratio descriptor-only; slope clause's seed spread ≈ 10× its SE). Post-hoc reading (labelled) of full vs control on separation and precondition = the block's demonstrated function, not the basis of admission. Scientific record: the block does not carry the level-vs-bond response; it carries the p_σ/p_π splitting at the flanking Pb that stabilises the Pb–Pb bound state — this supersedes the original hopping-saturation hypothesis |
| C9 | **v4.5 amendment (user, 2026-09-07 12:40; `DSCC_PLAN_V4_5_AMENDMENT.md`, verbatim): SCF initialisation during training.** Trigger: the coupled Arm-2+3 epochs took 20–33 min at four per GPU (the Φ = 0 arms 90 s) — a profile of one coupled step put 65 % of the GPU time in 32 batched eigendecompositions per batch (4 continuation stages × Newton with trial re-diagonalisations) against one for Φ = 0. Ruling: first visit by the registered four-stage continuation with a tangent predictor, the converged `dq` stored per frame; later visits warm-started, Newton with LM damping (no trial re-diagonalisations), converged to `tol_q`, `tol_E`, `tol_c`; a registered 5 % per-epoch subsample re-run by continuation against the warm start (`tol_root`), the failing fraction under the registered ceiling, failure fails the arm; solver engineering permitted without further ruling; gate before restart (fixed points, band identity, FD forces, implicit gradients vs the previous implementation). | **implemented and gated (commit `997e53a`); campaign restarted 2026-09-07 13:53 (b3 clock)** — see P3.2 and the registers (§2.1) for the gate numbers and one recorded deviation: the damping is a deferred-evaluation damped Newton, not Levenberg–Marquardt (below) |
| C10 | **Arm 2+3 outcome (campaign complete 2026-09-08 13:08, report 13:20; `~/runs/dscc/arm23_report.json`; note `ARM23_COADVISOR.md`; D12): the registered v4.3 rule selects `B_A_phi0` — the Φ = 0 head — at a held-out charged force RMSE median of 39.9 meV/Å (seeds 36.8 / 41.2 / 40.0 / 39.1 / 41.6 / 39.8), `tau_noise` 1.7, margin `max(tau_phys, tau_noise)` = 3.0 meV/Å; the coupled medians are LR+U 41.3, LR-only 41.7, full 42.0, λ = 1 43.3 (regime-A ablation 40.9, not selectable). Every coupled arm that reaches the ranking is EQUIVALENT to Φ = 0 (none beats it, none is beaten beyond the margin; λ = 1, at 3.4 meV/Å above Φ = 0, is the one arm beyond the margin and leaves on `localisation_stable` before the ranking) and Φ = 0 is the simplest, so the selection is the same under every reading of the two contested gates (counterfactuals run on the same records: with the `f_SR` gate removed and the last-epoch root rule, the equivalent set is {Φ = 0, LR+U, LR-only, full}, λ = 1 excluded by `localisation_stable`; Φ = 0 selected in all four readings). The optimiser turned the coupling off: in the full arm `lambda_dir` trained from 0.05 to 0.004–0.018 and the hub-site `U_eff(Pb)` from 1.36 eV to 0.04–0.18 eV (LR+U: 0.03–0.18; λ = 1: 0.06–0.17); the full and λ = 1 arms are worse on the median inside 2 Å (93 and 90 vs 84 meV/Å, within their seed spreads of 26 and 15), LR + U is equal (84), and every arm equals Φ = 0 beyond 6 Å. Four rulings are needed.** (a) **Root-rule reading.** The report reads the v4.5 check at its worst epoch ≥ 1 (v4.5: "failure fails the arm"): full, LR+U and λ = 1 fail (worst-epoch failing fractions 8–26 %, 8–18 %, 10–36 % against the 10 % ceiling; LR-only and regime A 0–8 %; pooled over all 59 checked epochs, 2301 frames per run: full 0.3–2.3 %, LR + U 0.3–2.4 %, λ = 1 0.9–7.7 %, regime A ≤ 0.2 % — every run under the ceiling on the pooled reading). The over-ceiling epochs are the EARLY ones — full: epochs 1–5 and 14; LR+U: 1–8 and 18; λ = 1: 1–31 (seed 4 over on 22 epochs) — none within two epochs of a logged loss spike (spikes sit at epochs 20–58), and the final-epoch check is 0/39 on all 36 runs: the trained maps are single-valued and the failures are the transient from the multi-valued initialised map (28–51 % of frames at epoch 0, C9) that decays as `U_eff(Pb)` falls. Under the last-epoch reading full and LR+U pass and λ = 1 still fails `localisation_stable` (N_eff seed spread 0.88 > 0.5, seed 6 at 5.5). Selection unchanged either way; the record keeps the strict reading unless overruled. (b) **The `f_SR` gate is a criterion defect (ours).** `f_SR` = Σ|dq_i dq_j| K_SR_ij / Σ|dq_i dq_j| (K_SR + K_LR)_ij reads 1.5–1.8 (median 1.6) on every regime-B coupled arm, 1.2 on regime A, and 3.1 on one λ = 1 seed — above 1, which a fraction cannot be. Measured on one 79-atom charged frame with the trained full-s0 model: SR part +0.41 eV, LR part −0.19 eV, `K_LR` NEGATIVE on all 3081 pairs (range −0.97 to −0.31 eV/e²): in a 79-atom cell the long-range component of the intra-carrier interaction is negative (the neutralising background dominates it beyond `r_s`), so the quantity is unbounded and the registered `≤ 0.5` was set on a scale it does not live on. It also does not discriminate between arms (LR-only reads 1.6 with no `K_SR` in its Γ — the report computes it from the kernel split and the trained `dq` whatever the arm) and Φ = 0 passes it vacuously (no `dq`). Recorded, selection unaffected, not re-registered (same status as Arm 1's tensor-ratio clause). The plan's regime-B small-cell rule (λ → 0 on the ladder) is a different test and has not been run. (c) **Epoch-0 reading** (C9, pending): the initialised-model check is the diagnostic, the ceiling applies from epoch 1 — confirm or overrule (under the other reading every arm but LR-only fails at initialisation). (d) **Downstream with Φ = 0 selected.** Arm 4's F-SCC comparators and decision (1)–(6) compare against a coupled D-SCC model that the selection did not produce; the 2× benchmark engineering item is registered "on the Φ-on model" (the Φ = 0 benchmark already reads 2.07 / 2.22 at 79 atoms, P4.3); the Phase-4 ladder's `K_LR_ii` and `dq`-spread items are vacuous at Φ = 0 (its `E(+1) − E(0)` vs `1/L` remains readable as the base + `H0` size behaviour); the §9 robustness pass retrains "the selected configuration and its Φ = 0 reference", which are now one arm. Options: (i) close at the Arm-2+3 selection — §9 pass on Φ = 0 (mechanical, 1.5–2 h per seed on the A4000), P4.1 ladder on it, final artefact; (ii) additionally retrain LR+U (the best coupled arm) under the §9 protocol to test whether the equivalence survives the final-checkpoint noise — note that a coupled arm would have to BEAT Φ = 0 by more than `tau_phys` = 3 meV/Å to change the selection, and the trained couplings are near zero, so the pass cannot plausibly reverse it; (iii) open Arm 4 on the full arm as an exploratory, non-selected comparison of the D-SCC vs F-SCC feedback sign (§2.10). My reading: (i), with (iii) only if the feedback-sign question is wanted on its own. **§2.10 forecasts against the outcome:** the Route-A forecast ("`lambda_dir` and `U_eff` come out O(1); the λ = 0 ablation makes the level shallower and the carrier more extended") is NOT borne out in regime B — the couplings trained toward zero and the λ = 0 ablation (LR+U vs full) leaves `N_eff` at 3.7 vs 3.6 and forces 41.3 vs 42.0 meV/Å; in the regime-A ablation `U_eff(Cl)` did go to 2.3–8.9 eV (Pb 0.16–0.34) at an equivalent force error. The "practical consequence" forecast (distinct fixed points at asymmetric geometries) is borne out at initialisation and during the first epochs (a). | **RULED 2026-09-08 (`DSCC_C10_RULINGS.md`, verbatim; applied in `23a4639`, `be3c39b`; D13).** (a) Root rule: read on the final model and the last ten epochs (`sv_fraction_last10`), the early-epoch failures recorded as the initialised-map transient (`sv_transient_end`), ceiling from epoch 1 — every arm passes (λ = 1 at most 5 % on its last ten; transients end at epochs 2–14 full, 5–18 LR + U, 5–31 λ = 1). (b) `f_SR`: criterion defect (the user's: "in a small cell `K_LR` is self-image dominated and negative on every pair, so a signed fraction is meaningless"), withdrawn as a gate, reported alongside the absolute share `f_sr_abs` = Σ|dq_i dq_j K_SR_ij| / Σ|dq_i dq_j|(|K_SR| + |K_LR|) — 0.69–0.71 on every regime-B coupled arm, 0.46 on regime A. (c) Epoch 0: diagnostic, confirmed. (d) **Selection under the user's "v4.4" rule — `K_LR` required physics with no learnable coefficient, not subject to the equivalence rule; the minimal candidate LR-only; the Φ = 0 arms force references: LR-only SELECTED (`select_v44`; candidates LR + U 41.3 best, LR-only 41.7, full 42.0 equivalent; λ = 1 out on `localisation_stable`; cost against Φ = 0 +1.8 meV/Å inside the 3.0 margin, "costs nothing in forces"). No v4.4 document exists in the repository, the tracker or the session record (only the user's 2026-09-08 message states it), so this selection is labelled POST HOC; the v4.3 rule as registered still resolves to Φ = 0 (with the C10 gates: {Φ = 0, LR + U, LR-only, full} equivalent). `arm23_report_c10.json`. Not closed: Route B′ (six seeds, same protocol) and the matched-kernel F-SCC comparator (three seeds) LAUNCHED 2026-09-08 15:13 (b3 clock) — see P3.2b, P3.3. **Why Route B′ was absent (the user asked):** its registered v4.2 entry gate — the local-neutrality ladder on the trained `H0` — failed on all four Arm-1 winners (51–65 % off Madelung; the reference-fill compensation of the missing ion extends over 8–11 Å) and the tracker recorded "Route B′ arms are therefore not opened"; re-run on the Arm-2+3 Φ = 0 models s0–s3 it fails again (`q0` slope 51 / 53 / 61 / 51 % off Madelung, species negative test passes; `arm23_phi0_gates.json`). B′ therefore opens BY THE RULING over the C6/v4.2 entry gate; its registered selection gate "tiling-ladder `1/L` coefficient within 5 % of Madelung" fails by construction on this `H0` and awaits a ruling (the code keeps it as registered). **Prerequisite diagnostic (D13): the far-field premise was mis-scaled** — the trainer's held-out numbers are the RMS of the per-atom force-error VECTOR (√3 × the per-component RMS of MACE's logs and of the base's 13 meV/Å); in one convention the charged far field of the Φ = 0 head at 159 atoms sits AT the out-of-fold neutral floor beyond 10 Å (7.2 vs 6.9, 4.5 vs 4.7 meV/Å) and 5 meV/Å above it at 8–10 Å, while at 79 atoms it is 18 meV/Å above it in quadrature at 8–10 Å: by the user's own criterion "small-cell and manifold extrapolation", not the Coulomb channel, and B′ is not mandatory on that ground. Open item recorded: the near-field residual (0–2 Å: 48 per component / 83 in the trainer convention against a 31 out-of-fold neutral floor) is unmoved by every electrostatic arm. **Rulings still needed:** (e) the B′ ladder selection gate (waive, re-read, or B′ reads on forces only); (f) whether to trim the B′ queue to LR-only and LR + U given the diagnostic (the queue runs LR-only, LR + U, full, λ = 1, then the regime-A B′ ablation; stop with `scratchpad/kill_bp_b3.sh`); (g) confirmation of the far-field gate reading now implemented (4–6 and 6–8 Å pooled by atom count, gain over the same-coupling Route A arm beyond the Route A Φ = 0 seed spread); (h) the v4.4 document itself, if one exists. |
| C4 | **Regime-A placement check fails on the training set at the plan's defaults** (`r_d1 = 3.2`, `r_d2 = 3.6` Å, floor 1e-3). Identified first-shell Pb–Cl bonds beyond `r_d1`: 3.1 % on the 544 pristine frames, 5–7 % on the 79-atom vacancy frames (the six-nearest rule counts a Cl across the vacancy for the flanking Pb), 1.2–1.6 % at 159 atoms; intra-octahedron Cl–Cl edges below `r_d2`: 2.0–2.6 % (0 at 159 atoms). The first-shell Pb–Cl distribution is thermal and wide: p50 2.88 Å, fraction > 3.0 / 3.1 / 3.2 / 3.3 Å = 22.5 / 10.8 / 5.6 / 3.4 %. No window in the 3.2–3.6 Å gap clears a 1e-3 floor on these frames. Options for ruling: (a) register a higher floor (≈ 5e-2) with the physical consequence that a few % of first-shell pairs sit inside the switch; (b) move the window up (e.g. 3.5–4.0 Å, still below the 4.8 Å vacancy-spanning minimum, but then the Cs–Cl shell at 3.5–4.1 Å is inside it); (c) make regime B (no switch) the primary regime and report regime A as failing its placement gate. Also: exclude the vacancy-flanking Pb's sixth neighbour from the identified set, or run the check on pristine frames only. | **ruled (v4.1): regime B primary; regime A failed its gate, retained as a reduced ablation only** (retrospective floor 5e-2 for the ablation; flanking-Pb bond fraction and `m_sw` reported; excluded from selection) |

## 2. Registers

### 2.1 Registered values (plan §11)

| Quantity | Value | Source / status | Registered |
|---|---|---|---|
| `sigma_s` | 0.05 eV, Gaussian (`f = erfc(x)/2`, `R = -sigma_s sum exp(-x^2)/(2 sqrt(pi))`) | plan §1/§11; label-code convention | 2026-09-06 (plan) |
| `n0` | Cs 1, Pb 4, Cl 7 | plan §1 | 2026-09-06 (plan) |
| `eps_inf` | 4 | plan §1 (host input) | 2026-09-06 (plan) |
| `E_gap` | 2.40 eV | plan §1 (host input); convention C2 open | 2026-09-06 (plan) |
| Coulomb constant `C` | 14.399645 eV·Å | `defect_madelung.COULOMB_CONSTANT` | 2026-09-06 |
| Gaussian width convention | a charge of width `r_g` has density `∝ exp(-r²/(2 r_g²))`; two of widths `s_i`, `s_j` interact as `C erf(r / sqrt(2 (s_i² + s_j²))) / r` (equal widths: `erf(r/(2 r_g))/r`, self term `C/(sqrt(pi) r_g)`); the LES `sigma` is `sqrt(2) r_g` (measured: cross term of two unit charges, Richardson-extrapolated in 1/L, matches `r_g = sigma/sqrt(2)` to 4e-5 at r = 1–3 Å); LES with `remove_self_interaction` excludes the Gaussian self term (a lone unit charge gives `-alpha_M C/(2L)` exactly: `E·L = -20.428` at L = 20/40/80 Å) | `scratchpad/les_convention.py`, 2026-09-06 | 2026-09-06 |
| `U_max[Z]` | GFN1-xTB hardness `GAM` (Hartree): Cl 0.519712, Cs 0.085110, Pb 1.000000 → 14.142, 2.316, 27.211 eV; bounds only | `param_gfn1-xtb.txt` (grimme-lab/xtb, doi 10.1021/acs.jctc.7b00118) | 2026-09-06 |
| Background | VASP neutralising background; `E_bg = -pi C sigma^2 Q^2 / V` for Gaussian width `sigma` | plan §11; `latent_ewald.LatentEwald.energy` | 2026-09-06 (plan) |
| Spin split | majority-channel: `N_ref_up = ceil(N_ref/2)`, `N_ref_dn = floor(N_ref/2)` | plan §1 | 2026-09-06 (plan) |
| Base features | `l = 1` only, position derivatives through the recomputed first block | plan §11 | 2026-09-06 (plan) |
| Regime A `r_d1`, `r_d2` | 3.2 Å, 3.6 Å (defaults) | plan §2.4 | to register before Arm 2+3 |
| Placement-check floor | 1e-3 (default) | plan §2.4 | to register before Phase 0 gate |
| `m_sw` floor, `f_SR` floor | `m_sw` 0.05, `f_SR` ≤ 0.5 (in the Arm-2+3 thresholds row below) | plan §2.4 | registered 2026-09-07; **`f_SR` found unbounded on the Arm-2+3 outcome (C10 (b), D12): `K_LR` is negative on every pair of a 79-atom cell, so the "fraction" reads 1.2–3.1 on every coupled arm; recorded as a criterion defect, not re-registered** |
| Regime B `r_s` | 6.5 Å (default); `K_SR` image range: converged to 1e-10 eV automatically | plan §2.4 | to register before Phase 0 gate |
| `r_g`, `r_split` | — | plan §2.4/§2.5 | to register before Phase 0 gate |
| `lambda_0`, `lambda_max`, `U_max[Z]` (GFN1-xTB hardness, bounds only) | — | plan §2.4 | to register before Phase 1 |
| `n_max`, `tol_q`, `tol_E`, `tol_c`, `tol_root`, `rho` ceiling, continuation schedule, mixing | see the solver row below | plan §2.6 | confirmed 2026-09-07 |
| `Z_max`, Route B L2 weight | `Z_max = 2 × max_s |Zstar_init,s|` (`Z_MAX_FACTOR`); L2 weight: — | plan §2.5; `model.py` | default 2026-09-06, to confirm |
| `lambda_0`, `lambda_max`, `U_eff` init | 0.05, 2.0, `0.05 U_max[Z]` (`LAMBDA_0_DEFAULT`, `LAMBDA_MAX_DEFAULT`, `U_INIT_FRACTION`); bounded by `x_max sigmoid(raw)` | `model.py` | defaults 2026-09-06, to confirm before Phase 1 gate |
| `r_g`, `r_split`, `q_cut`, `a_max`, `b_max`, `r_cut` | 1.0 Å, 2.5 Å, 4.5 Å, 1.0 eV, 1.0 eV, 10.0 Å | `kernels.KernelConfig`, `model.py`, `hamiltonian.py` | defaults 2026-09-06, to confirm |
| Solver: `tol_q`, `tol_E`, `tol_c`, `n_max`, mixing, history, `tol_root`, `rho` ceiling, continuation steps | 1e-8 e, 1e-10 eV, 1e-7, 100, 0.3 (Anderson), 6, 1e-6, 0.9, 4 | `scf.ScfOptions` | confirmed 2026-09-07: unchanged since registration and used in every run (Arm 1 at Φ = 0 does not solve); D10/D11 added `method = newton` (damped, Anderson fallback), `max_backtrack` 6, continuation steps 4 |
| `s_tol`, `z`, `n_min`, coverage bin width, out-of-fold base protocol id | 0.05 eV/Å, 2.0, 3, 0.2 Å, `cf_base_4fold_seed0` (`admission.AdmissionConfig`) | plan §6 | defaults 2026-09-06, to confirm before Phase 2 results are opened |
| Gap-regulariser convention | static-lattice or thermal-mean (C2) | plan §6 | before Phase 2 |
| Benchmark hardware / batch / trajectory | local NVIDIA RTX A4000 (16 GB), CUDA, float64, GPU otherwise idle; batch = one frame; trajectory = the first N consecutive charged frames of one source in `dataset_pbe/train.xyz` at the stated size (N = 40 at 79 atoms, the 16 available at 159; re-run with N = 100 where available), warm-started from the previous frame's `dq` (initialisation iii); solver at the registered `ScfOptions` defaults; the base alone on the same frames at its own cutoff, same device; median and p95 of the per-frame ratio | `defect-perovskite/dscc_benchmark.py` | registered 2026-09-07 |
| Training defaults (`train.TrainConfig`) | epochs 60, lr 2e-3 (Adam), batch 4 (size-grouped), force weight 1, gap weight 1 eV⁻², grad clip 10, eval every 5, single-valued subsample 16 / ceiling 0.10, outer fold = seed mod 4 for the six seeds | `dscc/train.py` | defaults 2026-09-07, registered before Arm-1 results are opened |
| Arm-1 thresholds | (i) stop ≤ 1/6 seeds (`|tanh g| ≥ 0.98` on > 50 % of frames); (ii) flanking `|bQ|` > 3× bulk spread and control loses ≥ 30 % of the level-vs-bond slope; (iii) Cl `eps_sigma − eps_pi < 0`; (iv) `N_eff` seed spread halves at ≤ force RMSE, saturation < 50 % | `dscc/arm1.py` | registered 2026-09-07, before results are opened |
| Arm-2+3 thresholds (registered 2026-09-07 before any Arm-2+3 result is opened) | Entry: bound-state precondition on the Arm-1 winner (`Delta_c` 0.5 eV, `N_loc` 4, ≥ 95 %); Route B′ arms additionally the local-neutrality ladder gate (5 % of Madelung, species pattern fails). Selection (joint over route × coupling, regime B): held-out charged force RMSE and the 159-atom energy-shape prediction error (v4.2) must beat the best competitor by more than `max(tau_phys, tau_noise)` with `tau_phys` = 3 meV/Å (forces) / 15 meV/Å (159-atom shape slope), `tau_noise` = the six-seed standard error; localisation stability = seed spread of `N_eff` (p50 over held-out charged frames) ≤ 0.5; root rule passed on the per-epoch subsample with failing fraction ≤ 0.10 and SCF within `n_max` = 100 on ≥ 99 % of training frames (final epoch); `f_SR` ≤ 0.5 (floor: at most half the intra-carrier interaction in the short-range component); Route B′ additionally: the 4–8 Å far-field shell residual improves beyond `tau_noise`, `s` inside `[0.1, 1.9]` (unsaturated), tiling-ladder `1/L` coefficient within 5 % of Madelung. Regime-A ablation excluded from selection, reported with `m_sw` (floor 0.05) and the flanking-Pb fraction. `r_s` sensitivity of `Phi_cc` reported. | `dscc/arm23.py` | registered 2026-09-07; **C10 (2026-09-08): root rule read on the final model and the last ten epochs; the `f_SR` gate withdrawn (defect) and `f_sr_abs` reported; the far-field gate implemented as registered — the 4–6 and 6–8 Å shells pooled by atom count (shell-volume weights 152 : 296 for runs recorded before C10, which carry no counts), the B′ arm's median against its same-coupling Route A arm's, the gain beyond the Route A Φ = 0 seed spread; the > 8 Å shells (8–10, 10–12, > 12) resolved in the trainer from `be3c39b`; the B′ ladder gate kept as registered though it fails by construction on this `H0` (C10 (e))** |
| Arm-2+3 protocol (v4.3, registered before launch) | **60 epochs, constant lr 2e-3, batch 4, final checkpoint read — identical to Arm 1** (the earlier 30-epoch registration is withdrawn by v4.3); initialisation `h0_state.pt` of each admitted Arm-1 seed; per-frame SCF by continuation from Φ = 0; no LR schedule, no spike detection, no checkpoint averaging. Noise handling statistical only: `tau_noise` for every force comparison = the seed spread of the Φ = 0 arm under this protocol; comparisons on seed medians; arms inside `max(tau_phys, tau_noise)` are equivalent and the simpler one is chosen (order Φ = 0 < LR-only < LR+U < λ = 1 fixed < full; route A before B′); Arm-1 force numbers are not a baseline. Consequence recorded: the ±10 meV/Å final-checkpoint noise stays in every force reading and favours the simpler arms. Loss-spike events logged (epoch, loss, RMSE before/after) and not acted on. Seeds: the C5 gate admits {0, 1, 2, 3, 4, 6, 7, 9, 10}; **the user's instruction at launch (2026-09-07 08:55: "just use 6 seeds otherwise this will take too long") sets six: {0, 1, 2, 3, 4, 6}** — the five admitted round-1 seeds plus the lowest-numbered admitted repeat seed (precondition 97.8 %); folds {0, 1, 2, 3, 0, 2}; factorial v4.1, Route A opened, Route B′ only if its ladder gate passes on the trained `H0` (it has not) | `DSCC_PLAN_V4_3_AMENDMENT.md`; `dscc/arm23.py` | registered 2026-09-07 08:45, before launch |
| Sparse path (plan §8) | `k_buffer` = 8 (window = `|Q| + k_buffer` states, grown until the certified omitted-charge bound ≤ `tail_tol` = 1e-8 e); frontier eigenpairs to machine precision (residual ≤ 1e-8 checked); LDLᵀ factorisation residual ≤ 1e-8; dense regime (dense Ewald matrices, dense fills where used) ≲ 5 000 atoms; `max_active` (dense fallback) = the window size at which ARPACK exceeds the dense eigh — to register from the benchmark | `dscc/sparse.py` | registered 2026-09-07 |
| Arm-4 thresholds | — | plan §7 | before Arm-4 results are opened |


**v4.5 registers (2026-09-07, before the restart).**
- `warm_check_fraction = 0.05` of the frames visited in a pass (training epoch: of the training frames at the start of the epoch; evaluation: of the held-out frames), `tol_root = 1e-6` unchanged, ceiling `single_valued_ceiling = 0.10` unchanged; frames that fail leave the epoch's steps AND the store; the run-level flag `arm_failed_single_valuedness` (any epoch over the ceiling) is written to `held_final.json` and the report reads the worst epoch (`sv_fraction`).
- Epoch 0 has no warm solutions: the check there is the v4.2 form (zero start against the continuation) on the same subsample, so every epoch carries a logged check; from epoch 1 the v4.5 form (warm start against the continuation). Frames without a stored `dq` at any later epoch are checked in the v4.2 form.
- Held-out frames are warm-started from their previous evaluation (five epochs stale) with the same 5 % check logged under `held.single_valued`.
- Solver: continuation stages `s = k/4` unchanged; `tol_c = 1e-7` now enforced inside the loop (evaluated once `tol_q`, `tol_E` pass; the measured commutator floor at convergence on real frames is 1e-13–1e-10); `n_max = 100` bounds ACCEPTED iterates (the pre-v4.5 meaning), rejected fills are reported separately (`fills`).
- **Deviation, recorded:** Levenberg–Marquardt damping (normal-equations form, Marquardt scaling; `damping = "lm"`, kept selectable with its constants `lm_mu0 = 1e-6`, `lm_up = 100`, `lm_down = 0.1`, `lm_max_reject = 4`) was implemented first and found not robust far from the fixed point on real frames: from a zero start on dataset frame 2510 it ran 100 iterations without converging where the backtracking Newton converged in 12 (its gradient-like direction at large `mu` does not reduce the residual at a near-degenerate frontier). The production damping (`damping = "newton"`, default) is the deferred-evaluation damped Newton: the full Newton step is taken and judged by the NEXT iteration's fill (max-norm, as the old backtracking test), a rejected step is halved from the same Jacobian, six rejections fall back to the Anderson step on the accepted history — the pre-v4.5 iterate sequence exactly, without re-diagonalising the accepted point (one eigendecomposition fewer per accepted step). The amendment's stated intent (no trial re-diagonalisations) is met; its named method is not used. The pre-v4.5 solver remains selectable (`damping = "backtrack"`) for the gate.
- **Deviation 2, recorded:** the tangent predictor between continuation stages (`dq + Δs (I − s M Γ)⁻¹ M (Γ dq + W)` from each stage's spectrum, no extra diagonalisation; `predictor = True`, with a trust cap `predictor_trust = 1.0` on the predicted change) is implemented and selectable but OFF by default. Measured on real frames with the production damping: it saves one iteration in fifteen on 79-atom frames (15 → 14) and costs on the hard 159-atom class (40 → 42 iterations at λ = 0.05; 18 → 23 iterations, 47 fills against 19, at λ = 0.5), the fixed points the same to ≤ 2.4e-9 — on first visits only (one epoch in sixty plus the 5 % check). It does not pay for itself on this data. (An earlier gate run that showed a stage exhausting `n_max` with the predictor was running LM damping, not the production damping; LM alone took 69 and 90 iterations on the two hard frames where damped Newton takes 40 and 18.) The user may switch the predictor on by ruling.
- Gate (`defect-perovskite/dscc_v45_gate.py`, `~/runs/dscc/v45_gate.json`): **PASS** on 8 × 79-atom and 2 × 159-atom charged frames (the second the 40-iteration class) at λ = 0.05 and 0.5, on CPU: fixed points, energies, forces and implicit parameter gradients of the v4.5 defaults equal the previous solver's to ≤ 1.4e-12 (79 atoms: bit-identical iterates; 159: dq 8e-15–7e-13, gradients 3e-13–4e-12 relative), band identity ≤ 1.2e-12 eV, commutator ≤ 6e-10, FD forces (h = 2e-3) within 4.3e-6 eV/Å; accepted iterations equal the previous solver's (15–19 per 79-atom continuation, 17/40 and 17/18 on the two 159-atom frames), time per batch 3.5 → 3.0 s (79) and 10.4 → 8.6 s (159) without the base cache; warm starts after a 2 % move of λ and U and 1e-3 of H0 converge in 4–8 fills against 15–19 for the continuation, to the continuation's fixed point within 4e-10 (gradients 3e-9 relative), and the previous solver warm-started from the same stored dq lands on the same point (≤ 1e-10). Variants recorded there: predictor 15 → 14 / 40 → 42 / 18 → 23 iterations; LM 69 and 90 iterations on the hard frames.

### 2.2 Decisions

**D1 — build strategy (2026-09-06).** The head is built in a new namespace —
`mace/modules/dscc/` with a new model class `MACEDSCC(ScaleShiftMACE)` and tests in
`tests/extensions/dscc/` — and the old `defect_*` modules are left untouched until the Phase 1
gates pass; the §3 deletion is then one sweep commit ticked against the plan's delete list
(§3.6 below). Reuse is by import: `SlaterKosterH`, `harrison_initialise`, `find_mu`,
`occupation_of`/`entropy_of`/`occupation_slope` (Gaussian default), `_FermiDensityMatrix` /
`_dk_backward`, `site_charges`, `head_forces` (`defect_counting`); `ElectronicStateSpec`,
`reference_state` (`defect_state`); `profile_charge_constant`, strata (`defect_objective`);
`LatentEwald` (`latent_ewald`) as the Ewald *oracle*; the base feature cache
(`defect_cache.BaseOutputCache`). Not imported: gauge, windows, lift, boundary, composition,
`profile_intercepts` (all on the delete list). Every commit stays green; old checkpoints stay
loadable for reference until the sweep.

**D2 — state adapter (2026-09-06).** From the dataset's `carrier_counts = (e_maj, e_min,
h_maj, h_min)`: `dN_up = e_maj - h_maj`, `dN_dn = e_min - h_min`, `Q = -(dN_up + dN_dn)`;
`Q == cell_charge` asserted on every frame at load. Reference split per §1 (majority channel);
`m_s_ref_doubled` of the dataset is the same convention (79-atom V_Cl: N_ref = 409 → (205,
204); `vcl_qp1`: (204, 204), Q = +1).

**D3 — Ewald matrix (2026-09-06).** `E_PBC` is built as an explicit real + reciprocal Ewald
matrix of unit Gaussian charges in float64 with autograd through positions and cell and
cutoffs derived from the tolerance; `0.5 q^T E_PBC q` must equal `LatentEwald.energy(q)`
(LES + background) for random neutral and net-charged `q` — that is how "keep the verified
full-sum Ewald" is honoured. The LES `sigma` ↔ `r_g` relation and the diagonal convention
(`E_PBC_ii` includes the Gaussian self term, so `K_LR_ii = E_PBC_ii - 1/(sqrt(pi) r_g)`) are
verified numerically, not assumed (Phase 0 tests).

**D4 — batching by orbital count (2026-09-06).** The SCF's `eigh` is `[B, n_orb, n_orb]`;
316- and 636-orbital frames do not batch. Batches are grouped by atom count in the loader
(the old `--defect_size_grouped_batches` mechanism), registered now because the loader is
Phase 0.

**D5 — the neutral null is a control-flow short-circuit (2026-09-06).** At `S = S_ref`
`forward` never enters the head branch, so the base outputs are returned bit-identically
(adding an exact zero can change the last bit).

**D6 — base wiring (2026-09-06).** The frozen neutral base is
`/home/alex/runs/aprime_prod/aprime_prod.model` (the `--defect_base_init` of the last v8
wave): a `MACEDefect` object with `spectral_head = False`, `r_max = 5.0 Å`, two interactions,
block-0 irreps `128x0e+128x1o`, species [Cl, Cs, Pb]. `E_base`, `F_base` and the stress are
taken from `ScaleShiftMACE.forward(base, data)` (the parent class's forward: measured equal
to `base(data)` at `S_ref` to 1e-16), with every base parameter frozen; `out["node_feats"]`
`[N, 640]` is block 0 (`[:, :128]` scalars, `[:, 128:512]` the 128 `l = 1` vectors as
`[N, 128, 3]`) followed by block 1, attached to positions and cell. The head's neighbour
list is built in the data pipeline at the head's `r_cut` (registered; the old head used 10 Å
and the vacancy-spanning Pb–Pb pair is 4.8–7.1 Å, so the base's 5 Å list cannot serve) and
the trunk is fed the `≤ r_max` subset, as the old model did. `cf_base_f0..f2` are the
k-fold bases for the out-of-fold `s0` of §6 (Phase 2).

**D7 — the rank-2 descriptor (2026-09-06).** `Q_i = sum_j w(r_ij)(rhat rhat^T - I/3)` with
`w = (1 - (r/q_cut)^6)^2`, `q_cut = 4.5 Å` (registered default). A traceless rank-2 tensor is
even under inversion, so `Q_i` vanishes at sites of cubic symmetry (ideal Pb, Cs), not at
every centrosymmetric site as the plan's remark says: on the ideal Cl site (D4h) it is the
uniaxial quadrupole along the Pb–Cl–Pb axis, `2w(2.8) diag(-1/3, -1/3, 2/3)`, which is the
crystal field that splits Cl `p_sigma` from `p_pi` — the sign Arm 1 (iii) tests. The formula
is what is implemented; the remark is corrected here. The rank-1 term is the one that
vanishes at centrosymmetric sites.

**P0.4 at production size (2026-09-06).** On a 159-atom training frame with a random `dq`
of net +1 (energy 496 eV): `eta` = 1.5 / 2.5 / 4.0 Å against the default 3.14 Å gives
|ΔE| ≤ 2.6e-11, |ΔF| ≤ 2e-12, |Δstress| ≤ 5.2e-10 / 6.9e-11 / 2.3e-11 eV. The stress
figure at `eta = 1.5` is float64 rounding at the 500-eV test scale (1e-12 relative); at
physical `dq` (|Q| = 1, energies of a few eV) the absolute figure is 1e-11. Cost 0.1–0.4 s
per matrix on CPU at 160 atoms, 0.34 s on the local GPU.

**D8 — the static pristine cell for the gap regulariser (2026-09-06, after C2).** The
dataset has no relaxed pristine frame (every `ideal` frame is a thermal snapshot) and no new
DFT is allowed. The static pristine cell is therefore the label-free time-average of the
544 pristine frames — the registered site-median lattice symmetrised over the group it
generates (Pnma × the 2×2×1 supercell translations, 32 operations; built in the v8 programme
as `defect_composition.mean_pristine_lattice`, cell 16.035 × 16.076 × 11.395 Å, Cl rms
symmetrisation shift 0.10 Å). Its construction is ported into `dscc/` before the deletion
sweep (Phase 2 item P2.2a). The thermal ensemble-mean gap over pristine frames is reported
as a diagnostic. If the user can supply the paper's relaxed bulk structure, it replaces this
without other change.

**Flanking-Pb placement fraction (v4.1 pre-condition for any regime-A run; computed
2026-09-06 evening, `scratchpad/flanking_pb.py`).** Flanking Pb identified label-free as the Pb
whose sixth-nearest Cl is beyond 4.0 Å (a first shell of five): exactly 2 per frame on 1957 of
1985 79-atom vacancy frames (1 on 27, 3 on 1). Their real first-shell bonds beyond `r_d1 =
3.2 Å` against the other Pb's: **V_Cl⁰ 79 at.: 15.4 % vs 4.3 % (×3.6); 159 at.: 5.3 % vs 0.3 %
(×21)** — the expected enhancement, on the neutral frames (where `dq = 0`); **V_Cl⁺ 79 at.:
2.9 % vs 3.2 % (×0.9); 159 at.: 0 % vs 0.2 %** — no enhancement on the charged frames, where
`dq` sits (the flanking Pb of V_Cl⁺ move away from the vacancy and their remaining bonds are
not stretched). Either way the fraction is 30× the 1e-3 floor; the ruling stands.

**D9 — kernel regime (v4.1).** `KernelConfig.regime` defaults to `"B"`; regime A is
constructed only explicitly for the ablation arm. The flanking-Pb restricted placement
fraction is computed before any regime-A run (see the note under P0.5).

**D10 — the SCF solver is a damped Newton on the exact response Jacobian (2026-09-06
evening).** On the real 79-atom V_Cl⁺ frames at the initialised head, Anderson mixing
(0.3 / 6) converged on some frames in 30–50 iterations and not at all on others within
100 (residual oscillating at 0.05–0.17 e): the map `dq -> dq_new` is strongly non-linear
there (see the finding below). The plan's "broyden_or_anderson" is replaced by its exact
limit: `hole_response` gives `M = d dq_new / dV` in closed form from the eigenbasis
(Daleckii-Krein divided differences with the fixed-N correction, over the active levels
only — `O(N_act n_orb N²)`, FD-verified to 1e-6), the fixed-point Jacobian is `J = M Gamma`,
and each step solves `(I - J) delta = r` with backtracking on the unmixed residual. On the
same two frames: 6 and 8 iterations (Anderson 34/44 or failing). Anderson stays available
(`ScfOptions.method`). The Anderson least squares is now the Tikhonov-regularised normal
equations (a full-rank QR driver on CUDA returned garbage on nearly collinear residual
differences — the solve must not depend on the device). Convergence criteria unchanged
(unmixed residual `tol_q` and `|ΔJ| < tol_E`).

**Finding (2026-09-06 evening): the initialised `H0` has no bound vacancy state.** On the
real frames, Harrison-initialised `H0` gives the pristine cell a 2.400 eV gap (−9.93 →
−7.53 eV). On the V_Cl frames the reference majority electron (205th) sits in a
near-degenerate manifold at the conduction-band edge (−7.532, −7.529, −7.502, −7.45 eV;
gap(up) 0.003–0.045 eV): the vacancy-spanning Pb–Pb `pp` hopping at 5–7 Å does not split a
dimer state off the CBM at initialisation (the Arm-1 question). Consequences: the
`Phi = 0` hole is delocalised over the Pb sublattice (N_eff 17–18); the localising
feedback on a near-degenerate manifold produces **distinct fixed points** on some frames
(frame 2: max|dq| 0.287 by Newton vs 0.402 by Anderson on CPU, 0.290 on CUDA — different
converged solutions), i.e. the single-valuedness gate of plan section 5 fails at
initialisation on real frames for a physical reason. A full-set run with Newton is in
progress (`scratchpad/stability_probe.py cpu`); ruling needed on whether the Phase-1
single-valuedness gate is read at initialisation or on the trained model (C5).

**D11 — continuation from Φ = 0 (v4.2, C5).** With Φ on, every per-frame solve starts from
the Φ = 0 two-fillings solution and ramps `Gamma` and `W` together from zero to their full
values in `continuation_steps` (registered, 4) warm-started Newton solves, the last of which
is the production solution (implicit differentiation applies to it alone). The root-rule
harness keeps its zero-coupling variant (`lambda_dir = U_eff = W = 0`, `K_LR` on) as a
validation initialisation.

**Registered for the precondition (v4.2, before Arm-1 results are opened):** `Delta_c = 10
sigma_s = 0.5 eV`; `N_loc = 4` (default: a Pb-dimer state has `N_eff` 2–3; to confirm);
fraction 0.95; `s_max = 2` for the Route B′ scale; training-time single-valuedness subsample
and ceiling: to register before Arm 2+3.

**Arm-1 criterion (ii), as implemented, is partly geometric (found 2026-09-06 21:20, before
results are opened).** With a per-species `b`, the ratio "flanking-Pb `|bQ|` over its bulk
spread" is `|Q_flank| / std|Q_bulk|` — `b` cancels — and equals 11.9 on every full run (a
property of the vacancy geometry: the flanking Pb has a strongly anisotropic environment).
The discriminating content of (ii) is therefore the control-loss test alone (the scalar-only
control loses ≥ 30 % of the level-vs-bond slope). Flagged to the user; the threshold text is
not changed here.

**Mid-training monitor (2026-09-06 21:20, epochs 3–7; not the arm result).** On 48
neutral-vacancy frames the initialised head has `N_eff` 2.0, separation 283 meV,
precondition 2 %; after 3–7 epochs every run has `N_eff` 1.7–1.8, separation 750–930 meV,
precondition 85–92 % (full) / 85–88 % (control), level-vs-bond slope 280–340 meV/Å (control
within 6 % of full so far), Cl splitting −11 to −18 meV (registered sign), no saturation,
`pp` modulation 0.91–0.98 (no stop). Held-out force RMSE (24-frame subsample, 79 atoms):
full 39–46 meV/Å, control 43–46 meV/Å (initialised 96). The earlier "no bound state at
initialisation" finding was on the V_Cl⁺ geometries (flanking Pb far apart); on the
neutral-vacancy geometries the initial `H0` already splits a dimer state 283 meV off the
CBM, which is why the precondition is defined on those frames.

**Coupled-path cost (2026-09-07, CPU 8 threads, 4 real frames, trained `H0`):** per-graph
2.9 s/frame; batched solver 2.35 s/frame (forward 0.65 s/frame: 16–17 Newton iterations over
the 4-step continuation, one batched `eigh` per iteration; the training backward — implicit
derivative, divided-difference backward through the fills, block-0 double backward — is
60 % of the step). With the untrained `H0` (no bound state) the same path took 25 s/frame,
the solver backtracking through the degenerate manifold — the bound-state precondition is
also a throughput precondition.

**Base checkpoint re-saved (2026-09-07):** `/home/alex/runs/aprime_prod/aprime_prod_base.pt` is
the frozen base as a plain `ScaleShiftMACE` (same hyperparameters, `use_reduced_cg=False`),
equal to the `MACEDefect` object's parent forward to 0.0 (energy) / 1e-16 (forces) on real
frames; it is what survives the deletion sweep (the old pickle needs the deleted modules).

**Phase-4 gate results on the Arm-1 winner `full_s0` (2026-09-07 00:20, before the Arm-1
decision is opened; Φ = 0, Route A).** Dense vs sparse (frontier window `|Q| + 8` states):
real 79-atom frames ΔE 2e-10 meV, ΔF 2e-11 meV/Å, Δdq 1e-14; real 159-atom frames ΔE
1e-9 meV, ΔF ≤ 9e-10 meV/Å (window 18, certified tail ≤ 5e-12 e); static-cell ladder
80/159/319/639 atoms ΔF ≤ 4e-13 eV/Å on every cell (window 9–36). Sparse costs 2–4× the
dense at these sizes (the dense regime), as expected. `E(+1) − E(0)` on the ladder:
7.994 / 7.984 / 7.923 / 7.916 eV (the 2×2×1 slab has α = 1.56 against 2.72 for the
near-cubic cells — the Madelung coefficient is now computed per cell shape and the 1/L fit
restricted to one shape class); with Φ = 0 the head carries no electrostatics, so the
1/L test is read on the coupled models.

**Arm-2+3 entry gates on the trained Arm-1 winners (2026-09-07 00:40; `full_s0..s3`,
before the Arm-1 decision is opened).**
* Bound-state precondition (`Delta_c` 0.5 eV, `N_loc` 4, ≥ 95 % of the 1191 neutral-vacancy
  frames): **passes on five of six** — 96.1 / 96.3 / 97.2 / 96.8 / 95.5 % (s0–s4), HOMO
  separation p50 782–876 meV, `N_eff` p50 2.2–2.5; **`full_s5` fails at 92.9 %** (separation
  p50 768 meV). Each Arm-2+3 seed starts from its own Arm-1 `H0`, so seed 5's coupled arms
  do not open: Arm 2+3 runs with seeds 0–4 (recorded deviation from the six-seed protocol;
  `tau_noise` uses five seeds).
* Local-neutrality ladder gate (cubic CsPbCl3 ladder, fixed unit carrier on a flanking Pb,
  the model's `q0` pattern vs the species pattern): **fails on all four** — the `1/L` slope of
  `Phi_cc + E_SF` is 53–65 % off Madelung on the 2/3/4 ladder and still 51 % on the
  4/5/6 ladder (22–34 Å); the species pattern (centred/uncentred) fails too (negative
  test passes). Diagnosis (split against the cubic pristine cell's own `q0`): `Phi_cc` alone
  carries the Madelung slope; the missing ion's Mulliken charge (+0.36–0.46) is compensated by
  the reference fill only over ~10 Å (net compensation −0.03 within 5 Å, −0.12…−0.45 within
  8 Å, complete at 12 Å; `R_eff` 8–11 Å) — a neutral but extended distribution whose
  interaction with the carrier's images is not in the `1/L` regime at these cells and differs
  by ~0.1 eV between 79- and 159-atom-sized cells, an error no `C_Q` absorbs. **Route B′
  arms are therefore not opened** (v4.2: "re-run on the trained Arm-1 `H0` before any Route
  B′ arm opens"); Arm 2+3 runs Route A: 5 configurations × 6 seeds, plus the regime-A
  ablation (route A, 6 seeds). Route B′ stays available if a later `H0` compacts the
  compensation (or a compact-pattern variant is ruled).

**Coupled-path cost on the GPU (2026-09-07 00:34, A4000, `full` coupling from the Arm-1
`full_s0` H0, 48 frames + 16-frame single-valuedness check + 12 held frames in 63 s):**
≈ 0.9 s/frame → ~12 min per 785-frame epoch alone; three per b3 GPU ≈ 30 min/epoch →
30 epochs ≈ 15 h per run, 36 Route-A runs over 11 slots ≈ 2.5 days. Observed at the same
time: the per-epoch single-valuedness check failed on **6 of 16** frames (continuation vs
zero start) with the trained `H0` at the initial coupling — the registered ceiling is 0.10;
Arm 2+3 will read this fraction per epoch.

**Arm 1 — preliminary reading (2026-09-07 01:00, 11 of 12 runs; `ctrl_s5` still training;
the decision is opened on the final twelve-run report).** Held-out charged force RMSE, full
`H0`: 41–47 meV/Å (s0–s5), scalar-only control: ~50–56 (initialised head 96). Criteria: (i)
`pp` stop fraction 0/6 — pass; (ii) flanking-Pb tensor ratio 10.3 (geometric, see the note
above) but the control loses only **10 %** of the level-vs-bond slope (threshold ≥ 30 %) —
**fail**; (iii) Cl `ε_σ − ε_π` −40 to −155 meV, negative on every seed — pass; (iv) `N_eff`
seed spread 0.094 (full) vs 0.259 (control), halved at better force quality, no saturation —
pass. Routing: 3 of 4 → **D (conditional)**: Arm 2+3 opens on the full `H0` winners with the
(ii) outcome recorded — the scalar-only head reproduces most of the level-vs-bond response,
so the directional block's contribution is in the force quality and the localisation
stability, not in that slope. Thermal ensemble-mean gap of the pristine frames 2.30–2.44 eV
(static cell held at 2.40).

**Arm 1 — FINAL reading (2026-09-07 01:47, all twelve runs, final-epoch checkpoints,
`~/runs/dscc/arm1_report.json`; supersedes the preliminary reading).** Per seed — held-out
charged force RMSE (meV/Å) | `N_eff` p50 | HOMO separation p50 (eV) | report precondition
fraction (300 frames) | level-vs-bond slope | Cl `ε_σ − ε_π` (meV):

| run | RMSE | `N_eff` | sep. | precond. | slope | Cl split |
|---|---|---|---|---|---|---|
| full s0 | 43.0 | 2.50 | 0.86 | 0.987 | 0.342 | −12 |
| full s1 | 43.6 | 2.29 | 0.84 | 0.993 | 0.198 | −75 |
| full s2 | 42.5 | 2.38 | 0.89 | 0.993 | 0.338 | −155 |
| full s3 | 40.9 | 2.22 | 0.91 | 0.993 | 0.282 | −92 |
| full s4 | 43.6 | 2.31 | 0.81 | 0.990 | 0.213 | −40 |
| full s5 | 49.0 | 1.93 | 0.85 | 0.980 | 0.129 | −112 |
| ctrl s0 | 50.0 | 1.81 | 0.68 | 0.940 | 0.259 | 0 |
| ctrl s1 | 74.7 | 1.95 | 0.51 | 0.533 | 0.087 | 0 |
| ctrl s2 | 67.2 | 2.49 | 0.10 | 0.000 | 0.103 | 0 |
| ctrl s3 | 39.3 | 2.18 | 0.45 | 0.283 | 0.293 | 0 |
| ctrl s4 | 43.7 | 2.02 | 0.53 | 0.640 | 0.247 | 0 |
| ctrl s5 | 45.5 | 1.95 | 0.45 | 0.323 | 0.223 | 0 |

(The report's precondition column is a diagnostic over the first 300 neutral-vacancy frames; the registered entry gate is `dscc_arm1_gates.py` over all 1191 neutral-vacancy frames, the figures in the entry-gate entry above — seed 5 at 92.9 % — and it is what selects the seeds of any Arm 2+3.) Medians: full 43.3, control 47.7 meV/Å. Shell RMSE (0–2 / 2–4 / 4–6 / 6–8 / > 8 Å from the
vacancy), full s0: 99 / 77 / 55 / 34 / 33; control s0: 111 / 87 / 61 / 47 / 32 meV/Å — the
directional block's gain sits inside 8 Å; beyond 8 Å both heads are at the base's floor.
Criteria (registered thresholds): (i) `pp` stop fraction 0/6 — **pass**; (ii) flanking-Pb
tensor ratio 10.3 (> 3) but the control loses only **5 %** of the level-vs-bond slope
(median 0.247 vs 0.234; threshold ≥ 30 %) — **fail**; (iii) Cl splitting negative on all six
seeds — **pass**; (iv) `N_eff` seed spread full **0.175** vs control 0.217 (needs ≤ 0.108) at
better force quality (43.3 ≤ 47.7) and no saturation — **fail on the spread** (full s5 at
`N_eff` 1.93 widens the full spread; control s2, whose HOMO is not separated, narrows the
control's). **Routing: C** — (ii) fails, so the covariance/sign tests are repeated once; stop
if the repeat fails (plan §7, table of the v8.1 addendum §Stages 2 and 3).

*Code correction, recorded as an error:* `arm1.decide` routed by the pass count (4 → A,
3 → D, else C/B), which sends a lone (ii) or (iv) failure to D; the table sends any (ii)/(iii)
failure to C, a (iv) failure with (ii)+(iii) passing to B, and only a lone (i) failure to D.
So the preliminary "3 of 4 → D" above was a mis-route (under the table it was already C);
corrected to the table verbatim with a test over all 16 verdict combinations
(`test_arm1.test_decide_routing_table_exhaustive`); the saved report keeps the old field as
`decision_pass_count_code`. The final decision is C under both.

*Watch item outcome — the loss spike recurred.* Final-epoch spikes: ctrl s1 (epoch 59, loss
9.1e-5 vs its median 2.2e-5; held-out 48.7 → 74.7 meV/Å between the epoch-54 and epoch-59
evaluations), ctrl s2 (epoch 56; 52.8 → 67.2), ctrl s0 (epoch 56; 47.2 → 50.0), full s5
(epochs 48 and 57; 45.6 → 49.0); every run but full s2 shows at least one loss spike > 3× its
median after epoch 20 at constant lr 2e-3. **Proposed for the user, not applied:** a
registered decay of the learning rate to 2e-4 over the last 20 epochs (or reading an
epoch-averaged final checkpoint). The repeat runs the identical protocol (constant lr 2e-3,
60 epochs, final checkpoint read): a repeat with a changed protocol is not a repeat.

**Registered before the repeat is launched (2026-09-07 01:50):** the route-C repeat is both
arms on fresh seeds 6–11 (outer fold = seed mod 4), the identical protocol, the same
report script and thresholds; the decision is read on the repeat's own six seeds with the
corrected table; the combined twelve-seed reading is reported for information only. If (ii)
or (iii) fails on the repeat: **stop** — Arm 2+3 stays closed and the outcome goes to the
user. If both pass: route by the table on the repeat (A/D: Arm 2+3 opens on the repeat's
full-`H0` seeds that pass the bound-state precondition; B: stop and investigate
identifiability). Identical-code check before launch: one epoch of the full s0 configuration
on the launch commit (`5bd1683`) reproduces the archived epoch-0 loss exactly (6.8488e-05,
force 5.1701e-05; the training path is untouched by the routing fix and the merged
inference backward). **Launched 2026-09-07 01:54 (b3, GPUs 4/5/7, three per GPU:
`full/ctrl_s6`, `full/ctrl_s7`, `full/ctrl_s8`, `full/ctrl_s9`, `full_s10`) and 02:00
(local A4000, two at a time from a snapshot of the launch commit at
`~/runs/dscc_src_arm1r`: `ctrl_s10`, `full_s11`, then `ctrl_s11`).** Queue logs
`~/runs/queue_arm1r_{b3,local}.log`; run logs `~/runs/dscc_arm1_*_s{6..11}.log`.

**Arm 1 — route-C repeat, FINAL reading (2026-09-07 05:27; seeds 6–11, identical protocol,
final-epoch checkpoints; `~/runs/dscc/arm1_repeat_report.json`, gates
`arm1_repeat_gates.json`).** Same columns as above (the entry-gate column here is the
REGISTERED precondition over all 1191 neutral-vacancy frames):

| run | RMSE | `N_eff` | sep. | entry gate | slope | Cl split |
|---|---|---|---|---|---|---|
| full s6 | 55.2 | 2.00 | 1.04 | 0.978 pass | 0.164 | −71 |
| full s7 | 51.4 | 1.83 | 0.88 | 0.959 pass | 0.161 | −108 |
| full s8 | 47.7 | 2.44 | 0.76 | 0.941 fail | 0.077 | −103 |
| full s9 | 42.0 | 2.41 | 0.86 | 0.964 pass | 0.272 | −114 |
| full s10 | 44.2 | 2.29 | 0.93 | 0.974 pass | 0.249 | −46 |
| full s11 | 51.3 | 1.77 | 0.81 | 0.929 fail | 0.137 | −78 |
| ctrl s6 | 87.9 | 6.13 | 0.08 | — | 0.114 | 0 |
| ctrl s7 | 92.4 | 5.85 | 0.10 | — | −0.020 | 0 |
| ctrl s8 | 42.5 | 2.19 | 0.36 | — | 0.173 | 0 |
| ctrl s9 | 52.1 | 1.89 | 0.29 | — | 0.126 | 0 |
| ctrl s10 | 44.0 | 2.01 | 0.44 | — | 0.247 | 0 |
| ctrl s11 | 43.5 | 1.94 | 0.40 | — | 0.217 | 0 |

Medians: full 49.5, control 48.0 meV/Å. Criteria: (i) 0/6 stops — **pass**; (ii) tensor
ratio 10.3 but the control loses **8.0 %** of the level-vs-bond slope (medians 0.162 vs
0.149; ≥ 30 % needed) — **fail**; (iii) Cl splitting negative on all six — **pass**; (iv)
`N_eff` spread full 0.269 vs control 1.881 (halved) but the full head's median force RMSE
49.5 > the control's 48.0 — **fail on force quality**; no saturation. **Routing: C for the
second time. The registered rule applies: the repeat fails (ii) → STOP.** Arm 2+3 stays
closed; Arm 4 and the Phase-4 ladder on a coupled winner do not open; the outcome is
escalated to the user (item C8 in §1).

*What the two rounds say.* The scalar-only control reproduces the modelled level-vs-bond
slope (loses 5 % in round one, 8 % in the repeat, 4.9 % over all twelve seeds), so the
premise of criterion (ii) — that the response lives in the directional block — is not met
by this `H0` on this data; the seed spread of the slope (full 0.08–0.34, control −0.02–0.29)
dwarfs the medians' difference in both rounds. What the block does carry is the bound state:
every full seed keeps a separated vacancy level (HOMO separation p50 0.76–1.04 eV, registered
precondition 0.93–0.98) while the scalar-only control does not (separation 0.08–0.68 eV,
report precondition 0.00–0.94; repeat seeds ctrl s6/s7 diverged to `N_eff` ≈ 6 with 88–92
meV/Å) — but criterion (iv) as registered reads the `N_eff` seed spread at ≤ force RMSE,
and in the repeat the force medians went the other way (full s6/s7/s11 at 51–55 meV/Å
after mid-training spikes). The far-field readout is unchanged: beyond 8 Å both heads sit at
the base's floor (32–35 meV/Å); the full head's gain, where present, is inside 8 Å.

*Loss spikes (constant lr 2e-3, final checkpoint read):* ctrl s6 diverged at epoch 57 (loss
2.6e-3 vs its median 1.5e-5; held-out 43.1 → 87.9 meV/Å), ctrl s7 (57.9 → 92.4 at the final
evaluation), full s6 (8.6e-4 at epoch 38; 55–63 after), full s7 (9.9e-4 at epoch 52; 43.4 →
65.5 → 51.4). The final-checkpoint read is noisy at the ±10 meV/Å level — larger than the
full-vs-control force difference either round measures. The LR-schedule proposal above
stands; it was not applied (the rule says stop, not a third round).

*Combined twenty-four-run reading (information only, `arm1_combined_report.json`):* (ii) control loses 4.9 % (fail), (iv) N_eff spread full 0.239 vs control 1.481 at force medians 43.9 vs 47.7 meV/Å (pass), (i) and (iii) pass — route C.

*Not opened, recorded for completeness:* the repeat's entry gate passes on s6, s7, s9, s10
(95.9–97.8 %) and fails on s8 (94.1 %) and s11 (92.9 %).

**D12 — Arm 2+3 outcome and the two contested gates (2026-09-08).** Campaign of 36
runs complete (24 regime-B coupled runs on b3 at four per GPU on `c8382f0`, the six Φ = 0
and six regime-A ablation runs on the A4000; no run failed after the fifth relaunch; the
Φ = 0 arm spans `aaff845`/`997e53a` whose executed path is identical). Held-out charged
force RMSE, medians over the six seeds (meV/Å), with the six-seed values, the 159-atom
shape-slope error (median over four seeds: seeds 0 and 2 have no shape reading because their `C_Q`
window admits one and two 159-atom frames, below the three a slope needs), `N_eff` p50 median (seed spread), the worst-epoch (≥ 1) and last-epoch
single-valuedness failing fractions, `f_SR` median, trained couplings and loss spikes:

| arm | force median | seeds | shape err | `N_eff` (spread) | sv worst / last | `f_SR` | `lambda_dir` | `U_eff` Cl / Cs / Pb (eV) | spikes |
|---|---|---|---|---|---|---|---|---|---|
| Φ = 0 (**selected**) | **39.9** | 36.8 / 41.2 / 40.0 / 39.1 / 41.6 / 39.8 | 0.042 | 3.56 (0.29) | — | — | — | — | 0–2 |
| LR-only | 41.7 | 48.6 / 40.9 / 42.2 / 37.2 / 55.7 / 41.1 | 0.037 | 3.33 (0.32) | 0.00 / 0.00 | 1.61 | 0 | 0 | 0–5 |
| LR + U | 41.3 | 42.4 / 39.4 / 41.2 / 41.8 / 39.9 / 41.3 | 0.025 | 3.71 (0.14) | 0.08–0.18 / 0.00 | 1.65 | 0 | 0.27–0.68 / 0.08–0.11 / 0.03–0.18 | 0–3 |
| full | 42.0 | 39.2 / 42.5 / 44.9 / 39.3 / 41.4 / 47.7 | 0.040 | 3.61 (0.35) | 0.08–0.26 / 0.00 | 1.61 | 0.004–0.018 | 0.20–0.98 / 0.07–0.10 / 0.04–0.18 | 0–4 |
| λ = 1 fixed | 43.3 | 40.3 / 41.7 / 47.2 / 42.1 / 44.4 / 49.7 | 0.080 | 3.48 (0.88) | 0.10–0.36 / 0.00 | 1.63 (s6 3.08) | 1 | 0.22–0.33 / 0.08–0.09 / 0.06–0.17 | 1–5 |
| regime A, full (ablation) | 40.9 | 39.1 / 43.7 / 41.1 / 36.8 / 51.8 / 40.7 | 0.020 | 3.58 (0.34) | 0.00–0.08 / 0.00 | 1.19 | 0.019–0.063 | 2.3–8.9 / 0.09–0.10 / 0.16–0.34 | 0–3 |

Initial couplings: `lambda_dir` 0.05, `U_eff` 0.71 / 0.12 / 1.36 eV. `tau_noise` (Φ = 0 seed
spread) 1.7 meV/Å on forces, 0.018 on the shape slope; `tau_phys` 3 meV/Å / 0.015; every
run converged on 100 % of training frames within `n_max`; `s` = 1 throughout (Route A).
Force RMSE by shell (0–2 / 2–4 / 4–6 / 6–8 / > 8 Å, medians): Φ = 0 84 / 68 / 50 / 32 / 32,
LR + U 84 / 72 / 52 / 33 / 32, full 93 / 74 / 54 / 34 / 32, λ = 1 90 / 74 / 54 / 36 / 33 — full and
λ = 1 are worse on the median inside 2 Å but within their seed spreads (0–2 Å seed std:
Φ = 0 6, LR + U 11, λ = 1 15, full 26, LR-only 39 meV/Å); LR + U equals Φ = 0 there; every
arm matches Φ = 0 beyond 6 Å. The base alone reads
53.2 meV/Å on the fold-0 charged held-out set; the `H0` head takes it to 36.8 (Φ = 0, s0);
the coupling adds nothing on top.

*Selection (v4.3 rule as registered, `arm23.select`):* best by median force Φ = 0; the
report's ranking holds only Φ = 0 because every coupled arm fails a gate (`f_SR` all;
root rule full / LR + U / λ = 1; `localisation_stable` λ = 1). Counterfactuals on the same
records: `f_SR` gate removed → equivalent {Φ = 0, LR-only}; last-epoch root rule → no
change; both → equivalent {Φ = 0, LR + U, LR-only, full}, λ = 1 excluded by
`localisation_stable`; `beaten_beyond_margin` empty in every reading — λ = 1 (43.3, 3.4 above
Φ = 0, beyond the 3.0 margin) is the one arm that would be beaten, and it leaves on
`localisation_stable` before the ranking. Φ = 0 is selected in all four as the simplest equivalent configuration. (`tau_noise` is the seed standard
deviation, as v4.3 words it — "seed spread"; the §2.1 row says "standard error", which
would be 0.7 meV/Å: the margin is `tau_phys` = 3 either way.)

*Root rule (C10 (a)).* Worst-epoch reading per v4.5 ("failure fails the arm"): the
over-ceiling epochs are early — full s1 {3}, s2 {14}, s3 {1, 5}, s4 {1, 2, 3, 14}, s6 {1, 2};
LR + U s2 {2, 8}, s3 {2, 3, 5}, s4 {1, 2, 3, 5, 8, 18}; λ = 1 all seeds, s4 on 22 epochs up to
31 — none within two epochs of a logged loss spike (spikes at epochs 20–58), and every
run's final-epoch check is 0/39. Pooled over the 59 checked epochs (2301 frames per run) the failing fractions are full 0.3–2.3 %, LR + U 0.3–2.4 %, λ = 1 0.9–7.7 %, regime A ≤ 0.2 % — every run under the 10 % ceiling on that reading; the worst-of-59 reading of a 39-frame subsample is the harshest of the three (one epoch at 5 of 39 reads 12.8 % and is consistent with a true rate of 5 %). The failures are the transient from the multi-valued
initialised map (28–51 % at epoch 0, C9) that decays as the hub `U_eff(Pb)` falls from
1.36 eV to ≈ 0.1 eV; the trained maps are single-valued.

*`f_SR` (C10 (b)).* Unbounded: on frame 1616 (79 atoms, charged) with the full-s0 model the
short-range part of Σ|dq_i dq_j| K_ij is +0.41 eV and the long-range part −0.19 eV, `K_LR`
negative on all 3081 pairs (−0.97 to −0.31 eV/e²). A negative denominator term puts the
"fraction" above 1 whenever the SR part outweighs it. The registered `≤ 0.5` fails every
coupled arm identically, is passed by Φ = 0 vacuously, and does not depend on the arm's Γ
(LR-only reads 1.6). Criterion defect, ours; selection unaffected; not re-registered.

*Scientific record.* Under the registered protocol the data drove the D-SCC coupling
toward zero on the 79/159-atom V_Cl⁺ frames: `lambda_dir` ≈ 0.01 and `U_eff(Pb)` ≈ 0.1 eV
in the full arm, with no force gain at any shell (full and λ = 1 worse on the median inside
4 Å, within seed spread; forcing the short-range coupling on at full strength, λ = 1, is the
one arm worse than Φ = 0 beyond the margin); the bound state is
kept in every arm (`N_eff` 3.3–3.7) by `H0`, not by Φ. The §2.10 Route-A forecast
(couplings O(1); λ = 0 ablation extends the carrier) is not borne out in regime B (the
ablation LR + U vs full: `N_eff` 3.7 vs 3.6, forces 41.3 vs 42.0); the multi-valued-map
forecast is borne out at initialisation and in the early epochs. Whether the coupled head
has any role on this dataset is now a question for Arm 4's feedback-sign comparison or the
ladder, not for selection — C10 (d).

**D13 — C10 follow-up: the far-field diagnostic, the units of the force numbers, and the
reopening of Route B′ (2026-09-08).** *Units.* `Trainer.evaluate` reports the RMS of the
per-atom force-error vector, sqrt(Σ|ΔF|²/N_atoms); MACE's logs, the base's 11.8 / 13.1 meV/Å
and `base_rmse.py` report the per-component RMS, smaller by √3 (checked: Φ = 0 s0 on its
held-out fold reads 36.8 in the trainer's convention and 21.5 per component). Every Arm-1 /
Arm-2+3 comparison is internal and unaffected; the user's "13 on neutral frames vs ~33 beyond
8 Å, 30 in quadrature" compared the two conventions. The trainer's convention is now stated in
the code; this record uses per-component numbers.

*Far-field diagnostic (`scratchpad/farfield_diag.py`, `~/runs/dscc/farfield_diag.json`;
per-component RMS, meV/Å, by distance from the vacancy midpoint: 0–2 / 2–4 / 4–6 / 6–8 / 8–10 /
10–12 / > 12 Å; the 159-atom cell has no atom inside 2 Å of the midpoint).*

| set | frames | all | 0–2 | 2–4 | 4–6 | 6–8 | 8–10 | 10–12 | > 12 |
|---|---|---|---|---|---|---|---|---|---|
| base, charged 79 (all; out-of-sample for the base) | 1030 | 53.6 | 89.0 | 121.0 | 74.5 | 38.6 | 24.1 | 13.0 | 10.4 |
| base, charged 159 (all) | 17 | 30.0 | — | 98.6 | 58.9 | 20.7 | 16.5 | 10.0 | 5.0 |
| Φ = 0 head (s0–s3 on their folds), charged 79 held-out | 1028 | 23.0 | 48.1 | 39.1 | 28.1 | 18.7 | 20.3 | 13.5 | 10.3 |
| Φ = 0 head, charged 159 held-out | 19 | 10.0 | — | 20.5 | 14.9 | 9.3 | 10.6 | 7.2 | 4.5 |
| cross-fit bases, neutral vacancy 79, OUT-OF-FOLD floor | 1174 | 12.9 | 30.8 | 21.5 | 17.4 | 10.3 | 10.2 | 9.5 | 8.3 |
| cross-fit bases, neutral vacancy 159, out-of-fold floor | 17 | 12.0 | — | 33.2 | 19.9 | 10.8 | 9.4 | 6.9 | 4.7 |
| production base, neutral vacancy 79 (training frames included: optimistic) | 1174 | 13.1 | 22.6 | 21.0 | 17.0 | 10.9 | 10.7 | 9.7 | 8.9 |
| production base, neutral vacancy 159 | 17 | 6.5 | — | 15.5 | 11.2 | 5.6 | 5.7 | 4.4 | 3.7 |
| production base, pristine 80 | 200 | 5.8 | | | | | | | |

The cross-fit bases (`cf_base_f0..f3`, converted to plain `ScaleShiftMACE` under the old tree
as `cf_base_f{k}_plain.pt`, forces identical to 1e-16) give the honest neutral floor: 12.9 at
79 atoms — the same as the optimistic 13.1, so the base does not overfit its neutral frames.
Excess of the Φ = 0 head over that floor in quadrature: 79 atoms 17.6 (8–10 Å), 9.6 (10–12),
6.1 (> 12); 159 atoms 4.9 (8–10), 2.1 (10–12), none beyond 12 Å. The charged far field at
159 atoms is neutral-like: by the user's criterion the 79-atom excess is small-cell and
manifold extrapolation, not the Coulomb channel. No geometry is shared between the charged and
neutral frames (0 common group keys at either size), so the model-free label difference
F(Q) − F(0) cannot be read. Near field: 48 per component (83 in the trainer's convention) at
0–2 Å against a 31 floor, unmoved by any electrostatic arm (C10 open item).

*C10 readings on the Arm-2+3 records (`arm23_report_c10.json`, same runs).* Root rule on
the final model and the last ten epochs: every arm passes (λ = 1 at most 0.051); transients
(last over-ceiling epoch) full {0, 3, 14, 5, 14, 2}, LR + U {0, 0, 8, 5, 18, 0}, λ = 1 {6, 20,
13, 18, 31, 5}, regime A and LR-only none; pooled over epochs ≥ 1: full ≤ 2.3 %, LR + U ≤
2.4 %, λ = 1 ≤ 7.7 %. `f_sr_abs` medians: full 0.705, LR + U 0.694, LR-only 0.711, λ = 1 0.712,
regime A 0.462. v4.3 with the C10 gates: {Φ = 0, LR + U, LR-only, full} equivalent, Φ = 0
selected. "v4.4" (post hoc): LR-only selected, cost +1.8 meV/Å (trainer convention) vs Φ = 0.

*Route B′ reopened by the ruling.* Entry gate re-check on the Arm-2+3 Φ = 0 models: `q0`
ladder slope 51 / 53 / 61 / 51 % off Madelung (fails; Arm-1 winners 29–55 %). Engineering
before launch (`23a4639`): the pair force route extended to B′ — the `Γ_LR` geometry term
from `gamma_lr_pair_gradient` contracted with `−n_site (s q0)ᵀ`, the `q0(H0(R))` term through
the cotangent `(W, −n_site)` with `Γ_LR` detached; forces and the λ / U / `s` gradients equal
the cotangent route's to 1e-9 / 1e-8 on the toy, per-graph and batched, batched = per-graph
(`TestPairForcePath`, `test_batched_training_path_matches_per_graph[True]`). Memory on the
A4000 at four 159-atom frames: check 5.37 GB, step 4.39 GB (reserved 5.8–6.2), evaluation
3.66 GB — the Route A full-mode footprint, so four per GPU. Smoke on b3 (48 frames, two
epochs): warm-started 12/12 batches at epoch 1, `s` 1.000 → 0.983, held-out written. A
"B′ Φ = 0" arm is the Route A Φ = 0 path (`W` is built only under coupling) and is not queued:
the six Route A Φ = 0 runs are the force references. *F-SCC matched-kernel comparator:* smoke
51–59 s per 12-batch epoch (two cold absolute-charge solves per frame, 35 SCF iterations per
batch, per-graph path, the v4.5 warm starts unused — `dq_store` fills with `Dq_S − Dq_ref` that
the branch ignores, harmless) → ≈ 16 min per epoch alone, 35–45 h per run at three per GPU;
warm-starting the two solves is available "solver engineering" if a fourth seed is wanted.

*Launch 2026-09-08 15:13 (b3 clock), commit `be3c39b`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:*
`~/runs/arm23bp_queue_b3.txt` on GPUs 4 and 5 at four per GPU — B′ LR-only s{0,1,2,3,4,6},
then LR + U, full, λ = 1 (24 regime-B runs), then the regime-A B′ ablation (6, the natural
cut); `~/runs/arm4_queue_b3.txt` on GPU 7 at three per GPU — `dscc_arm4_matched_s{0,1,2}`
(`--coupling 1 --coupling_mode full --fscc matched`, λ and U learned, plan §2.9). Wave 1:
16.3 GB per B′ GPU at launch, 12.2 GB on GPU 7. Driver `scratchpad/bp_driver.sh` (fetch,
spikes, report over all `dscc_arm23_*` runs when 33 `held_final.json` exist), watcher
`bp_watch.sh`, stop `kill_bp_b3.sh`. The b3 tree is frozen until the B′ queue log shows all 30
lines launched.

## 3. Task list

Status: `todo` / `wip` / `done` / `blocked`.

### 3.0 Bookkeeping
| # | Task | Status |
|---|---|---|
| 0.1 | Cancel the v8 s14a wave; leave the v8 branch clean (`06c04c8`, 689 tests green) | done |
| 0.2 | Plan v4 saved verbatim; this tracker; memory updated | done |
| 0.3 | Report to the user: cancellation outcome, GPU 6 fault, C1–C4 | done 2026-09-06 (evening) |

### 3.1 Phase 0 — scaffold (plan §4)
| # | Task | Gate | Status |
|---|---|---|---|
| P0.1 | Species table `n0`, state adapter (D2), `N_ref`, spin split; dataset assertion `Q == cell_charge` | counts match on every frame | done — `dscc/species.py`; all 2877 train+valid frames: `Q == cell_charge`, (205,204)/(204,204)/(208,208)/(413,412)/(412,412) |
| P0.2 | Data pipeline: geometry-state groups formed before the split; strata keys; size-grouped batches (D4) | no group split across folds | done — `dscc/data.py` (`frame_meta`, `atomic_data` at `r_cut`, `split_by_group`, `assert_no_group_split`, `SizeGroupedSampler`) |
| P0.3 | `fill(H, N)`: Gaussian smearing, bisection `mu`, `P`, `F_band`, generalised entropy; matrix-function backward (batched Function over `_dk_backward`) | `dF_band/dH_ab = P_ba` to 1e-10 | done — `dscc/fill.py`; FD 1e-8, autograd exact; density response vs FD 1e-6; batched == per-frame |
| P0.4 | `E_PBC` Ewald matrix of Gaussians with derivatives (D3); LES oracle test; tiling-ladder `K_LR_ii` → `-alpha_M/L` | independent of the splitting parameter to 1e-10 eV (E, F, stress); oracle agreement | done at toy size — `dscc/ewald.py`; η-independence 1e-10 (E, F, stress), LES oracle 1e-6, ladder 1e-8, pair width convention; 159-atom gate: see log |
| P0.5 | Regime A: `K_SR` (erf, `w_dir` C2 switch), `K_LR`; placement check over identified first-shell bonds; `m_sw` | placement floors satisfied on the training set | **gate FAILED (C4, ruled v4.1)**: regime A is a reduced ablation only; flanking-Pb restricted fraction: `scratchpad/flanking_pb.py` (running) |
| P0.6 | Regime B: `K_SR` lattice sum with automatic image range; `K_LR`; `f_SR` | converged to 1e-10 eV; rewrapping-invariant | done — `K_SR + K_LR = E_PBC` to 1e-10, range-converged 1e-12, rewrapping 1e-12 |
| P0.7 | `Gamma` (both regimes, `lambda_dir`, `U_eff` bounded), `Gamma_LR` (`r_g`/`r_split` cross Ewald), `Zbar` centring, `W` | splitting-parameter independence of `Gamma_LR` | matrices done (`gamma_matrix`, `gamma_lr`, `centred_pattern`, `project_sum_rule`, `host_potential`); the bounded learnables live in the head module (P1) |
| P0.8 | `H0`: reuse `SlaterKosterH` (SK, Harrison init, decay lengths, log modulation, centred scalar onsite); rank-1 `l=1` descriptor; geometric rank-2 `Q_i`; head graph at its own `r_cut` (D6) | invariance under translation / rotation / permutation / rewrapping with covariant derivatives; `Q_i = 0` at cubic sites (D7) | done — `dscc/hamiltonian.py`, `dscc/graph.py`; rotation covariance `H' = D H D^T` 1e-10, permutation 1e-12, FD of the spectrum 1e-7; scalar-only control = block off |
| P0.9 | Serialisation and checkpoint round trip | loaded checkpoint reproduces an uncached forward bit-for-bit | done — `dscc/model.py` (`MACEDSCC`; registered numbers, `C_Q` table and record in `extra_state`); whole-object and state-dict round trips bit-identical |

### 3.2 Phase 1 — D-SCC forward and derivatives (plan §5)
| # | Task | Status |
|---|---|---|
| P1.1 | Batched SCF loop (Anderson/Broyden, unmixed residual, `tol_q`/`tol_E`/`tol_c`, `n_max` cap + flag, `rho`) | per-graph loop done — `dscc/scf.py` (`solve_dscc`: damped Newton on the exact Jacobian (D10) with Anderson fallback, unmixed residual, cap flagged, `rho`, commutator, primary vs band; implicit differentiation for training); cost at 79 atoms on 8 CPU threads ≈ 4.8 s/frame at the initialised head (many frames at the cap); **batched Φ = 0 path** (`H0.batched`, batched fills, one block-diagonal cotangent; equal to the per-graph reference to 1e-10/1e-9), **cached `E_base`/`F_base` per frame with block-0-only recomputation** (`first_block`; keys `dscc_base_*`), cached static-cell features for the gap term (2026-09-06 22:30, after the step profile: 67 % of a step was the double backward through the per-graph slices and the base's block 0, 24 % the five cotangent calls, 8 % forward). Coupled (SCF) path stays per-graph |
| P1.2 | Energy (band form vs primary functional to 1e-9 eV), HF forces, autodiff stress; neutral short-circuit (D5) | done — model `_charged_forward`: band form, HF cotangents `(H, dP)`, `(Gamma, -½ dq dqᵀ)`, strain-based stress; short-circuit bit-identical |
| P1.3 | Route B′ switch (off by default); both regimes selectable | done (v4.2) — `route_b`: `W = Γ_LR (s q0)`, `q0 = reference_charges(H0)` attached (its geometry derivative reaches the force through the `(H, dP)` cotangent = the Fréchet contraction; a detached `q0` measurably changes the force, test), one global `s = s_max σ(s_raw)` (init 1); `compensation_cloud` R_eff diagnostic; `pristine_gap` on `H0 − W` |
| P1.4 | Root-rule harness: initialisations (i)–(iii), `tol_root`, symmetry-equivalent collapse | `scf.root_rule` (zero / continuation / warm, spread, `passed`); production per-frame init = `continuation_solve` from Φ = 0 (D11); `forward(warm_start=...)` for initialisation iii; symmetry-equivalent collapse by identical observables: todo |
| P1.5 | Gates: neutral null; `sum dq = Q` to 1e-12; gauge shift; FD forces {1e-2,1e-3,1e-4} Å to 1e-4 eV/Å and six strains; invariances to 1e-10 eV; stability / single-valuedness on every frame of both charge states; Route B centred/uncentred ladder test | toy-size gates pass (`test_model.py`, `test_scf.py`): neutral null bit-identical; `sum dq = Q` 1e-12 (Newton-polished `mu`); gauge `-aQ` 1e-10; FD forces 3e-6 and strains 1e-7 (both regimes, Route B); invariances 1e-9; band vs primary 1e-9; root rule; unrolled vs envelope gradient. FD step ladder {1e-2,1e-3,1e-4} (< 1e-4 eV/Å, not worse at tighter SCF tolerance) and all six strains: pass (`test_gates.py`). **Real-data run at the initialised head** (regime B, Route A, Newton without fallback, every charged frame of train+valid, CPU): 79 at.: 900/1030 converged within n_max = 100 (iterations p50 7), 159 at.: 16/17; `sum dq = Q` to 6e-14 on every frame; |band − primary| ≤ 9e-3 (non-converged frames only); 0.95 s/frame (79), 1.8 s (159); J* 7.55 ± 0.05 eV; max|dq_i| p50 0.31. **Root rule on 20 frames: 8/20 pass** (spreads up to 0.40 e between the zero-start and continuation fixed points) — the single-valuedness gate FAILS at initialisation (no bound state, D10 finding; C5 ruled: read on trained models). With the Anderson fallback in Newton (final solver): 79 at. **1018/1030** converged, 159 at. 16/17, root rule 10/20, 0.98 s/frame — the initialised-model diagnostic of record. Route B model-level tiling ladder: needs a bound carrier (the toy cannot bind one; C6 records the kernel-level result) |

### 3.2b v4.2 items (C5–C7)
| # | Task | Status |
|---|---|---|
| V2.1 | Bound-state precondition for Arm 2+3 (`Delta_c`, `N_loc`, fraction; label-free on neutral-vacancy frames) | done — `dscc/precondition.py` (`bound_state_precondition`, per-frame separation and `N_eff`); to be run on the Arm-1 `H0` |
| V2.2 | Local-neutrality ladder gate with the species-pattern negative test | done — `ladder.local_neutrality_gate` (fixed carrier on a flanking Pb, `q0` vs species pattern centred/uncentred); kernel-level mechanism test in `test_kernels`; to be run on the trained Arm-1 `H0` |
| V2.3 | Post-hoc `C_Q` calibration (interpolation-only 159-atom frames, SE, "base support at 159 unverified" flag) | done — `dscc/calibration.py` (`profile_c_q`); the current `s0(159) = +0.08 ± 0.09` already raises the flag |
| V2.4 | Training-time single-valuedness monitoring (per-epoch subsample, registered ceiling, failing frames dropped from the step with the fraction logged) | todo — trainer |

### 3.3 Phase 2 — training protocol (plan §6)
| # | Task | Status |
|---|---|---|
| P2.1 | Energy admission per fold and size: coverage table, `s0 ± SE` (out-of-fold base), `sQ`; stored before results | module done — `dscc/admission.py` (label-free `d` = flanking-Pb distance, coverage bins, `slope_with_se`, decision, record); tables built (`defect-perovskite/dscc_admission_tables.json`, `dscc_base_residuals.json`; 4 outer folds; out-of-fold null from `cf_base_f0..f3` on their `null_oof` frames, 298/298/298/297): **no size admitted under the registered defaults — C7** |
| P2.2 | Objective (v4.2): forces at every size, no energy term; strata weights on the force loss; `L_gap` on the Hamiltonian actually filled at `dq = 0` on the static pristine cell (D8); `C_Q` outside the loss (V2.3); Route B′ has no pattern regulariser (no learnable pattern) | todo |
| P2.2a | Static pristine cell: port the symmetrised site-median lattice construction into `dscc/` (D8); thermal-mean gap diagnostic | cell done — `dscc/static_cell.py` (global anchor-site registration + species-wise assignment + iterated median + group symmetrisation); on the 544 training pristine frames: 32 operations, cell [16.035, 16.076, 11.395] Å, thermal rms Cl 0.52 / Cs 0.45 / Pb 0.22 Å, symmetrisation shift 0.19 Å, last median pass 0.022 Å; saved `defect-perovskite/static_pristine_cell.json` (the dataset directory is a symlink outside the repository) (fingerprint cc78a367e345352d). Thermal-mean gap diagnostic: todo |
| P2.2b | Trainer (`dscc/train.py`, launcher `defect-perovskite/dscc_train.py`): forces-only loss with frozen stratum totals, gap regulariser on the static cell, per-epoch single-valuedness subsample (C5), shell-resolved held-out force RMSE (0–2, 2–4, 4–6, 6–8, > 8 Å from the flanking-Pb midpoint), run record with every registered number; outer folds = cross-fit folds for neutral frames + seeded group split for charged frames; pristine frames train everywhere | code done; smoke run: see log |
| P2.3 | Loss-path audit (v4.2: no energy term — the audit reduces to the force path: masks, units, weights, restore; `C_Q` post hoc never sees validation/test/tiling frames) | todo |
| P2.4 | Fixed protocol frozen: splits, strata, weights, balance, six seeds, metrics, all §6/§7 thresholds | todo |
| P2.5 | Leak readout `d(J* + C_Q)/dd` per size | todo |

### 3.4 Phase 3 — arms (plan §7)
| # | Task | Status |
|---|---|---|
| P3.1 | Arm 1: full `H0` vs scalar-only control, `Phi = 0`, Route A; thresholds (i)–(iv) registered before opening | diagnostics done — `dscc/arm1.py` (`run_diagnostics`, `summarise`, `decide`: stop fraction from the flanking-pair `pp_sigma` modulation at its bound on > 50 % of frames; tensor ratio > 3; control loses ≥ 30 % of the level-vs-bond slope; Cl splitting sign registered NEGATIVE (`eps_sigma − eps_pi < 0`, the axial cation field lowers p_sigma); participation spread halves at ≤ force RMSE without saturation (`|tanh| ≥ 0.98` on < 50 % of coefficients)); routing A/D/C/B. Wave 1 first launched 2026-09-06 20:30 (commit 96d32b1; ~500 s/epoch shared, 295 s alone) and **cancelled at epoch 4–7 on the user's instruction after the step profile** (per-graph slicing and the full base forward: 7.46 s per 4-frame step on CPU → 1.14 s with the batched Φ = 0 path and the cached base, 6.5×). **Resubmitted 2026-09-06 22:40 on commit e03cad2**: `dscc_arm1_{full,ctrl}_s{0,1,2}` on b3 GPUs 4/5/7 (two per GPU), `full_s3` local, wave 2 (`ctrl_s3`, `full/ctrl_s4`, `full/ctrl_s5`) queued behind. Partial old runs moved to `~/runs/dscc_old/` and not read. Results are not opened until all twelve finish and the thresholds above stand as registered. Fast-path check: the resubmitted runs reproduce the cancelled ones epoch by epoch (same seeds: relative loss differences 1e-14 → 1e-3 over epochs 0–5, then the usual chaotic divergence at unchanged loss level). Watch item: the cancelled local `full_s3` showed a loss spike at epoch 14 (1.5e-5 → 6.0e-5, held 54 → 87 meV/Å) at lr 2e-3 — if it recurs, a registered LR schedule is proposed before Arm 2+3. **Final twelve-run reading 2026-09-07: route C** (criteria (ii) and (iv) fail; §2.2 final-reading note); the registered repeat on seeds 6–11 launched 2026-09-07 (same names `dscc_arm1_{full,ctrl}_s{6..11}`). **Repeat read 05:27: route C again ((ii) 8 %, (iv) on force quality) → STOPPED by the registered rule; C8 open** |
| P3.2 | Arm 2+3 (v4.1 layout, v4.3 protocol): regime B — route A × coupling {Φ = 0; LR-only; LR+U; full; λ = 1 fixed} = 5 configurations × 6 seeds {0, 1, 2, 3, 4, 6} = 30 runs, plus the regime-A ablation (full coupling, route A, 6 seeds), excluded from selection; Route B′ closed by its ladder gate; thresholds registered; forecasts §2.10 checked | **launched 2026-09-07 08:50** (v4.3 ruling; nine seeds queued, cut to six on the user's instruction five minutes later by truncating the b3 queue in place — no run of seeds 7/9/10 had started): first layout (three per GPU on b3) **aborted at 09:10: a coupled training step needs ≈ 13 GB (implicit-function backward through four fills with `create_graph`), three per 24 GB GPU OOM'd** — all crashed runs cleared, nothing read. **Relaunched 09:20:** b3 GPUs 4/5/7 at ONE coupled run per GPU (`~/runs/arm23_queue_b3.txt`: the 24 regime-B coupled runs, seed order), local A4000 one run at a time from a snapshot of HEAD (`~/runs/arm23_queue_local.txt`: the 6 Φ = 0 runs, then the 6 regime-A ablation runs) after the entry-gate jobs finish; 36 runs × 60 epochs; the Φ = 0 arm trains at ≈ 94 s/epoch. **Second abort 09:25: one coupled run alone exceeds 24 GB** (21.9 GB allocated at the head-force backward) — not a per-GPU count problem. Diagnosed on CPU with four real 79-atom frames (`scratchpad/mem_profile.py`): the coupled training step's `create_graph` force backward kept the second-order autograd graph of the Ewald lattice sums (K_SR, K_LR at tol 1e-16): peak 18.9 GB RSS (+16.9 GB per step), 10.9 s. **Fix (numerically identical, commit below):** the kernel terms' forces are taken from the kernels' PAIR derivatives `D_ij = dK_ij/d(r_i − r_j)` (constants of the geometry, one first-order backward on a pair-vector leaf: `ewald.pair_gradient`, `kernels.kernel_pair_gradients`) contracted with the attached charges and `lambda_dir` (`gradient_of_contraction`; Γ is linear in λ and its diagonal does not move) — no second-order graph through the lattice sums; the Hamiltonian cotangent `(H, dP)` keeps `create_graph` as in Arm 1; inference and stress keep the cotangent route (`gamma_force_mode = "autograd"` selects it for the tests). Peak 7.1 GB RSS (+4.9 GB), 5.5 s per step on CPU; **4.0 GB peak on the A4000**; forces equal the inference forces to 1e-9 and the parameter gradients equal the cotangent route's to 1e-8 relative on both regimes (`TestPairForcePath`). **Bug found by the new batched-vs-per-graph gradient test and fixed in the same commit:** the batched divided-difference backward (`legacy._dk_eigenbasis`) applied the fixed-N (moving-μ) correction to a batch only when EVERY frame's frontier was fractional (`.all()` on a per-frame test), so a batch mixing a gapped frame (f′ = 0) with a fractional one dropped the correction for both — the batched coupled force gradient disagreed with the per-frame one by orders of magnitude (λ: 47.3 vs −1.25, FD −1.249; U: 5.7 vs −0.75, FD −0.752), the energy gradient was unaffected. Now per frame (`test_fill` regression; batched = per-graph to 1e-11 on forces, energies and dq). Exposure of Arm 1 (batched Φ = 0): **none found where checked; forward quantities never affected.** On 400 of 400 sampled 79-atom charged frames at the TRAINED `H0` of seed 0 every fill has the same frontier class in every frame (state up / state down / reference down: gapped, f′ = 0; reference up: fractional), so within any such batch the per-batch test agreed with the per-frame one for each fill. Not checked: the frontier classes at the Harrison initialisation and during the first epochs, and the 159-atom frames. The bug is backward-only (the training gradient): forward values, forces and every reported Arm-1 number are unaffected regardless, and the Arm-1 decision was read on forward quantities. **Third abort 10:10 (commit `6a34c99`): the 159-atom batches still needed ≈ 12 GB** (the FIRST-order Ewald graph, kept for the energy's position derivative that training never uses). Fixed in `ba04223`: under the pair route the kernel components are computed without a graph (λ and U enter through `gamma_matrix` on detached kernels; the force from the pair derivatives) — GPU peak on the A4000 **1.66 GB at 79 atoms (2.3 s/step), 3.32 GB at 159 atoms**. The 159-atom step then took 78 s: the profile put 721 batched `eigh` calls in one step — one graph of the four hit `n_max` = 100 (residual 3.6e-3, ρ = 1) in the batched solver where the reference per-graph solver converges it in 40 iterations, because the batched solver's fallback after failed Newton damping was a plain damped step, not the graph's own Anderson step. **Fixed (commit below): the batched solver carries per-graph Anderson histories and takes the reference fallback** — iteration counts now identical to the per-graph solver on real 79- and 159-atom frames ([15, 15, 15, 16] and [17, 40, 17, 20], all converged); the 159-atom step 9.2 s, the 79-atom step 2.2 s (≈ 7.7 min per epoch alone on the A4000, ≈ 7.7 h per 60-epoch run). **Relaunched 2026-09-07 10:20 on commit `aaff845`** (every Arm-2+3 run on this commit): b3 four coupled runs per 24 GB GPU (12 slots for the 24 regime-B coupled runs, `~/runs/arm23_queue_b3.log`), local two at a time from a fresh snapshot (6 Φ = 0 then 6 regime-A ablation; three at a time put the A4000 at 15.6 of 16.4 GB). **First coupled epochs (10:41): 1246 s at four per GPU (5.2 GB per process, 21 GB per GPU), epoch-0 single-valuedness subsample 0/16 failing, 0 unconverged; ≈ 21 h per run, two b3 waves ≈ 1.8 days; the local Φ = 0 runs at 115 s/epoch**. **Fourth abort 11:00, before any evaluation: the held-out evaluation at epoch 4 would have OOM'd every wave-1 run.** `Trainer.evaluate` calls the model with `compute_force=True` and no `create_graph`, a path the pair route did not cover (it was gated on `training`), so the evaluation took the cotangent route and kept the first-order Ewald graph of the batch: measured 9.4 GB of working set on CPU and **8.10 GB peak allocated on a b3 GPU for four 159-atom frames** (3.42 GB at 79 atoms) against 3.4 GB free per GPU at four runs each (`scratchpad/eval_mem_cpu.py`, `eval_mem_b3.py`). The twelve runs were killed at epochs 1–2 and their directories cleared; nothing read. **Fix (commit `f76a6d0`): the pair route serves inference as well** — `use_pairs` no longer requires `training`, the pair force is added (detached) in the merged inference backward, F-SCC likewise; stress and Route B′ keep the cotangent route and `gamma_force_mode = "autograd"` remains the tests' reference. Evaluation peak **3.64 GB at four 159-atom frames, 1.67 GB at 79** (b3 GPU), inside the training footprint; forces equal the cotangent route's to 1e-9 on the per-graph, batched and F-SCC paths (`test_inference_forces_match_the_cotangent_route`). Same commit: the batched solver's `rho` read 1.00 for any graph that converged before the batch's last iteration (its frozen residual repeated in the shared history); each graph's tail is now cut at its own iteration count (asserted against the per-graph solver in `test_batched_training_path_matches_per_graph`); no report had read `rho` from a batched solve. **Relaunched 2026-09-07 11:03 on commit `f76a6d0`**, b3 four per GPU as before; the six local Φ = 0 runs (≈ 20 epochs in) were left running on `aaff845` — the change touches only coupled inference, which Φ = 0 never executes — and the local snapshot's `model.py` and `scf.py` were replaced from the new commit before the regime-A ablation runs (coupled) start from it. **b3 tree frozen** from this relaunch until the queue log shows all 24 runs launched (wave 2 starts from whatever is on b3 at that moment); commits continue locally. Watcher v2 (`scratchpad/arm23_watch_v2.sh`): failure grep widened to Traceback / Killed / OOM, each failure reported once with the watch continuing, live process counts on both hosts in the two-hourly heartbeat (the driver waits for 36 `held_final.json` files and would stall silently on a dead run; the queue runner refills a freed slot at once, so a dead run does not show in the live count — the per-run failure report does). **Contingency, decided in advance:** the epoch-4 evaluation (~12:50) is the first test of four per GPU with the evaluation inside the training footprint (3.64 GB allocated in a fresh process against ≈ 3.8 GB free per GPU: a fragmentation margin, not a capacity margin); if any run OOMs there, relaunch the whole wave at THREE per GPU with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (three waves, ≈ 2.7 days) and do not iterate at four. **Single-run redo:** `dscc_train.py` neither resumes nor refuses an existing `--run_dir` (it appends to `train.log` and retrains from scratch, overwriting `model.pt` / `held_final.json`), so a crashed run is redone by deleting ITS directory and `~/runs/<name>.log` only and queueing that one line (never the relaunch script's `rm -rf dscc_arm23_*`)). **v4.5 restart (C9): wave 1 on `f76a6d0` killed at 13:53 b3 time (epochs 3–5, its epoch-4 evaluation had passed at four per GPU: LR-only held-out 43 meV/Å, no OOM) and the 24 coupled runs relaunched on `997e53a` at 13:53 (b3 clock), four per GPU on 4/5/7; b3 tree frozen again until the queue log shows 24 launched.** Local: Φ = 0 s0, s1 finished and s2, s3 running on the `aaff845` snapshot; the remaining lines (Φ = 0 s4, s6; regime-A ablation × 6) queued from a fresh snapshot of `997e53a` (`~/runs/dscc_src_arm23_v45`, `~/runs/arm23_queue_local_rest.txt`), started by a detached waiter once s2/s3 write their final evaluation — the Φ = 0 arm thus spans two code versions whose executed path is identical (no coupling: no SCF, no store, no check). The report now reads the single-valuedness check at its WORST epoch (stricter than the last-epoch reading before). **Fifth abort 14:02 b3 time, eight minutes in: `full_s0` OOM'd at its first training step (GPU 4 at 23.7 of 23.5 GB; its slot was refilled by a wave-2 run at once, as the runner does).** Two causes, both measured on the A4000 with four 159-atom frames: the epoch-0 check held two batched evaluations at once (peak 3.1 GB each) and left the process's pool fragmented, and the batches that mix a checked (stored) frame with first visits took the PER-GRAPH path, which needs ≈ 0.4 GB more than the batched path (3.14 vs 2.80 GB at three frames) — together ≈ 0.5 GB more per process than the previous wave's 5.2 GB. Not the registered contingency case (the epoch-4 evaluation had passed at four per GPU on the previous wave). **Fixed in `c8382f0`:** the check frees the first solve before the second and returns the inference pool afterwards (`torch.cuda.empty_cache`), and a uniform batch mixing stored and first-visit graphs stays on the batched path (the first visits get their continuation on the sub-batch, detached, then every graph takes the batched warm solve — fixed points, forces and gradients equal the all-continuation batch's, `test_mixed_warm_batch_stays_batched_and_matches`; mixed 2.81 vs batched 2.80 GB). 120 dscc tests. **Relaunched 14:18 (b3 clock) on `c8382f0`**, four per GPU, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the queue environment as insurance; the local snapshot's dscc module refreshed to `c8382f0` before any local coupled run started (Φ = 0 s2/s3 still running on the old snapshot, unaffected). **First v4.5 epochs (b3 clock 14:39–14:48): LR-only epoch 0 = 1048 s with 47.7 fills per batch (the 39-frame check plus first-visit continuations), epoch 1 = 517 s with 13.2 fills per batch (every batch warm-started) — 2.3× the previous code's 1185 s; the full / LR+U / λ = 1 modes spend 69–76 fills per batch at epoch 0. **Full mode (s0): epoch 0 = 1461 s (69.1 fills per batch), epoch 1 = 760 s (24.2 fills, 19.3 accepted iterations per batch, all 197 batches warm-started) — 2.3–2.6× the previous code's 1700–1980 s; the epoch-1 check (warm start against the continuation, 39 frames) failed 1/39 = 2.6 %, under the ceiling, so the warm-started solutions stay on the continuation's branch. Projected: LR-only ≈ 8.8 h per run, the other modes ≈ 13 h; wave 1 ends ≈ 03:30 on 8 Sep (b3 clock), the campaign ≈ 17:00 on 8 Sep — about one day instead of 2.5.** **Finding at epoch 0 (the initialised model, the v4.2 zero-start form of the check on 39 frames):** LR-only 0/39 failing on all three seeds, but full 14, 11, 20 of 39; LR+U 13, 11, 20; λ = 1: 14, 18 — 28–51 % of the 79-atom charged frames reach a different fixed point from a zero start than from the continuation at the initial coupling (U ≈ 0.7/0.1/1.4 eV): on three such frames the zero start lands on a branch 0.05–0.16 eV LOWER in J* (e.g. frame 2510: −262.941 vs −262.781 eV), batched and per-graph paths and both dampings agreeing to 1e-15, so this is the map, not the solver. The previous wave's 0/16 readings were the LR-only runs; the full modes' epoch-0 check was never read there. **Reading applied (my interpretation, for the user to confirm or overrule):** v4.2 §5 — "the root rule and convergence gates apply to trained models; the initialised-model outcome is recorded as a diagnostic" — so epoch 0's fraction is the initialised-model diagnostic (`sv_init_fraction`, recorded per run, failing frames dropped from epoch 0 as v4.2 says) and the arm-failure ceiling is read over epochs ≥ 1, where the check is the v4.5 warm-vs-continuation form; the report computes both from the history (the running processes' own flag counted epoch 0 and is not used). Under the other reading (the ceiling from epoch 0) every arm but LR-only fails the root rule at initialisation and the campaign is decided before training. The multi-valuedness of the initialised map on a third of the frames is a scientific finding in its own right (it was 12/20 on the v4.2 root-rule diagnostic) **DONE — campaign complete 2026-09-08 13:08 (last local ablation run), report 13:20 (`~/runs/dscc/arm23_report.json`, spikes `arm23_spikes.json`, all 36 run directories fetched to `~/runs/dscc/`); no run failed after the fifth relaunch. Selected `B_A_phi0` (median 39.9 meV/Å); coupled arms equivalent within the margin or gate-failed; the coupling trained toward zero. Full numbers, gates and the two contested readings in D12; rulings requested in C10; note `defect-perovskite/ARM23_COADVISOR.md`. **C10 ruled 2026-09-08: LR-only selected post hoc under the user's "v4.4" rule (Φ = 0 the reference); Route B′ reopened over its failed entry gate — P3.2b.** |
| P3.2b | Route B′ arms (C10, 2026-09-08): regime B × route B′ × coupling {LR-only; LR + U; full; λ = 1} × 6 seeds = 24 runs, plus the regime-A B′ ablation (6); Φ = 0 reference = the Route A Φ = 0 runs; same protocol (60 epochs, lr 2e-3, final checkpoint), each seed from its Arm-1 `H0`; pair force route extended to B′; selection gates as registered with the C10 readings (far-field 4–8 Å pooled gain over the same-coupling Route A arm beyond the Φ = 0 seed spread; `s` in [0.1, 1.9]; the ladder gate awaiting C10 (e)) | **launched 2026-09-08 15:13 (b3 clock) on `be3c39b`**, GPUs 4 and 5 at four per GPU, LR-only and LR + U first, the ablation last (cut candidate); ≈ 13 h per run, three waves ≈ 40 h; D13 for the entry-gate re-check, memory, smoke; driver / watcher / stop scripts in the scratchpad |
| P3.3 | Arm 4: matched-kernel and full-kernel F-SCC comparators; decision (1)–(6) | **matched-kernel comparator launched 2026-09-08 15:13 (b3 clock) on `be3c39b`: `dscc_arm4_matched_s{0,1,2}` on GPU 7 at three per GPU (C10: three seeds; λ, U learned; ≈ 35–45 h per run, D13); the full-kernel comparator and the frozen-coupling variant not queued; decision (1)–(6) reads against the C10 candidate (LR-only post hoc) once the runs finish** — was: todo after the Arm-2+3 selection; F-SCC training path smoke-tested (`--fscc matched`); its kernel force routed through the pair derivatives too (commit `8a5ca51`, `TestFsccPairForcePath`), so Arm 4 trains at the same memory as Arm 2+3. **Engineering item before Arm 4 (v4.3 §8):** base features and head features in one graph with a single backward (no early-block recomputation), then re-measure the 2× benchmark with ≥ 100 frames per size on the Φ-on model |

### 3.5 Phase 4 — ladder, sparse solver, benchmark (plan §8)
| # | Task | Status |
|---|---|---|
| P4.1 | Tiling-ladder generator and report (`E(+1) - E(0)` vs `1/L`, `K_LR_ii`, active states, `dq` spread, zero total force) | script done — `defect-perovskite/dscc_ladder.py` (static-cell tilings, Madelung coefficient computed for the cell shape, dense below `--dense_max`, sparse above, dense/sparse agreement per cell); to run on the Arm-1 winner and the Arm-2+3 selection |
| P4.2 | Sparse path (CSR `H0`, SP2/LDL inertia, Chebyshev/LOBPCG, tail bounds, PME/FMM) | frontier path done — `dscc/sparse.py`: CSR `H0` from the edge blocks; below-slice count by the LDLᵀ inertia of an unpivoted sparse LU (Sylvester; factorisation residual-checked); mid-gap shift by inertia bisection; window eigenpairs by shift-invert ARPACK to machine precision (residual-checked), `k = |Q| + k_buffer` (8) grown until the certified Fermi-tail bound on the omitted charge is ≤ 1e-8 e; frontier D-SCC solve (window quasi-Newton, Anderson fallback, unmixed residual); rank-k frontier forces by block cotangents. Dense/sparse agreement on the toy: energy 1e-7, forces 1e-6, dq 1e-7. **Not done:** PME/FMM (electrostatics stay the dense Ewald matrix: the dense regime is ≲ 5k atoms), Route B′ on the sparse path (needs the full reference density), the 159-atom real-frame agreement run and the ladder cells (after Arm 1) |
| P4.3 | 2x benchmark per the registered definition | harness done — `defect-perovskite/dscc_benchmark.py` (per-frame energy + forces along a charged trajectory, warm-started, base included; median and p95 of the ratio). **Run 2026-09-07 on the local A4000 (idle), `dscc_arm1_full_s0` (Φ = 0), warm-started along one charged trajectory, after merging the inference backward into one autograd pass (E_base + the Hellmann–Feynman contraction; identical numbers, tests): 79 atoms base 41.9 ms / model 82.9 ms, ratio median 1.98, p95 2.36 (40 frames); 159 atoms 73.2 / 141.0 ms, median 1.92, p95 5.66 (16 frames; the p95 is the slowest frame). Median within 2×, p95 not → fails the registered criterion as it stands** (`~/runs/dscc/benchmark_merged_{79,159}.json`). A first run at 01:35 overlapped the final report on the same GPU and is discarded (medians 2.4–2.5, p95 up to 9). The floor is the base's full forward inside the model (the head needs block 0 attached and E_base is needed anyway); the p95 tail is to be profiled per frame (graph build at the head's 10 Å cutoff) before the selected coupled model is benchmarked. **Profile (2026-09-07, CPU, Φ = 0 winner, 79 atoms): base 187 ms, model 228 ms per frame — ratio 1.22 on CPU against ≈ 2.0 on the GPU**, so the GPU deficit is kernel-launch overhead in the head (per frame ≈ 2700 reshapes, 2100 muls, 1200 copies, 170 einsums from the per-edge Slater–Koster assembly and the block cotangents), not arithmetic (the eigh is 5 % of the head). The v4.3 engineering item before Arm 4 is therefore a launch-count reduction of the head's assembly and backward (fused/vectorised SK blocks, fewer views), to be measured on the registered protocol. **Re-run 05:55 on the idle A4000 with 100 frames at 79 atoms: base 42.4 ms / model 87.9 ms, median 2.07, p95 2.22; 159 atoms (16 frames) 74.1 / 145.0 ms, median 1.96, p95 5.79** (`benchmark_idle_{79,159}.json`) — the Φ = 0 head costs about one base forward, so the ratio sits at 2.0 ± 0.1 and the registered criterion (median AND p95 ≤ 2) fails; the 159-atom p95 is one slow frame of sixteen — note (`f76a6d0`): the head's inference now takes the pair route by default (the cotangent route by `gamma_force_mode = "autograd"`), so the Arm-4 benchmark states which route it timed (on b3 the two are within a second at four 159-atom frames) |

### 3.6 Deletion sweep (plan §3, after the Phase 1 gates)
Done 2026-09-07 02:20 (while the Arm-1 repeat trains from its own snapshot; b3 is synced
and its stale copies removed after the repeat finishes). Method: every core file the v8
programme had modified is restored to the pre-defect upstream commit `050b791` (the
branch base; all 420 branch commits are the programme's own, no upstream merge among
them), every module and test the programme added is removed, the dscc package and its
suite are the only additions that stay.
- [x] Tier-1/Tier-2 constructor and `u_al` (`defect_composition`, `defect_rank`, `defect_constructor_cache`, `defect_seed`)
- [x] static density, residual monopole, covariant registration (`defect_density`, registration in `defect_composition`)
- [x] spectral windows, `r_+` matrix functions, channel normalisation, localisation switches, `rho_img` (`defect_windows`, `defect_frontier`, `defect_image`, `defect_spectral*`)
- [x] canonical lift, branch/cut/tail certificates, `IsoOK` (`defect_lift`)
- [x] separate `Phi_SF`/`Phi_img` objects (`defect_boundary`, `defect_image`)
- [x] spectral-gauge record (`defect_gauge`)
- [x] nuisance intercepts (`defect_objective.profile_intercepts`)
- [x] energy-term registry (`defect_terms`)
- [x] induced-polarisation stage; isolated-boundary outputs
- [x] their tests, launcher flags and docs: removed `mace/modules/defect_*.py` (30 modules), `mace/data/{defects,two_size,dilution,size_sampler}.py`, `mace/modules/latent_ewald.py` (the LES oracle test now calls the `les` package directly), `tests/extensions/defect/` and the 38 programme tests under `tests/unit` and `tests/workflows`, the programme's top-level plans/diagnoses/scripts; restored `run_train.py`, `arg_parser.py`, `tools/train.py`, `model_script_utils.py`, `scripts_utils.py`, `loss.py`, `calculators/mace.py`, `data/{__init__,atomic_data,utils}.py`, `modules/{__init__,models,extensions}.py`, `tools/{default_keys,finetuning_utils,tables_utils,torch_tools}.py` (4092 programme lines). Kept: the dscc package (`mace/modules/dscc/`, imports only upstream `ScaleShiftMACE`, `AtomicData.from_config` with its extra-property pass-through, `prepare_graph`, `tools`), `defect-perovskite/` as the programme's record (67 of its 124 old analysis scripts import deleted modules and are archival, not runnable), the two size-extensivity notes at the top level (uncommitted edits), and the untracked `mace/modules/defect_size.py` (never committed; not touched). Verification: `import mace` and the dscc launcher/scripts on the stripped tree; the dscc suite and the upstream unit tests (results in the commit message).

### 3.7 Deferred register (plan §9)
- [ ] **End-of-programme robustness pass on the selected model only (v4.3):** LR decay 2e-3 → 2e-4 over the last 20 epochs, spike detection with rollback, epoch-averaged final checkpoint; retrain the selected configuration and its Φ = 0 reference under that protocol and report whether any selection decision changes; no decision made before the pass is revisited unless it does. **After Arm 2+3 (C10 (d)): the selected configuration IS the Φ = 0 reference; whether the pass also covers the best coupled arm (LR + U) awaits the ruling.**
- [ ] PME/FMM electrostatics for the sparse path beyond the dense regime (≲ 5k atoms) — plan §8, not needed by any registered gate so far.

## 4. History note (v8 programme, closed 2026-09-06)

Last v8 commit `06c04c8`; full old defect suite 689 passed. The s14a wave: s1/s3/s4/s6
finished before cancellation, s2 stopped on the §3.1 gauge guard (gap floor), s5 was killed
on cancellation. No analysis of these runs.
