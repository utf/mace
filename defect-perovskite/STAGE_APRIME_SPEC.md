# Stage A′ base refit, head edits, and Stage B objective — implementation spec

Received 3 Sep 2026 (verbatim, the plan of record for this cycle).

V_Cl in orthorhombic CsPbCl3 (Mosquera-Lois & Walsh, PRX Energy 4, 043008); PBE labels.
Everything below is host-agnostic by construction; per-host inputs remain pristine geometry,
E_gap and eps_inf. Regime tag every trained number as now.

## 0. Decisions of record

- Joint training of base and head on charged labels is retired. The base is trained on
  neutral labels only and is frozen during head training. Criterion 1 (charged 159-atom
  residual slope) becomes a regression test rather than a gate.
- The latent-Ewald branch E_LR is retained unchanged in form. The carrier density entering it
  is detached; its long-range parameters are frozen at physical values.
- Decay lengths become four learned universal scalars, one per Slater–Koster integral type.
  Nothing is tuned per host.
- Precision: trunk f32, head f64. Base outputs cached (base frozen); only the first
  interaction block is recomputed per step.
- Not in this cycle: eigensolve batching by size group; per-frame Ewald geometry precompute;
  SCC term; occupation/SiC work.

## 1. Stage A′ — base refit on neutral data only

Data: all neutral frames, both sizes. Neutral 159-atom frames upweighted so their realised
share is the same target (default 0.25) in the energy loss AND the force loss; log
`realised_share_E`, `realised_share_F` per epoch. No charged frame anywhere in the base's
training or validation.

Folds: four fold bases (for out-of-fold nulls) plus one production base, same recipe. Every
non-parameter float that touches the forward (`avg_num_neighbors` and any other) is
serialised with the checkpoint and covered by the config round-trip test (standing rule).

Re-derived references, against the production A′ and out-of-fold, with 95% intervals:
- 79-atom neutral window null (E and F slopes vs d)
- 159-atom neutral null (E and F)
- charged 159-atom residual slopes d(E_label − E_base)/dd and force analogue — the new F4
  reference and its interval
- 79-atom charged residual slopes (full range and window), for the record

Extrapolation indicator, per frame (label-free):

    ood_E(frame) = std over fold bases of E_base/N_atoms        [eV/atom]
    ood_F(frame) = rms over atoms of std over fold bases of F_base   [eV/Å]
    s_E = 95th percentile of ood_E over neutral frames
    w_E(frame)   = min(1, (s_E / ood_E)^2)

Store `ood_E`, `ood_F`, `w_E` per frame in the dataset metadata. Report w_E versus d for the
charged 79-atom frames and the charged residual slope restricted to w_E > 0.5.

## 2. Head edits

### 2.1 Centred on-site correction
Replace the local on-site correction by its deviation from the pristine environment:

    corr_i = gamma * [ tanh(h(x_i)) − tanh(h(xbar_{s(i)})) ]

`xbar_s` = mean first-block feature of species s over the pristine reference cell, computed
at build time from the pristine geometry the model already requires and stored as a buffer.
Tests: corr_i = 0 to float tolerance on every atom of the pristine cell; pinned-continuum
test unchanged; `gamma`, `xbar` in the config round-trip.

### 2.2 E_LR: detach the density, freeze the long-range parameters
Keep q = a·alpha from the density-matrix difference exactly as now; detach before the Ewald
energy:

    q_i = a * sum_{mu in i} P_diff[mu, mu]      (spin-summed)
    q   = q.detach()
    assert |sum_i q_i + Delta_n| < 1e-8
    E_LR = E_per(q; R) − E_iso(q; R)            (positions carry gradient; q does not)

Forces from E_LR are dE_LR/dR at fixed q. All long-range parameters: `requires_grad=False`;
eps_inf from config; Gaussian width = the Ewald smearing (one global constant); any
per-species width collapsed to that constant. Test: on ten charged frames of each size,
forces with and without the detach differ by < 1 meV/Å rms (F19); record the measured value.
The isolated-gauge switch and the tiling extrapolation are unchanged.

### 2.3 Image compensation in H — forward probe first, adopt on evidence
Term (electron sign convention of Edit 1, eps_i = eps_local_i − phi_i/eps_inf):

    rho   = carrier density from a first solve of H without the term, detached
    q_c   = −rho                                  (electron; sum = −1)
    phi_img_i = [A_per q_c]_i − [A_iso q_c]_i      (same smearing, gauge and kernel object as E_LR)
    eps_i += −phi_img_i / eps_inf
    second solve of H gives energies, P, forces

One shot, no iteration; no gradient through rho (position dependence at fixed q_c kept).
Identities: zero on any neutral pristine cell; constant across atoms (variance < 1e-6 eV)
for a delocalised carrier on a pristine cell. The term is always on when adopted; E_LR
remains the only gauge switch.

Probe, forward-only on the s7 cohort, both sizes: depth from CBM, participation ratio,
R_bound, 79-atom force loss, with and without the term (F15). Adoption test on ideal
geometries: single vacancy in 1×, 2×, 3× tilings of the pristine supercell; the
frontier-weighted on-site potential sum_i rho_i [madelung_i + comp_i] drifts as ~1/L without
the term (D0) and must drift by ≤ 0.3·D0 with it. Adopt as config default only if both pass.

### 2.4 Decay lengths and modulation
Radial envelope per Slater–Koster type b in {ss-σ, sp-σ, pp-σ, pp-π}:

    L_b = L0 * exp(beta_L * tanh(u_b)),   L0 = 1.0 Å (current), beta_L = ln 2   →  L_b in [0.5, 2.0] Å
    u_b learned, four scalars total, shared across all hosts, init u_b = 0

Modulation switched to the implemented log form `exp(beta * tanh g)` at the range-equivalent
`beta = ln 1.5` (no capacity change, no sign-flip hazard). Escape guards unchanged: pinned
continuum, pristine-gap equality, init gate; log pristine bandwidth. Tests: config round-trip
for `u_b` and the modulation form; hopping magnitudes positive for all g.

### 2.5 Precision and caching
- Trunk f32; head f64. Explicit `.to(torch.float64)` for features, positions, cell and
  counters at the head boundary; E_total = E_base.double() + E_head (+ E_LR); loss in f64.
- Base frozen ⇒ cache E_base and F_base per frame from the production A′ base, keyed by a
  checksum of its state dict. Per step recompute only the first interaction block (f32) to
  obtain features with position gradients. Drift guard: each epoch, one fresh full forward on
  a random frame must match the cache to f32 tolerance.
- Profile one training step (torch.profiler) before and after; record per-component time in
  the run log. Do NOT batch the eigensolve or precompute Ewald geometry in this cycle.
- Test: head outputs (eps, delta_sr, E_head, F_head) identical to the all-f64 run within
  1e-6 eV / 1e-6 eV/Å on a fixed frame; E_total within 1e-3 eV/frame (F20).

## 3. Stage B objective (head only, base frozen)

- Forces: all charged frames, both sizes; large-cell realised share 0.25 as now.
- Energies: charged 159-atom frames at a realised energy share of 0.25–0.5 (config);
  charged 79-atom frames weighted by w_E(frame) from §1; log realised shares per channel.
- `loss_gap` as now (composition-selected).
- c per (charge, size): deterministic calibration over all charged frames at
  initialisation, with E_LR's known values included in the residual.
- Smearing: Gaussian at the confirmed SIGMA (currently 0.05 eV).
- Trainable mask: head parameters only; `u_b`, gamma-channel, Z (projected), c.

## 4. Gates and reports (six seeds on A′)

1. Regression: charged 159-atom residual slopes (E, F) against A′ equal the §1 reference.
2. F4: head d(delta_sr)/dd on the 16 charged 159-atom frames, correct sign, within 1.5× of
   the new reference, in ≥ 4/6.
3. c-consistency: report c(79), c(159) and the predicted size difference (E_LR constant
   difference plus the Madelung G=0 shift difference); promote to a gate once the tolerance
   is known.
4. F10: ligand-Cl − bulk-Cl correction > 50 meV in ≥ 4/6.
5. Pristine gap 2.4 ± 0.1; pinned continuum exact; bandwidth logged.
6. Dilution R ≤ 1.3 with bound fraction; depth from CBM with seed spread (report, no gate).
7. No integral type at its stop in more than 1/6 seeds; report learned L_b.
8. Participation ratio (charged/pristine) with seed spread (report).

Forecasts on record: F15 (probe: depth increases more at 79 than 159, ratio falls, R → 0.9–1.0,
79-atom force loss does not rise); F16 (the +0.36 slope is carried by low-w_E frames;
w_E > 0.5 frames reproduce the window value); F17 (159 null within ±0.05 of zero; charged 159
residual within the old interval or shifted by less than the old null's width; 79-atom charged
residual magnitude < 0.2); F18 (participation-ratio spread halves relative to the joint cohort;
no stop hit); F19 (< 1 meV/Å); F20 (≥ 3× per-epoch speedup with identical head outputs).

## 5. Order

1. Standing-rule updates: serialise non-parameter floats; config round-trip covers `u_b`,
   modulation form, detach flag, compensation flag, precision policy, cache checksum.
2. §2.5 precision and caching; profile; F20.
3. Forward-only on the existing s7 models: §2.3 probe and tiling drift test; §1 indicator on
   the existing Stage-A folds (F15, F16).
4. Stage A′ (folds + production); re-derived references and w_E (F17).
5. Head edits §2.1, §2.2, §2.4 (§2.3 if adopted); six seeds under §3; §4 gates.
6. Report with forecasts scored.
