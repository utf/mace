# Cycle spec: speed, corrected head forms, null-gated objective, two arms

Received 4 Sep 2026, verbatim, as the plan of record for this cycle.

V_Cl in orthorhombic CsPbCl3 (Mosquera-Lois & Walsh, PRX Energy 4, 043008); PBE labels.
Base: `aprime_prod` frozen throughout; folds `aprime_f0..3` for nulls. Regime-tag every
trained number as now. No charged label ever touches the base.

## 0. Standing rules added this cycle

- No per-host constant may live in a default. Per-host *inputs* are exactly: pristine
  geometry (and the species centres and composition derived from it), E_gap, eps_inf, and
  the label pipeline's smearing family and width. Anything else host-specific found in code
  (e.g. `d_ref = 2.861`) is a defect.
- Charged-frame energies enter the head's loss only for a size class that has a neutral
  null (out-of-fold neutral frames of the same size). The trainer checks for the null file
  per size class and refuses charged energies where it is absent. Forces enter for every
  charged frame. The extrapolation weight `w_E` is removed from the objective (delete the
  code path, not just force it to 1).
- Formation-energy outputs are referenced to c of the largest size class that has a
  neutral null (here 159), until the origin of Delta c is settled.

## 1. Speed (do first; everything after is a six-seed arm)

Target: >= 4x per epoch on top of the 1.5x cache gain (F24). The step is CPU-bound in the
per-graph eigensolve and Ewald loops.

1.1 Eigensolve batching by size group. The sampler groups frames by atom count so each
batch has a single n; `eigh`, the divided-difference backward, the Fermi bisection and
(arm B) the second solve all run on [B, n, n] tensors. Bit-identity test against the
per-graph path on a fixed batch (1e-8 eV in eigenvalues, 1e-8 in P).

1.2 Ewald geometry precompute. Training geometries are fixed per frame, so per frame and
keyed by the geometry hash: the smooth periodic kernel `A_per` (N x N, f32 storage, f64 use),
the isolated kernel `A_iso` (N x N) used by E_LR and the compensation, and the per-species
potentials and fields for the element baseline. Per step: phi = A_per . Z as a matvec; the
carrier-density field at the ions (Hellmann-Feynman force) and the per-site deviation
charges by one batched reciprocal-space pass. Drift guard: one frame per epoch recomputed
from scratch to 1e-8.

1.3 Profile one step before and after; record per-component time in the run log.

## 2. Model edits (both arms)

2.1 On-site correction, corrected form (replaces the output-centred form):

    corr_i = gamma * tanh( h(x_i) - h(xbar_{s(i)}) )

Zero on the pristine cell by construction; invariant to a constant shift of h; no penalty.
Test: pristine identity; gradient of corr_i w.r.t. h non-zero on every pristine atom.

2.2 Per-site charges, same form:

    Z_i = Z0[s(i)] + zeta * tanh( z(x_i) - z(xbar_{s(i)}) ),   zeta = 1.0 e (bound)

Neutrality projection on the per-cell sum, as now. Tests: on any pristine cell Z_i equals
the element baseline; round-trip for `zeta` and `z`. Report the per-site deviation
magnitude (max and rms) per seed; on this dataset it is expected to stay small.

2.3 Envelope anchor. Remove `d_ref`. Initialise each bond's Harrison baseline at that
bond's actual length (eta hbar^2/(m d^2) at d = r_ij), or anchor the envelope at a
per-species-pair reference from a universal covalent-radius table; either is acceptable,
choose one and record it. Init gate unchanged (edges, bandwidth).

2.4 E_LR unchanged from Stage B: density detached, branch frozen (host charges zero,
polarisation off, amplitude 1/sqrt(eps_inf)); neutrality invariant Sum q/a + Delta n = 0
checked in the forward.

2.5 Image compensation (arm B only), with the corrected identity and a validity switch:

    rho    = carrier density from the first solve, detached
    q_c    = -rho
    phi_img = A_per q_c - A_iso q_c
    eps_i += -s(frame) * phi_img_i / eps_inf
    s      = sigmoid( (depth/delta_L - 2) / 0.5 )      (bound flag, smooth; 1 when bound, 0 when resonant)

Identity tests: zero at zero charge; for a uniform carrier on a pristine cell the
*periodic* part A_per q_c is constant across atoms (the isolated part is not, so no
constancy assertion on the difference). Log `s` per frame. The tiling drift test stays as
the structural test; run it on thermal frames too.

## 3. Stage B objective (both arms)

- Forces: every charged frame, both sizes; large-cell share 0.25 (realised share logged on
  the column the term reads).
- Energies: charged frames of nulled size classes only (159 here), realised energy share
  0.25-0.5 of the charged energy loss; no 79-atom charged energies.
- `loss_gap` as now; c per (charge, size) calibrated with E_LR in the residual; head-only
  mask; trunk f32 / head f64 with the base cache; smearing Gaussian 0.05 eV.
- Learned decay lengths and log modulation at ln 1.5 as in Stage B.

## 4. Arms

- Arm A: sections 2.1-2.4 with the section 3 objective. Six seeds.
- Arm B: arm A plus 2.5. Six seeds.
- Conditional arm C (only if gate 7 fires on arm A): arm A with the log modulation bound
  widened to ln 2, all escape guards unchanged. Six seeds.

## 5. Readouts before training (forward-only, existing Stage B cohort)

5.1 The head's d(delta_sr)/dd on the 79-atom charged frames (b2 statistic) for all six
Stage B seeds. This tells whether the full-weight 79-atom energies taught the head the
+0.37 base artefact.

5.2 Optional diagnostic, no training role: per-atom kNN distance in the cached first-block
feature space to the neutral training frames, max over atoms per frame, against d. Report
only.

## 6. Gates (per arm, six seeds)

1. Regression: charged 159 residual slopes (E, F) against `aprime_prod` equal the A'
   reference (identity under a frozen base).
2. F4: head d(delta_sr)/dd on the 16 charged 159-atom frames, correct sign, inside
   [-0.142, -0.063], >= 4/6.
3. Report: c(79), c(159), Delta c, predicted Delta c; Delta c compared across arms.
4. F10 on the corrected form: ligand-Cl - bulk-Cl correction > 50 meV and > 2 sigma of
   bulk-Cl in >= 4/6.
5. Pristine gap 2.4 +- 0.1; pinned continuum exact; bandwidth logged; init gate passed.
6. Dilution R inside R_DFT's interval [0.66, 1.34], bound fraction reported; depth from CBM
   with seed spread (report).
7. Stops: fraction of seeds at |tanh g| > 0.98 per integral type (report by seed and type);
   learned L_b reported.
8. Participation ratio (charged/pristine) with seed spread (report).
9. Head 79-atom d(delta_sr)/dd (report): negative, |slope| within 2x of the forces-only
   -0.17; must not approach +0.37.
10. Arm B only: tiling drift on ideal and thermal frames <= 0.3 D0; `s` distribution logged.

## 7. Forecasts on record

- F21 (arm A): F10 passes >= 4/6.
- F22 (arm A): seeds at a pp stop fall by half relative to Stage B.
- F23 (arm B): Delta c falls by 0.2-0.4 eV relative to arm A; R inside [0.66, 1.34];
  thermal-frame tiling drift <= 0.3 D0; F4 holds.
- F24: >= 4x per epoch from section 1.
- F25: the existing Stage B cohort's 79-atom head slope has moved toward +0.37 relative to
  -0.17; arm A restores it (gate 9).

## 8. Order

1. Section 1 speed, with bit-identity and drift guards; profile; F24.
2. Section 5 readouts on the existing Stage B cohort.
3. Section 2 edits and section 0 rules as config; round-trip test flips every new knob;
   `d_ref` removed; `w_E` code path deleted.
4. Arm A, then arm B; gates; forecasts scored. Arm C only if gate 7 fires on arm A.
5. Report; Delta c origin decision; then the occupation/SiC design.

## 9. Not in this cycle

SCC term; occupation/SiC; promoting c-consistency to a gate; three-centre hopping terms;
any per-host weighting or selection of frames beyond the null-gated size rule.
