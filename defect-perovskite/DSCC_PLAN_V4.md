# D-SCC charge head: implementation plan (v4, consolidated)

Self-contained specification for a differential self-consistent-charge tight-binding head
(D-SCC) on a frozen neutral MLIP. Production target: PBC energies and forces for the chlorine
vacancy in orthorhombic CsPbCl3 (79- and 159-atom cells, charge states 0 and +1).
Constraints: no new DFT, no literature or database charges, frozen neutral base, no defect
labels or active regions reach the model, inference cost at scale within 2x of the base.
Each phase has hard gates; a later phase does not start until the previous phase's gates
pass. Every threshold is written down before the corresponding result is opened.

Status of the model: a 0/+1 D-SCC MVP and a foundation-model stepping stone. It is not a
general multi-carrier foundation head (§2.8).

Changes from v3: kernel components renamed `K_SR`/`K_LR` and a second, periodic
range-separated kernel regime added as a registered arm (§2.4); saddle-consistent root rule
replacing "lowest root" (§2.6, §5); D-SCC's feedback stated as localising, not screening,
with forecasts rewritten accordingly (§2.10); multi-carrier scope restated (§2.8); Route B
gap regulariser on the Hamiltonian actually filled (§6); two F-SCC comparators (§2.9, §7);
`+C_Q` explicit in the residual with the quadratic-loss assumption stated (§6); admission
diagnostics computed inside each outer fold with uncertainty (§6); separately trained
scalar-only control in Arm 1 (§7); placement check over identified first-shell bonds
(§2.4); PME/FMM at scale and a defined 2x benchmark (§8); summary language corrected (§0, §1).

---

## 0. Summary of the model

```
E(R, S) = E_base(R) + J*(R, S) + C_Q          (C_0 = 0)
J*      = stationary value of the D-SCC functional (§2.6)
S       = electronic state {Q, dN_up, dN_dn}; S_ref = {0, 0, 0}
```

- `E_base`: frozen neutral MLIP (periodic, local). Never retrained here.
- `J*`: difference between two fillings of ONE self-consistent tight-binding Hamiltonian,
  plus electrostatics in the charge difference `dq`. Exactly zero at `S_ref`.
- `C_Q`: one constant per formal charge, profiled on admitted training energies, shared by
  all sizes.
- No frontier windows, no valence-rank constructor, no pristine registration, no
  periodic-to-isolated lift, no isolated outputs. PBC outputs only.
- The image ($1/L$) physics is monopole-exact by construction: the reference is neutral, so
  the monopole of `dq` is the total charge of the requested state; the quadrupole term
  depends on the model's own `dq`.
- The model does contain a learned per-species static charge pattern when Route B (§2.5)
  is active, and stored pristine statistics (species feature means, pristine gap target,
  pristine reference-fill charges used for initialisation).
- D-SCC has no density-matrix screening. Its self-consistency is a localising feedback on
  the vacated orbital (§2.10); host screening is carried by `eps_inf` and, in Route B, by
  the static pattern. This is the property the F-SCC comparator (§2.9) tests.

---

## 1. Notation, conventions, external inputs

- Atoms `i = 1..N_at`, species `Z_i`, positions `R`, cell `h`, volume `Omega`, shortest
  cell vector `L_min`.
- Basis: `s, p_x, p_y, p_z` per atom, orthonormal representation, `n_orb = 4 N_at` per spin.
- `Pi_i`: projector onto atom i's 4 orbitals; `Tr(Pi_i X)` = sum of that block's diagonal.
- Species table `n0[Z]` = neutral valence count in the basis: Cs 1, Pb 4, Cl 7.
  `N_ref = sum_i n0[Z_i]`.
- Spin split of the reference: majority-channel default, `N_ref_up = ceil(N_ref/2)`,
  `N_ref_dn = floor(N_ref/2)`, overridable by the state record.
  `N_S_sigma = N_ref_sigma + dN_sigma`, `Q = -(dN_up + dN_dn)`.
- Smearing: Gaussian, `sigma_s = 0.05 eV` (§11): `f(x) = 0.5*erfc(x)`,
  `x = (eps - mu)/sigma_s`, generalised entropy
  `R = -sigma_s * sum_a exp(-x_a^2)/(2 sqrt(pi))`.
  `F_band(H, N) = min_P [Tr(P H) + R(P)]` s.t. `Tr P = N`. Required identity:
  `dF_band/dH_ab = P_ba`.
- Charges in units of +e. An electron at site i with site potential `V_i` has energy `-V_i`.
- Head arithmetic in float64. All thresholds, widths, cutoffs and conventions serialised.

**External-information rule.** The host-specific physical inputs are two scalars,
`eps_inf = 4` and the host gap `E_gap = 2.40 eV`. Other registered items (smearing,
background convention, species table, cutoffs, hardness bounds) are label-code or model
conventions, not host physics. No oxidation states, nominal charges, Born charges or
literature site quantities enter anywhere. External parameter sets (e.g. GFN1-xTB
hardness) may set the *bounds* of learnable quantities, never their values. Every
charge-like learnable is initialised from the model's own reference fill.

---

## 2. Model definition

### 2.1 Runtime Hamiltonian H0(R; theta)

Real symmetric, sparse on the base neighbour graph (cutoff `r_cut`), spin-independent.

```
H0 = H_SK + H_onsite_scalar + H_onsite_dir
```

- `H_SK`: Slater–Koster `ss_sigma, sp_sigma, pp_sigma, pp_pi` per species pair; Harrison
  initialisation at the actual bond length; four learned positive decay lengths; bounded
  log-form modulation from frozen early-block base features; smooth cutoff with full
  position and cell derivatives.
- `H_onsite_scalar`: species/shell baselines plus bounded corrections centred on the
  pristine species feature means (one forward pass on the pristine cell, stored).
  No unbounded constant channel.
- `H_onsite_dir = a_i (v_i . T_sp_i) + b_i (Q_i : T_pp_i)`, bounded species-level
  coefficients. Rank-1 descriptor `v_i`: the base's `l=1` features (position derivatives
  through the recomputed first block). Rank-2 descriptor, geometric:
  `Q_i = sum_j w(r_ij) (rhat_ij rhat_ij^T - I/3)`, smooth cutoff `w`; vanishes at
  centrosymmetric sites.
- Base features: detached w.r.t. base parameters, NOT w.r.t. positions or cell.

Gauge: `H0 -> H0 + a I` changes `E` by `-a Q` (absorbed by `C_Q`) and leaves all
densities, forces and stresses unchanged. Bounds, centring and the gap regulariser
prevent drift.

### 2.2 State schema

`state = {schema_version, Q, dN_up, dN_dn, occupation_policy="count_fill"}`.
Canonical equality to `S_ref` triggers the exact neutral short-circuit: return `E_base`,
its forces and stress without evaluating the head.

### 2.3 Charge difference

```
dP_sigma = P_S_sigma - P_ref_sigma
dq_i     = - sum_sigma Tr(Pi_i dP_sigma)          # sum_i dq_i == Q exactly
```

### 2.4 Same-carrier kernel: common structure and two registered regimes

Common structure for every regime:

```
Gamma_ij (i != j) = [K_LR_ij + lambda_dir * K_SR_ij] / eps_inf
Gamma_ii          =  K_LR_ii / eps_inf + U_eff[Z_i]
Phi_cc            =  0.5 * dq^T Gamma dq
E_PBC             :  Ewald matrix of unit Gaussian charges (width r_g) under PBC with
                     neutralising background and alpha-cancellation term
```

- `K_LR` is kept at full weight; it carries the $1/L$ physics. It is not a pure image
  kernel in either regime: it retains ordinary `1/r` interaction for pairs outside the
  short-range component, and the name "image" is not used.
- `lambda_dir` in `[0, lambda_max]`, learnable, initialised at registered `lambda_0`
  (small). `U_eff[Z]` in `[0, U_max[Z]]`, learnable, initialised small; `U_max[Z]` from
  GFN1-xTB hardness (bounds only).
- Diagonal convention (both regimes): with `U_eff -> 0` the diagonal short-range self-term
  vanishes; `U_eff` replaces both the scaled Gaussian self-energy and the hardness.
- `Gamma` need not be positive definite (the self-image term is negative); stability is a
  numerical gate (§5).
- Tests (both regimes): independence of every prediction from the Ewald splitting
  parameter for `E_PBC`, `K_LR` and `Gamma_LR` separately from `r_g`, `r_split` and
  `r_s`; tiling-ladder check that the size-dependent part of `K_LR_ii` tends to
  `-alpha_M / L`.

**Regime A — shell switch (v3 kernel).**

```
K_SR_ij (i != j) = erf(r_ij/(2 r_g))/r_ij * w_dir(r_ij), r_ij minimum-image distance
K_SR_ii          = 1/(sqrt(pi) r_g)
K_LR             = E_PBC - K_SR
w_dir            : C2 switch 1 -> 0 on [r_d1, r_d2] in the first–second shell gap
                   (registered defaults r_d1 = 3.2 Å, r_d2 = 3.6 Å), identical at all
                   sizes; both below the minimum vacancy-spanning distance (4.8 Å)
```

- Consequence, registered: `lambda_dir` acts only within the ligand shell; the
  vacancy-spanning Pb–Pb pair term sits entirely in `K_LR` at weight 1 at both sizes. In
  the 79-atom cell (`L_min/2 = 5.6 Å`) the flanking Pb are, for `d > 5.6 Å`, closer to
  each other's periodic images than to each other; a direct/image split of the dimer is
  not meaningful there and only the lattice total is.
- Placement check (label-free, from training-frame histograms) over *identified* bonds:
  first-shell Pb–Cl bonds (six nearest Cl of each Pb by distance rank per frame) with
  `r > r_d1`, and intra-octahedron Cl–Cl nearest neighbours with `r < r_d2`, must each be
  a fraction below a registered floor (default 1e-3). Cs–Cl pairs (≈3.5–4.1 Å) straddle
  any window in this range, carry negligible `dq`, and are covered by the diagnostic.
- Switch diagnostic, reported in every arm using this regime:
  `m_sw = sum_{i<j, r_d1 < r_ij < r_d2} |dq_i dq_j|`, per frame with trajectory
  statistics, plus the `(r_d1, r_d2)` sensitivity of `Phi_cc`, the level's `d`-response
  and the forces. If `m_sw` exceeds its registered floor, revisit the radii before
  Arm 2+3 opens.

**Regime B — periodic range separation (added in v4).**

```
s(r)     = erf(r/(2 r_g))/r - erf(r/r_s)/r         # smooth, short-ranged, r_s > 2 r_g
K_SR_ij  = sum_L s(|r_ij + L|)                       # lattice sum over images in range
K_SR_ii  = s(0) + sum_{L != 0} s(|L|)
K_LR     = E_PBC - K_SR = Ewald kernel of the broad Gaussian erf(r/r_s)/r, with background
```

- Rewrapping-invariant by construction (a lattice sum); no switch; `r_s` registered
  (default 6.5 Å) so the short-range component reaches the vacancy-spanning Pb pair and
  its near images symmetrically.
- Diagnostic analogue of `m_sw`: the short-range fraction of the intra-carrier
  interaction, `f_SR = sum_{i<j} |dq_i dq_j| K_SR_ij / sum_{i<j} |dq_i dq_j| (K_SR + K_LR)_ij`,
  plus the `r_s` sensitivity of `Phi_cc`, the level's `d`-response and the forces.
- Registered rule: if suppressing the short-range component (`lambda_dir -> 0`) destroys
  the finite-size slope on the 79->159 ladder or the model tiling ladder, record a
  small-cell incompatibility of the range split; do not resolve it by moving the split.

Both regimes are arms of the same experiment (§7). Regime A is not the only control on the
direct self-term; neither regime is described as self-interaction control (§2.10).

### 2.5 Host coupling (Route B; a registered arm, off in Route A)

Far-field force channel and long-range well, linear in `dq`, conservative:

```
Zbar_i = Zstar[Z_i] - (1/N_at) * sum_j Zstar[Z_j]        # per-cell centring (mandatory)
sum_s c_s Zstar_s = 0 over the pristine composition       # parameterisation (2 free scalars)
Gamma_LR : Ewald matrix between a unit Gaussian of width r_g at i and width r_split at j,
           divided by eps_inf; r_split > r_g registered so H0 owns the sharp near field
W        = Gamma_LR @ Zbar                                # geometry-only, once per frame
E_SF     = dq^T W
```

- Centring is not optional: an uncentred per-species pattern on the present ions of a
  vacancy cell carries net charge `-Zstar_Cl`, whose cross term with `dq` multiplies the
  physical $1/L$ term by `(1 + 2z)`, an error no `C_Q` can absorb. Centred, `dq` alone
  carries the total charge and the missing ion still appears as a local feature against
  the background.
- Initialisation: `Zstar_s <- species average over the pristine cell of
  q0_i = n0[Z_i] - sum_sigma Tr(Pi_i P_ref_sigma)` evaluated at `H = H0` with the current
  checkpoint (a fill already performed), projected onto the sum rule. Bounds
  `|Zstar_s| <= Z_max` registered from the init magnitudes. Weak L2 regulariser toward the
  init (registered weight). `q0_i` per site is reported as a diagnostic only.
- One `Zstar` serves both the potential and the force; a conservative functional with
  fixed charges cannot use different values for the two roles. The range split confines
  the resulting error to the long-range well.
- Effects: site potential `V_i += W_i`; force on every atom j
  `-dq^T (dGamma_LR/dR_j) Zbar`, nonzero outside the carrier's support. `E_SF = 0` at
  `S_ref`. No change to SCF structure (external potential).
- The Hamiltonian actually filled at `dq = 0` in Route B is `H0 - W`; the gap regulariser
  targets it (§6).

### 2.6 Functional, stationarity, SCF loop, root rule

```
J[P_S, P_ref] = sum_sigma { Tr[(P_S_sigma - P_ref_sigma) H0]
                            + R(P_S_sigma) - R(P_ref_sigma) }
                + Phi_cc[dq] + E_SF[dq]           # E_SF only when Route B is active
```

Stationary (saddle) point, one Hamiltonian for both fillings:

```
V = Gamma @ dq + W                               # W = 0 in Route A
H = H0 - sum_i V_i Pi_i
P_S_sigma = fill(H, N_S_sigma);  P_ref_sigma = fill(H, N_ref_sigma)
```

Band form (hard identity, tolerance 1e-9 eV; the linear term cancels exactly):

```
J* = sum_sigma [ F_band(H, N_S_sigma) - F_band(H, N_ref_sigma) ] - 0.5 * dq^T Gamma dq
```

SCF loop (batched over frames):

```
W  = Gamma_LR @ Zbar (if active); dq = registered initialisation
repeat:
    V = Gamma @ dq + W;  H = H0 - blockdiag(V_i * I_4)
    for sigma: eps, U = eigh(H); mu_S, mu_ref by bisection;
               P_S = U f((eps-mu_S)/sigma_s) U^T; P_ref likewise
    dq_new = -sum_sigma [Tr(Pi_i (P_S - P_ref))]_i
    res    = dq_new - dq                          # UNMIXED residual for convergence
    dq     = broyden_or_anderson(dq, dq_new)
until |res| < tol_q and |J change| < tol_E and commutator norm < tol_c
```

Iteration count logged per frame and capped (registered `n_max`); a frame hitting the cap
is flagged, never silently accepted. Contraction diagnostic: the ratio
`rho = |res_{k+1}| / |res_k|` over the last unmixed iterations, reported per frame;
`rho` close to 1 flags a near-unstable fixed point.

**Root rule (saddle-consistent).** `J` is a saddle, so "lowest root" is not a selection
rule. Registered initialisations: (i) `dq = 0`; (ii) deterministic continuation from zero
coupling (`lambda_dir = 0`, `U_eff = 0`, `W = 0`) to the trained values in registered
steps; (iii) warm start from the previous frame along a trajectory. All must converge to
the same fixed point (`|dq_a - dq_b| < tol_root`), with symmetry-related solutions of
identical observables counted as one. Distinct converged branches on any frame set fail
the arm regardless of `J`. Production uses (i); (ii) and (iii) are validation checks.

### 2.7 Energy, forces, stress

```
E(R, S) = E_base(R) + J*(R, S) + C_Q
F_I = F_I^base - sum_sigma Tr[dP_sigma dH0/dR_I] - 0.5 dq^T (dGamma/dR_I) dq
      - dq^T (dGamma_LR/dR_I) Zbar                                (Route B)
stress: (1/Omega) dE/d(strain), explicit cell dependence of H0, Gamma, Gamma_LR, Gaussian
        widths (fixed in Cartesian units), plus base stress
```

Envelope theorem at the saddle: no derivative of `P_S`, `P_ref` or chemical potentials.
`Zbar` and `C_Q` carry no force or stress. Matrix-function backward for training:
`dP = U [F o (U^T dH U)] U^T + (dP/dmu) dmu`, `F_ab = (f_a - f_b)/(eps_a - eps_b)`,
`F_aa = f'(eps_a)`, `dmu = sum_a f'(eps_a) deps_a / sum_a f'(eps_a)`. Never differentiate
through raw eigenvectors.

### 2.8 Scope statement (multi-carrier)

- `Phi_cc` contains interactions among the *changes* only. Carriers on the same side of
  the reference are both in `dq` and interact through `Phi_cc`; a change interacts with a
  carrier that remains in the reference only through the shared `H`, never through
  `Phi_cc`. Supported: `0 -> +1` (validated here), `0 -> +2` (structurally supported,
  unvalidated; for `V_Cl` the second hole is a valence-band hole and delocalised, so this
  host is a poor test). Not supported: `0 -> -1`, the `+1` state of a double donor, and any
  state whose change Coulomb-couples to a reference carrier. The extension for these is an
  occupation-dependent on-site term or F-SCC (§9), not a self-interaction correction.
- One `lambda_dir` scales single-carrier short-range self-terms and the carrier–carrier
  cross term together. For PBE-type labels this is consistent: PBE's E(N) is smooth and
  convex, so the fractional curvature of one carrier and the integer second difference for
  two carriers in the same level are the same quantity. For beyond-PBE labels (hybrid,
  experiment) it is not, and a separate self/cross treatment would be required. F-SCC has
  the same single-kernel structure and does not remove this.
- Claim: 0/+1 MVP; multi-carrier structural support only.

### 2.9 F-SCC comparators (offline only)

Same `H0`, same `fill`. Two independent SCC solves with absolute charges
`Dq_X_i = n0[Z_i] - sum_sigma Tr(Pi_i P_X_sigma)`, `X in {S, ref}`, each self-consistent in
its own potential `V_X = Gamma_F @ Dq_X`;
`E_SCC(X) = sum_sigma[Tr(P_X H0) + R] + 0.5 Dq_X^T Gamma_F Dq_X`,
`head = E_SCC(S) - E_SCC(ref)`. Host coupling is intrinsic (absolute charges); no Route B.
float64 mandatory.

Two comparators:
- **Matched-kernel F-SCC:** `Gamma_F` has the selected D-SCC kernel regime and bounds,
  with `lambda_dir`, `U_eff` learned under the same protocol (and additionally reported
  frozen at the D-SCC values). Isolates full versus differential SCC.
- **Full-kernel F-SCC:** `Gamma_F = E_PBC / eps_inf + diag(U_eff)`. Upper-fidelity
  ceiling.

Report for each: the trace norm of `P_S - P_ref` beyond its first `|Q|` singular values
(diagnostic only) and the carrier `N_eff`/`R_eff` versus `lambda_dir` (expected opposite
trend to D-SCC, §2.10).

### 2.10 Registered forecasts and the feedback sign

- **Sign of the D-SCC feedback.** With `dq > 0` where the hole is, `V = Gamma dq > 0`
  there and `H = H0 - sum V_i Pi_i` lowers those sites, so the orbital being vacated (the
  reference HOMO of the shared `H`) is pulled toward them and `dq` grows there. Positive
  `Gamma` therefore *localises* in D-SCC, weakly below the threshold
  `Gamma_ii - Gamma_ij ≈ 2t` (t the relevant hopping) and strongly above it. D-SCC has no
  density-matrix screening; its self-consistency is a polaron-like well on the vacated
  orbital. In F-SCC the occupied electrons of the charged state respond to the deficit and
  screen it, so positive `Gamma` *delocalises*. The two functionals have opposite shape
  feedback; this is expected to appear as opposite `N_eff`/`R_eff` trends versus
  `lambda_dir` in Arm 4.
- **Route A forecast.** No static object exists, so `Gamma dq` on the shared `H` is the
  only representation of the core's field on the reference electron (`dq` is spatially
  the core's compensating charge; `U_eff` mimics the missing-neighbour well at the hub
  sites). `lambda_dir` and `U_eff` come out O(1); the `lambda_dir = 0` ablation makes the
  level shallower and the carrier more extended. A `lambda_dir` obtained in Route A is
  NOT interpretable as self-interaction.
- **Route B forecast.** With the centred pattern supplying the intermediate-range well and
  `H0` the near field, `lambda_dir` and `U_eff` move toward the values that reproduce the
  labels' curvature. The direction of the `lambda_dir = 0` ablation on `N_eff`/`R_eff` is
  the discriminating test between the well-proxy and curvature interpretations.
- **Arm-1 precondition diagnostic:** saturation fraction of the hub-Pb scalar onsite
  corrections with `Phi = 0`. Saturation predicts the Route A proxy behaviour.
- **Practical consequence:** the localising feedback can produce distinct fixed points at
  asymmetric geometries; the root rule (§2.6) and contraction diagnostic are mandatory.

---

## 3. Codebase changes

**Delete:** Tier-1/Tier-2 constructor and `u_al`; static density, residual monopole,
covariant registration; spectral windows, `r_+` matrix functions, channel normalisation,
localisation switches, `rho_img`; canonical lift, branch/cut/tail certificates, `IsoOK`;
separate `Phi_SF`/`Phi_img` objects; spectral-gauge record; nuisance intercepts;
energy-term registry; induced-polarisation stage; isolated-boundary outputs.

**Keep:** `H_fix -> H0` (SK, scalar onsite, bounds, feature modulation); `count_fill` and
bisection; divided-difference backward; base feature cache; the verified full-sum Ewald;
the per-site charge module; state schema; data loaders; loss code pending the §6 audit.

**Add:** species table; `dq`; kernel regimes A and B (`K_SR`, `K_LR`, placement check,
`m_sw`, `f_SR`); `Gamma`; `Gamma_LR`, `Zbar`, `W`; geometric `Q_i` descriptor; D-SCC loop,
band-form identity, root rule, contraction diagnostic, HF forces, autodiff stress;
batching over frames; SCF logging; both F-SCC loops; scalar-only `H0` control
configuration; fold-wise base-support and coverage diagnostics; test harness;
tiling-ladder generator; sparse path with PME/FMM (§8).

---

## 4. Phase 0 — Scaffold

Tasks: data pipeline (states, species table, label conventions, geometry-state groups
formed before the split); `H0` with both descriptors; `fill` with Gaussian smearing and
matrix-function backward; `E_PBC`, `K_SR`/`K_LR` for both regimes, `Gamma_LR`, with
derivatives; placement check over identified bonds; serialisation and checkpoint round
trip.

Exit gates: `dF_band/dH_ab = P_ba` to 1e-10; `E_PBC`, `K_LR` (both regimes) and
`Gamma_LR` each independent of the Ewald splitting parameter to 1e-10 eV (energy, forces,
stress), separately from `r_g`, `r_split`, `r_s`; regime-B `K_SR` lattice sum converged
with respect to its image range to 1e-10 eV and rewrapping-invariant; placement-check
floors satisfied; `H0` invariant under translation, rotation, permutation, rewrapping with
covariant derivatives; loaded checkpoint reproduces an uncached forward pass bit-for-bit.

---

## 5. Phase 1 — D-SCC forward and derivatives

Tasks: batched SCF loop; energy; HF forces; autodiff stress; neutral short-circuit;
SCF logging/cap; contraction diagnostic; root-rule harness; test harness; Route B switch
(off by default); both kernel regimes selectable.

Hard gates (defaults, to be registered):
- Neutral null: `S = S_ref` gives energy, forces, stress bit-identical to the base.
- `sum_i dq_i = Q` to 1e-12 for every converged state.
- Gauge: random `a I` on `H0` leaves `dP`, forces, stress unchanged to 1e-10; `E` shifts
  by exactly `-a Q`.
- Primary functional and band form agree to 1e-9 eV (with and without `E_SF`).
- Central-difference forces at steps {1e-2, 1e-3, 1e-4} Å agree with analytic forces to
  1e-4 eV/Å at tight SCF tolerance, discrepancy decreasing with tolerance; same for all
  six strain components; both routes, both regimes.
- Invariance of `E` under translation, rotation, permutation, rewrapping to 1e-10 eV.
- Stability and single-valuedness: SCF converges within `n_max` on every frame of both
  charge states; energy independent of mixing history to `tol_E`; the three registered
  initialisations (§2.6) reach the same fixed point on every frame; `rho` below a
  registered ceiling.
- Route B negative test: the UNCENTRED pattern must fail the tiling-ladder $1/L$ check
  (§8); the centred pattern must pass.

---

## 6. Phase 2 — Training protocol (shared by every arm in §7)

**Energy admission (null-gated, numeric, out-of-fold, coverage-gated, per fold).**

```
Inside each outer training fold, for each size L, within the charged window of the
collective coordinate d (Pb–Pb across the vacancy):
  coverage(L): bin the charged window (registered bin width); every bin must contain at
               least n_min out-of-fold neutral frames of this fold. Any empty bin ->
               s0(L) is "unmeasurable" and charged ENERGIES at L are not admitted.
  s0(L) ± SE:  slope of E_label(V0) - E_base vs d with standard error, base residuals
               out-of-fold only (k-fold fine-tuned base or held-out neutral frames of this
               fold). The production base's residual on its own training frames is
               inadmissible.
  sQ(L):       slope of E_label(V+) - E_base vs d, before any head.
Admit charged energies at L only if coverage(L) passes AND |s0(L)| + z * SE(s0) <= s_tol
(z registered). Forces are admitted at every size. Store the coverage table, s0, SE, sQ
per size and fold before that fold's results are opened. Expectation to record: at 79
atoms the neutral frames end at 6.0 Å and the charged frames run to 6.8 Å, so coverage
fails and s0(79) is unmeasurable.
```

Consequences to record: with the current base, energy shape and `C_Q` are fitted on the
sparse 159-atom set; forces carry most of the training signal; the 79-atom energy
residual is a held-out prediction read against the stored diagnostics; the size-transfer
test runs 159 -> 79. If the base is later improved, readmission is automatic through the
same criterion.

Open item (not a blocker): the `d`-criterion is benchmark-specific. A foundation protocol
needs a defect-agnostic coverage rule in the base's own feature space; candidate:
per-atom novelty of charged-frame base features relative to the out-of-fold neutral
training distribution, aggregated by max over atoms, label-free.

**Leak readout.** Head slope `d(J* + C_Q)/dd` at each size; the 79-vs-159 difference must
stay within a registered margin.

**Objective.**

```
r_i(theta) = E_base(R_i) + J*(R_i, S_i; theta) + C_{Q_i} - E_label_i   # total-cell eV
L_E = sum_g W_g sum_{i in g} w~_i r_i^2                                   # weighted quadratic
L   = w_E L_E + w_F L_F + w_gap L_gap + regularisers                      # stress weight: 0
```

- Closed-form profiling of `C_Q` assumes the weighted quadratic energy loss above; any
  other loss requires the registered convex one-dimensional solve. `C_Q` per non-zero
  charge is profiled on admitted training frames only, recomputed at every gradient
  evaluation; `C_0 = 0`; stale calibration is a loading error; no recalibration when a
  size is held out.
- Never divide energy residuals by `N_at`. Strata keyed by (source, host, Q, composition
  hash, cell convention, size class), frozen total weight per stratum, frame weights
  normalised within it; the key is loss metadata, statically unreachable from the model.
- Gap regulariser (mandatory): `L_gap = (E_gap_model - E_gap)^2` with `E_gap_model` =
  LUMO−HOMO at the exact valence count of the Hamiltonian actually filled at `dq = 0`:
  `H0` in Route A, `H0 - W` in Route B (pristine cell, sum-rule pattern). Apply it to the
  cell the registered `2.40 eV` refers to: if static-lattice, regularise the static
  pristine cell and report the thermal ensemble-mean gap over pristine frames as a
  diagnostic; if thermal-average, regularise the ensemble mean over a batch of pristine
  frames per step. Record which convention is in force.
- Route B regulariser: weak L2 of `Zstar` toward its model-derived init.
- Gradients: unrolled autodiff through the converged SCF at 636 orbitals; implicit
  differentiation later.

**Loss-path audit (once, before any result is read):** inject ±1 eV into `C_Q` and compare
loss change and gradient with the analytic result; trace every admitted energy through
masks, units, reductions, weights, checkpoint restore; show profiling removes an injected
constant to the numerical floor; verify no validation, test or tiling frame enters `C_Q`.

Fixed protocol: splits, strata, weights, energy/force balance, six seeds, metric list,
`s_tol`, `z`, `n_min`, bin width, the `m_sw`/`f_SR` floors, `tol_root`, the `rho` ceiling
and all arm thresholds (§7) frozen before training.

---

## 7. Phase 3 — Attribution arms, in order

Each arm uses the §6 protocol and six seeds; each arm's thresholds are written before its
results are opened. Report per arm: charged force error, energy shape on admitted sizes,
raw and calibrated energy means, 159->79 residual, `N_eff`/`R_eff`, seed spreads, bound
saturation fractions, SCF iteration and `rho` statistics, root-rule outcome, `m_sw` or
`f_SR`, far-field force residual by distance shell (0–2, 2–4, 4–6, 6–8, >8 Å from the
vacancy).

**Arm 1 — H0 with and without the tensor block, Phi = 0, Route A.** Two fillings of `H0`,
no SCF (stationary in its own right, HF forces). Two separately trained models under the
identical protocol: full `H0` and a scalar-only control (`H_onsite_dir = 0`); a
post-training ablation is reported in addition but is not the attribution experiment.
Decision experiment on the response decomposition, not RMSE: (i) vacancy-spanning `pp`
stop fraction <= 1 of 6 seeds; (ii) flanking-Pb tensor coefficient > 3x its bulk spread and
the scalar-only control loses >= 30% of the modelled level-vs-bond response; (iii)
`p_sigma/p_pi` splitting has the analytically frozen sign; (iv) participation-ratio seed
spread halves at unchanged or better force quality without coefficient saturation. Routing
as in the original Stage-2 table (A: proceed; D: conditional edge residual; B: stop and
investigate identifiability; C: repeat covariance/sign tests once). Record the Route A
far-field readout and the hub-Pb onsite-correction saturation fraction (§2.10).

**Arm 2+3 — joint factorial on the Arm-1 winner.**
Route ∈ {A, B} × kernel regime ∈ {A: shell switch, B: range-separated} × coupling ∈
{LR-only (`lambda_dir = 0, U_eff = 0`); LR + U (`lambda_dir = 0`, `U_eff` learned);
full (`lambda_dir`, `U_eff` learned); `lambda_dir = 1` fixed, `U_eff` learned}, plus
`Phi = 0` under each route: 18 configurations × 6 seeds. Thresholds written before results
are opened. Selection of route, regime and coupling is joint, on: charged force error and
energy shape on admitted sizes beyond `max(tau_phys, tau_noise)`; localisation stability
across seeds; root rule passed and bounded SCF iterations; `m_sw`/`f_SR` below floor; for
Route B additionally the 4–8 Å far-field force residual beyond noise, `Zstar` inside
bounds and unsaturated, tiling-ladder $1/L$ coefficient unchanged. Report `lambda_dir`,
`U_eff` and the `lambda_dir = 0` ablation direction on `N_eff`/`R_eff` under every route
and regime; check the §2.10 forecasts; apply the regime-B small-cell incompatibility rule
(§2.4). If Route A's far-field residual is already within tolerance, Route B may still be
reported but the scope is stated.

**Arm 4 — F-SCC comparators and decision.** Both comparators of §2.9, same seeds and
protocol. Select D-SCC as production only if all hold against the matched-kernel
comparator: (1) charged force and energy-shape errors equivalent within the predeclared
noise margin; (2) Pb–Pb localisation and force response stable across seeds; (3) the
shared `C_Q` leaves no larger 159->79 bias; (4) dense and sparse implementations agree to
force tolerance (once §8 exists); (5) measured wall time below 2x the base under the §8
benchmark definition; (6) thermally active states bounded over the 79->159 and tiling
ladders. Report the full-kernel comparator as the fidelity ceiling. Interpretation rule:
the matched comparator differs from D-SCC in density-matrix screening (opposite feedback
sign, §2.10), far-field response and reference-carrier coupling; a win there is physics
D-SCC delegates to `H0`, `eps_inf` or Route B, not a failure of the differential
structure. If the matched comparator materially wins after Route B, the next experiment is
a first-order occupied-space response correction (§9).

---

## 8. Phase 4 — Size ladder, sparse solver, cost benchmark

Model-only tiling ladder: pristine supercells with one vacancy. Report `E(+1) - E(0)`
versus `1/L` with the Madelung coefficient of the registered convention (centred and
uncentred Route B as positive/negative tests), the size-dependent part of `K_LR_ii`
against `-alpha_M / L`, active-state count versus size, `dq` spread, zero total force to
the numerical floor.

Sparse path: CSR `H0` from the base neighbour list; below-slice count integer from a
one-off SP2 purification at a mid-gap chemical potential (trace within 0.05 of an integer)
or sparse LDL^T inertia, then tracked; Chebyshev-filtered subspace iteration or LOBPCG for
`|Q| + k_buffer` states around the two chemical potentials, warm-started (vectors,
chemical potentials, `dq`); certified Fermi-tail bounds on omitted energy, charge and force;
electrostatics by PME or FMM at every size beyond the dense regime (plain Ewald is not
acceptable for the scaling claim), one evaluation per SCF iteration plus one geometry-only
`W` evaluation per frame; dense fallback above `max_active` states.

Gates: dense and sparse agree to force tolerance at 159 atoms and on every ladder cell;
wall time versus base at 79, 159, ~1k, ~5k, ~20k, 50k atoms with the regime (isolated gap
state, bounded active set) stated explicitly.

**2x benchmark definition.** Wall time per MD step for energy and forces on registered
hardware (one named GPU/node), fixed batch size, fixed `tol_q`/`tol_E`, warm-started along
a registered trajectory, including recomputation of the base's early feature block for
derivatives, compared with the base alone under identical conditions. Report median and
95th percentile; both must be <= 2x.

---

## 9. Phase 5 — Deferred register (do not build without a named observable)

Route C (parameter-free host coupling with the reference-fill response and a bare kernel);
first-order occupied-space response correction; occupation-dependent on-site term for
reference-carrier coupling (negative states, partial removals); low-rank orbital
self-interaction correction only if beyond-PBE labels are ever used; learned short-range
pair kernel `g(r)`; induced dipoles and site multipoles; isolated-boundary postprocessor
(model-charge lattice term plus potential alignment, compactness checks); source-wide
energy gauge replacing `C_Q`; DFTB3 hardness (frozen to external priors if ever used);
spin term; spin–orbit coupling; electronic-structure pretraining of `H0`; defect-agnostic
feature-space coverage rule (§6 open item).

---

## 10. Rules

- Identity and derivative gates are never traded against predictive gains.
- Every threshold is written down before the corresponding result is opened; thresholds
  chosen afterwards are labelled retrospective.
- No defect position, defect type, oxidation state, manually designated site, stratum key
  or active region reaches the model.
- Charged energies are admitted per size and per fold by the registered coverage and
  out-of-fold base-support criteria (§6); charged forces are admitted at every size; the
  excluded size's energy residual is reported as a prediction. No per-size energy
  constants; no per-atom scaling of the charged residual.
- No literature or database charges, oxidation states or Born charges; charge-like
  learnables are initialised from the model's own reference fill; external parameter sets
  set bounds only; the host-specific physical inputs are `eps_inf` and `E_gap`.
- Route B patterns are centred per cell; an uncentred pattern is a bug.
- Kernel components are named `K_SR`/`K_LR`, never "image"; the split is a kernel
  definition, not a boundary claim; `m_sw` or `f_SR` is reported in every arm.
- The root rule is single-valuedness under registered initialisations, never "lowest
  root"; `rho` is reported in every arm.
- No isolated or boundary-complete outputs from this model.
- SCF iterations logged and capped; electrostatics by PME/FMM beyond the dense regime.
- Head arithmetic in float64; all conventions serialised; a checkpoint without its exact
  calibration record has no calibrated charged energy.
- The model is described as a 0/+1 D-SCC MVP and foundation stepping stone; multi-carrier
  support is claimed as structural only.

---

## 11. Registered values, defaults, confirmations

Registered: Gaussian smearing `sigma_s = 0.05 eV`; label code VASP with neutralising
background (the electron-count-linear alpha-Z term is intensive and size-independent, hence
one `C_Q` across sizes); `n0`: Cs 1, Pb 4, Cl 7; `eps_inf = 4`; `E_gap = 2.40 eV`; base
exposes `l=1` features only, with position derivatives through the recomputed first block;
`U_max[Z]` from GFN1-xTB hardness (bounds only); neutral labels spin-polarised,
majority-channel default; 79-atom cell `L_min/2 = 5.6 Å`, vacancy-spanning Pb–Pb distance
range 4.8–7.1 Å (non-separability registered); neutral 79-atom frames end at `d = 6.0 Å`,
charged frames run to 6.8 Å.

Defaults to register before use: regime A `r_d1 = 3.2 Å`, `r_d2 = 3.6 Å`, placement-check
floor 1e-3, `m_sw` floor; regime B `r_s = 6.5 Å`, image range for the `K_SR` lattice sum,
`f_SR` floor; `r_g`, `r_split`; `lambda_0`, `lambda_max`, `n_max`, `Z_max`, `tol_root`,
`rho` ceiling, continuation step schedule; `s_tol`, `z`, `n_min`, coverage bin width; the
out-of-fold base protocol identifier used for `s0`; the gap-regulariser convention in
force; benchmark hardware, batch size and trajectory; every arm threshold in §7.

Confirm before Phase 0: the smearing type and width from the label paper's methods;
whether `2.40 eV` is the static-lattice or thermal-average PBE gap.
