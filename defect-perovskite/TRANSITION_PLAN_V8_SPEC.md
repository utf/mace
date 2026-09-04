# Transition plan v8: single-functional charge architecture

Received 4 Sep 2026, verbatim, as the plan of record. Supersedes the cycle spec
(`CYCLE_SPEED_ARMS_SPEC.md`), whose report is `CYCLE_SPEED_ARMS_REPORT.md` on
`size-extensivity` at 4ad038f. The occupation/SiC design that spec's §8.5 asked for is
absorbed by §2.1 below (`ElectronicStateSpec`, occupation-policy dispatch) and is not written
separately.

Applies to the existing codebase. The current Stage 0–6 order is fixed; no stage starts until
the previous stage's acceptance tests pass. Every trained number carries its regime tag.

## 0. Principles (frozen)

1. The Stage 0–6 charge evaluator uses one differentiable energy functional; every
   density-dependent term contributes δE/δP to H, and forces and cell derivatives come from
   the same object.
2. The formal total monopole is exact from counting; effective charges may redistribute it,
   never change it. The core/frontier split of that monopole is a composition-level integer.
   Electronic state is a boundary condition separate from formal charge. The head is exactly
   null only for the designated electronic reference state.
3. Same-channel frontier electrostatics uses G_img = G_PBC − G_∞ only; no isolated classical
   self-repulsion; no U term.
4. Scalar static electrostatics is range-separated from the learned local Hamiltonian.
5. Rank-1 s–p and rank-2 traceless p–p on-site channels from a damped field basis with
   bounded, centred scalar coefficients.
6. Frontier response is solved to stationarity and is never detached.
7. Explicit polarisation last, constrained by the computed pristine dielectric response.
8. Independent lengths: σ_Ewald (numerical), r_split, r_orb[Z], r_res (model).
9. No defect position, oxidation-state label, species "normal geometry", or softmax
   localisation target in the model, loss or training. Per-host inputs: pristine geometry,
   E_gap, ε∞, label smearing. The pristine geometry is also used, as a periodic density, in
   the isolated-gauge transform (§2.1), and, in the composition-class constructor only, to
   establish integer counts; the union/ghost basis and any site correspondence built there
   never enter H^PBC, the loss, forces, training or inference. An occupation policy is an
   electronic boundary condition, not a learned graph embedding or atom/defect label.
10. The Stage-2 experiment carries the pre-registered criterion of §8.

## 1. Inventory

Retained unchanged: counting head; s+p SK assembly; four learned decay lengths; log
modulation ln 1.5; divided-difference density response; per-site charges
Z_i = Z0[s] + ζ·tanh(z(x_i) − z(x̄_s)) with Σ_i δZ_i = 0 and baselines neutral on the pristine
composition; argument-centred scalar on-site correction; frozen cached neutral base with
drift guard; two-stage protocol; pristine-gap constraint; Gaussian smearing at the label
width; reference-state-null-gated energies (the current reference remains Q = 0); large-cell
upweights; nulls, F4, dilution, tiling and
pinned-continuum tests; batched eigensolve and Ewald precompute.

Transformed: Madelung term → V_static (Stage 4); E_LR → Φ_FF (un-detached, kernel unchanged,
parameters retired); one-shot compensation → V_F inside frontier stationarity (Stage 5);
c per (charge, size) → single C_Q (Stage 5); Harrison init at d_ref → at actual bond length.

Retired: detached densities; the depth sigmoid; E_resp; per-host constant defaults;
species-projected n_occ; instantaneous spectral counting for Q_core; hard spectral inclusion
rules for the frontier density; signed normalisation of a mixed electron/hole density; any
rule moving charge between ions and carrier as a function of a spectral variable; a separate
ρ_res; non-reference charge energies without a same-size reference-state null; joint base
training.

Interface-only in the current programme: `ElectronicStateSpec = {state_id, Q_formal, ΔN_σ,
occupation_policy, occupation_payload}` and an occupation-policy dispatch. The only executable
production policy through Stage 6 is the existing count-filled policy with an empty payload;
the dispatch must call the exact legacy path so that chemical potentials, occupations,
density matrices, band free energies, energies, forces and stresses are numerically
bit-identical. Legacy checkpoints load with deterministic defaults; cached values must agree,
although the cache-key and checkpoint schemas are intentionally versioned.

## 2. The functional

### 2.1 Charge inventory and electronic-state interface (label-free)

Electronic-state boundary condition (interface active in Stage 0; alternate policies deferred).

    ElectronicStateSpec S:
        state_id             bookkeeping/logging only; excluded from physical equality
        Q_formal             exact formal charge
        ΔN_σ                  exact per-spin electron-count changes relative to S_ref
        occupation_policy    serialised policy key
        occupation_payload   empty in the current Stage 0–6 production programme

    S_ref                   designated electronic reference for the composition class
    current production S    occupation_policy = count_fill, occupation_payload = empty

The state specification contains physical counts and an occupation policy. Define its
canonical physical key as (schema version, Q_formal, ΔN_σ, occupation_policy,
occupation_payload). The bookkeeping `state_id` is excluded. The identifier and policy name
never enter H, an atomic feature or a learned network. Equality of the physical key to that of
S_ref, rather than Q_formal = 0 by itself, defines the exact head null. The current adapter
from the existing charge/spin counters to S calls the legacy count-fill path exactly.
Define Q_ref = Q_formal(S_ref) and ΔN_σ(S_ref) = 0. For every same-composition state,

    Tr[P_S,σ − P_ref,σ] = ΔN_σ(S)
    Q_formal(S) − Q_ref = −Σ_σ ΔN_σ(S).

Equal formal charge and equal ΔN_σ do not in general identify an occupation policy's density
matrix. Alternate policies are not active in Stage 0–6, and no physical alternate-state
evaluator is defined in this plan.

Static part.

    Z_i            static per-site charges on present ions (retained form)
    ρ_Z^present    = Σ_i Z_i g(r − R_i)                       (present atoms, frame positions)
    ρ_Z^pristine   = Σ_j Z0[s_j] g(r − R_j⁰)                  (pristine reference tiled to the
                                                             frame's supercell; a periodic density)
    ρ_static^raw   = ρ_Z^present − ρ_Z^pristine               (density difference; no assignment)
    q_raw          = ∫ρ_static^raw = Σ_i Z_i                  (zero on any pristine cell)
    ω_i            = |δZ_i| / Σ_j |δZ_j|,   g_res(r) = Σ_i ω_i g(r − R_i; r_res)
    ρ_static^def   = ρ_static^raw + (Q_core − q_raw)·g_res     ∫ = Q_core  (single mechanism)

Composition-level counting (integers; established once per composition class; cached).

    A composition class is the multiset of species in the cell together with its pristine
    reference. The class reference geometry is the ideal defect geometry built from the
    pristine cell when constructible, otherwise the class's first frame.

    Tier 1 (direct edge count), on the class reference geometry:
        VBM_al       pristine valence edge aligned to the cell by the continuum manifold
        M_VB,σ       = #{ε_k(H_0) ≤ VBM_al + δ},   δ = 2·(smearing width)
        Accept if no eigenvalue lies within ±δ of VBM_al and the pristine gap ≥ 4·smearing.

    Tier 2 (valence-subspace continuation), if Tier 1 is ambiguous:
        1. Site correspondence between the class reference geometry and the tiled pristine
           cell by automatic minimum-cost assignment on positions; flag if any matched
           displacement exceeds r_match (config). Unmatched pristine sites are ghosts
           (vacancies), unmatched reference atoms are additions (interstitials), species
           mismatches are substitutions. The correspondence is used only here.
        2. Union basis: physical orbitals of the reference geometry plus ghost orbitals for
           pristine sites absent from it (and for added atoms, present at the pristine end).
        3. Interpolate from H^(0)_union (pristine; added-atom orbitals decoupled and sunk at
           ε = +E_sink) to H^(1)_union (class reference; ghost orbitals decoupled and sunk)
           along at least two geometrically distinct paths:
             Path A: simultaneous fade of the affected hoppings and motion of the affected
                     on-site levels to/from the sink; substitutions by simultaneous
                     alchemical switching of species parameters.
             Path B: removals: fade the affected hoppings to zero, then move the decoupled
                     on-site levels to the sink; additions: bring on-site levels from the
                     sink first, then couple hoppings; substitutions: decouple the site,
                     swap species parameters, recouple.
           On each path, two step schedules (Δλ and Δλ/2) as a transport-convergence test.
        4. Parallel-transport the pristine occupied valence projector P_V^(0), rank M_V^(0),
           by maximum subspace overlap over eigenvectors of H at each step; track the
           subspace, never individual eigenstates.
        5. Endpoint classification, basis-invariant: with P_ghost the projector onto ghost
           orbitals and P_V the transported projector at λ = 1, compute the eigenvalues
               γ_a = eig( P_V P_ghost P_V ) restricted to ran P_V.
           γ_a < η → physical;  γ_a > 1 − η → ghost;  η ≤ γ_a ≤ 1 − η → closure.
           M_VB,σ^class = #{a : γ_a < η}.
        Accept if: both paths give the same M_VB,σ^class; no closure eigenvalue occurs on
        either path; the two step schedules agree on each path; and the physical part of
        the transported subspace has overlap > 1 − η with a spectrally contiguous set of
        eigenvectors of H^(1). Otherwise the class is genuinely ambiguous: it carries no
        core/frontier decomposition and the isolated gauge is unavailable for it. No
        override input exists.

    From either tier:
        n_e,σ^class  = max(N_σ − M_VB,σ^class, 0),   n_h,σ^class = max(M_VB,σ^class − N_σ, 0)
        Q_core       = Σ_σ (n_e,σ^class − n_h,σ^class)
    No site-resolved oxidation state or orbital count is formed at any point. The integers
    are cached with the tier used, the path and schedule agreement, and the endpoint
    eigenvalue spectrum γ_a, and are invariant under R, Q, cell tiling, localisation, band
    resonance, spectral motion and SCF iteration. Classes of the same defect type at
    different cell sizes must yield the same Q_core (test §7.1).

Per-frame charge counts from the counters (exact integers; current policy).

    n_e,σ(Q) = n_e,σ^class + (electrons added to σ) − (electrons removed from σ)
    n_h,σ(Q) = n_h,σ^class + (holes added to σ) − (holes removed from σ)
    q_F(Q)   = Σ_σ (n_h,σ − n_e,σ);   assert q_F(Q) = Q_formal(Q) − Q_core for every frame
    Benchmark: V_Cl⁰: Q_core = +1, n_e,maj = 1, q_F = −1;  V_Cl⁺: Q_core = +1, n_e = 0, q_F = 0.

These are absolute composition/charge-state carrier counts. They depend only on the
composition-level integers and exact carrier counters. An inactive occupation-policy payload
cannot modify Q_core, q_F or Q_formal.

Spatial frontier density (smooth; determines where the frontier charge resides, never how much).

    s_k^e  = sigmoid((ε_k − VBM_al − δ)/Δ_s),    s_k^h = sigmoid((CBM_al − δ − ε_k)/Δ_s),  Δ_s = smearing width
    ρ̂_e,σ = Σ_k f_k s_k^e |ψ_k|² / ∫(same);       ρ̂_h,σ = Σ_k (1 − f_k) s_k^h |ψ_k|² / ∫(same)
             (each a positive density normalised to 1; unused when its count is 0)
    ρ_F(P) = −Σ_σ n_e,σ ρ̂_e,σ + Σ_σ n_h,σ ρ̂_h,σ                 ∫ρ_F = q_F exactly
    w(P)   = sigmoid((p* − p)/Δp),  p = participation fraction of ρ_F
    ρ_F    = w·ρ_F,loc + (1 − w)·ρ_F,ext,  ρ_F,ext uniform with the same integral
    ρ_ind  induced dipoles (Stage 6; the reference contribution cancels exactly at S = S_ref)
    Δρ_def = ρ_static^def + ρ_F + ρ_ind;   ∫Δρ_def = Q_core + q_F = Q_formal exactly

ρ̂, w and the projectors are smooth functions of (H, P). A frontier state whose projector
weight falls below 0.5 in a frame (a level merging into a band) is logged; the class counts
are not changed by it.

The positive electron and hole channel objects are stored separately; no implementation may
reconstruct either channel from the sign of q_F. The signed ρ_F and its participation switch
above are defined only for the current `count_fill` charged-carrier construction. No
behaviour for any other occupation policy is implied.

Note: g_res and ρ_static^raw are physical densities; r_res and the ω rule are model choices
with a sensitivity test (§7.6). ρ_static^raw includes thermal-displacement dipoles of all
atoms (it vanishes identically only on the pristine reference geometry); their image
contribution is O(1/L³), random-signed, and is reported separately on the ladder (ideal
versus thermal frames).

### 2.2 Energy

In Stages 0–6, E(Q) below is shorthand for E(S_Q) under the active legacy count-filled
occupation policy. The notation is retained to make explicit that the charge functional and
its fitted quantities are unchanged.

    E(Q) = E_base + [F_band(H_Q, N_0−Q) − F_band(H_0, N_0)]
                  − [Tr(P_Q V_F,Q) − Tr(P_0 V_F,0)]
                  + [Φ_F(Q) − Φ_F(0)]  + C_Q

    F_band(H, N)  smeared band free energy consistent with the implemented occupation rule,
                  defined by ∂F_band/∂H_ab = P_ba at fixed N per spin (unit test §7.1).
                  Fermi–Dirac → Mermin; Gaussian f = ½erfc(x) → F = Σ f_k ε_k − (σ/2√π)Σ exp(−x_k²).
    V_F           = V_FF + V_ind: every density-dependent potential in H other than V_static
    Φ_F           = Φ_FF + Φ_F,ind + Φ_ind,ind + Φ_S,ind   (explicit terms; §2.8)
    C_Q           one constant per charge state, shared by all cell sizes and not indexed by
                  electronic-state bookkeeping; C_{Q_ref} = 0 (Stage 5)

Head correction ≡ 0 at S = S_ref. In the current charge programme S_ref is the count-filled
Q = 0 reference, so this is bit-identical to existing behaviour; Q = 0 alone is not the
general null condition. Frontier–static interaction is counted once, in the band term through
V_static^B. Static–static image term is charge-state independent and not in the head.

### 2.3 Hamiltonian

    H_B^0   = H_local^sp + H_onsite^(E,∇E) + H_edge^add + V_static^B
    H_B[P]  = H_B^0 + V_F,B[P]

    H_local^sp        existing SK head
    H_onsite^(E,∇E)   Stage 2: a_i(h_i) E_i·T^sp + b_i(h_i) Q_i:T^pp, Q_i = ∇E_i − ⅓(∇·E_i)I,
                      fields from the damped full-range potential φ_d = Σ_j Z_j f(r; r_orb)/r
                      (+ residual object from Stage 4); a_i = a_Z^0 + δa(h_i), b_i = b_Z^0 + δb(h_i),
                      bounded, centred on element defaults; no feature-learned tensor.
    H_edge^add        Stage 3 only: additive, short-range, bounded, centred on the pristine bond
                      environment.
    V_static^B        Stage 4 onward. Periodic gauge: V_full(Z; present ions) − V_SR(r_split)
                      + V_LR[(Q_core − q_raw)·g_res]. Isolated gauge: the same minus
                      G_img ⋆ ρ_static^def. Both gauges act on exactly the same defect-induced
                      object. Before Stage 4: the existing Madelung term.
    V_F,B             Stage 5 onward: δΦ_F/δP under kernel B, differentiated through ρ̂_e, ρ̂_h,
                      their normalisations, w(P) and the projectors — the complete P-dependence
                      of Φ_F. Zero before Stage 5.

### 2.4 Frontier stationarity (Stage 5; current count-filled policy)

    solve(H_B, S):
        iterate P ← occupations(H_B[P], S.occupation_policy) with Anderson mixing until
            ‖P_{k+1} − P_k‖ < ε_P  and  ‖[H(P_k), P_k]‖_T < ε_H
            (finite-temperature form)
    reference and charged fills both to stationarity; reference solve cached per geometry.
    Iteration counts logged per frame.

Only `count_fill` executes in Stages 0–6. Its dispatch calls the existing fill implementation
without rebuilding occupations, entropy or density matrices. The generic signature is dormant
infrastructure; no alternate occupation policy is reachable from Stage 0–6 training or
inference.

### 2.5 Derivatives

At stationarity: F = −∂E/∂R|_{P,u}, σ = (1/Ω)∂E/∂ε|_{P,u}. Before Stage 5, every term
depending on P outside the band energy contributes (∂E/∂P)(dP/dR) via the divided-difference
response: one backward on V = ∂E/∂P gives P̃; add Tr(P̃ ∂H/∂R). No detached quantity
anywhere. Every term registers: energy, δE/δP as an orbital-space potential (autograd through
its complete construction from H and P), ∂E/∂R|_P, ∂E/∂h|_P, and its gauge dependence.
Autograd may implement δE/δP; §7.2 verifies it.

### 2.6 Gauges

One functional, two kernels: G_PBC (training) and G_∞ (inference). Gauge-dependent terms:
V_static (through ρ_static^def), Φ_FF/V_FF, Φ_F,ind/V_ind, Φ_S,ind, Φ_ind,ind. Ground truth:
the tiling ladder of periodic-gauge energies (§7.5).

### 2.7 Lengths and constructor tolerances (config keys)

    ewald_sigma   numerical; predictions invariant to the numerical floor (§7.3)
    r_split       model; default = first-block cutoff
    r_orb[Z]      model; element table; independent of ewald_sigma
    r_res         model; width of g_res and of the density-difference Gaussians (§7.6)
    r_match       constructor; site-correspondence rejection distance (default: half the
                  pristine nearest-neighbour distance, evaluated per host)
    E_sink        constructor; ghost/addition on-site sink energy (default 50 eV above the
                  pristine conduction edge); convergence test §3
    eta           constructor; endpoint classification threshold (default 1e-3)
    dlambda       constructor; base step; halved for the schedule test

### 2.8 Interaction matrix (every pair in exactly one row)

    pair     kernel                       energy location                    H derivative   double count
    S–S      G_B                          not in head (cancels in ΔE)        none           none
    S–F      G_B; scalar range-separated, vector/tensor damped full-range
                                          band term via V_static^B           V_static^B     none
    F–F      G_img on w·ρ_F,loc           Φ_FF = ½ B_img[wρ_loc, wρ_loc]     V_FF = δΦ_FF/δP   −Tr(P V_FF)
    F–ind    G_B full                     Φ_F,ind = B_B[ρ_F, ρ_ind]          V_ind = δΦ/δP  −Tr(P V_ind)
    S–ind    G_B full                     Φ_S,ind = B_B[ρ_static^def, ρ_ind]  none (u stationary)  none
    ind–ind  A (Stage 6)                  Φ_ind,ind = ½ uᵀAu                 none (u stationary)  none
    B_img[ρ_F,ext, ·] := 0 (band electrons have no isolated counterpart; stated choice).

This table is complete only for the current `count_fill` charge evaluator. This plan assigns
no energy, electrostatic or response functional to an alternate occupation policy.

## 3. Stage 0 — registry and harness (no physics change)

- Term registry per §2.5 and two-kernel interface per §2.6; compensation and depth-sigmoid
  paths deleted; E_LR density un-detached, parameters frozen at ε∞-only values.
- Introduce immutable, serialised `ElectronicStateSpec`, `S_ref` and the occupation-policy
  dispatch. Register `count_fill` as the only production policy in Stages 0–6. The default
  dispatch must invoke the exact legacy implementation and be bit-identical in chemical
  potentials, occupations, P, band free energy, total energy, forces and stress on fixed
  pristine, V_Cl⁰ and V_Cl⁺ batches. Geometry-only electronic caches are forbidden: every
  cache key includes the canonical physical state key and occupation-policy implementation
  version; the bookkeeping `state_id` is excluded.
- Add a test-only mock occupation policy on small synthetic Hamiltonians; it is unreachable
  from production inference and training. Verify schema validation, exact per-spin
  electron-count/formal-charge consistency, policy dispatch, cache-key separation, canonical-
  physical-state reference-null semantics, and the matrix-level ∂F/∂H_ab = P_ba contract for
  a known occupation vector. Assign the mock no physical state, donor/acceptor interpretation,
  geometry-continuation rule or production force/stress claim. Reject inconsistent charge,
  spin-trace or occupation inputs.
- Finite-difference harness (§7.2), forces and strain, per term and assembled.
- Composition-class constructor (§2.1), both tiers, with the site-correspondence and
  union-basis code confined to the constructor module and unit-tested for absence from every
  production call path (an AST test, as for the protocol module). Per-frame counts from the
  counters; smooth edge projectors; separately normalised channels.
  Unit tests, Tier 1: pristine class (all counts 0, Q_core = 0, ρ_F = 0); V_Cl class at 79
  and 159 atoms (n_e,maj = 1, Q_core = +1, identical across sizes); Q = +1 frames (n_e = 0,
  q_F = 0); ρ_static^raw ≡ 0 on the pristine reference geometry; ∫ρ_static^raw = 0 on
  thermal pristine frames, with ‖ρ_static^raw‖ → 0 continuously as the displacement from
  the reference is scaled to zero; ∫ρ_static^raw = Σ_i Z_i on every frame; a synthetic
  class with a level inside ±δ of VBM_al triggers Tier 2. A parameterised synthetic class
  with m > 1 explicitly occupied frontier levels verifies n_e = m and Q_core = m; when its
  synthetic counter state has Q_formal = 0, it verifies q_F = −m and ∫Δρ_def = 0. The
  expected integer is defined solely by the toy spectrum and electron count; it assigns no
  value to any physical composition class.
  Unit tests, Tier 2: on the V_Cl class, Tier 2 reproduces Tier 1 exactly on both paths and
  both schedules; synthetic mixed-valence Hamiltonians in which (i) two same-species sites
  exchange charge character along λ and (ii) individual d-like eigenvalues cross inside the
  valence manifold give unchanged M_VB and Q_core on both paths; a synthetic interstitial and
  a synthetic substitution give the expected integers on both paths; a synthetic genuine
  valence/frontier closure raises the ambiguity flag with either a path disagreement or a
  closure eigenvalue reported; the endpoint classification is invariant under random
  unitary rotations within the transported subspace; the class integers and endpoint γ
  spectrum are unchanged when E_sink is increased by 4× and 16× and when dlambda is halved.
- Band-functional finite differences for production `count_fill` and the test-only mock
  policy's matrix-level contract (§7.1).
- Config round-trip covers r_split, r_orb, r_res, ewald_sigma, δ, Δ_s, r_match, E_sink, eta,
  dlambda, w parameters, a/b bounds, gauge flag, SCF tolerances, C_Q mode,
  `ElectronicStateSpec` schema version, occupation-policy key and payload, S_ref, and the
  cached class integers with their tier, path/schedule agreement and γ spectrum; every
  non-parameter float serialised.
Acceptance: harness runs; the default policy is bit-identical to v6; the production band
functional passes the registered finite-difference tests; the mock policy passes only its
interface/matrix contract; other terms' status is recorded; and `count_fill` is the sole
policy reachable from every Stage 0–6 configuration.

## 4. Stage 1 — formalise the existing functional; the SR/LR diagnostic

- Assemble E per §2.2 with V_F = 0; Φ_FF with the channel-normalised ρ_F; response-density forces.
- FD tests per term and assembled: pass required, including the continuity frames of §7.2.
- Diagnostic, forward-only then retrained (six seeds each): Madelung term (a) as is,
  (b) short-range part removed at r_split = first-block cutoff, (c) removed entirely.
  Report F4, depth, participation ratio, R_bound, 79-atom force loss, learned static charges
  under (a) and (b). Charges logged, never targeted. Select the Stage-1 reference by the gates.
- Tiling ladder baseline (§7.5), periodic gauge.

## 5. Stage 2 — tensorial on-site block (the decision experiment)

- Implement H_onsite^(E,∇E) with one damped primitive for φ, ∇φ, ∇∇φ.
- Toys (§7.1): sign derived from ⟨p_a|−eφ|p_b⟩ with the code's conventions, then frozen;
  rotational covariance under random rotations.
- Stage 2 retains the scalar-static configuration selected as the Stage-1 reference; no new
  scalar range separation until Stage 4.
- Six seeds head-only on the A′ base, Stage-B objective. Readouts per §7.4 and §8.

## 6. Stages 3–6

Stage 3 (conditional): H_edge^add only if the pp stop persists after Stage 2; its own
bound-saturation test is a gate for Stage 3.
Stage 4: V_static^B per §2.3 with r_split = first-block cutoff; residual object in the LR
channel; ρ_static^def for the isolated gauge. Tests: §7.3; §7.6; class-count invariance;
pristine identities. Six seeds; gates.
Stage 5: frontier stationarity (§2.4) with the complete V_F; single C_Q; per-size c retired.
Tests: FD sweep in h and (ε_P, ε_H); ladder in both gauges with fitted leading exponents;
w and Δ_s steepness-insensitivity; previous cohort's c's reported against C_Q with the
ladder intercept attributed (base coverage), not absorbed.
Stage 6 (last): induced dipoles p_i = α_i(E_ext + Σ_j T_ij p_j), Thole damping, bare Coulomb,
α_Z^0 first, δα(h_i) bounded later; the model's pristine-cell response under a uniform field
(tin-foil) reproduces ε∞; species partition regularised, reported as unidentified. Adopt only
on held-out improvement, a removed cell-size error, or a stabilised charge response.
Stress support: every stage returns the analytic total stress from the same energy functional
used for energies and forces. The current reference data contain no stress labels, so the
stress-loss weight is zero and no claim of reference-stress accuracy is made. Analytic stress
is checked against finite-strain derivatives at every stage, and the tiling ladder checks
model-only convergence. These are consistency tests, not benchmarks against DFT stress.

## 7. Tests

7.1 Identities and toys: head correction ≡ 0 at S = S_ref; the current count-filled Q = 0
reference is bit-identical to v6; Q_ref ≡ Q_formal(S_ref); on the charge path
Tr(P_Q,σ − P_ref,σ) = ΔN_σ and Q − Q_ref = −Σ_σ ΔN_σ;
∫Δρ_def = Q_formal; ∫ρ_F = q_F exactly;
q_F = Q_formal − Q_core on every frame; Q_core identical across cell sizes of the same
defect type and across tiers and paths where both apply; Σδz = 0; ρ_static^raw ≡ 0 on the
pristine reference geometry and ∫ρ_static^raw = 0 on thermal pristine frames;
translation/rotation/permutation/cell-wrap invariance; sign and covariance toys for the
on-site channels; centred-term identity on bulk-like atoms; band-functional identity
∂F_band/∂H_ab = P_ba by finite differences in H for the production smeared count fills,
both spins. A test-only mock policy with unchanged ΔN_σ but a distinct known occupation
verifies dispatch, cache separation, and that equality to S_ref's canonical physical key—not
charge alone—controls reference identity. The mock carries no physical-state interpretation
and is never exercised through the production energy, force or stress path.
Continuity: along a synthetic on-site sweep on one frame that drives the frontier level from
deep in the gap through the middle of the gap, through both projector windows and through
the conduction edge into the band, and along a geometric path between two thermal frames:
Q_core, all per-spin counts and the total monopole are unchanged; ρ_F, w, E and F are
continuous (no jump above the FD floor). An integer changes only when the counter state is
changed explicitly, and the test asserts that too.

7.2 Finite differences: central differences of the recomputed energy, per term and
assembled, h ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4} Å; pattern Ah² + ε/h with a clean minimum; an
h-independent floor is a missing derivative. From Stage 5: repeat at (ε_P, ε_H) ∈
{1e-4, 1e-6, 1e-8}; discrepancy must fall with tolerance. Frames: ordinary, near a level
crossing (selected by the model's spectrum), inside a projector window, both gauges.
Strain: homogeneous ε_αβ ∈ {±1e-3, ±3e-4} against the analytic stress.

7.3 Length separation: V_full = V_SR(r_split) + V_LR(r_split) and invariance of all
predictions under ewald_sigma ∈ {0.7, 1.0, 1.5} Å, both to the numerical floor established by
the same-model repeat procedure; B_img invariance under ewald_sigma asserted by construction.
Changes of r_split, r_orb, r_res, δ, Δ_s are model changes and are logged as such.

7.4 Hub-bond diagnostics (every stage): hopping magnitude and dt/dr on the vacancy-spanning
bond; environment-response contribution; carrier bond order; on-site crystal-field splitting
at the flanking Pb; each term's contribution to the projected Pb–Pb force (ablation);
fraction of integrals at the stop by type and seed; bound saturation of every parameter
family introduced since Stage 1.

7.5 Tiling ladder (model only): one ideal vacancy in 1×, 2×, 3× tilings, single C_Q, no
size-dependent reference; also thermal frames embedded at each size. Require: forces within
a fixed radius converge; the correction converges after switching to G_∞; total force zero;
volume-scaled defect stress converges; no discontinuity from any switch (vary w and Δ_s
steepness). Fit the leading exponent of E_PBC − E_∞ per component; monopole components ∝ 1/L;
monopole-neutral objects must move away from 1/L; nothing assumed. Report the thermal-
displacement image contribution separately. Stress convergence here is an internal model
test and is not a comparison with a reference stress label.

7.6 Model-charge sensitivity: vary r_res over {0.7, 1.0, 1.5}× default and the ω rule;
observables on the ladder change by less than the registered tolerance and the change
decreases with L. A sensitivity/convergence test of a physical density, not an invariance.

7.7 Retained gates: A′ regression; F4 within 1.5× of the A′ reference; pristine gap; pinned
continuum; dilution R against R_DFT's interval; F10 on the centred channel; participation
ratio and depth with seed spreads.

## 8. Pre-registered decision rule (Stage 2)

Hypothesis: scalar on-site p energies cannot split p_σ from p_π, so the optimiser obtains
the vacancy-directed lowering through the spanning hopping, which saturates; the rank-1 and
rank-2 channels transfer that response into the on-site block.

Readouts, all four required for Outcome A:
1. the pp stop fraction on hub bonds falls to ≤ 1/6 seeds (tests the old failure);
2. δb on the flanking Pb is active (|δb| > 3× its bulk-Pb spread) and its ablation removes
   ≥ 30% of the head's dλ/dd;
3. the flanking-Pb on-site block shows the p_σ/p_π splitting with the sign frozen in §7.1;
4. the seed spread of the participation ratio at least halves relative to the Stage-B
   cohort, at unchanged or better force fit, and neither a_i nor b_i shows systematic bound
   saturation.

Outcome A: representation was the deficiency; proceed to Stage 4.
Outcome B (2 and 3 hold, 4 fails): supervision/identifiability; freeze all electrostatic
structure, skip Stages 3 and 6, open a supervision cycle before any further head change.
Outcome C (2 or 3 fails): implementation or regularisation; re-run the toys, loosen
a_max/b_max by one step, repeat once; if still C, report and stop.
RMSE is not a criterion at this stage.

## Not in scope

For the Stage 0–6 charge programme: U/SIE terms; overlap matrices; extra shells;
three-centre expansions; a latent-Ewald neutral base; reference-stress supervision or stress-
accuracy benchmarking; and per-host tuning.

Also out of scope: every production occupation policy other than `count_fill`; state-specific
PES training or inference; donor/acceptor or active-subspace selection; spin-dependent
Hamiltonians; exchange/Hund terms; state-specific SCC; inter-state or nonadiabatic mixing;
spin–orbit coupling; and transition observables. The generic state schema, dispatch,
serialisation, cache tests and mock-policy contract are dormant infrastructure only.

Site-resolved oxidation-state assignment is neither supported nor required; mixed-valence
compositions use global valence-subspace continuation when the direct edge-count constructor
is ambiguous, and a class is excluded only when that continuation is genuinely non-unique
across distinct paths. No charge-architecture question is reopened unless a pre-registered
Stage 0–6 charge test fails.
