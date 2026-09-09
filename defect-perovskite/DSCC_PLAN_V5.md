# D-SCC programme v5 — implementation outline

*Received from the user 2026-09-09 (evening), verbatim.*

Six workstreams. W0 first; W1 and W2 in parallel; W3 needs both; W4 and W5 branch from
W3; W6 needs the W4/W5 outcome. No new DFT anywhere. The base sees neutral frames only.
Every threshold written before the corresponding result is opened. F-SCC finals (old base,
minimum-image centre) stay a separate record.

## W0 — Registration and evaluation protocol
- Shell centre: vacancy-side rule (commit 3a6a05c) everywhere; old files compared only
  with old files.
- C11 re-registered: near = 2–4 Å (flanking Pb, first-shell Cl reported separately),
  mid = 4–8, far = 8–10, 10–12, > 12 Å. Far-field gate reads the B′ gain on the full
  per-atom RMSE and on 2–4 / 4–8 Å against the same-coupling Route A arm.
- Statistics: per-component units throughout (`tau_phys` = 1.7 meV/Å per component);
  comparisons paired by fold (`fold = seed mod 4`); two one-sided tests against `±tau`:
  superiority / equivalence / inconclusive, simpler arm chosen only provisionally when
  inconclusive. Evaluation on the epoch-averaged checkpoint (last 10 epochs) —
  evaluation change only; optimiser unchanged.
- Kernel convention (regime B), stated explicitly: E_PBC includes own-cloud
  self-interaction, G = 0 removed by the background; `K_SR_ii = s(0) + Σ_{L≠0} s(|L|)`,
  `s(0) = 1/(√π r_g) − 2/(√π r_s)`; hence `K_LR_ii → 2/(√π r_s)` as L → ∞ (≈ 0.6 eV
  after eps_inf). Any implementation must reproduce this diagonal.
- Extrapolation-uncertainty proxy (new, for W1/W5): in the charged d-window, the
  disagreement between the MH foundation head and the fine-tuned head on neutral-state
  energy and forces at charged-frame geometries, plus the cross-fit spread; thresholds
  `u_E`, `u_F` registered now. Recorded as a proxy, not a bound.

## W1 — Base v2: multi-head fine-tune of MACE-MH-1 (neutral data only) — corrected
- Architecture: fixed by the MACE-MH-1 checkpoint (ScaleShiftMACE, r_max 6.0 Å, two
  RealAgnosticResidualNonLinear interactions, hidden 512x0e+512x1o, edge 128x0e+128x1o,
  max_ell 3, correlation 3, Agnesi distance transform, pair repulsion). Not modified.
  Receptive field 12 Å exceeds the 11.1 Å cell thickness: the base sees the periodic
  image, which matches the labels and is intended.
- Procedure: follow the multi-head fine-tuning guide in the MACE documentation
  (fetch the current page before starting and record its version). Use the foundation
  checkpoint with multi-head fine-tuning enabled, the replay set selected and sized as
  the guide recommends, and the guide's recommended handling of E0s for the fine-tuning
  head, learning rate, epochs, batch size and energy/force weights. Do not deviate from
  the documented defaults without a registered reason.
- Data: identical neutral sets (1616 train / 154 val) for the production base and the
  same 4-fold cross-fit splits for the out-of-fold bases. No charged frame enters any
  base training.
- Register before training: MACE version/commit, foundation checkpoint hash, the exact
  command line, replay selection and size, E0 handling, all optimiser settings, seeds.
- Feature export for the head: record which tensor is used (layer, irreps, dimension —
  the node features after the first interaction are 512x0e+512x1o, so the head's
  feature-modulation readouts and the rank-1 descriptor must be re-dimensioned), and
  confirm position and strain derivatives are available. Note for W2 item 1: the larger
  channel count makes the shared derivative graph more important, not less.
- Proxy inputs (W0): both heads evaluated at charged-frame geometries as neutral-state
  predictions; record how each head is selected at inference.
- Gates unchanged: validation force/energy RMSE ≤ current base (11.8 meV/Å per
  component, 4.9 meV/atom); out-of-fold neutral floors by shell (vacancy-side centre)
  at 79 and 159; `s0(L) ± SE`, coverage and proxy tables per size; per-size, per-fold
  admission decision recorded before W5 opens.

## W2 — Efficiency (no change to fixed points, energies, forces or gradients)
Order by expected impact; each item gated by agreement with the previous implementation
(energies 1e-9 eV, forces 1e-7 eV/Å, stress 1e-7 eV/Å³ on a registered frame set) and
by re-running the Phase-1 gates. Report measured speedup per item.
1. Shared derivative graph: base features computed once in the base forward and kept in
   the graph (base parameters frozen, coordinate/strain derivatives retained); inference
   forces from one backward of `E_base + J_stat`, with `J_stat` built from the converged
   `δP`, `q` as detached constants (envelope theorem); training keeps the implicit
   response derivative.
2. Frontier-only evaluation: skip the unaffected spin channel entirely; in the affected
   channel form `Δf_a` from the eigenvalues, active set `|Δf_a| > tol_f` (registered,
   1e-10), `δP = Σ_a Δf_a |a⟩⟨a|`, `q_i = −Σ_a Δf_a ‖Π_i a‖²`, HF force
   `Σ_a Δf_a ⟨a|∂H0/∂R|a⟩` on the sparsity pattern; band and entropy sums over all
   eigenvalues (O(N)); Newton Jacobian from the active levels' first-order response
   (dense: full eigenbasis; sparse path later: Sternheimer/Krylov, matrix-free).
3. Direct reciprocal `K_LR` for λ_dir = U_eff = 0:
   `K_LR_ij = (4π/Ω) Σ_{G≠0} e^{−G² r_s²/4}/G² cos(G·r_ij) − π(r_s² − 4 r_g²)/Ω`
   (Coulomb prefactor and 1/eps_inf outside). Evaluate through structure factors
   `S(G) = Σ_j dq_j e^{iG·r_j}` with phases precomputed once per frame: potential,
   energy `(2π/Ω) Σ c_G |S(G)|² − ½ κ Q²`, forces via `∂S/∂R_j`, stress via `∂/∂h` of
   `G`, `Ω`, `c_G`. Same treatment for `Γ_LR` with the combined width. Tests: direct
   formula vs `E_PBC − K_SR` including the diagonal to 1e-12 on a real frame;
   reciprocal-cutoff convergence (energy 1e-10, forces/stress 1e-8) replaces the
   Ewald-parameter test; `K_LR_ii` limit as in W0.
4. Uniform mode: `Γ = Γ̃ + κ(h) 11ᵀ` with `κ = −π(r_s² − 4 r_g²)/Ω` handled
   analytically: SCF on `Γ̃` (PSD), energy `½ κ Q²` restored, stress from `∂κ/∂h`,
   Newton updates projected onto `Σ δq_i = 0`. Reported eigenvalues carry the uniform
   shift back for cross-cell diagnostics.
5. Per-frame precompute: `H0` (with feature modulation), `W = Γ_LR (s q0)`, phases,
   `∂H0/∂R` sparsity structure — once per frame, never per SCF iteration.
6. B′ pair route as one Fréchet contraction: `A = Uᵀ[Σ_i (s Γ_LR dq)_i Π_i]U`,
   `B = F ∘ A`, `C = U B Uᵀ`, `F_J = −Tr(C ∂H0/∂R_J)`; profile before and after.
7. Eigensolver backend benchmark (CPU LAPACK across the batch vs cuSOLVER) at 316 and
   636 orbitals; batch frames by cell size; float32 pre-iterations with float64 final
   convergence as an optional last item.
8. Inference path: warm start along a trajectory; registered 2× benchmark (§8
   definition) on Φ = 0, LR-only and B′ LR-only after items 1–6; XL-BOMD recorded as a
   later item.

## W3 — Head baseline on base_v2 (previous strategy: forces only)
- Arms: Φ = 0, LR-only, B′ LR-only; six seeds; same folds; W2 code; epoch-averaged
  evaluation; paired TOST against each other and against the old-base runs on the same
  folds.
- Gates: C5 bound-state precondition on base_v2 (Arm-1-style check, thresholds
  unchanged); root rule on final model and last 10 epochs; by-shell tables under C11.
- Readout: whether the base change moved the 2–4 Å flanking-Pb residual at 79 atoms at
  matched d (D15 comparison repeated on base_v2).

## W4 — Capacity: bounded environment-dependent rank-2 coefficient
- `b_i = b_Z · (1 + β tanh(g_Z(h_i) − g_Z(h̄_Z^pris)))`, bounded, pristine-centred,
  from the early-block invariants; existing geometric `Q_i`. Factorial: {scalar-only,
  +rank-2 (species b), +rank-2 (b_i(h_i)), +rank-1, full}; Φ = 0; six seeds.
- Entry: C5 precondition must hold for any variant selected.
- Metric (registered, from D15): paired 79/159 flanking-Pb residual at matched d
  (0.2 Å bins), short-d bins (< 5.2 Å) reported separately; TOST reading; no regression
  in overall force RMSE or localisation.
- Selection: simplest variant that keeps the bound state and is not inferior; b_i(h_i)
  adopted only on superiority.

## W5 — Energies in the loss (frozen base_v2)
- Admission per size and fold by the W0/W1 rule (in-range `|s0| + z·SE ≤ s_tol`,
  coverage or proxy below `u_E`, `u_F`); frames outside remain excluded; decision
  recorded before training.
- Loss: registered strata, total-cell eV residuals with `+C_Q`, quadratic loss,
  closed-form `C_Q`, forces everywhere; capacity from W4 if adopted.
- Metrics: forces (TOST vs the forces-only twin on the same folds); energy shape at each
  admitted size and the 159-atom shape as prediction where not admitted; between-size
  residual mean after one `C_Q`; localisation (`N_eff`, separation, precondition);
  leak readout (head d-slope 79 vs 159, primary guard); model tiling ladder 1/L.
- Gate "energies do not compromise": force RMSE equivalent to forces-only within tau;
  localisation unchanged within registered tolerance; leak readout within margin.
  Failure → forces-only model retained, energies reported as diagnostics.

## W6 — Simplified fast model (SCF-free)
```
E = E_base + ΔF_band(H0; N_S, N_ref) + E_M(Q; h) + E_host + C_Q
ΔF_band : fill of H0 (one eigh, both counts, smearing as registered), HF forces
E_M     : point-charge Madelung energy of the registered background convention for the
          actual cell h, divided by eps_inf; stress only, no forces
E_host  : dq(H0 fill)ᵀ Γ_LR (s q0(H0 fill)) — non-self-consistent; forces need the two
          Fréchet contractions (∂dq/∂R, ∂q0/∂R) from the same eigh
```
- One eigendecomposition per frame; cost ≈ Φ = 0 plus two contractions.
- Comparison against the W3/W5 winner on the same folds: forces (TOST within tau),
  159-atom shape (post-hoc `C_Q`), localisation (`N_eff` expected lower without
  self-consistent reshaping), tiling ladder (E_M exact by construction; host term as
  before), 2× benchmark.
- Adopt only on: force equivalence, shape not inferior, ladder pass, benchmark met.
  Optional registered extension: analytic quadrupole correction from dq's second
  moment if the ladder shows a resolvable 1/L³ term.

## Record and boundaries
- Base v2 trained on neutral frames only; charged frames never enter base training.
- All new rules (proxy admission, epoch-averaged evaluation, C11 shells, W4 metric)
  registered in W0 before any W3+ result is opened; anything changed later is labelled
  retrospective.
- Deferred and unchanged: Route C / first-order response, induced multipoles, isolated
  postprocessor, DFTB3, SOC, LR schedule and spike handling (end-of-programme pass).
