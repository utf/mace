# D-SCC plan: v4.1 amendment — kernel regime ruling (apply to §2.4, §7, §11)

Ruled by the user, 2026-09-06 (evening), on the Phase-0 placement-gate outcome (tracker C4).

Ruling: regime B primary. Regime A is recorded as failing its registered gate and is
retained only as a reduced ablation, not rescued by a raised floor.

Reasoning (the user's):

* The placement check was a Phase-0 gate to be evaluated on label-free geometry before any
  training, and it has been. Choosing the regime on its outcome is inside the protocol; it
  is not a retrospective threshold change. Raising the floor to 5e-2 after seeing the
  distributions would be one, and would have to be labelled as such.
* A raised floor does not fix what the gate was protecting. The tail is not uniformly
  distributed: 3.1 % pristine versus 5–7 % in vacancy frames says the bonds beyond 3.2 Å
  are concentrated on the two flanking Pb, which is exactly where `dq` sits. So the switch
  would breathe on the hub bonds, and `m_sw` would be large where it matters most. Before
  running A at all, compute the same fraction restricted to the flanking-Pb bonds; I expect
  it to be several times the cell average.
* Moving the window up does not help. Anywhere between the Cl–Cl shell (4.0 ± 0.3 Å) and
  the vacancy-spanning pair (4.8 Å) is too narrow for a C² switch with thermal tails, and it
  puts the switch back on the collective coordinate.
* Regime B has no switch and no shell-gap premise: the short-range lattice sum is smooth in
  every pair distance, treats the direct pair and its near images identically, and its
  diagnostic (`f_SR`, `r_s` sensitivity) now carries the burden the placement check carried
  in A. The physics interpretation caveat on `λ_dir` (§2.10) is unchanged.

## §2.4 — regime status
- Regime B (periodic range separation) is the primary kernel regime.
- Regime A (shell switch) FAILED its registered placement gate on the real frames at the
  1e-3 floor: first-shell Pb–Cl beyond 3.2 Å = 3.1 % (pristine), 5–7 % (vacancy frames);
  intra-octahedron Cl–Cl below 3.6 Å = 2–2.6 %; thermal first-shell distribution p50
  2.88 Å, 22 % > 3.0 Å, 3.4 % > 3.3 Å. No window in the shell gap clears the floor. The
  shell-gap premise of regime A does not hold for this lattice at 300 K; this is recorded
  as the gate outcome, not adjusted.
- Regime A is retained only as a reduced ablation arm. If run: the floor used is
  recorded as retrospective (5e-2); the placement fraction is additionally reported
  restricted to the bonds of the two flanking Pb; `m_sw` computed with the Arm-1 `dq` is
  the operative admission quantity for the arm, against its registered floor. Regime A
  results are never used for the production selection.

## §7 — Arm 2+3 factorial (replaces the 18-configuration layout)
Regime B: route ∈ {A, B} × coupling ∈ {LR-only; LR + U; full learned; λ_dir = 1 fixed}
plus Φ = 0 under each route: 10 configurations × 6 seeds. Selection as before.
Regime A ablation: full learned coupling under both routes only: 2 configurations × 6
seeds, reported with `m_sw` and the flanking-Pb bond fraction, excluded from selection.

## §11 — registered defaults
Regime B primary: `r_s = 6.5 Å` with the `r_s` sensitivity of `Phi_cc`, the level's
`d`-response and the forces reported in every arm; `K_SR` image range; `f_SR` floor.
Regime A: gate outcome recorded (failed at 1e-3); retrospective floor 5e-2 for the
ablation only; `m_sw` floor.

## Confirmations ruled at the same time
- C1 — smearing: confirmed. Gaussian, `sigma_s = 0.05 eV`, from the label paper's methods.
- C2 — `E_gap = 2.40 eV` is the **static-lattice** PBE gap: the gap regulariser acts on the
  static pristine cell; the thermal ensemble-mean gap over pristine frames is reported as a
  diagnostic (§6).
- C3 — b3 GPU 6: leave for now provided GPUs 4, 5, 7 remain accessible (they do:
  `nvidia-smi` lists 1–5 and 7 at 0 MiB); a cold power cycle only if they are not.
