# D-SCC plan: v4.3 amendment (revised) — C8 ruling; training protocol unchanged for now

## Arm 1 outcome (recorded)
- Criteria: (i) pass 0/6 both rounds; (iii) pass all seeds; (ii) FAIL both rounds
  (control loses 5.2 % / 8.0 % of the level-vs-bond slope vs 30 % registered); (iv) not
  resolved (final-checkpoint force noise ±10 meV/Å exceeds the arm difference).
- Route C applied; repeat failed; architecture growth closed: no Stage-3 edge residual,
  no bound release, no further onsite capacity.
- Criterion defects, ours: the tensor-ratio clause is descriptor-only under species-level
  coefficients (`b` cancels); the slope clause's seed spread (≈10× its SE) cannot resolve
  a 30 % difference. Recorded; not re-registered.

## Precedence ruling
"Stop" in route C stops capacity growth (original §17: optional capacity is removed only
while preserving earlier gates). The C5 bound-state precondition (registered before Arm 1
opened) is an earlier gate for Arm 2+3. Scalar-only `H0` fails it on every seed; full `H0`
passes on 9/12. The block is retained on the C5 gate, not on (ii).
Winner: full `H0`, seeds {s0, s1, s2, s3, s4, s6, s7, s9, s10}; excluded {s5, s8, s11}
(precondition 92.9 / 94.1 / 92.9 % < 95 %).
Post-hoc reading (labelled): full vs control on HOMO separation and precondition
fraction — reported as the block's demonstrated function; not the basis of admission.

## Arm 2+3 protocol (register before launch; training protocol identical to Arm 1)
- Initialisation: `h0_state.pt` of each admitted Arm-1 seed; per-frame SCF by
  continuation from Φ = 0 (v4.2).
- Training: 60 epochs, constant LR 2e-3, final-checkpoint read — unchanged from Arm 1.
  No LR schedule, no spike detection, no checkpoint averaging at this stage.
- Noise handling is statistical only: `tau_noise` for every force comparison is the seed
  spread of the Φ = 0 arms under this protocol; comparisons are made on seed medians;
  arms inside `max(tau_phys, tau_noise)` are equivalent and the simpler one is chosen.
  Arm-1 force numbers are not used as a baseline.
- Record the consequence: the ±10 meV/Å final-checkpoint noise remains in every force
  reading and favours the simpler arms at selection. Loss-spike events are logged (epoch,
  loss, RMSE before/after) but not acted on.
- Factorial unchanged (v4.1): regime B primary, route ∈ {A, B′}, coupling × 4, plus Φ = 0
  under each route; regime A as the reduced ablation.

## §9 deferred register — add
End-of-programme robustness pass on the selected model only: LR decay 2e-3 → 2e-4 over
the last 20 epochs, spike detection with rollback, epoch-averaged final checkpoint.
Retrain the selected configuration and its Φ = 0 reference under that protocol and report
whether any selection decision changes; no decision made before the pass is revisited
unless it does.

## 2× benchmark (§8)
Criterion unchanged; applied in Arm 4 on the Φ-on model. Engineering item before Arm 4:
compute base features and head features in one graph with a single backward (no
early-block recomputation); re-measure with ≥ 100 frames per size. Current reading
(2.07 / 2.22 at 79; 1.96 / 5.79 at 159, 16 frames) recorded as diagnostic.

## Scientific record
The directional block does not carry the level-vs-bond response (scalar term does). It
carries the p_σ/p_π splitting at the flanking Pb that stabilises the Pb–Pb bound state
(criterion (iii) every seed; separation 0.76–1.04 eV vs 0.08–0.68 eV in the control).
Original hypothesis (hopping saturation → directional onsite) is superseded by this
statement.
