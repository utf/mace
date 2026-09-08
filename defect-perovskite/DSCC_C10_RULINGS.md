# D-SCC plan: C10 rulings (user, 2026-09-08, verbatim)

*Saved verbatim from the user's message answering the Arm 2+3 read-out (`ARM23_COADVISOR.md`,
tracker C10). The message applies the v4.4 amendment (`K_LR` required physics, LR-only the
minimal candidate, Φ = 0 arms as force references), which had not reached the record when the
campaign ran; the user sent it at ~15:35 the same day and it is saved verbatim as
`DSCC_PLAN_V4_4_AMENDMENT.md`. The selection under it is labelled post hoc (tracker C10 / D13).*

---

Three things, in order of importance.
1. The most informative number in the report is not in the selection table. The base makes 13 meV/Å on neutral vacancy frames and ~33 meV/Å on charged frames beyond 8 Å from the vacancy, where every local environment is bulk-like in both charge states. That excess, roughly 30 meV/Å in quadrature, is force error that no local model can remove and that none of the arms you ran can either: the same-carrier terms (image, direct, `U`) put no force on an ion that carries no `dq`. The one term in the plan that acts there, Route B′, is missing from the campaign. The registered factorial was route ∈ {A, B′} × coupling; the report shows five couplings under Route A plus the regime-A ablation, and B′ is not mentioned. I would want to know why before anything closes, because the far-field floor is exactly the signature Route B′ was designed for, and it is larger than every difference the campaign resolved. One hour of diagnostics sharpens this: the by-shell residual of the base and the Φ = 0 head on the 159-atom charged frames and on neutral 79-atom frames. If the charged far-field floor at 159 atoms is neutral-like, the 79-atom excess is small-cell and manifold extrapolation; if it is ~30 meV/Å there too, it is the Coulomb channel and B′ is mandatory.
2. The campaign is a clean negative for what it tested, and that outcome was predicted. The image term is nearly geometry-independent and invisible in forces; the direct term and `U` only reshape a carrier that `H0` already places. Force-only training of a saddle functional cannot reward PBE's curvature, so `U → 0.1 eV` and `λ → 0.01` say what the forces want, not what the energy curvature is. Note that the one energy observable you have, the 17-frame 159-atom shape error, mildly favours LR + U and regime A (0.025 and 0.020 against 0.042); it is a small sample and not the registered selector, but it is the only observable that can see this physics, and it points the other way. The record should say: same-carrier electrostatics contributes nothing resolvable to forces at 79/159 atoms; the image term was not tested by this campaign; host coupling was not run.
3. The selection applied v4.3, not v4.4. Under v4.4, `K_LR` is required physics with no learnable coefficient and is not subject to the equivalence rule; the minimal production candidate is LR-only and the Φ = 0 arms are force references. LR-only is equivalent to Φ = 0 within the margin (41.7 vs 39.9, margin 3.0), passes the root rule, and its only failed gate is the defective one. If v4.4 was in the tracker before the results were opened, LR-only is the selected candidate; if it was not, the choice must be labelled post hoc, but the argument does not depend on the results and the production model should carry the 1/L term regardless, since a cross-size energy claim is impossible without it and it costs nothing in forces.
On the four rulings:

* Root rule: agree. The early-epoch failures are the initialised coupled map at `U = 1.36 eV`, above the localising threshold; the trained models are single-valued. Read the gate on the final model and the last ten epochs, record the transient, ceiling from epoch 1.
* `f_SR`: agree, and the defect is mine: in a small cell `K_LR` is self-image dominated and negative on every pair, so a signed fraction is meaningless. The intended quantity is the absolute share `Σ|dq_i dq_j K_SR_ij| / Σ|dq_i dq_j|(|K_SR_ij| + |K_LR_ij|)`. Record, do not re-register.
* Epoch 0: diagnostic; agree.
* Downstream: do not close. Run Route B′ under the same protocol (six seeds), run at least the matched-kernel F-SCC comparator on three seeds, since it has host coupling and the opposite feedback sign and is the direct test of whether the 33 meV/Å floor is screening physics, and run the §9 retrain, the ladder and the benchmark on whichever candidate survives. Closing now would publish "electrostatics does not help" on the basis of the terms least able to help.

One more thing the table says: the first-shell residual is still ~100 meV/Å against a 13 meV/Å neutral floor, and no electrostatic arm moved it. That is a local problem, `H0` capacity or base extrapolation on the charged manifold near the vacancy, and it is the other open item the paper will have to name.

```markdown
# C10 rulings (apply to tracker)
- Root rule: gate read on final model and last 10 epochs; early-epoch failures recorded
  as the initialised-map transient; ceiling from epoch 1.
- f_SR: criterion defect (signed fraction unbounded in small cells); recorded; replaced
  for reporting only by the absolute share Σ|dq_i dq_j K_SR_ij| / Σ|dq_i dq_j|(|K_SR|+|K_LR|).
- Selection: v4.4 makes K_LR required and LR-only the minimal candidate; LR-only
  selected (equivalent to Φ = 0 within margin, passes all non-defective gates). Label
  post hoc if v4.4 postdates the campaign launch.
- Not closed. Remaining arms: Route B′ (six seeds, same protocol); matched-kernel F-SCC
  (three seeds). Prerequisite diagnostic: by-shell base/Φ = 0 residuals on 159-atom
  charged and 79-atom neutral frames. Then §9 retrain, ladder and benchmark on the
  surviving candidate.
- Record: near-field (first-shell) residual ~100 meV/Å unmoved by any arm; open item.

```

---

*C10 addendum (user, 2026-09-08 ~15:50, verbatim). The phrase "subject to the four checks
above" refers to text that did not reach this session; the four checks are not on file.*

# C10 addendum (record in D13)
- B′ ladder selection gate: not waived, re-read. It cannot pass on cells with L_min < ~3 R_c;
  recorded as a model property (compensation cloud 8–11 Å), not a code fault, subject to
  the four checks above. B′ is judged on forces; no cross-size energy claim includes E_SF
  until a ladder with L >> R_c exists (sparse solver). Double-screening caveat recorded.
- B′ queue: trim to LR-only and LR + U (the far field is periodicity-suppressed at these
  sizes; the remaining arms cannot add resolvable information).
- Far-field gate: read the B′ gain against the same-coupling Route A arm on the full
  per-atom RMSE and on the 0–2, 2–4, 4–8 Å shells at 79 atoms, beyond the Φ = 0 seed
  spread; the 4–8 Å pooled reading is reported, not decisive.
- tau_phys = 3 meV/Å was written against the trainer's held-out numbers (per-atom vector
  RMS); it stays in that convention (per-component equivalent 1.7). Recorded.
- Correction of record (C6): "compensated within the bound-state region" was wrong; the
  fill compensates over its screening length.

---

*The "four checks" context (user, 2026-09-08 ~16:00, verbatim).*

Here is the missing context: Why the number is what it is. Relative to the pristine pattern, the defect cell's `q0` is a compact charge `+z` where the anion is missing (z ≈ 0.6–0.8 in Mulliken terms) plus a compensating `−z` spread over the fill's screening cloud. Under PBC the carrier's cross term with the compact part carries an image contribution of `2z` times the carrier's own Madelung term; the cloud cancels it only to the extent it is compact relative to the cell. With the cloud at 8–11 Å and ladder cells with `L_min` of 11–16 Å, the cancellation is partial, roughly 40 %, giving `2z(1 − f) ≈ 0.5–0.6`. That is your 51–65 %, and it is the same on every seed, which is what a model property looks like rather than a bug. The cloud itself is not surprising: a tight-binding Mulliken fill compensates a missing anion through hybridisation changes over two or three shells, with a decay length of order hopping over gap, which is 6–10 Å here. My C6 statement that `q0` is compensated "within the bound-state region" conflated the bound state (two Pb, `N_eff` ≈ 2.3) with the fill's charge-transfer response. The gate's 5 % tolerance assumed the former; the trained `H0` delivers the latter. "By construction" is the right description.
Checks that would expose a bug if there were one, an hour of work:

* On the pristine tiled cells, `q0` must be exactly periodic and the carrier's cross term with it size-independent. If it drifts, the fill or the Ewald background is inconsistent across sizes.
* Fit the slope from the largest two cells only. With a fixed cloud the deviation must shrink with `L` roughly as `(R_c/L)²`; a deviation that does not shrink is a convention error.
* Cumulative compensation charge versus radius from the vacancy. A 50 % slope error needs a substantial fraction of the `−z` sitting at 5–10 Å. If more than ~80 % is inside 4 Å, the gate result is wrong and the code is at fault.
* The synthetic locally neutral pattern you already ran (Madelung to 5 %) validates the kernel; sweep its width and confirm the deviation curve passes through the model's value at the model's `R_c`.

What it means for B′. The cross term's size dependence is nearly geometry-independent, so it does not reach the forces. Judge B′ on forces, and withhold any cross-size energy claim that includes `E_SF` until the ladder can be run on cells with `L ≫ R_c`, which needs the sparse solver. Record one physics caveat with it: the cloud is the model's own charge-transfer screening of the missing ion, and `Γ_LR` is already divided by `eps_inf`, so the static well is partly double-screened. That is the Route C question and stays deferred.

---

*Three notes for the record (user, 2026-09-08 ~16:45, verbatim).*

1. The non-converging compensation is expected, and no H0 will fix it. Local neutrality is an electrostatic effect: in the real system the fill compensates a missing anion within a few ångströms because leaving +z uncompensated costs Hartree energy. q0 is the fill of H0 with no electrostatics in it, so nothing in it enforces local neutrality; the compensation is whatever hybridisation dictates, plus a delocalised piece from the valence states renormalising after four orbitals are removed, which spreads over the whole cell like a uniform background. That is why the ladder behaves like the centred species pattern: a uniform component produces exactly the centred-pattern 1/L error. So the sentence "waits on an H0 whose reference fill localises the compensation" should read "waits on a self-consistent static pattern". The pattern that is locally neutral by physics is the SCC-converged reference charge, which is what the matched-kernel F-SCC runs compute as their reference solve, and using it in the energy with its response derivative is Route C. The B′ record stands: judged on forces, E_SF excluded from any cross-size energy claim by construction, double-screening caveat with the deferred item. A cheap confirmation if you want it: the far-field δq0 per site should scale as 1/N_at across the tilings if it is the uniform component, and decay with distance if it is a screening cloud.

2. Γ-point sampling of q0. A 0.24 e per-species shift between the static cell and the tiled cells means any absolute use of the pristine pattern is sampling-dependent. It only enters the head through W in the Route B′ gap regulariser and through the ladder baseline you have already corrected; keep each cell's own pristine fill at its own sampling and record the convention.

3. The ARPACK step. A residual of 3.8e-8 against 1e-8 on a 2556-orbital cell is a solver-tolerance item for Phase 4, not a ladder problem; the dense CPU path is the registered exact one and skipping the sparse agreement for the ladder is fine. Log it against the sparse-solver gate so it is not forgotten.

Nothing else to change: the kernel and ladder convention are validated by checks 1 and 3, the checks were the right ones, the B′ pattern scale drifting down is what the periodicity-suppressed far field predicts, and the queue split is sensible.
