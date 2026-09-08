# D-SCC plan: C10 rulings (user, 2026-09-08, verbatim)

*Saved verbatim from the user's message answering the Arm 2+3 read-out (`ARM23_COADVISOR.md`,
tracker C10). The message applies a "v4.4" amendment (`K_LR` required physics, LR-only the
minimal candidate, Φ = 0 arms as force references); no v4.4 document exists in the repository,
the tracker or this session's record, so the paragraph below is the only text of v4.4 on file
and the selection under it is labelled post hoc (see tracker C10 / D13).*

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
