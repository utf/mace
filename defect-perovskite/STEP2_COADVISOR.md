# Carrier localisation in V_Cl+ orthorhombic CsPbCl3 — status, 1 Sep 2026

*Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025). Labels are that
paper's low-fidelity PBE set (scalar-relativistic, no SOC).*


**Question.** Our defect model (MACE trunk + tight-binding carrier head) must decide
whether the hole in a chlorine vacancy is *bound* to the two vacancy-adjacent Pb ("hub")
or *delocalised* over the cell. DFT says bound: the force-response ratio against DFT is
R = 0.95, CI [0.66, 1.34] over 13/17 matched frames, with a null magnitude 5–7% of signal.

> **Cell-size caveat on R_DFT.** The 80-atom training cell is a 2x2x1 orthorhombic
> expansion with c = 11.2 A, which constrains Pb-Pb separations above ~5.5 A when the
> vacancy axis lies along c. The 79-atom d(Pb-Pb) distribution is therefore partly
> cell-limited. The matched-d comparison against the 2x2x2 (159-atom) frames is
> unaffected in principle, but the unmatched long-d frames are exactly the ones this
> constraint removes from the small cell.

The open question is whether the model reproduces that, and if so, why.

## Background: where we were

Two facts constrained the design. (i) The hole's force footprint reaches ~6 Å, with two
thirds of the squared residual sitting *off* the hub — so any mechanism has to act at
range. (ii) A short-ranged TB head cannot produce that reach from a compact state: its
forces come from dt/dR and deps/dR, both dying with the hopping envelope.

That was confirmed empirically. In a clamp matrix ("R1") on a corrected 10 Å graph, every
compact clamp gave axial_red ≈ 0 (hub2: +0.059, best seed +0.264) while the unclamped
`free` head reached +0.807. A delocalised solution *exists* that fits the axial residual;
no compact one did. The head also showed no site preference — hub2 beat neither a
distance-matched ligand pair nor random pairs by more than the seed spread.

*(axial_red = 1 − RMSE_axial(head)/RMSE_axial(base-only), along the hub–hub axis, pooled
over held-out charged frames. 0 = no better than the frozen base, 1 = perfect.)*

## The hypothesis under test

Electrostatics has the reach the TB head lacks. We added a **carrier-field response
channel**: the carrier's own charge density generates a periodic potential V, and each ion
responds through a learned effective charge (linear in V) plus a learned polarisability
(quadratic, softplus-constrained so it is always stabilising). V enters the energy readout
only, never the Hamiltonian — deliberately, since feeding V back into H would install a
polaron prior whose sign here is unmeasured, i.e. site selection by construction rather
than by data.

Gate set in advance, both clauses required:
1. axial_red(hub2) rises from ≈0 to **≥ 0.4**, with rmse_nbhd falling toward `free`.
2. hub2 beats a distance-matched ligand pair and random pairs **by more than the seed
   spread** — a physical kernel should distinguish sites through the *shape* of the
   response.

Prediction recorded before results: *clause 1 passes, clause 2 is the real unknown.*

## Result 1 — clamped comparison (complete, 20 cells, 5 seeds/cell, H3)

| clamp | axial_red mean | sd | median | rmse_nbhd med | lig_shell | cs_shell |
|---|---|---|---|---|---|---|
| hub2 full | +0.306 | 0.324 | +0.392 | 84.5 | 72.0 | 35.9 |
| hub2 nbhd | +0.332 | 0.309 | +0.394 | 65.5 | 55.5 | 34.3 |
| lig2 full | −0.272 | 1.368 | +0.679 | 46.1 | 43.2 | 30.4 |
| lig2 nbhd | +0.236 | 0.824 | +0.678 | 45.8 | 43.9 | 38.4 |

Reference (no channel): hub2/full +0.059, best seed +0.264; `free` +0.807, rmse_nbhd 34.0.

**Both clauses fail — but they fail differently.**

*Clause 1 fails on the threshold, not the mechanism.* hub2 misses 0.4 under every
aggregation (mean 0.306, median 0.392, 2/5 seeds clear). But the mean now exceeds the best
single seed of the no-channel baseline, and rmse_nbhd fell 147 → 84.5 toward `free`'s 34.0.
The response channel genuinely does give compact states the long-range footprint that the
TB head could not produce. That part of the physics works.

*Clause 2 fails outright.* hub2 − lig2 is +0.577 (full) / +0.096 (nbhd) against seed sd of
1.368 / 0.824. Reading the fine structure, as the protocol requires when clause 2 fails:
on medians the ligand clamp is **better** than the hub on every metric. The operative
conclusion is the protocol's "no hub preference" category. The point estimates lean past
indifference toward the ligand actually fitting better, but that rests on a median over 3
converged ligand seeds against 5 hub seeds, so we report it as a lean, not a verdict.

The ligand control is distance-matched (5.57 Å, 100% coupled, same as hub) — an earlier
version was the maximally-separated pair and compared coupling rather than site.

## Result 2 — from scratch, 8 seeds (interim, epoch 28/50)

Trained jointly from scratch with the mechanism available, the supervised carrier channel
is delocalised in **8/8 seeds**: N_eff 29.5–51.6 against a gate of ≤8, and null ratio
0.40–0.70 against ≤0.15. The bandwidth anneal (hoppings scaled 4× → 1× over epochs 0–20,
so a level can separate gradually rather than tunnel out of a converged delocalised
solution) completed at epoch 20; N_eff has been flat since, so these numbers are now
diagnostic rather than a transient of the schedule. Three ungradiented channels provide a
per-run null baseline and sit stably at ~74 throughout.

## Reading

These are one finding, not two. **The response channel makes a compact state nearly as
good as the delocalised one (0.68 vs 0.807) regardless of which site is clamped, and when
the model is free to choose it goes delocalised in 8/8.** Nothing in the current data pulls
the solution onto the hub.

Worth keeping distinct: this is a statement about *preference*, not *representability*. A
bound solution that fits now exists in the model — that is new, and it is what the
electrostatic channel bought us. What is missing is any force in the objective that
selects it, or that distinguishes hub from ligand.

The plausible reason clause 2 failed is physical, not a bug: a cage-centred state also
produces an axial hub force, because each hub Pb sits inside five ligand charges instead
of six, so the net field is axial by the same asymmetry that made the earlier hopping
fingerprint non-specific. Both states reproduce the coarse footprint.

## Caveats

- Arm 1b is at epoch 28/50; final call at 50. Post-anneal flatness makes a reversal
  unlikely but it is not yet final.
- Clause 2 is formally incomplete — the random-pair controls are still queued — but it is
  a conjunction, and failing against the ligand pair already decides it.
- The `free` reference above (+0.807 / 34.0) is the *without-channel* value; the queued
  with-channel control will recalibrate that ceiling, so "toward free" comparisons are
  provisional.
- 5 seeds per clamp cell, 3 for controls. The ligand cell has 2/5 diverged seeds, which is
  why medians are quoted; hub2 converged 5/5. That stability difference is not significant
  at n=5 (Fisher p ≈ 0.44) and is not a site preference.

## Next

The protocol's prescribed measurement at this branch is an SCF diagnostic that turns an a
priori modelling choice into a measurement: evaluate the trained response energy at fixed
carrier amplitude under a hub clamp versus a cage clamp on identical frames. If
E_resp(hub) < E_resp(cage), the induced polarisation favours the hub and feeding V back
into H would help; the reverse means it would drive the carrier onto the cage and must
stay out. Not yet implemented.

Note that the alternative branch — abandoning electrostatics for an s+p Slater–Koster
basis — is ruled out by these data. It was conditioned on hub2 remaining ≈0, and it did
not.

---

*Superseded in part on the same day: a band-edge diagnostic (T-A) now shows Delta_bind =
lambda_1(pristine host) - lambda_1(defect) ~ 0 across 12 models, i.e. nothing anchors the
carrier level to the host continuum. See STEP2_RESULTS.md and the T-A/T-B work for the
follow-on.*
