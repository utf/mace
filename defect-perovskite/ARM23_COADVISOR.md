# Arm 2+3 — the registered rule selects Φ = 0; the coupling trained itself off

*V_Cl+ orthorhombic CsPbCl3, PBE labels of Mosquera-Lois & Walsh, PRX Energy **4**, 043008
(2025). Plan: D-SCC charge head v4 with amendments v4.1–v4.5. Arm 2+3 = regime B, Route A,
coupling ∈ {Φ = 0, LR-only, LR + U, full, λ = 1 fixed}, six seeds {0, 1, 2, 3, 4, 6} each
from the Arm-1 full-`H0` winner of the same seed, 60 epochs at constant lr 2e-3, final
checkpoint read (the Arm-1 protocol, v4.3); regime-A full coupling as a six-seed ablation
outside selection. Thresholds registered 2026-09-07 before any result was opened. Campaign
complete 2026-09-08 13:08; 36 runs, none failed.*

**Status.** The v4.3 selection rule chooses **`B_A_phi0`**, the Φ = 0 head, at a held-out
charged force RMSE median of 39.9 meV/Å. Every coupled arm is either equivalent to it
within the registered margin or excluded by a gate, and Φ = 0 is the simplest equivalent
configuration, so the selection is the same under every reading of the two gates that are
in question below. The optimiser turned the coupling off: `lambda_dir` trained from 0.05
to 0.004–0.018 and the hub-site `U_eff(Pb)` from 1.36 eV to 0.04–0.18 eV. Nothing downstream
has been opened; item C10 in the tracker carries four rulings.

## Per arm

Held-out charged force RMSE (meV/Å; median over six seeds, then the seeds), 159-atom
shape-slope error (median over four seeds: seeds 0 and 2 have no reading because their `C_Q`
window admits one and two 159-atom frames, below the three a slope needs),
`N_eff` p50 median with the seed spread, the single-valuedness failing fraction at its worst
epoch ≥ 1 and at the last epoch, the trained couplings, and loss spikes per run.

| arm | force | seeds | shape err | `N_eff` (spread) | sv worst / last | `lambda_dir` | `U_eff` Cl / Cs / Pb (eV) | spikes |
|---|---|---|---|---|---|---|---|---|
| **Φ = 0 (selected)** | **39.9** | 36.8 / 41.2 / 40.0 / 39.1 / 41.6 / 39.8 | 0.042 | 3.56 (0.29) | — | — | — | 0–2 |
| LR-only | 41.7 | 48.6 / 40.9 / 42.2 / 37.2 / 55.7 / 41.1 | 0.037 | 3.33 (0.32) | 0.00 / 0.00 | 0 | 0 | 0–5 |
| LR + U | 41.3 | 42.4 / 39.4 / 41.2 / 41.8 / 39.9 / 41.3 | 0.025 | 3.71 (0.14) | 0.08–0.18 / 0.00 | 0 | 0.27–0.68 / 0.08–0.11 / 0.03–0.18 | 0–3 |
| full | 42.0 | 39.2 / 42.5 / 44.9 / 39.3 / 41.4 / 47.7 | 0.040 | 3.61 (0.35) | 0.08–0.26 / 0.00 | 0.004–0.018 | 0.20–0.98 / 0.07–0.10 / 0.04–0.18 | 0–4 |
| λ = 1 fixed | 43.3 | 40.3 / 41.7 / 47.2 / 42.1 / 44.4 / 49.7 | 0.080 | 3.48 (0.88) | 0.10–0.36 / 0.00 | 1 | 0.22–0.33 / 0.08–0.09 / 0.06–0.17 | 1–5 |
| regime A, full (ablation) | 40.9 | 39.1 / 43.7 / 41.1 / 36.8 / 51.8 / 40.7 | 0.020 | 3.58 (0.34) | 0.00–0.08 / 0.00 | 0.019–0.063 | 2.3–8.9 / 0.09–0.10 / 0.16–0.34 | 0–3 |

Initial couplings 0.05 and 0.71 / 0.12 / 1.36 eV. `tau_noise` (Φ = 0 seed spread) 1.7 meV/Å
on forces and 0.018 on the shape slope; `tau_phys` 3 meV/Å and 0.015; the margin is 3.0.
Every run converged its SCF on 100 % of training frames. Force RMSE by distance from the
vacancy (0–2 / 2–4 / 4–6 / 6–8 / > 8 Å, medians): Φ = 0 84 / 68 / 50 / 32 / 32, LR + U 84 / 72 /
52 / 33 / 32, full 93 / 74 / 54 / 34 / 32, λ = 1 90 / 74 / 54 / 36 / 33. Full and λ = 1 are worse
on the median inside 4 Å but within their seed spreads (0–2 Å seed std 26 and 15 against 6
for Φ = 0); LR + U is equal; every arm matches Φ = 0 beyond 6 Å. The base alone reads 53.2 meV/Å on the fold-0 charged
held-out frames; the `H0` head takes it to 36.8 (Φ = 0, seed 0); the coupling adds nothing.

## The selection, and the two gates in question

The registered rule (`arm23.select`): configurations passing the gates are ranked by median
force; any within `max(tau_phys, tau_noise)` of the best on forces and not worse beyond the
same kind of margin on the shape slope is equivalent to it; the simplest equivalent wins.
As reported, only Φ = 0 passes the gates, so the ranking holds one entry. We re-ran the
selection on the same records under every reading of the two contested gates:

| reading | equivalent set | selected |
|---|---|---|
| as registered (worst-epoch root rule, `f_SR` gate) | {Φ = 0} | Φ = 0 |
| `f_SR` gate removed | {Φ = 0, LR-only} | Φ = 0 |
| last-epoch root rule | {Φ = 0} | Φ = 0 |
| both | {Φ = 0, LR + U, LR-only, full}; λ = 1 out on `localisation_stable` | Φ = 0 |

No coupled arm beats Φ = 0. λ = 1, at 43.3, is 3.4 meV/Å above Φ = 0 and beyond the 3.0
margin — the one arm worse than nothing beyond the margin; it leaves on `localisation_stable`
before the ranking sees it, which is why `beaten_beyond_margin` is empty in every reading.

**(a) Root rule.** The v4.5 check (a 5 % per-epoch subsample re-run by continuation against
the warm start, failing fraction ≤ 10 %, "failure fails the arm") is read at its worst epoch
≥ 1, and on that reading the full, LR + U and λ = 1 arms fail. The over-ceiling epochs are
the early ones — full: epochs 1–5 and 14; LR + U: 1–8 and 18; λ = 1: 1–31 (seed 4 over on 22
epochs) — none within two epochs of a logged loss spike (the spikes sit at epochs 20–58),
and the final-epoch check is 0/39 on all 36 runs. Pooled over the 59 checked epochs (2301 frames per run) the failing fractions are full 0.3–2.3 %, LR + U 0.3–2.4 %, λ = 1 0.9–7.7 %, regime A ≤ 0.2 % — every run under the 10 % ceiling on that reading; the worst-of-59 reading of a 39-frame subsample is the harshest of the three (one epoch at 5 of 39 reads 12.8 % and is consistent with a true rate of 5 %). So the trained maps are single-valued;
the failures are the transient from the multi-valued initialised map (28–51 % of the
79-atom charged frames at epoch 0 reach a different fixed point from a zero start than
from the continuation, C9) and they fade as `U_eff(Pb)` falls to ≈ 0.1 eV. Under a
last-epoch reading full and LR + U pass; λ = 1 fails `localisation_stable` regardless
(`N_eff` spread 0.88 > 0.5, seed 6 at 5.5). Three readings are on the table — worst epoch (strict, as
registered), pooled, last epoch; the record keeps the strict one unless you overrule it, and
the selection does not depend on it.

**(b) `f_SR` is a criterion defect, ours.** The registered gate is `f_SR` ≤ 0.5 with
`f_SR = Σ|dq_i dq_j| K_SR_ij / Σ|dq_i dq_j| (K_SR + K_LR)_ij`. It reads 1.5–1.8 on every
regime-B coupled arm, 1.2 on regime A and 3.1 on one λ = 1 seed — above 1, which a fraction
cannot be. On one 79-atom charged frame with the trained full-s0 model the short-range part
is +0.41 eV and the long-range part −0.19 eV: `K_LR` is negative on all 3081 pairs (−0.97 to
−0.31 eV/e²) because in a cell this small the neutralising background dominates the
interaction beyond `r_s`. The quantity is unbounded; the floor was set on a scale it does not
live on. It also does not depend on the arm's Γ (LR-only reads 1.6 with no `K_SR` in its
kernel — the report computes it from the kernel split and the trained `dq`), and Φ = 0 passes
it vacuously. Recorded; selection unaffected; not re-registered — the same status as
Arm 1's tensor-ratio clause. The plan's regime-B small-cell rule (λ → 0 on the ladder) is a
different test and has not been run.

**(c) Epoch-0 reading** (C9, still pending): the initialised-model check is a diagnostic and
the ceiling applies from epoch 1 (v4.2 §5). Under the other reading every arm but LR-only
fails at initialisation and the campaign is decided before training.

## What the outcome says

Under the registered protocol the data drove the D-SCC coupling toward zero on these
frames. The bound state is kept in every arm (`N_eff` 3.3–3.7) by `H0`; Φ neither
sharpens nor extends it (LR + U vs full — the λ = 0 ablation — `N_eff` 3.7 vs 3.6, forces
41.3 vs 42.0). The §2.10 Route-A forecast ("`lambda_dir` and `U_eff` come out O(1); the
λ = 0 ablation makes the level shallower and the carrier more extended") is not borne out in
regime B; in the regime-A ablation `U_eff(Cl)` did go to 2.3–8.9 eV at an equivalent force
error. The forecast of distinct fixed points at asymmetric geometries is borne out at
initialisation and in the early epochs. Inside 2 Å the full and λ = 1 arms are worse than Φ = 0
on the median (93 and 90 against 84 meV/Å) but within their own seed spreads; forcing the
short-range coupling on at full strength (λ = 1) is the one arm worse than Φ = 0 beyond the
selection margin.

Two caveats on the reading. The 159-atom shape criterion rests on four frames per fold and
is absent on two of the six seeds; it cannot separate the arms (LR + U 0.025 vs Φ = 0 0.042
against a margin of 0.018). And the ±10 meV/Å final-checkpoint noise recorded in v4.3
favours the simpler arms at selection by construction — but a coupled arm would have to
beat Φ = 0 by more than `tau_phys` = 3 meV/Å to change the outcome, and with the couplings
trained to ≈ 0 there is no mechanism for that.

## (d) What we ask about the rest of the plan

With Φ = 0 selected the downstream items lose their object: Arm 4's F-SCC comparators
compare against a coupled D-SCC model; the 2× benchmark engineering item is registered on
"the Φ-on model" (the Φ = 0 head already reads median 2.07 / p95 2.22 at 79 atoms, P4.3);
the Phase-4 ladder's `K_LR_ii` and `dq`-spread items are vacuous at Φ = 0 (its `E(+1) − E(0)`
vs `1/L` remains readable as the base + `H0` size behaviour); and the §9 robustness pass
retrains "the selected configuration and its Φ = 0 reference", which are now one arm.

- **(i) Close at the selection.** §9 pass on Φ = 0 under the LR-decay protocol (1.5–2 h per
  seed on the A4000), the P4.1 ladder on it, and the final artefact.
- **(ii) (i) plus LR + U under the §9 protocol**, to test whether the equivalence survives the
  final-checkpoint noise. It cannot plausibly reverse the selection (above), but it makes the
  "no gain from Φ" statement at lower noise.
- **(iii) Open Arm 4 on the full arm** as an exploratory, non-selected comparison of the
  D-SCC vs F-SCC feedback sign (§2.10) — a science question, not a selection one.

Our reading: (i), with (iii) only if the feedback-sign question is wanted for its own sake.

Artefacts: `~/runs/dscc/arm23_report.json` (records, gates, decision),
`arm23_spikes.json`, `v45_gate.json`; the 36 run directories `~/runs/dscc/dscc_arm23_*`
(`model.pt`, `held_final.json`, `train.log`); tracker
`defect-perovskite/DSCC_V4_IMPLEMENTATION_SPEC.md` (§1 C10, §2.2 D12, §3.4 P3.2).


## C10 follow-up (2026-09-08, after the rulings)

**Why Route B′ was absent.** Its v4.2 entry gate — the local-neutrality tiling ladder on the
trained `H0` — failed on every Arm-1 winner (51–65 % off Madelung: the reference fill
compensates the missing ion only over 8–11 Å) and the tracker closed the B′ arms before Arm 2+3
launched. Re-run on the Arm-2+3 Φ = 0 models it fails again (51 / 53 / 61 / 51 %). B′ now runs
by the C10 ruling over that gate; its registered selection gate (ladder `1/L` within 5 %) fails
by construction on this `H0` and needs a ruling.

**Units, and the far-field premise.** The trainer's held-out numbers are the RMS of the per-atom
force-error vector, √3 times the per-component RMS that MACE's logs and the base's 13 meV/Å use;
"13 vs 33 beyond 8 Å" compared the two. Per component, by shell from the vacancy (8–10 / 10–12 /
> 12 Å): Φ = 0 head at 79 atoms 20.3 / 13.5 / 10.3 against the out-of-fold neutral floor
10.2 / 9.5 / 8.3; at 159 atoms 10.6 / 7.2 / 4.5 against 9.4 / 6.9 / 4.7. The charged far field
at 159 atoms is neutral-like; by the criterion in the ruling the 79-atom excess is small-cell
and manifold extrapolation, not the Coulomb channel. The near field (0–2 Å: 48 against 31) is
the open item. Table and details: tracker D13.

**Selection under v4.4.** The amendment reached the record at ~15:35 (after the campaign) and
is on file; applied post hoc as the ruling anticipated: LR-only selected (LR + U 41.3 best candidate, LR-only 41.7, full 42.0 equivalent;
λ = 1 out on localisation), +1.8 meV/Å against Φ = 0 inside the margin. Root rule on the final
model and the last ten epochs: every arm passes. `f_sr_abs` 0.69–0.71 on every regime-B arm.

**Running since 15:13 (b3):** Route B′ LR-only ×6 and LR + U ×6 (trimmed to these by the C10
addendum at 15:41; full, λ = 1 and the ablation not run) at four per GPU on GPUs 4 and 5
(≈ 13 h per run, eight slots per wave, ≈ 26 h in all); the
matched-kernel F-SCC comparator on seeds 0, 1, 2 on GPU 7 (≈ 35–45 h). Stop:
`scratchpad/kill_bp_b3.sh`. The C10 addendum settled the B′ ladder gate (re-read as a model
property, B′ judged on forces), the trim, the far-field reading (gain on the full RMSE and the
0–2 / 2–4 shells decisive, 4–8 Å reported) and `tau_phys` (trainer convention); the v4.4 text
is on file. The four checks arrived and were run (tracker D13): the Ewald background and the
kernel are exact for a fixed pattern (a point-charge neutral pattern reads Madelung to 0.0 %
from 16.8 Å up), `q0` is exactly periodic, and the compensation is 18–29 % inside 4 Å — not a
code fault. But the model's compensation of the missing ion is not a fixed 8–11 Å cloud: read
against the same-size pristine cell its 85 % radius grows from ≈ 10 Å at 22 Å to > 12 Å at
34 Å, so the consecutive-pair slopes never settle (41 / 78 / 2 / 97 % off Madelung). B′ stays
judged on forces; the ladder and `E_SF` wait on the sparse path and on a self-consistent static
pattern (Route C, deferred) — the non-converging compensation is expected: `q0` is the fill of an
`H0` with no electrostatics, and its delocalised piece (≈ −0.1 e spread uniformly, confirmed to
scale as 1/N_at) produces the centred-pattern `1/L` error. LR + U seeds 4 and 6 run locally
(two on the A4000) to finish the B′ set earlier.

## Route B′ outcome (2026-09-09; tracker D14)

**Both selection rules select `B_Bp_lr_only`.** Held-out charged force RMSE on seed medians,
per component (trainer convention in brackets): B′ LR + U 18.9 [32.8], B′ LR-only 19.0 [32.9],
Φ = 0 23.0 [39.9], Route A LR + U 23.8 [41.3], LR-only 24.1 [41.7], full 24.2 [42.0]. The two
B′ arms are equivalent within the 3.0 margin and the simpler wins; Φ = 0 and every Route A arm
are beaten beyond the margin, every winner seed (18.0–19.1) outside the Φ = 0 seed range
(21.2–24.0). Under v4.4 the `K_LR` candidate costs −4.1 [−7.0] meV/Å against the Φ = 0
reference, i.e. nothing. B′ entered post hoc (the C10 ruling over its failed entry gate); within
the B′ set the pre-registered rule applied unchanged.

**Gates.** Root rule 0 % on the last ten epochs of every seed; `N_eff` 3.3–3.5; converged 100 %;
`s` 0.49–0.63 (unsaturated, about half its initial scale); the far-field gains over Route A
LR-only beyond the Φ = 0 spread on the full RMSE, 0–2 and 2–4 Å (decisive) and on 4–6, 6–8 and
4–8 Å (reported). Shells per component (0–2 / 2–4 / 4–6 / 6–8 / 8–10 / 10–12 / > 12 Å): B′
43 / 35 / 24 / 15 / 14 / 12 / 8.5 against Φ = 0 48 / 40 / 29 / 19 / 20 / 14 / 10 and the
out-of-fold neutral floor 31 / 22 / 17 / 10 / 10 / 9.5 / 8.3 — B′ reaches the floor beyond 12 Å
and closes most of the 8–10 Å gap; the near field (43 vs 31) stays open. Shape error 0.015
against 0.042. `U_eff(Pb)` in LR + U trains to 0.16–0.31 eV: "off" is the selected value.

**Checks on the winner.** Pair-route forces agree with central differences on a real held-out
frame to 2e-5 eV/Å; the `q0` ladder slope is 29–40 % off Madelung on the six `H0` (model
property, reported); the `1/L` coefficient of `E(+1) − E(0)` is 67 % of Madelung (Route A
LR-only 24 %), the head's kernel term 90 %. **The 2× benchmark fails on the winner:** ratio
7.0 at 79 atoms and 8.6 at 159 (p95 7.4 / 13.7) against 2.0 / 1.9 for Φ = 0 — the coupled head
costs about six base forwards; not profiled.

**Status.** Production status is provisional on Arm 4: the matched-kernel F-SCC (three seeds,
λ and U learned) is at epoch 30 of 60 and reads 17.2 / 19.2 / 17.4 per component at epoch 29 —
at or below the B′ finals at half training; finals ≈ 07:00 on 10 Sep. Decision (4) cannot be
read for B′ (no sparse Route B′ path). The §9 robustness pass is not run (user, 9 Sep: no
retraining); the ±10 meV/Å final-checkpoint noise stays in every reading. The earlier
"electrostatics adds nothing resolvable" was about the same-carrier terms; the host term
reversed it, as the ruling anticipated.

## D15 — who owns the near-field residual (2026-09-09; evaluation only)

**The item was mis-centred.** The 79-atom cell is two octahedra thick along z (11.1 Å), so the
flanking Pb pair are neighbours through both images and share an occupied bridging Cl on the
other side. The minimum-image midpoint used for every shell table so far lands on that Cl in
114 of 256 charged fold-3 frames (the +1 Pb relax away from the vacancy, making the occupied
path the shorter one); the "0–2 Å charged atoms, 43 meV/Å" were bridging Cl 5.5 Å from the
vacancy, while the neutral "0–2 Å atoms, 31" were dimerised flanking Pb. With a vacancy-side
centre no charged frame at either size has an atom inside 2 Å. Shells from 2 Å out change by
< 1 meV/Å, so the far field, the B′ gains and the selection stand; the far-field gate's
"decisive 0–2 Å" shell is empty under a correct centre and needs re-registration (C11). The
evaluator now uses the vacancy-side rule.

**Corrected near field (2–4 Å, per component).** Winner 33.0 against a 21.5 neutral floor at 79
atoms (flanking Pb 47.8 vs 33.4; first-shell Cl 26.2 vs 16.9; Cs 19.1 vs 10.7); at 159 atoms
14.3 against 33.2 (4 held-out frames, 31 atoms). Base alone 117.2 at 79 with a systematic −258
mean radial residual on the flanking Pb (it predicts the neutral dimer attraction); the head
removes it (+3.9) and leaves an unbiased scatter.

**Reading by the registered rule: mixed** (on the substitute 2–4 Å shell; the registered 0–2 Å
shell is not readable). `x_159` passes strongly, no radial signature on the pooled atoms at
either size, but the novelty-predicted excess is 26.4 (15.8–26.4 across the five ways of
extrapolating the base's calibration, which is shell- and size-blind: it under-predicts the
neutral 2–4 Å floor by 12 at 79 and by 31 at 159). **Evidence:** at matched Pb–Pb distance the
base's flanking-Pb residual is the same at both sizes (150–285) while the head's is 43–53 at 79
and 8–35 at 159 in every overlapping bin; the 79-atom flanking Pb carry 1.7 × the label force of
the 159-atom ones at the same d (389 vs 236 overall). The remaining 79-atom excess is a property
of the 79-atom labels and cell, not a deficit the head shows on the geometry. The `b_i(h_i)`
capacity arm is not indicated on this evidence (C12); if it runs, its metric must be the paired
79 / 159 flanking-Pb residual at matched d. Caveat: four 159-atom held-out frames.
