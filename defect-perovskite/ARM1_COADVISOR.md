# Arm 1 — route C twice; the registered rule has stopped the programme before Arm 2+3

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set. Plan: D-SCC charge head v4 with amendments
v4.1 and v4.2. Arm 1 = full `H0` (scalar + directional onsite block) against a scalar-only
control, Φ = 0, Route A, six seeds each, 60 epochs, outer fold = seed mod 4, final
checkpoint read. Thresholds registered 2026-09-07 before any result was opened.*

**Status.** Arm 1 routed **C** on the final twelve-run report (seeds 0–5) and **C again** on
the registered repeat (fresh seeds 6–11, identical protocol). Criterion (ii) fails on both
rounds: the scalar-only control reproduces the level-vs-bond slope to within 5 % and then 8 %,
against the registered 30 % loss. The rule we registered before launching the repeat —
"repeat the covariance/sign tests once; stop if the repeat fails" — has been applied. Nothing
is running. Arm 2+3, Arm 4 and the Phase-4 ladder on a coupled winner do not open without a
ruling. Item C8 in the tracker carries the options; this note is the case for each.

## The criteria, both rounds

| criterion (registered) | threshold | round 1 (s0–s5) | repeat (s6–s11) |
|---|---|---|---|
| (i) vacancy-spanning `pp` stop fraction | ≤ 1 of 6 seeds | 0/6 — pass | 0/6 — pass |
| (ii) flanking-Pb tensor ratio; control's loss of the level-vs-bond slope | > 3; ≥ 30 % | 10.3; **5.2 %** — fail | 10.3; **8.0 %** — fail |
| (iii) Cl `ε_σ − ε_π` sign | negative, every seed | −12 to −155 meV — pass | −46 to −114 meV — pass |
| (iv) `N_eff` seed spread halves, at ≤ force RMSE, no saturation | | 0.175 vs 0.217 (not halved); forces 43.3 ≤ 47.7 — **fail on spread** | 0.269 vs 1.881 (halved); forces **49.5 > 48.0** — **fail on forces** |
| routing (Stage-2 table) | | C | C |

Over all twelve seeds of each arm (information only, not a registered reading): (i) and
(iii) pass, (iv) passes (`N_eff` spread 0.239 vs 1.481, forces 43.9 ≤ 47.7 meV/Å, no
saturation), (ii) fails at 4.9 % — route C.

Two remarks on (ii). The tensor ratio (10.3) is identical on every seed: with a learnable
coefficient `b` per species and a geometric `Q_i`, the flanking-Pb `|bQ|` over the bulk
spread of `|bQ|` is a ratio in which `b` cancels — it measures the descriptor's geometry,
not the trained model. Only the slope clause discriminates, and it does not discriminate in
the direction the criterion assumed. The slope itself is noisy across seeds (full 0.08–0.34,
control −0.02–0.29 eV/Å, per-seed SE ≈ 0.013), so the medians' difference (0.247 vs 0.234;
0.162 vs 0.149) is well inside the seed spread in both rounds.

## Per seed

Held-out charged force RMSE (meV/Å), `N_eff` p50, HOMO separation p50 (eV), registered
bound-state precondition (`Δ_c` 0.5 eV, `N_loc` 4, over all 1191 neutral-vacancy frames;
full runs only), level-vs-bond slope (eV/Å), Cl splitting (meV).

| full `H0` | RMSE | `N_eff` | sep. | precond. | slope | Cl split |
|---|---|---|---|---|---|---|
| s0 | 43.0 | 2.50 | 0.86 | 96.1 % | 0.342 | −12 |
| s1 | 43.6 | 2.29 | 0.84 | 96.3 % | 0.198 | −75 |
| s2 | 42.5 | 2.38 | 0.89 | 97.2 % | 0.338 | −155 |
| s3 | 40.9 | 2.22 | 0.91 | 96.8 % | 0.282 | −92 |
| s4 | 43.6 | 2.31 | 0.81 | 95.5 % | 0.213 | −40 |
| s5 | 49.0 | 1.93 | 0.85 | 92.9 % | 0.129 | −112 |
| s6 | 55.2 | 2.00 | 1.04 | 97.8 % | 0.164 | −71 |
| s7 | 51.4 | 1.83 | 0.88 | 95.9 % | 0.161 | −108 |
| s8 | 47.7 | 2.44 | 0.76 | 94.1 % | 0.077 | −103 |
| s9 | 42.0 | 2.41 | 0.86 | 96.4 % | 0.272 | −114 |
| s10 | 44.2 | 2.29 | 0.93 | 97.4 % | 0.249 | −46 |
| s11 | 51.3 | 1.77 | 0.81 | 92.9 % | 0.137 | −78 |

| scalar-only control | RMSE | `N_eff` | sep. | slope |
|---|---|---|---|---|
| s0 | 50.0 | 1.81 | 0.68 | 0.259 |
| s1 | 74.7 | 1.95 | 0.51 | 0.087 |
| s2 | 67.2 | 2.49 | 0.10 | 0.103 |
| s3 | 39.3 | 2.18 | 0.45 | 0.293 |
| s4 | 43.7 | 2.02 | 0.53 | 0.247 |
| s5 | 45.5 | 1.95 | 0.45 | 0.223 |
| s6 | 87.9 | 6.13 | 0.08 | 0.114 |
| s7 | 92.4 | 5.85 | 0.10 | −0.020 |
| s8 | 42.5 | 2.19 | 0.36 | 0.173 |
| s9 | 52.1 | 1.89 | 0.29 | 0.126 |
| s10 | 44.0 | 2.01 | 0.44 | 0.247 |
| s11 | 43.5 | 1.94 | 0.40 | 0.217 |

Force RMSE by distance from the vacancy (0–2 / 2–4 / 4–6 / 6–8 / > 8 Å), round 1 seed 0:
full 99 / 77 / 55 / 34 / 33, control 111 / 87 / 61 / 47 / 32 meV/Å. Beyond 8 Å both heads sit
at the base's floor; the full head's gain, where it exists, is inside 8 Å.

## What the directional block does and does not carry

It does not carry the level-vs-bond response. That is the premise of criterion (ii), and on
this `H0` and this data the scalar onsite term reproduces the slope. We read that as a
negative attribution result, not as a failure of the experiment.

It does carry the bound state. Every full seed keeps a separated vacancy level (separation
0.76–1.04 eV, precondition 92.9–97.8 %); the control does not (separation 0.08–0.68 eV;
on the report's 300-frame diagnostic its precondition fraction is 0.00–0.94), and two repeat
control seeds (s6, s7) lost the state altogether — `N_eff` ≈ 6 and 88–92 meV/Å at the final
checkpoint. Criterion (iv) as registered reads the `N_eff` seed spread at unchanged-or-better
force RMSE; it does not read separation or the precondition, and in the repeat the force
medians went the other way (three full seeds at 51–55 meV/Å after mid-training loss spikes).
So the block's one measurable contribution is exactly the quantity the registered criteria
do not test.

## Three things about the record

1. **Routing-code correction, ours.** The preliminary reading in the tracker said "3 of 4 →
   route D". `decide()` routed by the pass count, which sends a lone (ii) or (iv) failure to
   D; the Stage-2 table sends any (ii)/(iii) failure to C, a (iv) failure with (ii)+(iii)
   passing to B, and only a lone (i) failure to D. Corrected to the table verbatim, with a
   test over all sixteen verdict combinations; the final decision is C under both versions,
   and the preliminary D was a mis-route even on its own numbers.
2. **The loss-spike watch item fired in both rounds.** At constant lr 2e-3 with the final
   checkpoint read: control s1 (48.7 → 74.7 meV/Å between the epoch-54 and epoch-59
   evaluations), control s2 (52.8 → 67.2), repeat control s6 (diverged at epoch 57, loss
   2.6e-3 against a median of 1.5e-5; 43.1 → 87.9), control s7 (57.9 → 92.4), full s6 and s7
   (spikes of 8.6e-4 and 9.9e-4 at epochs 38 and 52). The final-checkpoint read is noisy at
   ±10 meV/Å, which is larger than the full-versus-control force difference either round
   measures. A registered LR decay to 2e-4 over the last 20 epochs (or an epoch-averaged
   final checkpoint) is proposed; it was not applied, because a repeat under a changed
   protocol is not a repeat and the rule says stop.
3. **Route C was applied in the v4 wording** ("repeat covariance/sign tests once"). The v8.1
   table's extra clause, "and the single registered bound release", was not carried into v4
   and was not done.

## Where this leaves the programme

- The deletion sweep (plan §3) is done: the v8 modules, tests and documents are removed,
  seventeen core files are back at upstream `050b791`, and only `mace/modules/dscc/` with
  its suite remains (102 dscc tests, 260 upstream unit tests, all green; b3 synced).
- Registered 2× benchmark on the Φ = 0 winner (idle A4000, warm-started trajectory): 79
  atoms median 2.07, p95 2.22 (100 frames); 159 atoms median 1.96, p95 5.79 (16 frames). The
  head costs about one base forward, so the ratio sits at 2.0 ± 0.1 and the criterion (median
  and p95 both ≤ 2) fails as it stands.
- Arm 2+3 (Route A, regime B, coupling factorial, seeds by explicit list) and Arm 4 (F-SCC
  comparators) launch paths are smoke-tested on the swept tree and take the converted
  `h0_state.pt` of any Arm-1 seed. Nothing has been launched.

## What we ask

A ruling on C8. The options, with our read:

- **(a) Accept the stop.** The attribution experiment is negative as registered: the tensor
  block is not needed for the modelled level-vs-bond response. Clean, and it ends the
  programme at the Arm-1 gate.
- **(b) Re-register the attribution, post hoc.** Replace the slope clause of (ii) with a
  localisation criterion (registered precondition or HOMO separation, full versus control)
  and read the twelve existing seeds of each arm against it — no retraining. This is the
  cheapest resolution and it tests what the block demonstrably does; it is also a change of
  the test after two rounds have been opened, and should be recorded as such.
- **(c) A third round under a registered LR schedule.** Outside the rule. It would tighten the
  force reading (the ±10 meV/Å final-checkpoint noise), but it cannot rescue (ii): the
  slope loss was 5 % and 8 %, not 25 %.
- **(d) Open Arm 2+3 on the full `H0` as an exploratory, non-registered arm.** About 2.5 days
  of GPU on the current layout; every downstream selection would then rest on an arm the
  plan did not admit.

Our preference is (b), on the record as a post-hoc re-registration, with the LR schedule
registered before any further training under any option. If the ruling is (a), the
benchmark and the sweep are the programme's closing state and we will write the final
artefact from the tracker as it stands.

Artefacts: `~/runs/dscc/arm1_report.json` (round 1), `arm1_repeat_report.json`,
`arm1_repeat_gates.json`, `arm1_combined_report.json`; tracker
`defect-perovskite/DSCC_V4_IMPLEMENTATION_SPEC.md` (§1 C8, §2.2 final readings).
