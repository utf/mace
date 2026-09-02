# Full-sum Madelung kernel: sections 0–6 — report

V_Cl⁺, orthorhombic CsPbCl₃ (Mosquera-Lois & Walsh, PRX Energy 4, 043008, 2025; PBE
scalar-relativistic labels). **Regime tag on every trained number: lr 0.01, AdamW, 60 epochs,
frozen Stage-A base, forces-only + loss_gap.** Nothing here transfers to another optimiser
regime without a cross-regime check.

---

## §0 — record

Entry 7 of `LEDGER.md`, replacing the previous #7, carries the convention statement, the
analyst's rewritten entry and the statement of record verbatim. What it invalidates: every
number computed with `phi_LR` under the subtraction convention — Stages 1–3 and every gate
scored on those models. What it does **not** invalidate: the §1 wiring evidence (a statement
about the gradient, independent of the kernel), the tiling test's element and spectrum
clauses, and the F4 head-vs-base discrimination (both slopes measured through the same
kernel).

---

## §1 — the kernel

`phi_LR` is now the full lattice sum with only the true self term excluded.
`site_potential` **refuses** a `self_potential` argument rather than ignoring it.

**One deviation from the plan's letter, in service of its content.** The plan says to keep
the Gaussian self-energy removal (+11.49 eV/e). There is nothing left to remove.
`self_potential_of` returns the Makov–Payne potential of a point charge in jellium:

| L (Å) | 5.6 | 11.2 | 16.8 | 22.4 | 33.6 |
|---|---|---|---|---|---|
| kernel `A_ii` | −7.320 | −3.666 | −2.450 | −1.843 | −1.235 |
| `−α_M C / L` | −7.297 | −3.648 | −2.432 | −1.824 | −1.216 |

~1.5% at every size. A Gaussian self-energy would add a constant +11.49 eV/e and swamp the
table, so LES's `k ≠ 0` sum carries no self term and what was being subtracted was entirely
ion i's own periodic images. Now a standing test rather than a remembered claim.

---

## §2 — test table

| | test | status |
|---|---|---|
| (a) | `(AZ)_i` invariant under three tilings, 1e−8 | **pass** |
| (a′) | counterfactual: retired convention still drifts ~1.3 eV | **pass** (so (a) is not vacuous) |
| (b) | kernel vs `−α_M/L` | **pass**, ≤3% at five sizes |
| (b′) | φ shares E_LR's object *by identity* (`E = qᵀAq/2`) | **pass**, rel 1e−10 |
| (c) | prohibition on a constant-difference test across **defect** sizes | recorded in-suite with the `O(Q/εL)` reason |
| (d) | 79↔159 pristine on-site equality | **pass**, strict |
| (e) | `changed_level_index` property test | **pass**, 5 cases |
| — | no isolated-mode branch in the φ path | **pass**, enforced against the **AST**, not a text grep |

The old `xfail` is now a strict pass for the right reason: not a loosened tolerance, but the
description-invariant object being computed.

**126 → 142 tests pass** across the defect suite and flag plumbing.

---

## §3 — plumbing and bit-identity

Five flags through `arg_parser`, both `MACEDefect` construction sites and the launcher, via
one shared `_defect_madelung_kwargs` that **refuses** the counting head without the spectral
branch and the Madelung term without a composition. No P-backward flag exists to plumb: the
density response is requested structurally (training ∧ compute_force), so no configuration
can silently train the frozen-P gradient.

**The gate found a defect on its first run.** `extract_config_mace_model` read
`model.spectral.smearing` and `.t_min` by plain access; the counting head has neither, so a
Stage-3 model **could not be round-tripped through the production path at all** — could not
be saved and reloaded, and the cuEq conversion performs exactly that round trip at the end of
every run.

| | cross-path | same-model repeat floor |
|---|---|---|
| eps (site) | **0.000e+00** | 0.000e+00 |
| delta_sr | **0.000e+00** | 0.000e+00 |
| energy | 3.05e−05 | 9.16e−05 |
| forces | 1.01e−06 | 8.05e−07 |

The head's own outputs are exactly identical. Energy and forces sit inside the floor from
running the *same* model twice (MACE's scatter reductions use atomics); the floor is measured
in the same script, so a real 1e−6 difference cannot be dismissed as noise and noise cannot be
reported as a difference. **`SECTION 3 BIT-IDENTITY: PASS`.**

---

## §4 — forward-only, on the six pre-correction models

Read-only. These are *what the old weights predict through the corrected kernel* — the
weights were fitted under the subtraction, so `eps0`, `Z`, `c` are off their optimum.

### The quantity that motivated the change

On-site error between the 79- and 159-atom cells under the retired convention, at the learned Z:

| Cl | Cs | Pb |
|---|---|---|
| **+0.1094 ± 0.0013** | −0.0454 ± 0.0062 | **−0.2828 ± 0.0040** eV |

Species-proportional, non-cancelling. Now zero by construction.

### Gates re-evaluated

| | pre-correction | post-correction |
|---|---|---|
| F4 slope | −0.0607 ± 0.0091 | −0.0540 ± 0.0079, sign 6/6, 5/6 in band |
| F5 corr | +0.898 ± 0.036 | +0.874 ± 0.042, positive 6/6 |
| dilution R | 0.85 ± 0.04, bound 77 ± 5% | **0.82 ± 0.05, 6/6 pass, bound 67 ± 11%** |

F5's absolute level moved ~+1.7 → ~+2.4 eV — the on-site shift, exactly where it should
appear. F4 barely moves, which is the plan's forecast holding: the correction is size-level,
not d-level.

### T_el scan — a finding that bears on the N_eff story

| T_el (meV) | f(changed level) | fractional levels | N_eff |
|---|---|---|---|
| 5 | 0.000 | 0.0 | 5.51 |
| 10 | 0.000 | 0.0 | 5.75 |
| 25 | 0.000 | 0.0 | 6.93 |
| 50 | 0.000 | 0.0 | 8.76 |
| 100 | 0.000 | 0.0 | 14.47 |

`f(changed level) = 0` at every temperature is **correct, not a bug**: the counters are a hole
(`h_maj = 1`), so the changed level is the one the carrier *vacated*. Zero levels carry a
fractional occupation at any T_el, i.e. μ sits in a real gap and the majority fill is
integer — so E_head is nearly T_el-independent (−2.40 → −2.26 eV over a 20× range).

**But N_eff nearly triples over that range.** N_eff is therefore *not* a T_el-independent
property of the state; part of it is thermal smearing, entering through the minority channel
and the reference fills rather than the majority fill. **This is load-bearing for the
localisation narrative**: the N_eff numbers quoted throughout (2.92 → 4.75 → 5.26) are all at
T_el = 25 meV, and are only comparable to each other. The dilution ratio is the boundness
statement that does not have this sensitivity.

**Config decision recorded: keep T_el = 25 meV.** Nothing in the scan argues for a change —
the fill is integer at every temperature tried and E_head is flat — and changing it would
break comparability with every prior number for no measured gain.

### Saturation audit — INCOMPLETE, and reported as such

The audit ran on all six models but produced **no channel rows**: its spy looks for tanh-
argument submodules under `head.h` by name (`g`, `h`, `g_mlp`, `h_mlp`, `hop_mlp`,
`site_mlp`) and **none matched**. It fell back to parameter norms (`|v0|max = 3.087`) and the
radial envelope, and recorded the miss in the json rather than reporting a spurious zero.

**So the second §4 config decision is not yet made.** The saturation fractions, the `sech²`
gradient attenuation and the γ-bound judgement are all unmeasured. This needs the real
attribute names from `defect_bounded.BoundedLocalHead` before it is worth anything. It does
not block §5 — the decision it feeds is "keep the bounds as they are" by default — but it is
an owed item, not a completed one.

---

## §5 — six-seed rerun under the corrected kernel

6/6 trained, no init-gate trips.

| | wired (old kernel) | **§5 (full sum)** |
|---|---|---|
| trained | 6/6 | **6/6** |
| axial_red | +0.645 ± 0.010 | **+0.629 ± 0.009** |
| rmse_all | 25.3 ± 0.6 | **25.0 ± 0.5** |
| N_eff | 4.75 | **5.26** |
| pristine frontier gap | 2.400 | **2.399** |

Per seed: axial_red +0.610 to +0.638, N_eff 4.73–5.90, gap 2.383–2.416.

### Forecasts scored

| forecast | outcome |
|---|---|
| eps0 / Z / c re-settle | **met** — fit quality unchanged (RMSE 25.0 vs 25.3) |
| pristine gap re-passes within 0.1 eV | **met**, \|Δ\| = 0.001 on a 0.10 gate |
| F4 slope unchanged within CI | §6, in flight |
| dilution and F5 within spread | §6, in flight |
| cross-size consistency improves | zero by construction; the §6 dilution number is the empirical read |

### Two deviations, stated

1. **"Config path only" was not honoured.** This ran through `stage_run.py`. §3 established
   the two build a bit-identical model, so the *model* is the one the config path would build
   — but the production trainer does not carry the Stage-3 training *protocol*: Harrison
   init, the c-shift calibration, the warmup, `loss_gap` and the init gate all live in
   `stage_run`. Porting them into `mace.cli.run_train` is real work. **Owed.**
2. **§5 was launched before §4's config decisions landed**, on explicit instruction to start.
   The T_el decision has since come back "no change", which retrospectively validates the
   run. The saturation decision is still unmade (above), so if it eventually calls for a
   bound change, this run is superseded.

### On localisation

N_eff 4.75 → 5.26 is a ~10% loosening, outside the seed spread. Direction is what the
corrected kernel predicts — Pb's on-site energy moved most (−0.283 eV cross-size) and Pb is
where the carrier lives. For scale, the Stage-2 s-only control sat at N_eff 33.1 with its
best-fitting seeds at 64–70; this is an order of magnitude tighter, at better fit, with the
gap on target. **And N_eff is T_el-sensitive (above), so the dilution gate is the boundness
statement to rely on** — 0.82 ± 0.05 with 6/6 passing on the pre-correction models.

---

## §6 — gate table

Launched 18:03 on the §5 models: F4, F5, dilution and the head-vs-base discrimination, one
per GPU. F4/F5/discrimination are minutes; dilution ~27 (it forwards 892 matched small-cell
frames per model). **Results pending at the time of writing.**

---

## Owed items

1. **Saturation audit** — spy binds to no module; needs the real `BoundedLocalHead` attribute
   names. The second §4 config decision is unmade.
2. **Stage-3 protocol in the production trainer** — Harrison init, c-shift, warmup,
   `loss_gap`, init gate. Until then "config path only" cannot be honoured for a real run.
3. **F4's amplitude shortfall** — pre-correction, the head delivered 46% of what its own base
   left (residual slope −0.1310 [−0.1422, −0.1198], bracketing the −0.134 reference in 6/6),
   so the joint run will not fix it. Per the plan this is close-or-explain at R3, not a
   joint-run gate. Candidates in order: E_LR re-enable, then SCC.
4. **Z endpoints** for the §5 run not yet compared against the pre-correction run.
