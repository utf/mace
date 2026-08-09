# MACEDefect — forward plan

Supersedes `macedefect_correction_fix_plan.md`. Section references `plan §N.M` point at
`charge_aware_defect_mlip_plan.md`, `impl §N` at `charge_aware_defect_mlip_implementation.md`.

**Where we are.** The correction branch works. The 4H-SiC divacancy baseline plateaus for
~21 epochs, escapes sharply at ~22, and reaches `RMSE_dE = 43.67 meV` by epoch 42 — below
the 48 meV linear-probe floor — localising to participation 2.5 with 99.96% of attention on
the six defect first-shell atoms (`Δℓ ≈ 11.7`, meeting plan §3.2's requirement with
`λ_u = 0` and no regulariser). Stage 0 controls and the Stage 1 data-layer work are
complete and the suite passes.

**What this plan is for.** Getting from a model that learns to a model that is accurate
enough to use, without re-running anything already settled. The primary accuracy lever
(production width and `max_L`) has never been tested on a branch that actually learns, so
it comes first. Architecture changes are framed as challengers against a recorded baseline,
not as rescues.

---

## 0. Settled — do not re-litigate

Each of these was tested or argued out. Re-opening any of them needs new evidence, not new
reasoning.

| Item | Status | Why |
|---|---|---|
| Zero-init of `MLP_u` last layer | **Keep** | Measured: no-zero-init arm still on the plateau at epoch 40 (`dF 49.29`) while zero-init escaped (`dF 37.54`). Random `u` is structureless noise the optimiser must first destroy, arriving back at `u ≈ 0` with `α` uniform. Zero-init grows `u` along the gradient, so contrast is target-aligned from step 1. |
| Structured / novelty-shaped `u` init | **Rejected** | Same mechanism: a seed is a hypothesis the optimiser must test and mostly discard. Gradient-grown contrast is aligned for free. |
| `β` anneal | **Not needed** | `∂ΔE/∂u_j = n_c α_j[1 − β(u_j − ⟨u⟩_α)]` is `n_c/N ≠ 0` at `u ≡ 0`; nothing needs seeding. |
| L2 on `u` (`λ_u`) | **Keep at 0** | `Δℓ ≈ 11.7` was reached with no regulariser. The plan's bet that the data prefers localisation is empirically supported here. |
| Entropy penalty / learnable temperature | **Rejected** | Biases delocalised carriers (band-edge, shallow) toward localisation; forces spurious symmetry breaking on pristine cells; `α` is exported as `q_i^carrier`, so distorting it distorts the dilute limit. |
| Top-k / sparsemax pooling | **Rejected** | Discrete support ⇒ non-smooth PES where the support changes; fatal for phonons, Hessians, CC diagrams, NEB. Normalised softmax is also what buys intensivity and pristine exactness. |
| Separate LR for logit nets | **Rejected** | Premise refuted: `\|∇logit\|/\|∇u\| = 0.0330 ± 0.0001` across N = 286→398 at 16 and 128 channels. The `α_j` prefactor cancels; the ratio tracks `std(u)`, not `1/N`. |
| Defect-position inputs, defect-centre detection, site-similarity or novelty features | **Rejected** | plan §3.6. Non-smooth where detection flips; undefined for mobile defects, donor–acceptor pairs, delocalised states. |
| Synthetic anchor frames | **Removed** — see §A2 | Cost; and rattled anchors assert a zero target that is false, since a strained pristine cell's edge genuinely shifts by the deformation potential — which plan §3.2 wants `u` to carry. Superseded by the optional gauge penalty (§D). |
| Metrics off EMA weights | **Fixed** | Caused the original misdiagnosis. Keep EMA for the shipped model, never for training diagnostics. |
| Judging on ≤6 epochs or a 5-frame split | **Fixed** | Escape is at ~epoch 22; anything shorter measures the plateau. |

---

## Invariants — CI gates, re-run after every stage

- `E_total(R, 0) == E_base(R)` and `E_dilute(R, 0) == E_base(R)` exactly, random `R`,
  random parameters.
- `Σ_i α_i^c == 1` per cell per channel (float64, no eps).
- `Σ_i q_i^host == Σ_i q_i^pol == 0` per cell; `Σ_i q_i == a·q` (not `== q`).
- Autograd forces vs finite differences, per branch and per evaluator; Hessian symmetry;
  translation / rotation / permutation invariance.
- Optimizer coverage guard (every trainable parameter in a group).
- **Dead-channel invariant.** For any channel with `n_c = 0` across all frames, that
  channel's readout parameters must be bit-identical before and after training. `h_maj` is
  dead in this dataset and its `std(u) = 0.0534` (unchanged from init) is the control that
  proves live-channel shrinkage is gradient-driven, not weight decay. It also catches
  gradient-leak bugs.
- **Precision policy.** Train float32 (4.5× on the A4000; FP64 is 1:32 there). Run the
  invariant suite in float64 — exactness assertions, finite differences, and from §E the
  Ewald k-space sums. Energy conservation in MD must be checked at production precision.

**Do not change:** counter algebra and canonicalisation, band-edge referencing on raw
energies, residual-on-reference-base decomposition, structured latent-charge assembly, the
three-piece decomposition and `E_dilute`, absence of a separate force head, absence of any
defect-position input.

---

## Stage A — consolidate, then set a baseline of record

### A1. Flip the zero-init default

When seeds 2 and 3 agree with seed 1: set `DEFECT_ZERO_U_INIT = True` as the default, fix
the test that asserted its removal, and record the rationale from §0 in the code comment
**and** in plan §9.6. The original §9.6 instruction was right for a reason it did not state
(optimisation conditioning, not the `n = 0` identity, which is structural via the counter
prefactor) — record the correct reason so it is not removed again by someone re-deriving
from the stated one.

Judge the seeds on **escape epoch**, not on validation RMSE. Escape is sharp and cheap;
`43.67` vs `48 meV` on 5 frames is inside that split's noise.

### A2. Remove anchors and regenerate the dataset

Remove from the generator and the loader:

- synthetic anchor frame generation (all single- and multi-carrier variants);
- `--anchor_weight`, `--anchor_weight_rattled`, and any `synthetic_anchor` weighting path;
- the anchor↔referencing edge-consistency assertion.

Keep, unchanged:

- corrected plan-convention labels — ground `(1,0,0,1)`, excited `(1,1,0,2)`, pristine `0`;
- `multiplicity` required for spin-polarised frames, with `M_s == multiplicity − 1`;
- the generalised pair join (reference member = unique member with minimal `Σ_c n_c`);
- `n_ref` plumbing in `MACEDefect.forward` — **this is reused by §D**, do not remove it;
- `L_tot` weight 0 (no base labels at defect geometries under corrected labels);
- pristine `n = 0` frames — they train the base branch and supply the geometries §D needs;
- `--max-natoms`, the 129-frame `dataset_beta`.

Bump the dataset version string. Removing frames changes loss composition and epoch length,
so **no metric from the anchored dataset is comparable to one from the regenerated dataset**
— including the 43.67 meV figure above.

Replace the removed assertion with a weaker one that still has value: the referencing table
must resolve to identical edge constants for every frame sharing a `(host, cell)` key.

### A3. Confirm the referencing state

Verify whether band-edge referencing is live or whether the interim fitted gauge
(`E_gap = ⟨ΔE_vert⟩ = 0.943 eV`) is still in use. If the latter, it is a real error source
*within this dataset*, not just a transferability issue: `E_g^cell` differs between the 286-
and 398-atom cells, so a single fitted constant injects a systematic per-cell offset that
will surface the moment held-out sets are split by cell size. Resolving it needs `E_g^cell`
per distinct pristine supercell — see the external-inputs table. Until it lands, split
held-out sets *within* cell size and record the limitation.

### A4. Baseline of record

Re-run on the regenerated dataset: ≥3 seeds, EMA off for metrics, 200 epochs, validation
split expanded to ≥20–30 frames. Recompute the linear-probe floor on that same split — a
probe fitted on all data compared against a 5-frame model score is not like-for-like.

Log per epoch, per channel: `RMSE_dE`, `RMSE_dF`, escape epoch, participation ratio,
`Δℓ`, `mean(u^c)`, `std(u^c)`. Compute and record the §8.2 size-error bound.

Everything downstream competes against this number.

### RESULTS — Stage A as executed (2026-08-08)

Full detail in `defect-example/HANDOFF.md` §12–§15. Dataset is now **v4** (v3 = anchors
removed; v4 = unpaired frames restored for A5). Metrics are not comparable across versions.

**A1 — zero-init default flipped to `True`.** Done in the arg parser, the model, the
pooling block and the run script, with the conditioning rationale recorded in code and an
explicit warning against re-deriving from the `n = 0` identity. Two tests now construct
with `zero_u_init=False` explicitly so they pin mechanisms rather than the default.

> Evidence caveat: §0 justifies this from seed 1 alone. Seeds 2–3 only partly agreed —
> `ab_noinit_s3` escaped at epoch 37 and finished best of all six runs. The real tally is
> escape in **3/3 with the zero-init vs 1/3 without** (Fisher p ≈ 0.4). The flip follows
> the escape-reliability tally, not a settled result.

**A2 — anchors removed, `v3`.** Generation, CLI flags, config types, `_is_anchor` and the
anchor-vs-referencing assertion all gone. The replacement check — frames sharing a
`(host, cell)` key must resolve to identical edges — was **upgraded from a warning to a
hard error**; as a warning it was invisible.

**A3 — referencing is still the interim fitted gauge.** `E_gap = 0.9430 eV`, exactly the
mean vertical excitation, not `E_g^cell` per supercell. Needs charged pristine totals,
which cannot be produced now. Every defect cell size appears in **both** splits already,
so held-out sets are split within cell size as this section requires.

**A4 — diagnostics landed; the floor was wrong.** `DefectRMSE` now logs participation,
`mean(u^c)`, `std(u^c)` and the logit gap per channel every epoch.

> **The long-quoted 48 meV probe floor was in-sample.** Fitted on train and scored on the
> same validation split the model uses, the probe gives **53.72 meV (R² = 0.806)** on
> v4 — that is the number to beat. Earlier claims of "beating the 48 meV floor" compared a
> held-out model score against an in-sample probe score.

**Two invariant violations were caught by the new diagnostics.**

1. *Dead-channel invariant failed on first run.* Not structural — the counter prefactor is
   sound (dead-channel gradient exactly `0.000000e+00` at `u_l2 = 0`). The leak was the
   **L2 on `u`**, which is a mean over all channels and so updates channels no frame
   occupies. `--defect_u_l2` and `DefectLoss` now default to `0.0`, matching §0.
2. *The per-channel decomposition is seed-dependent.* Same data, same config: seed 1
   localises `e_maj`/`h_min` (participation 2.4 and 3.2 atoms), seed 2 localises `e_min`
   (2.8) and leaves the others near 85. Only the total `ΔE` is constrained, so the channel
   split is not identified — measured confirmation of §A5.1, and the reason `Δu` cannot be
   read as a per-channel binding energy without §D-opt.

**A5 — `L_tot` restored, `RMSE_F` fixed.** Applies at every `n ≠ 0` frame with `E_base`
receiving gradient. Forces now train (802 → 57 meV/Å and falling) where they were pinned
at ~145 and never moved. The gauge report logs every run and confirms **1 free direction**
for this counter set, exactly as §A5.1 predicts.

> **§A5.3's `base_lr_factor` is wrong at this stage.** Measured one knob at a time,
> 25 epochs: `0.25` gives RMSE_E 789 meV/atom, `1.0` gives **18**. A 40× effect, dwarfing
> every other knob. The "moving target" argument assumes an already-trained base branch;
> this one starts from random init and must learn the whole SiC potential, so quartering
> its LR merely leaves it undertrained — which `L_tot` then exposes. Default is **1.0**;
> lower it only when warm-starting, e.g. the §E retrain.

> **A correctness trap in restoring `L_tot`:** `delta_energy` is now the paired difference
> against `n_ref` and equals the correction only when `n_ref = 0`. The term must use
> `pred["energy"]`. Using `delta_energy`, as the old unpaired-only version did, would have
> compared the total against `E_base + [corr(n) − corr(n_ref)]` once extended to paired
> frames.

**Model size and speed** (§15). The correction branch is 68.8% of parameters at 8ch but
only **9.7% at 128ch/`max_L=1`** — it is nearly width-independent (52k → 60k), and
`carrier_pooling` is 96.5% of it. `counter_embedding_dim` and `carrier_mlp_hidden` halved
to 16/32. But parameters are *not* the cost: the correction's forward pass is free
(38.90 vs 38.97 ms) and shrinking the heads changes nothing; its entire ~14 ms overhead is
the two extra `autograd.grad` calls for `n_ref`.

**cuEquivariance is enabled for `MACEDefect` and is a memory win.** Verified numerically
identical to e3nn (9e-14 energies, 2e-16 forces, float64). At 128ch/`max_L=1` on a 16 GB
A4000 it uses **4.5× less memory** than e3nn, which lets the batch grow from 2 to 8, giving
**17.6 ms/frame vs 171.7 — about 10×**. OpenEquivariance is the faster single step at
batch 2 but takes 2.5× cueq's memory and OOMs at batch 8. cueq *loses* below ~32 channels
(0.85× at 8ch), so it stays off by default and on for production width. The cueq+oeq hybrid
does **not** work in this tree: the cueq conversion does not persist `cueq_config` on the
model, so chaining drops the cueq half and fails on a state-dict mismatch.

## Stage A5 — restore `L_tot` without closed-shell reference calculations

**Problem.** `L_tot` is off because corrected labels leave no `n = 0` reference at defect
geometries. Consequences: total forces at defect geometries are supervised by nothing
(`RMSE_F ≈ 145 meV/Å`, not a regression — a gap), and unpaired defect frames contribute
nothing and were dropped. The original remedy — constrained `M_s = 0` singlepoints — will
not be run for spin-polarised systems in practice, so the gauge must be fixed another way.

### A5.1 The gauge, stated

`E_base` at defect geometries is a **gauge**, not an observable. Under
`E_base → E_base + f(R)` with the correction absorbing `−f`, every label at every observed
counter is unchanged. Since the correction is `Σ_c n_c v_c(R)`, absorbing `f` requires
per-channel factors `s_c` with

```
Σ_c n_c^(k) s_c = −1     for every observed counter k
```

With the two counters here — `(1,0,0,1)` and `(1,1,0,2)` — and three live channels, this is
one equation short of determined. A one-parameter family of solutions exists using linear
channel trading alone. **This family is what the seed-dependent channel split is sampling**
(seed 1 localises `e_maj`/`h_min`, seed 2 localises `e_min`; `mean_u` for `e_maj` differs by
0.65 eV). Restoring `L_tot` adds no new counter and will therefore **not** resolve the
channel split — do not expect that symptom to clear here.

### A5.2 What the architecture already protects

The correction is `Σ_i α_i u_i` with `Σ_i α_i = 1` — **intensive**. `E_base` is a sum of
local energies — **extensive**. A bulk-wide error in the base is therefore structurally
unabsorbable by the correction, at any localisation of `α`, including channels sitting at
participation ≈ N. Normalisation does this, not localisation.

Consequence: un-detaching `E_base` in `L_tot` is safe on the part that matters. Force
residuals on atoms away from the defect can only be reduced by the base branch, which is the
mechanism that teaches it undercoordinated chemistry and fixes `RMSE_F`. The residual
freedom is confined to the defect-localised intensive part — the physical spin-polarisation
energy — and nothing else.

### A5.3 Change

Enable `L_tot` at `n ≠ 0` with `E_base` receiving gradient (remove the `stopgrad`). This
supervises total forces at defect geometries and restores the unpaired frames in one change.

Settings:

- base branch on a **low learning rate**; pristine frames replayed at high weight (the base
  is otherwise a moving target for the correction);
- unpaired frames: **full** force weight, **moderate** energy weight — their energies are
  not useless, they constrain the extensive part;
- weak L2 on the `z(n)` input weights of `MLP_u` — see A5.4 for the rationale, which is
  *not* gauge suppression;
- log `E_base` predictions at defect geometries against the pristine-only-trained
  predecessor each epoch, and watch the drift.

### A5.4 What actually kills the gauge: counter dependency

Not counter volume — counter **dependency**. `g(n) = Σ_c n_c s_c` is linear, so
`g(n₁ + n₂) = 2`, contradicting the required `g = 1`. Any observed counter that decomposes
into a sum of two other observed counters over-determines the system and forces `f = 0`.

For this defect:

```
anion   (1,0,0,0)   doublet, q = −1
cation  (0,0,0,1)   doublet, q = +1
neutral (1,0,0,1) = (1,0,0,0) + (0,0,0,1)     ← the dependency
```

The `q = ±1` states required for transition levels are exactly the states that close the
gauge. Treat them as **identifying data, not optional**.

The over-determination is exact only if the correction is linear in `n`; genuine
multi-carrier interaction (plan §3.3) makes it approximate. The weak `z(n)` L2 above exists
to keep that nonlinearity small enough for the cancellation to bite, while leaving real
electron–hole physics representable. That is its rationale — it is not a gauge-fixing
device, and an earlier justification claiming it suppresses `f` directly was wrong.

**Loader check to add:** for each `(host, defect, geometry family)`, list the observed
counter vectors and report whether any is an integer sum of two others. Report identified /
under-determined, with the remaining null-space dimension.

### A5.5 Second route: closed-shell coverage from naturally singlet sources

Any defect or configuration in the same host with a genuinely singlet ground state supplies
real `n = 0` labels at undercoordinated environments — no constrained calculation. `E_base`
at the magnetic defect then becomes a prediction from real chemistry rather than a free
function. This is a data-coverage need, not a defect-chemistry constraint, and it is the
route available before charged data exists. The two routes compose.

### A5.6 Status where neither route is available

`E_base` at defect geometries is **latent**: `E_total(R, 0) = E_base(R)` remains
structurally exact but is no longer a validated prediction. No deliverable requires it —
formation energies, transition levels, CC diagrams, phonons and barriers all live at
physical counters, and pristine references are labelled. The single exposure is **counter
extrapolation**, with error `∝ f`, which is the transition-level path.

Therefore: do not ship transition levels from a neutral-only model.

### A5.7 Gate

Hold out a counter entirely — train on the ground state (plus whatever else is available),
predict the excited state, compare against the paired-training result. This measures
counter extrapolation directly rather than by proxy, and is the acceptance test for A5.

If a naturally singlet defect in the same host ever becomes available, score the predicted
`n = 0` surface against it once. That validates the whole scheme and need not be repeated.

### A5.8 Magnetic hosts

If the **host** is magnetic, `E_base` is ill-defined rather than merely unobserved. The fix
is to redefine the reference as the host's own ground-state spin configuration and re-derive
canonicalisation against it. The model algebra is untouched; only the gauge definition
moves. Decide this explicitly before any magnetic host enters the roadmap, since it changes
what "canonical" means across systems.

---

## Stage B — capacity: the accuracy lever that has never been tested

User comment: skip this stage for now.

The branch has only ever been trained at 8/16 channels with `max_L = 0` and `r_max = 4.0`.
Descriptor resolution was never the blocker for *learning* — the baseline beats the linear
probe — but it is the obvious ceiling on *accuracy*, and `max_L = 0` gives no angular
resolution at all around a defect whose gap states are directional dangling bonds.

Sweep, in this order, ≥2 seeds each, held-out metrics from A4's split:

1. 128 channels, `max_L = 0` — separates width from angular resolution.
2. 128 channels, `max_L = 1` — production configuration.
3. If (2) helps and is affordable, probe `r_max` at 5.0–6.0 Å.

Also record **escape epoch at production width**. The gradient ratio was measured
width-independent, but plateau length at 128 channels is unmeasured and feeds §C's decision.

Retune the LR schedule for the post-escape regime — the 200-epoch run was a diagnostic, not
a tuned schedule, and the useful training only starts around epoch 22. A schedule whose
decay is calibrated to a 200-epoch run spends most of its budget on the plateau.

**Gate:** held-out `RMSE_dE` improves materially over A4. If width and `max_L` give little,
the limit is data volume (129 frames) and the next investment is the dataset, not the
architecture.

---

## Stage C — is the plateau a production hazard?

One measurement, decides whether §D is worth doing. The refuted claim was that the *logit*
gradient is `1/N`-suppressed; the untested one is that the defect-specific fraction of the
`u` gradient is `n_d/N`, which predicts plateau length ∝ N.

- **Cheap version, no need to wait for escape:** log `std(u)` restricted to the six defect
  atoms per epoch during epochs 1–20 and fit the growth rate across cell sizes. Prediction
  if the mechanism holds: rate ∝ 1/N.
- **Direct version:** train on N = 286-only and N = 398-only subsets — equal frame counts,
  equal steps/epoch, matched LR, 3 seeds — and compare escape epoch. Linear scaling
  predicts 22 → ~30. If small cells (≤128 atoms) are cheap to generate, add them; the 39%
  lever arm here is narrow.

**Decision:** rate ∝ 1/N or escape epoch rising with N ⇒ the plateau is a production-scale
hazard and §D is justified on dynamics. Both flat ⇒ the plateau is a fixed ~20-epoch cost
and §D must justify itself on the `Δu` diagnostic alone.

### RESULT (2026-08-08): escape epoch rises with N. §D is justified on dynamics.

Direct version, matched subsets on dataset v4 — 135 configs, 9 base labels, 63 delta
targets and 126 `L_tot` frames in **each** arm, identical 288-atom pristine pool, 8ch
`max_L=0`, EMA off, 150 epochs, 3 seeds. Only the defect cell size differs.

| seed | 286 atoms | 398 atoms | ratio |
|---|---|---|---|
| 1 | 22 | 116 | 5.27 |
| 2 | 50 | 59 | 1.18 |
| 3 | 43 | 57 | 1.33 |
| **median** | **43** | **59** | **1.37** |
| mean | 38.3 | 77.3 | 2.02 |

Later at 398 in **3/3 seeds**, and the median ratio 1.37 lands on the cell-size ratio
398/286 = **1.39** — the linear scaling this section predicts.

Escape is defined as the first epoch whose `RMSE_dF` drops 10% below *that arm's own*
plateau value. An absolute threshold would be wrong: `RMSE_dF` averages over all atoms
while the delta forces sit on six, so the larger cell starts lower (37.81 vs 50.66) by
dilution alone, before any training.

Control: the base branch trained essentially identically in both arms (`RMSE_F` 694→42 vs
698→43, and likewise for seeds 2–3), so cell size is doing the work.

Trust: 3 seeds, paired sign test p = 0.125 — consistent, not conclusive. Variance is large
and the mean ratio is inflated by seed 1. The 1.37-vs-1.39 agreement is closer than three
noisy seeds can support; read it as *consistent with linear*, not a measured exponent. The
1.39× lever arm is narrow, as noted above, and no cells ≤128 atoms exist in this dataset.

**A prerequisite this measurement discovered.** The first attempt was void: both arms were
built with `--frac-ideal 0` to match loss composition, which removed the only `n = 0`
frames and so *all* base-branch labels. `RMSE_dF` was then bit-identical for 120 epochs in
every run, and `RMSE_F` sat at its untrained value. The finding is worth keeping in its own
right: **the correction branch cannot bootstrap from delta targets alone.** With a shared
trunk and no base labels, the trunk's only gradient arrives through the correction, which
during the plateau is small and structureless, so the features never develop and the
attention never has anything to localise onto. Never train this model on a delta-only
dataset. `--defect-natoms` / `--ideal-natoms` were added so cell size can be varied while
the pristine pool is held fixed.

---

## Stage D — the tie (challenger, not a rescue)

### D1. Form

Delete `MLP_ℓ` and `--share_logits_across_spin`. In `mace/modules/defect_blocks.py`:

```
u_i^c  = MLP_u^c([h_i, z(n)])            # carrier site energy, eV
α_i^c  = segment_softmax(−β · u_i^c)     # per cell, per channel
ΔE_SR  = Σ_c n_c Σ_i α_i^c u_i^c         # plain expectation — no entropy term
```

`β = 10 eV⁻¹`, fixed and recorded with the model like `σ` and `k_c`. Do not use the
free-energy form `−β⁻¹ log Σ e^{−βu}`: it drifts as `log N` on pristine cells and breaks
plan §3.2's exactness. Keep zero-init (§0). Everything downstream of `α` is unchanged.

### D2. Why it might still be worth it

1. **Plateau elimination.** `α` responds to `u` instantaneously rather than through
   separate parameters that must themselves train, so the tied model should have a much
   shorter plateau or none. Worth real money only if §C came back scaling.
2. **`Δu` as a physical binding energy.** Turns the §8.2 bound from a statement about an
   arbitrary logit scale into a checkable energy. Strongest remaining reason, independent
   of §C — but only meaningful with §D-opt (below) on, since an open level mode makes
   `mean(u^c)` arbitrary.
3. Removes the additive gauge freedom in `ℓ`; forbids incoherent states (bound but
   delocalised, localised but unbound); fewer parameters.

### D3. Diagnostics and tests

Replace `logit_gap` in `DefectRMSE` with `Δu` per channel (eV) and participation ratio per
channel; `Δℓ = β·Δu`. Restate the size-error bound as `(N/n_d)·Δu·e^{−βΔu}`.

Add: `α` exactly uniform per symmetry class on a pristine cell; re-run finite-difference
forces and Hessian symmetry (forces now carry `−β·n_c·Cov_α(u, ∇u)`); and a
**carrier-transfer scan** — two competing localisation sites, sweep a coordinate that moves
the carrier between them, check `ΔE(Q)` and `ΔF(Q)` for kinks. The tie renders the transfer
as a softmin switch of width `~1/β`; this scan is what decides whether D4 is needed.

### D4. Residual head — only if D3's scan or the gate demands it

```
ΔE_SR = Σ_c n_c Σ_i α_i^c (u_i^c + v_i^c),   α = softmax(−β u)   # v does not enter α
```

with a weak L2 **on `v`** — the head that does not control the weights. Do not add
pre-emptively.

### D5. Gate

Held-out `RMSE_dE`/`RMSE_dF` no worse than the Stage B best within seed noise, **and** at
least one of D2(1) or D2(2) realised. Reject on any accuracy regression: the baseline works,
and the tie does constrain how `α` and the energy move together along a coordinate even
though any single `(α, E)` pair stays representable.

### RESULT (2026-08-08): gate REJECTS the tie as it stands. `MLP_ℓ` not deleted.

3 seeds, v4 dataset, 40 epochs, identical to the A4 baseline except `alpha_mode`:

| run | escape | final dE | final dF |
|---|---|---|---|
| baseline s1/s2/s3 | 15 / 14 / 16 | 37.16 / 38.50 / 50.56 | 26.63 / 27.16 / 29.64 |
| tied s1 | **none** | **130.52** | 45.49 |
| tied s2 | **1** | 36.30 | 27.88 |
| tied s3 | 16 | 55.74 | 27.23 |

**D2(1) is realised, spectacularly, in 1/3 runs** — escape at epoch 1 against a baseline of
14–16, and the best `dE` of all six runs. **Accuracy regresses in 1/3 runs**, not
marginally: 130.52 meV, worse than predicting the mean. D5 rejects on any accuracy
regression, so the tie is not adopted and `MLP_ℓ` stays.

**The failure is lock-in onto the wrong atoms, and it is measured.** Autopsy of the final
models (6 defect-shell atoms of 382; uniform attention gives 0.0157):

```
                partic  partic/N  mean_u  std_u   a_defect
tied s2  e_maj     2.5     0.007   2.274  0.325     0.9892
tied s1  e_maj    92.5     0.253   2.181  1.924     0.0000
```

The successful run puts **98.9%** of its attention on the six defect neighbours — the
cleanest localisation obtained so far. The failed one puts **exactly zero** there, having
committed to ~92 non-defect atoms. Both show a large `Δu` and a wide logit gap, so every
diagnostic except `a_defect` looks healthy in the failed run.

**§0's dismissal of the `β` anneal answers a different question.** The stated reason —
`∂ΔE/∂u_j = n_c α_j[1 − β(u_j − ⟨u⟩_α)]` is `n_c/N ≠ 0` at `u ≡ 0`, so nothing needs
seeding — is correct, and the runs do seed fine. The failure is *premature commitment*:
`β = 10 eV⁻¹` makes a 0.3 eV accident in `u` a factor ~20 in attention immediately, and the
tie is a positive feedback loop with no slowly-training logit network to damp it. The
superseded fix plan's `β` anneal (2 → 10 over the first epochs) targets exactly this and
should be tried before the tie is abandoned. §D-opt would not help: it constrains the
*level* of `u`, not which atoms are selected.

Caveat: 3 seeds, 40 epochs; a 1/3 failure rate rests on one occurrence. The existence of
the failure mode, and that `a_defect` identifies it, are not in doubt.

### D6. Re-diagnosis: the plateau is exploration

The gradient reaching site `j` is gated by `α_j`, so unattended atoms are gradient-starved
and whichever atoms hold attention when contrast develops are the only ones that can later
be refined. The baseline survives because `MLP_ℓ` starts uniform and trains slowly, holding
the gate open for 14–16 epochs while the data decides. The tie closes it immediately, with
no slow network to damp the feedback. Stage C measured the *exploration cost*; eliminating
it is only desirable if the exploration completes.

**The recovery window is `1/β`.** With `x = β(u_j − ⟨u⟩_α)`, the gradient magnitude goes as
`x·e^{−x}` — peaking at `x = 1`, decaying exponentially beyond. At `β = 10 eV⁻¹` that window
is 0.1 eV; atoms outside it are permanently unreachable.

**The order parameter is `β·std_u`**, and it is label-free:

```
tied s2 (good)   std_u 0.325   β·std_u ≈ 3.3    a_defect 0.9892
tied s1 (failed) std_u 1.924   β·std_u ≈ 19.2   a_defect 0.0000
```

Lowering production `β` is not an alternative: `Δℓ = β·Δu ≳ 11` (§3.2) puts a floor under
it. `Δu` also remains uninterpretable — the channel split is unidentified with two counters,
and the identifying `q = ±1` data is deferred to the next test case. **D2(2) is retired;
D2(1) is the tie's only remaining justification.**

### D7. Two mechanisms

**Seeding (D7.1)** attacks the plateau's cause — `u ≈ 0` zeroes the logit gradient — and is
usable in the untied model, where `MLP_ℓ` still starts uniform so there is no feedback loop.
**Annealing (D7.2)** slows commitment and is required wherever the tie is used. They
compose; the experiment order (D8) tries the cheaper one first.

#### D7.1 Data-derived contrast seeding (training-only initialisation)

**Descriptor.** Reuse the model's own one-particle basis with the species embedding replaced
by one-hot, making it fully deterministic in the geometry, and contract over `m` for
rotation invariance:

```
p_i^{(n,l,Z)} = Σ_m | Σ_{j∈N(i)} R_{n l}(r_ij) Y_l^m(r̂_ij) δ_{z_j,Z} |²   # power spectrum
p̂_i          = p_i / std_dataset(p)                                       # per (n,l,Z) channel
s_i          = ‖ p̂_i − mean_{j : Z_j = Z_i} p̂_j ‖                          # same frame, same species
ŝ_i          = s_i / σ_dataset                                             # one global constant
û_i          = −ε · ( ŝ_i − mean_frame(ŝ) )                                # CENTRED — the seed target
```

- **Equivariant components must not be used raw** — `‖A_i − mean(A)‖` over components is not
  rotation-invariant and would seed rotated copies of a frame differently. The per-`(n,l)`
  power spectrum is invariant by construction.
- **Whiten before the norm.** Low-`n`, low-`l` blocks dominate by orders of magnitude, so an
  unweighted norm collapses to a coordination count and discards the angular information.
- **Angular terms are available even at `max_L = 0`**: the edge basis computes them
  regardless of output irrep order, so the seed can see directional dangling-bond asymmetry
  that the current readout cannot.
- Inherits the model's `r_max` and cutoff envelope, so no second notion of locality enters.
- **Computable in one pass at dataset build time** — geometry only. Cache `ŝ_i` per frame;
  the seed is exact at step 0 with no random-feature noise, and the descriptor never enters
  the training loop.

**Centring is what makes this work.** `mean(u) = 0` gives exactly zero-init's energy under
uniform `α`, so nothing pressures the seed's destruction (the failure of the withdrawn §2.3
seed), while `∂ΔE/∂ℓ_j = n_c u_j/N` is live and correctly directed from step 0. Global
scale, per-frame centring, in that order: pristine frames then get `ŝ ≈ 0` and `α` stays
uniform, preserving §3.2's pristine exactness at init. The per-frame per-species mean needs
no pristine reference — bulk atoms dominate (376/382 here) — and catches antisites, which
coordination counting misses. Sign: carriers bind where `u` is low, so novel atoms get
negative `u`.

**Application.** Solve for `MLP_u`'s last layer (weights + bias) to reproduce `û_i` from the
hidden activations over a sample of frames — one linear solve against a fixed target.
Magnitude `ε ~ 1/β`; not a critical hyperparameter when D7.2 is active, since the
order-parameter schedule starts at `β·std_u = τ_0` for any seed scale.

**Compliance and recording.** Training-only: not an input, not in the energy function,
absent at inference, no smoothness implications — §3.6 untouched. Record
`(n_max, l_max, r_max, envelope)`, the per-channel scale constants, `σ_dataset` and `ε`; the
seed is then exactly reconstructible without a checkpoint.

**Risk.** The prior is wrong for shallow and effective-mass states, which §2.3 requires to
stay representable, and a wrong seed looks healthy. Seed only, never a penalty; keep
`ε ~ 1/β` so the data can override; state it as a per-host assumption.

**Free by-product.** While building the statistic, histogram `ŝ_i` computed with `l = 0`
only against all `l`. If the defect shell separates more cleanly with angular terms in, that
is direct evidence for Stage B's `max_L = 1` sweep — obtained before paying for equivariant
message passing.

#### D7.2 `β` anneal on the order parameter

Replace the fixed training `β` with

```
β_t = clip( τ_t / max(std_EMA(u), ε),  β_min,  β_prod )
```

`std_EMA(u)` is an EMA over training steps of per-cell `std(u)` pooled over live channels
(use the EMA, not the batch value, to avoid stochastic `β`). `τ_t` ramps from `τ_0 ≈ 1` so
`β_t` reaches `β_prod = 10 eV⁻¹` and holds for the final third of training; `β_min ≈ 1`,
with `ε` flooring `std_EMA` at the zero-init start. Scheduling on the order parameter rather
than on epochs needs no tuning against dataset size and — given Stage C's linear N-scaling —
automatically lengthens the soft phase on larger cells.

Required alongside:

- **Do not let the LR decay through the ramp.** As `β` rises, `⟨u⟩_α` falls below `mean(u)`,
  so the energy fit tracks a moving target by shifting `u` upward.
- **Keep delta-force weight high during the ramp.** Under uniform `α` the energy data
  constrains only `mean(u)`; delta-forces are the only channel resolving *which* atoms
  matter. If the anneal fails, check whether `RMSE_dF` fell during the soft phase — if not,
  the soft phase carried no information and no schedule will help.
- **Termination is a gate.** If `β_t < β_prod` when the ramp ends, `std_u` never settled;
  report as failure rather than forcing the switch.
- Run FD-force and Hessian-symmetry tests at production `β` only.

### D8. Experiment order

Run in order, stop as soon as the objective is met. Three seeds each; adopt only on 3/3
reliability with held-out `dE`/`dF` inside the Stage A baseline spread (37–51 meV `dE`).

1. **Seeded baseline** (untied, D7.1, no anneal). Cheapest, lowest risk, targets D2(1) in
   the architecture already known to be 3/3 reliable. *If escape falls from 14–16 to a few
   epochs at baseline accuracy, D2(1) is realised without the tie: park the tie permanently
   and keep `MLP_ℓ`.*
2. **Annealed tie** (D7.2) — only if (1) fails to shorten the plateau.
3. **Seeded + annealed tie** — only if (2) is reliable but slower than target.
4. **Re-run the Stage C protocol** on whichever variant is adopted. Whether the linear
   N-scaling survives is the entire case, and no reliability result answers it.

If lock-in persists at low `β`, the exploration phase is uninformative and scheduling cannot
fix it: park the tie. Do **not** reach for D4 — with `ΔE = Σ α_i(u_i + v_i)`,
`∂/∂v_j = α_j` is still gated by `α`, so the residual head addresses derivative coupling and
carrier-transfer smoothness, not starvation.

Stage E proceeds in parallel throughout. The plateau is a bounded training-time cost; the
long-range branch letting `ΔE_SR` absorb electron–hole physics is a correctness issue.

### D9. Diagnostics and early abort (supersedes D3's monitoring set)

Both tied runs showed a large `Δu` and a wide logit gap, so those do not separate success
from catastrophic failure. Log per channel per epoch:

- `β·std_u` — commitment order parameter, label-free, primary abort signal (abort above ~5);
- `Σ_i α_i ŝ_i` — attention–novelty overlap from D7.1's statistic, label-free, the abort-rule
  substitute for `a_defect`;
- `a_defect` — summed attention on the defect shell (uniform gives `n_d/N` = 0.0157 here).
  Research diagnostic only; never a selection rule, since it needs labels unavailable on an
  unknown system;
- `mean_u`, `std_u`, participation ratio, `Δu`, held-out `dE`/`dF`.

**Safety net, adopt regardless of which variant wins.** Failure is detectable within a few
epochs, so run k short probes, abort divergent ones, continue the survivor — ~1.2× cost
rather than 3×. Select on held-out `dE` and `β·std_u` only. If the seed and the abort rule
share a statistic they partly measure each other, so held-out `dE` is the primary gate and
overlap is secondary.

---

## Stage D-opt — level-mode gauge penalty (optional, flag-gated, default off)

**What it fixes.** The softmax is shift-invariant, so a uniform offset in `u^c` moves the
energy without moving `α` — one free direction per channel. With `Δn = (0,1,0,1)`, three of
four channel directions are unidentified here. Measured: `mean(u^{e_min}) = 0.1699` against
an anchored `h_min` at `0.0014`.

**What it does not fix, and never did.** Not extensivity — a uniform offset contributes
`n_c·c` independent of N because `Σα = 1`. Not the §8.2 bound's validity — both places `Δu`
enters are *differences*, invariant to the offset. The level mode matters only for
cross-counter transfer and for reading `Δu` as a binding energy.

**Form.** Not extra frames. Reuse the `n_ref` plumbing: on pristine frames already in the
batch, evaluate the readout at the dataset's counters from the trunk output already
computed, and penalise each channel's pooled value.

```
L_gauge = Σ_c w_c ⟨u^c(h_i, z(n))⟩_α ²      # pristine frames only; α is uniform there
```

Cost is a few readout MLP passes against a message-passing forward — negligible. No new
DFT, no new frames, no thermal-disorder assertion, nothing per-site: the constrained
quantity is a cell-level pooled scalar on a defect-free cell.

**Implementation.** `--defect_gauge_weight` (default `0.0`), applied only to frames flagged
pristine, evaluated at every counter vector present in the training set. Log `mean(u^c)`
per channel per epoch whether or not the penalty is on — that is the free diagnostic that
tells you whether the level mode is drifting.

**When to run it.** As an A/B against the Stage B or Stage D best: (i) when `Δu` is to be
read as a binding energy; (ii) before adding a second counter direction, a second defect,
or charged states, where the unidentified offsets become a real transfer error; (iii) if
cross-cell or cross-counter errors appear. **Acceptance:** `mean(u^c)` collapses toward the
`1e-3` scale on all live channels *and* held-out error is not worse. If held-out error
degrades, the penalty is fighting the data — report and leave it off.

**Known limitation.** A smooth network could satisfy the penalty at pristine geometries and
drift elsewhere. That is no longer a flat direction, so it is disfavoured; if it appears,
evaluate on all available distinct pristine supercells.

---

## Stage E — restore the long-range branch

The LR branch is **not** inert at `q = 0`. With `n = (1,0,0,1)`,
`q_i^carrier = a(α_i^{e·maj} − α_i^{h·min})` sums to zero but is pointwise non-zero wherever
the electron and hole distributions differ — a compensated dipolar latent charge whose
self-term is the electron–hole interaction. `q_i^pol` is live too, carrying `Σ_c n_c`. Part
of the observable belongs there, and `ΔE_SR` is currently absorbing it.

1. Re-enable `use_long_range=True` with `a` **frozen** at `1/√ε_∞` (no gradient to
   `MLP_a`). `a` is not identifiable from a dipole term alone; it stays frozen until §F.
2. Retrain from the Stage B/D checkpoint; measure how much energy `ΔE_SR` gives up to
   `E[S_carrier]`. A material transfer is the expected result.
3. Re-check the Stage B/D held-out metrics after the transfer, and the invariant suite in
   float64 (k-space sums).

**Precondition, deferred until now:** sanity-check `α` before it is exported as
`q_i^carrier`, since an over-localised `α` gives a wrong e–h term even when `ΔE` is right.
Two eigenvalue-free checks: (i) on a relaxed symmetric ground-state geometry, permutation
invariance forces `α` equal across the three equivalent dangling bonds, flooring
participation at ≈3 — the measured 2.5 came from a lower-symmetry frame and should be
re-measured on the relaxed structure; (ii) `α` on corresponding atoms should be stable
between the 286- and 398-atom cells. Drift with N means `α` is absorbing fit.

---

## Stage F — production scale and a charged system

1. Production width and `max_L` from Stage B; full invariant suite.
2. First charged defect in the same host. Unfreeze `a`; the plan §3.4 monopole identifiers
   become live. Only now is the plan §9.7 step-7 gate (`1/a²` vs DFPT `ε_∞`) meaningful.
3. Run the multi-cell size-series discriminator (plan §9.7 step 5) on the undertrained
   charged model before scaling the data campaign.
4. Note that a single counter vector per charge state means the counter embedding is
   currently untested for multi-state conditioning; that only gets exercised here.

---

## External DFT inputs

| Input | Needed by | Size |
|---|---|---|
| `E_g^cell` per distinct pristine supercell (plan §2.5 convention: `E(N±1) − E(N)`, same cell, same background, no image correction, band-filling handled) | §A3 referencing | 2 calcs × ~4 cells |
| DFPT `ε_∞` (PBEsol, 4H-SiC; uniaxial — record both components, use the isotropic average) | §E frozen `a` | 1 calc |
| `q = ±1` states of the same defect at shared geometries | **identifying data** for the `L_tot` gauge (A5.4) and for the channel split; also required for transition levels | as for the charged campaign |
| Naturally singlet defects/configurations in the same host, labelled `n = 0` | A5.5 — closed-shell coverage at undercoordinated environments without constrained calculations | opportunistic |
| ~~Closed-shell `M_s = 0` constrained singlepoints~~ | **Superseded by A5.** Retain only if a system is naturally closed-shell | — |
| More defect frames beyond the 129-frame `dataset_beta` | §B fallback if capacity gives little | as affordable |

---

## Document edits

| Section | Change |
|---|---|
| plan §2.1 / §5.2 | `multiplicity` mandatory for spin-polarised frames; labelling always relative to the closed-shell `M_s = 0` surface, never a per-system ground state |
| plan §3.2 | record that `Δℓ ≈ 11.7` was reached with `λ_u = 0` — the localisation preference is data-driven here; if §D adopted, logits derived from `u`, bound restated as `(N/n_d)·Δu·e^{−βΔu}`, `β` recorded as a gauge constant |
| plan §3.3 | if §D adopted: remove `MLP_ℓ` and the shared-logit option; add the conditional `v` head |
| plan §4 | `λ_u = 0` with the reason; add `--defect_gauge_weight` as an optional diagnostic term, default off |
| plan §5.3 | pristine `n ≠ 0` anchor frames **rejected** — record the reason (cost; rattled anchors assert a false zero target against the deformation-potential physics §3.2 requires) so they are not reproposed |
| plan §8.2 | restate in terms of `Δu` if §D adopted — **blocked:** the working copy is missing §6, §8.2–§8.9, §10, §11; restore before editing |
| plan §9.6 | **keep** the `MLP_u` zero-init; replace the stated rationale with the conditioning one; explicitly rule out structured init |
| impl §3, §4, §6 | anchor removal and dataset version bump; `n_ref` retained and reused by §D-opt; `DefectRMSE` columns (`mean(u^c)`, `std(u^c)`, participation, escape epoch, `Δu`) |
| impl §12 | record: labelling correction, `L_tot = 0` interim, referencing state, float32 training / float64 tests, and every frozen constant |

