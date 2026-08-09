# MACEDefect — correction-branch fix: implementation plan

Work plan for branch `defect`. Section references `plan §N.M` point at
`charge_aware_defect_mlip_plan.md`, `impl §N` at
`charge_aware_defect_mlip_implementation.md`, `diag §N` at
`defect_correction_branch_diagnosis.md`.

**Problem being fixed.** The short-range correction branch does not learn on the 4H-SiC
divacancy test system: attention stays uniform (participation ratio ~380/382), predicted
ΔE is a near-constant, predicted |ΔF| is ~4 orders below target, `RMSE_dF` is pinned at
the target norm. Root cause: the bilinear `α·u` parameterisation admits a "predict the
mean" attractor (bulk-dominated gradient direction), the zero-init of `MLP_u` kills the
attention gradient at step 0, and nothing in the data pins the bulk response of `u`.
Separately, the carrier labelling of this dataset deviates from plan convention and must
be corrected at the data layer before anything else is built on it.

**The route.** (1) Controls to confirm the mechanism. (2) Data correctness: plan-convention
carrier labels, band-edge referencing, pristine zero-anchor frames. (3) Tie the attention
logits to the energy readout, `α = softmax(−β·u)`. (4) Re-enable the long-range branch with
`a` frozen at the DFPT gauge. (5) Production width, then a charged system.

---

## Invariants — must hold after every stage

Re-run these after each stage; they are CI gates, not metrics (plan §8.1):

- `E_total(R, 0) == E_base(R)` and `E_dilute(R, 0) == E_base(R)` exactly, random `R`,
  random parameters.
- `Σ_i α_i^c == 1` per cell per channel (float64, no eps).
- `Σ_i q_i^host == Σ_i q_i^pol == 0` per cell; `Σ_i q_i == a·q` (not `== q`).
- Autograd forces vs finite differences, per branch and per evaluator; Hessian symmetry;
  translation/rotation/permutation invariance.
- The optimizer coverage guard (every trainable parameter in a group) stays active.

Do **not** change: counter algebra and canonicalisation, band-edge referencing on raw
energies, the residual-on-reference-base decomposition, the structured latent-charge
assembly, the three-piece decomposition and `E_dilute`, absence of a separate force head,
absence of any defect-position input (plan §3.6). Diagnostic-only use of defect positions
(Stage 0 oracle) is permitted.

---

## Stage 0 — controls (run before any code change)

All five are cheap; two could change the conclusions downstream.

**0.1 Oracle-attention ablation.** Hard-set the logits so `α` is uniform over the six
defect-neighbour atoms (identify them by hand from the vacant sites; diagnostic only,
never shipped) and train `u` and the counter embedding alone. Run at 8 channels and at
128 channels / `max_L=1`.
*Gate:* `RMSE_dE` approaches the 48 meV linear-probe floor (diag §3).
*If it fails:* stop — the problem is features/receptive field (`r_max`, `max_L`) or
capacity, and Stages 1–2 will not fix it. Investigate descriptors before proceeding.

**0.2 Delta-forces gradient path.** Verify both `autograd.grad` calls in training use
`create_graph=True`, and add a unit test: tiny model, loss from `delta_forces` **only**,
backward, assert non-zero gradients on `MLP_u` parameters. A missing `create_graph` on the
second call reproduces "`RMSE_dF` never moves" exactly, and the two prior bugs in this
family (diag §4) were both plumbing.

**0.3 EMA / epoch-count control.** Confirm validation metrics are not read off EMA weights
while the raw weights move; run one 200-epoch baseline of the current model so slow escape
is ruled out (or observed).

**0.4 Pristine-frame counters.** Confirm all pristine frames currently carry `n = 0`
(expected). This is the missing anchor subset added in Stage 1.

**0.5 Gradient-ratio N-test.** Measure the logit-gradient : u-gradient ratio on a
286-atom and a 398-atom cell. Expected: N-independent (it tracks the spread of `u`).
Record the numbers.

---

## Stage 1 — data correctness: labelling, referencing, anchors

### 1.1 Fix the carrier labelling (plan convention)

The dataset currently labels the divacancy triplet **ground state** as `n = 0` and the
excited state as `(0,1,0,1)`. Under plan §2.2 the reference `n = 0` is the closed-shell
`M_s = 0` surface only; the triplet ground state is not it. Correct labels:

| state | counter vector | q | M_s | canonical? |
|---|---|---|---|---|
| divacancy ground (triplet) | `(1,0,0,1)` | 0 | 2 | yes |
| divacancy excited (triplet, β-channel promotion) | `(1,1,0,2)` | 0 | 2 | yes |
| pristine cell | `(0,0,0,0)` | 0 | 0 | yes |

Tasks:

1. Relabel every frame in the dataset files to the true counter vectors above. Store the
   canonical form. Record the frontier occupations that justify each assignment
   (plan §2.1: a labelling not reproducible from stored metadata is a data-integrity
   failure).
2. Make `multiplicity` **required** for every spin-polarised frame (currently optional),
   and keep the assertion `M_s == multiplicity − 1`. This assertion would have caught the
   original mislabelling; its being optional is why it did not.
3. **Generalise the pair join** (`mace/data/defects.py`). The reference member of a
   `pair_id` group is the unique member with minimal `Σ_c n_c` (error if not unique),
   not necessarily `n = 0`:
   - `delta_energy = E_target(R, n) − E_target(R, n_ref)`, `delta_forces = F(n) − F(n_ref)`,
     computed between actual labels;
   - `base_*` targets are emitted **only** when the reference member is genuinely `n = 0`
     (here: pristine groups only);
   - the geometry-consistency assertion within a group is unchanged.
4. **Model forward with a reference counter.** `MACEDefect.forward` accepts an optional
   per-graph `n_ref`; it evaluates `ΔE_SR` and `E_LR` at both `n` and `n_ref` on the one
   shared trunk pass and returns `delta_energy = [ΔE_SR + E_LR](n) − [ΔE_SR + E_LR](n_ref)`
   with `delta_forces` by autograd of that difference. `n_ref = 0` reproduces current
   behaviour exactly (the correction vanishes at 0), so existing tests must pass unchanged.
5. **`L_tot` weight → 0 for this dataset.** With true labels there are no base labels at
   defect geometries, so `stopgrad(E_base)` there is untrained and `L_tot` would push base
   error into the correction. The base branch trains on pristine frames only for now. The
   delta observables — the point of the probe — do not depend on `E_base` at all.
6. *(Optional, external DFT, restores `L_tot` and the base branch later)*: closed-shell
   constrained singlepoints (`M_s = 0`, e.g. VASP `ISPIN=2, NUPDOWN=0`) at ~100–200 of the
   paired geometries, labelled `n = (0,0,0,0)`, smearing and frontier occupations recorded
   (plan §2.2 reference-surface hazard). When present, the ground/closed-shell pairs also
   exercise the spin-channel structure, which this dataset currently does not test.

### 1.2 Data-generation rules going forward (applies to all future campaigns)

Encode these as loader assertions where possible, not documentation:

- Counters are assigned as the **minimal vector consistent with the converged SCF
  occupations, relative to the closed-shell `M_s = 0` surface** — never relative to a
  per-system ground state, whatever its spin.
- `multiplicity` required whenever `ISPIN=2`; assert `M_s == multiplicity − 1`.
- Enumerate carrier configurations directly; derive `(q, M_s)`; canonicalise at write
  time; store frontier occupations, local moments, smearing scheme and width per frame.
- Reference-state (`n = 0`) calculations are spin-constrained to `M_s = 0`; do not let the
  SCF relax the moment.
- A frame with counters but no resolvable band edges is a hard error (already implemented;
  keep it).

### 1.3 Band-edge referencing (replace the fitted gauge)

The current delta gauge `E_gap = ⟨ΔE_vert⟩ = 0.943 eV` is replaced by proper band-edge
referencing (plan §2.3). For this dataset every state has `n_e_total = n_h_total`, so
**only `E_g^cell = E_CBM^cell − E_VBM^cell` ever enters**: the ground state is referenced
by `1·E_g^cell`, the excited state by `2·E_g^cell`, and the delta target becomes
`E_raw(ex) − E_raw(gs) − E_g^cell`.

- **External input required:** `E_g^cell` per distinct pristine supercell, from
  total-energy differences in the §2.5 convention (`E(N±1) − E(N)`, same cell, same
  background convention, **no image correction**, band-filling handled). Two charged
  pristine calculations per distinct cell. Until these land, a documented interim constant
  may be used, but see the consistency assertion in 1.4.
- Enter the edges in the `--band_edges_file` table per `(host, cell)`; the registry ships
  with the model (impl §3.6).

### 1.4 Pristine zero-anchor frames

Purpose: pin `u_bulk^c = 0` per channel from data, removing the bulk direction as a way to
reduce loss. Implementation: **synthetic siblings of existing pristine frames**, trained
through `L_Δ`:

1. For each distinct ideal pristine supercell in the dataset, add four synthetic frames —
   `(1,0,0,0)`, `(0,1,0,0)`, `(0,0,1,0)`, `(0,0,0,1)` — sharing the pristine frame's
   `pair_id`, positions and cell. Their referenced delta target is **exactly 0**: this is
   definitional under band-edge referencing (the cell edges are *defined* as those
   total-energy differences), not measured, so no SCF is run. Mark them
   `synthetic_anchor=True` in the file.
2. `delta_energy` weight: configurable CLI weight (`--anchor_weight`), default such that
   anchors contribute ~10–20% of `L_Δ`. `delta_forces` weight **0** on synthetic anchors
   (the zero-force idealisation is not exact).
3. **Consistency assertion in the loader:** anchors and referencing must resolve to the
   same edge constants for the same `(host, cell)`. An anchor whose implied edges differ
   from the referencing table is a hard error — this is the silent-uniform-shift failure
   mode of plan §7.2 relocated to training.
4. *(Second pass, when pristine thermal/rattled frames with real `n = 0` labels exist)*:
   add rattled anchors at reduced weight; their zero target is approximate (deformation
   terms, tens of meV), so grade `--anchor_weight_rattled` below the ideal-lattice value.
   Purpose: pin `u_bulk` across the thermal descriptor range, not only at the ideal
   lattice.

### 1.5 Initialisation and regularisation changes

- Remove the hard zero-init of `MLP_u`'s last layer (plan §9.6 is amended; the `n = 0`
  identity is structural via the counter prefactor and owes nothing to the init). The
  init scale is set in Stage 2.3.
- Set `λ_u = 0` (`--defect_u_l2 0`) and record it. Note in the code comment: this is safe
  only together with the Stage 2 tie; if the tie is ever removed, a regulariser must
  return **on the residual head `v`** (see 2.6), not on `u`.
- `p_l2` and `qhost_l2` unchanged.

**Stage 1 gate:** loader tests pass (join with non-zero reference member; anchors present
with correct targets; edge-consistency assertion fires on a constructed mismatch);
participation ratio moves measurably off N in a training run; `RMSE_dF` shows a trend.
A partial improvement is the expected result — proceed to Stage 2 regardless, unless
Stage 0.1 failed.

---

## Stage 2 — tie the attention logits to the energy readout

### 2.1 The form

Delete `MLP_ℓ` (and the `--share_logits_across_spin` argument). In
`mace/modules/defect_blocks.py`:

```
u_i^c  = MLP_u^c([h_i, z(n)])            # carrier site energy, eV
α_i^c  = segment_softmax(−β · u_i^c)     # per cell, per channel
ΔE_SR  = Σ_c n_c Σ_i α_i^c u_i^c         # plain expectation — no entropy term
```

`β = 10 eV⁻¹`, a **fixed constant recorded with the model** (like `σ` and `k_c`), not a
learnable or tuned hyperparameter. The energy is the plain expectation; do not add the
`−β⁻¹ log Σ e^{−βu}` free-energy form (it drifts as `log N` on pristine cells and breaks
plan §3.2's exactness).

Everything downstream of `α` is unchanged: `q_i^carrier = a Σ_c s_c n_c α_i^c`, the
three-piece decomposition, `E_dilute`.

### 2.2 Diagnostics

Replace `logit_gap` in `DefectRMSE` and the per-epoch log with:

- `Δu` per channel (eV): bulk-minus-defect site-energy gap, `Δℓ = β·Δu`;
- participation ratio `1/Σα²` per channel.

With the anchors pinning `u_bulk = 0`, `Δu` is the carrier binding energy relative to the
band edge — sanity-check it against the known level depth. The size-error bound becomes
`(N/n_d)·Δu·e^{−βΔu}`; update the bound computation accordingly.

### 2.3 Initialisation

Initialise `MLP_u`'s last layer so the `u` field has spread ≈ `1/β` = 0.1 eV, shaped by
descriptor novelty: `u_init,i ≈ −ε·ŝ(h_i)` with `ε ≈ 0.1 eV` and
`ŝ(h_i) = ‖h_i − mean_{j∈species(i)} h_j‖` normalised to unit scale per cell. This is an
**initialisation only** — no architecture term, no extra head — so `α = softmax(−βu)`
holds exactly and the seed is trained away where wrong. It breaks the uniform degeneracy
toward physically unusual environments instead of a random atom (the lock-in risk).

If lock-in onto wrong atoms is still observed: anneal `β` from ~2 to 10 over the first few
epochs. `β` changes the energy function, not just the optimiser — the anneal must
terminate at the recorded production `β` before any reported number.

### 2.4 Tests (add to the invariant suite)

- `α` exactly uniform (per symmetry class) on a pristine cell, to numerical precision.
- Re-run finite-difference forces and Hessian symmetry — forces now contain a
  `−β·n_c·Cov_α(u, ∇u)` term.
- **Carrier-transfer scan:** construct a configuration with two competing localisation
  sites, sweep a coordinate that transfers the carrier between them, and check `ΔE(Q)` and
  `ΔF(Q)` for kinks. The tie renders the transfer as a softmin switch of width ~`1/β`;
  this scan is the discriminating test for whether the residual head (2.6) is needed.

### 2.5 Stage 2 gate

Beat the linear probe on held-out supercells: `R² > 0.85`, residual `< 48 meV`, with
`RMSE_dF` trending toward the target scale rather than pinned at it. If the gate fails,
check anchor coverage first (the level mode of `u` reopens wherever bulk descriptors are
not spanned by anchors) before concluding against the tie.

### 2.6 Residual head — only if 2.5 or the carrier-transfer scan demands it

```
ΔE_SR = Σ_c n_c Σ_i α_i^c (u_i^c + v_i^c),    α = softmax(−β u)   # v does not enter α
```

with a weak L2 **on `v`** (the head that does not control the weights). Do not add `v`
pre-emptively.

---

## Stage 3 — re-enable the long-range branch

The long-range branch is **not** inert at `q = 0`: `q_i^carrier = a(α_i^{h} − α_i^{e})` is
a compensated but pointwise non-zero charge whose self-term is the electron–hole
interaction, and `q_i^pol` carries a factor `Σ_c n_c`. Part of the 0.94 eV observable
belongs there.

1. **External input required:** DFPT `ε_∞` for 4H-SiC at the dataset's functional
   (PBEsol), once. 4H-SiC is uniaxial; use the isotropic average and record both
   components.
2. Re-enable `use_long_range=True` with `a` **frozen** at `1/√ε_∞` (no gradient to
   `MLP_a`). `a` is not identifiable from a dipole term; it stays frozen until a charged
   system exists (Stage 4).
3. Retrain from the Stage 2 checkpoint; measure how much energy `ΔE_SR` gives up to
   `E[S_carrier]`. A material transfer (order 0.1–1 eV before geometry dependence) is the
   expected result. Re-check the Stage 2 gate metrics after the transfer.

---

## Stage 4 — production width, then a charged system

1. 128 channels, `max_L = 1`; re-run the invariant suite and the Stage 2 gate.
2. First charged defect in the same host. Unfreeze `a`; the monopole signals of plan §3.4
   (far-field forces, shape-varied cells, optionally the verified hydrostatic pressure
   term) become live. Only now is the plan §9.7 step-7 gate — `1/a²` vs DFPT `ε_∞` agree
   or the disagreement is understood — meaningful.
3. Run the multi-cell size-series discriminator (plan §9.7 step 5) on the undertrained
   charged model before scaling the data campaign.

---

## External inputs (DFT — cannot be produced by this codebase)

| Input | Needed by | Size |
|---|---|---|
| `E_g^cell` per distinct pristine supercell (§2.5 convention, charged pristine totals) | Stage 1.3 referencing + 1.4 anchor consistency | 2 calcs × ~4 cells |
| DFPT `ε_∞` (PBEsol, 4H-SiC), once | Stage 3 frozen `a` | 1 calc |
| Closed-shell `M_s = 0` constrained singlepoints at paired geometries (optional) | restores `L_tot` and defect-geometry base labels | ~100–200 singlepoints |
| Rattled pristine `n = 0` frames (optional) | Stage 1.4 rattled anchors | small |

Interim behaviour where an input is missing is specified inline above; every interim must
be recorded, and the loader consistency assertion (1.4.3) must never be bypassed.

---

## Document edits to make alongside the code

| Document | Change |
|---|---|
| plan §2.1/§5.2 | `multiplicity` mandatory for spin-polarised frames; labelling always relative to the closed-shell surface |
| plan §3.2 | logits derived from `u`; drift bound restated as `(N/n_d)·Δu·e^{−βΔu}`; `β` recorded like `σ`, `k_c` |
| plan §3.3 | remove `MLP_ℓ` and the shared-logit option; add the conditional `v` head |
| plan §4 | `λ_u = 0`; L2 belongs on `v` if present; add the anchor subset and its weight |
| plan §5.3 | pristine `n ≠ 0` anchors promoted from verification identity to required training subset (synthetic, definitional targets) |
| plan §8.2 | rewrite the size-error diagnostic in terms of `Δu` — **blocked**: the supplied plan is missing §6, §8.2–§8.9, §10, §11; restore them before editing |
| plan §9.6 | remove the `MLP_u` zero-init; add the spread-`1/β` novelty-shaped init and the `β` gauge |
| impl §3, §4, §6 | generalised join / `n_ref`; head structure; loss weights and `DefectRMSE` columns; anchor keys |
| impl §12 | record the labelling correction, the gauge replacement, `L_tot = 0` interim, and all frozen/interim constants |
