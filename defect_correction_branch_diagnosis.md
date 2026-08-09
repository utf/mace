# MACEDefect: the carrier correction branch does not learn

Briefing for architecture co-design. Self-contained — no repo access needed.

**What I need:** an evaluation of the diagnosis in §5–6 and of the proposed changes in §7,
which deviate from the original design plan in ways I want scrutinised before I commit.

---

> ## ⚠️ CORRECTION — 2026-08-08, after this document was written
>
> Two of the central claims below have since been **measured and refuted**. Read this box
> before acting on §5–§7. The title of this document is itself wrong: the branch does
> learn.
>
> **1. It is a plateau, not an attractor. The title claim is false.**
> §5–§6 rest on short runs (≤6 epochs) whose validation metrics were additionally read off
> **EMA-averaged weights**, not the weights being optimised (`train.py:299` wraps
> validation in `ema.average_parameters()`, and the run script hard-coded `--ema`).
> A 200-epoch run with EMA off shows a flat plateau through epoch 21 — `RMSE_dE` at the
> target σ of 109.8 meV, `RMSE_dF` pinned at 46.11 — and then a **sharp escape at epoch
> ~22**, reaching `RMSE_dE = 43.67 meV` by epoch 42 and 26.6 meV/Å on `dF` by epoch 68,
> both still falling. 43.67 meV is already **below the 48 meV linear-probe floor** that
> §5 sets as the target. So the correction branch does fit the binding; it just takes
> ~2000 gradient updates to leave the plateau. §6.3's "the optimiser has a third option:
> don't fit the binding at all" is not what happens.
>
> **2. The attention gradient is not suppressed by cell size.**
> §6.1 argues the attention gradient is ~260× weaker because `α ~ 1/N`, and §7.4 proposes
> replacing the softmax on that basis. Measured directly, with one model instance across
> four cell sizes, the ratio `|∇logit| / |∇u|` is **0.0330, 0.0329, 0.0330, 0.0329** for
> N = 286, 318, 382, 398 — constant to 0.1% across a 1.39× change in N, at both 16 and 128
> channels. The `α_j` prefactor is cancelled by the sum over sites, as the algebra says.
> The ratio is small (~0.03) but it is a **fixed** factor set by the spread of `u`, not a
> growing one. **§7.4 is not motivated by size-scaling.**
>
> **What survives.** The zero-init mechanism of §6.1(a) is real and now has a unit test:
> with `u ≡ 0`, `∂ΔE_SR/∂ℓ` is *exactly* `0.0` at every cell size, so the attention starts
> with no gradient at all. But `u` itself keeps receiving gradient, climbs away from zero,
> and switches the attention on — hence a **delay**, not a trap.
>
> **3. §7.1 (remove the zero-init) is not supported — but neither is its opposite.**
> Three seeds per arm, 200 epochs, identical otherwise. Escape epoch (first epoch with
> `RMSE_dF < 45`; the metric is flat at 49.3 on the plateau):
>
> | zero-init | seed 1 | seed 2 | seed 3 | best final dF |
> |---|---|---|---|---|
> | off (§7.1's proposal) | never | never | 37 | **23.92** |
> | on (status quo) | 10 | 78 | 102 | 25.44 |
>
> Zero-init escaped 3/3 versus 1/3 — suggestive but Fisher-exact p ≈ 0.4, not a result.
> Escape times vary 10× *within* the winning arm, so **seed dominates the init**, and once
> a run escapes the final quality is the same either way (the single best run of the six
> is a no-zero-init one). So the real finding is that **the plateau fails to resolve
> within 200 epochs in a third of runs regardless of initialisation.** That is an
> optimisation pathology worth fixing on its own terms, and it is direct evidence *for*
> plan §2.3's structured init — seed `u` with descriptor novelty at controlled scale, so
> it is neither zero (no attention gradient) nor noise (crushed back toward zero by the
> cheapest early loss reduction). Under the §2.2 tie `α = softmax(−βu)` this matters more,
> not less, since the attention would inherit a random `u`'s noise directly.
>
> A per-channel autopsy of one escaped and one stuck run is in `HANDOFF.md` §11. The
> escaped run localises `e_maj` onto **2.5 of 286 atoms with 99.96% of the attention on
> the six defect first-shell atoms** — the architecture does what it was designed to do.
> That autopsy also exposes a gap not in the plan: the anchors pin `e_maj` and `h_min`,
> `h_maj` is dead in this dataset, and **`e_min` — the channel that carries the excitation
> — has no anchor at all**, leaving its bulk level free (measured `mean(u)` = 0.1699
> against 0.0014 for anchored `h_min`).
>
> Also corrected since: the long-range branch is **not** inert at q = 0 (`q^carrier` is a
> compensated but pointwise non-zero charge; `q^pol` carries a factor `Σ_c n_c`), and the
> carrier labelling used for every number below was wrong — the divacancy triplet ground
> state was labelled `n = 0` when the plan's reference is the closed-shell surface, so it
> is `(1,0,0,1)`. Relabelling leaves the delta *targets* numerically unchanged, so the
> measurements above still stand, but it changes what supervises the base branch.
>
> Current state of the work: `defect-example/HANDOFF.md`.

---

## 1. What the model is

`MACEDefect` is a charge-aware defect potential built on MACE. A geometry-only message-passing
trunk produces a **base branch** `E_base(R)`, and a separate **correction branch** adds a
carrier-conditioned term. The conditioning variable is a carrier counter vector

```
n = (n_e^maj, n_e^min, n_h^maj, n_h^min)     # electrons/holes, majority/minority spin
```

The counters enter **only** the correction heads, never the trunk, so that
`E_total(R, n = 0) ≡ E_base(R)` holds exactly for any parameters.

The short-range correction is an **attention-pooled** per-atom readout:

```
ℓ_i^c = MLP_ℓ^c([h_i, z(n)])            # per-atom logit,  channel c
α_i^c = softmax_i(ℓ_i^c)                # per cell, per channel:  Σ_i α_i^c = 1
u_i^c = MLP_u^c([h_i, z(n)])            # per-atom energy readout
ΔE_SR = Σ_c n_c Σ_i α_i^c u_i^c
```

`h_i` are invariant node features from the trunk; `z(n)` is an embedding of the counter
vector. `α^c` is meant to be the carrier's occupation distribution over atoms, so
`Σ_i α_i^c u_i^c` is its expectation energy.

The design intent (plan §3.2): because α is **normalised**, the pooled quantity is
*intensive* — the correction scales with carrier count, not atom count — which is what makes
the model size-consistent. Pristine cells cancel exactly; defective cells drift as
`≈ N·r·(u_b − u_d)` with `r = e^{ℓ_b − ℓ_d}`, so a large **logit gap** between defect and
bulk atoms is what suppresses the size error.

There is also a long-range branch (latent-Ewald with structured latent charges). It is
**inert for this test system** — see §3 — and is not the subject of this briefing.

### 1.1 Two initialisation choices from the plan

- **Plan §9.6:** "Initialise `MLP_u`, `MLP_p` and `Q_host` near zero so the model starts
  close to the base potential." Implemented as a hard zero of the last layer of each
  `MLP_u^c`, so `u ≡ 0` at step 0. Logits are *not* zeroed ("a zero logit field is a
  uniform, maximally delocalised carrier").
- **Plan §3.2 / §4:** a weak L2 on `u` is declared **load-bearing**, not housekeeping:

  > fitting a 1 eV binding with `α_d = 1/216` requires `u_d ≈ −216 eV`, which the penalty
  > makes expensive, so the optimiser buys localisation instead. That regulariser is
  > therefore load-bearing for extensivity rather than incidental housekeeping — tune it
  > deliberately, record the value, and do not remove it as cleanup.

Implemented as `1e-4 × torch.mean(u²)`.

---

## 2. The defect system

4H-SiC **divacancy** (V_Si–V_C), neutral, spin triplet, from a published DFT dataset
(PBEsol, VASP, 520 eV cutoff) accompanying a paper on optical line shapes of colour centres.

The physics being learned is the **first excited state**: an intra-defect promotion within
the minority (β) spin channel — one electron promoted between two localised defect levels.

- counter vector: ground `(0,0,0,0)`, excited `(0,1,0,1)` → canonicalised to `(1,0,1,0)`
- **net charge is zero on every frame** (one electron + one hole)
- vertical excitation ⟨ΔE⟩ = 0.943 eV, σ = 0.124 eV; published ZPL 0.97 eV

The published data encodes the electronic state in the **chemical species** of the six defect
neighbours (Si_gs→P, C_gs→N, Si_ex→S, C_ex→O), because the NEP model it was built for has no
other channel for it. For MACEDefect these are mapped back to Si/C. **This makes the task
strictly harder than the original:** the trunk sees *identical input* for a ground/excited
pair — same species, same positions, same cell — so the entire 0.94 eV must come through the
correction branch.

Consequence of `q = 0`: the long-range branch is unconstrained here (no monopole; the
screening amplitude `a` never moves off its initialisation), so all runs below use a
short-range-only model.

---

## 3. Training data

800 frames, 340 ground/excited **pairs** at identical geometry (`pair_id` groups), plus
unpaired ground-state frames, unpaired excited frames, and pristine 4H-SiC supercells.
Cells are 286–398 atoms. Train 720 / valid 80, split by whole groups.

Targets, built at load time:

- `base_energy`, `base_forces` — the `n = 0` labels (base branch)
- `delta_energy = E_ex(R) − E_gs(R) − E_gap`, `delta_forces = F_ex(R) − F_gs(R)`

`E_gap` is a fitted gauge (only `E_CBM − E_VBM` enters, since every non-zero counter here has
one electron and one hole). Set to ⟨ΔE_vert⟩, so **delta targets are centred at zero with
σ = 0.124 eV**, range −0.37 … +0.43 eV. Delta forces reach ~0.6 eV/Å.

**The target is strongly and locally determined.** A *linear* regression of ΔE_vert on the 15
pairwise distances among the six defect neighbours gives **R² = 0.85**, residual 48 meV, over
all 504 available pairs. Held-out supercells: 318-atom 0.86, 382-atom 0.86, 398-atom 0.71.
(286-atom holdout is −4.05 — distinct cell shape, smallest subset; a limit of the linear
probe, not of the signal.) **This is not a data problem.**

---

## 4. Two plumbing bugs already fixed

Mentioned only so they are not re-diagnosed:

1. **The correction branch was never passed to the optimizer.** MACE builds optimizer groups
   from an explicit whitelist of submodules; the new defect modules matched none of them, so
   38 of 64 parameter tensors (69% of parameters) sat at initialisation for every run. Fixed,
   with a coverage guard that raises if any trainable parameter is in no group.
2. **Per-epoch metrics were silently dropped** for the defect error table (no branch and no
   `else` in the logging chain), which is why (1) went unnoticed.

Both are fixed. **Everything below is from runs after those fixes**, with the correction
branch verifiably receiving gradients.

---

## 5. The problem, measured

Base branch trains normally. Correction branch does not.

```
Epoch 0: RMSE_E= 198.33 meV/atom  RMSE_F= 200.26 meV/Å  RMSE_dE= 110.98 meV  RMSE_dF= 46.11 meV/Å
Epoch 1: RMSE_E=  29.28           RMSE_F= 125.54        RMSE_dE= 116.39      RMSE_dF= 46.11
Epoch 2: RMSE_E=  21.78           RMSE_F=  92.89        RMSE_dE= 110.81      RMSE_dF= 46.11
Epoch 3: RMSE_E=  20.53           RMSE_F=  83.16        RMSE_dE= 115.43      RMSE_dF= 46.11
Epoch 4: RMSE_E=  22.11           RMSE_F=  77.89        RMSE_dE= 111.93      RMSE_dF= 46.11
Epoch 5: RMSE_E=  21.33           RMSE_F=  74.69        RMSE_dE= 110.62      RMSE_dF= 46.11
```

`RMSE_dE` fluctuates around the target σ without trending. `RMSE_dF` is **constant to two
decimals** for every epoch.

Probing the trained model:

| quantity | measured | expected if working |
|---|---|---|
| attention participation ratio `1/Σα²` | **380.7 of 382 atoms (99.7%)** | ~6 (defect neighbours) |
| max α | 0.00279 (uniform = 0.00262) | ≫ 1/N |
| predicted ΔE, four frames | 0.0048, 0.0046, 0.0055, 0.0048 | should track targets |
| target ΔE, same frames | −0.183, +0.005, +0.258, −0.139 | — |
| predicted \|ΔF\| max | 9e-5 eV/Å | ~0.6 eV/Å |

**α is 6% away from perfectly uniform.** The attention has not localised at all.

---

## 6. Diagnosis

`ΔE_SR = Σ_c n_c Σ_i α_i^c u_i^c` with `Σ_i α_i = 1` is a **convex combination** — an average
over atoms. With α uniform it is the cell mean of `u` over ~300 atoms, dominated by bulk atoms
whose descriptors are near-identical in every frame. The output is therefore a near-constant.
Measured predictions span 0.0046–0.0055 against targets spanning −0.18…+0.26.

### 6.1 Three compounding mechanisms

**(a) At initialisation the attention gradient is exactly zero.**

```
∂ΔE_SR/∂ℓ_j^c = n_c · α_j^c · (u_j^c − ⟨u^c⟩_α)
```

`u ≡ 0` (plan §9.6) makes this vanish identically. Measured `|grad| logit_readouts =
0.000000e+00`. The zeroed last layer also blocks the backward path *through* `u`, so the
counter embedding and the feature projection measure exactly zero too. **At step 0 only the
final layer of the u-MLP trains.**

**(b) Once `u ≠ 0`, the attention gradient is ~260× weaker than the readout gradient**
(measured ratio 3.8e-3, since `α_j ~ 1/N`). `u` converges to the best *uniform-attention*
solution — the cell mean — long before α moves. That solution has `u` nearly uniform, which
sends `(u_j − ⟨u⟩) → 0` and kills the attention gradient again. Self-reinforcing.

**(c) The L2 that was supposed to buy localisation is seven orders of magnitude too weak.**
`1e-4 × torch.mean(u²)` = **6.2e-9** measured, against a delta data term of **1.8e-1**. The
`mean` also makes it intensive, so 6 large `u` and 300 small `u` are penalised almost
identically — it cannot express the preference it was written for.

### 6.2 Why dE and dF differ — the sharper symptom is dF

`ΔE` can absorb a **constant**: with uniform α it fits the mean of the target distribution, so
`RMSE_dE` sits at σ and drifts as that constant moves. A force field has **no constant
component to fit**. `ΔF_j = −∂ΔE_SR/∂r_j ≈ −(1/N) Σ ∂u/∂r_j` is suppressed by ~N with bulk
terms largely cancelling, giving a prediction four orders low. So `RMSE_dF` is pinned exactly
at the target norm. **A constant `RMSE_dF` is the signature of no localised structure at all.**

### 6.3 Where the plan's reasoning breaks

Plan §3.2's argument is sound as far as it goes: fitting the binding with delocalised α would
need enormous `u`, and the L2 makes that expensive, so localisation is the cheaper of those
two options.

The gap is that the optimiser has a **third option the argument does not consider: don't fit
the binding at all.** Predicting the mean costs only `≈ σ² × w_ΔE` in the data term while
keeping `u` small, so it is cheap under *both* the data loss and the L2. It is also the basin
that gradient descent falls into from the prescribed initialisation, because of (a) and (b).

So the dichotomy "large cancelling readouts vs localisation" is real but incomplete — and the
regulariser meant to arbitrate it is, as implemented, numerically absent anyway.

**A second, independent concern about §3.2:** the extensivity argument requires a logit gap of
`Δℓ ≈ 11` at production size. The measured gap is **0.055**. Even if the model did start
fitting, nothing in the objective drives the gap to a specific large value — the plan's own
diagnostic (read the gap off a trained model to bound the size error) would currently report a
size error bound of order `N·(u_b − u_d)`, i.e. no suppression whatsoever.

---

## 7. Proposed changes, and how they differ from the plan

Ordered by confidence. (1) is verified; (2)–(4) are proposals I want evaluated.

### 7.1 Remove the zero-initialisation of `MLP_u` — *contradicts plan §9.6*

The plan zeroes `MLP_u` so the model "starts close to the base potential". **This is
unnecessary:** the `n = 0` identity is guaranteed *structurally* by the counter prefactor,

```python
delta_sr = (counts * pooled).sum(dim=-1)     # ≡ 0 whenever n = 0, for any u
```

Verified: with `u` randomly initialised at O(1), at `n = 0` all of `max|E_total − E_base|`,
`max|Δ_energy|` and `max|Δ_forces|` are exactly `0.000e+00`.

So the zero-init buys nothing the prefactor does not already give, and it costs the entire
attention gradient at step 0 plus the gradient to the counter embedding and feature
projection. It does mean *charged* configurations no longer start at zero correction — which
is the property §9.6 actually wanted, and which I claim is not worth a dead gradient.

**Expected insufficient alone**, since (b) and (c) persist.

### 7.2 Make the localisation pressure real — *reinterprets plan §3.2/§4*

The plan's mechanism is right in spirit and inert in implementation. Options:

- make the `u` penalty **extensive** (sum, not mean) and calibrate the weight against the
  data term, so the ratio the plan's argument depends on actually exists; or
- target the symptom directly with an **entropy penalty on α**, or a **learnable inverse
  temperature** on the logits initialised sharp.

The second is a bigger departure: the plan deliberately makes localisation an *emergent*
consequence of a penalty on `u` rather than something imposed on α, because it wants the
logit gap to be a *diagnostic* read off a freely-trained model (§8.2). An entropy penalty
would make the gap partly a hyperparameter, weakening that diagnostic. **This is the main
trade-off I want your view on.**

### 7.3 Break the α/u scale imbalance — *not in the plan*

A separate, higher learning rate for the logit networks, or a reparameterisation so the
correction is not a pure product of two mutually degenerate factors. The 260:1 gradient ratio
is a direct consequence of `α ~ 1/N`, so it worsens with cell size — exactly the regime the
model is for.

### 7.4 Reconsider softmax-over-all-atoms — *questions plan §3.2*

The correction lives on ~6 of ~300 atoms: a ~50× upweight, `ln 50 ≈ 3.9` in logits, and §3.2
wants `Δℓ ≈ 11` at production size. Softmax from a near-uniform start has to *search* for
that. Alternatives that begin in the right regime: logits biased by a coordination-deficit
feature; top-k or sparsemax pooling; or an explicit locality prior.

This is the largest departure — normalised softmax pooling is the mechanism §3.2's whole
size-consistency argument is built on, so replacing it means re-deriving that argument. I do
not propose it lightly and would rather fix the optimisation path if that is enough.

---

## 8. Questions I want answered

1. Is the "predict the mean" basin escapable by any fix to initialisation and learning rates
   alone, or is the product parameterisation `Σ α·u` with normalised α intrinsically prone to
   it at large N?
2. If localisation is imposed (entropy penalty / temperature), what remains of §3.2's use of
   the logit gap as an *a priori* size-error diagnostic?
3. Is there a formulation that keeps intensivity and size-consistency but is **not** a product
   of two separately-degenerate factors? (e.g. pooling with a fixed, physically-motivated
   locality kernel and a learned residual.)
4. Should the correction be forced to be **local by construction** — e.g. non-zero only within
   some cutoff of a detected defect centre — with the attention refining it rather than
   discovering it?
5. Does the `q = 0` character of this test system (a neutral exciton, no monopole) change any
   of the above, or is the failure generic to the short-range branch?

---

## 9. Reproduction notes

- All numbers above from a short-range-only model (`use_long_range=False`), 8 channels,
  `max_L=0`, `r_max=4.0`, batch 8, lr 5e-3, Adam + EMA, 720 train / 80 valid frames.
- Model width is deliberately small. The failure is argued **not** to be a capacity limit,
  but be precise about what was measured where: the α-uniformity and predicted-ΔE numbers in
  §5 are from the trained 8-channel model; the gradient measurements in §6.1(a)–(b) and the
  `n = 0` identity check in §7.1 were made at 16 channels. Mechanism (a) is exactly zero by
  algebra and so is width-independent; (b) scales as `1/N` in the *atom* count, not the
  channel count. **Neither has been checked at production width (128 channels, `max_L=1`)**
  — worth doing before acting on this, since it is cheap and would close the gap.
- Loss weights: `energy 1.0, forces 100.0, delta_energy 10.0, delta_forces 100.0,
  total_energy 0.1, u_l2 1e-4, p_l2 1e-4, qhost_l2 1e-4`.
- Reference target for any fix: beat **R² = 0.85 / 48 meV residual**, the linear
  defect-neighbour-geometry probe of §3.
