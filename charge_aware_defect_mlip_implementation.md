# MACEDefect — implementation blueprint

Implementation plan for `charge_aware_defect_mlip_plan.md`, as an extension of the existing
MACELES architecture in this repository. Section references of the form §N.M point at the
scientific plan; this document only records *how* it is built here.

Reference-data generation and pre-processing (plan §5, §6) are out of scope.

---

## 1. Scope

**In** — plan build order §9.7 steps 1–7:

1. counters, canonicalisation, band-edge-referenced targets, data loading, validity assertions
2. base branch (inherited from `ScaleShiftMACE`, unchanged)
3. carrier attention pooling and `ΔE_SR`, plus the §8.1 unit tests
4. structured latent charges (`q_host`, `q_pol`, `q_carrier`, amplitude `a`) on top of LES
6. `L_Δ`, `L_tot` with detached base, optional `L_P`, regularisers
7. isolated evaluator, three-piece decomposition, `E_dilute`

**Out**, deferred explicitly:

- §3.5 multi-fidelity `Δ_fid`
- §7.2 formation-energy / transition-level assembly (thin `defect_thermo` utility, later)
- §7.3 bounded counter enumeration with UQ gating (needs an ensemble)
- §8.2–§8.9 validation campaign — only the §8.1 unit tests land now
- `correction_trunk: separate` — the constructor argument exists and raises

---

## 2. Placement

| Location | Contents |
|---|---|
| `mace/data/defects.py` *(new)* | counter algebra, canonicalisation, validity assertions, band-edge table, the `Configuration`-level transform, bounded enumeration helper (used later by §7.3) |
| `mace/modules/defect_blocks.py` *(new)* | `CounterEmbedding`, `CarrierAttentionPooling`, `StructuredLatentCharges` |
| `mace/modules/latent_ewald.py` *(new)* | thin wrapper over `les`: periodic and isolated evaluators, three-piece decomposition |
| `mace/modules/defect_models.py` *(new)* | `MACEDefect(MACELES)` |
| `mace/modules/loss.py` | `DefectLoss` |
| `mace/data/{utils,atomic_data}.py`, `mace/tools/default_keys.py` | new keys and fields |
| `mace/tools/{arg_parser,model_script_utils,scripts_utils,tables_utils,train}.py`, `mace/cli/run_train.py`, `mace/calculators/mace.py` | plumbing |
| `tests/unit/test_defects.py`, `tests/extensions/defect/` | tests; LES-dependent ones marked `les` |

`extensions.py` is already at a `too-many-lines` pylint disable and every `MACELES` import
site is lazy (inside a function), so nothing forces co-location. Naming follows the repo's
`<Physics>MACE` convention.

`MACEDefect` subclasses `ScaleShiftMACE` and **composes** the LES evaluator through
`LatentEwald` rather than subclassing `MACELES`. Reason, found while building it:
`MACELES.__init__` imports `les` unconditionally and raises without it, which would make the
short-range branch — and the `E_total(R, 0) == E_base(R)` identity that gates everything —
untestable without an optional dependency that is not installed by default. `forward` is
overridden wholesale in either case, so inheritance bought only the `les` handle. A
`use_long_range=False` flag builds a short-range-only model; the default is `True` and then
`les` is required, with the same error message MACELES gives.

---

## 3. Data layer

### 3.1 File format

One frame per DFT calculation. Pairing between a charged calculation and the neutral
single-point at the same geometry is carried by a `pair_id` info key.

```
# neutral single-point at geometry R1
REF_energy=-1234.50  carrier_counts="0 0 0 0"  host=GaN_216  pair_id=GaN_216_VGa_R1

# same positions, q = -1 (one electron, majority channel)
REF_energy=-1238.70  carrier_counts="1 0 0 0"  host=GaN_216  pair_id=GaN_216_VGa_R1
```

Rejected alternative: one frame per *geometry* with the charge states stacked as columns
(`carrier_counts="0 0 0 0 1 0 0 0"`, forces `(N, 3K)`), demultiplexed at `Configuration`
level. It is downstream-equivalent and round-trips through ASE cleanly, and it makes pairing
structural. It was not chosen because dataset construction is a per-calculation loop (stacking
needs an assembly pass and blocks until every state of a geometry converges); because
row operations — append for active learning, filter, merge across hosts — are rewrites in the
stacked format; because stock tooling (`ase.io.read`, `mace_eval_configs`, `fine_tuning_select`)
reads a stacked frame *successfully* and yields an uninterpretable `(N, 3K)` force array; and
because the integrity it guarantees is obtained instead by a loader-side assertion (§3.4) that
additionally catches the error stacking cannot express — two states relaxed separately that
are not in fact the same geometry.

The two formats meet at `K = 1`, so a stacked front-end can be added later as a pure
`Configuration`-level reader without touching anything below it.

### 3.2 Keys

Added to `DefaultKeys` **and** to the hardcoded `infos`/`arrays` lists in
`update_keyspec_from_kwargs` (`mace/data/utils.py`) — a key added to only one of the two
silently vanishes from the keyspec with no error.

| Key | Where | Required |
|---|---|---|
| `carrier_counts` | info, 4 non-negative ints `n_e^maj n_e^min n_h^maj n_h^min` | every frame |
| `host` | info, string | whenever more than one host is present |
| `pair_id` | info, string | optional; enables `L_Δ` |
| `multiplicity` | info, int | optional; cross-checked against `M_s + 1` |
| `e_cbm_cell`, `e_vbm_cell` | info, float | on `n ≠ 0` frames, unless supplied by the band-edge table |
| `base_energy`, `base_forces`, `base_stress` | info / arrays (`REF_base_*`) | optional; normally produced by the join |

The `base_*` keys are both the optional user input and the derived property name: the join
fills them from the `n = 0` partner when they are absent, so the loss reads one field either
way.

### 3.3 Band edges

`E_CBM^cell` and `E_VBM^cell` are properties of (host, supercell), not of a configuration, so
the canonical source is a JSON table passed as `--band_edges_file` and keyed on `host`:

```json
{"GaN_216": {"e_cbm_cell": -3.20, "e_vbm_cell": -6.85},
 "ZnO_192": {"e_cbm_cell": -4.05, "e_vbm_cell": -7.31}}
```

Per-frame keys override the table where present. This removes the possibility of two frames of
the same host carrying different edges; where per-frame keys *are* used, frames sharing
`(host, natoms)` whose edges differ beyond a tolerance are flagged, because that is the silent
uniform shift of every transition level warned about in §7.2.

`E_CBM^∞`, `E_VBM^∞`, chemical potentials and DFPT `ε_∞` are **not** training inputs — nothing
in the loss uses them. They live in the per-host reference table consumed only by the
inference-side §7.2 assembly, which makes "never mix cell and ∞ edges" structural rather than
a discipline.

### 3.4 Transform — `mace/data/defects.py`, applied to `Configuration` objects

Applied immediately after `config_from_atoms_list` in `load_from_xyz`, gated on
`carrier_counts` being in the keyspec. It operates on `config.properties`, so the HDF5
preprocessing path (`save_configurations_as_HDF5`, generic over `properties`) carries the
derived targets without further work.

1. **Canonicalise and validate.** `M_s > 0` → unchanged; `M_s < 0` → swap the spin channels;
   `M_s = 0` → lexicographic maximum of `{n, swap(n)}` (§2.1). Assert four non-negative
   integers and, where a multiplicity is recorded, `M_s == multiplicity − 1`.
2. **Join.** Group by `pair_id`; within a group locate the unique `n = 0` member and copy its
   raw energy, forces and stress onto every sibling. Rules:
   - no `pair_id` → the frame trains through `L_tot` (§4c); this is not an error
   - `pair_id` present with zero or several `n = 0` members → error
   - all members of a group must share atomic numbers, positions and cell to tight tolerance,
     else error — this is the check that replaces the stacked format's structural guarantee
   - explicitly supplied `base_*` keys are respected and skip the join
   - a frame with no counters is treated as `n = 0`, unless it carries band edges or a
     non-zero `total_charge`, which is a labelling error rather than an implicit neutral
3. **Reference.** `energy ← E_raw − Σ_σ[n_e^σ·E_CBM^cell − n_h^σ·E_VBM^cell]` (§4 `E_target`),
   per frame, from that frame's own edges; `raw_energy` is retained. Forces and stress are
   untouched — the referencing term is a per-config constant, so `F_target = F_raw` always.
4. **Derive targets.** `base_energy`/`base_forces`/`base_stress` (the `n = 0` labels for the
   base branch) and `delta_energy = E_target − E_raw(0)`, `delta_forces = F(n) − F(0)`, each
   with its weight set to 1.0 where defined and 0.0 otherwise.

Guardrails: `n ≠ 0` with no edges from either source is an error; `n = 0` with no edges is
fine, since the referencing constant is identically zero.

**`L_base` is weighted once per group.** `base_energy` is a model output on *every* frame and
depends on geometry alone, so a paired charged frame trains `L_base`, `L_Δ` and `L_tot` from a
single forward pass and the neutral single-point does not need to be a training frame in its
own right. To stop one geometry contributing the same base-branch target once per charge
state, exactly one member of each group carries `base_*` weight 1: the `n = 0` frame if
present, otherwise the first charged member.

**Why reference at load rather than inside the loss.** `compute_average_E0s` and
`scaling_classes[args.scaling]` run over the train loader *before* the model is built, so a raw
`energy` would put band-gap-scale constants into the E0 fit and the scale. The corollary is
that the calculator returns referenced energies and §7.2's affine map becomes a required
inference step (§8.2).

**Statistics use the `n = 0` subset only** — `reference_state_configurations` for the E0 fit
and `reference_state_data_loader` for the energy scale, both falling back to the full set with
a warning if no reference-state configuration exists. Even after referencing, the base surface is defined
by `n = 0` alone (§2.2, §4a); charged frames in the E0 fit or in `std` would push carrier
binding energy into the base branch's baseline and scale. In practice only `std` is at risk —
`atomic_inter_shift` is hard-coded to zeros on the MACELES factory path, which `MACEDefect`
follows — but the filter applies to both.

### 3.5 `AtomicData`

Explicit fields, because the generic extra-property passthrough at the end of `from_config`
promotes a `(4,)` tensor to `(4, 1)` and concatenates it along dim 0. Per-graph vectors follow
the `dipole` / `external_field` convention of a leading dimension of 1.

- `carrier_counts` `(1, 4)`
- `raw_energy`, `base_energy`, `delta_energy` scalars; `base_forces`, `delta_forces` `(N, 3)`;
  `base_stress` `(1, 3, 3)`
- weights `base_energy_weight`, `base_forces_weight`, `base_stress_weight`,
  `delta_energy_weight`, `delta_forces_weight`

### 3.6 Multi-host training

Nothing in the referencing is global, so one model over many hosts needs no extra machinery.

- **Hosts are not heads.** MACE `head`s exist for multi-dataset and multi-fidelity splits;
  hosts are distinguished by geometry and composition through the shared trunk. `head` stays
  reserved for the PBE/HSE split of §3.5.
- **`a` is per host automatically.** `a = softplus(MLP_a(mean_i h_i))` reads the mean host
  descriptor, so a multi-host model learns a different `ε_∞` per host without a host label
  ever entering the network. `host` is bookkeeping only.
- **The registry ships with the model.** Training collects `(host, natoms) → {E_CBM^cell,
  E_VBM^cell}` into `model.band_edge_registry`, which is saved with the model, so inference
  reproduces exactly the constants training used. The transform writes the resolved edges back
  onto each configuration first, so entries that came from the band-edge table are recorded
  too. An explicit `cell_id` key can override the `natoms` component where two distinct
  supercells share an atom count.

---

## 4. Model — `MACEDefect`

The trunk is geometry-only. Counters enter **only** the correction heads — never
`joint_embedding`, which feeds the trunk and would destroy `E_total(R, 0) = E_base(R)`.

```
z            = CounterEmbedding(n)                     per graph → broadcast to nodes
ℓ_i^c, u_i^c = MLP^c([h_i, z])                         four channels
α^c          = segment_softmax(ℓ^c, batch)             per cell, per channel
ΔE_SR        = Σ_c n_c Σ_i α_i^c u_i^c
q_host       = Q(h_i) − scatter_mean_j
q_pol        = (Σ_c n_c)(MLP_p([h_i, z]) − scatter_mean_j)
a            = softplus(MLP_a(scatter_mean_i h_i))     one per graph, no n dependence
q_carrier    = a Σ_c s_c n_c α_i^c
```

- **Every reduction is per cell, keyed on `batch`** — the softmax (§9.3) and equally both mean
  subtractions and the descriptor pool feeding `a`. A batch-global mean in `MLP_a` yields one
  `a` per *batch* rather than per configuration, and global mean subtraction breaks
  `Σ_i q_i^host = 0` per cell.
- **`ΔE_SR` and `E_LR` are graph-level terms added after `scale_shift`**, as MACELES does with
  `les_energy`; folding them into `node_inter_es` would apply a per-atom shift to a quantity
  that is intensive by construction.
- **`base_energy ≡ E_short + E[q_host]`** — the host Ewald term belongs to the base branch
  (§3.1). If it does not, the §8.1 identity passes trivially while `L_tot`'s detached base is
  wrong.
- Outputs: `energy`, `base_energy`, `delta_energy`, `forces`, `base_forces`, `delta_forces`,
  `stress`;
  diagnostics `latent_charges`, `alpha`, `logit_gap`, `screening_amplitude`; and the
  regulariser tensors `u`, `p`, `q_host`. Forces and delta-forces are two `autograd.grad` calls
  on one graph (`retain_graph`, and `create_graph` in training); `base_forces` is their
  difference, which is exact and free, and is what `L_base` is trained against.
- The host long-range term is part of the base branch but still depends on the positions, so
  it must enter the differentiated energy alongside the correction. Leaving it out costs the
  entire host electrostatic contribution to the forces and no `n = 0` test can see it.
- A fresh model starts at the base potential for the short-range branch, and `n = 0` is exact
  always. With the long-range branch on it does **not** start there for charged
  configurations: the amplitude is deliberately initialised at a stated dielectric gauge
  rather than at zero (plan section 9.6), so `q_carrier` is non-zero from the first step.
- Numerics (§9.6): logits clamped before the max-subtraction; float64 accumulation in the
  softmax; `MLP_u`, `MLP_p`, `Q_host` initialised near zero; `MLP_a` bias initialised at
  `softplus⁻¹(1/√ε_∞)`. Segment softmax via `torch.scatter_reduce(..., "amax")` if it scripts
  cleanly, otherwise a local helper in `defect_blocks.py` — not in `mace/tools/scatter.py`,
  which is excluded from lint and coverage as vendored.

---

## 5. Long-range branch — `latent_ewald.py`

Verified against the `les` commit pinned in `requirements/les.txt`: multi-channel charges
`[n_atoms, n_q]`; `norm_factor = 90.4756 ≈ 1/(2ε₀)` in eV·Å/e²; and a real-space fallback
selected by `det(cell) < 1e-6` which is pure-torch, shares the `make_kernels` smearing and the
same self-interaction treatment, and is cell-independent.

- **Three-piece decomposition without touching LES internals.** `E_LR` is quadratic in `S`
  *including* the self-energy subtraction, so
  `2·E_cross[env, carrier] = E[q_env + q_carrier] − E[q_env] − E[q_carrier]` is exact. This
  satisfies both "a thin layer, not a reimplementation" (§9.1) and "expose the three pieces"
  (§9.4).
- **Training costs two evaluations, not three**: `E[q_host]` (which is `E_LR(R, 0)`) and
  `E[q_host + q_pol + q_carrier]`. The full decomposition is inference-only.
- **The isolated evaluator is the same LES module called with a zero cell**, so it inherits
  identical `σ` and self-energy handling. It contributes no stress term — variable-cell
  relaxation under `E_dilute` is therefore unsupported, consistent with §9.5 — and is O(N²).
- `E_dilute = E_base + ΔE_SR + E[S_env]^per + 2·E_cross^per + E[S_carrier]^iso`, behind a
  `dilute` forward flag, with the factor of 2 written explicitly and unit-tested.

`σ` and `k_c` come from `les_arguments` (`sigma`, `dl`); they are convergence parameters to be
recorded, not hyperparameters to tune against validation loss. `LatentEwald` wraps
`les.module.ewald.Ewald` directly rather than the `Les` façade, which would additionally build
an unused descriptor-to-charge network; the argument names and defaults are LES's own, so an
`--les_arguments` file written for MACELES is valid here.

Conventions pinned empirically against the pinned LES commit, now covered by tests:

- the kernel `exp(−σ²k²/2)/k²` transforms to `erf(r/(σ√2))/r`, with
  `1/(4πε₀) = 14.3996 eV·Å` (LES stores `2π` times this as `norm_factor = 90.4756`);
- the periodic sum reproduces the NaCl Madelung energy to 1e-6 relative at `σ = 0.5`,
  `dl = 0.6`. At the production `σ = 1.0` it differs from the point-charge value by ~1.6% at
  nearest-neighbour separation: that is the smearing, not an error;
- a single isolated charge has exactly zero energy, so in a two-charge case the whole pair
  energy sits in `2·E_cross` — which is how the factor of 2 is tested.

`E_dilute` is assembled through the difference form: the environment and cross terms carry the
same boundary condition in both evaluators and cancel exactly, so only
`E[q_carrier]^isolated − E[q_carrier]^periodic` is evaluated. `decompose()` still exposes all
three pieces and is tested against the total, so the cross term cannot be silently dropped.

The host latent charge comes from an MLP on the projected invariant features rather than from
copies of the MACE readouts as in MACELES — a strict generalisation of the linear readout,
and it keeps one feature pathway for all correction heads.

---

## 6. Loss — `DefectLoss`

Masks derive from the data rather than from config types: `base_*_weight` selects (a), a
non-zero `delta_energy_weight` selects (b), the remainder selects (c).

- **(a) `L_base`** — energy and forces against `base_energy` / `base_forces`
- **(b) `L_Δ`** — `delta_energy`, `delta_forces`; weighted highest, since fixed-geometry
  charge-state differences are exactly free of base-model error
- **(c) `L_tot`** — `pred["base_energy"].detach() + delta_energy` against `energy`, **plus the
  ordinary total-force term**: forces carry no referencing constant, so they are valid here and
  must not be masked
- **(d) `L_P`** — hydrostatic trace only, off by default pending the §9.5 finite-difference
  verification
- **Regularisers** — L2 on `u`, `p`, `q_host`; optional `λ(a − 1/√ε_∞)²` prior; the logit gap
  is logged, never penalised

New CLI: `--loss defect`, `--delta_energy_weight`, `--delta_forces_weight`,
`--total_energy_weight`, `--pressure_weight`, `--defect_u_l2`, `--defect_p_l2`,
`--defect_qhost_l2`, `--eps_inf`, `--eps_inf_prior`.

---

## 7. Plumbing

- `arg_parser.py` — `--model MACEDefect`, `--error_table DefectRMSE`, `--band_edges_file`,
  the loss weights above, the model-shape arguments (`--carrier_feature_dim`,
  `--counter_embedding_dim`, `--carrier_mlp_hidden`, `--share_logits_across_spin`,
  `--use_long_range`, `--eps_inf`), and the new `*_key` arguments. The `*_key` arguments are
  **required, not cosmetic**: `run_train` builds its keyspec from `vars(args)`, so a key that
  has an entry in `DefaultKeys` but no CLI argument never reaches the keyspec and its property
  silently stays `None`
- `model_script_utils.py` — the `MACEDefect` build branch. The foundation warm-start branch
  inherits a known-broken path: the documented MACELES xfail is `'ScaleShiftMACE' object has
  no attribute 'les'`, i.e. the finetuning path never wraps the loaded model in the LES
  subclass. §9.6 wants warm-start with `E_base` frozen for the first epochs, so the wrapping is
  fixed once in the shared path and covers `FoundationMACELES` and `FoundationMACEDefect`
  alike; the MACELES xfail is removed when it passes.
- `scripts_utils.py` — the `get_loss_fn` branch, and `MACEDefect` added to the model-name
  lists in `extract_config_mace_model`
- `run_train.py` — loads `--band_edges_file` and threads it through `get_dataset_from_xyz`;
  raises a clear `NotImplementedError` if cueq or oeq conversion is requested for
  `MACEDefect`, since converting the new heads is untested
- `train.py` `MACELoss` and `tables_utils.py` — a `DefectRMSE` table reporting E, F, ΔE and
  ΔF, with the fitted `a` and the logit gap logged per epoch
- `calculators/mace.py` — `carrier_counts` in the default `info_keys`, torch-side
  canonicalisation at the entry point, and the `energy_scale` / `dilute` options of §8

---

## 8. Inference

The calculator reads the same info keys it was trained with and canonicalises the counters at
the entry point. Two options:

- `energy_scale="referenced"` (default) — returns the model's own output and never depends on
  constants read off the `Atoms` object. `"raw"` is opt-in and requires an explicit reference
  table (checkpoint registry or JSON), *not* per-frame info keys: a frame whose
  `(host, natoms)` is missing from the table is an error, never a silent zero. Reading the
  constants per frame would let two curves of a configuration-coordinate diagram shift relative
  to one another because one frame's `e_cbm_cell` was absent or slightly different — the §7.2
  failure mode relocated into the calculator, and invisible to the `n = 0` unit tests. Forces
  and stress are identical under either scale, so single-`n` relaxation and MD are unaffected.
- `dilute=False|True` — selects `E_total` (periodic, comparable with DFT in the same cell) or
  `E_dilute` (isolated-limit carrier self-term, §7.1). Relaxations run under whichever is
  selected, which makes "relax the charged defect in the isolated limit" a plain ASE
  optimisation.

**Configuration-coordinate diagram.** Interpolate geometries between `R*_q` and `R*_q′`; at
each point call the calculator twice with the two counter vectors and `dilute=True`. The
`defect_thermo` utility puts the two referenced curves on a common raw scale.

**Defect thermodynamics.** For each (defect, `q`): relax under `dilute=True`, then assemble
`E_f(D^q, E_F)` and `ε(q/q′)` from the relaxed energies, the per-host `E_VBM^∞` and the
chemical potentials. Only that utility touches the `∞` edges.

`defect_thermo` and the §7.3 bounded enumeration with UQ gating are deferred; the evaluator
beneath them is in scope.

---

## 9. Tests

`tests/unit/test_defects.py`, requiring no LES:

- canonicalisation: `canonical(n) == canonical(swap(n))` for random `n`; every worked example
  in §2.1 is already canonical; `q` and `M_s` invariant / antivariant under the swap
- counter validity assertions, including the multiplicity cross-check
- the join: correct delta and base targets, `pair_id` groups with zero or several `n = 0`
  members rejected, mismatched geometries within a group rejected, `L_base` weighted once per
  group
- band-edge referencing arithmetic, including the pristine identity that a band-edge carrier
  gives exactly zero
- `Σ_i α_i^c == 1` in float64 without `eps`; all four reductions per-cell in a batch of at
  least three cells of different sizes, with `a.shape == (num_graphs,)`
- permutation invariance of the pooled quantities

`tests/extensions/defect/`, marked `les`:

- `E_total(R, 0) == E_base(R)` and `E_dilute(R, 0) == E_base(R)` for random `R` and random
  parameters
- `Σ q_host = Σ q_pol = 0`, and `Σ_i q_i == a·q` — **not** `== q`
- autograd against finite differences for each branch separately and for both evaluators;
  Hessian symmetry; translation and rotation invariance
- an analytic Madelung constant, to pin prefactors before any absolute energy is trusted
- the cross-term factor of 2 against a two-charge analytic case
- periodic converging to isolated under cell growth
- an end-to-end `run_train` smoke test

---

## 10. Build order, with gates

1. `mace/data/defects.py`, the data-layer keys and fields, and the unit tests — no model yet
2. `MACEDefect` skeleton: counter embedding, pooling, `ΔE_SR`
   — **gate: `E_total(R, 0) == E_base(R)` exactly; the logit gap stable during training**
3. `latent_ewald.py`: wire in LES, pin the prefactors against an analytic Madelung constant,
   expose the three pieces, then add the structured charge assembly including `q_pol`
4. `DefectLoss`, the plumbing and the `DefectRMSE` table; train on paired data
5. `L_tot` with detachment; confirm base-branch predictions on reference-state holdouts have
   not moved
6. isolated evaluator, `E_dilute`, and the calculator options
   — **gate: `E_dilute(R, 0) == E_base(R)`; periodic converging to isolated under cell growth**

Per §9.7, do not proceed past step 2 until the `n = 0` identity holds exactly, and not past
step 6 until `1/a²` and DFPT `ε_∞` agree or the disagreement is understood.

---

## 11. Known risks

- `les` is an optional dependency, not on PyPI, pinned by commit in `requirements/les.txt`, and
  not installed in the default development environment. New LES-dependent tests follow the
  existing convention — `@pytest.mark.les` plus `skipif(not LES_AVAILABLE)` from
  `tests/helpers.py` — rather than introducing a new CI job. They therefore run wherever the
  existing LES tests run, and skip elsewhere. To run them against a local checkout without
  installing anything: `PYTHONPATH=../les/src python -m pytest tests/extensions/defect`.
- The foundation warm-start path is broken today (`'ScaleShiftMACE' object has no attribute
  'les'`: the finetuning path never wraps the loaded model in the LES subclass). The fix is in
  scope and is applied to **both** MACELES and MACEDefect, since it is one bug in the shared
  path; the corresponding MACELES xfail in `tests/extensions/les/test_maceles.py` is removed
  when it passes.
- cueq and oeq conversion of the new heads is untested and excluded initially.

---

## 12. Implementation status

Last updated 2026-08-08. Branch `defect`, not yet committed.

### 12.1 Stages

| Stage (§10) | State | Notes |
|---|---|---|
| 1. Counters, targets, data loading | **done** | `mace/data/defects.py`; keys, `AtomicData` fields, `load_from_xyz` hook |
| 2. Pooling and `ΔE_SR` | **done** | `defect_blocks.py`, `defect_models.py`; gate on the `n = 0` identity passes |
| 3. Latent Ewald, structured charges | **done** | `latent_ewald.py`; prefactors pinned against analytic results |
| 4. `DefectLoss`, plumbing, `DefectRMSE` | **done** | Trains end to end via `mace_run_train` |
| 5. `L_tot` with detachment | **code done** | Shipped with `DefectLoss`; the empirical holdout check needs data |
| 6. Isolated evaluator, `E_dilute`, calculator | **done** | `dilute` / `energy_scale` / `band_edges` options |
| Deferred | — | `defect_thermo` (§7.2), enumeration with UQ gating (§7.3), multi-fidelity (§3.5), `correction_trunk: separate` |

### 12.2 Files

New: `mace/data/defects.py`, `mace/modules/defect_blocks.py`,
`mace/modules/defect_models.py`, `mace/modules/latent_ewald.py`,
`tests/unit/test_defects.py`, `tests/unit/test_defect_model.py`,
`tests/unit/test_defect_loss.py`, `tests/unit/test_defect_calculator.py`,
`tests/extensions/defect/test_defect_long_range.py`,
`tests/workflows/test_run_train_defect.py`.

Modified: `mace/data/{__init__,atomic_data,utils}.py`, `mace/modules/{__init__,loss}.py`,
`mace/tools/{arg_parser,default_keys,finetuning_utils,model_script_utils,scripts_utils,tables_utils,train}.py`,
`mace/cli/run_train.py`, `mace/calculators/mace.py`.

### 12.3 Tests

110 unit tests (49 data / 26 model / 14 loss / 21 calculator) plus 26 LES-marked tests and
one `slow` workflow test. Full unit suite: 365 passed. `tests/unit/test_compile.py` fails,
identically on a clean tree (a pre-existing torch inductor error), and is deselected above.

```
python -m pytest tests/unit -q --deselect tests/unit/test_compile.py
PYTHONPATH=../les/src python -m pytest tests/extensions/defect tests/workflows/test_run_train_defect.py -q
```

Identities that hold exactly (`atol=0`, `rtol=0`), for random geometries and random
parameters: `E_total(R, 0) == E_base(R)` with and without the long-range branch,
`E_dilute(R, 0) == E_base(R)`, `Σ_i q_i^host = Σ_i q_i^pol = 0` per cell, and
`Σ_i q_i == a·q`. Prefactors: the NaCl Madelung energy to 1e-6 relative, the smeared
kernel `erf(r/(σ√2))/r` with `1/(4πε₀) = 14.3996 eV·Å`, and the cross-term factor of 2
against a two-charge case.

### 12.4 Open gates

Both need real training data:

- **the logit gap must stay stable during training** (§10 step 2). Monitored and logged, but
  never yet observed on a real run; §3.2's extensivity argument depends on it;
- **`1/a²` must agree with DFPT `ε_∞`**, or the disagreement must be understood, before
  anything downstream of step 6 is trusted (§9.7).

Also unverified: that base-branch predictions on reference-state holdouts do not move once
`L_tot` is switched on (§10 step 5).

### 12.5 Findings worth carrying forward

Four things that were wrong or missing and are now fixed, each found by a test rather than by
inspection:

- **the host long-range term was missing from the differentiated energy.** `E_LR[q_host]`
  belongs to the base branch but depends on the positions, so omitting it cost the entire host
  electrostatic contribution to the forces (−1.03 vs 2.23 eV/Å in the finite-difference check).
  No `n = 0` test could have seen it, since the term is present on both sides of that identity;
- **ASE's `check_state` ignores `atoms.info`.** A calculator shared across charge states
  returned the first energy for all of them — silently, and precisely in the
  configuration-coordinate workflow. Fixed with an override that also compares the counters
  and host;
- **the `*_key` CLI arguments are load-bearing.** `run_train` builds its keyspec from
  `vars(args)`, so a key with a `DefaultKeys` entry but no CLI argument never reaches the
  keyspec. The symptom was empty ΔE/ΔF columns in an otherwise healthy run;
- **the foundation warm-start bug was in the shared path**, not in MACELES: the generic
  child-copy loop in `load_foundations_elements_default` assumed every child of the new model
  exists on the foundation. Fixed once, covering MACELES and MACEDefect alike.

### 12.6 Caveats

- **A fresh model does not start at the base potential when the long-range branch is on.**
  `MLP_u`, `MLP_p` and `Q_host` start at zero, but the screening amplitude starts at a stated
  dielectric gauge rather than zero (§9.6), so `q_carrier ≠ 0` from the first step and charged
  configurations carry a non-zero correction at initialisation. `n = 0` stays exact.
- **The MACELES network-marked xfail in `tests/extensions/les/test_maceles.py` was left in
  place.** The underlying bug is fixed and covered by local tests, but that test needs network
  access plus `les`, so it was never executed here; it should now XPASS rather than fail.
- **`dilute=True` on a short-range-only model is a no-op**, not an error: there is no carrier
  charge distribution whose boundary condition could change. Tested, but worth knowing.
- **cueq and oeq conversion raise `NotImplementedError`** for this model rather than tripping a
  bare assert.
- The band-edge registry is keyed `(host, natoms)`; two distinct supercells of one host with
  the same atom count need the `cell_id` override, which is specified but not yet implemented.
