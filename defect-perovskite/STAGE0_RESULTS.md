# Transition plan v8, Stage 0 — registry and harness: results and decisions of record

Plan of record: `TRANSITION_PLAN_V8_SPEC.md` (received 4 Sep 2026), section 3. Branch
`single-functional`, from `size-extensivity` @ 4ad038f. Every number below carries its regime
tag. Acceptance run: `defect-perovskite/stage0_acceptance.py` → `golden/stage0_acceptance.json`.

**Regime for every trained number in this document:** `arma_s1` (`~/runs/arma_models/arma_s1.model`,
the arm-A seed-1 model of the speed cycle; uniform float64), evaluated on CPU, float64 default
dtype, 1 thread, `torch.use_deterministic_algorithms(True)`, TorchScript profiling executor off.
No model was trained in Stage 0. The toy numbers are from a Harrison-initialised 2-block test
model (`tests/unit/test_base_cache_precision._model` + `apply_harrison`) on cubic CsPbCl3
supercells rattled 0.02 Å.

---

## Acceptance (section 3), all four items

| item | result |
|---|---|
| golden bit-identity, default policy | 10/10 records (378 fields: chemical potentials, occupations, P, band free energy, energies, forces, stress; pristine, V_Cl⁰, V_Cl⁺; both solver paths) identical to the v6 capture at f8c61eb, at every commit of Stage 0 |
| band functional ∂F/∂H_ab = P_ba | passes for `count_fill` (Fermi–Dirac and Gaussian) and for the mock's matrix-level contract (`test_band_functional.py`) |
| mock policy | passes only its own contract; unreachable from production (AST: `register_test_policy` appears in no module under `mace/` but `defect_state.py`) |
| other terms' harness status | recorded, below (`golden/stage0_fd_v6.json`, 96 summary rows at 2c3fd96) |
| `count_fill` the sole reachable policy | `registered_policies() == ('count_fill',)` in a fresh process that imports the model and the trainer |
| test suite | 409 passed (`tests/extensions/defect`, `test_base_cache_precision`, `test_flag_plumbing`, `test_defect_spectral_range`) |

Commits: f8c61eb (spec) · 4eb39e2 (golden) · 7bf6344 (0.1 state) · 0f4d6ac (0.2 cache keys) ·
691d18e (0.3 mock, 0.6 band functional) · 2a313e7 / d4086dd (0.4 registry, kernels, deletions,
un-detach) · 2c3fd96 / cad0fcd (0.5 harness, status record) · 790e410 (0.7 Tier 1) · 121e1cd /
420d684 (0.8 Tier 2) · d0cb388 (0.9 densities) · 37557f5 (0.10 round-trip) · this commit (0.11).

---

## What was built (section 3, item by item)

- **Term registry and two kernels (§2.5, §2.6).** `defect_terms.registry(model)`: `base`
  (`base_trunk_energy`), `lr_host` (`energy_lr_host`), `band` (`delta_sr_energy`, potential
  "band"), `lr_carrier` (`delta_lr_energy`, potential "absent" — the v6 fact Stage 5 changes).
  Kernels `periodic` / `isolated` behind one entry point; the model's `gauge` flag is config
  and `gauge="isolated"` is bit-equal to `dilute=True` on the periodic model.
- **Deleted:** `defect_image.py` (image compensation; `image_compensation=True` now refuses),
  the depth sigmoid, `lr_detach_density` (ignored with a warning), `lr_freeze=False` (ignored;
  the branch is always frozen at its ε∞-only values), the `data["occupations"]` override and
  `CountingHead(occupations=...)` (both refuse). E_LR's density is no longer detached — golden
  identical because the frozen branch carried no gradient.
- **`ElectronicStateSpec`, `S_ref`, dispatch (§2.1).** Physical key `(schema, Q_formal, ΔN_σ,
  policy, payload)`, `state_id` excluded; the null is key-equality to `S_ref`, not `Q = 0`.
  The adapter from the `(e_maj, e_min, h_maj, h_min)` counters calls the legacy fill exactly.
  Cache format 2: entries keyed `(frame_key, state digest)`, the checksum carries the policy
  version. Both the loop and the batched solver report `mu`, `occupations`, `fills` and the
  Hamiltonian they diagonalised (`H_orbital`) through `internals`.
- **Finite-difference harness (§7.2).** `defect_fd.py`: forces and strain, per term and
  assembled, both gauges; `err(h) = c + A h² + ε/h` fitted over eight steps; verdicts pass /
  missing_derivative / fail; strain applied to the geometry (positions, cell, shifts), never
  through a displacement tensor; stress convention `(1/V) ∂E/∂D` with the off-diagonal factor.
- **Composition-class constructor (§2.1), both tiers** — `defect_composition.py`; below.
- **Density objects** — `defect_density.py`; below.
- **Config round-trip** — every §3 key, named verbatim in `test_config_round_trip.py`.

---

## Harness status of the v6 functional (`golden/stage0_fd_v6.json`, arma_s1, uniform float64)

The record §3 asks for: what the existing functional's derivatives are, per term, before
anything changes. Steps h ∈ {8e-4 … 1e-1} Å on twenty random atoms per frame; tolerances
1e-4 eV/Å (force), 1e-6 eV/Å³ (stress); frames: two neutral pristine, V_Cl⁰ at 79, V_Cl⁺ at 79
(one ordinary, gap 0.161 eV; one near-crossing, gap 0.048 eV) and at 159.

| term | forces | stress |
|---|---|---|
| base | pass everywhere, floor 1.5e-8 | pass, 5.5e-8 |
| lr_host | pass (exactly 0 — the frozen branch) | pass |
| band | pristine/neutral pass; **charged: missing_derivative**, floor 3.4e-2 (ordinary) / 7.4e-2 eV/Å (near-crossing) | **charged: missing_derivative**, 4.1e-4 – 8.5e-4 eV/Å³ |
| lr_carrier | **charged: missing_derivative**, 2e-3 – 2.1e-2 eV/Å | **missing_derivative**, 2e-5 – 4.8e-5 eV/Å³ |
| assembled | as band | as band |

Causes, named by the harness: the head detaches `node_feats` and the density matrix in its
energy (the force response covers the position dependence of H but not of the features), and
the Ewald terms are handed the undisplaced `data["cell"]`. These are the Stage 1 items.

---

## Decisions of record — where the implementation reads the plan against its letter

Each is in the module docstring it belongs to; collected here because a reader who only sees
"Tier 1 and Tier 2 pass" would be surprised by them.

1. **The Tier 1 ambiguity window is centred on the counting cut, not on VBM_al**
   (`defect_composition.tier1`). The clause "accept if no eigenvalue lies within ±δ of VBM_al",
   read literally, excludes every class including the pristine one: the valence edge *is* at
   VBM_al by definition of the alignment, and every class cell's own valence top scatters about
   it (7 meV on a thermal 80-atom pristine frame, 22–139 meV on vacancy cells). The operative
   rule: levels ≤ VBM_al + δ − Δ_s are valence, levels ≥ VBM_al + δ + Δ_s are frontier, a level
   in between makes the class Tier 1-ambiguous. With δ = 2Δ_s = 0.1 eV the ambiguous band is
   (VBM_al + 0.05, VBM_al + 0.15). The margin absorbs edge scatter; the window is one smearing
   width either side of the cut. Both are config (`class_constructor.delta`, `.window`).

2. **The class reference geometry is the class's first frame, always.** The stoichiometric
   frames are thermal snapshots (the two 80-atom pristine golden frames have cells 15.80 ×
   16.08 × 11.38 and 16.17 × 15.88 × 11.40 Å), so "the ideal defect geometry built from the
   pristine cell" is not constructible; and building V_Cl from one would choose *which* Cl —
   a defect position, which principle 9 forbids. The frame key of the geometry used is cached.

3. **One alignment per class, against the pristine reference tiled to the class cell.** The
   head is spin-restricted, so M_VB,σ is one integer for both spins (aligning per spin gave two
   edges a one-level quantile shift apart — noise). The pristine spectrum aligned to is the
   reference **tiled** to the class cell so both sample the same k-points: the 159-atom cell is
   the 80-atom cell doubled along c with the axes relabelled (`tiling_map` → factors (1,1,2),
   perm (0,2,1)); an untiled reference put the 2×1×1 cell's valence top 0.14 eV above VBM_al
   and would have counted a valence level as frontier. Quantile window 0.05–0.60 of the
   occupied manifold (the depth scorer's, `b6_depth_edges.QUANTILES`).

4. **Tier 2: the sink is never crossed while coupled** (`interpolated_hamiltonian`). The plan's
   Path A — the ghost level riding to +E_sink while its hoppings fade — sweeps every state above
   it with a residual coupling; the fine-step limit of that sweep is *adiabatic* following,
   which swaps the ghost for a physical state at each crossing and converges to the lowest-M
   subspace of H⁽¹⁾, i.e. the vacancy level counted as valence. Measured on the toy: Path A
   gave M_VB = 104 (Q_core = 0) against Path B's 100 (Q_core = +1), with the finer schedule
   *worse*. Implemented: a ghost's on-site level stays where the pristine put it until its
   hoppings are exactly zero and only then is parked at the sink (an exact relabelling of
   decoupled eigenstates); an addition leaves the sink at λ = 0⁺ while still decoupled. The
   ghost levels then sit inside the occupied manifold throughout, their crossings are within the
   transported subspace, and E_sink cannot enter the transport at all (the ×4 / ×16 invariance
   holds by construction). Paths A and B still differ in the ordering of the physical changes
   (simultaneous vs. fade → switch → recouple); min separation 1.000 on every path and schedule.

5. **r_match = 2.0 Å, against §2.7's "half the pristine nearest-neighbour distance" (1.4 Å for
   Pb–Cl).** On arma_s1 the Cl of a V_Cl snapshot sit up to 1.58 Å from their sites in the
   pristine snapshot (mean 0.5–0.7 Å); at 1.0 and 1.5 Å one swung Cl became a spurious ghost +
   addition whose in-band level broke contiguity and left the class uncounted. 2.0 Å is half
   the Cl–Cl distance. The augmented assignment charges r_match² per unmatched *pair*, so an
   atom beyond r_match is a ghost plus an addition rather than a stretched match.

6. **The pristine placement for ρ_static is a density alignment, not a site correspondence**
   (`defect_density.align_pristine`): the fractional shift minimising ‖ρ_static^raw‖² on the
   class reference frame (analytic objective, Newton-refined), cached per class and reused by
   every frame, whose ‖ρ_static^raw‖ is reported. The 80-atom cell holds four 20-atom
   orthorhombic cells, so the density objective and the correspondence's squared-displacement
   objective can pick minima one sub-lattice vector apart; what is checked is the residual at
   each (the acceptance JSON records both).

7. **δZ_i for the ω rule is the raw density read at each probe point in charge units**
   (`residual_shape`), with pristine sites within 0.5 r_res of a present atom not counted as a
   second probe. A vacancy's charge reads at its 2.8 Å neighbours at 2 %; a 0.5 Å thermal pair
   at 12 %. Toy V_Cl: the vacancy carries ω = 0.96 on the exact frame, 0.76 on a 0.05 Å-rattled
   one. §7.6 is its sensitivity test (r_res).

8. **Edge-projector leakage is as specified.** With s^e = sigmoid((ε − VBM_al − δ)/Δ_s), a
   fully occupied level *at* the aligned edge carries weight sigmoid(−2) = 0.12 in the electron
   channel; the channels are separately normalised, unneeded ones are unused (count 0), and a
   needed channel with zero weight refuses. Noted, not changed.

9. **Per-frame counts are netted per spin** (`frame_counts`): `net_σ = n_e^class − n_h^class +
   ΔN_σ`, `n_e = max(net, 0)`, `n_h = max(−net, 0)`. The plan's literal per-channel formula
   gives V_Cl⁺ one electron *and* one hole (q_F = 0 either way, so the assertion does not
   discriminate); the plan's own benchmark says n_e = 0, and the netted form is the one that
   produces it. The benchmark is a test.

10. **E_sink and η defaults per §2.7** (50 eV above the pristine CBM; 1e-3), resolved to numbers
   in the table whether or not a class needed Tier 2. **Every None default is resolved at
   construction** (r_split = first-block cutoff, r_orb = covalent radii, δ = 2·smearing,
   Δ_s = smearing) so the extracted config carries numbers.

---

## The class table on arma_s1 (golden frames; `golden/stage0_acceptance.json`)

Smearing Gaussian 0.05 eV → δ = 0.10, window 0.05; pristine gap 2.391 eV; E_sink = CBM + 50.

| class | tier | M_VB | N_σ | n_e | Q_core | shift / spread (eV) | nearest level to cut (eV) | tiled |
|---|---|---|---|---|---|---|---|---|
| Cl48 Cs16 Pb16 (pristine) | 1 | 208 | 208/208 | 0 | 0 | 0 / 0 | −0.100 (its own VBM) | 1×1×1 |
| Cl47 Cs16 Pb16 (V_Cl, 79) | 1 | 204 | 205/204 | (1, 0) | **+1** | −0.082 / 0.119 | −0.051 | 1×1×1 |
| Cl95 Cs32 Pb32 (V_Cl, 159) | 1 | 412 | 413/412 | (1, 0) | **+1** | +0.280 / 0.082 | −0.409 | 1×1×2, perm (0,2,1) |

Q_core identical across sizes (§7.1's requirement); the benchmark V_Cl⁰ → Q_core = +1,
n_e,maj = 1, q_F = −1 and V_Cl⁺ → n_e = 0, q_F = 0 hold on the netted per-frame counts.

**Tier 2 forced on both V_Cl classes** (site correspondence: one Cl ghost, no addition, no
substitution, max matched displacement 1.45 / 1.58 Å): accepted on both paths and both
schedules, four ghost γ (= 1.000) and 204 / 412 physical (γ < 1e-3), M_VB identical to Tier 1,
min separation ≥ 0.997; 3 s at 79 atoms, 12.6 s at 159 on one CPU thread.

**Alignment noise, to be read with the table.** The quantile alignment between thermal
snapshots is good to ~0.1 eV: the second 80-atom pristine snapshot aligned to the first gets
shift −0.125 eV with its valence top at +0.067 eV; the 79-atom class is accepted with 1 meV to
spare from the window (nearest −0.051 vs window 0.050); the 159-atom class's VBM_al sits
~0.3 eV above its own valence top (shift +0.280, spread 0.082) — a cell from a different MD
with a 2 % larger lattice constant, so the deep bands and the edge do not shift together. None
of this moves an integer here (the vacancy level is 0.7–1.0 eV above the edge and the gap is
2.4 eV), and Tier 2 is the robust path when it would; Stage 4's class-count invariance test
(`verify_class_table`) recomputes the table after training and reports any integer that moved.

Placement vs correspondence on the V_Cl classes: the density shift and the correspondence's
translation differ by 8.06 Å (79) and 7.96 Å (159) — one 20-atom orthorhombic sub-cell, a
near-symmetry of the pristine density. ‖ρ_static^raw‖ at the density's shift is 0.661 vs 0.710
at the correspondence's (79) and 0.808 vs 1.016 (159), against ‖ρ_Z^present‖ = 1.44 / 2.01: the
density's own minimum is the lower, as it must be, and both residuals are ~40 % of the present
density — the thermal-displacement dipoles of every atom (a 0.6 Å mean Cl swing at r_res = 1 Å
gives a 41 % relative norm for one displaced pair), which the plan says ρ_static^raw includes
and reports separately on the ladder (ideal vs thermal frames).

## Toy (Harrison-initialised test model, cubic 2×2×2 and 2×2×4, rattle 0.02 Å)

Pristine 40: Q_core = 0. V_Cl 39: Tier 1, M_VB = 100, N = 101/100, Q_core = +1. Pristine 80
(tiled 1×1×2 reference): Tier 1, Q_core = 0. V_Cl 79: **Tier 1 ambiguous** (a vacancy-perturbed
valence level at VBM_al + 0.139 eV, inside the window) → **Tier 2**: one ghost, four ghost γ,
M_VB = 204, Q_core = +1 — the 39-atom integers, on both paths and schedules. Tier 2 forced on
the 39-atom class reproduces Tier 1's 100 exactly.

Synthetic Tier 2 tests (gapped tight-binding toys, `test_composition_classes.TestTier2Synthetic`):
removal → one site's worth of ghost γ; two same-species sites exchanging charge character and
levels crossing inside the valence manifold → unchanged M_VB with unit separation; an
interstitial with its level in the gap and a substitution → the expected integers on both paths;
a genuine valence/frontier closure → flagged; the endpoint classification invariant under random
rotations within the transported subspace; E_sink ×4 / ×16 and dlambda halved → γ unchanged to
1e-8.

Density tests (`test_defect_density.py`): ρ_static^raw ≡ 0 on the pristine reference geometry
(norm < 1e-9); ∫ = 0 on thermal pristine frames; ‖ρ_static^raw‖ linear in a scaled
displacement down to zero; ∫ρ_static^raw = −Z_Cl for V_Cl at both sizes with the heaviest ω at
the vacancy site; ∫ρ_static^def = Q_core; channels separately normalised; degeneracy-safe
site densities; ∫ρ_F = q_F at every w; the synthetic m = 3 frontier class with Q_formal = 0 →
q_F = −3 and ∫Δρ_def = 0.

---

## Not done / carried forward

- No forward consumes the class integers or the density objects yet (Stage 0 scope is the
  objects and their tests); `frame_counts_batch` and `frame_static_densities` are the entry
  points Stage 1 and Stage 4 will call.
- The trainer builds the table over the train and validation sets before training and skips
  with a warning when the model has no Madelung composition (no pristine formula → no isolated
  gauge for that run).
- Interstitials whose occupied level lands inside the valence band are, by the plan's own
  rule (transported rank = the pristine one; contiguity required), genuinely ambiguous —
  correct by the rule, noted here because it is the rule's consequence, not a bug.
- `r_match`'s §2.7 default is departed from (item 5); `r_match` remains config.
