# Transition plan v8, Stage 1 — the existing functional formalised: results and decisions of record

Plan of record: `TRANSITION_PLAN_V8_SPEC.md`, section 4. Branch `single-functional`. Stage 0's
record is `STAGE0_RESULTS.md`; this file continues its numbering of the decisions of record.
(`STAGE1_RESULTS.md` is the earlier programme's Stage 1 — the Madelung retrain — not this one.)

**Regime for every trained number in this document:** `arma_s1` (`~/runs/arma_models/arma_s1.model`,
uniform float64), CPU. Golden comparisons: 1 thread, `torch.use_deterministic_algorithms(True)`,
TorchScript profiling executor off. Finite differences and sweeps: 8 threads. No model was trained
in Stages 1.0–1.2. Toy numbers are from the Harrison-initialised 2-block test model on cubic
CsPbCl3 2×2×2 cells (`tests/extensions/defect/test_frontier_term.py`).

---

## Stage 1.0 — the hook and the golden's second mode (commit 08d4747)

- The trainer's class-table hook is device-safe (species charges to CPU; densities and the
  placement on CPU; smoked on b3 GPU 4).
- `stage0_golden.py compare --energies-only`: derivative fields may differ, every value field
  (energies, chemical potentials, occupations, density matrices, Madelung shifts) must not.

## Stage 1.1 — the derivatives the harness named (commit 9b5da41)

Three detached or undisplaced quantities, each found by holding one piece fixed on both sides of
the finite difference until the floor vanished:

1. the counting head detached the trunk features at entry — removed;
2. the Madelung per-site charge channel detached its features (`deviation`) — removed (the 6e-2
   eV/Å band-force floor on charged frames);
3. the head read the undisplaced cell and positions — it now reads `cell + cell·sym(D)` and the
   displaced positions (the 6e-4 eV/Å³ band-stress floor).

`golden/stage1_fd_derivatives.json` (96 rows uncached, 10 cached; arma_s1, 8 threads):

| term | forces | stress |
|---|---|---|
| base | pass, 1.5e-8 | pass, 5.5e-8 |
| band | **pass everywhere**, 1.2e-6 eV/Å (was 3.4e-2 – 7.4e-2) | **pass**, 3.4e-6 eV/Å³ (was 4e-4 – 8e-4) |
| lr_host | pass (exactly 0) | pass |
| lr_carrier | missing_derivative, 1.5e-2 – 2.1e-2 | missing_derivative, 1.6e-5 – 5e-5 |

Energies bit-identical to the v6 golden (energies-only mode 10/10). Base cache: the head's force
under the cached forward is exact (band 1.2e-6); the cached total carried only the lr_carrier
floor, so the campaign can train cached — decision 11 below.

---

## Stage 1.2 — Φ_FF on the channel-normalised ρ_F, response-density forces (this commit)

### What was built

- **`defect_counting.channel_matrix`** — `Q = U diag(f s^e) Uᵀ` / `U diag((1−f) s^h) Uᵀ` as an
  autograd Function of ONE Hamiltonian whose backward is the Daleckii–Krein map of the general
  matrix function `g = f·s` (divided differences, `g′ = f′s + fs′` at coincidence, the fixed-N
  μ-correction through `∂g/∂μ = −f′s`), never the eigenvectors. Verified against central
  differences in H on a gapped toy (both smearing families, fills with μ in the gap and at a
  partially occupied frontier, an exactly degenerate pair; `test_channel_matrix.py`, 14 tests),
  and shown to reduce to the density-matrix Function at `s ≡ 1`.
- **The head hands back its Hamiltonian** (attached) with the detached spectrum, the four fills'
  occupations and chemical potentials and the graph's rigid level shift, per graph, on
  `SpectralOutput.frontier` — on both solver paths.
- **`defect_frontier.py`** — `Φ_FF = ½ B_img[wρ_F,loc, wρ_F,loc]`, `B_img = G_PBC − G_∞`
  (§2.8), `B_img[ρ_ext, ·] = 0`, carried as §2.2's `Φ_F(Q) − Φ_F(0)` (decision 12). The
  channels are read from the head's own fills through the edge projectors, normalised
  separately, combined with the per-frame integers of the class table (`frame_counts`), switched
  by `w(P)`; the Ewald evaluator is the model's own at `σ = r_res` (`MACEDefect.frontier_ewald`,
  no parameters; decision 14). Zero under the isolated gauge. Two Ewald passes per batch, four
  matrix functions per off-reference graph.
- **The registry** (`defect_terms.registry`): `base`, `band`, `frontier`; `lr_host` and
  `lr_carrier` retired (decision 13). A new column `response` says how δE/δP reaches the FORCES
  (`band` / `divided_difference` / `none` / `absent`), separately from `potential`, which says
  whether it enters H (`absent` for the frontier term until Stage 5). No term of the Stage 1.2
  functional has an absent response.
- **`defect_composition.ensure_class_table`** — the one helper every driver uses (trainer, FD
  driver, golden, continuity, scorers): builds the table if the model has none, from frames in
  the order given (first frame of each class = its reference geometry), and logs the reference
  frame keys. The FD/golden drivers build it from the golden frames, pristine first.
- **The forward refuses** a frame off its reference state when the model has no table, naming
  the remedy; a reference-state batch runs without one (the term is an exact zero there).
- **Drivers**: `stage0_fd.py` gains the projector-window frame (the pool frame whose level is
  nearest `VBM_al + δ`, from the class record and the model's spectrum at S_ref) and the
  159-atom charged frame (`--frames qp1_159`), and builds the table; `stage0_golden.py compare --stage12` (the allow-list: retired and new fields may be
  present on one side; on a record with a charged graph the totals and derivatives may differ;
  on a neutral record NOTHING may); `stage12_continuity.py` (§7.1 on arma_s1).
- **Retired with the branch**: `_isolated_carrier_self`, the `lr_start_epoch` gate (the buffer
  stays for checkpoints), the `dilute_correction`, `latent_charges*`, `screening_amplitude`,
  `polarisation`, `delta_lr*`, `energy_lr_host` outputs; the tests written against them
  (`test_carrier_self_isolated`, `test_host_carrier_repartition`, `test_host_charge_alive`,
  `test_lr_start_epoch`, the structured-charge and dilute-limit classes of
  `test_defect_long_range`, the amplitude/invariant tests of `test_defect_model`). The
  `StructuredLatentCharges` module is still constructed (checkpoint compatibility) and is never
  called — asserted by a test.

### Acceptance

| item | result |
|---|---|
| channel-matrix derivative in H | 14/14 against central differences (toy, 1e-6 relative) |
| identities (toy) | head correction an exact zero at S_ref (no graph; bit-identical); `∫ρ_F = q_F` to 1e-10; terms sum to the assembled energy to 1e-12; isolated gauge zero; `w → 0` ⇒ `Φ_FF → 0`; batched = per-graph path to 1e-9; rigid c-table shift leaves the term unchanged to 1e-9 |
| FD, toy (V_Cl⁺ 39 atoms) | frontier force pass (4e-11 eV/Å) both gauges, stress pass (9e-8 eV/Å³); with the channel objects held fixed the harness reports `missing_derivative` with floor > 1e-4 — the density response is a visible part of the force |
| FD, arma_s1 | table below |
| continuity, toy (§7.1) | on-site sweep −3…+3 eV on the Pb nearest the vacancy through both windows into the band, and a geometric path between two thermal frames: the largest consecutive change of E, Φ_FF, w and F falls by ≥ 2× when the step is quartered (ratio ≤ 0.5); `q_F` constant; integers unchanged |
| continuity, arma_s1 (§7.1) | table below (`golden/stage12_continuity.json`) |
| golden vs v6 (`--stage12`) | 10/10 on the final code: every head field (μ, occupations, P, band term, Madelung shift, trunk energy) and every neutral record bit-identical; charged records differ only in the totals — by 0.43 eV, the retired E_LR carrier out and Φ_FF in — and their derivatives |
| GPU (b3, GPU 4) | arma_s1 on cuda: class table built on the GPU model in 3.8 s (`ensure_class_table` infers the model's device); training forward + backward on the qp1_79 pair 1.9 s, Φ_FF 0.41889 / 0.41344 eV (CPU: 0.41889 / 0.41344), forces finite, gradients on all 18 head parameters; qp1_159 on the per-graph path likewise |
| batched = per-graph solver path, forces | qp1_79 pair, eval and training forwards: forces agree to 8e-15 eV/Å, Φ_FF to 3e-15 eV |
| test suite | 524 passed, 1 skipped (`tests/extensions/defect`, `test_base_cache_precision`, `test_flag_plumbing`, `test_defect_spectral_range`, `test_defect_model`, `test_defect_cueq`, `test_pol_gate`, `test_defect_loss`); four unit files retired with the branch |

### FD on arma_s1 (`golden/stage12_fd.json`; h ∈ {1e-2 … 1e-4} Å, 4 components per frame; tolerances 1e-4 eV/Å, 1e-5 eV/Å³)

Frames: charged ordinary (gap 0.161 eV), charged near-crossing (0.048 eV), charged
projector-window (a level 0.000 eV from the cut — the near-crossing frame's level sits on
the cut), V_Cl⁰ 79, pristine 80; both gauges.

| term | forces (periodic / isolated) | stress (periodic / isolated) |
|---|---|---|
| base | pass, 1.5e-8 / 1.5e-8 | pass, 5.5e-8 / 5.5e-8 |
| band | pass, 1.2e-6 / 1.2e-6 | pass, 3.4e-6 / 3.4e-6 |
| frontier | **pass**, 9.0e-8 (ordinary 5.2e-8, crossing 9.0e-8, window 8.5e-9) / exactly 0 (no image term) | **pass**, 3.0e-6 (ordinary; crossing 6.1e-8, window 2.3e-8) / exactly 0 |
| assembled, and the model's own `forces`/`stress` | pass, 9.5e-7 / 1.3e-6 | pass, 6.6e-6 / 3.6e-6 |

100 rows, every one a pass. The 159-atom charged vacancy (`golden/stage12_fd_159.json`,
periodic, the tiled 1×1×2 class): base 4.5e-9, band 8.9e-6, frontier 7.7e-6, assembled 2.1e-6
eV/Å; stress base 5.3e-8, band 5.8e-6, frontier 3.9e-6, assembled 9.1e-6 eV/Å³ — all pass. The first run of this table, before
decision 15, also passed the frontier term — at 1e-16 eV/Å, because the term was 1e-11 eV: a
vacuous pass the diagnostics (w_ref = 1e-4, projector weight 132) exposed. The values above are
of a term of 0.42 eV whose force response is real (the toy's frozen-channel check).

Under the base cache (`~/runs/stage12_fd_cache.json`, periodic, charged ordinary and window):
`base` and `assembled` read `missing_derivative` (5.5e-1 eV/Å) by construction — under the cache
the trunk energy is a constant and its force is the cached one — while `assembled_model` (the
model's own force output) passes at 9.5e-7, `band` at 1.2e-6 and `frontier` at 5.2e-8 eV/Å: the
cached training force is the derivative of the physical energy, frontier term included
(`golden/stage12_fd_cache.json`; decision 11 stands).

### What the term is on the golden frames (arma_s1)

| frame | counts (n_e, n_h, q_F) at S | Φ_FF(S) − Φ_FF(S_ref) | w at S | reference cloud: p, w, projector weight, sites |
|---|---|---|---|---|
| V_Cl⁺ 79 (`eval_qp1` first) | (0,0), (0,0), 0 | **+0.419 eV** | — (ρ_F = 0) | 0.148, 0.885, 1.531, Pb 25 / 21 (−0.21 / −0.15 e) |
| V_Cl⁺ 159 | (0,0), (0,0), 0 | **+0.410 eV** | — | 0.081, 0.967, 1.000, Pb 53 / 39 (−0.18 / −0.18 e) |
| V_Cl⁰ 79 (train first) | (1,0), (0,0), −1 | 0 (S = S_ref) | 0.988 (p 0.032; Pb 21 carries −0.62 e) | same |
| pristine 80 | (0,0), (0,0), 0 | 0 | — | — |

So on V_Cl⁺ the term is minus the image energy of the neutral vacancy's own electron cloud
(§2.2's Φ_F(0)), 0.42 eV at both sizes: the bare (unscreened) image attraction of a cloud that
is 89–97 % "localised" by the participation switch. The projector weight of 1.53 on the 79-atom
reference cloud is the vacancy level (f = 1, s ≈ 1) plus the valence-edge leakage of the
sigmoid projectors as specified (s^e = 0.12 at the edge; Stage 0 decision 7): a third of the
normalised "electron" density is valence-edge density spread over the cell, which is what pulls
p from 0.03 (V_Cl⁰'s own fill, where the level sits deeper) to 0.15.

### Continuity on arma_s1 (§7.1)

On-site sweep of the Pb carrying the most carrier density (site 25, α = 0.32) on the charged
79-atom vacancy frame, λ ∈ [−3, +3] eV, 25 then 97 points; geometric path between the two
closest thermal charged frames of the pool (frames 3 and 4, largest atom displacement 0.69 Å),
9 then 33 points. Largest consecutive change at step h against h/4:

| quantity | on-site sweep: h, h/4, ratio | geometric path: h, h/4, ratio |
|---|---|---|
| E | 1.89e-1, 4.94e-2, 0.26 | 4.62e-1, 1.28e-1, 0.28 |
| Φ_FF | 1.38e-1, 4.27e-2, 0.31 | 1.61e-2, 4.07e-3, 0.25 |
| w (reference cloud) | 2.36e-1, 6.88e-2, 0.29 | 1.55e-2, 3.93e-3, 0.25 |
| band term | 1.86e-1, 4.65e-2, 0.25 | 3.06e-2, 8.24e-3, 0.27 |
| F (max component) | 6.26e-1, 1.90e-1, 0.30 | 2.20e-1, 5.71e-2, 0.26 |

Every ratio is at the smooth-curve value of ~0.25 (a jump would give 1). The sweep does cross
the windows: the reference cloud's projector weight runs from 1.42 to 2.37 and its w moves by
0.24 per coarse step at the steepest point. `q_F = 0` at every point; the class integers
(`n_e = (1, 0)`, `Q_core = +1`) are read from the table and cannot move. At the charged state
ρ_F = 0 (its w is a constant 3e-7), so the continuity of the term is that of Φ_F(0).

---

## Stage 1.3 — the SR/LR diagnostic (three Madelung arms)

### The switch

`madelung_range` on the model (`--defect_madelung_range`, checkpoint default `full`):

| arm | mode | what the head's on-site energies see |
|---|---|---|
| (a) | `full` | the whole lattice potential `−φ/ε∞` (as is) |
| (b) | `long_range` | `−(φ − V_SR(r_split))/ε∞`: the short-range part within `r_split = functional["r_split"]` (the first-block cutoff, 5.0 Å) removed — `V_SR` is the smeared Coulomb potential of the ions on the head's own neighbour list, LES's kernel `C q erf(r/σ√2)/r`, switched off by MACE's p = 6 polynomial at `r_split`, so `V_full = V_SR + V_LR` exactly (§7.3) and `V_LR` is as smooth as `V_full` |
| (c) | `off` | nothing; the module stays (the pristine formula for the class table, the static charges for Stage 4) |

Tests (`test_madelung_range.py`, 7): `V_SR` against a brute-force image sum on a 5-atom cell;
the identity read off the module with the head's edges; three modes = three Hamiltonians and
`off` an exact zero; the band and frontier forces pass the harness under `long_range`; the
config round trip; a head list shorter than `r_split` refused.

### Forward-only, on the Stage B cohort (six seeds, re-saved per arm with the class table)

STAGE13_FORWARD_PENDING

### Retrained, six seeds per arm (the Stage B recipe with `DEFECT_MADELUNG_RANGE`)

Recipe differences from Stage B, both forced by the current trainer: standing rule 2's null
gate (`--defect_null_reference aprime_nulls.json`) admits charged ENERGIES only for the size
class with a neutral null — the 159-atom class — so the 928 charged 79-atom energies Stage B
fitted at `w_E` are out of the loss (forces stay); and the class table is built on the
Harrison-initialised head before epoch 0 (decision 19 below).

STAGE13_RETRAIN_PENDING

---

## Stage 1.4 — the tiling ladder (§7.5), periodic gauge

`stage14_ladder.py`. The IDEAL pristine cell is the population mean of the stoichiometric
training frames of ONE domain: matched atom by atom (species, minimum image, one to one), the
544 pristine frames are not one crystal — matched to any one of them most others sit 1.5–4 Å
away (different runs, origins and tilt domains) — so the reference is the frame with the most
neighbours within 0.8 Å (frame 100, 23 members) and the ideal cell their mean cell and mean
matched fractional positions (residual 0.29 Å mean, 0.80 Å worst; cell 15.68 × 16.18 × 11.36 Å;
`golden/ideal_pristine_80.xyz`). The vacancy is the Cl the closest thermal 79-atom training
frame lacks (frame 793, matched to 0.67 Å mean), the same site at every size; 1×/2×/3× =
79/639/2159 atoms, L = 14.2/28.5/42.7 Å. §7.5's "single C_Q": the head's c table is
collapsed to its 1× column per charge class for the ladder (the 2× column is 0.81 eV higher for
the charged class, and with it the band term would carry a size-dependent constant). Each
tiling is its own class, counted against the ideal pristine cell tiled (Tier 1, `n_e = (1, 0)`,
`Q_core = +1` at 1× and 2×).

### The ladder on arma_s1 (`golden/stage14_ladder_arma_s1.json`; CPU, 32 threads; 3× class table 349 s, 3× forward with forces 40 s)

| size | N | L (Å) | E_base (eV) | band term | Φ_FF (w_ref) | E_PBC − E_∞ | correction under G_∞ | Σ F | N_eff |
|---|---|---|---|---|---|---|---|---|---|
| 1× | 79 | 14.2 | −275.673 | −4.5203 | +0.5908 (0.983) | +0.5908 | −4.5203 | 9e-16 | 3.7 |
| 2× | 639 | 28.5 | −2243.715 | −4.7204 | +0.2798 (0.992) | +0.2798 | −4.7204 | 2e-15 | 3.2 |
| 3× | 2159 | 42.7 | −7585.537 | −4.7779 | +0.1505 (0.993) | +0.1505 | −4.7779 | 4e-15 | 3.2 |

V_Cl⁺ (counter (0, 0, 1, 0)) in every row; the neutral vacancy (S = S_ref) has band term and
Φ_FF exactly zero at every size, as it must. Each tiling is its own Tier 1 class
(`n_e = (1, 0)`, `Q_core = +1`; the 3× class's spectrum shift +0.006, spread 0.005 eV).

- **Exponent of E_PBC − E_∞ (= Φ_FF before Stage 4):** log-log slope −1.23 over the three
  sizes; the `a/L + b` fit gives a = 9.27 eV·Å, b = −0.058 eV, residual 0.009 eV. A monopole
  (the reference electron cloud, q = −1, w_ref → 0.99) going as 1/L with a small offset — the
  bare image attraction of one localised electron, 0.59 eV at the training cell.
- **The band term moves with L**: −4.52 → −4.72 → −4.78 eV, fitted −4.91 + 5.54/L (residual
  0.003 eV). This is the correction under G_∞ too (Φ_FF is the only gauge-dependent term
  before Stage 4), so §7.5's "the correction converges after switching to G_∞" is **not yet
  met**: the Madelung shift inside H is the periodic potential of a lattice that is net
  charged once a Cl⁻ is removed, and its image part is what Stage 4's `V_static^B` in the
  isolated gauge removes (`− G_img ⋆ ρ_static^def`). Recorded as the Stage 1 baseline.
- **Forces within 6 Å of the vacancy** (21 atoms), against 3×: charged, periodic 0.50 (1×) →
  0.19 eV/Å (2×) → 0; isolated 0.32 → 0.12 → 0; neutral 0.035 (1×) → 1e-14 (2×): the neutral
  cell's forces converge at 2× (the trunk's and the head's neighbourhoods fit the box), the
  charged cell's carry the same slowly converging image terms as its energy. Total force
  ≤ 4e-15 eV/Å at every size.
- **Volume-scaled defect stress** (vacancy cell's virial minus the tiled pristine's, trace):
  neutral +1.52 / +1.54 / +1.54 eV — converged at 2×; charged −1.48 / −5.68 / −6.94 eV — the
  charged − neutral difference (−3.0, −7.2, −8.5 eV) converges geometrically towards ≈ −9 eV,
  the strain derivative of the frontier electron's on-site energy through the Madelung
  potential of the charged lattice (|c| ≈ 10 eV). Same Stage 4 item.
- **Thermal frame embedded** (train frame 793, matched to the ideal cell at 0.67 Å mean):
  Φ_FF +0.633 / +0.222 / +0.121 eV; the thermal-displacement image contribution
  (E_PBC − E_∞ thermal minus ideal) +0.042 / −0.058 / −0.030 eV — small, not monotone, of
  the order §2.1 expects (O(1/L³), random-signed); w_ref 0.988 / 0.969 / 0.977.
- **Switches** (Φ_FF over δ_p ∈ {½, 1, 2}×0.05 and Δ_s ∈ {½, 1, 2}×0.05, per size): 1×
  0.455–0.613, 2× 0.202–0.303, 3× 0.117–0.182 eV. Continuous in both (no jump between
  neighbouring settings larger than the change of w_ref it implies: doubling δ_p moves w_ref
  from 0.98 to 0.88 at 1×, halving Δ_s moves Φ by 3 %); a factor of two in either switch
  moves Φ_FF by up to 25 %, so §7.6's sensitivity is real and the defaults are model choices,
  as the plan says.

---

## Decisions of record (continuing Stage 0's numbering)

11. **Training can use the base cache** (Stage 1.1). The cached forward's head force is the
    exact derivative of the head energy (band 1.2e-6 eV/Å under the cache); what the cache cannot
    supply is the derivative of later-block features, which the head does not read
    (`spectral_first_shell`). The FD driver's `--base-cache` mode measures exactly this: analytic
    forces from the cached forward, numerical derivative of the full energy. Per-term `base` is a
    constant under the cache by construction (its force is the cached one, seen in
    `assembled_model`).

12. **Φ_FF is carried as Φ_F(Q) − Φ_F(0), with Φ_F(0) at the neutral fill of the SAME
    Hamiltonian** (`defect_frontier`). §2.2 writes `[Φ_F(Q) − Φ_F(0)]`; on a vacancy frame Φ_F(0)
    is not zero — V_Cl⁰ carries one frontier electron at Q = 0 (the plan's own benchmark) whose
    image energy is 0.4 eV. Both fills come from one diagonalisation (the head already builds the
    reference fills), so the subtraction costs one extra Ewald pass and no second head pass. At
    S = S_ref the difference is written as the exact zero it is, with no graph, which is what
    keeps neutral records bit-identical to v6 and the reference-branch skip an identity.

13. **`E_LR[q_host]` and the learned carrier branch are retired.** The host-charge readout is
    frozen at exactly zero on every model in hand (the harness read 0.0 for `lr_host` on every
    frame), §2.8 puts S–S "not in head (cancels in ΔE)", and the plan's functional has no learned
    charge cloud. The module is still constructed so checkpoints load; the forward never calls
    it; the flags are accepted and ignored. `lr_start_epoch`'s gate goes with it: Φ_FF is a term
    of the functional, not a staged branch.

14. **B_img's evaluator is the model's own Ewald at `σ = r_res`**, a second `LatentEwald` with
    the same `dl` and normalisation. The image term is then a property of the physical density
    (the Gaussian charges of width r_res that §2.1 defines) and invariant under `ewald_sigma` by
    construction, which §7.3 asserts; changing `r_res` is a model change (§7.6).

15. **The projector edges follow the head's rigid level shift.** The c table shifts every level
    of a charged graph by its charge class (§2.2's C_Q realised on the levels, from Stage A′).
    The class edges were aligned at S_ref, so a charged frame's spectrum sits ~2.7 eV above them;
    read against the unshifted edges, the "electron" channel of V_Cl⁺'s reference fill summed 132
    states (the whole valence band), p was 0.74, w was 1e-4 and Φ_FF was 1e-11 eV — the first
    arma FD run passed the frontier term at 1e-16 eV/Å for that reason. The head now records the
    shift relative to charge class 1 per graph and the term applies its projectors at
    `edges + shift`; the toy test asserts a +2.5 eV table entry leaves the term unchanged.

16. **The class table is required only off the reference state.** A frame at S_ref has an exact
    zero regardless of composition (decision 12), so a model without a table evaluates every
    reference-state batch; a charged frame on a table-less model raises with
    `ensure_class_table` named. An uncounted class (ambiguous at both tiers) has no counts and
    the forward refuses it — the plan's "no override input exists".

17. **A level merging into a band is logged, not acted on** (`LOG_BELOW = 0.5` on the channel's
    projector weight), and the normalisation is floored at 1e-12 so a synthetic spectrum with no
    weight in a window is a finite density with a warning rather than a division by zero. On a
    real spectrum the projector leakage keeps every needed channel's weight ≥ 1 (the 159-atom
    reference cloud reads exactly 1.000).

19. **Tier 2 requires identification, and spectral contiguity only across the gap.** On the
    Harrison-initialised head the 159-atom V_Cl class was Tier 1 ambiguous and Tier 2, read
    literally, refused it: the transported physical part was identified with class
    eigenvectors to overlap 1.000, but they spanned indices (0, 414) for a 412-dimensional
    manifold — two frontier levels resonant just below the top valence level. The plan says the
    integers are invariant under band resonance, and the literal clause would have left every
    159-atom charged frame with no frontier term (the training smoke crashed on it). The rule
    now: the physical part must be identified (overlap > 1 − η) and, when the identified set
    is not contiguous, every identified level must lie below the conduction-side cut
    `CBM_al − δ` — the valence manifold's vectors may sit in the gap (on the fresh head the
    159-atom class's top identified level is 0.2 eV above VBM_al, a valence-derived level the
    vacancy pushes up, with the two frontier levels resonant below it), and the integer is the
    identified manifold's dimension either way (412: `n_e = (1, 0)`, `Q_core = +1`, the same
    as the 79-atom class; an energy count would have given `Q_core = +3`); a valence vector
    that ended ACROSS the gap (the synthetic closure toy, level at +2.5 eV in the conduction
    manifold) is still refused. The record keeps `contiguous`, `identified`,
    `top_identified_level` and a note.

20. **The ladder's ideal cell is a domain mean, not a population mean** (above): the frames
    are several crystals. And the ladder collapses the c table to one constant per charge
    class — §7.5's single C_Q — which the production forward does not do.

21. **The class table's alignment follows the head; the integers do not.** The Stage B
    recipe's Harrison-initialised head has a 0.2 eV pristine gap (`loss_gap` opens it to
    2.4 eV over training), so a table built before epoch 0 — where the trainer built it —
    carried projector edges 2 eV stale for the whole run, and the 159-atom class was uncounted
    there (a valence-derived level at the conduction edge with two frontier levels below it,
    genuinely ambiguous on THAT head). `refresh_class_table` rebuilds the table on the current
    head at the start of every epoch: `VBM_al`, `CBM_al`, shift, spread, nearest level and the
    placement are taken fresh for every class; a class uncounted so far adopts the fresh count;
    a counted class keeps its integers, and a fresh count that disagrees is logged as a failed
    invariance and never adopted. The frontier term, in a TRAINING forward only, gives an exact
    zero for a graph of a still-uncounted class and warns once per (class, reason); an
    evaluation forward refuses it, as the plan says. `test_class_table_refresh.py` pins a
    rigid head shift moving the edges and not the integers, the adoption after the Harrison
    initialisation, and the training zero / evaluation refusal.

18. **`base_forces` on charged records move by ≤ 1e-6 eV/Å against the v6 golden.** Measured:
    the current `base_forces` equals the autograd of the trunk energy alone to 1.5e-15 eV/Å
    (qp1_159), so the movement is on the v6 side — its `F_total − F_correction` carried the
    cancellation roundoff of the retired Ewald branch's large opposing pieces. Neutral records
    (no E_LR contribution) are bit-identical. Three orders below the force RMSE the base loss sees.

---

## Not done / carried forward

- Stage 1.3 (the SR/LR diagnostic and its six-seed retrains), 1.4 (the tiling ladder) and 1.5
  (the reference selection by the §7.7 gates; the Stage 1 golden) follow.
- The golden of record is still the v6 capture (`--stage12` allow-list); the Stage 1 golden is
  captured at the selected reference (1.5).
- V_FF = δΦ_FF/δP does not enter H (Stage 5); the registry says `potential="absent"`.
- The projector leakage (a third of a 79-atom reference cloud's normalised density is
  valence-edge density) is the spec's own construction; its effect on w and on the ladder is
  measured in 1.4, not adjusted here.
