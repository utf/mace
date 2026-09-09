# D-SCC v5 — implementation tracker

Governing document: `DSCC_PLAN_V5.md` (user, 2026-09-09, verbatim). The v4 tracker
(`DSCC_V4_IMPLEMENTATION_SPEC.md`, D1–D15, C1–C12) is the record of everything before v5 and
stays read-only except for the F-SCC finals (Arm 4), which are written there. Units: per
component (meV/Å) everywhere in this file unless a value is marked [trainer] (per-atom vector
RMS = √3 × per component). Every rule below is registered before the corresponding result is
opened; "ours, provisional" marks a value or rule the plan left open and we set — the user may
overwrite it before the result is read, after which a change is labelled retrospective.

## W0 — registration and evaluation protocol (registered 2026-09-09, evening)

**W0.1 Shell centre.** The vacancy-side rule (`train.vacancy_centre`, commit `3a6a05c`; D15)
everywhere. Every held-out file written before `3a6a05c` — all 48 Arm-2+3 runs and the three
F-SCC finals (b3 tree at `be3c39b`) — is minimum-image and is compared only with old files.

**W0.2 Shells (C11 re-registered).** near = 2–4 Å with the flanking Pb and the first-shell Cl
(Cl within 3.5 Å of a flanking Pb) reported separately; mid = 4–8 Å (pooled by atom count);
far = 8–10, 10–12, > 12 Å. The 0–2 Å shell is not reported (empty under a correct centre for
charged frames; the neutral dimerised pair is a diagnostic). Far-field gate = the B′ gain on the
full per-atom RMSE and on the 2–4 and 4–8 Å shells against the same-coupling Route A arm, each
beyond the Φ = 0 seed spread of that quantity. Gate keys to become `("force_rmse", "2-4",
"4-8")` in `arm23.gates()`; `Trainer.evaluate` to write the 4–8 pooled shell and the two
near-field categories — code change queued before W3 (not done).

**W0.3 Statistics.** Per-component units; `tau_phys` = 1.7 meV/Å per component (the registered
3.0 [trainer] ÷ √3). Comparisons paired by seed: seeds {0, 1, 2, 3, 4, 6} → folds {0, 1, 2, 3,
0, 2}; the same seed means the same fold, the same held-out set and the same Arm-1 `H0`
initialisation, so the pair is the seed (n = 6). Two one-sided tests on the paired differences
`d_k = RMSE_A,k − RMSE_B,k` (ours, provisional: paired t, n − 1 = 5 degrees of freedom,
one-sided alpha = 0.05, so `t_0.95,5` = 2.015; SE = sd(d)/√6): **B superior** to A if the lower
one-sided 95 % bound of mean d is above 0 (mean d − 2.015 SE > 0); **equivalent** if both
one-sided bounds lie inside (−tau, +tau) (mean d + 2.015 SE < tau and mean d − 2.015 SE > −tau —
the TOST); **inconclusive** otherwise, the simpler arm chosen provisionally. Superiority and
equivalence can hold together (a real but sub-tau difference): superiority is reported, the
selection rule uses equivalence. With n = 6 and seed spreads of 1–3 per component the equivalence bound needs
SE ≲ 0.8, so "inconclusive" will be frequent — recorded, not hidden. Implementation
`mace/modules/dscc/stats.py::tost` with a test (queued, this turn).

**W0.4 Epoch-averaged evaluation.** The evaluation model is the parameter average over the last
10 epochs' checkpoints (ours, provisional: uniform average of the trainable parameters, taken
after each of the last 10 accepted epochs; the last-epoch model is also evaluated and both
numbers are written). Optimiser, learning rate, epochs unchanged. This needs a running average in
`Trainer.fit` — a W3 prerequisite, not done; every pre-v5 run has only its final checkpoint.

**W0.5 Kernel convention (regime B), verified 2026-09-09 (`scratchpad/w0_kernel_diag.py`).**
`E_PBC` includes the own-cloud self-interaction, G = 0 removed by the background;
`K_SR_ii = s(0) + Σ_{L≠0} s(|L|)`, `s(0) = 1/(√π r_g) − 2/(√π r_s)`, `K_LR_ii → 2/(√π r_s)`.
Naming: the W0 `r_s` is `KernelConfig.r_s` = 6.5 Å (the screening/smoothing width of the
regime-B split), NOT `MACEDSCC.r_split` = 2.5 Å (the Ewald real/reciprocal computational split);
`r_g` = 1.0 Å; the Coulomb prefactor 14.3996 eV·Å is inside `K`, `1/eps_inf` (= 1/4) is applied
in Γ. Measured on the static-cell tilings: `K_SR_ii` = 5.6595 (79 atoms, L = 14.3 Å, with the
image sum), 5.6244 (639), 5.6244 eV (2159 atoms) against `14.3996 × s(0)` = 5.6244 eV;
`K_LR_ii` = −0.2728 / 1.1310 / 1.5872 eV at L = 14.3 / 28.6 / 43.0 Å, and the `a + b/L` fit
through the two larger cells gives a = 2.4995 eV = `14.3996 × 2/(√π r_s)` (2.4995) to four
digits — 0.625 eV after `eps_inf`, the plan's "≈ 0.6 eV". Any W2 implementation (items 3, 4)
must reproduce these diagonals.

**W0.6 Extrapolation-uncertainty proxy (for W1 / W5).** In the charged d-window (flanking Pb–Pb
distance across the vacancy, 4.8–7.1 Å at 79 atoms, D15), the proxy per atom is the disagreement
between the MH foundation head (`omat_pbe`, frozen) and the fine-tuned head on the neutral-state
force at charged-frame geometries, `|F_ft − F_found|` per component, plus the cross-fit spread
(the per-atom std of the four out-of-fold bases' forces); per frame the energy proxy is
`|E_ft − E_found| / N` plus the cross-fit spread. Thresholds (ours, provisional, fixed the moment
base_v2 exists and BEFORE any charged-geometry proxy is opened): `u_F` = the 95th percentile of
the same force proxy on the neutral out-of-fold frames in the 2–4 Å band, `u_E` = the 95th
percentile of the energy proxy on the neutral out-of-fold frames; both recorded per size (79,
159). Recorded as a proxy, not a bound. `s_tol` and `z` for the W5 admission rule (in-range
`|s0| + z·SE ≤ s_tol`): z = 2 (ours, provisional); `s_tol` = the registered v4 shape margin
(`tau_noise_shape` of the Φ = 0 arms = 0.018 in the v4 units) unless the user sets one.
Localisation tolerance for W5 (ours, provisional): `N_eff` p50 within ±0.3 of the forces-only
twin, separation p50 within ±0.1 eV, precondition fraction within ±2 %.

**W0.7 F-SCC finals.** Old base, minimum-image centre: written to the v4 tracker (P3.3) as a
separate record; not compared with any v5 number.

## W1 — base v2: multi-head fine-tune of MACE-MH-1 (neutral data only)

*Decision trail (user, 2026-09-09 evening): W1 was outlined on MH-1; "looks too heavy, let's do
MACE-MPA-0"; after the measurement below, "since MH-1 is smaller and faster, let's stick with
that".* Measured on the A4000 (ASE calculator path, energy + forces per frame, median):
MACE-MH-1 6.44 M parameters, float64 121 ms (79 atoms) / 223 ms (159), float32 72 / 100 ms;
MACE-MPA-0 9.06 M, float64 160 / 313 ms, float32 56 / 46 ms. MH-1's 512-channel node features
cost activation memory in training, not inference time.

**W1.1 Registered before training (2026-09-09).** MACE 0.3.17, worktree `b37b298` (b3 tree frozen
at `be3c39b`; `run_train.py`, `multihead_tools.py`, `scripts_utils.py`, `fine_tuning_select.py`
and `arg_parser.py` identical between the two — checked with `git diff --stat`). Foundation
checkpoint `~/.cache/mace/macemh1model` (mace-foundations release `mace_mh_1`, 59 MB, sha256
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`): `ScaleShiftMACE`, r_max
6.0 Å, two `RealAgnosticResidualNonLinearInteractionBlock`s, node features 512x0e → hidden
512x0e+512x1o (block 0) → 512x0e (block 1), edge attrs up to l = 3, 10 Bessel edge features,
Agnesi transform, pair repulsion, 89 elements, six heads {matpes_r2scan, mp_pbe_refit_add,
spice_wB97M, oc20_usemppbe, omol, omat_pbe}. Foundation head: `omat_pbe` (the release notes'
general inorganic head; initial errors on the neutral validation set with the foundation E0s —
`omat_pbe`: force RMSE 43.9 per component, energy offset +0.004 eV/atom mean, per-species
offsets Cl +0.51 / Cs −0.75 / Pb −0.75 eV, residual after the per-species fit 1.9 meV/atom;
`mp_pbe_refit_add`: 43.1, −0.036 eV/atom, +0.51 / −0.85 / −0.85, 1.8; `matpes_r2scan`: 55.5,
−18.2 eV/atom — excluded). E0s of the fine-tuning head: `foundation` (the guide: compute your own
E0s or use the foundation's for MP-compatible DFT; no new DFT is allowed, the labels are PBE, and
the initial per-atom error is 0.004 eV/atom against the guide's 0.4 eV/atom limit; the
per-species constant offsets are absorbed by the species-dependent readout; `average` is refused
by the code in multihead mode). Replay: `--pt_train_file omat` (MPtraj structures with
OMat-compatible labels, the set matching the `omat_pbe` head; 480 MB, cached on both machines),
`--num_samples_pt 10000` (the code default and the guide's examples; the guide's "30000" trips
the 0.1 real-data ratio and duplicates the fine-tuning set — not used), `--subselect_pt fps`
(guide), `--filter_type_pt combinations` (guide), `--atomic_numbers "[17, 55, 82]"`. Learning
rate 1e-4 and EMA (decay 0.99999) set by the code in multihead mode (`force_mh_ft_lr` False);
energy weight 1.0, forces weight 100.0 (guide); `--max_num_epochs 30` (the upper end of the
guide's 10–30) with the best-validation checkpoint; batch size 8 (the old base); `default_dtype`
float64 (ours: matches the head, which converts the base to float64; the old base trained in
float32); no SWA; seed 1 for the production base and the four out-of-fold bases (the old bases
used seed 1); `--E0s foundation`, keys `REF_energy / REF_forces / REF_stress`, `compute_stress`
default (off). Data: production base `dataset_e0/train.xyz` (1616 neutral: 1057 × 79, 544 × 80,
15 × 159) / `valid.xyz` (154); out-of-fold bases `dataset_cf/fold{k}/train.xyz` (1379–1380) /
`valid.xyz` (153) — the same files as the old bases (md5 checked on both machines); no charged
frame in any of them. Command (one line per run, through `b3_run.sh` on GPUs 4/5 by UUID):
`python -c "import sys; sys.path.insert(0, W); from mace.cli.run_train import main; main()"` with
the arguments of `scratchpad/w1_smoke.sh` at `--max_num_epochs 30 --num_samples_pt 10000
--subselect_pt fps` and `--name base_v2_{prod,f0,f1,f2,f3}`. **Smoke (2026-09-09, `w1_smoke_b3`
on b3 GPU 4; one epoch, 200 replay samples, otherwise the registered command):** the local A4000
(16 GB) OOMs at batch 8 in float64 (12.5 GB allocated at the first step), so W1 runs on b3 only;
peak 17.2 GB on a 24 GB card → ONE run per GPU. The `omat` replay filtered by `combinations` on
{Cl, Cs, Pb} holds 41 configurations; the remainder is random padding (`allow_random_padding_pt`,
the code default) — with 10000 samples the replay is 41 relevant + 9959 random MP structures,
registered as such. Initial errors with the foundation E0s: Default head 5.53 meV/atom,
43.86 meV/Å (per component, the MACE table), stress 0.11 meV/Å³; pt_head 99.5 meV/atom, 79.1.
After one epoch at lr 1e-4: Default 3.61 meV/atom, 13.00 meV/Å (the current base: 4.9 / 11.8
after 140 epochs); pt_head 96.3 / 79.4. Epoch time 6.9 min for 1796 frames (≈ 1.8 s per step of
8) → ≈ 35–45 min per epoch at 10000 replay samples, 18–22 h per 30-epoch run. The saved model
(`w1_smoke_b3.model`, float64, `--save_cpu`) carries heads ['pt_head', 'Default'] and 76 elements
(the replay's) — the head selects 'Default' (`remove_pt_head`) and prunes to {Cl, Cs, Pb}
(W1.2). **Launched 2026-09-09 ~17:52 (b3 clock):** `~/runs/w1_queue.sh` launches
`~/runs/w1_queue.txt` (base_v2_prod, base_v2_f0, f1, f2, f3; `~/runs/w1_run.sh NAME TRAIN VALID`
= the smoke command at 30 epochs / 10000 fps-selected replay samples) one per free GPU among 4, 5
and 7 (GPU 7 frees when the F-SCC comparators finish, ≈ 05:00 on 10 Sep), never GPUs 0–3, by
UUID through `b3_run.sh`; logs `~/runs/base_v2_*.log`, queue log `~/runs/w1_queue.log`. The b3
tree stays at `be3c39b` (run_train identical to HEAD). Expected completion: the first two runs
≈ 14:00 on 10 Sep, all five ≈ 10 Sep late evening.

**W1.2 Feature export for the head.** Block-0 features of MH-1 are 512x0e+512x1o (n_scalars 512,
n_vectors 512, from `products[0]`); `MACEDSCC.first_block` / `features` slice them; the head's
feature-modulation readouts and the rank-1 descriptor are re-dimensioned by construction from
`n_scalars`. The fine-tuned base keeps every element of its fine-tuning and replay data (the
smoke's model table ran to dozens of elements through the replay's random padding), and the
head's host-free per-species tables (covalent radii, `U_max`, valence) raise on elements they do
not cover — so the base is PRUNED to {Cl, Cs, Pb} before the head wraps it
(`mace.modules.dscc.prune.prune_elements`: slices the node embedding and the interaction blocks'
source/target embeddings with e3nn's `1/sqrt(mul_in)` path normalisation restored, the atomic
energies and the element buffer; the MH-1 symmetric contractions are element-agnostic and the
radial embedding / ZBL index full tables by Z; predictions on kept-element frames unchanged to
1e-10, `test_prune.py`). Wrap test on the pruned MH-1 (`scratchpad/w1_wrap_test.py`, 2026-09-09):
`MACEDSCC` constructs with `n_scalars` 512, `n_vectors` 512, r_max 6.0, three species;
`first_block` equals the full forward's block-0 slice to 9e-16 (79 atoms × 2048 features); the
pristine centre sets on 16 frames; a coupled B′ forward with forces on a real charged 79-atom
frame converges (22 iterations) at a 3.2 GB peak. The W3 head on base_v2 therefore needs no
re-dimensioning code; only the cost of the 512-wide features (W2 item 1).

**W1.3 Gates (unchanged).** Validation force / energy RMSE ≤ the current base (11.8 meV/Å per
component, 4.9 meV/atom); out-of-fold neutral floors by shell (vacancy-side centre) at 79 and
159; `s0(L) ± SE`, coverage and proxy tables per size; per-size, per-fold admission recorded
before W5 opens.

## W2 — efficiency

**Baseline profile (2026-09-09, `scratchpad/w2_profile.py`; B′ LR-only s3, eleven held-out 79-atom
charged frames after a five-frame warm-up, cold SCF start, A4000, float64, synchronised timers on
the entry points):** 637 ms per frame with forces at 12–14 SCF iterations. `torch.linalg.eigh`
182 ms (18.3 calls per frame — about 1.5 diagonalisations per SCF iteration plus the reference
fills), `autograd.grad` 72 ms (4 calls: the merged base + Hellmann–Feynman backward and the
pair-route terms), the Ewald terms 96 ms (`kernel_pair_gradients` 52, `kernel_components` 20,
`gamma_lr_pair_gradient` 18, `gamma_lr` 7), the base forward 24 ms (block 0 + block 1 on the
r_max graph; the base alone with forces is 50 ms), `fill` 7 ms; the remaining ≈ 250 ms is the SCF
loop's CPU-side work (the torch profiler on one frame: self CPU 776 ms against self CUDA 383 ms;
`_DensityMatrix` 70 calls, 3800 small `div`s). A first-sight transient of 130–340 ms on the first
four or five distinct graph shapes (allocator growth, not compilation: later new shapes cost
23 ms) inflates any short benchmark and must be excluded by a warm-up. The registered 2×
benchmark (warm-started trajectory, median 4 SCF iterations) read 305 ms per frame at 79 atoms
(D14). Order of attack by measured cost: the diagonalisation count and the SCF loop's CPU
overhead (items 2, 5, 7), the Ewald terms (item 3), the backward (items 1, 6); item 1's inference
half (one merged backward) is already in place since P4.3, so item 1 is about the training path
and the 512-wide MH-1 features.

**Item 3 — direct reciprocal `K_LR` and `Γ_LR` (done 2026-09-09; `ewald.reciprocal_matrix`,
`ewald.reciprocal_pair_gradient`, `kernels.lr_route`, `KernelConfig.lr_route` = 'reciprocal'
(default, also for models pickled before v5) or 'ewald' (the v4 route, kept for the gate);
`tests/extensions/dscc/test_reciprocal.py`).** The periodic kernel of the broad Gaussian
(pair width `r_s` = 6.5 Å) is evaluated in reciprocal space alone, `C[(4π/V) Σ_{G≠0}
e^{−G² r_s²/4}/G² cos(G·r_ij) − π r_s²/V]`, the diagonal included; `Γ_LR` likewise with the
combined width `sqrt(2 (r_g² + r_split²))` = 3.81 Å; the pair derivatives `D_LR`, `dΓ_LR/dr`
analytically through the structure factors (three matrix products per component, no backward);
stress by autograd through `G(h)`, `V(h)`, `c_G(h)`. With `lambda_dir` held at zero (LR-only,
LR + U) `K_SR` and `D_SR` are skipped (`need_sr=False`; `K_SR` never enters `Γ` there). **Unit
tests:** reciprocal = Ewald route to 1e-12 on a triclinic 7-atom cell (both widths, diagonal
included), analytic pair gradient = autograd pair gradient to 1e-10 (and = the Ewald route's),
cell derivative = finite differences to 1e-6 relative, cutoff convergence (production tol 1e-16
against 1e-20: energy < 1e-10 eV, pair forces < 1e-8 eV/Å — the registered criterion; the
1e-10 cutoff sits < 1e-6 eV / 1e-5 eV/Å from converged). On the 79-atom static cell the
direct formula agrees with the v4 `K_LR` to 3e-14 (1.1e-13 at 319 atoms) in 2–3 ms against
70–560 ms. **Agreement gate (`scratchpad/w2_gate_item3.py`, `~/runs/dscc/w2_item3_gate.json`;
the registered frame set — five fold-3 held-out 79-atom charged frames 1622 / 1631 / 1659 /
1670 / 1672, two fold-3 159-atom held-out frames, the 2×2×2 static tiling (energy only, on the
CPU: the v4 lattice sums exceed 16 GB there); models B′ LR-only s3, Route A full s0, LR + U s1):**
|ΔE| ≤ 3.6e-12 eV, |ΔF| ≤ 1.7e-14 eV/Å, |Δσ| ≤ 5.0e-17 eV/Å³ — the gate (1e-9 / 1e-7 / 1e-7)
passed by three orders. **Phase-1 gates:** the D-SCC suite, 140 tests, passes on the new default.
**Measured speed (per frame with forces, cold start, first-sight transients included):** B′
LR-only 79 atoms 705–1155 ms → 527–600 ms, 159 atoms 1318–1440 → 1071–1172; full / LR + U (K_SR
still needed) 545–611 → 498–574; 639 atoms energy-only on the CPU 24.8 → 20.6 s. The Ewald
terms' 96 ms of the baseline profile fall to ≈ 5 ms on LR-only; the remaining cost is the SCF
loop (items 2, 5, 7).

**Convention flag (C13, for the user).** The outline's item-3 formula carries the constant
`−π (r_s² − 4 r_g²)/Ω`; the code's `E_PBC` (all v4 results) carries the physical
Gaussian-background convention, under which `K_LR = E_PBC − K_SR` has `−π r_s²/Ω` — the two
differ by the uniform `4π r_g² C/Ω` (0.0616 eV per pair at 79 atoms, 0.0154 at 319), i.e. the
item-4 `κ` differs by that amount and `E(+1) − E(0)` by `½ · 4π r_g² C/(eps_inf Ω) Q²` = 7.7 meV
at 79 atoms (a 1/Ω term). Item 3 keeps the code's convention (verified to 1e-13 against the v4
route and against W0.5's diagonal); the outline's constant is adopted only if the user rules
that the Gaussian-width background term is to be dropped.

## W3–W6 — not opened.
