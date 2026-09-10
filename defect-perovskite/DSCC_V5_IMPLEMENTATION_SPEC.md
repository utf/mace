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

**W0.6a Proxy definition closed (registered 2026-09-09 23:05, before any fold model was
opened).** Two holes in W0.6 as first written, closed here so the thresholds can be computed
before the charged window is read.

1. *The two terms and how they combine.* The per-atom force proxy is the SUM of the two terms
   the outline names, `u_F(atom) = |F_ft − F_found| + sd_folds(F)`, per component, and the two
   terms are ALSO written separately. `F_ft` is the out-of-fold base for that frame (the frame's
   own fold); `F_found` is the frozen foundation `omat_pbe` head of the original
   `~/.cache/mace/macemh1model`, never the fine-tuned model's drifted `pt_head` (W1.1 replay
   trajectory). The energy proxy is `u_E(frame) = |E_ft − E_found_aligned| / N + sd_folds(E)/N`.
2. *The cross-fit spread on neutral frames is biased low, and the bias is recorded, not
   corrected.* A neutral frame is out-of-fold for exactly one of the four bases; the other three
   trained on it, so `sd_folds` over the four is smaller there than on a charged geometry, where
   all four are out-of-fold. The consequence is a threshold that is too TIGHT, i.e. charged frames
   are flagged out-of-range more readily — the safe direction for an uncertainty flag. Recorded as
   a known bias of the proxy; no correction factor is applied (ours, provisional).
3. *Energy alignment (new).* The foundation head and the fine-tuned head carry different
   isolated-atom references (estimated E0s vs the foundation's), so the raw `E_ft − E_found` is
   dominated by a per-composition constant and the 79 / 80 / 159-atom compositions differ. Before
   the proxy is formed, the foundation energies are aligned by ordinary least squares on the
   neutral out-of-fold frames against the three species counts,
   `E_found_aligned = E_found + Σ_s n_s a_s` with `s ∈ {Cl, Cs, Pb}`; the fitted `a_s` are
   recorded, and the alignment costs three degrees of freedom on the same frames that set `u_E`
   (recorded, not cross-fitted; ours, provisional).
4. *Isolated-atom frames are excluded* from every out-of-fold evaluation and from both
   thresholds (the weight-1000 `IsolatedAtom` rows of `train_iso.xyz`; the 437 meV Cs miss would
   otherwise dominate).
5. *Order of operations.* Stage 1 computes `u_F`, `u_E` per size (79, 159) on neutral out-of-fold
   frames and WRITES them to `~/runs/w1_proxy_thresholds.json`; only then does stage 2 open the
   charged d-window. The thresholds are the 95th percentiles named in W0.6 (`u_F` in the 2–4 Å
   band, `u_E` over frames).

**W0.2 / W0.4 implemented (2026-09-09 23:20, commit `722f17a`; suite 155 passed).** The three
code items queued before W3 are done, on the local tree only (b3 stays frozen at `be3c39b`).
- `Trainer.evaluate` writes the 4–8 Å pooled shell as a first-class `shell_rmse` key and splits
  the 2–4 Å shell into `2-4:pb_flank`, `2-4:cl_first` (Cl within 3.5 Å of a flanking Pb) and
  `2-4:other`, each with its atom count; the unrestricted chemical sets are written as
  `near_rmse_all_radii` (information), so the windowing choice is auditable. `vacancy_centre_full`
  returns `(radii, flanking pair)`; `vacancy_centre` keeps its old signature.
- `arm23.gates(..., spec=)`: `spec="v4"` reproduces the recorded campaign reading (decisive
  `force_rmse`, `0-2`, `2-4`, gate = ANY beyond noise, 4–8 reported) so no v4 number moves;
  `spec="v5"` is W0.2 — decisive `("force_rmse", "2-4", "4-8")`, `0-2` demoted to information.
  *Reading choice, ours, registered here before any W3 result exists:* W0.2's "each beyond the
  Φ = 0 seed spread" is ambiguous between "each compared against its own noise" and "all three
  must clear it"; we take the strict reading, ALL three, and report `..._any` and `..._all`
  side by side so the other reading is always visible. The v4 `any` is not changed retroactively.
  The pooled 4–8 reading is taken from the trainer's new `shells["4-8"]` when present, falling
  back to the pre-v5 `far_field_4_8` field.
- `Trainer.fit` keeps a float64 uniform running average of the TRAINABLE parameters over the
  last `TrainConfig.avg_window` (= 10) epochs, and after the last-epoch `held_final` reading —
  in that order, so the warm-start store behind the last-epoch numbers is not overwritten first —
  evaluates it as `held_final_avg`, writes `held_final_avg.json` and `model_avg.pt`, and restores
  the last-epoch parameters. Every pre-v5 run has no `avg` reading; readers fall back to the
  last-epoch numbers. Helpers `average_into` / `load_average` / `restore_parameters` are tested
  directly.
- `dscc_arm23_report.py` (commit `e8e0d1e`) reads the averaged evaluation where a run wrote one
  and falls back to the last epoch otherwise, recording which reading each run contributed
  (`--reading avg|last`); it takes the pooled 4–8 Å value from the trainer's own key when present
  and only then falls back to pooling 4–6 and 6–8 by atom count; `--spec v5` additionally writes
  the v5 far-field gate table alongside the v4 decision, which is left untouched. This was the
  last code prerequisite for W3.

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
tree stays at `be3c39b` (run_train identical to HEAD). **Restart 18:18 (b3 clock) in float32:** `base_v2_prod` died at
22.9 GB (CUDA OOM in a backward on a large replay structure, 25 min in, before its first epoch
completed); `f0` sat at 23.3 GB and `f1` had just been launched onto the freed GPU by the
runner — both stopped, the three directories moved aside (`*_f64_oom_*`). Registered deviation
(ours, with the reason): `--default_dtype float32` — the dtype every MACE foundation model and
the old base were trained in, halving the activation memory; the head still converts the base
to float64 at load. All five runs requeued from scratch under the same script otherwise; the
float64 smoke stands as the smoke. The first production epoch (10616 frames) had not completed
in 25 min at float64, so the user's 6.9-min-per-epoch estimate (the smoke's 1796 frames) does not
transfer; the float32 epoch time: **7.5 min per epoch** (10616 frames; `base_v2_prod` epoch 0 at 18:28:30,
training started 18:21:02) at 12–16 GB, so ≈ 3.8 h per 30-epoch run and all five by ≈ 04:00 on
10 Sep on two GPUs (three once the F-SCC runs finish). Epoch 0 (float32, lr 1e-4): Default head
4.32 meV/atom, 13.56 meV/Å, stress 0.22 meV/Å³.

**W1.1 revised — v2 registration (user, 2026-09-09 ~20:20: "You are using out of date settings";
the MACE fine-tuning-guidance page, `guide/finetuning_guidance.html`, supersedes the multihead page's
1 : 100 + SWA recipe and the Q&A's 1 : 10).** The guidance, as relayed and as fetched: constant
target weights throughout, e.g. `--energy_weight=10 --forces_weight=10`, no SWA / stage-two
schedule for multihead fine-tuning (the force-then-energy transition destabilises a model that
already has coherent forces); for multihead / pseudolabel replay: learning rate 1e-4, EMA decay
0.9999, grad clip 1.0, weight decay 0 ("weight decay pulls parameters toward zero, i.e. away from
the pretrained solution"); E0s: explicit isolated-atom DFT (spin-polarised, asymmetric box) if
available, otherwise `--E0s="estimated"` ("recommended default when explicit isolated-atom
energies are unavailable"), never "average"; the MH-1 warning verbatim: "Their isolated-atom
reference is shared between the explicit E0s and learnable bias terms, so setting the E0s
correctly does not by itself guarantee accurate isolated-atom energies. If your application
depends on absolute atomic energies (e.g. defect formation energies), also include true
isolated-atom configurations as explicit training points"; an initial error > 500 meV/atom is
the E0-mismatch symptom (ours was 4 meV/atom); replay 10k–30k with `fps` + `combinations`; the
MH-1 release ships the head's own replay set `replay-data-mh-1-omat-pbe.xyz` (500 MB). The runs
of 18:18 (energy 1 / forces 100 from `run_train`'s defaults and the multihead page's example,
EMA 0.99999 set by the code, weight decay 5e-7, `foundation` E0s, the MPtraj-OMat replay) are
stopped at epoch ≈ 18 and recorded as out of date: their energy RMSE per atom rose monotonically
(prod 4.32 → 6.06, fold 0 3.30 → 6.15 by epoch 15) while forces fell (13.56 → 10.67; 6.15 →
4.14), the signature of the 1 : 100 weighting against the old base's 10 : 100.

*v2 settings (registered before launch):* `--energy_weight 10 --forces_weight 10` (stress at the
universal loss default 1.0), `--force_mh_ft_lr True --lr 0.0001 --ema --ema_decay 0.9999`
(the code's automatic override would set 0.99999), `--weight_decay 0.0 --clip_grad 1.0`, no
SWA, `--max_num_epochs 30`, batch 8, float32 (the memory deviation stands), `--enable_cueq
True` (cuEquivariance 0.11.1 on both machines), replay `replay-data-mh-1-omat-pbe.xyz` with
`--num_samples_pt 10000 --subselect_pt fps --filter_type_pt combinations`, seed 1. *E0s:* no
DFT isolated atoms exist and none may be run, so `estimated` is the registered route; MACE's own
estimator (`data.utils.estimate_e0s_from_foundation`: the foundation head's E0s corrected by a
least-squares fit of its residuals on the training set) gives, on the production set, Cl
−0.7159, Cs +0.5417, Pb −0.1022 eV (foundation omat_pbe: −0.2574, −0.1297, −0.7736; folds 0–3
within 0.02 of these: `~/runs/w1_e0s_estimated.json`), with the warning "system is rank
deficient (rank 2/3)": Cs and Pb counts are always equal in this data, so only E0(Cl) and
E0(Cs) + E0(Pb) are determined and the Cs/Pb split is the minimum-norm convention — harmless for
any Cs:Pb-stoichiometric comparison (the Cl-vacancy formation energy depends on E0(Cl) and the
Cl reference only), recorded. Per the MH-1 warning, the three isolated-atom configurations
(one atom in a 20 × 21 × 22 Å box, `config_type IsolatedAtom`, `REF_energy` = the estimated E0,
zero forces, `config_weight 1000` as in the paper's Table 8) are appended to every training file
(`train_iso.xyz` beside each `train.xyz`) and kept as training points (`--keep_isolated_atoms
True`); MACE reads the Default head's E0s from them. These are "fake" isolated atoms labelled
with estimated, not DFT, energies — recorded as such.

*v2 smoke and launch (2026-09-09).* A custom `--pt_train_file` path is loaded WHOLE in this MACE
version (`--num_samples_pt` / `--subselect_pt` / `--filter_type_pt` act only on the built-in
shortcuts): the first smoke pulled all 372k replay configurations. The docs' Method 1 applies:
`mace.cli.fine_tuning_select` on the MH-1 omat_pbe replay file with `--filtering_type
combinations --subselect fps --num_samples 10000` (head_pt `omat_pbe`) — 107 Cl/Cs/Pb-combination
structures exist in it, the rest is random padding — writes
`~/.cache/mace/replay_mh1_omat_pbe_sel10k.xyz` (10000 frames, 1–148 atoms, `head=omat_pbe`, the
`REF_*` keys), which is the registered replay set on both machines. The replay's labels are the
foundation head's own predictions (its initial error on the replay is exactly zero): a
pseudolabel replay. Local smoke (A4000, float32, cueq, one epoch on `train_iso.xyz` with the
selected replay): E0s read from the isolated-atom frames, `UniversalLoss(10, 10, 1)`, lr 1e-4,
weight decay 0, clip 1.0; initial Default-head error 1.69 meV/atom (5.53 under the foundation
E0s), 43.9 meV/Å; after one epoch **0.90 meV/atom, 13.86 meV/Å**, stress 0.20 (v1's epoch 0:
4.32 / 13.56); 4.3 min per epoch with cuEquivariance (7.0 without, on b3); the model saves after
conversion back to e3nn. **Restart on b3 at 20:44 (b3 clock):** the v1 runs stopped at epoch ≈ 18
(directories `*_v1_1to100`), all five requeued on `train_iso.xyz` through `w1_run.sh` (= v2),
`base_v2_prod` on GPU 4, the others as GPUs 5 and 7 free. Expected ≈ 2.5 h per run, all five by
≈ 04:00 on 10 Sep.

**W1 v2 results, first two runs (2026-09-09 22:40–22:42 b3 clock; 30 epochs, best = last, ≈ 2 h
each).** `base_v2_prod` (validation, the same 154 frames the old base's gate numbers were read on):
energy **1.16 meV/atom**, forces **10.23 meV/Å** per component, stress 0.16 meV/Å³ — the old base
4.9 / 11.8: **both W1.3 validation gates pass.** By size: 79 atoms 1.05 meV/atom (mean offset −0.66),
80 atoms 1.20 (+1.06), 159 atoms 2.00 (−1.69; two frames). Trajectory (epochs 0 / 4 / 9 / 14 / 19 /
24 / 29): energy 0.89 / 1.03 / 1.26 / 1.22 / 1.19 / 1.19 / 1.16 meV/atom, forces 13.82 / 11.97 /
11.23 / 10.92 / 10.64 / 10.41 / 10.23 meV/Å, stress 0.24 / 0.14 / 0.19 / 0.17 / 0.16 / 0.17 / 0.16,
loss 6.1e-5 → 3.4e-5 — forces still falling at epoch 29, energy flat since epoch 9 at the
10 : 10 weighting (v1 at 1 : 100 had reached 6.06 / 10.67 by epoch 15 with the energy rising).
`base_v2_f0` (its fold's validation split): 0.43 meV/atom, 3.88 meV/Å, stress 0.11 (epochs 0 → 29:
0.76 → 0.43, 6.27 → 3.88). **Isolated-atom check on `base_v2_prod`** (the MH-1 caveat): the
predicted isolated-atom energies against their weight-1000 labels are Cl −0.754 vs −0.716
(−38 meV), Pb −0.115 vs −0.102 (−13 meV), **Cs +0.105 vs +0.542 (−437 meV)** — the learnable bias
did not pin the Cs reference even at weight 1000 (this is the source of the 11–13 meV/atom
train-set energy RMSE in the final error table; the bulk fit is 1.2 meV/atom). Because the data
constrain only E0(Cl) and E0(Cs) + E0(Pb), the Cs/Pb split of the labels was itself the
minimum-norm convention; the Cl reference (the one a Cl-vacancy formation energy uses) is
reproduced to 38 meV. Recorded: absolute formation energies referenced to isolated Cs are not
supported by this base; Cs:Pb-stoichiometric comparisons are. Remaining: `f1` (GPU 5, 22:41),
`f2`, `f3` as GPUs free (GPU 7 after the F-SCC finals); W1.3's out-of-fold floors by shell and
the coverage / proxy tables wait for all four folds.

**Replay (pt_head) validation trajectory, read 2026-09-09 23:00 after the user's question.** The
replay head's validation error against the MH-1 pseudolabels (initial error 0) reaches its minimum
around epoch 8–11 and then RISES for the rest of the run at the constant lr 1e-4. `base_v2_prod`
(epochs 0 / 5 / 8 / 14 / 20 / 26 / 29): energy 5.77 / 3.61 / 3.64 / 3.47 / 4.17 / 3.92 / 4.05
meV/atom, forces 17.42 / 14.67 / 14.15 / 14.62 / 15.68 / 16.38 / 17.11 meV/Å, stress 1.35 / 1.11 /
0.98 / 1.05 / 1.06 / 1.11 / 1.17 meV/Å³; `base_v2_f0`: energy 5.75 → 3.16 (epoch 11) → 3.70,
forces 18.51 → 14.38 (epoch 11) → 16.49, stress 1.39 → 0.87 → 1.04. At epoch 29 the replay forces
are back at their epoch-0 level (the representation has drifted from MH-1 by ≈ 3 meV/Å on the
replay frames, ≈ 20 % over the minimum). Over the same epochs the Default head was still improving,
but slowly: prod forces 10.57 → 10.23 meV/Å over epochs 20 → 29 (≈ 0.04 meV/Å per epoch, energy
flat at 1.16–1.19); f0 4.07 → 3.88. The trainer selects the checkpoint on the LAST head only
(`mace/tools/train.py:213`, "consider only the last head for the checkpoint"), i.e. on Default, so
the saved model is epoch 29 and the replay rise does not enter the selection. Reading: the run is
past the point where the replay constrains it; continuing at constant lr 1e-4 buys ≈ 0.3 meV/Å of
Default forces per 10 epochs at the price of further replay drift. Registered protocol (30 epochs,
best = last) is kept for the four folds and prod; a decay tail (lr 1e-4 → 1e-5 over 10 epochs from
the epoch-29 checkpoint) is the candidate if the replay drift is to be reclaimed — a protocol
change, to be registered before use, and applied to all five runs or none. User decision pending.

**W1 v2 complete: all five bases (2026-09-10 01:02 b3 clock).** Thirty epochs each, best = last,
float32, 10 : 10 weights, lr 1e-4, EMA 0.9999, cuEquivariance. Validation at epoch 29, per
component:

| run | split | energy meV/atom | forces meV/Å | stress meV/Å³ | replay forces meV/Å |
|---|---|---|---|---|---|
| prod | the 154-frame gate split | 1.16 | 10.23 | 0.16 | 17.11 |
| f0 | fold 0 valid (153 pristine) | 0.43 | 3.88 | 0.11 | 16.49 |
| f1 | fold 1 valid | 0.55 | 3.88 | 0.12 | 16.45 |
| f2 | fold 2 valid | 0.49 | 3.89 | 0.12 | 16.70 |
| f3 | fold 3 valid | 0.47 | 3.89 | 0.12 | 17.32 |

The four cross-fit bases agree to 0.01 meV/Å on forces and 0.12 meV/atom on energy: the fold
split contributes almost nothing to the base, which is the condition the cross-fit spread term of
the W0.6 proxy relies on. Every run shows the same replay drift (minimum near epoch 8–11, back to
its epoch-0 level by 29). prod's larger numbers are its split, not its quality: it is read on the
mixed 154-frame set the old base's gate numbers came from, while each fold is read on 153
pristine 80-atom frames. Wall clock: 20:49 to 01:02 on three GPUs, the last fold co-located.

**Co-location measurement (b3, Quadro RTX 6000, 24 GB).** One fine-tune holds 4.4 GB and 12–19 %
of a card. Two on one card cost about 5 % per epoch (3.67 → 3.8–3.9 min) and double throughput;
fold 3 was launched alongside fold 1 on GPU 5 at 22:59 rather than waiting for a free card, which
moved the last fold from ≈ 03:30 to 01:02. Three per card is the expected comfortable ceiling.

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

**W1.3 procedure registered (2026-09-09 23:30, before any fold model beyond f0 was opened).**
Two scripts, run locally on the A4000; b3 stays frozen at `be3c39b`.
- *Data.* Each fold holds out 451 neutral frames: `valid.xyz` (153 pristine, 80 atoms — the
  split the training log's Default-head numbers were read on) and `null_oof.xyz` (298 neutral
  vacancy frames, 293 at 79 atoms and 5 at 159), which no base of that fold ever saw. Across the
  four folds every neutral frame is out-of-fold exactly once. The charged set is `eval_qp1.xyz`
  (1047 frames, 1030 at 79 and 17 at 159); no base saw any of it.
- *Stage 1* (`scratchpad/w13_eval.py`): per out-of-fold neutral frame, the fold's own base gives
  the force error per component by shell (`2-4`, `4-8`, `8-10`, `10-12`, `12+`, vacancy-side
  centre) with the 2–4 Å split into flanking Pb / first-shell Cl / other, the energy residual,
  the collective coordinate `d`, and the two proxy terms — the frozen foundation (`omat_pbe`)
  disagreement and the cross-fit spread over the four fold bases. A per-atom force quantity is
  always the RMS over the three Cartesian components (per component, as everywhere in v5).
- *Stage 1b* (`scratchpad/w13_thresholds.py`): the OLS species alignment of W0.6a.3 and the
  thresholds `u_F`, `u_E` per size, written to `~/runs/w1_proxy_thresholds.json` before stage 2
  runs. Note on the alignment (from the fold-0 dry run): the design matrix is rank-deficient —
  these cells are Cs : Pb = 1 : 1, so only `a_Cl` and `a_Cs + a_Pb` are identifiable and
  `lstsq` splits the sum equally. The predicted alignment `Σ n_s a_s` is unique, which is all the
  proxy uses; the individual `a_Cs`, `a_Pb` are not readings.
- *Stage 2* (`scratchpad/w13_charged.py`, run only after the thresholds exist): the charged
  d-window, the proxy against `u_F` / `u_E` per size, and `admission_table` (coverage bins of
  width 0.2 Å with `n_min` 3, `s0 ± SE` from the out-of-fold NEUTRAL residuals, `sQ` from the
  charged ones), with `s_tol` = 0.018 and `z` = 2 as W0.6 registered.
- *Which model is `F_ft`* (choice, ours, registered here): on the neutral threshold frames it is
  the frame's own out-of-fold base — the production base trained on all of them, so it would be
  an in-sample reading. On charged geometries no base is in-sample, and `F_ft` is the PRODUCTION
  base, the model W3 will wrap; the spread term is over the four fold bases in both cases.
- *Dry run on fold 0 alone* (2026-09-09 23:12, one base, so no spread term): out-of-fold neutral
  force RMSE per component, median over frames — 79 atoms 7.78 meV/Å overall, by shell
  11.20 / 7.40 / 6.84 / 5.88 / 4.80, near-field 15.26 (flanking Pb) and 9.19 (first-shell Cl)
  against 5.57 for the rest of the 2–4 Å band; 159 atoms (5 frames) 9.26 overall, 23.44 in 2–4 Å
  (38.34 at the flanking Pb); 80-atom pristine 3.63. Five of the 293 79-atom frames have no
  identifiable flanking pair and carry no shell reading. Recorded as a dry run of the pipeline,
  not as the W1.3 numbers — those wait for all four folds.

**W1.3 expected-reading note (registered 2026-09-09 23:40, BEFORE stage 2 runs).** The fold-0 dry
run puts the foundation-disagreement term at 175 meV/Å (79 atoms) and 200 meV/Å (159) p95 in the
2–4 Å band, against a model force error of about 11 meV/Å in the same band. That term measures how
far the fine-tune moved the foundation, not how far a charged geometry sits outside the training
distribution, and it will dominate the registered sum `u_F = p95(dis + sd)`. We therefore expect
the stage-2 table to report a within-`u_F` fraction near 100 % and we say so before opening it:
that pass is vacuous, not a validation. The informative component is the cross-fit spread. The
registered rule is NOT changed — `u_F`, `u_E` stay the sums — but stage 1b now also writes the
per-term thresholds `u_F_dis`, `u_F_sd`, `u_E_dis`, `u_E_sd`, and stage 2 reports the within-fraction
for each term, so the discriminating component is visible next to the registered one.
Two further choices registered here: a charged frame counts as "within `u_F`" when the 95th
percentile over ITS OWN band atoms is at or below the threshold (the same statistic the threshold
was formed from, stricter than an atom-fraction reading); and the admission table is written per
fold as well as pooled, `s0(L)` from that fold's own out-of-fold neutral residuals while the
charged side (`sQ`, window, bins) is the production base on the same 1047 frames and so is
identical across folds. With five 159-atom neutral frames per fold against 0.2 Å bins and
`n_min` = 3, coverage at 159 is expected to fail per fold and probably pooled; that is a recorded
limitation of the data, and the 159-atom energy admission will read "unmeasurable", not "refused".
One-shot driver `defect-perovskite/w13_run.sh` (fetch f1–f3, stage 1 → 1b → 2, log `~/runs/w13.log`).

## W1.3 RESULT (2026-09-10 01:34; `~/runs/w1_oof_stage1.json`, `w1_proxy_thresholds.json`, `w1_charged.json`)

**Gate 1, validation vs the current base: PASS** (recorded above: 1.16 meV/atom and 10.23 meV/Å
per component against 4.9 and 11.8).

**Gate 2, out-of-fold neutral force floors by shell** (per component, median over frames,
vacancy-side centre; all four folds pooled, each neutral frame read by the one base that did not
train on it):

| size | frames | overall | 2–4 Å | 4–8 | 8–10 | 10–12 | >12 | flanking Pb | first-shell Cl | rest of 2–4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 79 | 1174 | 7.89 | 11.38 | 7.39 | 7.11 | 5.77 | 4.83 | 15.13 | 9.54 | 5.59 |
| 159 | 17 | 8.69 | 17.98 | 9.38 | 9.80 | 6.43 | 4.12 | 25.69 | 9.79 | 9.62 |
| 80 (pristine) | 612 | 3.63 | — | — | — | — | — | — | — | — |

Fold spread at 79 atoms is 7.70–8.11 overall and 10.75–12.11 in the 2–4 Å shell; at 80 atoms
3.61–3.64. (The admission table's `n_neutral_oof` at 79 atoms is 1153, not the 1174 of this table:
the 21 frames without an identifiable flanking pair have no `d` and drop out of the slope fit.) The 159-atom row rests on four frames per fold and its fold spread (5.79–18.16) is
noise, not structure. Twenty-one of the 1174 79-atom frames have no identifiable flanking pair and
carry no shell reading. The near-field ordering of D15 survives on neutral data and on the new
base: the flanking Pb pair is the hardest site (15.13 against 5.59 for the rest of its own shell),
and the error falls monotonically outwards.

**Gate 3, `s0(L) ± SE`, coverage, and the admission decision — CHARGED ENERGIES ARE NOT ADMITTED
AT EITHER SIZE, on two independent grounds.**

*Coverage.* At 79 atoms the charged frames span `d` = 4.93–6.80 Å; the out-of-fold neutral frames
span 3.68–6.02 and their 0.2 Å bin counts run 152, 97, 84, 33, 9, 3, then **0, 0, 0, 0** above
6.13 Å. The neutral geometries never reach the stretched flanking Pb–Pb distances the +1 state
explores, so four of the ten bins of the charged window have no neutral support at all and `s0`
is unmeasurable there. At 159 atoms there are 17 neutral out-of-fold frames against a 2.3 Å
window: twelve of twelve bins are under `n_min`, and no coverage test can pass on this data. No
new DFT is allowed in v5, so the gap cannot be closed by sampling.

*Why the coverage fails — the two charge states barely overlap in `d`.* This is not
under-sampling that a reweighting could repair. At 79 atoms the out-of-fold neutral frames have
`d` = 4.71 ± 0.47 Å and the charged frames 5.57 ± 0.31 Å: the distributions are offset by 0.86 Å,
nearly two neutral standard deviations, and only 378 of 1153 neutral frames fall inside the
charged range at all while 109 charged frames (10.7 %) lie beyond the neutral maximum. Counts per
0.4 Å bin, neutral against charged:

| `d` Å | 3.6 | 4.0 | 4.4 | 4.8 | 5.2 | 5.6 | 6.0 | 6.4 |
|---|---|---|---|---|---|---|---|---|
| neutral out-of-fold | 77 | 265 | 288 | 332 | 162 | 28 | 1 | 0 |
| charged | 0 | 0 | 0 | 47 | 616 | 240 | 93 | 23 |

At 159 atoms the same offset appears (neutral 5.53 ± 0.76, charged 6.09 ± 0.69, 29 % of charged
frames beyond the neutral maximum) on seventeen frames a side. The physics is the expected one:
removing the electron from the Cl vacancy leaves the flanking Pb pair less screened and it relaxes
outward, so the +1 manifold sits at systematically larger `d` than any neutral frame the base was
trained on. Restricting the charged set to `d` < 6.13 Å keeps 93 % of the frames but does not put
them where the neutral data are — that sub-window holds 949 charged frames against 191 neutral
ones. Any W5 energy claim on this base is an extrapolation in the one coordinate the defect
energy depends on, whichever way the two knobs above are ruled.

*Slope, and a code-versus-spec discrepancy found while reading it.* `admission.py`'s header scopes
the test to "within the charged window of the collective coordinate `d`", but the campaign code
fitted `s0` over ALL out-of-fold neutral frames of the size — at 79 atoms 1153 frames spanning
3.68–6.02 Å, of which only 378 lie inside the charged window, the rest supplying a 1.25 Å lever arm
below any charged frame. Both readings are now available (`AdmissionConfig.s0_window_only`,
default False so every recorded v4 number reproduces; test in `test_admission.py`) and both are
reported here:

| size | fit | frames | `s0` eV/Å | SE | `\|s0\| + 2 SE` | vs 0.018 | vs 0.05 |
|---|---|---|---|---|---|---|---|
| 79 | all neutral (registered) | 1153 | −0.0237 | 0.0030 | 0.0297 | fail | pass |
| 79 | charged window only | 378 | −0.0007 | 0.0130 | 0.0268 | fail | pass |
| 159 | all neutral (registered) | 17 | +0.1433 | 0.0356 | 0.2145 | fail | fail |
| 159 | charged window only | 13 | +0.2547 | 0.0595 | 0.3738 | fail | fail |

The decision is the same under both fits, so the conclusion is robust — but the reason differs and
that matters for what comes next. Over the full neutral range the 79-atom base carries a genuine
energy-shape slope (−0.024 eV/Å, eight standard errors from zero). Restricted to where the charged
frames actually live it is indistinguishable from zero (−0.0007 ± 0.0130): there the failure is
driven entirely by the standard error, i.e. by how few neutral frames sit in that window, not by a
measured bias. Per fold, in-window, `s0` runs −0.0542 to +0.0338 with SE ≈ 0.024 — every fold
consistent with zero individually and every fold failing 0.018 on precision alone. At 159 atoms the
in-window slope is large and significant (+0.2547 ± 0.0595): that size fails on a measured bias.
For the record `sQ` = 0.0901 ± 0.0077 at 79 and −0.0571 ± 0.0026 at 159.

*Sensitivity, so the decision surface is visible* (the registered reading is the first row):

| charged window | `s_tol` | coverage at 79 | 79 admitted | 159 admitted |
|---|---|---|---|---|
| full (registered) | 0.018 (registered) | fail | **no** | no |
| full | 0.05 | fail | no | no |
| `d` < 6.13 Å (950 of 1019 frames, 93 %) | 0.018 | pass | no (0.0297 > 0.018) | no |
| `d` < 6.13 Å | 0.05 | pass | yes | no |

The coverage pass under restriction is marginal: the top bin of the sub-window holds exactly three
out-of-fold neutral frames, `n_min` itself.

So at 79 atoms the outcome turns on two rulings that are the user's, not ours: whether the charged
set may be restricted to the neutral-supported sub-window, and which `s_tol` applies (W0.6 took
0.018, the v4 `tau_noise_shape`; `AdmissionConfig`'s own pre-v5 default was 0.05). At 159 atoms
nothing changes the answer. **Forces are admitted at every size** by the registered rule, so W3
is not blocked; W5's charged-energy claims are.

**Gate 3b, the proxy tables.** Thresholds (95th percentiles on out-of-fold neutral frames):
`u_F` = 218.4 / 293.5 / 17.7 meV/Å and `u_E` = 3.01 / 4.31 / 0.41 meV/atom at 79 / 159 / 80 atoms.
The registered `u_F` is 98 % foundation-disagreement, exactly as flagged before opening, and the
charged set passes it 100 % at both sizes — the vacuous pass we predicted, recorded as such. The
components carry the information:

| size | force spread term within its threshold | energy proxy within `u_E` | energy proxy median vs `u_E` |
|---|---|---|---|
| 79 | 75 % | **0 %** | 7.65 vs 3.01 meV/atom |
| 159 | 100 % | 6 % | 4.36 vs 4.31 |

The two instruments agree with the admission table: the forces largely transfer to charged
geometries (three quarters of frames inside the neutral spread envelope, per-frame p95 median
2.14 against a 2.78 meV/Å threshold), while the energies do not — not one 79-atom charged frame
sits inside the neutral energy envelope, and it misses by a factor 2.5. The species alignment of
W0.6a.3 works as intended (energy disagreement 3.89 → 1.27 meV/atom RMS after the fit).

**Reading.** The base passes W1 for forces and fails it for charged energies at both sizes. This
is a property of the data, not of base v2: the neutral frames do not reach the geometries the
charged state occupies, and no new DFT is permitted. Nothing further is launched pending the
user's ruling on the two knobs above and on what W5 may claim.

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

**Items 2 and 5 — frontier-only fillings and one chemical-potential solve per pass (done
2026-09-09; `scf.two_fillings`, `scf.TwoFillings`, `fill.FillResult.density()`, `fill.fill(mu=)`,
`scf.hole_response(mus=)`, `scf.FRONTIER_TOL` = 1e-10 (registered); `test_frontier.py`).** The
cProfile of the inference forward put 47 % of the wall time in `chemical_potential` (174 calls per
frame: two per fill — `fill` and `_DensityMatrix.forward` each solved it — eight per SCF pass,
plus four per Jacobian) and the density matrices of the unaffected spin channel were formed and
subtracted every pass. Now: one `eigh` per pass, ONE batched chemical-potential solve for the
distinct counts (the solves are element-wise, so the values equal separate solves to 1e-14), the
channel whose state and reference counts agree skipped exactly (one fill object reused for both,
its contribution to `dP`, `dq` and the energy identically zero), the fills' potentials reused by
the Jacobian; under `no_grad` (the SCF loop and inference) no density matrix is formed: `dq_i =
−Σ_a Δf_a |Π_i a|²` over the active levels `|Δf_a| > 1e-10`, `energy = Σ_a Δf_a ε_a + R_S −
R_ref` over all levels, `dP` from the active vectors on demand (the force cotangent, the primary
functional), `P_S` on demand for the commutator test (once per solve). With gradients enabled
(training, the attached pass) the eager fills with the divided-difference backward run as before,
so training gradients are unchanged (`test_model.py` batched-vs-per-graph gradient and FD tests
pass). **Agreement gate against the saved reference of the committed item-3 code**
(`~/runs/dscc/w2_reference_item3.json`, `w2_item2_gate.json`; the registered frames plus a neutral
79-atom frame and a two-frame batch; three models): worst |ΔE| 9.1e-12 eV, |ΔF| 1.9e-12 eV/Å, |Δσ|
1.2e-15 eV/Å³ — passed. **Phase-1 gates:** the suite (145 tests with the new ones) passes.
**Profile after items 2, 3, 5 (same protocol as the baseline):** 637 → **362 ms per frame** at
12–14 cold-start SCF iterations: `eigh` 182 ms (18.3 calls, unchanged, now 50 % of the total),
backward 39 ms (one call), base forward 23 ms, fills 4 ms, kernels 6 ms; the SCF loop's CPU-side
remainder ≈ 110 ms (from ≈ 250). Gate timings per frame (cold, first-sight transients on the first
frame): B′ LR-only 79 atoms 636–1005 → 358–740 ms, 159 atoms 1166–1298 → 846 ms; full 537–570 →
379–426; LR + U 509–538 → 361–417. Next by measured cost: the diagonalisations (items 7, 8 — backend,
batching by cell size, warm starts) and the loop's remaining CPU work.

**Item 7 — eigensolver backend and float32 pre-iterations (done 2026-09-09; `ScfOptions.mixed_precision`
= True, `pre_tol_q` 1e-5, `pre_tol_E` 1e-5 eV, `pre_tol_c` 1e-2 (registered); `test_frontier.py`).**
Backend benchmark on the A4000 (float64 symmetric `eigh`, ms per matrix): 316 orbitals —
cuSOLVER single 13.0, batched ×4 5.5, ×8 3.3; CPU LAPACK (8 threads) single 9.5, batched 9.3–9.9;
cuSOLVER float32 single 2.5. 636 orbitals — cuSOLVER 23.5 / 17.5 (×4) / 13.8 (×8); CPU 24.3 /
23.8; float32 7.4. So: batching frames of one cell size pays 3–4 × per matrix on the GPU (the
evaluation path already batches); a single 316-orbital float64 `eigh` is as fast on the CPU as
on the GPU; float32 is 3–5 × cheaper. Implemented: under no implicit derivative (i.e. not
training) the batched solver first iterates in float32 to the pre-tolerances and then continues
in float64 from that iterate to the registered tolerances; the continuation ramp's intermediate
stages (which only seed the next stage) run entirely in float32; the chemical-potential solve has
dtype-aware tolerances. The fixed point is judged in float64 and unchanged: toy batched solves
agree with the plain solver to 1e-15; **agreement gate against the item-3 reference
(`w2_item7_gate.json`): worst |ΔE| 9.1e-12 eV, |ΔF| 2.4e-12 eV/Å, |Δσ| 2.1e-15 eV/Å³ — passed.**
`test_batched_lm_solver_matches_per_graph` now compares iteration counts under
`mixed_precision=False` (the reference algorithm); a new test checks the mixed fixed point.
**Profile after items 2, 3, 5, 7:** 637 → 362 → **277 ms per frame** (cold start): `eigh` 94 ms
(22 calls, most of them float32), backward 42, base forward 22, fills 4, kernels 6. Gate
timings per frame (cold, first frame with the first-sight transient): B′ LR-only 79 atoms
636–1005 → 226–581 ms; 159 atoms 1166–1298 → 665–701; full 537–570 → 224–323; LR + U 509–538 →
193–327. The CPU backend for single frames and float32 pre-iterations in training are not used
(training gradients untouched). The pre-stage is a model-level inference decision (`mixed = not
training` at the solver calls, so the training path — including its detached first-visit seed —
is bitwise as before and `test_batched_training_path_matches_per_graph` compares like with like),
and it is skipped from a warm start (`dq0` given: the float64 solve needs its two or three fills
anyway; measured 164 vs 167 ms per frame either way).

**Item 8 — warm-started inference and the registered 2× benchmark (measured 2026-09-09).**
Warm-started along a charged 79-atom trajectory (each frame from the previous fixed point,
`scratchpad/w2_warm_profile.py`): **167 ms per frame** at 3–5 float64 SCF iterations —
`solve_dscc_batched` 79 ms (six `eigh` 60 ms, the chemical potentials ≈ 14 ms with their host
syncs, `hole_response` 8), the merged backward 38 ms, the base forward 22 ms, the B′ reference
fill (`q0`, one `eigh`) 14 ms, kernels 6 ms. **Registered 2× benchmark (P4.3 protocol,
`dscc_benchmark.py`, idle A4000, `B_Bp_lr_only_s3`):** 79 atoms base 42.5 ms, model 164 ms,
ratio median **3.90**, p95 4.50 (100 frames, SCF median 2 iterations); 159 atoms 73.8 / 322 ms,
median **4.26**, p95 8.33 (16 frames) — down from 7.03 / 8.59 at D14, still above the criterion.
What is left is structural: one full base forward and its backward (≈ 50 ms, the criterion's own
denominator) plus one `eigh` per SCF fill on a dense 316-orbital `H`; the head's remaining
budget under 2× would be ≈ 43 ms at 79 atoms. That is the W6 question (one `eigh`, SCF-free),
not a further W2 item; items 1 and 6 (the training-path shared graph, the single Fréchet
contraction) can still trim the 38 ms backward.


**C13 — registered background convention (ruled by the user 2026-09-09 evening; verbatim):**

```markdown
# C13 — registered background convention (v5, all W2+ code)
E_PBC_ij = (4π/Ω) Σ_{G≠0} e^{−G² r_g²}/G² cos(G·r_ij)          (own-cloud included; no constant;
                                                              an α-split implementation must add
                                                              −π(1/α² − 4 r_g²)/Ω, not −π/(α² Ω))
K_LR_ij  = (4π/Ω) Σ_{G≠0} e^{−G² r_s²/4}/G² cos(G·r_ij) − π(r_s² − 4 r_g²)/Ω
κ(h)     = −π(r_s² − 4 r_g²)/Ω     (item 4; energy ½ κ Q², stress ∂κ/∂h, no forces)
Γ_LR     : G ≠ 0 sum with the combined width, no constant (q0 net neutral)
Conversion, old → new energies: add ½ · 4π r_g² · C/(ε∞ Ω) · Q²  (C = 14.4 eV·Å);
forces, stress-free quantities, fixed points and selections unchanged.
Old-convention files: everything before the change, including the F-SCC finals.
W6: E_M(Q; h) includes the same model-density second-moment term.
```

*The user's reasoning, recorded:* both conventions are α-independent (so the Ewald test could not
decide) and share the inclusive diagonal (verified in W0.5); they differ only in how the model
density interacts with the neutralising background. The declared density is a superposition of
Gaussian clouds of width `r_g`; the exact energy of that density plus its background in a
periodic cell is the G ≠ 0 sum with no constant, and subtracting the `K_SR` lattice sum leaves
`−π(r_s² − 4r_g²)/Ω`. The code's `−π r_s²/Ω` was the point-charge background convention (the
clouds interact with themselves as Gaussians but with the background as points — no single
density does that). Physically the term is the Makov–Payne second-moment contribution
`(2π/3ε∞Ω) Q Q_2` of the model density; a 1 Å cloud is a fair stand-in for a Pb 6p orbital.
Consequences ≈ 8 meV at 79 atoms, 4 meV at 159 — a `1/Ω` term a size-independent `C_Q` cannot
absorb but the 17-frame shape test cannot resolve. The uniform mode never enters the fixed point:
`δP`, `q`, forces, the root rule, the selection and the 2× benchmark are identical; energies,
stress, item 4's `κ` and the `1/Ω` component of the ladder change; `Γ_LR` is unaffected because
`q0` is net neutral. *Actions ruled:* reprofile the post-hoc `C_Q` under the new convention;
re-run the tiling ladder before W3 opens; W6's `E_M` carries the same second-moment term; the
F-SCC finals and every earlier file stay in the old convention, compared only with old files.

*Implementation (2026-09-09, this commit):* `KernelConfig.background` = 'density' (default, also
for configs pickled before v5) | 'point' (v4, for conversions); `ewald.ewald_matrix(background=)`
— under 'density' the α-split background is `−π(4η² − w²)/Ω` per pair (the user's
`−π(1/α² − 4r_g²)/Ω` with `1/α² = 4η²`, generalised to mixed widths) so the total constant
vanishes; `ewald.reciprocal_matrix(constant=)` with 0 by default; `kernels.k_lr_constant(cfg)` =
`−π(r_s² − 4r_g²)` ('density') or `−π r_s²` ('point'), added to the G ≠ 0 sum of the broad
Gaussian on both routes; `gamma_lr` carries no constant under 'density'. Regime A follows the
same rule through `ewald_matrix`. **Tests** (`test_reciprocal.py`, `test_ewald.py`, `test_kernels.py`):
K_LR(density) − K_LR(point) = 4π r_g² C/V on every entry on both routes; Γ_LR(density) −
Γ_LR(point) = π w² C/(ε V) and a net-neutral pattern feels no difference in `W`; the
density-convention E_PBC equals the G ≠ 0 sum with no constant; the LES oracle is the point
convention (kept as the point-convention check) and the density energy differs from it by
2π r_g² C Q²/V; the Makov–Payne diagonal check gains the cloud's 4π r_g² C/L³ under 'density'
(both conventions tested); the regime-B short-range identity `K_SR = E(r_g) − E(r_s/2)` holds
under 'point' and under 'density' with the summand's own G = 0 term π(r_s² − 4r_g²)C/V made
explicit. **Model-level conversion gate (`scratchpad/w2_gate_c13.py`, `~/runs/dscc/c13_gate.json`,
against the point-convention reference of the registered frame set, three models):** every
charged-frame energy shifts by exactly `½ · 4π r_g² C/(ε∞ V) · Q²` — 7.94 meV at 79 atoms
(V = 2937 Å³), 3.97 meV at 159, 0.99 meV on the 639-atom tiling — residual |ΔE − conversion| ≤
9.4e-12 eV; forces unchanged to 1.9e-12 eV/Å; the stress shifts by the pure pressure term
`−A/V²` (A = the energy constant × V) to 1.2e-15 eV/Å³; fixed points unchanged. **Ladder
reading under C13:** the second-moment term is a `1/V` piece (22.6 eV·Å³ / L³ for Q = 1, ε∞ = 4,
r_g = 1 Å), which a three-cell `1/L` fit reads into its slope (the locally neutral toy ladder
moved from −5.11 to −4.82 eV·Å, 5.7 %); `ladder.second_moment_term` is subtracted before the
monopole `1/L` reading (the kernel test, `dscc_ladder.py`'s `dE_slope_minus_second_moment`), and
`dscc_ladder.py`'s cell-shape Madelung coefficient is the point-charge one by construction
(`background="point"`). Old-convention files: every result before this commit, including the
F-SCC finals; the conversion above applies to their energies.
**C_Q reprofiled under C13 (`scratchpad/c13_cq_reprofile.py`, `~/runs/dscc/c13_cq_reprofile_s3.json`;
the winner's held residuals converted analytically — the gate shows the conversion is exact —
and the registered post-hoc profile re-run on the interpolation-only 159-atom frames against the
out-of-fold neutral base residuals):** `C_Q` 12.5651 → 12.5691 eV (+4.0 meV, the 159-atom
conversion; SE 0.0026 unchanged), `s0` 0.117 ± 0.064 and the shape residual slope −0.0076
unchanged (a constant shift within a size), the 79-vs-159 mean residual after one `C_Q` −0.8226 →
−0.8187 eV (the two sizes' conversions differ by 3.9 meV; the −0.82 eV between-size offset itself
is the E_SF / small-cell term recorded at D13–D14, untouched). **Tiling ladder under C13 (`dscc_ladder.py` on the winner `B_Bp_lr_only_s3`, CPU, static-cell
tilings, `~/runs/dscc/ladder_bp_lr_only_s3_c13.json`):** `E(+1) − E(0)` = 8.1685 / 8.2536 / 8.3282 /
8.2974 eV on the 79 / 159 / 319 / 639-atom cells (L = 14.3 / 18.0 / 22.7 / 28.6 Å); the same-shape
`1/L` fit (79, 159, 639) gives −3.58 eV·Å raw and **−3.77 eV·Å after the second-moment term** is
removed (`dE_slope_minus_second_moment`), i.e. 77 % of the expected −4.89 (the old-convention s0
reading was 67 %; the v4 Route A LR-only 24 %); `K_LR_ii`'s size-dependent part matches the
cell-shape Madelung coefficient to 1.8 %. Dense = sparse where the sparse step converged. The
`E_SF` exclusion from cross-size energy claims (C10 addendum) stands; the ladder is the model
property W3's B′ arm inherits. **W6 note (ruled):** `E_M(Q; h)` must carry the same model-density
second-moment term so that W6 and the SCF models agree at fixed cell.

**Item 7b — CPU LAPACK for the single float64 `eigh` (done 2026-09-09; `fill.eigh_for`,
`ScfOptions.eigh_device` = 'auto' (registered: a single float64 matrix of ≤ 512 orbitals on a
CUDA tensor goes through MKL and back; batched matrices and float32 stay on the GPU), threaded
through the batched solver and the model's reference fills).** Measured (Xeon Gold 6248R, MKL;
316 orbitals, float64): torch/MKL 6.5 / 5.0 / 5.5 ms at 1 / 4 / 8 threads, scipy evd 8.3, evr
10.8, numpy 7.9; the GPU→CPU→GPU round trip 5.9 ms against cuSOLVER's 11.8; float32 stays faster
on the GPU (2.5 vs 4.0). Gate: the C13 gate above was run with the backend on (forces to 1.9e-12,
fixed points unchanged). **Profiles:** cold start 277 → **241 ms** per frame (`eigh` 65 ms for 18
calls); warm-started 167 → **149 ms** (six `eigh` 34 ms, backward 39, base forward 23, `q0` fill
12). **Registered 2× benchmark:** 79 atoms base 43.3 ms, model 156 ms, ratio median **3.59**, p95
3.96 (from 3.90 / 4.50; D14: 7.03 / 7.43).

**Item 1 — the "duplicated base pass" (measured 2026-09-09, on the user's pushback).** The
benchmark's model time is one call of the head, which runs the base ONCE (its forward, 22–23 ms,
the same as standalone at 20 ms) and takes E_base and the head's Hellmann–Feynman terms in ONE
merged backward (38–40 ms, of which the base's own backward is ≈ 28 ms standalone). The base's
share inside the head is therefore ≈ 50 ms of the 149, the same as the base alone (43–48 ms);
there is no second base pass to remove — item 1's inference half has been in place since P4.3.
What item 1 can still return is the ≈ 10 ms the head's cotangent terms add to the backward
(item 6's contraction), not 50 ms. The remaining warm-frame budget: base 50, six `eigh` 34,
backward extra 10, `q0` 12, chemical potentials ≈ 14 (host-synchronised loops), `hole_response`
8, kernels 6, misc ≈ 15; at the criterion's 2 × 43 = 86 ms the head's own share must fall to
≈ 40 ms.


**W2 closing items (user, 2026-09-09 evening; registered verbatim before any result is read):**

```markdown
# W2 closing items (registered; gated against the saved reference outputs as before)
- chemical potentials: vectorised CPU root-find on the returned eigenvalues
- misc: launch/sync audit of the warm-frame path
- predictor: tangent extrapolation of dq along a trajectory (warm start only)
- inference tolerance: tol_q,inf = 1e-6 registered separately from the gate tolerance
  1e-8, with the force-error bound |ΔF| ≤ ‖Γ‖·tol_q,inf recorded and checked once
- q0 fill without forming P; item 6 (cotangent trim)
- active-window partial solve inside Newton (Rayleigh–Ritz on previous window vectors +
  dense inverse iteration), same fixed point within tol_f; shared with Phase 4
- re-read the registered benchmark at 79 and 159 (median and p95) after these; if p95
  fails, the 2× reading passes to W6 and the record says so
```

*The user's budget reading, recorded:* the head's own share is 99 ms against a target of ≈ 50;
mechanical floors — chemical potentials 14 → ≈ 1 (a vectorised root-find on eigenvalues that are
already on the CPU after the MKL path; the 14 ms is Python bisection with device syncs), misc
15 → ≈ 5 (launch overhead), six `eigh` 34 → 17 (a predictor along the trajectory saves one, an
inference tolerance of 1e-6 one or two), `q0` fill 12 → ≈ 7 (`q0_i` needs `Σ_occ f_a |Π_i a|²`,
not `P`), backward extra 10 → ≈ 3 (item 6); hole response and kernels 14 are physics. ≈ 50 ms of
mechanical savings lands the head near 45–50 ms and the ratio near 2.0 at 79 atoms; the p95 and
the 159-atom number decide, and 2× *with margin* stays W6's (≈ 1.9 without an SCF loop). The
criterion is harsh at this size: a 6 ms diagonalisation is already 12 % of a 50 ms base, so any
electronic head sits near the line at 79 atoms; the ratio becomes a fair test of the solver at
sizes where the sparse path is the solver. The active-window partial solve is the Phase-4 solver
in embryo. On the ladder: the kernel is clean (1.8 %); the 23 % shortfall of the total slope after
the second-moment subtraction is the band-term drift and the static pattern's non-local
compensation of D13–D14; the B′ energy record is unchanged (no cross-size claim); the `C_Q` and
offset shifts are exactly the conversion. Item 1's "duplicated base pass" withdrawn by the user
on the measurement.

**Choices ours, before reading (registered here):** (a) the CPU root-find is the same safeguarded
Newton-with-bisection on the Gaussian count, vectorised in NumPy over the stacked counts, with
the same stopping rule (|count| < 1e-13 or bracket < 1e-12) and the same three polish steps —
the values must equal the torch solve to 1e-13; (b) "tangent extrapolation along a trajectory"
is implemented as the secant of the last two fixed points, `dq_pred = 2 dq_n − dq_{n−1}` (the
geometry response `∂dq/∂R` is not formed), applied by the caller through the existing
`warm_start` interface; (c) `tol_q,inf` = 1e-6 lives in `ScfOptions.tol_q_inference` and is used
when the model is called with `training=False`; `tol_E` and `tol_c` keep their registered values;
the bound `|ΔF| ≤ ‖Γ‖₂ · tol_q,inf` is recorded per frame (‖Γ‖₂ from the frame's Γ) and checked
once on the registered frames against the 1e-8 solve; the W2 agreement gates keep 1e-8;
(d) the active window is the levels with `|f_S − f_ref| > tol_f` at the previous solve plus
`n_buffer` = 8 levels on each side; the below-window count is tracked as an integer and the
window is rebuilt from a full `eigh` whenever the Ritz residual `‖H x − θ x‖` of any window
vector exceeds `1e-8`, the count changes, or a Ritz value approaches the window edge within
`4 σ_s`; the Newton Jacobian uses the last full basis; the fixed point is required to agree with
the full-`eigh` solve within `tol_f` in `dq` (the item gate).

**Closing items, first pass (2026-09-09 evening; gate at the gate tolerance against the point-convention
reference with the C13 conversion: |ΔE − conversion| ≤ 9.4e-12 eV, |ΔF| ≤ 1.9e-12 eV/Å, stress
by the pressure term to 1.2e-15; suite 148 passed).** (a) *Chemical potentials:* the same
safeguarded Newton, vectorised in NumPy on the host (`fill._chemical_potential_numpy`, taken for
float64 spectra up to 256k eigenvalues; equal to the device solve to 1e-13, `test_frontier.py`):
14 → 0.4 ms per warm frame. (b) *`q0` without `P`:* `MACEDSCC._occupied_from_spectrum` gives
`Σ_σ Σ_a f_σ,a |Π_i a|²` from the spectrum; it is taken when no graph is needed (true
`no_grad`); at inference the head builds `H0` under gradients for the merged backward, so the
`(W, −n_site)` cotangent still flows through the fills' divided-difference backward there — the
reference fill reads 9.6 ms (from 11.9) with the host potentials, and the remaining trim is
item 6 below. (c) *Inference tolerance:* `ScfOptions.tol_q_inference` = 1e-6, applied by
`MACEDSCC._inference_options` when `training=False`; the bound check
(`scratchpad/w2_tolbound.py`, `~/runs/dscc/w2_tolbound.json`; seven registered frames, two
models): ‖Γ‖₂ = 11.2–12.2 eV, bound 1.1–1.2e-5 eV/Å, measured |ΔF| between the 1e-6 and 1e-8
solves ≤ 5.8e-15 eV/Å — the bound holds with ten orders to spare because the two solves are
the same solve: the last Newton step from the float32 seed (residual ≈ 1e-5) lands near 1e-11,
and the convergence test's other members (`tol_E` = 1e-10 eV, `tol_c` = 1e-7) still have to be
met, so relaxing `tol_q` alone changes neither the iteration count (median 4 either way) nor the
benchmark (147 vs 151 ms). An inference *triple* (`tol_q`, `tol_E`, `tol_c`) would be needed for
a saving — not registered; the user's call. (d) *Predictor:* the secant of the last two fixed
points, `2 dq_n − dq_{n−1}`, on the benchmark trajectory (100 frames, 79 atoms) — **worse**:
151 ms per frame, ratio 3.56 (p95 3.99) against 135 ms, **3.16 (p95 3.31)** with the plain warm
start at the same median of 4 iterations (the extrapolated start costs rejected fills); the
predictor is implemented in `dscc_benchmark.py --predictor` and OFF by default, the plain warm
start remains the registered initialisation. (e) *Benchmark after (a)–(d):* 79 atoms base
42.7 ms, model 135 ms, ratio median **3.16**, p95 3.31; 159 atoms 73.7 / 315 ms, **4.26**, p95
7.97 (the 636-orbital `eigh` stays on the GPU at 23 ms). Warm-frame profile: 142 ms with the
predictor on (six `eigh` 33, backward 41, base 24, `q0` 10, hole response 7, kernels 6, 38
device syncs ≈ 15 ms of waiting). Remaining W2 items: the launch/sync audit, item 6, the
active-window partial solve.

**Closing items, second pass and the benchmark re-read (2026-09-09, ~19:50).** (f) *Item 6:*
`fill.site_occupation` / `_SiteOccupation` — the two-spin reference occupations from the
spectrum with ONE Daleckii–Krein contraction as the backward (`Ĝ = Uᵀ diag(g) U` once, both
spins' divided-difference maps on it, one `U M Uᵀ`; ten products to three); equal to the
density route in value and H-gradient to 1e-10 (single and batched; the batched-vs-per-graph
training gradient test passes); the reference fill 9.6 → 8.5 ms (its `eigh` is 6 of it).
(g) *One free diagonalisation:* the batched solver's final attached pass re-filled the very `H`
the converged pass had just solved; without an implicit derivative (inference) that fill is
now reused exactly — six → **five** `eigh` per warm frame. With the frontier `dq` now summed
over all levels (the active set serves the low-rank `dP` only) the registered sum rule
`Σdq = Q` holds to rounding again (it had slipped to 9e-12 against the 1e-12 gate with the
reused frontier fill). Gate at the gate tolerance: |ΔE − conversion| 9.4e-12, |ΔF| 1.1e-11,
stress 4e-14 — passed. (h) *Active-window partial solve* (`mace/modules/dscc/window.py`, tests
`test_window.py`): Rayleigh–Ritz on the carried window basis, two-shift dense inverse
iteration with adaptive rounds, the residual check on the levels inside the tails' range, the
two LDLᵀ inertia counts certifying that exactly `m` eigenvalues lie in the window's range, and
the fallback on any failed check; on the toy it reproduces the full fillings (`dq`, energy,
`dP`) to 1e-10 and rejects a level pushed into the window. **Measured at 316 orbitals it does not
pay:** even for potential changes of 3e-5 eV the inner levels need 7–8 refinement rounds (two
316-orbital solves each) — 32 ms against 5.7 ms for the MKL `eigh`; a 3e-2 change falls back.
Dense inverse iteration converges the band-edge levels of the window too slowly at this size;
the item's value is where a dense `eigh` does not exist (the sparse path, with a
folded-spectrum Chebyshev filter instead of shifted solves). Kept, tested and certified as the
Phase-4 embryo; **not wired into the production solver.** (i) *Sync/launch audit* (torch
profiler, warm frame): self CPU 145 ms against self CUDA 129 ms; ≈ 3000 kernel launches
(18.5 ms), 180 stream syncs (10.6 ms), 261 copies (6.9 ms) per frame — the "misc" floor is
kernel fusion of the SCF loop's small ops (CUDA graphs or a fused loop), not a few stray syncs;
not done in W2.

**Registered benchmark re-read (P4.3 protocol, `dscc_benchmark.py`, idle A4000, the D14 winner
`B_Bp_lr_only_s3`, plain warm start, `tol_q,inf` 1e-6):**

| size | base | model | ratio median | p95 | SCF median | at D14 |
|---|---|---|---|---|---|---|
| 79 atoms (100 frames) | 42.2 ms | 128 ms | **3.03** | 3.27 | 4 | 7.03 / 7.43 |
| 159 atoms (16 frames) | 73.2 ms | 269 ms | **3.66** | 7.54 | 4 | 8.59 / 13.7 |

The head's own share at 79 atoms is now ≈ 86 ms (five `eigh` 30, backward 37 of which the
base's own ≈ 28, base forward 22 shared with the denominator, `q0` 8, hole response 7, kernels
6, the rest launch and sync overhead). **By the registered rule the 2× reading passes to W6:**
the median is 3.0 at 79 atoms and the p95 fails at both sizes; W2 delivered 637 → 128 ms per
warm frame (7.03 → 3.03) with every fixed point, energy (up to the C13 conversion), force and
stress preserved to the gates. What W2 leaves for W6 is structural at this size: one full
diagonalisation per SCF fill on a 316-orbital `H` (six milliseconds each, 12 % of the base
per fill) and the SCF loop's kernel count.

## W3 — head baseline on base v2 (prerequisites cleared; the arm itself is NOT opened)

**Base artefact.** `defect-perovskite/w3_make_base.py` converts a `base_v2` multi-head fine-tune
into the plain `ScaleShiftMACE` the head takes (`dscc_train.py --base`): drop the replay head,
prune to {Cl, Cs, Pb}, float64, checked on real out-of-fold frames before writing.
`~/runs/base_v2_prod/base_v2_prod_base.pt` written 2026-09-09 23:18 (heads ['Default'], elements
[17, 55, 82]).

**Path smoke (2026-09-09 23:19, local A4000, 1 epoch, 12 charged frames, Φ = 0, regime B).** The
head trains on base v2 unchanged: epoch 4 s, no unconverged frames, and the run directory carries
`model.pt`, `model_avg.pt`, `held_final.json`, `held_final_avg.json`. The W0.2 keys are present
and internally consistent — the 4–8 Å pooled count equals 4–6 plus 6–8 (173 = 42 + 131) and the
three near-field categories partition the 2–4 Å shell (8 + 20 + 3 = 31) — and the W0.4 averaged
model over a single epoch reproduces the last epoch to 1e-16. The force RMSE of an untrained head
on twelve frames is not a reading and is not recorded.

**W3 protocol registered (2026-09-10 03:20, before any W3 run is launched).** The plan's W3 is
three arms — Φ = 0, LR-only, B′ LR-only — over six seeds on base v2, read against each other and
against the old-base arms. Registered here, with the points the plan leaves open resolved:

- *Seeds and folds.* Seeds {0, 1, 2, 3, 4, 6} → folds {0, 1, 2, 3, 0, 2} (W0.3); the same seed is
  the same fold, the same held-out set and the same pairing unit for every TOST (n = 6).
- *Optimiser unchanged* from v4.3, as the v5 outline requires: 60 epochs, constant lr 2e-3,
  batch 4, no schedule, no spike action. `avg_window` = 10, so each run writes both the
  epoch-averaged and the last-epoch reading (W0.4).
- *Initialisation.* The Arm-1 `h0_state.pt` files CANNOT be used: they were fitted to the old
  base's feature width, and base v2's block-0 features are 512x0e+512x1o (W1.2). Every W3 head is
  randomly initialised from its seed. Recorded as a forced deviation from the Arm 2+3 practice of
  starting from the Arm-1 winner.
- *Base.* `~/runs/base_v2_prod/base_v2_prod_base.pt` for every arm — the production base, not a
  fold base. The fold bases exist for the W1.3 out-of-fold reading and for the proxy's spread
  term, not for training heads.
- *Which reading is compared.* Within W3, the epoch-averaged reading. Against the old-base arms,
  the LAST-EPOCH reading of both, because no pre-v5 run has an averaged one and comparing an
  averaged model with a final checkpoint would credit the averaging to the base change.
- *Centre convention for the cross-base comparison.* The old-base arms' stored held-out files are
  minimum-image (W0.1) and may not be compared with a vacancy-side reading. The old models
  themselves are kept, so the comparison is made by RE-EVALUATING the old-base arms with the
  current `evaluate` (vacancy-side centre, W0.2 shells) — an evaluation-only pass, no retraining,
  permitted under "don't retrain any models". Until that pass exists there is no cross-base
  number; the within-W3 comparison does not depend on it.
- *Gates.* The C5 bound-state precondition on base v2 (Δ_c 0.5 eV, `N_loc` 4, ≥ 95 % of the
  neutral-vacancy frames, thresholds unchanged from Arm 1); the root rule on the final model and
  each of the last ten epochs (≤ 0.10) with SCF within `n_max` on ≥ 99 % of training frames; the
  by-shell tables of W0.2; the far-field gate at `spec="v5"` for the B′ arm.
- *Readout.* The D15 near-field comparison repeated on base v2: whether the base change moved the
  2–4 Å flanking-Pb residual at 79 atoms at matched `d`. The W1.3 out-of-fold neutral floors are
  the reference the head is read against — 79 atoms 7.89 overall, 11.38 in 2–4 Å, 15.13 at the
  flanking Pb, per component.
- *Old-base re-evaluation is feasible* (checked 2026-09-10 03:19,
  `defect-perovskite/w3_oldbase_check.py`, evaluation only, nothing rewritten). An Arm 2+3 model
  (`dscc_arm23_B_A_lr_only_s0`, old base, trained before C13 and before every W2 change) loads
  into the current code unchanged — `MACEDSCC`, `n_scalars` 128, r_max 5.0, coupling on — and
  `evaluate` runs on it with the vacancy-side centre and the W0.2 shells. On 24 held-out charged
- *Old-base re-evaluation, the full pass* (`defect-perovskite/w3_oldbase_reeval.py`, started
  2026-09-10 04:19 on the local A4000): the three arms W3 repeats — Φ = 0, LR-only and B′ LR-only,
  six seeds each — re-read on their own held-out charged frames (≈ 262 per fold) under the current
  `evaluate`. All eighteen models are mirrored locally, so b3 is not touched. **The comparison is
  on FORCES only.** C13 changed the energy background, so an old-base energy and a v5 energy are
  not the same quantity and must never be differenced; forces and fixed points are unchanged by
  C13, which is what makes the force comparison legitimate. Output `~/runs/dscc/w3_oldbase_reeval.json`.

  **Complete 2026-09-10 04:32, all 18 runs, 727 s.** Medians over the six seeds, per component in
  meV/Å, on the vacancy-side centre and the W0.2 shells:

  | old-base arm | overall | 2–4 | 4–8 | 8–10 | 10–12 | >12 | flanking Pb | first-shell Cl | rest of 2–4 |
  |---|---|---|---|---|---|---|---|---|---|
  | Φ = 0 | 23.03 | 39.40 | 21.62 | 20.20 | 12.71 | 8.67 | 54.23 | 32.41 | 23.67 |
  | LR-only (route A) | 24.05 | 42.39 | 22.89 | 20.59 | 13.07 | 9.07 | 60.39 | 34.00 | 29.35 |
  | B′ LR-only | 18.97 | 34.65 | 17.84 | 14.13 | 12.04 | 8.53 | 50.33 | 27.44 | 20.60 |

  *The pass reproduces the recorded campaign numbers exactly*, which is the strongest regression
  check the W2 programme has had: the overall force RMSE does not depend on the centre convention,
  and the v4 record's medians (Φ = 0 23.0, route-A LR-only 24.1 with range 21.5–32.1, B′ LR-only
  19.0 with seed spread 0.5) come back as 23.03, 24.05 (21.5–32.1) and 18.97. Every W2 change and
  the C13 convention therefore left trained-model forces untouched, measured on eighteen trained
  models rather than on gate frames. Only the shell decomposition moves, which is the point of the
  new centre.

  Paired TOST over the six seeds at `tau` = 1.7 (positive mean d favours the second arm):
  Φ = 0 against route-A LR-only is **inconclusive** on all three keys (overall −2.63,
  [−5.81, +0.56]) — the wide LR-only seed spread swallows it, and the v4.3 rule's "choose the
  simpler when inconclusive" is what selected Φ = 0. Φ = 0 against B′ LR-only is **superior on
  every key** (overall +4.20, [+3.37, +5.03]; 2–4 +4.56; 4–8 +4.12), reproducing D14's selection
  under the corrected centre and the v5 statistics. These are the numbers the W3 arms are read
  against, arm by arm and seed by seed.
- *Feasibility check that preceded it* (2026-09-10 03:19, `defect-perovskite/w3_oldbase_check.py`).
  An Arm 2+3 model (`dscc_arm23_B_A_lr_only_s0`, old base, trained before C13 and before every W2
  change) loads into the current code unchanged — `MACEDSCC`, `n_scalars` 128, r_max 5.0, coupling
  on — and `evaluate` runs on it with the vacancy-side centre and the W0.2 shells.
- *Cost and memory, measured 2026-09-10 03:20–03:40 on the local A4000 (16 GB), 64-frame subset,
  batch 4.* Per epoch: Φ = 0 10.8 s over 17 batches (0.64 s/batch); LR-only 15.6 s warm
  (0.92 s/batch, 12.3 SCF iterations per batch after the first epoch's 47.5). Scaled to a full
  fold (≈ 785 charged training frames, 196 batches): **≈ 125 s/epoch for Φ = 0 and ≈ 180 s for
  LR-only, so ≈ 2.1 h and ≈ 3.0 h per 60-epoch run**; the eighteen runs of W3 are ≈ 49 GPU-hours.
  The coupled arm costs only 1.4× the uncoupled one — the W2 solver work is what makes that true.
- *Memory: the setup pass was the whole high-water mark, and it is now configurable.* On base v2's
  512-wide features the pristine-centre and base-cache passes at the campaign's hardcoded batch
  sizes (16 and 8) peak at **12.64 GB allocated and do not fit a 16 GB card at all**; at batch 2
  the same run peaks at **5.78 GB allocated / 6.60 GB reserved** and completes. `TrainConfig`
  gained `setup_batch_size` (16) and `base_cache_batch_size` (8), defaults exactly the literals the
  campaign ran with, exposed as `--setup_batch_size` / `--base_cache_batch_size`. These passes are
  geometry-only caches computed once before training, so the batching cannot change a reported
  number, and that was checked rather than assumed: two otherwise identical two-epoch runs (seed 3,
  LR-only) at batch 8 and batch 2 agree to relative 3e-14 and 5e-13 on the epoch losses, 1e-14 on
  the held-out force RMSE and 2.4e-15 absolute on every shell. **W3 therefore registers
  `--setup_batch_size 2 --base_cache_batch_size 2`**, which turns one run per card into three on
  b3's 24 GB cards and two on a 16 GB card. Wall clock for W3 on GPUs 4, 5 and 7 at three per
  card: ≈ 6–8 h including contention.
- *Not yet done, required before launch:* the b3 tree is frozen at `be3c39b` until the Arm 4
  F-SCC reading is written, so nothing can be launched there yet.

## W3–W6 — not opened.
