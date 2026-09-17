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

**Base v2 against the old base, by data set and by range to the defect (2026-09-10 08:35,
`defect-perovskite/base_compare.py`, `~/runs/base_compare.json`).** Both PRODUCTION bases, the
same 2877 frames, the same code, forces per component in meV/Å, energies in meV/atom after a
per-species offset fitted by ordinary least squares on the neutral frames of each base separately
(the two carry different isolated-atom references, so a raw difference is a per-composition
constant; the raw median is given in brackets).

| data set | frames | old base E | base v2 E | old base F | base v2 F |
|---|---|---|---|---|---|
| pristine, 80 atoms | 639 | 1.15 (2.93) | **0.30** (0.94) | 6.82 | **5.05** |
| neutral vacancy, 79 | 1174 | 1.31 (6.69) | **0.40** (0.65) | 13.07 | **10.96** |
| neutral vacancy, 159 | 17 | 3.08 (8.02) | **0.74** (0.63) | **6.51** | 10.96 |
| charged vacancy, 79 | 1030 | 42.16 (49.02) | **38.37** (37.66) | 53.65 | **33.22** |
| charged vacancy, 159 | 17 | 24.58 (29.52) | **18.89** (19.00) | 29.95 | **22.14** |

Force RMSE by range from the vacancy (per component, meV/Å; the 2–4 Å shell split into the
flanking Pb pair, the first-shell Cl and the rest):

| data set | base | 2–4 | 4–8 | 8–10 | 10–12 | >12 | flanking Pb | first Cl | rest of 2–4 |
|---|---|---|---|---|---|---|---|---|---|
| neutral 79 | old | 21.0 | 12.5 | 10.7 | 9.7 | 8.9 | 31.1 | 17.3 | 12.4 |
| neutral 79 | v2 | **20.0** | **10.0** | **8.8** | **7.1** | **6.5** | 34.0 | **13.5** | **9.2** |
| neutral 159 | old | **15.5** | **7.2** | **5.7** | **4.4** | **3.7** | **23.7** | **10.3** | **13.0** |
| neutral 159 | v2 | 26.9 | 11.0 | 10.4 | 8.6 | 4.7 | 43.1 | 18.2 | 12.8 |
| charged 79 | old | 120.3 | 49.4 | 24.2 | 12.9 | 10.5 | 181.2 | 91.4 | 60.6 |
| charged 79 | v2 | **74.9** | **30.4** | **15.3** | **8.9** | **8.0** | **106.1** | **61.8** | **37.6** |
| charged 159 | old | 98.6 | 33.6 | 16.5 | 10.0 | 5.0 | 170.8 | 51.2 | 33.8 |
| charged 159 | v2 | **66.4** | **26.1** | **14.3** | **8.6** | **4.5** | **110.2** | **40.6** | **27.9** |

**Reading.** (i) Base v2 wins almost everywhere, and by most on the frames that matter: on the
1030 charged 79-atom frames it takes the force error from 53.7 to 33.2 meV/Å per component, a
38 % reduction, and the near-field 2–4 Å shell from 120 to 75. The gain grows towards the defect
(flanking Pb 181 → 106) — the opposite of what a base that merely fits the bulk better would do.
(ii) Energies improve by a factor 3–4 on every neutral set, and base v2's RAW and ALIGNED medians
nearly coincide (0.65 vs 0.40 at 79 atoms) whereas the old base's differ by a factor five
(6.69 vs 1.31): base v2's isolated-atom references are close to the labels' own, the old base's
are not. (iii) The one place the old base is better is the 159-atom NEUTRAL set — 6.5 against
11.0 meV/Å, and worse in every shell — on SEVENTEEN frames, the same seventeen whose fold spread
in W1.3 ran 5.8 to 18.2. Followed up per frame
(`defect-perovskite/base159.py`): the reversal is SYSTEMATIC, not a few frames — base v2 is worse
on 15 of the 17, and the distributions barely overlap (old 3.0–13.8, v2 5.1–19.3 meV/Å per
component; v2's best frame is worse than the old base's median). It is uniform across shells, far
field included, so it is a global fit difference at that size and not a defect-local one. On the
17 CHARGED 159-atom frames the order reverses again and base v2 is both better and much tighter
(11.3–33.8 against 7.8–48.9), so it does not carry into the charged set.
*A hypothesis raised and killed the same hour, recorded so it is not raised again.* Base v2's
receptive field is 12.0 Å (r_max 6.0, two interactions) against the 79-atom cell's shortest axis
of 11.3 Å, while the old base's is 10.0 Å and fits inside it; the natural guess was that base v2
learned a self-image-wrapped environment on the small cells (99 % of the data) and mistransfers it
to the 159-atom cell. **The label-free test refutes the mechanism as stated**
(`defect-perovskite/base_extensivity.py`): tiling a 79-atom frame 1×1×2 and comparing against
twice the single-cell prediction gives EXACTLY zero difference for both bases, to 0.0000 meV/atom
and 0.0000 meV/Å — as it must, since tiling a periodic cell reproduces the same infinite crystal,
so the test cannot remove a self-image and proves only that neither base has an extensivity bug.
The cause of the 159-atom neutral reversal is therefore UNKNOWN. It is flagged for the W4 size work
rather than explained, and no receptive-field claim should be repeated without a test that
actually discriminates. Note that at 159 atoms with CHARGE base v2 is ahead again
(22.1 against 30.0), so the effect does not survive into the charged set.
(iv) The charged-frame ENERGY error of both bases (19–42 meV/atom) is not a defect of either: a
base predicts the neutral-state energy and the charge physics is exactly what the head supplies.
It is recorded as the level the head works against, not as a base error.
**Caveat on provenance:** both are production bases, so every neutral frame is in-sample for both
and those rows measure fit, not generalisation. Base v2's out-of-fold neutral numbers are the
W1.3 table (7.89 meV/Å per component at 79 atoms against the 10.96 in-sample figure here); the old
base has no fold siblings, so no out-of-fold comparison exists for it and none is implied.

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
  `--setup_batch_size 2 --base_cache_batch_size 2`.** *Correction, 2026-09-10 06:10: the packing
  claim in the first version of this line was wrong and is retracted.* That 5.78 GB was measured
  with `--subset 64`, which also shrinks the pristine set (32 frames instead of 639) and the base
  cache, so it never exercised the real setup pass. **On the full data set a training step peaks at
  11.6 GB allocated for BOTH arms** (Φ = 0 11.55, LR-only 11.59), so the base forward on the
  159-atom batches dominates and the solver does not: two runs do not share a 23.5 GB card. Three
  facts came out of the sizing, all measured on b3:
  (i) without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the many small setup batches
  fragment the reserved pool to 14.2 GB against 6.5 GB live, and one run then starves the next —
  this is what killed the first launch attempt at 05:24; with it, reserved tracks live (6.69 vs
  6.47 through setup, 11.98 vs 11.59 through training);
  (ii) the setup batching still matters, because the pristine-centre pass at batch 16 does not fit
  a 16 GB card at all;
  (iii) `dscc_train.py --gpu_memory_fraction` was added to bound the allocator (no numerical
  effect); at one run per card it is insurance, set to 0.65.
  **Layout as launched (2026-09-10 06:11): one run per card.** b3 GPUs 4/5/7 take the twelve
  coupled runs (`w3_queue.sh`, `PER_GPU=1`, `HEADROOM=13000`); the local A4000 takes the six Φ = 0
  runs one at a time (`scratchpad/w3_queue_local.sh`, fraction 0.85). Expected ≈ 12 h for both
  halves. Three runs launched at 05:24 under the wrong packing were killed during setup and their
  directories removed; no number was read from them.
- *Not yet done, required before launch:* the b3 tree is frozen at `be3c39b` until the Arm 4
  F-SCC reading is written, so nothing can be launched there yet.

## A1 — reference-free environment dependence (user amendment, 2026-09-10 12:15)

Received verbatim as `defect-perovskite/DSCC_PLAN_V5_A1_AMENDMENT.md`; it removes pristine
centring from every environment-dependent term of `H0` and applies "to W3 onward".

**State when it arrived, recorded because it conflicts with "registration before W3 opens":
W3 was already open.** The Φ = 0 arm had completed all six seeds and been read; the six Route A
LR-only runs were at epochs 20–26 of 60; the six Route B′ runs had not started. The coupled runs
were killed on arrival rather than spend four more hours on a superseded form, and every
pre-amendment artefact is archived under `~/runs/dscc/pre_a1/` (six Φ = 0 complete, six LR-only
partial). **The completed Φ = 0 arm and its two readings above are therefore a PRE-A1 BASELINE,
not a W3 result**: they stand as recorded, on the centred form, and W3 proper re-opens on the A1
form. The cross-base numbers in that section keep their value as a base comparison, since both
sides used the centred head.

### §8 2× benchmark on base v2 + the A1 head (2026-09-10 15:05, idle A4000, Φ = 0)

Base alone against base + head, energy and forces per step, warm-started along consecutive
charged frames, no SCF (Φ = 0 arm). Head weights are `dscc_w3a1_phi0_s0` at epoch ~20; the
ratio is architectural, and with coupling off there is no solver whose iteration count could
depend on the weights (`scf_iterations_median` 0 on both sizes).

| size | frames | base (ms) | base + head (ms) | ratio median | ratio p95 | criterion |
|---|---|---|---|---|---|---|
| 79 | 100 | 39.8 | 78.9 | **1.94** | **2.21** | fails on p95 |
| 159 | 16 | 68.0 | 142.7 | **2.11** | **8.93** | fails on both |

Recorded against the Arm-1 reading on the old base (79: median 2.07, p95 2.22, 100 frames;
159: median 1.96, p95 5.79, 16 frames). **The 79-atom median improving 2.07 → 1.94 is not the
head getting cheaper — it is the denominator getting dearer.** Base v2 is 512-wide at r_max
6.0 against the old base's 128-wide at r_max 5.0, so the same head is a smaller fraction of a
larger base forward. Read as an absolute cost the head is unchanged: about one base forward.

- 159 atoms has only sixteen charged frames in the whole dataset, so the v4.3 amendment's
  "re-measure with ≥ 100 frames per size" **cannot be met at that size** with this data. The
  p95 there is one or two frames out of sixteen and moved 5.79 → 8.93 between the two bases;
  at n = 16 that is a single frame changing places and it should not be read as a trend.
- The engineering item registered in v4.3 is still **not done**: "compute base features and
  head features in one graph with a single backward (no early-block recomputation)". The head
  recomputes the base's first block for its derivatives, so part of the head's cost is a
  second pass over work the base already did. That is the lever if the criterion matters; it
  is a code change that moves no trained number.
- `dscc_benchmark.py` was fixed here: it predated W1's float32 base and fed the base float64
  inputs, which now crashes outright (`both inputs should have same dtype`). The base leg is
  timed in the base's own precision with the cast OUTSIDE the timed region — that is what a
  base-only MD step costs, and putting the cast inside would charge the denominator for work
  the comparison does not do. The model leg pays its own casts, which is correct: they are
  part of what the head costs. Reports `base_dtype`, `r_max_base`, `r_cut_head`.

Artefacts `~/runs/dscc/bench_a1_79.json`, `bench_a1_159.json`.

### Ruling: `η` back to ln 3 (user, 2026-09-10 13:40)

The amendment's literal `η = 0.5` is **overruled**; the SK modulation bound returns to
`ln 3 = 1.0986`, `×[1/3, 3]`. `ETA_DEFAULT` is `HOP_LOG_BETA_DEFAULT` again and the queue
generator writes `--eta 1.0986` on every line. **Every other A1 value stands**: no pristine
reference anywhere, `Δ_Z = 3.079` eV, `β_b = β_a = 0` for W3, the readouts, the L2 at 1.8e−6.

Superseded by this ruling, and archived rather than deleted at
`~/runs/dscc/a1_eta05/` (six Φ = 0 runs, epochs 4–16 of 60, plus the queue file and log):

- Matched-epoch held force RMSE against the pre-A1 base-v2 Φ = 0 runs, per component:
  epoch 4, all six seeds, paired mean **+1.68 ± 0.31** meV/Å (same sign every seed);
  epoch 9, three seeds, **+1.04 ± 1.02**. Not a registered reading and not a clean
  attribution — the runs carried the centre removal and the narrower `η` together — but
  the sign was consistent and it is the only measurement anyone has of `η = 0.5` on real
  data. Recorded so the choice is not re-litigated from memory.
- The gap regulariser held at 2.399–2.400 against the registered 2.40 on every seed with no
  pristine centre, which is the term A1's change most exposed.
- The L2 ran four to five orders below the force loss (2.5e−10 to 8.1e−8 against 1.4–2.2e−5):
  inert, as designed, since the corrections never approached the bound.
- The toy-fixture multi-valuedness recorded below was an `η = 0.5` effect and does not apply
  to the relaunched configuration. The branch pin added to `TestPairForcePath` stays anyway:
  the test had been asserting branch selection by accident, and a bound it does not choose
  should not decide whether it measures what it says.

W3 relaunched on (A1 form, `η = ln 3`) — same eighteen runs, same names `dscc_w3a1_*`, the
names freed by the archive move.

### C5 read at last, and the rank-1 deadlock (2026-09-11)

**C5 could not be run against a float32 base.** `bound_state_precondition` called
`ScaleShiftMACE.forward(model.base, ...)` directly with float64 inputs and died with "both
inputs should have same dtype" — the same incompatibility `dscc_benchmark.py` had. That, not
oversight, is why the gate stayed open from W3 onward. Routed through `base_forward`
(`715bb9d`).

**Result: A1.2 fails C5 on every head, and the failure is SEPARATION, not localisation.**
The gate is `separation >= 0.5 eV AND n_eff <= 4.0`, per frame, on >= 95 % of 1191
neutral-vacancy frames.

| head | pass fraction | separation p50 (meV) | N_eff p50 |
|---|---|---|---|
| Φ = 0 s0 | 0.718 | 599 | 1.66 |
| Φ = 0 s1 | 0.081 | 404 | 1.53 |
| Φ = 0 s2 | 0.034 | 408 | 1.34 |
| Φ = 0 s3 | 0.207 | 440 | 1.44 |
| Φ = 0 s4 | 0.000 | 126 | 2.82 |
| Φ = 0 s6 | 0.000 | 331 | 1.33 |
| B′ s0 | 0.382 | 477 | 1.32 |

**The localisation half passes on every seed**: `N_eff` 1.32–2.82 against a ceiling of 4.0,
tighter than Arm 1's full `H0` on the old base (2.2–2.5) and far tighter than its scalar-only
control (up to 6.1). The carrier is tight; the level is not gapped. Seed 0 shows the shape:
its median separation (599 meV) clears 500, yet only 855/1191 frames pass, so the
distribution straddles the threshold rather than the state being absent. **Correction to our
first reading of this table, recorded because it was wrong in substance: we described it as
"losing the bound state", which the N_eff column contradicts.**

**The rank-1 deadlock.** `sp_block = a_Z · (vector_mix[Z] · vectors)`, with `alpha` (through
`a_Z = a_max tanh alpha`) and `vector_mix` BOTH zero-initialised, and each one's gradient
proportional to the other. Zero times zero is a saddle the optimiser cannot leave, so the
rank-1 s–p block has been **identically zero after 60 epochs in every production run of the
campaign** — measured, `|vector_mix|` and `|alpha|` exactly 0.0 in the converged W3 Φ = 0 and
B′ heads. `beta`/`b_Z` escapes only because it multiplies the geometric quadrupole `Q_i`,
which is nonzero whatever the parameters do. No test caught it: the fixtures break the
deadlock by hand (`test_hamiltonian._model` sets `vector_mix.normal_(0, 0.5)`,
`alpha.fill_(0.7)`).

Consequences: W4's `rank1` and `full` variants would have measured nothing (`f_Z`'s gradient
is also ∝ a_Z = 0); the rank-2 readout `g_Z` is *starved* rather than useless (its gradient is
∝ b_Z ≈ 0.03 early), so the observed rank2 ≈ spec agreement at epoch 9 is not evidence that
the capacity is unneeded; and **Arm 1's attribution of the Pb–Pb bound state to "the
directional block" must be narrowed to the rank-2 quadrupole term**, the only half that was
ever active.

Fix: `vector_mix` initialised fan-in scaled off zero, `alpha` left at zero, so the block still
starts exactly absent (tested: s–p and p–p entries identically zero at init, so the neutral
null and the Phase-1 gates are untouched) while `∂L/∂alpha` is now nonzero. Three tests pin
it. A first version of the gradient test was itself wrong — it used `(H²).sum()`, whose
cotangent `2H` vanishes at the s–p entries *because* the block starts absent, reporting no
gradient either way; a generic cotangent is the correct probe.

**Since the rank-1 term is a separation mechanism (Arm 1: the p_σ/p_π splitting at the
flanking Pb), whether the fix recovers C5 is the primary readout of the W3 re-run — not the
forces.** Suite 179 green.

### A1 registration — the values the amendment leaves to us (2026-09-10 12:40, written before any A1 code)

A1 asks for six items to be registered before W3 opens. Four are ours to propose; two A1
fixes itself. **Two readings below need a user ruling before launch, not before code**, and
both are marked ⚑.

**⚑ 1. `η` (SK hopping modulation bound) — A1 says 0.5; the running code says ln 3 = 1.0986.**
`t_ij = t^SK(r)·exp(η·tanh(m(h_i,h_j)))` is the code's `hop_log_beta`, and its registered
value has been `ln 3` (×[1/3, 3]) since Stage A′, where it was widened deliberately: the
trained cohort sat **at** the narrower stop on the vacancy-flanking Pb–Pb bond (ss-σ pinned
low, pp-σ and pp-π pinned high, in every d bin), a parameter at its stop has `sech² ≈ 0`
gradient, and a scaling what-if on that bond moved F4 from −0.061 to −0.090 while the force
loss fell. `η = 0.5` gives ×[0.607, 1.649] — narrower than what that measurement said was
binding. A1's own annotation on that line is "(unchanged form, reference removed)", and the
SK term never had a reference to remove, so the value may not have been the intent.
**Registered as A1 states: `η = 0.5`**, because it is the user's number and because A1's
required hop saturation fraction is exactly the diagnostic that tests whether 0.5 binds —
running ln 3 against their number would foreclose that test. Exposed as `--eta`; flipping it
back is one flag and a full W3 re-run (≈ 11 h wall on three cards).

**⚑ 2. Which terms are ON in W3.** A1 replaces the centred clauses of §2.1 (scalar onsite,
SK) *and* the W4 formula. The `b_i(h_i)` / `a_i(h_i)` readouts belong to W4 — A1's own
restated factorial still lists "+ rank-2 with species `b_Z` (β = 0)" as a W4 variant.
**Registered reading: W3 runs `β_b = β_a = 0`** — the reference-free scalar onsite and SK,
with the directional block at its species-level coefficients, i.e. today's architecture minus
the centring. The readouts are built, tested and registered now and enter at W4. Running W3
at β = 0.5 would fold W4's capacity question into the head baseline and leave W4's "full"
variant already spent. `beta_b` and `beta_a` are separate floats so W4 can move them
independently; both are written into every run record.

**3. Readout architectures and widths.** `e_Z` and `m_{ZZ'}` are the existing MLPs, unchanged:
`site = [F+8 → 64 → 64 → 2]`, `hop = [2F+16 → 64 → 64 → 4]`, SiLU, final layer scaled by 0.05.
`g_Z` and `f_Z` are new, A1's "one hidden layer of registered width": `[F+8 → 64 → 1]`, SiLU,
final layer (weights **and** bias) scaled by 0.05, over a species embedding of their own
(`elem_env`, dim 8) so that turning a readout off cannot perturb the SK path. `F = 512`
(base v2 block-0 invariants). Near-zero init means the A1 head starts exactly at the
species-default form, which is the W4 `β = 0` variant — the factorial's own baseline.

**4. `β` = 0.5** (A1), on both `(1 + β tanh)` factors; `b_i ∈ [0.5 b_Z, 1.5 b_Z]`.

**5. `Δ_Z`.** Rule: the population standard deviation of the `2·n_el` Harrison onsite
baselines of the model's elements, times a registered fraction `c = 0.40`, shared across both
shells, stored as an `[n_el, 2]` buffer filled uniformly so a per-species rule can replace it
without an interface change. For CsPbCl3 (Cl −24.63/−11.74, Cs −3.36/−1.80, Pb −15.19/−8.04):
spread = **7.696 eV**, **Δ_Z = 3.079 eV** for every species. Chosen so that A1 changes the
FORM and not the capacity: the current registered `on_site_range` is 3.0 eV, itself set by a
saturation audit, and 3.079 is within 3 % of it. Recorded alternative, so the user sees what
they would be choosing: reading `Δ_Z` as species Z's own s–p splitting gives Cl 12.89, Pb
7.15, Cs 1.56 eV before the fraction — at any `c` that keeps Cl sane it cuts **Pb** below
today's 3.0 (c = 0.25 → Pb 1.79 eV), and the flanking-Pb residual is the programme's open
question, so we do not propose it.

**6. L2 weights per term.** Rule registered here, number computed from neutral training
history only (no held-out result opened): `w = 0.05 · L_conv / 0.25`, i.e. the penalty is 5 %
of the converged force loss when a term sits at `mean tanh² = 0.25`. `L_conv` = median over
the six archived pre-A1 Φ = 0 seeds of their median last-10-epoch **train** force loss =
8.94e−6 (per seed 8.45, 7.42, 9.42, 5.45, 15.01, 12.54 e−6). **`w = 1.8e−6`, the same for
every term** (`e`, `m` in W3; `g`, `f` inherit it at W4). Penalty = `w · Σ_terms mean(tanh²)`,
logged per epoch beside `force` and `gap`. Calibrating at epoch 1 was rejected: the init loss
is orders of magnitude above converged, so an epoch-1 calibration becomes a large pull once
training settles.

**7. Foundation embedding dimension (form only, nothing built).** Species embedding of width
8 (the existing `elem_dim`), one shared readout per term taking `(embedding, h_i)`; `b_Z, a_Z,
Δ_Z, ε_Z` become outputs of that embedding rather than per-species tables. Bounds and L2
unchanged. `Δ_Z`'s rule above is already host-free — it reads the Harrison table, not the host.

**Compatibility, forced.** A1's static check ("no runtime module reads pristine feature
statistics") and the codebase's usual `getattr` back-compatibility branch cannot both exist:
a legacy branch would put `centre` back in the H0 assembly path, which is the symbol the test
asserts absent. So the `centre` argument is removed from `SlaterKosterH.on_site` outright and
HEAD before A1 is tagged **`pre-a1`** (`3a98e05`). Any re-evaluation of a pre-A1 checkpoint —
the old-base re-eval, `arm4_read`, the cross-base comparison — runs at that tag. Consequence,
stated plainly: **the old-base regression check and the base-v2-vs-old-base comparison are
frozen at their recorded numbers**; they will not be recomputed under A1 code.

**Still outstanding and unchanged by A1: the C5 bound-state precondition on base v2 has never
been read.** A1 names it as W4's entry gate and W3's gates already list it. It is a diagnostic
on trained heads, so it cannot be read until the A1-form heads exist — it does not block
implementation, and it does block adoption of any W3 result.

### A1 implementation (2026-09-10 13:30, committed, suite green)

`3a98e05` registers the amendment, `a879576` the values, `e1b5709` the code; HEAD before A1
is tagged **`pre-a1`**. What changed in the runtime path:

- `SlaterKosterH.on_site` is `eps0[Z] + Delta_Z tanh(e_Z(h_i))`; the `centre` argument, the
  `centre_form` switch and both retired forms are **gone**, not deprecated. `H0` has no
  `centre` buffer, no `centre_set`, no `set_centre`, and needs no setup call before its first
  forward.
- `Delta` is an `[n_el, 2]` **buffer** (a registered bound is not something the fit may
  widen), filled by `delta_from_baselines(eps0, 0.40)` = 3.079 eV.
- `H0.site_coefficients` adds A1's `b_i = b_Z(1 + β tanh g_Z(h_i))` and `a_i = a_Z(1 + β tanh
  f_Z(h_i))` over their own species embedding. At `β = 0` the readout is **not evaluated**:
  W4's species-coefficient variant is the species model exactly, with no gradient path left
  open (tested).
- `l2_weight · Σ_terms mean(tanh²)` is in the loss, pooled over every `H0` of a step (SCF
  iterations and the gap regulariser included) rather than averaged per call, and logged per
  epoch as `l2`.
- `saturation_report` writes the `|tanh| > 0.95` fraction per term and species plus the
  §2.10 readout at the flanking Pb into `held_final.json` and `held_final_avg.json`.
- `set_pristine_centre` → `set_pristine_reference`, which now does only what A1 keeps: the
  pristine composition, the atom count, and Route B′'s `q0` charge reference. Its feature
  pass is deleted — one of the two base forwards over the pristine set, so setup is faster;
  the `q0` pass remains and the per-batch memory peak is unchanged, so the registered 7.2 GB
  and `--gpu_memory_fraction 0.42` stand.
- `H0.__setstate__` refuses a pre-A1 pickle. Unpickling restores `_buffers` whether or not
  `__init__` would create them, so a centred checkpoint would otherwise load at HEAD and run
  the uncentred `on_site` on centred weights — silently, with no error. The tag is now
  enforced rather than merely recorded.
- `static_cell.py`'s local `centre` → `median_shift` (a fractional-coordinate median, not a
  feature mean) so the static check can be exact.

**Tests** (`tests/extensions/dscc/test_a1.py`, 11 new): A1's static check is on
**identifiers, parsed**, not on text — grepping the source would fail on the amendment's own
record of what was removed, and passing that grep would mean deleting the history rather than
the code. Exemptions are named in the test: `q0_pristine`, `pristine_atoms`, `pristine_gap`,
`static_pristine_cell`, `vacancy_centre`. Also: zero-readout limit to 1e-12 against a
species-default `H0` **built in the test** from `eps0` and `v0·radial` (comparing one `H0`
against another only proves the code agrees with itself); `β = 0` inertness; the bound
`b_i/b_Z ∈ [0.5, 1.5]`; `batched` ≡ per-graph with the factors on; the L2 pooling and its
gradient; and `Delta`'s rule, its buffer status and its host-freedom. Full suite 167 tests
green.

**Finding, recorded because it is about A1's `η` and not about A1's code: `η = 0.5` makes the
toy fixture's SCF multi-valued.** `TestPairForcePath[B-True]` failed after the change. The
cause is not the force routes — `pairs` and `autograd` agree with each other **exactly**
(both differ from the inference reference by the identical 0.3136) — it is that the inference
and training forwards converge to **different fixed points** from the same Φ = 0 start: dq
apart by 0.256, energy by 0.55 eV, 3 iterations against 21, both flagged converged. Forcing
`η = ln 3` with everything else at A1 values restores agreement to 6e−13, and forcing
`Δ = 3.0` at `η = 0.5` does not. The test now warm-starts all three forwards at the inference
solution, which is what it always meant to test (force routes at a given fixed point, not
branch selection). **The programme consequence: the registered single-valued ceiling (0.10)
and the warm-start check are the things to watch in W3's two coupled arms under the narrower
`η`.** This is a toy with no bound state, so it is a flag, not a prediction.

### W3 first result — the Φ = 0 arm on base v2 (2026-09-10 10:47, six seeds complete)

Cross-base, last-epoch reading on both sides as registered (no pre-v5 run has an epoch average),
forces per component in meV/Å, paired by seed, `tau` = 1.7, n = 6:

| reading | old base median | base v2 median | mean d | one-sided 95 % | verdict |
|---|---|---|---|---|---|
| full per-atom | 23.03 | **18.79** | +3.53 | [+0.66, +6.40] | **superior** |
| 2–4 Å | 39.40 | 34.29 | +3.60 | [−2.61, +9.81] | inconclusive |
| 4–8 Å | 21.62 | 17.94 | +2.97 | [−0.12, +6.06] | inconclusive |
| 8–10 Å | 20.20 | **14.44** | +5.68 | [+4.76, +6.59] | **superior** |
| 10–12 Å | 12.71 | **10.09** | +2.36 | [+0.42, +4.31] | **superior** |
| > 12 Å | 8.67 | **6.86** | +1.58 | [+0.72, +2.45] | **superior** |

Per seed (old → new): 21.2→18.0, 23.8→16.5, 23.1→19.6, 22.6→15.0, 24.0→**25.2**, 23.0→22.2 —
five of six improve, seed 4 does not.

**The structure of the gain is the finding, not its size.** The base change is decisive in the FAR
field — 8–10 Å improves by 5.68 meV/Å with a lower bound of +4.76, a margin of nearly three
`tau` — and is NOT resolved in the near field, where the 2–4 and 4–8 Å shells improve by a similar
absolute amount but carry a seed spread that swallows it. That is the same shape as D15's reading
on the old base: the near-field residual is the part that no change of this kind has yet moved.
The far-field gain is consistent with W1's base comparison, where base v2's advantage on charged
frames also grew with distance once the near field is excluded.

**The epoch-averaged reading costs about 1.2 meV/Å** (median 19.97 against 18.79 last-epoch) and
is worse than the last epoch on every seed. The runs are still descending at epoch 59, so the
average over the last ten epochs mixes in visibly worse models. This does not bias the cross-base
table above (last-epoch on both sides, as registered) and it is applied identically to every W3
arm, so within-W3 comparisons stay fair; but it means the registered evaluation model understates
this arm in absolute terms, and the remedy if one is wanted is more epochs, not a different
reading. Recorded for the user before the arm comparison is read.

**The registered W3 readout, answered for this arm (2026-09-10 10:55).** "Whether the base change
moved the 2–4 Å flanking-Pb residual at 79 atoms at matched `d`" — the two arms are read on the
SAME held-out frames per seed, so `d` is matched by construction. Paired over the six seeds:

| 2–4 Å category | old base | base v2 | mean d | one-sided 95 % | verdict | per-seed base v2 |
|---|---|---|---|---|---|---|
| flanking Pb pair | 54.23 | 49.59 | +2.63 | [−6.55, +11.81] | **inconclusive** | 47, 46, 52, 39, 68, 64 |
| first-shell Cl | 32.41 | 27.81 | +4.57 | [−0.75, +9.88] | inconclusive | 27, 23, 29, 20, 38, 29 |
| rest of the shell | 23.67 | 19.20 | +5.22 | [+1.52, +8.93] | **superior** | 17, 24, 15, 13, 23, 21 |

**No: the base change does not move the flanking-Pb residual.** The ordinary atoms of the same
shell improve decisively (+5.22, lower bound +1.52), the first-shell Cl improves by a similar
amount but is not resolved, and the flanking Pb pair — the site D15 identified — is unmoved within
a seed spread that runs 39 to 68 meV/Å. This is not a counting limitation: the flanking category
pools 519 atoms over the held-out set (two per frame × 262 frames); it is seed spread. The reading
is the same one D15 reached on the old base and Arm 1 reached before it: the near-field residual at
the flanking Pb is owned by something the electrostatic arms and now the base change all leave
alone, and by D15's evidence it is a property of the 79-atom labels and cell rather than of the
model. Recorded as the third independent arrival at that conclusion.

Coupled arms (Route A LR-only, Route B′ LR-only) still running; W3 selection waits for all three.

## W3–W6 — not opened.

## W6 — SCF-free model: REGISTRATION (2026-09-13, written before any W6 result)

Implemented this commit (`MACEDSCC(scf_free=True)`, `dscc_train.py --scf_free 1`,
`ewald.madelung_self`, `tests/extensions/dscc/test_w6.py`, 16 gates; the 196-test dscc suite
passes). The model is

```
E = E_base + dF_band(H0; N_S, N_ref) + E_M(Q; h) + E_host + C_Q
dF_band : the Phi = 0 path, unchanged (one eigh, two fills, HF force through (H, dP))
E_host  : dq^T Gamma_LR (s q0), NON-self-consistent -- dq and q0 are both fills of H0 itself
E_M     : 1/2 Q^2 [xi(h) + 4 pi r_g^2 C / Omega] / eps_inf
```

**The two Frechet contractions.** `occ_S = occ(H0; N_S)` and `occ_R = occ(H0; N_ref)` come from
the SAME eigendecomposition as the fill (`site_occupation`, one Daleckii-Krein contraction each);
`dq = occ_R - occ_S` and `q0 = n0 - occ_R` SHARE `occ_R`, so the backward costs two contractions
and not three, which is the plan's cost statement. `E_host` is not a Hellmann-Feynman term -- the
energy is not stationary in a non-self-consistent charge -- so its force keeps the full
`d dq/dR` and `d q0/dR`; both are carried by an `(E_host, 1)` cotangent. `Gamma_LR`'s own geometry
derivative comes from its pair derivatives under the pair force route and from the attached
lattice sum under the autograd route (stress); the two agree to 1e-9 eV/A (registered gate).
Dropping the two contractions moves a force by more than 1e-4 eV/A, so the finite-difference gate
covers them (gate `test_the_charge_derivatives_are_not_negligible`).

**`E_M` and C13.** `xi(h)` is the point-charge Ewald self constant of the ACTUAL cell
(`madelung_self`; `-alpha_M C / L` for a cubic cell, checked to 1e-9 relative, independent of the
splitting width to 1e-10). The C13 ruling of 2026-09-09 ("W6: `E_M(Q; h)` includes the same
model-density second-moment term") is honoured by the `+ 4 pi r_g^2 C / Omega` term: verified to
machine precision, `xi + 4 pi r_g^2 C / Omega = E_PBC_ii(density, r_g) - C / (sqrt(pi) r_g)`, i.e.
`E_M` is the Gaussian-cloud periodic self energy with the own-cloud self term removed. That
removed term is size-independent at fixed `Q` and `C_Q` absorbs it exactly; a point charge has no
self term, which is what the plan's `E_M` line asks for. (A review proposed dropping the
second-moment term as the plan's deferred `1/L^3` extension. It is not that term -- the deferred
one is the quadrupole of `dq` -- and C13 is explicit, so C13 governs. Recorded because the two
readings differ by 7.7 meV at 79 atoms and 4.0 at 159.)

**`E_M` on the actual cells, computed before training (eps_inf = 4.0, r_g = 1.0 A):**

| cell | V (A^3) | xi (eV) | second moment | `E_M(Q=+1)` | `E_M(Q=+2)` |
|---|---|---|---|---|---|
| 79-atom  | 2949.37 | −2.719862 | +7.67 meV | **−332.31 meV** | −1329.26 meV |
| 159-atom | 5697.70 | −2.187007 | +3.97 meV | **−269.41 meV** | −1077.62 meV |

The 79 → 159 difference is **+62.90 meV at `Q = +1`** (+251.6 at `Q = +2`): this is the
between-size term `C_Q` cannot absorb, supplied analytically in W6 where the SCF models get it
from `1/2 dq^T Gamma dq`. For scale, the W5 reading of the B′ comparator's between-size residual
after one `C_Q` was −0.4 meV at `energy_weight = 0.05`.

**Runs.** Six seeds {0, 1, 2, 3, 4, 6} → folds {0, 1, 2, 3, 0, 2}, the same folds and strata as
every W3/W4/W5 arm. Flags identical to the B′ `energy_weight = 0.05` arm now running
(`--directional 1 --regime B --base base_v2_prod --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05
--base_float32 1 --epochs 60`, W4's `spec` capacity) except `--coupling 0 --scf_free 1
--route_b 1` in place of `--coupling 1 --coupling_mode lr_only --route_b 1`.

**Gates (registered now; the comparator is the B′ `energy_weight = 0.05` arm, unread at this
writing). Adoption requires all four.**

- **G1 forces.** Paired TOST by seed (n = 6) against B′-0.05 on the same held-out folds,
  `tau = 1.7` meV/A per component. Pass = equivalent (both one-sided 95 % bounds inside
  ±tau). A one-sided superiority for B′ beyond tau fails the gate.
- **G2 energy shape.** (a) energy RMSE per atom after one `C_Q`, paired by seed: W6 no worse than
  B′-0.05 by more than **0.10 meV/atom** (one-sided 95 %). (b) between-size residual after one
  `C_Q`: **|residual| <= 10 meV** in absolute terms. (The W5 spec arm at `w = 0` read +21.6 meV
  and B′-0.05 read −0.4 meV, so 10 meV is a real constraint, not a formality.)
- **G3 tiling ladder.** `dscc_ladder.py` on the W6 winner over the registered static-cell ladder
  (79 / 159 / 319 / 639 atoms). Fit `a + b/L` to `E(+1) − E(0) − E_M` (E_M removed, since W6
  supplies it exactly): pass if **|b| <= 0.10 eV·A**, i.e. below 2 % of the expected monopole
  slope `madelung_slope(4.0) = −5.11` eV·A. Run on the B′-0.05 winner too, same reading.
- **G4 benchmark.** W6 median s/epoch on one card at batch 4, 79 atoms, **<= 0.5 ×** the
  B′-0.05 arm's on the same card and batch size (the plan's 2×). Measured on seed 0 before the
  other five are queued; recorded here whatever it says.

**Gradient fidelity, recorded rather than claimed.** `_SiteOccupation.backward` holds `eps`, `U`
detached, so under `create_graph = True` the second derivative of the occupations in `H` is
dropped from `d(force)/d(theta)`. Route B′ has trained on exactly this for `q0` since v4.2; W6
adds the same class of term for `dq`. The statement is "the same fidelity as the B′ comparator",
not "exact". The `Phi = 0` band term is unaffected (its density enters as `grad_outputs`).

**Gates already passed (implementation, not results; `test_w6.py`, 18 gates, and the 198-test
dscc suite):** neutral null bit-identical to the base; `sum dq = Q` to 1e-12 and `sum q0 = 0` to
1e-9; `dq` equals `two_fillings`' charge on an arbitrary `H` to 1e-10; batched == per-graph to
1e-12 (energy) and 1e-10 (forces); pair route == autograd route both at inference (forces) and in
training (forces AND every parameter gradient under `create_graph`, `s_raw` included); forces vs
central differences to 3e-6 eV/A (h = 1e-4); stress vs strain differences to 1e-7 eV/A^3
(h = 1e-5); gauge `H0 -> H0 + a I` leaves `dq`, `q0`, `E_host` and the forces unchanged and shifts
`E` by exactly `-a Q`; `E_M = 0` exactly at `Q = 0` (a neutral excited state still carries
`E_host`); the pristine-gap regulariser reads `H0`.

**Two pre-result corrections, recorded because both were found after the first seed-0 launch and
that launch was discarded (no result was read from it).**

1. *The gap regulariser was reading `H0 - W`.* `pristine_gap` branches on `route_b`, and W6 sets
   `route_b`. In W6 the host term is an ENERGY and never a potential: the Hamiltonian W6 fills is
   `H0`, so the regulariser (and C5's localisation reading) must be `H0`'s gap. On the toy the two
   gaps differ by more than 1e-4 eV, so this was not cosmetic. Fixed and gated
   (`test_the_gap_regulariser_acts_on_h0_not_h0_minus_w`).
2. *The pair force route detached the `Gamma_LR` geometry contraction.* Written as
   `gradient_of_contraction(dq.detach() (x) sq0.detach(), d_w)` it gives the right forces at
   inference -- every inference gate passed -- but under `create_graph` the force loss loses
   `d/dtheta` of `dq^T (dGamma_LR/dR) (s q0)`, and `s` gets no force gradient at all from the term
   it owns. Route B's own `A_w` is attached for exactly this reason. Fixed; the new training gate
   fails against the detached version and passes against the attached one (checked both ways).

### W6 RESULT — G4 benchmark (2026-09-13 04:5x; the first W6 gate to close). **FAIL.**

Card-matched on the local A4000, every run alone on the card, 79-atom frames, batch 4, median
over all logged epochs:

| arm | s/epoch | n epochs | ratio to B' |
|---|---:|---:|---:|
| `Phi = 0` (`dscc_w3fix_phi0_s0`, `dscc_w5e0p05_phi0_s0`) | 82 / 83 | 60 | 0.46 |
| **W6** (`dscc_w6_s1`) | **124** | 6 (122-126, flat) | **0.69** |
| B' 0.05 (`dscc_w5bp_s0`) | 180 | 60 | 1.00 |

**G4 as registered is <= 0.50 x; W6 reads 0.69 x. The gate fails.** W6 is 1.45 x the comparator's
speed, not the plan's 2 x. Recorded as registered ("whatever it says"), before the other gates are
read; nothing about G1-G3 is prejudged by it.

Where the time goes, since the plan's estimate was "one eigh; cost ~ `Phi = 0` plus two
contractions": `Phi = 0` is 83 s, so W6's own additions cost **+41 s/epoch** and the whole SCF loop
costs B' only **+56 s** over that. The additions are the two Daleckii-Krein contractions AND, at
least as importantly, `Gamma_LR` plus its pair derivative, both built per graph in a Python loop
(two lattice sums x 4 graphs x ~196 batches = ~1600 calls an epoch) -- a cost B' pays too, which is
why removing the SCF loop buys less than the ratio of solves suggests. No optimisation was
attempted before this reading and none is claimed; a batched `Gamma_LR` would be the obvious
target if the science gates make W6 worth pursuing.

Memory, same measurement: W6 reserves 11.6 GB on the A4000 and 11.9 GB on b3 against B' 0.05's
10.1 GB on b3 -- **W6 is the heavier model despite having no SCF loop**, so the b3 queue runner's
headroom was raised 8.5 -> 12.5 GB to stop it packing two W6 runs onto one 23.5 GB card.

### W6 — G4 CORRECTED, and the benchmark in the plan's own terms (2026-09-13 10:08)

**The G4 I registered this morning was the wrong criterion.** I wrote "W6 s/epoch <= 0.5 x the
B' arm's, the plan's 2x". The plan's 2x benchmark is not a ratio between two heads: it is the
registered P4.3 protocol (`dscc_benchmark.py`, W2 item 8) -- **model wall time divided by the
BASE's, median and p95, criterion 2**, warm-started along a charged trajectory on an idle A4000.
Both readings are recorded below; the P4.3 one governs, because it is the plan's. This is a
retrospective correction to a gate I mis-worded, made after the mis-worded gate was measured and
failed; it is labelled as such, and the number that failed is kept.

- **Training throughput (the mis-worded G4):** W6 124 s/epoch against B' 0.05's 180 on the same
  idle A4000, alone on the card = **0.69 x**, against the 0.50 x I wrote. FAIL, as recorded.
- **P4.3, model / base (the plan's benchmark), measured on today's code, one clean pass:**

| arm | 79: base / model / head (ms) | 79 ratio (p95) | 159: base / model / head (ms) | 159 ratio (p95) |
|---|---|---:|---|---:|
| `Phi = 0`, w = 0    | 42.5 / 83.4 / 41.0  | **1.95** (2.69) | 67.1 / 147.6 / 80.5  | **2.21** (7.24) |
| `Phi = 0`, w = 0.05 | 37.3 / 79.3 / 42.0  | **2.12** (2.59) | 67.2 / 148.0 / 80.8  | **2.19** (7.25) |
| **W6, w = 0.05**    | 44.4 / 91.6 / 47.2  | **2.10** (3.56) | 67.9 / 167.4 / 99.4  | **2.47** (7.48) |
| B', w = 0           | 37.6 / 155.1 / 117.5 | 4.12 (5.09)    | 67.0 / 293.6 / 226.6 | 3.91 (8.80) |
| B', w = 0.05        | 44.7 / 152.0 / 107.3 | 3.55 (4.69)    | 67.2 / 295.8 / 228.6 | 4.40 (8.88) |

**W6 misses 2x by 0.10 at 79 atoms and by 0.47 at 159; B' missed it by 1.6 and 2.4.** The head's
own cost is 47 ms at 79 and 99 at 159, against `Phi = 0`'s 41 and 81: the whole electrostatic
apparatus -- two Frechet contractions, `Gamma_LR`, the host term, `E_M` -- costs **+6 ms (+15 %)
at 79 and +19 ms (+23 %) at 159**, and the SCF loop it replaces cost B' +66 ms and +146 ms.
The energy term is free (`Phi = 0` head 41.0 at w = 0 against 42.0 at w = 0.05).

Two readings that outlive the W6 question. (i) `Phi = 0` ALONE reads 1.95-2.21 x, so at these
sizes the criterion is a statement about the base plus one `eigh`, not about the electrostatics;
the base leg is 42 ms at 79 atoms and the whole head budget under 2 x is ~43 ms, which one
316-orbital float64 `eigh` plus `H0` already spends. (ii) The p95 at 159 atoms is 7.2-8.9 for
EVERY arm including `Phi = 0`, so it is the 13-frame sample and not a model property -- the
registered B' p95 of 8.33 (2026-09-09) should be read the same way. The base leg itself scatters
+-10 % run to run at 79 atoms, so the `head` column is the comparable one, not the ratio.

### Two exact optimisations (2026-09-13), landed before the G3 ladder

1. **One `autograd.grad` for all Hellmann-Feynman cotangent terms** (`model.FUSED_HF_BACKWARD`).
   The terms share the base's block-0 graph and `H0`'s, and the per-term loop re-walked the
   expensive part once per term. `grad([t1, t2], inputs, [c1, c2])` is their sum by linearity.
   (NOT the same as differentiating `sum (cot * tensor)`, which adds a `d cot/dR` term the
   Hellmann-Feynman form must not have.) **One training step, batch 4 x 79 atoms: W6 500 -> 362
   ms (-28 %), B' 1070 -> 928 ms (-13 %).** The inference path already fused, so the benchmark
   above is unaffected by it.
2. **Cell-level memoisation in `ewald.py`** (`_CELL_CACHE`): the reciprocal-vector set and the
   point-charge Madelung constant are functions of the cell alone, and the data set has two
   distinct cells. Bypassed whenever the cell carries a graph, so the stress path is untouched.
   **Inference A/B (caches off -> on): W6 head 51.1 -> 47.2 ms at 79 (-7 %) and 102.4 -> 99.4 at
   159; B' 109.1 -> 107.3 and 227.3 -> 228.6 (inside its scatter).**

**Evidence that neither changes what the model computes.** Fused against looped in one process,
real models, CPU float64: forces differ by 5e-16 (2e-15 relative), energies by **exactly zero**,
every parameter gradient by 6e-14 (3e-16 relative) -- float reassociation. The inference path is
**bit-identical** before and after (SHA-256 of the force array on 79 x 3, 159, mixed-batch and
stress cases). Gate `test_fused_and_looped_hellmann_feynman_agree` holds the two forms to 1e-10 on
forces, energy and every parameter gradient, for W6 and for route B'. Full suite 200 passed.

*Collateral, recorded:* adding `scf_free` as an instance attribute broke every checkpoint pickled
before W6 -- `torch.load` of a whole module restores `__dict__` and never calls `set_extra_state`,
so the `Phi = 0` arms died with `AttributeError` on the first benchmark. Fixed by declaring
`scf_free` as a CLASS attribute (the pattern the other late flags use through `getattr`).

### W6 RESULT — G1 and G2 at n = 6 (all six seeds, folds {0,1,2,3,0,2} as registered)

| arm | n | force (meV/A) | flanking Pb | energy 79 (meV/atom) | energy 159 (pred) | between-size (meV) |
|---|---:|---:|---:|---:|---:|---:|
| B' 0.05 | 6 | 11.64 | 28.64 | 0.471 | 0.042 | +3.7 |
| W6 0.05 | 6 | 11.58 | 27.40 | 0.515 | 0.041 | +5.0 |

- **G1 forces** (paired TOST, tau = 1.7): mean d **+0.14**, 90 % CI [-0.21, +0.50] -> equivalent.
  **PASS.**
- **G2a energy RMSE at 79** (margin 0.10 meV/atom): d -0.0485, lower bound -0.0688. **PASS.**
- **G2b between-size after one `C_Q`** (|.| <= 10 meV): **+5.0** meV (B' +3.7). **PASS.**

G3 (the tiling ladder) is the only science gate left.

### W6 — localisation at n = 6 (`c5_w6.json`, `c5_w6_rest.json`; comparators re-measured today
at the current threshold, `c5_w5bp.json`, `c5_w5e0p05.json`)

Delta_c = 0.20 eV (4 sigma), N_loc = 4.0, all 1191 neutral-vacancy frames. Cells are
separation p50 (meV) / N_eff p50 / C5 pass fraction; a tick fails the 95 %-of-frames rule.

| seed | `Phi = 0` w = 0 | `Phi = 0` w = 0.05 | B' w = 0.05 | W6 w = 0.05 |
|---|---|---|---|---|
| s0 | 562 / 1.57 / 0.985 | 556 / 1.85 / 0.967 | 348 / 1.36 / 0.976 | 422 / 1.35 / 0.978 |
| s1 | 437 / 1.62 / 0.971 | 327 / 1.73 / 0.887 X | 220 / 1.29 / 0.640 X | 230 / 1.35 / 0.706 X |
| s2 | 364 / 1.35 / 0.977 | 313 / 1.41 / 0.950 | 305 / 1.30 / 0.960 | 245 / 1.33 / 0.772 X |
| s3 | 468 / 1.52 / 0.983 | 408 / 1.51 / 0.954 | 316 / 1.33 / 0.971 | 322 / 1.26 / 0.929 X |
| s4 | 355 / 1.49 / 0.982 | 292 / 1.57 / 0.908 X | 395 / 1.40 / 0.982 | 320 / 1.37 / 0.945 X |
| s6 | 472 / 1.59 / 0.984 | 349 / 1.49 / 0.843 X | 470 / 1.42 / 0.982 | 245 / 1.26 / 0.793 X |
| **median (C5)** | **452 / 1.55 / 0.983 (6/6)** | **338 / 1.54 / 0.929 (3/6)** | **332 / 1.35 / 0.973 (5/6)** | **283 / 1.34 / 0.861 (1/6)** |

Paired against B' 0.05 on the same six seeds (d = B' - W6, + means B' better):

| quantity | B' median | W6 median | mean d | 90 % CI | reading |
|---|---:|---:|---:|---:|---|
| separation (meV) | 332.2 | 282.7 | +45.1 | [-39.8, +129.9] | inconclusive (tau 50) |
| `N_eff` | 1.345 | 1.342 | +0.030 | [-0.036, +0.095] | **equivalent** (tau 0.2) |
| C5 pass fraction | 0.973 | 0.861 | +0.064 | [-0.021, +0.150] | inconclusive (tau 0.05) |

**Correction to the n = 4 preview recorded earlier today.** On the first four seeds I wrote that
"W6's localisation is B' localisation"; seeds 4 and 6 both lean the other way (separation -74 and
-225 meV against B'), and at n = 6 W6 passes C5 on **1 seed of 6 against B's 5 of 6**. The paired
differences in separation and pass fraction are NOT resolved at 90 % -- the seed spread is larger
than the effect, 220-470 meV within B' alone -- but the binary gate outcome is stark because the
0.95 rule cuts through a distribution that mostly sits between 0.77 and 0.98. What is settled is
that the hole is as tight in W6 as in B' (`N_eff` equivalent, 1.342 vs 1.345, both well below the
`Phi = 0` arms' 1.54-1.55); what is not settled is where the level sits relative to the band edge,
and on the point estimate W6's sits nearer it.

The W5 ruling applies unchanged: at w = 0.05 the energy term pulls the level toward the band edge
in EVERY arm (`Phi = 0` 452 -> 338 meV, 6/6 -> 3/6 on the same threshold), and the measured
leakage at sub-threshold frames was >= 98.8 % of the hole on the defect state. C5 as written is a
threshold on a proxy, not a measurement of delocalisation, and the open item from W5 -- whether
the 95 %-of-frames rule should be re-expressed as a leakage bound -- is now the item that decides
whether W6 has a localisation problem or the gate does.

## W3, W4 and W5 — results written up (2026-09-13)

*Recording note.* These arms were run and read between 2026-09-10 and 2026-09-12, against
thresholds registered before each was opened (W0 statistics, C11 shells, the W4 metric, the W5
gate). They were reported to the user as they landed but never written into this tracker; the
tables below are recomputed from the run artefacts (`held_final.json` per run) today, so the
numbers are the artefacts' and not a transcription. Everything is per component, meV/A, medians
over the six seeds {0,1,2,3,4,6} on folds {0,1,2,3,0,2}.

### W3 — head baseline on base v2, forces only (three arms)

| arm | n | force | flank Pb | first Cl | other 2-4 | 4-8 | >8 | E79 (meV/atom) | between (meV) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `Phi = 0` | 6 | 12.22 | 30.40 | 15.92 | 10.50 | 11.13 | 10.77 | 1.263 | +21.6 |
| Route A LR-only | 6 | 12.31 | 31.30 | 16.20 | 11.12 | 11.19 | 10.86 | 1.345 | +76.6 |
| **B' LR-only** | 6 | **9.45** | **22.81** | **12.59** | **9.19** | **8.99** | **7.91** | 1.160 | +103.9 |

| pair (force TOST, tau 1.7) | mean d | 90 % CI | reading |
|---|--:|--:|---|
| `Phi = 0` -> Route A | -0.24 | [-0.39, -0.10] | equivalent (inferior within tau) |
| `Phi = 0` -> B' | **+2.69** | [+2.36, +3.01] | **superior** |

**W3 selection: Route B' LR-only.** It is superior to `Phi = 0` by 2.69 meV/A with the whole
interval clear of tau, and it wins on every C11 shell, not just in the near field: flanking Pb
30.40 -> 22.81, first-shell Cl 15.92 -> 12.59, 4-8 A 11.13 -> 8.99, beyond 8 A 10.77 -> 7.91.
Route A LR-only is indistinguishable from no electrostatics at all (-0.24, inside tau): the
directional kernel without the static pattern buys nothing on this system. The energy columns are
diagnostics here (forces-only arms); the +103.9 meV between-size residual of B' is the number W5
later moves to +3.7 by putting energies in the loss.

### W4 — capacity factorial (`Phi = 0`, six seeds, W4 metric registered before opening)

| variant | n | force | flank Pb | 4-8 | E79 (meV/atom) | force TOST vs spec |
|---|--:|--:|--:|--:|--:|---|
| **spec** (species coefficients, directional block, no readouts) | 6 | **12.22** | 30.40 | 11.13 | 1.263 | baseline |
| scalar | 6 | 12.51 | 32.82 | 11.49 | 1.243 | -0.47 [-0.72, -0.22] equivalent (inferior within tau) |
| rank1 | 6 | 12.30 | 30.86 | 11.23 | 1.246 | -1.07 [-3.11, +0.98] inconclusive |
| rank2 | 6 | 11.98 | 29.91 | 10.94 | 1.236 | +0.12 [-0.01, +0.25] equivalent |
| full | 6 | 12.14 | 30.17 | 11.06 | 1.252 | +0.09 [-0.12, +0.29] equivalent |

**W4 selection: `spec` retained.** The rule was "the simplest variant that keeps the bound state
and is not inferior; `b_i(h_i)` adopted only on superiority". No variant is superior: `rank2`'s
+0.12 meV/A has a CI of [-0.01, +0.25], two orders inside tau, and `full` is the same. `scalar`
(dropping the directional block) is inferior within tau and `rank1` is inconclusive with a spread
four times the effect. `spec` is also the cheapest (82 s/epoch). The environment readouts built in
A1.1-A1.3 therefore stay in the code, exercised by the tests, and out of the production model.

### W5 — energies in the loss (`Phi = 0` sweep, plus the B' and W6 arms at the chosen weight)

| arm | n | force | flank Pb | 4-8 | E79 (meV/atom) | between-size (meV) | force TOST vs w=0 |
|---|--:|--:|--:|--:|--:|--:|---|
| w = 0 (spec) | 6 | 12.22 | 30.40 | 11.13 | 1.263 | +21.6 | baseline |
| w = 0.01 | 6 | 12.28 | 30.69 | 11.32 | 1.020 | +5.6 | -0.14 [-0.36, +0.09] equivalent |
| **w = 0.05** | 6 | 13.51 | 34.56 | 12.69 | **0.695** | **-0.4** | -1.30 [-1.61, -0.99] equivalent (inferior within tau) |
| w = 0.1 | 6 | 13.93 | 34.83 | 13.41 | 0.463 | -4.3 | -3.18 [-5.54, -0.82] **inferior** |
| w = 1.0 | 6 | 17.00 | 42.41 | 16.66 | 0.274 | +0.3 | -5.07 [-5.79, -4.35] **inferior** |
| B' w = 0.05 | 6 | 11.64 | 28.64 | 10.89 | 0.471 | +3.7 | +0.38 [+0.16, +0.61] equivalent (superior within tau) |
| W6 w = 0.05 | 6 | 11.58 | 27.40 | 10.92 | 0.515 | +5.0 | +0.53 [+0.10, +0.95] equivalent (superior within tau) |

**W5 ruling: `energy_weight = 0.05`, chosen by the user on the sweep.** The energy error falls by
a factor 1.8 (1.263 -> 0.695 meV/atom on `Phi = 0`) and the between-size residual after one `C_Q`
collapses from +21.6 to -0.4 meV, for a force cost of 1.30 meV/A that stays inside tau. At 0.1 and
above the force cost leaves tau and the gate fails. On the electrostatic arms at the same weight
the force cost does not appear at all: B' and W6 at w = 0.05 are both slightly BETTER on forces
than the forces-only `Phi = 0` baseline (+0.38 and +0.53, inside tau), so the trade the `Phi = 0`
sweep shows is not a property of the energy term but of what the model has to spend to fit
energies without electrostatics.

**Admission (supersedes W1.3 for W5, ruled by the user 2026-09-12).** W1.3 recorded "charged
energies are NOT admitted" from the `s0(L) +- SE` and coverage tables. For W5 the user ruled:
"count all frames as admissible, as if we had generated the dataset and were trying unbiased
training." Every charged frame therefore entered the energy loss, at both sizes, and the W5
numbers above are on that basis. Recorded here as a registered deviation rather than left in a
transcript; the W1.3 analysis stands unchanged as the measurement it was.

### C5 — demoted to analysis (user, 2026-09-13)

"Treat C5 as analysis not a gate." The bound-state precondition is no longer an entry gate for
W4, a pass/fail on W5 or a W6 adoption condition; the `N_eff`, separation and pass-fraction
tables stay in the record as diagnostics. This resolves the open W5 item (whether the
95 %-of-frames rule should be re-expressed as a leakage bound) by removing its consequence: the
measured leakage was >= 98.8 % of the hole on the defect state even for sub-threshold frames, so
nothing in the forces or energies was ever contingent on the threshold. The W6 reading -- 1/6
seeds passing against B's 5/6, with `N_eff` equivalent at 1.34 -- is therefore recorded and not
adjudicated.

### W6 — G3, the tiling ladder (2026-09-13; `~/runs/dscc/ladder_v5/`)

Static-cell tilings 1,1,1 / 1,1,2 / 2,2,1 / 2,2,2 = 79 / 159 / 319 / 639 atoms, dense on every
cell, `E(+1) - E(0)` fitted as `a + b/L`. The exact monopole coefficient is
`-alpha_M C / (2 eps_inf)` = **-5.107 eV.A**.

| arm | dE 79 / 159 / 319 / 639 (eV) | raw `1/L` slope | minus second moment | % of exact |
|---|---|--:|--:|--:|
| `Phi = 0`, w = 0.05 (control) | 5.86 / 6.10 / 6.42 / 6.48 | +1.979 | +1.785 | **-35 %** |
| **W6, w = 0.05** | 6.745 / 6.808 / 6.937 / 6.870 | -4.538 | **-4.732** | **93 %** |
| B', w = 0.05 | 7.310 / 7.389 / 7.505 / 7.466 | -5.195 | **-5.389** | **106 %** |

The `Phi = 0` control is the informative row: with no electrostatics the band term alone carries
**+1.98 eV.A** of `1/L`, the wrong sign entirely. Both electrostatic arms land within 7 % of the
exact coefficient, W6 under and B' over. The C13-era reading on the old base was 77 % of exact,
so this is a large improvement in both arms and W6's analytic `E_M` is doing what it was built to
do. **G3 passes on the programme's reading of the ladder.**

**The G3 threshold I registered was mis-specified, as G4 was.** I wrote "fit `a + b/L` to
`E(+1) - E(0) - E_M`; pass if |b| <= 0.10 eV.A", which presumes the band and host terms carry no
`1/L` physics. The control shows the band term alone carries +1.98 eV.A, so no model with a band
term can meet it. Measured against it anyway: W6 residual **+0.375**, B' **-0.282** eV.A. The
programme's own reading -- the percentage of the exact coefficient, which is how C13 read the
ladder -- is the one quoted above. Both numbers are recorded; the criterion is not silently
replaced.

*Two fixes the ladder needed first.* `sparse.model_forward_sparse` called `ScaleShiftMACE.forward`
directly instead of `model.base_forward`, so a float32 base met float64 data and every ladder run
died -- the same bug class as the C5 precondition's, and the third instance of it. The ladder's
dense/sparse cross-check now records any exception as a note instead of losing the row that is the
actual measurement (Route B' and W6 raise `NotImplementedError` there by design).

### W5 — the leak readout, built to the plan's wording (2026-09-13; `defect-perovskite/w5_leak_readout.py`)

The registered metric is "leak readout (head d-slope 79 vs 159, primary guard)". No instrument
existed for it on the v5 models: `b8_model_force_slope.py` is the v8/Stage-3 tool and takes a
`ForwardContext`, not a D-SCC model. Built new: the HEAD's own axial force on the flanking Pb pair
(`1/2 (F_i - F_j) . u`, `u` along the minimum-image pair vector, `F` = model minus base) against
the collective coordinate `d`, fitted separately at each size over every charged frame. The label
slope (`F_DFT - F_base`, same projection) is the primary guard: the head is asked to reproduce the
labels' own `d`-trend, not to be flat.

**The labels themselves have a large size-dependent slope: +189.0 meV/A per A at 79 atoms and
-109.4 at 159, a difference of +298.4.** A head that leaks across the cell would not track that.
W6, six seeds:

| seed | head slope 79 | head slope 159 | head 79-159 | label 79-159 | head - label |
|--:|--:|--:|--:|--:|--:|
| 0 | +167.1 | -113.6 | +280.6 | +298.4 | -17.8 |
| 1 | +158.4 | -126.8 | +285.2 | +298.4 | -13.2 |
| 2 | +147.5 | -90.7 | +238.1 | +298.4 | -60.3 |
| 3 | +163.2 | -117.8 | +281.0 | +298.4 | -17.4 |
| 4 | +155.3 | -143.7 | +299.0 | +298.4 | +0.6 |
| 6 | +178.2 | -112.2 | +290.4 | +298.4 | -8.0 |

**Median head-minus-label size difference: -15.3 meV/A per A on a label difference of +298.4 --
5 %.** The head's `d`-slope changes between 79 and 159 atoms by what the labels change by. That is
the leak readout passing, and it is the measurement the W5 gate asked for.

All four arms, six seeds each (`~/runs/dscc/leak_v5/`):

| arm | n | head 79-159 (median) | label 79-159 | head - label | as % of label |
|---|--:|--:|--:|--:|--:|
| **W6, w = 0.05** | 6 | +283.1 | +298.4 | **-15.3** | **5 %** |
| B', w = 0.05 | 6 | +267.2 | +298.4 | -31.2 | 10 % |
| `Phi = 0`, w = 0.05 | 6 | +262.3 | +298.4 | -36.1 | 12 % |
| `Phi = 0`, w = 0 | 6 | +266.0 | +298.4 | -32.4 | 11 % |

**W6 tracks the labels' size-dependence about twice as closely as anything else in the
programme.** The analytic `E_M` plus the non-self-consistent host term reproduces how the axial
`d`-slope changes between cells better than the self-consistent loop does, and the two `Phi = 0`
arms -- with and without energies in the loss -- sit together at 11-12 %, which says the residual
11 % is the band term's and not the energy term's. This is the same ordering the tiling ladder
gives from a different quantity (W6 93 % and B' 106 % of the exact monopole slope against `Phi =
0`'s -35 %), so two independent size-dependence instruments agree.

### `C_Q` written into the checkpoints (2026-09-13; `defect-perovskite/dscc_calibrate.py`)

`C_Q` was never a learned parameter: under the quadratic energy loss its optimum is the mean
residual, so W5's trainer profiles it out, tracks a running estimate per charge state and records
it every epoch in `history.json`. It was never written into the model, so every checkpoint
returned `calibrated: false` with an empty `c_q_table` and energies offset by ~10 eV per charge --
correct for the loss, wrong for a caller. Now fitted in closed form on each run's OWN TRAINING
fold (never the held-out fold: the constant is a parameter) and stored with its record through
`set_calibration`, written to `model_calibrated.pt` beside the untouched `model.pt` that every
measurement in this tracker was taken on. All 24 production checkpoints:

| arm | n | `C_Q(+1)` median (eV) | spread across seeds | training residual sd (eV) |
|---|--:|--:|--:|--:|
| `Phi = 0`, w = 0 | 6 | -10.6636 | 0.048 | 0.0993 |
| `Phi = 0`, w = 0.05 | 6 | -10.7350 | 0.305 | 0.0525 |
| B', w = 0.05 | 6 | -9.9394 | 0.476 | 0.0361 |
| W6, w = 0.05 | 6 | -9.3257 | 0.787 | 0.0407 |

The closed-form constant agrees with the trainer's running estimate to a few meV (W6 s1 9.4819 vs
9.4925; B' s0 10.0221 vs 10.0196). The residual sd is the per-cell energy error the constant
leaves behind: 36-41 meV on the arms trained with energies, 99 meV on the forces-only arm, which
is the same story the held-out meV/atom columns tell. The seed-to-seed spread of `C_Q` itself
(0.3-0.8 eV) is the head's zero moving between independently trained models -- each model carries
its own constant, which is exactly why the constant is per model and not a programme-wide number.

### Localisation across every arm, at `Delta_c` = 0.20 eV (analysis, not a gate)

| arm | n | C5 | separation p50 (meV) | `N_eff` p50 | pass fraction |
|---|--:|---|--:|--:|--:|
| `Phi = 0`, w = 0 | 6 | 6/6 | 452 | 1.55 | 0.983 |
| Route A LR-only, w = 0 | 6 | 6/6 | 495 | 1.72 | 0.984 |
| B' LR-only, w = 0 | 6 | 6/6 | 392 | 1.32 | 0.983 |
| `Phi = 0`, w = 0.01 | 6 | 6/6 | 444 | 1.58 | 0.982 |
| `Phi = 0`, w = 0.05 | 6 | 3/6 | 338 | 1.54 | 0.929 |
| `Phi = 0`, w = 0.1 | 6 | 0/6 | 250 | 1.85 | 0.664 |
| `Phi = 0`, w = 1.0 | 6 | 0/6 | 166 | 1.62 | 0.295 |
| B', w = 0.05 | 6 | 5/6 | 332 | 1.35 | 0.973 |
| W6, w = 0.05 | 6 | 1/6 | 283 | 1.34 | 0.861 |

The `Phi = 0` sweep is a clean dose-response: separation p50 452 -> 444 -> 338 -> 250 -> 166 meV
as w goes 0 -> 0.01 -> 0.05 -> 0.1 -> 1.0, and C5 6/6 -> 6/6 -> 3/6 -> 0/6 -> 0/6. The energy term
pulls the defect level toward the band edge, monotonically, in proportion to its weight.

**Every arm passes at w = 0 and the failures appear only when energies enter the loss** -- in all
three w = 0.05 arms, including the one with no electrostatics at all. The electrostatic arms
(B', W6, and B' at w = 0) hold `N_eff` at 1.32-1.35 against 1.55-1.72 for the arms without a
static pattern: the pattern tightens the hole. What moves with the energy term is where the level
sits, not how localised it is.

### What two cell sizes can and cannot constrain (2026-09-13, from the convergence study)

Asked why the `Phi = 0` and W6 ladders diverge when both arms fit the 159-atom energies well.
They do both fit them -- the finding is that the training data cannot see the difference.

Mean HEAD energy per size on all 1047 charged frames (the base and the labels are common to both
models, so the difference between two models of `mean head(159) - mean head(79)` IS the difference
of their between-size residuals):

| model | mean head at 79 | at 159 | delta | of which analytic `E_M` |
|---|--:|--:|--:|--:|
| W6 | 6.5200 eV | 6.4818 | **-38.2 meV** | **+66.4** |
| `Phi = 0`, w = 0.05 | 7.7141 eV | 7.6624 | **-51.7 meV** | — |

The two heads differ by **13.5 meV** in how they move between the trained sizes, which is what the
between-size residuals (+5.0 and -0.4 meV) already said. But the decomposition differs completely:
W6's analytic `E_M` supplies +66.4 meV and its band and host terms supply -104.6, nearly
cancelling it, while `Phi = 0`'s band term supplies -51.7 on its own. **Two cell sizes contain
exactly one measured size difference, and any split of it between an analytic monopole and a
fitted band term reproduces that difference equally well.** The second size is thin as well: 17
charged frames at 159 atoms against 1030 at 79, about four per held-out fold.

The split stops being arbitrary outside the trained sizes. `E_M` is a function of the cell and
keeps obeying `-alpha_cell C / (2 eps_inf L)` at 319, 639 and 959 atoms and in the limit; the band
term was fitted on 79- and 159-atom thermal snapshots and has no reason to do anything particular
beyond them. That is exactly where the two ladders separate, and they separate along the exact
monopole law for W6 (82 % of it) and with the wrong sign for `Phi = 0` (+0.55 against -1.80 eV.A).

**Recorded as a limit of the evidence, not as a claim that one extrapolation is verified.**
Confirming which is right needs DFT at 319+ atoms, which the programme forbids. The indirect case
for W6 is that its size dependence follows an analytically known law while `Phi = 0`'s does not,
and that the independent leak readout has W6 tracking the labels' own size dependence to 5 %
against 11-12 % for every other arm. The same measurement also shows W6's decomposition is not
clean: its band term cancels most of its own `E_M` on the trained structures, which is the 18 %
the ladder slope is short.

**Convergence numbers, calibrated** (`C_Q` added per model; the two curves cross at the 79-atom
cell where the constant was fitted):

| cell | W6 | `Phi = 0` | gap |
|---|--:|--:|--:|
| 79 atoms | -2.7434 | -2.7557 | 12 meV |
| 159 | -2.6797 | -2.7609 | 81 |
| 639 | -2.6183 | -2.8163 | 198 |
| 959 | -2.6025 | -2.8152 | 213 |
| extrapolated limit | **-2.4636** | -2.8529 | **389 meV** |

Instruments: `defect-perovskite/dscc_convergence_plot.py` (the plot and these tables) over
`dscc_ladder.py` runs in `~/runs/dscc/ladder_conv/`. The `1/L` axis is WRONG for a mixed-shape
ladder -- two 319-atom cells of the same volume differ by 85 meV because `alpha_cell` is 2.45 and
1.56 -- so the variable is `alpha_cell / L` and the exact slope against it is `-C / (2 eps_inf)` =
-1.800 eV.A, shape-independent.

### Convergence of `E(+1) - E(0)` with cell size, all three arms (2026-09-13)

Seven cells, 79 to 959 atoms, two shapes at 319; dense throughout; `C_Q` added per model so the
curves are on the DFT scale (`defect-perovskite/dscc_convergence_plot.py`,
`defect-perovskite/figures/convergence_e1_e0.png`).

| model | fitted slope vs `alpha/L` | % of exact (-1.800) | dilute limit (calibrated) | residual after the monopole correction, 79 -> 959 |
|---|--:|--:|--:|---|
| B' (self-consistent) | -1.620 | **90 %** | -2.3945 eV | **+18 -> +16 meV** |
| W6 (SCF-free) | -1.470 | 82 % | -2.4636 eV | +55 -> +16 meV |
| `Phi = 0` | +0.546 | -30 % | -2.8529 eV | +432 -> +193 meV |

**The two electrostatic arms agree on the dilute limit to 69 meV** -- from an SCF loop and from an
analytic term respectively, with no reason to agree there unless both carry the same physics --
while `Phi = 0` sits 389-459 meV away with the wrong sign. That mutual agreement is the strongest
evidence in the programme that the extrapolation means something; it is still not a verification,
which would need DFT at 319+ atoms.

**B' is modestly better than W6 on this test**: 90 % against 82 % of the exact slope, and +18 meV
against +55 at the 79-atom cell after the analytic correction. The self-consistent charge can
respond to the compensating background where W6's `E_M` is a fixed point-charge term. By 639 atoms
they are the same (+10 and +15 meV).

**Consequence for the W6 adoption recorded above.** On forces, energies, speed and the leak
readout W6 matches or beats B'; on finite-size extrapolation from small cells B' is better by
about 25 meV at 79 atoms, for 1.7x the cost per step. The adoption stands for MD and for
same-size work, which is what the plan's gates were written for; a use that extrapolates to the
dilute limit from 79-atom cells should read this table first. Recorded rather than re-adjudicated.

### The ladder extended to 5119 atoms, and what it says about the two models (2026-09-13)

Eleven cells, 159 to 2159 atoms, plus a 4,4,4 cell of **5119 atoms (20476 orbitals)** for W6 on
b3. Instruments: `--reuse` (a ladder cell is a deterministic function of (model, tiling), so
earlier runs are copied rather than recomputed), `--head_only_above` (`E_base` cancels exactly in
`E(+1) - E(0)` at fixed geometry, so the neutral pass is skipped and the charged one takes the
cached-base route) and `no_grad` on energy-only cells (`two_fillings` otherwise takes its eager
path and forms density matrices -- 3.35 GB each at 5119 atoms). Gated against the two-forward
result on the 639-atom cell: 6.869712 vs 6.869712 eV.

**W6, same-shape family 639 / 2159 / 5119 (alpha = 2.7225 throughout):**

| tiling | n | L (A) | alpha/L | dE (eV) | residual of the 2-parameter fit |
|---|--:|--:|--:|--:|--:|
| 2,2,2 | 639 | 28.64 | 0.0951 | 6.8697 | +0.5 meV |
| 3,3,3 | 2159 | 42.96 | 0.0634 | 6.9211 | -1.4 meV |
| 4,4,4 | 5119 | 57.28 | 0.0475 | 6.9499 | +0.9 meV |

**Slope -1.678 +- 0.050 = 93 % +- 3 % of exact; intercept 7.0288 +- 0.0035 eV.** Collinear in
`alpha/L` to +-1.4 meV over a factor of eight in volume. Adding the 79-atom cell to the same
family drags the slope to 76 %: that cell sits **+35 meV above** the line the three large ones
define, which is the higher-order term measured rather than fitted (the degenerate
three-parameter fit had assigned +47 meV to a `1/L^3` contribution). **Beyond ~640 atoms the
convergence is pure monopole and the residual after the analytic correction is 6-11 meV; at 79
atoms about 35 meV is left that the monopole correction does not reach.**

**Subset stability, the same eleven cells fitted three ways:**

| subset | n | W6 slope (% of exact) | W6 limit | B' slope (% of exact) | B' limit |
|---|--:|--:|--:|--:|--:|
| all 11 cells | 11 | -1.371 +- 0.156 (76 %) | -2.4758 | -1.750 +- 0.159 (97 %) | -2.3753 |
| >= 639 atoms | 6 | -1.733 +- 0.145 (96 %) | -2.4541 | -2.508 +- 0.057 (**139 %**) | -2.3169 |
| >= 1279 atoms | 3 | -1.390 +- 0.716 (77 %) | -2.4803 | -2.868 +- 0.174 (**159 %**) | -2.2919 |

**W6's answer does not depend on which cells are used (26 meV across the three subsets); B's does
(83 meV).** B's slope steepens monotonically as small cells are dropped -- its large cells lie on
a visibly steeper line than its small ones.

*Two artefacts ruled out.* SCF convergence: 11-19 iterations with no size trend, residual total
force ~1e-7 eV/A at every cell. The electrostatic kernel: `K_LR_ii` against the cell-shape
Madelung coefficient reads 1.3 % at 159 atoms falling monotonically to **0.25 % at 2159**. So B's
overshoot is its self-consistent charge response: a larger cell lets the compensating cloud
spread further, which is a size-dependent contribution the analytic monopole term does not
contain. W6, with a fixed `E_M` and a non-self-consistent host term, has no such freedom.

**This inverts the reading recorded earlier today** from the seven-cell set, where B' looked
better on finite-size extrapolation (90 % against 82 %). With eleven cells to 2159 atoms **W6 is
the stable extrapolator and B' is not.** Which behaviour is physically right cannot be settled
without DFT at these sizes; what can be said is that B's extrapolated dilute limit is sensitive to
the choice of cells at the 80 meV level and W6's is not. The consequence recorded for W6 adoption
("B' is better for extrapolating from small cells") is therefore withdrawn -- on the evidence now
available it is not.

### Relaxed geometries change the conclusion (2026-09-13, user-directed)

New instrument: `defect-perovskite/dscc_relax.py` -- an ASE calculator around `MACEDSCC` plus
BFGS at FIXED cell. The head short-circuits in the reference state, so the neutral cell relaxes
identically under every arm and `E(0)` is the base alone, shared by all three.

**The +1 vacancy relaxed in the 159-atom cell, W6 against B'.** Both converge (fmax 0.015 and
0.019 eV/A, 66 and 61 steps). The **structures are the same**: rms difference 0.0103 A, max
0.0252 A; the flanking Pb-Pb distance opens 5.676 -> 5.973 A (W6) and 5.978 A (B'); the first
shells sit at 2.99 / 3.53 / 3.61 / 3.63 / 3.74 A in both. Cross-evaluating each model on the
other's minimum costs 4.3 meV (W6) and 0.1 meV (B'), so the two surfaces have the same minimum
and differ only by an offset: **48.5 meV on the ideal lattice, 23.8 meV after relaxation.**

**Does the model-to-model gap grow with cell size? No -- it scatters.** Relaxing with W6 and
evaluating both models at that one geometry (so the structure is identical for both energies):

| cell | n | `alpha/L` | gap at ideal geometry | gap at the relaxed geometry |
|---|--:|--:|--:|--:|
| 1,1,2 | 159 | 0.1503 | +48.6 meV | +23.7 meV |
| 1,2,2 | 319 | 0.1076 | +59.4 meV | +4.0 meV |
| 2,2,1 | 319 | 0.0685 | +35.0 meV | +74.2 meV |
| 2,2,2 | 639 | 0.0951 | +64.0 meV | +10.6 meV |

**Two cells of the SAME size differ by 70 meV** in the relaxed gap. The scatter is the scale of
the models' own per-cell energy error (held-out RMS after one `C_Q`: W6 45.3 meV/cell, B' 37.2 at
79 atoms, n = 256 and 258; the 159-atom columns quoted throughout this programme rest on 6 and 4
held-out frames and establish no error scale at all).

*Why the two 319-atom cells differ:* they relax to different local structures. 2,2,1 is
32 x 32 x 11 A -- two octahedra thick -- so the relaxation field meets its own image along the
short axis and is cut off. d(Pb-Pb) relaxes to 5.8729 A in 1,2,2 against 5.8423 in 2,2,1, and the
first fourteen shell radii differ by 0.042 A rms. Same size, same defect, different confinement.

*Also recorded:* the relaxation energies are -1742 / -3207 / -3250 / -6105 meV, i.e. **-9.6 to
-10.9 meV per atom regardless of size** -- these relaxations are dominated by the whole lattice
settling out of the MD-averaged static cell, not by the defect. Any comparison built on relaxed
energies inherits that.

**`E(+1) - E(0)` at the relaxed geometries, all three arms** (2,2,1 excluded; `E(0)` is the base
alone at the same geometry; `relaxed_structures.json` holds the three geometries so none of this
needs recomputing):

| cell | `alpha/L` | W6 | B' | `Phi = 0` |
|---|--:|--:|--:|--:|
| 1,1,2 (159) | 0.1503 | -2.9122 | -2.8886 | -2.9619 |
| 1,2,2 (319) | 0.1076 | -2.8548 | -2.8508 | -2.9900 |
| 2,2,2 (639) | 0.0951 | -2.8344 | -2.8220 | -2.9889 |
| **fit** | | **-1.393 +- 0.051 (77 %)** | **-1.124 +- 0.259 (62 %)** | **+0.532 +- 0.138 (-30 %)** |
| **limit** | | **-2.7033 +- 0.006** | **-2.7215 +- 0.031** | **-3.0429 +- 0.017** |

**What survives and what does not.** `Phi = 0` reads +0.55 eV.A on ideal cells and +0.53 +- 0.14
on relaxed ones -- the same wrong-sign slope, robust to geometry, to relaxation and to the choice
of cells, landing 320-340 meV from the electrostatic arms. That is the case for carrying
electrostatics, reproduced three independent ways. Everything that distinguished W6 from B' does
NOT survive: on ideal cells they sat 166 meV apart with slopes of 93 % and 154 %; on relaxed cells
they are **18 meV apart** with slopes of 77 % and 62 %, both inside B's own +-26 % uncertainty.

**Withdrawn: "W6 is the stable extrapolator and B' is not."** That reading came from idealised
unrelaxed cells. On relaxed geometry the two agree within their uncertainties, and the
model-to-model differences at any single cell are smaller than either model's own energy error.
The surviving statement is narrower: the two electrostatic arms agree with each other; a head
without electrostatics does not converge at all.

## Ladder v6 — frozen-core embedding (coadvisor design, 2026-09-13)

**Why the relaxed ladder fitted badly, and why its 77 % meant nothing.** The finite-size law
depends on what is allowed to move. With ions frozen, a charge in a periodic cell follows the
`eps_inf` Madelung law -- exactly what `E_M` encodes. With ions relaxed the lattice screens the
charge too and the leading coefficient becomes the STATIC one, `-C alpha / 2 eps_0`; with
`eps_0` several times `eps_inf` (order 20 against 4) the fully relaxed slope is a fraction of the
`eps_inf` value, not 100 % of it. In 16-45 A cells the relaxation field is clipped, so each cell
sits at a different point between the two laws and cell shape controls how much is clipped. No
single `1/L` coefficient exists to fit. That is what the relaxed ladder showed, and it is why
77 % should never have been read against the `eps_inf` reference. The ideal-lattice ladder fitted
well because it IS the frozen-ion case.

Two ladders now, each testing one thing.

### Part 1 — frozen-core embedding ladder (`defect-perovskite/dscc_embed_ladder.py`)

The core is relaxed once (in the 639-atom cell, fixed volume) and embedded RIGID in larger
pristine tilings: atoms within `R_core` of the vacancy carry the relaxed displacements, everything
beyond sits at pristine positions. The local geometry is then identical in every cell and inside
the training domain; only the periodic environment changes with size. The core's polarisation is
compensated by a bound surface charge at `R_core`, so the object still carries `+1` and its images
interact through the unrelaxed medium -- the leading term is the `eps_inf` monopole whatever
`R_core` is. Energies only, `no_grad`, CPU eigendecomposition, no forces and no contractions: the
2879-atom cell (11516 orbitals) costs 76 s.

*Gate:* embedding the full core back into its own cell moves 639 of 639 atoms with none missing
and reproduces that structure's energy.

**Result, `R_core` = 8 A** (639 kept as a check, not in the fit -- its shortest dimension is 22 A):

| tiling | n | L (A) | alpha | dE (eV) | remainder after the EXACT monopole |
|---|--:|--:|--:|--:|--:|
| 2,2,2 | 639 | 28.64 | 2.7225 | -2.7928 | -2.6217 (check) |
| 2,2,3 | 959 | 32.79 | 2.8335 | -2.7769 | **-2.6214** |
| 3,3,3 | 2159 | 42.96 | 2.7225 | -2.7343 | **-2.6203** |
| 3,3,4 | 2879 | 47.29 | 2.8344 | -2.7277 | **-2.6198** |

**The remainder is flat to 1.6 meV across a 4.5x range in volume**, and the 639 check sits 1.9 meV
from the fit. `E_inf = -2.6191 +- 0.0002 eV`, `c = -79.4 +- 11.0 eV.A^3`, fit residuals +-0.1 meV.
The leading term is the exact `eps_inf` monopole -- confirmed, not fitted. The two-parameter form
needs no `1/L` term; forcing one (three points, three parameters) is unconstrained and is NOT read
as a D13 measurement.

**`R_core` sweep {6, 8, 10 A} on 959 and 2159:** `E_inf` = -2.6170 / -2.6191 / -2.6203 eV, a
**3.3 meV spread** -- the dilute limit is insensitive to the truncation radius. The `c` values
(-55, -79, -417) are not comparable: the 6 and 10 A fits have two cells each, and with `1/L^3`
differences of ~1.6e-5 A^-3 a 1 meV energy error moves `c` by ~60 eV.A^3. Only the three-cell
`c = -79 +- 11` is meaningful; after removing the model's own C13 second moment (+22.6 eV.A^3,
inside `E_M` by construction) the intrinsic part is **-102 eV.A^3**.

**`E_inf`(frozen core) = -2.619 +- 0.003 eV**, the `R_core` spread taken as the uncertainty.

### Part 2 — relaxation ladder (ionic screening), in progress

`dE_relax(L) = E_relaxed(L) - E_embedded(L)` starting from the embedded structure, near-cubic
cells only (959 and 2879), fitted as `dE_relax(inf) + (C alpha / 2)(1/eps_inf - 1/eps_0)/L` with
`eps_0` the single free parameter besides the constant -- a label-free test of the model's
far-field ionic response. A slope near zero would mean that response is missing.
(`defect-perovskite/dscc_relax_ladder.py`.)

### Part 2 as specified will not work, and the reason is not cost (2026-09-13)

The 959-atom relaxation ladder ran to convergence: **77 BFGS steps, 740 s, 9.6 s/step, fmax
0.0138 eV/A**, starting from the `R_core` = 8 A embedded structure.

**Cost, measured.** 3.8 s per energy+force call at 639 atoms, 6.0 s at 959, and 9.6 s per BFGS
step at 959 including the optimiser. The force path scales as `n^3` through the
eigendecomposition, so 2879 atoms (11516 orbitals) is ~27x that -- of order 260 s per step, and
with a comparable or larger step count (more, softer, longer-wavelength modes) that is **6-20
hours**, with a memory footprint beyond a 16 GB card. b3 is not a reliable fallback for this
workload: the same job there sat at 0 % GPU and 1200 % CPU with 13 GB allocated and produced no
step in ten minutes, against 9.6 s/step locally.

**But the arithmetic kills it before the cost does.** The measured relaxation energy is

    dE_relax(959) = -8.5083 eV = -8.87 meV per atom,

which is the WHOLE LATTICE settling out of the MD-averaged static reference cell -- the same
-9.6 to -10.9 meV/atom seen in every earlier relaxation, and proportional to `N`, i.e. to `L^3`.
The far-field ionic screening term the ladder is meant to measure,
`(C alpha / 2)(1/eps_inf - 1/eps_0)/L`, is +0.09 to +0.14 eV at this cell size for `eps_0` between
10 and 40 -- **1 to 2 % of what was measured**. Fitting a `1/L` law to a quantity dominated by an
`L^3` bulk term, with two cells and two parameters, cannot recover `eps_0`; the bulk term would
have to be subtracted to better than a part in a hundred first.

**Two ways to make it work, both cheaper than the ladder as specified.**

1. *Remove the bulk term by construction.* Relax the 80-atom PRISTINE cell with the base first
   (seconds), and build every tiling from that relaxed unit cell. Its tilings are then stationary
   by periodicity, the bulk term vanishes, and `dE_relax` is the defect's own far-field response.
   This also fixes the frozen-core ladder's reference, though part 1 is insensitive to it because
   the bulk term cancels in `E(+1) - E(0)` at fixed geometry.
2. *Get `eps_0` without the ladder at all.* `eps_0` is a property of the pristine lattice -- Born
   charges and zone-centre phonons -- and on a neutral pristine cell the head short-circuits, so
   it is the BASE's ionic response and computable on the 80-atom cell in minutes. That gives
   `eps_0^model` directly, after which `dE_relax(inf)` needs one relaxed cell rather than a ladder.

**Recommendation:** do not run 2879. Take route 2 for `eps_0^model`, and if `dE_relax(inf)` is
wanted, run route 1 at one or two sizes. Part 1's `E_inf = -2.619 +- 0.003 eV` stands on its own
and is the expensive half already paid for.

## W6 speed plan — O1, O2, O3 (coadvisor design, 2026-09-13)

Scope: the shipped W6 (w = 0.05) only. Targets at 79 atoms, model / base: current 2.10 ->
O1 ~1.7 -> O1+O2 ~1.5 -> O3 ~1.2.

### O1 — assembly (done; no model change)

**The premise needed correcting.** The scatter is already ONE vectorised `index_put` (~1.5 ms)
with no per-edge or per-species Python loop left to remove. The assembly cost is a float64
GEMM: the hopping MLP's descriptor is `feats_i + feats_j`, `|feats_i - feats_j|` and the same
for the species embedding -- all symmetric and none of them a function of `r` -- so `pre` is a
function of the unordered NODE PAIR, while the edge list carries both directions and every
periodic image of each pair. `standardise` is likewise a per-node map applied to edge rows.

Deduplicating by node pair and standardising per node (`SlaterKosterH.pair_correction`,
`integrals_fast`): on a 79-atom frame **9122 edges collapse to 2872 unique pairs**.

| | reference | O1 |
|---|--:|--:|
| `integrals` | 8.90 ms | **2.54 ms** |
| H0 assembly, 79 / 159 atoms | 10.3 / 19.5 ms | **5.3 / 10.0 ms** |
| model / base, 79 atoms | 2.10 (p95 3.56) | **1.94** (p95 2.54) |
| model / base, 159 atoms | 2.47 (p95 7.48) | **2.19** (p95 7.34) |

Agreement with the reference path **1.3e-15** (gate: 1e-12). Suite 200 + 2 new gates
(`test_integrals_fast_matches_the_reference`, and a check that the deduplication is real so
the gate cannot silently pass on a future graph convention). The assembly landed at 5.3 ms
against the <= 3 ms target; the rest is kernel-launch overhead, where `torch.compile` would
act -- deferred until O3 settles what is on the critical path.

### O2 — folded into O3 (ruled by the user)

Measured ceiling: only the base TAIL (blocks 1+ and the readout, ~9 ms of the 24.2 ms base) is
independent of the head, so overlapping it with the CPU eigendecomposition saves ~9 ms and
lands near 1.66, not 1.5 -- and it requires a maintained fork of `ScaleShiftMACE.forward`.
Ruled: a maintained fork for 9 ms is a bad trade. Under O3 the head reads nothing from the
base, so H0 is assembled first, copied to the CPU, the WHOLE base issued, and the
eigendecomposition and fills run underneath it: larger and free. **If O3 fails its gates,
1.94 at 79 atoms is where the coupled model stops and this record says why.**

### O3 — decoupled H0 (registered 2026-09-13, BEFORE the retrain)

```
A_i^(l) = sum_j c_{Z_i Z_j}^{(n,l)} R_n(r_ij) f_cut(r_ij) Y_l(rhat_ij)   l = 0,1,2; n = 1..6
v_i     = A_i^(1)                (replaces the base l = 1 feature in the rank-1 onsite block)
Q_i     = traceless A_i^(2)
t_ij    = t^SK(r_ij)             (hopping modulation removed)
```

**Registered choices (user ruling on the two open questions).**

1. *Scalar on-site input:* the species-resolved `l = 0` basis
   `A_{i,n,Z'} = sum_{j in Z'} R_n f_cut`, `N_r x n_species = 18` numbers, fed DIRECTLY to the
   production bounded, pristine-centred readout; **no `c` for `l = 0`** (the readout's first
   layer is that projection, so a separate `c` would be redundant). `c` remains for `l = 1, 2`.
   Species-only was rejected: the environment dependence of that term is not decorative (the
   centred head used its full +-3 eV range at the flanking Pb, the missing-neighbour well, and
   the one model whose corrections failed to turn on lost 4.8 meV/A everywhere). The A1.1
   standardisation path was rejected FOR THIS RUN under one-change-at-a-time: it has never
   carried a passing model, so a failed gate must be attributable to the descriptor and not to
   normalisation. It stays in the code for a later capacity arm.
2. *Centring:* production convention (static pristine cell, per species). The ensemble-mean
   centre is the right foundation form and a one-line change, but it is a SECOND change --
   deferred list.
3. *Radial family:* orthogonal Bessel, `N_r = 6`, on `[0, r_cut]`, times H0's own taper
   `(1 - (r/r_cut)^6)^2` (C1 at `r_cut`, as for the hoppings), so the head carries one cutoff
   radius and one smoothness class. Harrison powers `(d_ref/r)^n` were rejected: over a
   2.8-6 A window they are a nearly collinear monomial basis in `1/r`, ill conditioned and
   divergent at short range, and nothing reads the coefficients (they feed bounded readouts) --
   the single-convention argument concerns the hopping ENVELOPE, one specific function, not a
   basis that must span smooth functions. Gaussian shells would work but add registered centres
   and widths for no gain.

**Required reading (registered now):** the per-species median and p95 of the trained scalar
on-site correction, against the shipped W6's **0.2 / 1.85 eV** -- the turn-on check, so that
"forces equivalent" and "the term never activated" cannot be confused.

**Gates, paired by fold against the shipped W6 seeds:** force TOST (tau = 1.7 meV/A per
component) equivalent or superior; energy RMSE at 79 not inferior beyond 0.10 meV/atom;
between-size after one `C_Q` <= 10 meV; frozen-core ladder remainder flat as in v6; C5 read as
analysis; neutral null and FD-force gates re-run. Adopt on pass; on failure keep the coupled
W6 with O1 and register the three-body bond screening
`S_ij = sum_k g(r_ik) g(r_jk) P(cos theta_ikj)` as the single follow-up arm.

**Implementation gates already passed** (`decoupled=True`, toy model): FD forces 2.0e-9,
1.7e-9, 6.3e-10 eV/A; FD stress 1.4e-11, 4.1e-12 eV/A^3; gauge `H0 -> H0 + aI` shifts `E` by
exactly `-aQ` with forces unchanged to 1.1e-14 and `dq` to 4.4e-15; batched == per-graph to
0.00e+00; `sum dq = Q` to 4e-16; neutral null short-circuits; feature statistics not required
(there are no base features to standardise). With a cached base the decoupled head evaluates
NO part of the base -- block 0 and its `create_graph` traversal leave the training step.

### O4 — the analytic head force (coadvisor 2026-09-13; implemented, gated, measured)

The head's force is `d <X, H0(R)>/dR` at fixed `X`, and autograd was computing it by walking
the assembly graph backwards. The contraction is `O(edges x 16 x 3)` arithmetic, so the
traversal was nearly all overhead. `mace/modules/dscc/analytic.py` writes it out:

* per edge, closed form -- the Slater-Koster block's `dS/dv` and `dS/du`, the descriptor's
  three channels, and the projector `du/dd = (I - u u^T)/r`;
* per node, a small autograd on leaf descriptors through the PRODUCTION `on_site` and
  `directional_site_blocks`, so A1's environment factors and the block layout are not
  transcribed a second time;
* ONE `X` for both consumers. The band term's `dP` and `E_host`'s two Frechet pullbacks share
  `U`, so `X = U (diag(sum_sigma df) + M_S + M_R) U^T` is formed in the eigenbasis and rotated
  out once -- four `n^3` GEMMs where the two separate routes did about ten.

With it the whole head runs under `no_grad` at inference, which closes the coadvisor's fills
item as a side effect: `two_fillings` takes its frontier path and forms no density matrix at
all (8.0 -> 2.4 ms at 159 atoms). **Scope, stated not hidden:** forces only. The stress keeps
the autograd cotangent route -- it needs the strain graph, and it is not on the timed path.
Decoupled heads only: with the base's features in `H0` the derivative runs through the base's
graph and there is nothing to write out.

**Gates.** `test_the_analytic_force_route_matches_the_autograd_route` (same model, same frame,
two derivative routes) 1e-9; compiled vs eager 2e-14; a 5.6 A cell with 30 periodic self-image
edges 2.7e-15 (the assembly overwrites the matrix diagonal with the on-site levels, so a
`src == dst` edge contributes its off-diagonal entries and not its diagonal ones -- masked);
`test_the_analytic_route_forms_no_density_matrix`. Nine more O3 gates were added at the same
time, because **O3 had no test of its own before 2026-09-13**: FD forces, FD stress, batched
== per-graph, training == inference, and "perturbing every base parameter leaves the head
energy bit-identical". Full `tests/extensions/dscc` suite: 211 passed.

**Two O3 training-path bugs the one-epoch smoke found** (both in `Trainer.prepare`, both fatal
to all six seeds): `set_feature_stats` standardised the base's 512 block-0 invariants into
buffers sized for the head's 18-channel descriptor and raised; `saturation_report` fed those
same base features to the head's readouts -- once the shapes agreed that would have been a
wrong number and not an error, so diagnostics now go through `MACEDSCC.readout_scalars`.
Recorded also: under O3 the saturation report still carries a `hop` entry computed from
`sk.hop`, the MLP that `hop_modulation = False` never evaluates. **It is meaningless for O3
and should not be read as the head saturating.**

**A third O3 bug killed the first launch** (2026-09-13 ~20:03 b3 clock, four minutes in, at the
first evaluation batch): the per-graph path -- taken by any batch that mixes a charged frame
with a reference one, or mixes sizes -- slices the features it hands `H0`, and a decoupled head
has none (`scalars[nodes]` on `None`). The uniform path passes them straight through, which is
why every gate and both smokes missed it: the smoke ran `--subset 12`, all one size. Fixed with
explicit placeholders (`MACEDSCC._descriptor_placeholders`; `H0` overwrites them with its own
descriptor), gated by `test_the_per_graph_path_matches_the_batched_one`, and
`set_feature_stats` now refuses a decoupled head loudly instead of raising a shape error.
Runners stopped before the fix and the six seeds relaunched; cost one wave, ~10 minutes.

**Timing (idle A4000, fake O3 weights, P4.3 protocol, both force routes interleaved frame by
frame).** The interleave is necessary: the GPU's clocks drift ~15 % between separate runs, so
the base leg alone reads 38-45 ms across invocations. The neighbour list is NOT static -- 60
(16 at 159) different frames, `torch.compile(dynamic=True)`.

| | base | model, analytic + compiled | ratio | model, autograd | ratio |
|---|---|---|---|---|---|
| 79 atoms | 38.4 ms | 74.0 ms | 1.88 | 74.4 ms | 1.94 |
| 159 atoms | 67.7 ms | 106.5 ms | 1.57 | 114.0 ms | 1.68 |

**A wash at 79 atoms, 7.5 ms at 159.** Why: the kernel's own arithmetic is 2.7 ms of GPU time
at 79 atoms but it issues 542 CUDA kernels eager (262 compiled) and costs ~4.5 ms of launch
latency; `head_cotangent` is another 3.6 ms of the same kind. At 79 atoms that is about the
size of the traversal it removes. The compiled H0 assembly went 3.2 -> 1.1 ms.

**The CUDA-graph item, measured and NOT implemented.** Base forward+backward at 79 atoms:
eager 41.6 ms wall of which **41.4 ms is CPU issue** -- entirely launch-bound; captured as a
CUDA graph, 30.6 ms wall and **0.023 ms of CPU**. So graphing the base both removes 11 ms and
frees 30 ms of CPU time in which the head -- which reads nothing from the base under O3 --
could issue its own kernels. Capture succeeds on this base (openequivariance's JIT kernels do
not refuse it); energies and forces agree with eager to 9e-7, which is the base's own
run-to-run scatter nondeterminism and not an error of the graph. What it needs before it can
be used is edge-bucket padding: capture fixes the shapes, and every frame has a different
edge count, so the neighbour list must be padded to a bucket with dummy edges beyond `r_max`
(zero through MACE's polynomial cutoff). Node counts are already fixed per size class. Not
done; it is the largest remaining lever and it is a day's work with production caveats.

**The `eigh` rule on the deferred-base path, registered with its measurement.** Not a size
threshold: the base is issued first and the head reads nothing from it, so a CPU `eigh` runs
underneath the base's GPU window and costs nothing, while a GPU one is serial work behind that
window. Measured (float64, one matrix, ms): 316 orbitals cuSOLVER 12.4 / MKL 8.6 / base window
45.9; 636 -> 23.6 / 26.0 / 64.2; 1272 -> 84.7 / 101.3 / 127.2; 2544 -> 433.8 / 389.2 / 253.6.
The CPU `eigh` hides completely up to 1272 orbitals and is cheaper than cuSOLVER at 2544
anyway, so the deferred path uses the CPU at every size measured (`EIGH_DEFERRED_ON_CPU`,
`mace/modules/dscc/fill.py`). Note the head-to-head crossover is NOT monotone -- the GPU wins
at 636 and 1272 and loses again at 2544 -- so `EIGH_CPU_MAX_ORBITALS = 512` must not be
generalised into an "above this size, use the GPU" rule.

**Six-seed O3 retrain launched 2026-09-13 ~19:59**: seeds 2, 3, 4, 6 on b3 GPUs 5/6/7
(`~/runs/o3_queue.txt`, runner log `~/runs/o3_queue.log`), seeds 0, 1 on the local A4000
(`~/runs/o3_local_queue.txt`). **b3 GPU 4 is off the bus** ("Unable to determine the device
handle for GPU4: 0000:89:00.0: Unknown Error") -- the same fault GPU 6 had on 2026-09-06,
which a cold power cycle cleared; not cycled, and the run does not need it (three usable cards
of the permitted four, so four seeds run as 3 + 1).

### O5 — CUDA-graph capture (coadvisor ruling 2026-09-13; implemented and gated, timing pending)

The coadvisor's reading of the O4 numbers, adopted: the cost model was the mistake, not the
arithmetic. At these sizes nothing in the model is compute-bound, so the only lever is
eliminating launches -- **for the head as well as the base** -- and the deliverable is
registered in ABSOLUTE terms, **model <= 53 ms per step at 79 atoms** (the 1.2x of the eager
base the criterion was written against), with both ratios reported. Graphing the base makes
the model faster and the ratio harder; milliseconds are what carries across versions.

**Structure** (`mace/modules/dscc/capture.py`):

```
graph A0 : H0 assembly                         ~1 ms of GPU, issued FIRST
D2H      : H to pinned host memory
graph A1 : base forward + backward             ~30 ms of GPU, 0.02 ms of CPU
CPU      : eigh, chemical potentials, occupations -- under A1's window
graph B  : site charges, E_host, E_M, X, the analytic contraction, accumulation
```

The assembly goes first rather than alongside the base as sketched: recorded after it, `H`
would not be ready until the base's window closed, and the host would idle through 30 ms and
then work alone. First, the host diagonalises while the base runs -- which is the
`EIGH_DEFERRED_ON_CPU` rule doing its job inside a captured step.

**Gates.** Reproduces the eager model to **1e-14** in energy, head energy, `dq` and forces; a
frame whose atoms have moved -- a different neighbour list of the same length -- replays
through the same graphs to the same tolerance (only the LENGTH is fixed by capture; the edge
list's values are read at replay). A coupled or base-reading head is refused. Bucket rule:
count x 1.10 rounded up to 512, shrunk only below 0.75 x bucket, so a relaxation drifting by a
few edges keeps one graph. Padding is **bit-identical** (see O4 above). Full suite 225 passed.

**What had to change, all of it removing a device-to-host read or a value-dependent shape, and
none of it changing a number** (the suite is the evidence): `ReciprocalPlan` holds the
cell-dependent half of the Ewald kernel and the eager entry points delegate to the same
arithmetic; `site_occupation` accepts an already-solved fill (`chemical_potential` is a host
Newton iteration and belongs in the CPU segment beside the eigh); `_dk_eigenbasis` applies its
fixed-`N` correction unconditionally (`if bool(ok.any())` was a host read that skipped only
work `torch.where` already makes a no-op); `standardise` mirrors its buffer flag in a Python
attribute; `_trunk_data` can pad the base's subset instead of compacting it; and `head_tail` is
the head's GPU work as a function of tensors alone -- no potential solve, no `nonzero` for the
frontier set (the analytic route needs no `dP`), no lattice sums -- gated against the model's
own path at 1e-14.

**Two capture traps worth recording.** (1) `padded[k].clone()` of a batch that has been through
an eager forward is a NON-leaf: the model calls `positions.requires_grad_(True)` in place, so
the clone carries that forward's graph and capture refuses it ("stale reference to the default
stream"). `.detach().clone()`. (2) A `for` loop's bindings outlive the loop in Python, so the
warm-up's autograd graph -- and its stream reference -- is still alive at capture; deleted
explicitly.

**Open:** the analytic stress (needed for NPT and nothing else, a follow-up not a blocker), and
the timing, which waits for the local A4000 to come free -- every registered P4.3 number is on
that card and the six-seed retrain is using it.

**O5 TIMING (idle A4000, fake O3 weights, P4.3 protocol, `dscc_benchmark.py --graphed`).**

| | base eager | base captured | model eager | model graphed | model / eager base | model / captured base |
|---|---|---|---|---|---|---|
| 79 atoms, fixed cell | 40.4 ms | 32.4 ms | 63.6 ms | **39.7 ms** | 1.08 | 1.23 |
| 159 atoms (registered set) | 67.2 ms | 67.3 ms | 100.6 ms | **89.0 ms** | 1.33 | 1.32 |

**The registered deliverable is met at 79 atoms: 39.7 ms against <= 53 ms.** Both ratios are
quoted because graphing the base moves the denominator, exactly as the coadvisor said it would.

*Why the eager column differs from O4's* (79: 63.6 here against 74.0; 159: 100.6 against 106.5).
Different trajectories, not a code change. This table's 79-atom row is a FIXED-CELL trajectory,
so the eager path's reciprocal-vector cache (`_CELL_CACHE`) hits on every frame; O4's row walked
the registered NPT frames and rebuilt the `k` set per frame. Against O4's 74.0 the graphed model
is a 34.3 ms gain, of which 10.4 ms is the cell cache and 23.9 ms the capture. The 159-atom
difference is within the card's run-to-run drift.

**At 159 atoms the captured base is within noise of the eager one** (67.3 against 67.2 -- inside
the card's ~15 % drift, so this pair alone proves nothing; what carries the conclusion is the
79-atom split below). The mechanism is the launch-bound-to-compute-bound crossover: at 79 atoms
the base is 41.6 ms of wall
time on 30.6 ms of GPU work, so removing its 41.4 ms of CPU issue is the whole win; at 159 the
kernels are large enough that the GPU is the constraint and there is no CPU time left to
remove. The model still gains (100.6 -> 89.0) because the HEAD is launch-bound at both sizes.

**Capture is a FIXED-CELL path.** The reciprocal plan, the Madelung constant `xi` and the
padding translation are constants of the cell, and the formal charge enters `E_M` directly, so
`GraphedStep.fits` now refuses a frame whose cell, composition, charge state or node count
differs -- a ~50 us host comparison against a silently wrong answer. This was not hypothetical:
the registered 79-atom P4.3 frames are NPT snapshots whose cells span 15.76-16.07 A, and
replaying them through one capture read **5.5 meV in energy and 9.3 meV/A in force** with
nothing raised. The 79-atom row above therefore uses a fixed-cell trajectory (the first frame
plus 0.02 A thermal displacements, `--fixed_cell`), which is what a relaxation or an NVT/NVE
step actually is; the 159-atom set shares one cell already and is used as registered. NPT pays
a re-capture per cell: **0.25 s at 79 atoms, 0.38 s at 159**, so a barostat that updates the
cell every few hundred steps is fine and one that updates every step is not.

**Agreement on the PRODUCTION base** (float32, with its ZBL pair repulsion): energy 2e-5 to
8e-5 eV, forces <= 5e-6 eV/A over six real frames at each size. That is the float32 base's own
reproducibility under a changed scatter order -- padding reorders the edge list -- and not an
error of the capture: the same base evaluated twice unpadded moves by 1.6e-7 eV and
1.6e-6 eV/A, and padded-vs-unpadded by 2.9e-6 eV. On a float64 toy base the captured step
reproduces the eager model to **1e-14**. Against `tau_phys` = 1.7 meV/A per component, 5e-6 eV/A
is four orders of magnitude clear.

**O3 TURN-ON READING (first four seeds, 2026-09-13) -- the registered check, and it reads the
OPPOSITE way from the failure it was written to catch.**

| | on-site shift p95 | max | fraction `|tanh| > 0.95` | held-out force RMSE |
|---|---|---|---|---|
| O3 s0 / s2 / s3 / s4 | 2.970 / 2.974 / 2.970 / 2.970 eV | 2.977-2.981 | **0.398-0.399** | 0.0323 / 0.0318 / 0.0353 / 0.0319 eV/A |
| W6 s0 / s1 / s2 / s3 / s4 / s6 | 0.465-1.147 eV | 0.538-1.314 | **0.000** | (shipped) |

The check was registered "so that 'forces equivalent' and 'the term never activated' cannot be
confused". The term did not fail to activate: **it is pressed against its stop on two atoms in
five**, at a p95 shift of 2.97 eV against a half-width `Delta_Z` of 3.079 eV, where the shipped
W6 cohort never came within a factor of two of the bound and reads a saturated fraction of
exactly zero. The four seeds agree to the third decimal, so this is the architecture and not a
seed.

Reading it precisely: the A1 barrier sits at `|pre| = 2.0` and `|tanh(2.0)| = 0.9640`, which is
past the `SATURATED = 0.95` counter by construction, so "40 % saturated" means 40 % of atoms sit
AT the barrier knee, not beyond it -- the logged barrier term (mean `relu(|pre| - 2)^2` about
5.8e-6) puts them within ~4 meV of pre-activation past it. The barrier is doing its job and it
is BINDING on 40 % of sites. A parameter at its stop has `sech^2 ~ 0`: this project has already
paid once for reading such a parameter as "learning" (the `eta` ruling, 2026-09-10).

Most likely cause, and it is a known deferral: O3 feeds the production readout the RAW
species-resolved `l = 0` basis. A1.1's per-species standardisation was deliberately excluded
from this run under one-change-at-a-time (see the O3 registration above), so the readout sees
whatever scale `sum_j sin(n pi x) f_cut / r` happens to have. The follow-up arm is therefore
the standardisation path, not the three-body screening -- but that is a SECOND change and the
adoption gates are decided on forces first.

Not a blocker for the gates; recorded before they are read, as the plan requires.

**Also recorded:** `c4_stage_b_extras.py` reads the audit's `hop` pre-activations, which an O3
head never produces; it now says so instead of raising `KeyError`.

### O3 record additions and the O3.1 arm (coadvisor ruling 2026-09-13, evening)

**Capture is a fixed-cell path, and NPT runs eager until further notice.** `fits()` refuses a
changed cell, composition, charge state or node count. Re-capture at 0.25 s per step would
dominate an NPT trajectory, and a drift threshold would reintroduce the staleness in a smaller
dose rather than remove it; **the follow-up is to make the cell a graph INPUT** -- the
reciprocal plan, `xi` and the padding translation computed inside the graph, with the G-vector
count bucketed by volume range. Until that exists, NPT does not capture.

**The production base's float32 reproducibility under edge reordering is 2e-5 to 8e-5 eV and
<= 5e-6 eV/A. That is a NOISE FLOOR, not a head error** -- it is the same base evaluated twice
with its edge list in a different order, which is what padding does -- and it is four orders
below `tau_phys`. Recorded here so it is not later read as a cost of the capture.

**O3 SATURATION -- and why it is disqualifying.** 40 % of sites at the on-site barrier, the
four seeds identical to the third decimal, so it is the architecture. Measured by species and
sign on 30 charged frames, the trained O3 correction and its WITHIN-SPECIES spread:

| | O3 s0: `corr_s` | sd | `corr_p` | sd | frac `pre > 1.83` / `< -1.83` |
|---|---|---|---|---|---|
| Cl (1410) | +2.672 | **0.028** | +2.952 | **0.008** | +0.498 / 0.000 |
| Pb (480) | +2.228 | 0.070 | +2.957 | 0.012 | +0.492 / 0.000 |
| Cs (480) | -0.391 | 0.220 | +2.188 | 0.097 | 0.000 / 0.000 |

| | W6 s1: `corr_s` | sd | `corr_p` | sd | frac |
|---|---|---|---|---|---|
| Cl / Cs / Pb | -0.018 / -0.024 / -0.083 | 0.060 / 0.027 / 0.069 | +0.539 / +0.471 / +0.624 | 0.081 / 0.081 / 0.076 | 0.000 |

The coadvisor's prediction holds exactly: **one sign, and two species** (Cl and Pb; the
saturated channel is the p shell, whose 2.95 eV is `|tanh| = 0.959` against the s shell's
0.868). The environment-dependent part of the correction is **0.3-1 % of its magnitude** for Cl
and Pb, against 10-15 % for the shipped W6 head. The term has degenerated into a per-species
constant -- the mode `eps0` already parameterises, and which A1 records as "redundant by one
constant per species". **This is the A1 failure inverted**: there the inputs were uncentred and
the corrections never turned on; here the centre is wrong and the scale is raw, and they slam
to the stop carrying no environment with them.

**O3 is therefore NOT ADOPTABLE regardless of the paired TOST** (the plan's Stage-2
no-systematic-saturation rule): a bound binding on 40 % of sites has no gradient there, so the
trained head is not at an optimum. The O3 gates still run, because they answer a different and
worth-answering question -- whether the descriptor swap itself is harmless.

**What A1.1 would freeze for O3's `l = 0` input**, measured on the same 30 frames (and
confirmed by the O3.1 trainer over its own 92 setup frames, which reads 4.9 / 6.0 / 8.6):

| species | atoms | median abs mean | median sd | median `|mu|/sd` | max | dead channels |
|---|---|---|---|---|---|---|
| Cl | 1410 | 0.713 | 0.096 | **4.8** | 66.1 | 0/18 |
| Cs | 480 | 1.177 | 0.086 | **6.1** | 84.4 | 0/18 |
| Pb | 480 | 1.073 | 0.075 | **8.0** | 53.8 | 0/18 |

Every channel is alive -- nothing would be zeroed -- and the species-constant component is 5 to
8 times the ensemble deviation in the median channel and up to 84 times in the worst. That is
the whole mechanism: the readout sees a large per-species offset, saturates on it with one
sign, and the environment signal (a fifth to an eighty-fourth of the input) is squashed by the
tanh's vanishing slope at the rail.

**O3.1 LAUNCHED 2026-09-13 21:27** (b3 clock), six seeds, folds as O3: the same descriptor, the
same readout, **standardised input** (ensemble mean and sd per channel per species, frozen once
from the training frames) and the readout final layer at A1.1's registered **0.1** rather than
the 0.05 the pre-A1.1 campaign, W6 and O3 all ran on raw inputs. Everything else as O3. The
turn-on diagnostic is read against BOTH W6 (0.47-0.62 eV p95, no saturation) and O3 (at the
stop). **The three-body bond screening stays deferred** -- the O3 failure is not a capacity
shortfall. Seeds 2, 3, 4, 6 on b3 GPUs 5/6/7 (`~/runs/o31_queue.txt`), 0 and 1 on the local
A4000. `--standardise` and `--readout_init_scale` are explicit flags, so which form a
checkpoint carries is a recorded fact and not an inference.

**Training step time, like for like on the local A4000** (the block-0 `create_graph` traversal
leaving the step): **O3 32 s/epoch against W6's 124 s -- 3.9x** (on b3, 29 s against 170 s
median, 5.9x; the b3 cards differ and some W6 runs shared one). Route B' (W5) is 182 s locally
and 330 s on b3, so O3 trains 6-11x faster than the shipped B' cohort. A six-seed O3 wave
finishes in about an hour where W6's took a night.

### O3 — G1 (2026-09-13, all six seeds). **FAIL: INFERIOR by 4.4 tau.**

| seed (fold) | W6 | O3 | d = O3 - W6 |
|---|---:|---:|---:|
| 0 (0) | 11.22 | 18.82 | +7.60 |
| 1 (1) | 11.62 | 19.71 | +8.09 |
| 2 (2) | 11.58 | 18.55 | +6.97 |
| 3 (3) | 11.23 | 20.48 | +9.25 |
| 4 (0) | 12.02 | 18.53 | +6.51 |
| 6 (2) | 11.84 | 18.79 | +6.95 |
| **median** | **11.60** | **18.81** | |

meV/A per component (the plan's convention: the trainer logs the per-atom force-error VECTOR
RMS, `sqrt 3` times this). The W6 median of 11.60 reproduces the registered W6 table's 11.58,
which confirms both the cohort and the convention.

**Paired TOST, `tau` = 1.7, `d = W6 - O3`: mean -7.563, se 0.407, 90 % CI [-8.384, -6.743] ->
INFERIOR.** Every seed the same sign; the interval is four standard errors clear of the
equivalence band. **O3 is worse than the shipped W6 head by 7.56 meV/A per component, 4.4 x
tau.**

**CORRECTION OF RECORD (mine).** An earlier entry in this section read "no worse", comparing
O3's 0.0318-0.0353 eV/A against "the shipped cohort's 0.0329 median". That 0.0329 is the
**Arm 2+3** figure for `B_Bp_lr_only` -- a different campaign, base and dataset -- quoted in the
trainer's per-atom-vector convention against W6 numbers that were never in it. The shipped W6
cohort is 0.0194-0.0208 eV/A, i.e. 11.58 per component, as the W6 RESULT table above always
said. The inference drawn from the wrong comparison -- that the on-site term's ENVIRONMENT
dependence "carries little at this accuracy" -- is **withdrawn**: it was built on a number that
did not belong to the comparison.

**What this changes.** The coadvisor's reason for letting the O3 gates run was that they answer
whether the descriptor swap is harmless. They answer it: **it is not.** O3 is disqualified twice
over -- on forces by G1, and on the Stage-2 no-saturation rule -- and the two are not
independent, since an on-site term pinned as a per-species constant is capacity removed. Note
O3 changes TWO things against W6 (the head-owned descriptor, and the hopping modulation
deleted) and saturates as well, so the 7.56 meV/A is not attributable to one cause. O3.1 tests
only the saturation half: it now has a 7.56 meV/A gap to close, not merely a bound to come off.
If it closes most of it, the descriptor is sound and the saturation was the cost; if it closes
little, the head-owned basis or the deleted modulation is carrying the loss and the next arm
is one of those, one at a time.

**O3 IS EXCLUDED FROM FURTHER ANALYSIS (user, 2026-09-13).** It is disqualified twice -- G1
INFERIOR by 4.4 `tau`, and the Stage-2 no-systematic-saturation rule -- so its numbers are a
record of what the unstandardised descriptor did, not a comparator. Comparisons are against the
shipped **W6** cohort (11.58 meV/A per component) and, once it lands, **O3.1**. The O3 runs
themselves are complete; nothing was left running.

**Card packing, measured rather than inherited.** The `PER_GPU = 1` rule in `w3_queue.sh` is the
W6/B' number: their training step peaks at 11.6 GB because the base's block 0 is recomputed
under `create_graph`. A decoupled head does not recompute it -- an O3.1 job measures **4.7 GB
steady on a 24.5 GB card, with the cards at 51-65 % utilisation running one job each**, i.e.
launch-bound and half idle, which is the same diagnosis as the inference path. The O3.1 wave
therefore runs **two per card** (`W3_PER_GPU=2`, `W3_HEADROOM=8000`), which starts the last two
seeds immediately instead of queueing them behind a 30-minute run. Neither number transfers to
a base-reading head.

### O3.2 — three-body bond screening: REGISTRATION (2026-09-13, written before any O3.2 result)

Opened on the user's instruction while the O3.1 wave is still running, on the trend rather than
on its gates: O3.1's held-out force is tracking ~14.5-15 meV/A per component against the shipped
W6's 11.58, so it recovers most of O3's 7.56 deficit and not all of it. **The O3.1 gates remain
the arbiter of O3.1**; this arm is queued behind them, not in place of them. The residual is
consistent with the one capability O3 deleted and O3.1 did not restore: the hopping's environment
dependence. O3 removed the MLP modulation (`hop_modulation = False`); this restores it in the
form the plan registered as the single follow-up arm,

    S_ij = sum_k g(r_ik) g(r_jk) P(cos theta_ikj)

with `theta_ikj` the angle AT k. Four specifics were left open by that line. Registered now:

1. **`g(r) = (1 - (r/r_s3)^6)^2` inside `r_s3`, zero outside** -- H0's own taper, so the head
   keeps one smoothness class (C^1) across hoppings, descriptor and screening alike.
2. **`r_s3 = 4.5 A`**, the registered `Q_CUT_DEFAULT`, and for the same reason it was registered
   there: it reaches the first coordination shell and the Cs neighbours (Pb-Cl 2.8, Cs-Cl
   3.5-4.1 A) and not the second Pb shell. A bond longer than `2 r_s3 = 9 A` therefore has no
   screener at all and `S = 0` exactly, which is a property of the cutoff and not an
   approximation.
3. **`P_l`, Legendre, `l = 0, 1, 2`, species-resolved in `k`** -> 3 x 3 = **9 channels per
   bond**. Species-blind screening would treat a Cs and a Cl third atom alike, which the
   on-site term's own species dependence says is wrong; `l <= 2` matches the head's existing
   angular content (the rank-2 on-site block).
4. **It enters through the EXISTING bounded modulation**: `pre_ij = A[Z_i, Z_j] . S_ij` with `A`
   a `[n_el, n_el, 9, 4]` linear map (one pre-activation per Slater-Koster integral type), then
   `t_ij = v0 . radial(r) . exp(eta tanh(pre))` through the `modulation()` already in the head.
   **No MLP** -- the point of O3 was to remove the readout from the hopping, and this does not
   put it back; it replaces a learned function of base features with a learned contraction of a
   geometric three-body invariant.

**`A` is ZERO-INITIALISED, so the arm starts EXACTLY at O3.1.** That is the entry gate: with
`A = 0` the energies and forces must equal O3.1's bit for bit. It also means O3.2 is one change
from O3.1 and two from O3, so a failure is attributable.

**Symmetry.** `S_ij = S_ji` by construction (`g(r_ik) g(r_jk)` is symmetric in `i, j` and the
angle at `k` does not depend on which end is named first), which is what keeps `H` Hermitian by
construction rather than by symmetrisation -- the property `sk_block` exists to preserve. Gated
at 1e-12 on the reversed edge.

**Known hole, registered rather than hidden: the analytic force kernel does not cover this.**
`h0_position_gradient` writes `d H0/dR` in closed form assuming `t_ij` depends on `r_ij` alone;
with screening it depends on `R_k` for every third atom, a genuine three-body force term. For
this arm `analytic_head_forces` is **off** and the head falls back to the autograd cotangent
route at inference (gated against finite differences). Extending the kernel -- `S` as a leaf in
the small per-node autograd, chained analytically through the triplet geometry, the way the
on-site readouts already are -- is a follow-up and is not on the training path, which never used
the analytic route.

**Gates, paired by fold against O3.1 and against the shipped W6**: as registered for O3 (force
TOST at `tau` = 1.7 per component, energy RMSE at 79 within 0.10 meV/atom, between-size after
one `C_Q` <= 10 meV, neutral null and FD forces re-run), plus the turn-on reading, plus the
screening's own diagnostic: **the distribution of `|pre|` and the fraction beyond the barrier
knee**, so that the modulation cannot be reported as "learned" while sitting at its stop, which
is the mistake O3 made with the on-site term.

**O3.2 registration, amendment (2026-09-13, still before any O3.2 result).** Four things the
implementation fixed that the four specifics above left implicit. Written now because the rule
is that the form in the record is the form that RAN, and the first held-out number lands at
epoch 5:

1. **`S` carries a fixed `1/N_ref` with `N_ref = 6`**, the octahedral coordination, so a full
   first shell gives `S ~ O(1)`. It is absorbed exactly by `A` and changes nothing about what
   the arm is; what it sets is Adam's effective step on `A` and the scale at which `|pre|` is
   read against the barrier knee.
2. **The screening's pre-activations go into the SAME `_reg_bin` as W6's `hop` did**, so they
   carry both the registered L2 (`mean tanh^2`, weight 1.8e-6) and the one-sided barrier
   (`mean relu(|pre| - 2.0)^2`, weight 1e-3). "Enters through the existing modulation" means
   the existing regularisation too; a modulation with a bound and no barrier would be the one
   thing A1 exists to prevent.
3. **`A` is symmetrised in the species pair at use** (`0.5 (A + A^T)`), not merely initialised
   symmetric. `S_ij = S_ji` alone does not make `t_ij = t_ji`.
4. **Two refusals, so neither can fail silently.** CUDA-graph capture refuses a screened head
   (the triplet list is built with `nonzero`, so the shapes are data dependent and a replay
   would reuse the captured frame's triplets), and the sparse/windowed assembly refuses it
   (it evaluates the two-body `integrals` and would drop the screening without a word). Both
   are registered follow-ups; neither is on the training path. The analytic-force refusal and
   the capture refusal ask `sk.screen is not None`, NOT the `screened` flag -- the flag
   travels in the extra state, so loading an unscreened parent's `state_dict` clears it while
   the module stays live, and a gate on the flag would then run the closed form on a screened
   Hamiltonian with wrong forces and no error. Gated.

**Cost, measured before launch on the local A4000, same data and same card as O3.1's
production run: 35 s/epoch against O3.1's 28 s -- 1.25x.** ~70k triplets at 79 atoms. The
smokes: 2 epochs at `--subset 60` and at `--subset 300` clean, then ONE FULL epoch on the real
held set (258 frames at 79 atoms and 4 at 159) because a `--subset` held split is all one size
-- which is precisely the blind spot that killed O3's first launch four minutes in. Suite: 254
passed, 0 failed, including 22 new gates in `tests/extensions/dscc/test_screening.py`.

### O3.1 RESULT (wave complete 2026-09-13 22:31 b3 clock, six seeds, all six finished clean)

**G1, the registered adoption gate.** Paired by seed/fold, `tau` = 1.7 meV/A per component,
the trainer's per-atom force-error VECTOR RMS divided by `sqrt 3`:

| seed (fold) | W6 | O3 | O3.1 | O3.1 - W6 | O3 - O3.1 |
|---|---:|---:|---:|---:|---:|
| 0 (0) | 11.22 | 18.82 | 14.59 | +3.37 | +4.23 |
| 1 (1) | 11.62 | 19.71 | 14.22 | +2.60 | +5.49 |
| 2 (2) | 11.58 | 18.55 | 15.06 | +3.48 | +3.49 |
| 3 (3) | 11.23 | 20.48 | 14.82 | +3.59 | +5.67 |
| 4 (0) | 12.02 | 18.53 | 14.28 | +2.26 | +4.25 |
| 6 (2) | 11.84 | 18.79 | 14.17 | +2.33 | +4.62 |
| **median** | **11.60** | **18.81** | **14.44** | | |

- **vs W6** (`d = W6 - O3.1`): mean **-2.937**, se 0.247, 90 % CI [-3.436, -2.439] ->
  **INFERIOR**. Every seed the same sign. O3.1 is worse than the shipped head by **2.94 meV/A
  per component, 1.7 x tau**. **O3.1 IS NOT ADOPTABLE.**
- **vs O3** (`d = O3 - O3.1`): mean **+4.626**, se 0.338, 90 % CI [+3.945, +5.307] ->
  **SUPERIOR**, 2.7 x tau.

**This answers the question O3.1 was registered to ask, and it answers it in the middle.** The
registration said: "If it closes most of [O3's 7.56 meV/A gap], the descriptor is sound and the
saturation was the cost; if it closes little, the head-owned basis or the deleted modulation is
carrying the loss." **A1.1 standardisation closed 4.63 of 7.56 -- 61 %.** So the saturation was
the larger half and the descriptor is not the problem; **2.94 meV/A is still unaccounted for,
and the one capability O3 deleted and O3.1 did not restore is the hopping's environment
dependence.** That is what O3.2 puts back, and it is the arm already queued.

**G2a, energy RMSE at 79 atoms** (margin 0.10 meV/atom, one-sided 95 %): mean `d` **-0.058**,
se 0.017, upper bound **-0.025** -- O3.1 is *better* than W6 on energies at every seed.
**PASS.** Worth stating plainly because it separates the two channels: the decoupled head is
not globally worse, it is worse **on forces**.

| seed (fold) | 0 (0) | 1 (1) | 2 (2) | 3 (3) | 4 (0) | 6 (2) |
|---|---:|---:|---:|---:|---:|---:|
| W6 | 0.4752 | 0.5739 | 0.5119 | 0.4996 | 0.5183 | 0.5373 |
| O3.1 | 0.4486 | 0.4618 | 0.4920 | 0.4074 | 0.4403 | 0.5171 |

**G2b, between-size after one `C_Q`: NOT RESOLVABLE on this split, and the estimator is what
fails, not the arm.** Taking the difference of the two sizes' fitted `C_Q`, O3.1 spans
-15.1 to +12.1 meV against the `|.| <= 10` gate -- but the SHIPPED W6 cohort spans -0.4 to
+20.5 meV on the same estimator and the same frames. The held split carries **4 frames at 159
atoms against 258 at 79**, so the 159 intercept is a four-point mean; neither cohort's spread
is a statement about between-size extensivity at this sample size. Recorded as unresolved
rather than scored either way. Fixing it needs a held split with a usable 159 count, which is
a change to the data plan and is not made silently here.

**Turn-on: the saturation is GONE, which is what A1.1 was for.** Median over the six seeds of
the on-site term: saturated fraction **0.000** (O3's was 0.40, i.e. 40 % of atoms AT the
barrier knee), p95 shift **0.740 eV** against W6's 0.846 and the half-width `Delta_Z` = 3.079,
max shift 1.595 eV against W6's 0.923. So the correction is live, of the same order as the
shipped head's, and nowhere near its bound. The Stage-2 no-systematic-saturation rule, which
disqualified O3 independently of G1, is **satisfied**.

**Reading, stated as what it is.** Standardising the input fixed the saturation completely and
bought 61 % of the force gap. The remainder is not a saturation problem and not an energy
problem; it is specific to forces, which is where a two-body hopping with no environment
dependence would be expected to lose -- the force is where the angular structure of the bond
enters. O3.2 is the test of that reading and it is running.

**O3.2 LAUNCHED 2026-09-13 22:31** (b3 clock), six seeds, the same folds as O3 and O3.1, every
flag identical to the O3.1 lines plus `--screened 1 --r_s3 4.5`, so the arm is ONE change from
O3.1 and two from O3. Seed 0 on the local A4000 (`~/runs/o32_local_queue.txt`), seeds 1, 2, 3,
4, 6 on b3 GPUs 5/6/7 at two per card (`~/runs/o32_queue.txt`, `W3_PER_GPU=2`,
`W3_HEADROOM=8000`). At 35 s/epoch the wave is about 35 minutes per seed.

**Operational notes, recorded because both have bitten before.** (i) b3's GPU 4 reported
`Unable to determine the device handle ... Unknown Error` during the changeover, so the runner
was started on cards **5 6 7** only; a faulted card has previously killed the driver context of
every job on the machine, not only its own. (ii) The default queue path `~/runs/w3_queue.txt`
still held six stale `dscc_w3_bp_lr_only_*` lines from 2026-09-10, which a runner started with
default settings would have launched ahead of this wave. It is renamed to
`w3_queue_stale_2026-09-10.txt` so it cannot fire, and O3.2 uses its own queue file.

**CORRECTION OF RECORD, O3.1 G2b (mine, same evening, before O3.2's first held number).** The
entry above called G2b "not resolvable" on the ground that the per-seed spread is wide. That
was my own reading substituted for the registered one. The registered reading is the **cohort
median** of `C_Q(159) - C_Q(79)`: the W6 median on this estimator is **+5.0 meV**, which
reproduces the W6 adoption table's +5.0 exactly, so it is the same estimator and the same
reading. Scored like for like: **O3.1 median -1.8 meV, |.| <= 10 -> PASS.** The wide per-seed
spread (O3.1 -15.1..+12.1, W6 -0.4..+20.5, on 4 held frames at 159 atoms) stands as a caveat on
the per-seed numbers, not as a reason to decline the gate. **O3.1: G1 INFERIOR, G2a PASS,
G2b PASS, turn-on PASS.** It fails on forces alone.

**Where the 2.94 meV/A sits -- by shell, cohort medians, meV/A per component.** The registered
localisation columns, which the G1 table above omitted:

| shell (A) | n | W6 | O3.1 | O3.1 - W6 |
|---|---:|---:|---:|---:|
| 2-4 | 1991 | 19.83 | 25.54 | +5.71 |
| 4-6 | 2877 | 13.81 | 17.49 | +3.68 |
| 6-8 | 8448 | 9.73 | 12.62 | +2.89 |
| 8-10 | 5578 | 9.86 | 11.11 | +1.25 |
| 10-12 | 1768 | 9.02 | 10.48 | +1.46 |
| **flanking Pb** | | **27.40** | **37.41** | **+10.02** |
| first-shell Cl | | 16.55 | 20.56 | +4.02 |
| other 2-4 | | 10.26 | 11.85 | +1.59 |

**The deficit is not uniform: it is concentrated exactly where a bond is.** The flanking Pb
carries +10.0 of it, the 2-4 A shell +5.7, and it decays monotonically to +1.3 by 8-10 A. A
hopping with no environment dependence is a two-body function of `r_ij` alone, and the atoms it
should hurt most are the ones whose bonds are screened by a third atom -- the near neighbours of
the vacancy. **Registered as the O3.2 expectation, before O3.2's first held number:** if the
three-body screening is the right diagnosis, its gain should be concentrated in the same places
-- the flanking Pb and the 2-4 A shell -- and largely absent beyond 8 A. If instead O3.2 gains
uniformly across shells, or gains at long range, the reading is wrong whatever G1 says, and the
next arm is the head-owned basis rather than the modulation.

**Two precisions on the O3.1 gate list.** The registered "neutral null and FD forces re-run"
were run in the unit suite on toy weights (255 passed, `tests/extensions/dscc`), not on the
trained checkpoints; that is what the phrase covers here. And O3.2's epoch-0 loss is NOT
expected to equal O3.1's -- `A` takes gradients from the first step -- the bit-for-bit property
is the first forward at `A = 0`, which is gated in `test_screening.py`, not an epoch aggregate.

## W7 — base v3: MACE-MH-1's architecture trained FROM SCRATCH, plain and with LES

*(User instruction, 2026-09-13 night: "retrain the base using the same configuration (width,
non linear blocks etc) as the MACE-MH-1 model we fine tuned earlier but this time from scratch
... follow the mace documentation to get the best strategy. Secondly, train a MACE-LES model
using the same configuration also from scratch ... report the errors using the same error table
as the last time we trained base v2.")*

**Registered before the runs (2026-09-13 23:00).** Two models, one change between them:

| | `base_v3_scratch` | `base_v3_les` |
|---|---|---|
| model class | `ScaleShiftMACE` | `MACELES` (latent Ewald summation) |
| everything else | identical | identical |

**Architecture = MH-1's, read off the checkpoint rather than retyped.**
`extract_config_mace_model` on `~/.cache/mace/macemh1model` gives the values, and every one is
passed explicitly: `r_max` 6.0, 10 Bessel, polynomial cutoff p = 5, `max_ell` 3, TWO
`RealAgnosticResidualNonLinearInteractionBlock`s, hidden `512x0e+512x1o`, edge irreps
`128x0e+128x1o` with `use_edge_irreps_first`, element-agnostic product, correlation 3,
`MLP_irreps` `16x0e`, silu, Bessel radial with radial MLP `[64, 64, 64]`, Agnesi distance
transform, ZBL pair repulsion, `apply_cutoff` False. **The match is verified by construction,
not asserted**: the saved model is re-read with the same function and diffed against MH-1's
config; the only permitted differences are the ones a single-material three-element model must
have (`num_elements` 3 not 89, one head not six, the atomic energies, and the scale/shift and
`avg_num_neighbors` refitted from this data). The first smoke FAILED that diff on
`apply_cutoff` (MH-1 False, the code's default True) and it is now passed explicitly — which is
the reason the diff exists.

**Training = the README's from-scratch recipe, NOT base v2's fine-tune settings.** lr 1e-2,
amsgrad, `ReduceLROnPlateau`, EMA 0.99, grad clip 10, Stage Two over the last 20 % of epochs,
batch 8, float32, cuEquivariance, seed 1. Loss `stress` (energy 1 / forces 100 / stress 1;
Stage Two 1000 / 100 / 10) because the labels carry `REF_stress` and the base v2 table has a
stress column — the documented from-scratch weighting plus the label the data already has, not
a hybrid with base v2's 10 : 10 : 1 fine-tune weights.

**E0s: the SAME estimated isolated-atom energies base v2 used** (Cl −0.7159, Cs +0.5417,
Pb −0.1022 eV), carried as `IsolatedAtom` frames in `train_iso.xyz` at `config_weight` 1000 and
kept as training points. They came from the foundation model's residuals, so they are a
constant reference and not DFT — used here precisely because sharing base v2's reference is
what makes the energy columns comparable. Data: `dataset_e0/train_iso.xyz` (1619 = 1616 neutral
+ 3 isolated atoms) and `valid.xyz` (154), the same files and the same split as base v2. **No
new DFT, neutral frames only**, as the programme requires.

**One code change, made deliberately and recorded.** `model_script_utils.py` built `MACELES`
with `atomic_inter_shift = [0.0] * len(heads)` while the `ScaleShiftMACE` branch uses the
fitted `args.mean`. Under `rms_forces_scaling` that shift is the training set's mean per-atom
interaction energy, so discarding it starts the model that far from its target: **measured on
this data, the LES smoke's initial validation energy RMSE was 1870 meV/atom against the plain
model's 205 for the same architecture.** That is a handicap the LES term has nothing to do
with, and it would confound the comparison the user asked for at equal epochs. The MACELES
branch now passes `args.mean`, as its sibling does.

**Epoch budget, and the deviation from the documented heuristic, registered before launch.**
The README's rule of thumb is ~200 k gradient updates (1619 frames / batch 8 × 1000 epochs =
202 k). Measured epoch time on b3 at 512 channels is ~84 s WITH an O3.2 job co-resident, so
1000 epochs is ≈ 23 h and does not fit one night on one card. The runs are therefore set to the
largest epoch count that completes by morning, with Stage Two kept at 80 % of it, and the
shortfall against the heuristic is recorded here rather than discovered in the morning. A
from-scratch 512-channel model on 1619 frames is not expected to match a fine-tune of a 89-element
foundation model; **if it does not, that is the finding**, not something to be tuned away.

### W7.1 — base v3 LES-FT: MACE-LES from the MH-1 weights, DIRECT fine-tuning

*(User instruction, 2026-09-13 night: "one more base model which is more experimental. MACE-LES
model but initialised with the MACE-MH-1 weights and direct Finetuning (not multi head)
following the guidance here: mace-docs finetuning_guidance".)*

**Registered before the run.** `--model MACELES --foundation_model <mh-1> --foundation_head
omat_pbe --multiheads_finetuning False`, which the code resolves to `FoundationMACELES`: MH-1's
architecture and weights, plus LES's latent-charge readouts. `load_foundations_elements`
transfers the backbone and the ENERGY readouts and does not touch `les_readouts` — those are
new modules and are learned here, which is the intended division of labour.

**Settings taken from the guidance page's NAIVE column, which is a different recipe from the
multihead one base v2 used:**

| | naive / direct (W7.1) | multihead replay (base v2) |
|---|---|---|
| learning rate | **1e-3** | 1e-4 |
| EMA decay | **0.999** | 0.9999 |
| grad clip | 1.0 | 1.0 |
| weight decay | 0.0 | 0.0 |
| loss weights | constant 10 : 10 (: 1 stress), no stage two | constant 10 : 10, no stage two |
| replay | **none** | 10k fps/combinations pseudolabels |

**Epochs: 60, with best-validation selection.** The guide gives 10–30 for multihead and is
silent for naive. It is also explicit that naive fine-tuning has the worst MD stability and the
greatest drift from the pretrained distribution of the methods it compares — so the epoch
trajectory is reported, and the minimum and any rise past it are put in the record rather than
assumed. Base v2's own multihead run showed exactly that shape on its replay head (minimum at
epoch 8–11, back to the epoch-0 level by 29) while its Default head still improved.

**E0s and data: the same as every other base here** — `dataset_e0/train_iso.xyz` (1619) and
`valid.xyz` (154), the estimated isolated-atom energies carried as `IsolatedAtom` frames at
`config_weight` 1000 and kept as training points. That is also the guide's own advice for
models with learnable E0 biases ("include true isolated-atom configurations as explicit
training points"), and it keeps the energy column comparable across all four bases.

**One thing to watch, registered before it is read.** The LES readouts start from a fresh
initialisation, so the latent charges — and therefore the Ewald term — start RANDOM on top of
an already-good backbone. If the initial validation energy error is large, the guide's ">500
meV/atom means an E0 mismatch" rule does NOT apply here: the E0s are shared with three other
bases that start at ~200 meV/atom, so a large number would be the untrained latent charges and
the remedy would be to zero the LES readouts' final layer (the model then starts exactly at
MH-1 and learns the long-range term from zero, the same "starts at its parent" discipline O3.2
uses). Which of the two it is will be measured in the smoke, not guessed.

**W7.1 smoke: two real bugs, both fixed before the run, both recorded.**

1. **`MACELES` could not be fine-tuned from a plain foundation at all.**
   `load_foundations_elements_default` walks the NEW model's children and looks each one up on
   the foundation by name; `MACELES` adds `les` and `les_readouts`, which a `ScaleShiftMACE`
   foundation does not have, so the loader raised
   `AttributeError: 'ScaleShiftMACE' object has no attribute 'les'` before any training. Fixed
   with a `hasattr` guard that skips modules the fine-tuned model ADDS — there is nothing to
   transfer for them; they are new and are learned.

2. **The latent-charge readouts started random on top of an already-good backbone**, exactly
   the case flagged in the registration above. Measured: **initial validation error 8332
   meV/atom and 1474 meV/Å**, against ~44 meV/Å for the same backbone without LES. The
   guide's ">500 meV/atom means an E0 mismatch" rule does not apply — those same E0s give
   ~200 meV/atom from scratch and 1.69 meV/atom for MH-1 itself. Remedy as registered: the
   `FoundationMACELES` path now ZEROES the last layer of each latent-charge readout, so the
   model starts exactly at the foundation and learns the long-range term up from zero.
   Gradients still flow into that layer; nothing is frozen.

   **Verified against an independent recorded number:** after the fix the initial error is
   **1.69 meV/atom, 43.86 meV/Å, 0.11 meV/Å³** — the same 1.69 / 43.9 that W1.1 recorded for
   the MH-1 foundation head on this validation set in September. The LES-FT model therefore
   starts at MH-1 exactly, which is what "initialised with the MH-1 weights" has to mean.

**Launched 2026-09-13 23:09** (b3 clock) on GPU 5 under a watchdog, 60 epochs, best-validation
selection.

**W7 epoch budget revised upward, 23:14 (b3 clock), 13 epochs in.** The 250-epoch figure was set
from an 84 s/epoch smoke measured with an O3.2 job co-resident. The real rates are **55 s/epoch
(plain) and 72 s (LES)**, still under contention, so 250 epochs would have finished around
03:20 and left roughly three hours of the night unused — and the instruction that opened this
work said from scratch "may need more epochs". Both runs were stopped and relaunched at **350
epochs with Stage Two at 280**; `--restart_latest` resumed them from their epoch-13 checkpoints,
so nothing already computed was discarded and Stage Two had not yet been reached. The two runs
keep the SAME epoch count as each other, which is what makes plain-versus-LES a comparison.
That is 71 k gradient updates against the README heuristic's 200 k — still short, and still
recorded as short.

**W7 incident, 23:14–23:23 (b3 clock): the rebudget ran TWO trainers per run for nine minutes.**
The trainer is a python CHILD of `scratch_run.sh`, so `pkill -f "scratch_run.sh <name>"` killed
the wrapper and ORPHANED the python. The supervisor then saw no wrapper, correctly concluded
the run was gone, and started a second one. Both wrote the same log and the same checkpoint
directory, which is what the interleaved epoch numbers in
`base_v3_{scratch,les}_doubled_2026-09-13.log` are — `Epoch 16, Epoch 12, Epoch 17, Epoch 13`,
one lineage continuing from 16 and one resumed at 12. Caught from the monitor: the LES run's
"epoch 9" came back at 151 meV/atom after an earlier epoch 9 of 4.61.

Both runs were stopped properly (supervisor, then wrapper, then python), the directories moved
aside rather than deleted, and **restarted clean at 320 epochs with Stage Two at 256**. The two
lineages had both written `epoch-13` and `epoch-14`, so resuming from the latest checkpoint
would have meant trusting a file two processes wrote: twenty minutes of training is not worth
that on an unattended run. `scratch_stop.sh` now does the three-step kill, and the supervisor
says in its header why a bare `pkill` on the wrapper is wrong.

`base_v3_les_ft` was never touched by the rebudget and ran single throughout.

### O3.2 RESULT (wave complete 2026-09-14 00:19 b3 clock, six seeds, all clean)

**G1, the adoption gate.** meV/Å per component, paired by seed/fold, τ = 1.7:

| seed (fold) | W6 | O3.1 | O3.2 | O3.2 − W6 | O3.1 − O3.2 |
|---|---:|---:|---:|---:|---:|
| 0 (0) | 11.22 | 14.59 | 13.78 | +2.55 | +0.81 |
| 1 (1) | 11.62 | 14.22 | 13.94 | +2.32 | +0.28 |
| 2 (2) | 11.58 | 15.06 | 13.85 | +2.27 | +1.21 |
| 3 (3) | 11.23 | 14.82 | 13.89 | +2.66 | +0.92 |
| 4 (0) | 12.02 | 14.28 | 13.67 | +1.65 | +0.61 |
| 6 (2) | 11.84 | 14.17 | 14.04 | +2.20 | +0.13 |
| **median** | **11.60** | **14.44** | **13.87** | | |

- **vs W6**: mean **−2.278**, se 0.144, 90 % CI [−2.568, −1.988] → **INFERIOR**, 1.3 τ.
  **O3.2 IS NOT ADOPTABLE.**
- **vs O3.1**: mean **+0.659**, se 0.166, 90 % CI [+0.326, +0.993] → better on every seed, but
  the whole interval lies **inside the equivalence band**: by the registered τ the two arms are
  **EQUIVALENT**. The screening helps, and the help is smaller than the threshold this
  programme set for "a difference that matters."

**G2a energy at 79 atoms: PASS** (mean d −0.0739, one-sided 95 % upper −0.0314), better than W6
at five of six seeds. **G2b between-size, cohort median: −0.3 meV, PASS** (W6 +5.0, O3.1 −1.8).
**Saturation: nothing at the knee**, on-site p95 shift 0.722 eV.

**THE PRE-REGISTERED EXPECTATION IS NOT MET, AND THAT IS THE RESULT.** Before the wave ran,
this was written down: "if the three-body screening is the right diagnosis, its gain should be
concentrated in the same places — the flanking Pb and the 2–4 Å shell — and largely absent
beyond 8 Å. If instead O3.2 gains uniformly across shells, or gains at long range, the reading
is wrong whatever G1 says, and the next arm is the head-owned basis rather than the
modulation." Cohort medians, meV/Å per component:

| shell (Å) | W6 | O3.1 | O3.2 | O3.2 − O3.1 |
|---|---:|---:|---:|---:|
| 2–4 | 19.83 | 25.54 | 24.99 | −0.55 |
| 4–6 | 13.81 | 17.49 | 17.00 | −0.50 |
| 6–8 | 9.73 | 12.62 | 12.09 | −0.53 |
| 8–10 | 9.86 | 11.11 | 10.25 | **−0.86** |
| 10–12 | 9.02 | 10.48 | 9.38 | **−1.10** |
| **flanking Pb** | 27.40 | 37.41 | 35.55 | **−1.86** |
| first-shell Cl | 16.55 | 20.56 | 20.26 | −0.31 |
| rest of 2–4 | 10.26 | 11.85 | 11.18 | −0.68 |

Half of it matches: the flanking Pb takes the largest single gain, −1.86, three times the
typical shell. The other half does not: the gain **does not decay with distance** — it is
LARGER at 8–12 Å (−0.86, −1.10) than at 4–8 Å (−0.57). A bond-local three-body screening with
`r_s3` = 4.5 Å has no business improving forces at 10–12 Å more than at 5 Å. By the criterion
registered before the result, **the reading that "the residual is the hopping's missing
environment dependence" is not supported.**

**And the screening barely engaged.** The diagnostic registered so that the modulation could not
be reported as "learned" while sitting at its stop shows the opposite failure — it is sitting at
its START. Cohort medians: saturated fraction **0.000**, `|pre|` p95 **0.038**, `|pre|` max
0.390, and the modulation factor `exp(η tanh pre)` at p95 **1.001**. The fit moved the hoppings
by about a tenth of a percent over most bonds. So the +0.66 meV/Å did not come from a strong
screening; it came from a nearly-absent one, which is consistent with the shell table saying the
gain is not where a screening would put it.

**What this means for the programme.** Three arms now agree on the same thing. O3 (descriptor
swap plus modulation deleted) lost 7.56; O3.1 (standardisation) recovered 4.63 of it; O3.2
(modulation restored, from geometry) recovers a further 0.66 and reads EQUIVALENT to its own
parent. **The remaining ~2.3 meV/Å is not the hopping modulation.** The two candidates left are
the head-owned descriptor itself — the one thing every O3-lineage arm has in common and W6 does
not — and the possibility that the base's block-0 features carry information no geometric basis
of this size reproduces. Per the registration, **the next arm is the head-owned basis**, and it
should be tested by the cheapest decisive experiment: give the decoupled head the base's
features back for the on-site term ALONE, changing nothing else. That isolates the descriptor
from the coupling, and it is one change.

**No new arm is opened here.** This entry records the result and the direction; the next arm is
registered before it runs, as every arm in this programme has been.

**W7 addition, 2026-09-14 00:50: `base_v3_les_zero`, and why.** `base_v3_les`'s validation
ENERGY is flat, not converging: median **127.2 meV/atom over epochs 40–60 and 122.1 over
80–100**, while its forces fall 15.62 → 12.16 and the SAME architecture without LES reaches
**1.26 meV/atom** at epoch 108. The criterion for calling this a real problem rather than slow
convergence was set before the numbers were read, and it is met.

Diagnosis: at the documented from-scratch weighting (energy 1, forces 100) the long-range Ewald
energy from the latent charges is a degree of freedom the force term barely constrains, and the
charges start from a fresh random initialisation — the from-scratch analogue of the 8332
meV/atom the fine-tune showed before its readouts were zeroed. The fix is therefore the SAME
one, and deliberately not a change of loss weights, which would have broken the controlled
plain-versus-LES comparison: a new `--les_zero_readouts` flag (default False; always on from a
foundation model) zeroes the last layer of each latent-charge readout so the model starts with
no long-range term and grows it. Gradients still flow there; nothing is frozen.

`base_v3_les` (the documented recipe, untouched) keeps running on GPU 7 to 320 epochs, because
its Stage Two answer — whether a 1000× energy weight over the last 64 epochs recovers a term
left at 122 meV/atom for 250 — is worth having and is not otherwise known. `base_v3_les_zero`
runs the identical configuration with the flag on, on GPU 5, same 320 epochs and same Stage Two
at 256, so all three from-scratch runs are directly comparable.

**W7.1 RESULT at 60 epochs, and why it was extended.** `base_v3_les_ft` finished 00:17 on
2026-09-14; validation, per component:

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 59 |
|---|---:|---:|---:|---:|---:|---:|---:|
| forces meV/Å | 11.99 | 8.06 | 7.32 | 6.75 | 6.46 | 6.28 | **6.16** |
| energy meV/atom | 1.07 | 1.76 | 0.55 | 0.49 | 0.59 | 0.38 | 0.74 |
| stress meV/Å³ | 0.36 | 0.18 | 0.17 | 0.18 | 0.17 | 0.17 | 0.24 |

Against **base v2's 30-epoch multihead fine-tune on the same split: 1.16 / 10.23 / 0.16**. The
direct LES fine-tune is at 6.16 meV/Å — **40 % below** the shipped base's forces — with energy
at or below it.

**The drift the guidance warns about did not appear.** The page is explicit that naive
fine-tuning has the worst MD stability and the greatest drift from the pretrained distribution;
60 epochs at lr 1e-3 gave a MONOTONIC force improvement to the last epoch and no rise in any
column. Whether that holds to 150 epochs is now being measured rather than assumed: the run was
**extended to 150 epochs on the local A4000** (b3's three cards are committed to the
from-scratch runs) from its epoch-58 checkpoint via `--restart_latest`. The 60-epoch log is kept
as `base_v3_les_ft_60ep.log` so the first wave's numbers stay readable on their own.

The registered 60 was chosen because the guide gives a range only for multihead and warns about
drift; the run ended still improving, so the cap was the binding constraint and not convergence.
That is the reason for the extension, and it is recorded before the longer run's result is read.

**`base_v3_les` (the documented from-scratch recipe with LES) DIVERGED TO NaN at epoch 118**
and stayed there for 41 epochs with no recovery; it was stopped at 01:20 on 2026-09-14 rather
than spend four more hours of a card on a dead run. Its best pre-NaN validation was **10.19
meV/Å at epoch 101** (energy 22.35) and **8.75 meV/atom at epoch 106** (forces 13.05) — forces
level with base v2's 10.23, energy an order of magnitude worse, and the energy never settled:
the last non-NaN epochs read 9.14, 30.30, 61.36 meV/atom on consecutive evaluations.

This is the finding the arm was for, and it is a clean one: **MACE-LES trained from scratch at
the documented 1 : 100 energy : forces weighting, with the latent-charge readouts at their
default random initialisation, is unstable on this data.** The long-range Ewald energy is a
degree of freedom the force term barely constrains; the validation energy wandered between 60
and 430 meV/atom for a hundred epochs and then the run blew up. `base_v3_les_zero` — the same
configuration with `--les_zero_readouts`, so the long-range term starts at zero and is learned
up — was at **4.41 meV/atom and 10.98 meV/Å by epoch 49**, against the unmodified run's 189
and 16.0 at the same epoch. The fix is not cosmetic.

**A second self-inflicted process bug, recorded like the first.** `scratch_stop.sh base_v3_les`
used an unanchored `pkill -f "$N"`, and `base_v3_les` is a PREFIX of `base_v3_les_zero`: it
killed the healthy zero-init run along with its target. Caught within a minute from the GPU
memory readout (card 5 at 1 MiB when it should have held a job), restarted from its epoch-56
checkpoint, and the script now anchors the name with `\b...\b`.

**W7.1 at 150 epochs, and a second extension to 300.** Resumed on the local A4000 (43 s/epoch,
faster than b3 for this model) and run to 150:

| epoch | 59 | 100 | 125 | 149 |
|---|---:|---:|---:|---:|
| forces meV/Å | 6.16 | — | — | **5.69** |
| energy meV/atom | 0.74 | — | — | **0.48** |

Still monotonic on forces, still no rise in any column: at 150 epochs of NAIVE fine-tuning at
lr 1e-3 **the drift the guidance warns about has not appeared**. Since the run again ended still
improving, and the A4000 is otherwise idle until morning, it is extended once more to **300
epochs**. The stopping rule for the night is stated here before the result: stop at 300, or
earlier if forces stop improving or any column starts to rise. Logs kept per wave
(`_60ep`, `_150ep`).

### W7 RESULT — `base_v3_scratch` (MH-1's architecture, from scratch), complete 2026-09-14 02:28

320 epochs, Stage Two from 256. Validation on the same 154-frame split base v2's gate numbers
were read on, per component:

| epoch | 49 | 99 | 149 | 199 | 249 | 255 (Stage One end) | 279 | 319 (final) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| energy meV/atom | 10.06 | 1.15 | 1.61 | 1.87 | 2.32 | 1.85 | **0.25** | **0.23** |
| forces meV/Å | 10.96 | 8.09 | 7.38 | 6.90 | 6.66 | 6.68 | 6.86 | **6.63** |
| stress meV/Å³ | 2.81 | 0.57 | 0.31 | 0.64 | 0.50 | 0.55 | 0.26 | 0.24 |

| | energy meV/atom | forces meV/Å | stress meV/Å³ |
|---|---:|---:|---:|
| base v2 (multihead fine-tune of MH-1, 30 epochs) | 1.16 | 10.23 | **0.16** |
| **base v3 from scratch (320 epochs)** | **0.23** | **6.63** | 0.24 |

**A from-scratch model with MH-1's architecture beats the fine-tune of MH-1 on this data: 5×
on energy, 35 % on forces.** base v2 keeps stress (0.16 against 0.24). Two things this does NOT
say: it is not a statement about MH-1's transferability (this model knows three elements and one
material, MH-1 knows 89 and everything), and it is not a like-for-like epoch budget (320 epochs
of 1619 frames against 30 epochs of 1619 + 10 000 replay). It says that for THIS host, with
this much data, the foundation initialisation and its replay were not what the accuracy was
coming from.

Stage Two did what it is for, exactly: energy 1.85 → 0.25 meV/atom over epochs 256–279 with
forces unchanged. Forces were flat from epoch ~200 (6.90 → 6.63 over 120 epochs), so the
registered shortfall against the README's 200 k-update heuristic (71 k here) cost little on
forces; whether it cost anything on energy cannot be told from one run.

### W7 — the base_compare reading, and a QUALIFICATION of "from scratch beats the fine-tune"

`base_compare_n.py`, 2877 frames, identical code, energies after a per-species OLS offset fitted
on the neutral frames of each base separately (raw median in brackets). Forces per component,
meV/Å; energies meV/atom.

| data set | frames | old base E | base v2 E | **v3 scratch E** | old base F | base v2 F | **v3 scratch F** |
|---|---:|---:|---:|---:|---:|---:|---:|
| pristine, 80 | 639 | 1.15 | 0.30 | **0.06** | 6.82 | 5.05 | **3.21** |
| neutral vacancy, 79 | 1174 | 1.31 | 0.40 | **0.06** | 13.07 | 10.96 | **4.83** |
| neutral vacancy, 159 | 17 | 3.08 | 0.74 | **0.09** | 6.51 | 10.96 | **5.35** |
| charged vacancy, 79 | 1030 | 42.16 | **38.37** | 40.52 | 53.65 | **33.22** | 57.23 |
| charged vacancy, 159 | 17 | 24.58 | **18.89** | 20.69 | 29.95 | **22.14** | 33.74 |

**The from-scratch base is far better on neutral data and WORSE on charged data, and the
charged rows are the only out-of-sample measurement in this table.** Every neutral frame is
in-sample for all three production bases, so those rows measure FIT. No charged frame was ever
seen by any base, so those rows measure GENERALISATION — and there `v3 scratch` reads **57.23
meV/Å at 79 atoms against base v2's 33.22**, worse even than the old base's 53.65, with the
same ordering at 159 atoms (33.74 vs 22.14). By shell the gap is worst exactly where the head
works: the 2–4 Å shell of a charged 79-atom cell is **132.5 against base v2's 74.9**.

This is the classic shape of a model that has fit its training distribution harder and
transfers less well off it, and it is the reason the foundation initialisation plus replay was
chosen in W1: base v2's numbers on neutral data are worse BECAUSE it is anchored to a
general-purpose representation, and that anchor is what pays on configurations it never saw.

**So the W7 result entry above stands as written but must be read with this:** "from scratch
beats the fine-tune" is true on the held-out NEUTRAL validation split (0.23 / 6.63 against
1.16 / 10.23), and false on the charged frames the D-SCC head actually operates on. **For the
programme's purpose — a base the charge head works against — base v2 remains the better base on
this evidence, and `base_v3_scratch` is not a candidate to replace it.** The comparison was
asked for and is reported; the recommendation follows the out-of-sample column, not the
in-sample one.

One thing `v3 scratch` does fix: the **159-atom neutral reversal** recorded in W1 (base v2 worse
than the old base there, 10.96 against 6.51, cause unknown). `v3 scratch` reads 5.35 and is
better than both at that size, uniformly across shells. The cause of base v2's reversal is still
not established, but it is not intrinsic to this data or this architecture.

**W7.1 FINAL at 300 epochs** (`base_v3_les_ft`, local A4000, complete 2026-09-14 03:47):

| epoch | 0 | 20 | 59 | 149 | 262 | 299 |
|---|---:|---:|---:|---:|---:|---:|
| energy meV/atom | 1.07 | 0.55 | 0.74 | 0.48 | 0.96 | **0.55** |
| forces meV/Å | 11.99 | 7.32 | 6.16 | 5.69 | 5.26 | **5.19** |
| stress meV/Å³ | 0.36 | 0.17 | 0.24 | 0.27 | 0.32 | **0.28** |

Against base v2 (1.16 / 10.23 / 0.16) on the same split: **energy 2.1× better, forces 49 %
lower, stress 1.75× worse.** The forces improved monotonically at every checkpoint — 6.16 at
60, 5.69 at 150, 5.19 at 300 — and the gain is flattening (0.47 over epochs 60–150, 0.50 over
150–300, i.e. half the rate per epoch).

**The drift the fine-tuning guidance warns about did not appear at 300 epochs**, five times the
upper end of the range the page gives for multihead replay, at lr 1e-3 with no replay head at
all. That is a real result about naive fine-tuning ON THIS DATA and should not be generalised
past it: this is one host, 1619 frames, and a fine-tune whose LES term started at exactly zero.

**Operational note.** b3's GPU 6 fell off the bus at ~03:45 (GPU 4 had gone earlier). Nothing
was lost: `base_v3_scratch` had finished on it at 02:28 and `base_v3_les_zero` on GPU 5 was
unaffected. Recorded because the machine has previously lost every job at once to a card fault.

### W7 — the out-of-sample reading, and a CORRECTION of "the drift did not appear"

`base_compare_n.py` pass B. Forces per component, meV/Å; energies meV/atom after the per-species
OLS offset.

| data set | frames | base v2 | v3 scratch | **v3 LES-FT** |
|---|---:|---:|---:|---:|
| pristine, 80 | 639 | 5.05 | 3.21 | **2.41** |
| neutral vacancy, 79 | 1174 | 10.96 | 4.83 | **3.70** |
| neutral vacancy, 159 | 17 | 10.96 | 5.35 | **4.75** |
| charged vacancy, 79 | 1030 | **33.22** | 57.23 | 58.54 |
| charged vacancy, 159 | 17 | **22.14** | 33.74 | 36.26 |
| charged 79, 2–4 Å shell | | **74.9** | 132.5 | 140.3 |

**CORRECTION OF RECORD (mine).** The W7.1 entry above says "the drift the fine-tuning guidance
warns about did not appear at 300 epochs". That was read off the NEUTRAL validation split, and
a neutral in-distribution split **structurally cannot see drift away from the pretrained
distribution** — it is the distribution being fitted. On the charged frames, which no base has
ever seen, the drift is plain: the LES fine-tune is **58.54 meV/Å against base v2's 33.22**, and
140.3 against 74.9 in the 2–4 Å shell where the head works. The claim is withdrawn. What holds
is the narrower statement: *no degradation appeared in-distribution* at 300 epochs.

**And the initialisation is not what protects base v2.** `v3 LES-FT` starts from MH-1's weights
and still loses out-of-sample — slightly MORE than the from-scratch model does. The one thing
base v2 has that neither W7 model has is the **multihead pseudolabel replay head**. On this
evidence the replay, not the initialisation, is what buys out-of-distribution robustness, which
is what the multihead recipe is for and what the guidance page claims when it says naive
fine-tuning drifts most. W1's choice of the multihead route is vindicated by a measurement it
did not have at the time.

**Recommendation, following the out-of-sample column.** For the D-SCC programme, whose head
operates on CHARGED frames, **base v2 remains the base.** Neither `base_v3_scratch` nor
`base_v3_les_ft` is a candidate to replace it, notwithstanding that both are two to three times
better on every neutral set. For a user who wants a neutral-only CsPbCl3 potential — bulk MD,
phonons, pristine or neutral-defect energetics — `base_v3_les_ft` is the best model produced
here by a wide margin (2.41–4.75 meV/Å against base v2's 5.05–10.96).

**Open question, being measured rather than argued** (`pass C`): whether the out-of-sample gap
GROWS with fine-tuning epochs, as the guidance implies. The 150-epoch LES-FT checkpoint was
kept, so 150 and 300 can be compared on the same charged frames.

**Pass C — does the out-of-sample gap grow with fine-tuning epochs? No: it saturates early.**
The 150-epoch LES-FT checkpoint against the 300-epoch one, same frames, same code:

| | neutral 79 (in-sample) | charged 79 (OUT of sample) | charged 79, 2–4 Å |
|---|---:|---:|---:|
| LES-FT @ 150 epochs | 4.41 | 57.12 | 136.4 |
| LES-FT @ 300 epochs | **3.70** | 58.54 | 140.3 |
| base v2 | 10.96 | **33.22** | **74.9** |

Doubling the epochs bought **16 % in-distribution** and cost **2.5 % out**. So the drift is not
a progressive penalty for training longer — it is incurred early, in the first hundred epochs or
sooner, and then roughly saturates while the in-distribution fit keeps improving. Two
consequences: (i) shortening a naive fine-tune would NOT have avoided the out-of-sample loss,
so "fewer epochs" is not the remedy; (ii) given that a naive fine-tune is being run at all,
running it long is nearly free.

The remedy the evidence points at is the replay head, which is the one structural difference
between base v2 and both W7 models — not the epoch count, not the initialisation, and not LES.

### W7 RESULT — `base_v3_les_zero` (MACE-LES from scratch, latent charges zero-initialised)

320 epochs, Stage Two from 256, complete 2026-09-14 04:40. Same architecture, data, E0s, split
and schedule as `base_v3_scratch`; the ONLY difference is the LES term.

| epoch | 49 | 99 | 149 | 199 | 249 | 255 | 279 | 319 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| energy meV/atom | 4.41 | 7.21 | 6.51 | 6.48 | 4.42 | 5.04 | **0.27** | **0.23** |
| forces meV/Å | 10.98 | 8.60 | 7.79 | 7.68 | 7.51 | 7.22 | 7.45 | **7.19** |
| stress meV/Å³ | 1.64 | 0.83 | 1.06 | 0.69 | 1.03 | 0.94 | 0.35 | 0.33 |

| | energy | forces | stress |
|---|---:|---:|---:|
| base v2 | 1.16 | 10.23 | **0.16** |
| base v3 from scratch (no LES) | **0.23** | **6.63** | 0.24 |
| **base v3 LES, zero-init** | **0.23** | 7.19 | 0.33 |

**Two findings, one about the method and one about the material.**

(i) **The zero-init fixes the instability completely.** The identical configuration with the
library's default random latent-charge initialisation wandered between 60 and 430 meV/atom and
diverged to NaN at epoch 118. With the term started at zero the run is stable for all 320
epochs and Stage Two behaves exactly as it does for the plain model (energy 5.04 → 0.27 over
epochs 255–279). This is not a tuning preference: one configuration completes and the other
does not.

(ii) **LES buys nothing here.** Against the plain from-scratch model it is identical on energy
(0.23), 8 % worse on forces (7.19 vs 6.63) and 38 % worse on stress. A latent Ewald term has
nothing to add to neutral CsPbCl3 frames at these cell sizes, which is the expected answer for a
material whose long-range electrostatics are screened and whose training set carries no net
charge — LES is designed for systems where charge transfer and long-range interactions are not
captured locally. **It is the wrong test bed for LES, and that is a property of the data, not a
failure of the method.** The charged frames, where a long-range term might matter, are exactly
the frames no base is trained on.

### W7 — THE COMPLETE ERROR TABLE (base v2's format, all four new bases)

`base_compare_n.py`, 2877 frames, identical code and frames for every model, each called through
its OWN forward so the LES models contribute their Ewald term. Energies meV/atom after a
per-species OLS offset fitted on the neutral frames of each base separately (raw in brackets);
forces per component, meV/Å. **Neutral rows are in-sample for every production base and measure
FIT. Charged rows were seen by no base and are the only GENERALISATION numbers here.**

| data set | frames | old base | base v2 | v3 scratch | v3 LES zero-init | v3 LES-FT |
|---|---:|---:|---:|---:|---:|---:|
| **energy, meV/atom** | | | | | | |
| pristine, 80 | 639 | 1.15 | 0.30 | **0.06** | **0.06** | 0.07 |
| neutral vacancy, 79 | 1174 | 1.31 | 0.40 | **0.06** | **0.06** | 0.08 |
| neutral vacancy, 159 | 17 | 3.08 | 0.74 | **0.09** | 0.12 | 0.10 |
| charged vacancy, 79 | 1030 | 42.16 | **38.37** | 40.52 | 41.11 | 41.68 |
| charged vacancy, 159 | 17 | 24.58 | **18.89** | 20.69 | 21.15 | 21.51 |
| **forces, meV/Å per component** | | | | | | |
| pristine, 80 | 639 | 6.82 | 5.05 | 3.21 | 3.50 | **2.41** |
| neutral vacancy, 79 | 1174 | 13.07 | 10.96 | 4.83 | 5.26 | **3.70** |
| neutral vacancy, 159 | 17 | 6.51 | 10.96 | 5.35 | 5.68 | **4.75** |
| charged vacancy, 79 | 1030 | 53.65 | **33.22** | 57.23 | 58.33 | 58.54 |
| charged vacancy, 159 | 17 | 29.95 | **22.14** | 33.74 | 33.90 | 36.26 |

Charged 79-atom forces by range from the vacancy, per component (the shells the head works in):

| base | 2–4 | 4–8 | 8–10 | 10–12 | >12 |
|---|---:|---:|---:|---:|---:|
| old base | 120.3 | 49.4 | 24.2 | 12.9 | 10.5 |
| **base v2** | **74.9** | **30.4** | **15.3** | **8.9** | 8.0 |
| v3 scratch | 132.5 | 51.8 | 21.9 | 10.7 | 7.3 |
| v3 LES zero-init | 138.7 | 51.2 | 22.1 | 11.0 | 7.6 |
| v3 LES-FT | 140.3 | 51.2 | 21.8 | 9.6 | **6.9** |

**The result is uniform across all three new bases and is the night's main finding.** Every one
of them is two to three times better than base v2 on neutral data and 1.7–1.8× WORSE on charged
data, with the damage concentrated in the 2–4 Å shell (133–140 against base v2's 74.9). They
differ in architecture (LES or not) and in initialisation (from scratch or from MH-1) and it
makes no difference to this pattern; base v2 differs from all three by having the **multihead
pseudolabel replay head**, and it is the only one that holds up off its training distribution.

**Recommendation: base v2 stays.** For the D-SCC programme, whose head operates on charged
frames, none of the three is a candidate. For neutral-only work on this host — bulk MD, phonons,
pristine or neutral-defect energetics — `base_v3_les_ft` is the best model produced here
(2.41–4.75 meV/Å against base v2's 5.05–10.96, energies 0.07–0.10 against 0.30–0.74).

**What would test the conclusion** (registered, not run): repeat the direct fine-tune WITH the
replay head and nothing else changed. If the charged-frame error returns to base v2's level,
the replay is confirmed as the mechanism; if it does not, the mechanism is something else and
this reading is wrong. That is one run and it is the obvious next experiment.

### W7 — MAJOR CORRECTION: the latent charges never trained in EITHER LES model

*(User question, 2026-09-14 morning: "compare the zero vs ft les models to see whether the
learnt charges actually trained in the ft model." They did not, in either.)*

**Measured, three independent ways, on `les_ft_300.model` and `base_v3_les_zero_stagetwo.model`:**

| reading | LES-FT (300 ep) | LES zero-init (320 ep) |
|---|---|---|
| final-layer `\|\|W\|\|` of both latent-charge readouts | **0.0000e+00** | **0.0000e+00** |
| latent charge `q` on 48 frames (neutral and charged) | identically 0 | identically 0 |
| `E_LES` per frame | identically 0 | identically 0 |
| ablation (zero the readouts and re-predict) | ΔE 0.0000, ΔF 0.0000 | ΔE 0.0000, ΔF 0.0000 |

**Mechanism, confirmed directly rather than inferred.** With the trained model's weights at zero
the gradient reaching the latent-charge readouts is **exactly 0.000000e+00** while the energy
readouts receive 9.03; after perturbing those weights by 1e-3 the gradient becomes 9.48e-03. The
Ewald energy is **quadratic** in the charges, so `q = 0` is a stationary point of it:
`dE_lr/dq = 0` there, and `dL/dW = dL/dE_lr · dE_lr/dq · dq/dW` vanishes identically however
large `dL/dE_lr` is.

**The claim written into `_zero_les_readouts` — "Gradients still flow into the zeroed layer, so
nothing is frozen: the term is learned if the data wants it, from zero rather than from noise"
— IS FALSE, and is the error this correction is about.** Zeroing the latent-charge readouts does
not start the long-range term small; it removes it. Both "fixes" of last night worked by
silently disabling the thing they were meant to stabilise.

**What this invalidates.**

1. **`base_v3_les_ft` is not a LES model.** It is a plain direct fine-tune of MH-1 with inert
   LES machinery. Its result (0.55 meV/atom, 5.19 meV/Å, 49 % below base v2 on forces) stands
   exactly as measured, but it is a result about NAIVE FINE-TUNING, not about LES. Every
   statement about it in the entries above should be read that way.
2. **`base_v3_les_zero` is not a LES model** either — it is `base_v3_scratch` with 528 dead
   parameters and a different RNG draw. Its agreement with that model (0.23 / 7.19 against
   0.23 / 6.63) is now fully explained and is not evidence about LES.
3. **"LES buys nothing here" is WITHDRAWN.** It was never tested. The only run in which the
   latent charges were live is `base_v3_les` with the library's default random initialisation,
   and that one diverged to NaN at epoch 118. **The question of whether LES helps on this data
   is OPEN and unmeasured.**
4. The out-of-sample finding is untouched: all three models lose on charged frames, and none of
   them had a working long-range term, so that conclusion never rested on LES.

**The fix, and why it is not "zero".** The initialisation has to be small enough that the Ewald
term does not swamp an already-good potential, and non-zero so the gradient exists — the
default random init is the second without the first, and zero is the first without the second.
`--les_zero_readouts` should become a SCALE (e.g. 1e-2 × the default draw), not a zeroing, with
the scale registered. A run at that setting is the experiment that would actually answer
whether LES helps here; it has not been run and no claim about LES should be made until it is.

## O3.3 / O3.D — REGISTRATION (2026-09-14, written before either arm runs)

*(Coadvisor ruling, relayed 2026-09-14. Adopted as written, with the mechanism check added.)*

**The coadvisor's mechanism for the 8–12 Å gain, which the O3.2 entry left unattributed.** The
far-field force channel is `E_host = dqᵀ Γ_LR (s q0)`, and `q0` is the Mulliken pattern of the
reference fill of `H0` on EVERY atom, including those 10 Å from the vacancy. The on-site term
sets those bulk atoms' thermal charges, so a better on-site environment model gives a better
thermal `q0` and better far-field forces. A gain that is largest at the flanking Pb and still
present at 8–12 Å is what that channel predicts **from an on-site change, with no three-body
screening required** — which is consistent with O3.2's modulation having moved the hoppings by
0.1 % while the gain appeared anyway. O3.1, not O3.2, is therefore the parent for what follows.

**Diagnostic, to run now (`q0` check).** Per-atom `|q0(W6) − q0(O3.1)|` on shared frames against
the per-atom force error, by shell. If they track, the target is the on-site term and nothing
else.

**What block 0 has that the head's basis does not: BODY ORDER, not range or radial richness.**
Both reach 6 Å and six Bessel functions with learned coefficients is a normal ACE A-basis. Block
0's invariants are products of that A-basis up to correlation 3 — three- and four-body content:
octahedral tilt angles, the Cs displacement relative to the cage, bond-polarity patterns. The
head's on-site readout sees only the two-body radial sum, and **the crystal-field shift of a site
in a tilted perovskite is a three-body quantity.**

### Arm O3.3 — decoupled, PRIMARY, one change from O3.1

O3.1 plus the **ACE correlation-2 invariants of the A-tensors the head already computes**, fed to
the on-site readout: the pairwise products of the `l = 0` channels, `A^(1)_n · A^(1)_m` over the
radial channels, and `‖A^(2)‖²_F`. Standardised by A1.1 exactly as the linear channels are.
**Zero runtime cost** — the tensors are already built and these are contractions of them — and
it keeps every property that makes the decoupled head worth having: nothing from the base, no
wait on block 0, no `create_graph` traversal, capture still possible.

### Arm O3.D — coupled, DIAGNOSTIC ONLY, not a production candidate

O3.1 with the base's block-0 invariants restored **for the on-site term alone**, hoppings and
directional block left on the head's own descriptor. **Registered as not adoptable whatever it
scores**: a head that reads block 0 anywhere must wait for it, needs the cotangent through it,
must expose block-0 outputs to capture, and returns the `create_graph` traversal to the training
step — back near the 1.66× ceiling measured for O2. It exists to attribute the gap, not to ship.

### How the pair reads (registered before either result)

- **B-basis recovers the 2.3** → adopt the decoupled head; the content was many-body and the
  head can make it itself.
- **Block-0 recovers it and the B-basis does not** → the content is many-body beyond
  correlation 2, or it is the base's pretrained chemistry. The choice is then accuracy against
  speed, made with numbers, and a registered "fast tier" becomes a legitimate product ruling.
- **Neither recovers it** → the deficit is in the rank-1 block, and the next arm is the
  equivariant correlation-2 product for `v_i`.

Gates for both: paired TOST vs W6 and vs O3.1 at τ = 1.7, by-shell gains, saturation (including
that the new channels are not merely saturating the readout), G2a and G2b as registered. Six
seeds each, folds as before.

### Recorded whatever happens

**The decoupled head WITH electrostatics (13.87) is worse than the coupled head WITHOUT them
(`Phi = 0`, 12.22).** The environment information in the on-site term is worth at least as much
to the forces as the entire host term. The on-site term was treated through this programme as
the inert piece of the head; it is not, and that reorders what is worth working on.

**Optional, not scheduled:** a learning curve for O3.1 and W6 at 50 % of the training frames,
to separate variance (learning the environment dependence from 785 charged frames) from capacity
(inheriting it from the neutral PES). It would change how a foundation version is built.

### The `q0` diagnostic (coadvisor's check), 2026-09-14 — the mechanism is SUPPORTED

30 charged 79-atom frames, `q0` recomputed from each model's own `H0` through
`reference_charges` (not by changing what production exports). W6 s0 against O3.1 s0, same
fold, same frames.

| shell (Å) | atoms | mean \|Δq0\| (e) | W6 F err | O3.1 F err | O3.1 − W6 | within-shell corr(\|Δq0\|, err) |
|---|---:|---:|---:|---:|---:|---:|
| 2–4 | 234 | **0.3418** | 17.09 | 26.92 | +9.82 | +0.446 |
| 4–6 | 320 | 0.1549 | 12.37 | 17.51 | +5.14 | +0.428 |
| 6–8 | 991 | 0.0728 | 9.31 | 13.66 | +4.35 | +0.060 |
| 8–10 | 601 | 0.0834 | 8.31 | 10.23 | +1.92 | +0.090 |
| 10–12 | 204 | 0.0790 | 8.37 | 10.74 | +2.36 | −0.224 |
| >12 | 20 | 0.0496 | 6.64 | 7.01 | +0.37 | +0.117 |

**The reference charges differ by 21 %.** Mean `|q0(W6) − q0(O3.1)|` is **0.1135 e** against a
mean `|q0|` of 0.5396 e, reaching 0.94 e on individual atoms. The two heads do not merely place
the defect differently; they disagree about the thermal Mulliken charge of ordinary bulk atoms.

**Across shells the two quantities track almost perfectly:** correlation between mean `|Δq0|`
and the O3.1 − W6 force deficit is **+0.939**, and between mean `|Δq0|` and O3.1's absolute
force error **+0.963**, over six shells spanning 2 to >12 Å. Both fall by roughly an order of
magnitude together. This is the coadvisor's prediction and it holds at the level the argument
is made: `E_host = dqᵀ Γ_LR (s q0)` with `q0` set on EVERY atom by the on-site term, so a worse
on-site environment model gives a worse far-field force through the host channel, with no
three-body screening required anywhere.

**Stated with its limit, because the limit matters.** The correlation is between SHELL MEANS.
The WITHIN-shell per-atom correlation is strong only at 2–6 Å (+0.45, +0.43) and is ~0 or
negative beyond (+0.06, +0.09, −0.22). So `|Δq0|` predicts which SHELL carries force error and
not which ATOM within a far shell does. That is consistent with a collective electrostatic
channel — `Γ_LR` mixes every atom's `q0` into every atom's force, so a per-atom correspondence
is not expected at long range — but it means the diagnostic supports the mechanism rather than
proving it atom by atom.

**Consequence for the programme.** The on-site term is not the inert part of the head. It sets
`q0` on all 79 atoms, `q0` drives the entire far-field force channel, and the O3 lineage's
remaining deficit tracks the error it makes there. O3.3 and O3.D both act on exactly that term,
which is now the right place to be working.

### O3.3 instability, diagnosed — and O3.3b REGISTERED (2026-09-14, before it runs)

O3.3 seeds show discrete loss spikes (up to 79× the median) that O3.1 never shows. Measured,
with O3.1 on the SAME seeds and folds as the control, and the cosine schedule (lr 2e-3 → 5 %
over 60 epochs) identical in both:

| run | epochs with `sat` > 0 | max loss / median | dEp95 mean (eV) | dEp95 max |
|---|---:|---:|---:|---:|
| O3.1 s1 / s2 / s3 | **0 / 0 / 0** | 19× / 11× / 12× | 0.65–0.67 | 0.88–0.90 |
| O3.3 s1 / s2 / s3 | **23 / 11 / 44** | 41× / 24× / **79×** | 1.10–1.36 | 1.53–**2.34** |

**It is not input outliers.** The standardised 211-wide block was measured over 4800 atoms:
max `|z|` is 4.1 (linear `l = 0`), **4.9 (the 171 product channels)**, 6.7 (`A¹·A¹`), 4.1
(`‖A²‖²`), with essentially nothing beyond 5σ. The quadratic channels are no heavier-tailed
than the linear ones — the hypothesis that products would blow the tails out is **wrong** and
is recorded as such.

**It is the A1 bound.** O3.3 spends its extra capacity on a roughly DOUBLED on-site correction
(dEp95 1.10–1.36 eV against O3.1's 0.65), and where O3.1 never once reaches the barrier knee in
60 epochs on any seed, O3.3 reaches it in 11 to 44 of them. The correction is
`Δ_Z tanh(pre)` with `Δ_Z` = 3.079 eV; as `|pre| → 2` the tanh flattens, gradients through it
become erratic, and occasional batches push past — which is the spike and the damaged fill.
**Dose-dependent, which is the evidence:** barrier incidence orders the seeds exactly as the
final error does — s2 (11 epochs → 11.99 meV/Å), s1 (23 → 13.91), s3 (44 → 15.97).

**The barrier is currently weightless.** `sat_weight` 1e-3 times `sat` ≈ 1e-6 is ~1e-9 against a
loss of 6e-6 — a millionth of the objective. The mechanism A1 registered to prevent exactly this
has never had the weight to do it; it has simply never been needed before, because no arm until
now went near the knee.

**Arm O3.3b (registered, user-approved 2026-09-14):** O3.3 with `--sat_weight 1.0`, one change,
six seeds, everything else identical. At the observed excursion size that makes the barrier
worth ~20 % of the loss when it fires and ~nothing otherwise, which is what a barrier should be.

**How it reads, written before the result:**
- **s2's 11.99 survives and the spikes go** → the excursions were optimiser noise, the
  correlation-2 content is real, and O3.3b is the adoption candidate.
- **Spikes go but accuracy degrades toward O3.1** → the on-site term genuinely NEEDS more range
  than A1 allows. Unlike O3 — which was stuck on a species offset with a badly conditioned
  input — this arm has a clean input and is using twice the correction, so the constraint is
  the BOUND and not the parameterisation. The next arm is then a registered increase of `Δ_Z`,
  which is an A1 amendment and needs its own registration.
- **Spikes remain** → the cause is not the bound and the diagnosis above is wrong.

### W7.2 — LES retrained with LIVE charges: the divergence was the LR SCHEDULE, and LES does help

*(User: "I know many people train les mace models" — correct, and the failure was mine.)*

**The reference recipe** (`ChengUCB/les_fit`, `MLIPs/MACE-LES-new/fit.sh`) does NOT set `--lr`
either: both it and the run that diverged use MACE's default 1e-2. What it sets and the failed
run did not is **`--scheduler_patience 15` against the default 50**. The diverged run's LR was
still 1e-2 at epoch 90 — the plateau scheduler essentially never fired — and the forces degraded
for ~17 epochs (10.19 → 12.00 → 15.56 → 21.40) before the NaN at 118. The other differences were
stress in the loss (reference: weight 0) and Stage Two at 50 % rather than 80 %.

**`base_v3_les_ref`** (reference recipe, charges at the library's own initialisation): **320
epochs, no NaN, 0.21 meV/atom and 6.43 meV/Å.** Against `base_v3_scratch`'s 0.23 / 6.63 — but
that is NOT a controlled LES comparison, because the recipe changed too.

**The controlled comparison is the fine-tune pair**: identical settings, identical 150 epochs,
one difference — latent charges live at a 1e-2 scale versus pinned at zero.

| | energy meV/atom | forces meV/Å |
|---|---:|---:|
| LES-FT, charges PINNED, 150 ep | 0.48 | 5.69 |
| **LES-FT2, charges LIVE, 150 ep** | **0.24** | **5.62** |

**Energy halved, forces level. LES helps on this data, modestly.** The earlier "LES buys nothing
here" was withdrawn as untested; it is now tested, and the answer is that it buys a factor two
on energy and nothing measurable on forces.

**The charges are live and doing work**, measured the same three ways that exposed the pinned
runs: readout `‖W‖` 6.4e-1 / 1.9e-1 (FT2) and 1.3e1 / 9.2e-1 (REF) against 0 before; `|q|` mean
0.028 / 0.053 e; `E_LES` 4.1 / 8.8 meV/atom; ablation moves forces by 9.6–18.4 meV/Å. **The
ablation only shows the model has ADAPTED to the term, not that it improves accuracy** — the
FT2-versus-pinned pair is the evidence for that, and it is the only controlled pair here.

**Flagged, not resolved: the net latent charge per cell is −2.06 e (FT2) and +2.47 e (REF)** in
cells that are neutral or singly charged. Latent charges are an auxiliary field, not Mulliken
charges, and the Ewald energy is invariant under a global sign flip — so this is not wrong on
its face. But a non-neutral distribution in a periodic cell needs a neutralising background and
the energy then carries a convention. Anyone reading these charges physically should check that
first. Per-species means (Cl / Cs / Pb): FT2 −0.040 / +0.006 / −0.017, REF +0.039 / −0.055 /
+0.091 — not an ionic pattern in either, which is expected for a latent field and is the reason
not to read them as charges.

### W7.3 — LES analyses: shell-resolved errors, energies, and the Born-charge check

All on the same 2877 frames and the same code as base v2's own table
(`base_compare_n.py`), each model called through its OWN forward so the LES bases
contribute their Ewald term.

#### (a) Forces by range from the vacancy, meV/Å per component

**Neutral vacancy, 79 atoms (1174 frames — IN-SAMPLE for every production base)**

| base | total | 2–4 Å | 4–8 | 8–10 | 10–12 | >12 |
|---|---:|---:|---:|---:|---:|---:|
| base v2 | 10.96 | 20.0 | 10.0 | 8.8 | 7.1 | 6.5 |
| v3 scratch (no LES) | 4.83 | 6.3 | 4.6 | 4.6 | 4.3 | 3.9 |
| LES-REF (scratch, live q) | 4.99 | 7.1 | 4.8 | 4.6 | 4.1 | 3.8 |
| **LES-FT2 (MH-1 init, live q)** | **4.18** | **6.2** | **3.9** | **3.9** | **3.5** | **3.1** |

**Charged vacancy, 79 atoms (1030 frames — OUT of sample for every base)**

| base | total | 2–4 Å | 4–8 | 8–10 | 10–12 | >12 |
|---|---:|---:|---:|---:|---:|---:|
| **base v2** | **33.22** | **74.9** | **30.4** | **15.3** | **8.9** | 8.0 |
| v3 scratch | 57.23 | 132.5 | 51.8 | 21.9 | 10.7 | 7.3 |
| LES-REF | 62.99 | 148.1 | 55.9 | 24.3 | 11.4 | 7.8 |
| LES-FT2 | 52.37 | 124.0 | 46.4 | 19.8 | 9.2 | **7.2** |

**The in-sample gain is UNIFORM across shells** — LES-FT2 beats base v2 by 2.6–3.2× at every
distance, not preferentially at long range. If the latent Ewald term supplied genuine long-range
physics the advantage should GROW with distance; it does not, so the improvement is better local
fitting. The only bin where the LES bases lead out of sample is **>12 Å (7.2–7.8 against 8.0)**,
which is suggestive and is the smallest bin — not claimed. **LES-REF is worse than the no-LES
from-scratch model at every charged shell** (148.1 vs 132.5 at 2–4 Å).

#### (b) Energies, meV/atom — median aligned (raw in brackets)

| data set | frames | base v2 | v3 scratch | LES-REF | LES-FT2 |
|---|---:|---:|---:|---:|---:|
| pristine 80 | 639 | 0.30 (0.94) | 0.06 (0.06) | 0.06 (0.17) | **0.05 (0.08)** |
| neutral vacancy 79 | 1174 | 0.40 (0.65) | 0.06 (0.06) | 0.07 (0.08) | 0.07 (0.07) |
| neutral vacancy 159 | 17 | 0.74 (0.63) | 0.09 (0.10) | 0.08 (0.08) | **0.03 (0.03)** |
| charged vacancy 79 | 1030 | **38.37** | 40.52 | 41.57 | 41.50 |
| charged vacancy 159 | 17 | **18.89** | 20.69 | 21.58 | 21.66 |

Aligned RMS (pristine / neutral 79 / neutral 159): base v2 0.43 / 0.61 / 1.60; v3 scratch
0.10 / 0.17 / 0.54; LES-REF 0.10 / 0.16 / 0.21; **LES-FT2 0.08 / 0.14 / 0.21**.

**Energies improve by 5–8× where forces improved 2–3×**, and the LES advantage seen in forces is
mostly absent — LES-FT2 and the no-LES scratch model tie at 79 atoms. The exception is
**159 atoms: LES-FT2 0.03 against scratch 0.09 and base v2 0.74**, the only size-transfer signal
in the table.

**Isolated-atom references, from the fitted per-species offsets (eV/atom):**

| base | a_Cl | a_Cs = a_Pb |
|---|---:|---:|
| old base | +0.2993 | −0.4566 |
| base v2 | −0.1299 | +0.1925 |
| v3 scratch | −0.0032 | +0.0048 |
| LES-FT2 | +0.0092 | −0.0136 |

base v2 needs a **130–193 meV/atom** correction to align with the labels; the new bases need
3–28 meV. **This retires the W1 caveat** that absolute formation energies referenced to isolated
Cs are unsupported: they are unsupported on base v2 and supported on these.

#### (c) Born effective charges — the CORRECT comparator, and two corrections of record

**CORRECTION 1.** An earlier entry compared the LES latent charges to W6's reference-fill
MULLIKEN charge `q0`. That is the wrong reference: LES latent charges are constrained by FORCES,
i.e. by the energy's response to displacement, which is what `Z*` is and what a static
population analysis is not. The `q0` correlation table is withdrawn.

**CORRECTION 2.** On the raw latent charges, Cl and Pb shared a sign in both models, and after
mean-centring LES-FT2 looked chemically right and LES-REF wrong. Both readings were artefacts.
The Ewald excludes `k = 0`, so nothing constrains the net latent charge (−2.08 e per cell for
FT2, +2.54 e for REF) and the global sign is free because the energy is quadratic in `q`.

`Z*` computed from the models themselves (`compute_bec`), isotropic part, 8 neutral 80-atom
frames:

| | Cl | Cs | Pb | \|Pb/Cs\| | Σ Z* per cell |
|---|---:|---:|---:|---:|---:|
| LES-FT2 | −0.466 | +0.382 | **+1.028** | 2.69 | **+0.18** |
| LES-REF | +0.507 | −0.409 | −0.955 | 2.34 | +2.50 |
| DFT expectation | − | + | + (anomalous) | ~3.4 | 0 |

`Z*` carries the same free global sign as `q` (it is linear in `q` while the energy is
quadratic). Flipping LES-REF gives Cl −0.507, Cs +0.409, Pb +0.955 — **agreeing with LES-FT2 to
~10 % on every species.** So the two models learned the SAME response, and both have the
halide-perovskite signature: Cl the anion, Cs and Pb cations, **Pb anomalously large at 2.3–2.7×
Cs**. Magnitudes run ~4× below DFT (Pb ≈ +4.4, Cs ≈ +1.3), expected because nothing pins the
absolute scale at `epsilon_factor = 1`; only ratios are meaningful.

**The acoustic sum rule separates them and favours LES-FT2 by 14×** (+0.18 against +2.50 per
cell). `Σ Z* = 0` is rigorous, so the fine-tune's response is nearly charge-neutral and the
from-scratch one is not — consistent with LES-FT2's better shell errors (4.18 vs 4.99 neutral,
52.4 vs 63.0 charged).

**Provenance, checked because it was asked:** the `les` library at `/home/alex/src/les` is
pristine upstream — clean tree, no commits ahead of `ChengUCB/les.git`, every commit by the LES
authors, HEAD `c8063fa`. `MACELES` in `mace/modules/extensions.py` differs from the branch point
by one cosmetic `node_energy` cast (commit `daaaf10`, pre-dating this work) on a quantity the
LES path never consumes. The changes made here are confined to `mace/tools/` and concern getting
INTO MACELES — the foundation-loader guard, the fitted shift, and the readout init scale — not
what it computes.

### O3.3 / O3.3b / O3.D RESULT (2026-09-14) — the three-way reading resolves to branch THREE

All six seeds each, folds as registered, τ = 1.7 per component.

| seed (fold) | W6 | O3 | O3.1 | O3.2 | O3.3 | O3.3b | O3.D |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (0) | 11.22 | 18.82 | 14.59 | 13.78 | 13.20 | 13.58 | 15.74 |
| 1 (1) | 11.62 | 19.71 | 14.22 | 13.94 | 13.91 | 14.08 | 14.65 |
| 2 (2) | 11.58 | 18.55 | 15.06 | 13.85 | 11.99 | 12.80 | 15.44 |
| 3 (3) | 11.23 | 20.48 | 14.82 | 13.89 | 15.64 | 13.07 | 14.16 |
| 4 (0) | 12.02 | 18.53 | 14.28 | 13.67 | 12.24 | 15.20 | 17.18 |
| 6 (2) | 11.84 | 18.79 | 14.17 | 14.04 | 15.31 | 13.06 | 14.06 |
| **median** | **11.60** | **18.81** | **14.44** | **13.87** | **13.55** | **13.32** | **15.04** |
| spread | 0.80 | 1.95 | 0.89 | 0.37 | 3.65 | 2.39 | 3.12 |

| comparison | mean d | 90 % CI | reading |
|---|---:|---|---|
| W6 − O3.3 | −2.127 | [−3.487, −0.767] | INFERIOR, 1.3 τ |
| W6 − O3.3b | −2.046 | [−2.678, −1.414] | INFERIOR, 1.2 τ |
| **W6 − O3.D** | **−3.620** | **[−4.521, −2.718]** | **INFERIOR, 2.1 τ** |
| O3.1 − O3.3 | +0.810 | [−0.551, +2.172] | inconclusive |
| O3.1 − O3.3b | +0.891 | [−0.046, +1.829] | inconclusive |
| **O3.1 − O3.D** | **−0.682** | [−1.702, +0.338] | inconclusive (O3.D WORSE) |

**O3.D — block 0 for the on-site readout — is the WORST arm since O3 itself.** Median 15.04,
worse than its own parent O3.1 (14.44), 2.1 τ inferior to W6, and worse than O3.1 at EVERY
shell including the flanking Pb (39.98 against 37.41). **The registered three-way reading
therefore lands on branch three: neither the head's own correlation-2 content nor the base's
block-0 content recovers the 2.3 meV/Å. The deficit is not in what the on-site readout is
fed.**

**And O3.D is the first arm since O3 to saturate.** On-site p95 shift **2.767 eV** against
`Δ_Z` = 3.079, with a nonzero saturated fraction (0.002). Giving the readout 512 base invariants
makes it ask for a larger correction than the bound allows — the same failure O3 had, reached by
the opposite route (O3 through a badly conditioned input, O3.D through too rich a one).

**O3.D is also the only arm to FAIL G2a.** Energy at 79 atoms 0.6621 meV/atom against W6's
0.5151: mean d **+0.134**, one-sided 95 % upper **+0.213**, outside the 0.10 margin.

| arm | E79 (meV/atom) | vs W6, 95 % upper | G2b median (meV) | on-site p95 shift (eV) |
|---|---:|---|---:|---:|
| W6 | 0.5151 | — | +5.0 | 0.846 |
| O3.1 | 0.4552 | −0.0245 PASS | −1.8 | 0.740 |
| O3.2 | 0.4340 | −0.0314 PASS | −0.3 | 0.722 |
| **O3.3** | **0.3387** | **−0.1108 PASS** | +3.2 | 1.526 |
| **O3.3b** | **0.3395** | **−0.1280 PASS** | +0.3 | 1.569 |
| O3.D | 0.6621 | **+0.2132 FAIL** | +0.6 | 2.767 |

**The finding worth carrying forward is a DECOUPLING of energy from forces.** O3.3 doubles the
on-site correction it uses (p95 1.53 eV against O3.1's 0.74) and converts that almost entirely
into ENERGY: E79 improves 26 % over O3.1 and is **34 % better than the shipped W6 head**, the
best energies anywhere in this programme. The same capacity buys only +0.81 meV/Å of force,
which does not clear τ. So the correlation-2 content is real, is used, and is worth having — it
simply is not what the force deficit is made of.

**Shell localisation, cohort medians:**

| shell (Å) | W6 | O3.1 | O3.2 | O3.3 | O3.3b | O3.D |
|---|---:|---:|---:|---:|---:|---:|
| 2–4 | 19.83 | 25.54 | 24.99 | 22.82 | 22.63 | 27.30 |
| 4–6 | 13.81 | 17.49 | 17.00 | 15.72 | 15.64 | 17.87 |
| 6–8 | 9.73 | 12.62 | 12.09 | 11.92 | 11.78 | 12.75 |
| 8–10 | 9.86 | 11.11 | 10.25 | 11.22 | 11.15 | 11.60 |
| 10–12 | 9.02 | 10.48 | 9.38 | 10.51 | 10.62 | 10.74 |
| **flanking Pb** | 27.40 | 37.41 | 35.55 | **33.64** | **33.21** | 39.98 |
| first-shell Cl | 16.55 | 20.56 | 20.26 | 17.96 | 18.14 | 21.43 |
| rest of 2–4 | 10.26 | 11.85 | 11.18 | 11.76 | 12.25 | 13.66 |

O3.3/O3.3b improve on O3.1 in the NEAR field (flanking Pb 37.41 → 33.2, first-shell Cl
20.56 → 18.0) and are flat-to-worse beyond 8 Å — the opposite localisation to O3.2's, and the
right one for an on-site change acting through the near field.

**Next arm indicated by the registration**, not opened: the deficit is in the rank-1 block, and
the arm is the equivariant correlation-2 product for `v_i`. It is registered before running.

## O5 — crystal-field head: REGISTRATION (2026-09-14, before either arm runs)

*(Coadvisor ruling. Adopted as specified. This restores in closed form what the original
specification's §8.2 carried and v4 simplified away.)*

**One damped-Coulomb primitive over neighbours generates all three on-site channels:**

    V_i   = sum_j q_{Z_j} erf(r_ij / a_{Z_i Z_j}) / r_ij  f_cut(r_ij)      potential  (l = 0)
    E_i   = -grad_i V_i                                                     field      (l = 1)
    grad E_i = traceless Hessian of V_i                                     gradient   (l = 2)

    d eps_i = s_Z (V_i - V_bar_Z)               on-site shift, linear, no MLP
    a_i (E_i . T_sp),  b_i (grad E_i : T_pp)    rank-1 and rank-2 blocks, species scalars
    t_ij -> t_ij [1 + beta_{ZZ'} (V_i - V_j)^2] hopping as bond polarity

**Parameters:** one charge and one damping length per species; a scale per species per channel;
one polarity coefficient per pair. **Charges are initialised from the pristine Mulliken means
the head already computes**, so nothing external enters and the arm keeps O3's defining property.

**Why this and not more angular machinery.** Everything is TWO-BODY: the analytic force kernel
extends by a closed-form `dV_i/dR_j`, no triplet list is needed, and CUDA capture still works —
all three of which O3.2's angular screening gave up. The rank-1 block gets the FIELD, which is
what drives s–p mixing at a Pb next to a vacancy; the rank-2 block gets the field GRADIENT,
which is the p-splitting. That is the physics Arm 1 found the directional block to carry, now
with its correct radial form instead of a neighbour-direction sum. The bulk `q0` modulation
follows from the potential's bond-length dependence rather than from a readout — which is
exactly the channel the `q0` diagnostic showed carries the residual (+0.939 across shells).
`(V_i - V_j)^2` is Harrison's polar-bond renormalisation and is a real candidate for the ~20 %
modulation W6 uses on the flanking bonds that O3.2's angular screening could not stand in for.

### Arm O5a — the crystal-field head

O3.3b with the on-site, rank-1 and rank-2 inputs replaced by `(V_i, E_i, grad E_i)`; signed
species charges initialised from the pristine Mulliken means; species damping lengths; linear
readouts. Hopping modulation as polarity, `beta` **zero-initialised** so the arm starts exactly
at O3.3b. A1.1 standardisation stays on the scalar input and **the B-basis is kept as a second
`l = 0` input** (it earned that: it is worth 34 % on energies).

### Arm O5b — radial-resolution control

O3.3b with `N_r = 16` against the standing 6. One flag. Part of the stall may be nothing more
than six Bessels, and this is the cheapest way to find out.

### Gates (both arms, registered before results)

Paired TOST vs W6 (τ = 1.7) and vs O3.3b; the by-shell table with flanking Pb and first-shell
Cl called out; **the `q0` disagreement with W6 by shell**, which is the diagnostic that
identified this target; saturation including whether the new channels merely re-saturate the
on-site bound; G2a and G2b; the analytic-force FD gate; `capture.fits()`.

### Reading, written before the results

- **O5a closes the near field and the `q0` disagreement** → adopt the crystal-field head.
- **O5b closes it and O5a does not** → the basis was simply under-resolved, and the physics
  argument was unnecessary.
  **VOID as written (2026-09-14, see the architecture audit below).** `N_r` enters the
  correlation-2 scalar block QUADRATICALLY, so O5b is not a resolution control: it is a 6.4x
  widening of the readout input and a 5x increase in live parameters, and it carries the only
  real generalisation gap in the programme (+1.1 meV/Å). Its result must be read as a CAPACITY
  arm. **The radial-resolution question is still open**, and the control that actually asks it
  is a SMALLER cutoff at fixed `N_r`, not a larger `N_r`.
- **Neither** → the deficit is the pretrained prior, and the cached-Jacobian W6 (below) is the
  model.

### Parked, explicitly

The `Δ_Z` amendment, triplet screening, and distillation. **Also registered but not scheduled:
the cached block-0 Jacobian.** Block 0 is frozen, so `d(features)/dR` is parameter-independent
and cacheable per frame (sparse within 6 Å, tens of MB/frame). The force loss would then need no
`create_graph` traversal of block 0, taking W6's training from 124 s/epoch to roughly the
decoupled head's speed **with no change to the model**. That is the fallback if the physics arms
do not land, and it is worth doing on its own merits.

---

## O5 — CORRECTION OF RECORD and RE-REGISTRATION (2026-09-14, written before either arm runs)

Three entries below are corrections to what this tracker already says, written because the
record has to describe what was BUILT, not what was intended. They are listed before the
re-registration because the re-registration only makes sense against them.

### Correction 1 — the cancelled `dscc_o5a_s*` runs were not O5a

The ruling says, twice and unambiguously: `delta_eps_i = s_Z (V_i - V_bar_Z)`, **"on-site
shift, linear, no MLP"**, and **"linear readouts"**. What was built and launched appended the
three invariants `(V_i, |E_i|, tr(grad E . grad E))` to the 211-channel scalar block and fed
the 214-wide result to the existing on-site MLP. That is a strictly larger hypothesis class
than the one registered, it confounds the crystal field with an MLP the ruling declined, and
the result would not have answered the question the arm was set to answer. **Those runs are
cancelled and their partial results are not to be read as O5a.** The queue file is at
`~/runs/o5a_mlp_CANCELLED.txt`.

### Correction 2 — O3.3's invariants were pairwise, not the specified norms

The ruling for O3.3 specified the correlation-2 invariants `|A^(1)_n|^2` of the head's own
A-tensors — 6 channels at `N_r = 6`. What was built is the full pairwise Gram
`A^(1)_n . A^(1)_m` — 21 channels, containing the 6 as its diagonal. This was not disclosed
when O3.3 was registered. **It propagates into O3.3b and into O5b**, both of which are built
on the O3.3 descriptor, so every reading of those arms is a reading of the 21-channel form.
The user's decision is not to re-run O3.3; the record states what the arm was.

### Correction 3 — gates named in the ruling that had not been run

`capture.fits()` and the analytic-force FD gate were named in the O5 gate list and neither had
been exercised. Both have now been run and one of them found a live bug; see **Gate status**
below. Separately: O3.3 and O3.D ran sequentially rather than in parallel, the optional
learning curve was never run, and **O3.3b was not the coadvisor's arm — it was proposed here**
(user-approved) after the instability diagnosis.

### Arm O5a (corrected) — the crystal field with LINEAR readouts

O3.3b with the damped-Coulomb primitive, **exactly as the ruling writes it**:

| term | form | flag |
|---|---|---|
| potential (l = 0) | `V_i = sum_j q_{Z_j} erf(r_ij / a_{Z_iZ_j}) / r_ij . f_cut(r_ij)` | |
| on-site | `delta_eps_i = s_Z (V_i - V_bar_Z)`, **linear, no MLP** | `--cf_linear` |
| rank-1 | `a_i (E_i . T_sp)`, `E_i = -grad_i V_i` | |
| rank-2 | `b_i (grad E_i : T_pp)`, traceless | |
| hopping | `t_ij -> t_ij [1 + beta_{ZZ'} (V_i - V_j)^2]`, `beta` zero-init, symmetrised | |

`q_Z` from the pristine Mulliken means (hard-fails if they are zero — the LES lesson), `a_Z`
per species, `V_bar_Z` the pristine per-species mean, `s_Z` zero-init so the arm **starts
exactly at O3.3b**. A1.1 standardisation stays on the scalar block; the B-basis is kept as the
second `l = 0` input; **the three invariants do not enter the readout** in this arm.

Two departures from the letter of the ruling, stated rather than left to be found:

1. **`s_Z` is `[n_elements, 2]`, not one scalar per species** — an independent shift for the s
   and p levels. The natural generalisation (the ruling's own `T_sp`/`T_pp` split treats the
   shells separately), one extra number per species, but it is a choice.
2. **`delta_eps_i` sits OUTSIDE A1's `Delta_Z tanh` bound by construction.** It is a linear
   term; bounding it would make it not the term that was specified. `V` is bounded in practice
   by the cutoff and the fixed charges, and the on-site saturation report still covers the
   tanh'd part.

### Arm O5c — the MLP fallback, run alongside rather than after

The ruling offered "a small MLP on `(V_i, |E_i|, tr(grad E . grad E))` with A1.1
standardisation" as "the lightweight place for one, but I would not start there". Both are
queued together, because the pair IS the measurement: same primitive, same everything else,
linear against MLP. `--cf_mlp` adds a dedicated `[3 + elem_dim, hidden, 2]` MLP with a
zero-initialised final layer, inside the `Delta_Z tanh`.

**One departure, stated:** the ruling's fallback reads as an MLP *instead of* the linear form
on those three inputs. O5c keeps the B-basis readout alongside it, so its on-site channel has
two MLPs. That is deliberate — the only difference between O5a and O5c is then the form of the
crystal-field term, which is the comparison worth making — but it is not the letter.

### The one unbounded term, and what watches it

`t_ij [1 + beta (V_i - V_j)^2]` is the only term in either arm that no `tanh` bounds, and
`(V_i - V_j)^2 >= 0`, so a `beta` that goes negative drives the factor through **zero** and
flips the sign of a hopping. That is the documented failure of the linear modulation form and
the reason W6's hopping is `1 + range tanh`. The ruling does not bound it, so it is **watched**
instead: `SlaterKosterH.polarity_floor` accumulates `min(1 + beta dV^2)` over every edge, the
trainer reads it once an epoch into `history.json` as `polarity_floor`, and **the run stops**
the moment it is not positive. A sign flip otherwise yields a complete run of plausible
numbers with nothing visibly wrong.

### Gate status (of the gates the ruling named)

| gate | status |
|---|---|
| paired TOST vs W6 (τ = 1.7) and vs O3.3b | on the result, as for every arm |
| by-shell table, flanking Pb / first-shell Cl | on the result |
| `q0` disagreement with W6 by shell | on the result — the diagnostic that chose this target |
| saturation, incl. whether the new channels re-saturate the bound | on the result |
| energies G2a / G2b | on the result |
| **analytic-force FD gate** | **not applicable, and now enforced** — see below |
| **`capture.fits()`** | **run; it found a live bug** — see below |

**`capture.fits()` found that capture was silently wrong for O5.** The ruling's argument for
the damped-Coulomb primitive was that it is two-body, so the capture and analytic paths keep
working. The primitive is two-body and its padding is exact — `V` and `E` are bit-identical
under 216, 1240 and 3288 dummy edges, and the force discrepancy (1.7e-18, autograd
reassociation in the traceless-Hessian scatter) does not move with the dummy count, which is
the gate that distinguishes reassociation from contamination. But the captured tail **is** the
closed-form gradient `h0_position_gradient`, and the closed form contains no `dV/dR`.
`MACEDSCC.forward` already refused the analytic route for `crystal_field`; `GraphedStep` kept
its own separate guard list and did not. An O5-**linear** model therefore captured without
complaint and replayed forces **19 meV/Å wrong**, the crystal-field terms simply absent from
the gradient, widths matching by coincidence so nothing raised. (The MLP form crashed instead,
101 channels against 104, which is how it was found.) Fixed by making both callers ask one
predicate, `analytic.analytic_refusal(model)`; capture now refuses O5 with a reason, and a gate
asserts the two callers agree. **Capture and the analytic force for O5 wait on extending the
kernel with `dV/dR`; neither is on the training path, so neither costs the arm anything.**

### Gates added with these arms (27 in `TestO5aCrystalField` / `TestO5OnSiteForms`, 4 in `test_capture.py`)

The potential is a RETURN VALUE of `geometric_inputs`, not stashed on the module (it was, and
it worked only because the producer happened to run immediately before the consumer); the
linear shift starts at zero and moves; with the crystal field on there is exactly **one**
vector channel and it is the field itself, so the rank-1 block is literally `a_i (E_i . T_sp)`
rather than the field broadcast across `N_r` channels and contracted by a redundant mix; the
charges come from the pristine means; forces match finite differences in both forms; the head
still reads nothing from the base; the polarity floor is recorded, reset, and acted on.

---

## W6 → O5 ARCHITECTURE AUDIT (2026-09-14, requested: has anything changed beyond the registered scope?)

### 1. Verified IDENTICAL between W6 and O5a (checked, not assumed)

The energy expression is gated on `scf_free`, **not** on `decoupled`: both arms sum
`E_base + dF_band(H0) + E_host + E_M + C_Q`, the same code path, no term dropped for the
decoupled head. Hyperparameters read from the runs' own logs, not from defaults:
`eta 1.0986`, `beta_a 0.00`, `beta_b 0.00`, `Delta_Z 3.079 eV`, `l2 1.8e-6`, `lr 2e-3` cosine
to `1e-4`, knee 2.0, grad_clip 10, 60 epochs, batch 4, `energy_weight 0.05`, `force_weight 1.0`,
`gap_weight 1.0`, `e_gap 2.4`, `r_cut 10.0`, `avg_window 10`, same folds and seeds.

`beta_a = beta_b = 0` in **both** means `a_i` and `b_i` are pure species scalars in W6 and in
O5a alike (`site_coefficients`: "at beta = 0 this is `b_max tanh(beta_Z)`, the species
coefficient EXACTLY") -- which is also exactly what the O5 ruling asked for.

### 2. Registered changes, all traced to their registration

`decoupled` + `hop_modulation = False` (O3); standardisation and `readout_init_scale`
0.05 -> 0.1 (A1.1 at O3.1, registered in as many words); correlation-2 (O3.3, with the
21-vs-6 correction already on record); `sat_weight` 1e-3 -> 1.0 (O3.3b); crystal field,
polarity, `cf_linear` (O5a); `n_vectors = 1` (today).

### 3. Registered but NEVER ISOLATED: every H0 input channel's REACH changed, and its RESOLUTION with it

Cutoffs read off the saved checkpoints, not inferred: `h0.r_cut` 10.0 and `h0.q_cut` 4.5 on every
model; `descriptor.r_cut` 10.0 on O3.3b, O5b and O5lin; `cfield.r_cut` 10.0 on O5lin. The base is
`r_max` **6.0 A** with two interaction blocks, and the head reads `first_block` ONLY -- so W6's
scalar and vector inputs have a ONE-HOP, 6.0 A receptive field, not the two-hop 12 A.

| H0 input | W6 | O3 / O3.1 / O3.3 / O3.3b / O5b | O5a / O5c |
|---|---|---|---|
| `l = 0` on-site | base block 0, **6.0 A**, learned many-body within the hop | descriptor, **10 A**, two-body sums | same (B-basis kept) |
| `l = 1` rank-1 | base `l = 1`, **6.0 A** | descriptor, **10 A** | crystal field `E`, **10 A** |
| `l = 2` rank-2 | fixed quadrupole, **4.5 A**, no learned coefficients | descriptor, **10 A**, learned `c2` | field gradient `grad E`, **10 A** |

Every channel's reach grew. This follows from the registered "Radial family" decision -- "Bessel,
`N_r = 6`, on `[0, r_cut]`, times H0's own taper, so the head carries ONE cutoff radius and one
smoothness class" -- so it is in scope. But that argument is UNIFORMITY, and this tracker records a
PHYSICS argument for 4.5 A on the `l = 2` channel specifically: "it reaches the first coordination
shell and the Cs neighbours (Pb-Cl 2.8, Cs-Cl 3.5-4.1 A) and **not the second Pb shell**".

**The operative variable may be RESOLUTION, not reach.** `N_r` Bessels spread over `[0, r_cut]`
resolve `r_cut / N_r`:

| basis | radial resolution | first shell (Pb-Cl 2.8 A) |
|---|---|---|
| W6's base features, 6.0 A | (learned, many-body) | one hop |
| O3+ descriptor, `N_r = 6` on 10 A | **1.67 A** | ~1.7 shells per basis function |
| the same 6 on 6.0 A | 1.00 A | |
| the same 6 on 4.5 A | 0.75 A | |

The same six functions spread over 10 A resolve the near field roughly **2x more coarsely** than
over the window W6's inputs actually covered. **O5b is evidence that this bites**: raising `N_r`
6 -> 16 improved the TRAINING fit markedly (12.1 against O3.3b's 13.7, section 6) -- which is what
buying back near-field resolution looks like -- but it bought it at 5x the live parameters and
brought the only real generalisation gap in the programme (+1.1).

**So the untried control is a SMALLER cutoff, not a larger `N_r`**: it buys the same resolution at
fixed `N_r` and fixed parameter count, and it is the opposite lever from O5b. Matched to what W6
actually had, that is **6.0 A for `l = 0` and `l = 1`, 4.5 A for `l = 2`** -- not 4.5 everywhere,
which would be smaller than anything W6 used on the first two channels. Note this is a control on
**O3.3b**, not on O5a: O5a's rank-1 and rank-2 slots carry the crystal field, whose reach is
`cfield.r_cut`, so a descriptor-cutoff arm tests the B-basis channel underneath both.

Not a flag: `GeometricDescriptor` carries one `r_cut` across all `l`, so per-`l` cutoffs are an
implementation and a registration, not a rerun. Registered here as the recommendation; not built.

### 4. Parameter counts: O5a has 6x FEWER LIVE parameters than W6

A `requires_grad` count is not a capacity count: modules that are never evaluated still report
their weights. `f_read`/`g_read` are unreachable whenever `beta_a = beta_b = 0` (every arm), and
`sk.hop` is unreachable whenever `hop_modulation = False` (every arm from O3 on; **live in W6**).
Verified by backward -- the listed parameters receive exactly zero gradient.

| model | total `h0` | **live** | structurally dead | scalar block |
|---|---:|---:|---:|---:|
| W6 | 177,132 | **110,314** | 66,818 | base block-0 invariants |
| O3.1 | 17,642 | **6,244** | 11,398 | 18 |
| O3.3b | 79,402 | **18,596** | 60,806 | 211 |
| **O5b** | **447,612** | **92,406** | 355,206 | **1,361** |
| O5a-lin | 79,414 | **18,608** | 60,806 | 211 |
| O5c-mlp | 81,266 | **19,692** | 61,574 | 211 |

O5a carries 18,608 live parameters against W6's 110,314 -- a sixth. Every decoupled arm EXCEPT
O5b is far smaller than the model it is trying to match; O5b at 92,406 is within 20 % of W6 and is
the exception (section 5).

**The single largest registered difference is `sk.hop`**: 71,044 of W6's 110,314 live parameters --
64 % of the model -- is the hopping-modulation MLP that made W6's `t_ij` depend on the two atoms'
BASE FEATURES. O3 deleted it (`hop_modulation = False`), leaving two-body hoppings `v0 . radial(r)`.
If the user asks what could be impacting performance, that is the answer: the programme's whole
premise is whether physics (three-body screening, then the crystal field) can replace a learned
environment dependence that was two thirds of the shipped model.

### 5. O5b was mis-registered as a cheap control. It is the largest arm in the programme.

O5b was registered as "O3.3b with `N_r = 16` against the standing 6. **One flag.**" With
correlation-2 on, `N_r` enters the scalar block QUADRATICALLY -- the `l = 0` Gram is
`n0 (n0+1) / 2` with `n0 = N_r x n_species`:

* `N_r = 6`  -> `n0 = 18`, block `18 + 171 + 21 + 1 = 211`, **18,596 live parameters**
* `N_r = 16` -> `n0 = 48`, block `48 + 1176 + 136 + 1 = 1,361`, **92,406 live parameters**

Not a 2.7x radial-resolution control: a **6.4x** widening of the readout input and a **5x**
increase in live parameters over its own parent -- the largest of the decoupled arms, and the only
one that approaches W6's live-parameter count (92,406 against 110,314). The registration did not
say so and I did not notice when writing it. Its result has to be read as
a capacity arm, not as a resolution control, and **the resolution question remains unanswered.**

### 6. Train-vs-held probe: the direct measurement, and a correction

`history.json`'s `force` is a stratum-WEIGHTED loss term accumulated over an epoch while the
weights were still moving. I used it to claim O5b "fits ~2 % worse on the data it trained on".
**That was wrong.** Re-evaluating the saved checkpoints with the same `evaluate` that produces
the held numbers (200-frame training sample; per-component meV/A):

| model | epoch | train | held | gap |
|---|---:|---:|---:|---:|
| W6 s0 | 60 | 11.1 | 11.2 | +0.1 |
| O3.1 s1 | 60 | 13.9 | 14.2 | +0.4 |
| O3.3b s0 | 60 | 13.7 | 13.6 | -0.1 |
| **O5b s0** | 60 | **12.1** | **13.2** | **+1.1** |
| O5a-lin s0 | 57 | 13.7 | 13.8 | +0.1 |
| O5c-mlp s0 | 54 | 14.7 | 14.6 | -0.1 |

**The O5a and O5c rows are mid-run checkpoints at DIFFERENT epochs (57 and 54) and are not
comparable to each other or to the finished ep-60 rows.** They are listed for their gap, which is
what this table is for, not for their level.

O5b fits the training set **better** than O3.3b (12.1 against 13.7) and carries the only real
generalisation gap in the table (+1.1 where every other arm is within 0.4 of zero). That is the
signature of the capacity increase in section 5, and it is consistent with O5b winning on seed 0
(13.2 against 13.6) while losing the six-seed mean (14.5 against 13.6): it uses the extra
capacity, and across seeds the added variance costs more than the added fit.

Every other arm sits at gap ~0 -- W6 included, at a LOWER absolute error with 6x more live
parameters. So the O3-lineage deficit is not overfitting and not a parameter shortfall; the
untested candidate the audit surfaces is section 3, the rank-2 reach.

---

## M1 / M2 — THE REGRESSIONS THAT DECIDE O6, AND THE DECISION (2026-09-15)

The O5 wave was **killed at 4 of 12 runs** on the user's explicit instruction (a stated deviation
from the plan's "let them finish for the record"). `dscc_o5lin_s0/s1` and `dscc_o5mlp_s0/s1`
completed 60 epochs and are the record; `s2`/`s3` were killed mid-run and their partial histories
are not to be read. Queues moved to `~/runs/o5{lin,mlp}_queue_KILLED.txt`.

### Before the numbers: `pre` is not a function of `r` by construction

`pair_correction` builds the hopping pre-activation from `feats_i + feats_j`,
`|feats_i - feats_j|` and the species embeddings, and its own docstring says none of them is a
function of `r`. Two periodic images of one node pair at different separations therefore share
one `pre`. So M1 does not ask whether the MLP was *given* the distance; it asks whether it
**learned** to encode it. That makes a high `R^2` the interesting outcome, and it is not what
came back.

### M1 — W6's hopping factor `exp(eta tanh pre)` against `r_ij`

60 training frames, 547,730 edges, `eta = 1.0986` (so the factor is bounded to [0.333, 3.0]).
`R^2` is the CEILING for any function of `r`: each edge predicted by its `r`-bin mean.

| pair | factor sd | observed range | `R^2(r)` all | `R^2(r)` r < 4 Å |
|---|---|---|---|---|
| Cl-Cl | 0.065-0.103 | 0.52 - 1.83 | 0.006-0.010 | 0.015-0.049 |
| Cl-Cs | 0.084-0.218 | 0.39 - 2.11 | 0.003-0.019 | 0.009-0.028 |
| Cl-Pb | 0.072-0.458 | 0.36 - 2.89 | 0.004-0.052 | 0.025-**0.122** |
| Cs-Cs | 0.094-0.253 | 0.42 - 2.33 | 0.009-0.013 | (no bonded edges) |
| Cs-Pb | 0.123-0.482 | 0.35 - 2.91 | 0.003-0.017 | (no bonded edges) |
| Pb-Pb | 0.058-0.585 | 0.36 - 2.95 | 0.018-0.026 | (no bonded edges) |

**The modulation is doing a great deal and none of it is radial.** It spans nearly the full
bound (0.36 to 2.95, a factor of 8) and rides its stops, yet distance explains **0.3-5 %** of its
variance overall and at most **12 %** in the bonded shell. The 71,044-parameter hopping MLP is a
genuine environment dependence, not a flexible radial correction in disguise.

### M2 — W6's on-site correction against the atom's 12 sorted neighbour distances

k-nearest-neighbour regression on a held-out half (so it is prediction, not memorisation),
alongside a linear fit:

| species | shell | p95 \|correction\| | `R^2` kNN | `R^2` linear |
|---|---|---|---|---|
| Cl | s / p | 0.144 / 0.445 eV | 0.64 / 0.72 | 0.58 / 0.76 |
| Cs | s / p | 0.232 / 0.475 eV | 0.48 / 0.39 | 0.51 / 0.47 |
| Pb | s / p | 0.197 / 0.550 eV | 0.70 / 0.77 | 0.64 / 0.61 |

The ON-SITE term **is** substantially radial -- roughly half to three quarters of it. But it is
the small term (p95 0.14-0.55 eV), it is not what O3 deleted, and a radial parameterisation of it
would be replacing the part of W6 that the geometric heads already reproduce.

### The decision, by the rule as written

> "Regression `R^2` high → O6. `R^2` low, or O6 stalls → block-0 copy with cached Jacobians is
> the model, the geometric head is closed."

**`R^2` is low. O6 is NOT indicated and is not built.** Its central premise -- that a learned SK
radial `t_Harrison(r)[1 + tanh(Σ c_n B_n(r))]` replaces the hopping MLP with ~300 parameters --
is refuted by M1: the function being replaced is not a function of `r`. Building O6 would spend
six seeds confirming a negative that two regressions already give.

**One thing the rule does not cover, flagged as outside it, not as a proposal.** M1 rules out a
function of `r_ij` ALONE. It does not rule out a learned function of both atoms' GEOMETRIC
descriptors on the bond -- `f(A0_i + A0_j, |A0_i - A0_j|, Z_i, Z_j)`, which is W6's `sk.hop` with
block 0 swapped for the head's own descriptor. No arm has run it: O3 deleted the hop MLP as a
founding principle, and O3.2 and O5 both put PHYSICS on the bond rather than a learned function of
the descriptor. The decision rule is binary and is followed as written; the coadvisor should see
that "closed" rests on that binary.

**The geometric head is closed.** The record states the finding: *the pretrained block-0
representation carries about 2 meV/Å of force accuracy (W6 11.6 against the best decoupled arm's
13.6) that geometric tight-binding parameterisations at this data scale do not recover, and the
carrier is a genuinely non-radial, environment-dependent modulation of the hoppings.*

## The guaranteed route: LESS work than assumed for accuracy, MORE for speed

**Accuracy: it already exists.** In training, with `cache_base` on, `MACEDSCC.forward` takes the
`cached_base` branch and evaluates **`first_block` alone** from the frozen base modules, with
`E_base`/`F_base` read from the cache -- the full base is never run. That IS "the head carries a
frozen copy of block 0, W6's weights load unchanged, bit-for-bit W6". It is the shipped model at
**11.6 meV/Å**, four seeds, today. What is missing is not the model but **capture**:
`GraphedStep` refuses any head that is not `decoupled`, because its captured tail is the
closed-form gradient. Extending capture to a block-0-reading head means bringing block 0 inside
the captured region -- real work, but bounded and two-body-free.

**Speed: the cached-Jacobian item does not survive its own arithmetic.** Measured on b3, warmed,
medians (the first call is 1.5 s of CUDA warm-up and was nearly reported as the cost):

| quantity | measured |
|---|---|
| channels the head reads | `n_scalars` 512 + 3 × `n_vectors` 512 = **F = 2048** |
| base graph mean degree within `r_max` | **22.8** |
| one vector-Jacobian product | **11.6 ms** |
| full `∂features/∂R`, forward mode (3N = 237 tangents) | **2.7 s/frame** → 0.6 h for 785 frames |
| full `∂features/∂R`, reverse mode (F = 2048) | 23.7 s/frame → 5.2 h |
| storage, sparse at degree 22.8 | **89 MB/frame → 70 GB** over the training set |
| block 0, forward + ONE backward | 78.6 ms / 4-frame batch → 15.4 s/epoch |
| **block 0, forward + `create_graph` + the SECOND traversal** | **115.2 ms / batch → 22.6 s/epoch** |

**Block 0's WHOLE training-step cost is 22.6 s/epoch — 18 % of W6's ~124 s.** Measured in
isolation, with nothing else in the timed region: forward, the `create_graph` backward, and the
second traversal that the loss makes through the retained graph, which is what `create_graph`
exists for. (An earlier version of this line reported only the first traversal and called it a
ceiling; it is a floor, and the isolated measurement was made because of that. The second
traversal turns out to be 0.5x the first, not the 2-4x that would have overturned the reading.)

W6 is ~124 s/epoch against O3's ~32, a gap of 92 s. Block 0 accounts for **23 of those 92 s**.
The rest is the head itself: W6's `sk.hop` is live and evaluated per unique node pair, and it is
71,044 of its 110,314 live parameters, where O3's is dead. **Removing block 0's traversal
entirely cannot take W6 below ~101 s/epoch**, so the cached Jacobian cannot deliver what the plan
asks of it — and it costs 70 GB plus per-step I/O to buy at most 23 s. **Recommend NOT building
it.**

**The version of the idea that DOES survive the arithmetic is not a cache.** The force loss never
needs `J` itself, only `J·r` for the force residual `r`: the parameter gradient of the force term
contracts to `(d2E/dtheta df)·(J r)`, and `J r` is ONE forward-mode JVP with tangent
`r` in R^{3N}. That is ~11.6 ms/frame — about 9 s/epoch — **exact, with no storage at all**.
Registered here as the speed item worth having; not built.

## Process fixes applied

* **Launch gate: `tests/extensions/dscc/test_analytic_composition.py`** (13 gates). For every
  composition of head modules the closed-form gradient either **equals autograd** or is
  **refused** — there is no third option, and a term added to `H0` but not to the kernel cannot
  pass silently. It also asserts capture declines exactly what the closed form declines, and that
  every H0-changing flag is either in the kernel with a covered case or named in
  `analytic_refusal`. This is the gate the 19 meV/Å bug would have caught.
* **The Gram is recorded as built** — O3.3, O3.3b and O5b are 21-channel results (§Correction 2).
* **O5b's reading is marked void** — it is a 5× capacity arm, not a resolution control.
* **The train/held gap table and the by-shell table are standard readouts** — they are what made
  this diagnosable, and both are in the O5 report.

---

## CAPTURE FOR A BLOCK-0-READING HEAD — BUILT (2026-09-15)

Registered as "the one real piece of engineering left on the production route" and built rather
than left open. Three blockers, all found by building it, and one of them is much larger than
this route.

### 1. The gradient — solved, and it fixes O5 as a side effect

The captured tail WAS the closed-form gradient and nothing else, which is why capture refused
every head the kernel could not differentiate, and why it silently mis-served an O5 head by
19 meV/Å before that refusal existed. **Capture does not need a closed form.** It needs a
gradient with a fixed kernel sequence and fixed shapes, and at fixed cotangent `X` that is one
ordinary backward: `h0_position_gradient_vjp` rebuilds the features and the assembly from
`positions` with grad on, contracts against a detached `X`, and takes one backward. No
`create_graph`, no parameter gradients.

**A backward IS capturable** — verified directly before anything was built on it: a capture
containing `torch.autograd.grad` replays **bit-identically** to eager (max |diff| 0.0).

`head_tail` now dispatches on `analytic_refusal`: closed form where valid (~2.7 ms), VJP where
not. Consequences:

* **O5 captures correctly now** rather than being refused — `test_an_o5_head_now_captures_through_the_vjp`, both forms, replay vs eager < 1e-9 on energy, `dq` and forces.
* **W6 captures** — `test_a_block_zero_reading_head_captures`, < 1e-9 on all three, and on a moved frame.
* `analytic_refusal` gained the term it was missing: **a non-decoupled head**. The kernel's own
  first line refuses one, but the predicate did not say so, so W6 routed into a function that
  raised. A predicate that callers trust has to be complete.
* `capture_refusal` is now its own, **shorter** list: what capture cannot take is a
  **value-dependent shape**, not a missing closed form.

### 2. Two value-dependent shapes on the block-0 path — both removed, both bit-gated

| where | was | now |
|---|---|---|
| `SlaterKosterH.pair_correction` | `torch.unique` over node-pair keys (the O1 dedup) | `dedupe_pairs=False` during capture: the same MLP per EDGE. Energy and `dq` bit-identical; forces within one ULP (3.5e-17 — the GEMM has a different row count, so its reduction is tiled differently). Costs the 2-3× O1 removed, for the life of the capture only. |
| `quadrupole_descriptor` | `r[inside]` boolean indexing | masked by multiplication: the weight is zero beyond `q_cut` by construction, so every excluded edge contributes exactly 0.0 and the sum is the same sum, with a shape that no longer depends on the positions. |

### 3. CORRECTION: production capture has REGRESSED in this environment. It is not my change, and it is not this route.

**An earlier draft of this section claimed "CUDA-graph capture has never worked at production
width, for any head". That was wrong, and the record refutes it.** §"O5 TIMING (idle A4000, fake
O3 weights, P4.3 protocol, `dscc_benchmark.py --graphed`)" is a methodical measurement at
production width on this very card: base captured 32.4 ms, **model graphed 39.7 ms** at 79 atoms
against a <= 53 ms deliverable. Capture worked. The claim was made from a single failed run
tonight and should not have been made.

What is true, established by reverting: **production capture fails today, and it failed before
any of this session's changes.** With the four modified modules reverted to HEAD, the same
benchmark on the same checkpoint fails identically. Two distinct faults, in order:

1. **`torch.compile` recompiled INSIDE the captured region** -- `h0_position_gradient` calls
   `_compiled(_site_blocks)`, and dynamo/inductor running during `g_tail`'s capture is a hard
   CUDA failure. **FIXED, by construction**: `GraphedStep` now turns `analytic.COMPILE` off for
   the life of the capture and restores it on `release()`. This costs nothing -- those kernels
   are compiled because they are LAUNCH-bound eagerly, and removing launch overhead is exactly
   what capture itself does, so inside a graph the compile buys nothing. Capture now BUILDS at
   production width with no environment variable.
2. **The replay then raised an illegal memory access** at `H_host.copy_(self.H)`, on the FIRST
   replay of the very frame the capture was built from. **FIXED.**

   Localisation, because the wrong answers were instructive: `g_assembly` alone captures and
   replays correctly; it is still correct after `_base` and `_tail` are WARMED UP; it breaks the
   moment a SECOND GRAPH IS CAPTURED, either one. Not the shared pool (one pool per graph changes
   nothing). Not the assembly's kernels (clean eagerly under `PYTORCH_NO_CUDA_MEMORY_CACHING=1`,
   which turns a benign overrun into a fault). Not a general torch bug (a minimal two-graph
   reproducer replays the first correctly after capturing the second, torch 2.13.0+cu129).

   **The cause: the assembly's OUTPUTS were allocated inside the captured region**, so `H` and
   `ev` lived in a graph pool -- and they are consumed by the two graphs captured after it, whose
   capture can reuse that memory underneath them. The inputs were already static buffers for
   exactly this reason; the outputs now are too. `_assembly` is run once before capture to size
   `self.H` and `self.ev`, and the captured region copies into them.

   **Verified against the recorded measurement.** `dscc_benchmark.py --graphed --fixed_cell`,
   production O3 checkpoint, 79 atoms, idle A4000:

   | | recorded (P4.3) | now |
   |---|---|---|
   | base captured | 32.4 ms | **32.27 ms** |
   | model graphed | **39.7 ms** | **39.67 ms** |

   Replay agrees with eager to 1.15e-06 eV/Å, in line with this tracker's own production
   tolerance for the captured path (1.6e-6 eV/Å).

Both reproduce on the reverted tree. The likely cause is an environment move (the suite now
warns on `torch.jit.load`, `torch.jit.script_method` and NumPy 2.5 deprecations, and
openequivariance reports its precompiled extension missing and falls back to JIT), but **that is
a hypothesis, not a measurement** -- what is measured is only that the fault predates this
session. Registered as a live regression against the recorded 39.7 ms, to be bisected against the
torch/openequivariance versions that measurement was taken on.

**It says nothing about the block-0-reading route.** The fault is in the decoupled head's own
captured assembly, which is the path that has always been captured. The design work in 1 and 2
above stands on its own gates, which run on the in-process bases and pass.

### Gates added (`test_capture.py`, now 23)

`test_an_o5_head_now_captures_through_the_vjp` (both forms), `test_a_block_zero_reading_head_captures`,
`test_a_moved_block_zero_frame_replays`, `test_the_pair_dedupe_switch_changes_no_value`,
`test_a_genuinely_COUPLED_head_is_still_refused`, `test_the_two_refusals_are_separate_and_each_is_honest`.
The composition gate is restated: capture's list is shorter than the closed form's, and what must
hold is that no head is accepted while `analytic_refusal` is `None` and the kernel nonetheless
raises — the exact gap W6 fell into.


### W6 captured: measured, and it is not a gain (2026-09-15)

Alternated in one loop (the card drifts ~15 % between separate runs): **O3 40.3 ms captured
against 76.2 eager, 1.89x; W6 97.0 ms captured against 92.9 eager, 0.96x.** Capture is now
AVAILABLE to a block-0-reading head and is not worth using on one. The captured step assembles
TWICE -- the closed form works on the already-captured `H`, whereas the VJP rebuilds features and
assembly under grad (~22 ms) -- and capture also forces the per-edge hop MLP, O1's dedup being
4.95 of the assembly's 7.7 ms, now paid twice. **W6's production speed is unchanged at ~93 ms
eager.** The avoidable half is the second assembly: capture the assembly WITH grad and capture
the backward as the tail, rather than rebuilding the forward inside the tail. Not built, saving
not measured, not estimated.


# LES head line (plan of 2026-09-15) — L0 registration, written before any run is opened

## L0 — `base_v4_lesmh`: base v2 recipe + LES readout

**Recipe (one addition to base v2, nothing else moves).** `~/runs/w1_run.sh` verbatim (MH-1
multi-head fine-tune, `omat_pbe` foundation head, the sel10k fps/combinations replay, energy 10 /
forces 10, lr 1e-4 forced, EMA 0.9999, clip 1.0, wd 0, 30 epochs, batch 8, float32, cueq, E0s as
IsolatedAtom frames at config_weight 1000 in `train_iso.xyz`) plus `--model MACELES
--les_readout_scale 0.01` — the scale LES-FT2 ran at (its debug log: `les_readout_scale=0.01`).
The code resolves this to `FoundationMACELES` with heads `['pt_head', 'Default']`; each head gets
its own latent-charge readout (`les_readouts`, per-head channels exactly as the energy readouts).
Runs: `base_v4_lesmh_prod` (`dataset_e0/train_iso.xyz` / `valid.xyz`) and `base_v4_lesmh_f{0..3}`
(`dataset_cf/fold{k}/train_iso.xyz` / `valid.xyz`), seed 1 — the same five files base v2 used.
Script `defect-perovskite/w1les_run.sh`; queue `w1les_queue.sh` on b3 GPUs **5 and 7 only** (GPU 4
has now joined 6 off the bus; 0–3 are never used), one run per card. Base v2 took 1 h 52 for 30
epochs (`base_v2_prod.log`, epoch 0 at 20:49 → epoch 29 at 22:41), so ≈ 6 h for the five.
**Nothing is tuned for the LES term** — not the scale, the lr, or the replay — because L1's
cross-base reference to W6-on-v2 needs the recipe identical apart from the readout.

**Known, registered, not gated.** The `pt_head` also acquires a latent-charge readout, trained
against MH-1 pseudolabels that contain no long-range term; it is discarded at conversion.

**LES conventions as run (library defaults, read off LES-FT2's `les` module and `les/module/ewald.py`).**
σ = 1.0 Å, dl = 2.0 (k-cutoff |k| ≤ 2π/dl), `remove_self_interaction = True`, `remove_mean = True`
(the BEC only), `epsilon_factor = 1`, reciprocal sum only with **k = 0 excluded** (`mask = k_sq > 0`),
so the net latent charge is unconstrained and a net charge carries the neutralising-background
Madelung energy. Measured (`scratchpad/les_convention2.py`, the library's own `Les`): the pair
kernel is `C erf(r / (2 r_g)) / r` with **r_g = σ/√2 = 0.7071 Å** — cross(∞) −9.8305 / −6.8686 /
−4.7836 / −2.8704 eV·e⁻² at r = 1 / 2 / 3 / 5 Å against −9.8305 / −6.8722 / −4.7869 / −2.8799
for r_g = σ/√2 and −7.4950 / −6.0673 at r = 1 / 2 for r_g = σ. A single unit charge in a 20 Å
cube: −1.0256 eV (background Madelung −αC/2L = −1.0213). **Registered for L2/L3:** `K_LR` is
built at `r_g = σ_LES/√2`, `k = 0` dropped in every kernel that touches `q_LES`; the global sign
of `q_LES` is fixed once per base so that `Z*(Cs) > 0`; the scale is untouched.

**Gates (thresholds fixed here; readers named; nothing opened yet).**
- G-L0.1 *Neutral validation* (the run's own epoch-29 `Default` row, the same 154 frames):
  **≤ 10.23 meV/Å (MACE table, per component) and ≤ 1.16 meV/atom** — base v2 prod epoch 29.
  Fold bases: reported, compared with `base_v2_f{k}` epoch 29 (f0: 3.88 / 0.43).
- G-L0.2 *Replay robustness — charged-frame base-alone shell table*, `base_compare_n.py
  --model` (each model through ITS OWN forward, so the Ewald term is in): the converted
  `base_v4_lesmh_prod_base.pt` on the charged 79-atom set must be **≤ base v2 at every shell**:
  total 33.22; 2–4 Å 74.9; 4–8 30.4; 8–10 15.3; 10–12 8.9; >12 8.0 (per-component meV/Å, the
  §"LES report" table). Neutral 79-atom rows reported against base v2's 10.96 / 20.0 / 10.0 /
  8.8 / 7.1 / 6.5 (in-sample; a floor, not a gate).
- G-L0.3 *Born check* (`les_gates.py`: `compute_bec=True` through the model's own forward,
  isotropic part, 8 neutral 80-atom frames, the LES-report protocol): after the sign fix
  (Z*(Cs) > 0) **Cl < 0, Cs > 0, Pb > 0; |Pb/Cs| ∈ [2, 3.5]; |Σ Z*| ≤ 0.3 e per cell.**
  LES-FT2 reference: −0.466 / +0.382 / +1.028, 2.69, +0.18.
- G-L0.4 *Liveness*: `les_live_check.py` — final-layer ‖W‖ > 0 on **both** heads' latent-charge
  readouts before conversion, and on the converted Default head.
- G-L0.5 *Conversion is exact*: `w1les_make_base.py` asserts, on real frames, that the
  single-head pruned model reproduces the multi-head model's Default-head `energy`, `forces`,
  `latent_charges` and `les_energy` to 1e-5 (eV, eV/Å, e) and that `vars(les)` is identical.
- Recorded, not gated: net latent charge per cell and per-species means (neutral 79/80),
  `E_LES` per atom, readout ‖W‖.
- *Fallback (plan):* if multi-head + LES cannot be built in ≈ 1 day, LES-FT2 with its
  charges-off control, recorded as lacking replay robustness; REF never used.

**Decision rule.** All of G-L0.1–G-L0.5 pass → `base_v4_lesmh` is the base of the line and L1
starts. G-L0.2 fails at any shell → the line has no replay-robust LES base; the fallback is
taken and the failure recorded. G-L0.3 fails → the charges are not a usable static pattern for
L2 (E_cross needs a physical q_LES); L1 still runs, L2 is not opened until the reading is
understood.

### L0 smoke and launch (2026-09-15, written before any epoch-29 row is opened)

**Head-side change, gated (`tests/extensions/dscc/test_les_base.py`, 7 gates; suite of
`test_les_base` + `test_capture` + `test_model` = 75 passing).** `MACEDSCC.base_forward` was pinned to
`ScaleShiftMACE.forward`, which silently drops a MACELES base's Ewald term; it now dispatches on the
base's class (`type(self.base).forward`), the neutral null is the LES base bit-identically, charged
forces stay exact against central differences (2e-6) with the Ewald term in the base energy, and the
base's `latent_charges` / `les_energy` are surfaced in the forward dict (`base_latent_charges`,
`base_les_energy`) — `q_LES` differentiable in the positions (autograd d(Σq)/dR against central
differences to 1e-4 relative). `capture_refusal` refuses a MACELES base first (the line runs eager).

**Local smoke (A4000, float32, cueq, one epoch, `train_iso.xyz` + the sel10k replay,
`v4_smoke_lesmh`).** Initial Default head 1.72 meV/atom, 43.88 meV/Å (base v2 smoke: 1.69 / 43.9 —
the 1e-2-scaled Ewald term adds 0.03 meV/atom at start); pt_head initial 0.98 / 1.72 (v2: exactly 0;
the LES term on the pseudolabels). **After one epoch 0.92 meV/atom, 13.90 meV/Å** (v2 smoke: 0.90 /
13.86). 4.5 min per epoch. Conversion (`w1les_make_base.py`) on 6 OOF-neutral + 6 charged frames:
max |multi(Default) − single| = 1.7e-13 eV, 9.4e-15 eV/Å, 3.5e-17 e, 4.2e-17 eV; `les` conventions
identical; latent-charge readouts live on both heads (Default final ‖W‖ 0.231 / 0.032, pt_head
0.227 / 0.029). Readers exercised on the smoke artefact (numbers are a one-epoch smoke, not a
result): `les_gates.py` runs (signs already Cl −/Cs +/Pb + after a −1 sign fix; ratio 4.5, ASR
0.54 — a one-epoch model), `base_compare_n.py --model` runs with the shell rows.

**Launched on b3 (10:04 b3 clock):** `w1les_queue.sh`, queue prod, f0, f1, f2, f3 on GPUs 5 and 7,
one per card, `~/runs/w1les_queue.log`, run logs `~/runs/base_v4_lesmh_*.log`. Expected ≈ 2 h 15
per run → all five by ≈ 16:30–17:00.

### L0 unattended pipeline, L1 registration and the L2 pre-test registration (2026-09-15, before any L0 row is opened)

**Pipeline (`l0_watch.sh`, running locally).** Waits for each `base_v4_lesmh_*.model` on b3, pulls it,
converts it (`w1les_make_base.py`: G-L0.4 liveness on both heads and G-L0.5 exactness asserted). For
the production base it then runs, in order, `les_live_check.py`, `les_gates.py` (G-L0.3),
`base_compare_n.py --model v2 --model v4` on the full 2877 frames (G-L0.2), `l2_pretest.py` (the L2
pre-test, sign from the Born fix) and `l0_gates.py`, which applies the registered thresholds and writes
`~/runs/base_v4_lesmh_prod/L0_GATES.json`. **On `L0: PASS` it launches L1 without further approval**
(the user is not monitoring); on FAIL it stops. The fold bases are converted afterwards for the OOF floor.

**L1 — plain W6 on `base_v4_lesmh_prod` (registered command).** Exactly `launch_w6_s0.sh`'s flags
with the new base: `dscc_train.py --name dscc_l1_s{s} --seed s --fold f --epochs 60 --directional 1
--regime B --base ~/runs/base_v4_lesmh_prod/base_v4_lesmh_prod_base.pt --coupling 0 --scf_free 1
--route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4
--base_cache_batch_size 4`. Six seeds {0,1,2,3,4,6} → folds {0,1,2,3,0,2}. Seeds 0 1 2 sequential on the
local A4000 (`l1_local_queue.sh`; W6 took 2 h 11 per seed there), seeds 3 4 6 on b3 GPUs 5/7 through
`w3_queue.sh` (`W3_GPUS="5 7"`, one per card, 13 GB headroom — they start as the fold bases free the
cards). E_host with the head's own `q0` pattern, as shipped; the base's Ewald term is inside `E_base`
through the dispatched `base_forward` (gated above); `E_base`/`F_base` cached per frame as before.
No capture (refused on a MACELES base); the step time is reported in absolute ms.
- *Gates, all from the shipped readers:* neutral null and FD forces (`test_les_base.py`, passing on
  a toy MACELES base; on the production base the trainer's own warm check); C5 localisation
  (`dscc_arm1_gates.py --no_ladder --winners <six runs>`: shell mass ≥ 0.70, Cs mass < 0.05,
  N_eff ≤ 8, the arm_gates thresholds); G2a energy RMSE at 79 after one `C_Q` and G2b
  |between-size| ≤ 10 meV (`held_final.json`, cohort median as for W6); by-shell force table
  (`held_final["shell_rmse"]`, per component) against the out-of-fold neutral floor of the fold
  bases; saturation (`held_final["saturation"]`: `hop` and `on_site` saturated fractions,
  `on_site_shift_eV_p95`); hop-modulation knee fraction = `saturation["hop"]` (the fraction of
  `|pre| > 2.0`, W6-on-v2: 1.4e-5).
- *Cross-base reference, reported, not a criterion:* paired TOST (τ = 1.7, six seeds by fold) of L1's
  epoch-averaged held force RMSE against W6-on-v2 (11.58 meV/Å cohort mean; G2a 0.515 meV/atom;
  G2b +5.0 meV). L1 is the baseline of the line; L2 is judged against L1, not against W6-on-v2.

**L2 pre-test (`l2_pretest.py`, kernel level, registered readings and thresholds).** Frozen-core
ladder (R_core 8 Å, the relaxed 2,2,2 core of `relax159`, tilings 2,2,2 / 2,2,3 / 3,3,3 / 3,3,4 =
639 / 959 / 2159 / 2879 atoms) with `q_LES` from the converted base on the actual cells (float32 base,
energies only) and a fixed localised unit carrier (Gaussian weights σ = 2 Å on the atoms within 4 Å of
the vacancy, identical in every cell because the core is rigid). `K_LR = gamma_lr(r_g = 1.0,
r_split = 2.5, ε∞ = 4)`, reciprocal, k = 0 dropped — the L2 kernel exactly.
- *Monopole exact (must hold):* `E_self − ½ ξ(cell)/ε∞` fits `a + c/V` with max residual ≤ 2 % of the
  monopole span over the ladder (the fitted-slope reading is NOT used: with four cells in x ∈
  [0.06, 0.095] the x and x³ columns are collinear and the linear slope under-reads the monopole by
  5–7 % on the smoke base while the subtraction test shows 0.9 %).
- *Local neutrality (recorded; gates the cross-size energy claim for `E_cross`, forces unaffected):*
  `Q_delta` = Σ(q_LES − q_pristine) over the cell (the base's response to the defect), its partial sums
  within 4 / 6 / 8 / 12 Å, the removed site's pristine charge `q_site`, and `Q_defect = Q_delta −
  q_site`. The 1/L slope of `E_cross` must agree with `−Q_defect C/ε∞` (consistency of the reading;
  on the smoke base 0.0779 fitted against 0.0793 expected). **The residual at 79 atoms,
  `b · α/L`, is the between-size energy term `C_Q` cannot absorb: ≤ 10 meV (the G2b tolerance) for
  the cross-size energy claim to stand at L2; above it, `E_cross` is registered as forces-only.**
  On the one-epoch smoke base: Q_delta −0.039 e, q_site −0.016 e, residual 7.4 meV — a scaffold
  reading, not a result.
- The pristine pattern's net charge per cell does not enter: a pattern periodic with the unit cell
  has an L-independent potential under the k = 0-dropped kernel; only the defect's net charge
  (`Q_defect`) carries 1/L. Verified on the smoke base (`E_host` slope −0.049 fitted, −0.059 from
  `q_site`).

### L2 built and gated (2026-09-15; NOT trained — L2 opens only after L1's six seeds are read)

**Implementation (`MACEDSCC(les_cross=True, les_sign=±1)`, `dscc_train.py --les_cross 1 --les_sign ±1
--les_cache FILE`).** In `scf_free_terms` the host term becomes `E_cross = s_c · dqᵀ K q_LES`:
`q_LES` is the frozen MACELES base's latent charge on the actual frame (sign-fixed by `les_sign`),
`K = gamma_lr(r_g = 1.0, r_split = 2.5)` **bare** (`eps_inf = 1`: the latent charges are the
base's effective response already), `s_c = s_max tanh(·)` in [−2, 2] initialised at
**+1/√ε∞ = 0.5**. No reference fill enters the energy (`q0` is not formed); `occ_R` is still
computed because `dq = occ_R − occ_S` needs it. `E_M(Q; h)` is unchanged. Serialised in the extra
state. **Interpretation registered:** the plan's `K_LR(r_g, σ_LES)` with "registered r_split" is
read as `gamma_lr(r_g, r_split = 2.5)`; the reading where σ_LES replaces r_split is not
constructible (`gamma_lr` requires r_split > r_g, and σ_LES/√2 = 0.71 Å < 1.0 Å) — the LES
density's own width is narrower than r_split and does not enter the long-range kernel.

**Forces.** `(e_cross, 1)` cotangent (dq's two Fréchet contractions, `s_c`); `K`'s geometry from
its pair derivatives under the pair route (autograd under the stress route), as E_host's;
`q_LES(R)`'s geometry **from a per-frame Jacobian cache** `J = ∂q_LES/∂R [N, N, 3]` (N first-order
backward passes through the float32 base, 1.7 s per 79-atom frame on the A4000, ~150 KB per
frame; `l2_les_cache.py`, keyed by geometry key, built once per base and loaded by the trainer):
`∂E/∂R_k += s_c Σ_j (K dq)_j J_jk`, J a constant of the geometry contracted with attached `s_c`
and `dq`. This is the plan's "base backward with cotangent `s_c K_LR dq` on `q_LES`" evaluated once
per frame rather than once per step — the training step runs no base (as W6 with the cached
base), and the force loss's parameter gradient needs no `create_graph` through the base. At
inference (ladders, new geometries) the base runs in the forward and the same cotangent covers
`q_LES` through the attached graph; the two routes are gated equal.

**Gap regulariser:** unchanged. In W6 it already acts on `H0` alone (the host term is an energy,
never a potential — `model.py` "NOT in W6"), so removing `q0` changes nothing there; `K q_LES` does
not enter `H0` at L2 (that is L3's physics).

**Gates (`test_les_base.py`, now 17; full dscc suite re-run below).** Construction guards (needs
`scf_free`, a MACELES base, sign ±1); `s_c` = 0.5 at init and bounded at ±s_max; state round trip
keeps flag and sign; neutral null is the LES base bit-identically; `E_cross` equals
`s_c dqᵀ K_bare q_LES` recomputed by hand and no `q0_sum` diagnostic exists; the sign fix flips
`E_cross` exactly; forces against central differences on the attached route (2e-6); **the cached
Jacobian route equals the attached route** (energy 1e-10, forces 1e-8); the cached route in
training mode against central differences (2e-6); dropping J moves a force by > 1e-5 eV/Å (the
term is not negligible). Trainer smoke on the one-epoch smoke base (8 frames, 1 epoch,
`--les_cross 1 --les_sign -1`): cache built, step runs, `held_final.json` written.

**Named superiority metric for L2 adoption (fixed now, before L1 or L2 is read):** the
**far-field shell force RMSE, 8–12 Å (per component, `held_final["shell_rmse"]` at 8–10 and 10–12
combined by count), epoch-averaged, paired by seed against L1**: L2 is superior if its paired mean
difference's one-sided 90 % CI upper bound is below 0 and it is lower on ≥ 5 of 6 seeds.
Not-inferior on forces = paired TOST at τ = 1.7 against L1. The 159-atom energy is reported, not
the criterion. The Born-fixed sign for the production base comes from `L0_GATES.json` (`sign_fix`).

### L3 built and gated (2026-09-15; NOT trained — L3 opens only if L2 is adopted, and starts from L2's model)

**Implementation (`MACEDSCC(les_well=True)`, requires `les_cross`; `dscc_train.py --les_well 1`).**
`W = s_W (K_bare q_LES − mean_i)` per frame, geometry-only, and the head fills `H0' = H0 −
diag(W ⊗ 1₄)` — both fillings (state and reference), so the level and the carrier shape see the
host's long-range well; `E_cross` is computed from the resulting `dq` with the same kernel
(computed once per frame, shared). **Centring registered:** under the k = 0-dropped kernel the
potential of a net-charged pattern carries the background's arbitrary constant, and a uniform
diagonal shift moves `dF_band` by `c·Q` for a charged state, so `W` is centred per cell. `s_W =
s_max sigmoid(·)` in [0, 2]; **initialised at 1e-3, not 0**: a smooth bounded map has zero gradient
at its boundary (the LES zero-pinning lesson), and Adam's normalised step moves the raw logit at
`lr` per step regardless of the gradient's size. Gated: at `s_W → 0` L3 equals L2 to 1e-9 eV /
1e-8 eV/Å. **The gap regulariser acts on `H0'`** (W6's rule: on the Hamiltonian actually filled);
the static cell's `q_LES` is attached once by the trainer (`dscc_base_q` on the static batch).

**Forces.** `(H', dP)` and `(E_cross, 1)` cotangents carry `s_W`, `q_LES` (attached route) and
`H0`; what they cannot carry — `K`'s geometry under the pair route (K detached) and `q_LES(R)`
on the cached route — uses the **total** `dE/dW` from one backward through the fill graph
(band energy through `dP` PLUS `E_cross` through the fill's response `d dq/dW`, which `−n_site`
alone misses: the first build used `−n_site` and failed FD by 8e-6 and the pair/autograd
agreement by 1.9e-4), contracted with `K`'s pair derivative and with the cached Jacobian.

**Gates (`test_les_base.py`, now 27 = 7 head-side + 10 L2 + 10 L3).** L3: needs `les_cross`; starts
at L2 within the registered init and equals L2 exactly at `s_W → 0`; bounded at `s_max`; state round
trip; neutral null is the LES base bit-identically; the well moves the charged energy (> 1e-5 eV at
`s_W = 0.5`); `pristine_gap` equals the by-hand gap of `H0 − diag(s_W (K q_LES − mean))` to 1e-9 and
differs from `H0`'s; forces against central differences on the attached route and on the cached
route in training mode (2e-6); cached Jacobian route equals the attached route (1e-10 / 1e-8);
the stress (autograd-kernel) route matches the pair route (1e-8). Trainer smoke (`--les_well 1`,
8 frames, 1 epoch): static-cell charges attached, step runs, `s_well` logged.

### Unattended chain L1 → L2 → L3 (registered 2026-09-15, before L0 is read)

`lx_watch.sh STAGE` (armed by `l1_arm.sh` once `l0_watch.sh` has launched L1): waits for the
stage's six seeds (0 1 2 local, 3 4 6 on b3, pulled when complete), reads the stage with
`lx_read.py` (the registered W6 reader generalised — self-test: reproduces the W6-vs-B′ record
exactly, G1 +0.14 [−0.21, +0.50], G2a −0.0688, G2b +5.0; **the epoch-averaged `held_final_avg`
is the line's primary estimator, the W6-era `held_final` is reported alongside**), runs C5
(`dscc_arm1_gates.py --no_ladder`, the W6 protocol: separation p50, N_eff p50, pass fraction at
the 95 %-of-frames rule), and writes `~/runs/dscc/{STAGE}_GATES.json` (`lx_verdict.py`).
- **L1 → L2:** L1 is the baseline and is judged against nothing; its C5, G2a/G2b, shells and
  saturation are recorded as its description and the cross-base TOST against W6-on-v2 is
  reported. L2 launches once L1's six seeds are read (`--les_cross 1 --les_sign <L0 sign fix>
  --les_cache ~/runs/base_v4_lesmh_prod/les_cache.pt`, the same six seeds/folds, the same
  local/b3 split).
- **L2 → L3:** L2 is ADOPTED iff (a) paired TOST on the epoch-averaged held force RMSE against
  L1 is not inferior (lower bound of L1 − L2 above −1.7) and (b) the named metric — far-field
  8–12 Å shell force RMSE — is superior (lower bound of L1 − L2 above 0 and L2 lower on ≥ 5 of
  6 seeds). Adopted → L3 launches from L2's same-seed models (`--init_state`, parameters copied,
  `s_W` at its init) with `--les_well 1`; not adopted → the chain stops, L1 stands, L4 runs on L1.
- **L3:** adopted by the same two-part rule against L2 (the plan's "named superiority" for L3 is
  read with the same far-field metric; C5 separation and N_eff and the energies are reported);
  the selected model goes to L4, which is not automated.
- No stage is opened before the previous stage's reading is written; L2/L3 results are never
  read against W6-on-v2 (cross-base), only against their own previous stage.

### L4 registration: the tabulated ε₀ (2026-09-15, before any ladder of this line is fitted)

The plan's L4 compares `ε₀^model` (from the relaxation-ladder fit and, independently, from base
phonons with `Z*_eff = s√ε∞ · q_LES`) with a tabulated ε₀ of the same status as the registered
`ε∞ = 4.0`. Registered now: **ε₀ = 7.2 ± 0.6 (cubic CsPbCl₃, DFPT Γ-point IR phonons through
Lyddane–Sachs–Teller; RSC Advances 2026, DOI 10.1039/D5RA07356A, the pristine reference of a
doping study), whose own ε∞ is 2.40 ± 0.06** — so the tabulated ratio is ε₀/ε∞ ≈ 3.0. Because the
registered `ε∞ = 4.0` is from a different source, the comparison is made on BOTH the absolute
ε₀ and the ratio ε₀/ε∞ (the model's ε∞ is the registered 4.0 by construction, so the model's
ratio is `ε₀^model / 4`). An earlier report line ("order 20 against 4 in this material") was an
order-of-magnitude remark without a source and is superseded by this registration; no
experimental cubic-phase capacitance value with a citable number was found in this search
(Raman/IR work on the phase transitions exists — J. Raman Spectrosc. 2003, DOI 10.1002/jrs.1005 —
without a static constant), so none is registered. The phonon route (`l4_eps0_phonon.py`):
finite-difference Γ-point force constants of the 5-atom pristine cell with the base alone,
acoustic modes removed, `ε₀ − ε∞ = (4π C / 3Ω) Σ_m Σ_α |Σ_κ Z*_κ e^α_{κm} / √M_κ|² / ω_m²`
with `Z*_κ = s√ε∞ · q_LES,κ` (rigid latent charges as Born charges, the plan's definition) and
`s` the trained L2 scale (`1/√ε∞` for a base-only pre-reading).
*Phonon-route note (exercised on the smoke base, not a result):* the 80-atom static cell folds the
primitive cell's R- and M-point octahedral-tilt modes to Γ, and they come out imaginary for the
base (20 of 237 optical modes on the smoke base); they are non-polar and are excluded from the
dipole sum, which is recorded as an approximation of the route. The smoke base's rigid `Z*_eff`
(0.003–0.03 e at `s = 1/√ε∞`) gives ε₀ − ε∞ = 0.004 — a one-epoch base, which is why the
reading waits for the trained base and the trained `s`.

### L0 option registered by the user (2026-09-15, 11:45): distil MH-1, train MACELES on the distilled labels, fine-tune on the defect set
Registered as an option, not taken: distil MACE-MH-1 (the neutral distribution) into a MACELES
trained from the distilled energies/forces, then fine-tune on the defect data set — the
distillation + fine-tuning route reported successful at arXiv:2606.15001. The user's caveat: the
point of fine-tuning from MH-1 is to capture the base error on the *charged* defect structures,
and MD-generated distillation data would relax back to the neutral distribution, so the
distillation geometries would need a different generator than MD. Sits alongside the three
options listed in the same turn (continue L0 for more epochs; a per-group lr for `les_readouts`;
the plan's FT2 fallback), to be chosen only after G-L0.3 is read at epoch 29.

# Distilled LES base — plan of 2026-09-15 (12:20): registration record

**Plan received (user, 12:20):** MH-1 → lightweight LES student trained on MH-1's predictions
(Wang & Cheng, arXiv:2606.15001) → DFT fine-tune with the distillation set as replay → head line
(D0 distillation set, D1 student with the pre-registered three-arm architecture control, D2
fine-tune, D3 hand-off). Design principle: the student must be cutoff- and capacity-limited
relative to the teacher so the Coulomb term is *needed* to reproduce the teacher's 12 Å behaviour.

**Record, as instructed by the plan.**
- *Retractions:* "MH-1 labels carry no long-range content" (coadvisor) and "no label in this
  lineage carries long-range content for the replay" (my L0 report and the pt_head answer of
  ~12:00) — both wrong per arXiv:2606.15001: latent electrostatics are extractable from local
  foundation models by distillation into a cutoff-limited student.
- *Mechanism of L0's shrinkage:* distillation into a student that *is* the teacher (the multi-head
  run: same architecture, MH-1's weights, MH-1's replay labels) gives the charges nothing to
  explain; identification requires a cutoff-limited student.
- LES-FT2 and LES-REF: references for the Born check only; not bases of this line.
- **Base v4 (multi-head + LES): closed at epoch 29 with its gate file; not continued.** Epoch 29
  Default row: 1.23 meV/atom (the force column is read from the gate file). The L0→L1
  automation (`l1_arm.sh`, the L1 launch inside `l0_watch.sh`) was stopped at 12:41 before the
  gate file was written; fold bases f1–f3 removed from the b3 queue (f0 completes and is
  converted, nothing else runs on it). The gate reading itself completes and is appended below
  when written — a record, not a decision input.
- The head line's L1–L4 registrations of this morning carry over to D3 on the student base;
  L2/L3 code and gates are unchanged (they take any MACELES base).

## Distilled LES base — the plan as pasted (2026-09-15 12:20), D0–D3

> **D0 — distillation set** (no MD; MH-1 `omat_pbe` head as teacher, PBE-consistent). Structures
> (target ~30–50k, all labelled by MH-1 with energy, forces, stress): the 1047 charged-frame
> geometries labelled as neutral systems; the 1770 neutral frames (train + val), teacher-labelled;
> 3–5 perturbed copies of every frame above (Gaussian σ ∈ {0.05, 0.1, 0.2} Å); strains of pristine
> and vacancy cells (isotropic ±2 %, ±4 %; shear ±3 %; ~2k); vacancy tilings at 159, 319, 639 atoms
> with thermal displacements (~300); polar sublattice displacements of the pristine 80-atom cell
> (Cs, Pb, Cl sublattices rigidly displaced along x, y, z by 0.02–0.10 Å, ~200) — identify Born
> charges; optional ~2k MP halide perovskites (not required). Register teacher hash and head, E0s,
> 90/10 split by structure family, the seed of every perturbation.
> **D1 — student:** lightweight MACELES from scratch on D0 (`les_fit` recipe): r_max 6.0,
> num_interactions 1, correlation 3, max_L 1, hidden 96x0e+96x1o, num_radial_basis 8, LES readout
> at the recipe's init scale, ε_factor 1, σ as measured (r_g = σ/√2); loss energy + forces (+ stress
> on the strained subset, small weight), scheduler patience 15, Stage Two at 50 %, lr 1e-2.
> Gates: G-D1.1 student vs teacher on the held 10 % ≤ 1 meV/atom, ≤ 5 meV/Å per family; G-D1.2 Born
> (8 neutral 80-atom frames, EMA weights, one global sign fix: Cl −, Cs +, Pb +, |Pb/Cs| ∈ [2, 3.5],
> |ΣZ*| ≤ 0.3); G-D1.3 charge necessity: the LES ablation raises the student-vs-teacher force error
> on the long-range families. Fail on G-D1.2 → one change (r_max 5.5 or a larger polar family), rerun.
> **D1 amendment — architecture control (pre-registered):** D1-light (primary; prediction: charges
> live, G-D1.2 passes, ablation raises long-range-family error); D1-full-scratch (MH-1 shape, scratch,
> les_fit; prediction: fits the teacher at least as well, charges smaller, ablation effect small);
> D1-full-init (MH-1 shape, MH-1 weights, lr 1e-4, EMA 0.9999, 30 epochs; prediction: E_LES → 0,
> |q| falls from init — reproduces L0's shrinkage). Reading rule: light passes and the full arms do
> not → identification confirmed, light is the base; full-scratch also passes → choose on G-D1.1 and
> robustness, speed deciding ties; light fails and full-scratch passes → one change to light, rerun
> light only; full-init not collapsing → L0 mis-diagnosed, revisit before D2.
> **D2 — DFT fine-tune with distillation replay:** init from D1; multi-head `dft` (1616 DFT frames)
> + `teacher` (D0), shared backbone, shared LES readout, per-head energy readouts, replay 1:1 by
> batch, lr 1e-4, EMA 0.9999, 30–60 epochs, float32; four cross-fit bases. Gates: G-D2.1 neutral
> validation ≤ 1.2 × base v2 (12.3 / 1.4); G-D2.2 charged shell table ≤ base v2 at every shell
> (decisive); G-D2.3 Born repeated, drift reported, must pass G-D1.2; G-D2.4 cross-fit agreement
> (0.01 meV/Å); recorded: isolated-atom offsets.
> **D3 — hand-off to the head line** on this base: pre-test, L1 (readouts re-dimensioned to the
> student's single block; W3-type retrain, six seeds), L2, L3; M1 the registered fallback,
> initialised from q_LES; ε₀^model from student phonons; absolute ms; no capture.

### D0 enumeration (registered before the set is built)

| family (`config_type` = `d0_family`) | source | count | perturbation |
|---|---|---:|---|
| `charged_relabelled` | `dataset_pbe/{train,valid}.xyz`, `cell_charge ≠ 0` | 1047 | none; written with `cell_charge 0`, `carrier_counts 0`, provenance in `d0_source_charge` |
| `neutral` | `dataset_e0/{train,valid}.xyz` | 1770 | none |
| `perturbed` | every parent of the two rows above | 25 353 | 9 copies: 3 each at σ = 0.05, 0.10, 0.20 Å Gaussian (seed = 1000·parent + copy) |
| `strained` | 330 parents (110 pristine-80, 110 neutral-vacancy, 110 charged-relabelled, every k-th by index) | 1 980 | isotropic ±2 %, ±4 %; xy shear ±3 % (6 modes); `config_stress_weight = 1` here, 0 elsewhere |
| `tilings` | static pristine cell tiled 1,1,2 / 2,2,1 / 2,2,2 minus one Cl (159 / 319 / 639 atoms) | 300 | 100 each, thermal Gaussian σ = 0.10 Å (seed 5000 + i) |
| `polar` | static 80-atom cell | 180 | one sublattice (Cs, Pb, Cl) rigidly displaced along x/y/z by ±{0.02, 0.04, 0.06, 0.08, 0.10} Å (90); Pb-vs-Cl and Cs-vs-Cl opposite (60); Cs+Pb vs Cl (30) |
| isolated atoms | 20 Å box, one Cl / Cs / Pb | 3 | teacher-labelled, `config_type IsolatedAtom`, `config_weight 1000`, train only |

Total 30 630 (+3). Teacher: `~/.cache/mace/macemh1model` (sha256 `a522eb7f59c78799…`), head
`omat_pbe`, float64 forward with stress, batched by size. Labels `REF_energy / REF_forces /
REF_stress`. **E0s:** the teacher's *predicted* isolated-atom energies (its stored `omat_pbe`
row −0.2574 / −0.1297 / −0.7736 eV is recorded alongside; a ScaleShiftMACE adds its shift and
readout on top of the row). **Split 90/10 by parent geometry, stratified by family, seed 0** —
perturbed and strained copies of a held-out parent are held out with it. Manifest
`dataset_d0/manifest.json` (counts per family and split, seeds, E0s, teacher hash). The optional
MP pool is not built.

### D1 flags and budgets (registered)

Common (`les_fit`): `--model MACELES --loss stress --energy_weight 10 --forces_weight 1000
--stress_weight 10` (stress acts only on the strained family through `config_stress_weight`),
`--lr 0.01 --amsgrad --scheduler ReduceLROnPlateau --scheduler_patience 15 --patience 200 --ema
--ema_decay 0.99 --clip_grad 10 --stage_two --start_stage_two <50 %>`, batch 8, float32, cueq,
`--keep_isolated_atoms True` (E0s from the isolated frames), `--les_readout_scale 1.0`,
`--error_table PerAtomRMSEstressvirials` (per-`config_type` = per-family rows).
- **D1-light:** `--r_max 6.0 --num_interactions 1 --hidden_irreps 96x0e+96x1o --max_ell 1
  --correlation 3 --num_radial_basis 8 --num_cutoff_basis 5 --MLP_irreps 16x0e --radial_MLP
  [64,64,64] --pair_repulsion --distance_transform Agnesi`, interaction blocks as the reference
  recipe. **80 epochs, Stage Two at 40.** Local A4000.
- **D1-full-scratch:** the reference recipe's MH-1 shape (`les_ref_run.sh`), **40 epochs, Stage
  Two at 20** (≈ 15 min/epoch on 27.6k training frames → ≈ 10 h). b3 GPU 5.
- **D1-full-init:** `ft_les_run.sh`'s path (MH-1 weights, `--multiheads_finetuning False`),
  lr 1e-4, EMA 0.9999, energy 10 / forces 10 constant, clip 1, 30 epochs (≈ 7.5 h). b3 GPU 7.
- Readers, ready before any result: per-family teacher-fit table (G-D1.1) from the run's own
  error table plus `d1_read.py` on the held split; `les_gates.py` on the saved EMA model
  (G-D1.2); `d1_ablate.py` — `les_readouts` zeroed, teacher-fit force error on `tilings` and
  `polar` (G-D1.3); `d1_traj.py` — |q|, E_LES, readout norms per checkpoint; inference ms per
  80-atom frame. Wall-clock: light ≈ 3 min/epoch (estimate; measured at the smoke).

*D0 built (12:52):* 30 618 structures — charged_relabelled 1047, neutral 1770, perturbed 25 353,
strained 1980, tilings 300, **polar 168** (the Cs+Pb-vs-Cl pattern has 18 members at d ∈ {0.04,
0.08, 0.10}, not 30 as enumerated; recorded, not changed). Label-stage smoke (18 structures, CPU):
labels, per-config stress weight and the isolated atoms written; **teacher-predicted E0s Cl −2.252 /
Cs −0.970 / Pb −1.037 eV against the stored `omat_pbe` row −0.257 / −0.130 / −0.774** — the
predicted values are the D1 E0s. Labelling of the full set queued behind the L0 gate readers on
the local card (`d0_label_run.sh`). b3: the fold bases f1 and f2, dispatched by the w1les runner
before the queue was emptied, killed at 12:58 — cards 5 and 7 free for the D1 full-shape arms.

### L0 gate file (record only; the v4 run is closed) — `~/runs/base_v4_lesmh_prod/L0_GATES.json`, 13:1x

- **G-L0.1** epoch 29 Default: **1.23 meV/atom, 9.81 meV/Å** against ≤ 1.16 / ≤ 10.23 → FAIL (energy).
- **G-L0.2** charged 79-atom base-alone shells (total / 2–4 / 4–8 / 8–10 / 10–12 / >12, per component):
  v4 **32.0 / 68.5 / 30.2 / 16.7 / 9.0 / 7.8** against base v2 (same run) 33.2 / 74.9 / 30.4 / 15.3 / 8.9 / 8.0 → fails at 8-10, 10-12 Å → **FAIL**.
  Neutral 79 (in-sample): v4 10.3 / 20.1 / 9.3 / 7.4 / 6.1 / 5.2 against v2's floor 10.96 / 20.0 / 10.0 / 8.8 / 7.1 / 6.5.
- **G-L0.3** Born (EMA model, sign fix -1): Cl -0.360, Cs +0.257, Pb +0.680; |Pb/Cs| 2.65 (in [2, 3.5]); **ΣZ* -2.27 e per cell** against ≤ 0.3 → **FAIL on the sum rule**; signs and ratio pass.
- Recorded: net latent charge -0.817 e/cell (raw), species means Cl -0.0117 / Cs -0.0038 / Pb -0.0121, |q| mean 0.0109 e, E_LES -0.45 meV/atom.
- Verdict `FAIL`. Consistent with the trajectory read in flight (charges ~40 % of FT2's, ΣZ* far from zero); the shell-table failure is at the far field (8–12 Å), where the base's Ewald term acts.

*D0 labelling precision (13:2x, user: "Ensure you're training the base models using f32 and with
cueq or oeq enabled"):* every D1 arm trains in float32 with cuEquivariance (`d1_run.sh`:
`--default_dtype float32 --enable_cueq True`), as the v4 runs did. The teacher labelling was
started in float64 (registered above) and ran at 99 % utilisation for minutes without completing a
batch — FP64 on the A4000 is ~30× slower than FP32; restarted at **float32 + cueq** (the teacher's
own training precision; force/stress values stored as float64), the manifest records
`precision` / `cueq`. Registration amended accordingly.

*D2 prerequisite built (13:35, not trained):* `MACELES(les_shared_readout=True)` / `--les_shared_readout`
— ONE latent-charge readout (one output channel, applied to every node regardless of head) beside
the per-head energy readouts; `remove_pt_head` carries it through unsliced. Gates
(`tests/extensions/les/test_shared_readout.py`, 2): charges identical under either head and
non-zero while the energies differ per head; the single-head model after head removal reproduces
them to 1e-10; the unshared default stays head-dependent.

### D0 amendment — cell sizes (user, 2026-09-15 13:40; applied before any D1 epoch completed)

- Removed: "vacancy tilings at 159, 319, 639 atoms with thermal displacements" — the 300
  `tilings` structures (270 train / 30 held) taken out of `dataset_d0/{train,valid}.xyz`
  (kept in `tilings_removed.xyz`, unused); D0 is now **30 318 structures** (27 327 train + 3
  isolated atoms / 2 994 held).
- Rule: every distillation structure has all cell vectors within the teacher's reach in the
  sense already satisfied by the 80- and 160-atom cells; **no structure larger than 160 atoms
  enters D0 or the D2 replay set.**
- Long-range identification comes from the 6 → 12 Å range gap on these cells plus the
  polar-displacement family (80-atom cell); the student's behaviour beyond 12 Å is evaluated on
  the D3 ladders, not trained. G-D1.3 (charge necessity) is therefore read on the `polar` family
  and on the whole held set, not on tilings.
- Reason recorded: labels from a local teacher on cells larger than its receptive field contain
  the *absence* of long-range physics and would train the LES term to suppress it.
- The three D1 arms had been launched at 13:32 on the unamended set and were still loading data;
  killed and relaunched on the amended set at 13:4x. Measured D1-light epoch time from the smoke:
  22 s per 1 600 frames → ≈ 6.3 min per epoch on 27.3k frames → 80 epochs ≈ 8.5 h (the
  registered 3 min estimate was wrong); full-shape arms ≈ 13 min/epoch → full-scratch ≈ 8.7 h,
  full-init ≈ 6.5 h.
*D1-light batch size (user, 13:55: "That seems slow for a light model. Can you increase batch size?"):*
at batch 8 the light student ran 110 ms per step — launch-bound, the card idle — so D1-light is
relaunched at **batch 32** (`D1_BATCH=32`; lr and every other setting unchanged; 80 epochs are now
≈ 850 steps each instead of 3 400). The full-shape arms stay at batch 8 (compute-bound). Registered
as the light arm's setting before any of its epochs is read.
*b3 arms measured, not assumed (user, 14:0x: "Have you confirmed or asserted that the b3 arms are
compute bound"):* asserted. Measured: GPU 5 (full-scratch) 31–38 %, GPU 7 (full-init) 41–46 %
utilisation at 3.8 GB — launch-bound at batch 8 too. **D1-full-scratch relaunched at batch 32**
(after its epoch 0: 16.51 meV/atom / 40.31 meV/Å; 40 epochs kept), the same optimiser regime as
D1-light. **D1-full-init stays at batch 8**: its settings are base v2's (lr 1e-4, EMA 0.9999,
batch 8) because it tests whether L0's shrinkage reproduces from a converged start, and its epoch
0 is already 1.38 / 6.74 (≈ 17 min/epoch → ≈ 8 h).
*Divergence guard (user, 14:30: "Those spikes look concerning. Was the LR set properly?"):* the
light arm's settings are as registered (`lr=0.01`, patience 15, factor 0.8, clip 10, EMA 0.99,
batch 32, Stage Two at 40). LES-REF on the same recipe showed the same early energy spikes
(epochs 5–8: 5.5 → 26.6 → 57.0 → 8.9 meV/atom; forces 29.7 → 26.3 → 29.5 → 23.4) and trained
320 epochs cleanly; the divergence of `base_v3_les` was patience 50 holding lr at 0.01 while the
forces degraded for ~17 epochs. D1-light epoch 7: 90.7 meV/atom, 71.8 meV/Å (from 18.9 / 61.2).
**Rule registered now:** `d1_guard.sh` (every 5 min, the two from-scratch arms) — force RMSE
NaN, or above 2× its running minimum on 3 consecutive epochs → the arm is killed and relaunched
from scratch at **lr 0.005** (`D1_LR`), once, as `<arm>_lr5e3`; the event is logged
(`~/runs/d1_guard.log`) and the relaunch is the arm of record.
*Caveat on the in-flight checkpoint reads (16:20):* the raw-weight snapshots reported for the v4
run at epochs 17 and 23 (readout norms, |q|, E_LES) were taken by loading a cuEquivariance
checkpoint into an e3nn template with `strict=False`; the cueq product weights
(`symmetric_contractions.weight/projection`) do not exist under those names in e3nn and were
silently left at the template's (epoch-0 EMA) values. The readout-norm numbers are exact (those
keys load); the charge magnitudes are approximate. The epoch-29 gate file used the saved EMA
model and is unaffected. `d1_common.load_checkpoint` now converts the template to cueq first and
refuses any non-trivial missing key; every D1 snapshot uses it.

### D1 amendment 2 — two more architecture classes (user, 2026-09-15 17:30; registered before either runs)

"Run two additional models: the light model with two interaction layers (small width, larger
receptive field) and the full-scratch model with one interaction layer (big width, smaller
receptive field). That way we cover the full set of model classes. The init model does not need to
be retrained as it is a clear dead end." Same D0 set and split, `les_fit` recipe, batch 32, lr 0.01,
same gates and readers; the divergence guard covers them.

| arm | architecture | reach | epochs | prediction (written now) |
|---|---|---|---:|---|
| `d1_light2` | 96x0e+96x1o, max_L 1, corr 3, **2 interactions** | 12 Å | 80 (Stage Two 40) | fits the teacher better than D1-light (its reach now spans the teacher's); charges smaller than D1-light's — with the 12 Å behaviour reachable locally the Coulomb term is needed less; if the identification argument is right its Born check is weaker than D1-light's |
| `d1_full1` | 512x0e+512x1o, edge 128, max_L 3, corr 3, **1 interaction** | 6 Å | 40 (Stage Two 20) | fits the teacher worse than D1-full-scratch but better than D1-light; charges larger than D1-full-scratch's and Born-physical — the cleanest test of the principle at high capacity: capacity alone cannot supply the 6 → 12 Å physics |

The 2 × 2 (width × reach) table then reads: reach is what identifies the charges (both 6 Å arms
Born-physical, both 12 Å arms weaker) → principle confirmed, base chosen among the 6 Å arms on
G-D1.1 and speed; width is what matters instead → the principle is wrong and the base is the
best-fitting arm with live charges. `d1_full_init` runs to completion as the mechanism control
(16/30 at registration: |q| 0.028 e, all species one sign, E_LES −3.8 meV/atom — the predicted
L0-like state) and is not retrained. Scheduling: `d1_light2` on the local card after `d1_light`
(≈ 18:15), `d1_full1` on b3 GPU 5 after `d1_full_scratch` (≈ 19:45); `d1_queue2.sh`.

### D1-light — RESULT (2026-09-15 18:30; final stage-two EMA model `d1_light_stagetwo.model`, 80 epochs, batch 32)

Run's own table: train 1.2 meV/atom / 39.0 meV/Å, valid 1.1 / 38.5 (relative F 4.4 %).

**G-D1.1 teacher fit on the 2 994 held frames, per family (E meV/atom / F meV/Å / S meV/Å³):**
charged_relabelled 0.68 / 25.2 / 0.59; neutral 0.73 / 24.8 / 0.69; perturbed 1.14 / 40.5 / 0.73;
strained 0.82 / 27.1 / 0.77; polar 0.93 / 6.3 / 0.43. Energies pass (≤ 1 meV/atom) on four families
and miss on `perturbed` (1.14); **forces fail every family** against ≤ 5 meV/Å (25–40, polar 6.3).
**G-D1.1 FAIL** — the light student cannot reproduce the teacher's forces on this set.

**G-D1.2 Born (8 neutral 80-atom held frames, sign fix −1):** Cl −0.730, Cs +0.568, Pb +1.372 —
signs right, |Pb/Cs| 2.42 in [2, 3.5]; **ΣZ* −4.00 e/cell against ≤ 0.3 → FAIL on the sum rule.**
Recorded: |q| 0.168 e (raw Cl +0.133, Cs −0.261, Pb −0.183), net −0.72 e/cell, E_LES −114 meV/atom.

**G-D1.3 ablation (LES readouts zeroed):** every family collapses — 125 meV/atom, 229 meV/Å
overall; polar 66 meV/Å (from 6.3). PASS as written, but not discriminating: with E_LES at
−114 meV/atom the Ewald term is a structural part of the light student's energy on every
frame, not a long-range correction, so the ablation measures the model's dependence on it, not
the 6 → 12 Å physics.

Reading: the light student is under-fitted (forces 8× the gate) and its charges, although
ionic-signed with the right Pb/Cs ratio, are 5× LES-FT2's and violate the acoustic sum rule by
4 e/cell. Per the amendment's reading rule the decision waits for D1-full-scratch (≈ 19:45) and the
two new classes (`d1_light2`, `d1_full1`).

### D1-full-scratch — RESULT (2026-09-15 20:45; stage-two EMA model `d1_full_scratch_stagetwo.model`, 40 epochs, batch 32)

Run's own table: train 1.2 / 14.0, valid 0.4 / 14.1 meV/Å (relative F 1.6 %). *Note:* the first
automatic read used `d1_full_scratch.model` (the stage-one best, 10.6 meV/atom / 15.7 meV/Å); the
registered model is the stage-two final and the watcher now prefers it.

**G-D1.1 per family (E / F / S):** charged_relabelled 0.26 / 7.5 / 0.34; neutral 0.21 / 7.2 / 0.46;
perturbed 0.36 / 14.9 / 0.44; strained 0.64 / 10.5 / 0.45; polar 0.56 / **0.76** / 0.46. Energies pass
everywhere (≤ 1); forces pass on `polar` only and fail on the other four (7–15 against ≤ 5) →
**G-D1.1 FAIL** on forces, by 1.5–3× (the light arm: 5–8×).
**G-D1.2 Born (sign fix −1):** Cl −0.487, Cs +0.438, Pb +1.064; |Pb/Cs| 2.43 ✓; **ΣZ* +0.64 e/cell**
(≤ 0.3) → **FAIL on the sum rule**, by 2× (the light arm: −4.0). Recorded: |q| 0.148 e (raw Cl
+0.134, Cs −0.082, Pb −0.256), net +1.02 e/cell, E_LES −98 meV/atom.
**G-D1.3 ablation:** LES zeroed → 99.7 meV/atom, 38.8 meV/Å (polar 0.76 → 4.8): the term is
structural here too (E_LES −98 meV/atom); PASS as written, not discriminating.

Against the amendment's prediction for this arm ("fits the teacher at least as well; charges
smaller than D1-light; ablation effect small"): the fit is 2.7× better on forces ✓; the charges
are smaller in |q| (0.148 vs 0.168) but not by much and E_LES is the same order (−98 vs −114); the
ablation effect is NOT small ✗. Both from-scratch students build ~100 meV/atom of their energy on the
Ewald term regardless of width; the Born pattern is the same in both (Cl −, Cs +, Pb +, ratio
2.4), and the sum rule is closer to holding in the full-shape one.

### D1-full-init — RESULT (20:42; `d1_full_init.model`, 30 epochs at base v2's optimiser settings)
Teacher fit: 0.39 meV/atom / 2.11 meV/Å overall (charged 0.19 / 0.84; neutral 0.40 / 0.87; perturbed
0.39 / 2.26; strained 0.39 / 1.22; polar 0.55 / 0.18) — **G-D1.1 PASS on every family** (it is the
teacher's own shape and weights). E_LES −4.0 meV/atom on every family (init scale). Born and
ablation appended below when read.
*D1-full-init, Born and ablation (20:55; read on b3 GPU 7 — the local card OOMs the BEC gradient
beside the light2 training):* Z* after the sign fix Cl −0.109, Cs +0.058, Pb +0.118 (|Pb/Cs| 2.04,
signs right) but **ΣZ* −2.45 e/cell** → G-D1.2 FAIL; |q| 0.027 e at the 0.01-init scale, raw species
means all negative (Cl −0.025, Cs −0.035, Pb −0.026: no ionic pattern), net −2.19 e/cell, E_LES
−3.7 meV/atom. Ablation: LES zeroed → 3.7 meV/atom, 2.95 meV/Å (from 0.39 / 2.11) — a ~1 meV/Å effect,
i.e. the term is a small correction that does not identify anything. **Matches the pre-registered
prediction** (E_LES stays at the init scale, no ionic pattern — the L0 mechanism reproduced from a
converged start), with the nuance that |q| did not fall below its init value here; it stayed there.

### D2 launched for the two students (user, 2026-09-15 21:05: "start doing the multi head finetuning with our actual data of the small and full scratch models"; registered before any epoch is read)

`d2_run.sh STUDENT NAME`: `--model MACELES --foundation_model <D1 stage-two model> --foundation_head
Default --multiheads_finetuning True --pt_train_file dataset_d0/replay_1616.xyz --les_shared_readout
True`, DFT data `dataset_e0/train_iso.xyz` / `valid.xyz` (the base v2 files), base v2's optimiser
settings (10/10, lr 1e-4 forced, EMA 0.9999, clip 1, wd 0, 30 epochs, batch 8, float32, cueq).
Interpretations registered: the plan's `dft` head is MACE's `Default`, the `teacher` head is
`pt_head`; **replay 1:1 by count** — a fixed stratified 1 616-frame subset of D0 train (seed 0;
charged_relabelled 56, neutral 94, perturbed 1 350, strained 107, polar 9) plus the teacher-labelled
isolated atoms, since this MACE loads a custom `--pt_train_file` whole and cannot resample per
epoch; the student's **latent-charge readout is transferred unscaled** into the shared readout
(`model_script_utils`, new: a MACELES foundation's `les_readouts` load when the shapes match;
previously they were re-initialised and scaled); each head keeps its own E0s from its isolated
atoms. Arms: `d2_light` (from `d1_light_stagetwo.model`, local card beside `d1_light2`) and
`d2_full_scratch` (from `d1_full_scratch_stagetwo.model`, b3 GPU 7). Gates G-D2.1–2.4 as in the
plan; G-D2.2 (charged shell table ≤ base v2 at every shell) decisive. Cross-fit bases (G-D2.4)
only for the arm that passes.
*D2 smoke and launch (21:40).* Two loader defects found and fixed by the one-epoch smoke before
launch: (i) the multi-head loader never transferred a MACELES foundation's latent-charge readouts
(re-initialised and scaled) — now copied unscaled when the shapes match; (ii) the cueq ⇄ e3nn
converters and `remove_pt_head` rebuild the model through `extract_config_mace_model`, which did
not carry `les_shared_readout`, so the shared readout silently became per-head (192 channels) and
the transferred charges were scrambled (initial replay error 270 meV/Å) — the flag is now part of
the config and gated (`test_shared_readout.py`, 3 incl. the cueq round trip). After the fixes the
replay head's initial error equals the student's own teacher fit (1.04 meV/atom / 37.95 meV/Å
against 1.09 / 38.5) — the transfer is exact — and the DFT head starts at 52.4 meV/Å. Launched:
`d2_light` (local card, beside `d1_light2`) and `d2_full_scratch` (b3 GPU 7), 30 epochs each.

### D2 — first results (2026-09-15 22:30; 30 epochs each)

**d2_full_scratch** (from the D1 full-shape student): final DFT head **2.28 meV/atom / 14.62 meV/Å**
(replay head 1.98 / 14.92); G-D2.1 (≤ 1.4 / 12.3) **FAIL** on both. Converted base
(exactness 1e-13): charged 79-atom shells **35.9 / 82.0 / 32.3 / 16.4 / 10.7 / 9.5** against base v2's
33.2 / 74.9 / 30.4 / 15.3 / 8.9 / 8.0 — worse at every shell → **G-D2.2 FAIL**; neutral 79 (in-sample)
17.8 / 32.1 / 16.4 / 14.1 / 10.6 / 9.0 against base v2's 10.96 / 20.0 / 10.0 / 8.8 / 7.1 / 6.5. Born:
Cl −0.518, Cs +0.444, Pb +1.143, |Pb/Cs| 2.57 ✓, **ΣZ* +0.52** (≤ 0.3) → G-D2.3 FAIL (the D1 student
was at +0.64: drift −0.12). Recorded: **|q| 0.332 e, E_LES −480 meV/atom** — the DFT fine-tune grew the
Ewald term 5× (student: 0.148 e, −98 meV/atom), net −1.99 e/cell.
**d2_light** (from the light student): final DFT head **4.23 / 37.54** (replay 3.01 / 44.81) — G-D2.1
FAIL by 3×; conversion first failed in `prune_elements` (the light student's product block carries
per-element weights and needs no pruning — the converter now prunes only when the element table
differs); shell table and Born being read.
Reading so far: the distilled students are weaker starting points for the DFT fine-tune than MH-1
itself (base v2 reached 10.23 meV/Å from MH-1's 43.9 start; d2_full_scratch reaches 14.6 from the
student's 15.4 replay error, i.e. the fine-tune barely moves the forces), and the replay does not
hold the charges: they grow rather than shrink. D2's decisive gate (G-D2.2) fails for the full-shape
student at every shell. The two remaining classes (light2, full1) are queued for the same fine-tune.

### D3 on the benchmark base (user, 2026-09-15 22:45: "Go ahead with the rest of the plan using the full scratch model as a benchmark. I will no longer be monitoring.")

Base of the head line: **`d2_full_scratch_base.pt`** — the DFT fine-tune of the D1 full-shape
student — used as a *benchmark* base: it failed G-D2.1 (2.28 / 14.62), G-D2.2 (every shell) and
G-D2.3 (ΣZ* +0.52), so nothing built on it is an adoption; the readings are the line's mechanics
on a base with live ionic charges. Sign fix -1 (Born: Cs > 0). Chain armed unattended:
- pre-test (`l2_pretest.py`, b3 GPU 7) and the L2 Jacobian cache (`l2_les_cache.py`, same card,
  ~30 min), then the L1 b3 seeds 3 4 6 (`w3_queue.sh`, GPUs 5/7, one per card, 13 GB headroom —
  they start as `d1_full1` / `d2_full1` free the cards); local seeds 0 1 2 start when `d2_light2`
  exits (`l1_local_after_d2.sh`);
- `lx_watch.sh L1` (env `lx_env.sh`: base, sign, cache) reads L1 against W6-on-v2 (reference
  only) and launches L2 (`--les_cross 1 --les_sign -1 --les_cache …/d2_full_scratch/les_cache.pt`),
  then L3 from L2's models if L2 is adopted by the registered rule; L4 is not automated;
- the remaining D1/D2 arms (`d1_light2`, `d1_full1`, `d2_light2`, `d2_full1`) complete and are read by
  their watchers; `morning_summary.sh` writes `~/runs/morning_summary.txt` at 07:30.

*L2 pre-test on the benchmark base (23:05, sign -1):*
    2,2,2 n=639 E_self 0.2219 E_cross 0.0849 (host 0.0862 delta -0.0014) Q_delta 0.145 q_site -0.253 Q_defect 0.398 dq(R<=4/8) -0.045/-0.004
    2,2,3 n=959 E_self 0.2364 E_cross 0.0958 (host 0.0936 delta 0.0022) Q_delta 0.145 q_site -0.253 Q_defect 0.398 dq(R<=4/8) -0.045/-0.004
    3,3,3 n=2159 E_self 0.2750 E_cross 0.1239 (host 0.1126 delta 0.0113) Q_delta 0.145 q_site -0.253 Q_defect 0.398 dq(R<=4/8) -0.045/-0.004
    3,3,4 n=2879 E_self 0.2809 E_cross 0.1285 (host 0.1156 delta 0.0129) Q_delta 0.145 q_site -0.253 Q_defect 0.398 dq(R<=4/8) -0.045/-0.004
    monopole test resid/span 0.0088
    E_cross cubic: b -1.2158 expected -1.4331 implied Q 0.338 residual@79 -115.6 meV
Monopole exact (registered ≤ 2 %). The 1/L residual of E_cross at 79 atoms against the ≤ 10 meV rule
decides whether L2's cross-size energy claim can stand on this base; the E_cross slope agrees with
−Q_defect·C/ε∞ as the consistency check requires.

### D3 run (user, 2026-09-16 09:15: "Run D3")
L2 running since 08:38 (its overnight launch was blocked by my verdict script — the epoch-averaged
block pairs only two W6-on-v2 seeds; fixed, recorded); seeds 1 and 2 moved from the local queue to
b3 (placeholders skip them locally; pulled back when done) so all six finish by ≈ 11:00. `lx_watch L2`
reads L2 vs L1 and launches L3 on adoption. **L4 armed (`l4_run.sh`):** on the chain's final line it
selects the last adopted stage's best seed — the lowest held force RMSE among C5-passing seeds, else
the lowest force — and runs the frozen-core ladder (R_core 8 Å; 959/2159/2879), the relaxation
ladder (959/2879), the leak readout, ε₀ by phonons with the trained `s_c` (0.5 if the stage has no
E_cross), and absolute ms per frame (`l4_ms.py`, held charged 79-atom frames, eager). The far-field
table against an out-of-fold neutral floor cannot be read on this base (no cross-fit bases: G-D2.4 was
not run because no D2 arm passed); the base's in-sample neutral shells stand in, marked as such.
M1 (the registered fallback) is built only if the L2 read triggers it (E_cross does not move the
far-field shells); the pre-test has already put E_cross forces-only on this base.

# Amendment — bond-transfer LES charges and the distillation fixes (user, 2026-09-16 09:40; registered verbatim before anything runs)

> LES charges: q_i = Σ_j [T(f_i,f_j,Z_i,Z_j) − T(f_j,f_i,Z_j,Z_i)] w(r_ij), T a small edge readout on
>   the student's node features; Ewald unchanged. Tests: Σq = 0 to 1e-12 on every structure;
>   vacancy Q_defect within 4 Å = 0 by construction; Born check reduces to sign and ratio.
> D0: drop σ = 0.2 Å (keep 0.05, 0.10); keep the rest.
> D1: full-shape student, MACE-standard schedule, train until teacher-fit forces ≤ 5 meV/Å
>   (this is the gate that was skipped); lightweight student only as a speed variant.
> D2: replay restricted to the charged-manifold family (the geometries whose MH-1 extrapolation
>   is the point), weight registered ≤ 0.3; lr for a from-scratch model; gates G-D2.1/2.2 must
>   pass before any head is trained — a head on a failing base measures the base.
> D3: re-run the L2 pre-test with the bond-transfer pattern (expected: no residual by
>   construction); E_cross cross-size claim reinstated if it passes.
> Parallel, base v2: M1 (head-learned bond-transfer host pattern) — same construction,
>   best base, identified by the charged forces. Whichever base passes its gates carries the
>   head; if neither LES base reaches base v2's robustness, M1 is the product.
> Record: identification-by-reach hypothesis withdrawn for ionic solids; ΣZ* = Σq identity;
>   the +0.40 e defect charge and the base's neutral-energy degradation share one cause.

**Corrections of record (user's reading, adopted).** (1) The students' large charges are the
expected result, not a defect: the Born pattern (Cl −0.5 to −0.8, Cs +0.4 to +0.6, Pb +1.0 to +1.4,
|Pb/Cs| 2.4–2.7) is the halide-perovskite signature at about Z*/√ε∞, which is what a latent charge
in an ε∞ medium should be; LES-FT2's 0.03 e charges were the anomaly. An E_LES of −100 to −300
meV/atom on every frame is the Madelung partition of an ionic crystal's cohesion between a local and
a Coulomb term — what LES does in an ionic solid; my report's "not a long-range correction" misread
the method and is withdrawn. (2) The head's 2× deficit on the benchmark base is the base, not LES:
the full-shape student fits its teacher to only 14 meV/Å where a distillation should reach a few,
the σ = 0.2 Å copies dominate every miss and are out of distribution for the teacher too, and D2's
1:1 replay anchored the fine-tune to that bad teacher fit (15.4 → 14.6); in base v2's recipe the
replay was free because MH-1's weights reproduced the replay labels exactly — here it is a constraint
fighting the DFT fit. (3) Non-neutrality has one root: for environment-dependent charges Σ_i Z*_i =
Σ_i q_i identically, so the sum-rule failure IS the net charge (the report scored two findings); the
+0.40 e compact defect charge carries an image self-energy ≈ αQ²C/2L (≈ 0.25 eV at 79 atoms, 0.16 at
159 — a spurious ≈ 1 meV/atom size dependence in the base's own neutral vacancy energies, the size of
the D2 energy degradation); and its sign is wrong — a neutral vacancy is locally neutral (the missing
Cl⁻ compensated by the electron on the flanking Pb), which an environment-to-charge map trained on
neutral data cannot express (under-coordinated Pb is made more positive), which is why E_cross had a
−116 meV residual. (4) "Cutoff-limited student" identification is withdrawn for ionic solids:
from-scratch students adopt the Ewald term as a cohesion device whatever their reach, because it is
the cheapest representation of ionic energetics; the paper's water and TiO₂ cases are different
physics. The mechanism control did what it was registered to do.

**Jobs cancelled 09:45:** L2 seeds (local s0, b3 s3/s4/s6/s1 queue), the L2→L3 chain, the L4 runner
(which had already started on L1 s1 from the stale 06:05 line), all D-watchers. Both cards idle.

## Plan P — polarisation coupling via MACE-Field: REGISTRATION (2026-09-16, before any Phase 0 result is opened)

**Governs.** `PLAN_P_polarisation_coupling.md` (user's, 2026-09-16) supersedes the LES head line
(L0–L4), the distilled-base line (D0–D3) and the bond-transfer amendment entirely. Nothing of those
lines runs again; their records above stand as history. Branch `plan-p-polarisation` off `defect`
at `97996ec`. Standing mandate unchanged: no new DFT, every threshold written before its result is
opened, F-SCC finals a separate record. Units as before (forces per component meV/Å, energies
meV/atom, τ = 1.7 meV/Å, six seeds {0,1,2,3,4,6} → folds {0,1,2,3,0,2}); b3 GPUs 5 and 7 only.

**MACE-Field port (done, gated).** mdi-group/mace-field at `45d5c5f` (mace 0.3.15) cherry-picked
onto this tree (0.3.17 + D-SCC): `MACEField` in `mace/modules/extensions.py` (field FCTP + Linear
after every interaction layer but the last, the fork's `h ← h − Linear(FCTP(h, E))` form),
`UniversalFieldLoss` + `fold_polarization`, `electric_field / polarization / becs` data keys and
weights, `--compute_{polarization,becs,polarizability}`, `PerAtomFieldRMSE` table, MACEField
paths in `configure_model` / `_build_model` / `remove_pt_head` / cueq-oeq asserts / the ASE
calculator (`model_type="MACEField"`, `electric_field=` override); `eval_configs`, `visualise_train`
and the LAMMPS wrappers of the fork NOT taken. Plan P extension: `electric_field_node [N, 3]`
replaces the broadcast field (`compute_node_dipoles` → `node_dipoles[a] = −dE/dE_node[a]`, e Å).
Tests `tests/extensions/field/test_macefield_node_field.py` (5 pass): node field = broadcast field
reproduces E, F, P, Z*, α bit-for-bit; Σ_a node_dipoles = Ω P to 1e-12; node dipoles = finite
difference; the delivered checkpoint loads; e3nn→cueq→e3nn round trip reproduces P, Z*, α to 1e-8.
Regression: `tests/extensions/dscc` + `tests/extensions/les` 351 pass; one shared-readout test
failed in the full run and passes alone (order-dependent, pre-existing, noted).

**Checkpoint (as delivered, `~/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model`,
sha256 `f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2`).** `MACEField`, heads
[`pt_head`, `mp-dielectric`, `mp-ferroelectric`], float32, r_max 6.0 Å, 2 interactions (one field
layer), 128x0e+128x1o, `RealAgnosticDensityInteractionBlock` (MACE-MH-0 / mp-0b3 lineage, per the
user: fine-tuned from MH-0, not MH-1 — differences from base v2's backbone expected), 79 elements,
8.17 M parameters. Head for all Plan P response work: `mp-dielectric` (the BEC/α head).
Load-gate readings (float64, informational): dataset frame 719 (CsPbCl3 polar path, mp-675524
branch) P_z 1.396e-3 vs label 1.409e-3 e/Å²; MP CsPbCl3 cells (mp-23037 cubic, mp-675524 Amm2,
mp-1539667 P4/mbm, mp-675022 R32): Z* tr/3 Cl −1.67…−1.74, Cs +1.22…+1.27, Pb +3.75…+4.01,
polycrystalline ε∞ 3.83–3.90 (MP DFPT e_electronic 3.62–3.64 for the two entries that have it).
The dataset `dielectric_ferroelectric_combined.xyz` (8446 frames) holds CsPbCl3 only as 10
polarisation-path frames (BEC weight 0), so the Plan P "tabulated PBE Born tensors" come from the
literature, recorded as validation not input; MP exposes ε tensors, not Born charges (DFPT task
docs carry no `born` field through the API).

**Definitions fixed for the implementation (plan §0, with the conventions this tree already has).**
Smoothed periodic kernels are the pure reciprocal-space sums of `mace/modules/dscc/ewald.py`
(`route="reciprocal"`, `background="density"`, k ≠ 0). `E_c(R_a)` uses the existing carrier–host
kernel `gamma_lr(r_g, r_split)` (pair width √(2(r_g²+r_split²)), r_g = 0.7071 Å) over ε∞; `T_LR`
is the Hessian of the same kernel form with both widths r_split (pair width 2 r_split), images
included, the a = b, L = 0 term removed analytically (−p_aᵀ∇∇Γ_w(0)p_a = 4C|p_a|²/(3√π w³)), the
k = 0 (surface) term excluded, over ε∞. Both matrix-free in k-space (structure factors, O(N·n_k)).
`p[a] = node_dipoles` of the MACE-Field provider at E_node = 0, attached to the positions (nested
autograd) so one backward of the composed energy delivers `dp/dR`. ε∞ appears in `E_M`, `E_c`,
`T_LR` only, one scalar per run: (R) 4.0, (M) the polycrystalline mean of the provider's ε∞ tensor
on the relaxed pristine cell.

**Phase 0 thresholds (verbatim from the plan).** 0.1 `r_split = max(3.0 Å, 1.2 × r90)`, r90 the
radius enclosing 90 % of Σ_b|dp[b]| per species for a 0.01 V/Å node field on ≥ 2×2×2 tilings of the
relaxed pristine cell; if r90 > 6 Å, accept and proceed (no second coupling representation);
partition-ambiguity RMS(p − p′)/RMS(p) recorded, not gated. 0.3 matching: RMS per component of
the `E_PP` forces on the neutral and charged training frames ≤ 3 meV/Å, else raise r_split until it
is. Decision 0: mechanism confirmed if `eps0_ladder` (route-1 relaxation ladder 639/959/2159/2879
atoms, fit `dE_relax(L) = dE_relax(∞) + (A/L)(1/ε∞ − 1/eps0_ladder)`, A = α(h)Q²C/2) is within 30 %
of `eps0_LST` (base v2 Γ Hessian + provider Z*, polycrystalline) AND the W6-as-shipped ladder is not.
Fallback: LO–TO splitting of the strongest IR mode at the smallest nonzero q of the 3×3×3 cell must
appear with `E_PP` and match the nonanalytic value to 20 %; splitting present + ladder failing →
base v2's restoring forces, Phase 1 proceeds; no splitting → stop and report. Run directory
`~/runs/dscc/p0/`, deliverable `P0_REPORT.md`. Phase 1–3 thresholds are in the plan file and are
copied here when each phase is registered.

**Phase 0.3 specifics registered before the ladders run (2026-09-16 22:05).** W6 checkpoint for
0.3: seed-median by held force RMSE over the six W6-on-v2 seeds (19.42/19.85/20.28/19.19/21.05/
20.68 meV/Å for s0/s1/s2/s3/s4/s6; median 20.06 → **s1**, `~/runs/dscc/dscc_w6_s1/model.pt`,
uncalibrated: at fixed cell and charge state `C_Q` and `E_M` cancel in `dE_relax`). The
polarisation variant is the SAME checkpoint with `host_coupling="polarisation"`, the MACE-Field
provider attached (head `mp-dielectric`), `r_split` set to the 0.1 value for both `E_cP` and
`E_PP`, ε∞ = 4.0 (variant R); W6's `s` is unused there. Gate before use (`p0_variant.py`): analytic
and capture routes refused; `dq`, band energy and `e_madelung` identical to W6 on a charged
79-atom training frame (only `e_host` → `e_cP` differs, plus `e_pp`); one central-difference
force check with the real base and provider. Ladder (`p0_ladder.py`): base-v2-relaxed pristine
80-atom cell (`~/runs/dscc/p0/pristine_relaxed_v2.json`: E −281.152 → −281.876 eV, rms
displacement 0.253 Å, max 0.312 Å; same tilt topology — every Cl bridges two Pb — with the
Pb–Cl–Pb angle 167.1° → 156.1° mean and Pb–Cl 2.860 → 2.905 Å; recorded, not gated) tiled
2,2,2 / 2,2,3 / 3,3,3 / 3,3,4 with the first Cl removed (639/959/2159/2879 atoms); core = full
BFGS relaxation of the +1 vacancy in the 639-atom cell (fixed cell, fmax 0.02 eV/Å, maxstep
0.1 Å, ≤ 400 steps); per tiling `E_frozen = E(+1)` at the ideal vacancy geometry, the core
displacements embedded within r_core = 8 Å (minimum-image keys, as `dscc_embed_ladder.py`), the
far field relaxed with the core fixed, `dE_relax = E_relaxed − E_frozen`; fit against
x = α(h)/L (cell-shape Madelung constant from `madelung_self`, not a bare 1/L: 2,2,3 and 3,3,4
are not cubic), `dE_relax = dE_∞ + (Q²C/2) x (1/ε∞ − 1/eps0_ladder)`, leave-one-out spread of
eps0 as the fit uncertainty. eps0_LST from `p0_eps0_lst.py` (base v2 Γ Hessian, h = 0.01 Å,
acoustic sum rule; provider Born tensors; polycrystalline). Sizes beyond what the local 16 GB
card or b3's 24 GB cards can hold are truncated with the reason written here before the fit is
read.

**Phase 0.1 read (2026-09-16 22:10; `~/runs/dscc/p0/p0_locality.json`).** 2×2×2 tiling of the
base-relaxed pristine cell (640 atoms), node field 0.01 V/Å on one atom of each species, three
directions each. r90 of Σ_b|dp[b]|: Cl 7.00–7.42 Å, Cs 7.71–8.04 Å, Pb 6.65 Å (the perturbed
atom itself carries 22–39 % of the response; 95–98 % is inside 8 Å, 99.9 % inside 10 Å — the
field couples after layer 0 and the second layer carries it one more r_max). r90 of Σ_b|dF[b]|:
8.0–8.5 Å. **r90_max = 8.04 Å > 6 Å: by the plan's rule the mid-field coupling reverts to H0 for
this iteration (accepted, no second representation), and r_split = max(3.0, 1.2 × 8.04) =
9.645 Å for both E_cP and E_PP.** Partition ambiguity RMS(p − p′)/RMS(p) = 1.11 (rms p 0.043 e Å,
rms p′ 0.077 e Å; both sums equal Ω P to 1e-9): the per-atom-energy partition is a different
object from the node-field partition — recorded, not gated. 0.2 on the same relaxed cell: ε∞
tensor diag (3.531, 3.535, 3.575), polycrystalline 3.547, anisotropy 1.2 % (tensor kernel not
needed); Z* trace/3 Cl −1.618 (Frobenius 3.52), Cs +1.277 (2.24), Pb +3.577 (6.21), ΣZ* = 0 to
1e-12; on the MP cells the same checkpoint gives ε∞ 3.83–3.90 (MP DFPT 3.62–3.64) — the spread
is the (M) variant's uncertainty. Variant (M) for this base: ε∞ = 3.547. eps0_LST (base v2 Γ
Hessian on the relaxed cell, no imaginary optical modes, optical range 1.9–27.9 meV; provider Z*):
ionic part 20.63 → **eps0_LST = 24.63 (R, ε∞ 4.0) / 24.18 (M)**; the four strongest modes at
8.6–10.5 meV carry 9.1 of the 20.6.

**Phase 0.3 matching read (22:35; `p0_matching_r9645.json`, all 2560 training frames, provider
dipoles with dp/dR).** At r_split = 9.645 Å the forces of E_PP alone are RMS 0.13 meV/Å per
component on the pristine-80 and neutral-vacancy-79 frames, 0.12 on charged-79, 0.05 / 0.04 on
the 159-atom families (E_PP itself −0.01 meV/atom): the 3 meV/Å threshold is met by more than
20×, no r_split raise. Noted for the reading of Decision 0: with w_PP = 2 r_split = 19.3 Å and
w_cP = 13.7 Å the two new kernels are smoother than the training cells (e^{−w²k²/4} at the
first k of a 32 Å edge is 0.03), so in the training domain E_PP is inert by construction and
only the ladder's larger cells exercise it — the plan's "mid-field reverts to H0" acceptance.
Scan for the record (same frames): F_PP RMS all-families 8.50 / 4.74 / 1.86 / 0.56 / 0.13 meV/Å at
r_split 3.0 / 3.6 / 4.5 / 6.0 / 9.645 Å (charged-159 the largest family at every width: 13.2 /
9.5 / 5.3 / 1.5 / 0.04); the 3 meV/Å threshold alone would have allowed r_split ≈ 5.5 Å.

**Variant gate and timing (22:50; `p0_variant_s1.json`; `~/runs/dscc/p0/w6_s1_polarisation.pt`).**
W6 s1 → polarisation, r_split 9.645, ε∞ 4.0. On the first charged 79-atom training frame: dq
identical (max |Δ| 0.0), Σdq = 1.000 both, band energy 6.816156 eV both, e_madelung −0.338319 eV
both; e_host −20.5 meV (W6) → e_cP −0.13 meV + e_PP −0.49 meV. Central-difference force check on
the variant (h = 1e-3 Å): 2.1 and 0.8 meV/Å from the analytic values (analytic route refused,
cotangent route in use; W6 as shipped is also a block-0-reading head, so it was never on the
analytic route either). **Forces of the untrained composition:** RMS difference to W6 16.9
meV/Å, RMSE against the label 17.1 (variant) vs 8.4 (W6) on that frame — the W6 head was trained
with E_host's geometry dependence in the loop and loses it here; this is the frozen-provider
mechanism model, not a force claim (Phase 2 retrains). Timing, one energy+force, charged +1, local
A4000 float64: 639 atoms W6 2.25 s / 5.8 GB, variant 3.75 s / 8.8 GB; 959 atoms W6 5.7 s / 8.5 GB,
variant 7.9 s / 13.0 GB. Linear extrapolation: variant ≈ 29 GB at 2159 and ≈ 39 GB at 2879 — beyond
both cards; W6 ≈ 19 / 25 GB. **Launched (22:52) on b3:** W6 ladder on GPU 5, variant on GPU 7
(`~/runs/dscc/p0/ladder_{w6,var}/`, runner `b3_ladder_runner.sh`: core stage at 2,2,2 then the
ladder at 2,2,2 and 2,2,3); the 2159/2879 rungs are being timed on the CPU (96 cores, local) before
their placement is decided — if unaffordable, the ladder is fitted on 639/959 (+2159 for W6) and
that truncation is written here before the fit is read.
CPU timing at 2159 atoms (local, 96 cores, float64; `p0_variant_cpu2159.json`): W6 45.1 s,
variant 65.7 s per energy+force — affordable (a 150-step far-field relaxation ≈ 2–3 h). **Placement
decided before any rung is read:** 639/959 on b3 GPUs 5/7 (running), 2159/2879 on the local CPU
(48 threads per arm, `p0_large_rungs.sh`, chained on the b3 `CORE_DONE` status files; the core
displacements come from the b3 639-atom relaxation of the same arm). No truncation of the
registered ladder.

## Plan P Phase 1 — base v5: REGISTRATION (2026-09-16 23:20, written before Decision 0 is read; the run opens only on a passing Decision 0 or the LO–TO fallback)

Recipe (`defect-perovskite/p1_run.sh`, the user's MACE-Field fine-tuning example adapted to the
base v2 registration): foundation = the delivered MACE-Field checkpoint, `--foundation_head
pt_head`; head `Default` on `dataset_e0/train.xyz` (1616 neutral: pristine 80, vacancy 79/159;
E/F/stress) with `valid.xyz` (154); replay `~/runs/dscc/p1/replay_fps1000.xyz` =
`fine_tuning_select` on `dielectric_ferroelectric_combined.xyz` with `--subselect fps
--num_samples 1000 --filtering_type combinations` (16 Cs/Pb/Cl configurations + 984 random
padding, as the filter gives on this set), `--pseudolabel_replay True` (E, F, stress, P, Z*, α
labels are the checkpoint's own predictions on the replay: the response losses on the bulk
labels are carried by the replay head, no new DFT); `--model MACEField --loss universal_field`,
`--E0s foundation`; weights E 10 / F 10 / stress 1 (base v2) and P 1000 / Z* 100 / α 100 (the
user's example, in place of "the original MACE-Field values" which the fork does not record);
lr 1e-4 (`--force_mh_ft_lr`), EMA 0.9999, weight decay 0, clip 1.0, no SWA, 30 epochs, batch 8,
float32, cueq, seed 1; **E_PP in the forward** with r_split 9.645 Å, ε∞ 4.0 (variant R);
`--compute_{polarization,becs,polarizability}`. Deviation if the response loss fails to hold:
halve the learning rate once (plan). Smoke: one epoch locally (`~/runs/dscc/p1/smoke`) to
validate the pipeline; production on b3 GPU 5 or 7 after Phase 0 closes.

Gates (plan §Phase 1, verbatim): (1) neutral validation forces ≤ 10.2 + 1.0 = 11.2 meV/Å per
component, energies ≤ 1.5 meV/atom; (2) pristine-cell Z* (per species, Frobenius) and
polycrystalline ε∞ within 10 % of the checkpoint's 0.2 values (Frobenius Cl 3.52 / Cs 2.24 /
Pb 6.21; ε∞ 3.547 on the relaxed cell); (3) base-alone force error on the charged frames by
shell not worse than base v2 (33 meV/Å total, 75 in 2–4 Å) by more than 3 meV/Å in any shell.
Deliverable `base_v5.model`, `P1_REPORT.md` with the three tables and ε∞(M) of v5.

**Ladder as first run (23:20; fmax 0.02 eV/Å, `ladder_fit.json`) — read, and the convergence
criterion found unfit before Decision 0 is taken.** dE_relax (eV) at 639/959/2159/2879 atoms: W6
−0.4219/−0.4255/−0.4266/−0.4278 → slope +0.133 eV Å, eps0_ladder **4.32** (leave-one-out 4.17–4.39);
variant −0.3847/−0.3890/−0.3884/−0.3876 → slope +0.056, eps0_ladder **4.13** (3.91–4.22). eps0_LST
24.6 would need a slope of +1.51 eV Å, i.e. ≈ 48 meV more relaxation between the 639- and
2879-atom cells; the ladders show 6 and 3 meV. Every far-field relaxation stopped after 6–7 BFGS
steps at fmax 0.015–0.017 eV/Å. **Why that criterion cannot measure this quantity:** the
far-field response to the +1 charge is a displacement field driven by forces F ≈ Z* E_c ≈ Z* C /
(ε∞ r²): 0.026 eV/Å on a Cl at 15 Å, 0.010 at 24 Å — at or below fmax = 0.02, so the optimiser
declares convergence with the whole dielectric shell unrelaxed; its energy, Σ F²/2k over ~2000
atoms at k ≈ 5 eV/Å², is ≈ 0.05–0.1 eV, exactly the size of the missing term. The threshold on
eps0 (30 %) is untouched; the relaxation tolerance is re-registered at **fmax 0.002 eV/Å** for
the core and the far field (≤ 1000 steps, BFGS maxstep 0.1), output `ladder2_*`, and the
fmax-0.02 numbers stand in the record as the first run. The LO–TO fallback (`p0_loto.py`) is run
in parallel so that Decision 0 can be read either way as soon as the tight ladders land.

**Precision of record (23:45).** The W6 checkpoints carry their base in float32 (`base_float32`
in the W6 training); every Phase 0 evaluation from here casts it to float64 at load
(`p0_ladder.py`, `p0_loto.py`, `p0_variant.py`), so the ladder energies are free of the ≈ 1 meV
float32 summation noise that the fmax-0.02 first run and the variant gate carried. The tight
ladders (fmax 0.002) launched at 23:19 with the float32 base are stopped and relaunched with
the float64 base; nothing else changes. The variant gate's timings (8.8 / 13.0 GB) were with the
float32 base and are indicative only.

**LO–TO fallback, first reading (float64 base v2, float32-base variant; A = 0.02):** strongest
IR mode of the base-v2 Hessian at 10.06 meV, Z̃ = −1.894 e/√amu along y; q = 2π/(3b) = 0.131 Å⁻¹.
Curvatures (eV Å⁻² amu): base alone LO 0.581 / TO 0.331 (finite-q dispersion, not a nonanalytic
term); E_PP alone LO +0.135 / TO −0.001, i.e. **the splitting appears with E_PP: +0.136**;
nonanalytic expectation for the same pattern from Z* and ε∞ = 4: 0.373 raw (q → 0 kernel),
0.077 at the registered smoothing (g = e^{−w²q²/4} = 0.206 at this q). The variant's total
curvature was float32 noise at this amplitude (−0.02 / −0.61) and is rerun at A = 0.05 with the
float64 base, with the provider's dipole response to the pattern (form factor against Z* u) as a
diagnostic before the 20 % criterion is read.

**LO–TO fallback, second reading (float64 base and variant, A = 0.05; `p0_loto.json`).** The
variant's curvature is now additive (LO 0.716 = 0.581 + 0.135; TO 0.331 = 0.331 − 0.001). E_PP
alone splits the strongest IR mode by **+0.136 eV Å⁻² amu**. Reference from Z* and ε∞ = 4 for the
same displacement pattern, with the ±q structure-factor pair counted (checked numerically: the
kernel on the pattern p = Z* u equals twice the single-q expression to 0.3 %): **0.746 unsmoothed
(q → 0 kernel)**, **0.154 at the registered smoothing** (g = 0.206 at q = 0.131 Å⁻¹), and 0.135
once the provider's own dipole response to the pattern is used (form factor 0.94 of Z* u along
the LO axis, 0.83 along TO). So: the splitting appears; it matches the registered kernel's
value to 0.3 % (the kernel and the provider do exactly what they are built to do), and it is 18 %
of the unsmoothed nonanalytic value — the 79 % deficit is the smoothing that r_split = 9.645 Å
(the plan's own locality rule, > 6 Å accepted) puts on the mid field at this q. Reading by the
plan's fallback text: "the splitting appears" — the deficit, if the tight ladder confirms one,
is not in the dipole mechanism; Phase 1 proceeds, and the plan's "mid-field reverts to H0"
acceptance is what limits E_PP/E_cP to the far field (beyond ≈ 2 r_split).
Placement update (23:58): with the float64 base the two arms no longer share a 24 GB card (W6
10.7 GB + variant > 12.2 GB at 639 atoms), so the variant's tight ladder runs entirely on the
local CPU (`p0_var_local.sh`, `ladder2_var_local/`, 48 threads) and the W6 arm keeps its b3
core/small rungs with the 2159/2879 rungs local (`p0_w6_large.sh`). Same tolerances, same
scripts, same cells.

**Jobs stopped (user, 2026-09-17 00:35).** Killed by PID: the local variant tight ladder
(`p0_var_local.sh`, its 639-atom far-field relaxation at step ≈ 30) and the local W6 large-rung
chain (`p0_w6_large.sh`, 2159-atom relaxation at step ≈ 12); b3 idle (W6 tight 639/959 done:
−0.4331 / −0.4369 eV, two-point eps0 5.3; the smoke `p1/smoke2` complete). Tight-ladder state
of record: W6 639/959 tight, variant core only (−0.3957 eV). Nothing else running on either
machine.

## Plan P — AMENDMENT A1 (user, 2026-09-17 00:45; registered before any amended result is opened)

Verbatim: (1) r_split is set by matching, not locality: the largest E_PP force on training
frames that stays under 2 meV/Å → **r_split = 4.5 Å** (all-families RMS 1.86 meV/Å) for both
E_cP and E_PP; the r90 rule and the "mid-field reverts to H0" clause are dropped. (2) 0.1 is
replaced by the form factor: the provider's modulated-displacement response at λ = 48, 24, 16,
12 Å along the three axes, f(q) with the fitted Gaussian w_eff (expected ≈ 4 Å; the LO–TO test
already gave f = 0.94 at q = 0.131 Å⁻¹ ⇒ w ≈ 3.8 Å); if w_eff exceeds r_split on any axis,
raise r_split to w_eff, no further. (3) 0.3 rerun at r_split 4.5 Å: LO–TO (expect E_PP ≈ g² f² ×
0.746 with g = 0.71 at this q, ≈ 0.45) and the variant ladder at fmax 0.002; the W6 tight ladder
is kept as the control. (4) Decision 0 restated: **variant eps0_ladder ≥ 8 and ≥ 2 × W6's, with
LO–TO matching the smoothed expectation to 10 %**; quantitative agreement with eps0_LST is a
Phase 3 question at sizes where g → 1. (5) Phase 1 runs with r_split 4.5 Å in the forward (the
1.9 meV/Å is what the fine-tune matches the local part to); the pseudolabel fix stands; the base
is MH-0 lineage, 128 channels, r_max 6 against base v2's 512-channel MH-1 — the relative
robustness gate is the one that matters most, and the head's readouts port to 128-dimensional
features. (6) Phase 2 is no longer a null; (M) ε∞ is v5's on the relaxed cell, the 3.55–3.90
spread its uncertainty. Flagged for after Phase 2: distil the provider's Z*(R) into a small
local bond-dipole model (unlimited labels at any geometry, locality by construction, no nested
autograd at inference). The Z* validation gap is accepted (ε∞ within 10 % of MP DFPT and the
anomalous pattern are what a tabulated tensor would have checked).

Reading of record of the r90-based 0.3: a null by construction (w_PP = 19.3 Å carries 20 % of
the dipole physics at λ = 48 Å and none in the 4–15 Å region; eps0_ladder 4.1–4.3 for both arms
is what that model must give); the LO–TO agreement to 0.3 % is the positive finding.

Implementation registered: `p0_formfactor.py` (3×3×3 tiling, 2160 atoms, strongest-IR-mode
pattern e/√M · cos(q·R), q = n · 2π/L_axis, n = 1…4, along x, y, z; f = |Σ (q̂·Δp_a) e^{iq·R_a}| /
|Σ (q̂·Z*_a u_a) e^{iq·R_a}| with Δp from ±A central differences of the provider's node dipoles;
w_eff per axis by least squares of −4 ln f against q²); the variant re-cut at r_split 4.5
(`w6_s1_pol_r45.pt`, gate as before); `p0_loto.py` at 4.5; variant tight ladder core + 639/959
on b3 GPU 7, 2159/2879 local CPU; W6 tight 2159/2879 completed locally (control). Decision 0 is
computed by `p0_decision.py` from the JSON records into `~/runs/dscc/p0/DECISION0_STATUS`, and
Phase 1 (`p1_run.sh`, `--pp_r_split 4.5`, 30 epochs, b3 GPU 5) launches on PASS.

**A1 variant gate (00:50; `p0_variant_s1_r45.json`, `w6_s1_pol_r45.pt`, float64 base).** Same
charged 79-atom frame: dq, band energy (6.816157 eV) and e_madelung identical to W6; e_host
−20.5 meV → **e_cP −40.9 meV**, e_PP +0.2 meV; central-difference checks 0.35 and 0.002 meV/Å.
Forces of the untrained composition: 11.1 meV/Å RMS from W6, 11.8 vs 8.4 against the label
(was 17.1 at r_split 9.645): the coupling is now active where the data are.

**A1 form factor (01:05; `p0_formfactor.json`; 3×3×3 cell 47.1 × 48.5 × 34.1 Å, strongest IR
mode, A = 0.05).** f(q): x 0.945 / 0.810 / 0.528 at λ = 48.1 / 24.1 / 12.0 Å; y 0.939 / 0.787 /
0.454 at 48.2 / 24.1 / 12.1; z 0.834 / 0.531 at 34.2 / 17.1 (z has the short axis, c = 11.4 Å).
The n = 3 points (λ = a, b, c: lattice-periodic patterns, not modulations; f = 0.11, 0.61, 1.58
with a 180° phase on x) and the z n = 4 point (λ = 8.5 Å, f = 0.09, phase −23°: sub-cell
aliasing) are outside the method and excluded from the fit. **w_eff (least squares of −4 ln f on
q²): x 3.09, y 3.43, z 4.21 Å; max 4.21 < 4.5 → r_split = 4.5 Å stands.** Noted: the z axis is at
the edge (its n = 1 and 2 points alone give 4.6 and 4.3 Å). Launched 01:07: LO–TO at 4.5 (local
CPU), the variant tight ladder at 4.5 (`ladder3_var`: b3 GPU 7 core + 639/959, local 2159/2879),
the W6 tight control rungs (running since 00:36), and the Decision 0 → Phase 1 chain
(`p1_launch_on_pass.sh`, status files).

**Phase 1 instruments (01:20).** `p1_make_base.py`: the MACE-Field density block's skip_tp
carries per-element weights the old `prune_elements` refuses, so the {Cl, Cs, Pb} base is built
through MACE's element-filtered loader and checked against the unpruned Default head (E 1e-13,
F 1e-14, P 1e-19, Z* 1e-14, α 1e-16 on six frames). `p1_gates.py`: the three registered gates
(neutral validation; Z* Frobenius per species and ε∞ within 10 % of the 0.2 values; charged
shells vs base v2 33.2 / 74.9 / 30.4 / 15.3 / 8.9 / 8.0 + 3). Exercised on the one-epoch smoke
(r_split 9.645, not a reading): the response of a one-epoch fine-tune already drifts 14–22 %
from the checkpoint (ε∞ 3.55 → 3.06, |Z*| −19…−22 %) — gate 2 is a live risk for Phase 1; if it
fails, the registered deviation is to halve the learning rate once (plan) and, failing that,
raise the response weights on the replay head with the change written here first.

**Correction of the LO–TO reference (01:30).** The single-q expression in `p0_loto.py` omitted
the −q partner of the structure factor; the k-space kernel counts both (checked numerically on
the pattern p = Z* u: kernel = 2 × single-q to 0.3 %). The reference fields are doubled in the
script and in the records (`p0_loto_fix2.py`); the model-side numbers are unchanged. Restated
9.645 Å reading: E_PP splitting 0.136 vs 0.154 smoothed (0.135 with the form factor), 0.746
unsmoothed — as already written in the report. Preliminary 4.5 Å LO branch: E_PP 0.461 vs
0.467 smoothed × ff² (g = 0.709, f = 0.939), i.e. 1.3 % — within the A1 10 % criterion; the TO
branch and the JSON follow.

**A1 LO–TO at r_split 4.5 Å (01:40; `p0_loto_r45.json`, ±q-corrected).** Same mode and q:
E_PP alone splits LO − TO by **+0.468** eV Å⁻² amu (LO +0.461, TO −0.007); expectation from Z*
and ε∞ = 4 at the registered smoothing (g = 0.709) with the provider's form factor (0.939):
**0.467** → deviation **0.17 %** (criterion ≤ 10 %: met). Unsmoothed 0.746; the smoothed
reference without the form factor 0.529. Variant total LO 1.042 = 0.581 + 0.461, TO 0.324.
The LO–TO half of Decision 0 (A1) passes; the ladder half is running.

**A1 variant ladder, small rungs (01:50; `ladder3_var`, tight, r_split 4.5).** 639: −0.4072 eV
(embed only −0.3222), 959: −0.4076 — a 0.4 meV size dependence (W6 tight: −0.4331 / −0.4369,
3.8 meV); two-point eps0_ladder 4.1. The 2159/2879 rungs are running. Registered now, before
the fit is read: Decision 0's outcomes are PASS (ladder + LO–TO), FALLBACK_PROCEED (LO–TO within
10 %, ladder criteria not met: the plan's unchanged fallback clause, "the deficit is in base v2's
restoring forces and Phase 1 proceeds anyway") or FAIL (LO–TO not met); `p0_decision.py` writes
the three states and the launcher opens Phase 1 on the first two. The LO–TO half is already met
(0.17 %). Diagnostic to add for the report, not a gate: the far-field displacement pattern of
the relaxed 2879-atom cells against the continuum q Z*/(ε0 r²) form, for both arms.

**Far-field displacement diagnostic on the tight 959-atom cells (02:00; `p0_farfield.py`,
`farfield_{w6,var45}_223.json`; not a gate).** Radial displacement from the ideal tiling, rms
by shell: W6 0.030 / 0.017 / 0.011 / 0.006 / 0.006 / 0.0035 / 0.0023 Å at 8–10 / 10–12 / 12–14 /
14–16 / 16–18 / 18–20 / 20–24 Å; variant (4.5) 0.024 / 0.016 / 0.009 / 0.005 / 0.005 / 0.0028 /
0.0021. Far-field exponents of |u_r| against r (continuum −2): W6 Cl −2.97, Cs −2.30, Pb −3.45;
variant Cl −2.53, Cs −2.31, Pb −3.03; amplitudes u_r r² (e Å²-scale, sign = outward): Cs
+1.05 / +0.88, Pb +0.89 / +0.71, Cl −0.22 / −0.10 (W6 / variant). Reading: both arms relax the
far field with the same magnitude and the same faster-than-r⁻² fall-off (image truncation in a
33 Å cell is expected to steepen it); the variant's active coupling does not enlarge the
far-field response at this size. The 2879-atom cells will be read the same way.

**W6 tight ladder complete (04:05; the control).** dE_relax at 639/959/2159/2879: −0.4331 /
−0.4369 / −0.4378 / −0.4392 eV (far-field relaxations 57/60/49/56 BFGS steps, fmax 0.0012–0.0016);
slope +0.137 eV Å → **eps0_ladder 4.33** (leave-one-out 4.16–4.40) — the same size dependence
as the loose run (4.32) with every rung 11–12 meV deeper: the tolerance was not what hid the
term; the shipped model's far-field relaxation carries no lattice screening beyond ε∞ at these
sizes. Far field of the 2879-atom cell: rms u_r 0.029 / 0.017 / 0.011 / 0.006 / 0.006 / 0.003 /
0.0023 / 0.0010 Å at 8–10 / … / 20–24 / 24+ Å; exponents Cl −3.24, Cs −2.70, Pb −3.01. Variant
(4.5) so far: −0.4072 / −0.4076 / −0.4058 (three rungs, eps0 3.9); its 2879 rung is at fmax
0.003.

**Variant tight ladder at r_split 4.5 complete; DECISION 0 (A1) READ (04:15).** dE_relax at
639/959/2159/2879: −0.4072 / −0.4076 / −0.4058 / −0.4017 eV (48/49/47/48 steps, fmax ≤ 0.0015);
slope −0.127 eV Å → **eps0_ladder 3.74** (leave-one-out 3.65–3.89), against W6's 4.33. Far field
of its 2879-atom cell: rms u_r 0.023 / 0.016 / 0.009 / 0.005 / 0.005 / 0.0025 / 0.0018 / 0.0008 Å
by shell, exponents Cl −3.13, Cs −2.70, Pb −2.88 — the same pattern and magnitude as W6.
Criteria: eps0_var ≥ 8 **not met**; eps0_var ≥ 2 × W6 **not met**; LO–TO within 10 % **met**
(0.17 %). **Outcome: FALLBACK_PROCEED** — by the plan's unchanged fallback clause ("the
splitting appears but the ladder fails → the deficit is in base v2's restoring forces and Phase
1 proceeds anyway"). Reading of record: with the coupling active (E_cP −41 meV on the training
frame, E_PP the exact smoothed nonanalytic response), the far-field relaxation energy of the
frozen composition still carries no lattice screening beyond ε∞ between 639 and 2879 atoms; the
relaxed far-field displacement pattern is the same as W6's. What the ladder measures is the
model's (base v2 + head) relaxation response to the carrier, not the kernel; at these sizes it
does not show the ε0 physics for either arm. Quantitative eps0 is a Phase 3 question on base v5.
`DECISION0_STATUS` written by `p0_decision.py` (a JSON-bool bug in its first version crashed
the chain once; fixed, rerun by hand, same outcome). **Phase 1 launched 04:16 on b3 GPU 5:**
`p1_runner.sh base_v5 30 ~/runs/dscc/p1/base_v5 5 --pp_r_split 4.5 --batch_size 4
--valid_batch_size 4` (batch 4: the registered batch 8 did not fit a 24 GB card with the BEC
second derivatives on the up-to-288-atom replay structures in the smoke; registered deviation).

## Plan P Phase 2 — head W7 on base v5: REGISTRATION (2026-09-17 04:50, written before Phase 1 closes)

Base: `~/runs/dscc/p1/base_v5/base_v5_base.pt` (`p1_make_base.py` from the Phase 1 model; opens
only if the Phase 1 gates pass, or with the user's written deviation). Head: `MACEDSCC` on v5's
first block (128x0e + 128x1o; `block0_layout` reads it; same bounds, pristine centring
recomputed on v5 by `set_pristine_reference` / `set_feature_stats`). The registered W6 command
(`launch_w6_s0.sh` flags: `--epochs 60 --directional 1 --regime B --coupling 0 --scf_free 1
--route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4
--base_cache_batch_size 4 --gpu_memory_fraction 0`, lr 2e-3 cosine, batch 4, grad clip 10,
w_E = 0.05, gap regulariser, τ = 1.7 meV/Å) for every arm; six seeds {0,1,2,3,4,6} → folds
{0,1,2,3,0,2}. Arms: **A** = W6 as shipped on v5 (`--host_coupling mulliken`, `--cache_base 1`),
names `dscc_p2a_s*`; **B** = `--host_coupling polarisation --field_provider base --cache_base 0
--r_split 4.5 --eps_inf 4.0` (the base is its own provider: dipoles from the base's forward,
E_PP inside the base, E_cP replaces E_host, no learned scale; the q0 contraction is gone),
names `dscc_p2b_s*`; **B(M)** = B with `--eps_inf` = v5's polycrystalline ε∞ on the relaxed cell
(from `P1_GATES.json`), seeds 0, 1, 2 only, names `dscc_p2bm_s*`. Run dirs `~/runs/dscc/p2/`.
Placement: b3 GPUs 5 and 7 one run each (`w3_queue.sh`, `W3_GPUS="5 7"`), arm A seeds on the
local A4000 in series; s/epoch reported per arm (arm B carries the base's nested autograd every
step: nothing cached). Gates (plan §Phase 2, verbatim): (1) paired TOST B vs A on charged-frame
forces, total and 4–12 Å shell, τ = 1.7 meV/Å per component; B adopted on superiority in
either; on equivalence B is still adopted; (2) `C_Q` profiled at 79 atoms transfers to 159
within 10 meV; 79-atom energy ≤ 0.6 meV/atom. Standing tests on the new composition
(`tests/extensions/dscc/test_polarisation_head.py`, the field-base block: neutral null
bit-exact, Σδq = Q, FD forces including the nested-autograd terms, batched = per-graph,
translation invariance, analytic/capture refused; the six strains are NOT covered — the stress
path raises under the polarisation coupling and is registered as out of scope for Phase 2's
force/energy gates). Deliverable: 15 checkpoints with calibration records, `P2_REPORT.md` with
the paired table, shell table, energy table and cost per frame at 79 and 159 atoms (eager;
memory reported).
Run-directory correction (04:58): `w3_queue.sh` places runs at `~/runs/dscc/<name>`; Phase 2 runs
live there (`~/runs/dscc/dscc_p2{a,b,bm}_s*`), not under a `p2/` subdirectory.

**Phase 1 training complete (08:34; b3 GPU 5, 30 epochs, 8.6 min/epoch).** Neutral head
(`Default`, 154 validation frames, MACE table = per component): epoch 0 → 29: 4.90 → **3.00
meV/atom**, 37.65 → **17.94 meV/Å**, stress 1.02 → 0.51 meV/Å³ (trajectory 29.3 / 25.3 / 22.2 /
20.5 / 19.3 / 18.4 meV/Å at epochs 2 / 5 / 10 / 15 / 20 / 25: still falling at 30). Replay head
(`pt_head`, 100 pseudolabelled frames): E 0.03 → 15.1 meV/atom, F 0.17 → 33.7 meV/Å, P 0.000 →
0.209 m|e|/Å², Z* 0.0005 → 0.036 e, α 0.004 → 0.452 ε0. Read against the registered gates
BEFORE the gate script runs: gate 1 (≤ 1.5 meV/atom, ≤ 11.2 meV/Å) is not met on either number;
the replay's α drift (0.45 ε0 against a response of ≈ 2.5) makes gate 2 (10 %) doubtful. The
gate chain (`p1_after.sh`) writes `P1_GATES.json` / `P1_STATUS`; the Phase 2 chain blocks on
anything but ALL PASS. Context for the reading: base v2 (512-channel MH-1, lr 1e-4, 30 epochs)
reached 3.9 meV/Å on the same split; the 128-channel MH-0-lineage checkpoint with the response
losses in the loop reaches 17.9 — the user's A1 note that the base change makes the relative
robustness gate the one that matters most applies here in full.
