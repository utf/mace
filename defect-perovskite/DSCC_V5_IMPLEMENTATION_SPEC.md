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
