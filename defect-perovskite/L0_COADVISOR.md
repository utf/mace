# L0 (LES head line) — status and options for the coadvisor (2026-09-15, 12:15)

Units: forces per-component meV/Å (the MACE table's RMSE_F is per component); energies meV/atom.
Nothing below is a gate reading: the registered gates are read automatically at epoch 29
(≈ 12:35 b3) by `l0_watch.sh` → `~/runs/base_v4_lesmh_prod/L0_GATES.json`.

## 1. What L0 is and how it is running

`base_v4_lesmh_prod`: base v2's registered multi-head fine-tune of MACE-MH-1 (`omat_pbe` head,
the sel10k fps/combinations replay, energy 10 / forces 10, lr 1e-4, EMA 0.9999, clip 1, wd 0,
30 epochs, batch 8, float32, cueq, E0s as IsolatedAtom frames) plus ONE addition: `--model
MACELES --les_readout_scale 0.01` (LES-FT2's scale). Heads `['pt_head', 'Default']`, each with
its own latent-charge readout. Four fold bases follow on the same recipe. b3 GPUs 5 and 7 only
(4 has now joined 6 off the bus). 5 min per epoch.

| epoch | Default energy | Default forces | base v2 prod at the same epoch |
|---|---:|---:|---|
| 16 | 1.35 | 10.41 | 1.19 / 10.77 |
| 24 | 1.31 | 10.05 | — |
| 25 | 1.27 | **10.01** | epoch 29: **1.16 / 10.23** |

Forces are already below base v2's final value; energies are 0.1 meV/atom behind. Fold base f0
at epoch 26: 0.62 / 3.27 (base v2 f0 final 0.43 / 3.88).

## 2. The latent charges are shrinking, not growing

Read off the raw-weight checkpoints (not the EMA), Default head after conversion, 8 neutral
80-atom validation frames; LES-FT2 (direct fine-tune, lr 1e-3, 150 epochs) for scale:

| | LES readout ‖W‖ Default (linear / nonlinear) | \|q\| mean | Cl / Cs / Pb means | net e / cell | E_LES |
|---|---|---:|---|---:|---:|
| init | 0.23 / 0.03 | 0.017 | −0.016 / −0.004 / −0.032 | −1.35 | −1.40 |
| epoch 17 | 0.41 / 0.14 | 0.018 | −0.020 / +0.003 / −0.026 | −1.31 | −1.31 |
| **epoch 23** | **0.44 / 0.16** | **0.013** | −0.015 / −0.001 / −0.018 | −1.01 | **−0.73** |
| LES-FT2 (149) | 0.64 / 0.19 | 0.031 | −0.042 / +0.010 / −0.017 | −2.11 | −4.39 |

The readout weights grow slowly (the fastest-moving part of the Default head — its energy
readouts moved 0.04 % in the same epochs), but the charges they produce fall: |q| 0.018 →
0.013 e, E_LES −1.3 → −0.7 meV/atom, the species pattern collapsing towards zero. The pt_head's
LES readout is exactly unmoved (0.222 / 0.022 at init, 17 and 23).

**Gradient check (epoch 17, 8 batches of Default frames, the 10/10 loss):** the LES readouts
receive 3–15× smaller gradients than the energy readouts (7.5e-4 vs 2.4e-3 / 1.2e-2) but just as
batch-consistent (consecutive-batch cosine +0.99 linear / +0.85 nonlinear against +1.00) — under
Adam the drift is set by consistency, so the LES readouts are not starved of updates. The
shrinkage is therefore a *representation* effect: 10 000 replay frames anchor the shared backbone
to MH-1's (no-long-range) features, and 1 616 fine-tuning frames ask the Default LES readout to
build charges from those features; the cheapest fit is a smaller Ewald term.

**pt_head.** Its replay labels are MH-1's predictions, which contain no long-range term, so its
correct latent charge is zero on every replay frame; the unmoved readout is Adam averaging a
zero-mean gradient (9 959 of the 10 000 replay frames are random MP structures). It nonetheless
*carries* an untrained 0.01-scale Ewald term on every replay frame — the initial replay error is
0.98 meV/atom / 1.72 meV/Å where base v2's was exactly zero — noise the backbone is asked to
reproduce MH-1 through. It is discarded at conversion, so it neither helps nor hurts the
converted base directly.

## 3. What is registered and what is expected at epoch 29

- G-L0.1 neutral validation ≤ 10.23 / 1.16 — forces will pass; energies are borderline (1.27 at 25).
- G-L0.2 charged base-alone shell table ≤ base v2 at every shell (33.2 total, 74.9 at 2–4 Å) —
  the replay-robustness gate; unknown until read.
- G-L0.3 Born check (Cl −, Cs +, Pb + after the sign fix; |Pb/Cs| ∈ [2, 3.5]; |ΣZ*| ≤ 0.3) — at the
  current trajectory the charges are ~40 % of FT2's and the Cs/Pb pattern is still turning over;
  a failure on ratio or sum rule is likely.
- Recorded, not gated: net latent charge, per-species means, E_LES, readout norms.

Decision rule as registered: all pass → L1 launches automatically (six W6 seeds on the new base);
G-L0.2 fails → fallback; G-L0.3 fails → L1 still runs, **L2 is not opened** until understood.

Everything downstream is built and gated but untrained: head-side `base_forward` dispatch (a
MACELES base's Ewald term was silently dropped before), L2 (`E_cross = s_c dqᵀ K_bare q_LES`,
cached-Jacobian force route ≡ base backward), L3 (`H0' = H0 − s_W(K q_LES − mean)`, gap
regulariser on H0'); 29 gates, full dscc suite 346 passing. The L2 pre-test measured the LES
kernel convention (`r_g = σ/√2 = 0.707 Å`) and the monopole exactness of the L2 kernel (0.9 %).

## 4. Options (none launched; each is a registered change, not a tweak in flight)

1. **Continue the same run** from the epoch-29 checkpoint to 60 epochs (`--restart_latest`):
   no new hyperparameter, ~2.5 h. Tests only whether the charges keep shrinking.
2. **LES off for the pt_head** (mask `les_q` on pt_head nodes in `MACELES.forward`): the replay
   head becomes exactly base v2's; removes the Ewald noise from the replay loss. A few lines; a
   cleanup that should be in any multi-head + LES recipe, but it does not change the replay's
   pull on the backbone.
3. **Per-parameter-group lr for `les_readouts`** (1e-3, FT2's value), everything else identical.
   Targets update speed, which the gradient check says is not the limiting factor.
4. **Fewer replay samples**: the guide's 0.1 ratio floor is 1 616 frames (6× fewer than the
   10 000 now); reduces the anchor, weakens replay robustness (G-L0.2 is the gate for that).
5. **Distillation (the user's option):** distil MH-1 into a MACELES trained on the distilled
   labels, then fine-tune on the defect set (arXiv:2606.15001). Caveat stated by the user: the
   point of fine-tuning from MH-1 is the base error on the *charged* structures, and MD-generated
   distillation geometries relax back to the neutral distribution, so the geometry generator
   would need thought. Sidesteps the replay entirely; the largest change.
6. **The plan's fallback:** LES-FT2 with its charges-off control (controlled pair: 0.48 → 0.24
   meV/atom, forces 5.69 → 5.62), recorded as lacking replay robustness (FT2 on charged frames:
   52.4 total / 124.0 at 2–4 Å against base v2's 33.2 / 74.9).

Not an option: a *trainable* pt_head LES or a shared LES readout — no label in this lineage
carries long-range content for the replay, and sharing would push the charges to zero faster.

## 5. Questions for the coadvisor

- If G-L0.3 fails on magnitude/ratio but G-L0.1/2 pass, does L1 still run on this base (the
  registered rule says yes) or is the base replaced first? L1's purpose is the baseline of the
  line; L2 is what needs physical charges.
- Which combination to register as L0.1: (2) alone is the minimum; (2)+(1) tests time; (2)+(4)
  tests the anchor; (5) is the reset. My reading of the trend is that (2)+(1) will not reverse
  the shrinkage and (2)+(4) is the cheapest experiment that addresses the identified cause, at
  the cost of the robustness gate.
- Whether the Born check should be read on the EMA weights (the saved model) or the raw
  weights: the EMA lags by ~7 epochs at decay 0.9999, so the saved epoch-29 model is roughly the
  raw epoch-22 state.
