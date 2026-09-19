# Plan P — preliminary report: Phase 1 base, override, and Phase 2 arms so far

*2026-09-19. Units: forces per component meV/Å, energies meV/atom, τ = 1.7 meV/Å. Base seeds and
folds as registered; the pair in every paired reading is the seed (= the fold = the held-out set).*

## 1. Phase 1 — the MACE-Field base

**Goal.** A MACE-Field (mp-dielectric head) base fine-tuned on the neutral CsPbCl3 defect data that
keeps its field response, so the head can use the base's own dipoles for the polarisation
coupling `E_cP = −Σ p·E_c` (+ `E_PP`).

**What did not work, and why (A2–A4).**

| step | change | outcome |
|---|---|---|
| A2 | replay and targets labelled with MACE-Field response (P, Z*, α), weights P 1000 / Z* 100 / α 100; E_PP in the forward at r_split 4.5 Å; float32; lr 1e-2 / 1e-3 / 1e-4 | labels made without E_PP, model trained with it: replay head destroyed at 1e-3 (1.1 → 172 meV/atom) |
| A3 | E_PP removed from the forward | lr 1e-3 still destroys the replay (148 meV/atom after one epoch): the lr, not the mismatch. Energy rise diagnosed as a vacancy-vs-ideal offset (formation energy drifting +0.12 eV/epoch at 1e-4) because the response anchors held ~99.7 % of the backbone gradient; the replay α labels are the dielectric head's extrapolations on metals (negative diagonals, values 40–80) |
| A4 | response labels stripped from the replay (kept only on the 16 dielectric-set frames) | target forces 46.7 meV/Å at epoch 0 (A3: 57.3); anchors on the target frames still 95 % of the gradient |
| anchor test | one epoch at lr 1e-4: response weights 0, or P 10 / Z* 1 / α 1 | 22.6 and 24.6 meV/Å (vs 46.7); at 10/1/1 the response is held to 0.4 % (Z*) / 0.2 % (α). Verdict: the weights 1000/100/100 were the brake |

Base v2 (MH-1, same E/F weights, same replay drift) had reached 13.8 meV/Å after one epoch, so the
energy weight was never the lever; with light anchors the energy falls instead of drifting.

**A5 recipe (base v7).** MACE-Field mp-dielectric head; E 10 / F 10 / S 1; P 10 / Z* 1 / α 1 on the
target frames; replay = the 9360 base v2 replay frames with E/F/S pseudolabels only; estimated
E0s + isolated atoms; float32 + cueq; batch 4; no E_PP in the forward; head-grouped batches so
the replay skips the field derivatives (epoch 23 → 9.5 min). 30 epochs.

**Phase 1 gates.**

| | base v7 (lr 1e-4) | fallback (lr 3e-4) | threshold |
|---|---|---|---|
| gate 1, neutral validation forces | 15.19 | 13.23 | ≤ 11.2 |
| gate 1, neutral validation energy | 0.73 | 0.88 | ≤ 1.5 |
| gate 2, ε∞ on the relaxed cell | 3.545 (ref 3.547) | 3.548 | within 10 % |
| gate 2, Z* Frobenius Cl / Cs / Pb | within 0.3 % | within 0.5 % | within 10 % |
| gate 3, charged shells vs base v2: total | +14.2 (47.4 vs 33.2) | +13.6 | each shell ≤ +3 |
| gate 3, 2–4 / 4–8 / 8–10 / 10–12 / 12+ Å | +31.2 / +13.9 / +6.1 / +4.1 / +1.0 | +30.0 / +13.5 / +5.2 / +2.8 / +0.2 | |
| replay drift at the end (E / F) | 5.2 / 23 | 17.3 / 72 | |

Both bases pass gate 2 and fail gates 1 and 3. The higher learning rate bought 2 meV/Å on the
neutral validation and nothing on the charged cells, at three times the replay drift. The response
problem is solved by the light anchors; the remaining gap is the MACE-Field checkpoint's transfer
to this DFT (neutral floor 13–15 vs base v2's 10.2; charged 2–4 Å shell +30 over base v2).

**Registered override (user, 2026-09-19).** Phase 2 opens on `base_v7_lr3e4/base.pt` although gates
1 and 3 failed. Consequence written before the arms ran: arm comparisons B vs A on the same base
are internally valid; absolute charged-shell numbers carry the base's excess over base v2.

**Phase 2 entry measurement (A3 requirement).** E_PP forces from the base's own dipoles on the 154
neutral validation frames at r_split 4.5 / ε∞ 4.0: 1.495 meV/Å RMS (τ 1.7 → PASS; per-frame max
4.28; species-signed means ≤ 0.08). No readout refit.

## 2. Phase 2 — head arms on base_v7_lr3e4

**Arms.** Same W6 schedule for all (60 epochs, lr 2e-3 cosine, batch 4, w_E 0.05, gap regulariser).
**A** = W6 as shipped (Mulliken `E_host`), the control for the base change. **B** = `E_cP` + `E_PP`
with the base as its own dipole provider (nested autograd every step, r_split 4.5, ε∞ 4.0, no
learned scale). **B(M)** = B with ε∞ = the base's polycrystalline 3.548 (three seeds, for Phase 3).
Registered gates: (1) paired TOST B vs A on charged held-out forces, total and 4–12 Å; B adopted
on superiority or equivalence in either; (2) C_Q profiled at 79 atoms transfers to 159 within
10 meV and the 79-atom energy after C_Q ≤ 0.6.

**Results so far** (epoch-averaged checkpoints on the fold's held-out charged frames; 262 frames
per fold, 258 at 79 atoms and 4 at 159):

| run (fold = seed) | total | 4–12 Å | 2–4 | 4–8 | 8–10 | 10–12 | 12+ | 79-atom E | C_Q transfer meV |
|---|---|---|---|---|---|---|---|---|---|
| A, Mulliken, seed 0 | 15.3 | 13.7 | 26.3 | 14.3 | 13.1 | 11.8 | 7.8 | 0.61 | 11.0 |
| A, Mulliken, seed 1 | 18.8 | 16.6 | 33.9 | 17.9 | 14.8 | 13.1 | 8.3 | 0.69 | 4.9 |
| B, polarisation, seed 0 | 29.7 | 25.2 | 57.5 | 28.1 | 20.8 | 16.3 | 11.7 | 0.52 | 13.4 |
| B, polarisation, seed 1 | 23.3 | 19.9 | 45.4 | 22.2 | 16.5 | 12.8 | 8.3 | 0.60 | 5.8 |
| W6 on base v2, median of 6 seeds | 11.6 | 10.5 | 19.3 | 11.0 | 10.0 | 9.0 | 7.1 | 0.52 | 4.3 |

Medians so far: A force_rmse 17.1, 4-12 15.2; B force_rmse 26.5, 4-12 22.5 (two seeds each).

W6 on base v2 is the same head (arm A) on the previous base, read with the same instrument; it is
the reference the Phase 1 gate 3 was written against.

**Paired reading, seeds 0 and 1** (TOST at τ = 1.7; positive mean favours B):

| metric | n | mean d (A − B) | 90 % CI | reading |
|---|---|---|---|---|
| force_rmse | 2 | -9.4 | [-40.5, +21.7] | inconclusive |
| 4-12 | 2 | -7.4 | [-33.1, +18.4] | inconclusive |
| 2-4 | 2 | -21.4 | [-83.6, +40.8] | inconclusive |
| 4-8 | 2 | -9.1 | [-38.9, +20.7] | inconclusive |
| 8-10 | 2 | -4.7 | [-23.9, +14.5] | inconclusive |
| 10-12 | 2 | -2.1 | [-17.3, +13.1] | inconclusive |
| 12-99 | 2 | -1.9 | [-14.2, +10.3] | inconclusive |

With two pairs the intervals are wide and every reading is inconclusive by construction; the point estimates favour A at every shell. Gate 2 so far: A seeds 0/1 fail on C_Q transfer (11.0) or 79-atom energy (0.61, 0.69); B seeds 0/1 pass the energy limit (0.52, 0.60) and split on transfer (13.4, 5.8).

**Reading so far.**

- **B is worse than A at every shell on the first two seeds**, by 12–15 meV/Å in total and 7–11 in the 4–12 Å shell; the excess is largest inside 4 Å (+21 to +31) and shrinks with distance (+2 beyond 12 Å).
- **It is a fit problem, not a generalisation problem.** Arm B's training force loss runs 2–3.5× above arm A's at every epoch of the schedule (epoch 58: 1.38e-5 vs 3.9e-6 for seed 0), so the polarisation coupling fits the charged training frames worse than the Mulliken term does.
- **Energies are not hurt.** Both B seeds pass the 0.6 meV/atom limit; the 79-atom energy after C_Q is 0.52–0.60 against A's 0.61–0.69.
- **The near field is where to look.** The shortfall concentrates inside the 4.5 Å kernel width, where E_cP couples the base's dipoles to a smoothed carrier field; Phase 0 validated the coupling beyond the kernel width (Born force = Z*·E_c, cosine ≥ 0.99), not inside it.
- **Against W6 on base v2** the control itself is 4–7 meV/Å worse at every shell inside 12 Å, the base's charged-cell gap (Phase 1 gate 3) carried through the head; beyond 12 Å the two controls agree within 1.

**Localisation (registered C5 bound-state precondition on the 1191 neutral-vacancy frames: level separation ≥ 0.2 eV and hole participation N_eff ≤ 4 on ≥ 95 % of frames; epoch-averaged checkpoints).**

| run | pass fraction | separation p50, meV | N_eff p50 | N_eff p90 |
|---|---|---|---|---|
| A, seed 0 | 0.981 | 439 | 1.54 | 1.94 |
| A, seed 1 | 0.034 | 340 | 14.47 | 22.61 |
| B, seed 0 | 0.000 | 32 | 34.08 | 38.95 |
| B, seed 1 | 0.000 | 45 | 10.03 | 18.33 |
| W6 on base v2, seed 0 | 0.978 | 422 | 1.35 | 1.76 |
| W6 on base v2, seed 1 | 0.706 | 230 | 1.35 | 1.82 |
| W6 on base v2, seed 2 | 0.772 | 245 | 1.33 | 1.71 |
| W6 on base v2, seed 3 | 0.929 | 322 | 1.26 | 1.59 |

- **Arm B has no bound vacancy state.** On both seeds the precondition passes on 0 % of frames: the level sits 32–45 meV from the band edge and the hole spreads over 10–34 atoms. The head trained under the polarisation coupling learned an H0 without a localised state, which is what a near-field force deficit looks like: a delocalised carrier produces no short-range force pattern, and the smooth E_cP field cannot supply it.
- **The control is seed-dependent on this base.** A seed 0 localises like W6 on base v2 (pass 0.98, 439 meV, N_eff 1.5); A seed 1 does not (pass 0.03, N_eff 14.5) although its level separation is 340 meV. W6 on base v2 passed on 71–98 % of frames with N_eff 1.3 on every seed. The base change alone weakens the bound state; the coupling change removes it.
- This is the Phase 0 attribution result seen from the head side: there the W6 carrier delocalised with cell size (N_eff 1.3 → 12 at 2879 atoms); here the B head is delocalised already at 79 atoms.

**Cost.** Arm A 40 s/epoch (local A4000), arm B 76 s/epoch (b3 RTX card): the base's nested
autograd every step, nothing cached.

**Status.** Finished: A seeds 0, 1 (local), B seeds 0, 1 (b3). Running: A seed 2 (local), B seeds 2, 3 (b3). Queued on b3: B seeds 4, 6; B(M) seeds 0–2; A seeds 3, 4, 6. The finisher writes `P2_GATES.json` and `P2_REPORT.md` when all fifteen arms
are in; the paired verdict is read on six seeds.
