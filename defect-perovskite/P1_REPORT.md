# P1_REPORT — Plan P Phase 1: base v5 (MACE-Field fine-tuned on the neutral defect data)

Run `~/runs/dscc/p1/base_v5/` (b3 GPU 5, 2026-09-17 04:04–08:34, 30 epochs, 8.6 min/epoch).
Recipe as registered (tracker, "Plan P Phase 1 — REGISTRATION" and A1): MACE-Field checkpoint,
`pt_head` foundation head, `Default` head on `dataset_e0` (1616 neutral train / 154 valid),
pseudolabelled replay (fps 1000, 16 Cs/Pb/Cl + 984 random; E, F, stress, P, Z*, α from the
checkpoint), `universal_field` loss E 10 / F 10 / stress 1 / P 1000 / Z* 100 / α 100, lr 1e-4,
EMA 0.9999, clip 1.0, weight decay 0, batch 4 (registered deviation from 8: memory), float32,
cueq, E_PP in the forward at r_split 4.5 Å, ε∞ 4.0. Units: meV/atom, meV/Å per component.

## Training

| epoch | Default E | Default F | Default stress (meV/Å³) | pt_head E | pt_head F | pt_head Z* (e) | pt_head α (ε0) |
|---|---|---|---|---|---|---|---|
| initial | 20.86 | 66.97 | 0.49 | 0.03 | 0.17 | 0.0005 | 0.004 |
| 0 | 4.90 | 37.65 | 1.02 | 8.09 | 34.10 | 0.0225 | 0.093 |
| 5 | 3.92 | 25.30 | 0.73 | | | | |
| 10 | 3.58 | 22.17 | 0.74 | | | | |
| 15 | 3.34 | 20.49 | 0.55 | | | | |
| 20 | 3.14 | 19.29 | 0.59 | | | | |
| 25 | 3.04 | 18.44 | 0.53 | | | | |
| 29 | **3.00** | **17.94** | 0.51 | 15.11 | 33.67 | 0.036 | 0.452 |

The neutral fit is still improving at epoch 30 (−0.3 meV/Å per epoch). The replay head's
response labels (its own predictions at the start) drift to Z* 0.036 e and α 0.45 ε0 RMSE.

## Gates

Base `base_v5_base.pt` (single head, Cl/Cs/Pb; conversion reproduces the multi-head model to
1e-13). All three gates fail.

| gate | value | limit | met |
|---|---|---|---|
| 1 neutral validation forces | 17.94 meV/Å | ≤ 11.2 | no |
| 1 neutral validation energies | 3.00 meV/atom (offset +1.41) | ≤ 1.5 | no |
| 2 ε∞ polycrystalline (relaxed cell) | 2.87 vs 3.55 (19 %) | 10 % | no |
| 2 Z* Frobenius Cl / Cs / Pb | 2.51 / 1.57 / 4.52 vs 3.52 / 2.24 / 6.21 (−29 / −30 / −27 %) | 10 % | no |

Gate 3 — base-alone force RMSE on the 944 charged frames by shell (meV/Å per component):

| shell | base v5 | base v2 | excess |
|---|---|---|---|
| total | 41.7 | 33.2 | +8.5 |
| 2-4 | 90.4 | 74.9 | +15.5 |
| 4-8 | 39.4 | 30.4 | +9.0 |
| 8-10 | 20.9 | 15.3 | +5.6 |
| 10-12 | 14.6 | 8.9 | +5.7 |
| 12+ | 10.2 | 8.0 | +2.2 |

Limit: no shell worse than base v2 by more than 3 meV/Å — not met (every shell but 12+).

## Reading

- The response did not hold under the fine-tune (Z* −27…−30 %, ε∞ −19 %; the replay head's α
  RMSE went 0.004 → 0.45 ε0), so the plan's one registered deviation applies: a rerun at
  lr 5e-5 (`base_v5_lr5`, launched 08:55 on b3 GPU 5), everything else identical.
- Gates 1 and 3 are a capacity/data statement, not a learning-rate one: the 128-channel MH-0
  lineage backbone reaches 17.9 meV/Å where the 512-channel MH-1 base v2 reached 3.9 on the same
  split, with the fit still falling at epoch 30, and it is 5–15 meV/Å worse than base v2 on
  every charged shell. The lr halving is not expected to change that; what would (more epochs,
  a replay with more Cs/Pb/Cl response frames than 16 of 1000, response weights, a larger
  MACE-Field backbone) is a recipe decision outside the registered plan.
- Phase 2 stays blocked on this base (`P2_STATUS = BLOCKED`).

