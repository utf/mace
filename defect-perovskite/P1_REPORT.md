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

_Filled from `P1_GATES.json` when the chain lands (conversion by `p1_make_base.py`, checked
exact; gates by `p1_gates.py`)._
