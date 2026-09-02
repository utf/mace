# Stage 1 — Madelung on-site (Edit 1), response channel deleted (Edit 2)

Head-only retrain, 48 charged frames, 40 epochs, force-only loss. Six seeds per arm,
differing **only** in whether the Madelung term is present. b3 GPUs 4–7.

The control is the OFF arm, not the archived cohort: those models trained under T-B's edge
and gap losses, so reading against them would conflate the edit with the loss change.

---

## Gates

| gate | result |
|---|---|
| A1 toy tables (rock salt, perovskite band order, G=0 convention, eval-mode forces) | **pass**, 13 tests |
| A2 pinned continuum (learned parts bit-identical, total = analytic −Δφ/ε∞) | **pass**, 6 tests |
| D-2 rerun: polarisation-binned error shrinking | **pass** — see below |
| force parity not degraded | **pass** — ON is better, within seed spread |
| no double counting (E_resp gone in the same commit) | **pass**, `85d195b` |

## The arms

| arm | n | axial_red | rmse_all | rmse_nbhd | N_eff | null ratio |
|---|---|---|---|---|---|---|
| **ON** | 6 | **+0.514** ± 0.240 | **32.5** ± 7.2 | 62.8 ± 19.3 | 43.0 | 0.803 |
| OFF | 6 | +0.452 ± 0.189 | 35.5 ± 7.0 | 65.6 ± 19.4 | **19.3** | **0.265** |

**The φ-binned gate passes.** Error binned by `|phi_LR|` at defect-near atoms, low vs high:

| arm | low \|φ\| | high \|φ\| | gap |
|---|---|---|---|
| ON | 30.6 | 32.6 | **+2.0** |
| OFF | 33.2 | 37.0 | +3.8 |

The gap the Madelung term is supposed to close does close — 3.8 → 2.0 meV/Å — and ON is
lower in *both* bins. This is the gate as stated, read ON vs OFF.

## The result we did not expect: the λ–d anomaly changes sign

| cohort | mean `corr(λ, d_hub)` | sign |
|---|---|---|
| archived (T-B loss, no Madelung) | **−0.422** | negative in 17/18 |
| Stage-1 **OFF** | +0.020 | ~flat |
| Stage-1 **ON** | **+0.422** | positive in 5/6 |

Positive is the physical direction: the level rises as the pair separates, so the state is
deepest when the pair dimerises. That is what R-A predicted and never found.

Two things stop this being a clean "Edit 1 fixed the anomaly". The archived cohort's strong
negative is **not** reproduced by the OFF arm, so the loss change alone already removed it —
the anomaly was never purely architectural. And the ON-vs-OFF difference, which *is* clean
(+0.422 against +0.020, one edit apart), says the Madelung term drives the trend positive
rather than merely removing a negative. Both statements are worth keeping separate.

The ledger item is therefore **not closed but substantially changed**: D-2 refuted resonance
as the explanation this morning, and Stage 1 now shows the anomaly does not survive either
the loss change or the Madelung term.

## The problem: Z's scale is not identified, and it costs two seeds

Per-seed, ON arm:

| seed | Z (Cl, Cs, Pb) | axial_red | rmse_all | N_eff |
|---|---|---|---|---|
| 1 | (−2.57, +2.33, +5.38) | +0.170 | 42.8 | 44.0 |
| 2 | (−2.59, +2.35, +5.42) | +0.184 | 41.6 | 43.8 |
| 3 | (−0.31, +0.29, +0.65) | +0.670 | 31.7 | 61.8 |
| 4 | (−1.59, +2.34, +2.42) | **+0.722** | **25.6** | 36.0 |
| 5 | (−1.54, +1.94, +2.69) | +0.634 | 27.5 | 30.2 |
| 6 | (−0.90, +1.08, +1.61) | +0.701 | 25.6 | 42.3 |

Neutrality holds to 1 part in 10⁷ on every seed — the projection works exactly.

**The arm is bimodal, and the split is by |Z|.** Seeds 1 and 2 inflated Z to roughly 2.5×
nominal and are the two worst-fitting seeds by a wide margin. The other four kept |Z| near
or below nominal and average **axial_red +0.68 and rmse_all 27.6** — decisively better than
OFF's +0.45 and 35.5. The arm mean of +0.514 is an artefact of averaging two failures with
four successes.

This is not gauge drift, it is an optimisation pathology, and it has a clear cause: the
smooth part of `phi_LR` is dominated by a per-species constant that the learned on-site MLP
can also produce, so only the small defect-induced deviation constrains `|Z|`. The plan's
claim that "the real-space part is absorbed by the learned local term, which is what makes Z
identifiable" is right in principle and too weak in practice.

A diagnostic arm with `Z` pinned at nominal is running; it says whether the learned scale
buys anything at all.

## One reading for Stage 2

The pristine spectrum check reports **split fraction ≈ 0.76 against an even-spacing reference
of 0.20** — a superatom-shaped spectrum on a defect-free cell. That is the V3 head, so it is
the *baseline* for Stage 2's gate rather than a Stage-1 failure. Edit 3 exists to remove
exactly this, and now has a number to remove it against.

## Also worth flagging

ON is **more delocalised** than OFF (N_eff 43.0 vs 19.3; null ratio 0.80 vs 0.27). This is
the third time in this project that a change improving the force fit has also delocalised the
carrier. It is not yet an explanation of anything, but it has stopped being a coincidence.
