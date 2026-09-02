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

## The three arms

The third arm — `Z` pinned at the nominal charges — was added mid-stage after the ON arm's
`Z` ran away. It is a diagnostic control, not an architecture change: same code, same seeds,
`Z` simply not in the optimiser.

| arm | n | axial_red | rmse_all | rmse_nbhd | N_eff | null ratio | corr(λ,d) | φ gap |
|---|---|---|---|---|---|---|---|---|
| OFF (no Madelung) | 6 | +0.452 ± 0.189 | 35.5 ± 7.0 | 65.6 ± 19.4 | **19.3** | **0.265** | +0.020 | +3.8 |
| ON, learned `Z` | 6 | +0.514 ± 0.240 | 32.5 ± 7.2 | 62.8 ± 19.3 | 43.0 | 0.803 | **+0.422** | +2.0 |
| **ON, nominal `Z`** | 6 | **+0.627 ± 0.184** | **30.4 ± 6.6** | 48.6–50.5 | 49.8 | 0.790 | +0.270 | **+1.2** |

**The φ-binned gate passes, and passes best with `Z` fixed.** Error binned by `|phi_LR|` at
defect-near atoms: the gap the Madelung term is supposed to close goes **3.8 → 2.0 → 1.2**
meV/Å across OFF → learned `Z` → nominal `Z`, and the nominal-`Z` arm is lowest in both bins
(29.0 / 30.3). This is the gate as stated, read against the OFF control rather than against
the archived numbers.

**Learning `Z` buys nothing and costs two seeds.** Nominal `Z` beats learned `Z` on
axial_red (+0.627 vs +0.514), on force error (30.4 vs 32.5) and on the φ gate (1.2 vs 2.0),
with a *smaller* seed spread (±0.184 vs ±0.240). Every headline number is better with the
parameter removed.

## The result we did not expect: the λ–d anomaly changes sign

| cohort | mean `corr(λ, d_hub)` | sign |
|---|---|---|
| archived (T-B loss, no Madelung) | **−0.422** | negative in 17/18 |
| Stage-1 **OFF** | +0.020 | ~flat |
| Stage-1 ON, learned `Z` | **+0.422** | positive in 5/6 |
| Stage-1 **ON, nominal `Z`** | **+0.270** | positive in 5/6 |

**The physical trend comes from the Madelung contrast at nominal charges, not from fitted
`Z`.** That is the stronger version of the claim: it does not depend on an optimiser finding
the right species charges, only on the term being present with the formal ones. Learned `Z`
adds a little more (+0.42 against +0.27) and pays for it with the two diverged seeds.

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

**The diagnostic settles it: the learned scale buys nothing.** With `Z` pinned at nominal,
every headline number improves and the seed spread narrows (table above). The two-seed
divergence disappears because there is nothing left to diverge.

**Proposed amendment, for confirmation rather than silent adoption** (`BUILD_CHOICES.md` item
11): either fix `Z` at the formal charges, or make it bounded in Edit 3's own idiom —
`Z[s] = Z_nominal[s] + δ·tanh(·)` with a small δ. The second is more consistent with the
spec's philosophy of bounded corrections over physical scales; the first is what the data
currently supports. Stage 2 runs **both** the learnable and the nominal arms so the answer
is complete either way, and the plan's learnable `Z` remains the default until overruled.

## One reading for Stage 2

The pristine spectrum check reports **split fraction ≈ 0.76 against an even-spacing reference
of 0.20** — a superatom-shaped spectrum on a defect-free cell. That is the V3 head, so it is
the *baseline* for Stage 2's gate rather than a Stage-1 failure. Edit 3 exists to remove
exactly this, and now has a number to remove it against.

## Also worth flagging

ON is **more delocalised** than OFF (N_eff 43.0 vs 19.3; null ratio 0.80 vs 0.27). This is
the third time in this project that a change improving the force fit has also delocalised the
carrier. It is not yet an explanation of anything, but it has stopped being a coincidence.
