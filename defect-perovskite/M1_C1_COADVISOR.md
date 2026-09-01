# M1 and the queued C1 → A/B — overnight state, 1 Sep 2026

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** M1 — the first gate, labels only, no trained head — does not give a clean verdict.
It splits by cell size, and the split runs along a known confound: the 17 large-cell frames
confirm the predicted negative d-trend at corr −0.989, while the 1030 small-cell frames give
the opposite sign and are the ones already on record as cell-limited in d(Pb–Pb). We judged
this as not falsifying §2 and let the chain proceed, but the judgement is contestable and is
the first thing to check. C1 is running; the A/B is queued behind its gate and will launch
only if it passes.

## M1 — the label's energy channel

`dE = E+_DFT − E_base` against d(Pb–Pb), all charged frames, frozen Stage-A base.

| subset | n | corr | slope | d range (Å) | dE swing |
|---|---|---|---|---|---|
| all | 1047 | +0.314 | +0.260 eV/Å | 4.84–7.12 | 2.05 eV |
| 79-atom | 1030 | **+0.447** | +0.369 eV/Å | 4.93–6.80 | 1.86 eV |
| **159-atom** | 17 | **−0.989** | −0.134 eV/Å | 4.84–7.12 | 0.28 eV |

§2 requires a negative trend. The large cells give it, and give it about as cleanly as 17
points can. The small cells give the reverse.

**Why we do not read the pooled number as the answer.** The 80/79-atom cell is a 2×2×1
orthorhombic expansion with c = 11.2 Å, which constrains Pb–Pb separations above ~5.5 Å when
the vacancy axis lies along c — the caveat already recorded beside the R_DFT result. So the
1030 frames driving the pooled correlation have a d-distribution that is partly an artefact of
the box, and the 17 frames without that constraint are the ones that can speak to a d-trend.
Our script's own pooled verdict prints FAIL; we judged it too blunt to act on and did not stop
the chain.

**Against our own reading**, and worth weighing: n = 17, and the large-cell swing is 0.28 eV
rather than the eV scale §2 expects from a level that moves > 1 eV with this coordinate. A
correlation of −0.989 on a narrow swing is consistent with a real but much weaker dependence
than the mechanism needs.

**The control we did not run.** The spec asks for per-fold bases as a null. `E_base` is a
model; if its error correlates with d, the small-cell trend could be the base's rather than the
label's. That is the cheapest way to settle which subset is telling the truth and we would run
it first.

## What is queued

**C1** — hub2 clamp × {ON, OFF} × 2 seeds, frozen base, V3 + response channel, 40 epochs.
Clamping fixes hub mass by construction, so the energy-channel claim is tested in isolation
from placement — which is the separation F1/F2 showed we need, since placement is where the
head has been evading.

**Gate, evaluated mechanically** because it fires unattended: ON mean `axial_red` ≥ 0.40 and
exceeding OFF by ≥ 0.20. The A/B's 12 cells launch only on a pass; on a fail the script stops
and the s+p branch is where things stand.

**The gate is half the one specified.** §4 also asks for autograd |t′| ≥ 0.1 eV/Å at the pair.
That machinery is M3, which is not built, so the |t′| half was **not evaluated** and is
recorded as unevaluated rather than passed. A C1 pass on `axial_red` alone should not be read
as clearing §4.

## Two things that could make the ON arm unreadable

Both are ours, not the physics, and both are flagged so a bad ON result is not misread.

**The ON arm starts ~4700× off.** Force loss 14.96 at epoch 0 against a 0.0032 baseline, 20.17
at epoch 1. `init_mu_for_gap` sets μ from the unsigned energy and does not know about
`channel_sign`, so flipping the sign leaves μ initialised for the other convention. It may
train down over 40 epochs. If ON returns deeply negative `axial_red` with a force loss orders
of magnitude above OFF, that is the likely cause and it is a μ-initialisation bug, not evidence
about the sign.

**Δ_bind is driven strongly negative under ON** (−2.6 to −3.1 in smoke). `Delta_bind` is
defined on `H_e` and is untouched by the flip, but the head now optimises a differently-signed
energy and the lower margin fights it. §5's contingency — lower margin on the charged-ensemble
median — is **not** wired in and would need a rerun of the affected arm.

## Fixed on the way, and worth recording as a class

Three clamp paths ran their own forward with `_clamp_mask = None`: the training loop applied
the clamp, but `evaluate()` and `capture()` did not. A hub2-clamped cell therefore reported
**N_eff 35 on a two-atom clamp** — the metrics of a different model than the one trained. Now
1.49 with region mass 1.00. This is the "guard measuring a reimplementation" pattern appearing
in a fourth place, and C1's entire gate would have been read off unclamped states.

## Not done

* **M2** (which matrix elements carry the model's d-trend) and **M3** (autograd `dH_ab/dd`).
  M3 matters twice over: it supersedes F2's profile-slope `t′` and it is half the C1 gate.
  Both are hours on saved heads with no training.
* The per-fold-base null for M1, above.
* The §7 electron-counting head spec.

## What we are asking

1. Whether M1 falsifies §2. If it does, C1 and the A/B are void rather than interpretable and
   should be discarded unread — which is why this is the first question.
2. Whether a C1 pass on `axial_red` alone is acceptable, or whether M3 must be built and the
   |t′| half evaluated before anything is adopted.
3. If C1 passes and the A/B runs, whether the μ-initialisation issue should be fixed and the
   ON arm rerun before its numbers are read at all. Our inclination is yes — an arm that starts
   4700× from any fit is not obviously testing the convention.
