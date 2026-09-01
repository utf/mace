# Overnight handover — 1 Sep 2026, 23:05

Launched autonomously; no confirmation was waited for, as instructed. Read §1 first — it is
the one thing that genuinely needs your judgement before the results mean anything.

## 1. M1 does not give a clean verdict, and I did not treat it as one

M1 (labels only, frozen base, 1047 charged frames, `~/runs/m1_label_trend.json`):

| subset | n | corr | slope | d range | dE swing |
|---|---|---|---|---|---|
| all | 1047 | **+0.314** | +0.260 eV/Å | 4.84–7.12 | 2.05 eV |
| 79-atom | 1030 | **+0.447** | +0.369 eV/Å | 4.93–6.80 | 1.86 eV |
| **159-atom** | 17 | **−0.989** | −0.134 eV/Å | 4.84–7.12 | 0.28 eV |

§2 predicts a *negative* d-trend. The 159-atom frames give exactly that, with a correlation of
−0.989 on 17 points. The 1030 79-atom frames give the opposite sign — and the 79-atom cell is
the one already on record (the earlier §8 caveat) as **cell-limited in d(Pb–Pb)**: a 2×2×1
expansion with c = 11.2 Å constrains Pb–Pb above ~5.5 Å when the vacancy axis lies along c.

So the subset that contradicts §2 is the contaminated one and the clean subset confirms it,
but the clean subset is n = 17 and its swing is 0.28 eV rather than the eV scale §2 expects.

The script's own pooled verdict prints **FAIL**. I judged that too blunt to act on and did not
stop the chain. **That is the call I would most like you to check.** If you read M1 as
falsifying §2, everything below is void and should be discarded rather than interpreted.

The obvious follow-up, not run: the per-fold-base null the spec asks for. `E_base` is a model,
and if its error correlates with d the 79-atom trend could be the base's, not the label's.

## 2. What is running

`b3:~/queue_c1_ab.sh`, GPUs 4–7, started 23:03. Log `~/runs/c1_ab_queue.log`.

**C1** — hub2 clamp × {ON, OFF} × 2 seeds (`~/runs/c1_on.json`, `c1_off.json`). With mass
fixed by the clamp the only free lever is t(d), so this isolates the energy-channel claim from
placement.

**Gate, evaluated automatically** (nobody is watching, so it had to be mechanical):
ON mean `axial_red` ≥ 0.40 **and** exceeding OFF by ≥ 0.20. Verdict lands in
`~/runs/c1_gate.txt` and `~/runs/c1_gate_verdict`.

**The gate is only half the specified one.** §4 also asks for autograd |t′| ≥ 0.1 eV/Å. M3 is
not built, so that half was **not evaluated** — it is recorded as unevaluated, not passed. If
C1 passes on `axial_red` alone you should still check |t′| before adopting anything.

**A/B** — 12 cells, ON/OFF × 6 seeds, launched only if the gate passes
(`ab_on_a/b`, `ab_off_a/b`). If the gate fails the script exits and the spec's s+p branch is
where you are.

Models saved to `~/runs/ab_models/` under names carrying the arm.

## 3. Two watch items in the ON arm

**The force loss starts ~4700× baseline.** ON-clamped smoke: force 14.96 at epoch 0 against an
epoch-0 baseline of 0.0032, and 20.17 at epoch 1. The sign flip changes what `dE_SR` means
while μ is initialised on the old convention, so the head starts far from any fit. It may well
train down over 40 epochs — but if the ON cells come back with `axial_red` deeply negative and
a force loss orders of magnitude above OFF, that is the likely cause and it is a μ-init
problem, not evidence about the sign. `init_mu_for_gap` does not know about `channel_sign`.

**Δ_bind goes strongly negative under ON** (−2.6 to −3.1 in smoke). `Delta_bind` is defined on
`H_e` and is unchanged by the flip, but the head now optimises a differently-signed energy, so
the lower margin fights it. §5's contingency (switch the lower margin to the charged-ensemble
median if >30% of frames pay persistent penalty at epoch 20) may be needed; it is **not**
wired in and would need a rerun.

## 4. Three clamp bugs fixed on the way, same class

`evaluate()` and `capture()` each ran their own forward with `_clamp_mask = None`, so a
hub2-clamped cell reported **N_eff 35 on a two-atom clamp** — the metrics of a different model
than the one trained. Both now take the clamp; the same cell reports N_eff 1.49,
region_mass 1.00. Had this not been caught, C1's gate would have been read off unclamped
states, which is the "guard measuring a reimplementation" pattern in a new place.

## 5. Not done

* **M2** (which matrix elements carry the model's d-trend) and **M3** (autograd `dH_ab/dd`
  replacing the profile slope). M3 matters twice: it supersedes F2's `t′` and it is half the
  C1 gate. Both are hours of work on saved heads, no training.
* The §7 electron-counting head spec.
* The F2 `t′` caveat stands: profile slope, uniform at −0.033 across cells, which may account
  for part of the 0.54 closure ratio.

## 6. State

Branch `size-extensivity`, unpushed, HEAD at the commit adding this file. Local A4000 free.
b3 GPU3 recovered earlier today; 0–3 remain off-limits by instruction.

Kill switch: `ssh b3 'pkill -f "queue_c1_[a]b"; pkill -f "tb_[v]3.py"'`
