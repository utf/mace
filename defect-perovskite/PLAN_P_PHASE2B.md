# Plan P — Phase 2b: range-separated carrier–host coupling

Arm B failed because `E_cP` applies fixed Born-scale forces inside the kernel width, where the
carrier–host interaction is electronic structure (H0's business), and the head escapes them by
delocalising the carrier. Phase 2b splits the coupling by ownership: the head owns the near
field, the provider's dipoles own the far field, with one smooth switch on the carrier-site–host
pair distance. Two candidate near fields are trained; the data decide between them. No auxiliary
losses; no learned coefficient on the far-field term; `E_PP`, `E_M`, `H0`, the fill, the loss
strata and the six-seed paired protocol are unchanged.

## 0. Immediate

- Stop arms B and B(M) (all queued and running seeds). Their remaining seeds measure a flawed
  coupling.
- Let arm A finish all six seeds; it is the control every Phase 2b arm is read against.

## 1. The switched coupling (implement once)

**Switch.** `S(r) = 0` for `r ≤ 6.0 Å`, `1` for `r ≥ 8.0 Å`, a quintic (C² continuous) ramp
between; `r_c = 8.0 Å`, `Δ = 2.0 Å`. Evaluated per pair (host atom `a`, carrier site `c`) on the
full neighbour list with image shifts (not minimum image); pairs beyond 8 Å are not enumerated.

**Far field (both arms).**

```
E_far = - Σ_a p[a] · ( E_c^periodic(R_a) - Σ_{c: r_ac < 8} dq[c] (1 - S(r_ac)) E_direct(r_ac) )
```

`E_c^periodic` is the existing smoothed periodic carrier field (r_g 1.0, r_split 4.5, ε∞ as
registered); `E_direct` is the direct (non-periodic) field of one carrier site's smoothed cloud
with the same widths, so the subtraction removes exactly the near-field part of the periodic
kernel. Beyond 8 Å the coupling is identical to arm B's `E_cP`; coefficient 1, nothing learned.

**Near field, arm B_s (Mulliken).**

```
E_near = Σ_{a,c: r_ac < 8} dq[c] (1 - S(r_ac)) s q0[a] γ_2.5(r_ac)
```

`γ_2.5` is the direct smoothed Coulomb kernel of W6's `E_host` (r_split 2.5, r_g, /ε∞), `q0` the
reference-fill Mulliken pattern, `s` the learned global scale, all exactly as in W6. This differs
from W6's `E_host` only by lacking its (weak) far-field Ewald part; that difference is reported
once on a training frame.

**Near field, arm B_0.** `E_near = 0`. Inside 8 Å the carrier–host interaction is H0 alone.

**Forces.** Autograd through `S`, `E_direct`, `γ_2.5`, `dq` (existing Fréchet contraction) and
`p` (existing nested pass through the base). Cost class unchanged (~76 s/epoch).

**Tests (all must pass before any training).**
- Neutral null: `E = E_base + E_PP` bit-exact at `S_ref`.
- `S ≡ 1` (set r_c = 0): B_s and B_0 reproduce arm B's energy and forces to 1e-9.
- `S ≡ 0` on B_s: `E_near` equals W6's `E_host` minus its far-field part; report the difference.
- Central-difference forces to 1e-4 eV/Å on a charged 79-atom frame and on a pair straddling
  6–8 Å (continuity across the ramp); Σdq = Q; batched = per-graph; translation invariance.

## 2. Kernel validation (no training, half a day)

On the embedded 2879-atom structure from P0, two rows each for the far-field force:
- unit Gaussian point charge at the vacancy site (kernel property): switched far-field force on
  each far ion against `Z*·E_c`, ratio by shell. Expect 0 inside 6 Å, ramp 6–8 Å, then the P0
  ratios (0.78 → 0.97) beyond 8 Å.
- the head's own `dq` at the sharpened fill (σ_s × 0.25, as in P0): same table.

On the 2560 training frames: RMS force per component of `E_far`, of `E_near` (B_s), and of their
sum, by family (pristine, neutral 79/159, charged 79/159). Expect `E_far` at the meV/Å level
(the far field is periodicity-suppressed in these cells) and `E_near` at W6's `E_host` level.
Record both; no threshold, these are the numbers the arms are interpreted with.

## 3. Brillouin-zone form of the gap regulariser (both arms and A's remaining seeds)

For a pristine periodic cell, build `H0(k) = Σ_R H0(0,R) e^{ik·R}` from the head's neighbour
graph with shifts and diagonalise on the Γ-centred 3×3×3 mesh of the 80-atom cell (27 k-points,
320×320 Hermitian, negligible cost). Penalty, added to the existing Γ gap term with the same
weight:

```
L_BZ = w_gap [ Σ_k relu(E_CBM(Γ) - E_CBM(k))² + Σ_k relu(E_VBM(k) - E_VBM(Γ))² ]
```

Registered host facts: gap 2.40 eV, direct at Γ. Evaluate on the label pristine cell, the
base-relaxed pristine cell, and four fixed thermal pristine snapshots, every step. Test: at Γ
alone it reproduces the existing regulariser bit-exact. Readout per checkpoint:
`min_k [E_CBM(k) − E_CBM(Γ)]` and `max_k [E_VBM(k) − E_VBM(Γ)]` on the label cell.

## 4. Arms (base `base_v7_lr3e4`, W6 schedule: 60 epochs, lr 2e-3 cosine, batch 4, w_E 0.05)

| arm | near field (< 8 Å) | far field (> 8 Å) | seeds |
|---|---|---|---|
| A (running) | W6 `E_host`, all ranges | — | 6 |
| B_s | Mulliken, learned s | provider dipoles, coeff. 1 | 6 |
| B_0 | none (H0 only) | provider dipoles, coeff. 1 | 6 |
| B_x(M) | as the adopted B_x with ε∞ = 3.548 | | 3, after the decision |

Same folds and seeds as A. `E_PP` on in all B arms.

## 5. Gates and decisions

**Forces (adoption).** Paired TOST at τ = 1.7 meV/Å on held-out charged forces, total and
4–12 Å shell.
- B_x is adopted over A on superiority or equivalence in either metric (the far field is
  invisible at 79/159 atoms; equivalence is the expected outcome and is sufficient).
- Between B_s and B_0: B_0 if it is within τ of B_s on both metrics (then the Mulliken term is
  dropped from the model); otherwise B_s.
- If both B_s and B_0 are inferior to A: set r_c = 10 Å (ramp 8–10), rerun B_s only, six seeds,
  once. If still inferior, stop and report.

**Energies.** `C_Q` profiled at 79 atoms transfers to 159 within 10 meV; 79-atom energy after
`C_Q` ≤ 0.6 meV/atom.

**Localisation (reported for every arm; gate for Phase 3 entry).**
- 79-atom C5 readout as in P2_PRELIM (pass fraction, separation, N_eff).
- Large-N readout on the embedded 3×3×3 structure at the registered fill: separation ≥ 0.15 eV,
  N_eff ≤ 4, Σdq within 5 Å ≥ 0.8, on the adopted arm's seed-median checkpoint. If it fails,
  `w_gap` × 3 once and retrain that arm; no other change.
- BZ readouts from §3 per checkpoint.

## 6. Report (`P2B_REPORT.md`)

Tables: §2 kernel validation; arm results by shell with A and W6-on-v2 as references; paired
tables B_s vs A, B_0 vs A, B_0 vs B_s; energies; localisation (79-atom and large-N) per seed;
BZ readouts; `E_near`/`E_far` energy and force decomposition on held-out charged frames; cost.
Decision statements written against the rules in §5.

## What not to do

- No auxiliary localisation or participation loss; no learned coefficient on `E_far`; no change
  to `E_PP`, its r_split, or `E_M`.
- No r_c scan beyond the single fallback in §5; no extra arms.
- Do not reuse arm B or B(M) checkpoints for anything but the `S ≡ 1` test.
