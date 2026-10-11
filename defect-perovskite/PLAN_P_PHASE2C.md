# Plan P — Phase 2c: the carrier level's depth at large N

*User's plan, received 2026-10-11, verbatim.*

The large-N localisation gate fails on every base and arm. The BZ readouts show the pristine
label cell's bands are direct at Γ in every checkpoint, so the k-folding explanation is retired
and the BZ regulariser stays off. Two remaining causes, tested in order: (1) the embedded
structures are tiled from the base-relaxed pristine cell (tilt 156° against the label cell's
167°, 0.25 Å rms apart), where the head's readouts extrapolate; (2) the level's depth is one
scalar the data do not identify, and on the A2B bases it is shallow. Step 0 tests (1) with no
training. Step 1 addresses (2) with one arm. Base: `base_v8_w01`. No auxiliary losses.

## 0. Geometry diagnostics (no training; hours)

On the seed-median B_s checkpoint from w01 (seed 2) and the base-v7 B_s seed median. "Label
cell" = the static pristine cell the Γ-gap term acts on; "relaxed cell" = the base-relaxed
pristine cell the ladders were tiled from.

0.1 Head eigenvalues (dense) on the pristine 2×2×2 tilings of both cells (640 atoms) and, if
memory allows, the 3×3×3 tilings (2160): VBM, CBM, gap, side by side. This is the band edge each
embedding sees, folding included.

0.2 Level separation, N_eff and Σdq within 5 Å of the vacancy for the +1 core embedded at 639
atoms in (a) the relaxed tiling (the existing structure) and (b) the label tiling, the core
re-relaxed there with the far field fixed (R_core 8 Å, the frozen-core-ladder protocol).

0.3 Separation and N_eff on the 79-atom +1 training frames against the Pb–Pb distance across
the vacancy, with the relaxed core's Pb–Pb marked.

0.4 Sign-resolved dq by shell on one 79-atom +1 frame (positive and negative parts): N_eff 1.4
with Σdq(< 5 Å) 0.47 cannot both describe one compact hole.

**Decision 0.** Candidate (1) is confirmed if 0.1 shows the CBM in the relaxed tiling ≥ 0.1 eV
below the label tiling's, or 0.2 shows the separation in the label tiling ≥ 3× that in the
relaxed tiling. Then: all localisation and frozen-core work embeds in the label tiling from
here on; the large-N gate is re-read on the existing B_s checkpoints there; the 0 K relaxation
ladder is recorded as sitting at the host geometry furthest from the training distribution (the
application is 300 K dynamics, where the host is near the label geometry). If it passes, Step 1
still runs but no longer blocks Phase 3. If 0.1 and 0.2 are clean, candidate (2) stands alone.

## 1. Arm B_w: the near-field host potential in H0

**Change.** B_s carries the near field as an explicit energy,
`E_near = Σ_{a,c} dq[c] (1 − S(r_ac)) s q0[a] γ_2.5(r_ac)`. B_w removes that energy and puts the
same potential on the diagonal of H0:

```
W_i = Σ_j (1 − S(r_ij)) s q0[j] γ_2.5(r_ij)      S, s, q0, γ_2.5 exactly as in B_s
H   = H0 − W_i on every orbital of atom i        both fillings (N_S, N_ref) from H
```

One learned scale `s` (initialised at B_s's value, sign free). `q0` is the reference-fill
Mulliken pattern of the *unshifted* H0, so there is no self-consistency loop; the cost is a
second eigendecomposition per frame (≈ 10 ms hidden on CPU at 79 atoms; accepted for the
experiment, and a species-mean `q_Z` variant is the production simplification if B_w succeeds,
not run now). `E_far` (provider dipoles beyond r_c) and `E_PP` are unchanged. The first-order
band response to `W` equals the removed near-field energy, so nothing is counted twice.

**Forces.** Hellmann–Feynman through the band term: `Σ_i dq_i ∂W_i/∂R`, with `∂q0/∂R` through
the existing Fréchet contraction of the unshifted reference fill; `E_far`, `E_PP` as before.

**Tests before training.** Neutral null bit-exact (ΔF_band vanishes at N_S = N_ref for any H);
`s = 0` reproduces B_0 to 1e-9; the identity `∂ΔF_band/∂s` (central difference in s) `=
Σ_c dq[c] W_c / s` with dq from H, to 1e-6 relative; central-difference forces to 1e-4 eV/Å on
a charged 79-atom frame; Σdq = Q; gauge shift −aQ; batched = per-graph.

**Training.** Three seeds (0, 2, 6; folds as registered), W6 schedule, w01 base, the P2b
settings otherwise. Per epoch: held-out force RMSE; separation and N_eff on the 79-atom +1
frames; separation, N_eff and Σdq(< 5 Å) on the 639-atom label-tiled embedding from 0.2(b)
(one eigenproblem per epoch; readout only, never a loss); `s`; `dEp95`; readout saturation.

**Gates.**
- Forces: paired mean difference against B_s on the same seeds (n = 3, point estimate, no
  TOST at this n), total and 4–12 Å. B_w kept if within +τ or better on either; Step 1 fails
  if worse by more than τ on both.
- Separation at the +1 training geometry: reported per seed, not gated (no PBE depth exists).
- Large-N gate on the seed-median checkpoint in the label tiling: separation ≥ 0.15 eV,
  N_eff ≤ 4, Σdq(< 5 Å) ≥ 0.8. Pass opens Phase 3.

## 2. Reporting and decision

`P2C_REPORT.md`: the §0 tables and Decision 0; B_w per-seed forces, shells and energies against
B_s; separation curves per epoch; `s` per seed; the large-N table for B_w and for B_s re-read
in the label tiling; cost per epoch.

Outcomes: (a) B_s passes large-N in the label tiling: candidate (1) was the cause; B_s is the
model, B_w is recorded. (b) B_w passes where B_s does not: B_w is the model; the depth is set
by the host potential. (c) neither passes: the depth is not identifiable from the data or the
host's electrostatics; report that large-cell charged dynamics for this defect needs the +1
state as the reference, which is a different construction, and stop.

## Not permitted

The BZ k-mesh regulariser; any loss on N_eff, separation or participation; a state-pinned
fill; new Γ-gap cells (the registered 2.40 eV is the label cell's value; the gap at the
relaxed tilt is not registered); a fourth arm; changes to `E_far`, `E_PP`, r_c, r_split or ε∞.
