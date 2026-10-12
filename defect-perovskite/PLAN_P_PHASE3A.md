# Plan P — Phase 3a: bw5 at fraction 0.2, then the far-field test

Model of record: **bw5** (arm B_w on `base_v8_w01`: near-field host potential in H0, flank bound
0.1 × the element's s–p splitting, smoothed gap edge σ = 0.0259 eV, gap weight 0.1). Working
checkpoint: seed 0 `model_avg.pt` (the seed median, 11.79 meV/Å). One variant run (§1) may replace
it; the far-field programme (§2) runs on whichever stands. No other training. Run directory
`~/runs/dscc/p3a/`; report `P3A_REPORT.md`.

## 0. Registrations (write to the tracker before any §2 result is read)

- **Large-N gate, re-registered on compactness.** On the +1 core embedded at 639 and 2159 atoms:
  Σdq(< 12 Å) ≥ 0.9 at both sizes; N_eff(2159)/N_eff(639) ≤ 1.2; far-field Born-force ratio
  (§2.1, actual dq) ≥ 0.85 in every shell beyond 12 Å. Separation is reported, not gated: at +1
  geometries the vertical neutral state is a shallow donor (the Pb–Pb dimer is broken), and the
  model's requirement is that the +1 charge sit inside the base's 12 Å reach, not that the level
  be deep.
- **C5** is read on the held-out +1 frames from now on; the neutral-vacancy readout is kept as a
  second column.
- **Ladder reading rule.** ε₀_ladder is read against the base's own ε(q) at the cells'
  wavevectors (§2.2), never against ε₀_LST as pass/fail. Registered expectation: ε₀_ladder ≥ 2ε∞
  and between ε(2π/L) of the largest and smallest cells.

## 1. bw6: bw5 at fraction 0.2 (three seeds, ~3 h on b3)

Runs `dscc_p2cwbw6_s{0,2,6}`: identical to bw5 (command, seeds, folds, schedule, σ, gap weight 0.1,
s_init) except `delta_rule="sp_splitting"` frac 0.2 (bounds Pb 1.43, Cs 0.32, Cl 2.58 eV). Per-epoch
readouts as bw5 (held F, s_near, dEp95, +1-frame separation/N_eff, 639-atom embed readout) plus the
flank p-channel |tanh| p50 on the held-out +1 frames.

**bw6 replaces bw5 if all hold on `model_avg.pt`:** flank |tanh| p50 ≤ 0.8 on every seed (settled
below the bound); paired forces against bw5 (n = 3, point estimate) within +τ or better on total and
4–12 Å; energy gate on ≥ 2 seeds; Σdq(< 12 Å) at 639 atoms on the seed median ≥ 0.9. If flank
|tanh| > 0.9 on ≥ 2 seeds: "runs to the bound", bw5 stands with the saturation recorded, the bound
question closed. Otherwise bw5 stands. The chosen arm's seed-median checkpoint (by held-out total) is
"the checkpoint" below.

§2.2 and §2.3 do not depend on the checkpoint and start now on the local card; §2.1, §2.4, §2.5
start when the checkpoint is fixed.

## 2. The far-field test on the checkpoint

All structures are tiled from the pristine 80-atom cell relaxed with the neutral production model
(base v8 w01 + E_PP, fmax 0.002); record its tilt angle. The base-v2 tilings of P0/P2c are not
reused. Everything at ≥ 2159 atoms runs on the 96-core CPU in float64 (the 24 GB card OOMs there with
this base); 639/959 on GPU.

### 2.1 Born-force table (switched kernel) — the direct measurement of the +1 far field

Structure: the +1 core relaxed at 639 atoms with the full model (fmax 0.002), embedded within
R_core = 8 Å in the 2879-atom tiling (CPU) and, as the cheap copy, in the 639-atom cell. At each
structure: dq from the checkpoint; the E_far force on every atom; Z*_a (base's own, pristine
geometry) · E_c(R_a) for (i) a unit Gaussian charge at the vacancy site and (ii) the actual dq, E_c
from the smoothed periodic kernel. Table by shell (4–6 / 6–8 / 8–10 / 10–12 / 12–14 / 14–17 /
17–20 / 20–24 Å): rms force, ratio, cosine, per species. Also Σdq within 3/5/8/12 Å and N_eff at
both sizes (the compactness gate).

Expected: unit charge ~0 below 6 Å, ramp 6–8, 0.8–0.97 beyond (P0); actual dq ≥ 0.85 beyond 12 Å
(P0 read 0.4–0.6 with the half-delocalised carrier).

### 2.2 Base v8 w01's lattice response (base alone; no head)

(a) Γ Hessian of the relaxed pristine cell (h = 0.01 Å, ASR), its own Z* → ε₀_LST polycrystalline
at ε∞ = 4.0 and at the base's 3.667. (b) Force constants by central finite displacement of the 80
inequivalent atoms in the 3×3×3 pristine tiling (480 base-alone force calls; the 12 Å receptive
field fits without wrap-around), Fourier-interpolated to D(q); mode charges with the base's Z* and
its form factor; ε_ion(q) from the longitudinal modes along x, y, z at q = 0, 2π/L for L = 47.3,
43.0, 32.8, 28.6 Å, and 0.26 Å⁻¹. Table ε(q) = ε∞ + ε_ion(q); comparison with base v2's values
(19.4 / 11.4 / 4.8 at 0 / 0.13 / 0.26 Å⁻¹ along y). (c) Form factor of the base's dipoles at
λ = 48 / 24 / 16 Å along the strongest IR mode (as P0 A1). (d) Polar branches along Γ–X/Y/Z against
a published PBE phonon dispersion of Pnma CsPbCl3 if one can be found; a gap otherwise.

### 2.3 LO–TO at q = 2π/(3b) in the 3×3×3 cell, base v8 + E_PP from its own dipoles

E_PP's LO − TO against the smoothed nonanalytic expectation g² f² (4π/Ω) Z̃²/ε∞ (P0's reference
computation). Criterion: within 10 %. Validates E_PP on this provider.

### 2.4 Frozen-core ladder (vertical; energies only)

The §2.1 core embedded within 8 Å in 959 / 2159 / 2879 tilings; E(+1) − E(0) at fixed geometry,
E(0) = base + E_PP; remainder after the exact ε∞ monopole; fit E∞ + c/L³ with 639 held out as
check. Criterion: remainder flat to 5 meV over 959–2879 (a forced 1/L term below 5 meV); report
E∞(frozen), and Σdq(< 5/8/12 Å), N_eff at every size (gate §0 at 639 and 2159).

### 2.5 Relaxation ladder (route 1, fmax 0.002) and the linear-response check

Tilings with one Cl removed at 639 / 959 / 2159 / 2879; the §2.1 core embedded within 8 Å; far
field relaxed with the core fixed; ΔE_relax(L) = E_relaxed − E_frozen; fit
ΔE_∞ + (Q²C α(h)/2L)(1/ε∞ − 1/ε₀_ladder). Read by the §0 rule against §2.2's ε(q); record the
previous values (W6 4.33, frozen-provider variant 3.74) alongside. Far-field profile: species-signed
radial displacement by shell and its exponent (continuum −2).

Linear-response check on the 2879-atom structure: F = the full model's forces on the far-field atoms
at the frozen embedded geometry; K = base force constants (§2.2b, real space, within 12 Å) plus
E_PP's dipole–dipole block (T_LR contracted with Z*) over the far-field atoms, core fixed;
u_pred = K⁻¹F; compare with the relaxed displacements by species and shell (rms ratio, cosine).
Criterion: ratio within 15 % and cosine ≥ 0.9 in every shell beyond 10 Å. Agreement means the
composition relaxes as its own linear response says, and any gap to ε₀_LST is the base's ε(q).

### 2.6 Decision

**Far field captured** if §2.1 ≥ 0.85 beyond 12 Å; §2.3 within 10 %; §2.4 flat and the §0
compactness gate passes; §2.5 ε₀_ladder ≥ 2ε∞ within §2.2's range and the linear-response check
passes. → Phase 3b (dynamics) opens on the checkpoint.

**Base-limited** if §2.1, §2.3 and the linear-response check pass but ε₀_ladder sits below §2.2's
range or §2.2 shows dispersion as strong as base v2's: the model relaxes consistently with its own
response and the deficit is the base's polar dispersion (compared with PBE in §2.2d). → Phase 3b
opens with the ε(q) characterisation as the stated limitation.

**Stop and report** if §2.1 fails (the carrier is not compact as the far ions see it) or §2.4 has
a 1/L term above 5 meV (E_far carries a monopole error).

## 3. Report

`P3A_REPORT.md`: §1 per-seed table and the replacement decision with the flank |tanh| curves; §2.1
tables at both sizes; §2.2 ε(q) table and the base-v2 comparison, form factor, dispersion plot;
§2.3; §2.4 ladder table and fit; §2.5 ladder table, fit, profile, and the linear-response table;
the §2.6 verdict; wall-clock per item.

## Not permitted

Retraining beyond bw6; changes to r_split, r_c, ε∞, E_PP or the switch; reading the ladder against
ε₀_LST as pass/fail; reusing base-v2 tilings; new arms.
