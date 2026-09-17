# P0_REPORT — Plan P Phase 0 (diagnostics on the universal MACE-Field checkpoint, no training)

Run directory `~/runs/dscc/p0/`. Checkpoint `MACEField-omat-dielectric-converged.model`
(sha256 f92e043a…4aa2; heads pt_head / mp-dielectric / mp-ferroelectric; head used:
mp-dielectric; r_max 6.0 Å, 128x0e+128x1o, one field layer; MACE-MH-0 lineage). Units: forces
meV/Å per component, energies meV/atom, dipoles e Å, Born charges e, ε dimensionless.
Everything below was registered in `DSCC_V5_IMPLEMENTATION_SPEC.md` ("Plan P — REGISTRATION"
and the dated addenda) before the corresponding result was opened.

## Port and standing tests

MACE-Field (mdi-group/mace-field 45d5c5f) is ported onto this tree with a per-node field
(`electric_field_node`), a frozen `FieldProvider`, matrix-free k-space kernels for `E_cP` and
`E_PP` (`mace/modules/dscc/polarisation.py`) and the head coupling `host_coupling="polarisation"`.
Tests (all pass): node field = broadcast field bit-for-bit for E, F, P, Z*, α; Σ node dipoles =
ΩP to 1e-12; node dipoles = finite differences; cueq round trip to 1e-8; E_PP = dense-Hessian
reference to 1e-9 and real-space image sum to 1e-4 relative; E_cP = −Σ p·E_c with E_c from the
dense kernel; forces of both kernels = finite differences; head: neutral null = base + E_PP,
charged forces = central differences including dp/dR, Σdq = Q, batched = per-graph, translation
invariance, analytic and capture routes refused.

## 0.2 — Response on the pristine cell (base-v2-relaxed 80-atom cell)

| quantity | value |
|---|---|
| ε∞ tensor (diag) | 3.531, 3.535, 3.575 |
| ε∞ polycrystalline (variant M) | **3.547**; anisotropy 1.2 % (tensor kernel not needed) |
| Z* trace/3: Cl / Cs / Pb | −1.618 / +1.277 / +3.577 |
| Z* Frobenius: Cl / Cs / Pb | 3.52 / 2.24 / 6.21 |
| ΣZ* (acoustic sum rule) | 0 to 1e-12 (by construction of the enthalpy derivative) |
| same checkpoint on the MP CsPbCl3 cells (cubic, Amm2, P4/mbm, R32) | ε∞ 3.83–3.90; Z* Cl −1.67…−1.74, Cs +1.22…+1.27, Pb +3.75…+4.01 |
| MP DFPT e_electronic (mp-675022, mp-1539667) | 3.64, 3.62 |

The spread 3.55 (relaxed v2 cell) vs 3.83–3.90 (MP cells) vs 3.62–3.64 (MP DFPT) is the (M)
variant's uncertainty. Tabulated PBE Born tensors for orthorhombic CsPbCl3: the Materials
Project API exposes ε tensors but no Born charges, the training set holds CsPbCl3 only as ten
polarisation-path frames (BEC weight 0), JARVIS-DFT has no CsPbCl3 entry, and phonondb was not
reachable from this machine; the pattern (Pb ≈ +3.6…+4.0, Cs ≈ +1.3, Cl trace/3 ≈ −1.7 with a
strongly anisotropic Cl tensor) is the anomalous-Born-charge signature reported for the
CsPbX3 family, but no tabulated tensor was obtained in this session — recorded as a gap, not a
validation. Base-relaxed cell: E −281.152 → −281.876 eV, rms displacement 0.253 Å, same tilt
topology (every Cl bridges two Pb), Pb–Cl–Pb 167.1° → 156.1°, Pb–Cl 2.860 → 2.905 Å.

## 0.1 — Locality (decides r_split)

2×2×2 tiling (640 atoms), node field 0.01 V/Å on one atom per species, three directions.

| species | r90 of Σ|dp| (x, y, z) | r90 of Σ|dF| | share on the perturbed atom |
|---|---|---|---|
| Cl | 7.17, 7.00, 7.42 Å | 8.5 Å | 0.22–0.27 |
| Cs | 8.02, 8.04, 7.71 Å | 8.0–8.4 Å | 0.26–0.27 |
| Pb | 6.65, 6.65, 6.65 Å | 8.0 Å | 0.35–0.39 |

r90_max = 8.04 Å > 6 Å: by the plan's rule the mid-field coupling reverts to H0 for this
iteration (accepted; no second representation). **r_split = max(3.0, 1.2 × 8.04) = 9.645 Å** for
both E_cP and E_PP. Partition ambiguity RMS(p − p′)/RMS(p) = **1.11** (rms p 0.043, rms p′ 0.077
e Å; both sum to ΩP) — the per-atom-energy partition is a different object from the node-field
one; recorded, not gated.

## 0.3 — Mechanism test with a frozen provider

**Matching (E_PP forces alone, all 2560 training frames, dp/dR included).** At r_split 9.645 Å:
RMS 0.13 (pristine-80), 0.13 (neutral vacancy 79), 0.05 (159), 0.12 (charged 79), 0.04 (charged
159) meV/Å; E_PP −0.01 meV/atom. Threshold 3 meV/Å met (no raise). Scan: 8.50 / 4.74 / 1.86 /
0.56 / 0.13 meV/Å at r_split 3.0 / 3.6 / 4.5 / 6.0 / 9.645. With w_PP = 19.3 Å and w_cP = 13.7 Å
the kernels are smoother than the training cells, so E_PP is inert in the training domain by
construction and only the ladder's larger cells exercise it.

**eps0_LST** (base v2 Γ Hessian on the relaxed cell, h = 0.01 Å, ASR; provider Born tensors;
polycrystalline): ionic part 20.63 → **24.63 (R, ε∞ = 4.0) / 24.18 (M)**. No imaginary optical
modes; optical range 1.9–27.9 meV; the four strongest IR modes at 8.6–10.5 meV carry 9.1 of 20.6.

**Model.** W6 s1 (seed-median by held force RMSE; `~/runs/dscc/dscc_w6_s1/model.pt`,
uncalibrated — C_Q and E_M cancel in dE_relax at fixed cell and charge) with E_host → E_cP,
plus E_PP, provider attached, ε∞ 4.0 (variant R). Gate on a charged 79-atom training frame: dq
identical, band energy identical (6.816156 eV), e_madelung identical (−0.338319 eV); e_host
−20.5 meV → e_cP −0.13 meV + e_PP −0.49 meV; central-difference force check 2.1 and 0.8 meV/Å.
The untrained composition's forces differ from W6's by 16.9 meV/Å RMS (17.1 vs 8.4 against the
label): the W6 head was trained with E_host in the loop — a mechanism model, not a force claim.

**Cost.** One energy+force, charged +1, float64: 639 atoms W6 2.25 s / 5.8 GB, variant 3.75 s /
8.8 GB (A4000); 959 atoms 5.7 s / 8.5 GB and 7.9 s / 13.0 GB; 2159 atoms on 96 CPU cores 45 s
(W6) and 66 s (variant).

**Relaxation ladders (route 1).** Cells tiled from the base-relaxed pristine cell with the
first Cl removed (639 / 959 / 2159 / 2879 atoms; α(h)/L = 0.0950 / 0.0864 / 0.0634 / 0.0599),
core = the +1 vacancy fully relaxed in the 639-atom cell, embedded within 8 Å (56 atoms), far
field relaxed with the core fixed; fit `dE_relax = dE_∞ + (Q²C/2) x (1/ε∞ − 1/eps0_ladder)`.

*First run (fmax 0.02 eV/Å; stands in the record).* Every far-field relaxation converged in 6–7
BFGS steps at fmax 0.015–0.017.

| atoms | W6 dE_relax (eV) | variant dE_relax (eV) |
|---|---|---|
| 639 | −0.4219 | −0.3847 |
| 959 | −0.4255 | −0.3890 |
| 2159 | −0.4266 | −0.3884 |
| 2879 | −0.4278 | −0.3876 |
| slope (eV Å) | +0.133 | +0.056 |
| **eps0_ladder** | **4.32** (leave-one-out 4.17–4.39) | **4.13** (3.91–4.22) |
| slope eps0_LST = 24.6 would give | +1.51 | +1.51 |

Read before Decision 0 was taken: the criterion cannot measure this quantity. The far-field
response is driven by forces Z* C/(ε∞ r²) ≈ 0.026 eV/Å at 15 Å and 0.010 at 24 Å — at or below
fmax = 0.02 — so the optimiser stops with the dielectric shell unrelaxed, whose energy
(Σ F²/2k over ~2000 atoms) is ≈ 0.05–0.1 eV, the size of the missing ≈ 48 meV between the
639- and 2879-atom cells. The tolerance was re-registered at fmax 0.002 eV/Å for the core and
the far field (nothing else changed) and the ladders rerun (`ladder2_*`); the LO–TO fallback
runs in parallel. 
*Tight run (fmax 0.002 eV/Å, float64 base; in progress at 00:25 on 2026-09-17).* The W6
checkpoint stores its base in float32 (from `base_float32` in W6 training); all Phase 0
evaluations from the tight run on cast it to float64 (the first run carried ≈ 1 meV summation
noise per energy). Core relaxations now take 84–86 BFGS steps (fmax 0.0019 / 0.0018), the
far-field relaxations 50–60 instead of 6–7.

| atoms | W6 dE_relax (eV), tight | first run | variant, tight | first run |
|---|---|---|---|---|
| 639 | **−0.4331** | −0.4219 | running (core −0.3957) | −0.3847 |
| 959 | step 49 of the far field at fmax 0.004 | −0.4255 | queued | −0.3890 |
| 2159 | step 8 (local CPU) | −0.4266 | queued | −0.3884 |
| 2879 | queued | −0.4278 | queued | −0.3876 |

The tight tolerance recovers 11 meV at 639 atoms (W6) and 11 meV in the variant's core
(−0.3957 vs −0.3847). Whether the size dependence changes is what the remaining rungs decide;
the fit needs ≈ 48 meV between 639 and 2879 atoms for eps0_ladder to reach eps0_LST.

**Decision 0 fallback: LO–TO splitting (`p0_loto.json`, float64 base and variant, A = 0.05).**
Strongest IR mode of base v2's Hessian: 10.06 meV, Z̃ = −1.894 e/√amu along y. Frozen phonon
in the 3×3×3 cell (2160 atoms) at q = 2π/(3b) = 0.131 Å⁻¹, curvatures in eV Å⁻² amu:

| | LO (q ∥ y) | TO (q ∥ z) | LO − TO |
|---|---|---|---|
| base v2 alone | 0.581 | 0.331 | 0.250 (finite-q dispersion, not nonanalytic) |
| E_PP alone | +0.135 | −0.001 | **+0.136** |
| variant total | 0.716 | 0.331 | 0.386 |
| nonanalytic from Z*, ε∞ = 4, unsmoothed (q → 0 kernel) | 0.746 | 0 | 0.746 |
| same, at the registered smoothing (g = e^{−w²q²/4} = 0.206) | 0.154 | 0 | 0.154 |
| same, with the provider's actual response (form factor 0.94 of Z* u) | 0.135 | 0 | 0.135 |

(The ±q pair of the structure factor is counted; checked numerically against the kernel on the
pattern p = Z* u to 0.3 %.) Reading: the splitting appears with E_PP and matches the value the
registered kernel and provider are built to give to 0.3 %; it is 18 % of the unsmoothed
nonanalytic splitting because r_split = 9.645 Å (the plan's own locality rule, > 6 Å accepted)
smooths the dipole kernel over w = 19.3 Å, four fifths of it at this q. By the plan's fallback
text the mechanism is present; the plan's "mid-field reverts to H0" acceptance is what confines
E_cP and E_PP to the far field (beyond ≈ 2 r_split).

## Phase 1 pipeline (smoke, one epoch, b3 GPU 5, batch 4)

`p1_run.sh`: MACE-Field checkpoint → multihead fine-tune on the neutral defect data with a
pseudolabelled replay (`fine_tuning_select`, fps 1000, combinations: 16 Cs/Pb/Cl + 984 random),
`universal_field` loss, E 10 / F 10 / stress 1 / P 1000 / Z* 100 / α 100, lr 1e-4, EMA 0.9999,
E_PP in the forward at r_split 9.645. A bug found by the first smoke and fixed: the pseudolabel
generator did not request the response, so the replay's P / Z* / α labels were zeros with unit
weight and the fine-tune drove the response to zero (replay α RMSE 3.87 → 0.79 in one epoch).
With the fix the replay head starts at 0.0005 e (Z*) / 0.004 ε0 (α) and after one epoch holds
0.019 e / 0.081 ε0 (≈ 2 % of the response), while the neutral head goes 20.9 → 4.7 meV/atom
and 67.0 → 37.3 meV/Å (stress 0.49 → 0.88 meV/Å³). Batch 8 does not fit the 16 GB card, and
with a second job on a 24 GB card batch 4 is the limit; the production run has its own card.

## Preliminary conclusions (to be confirmed by the tight ladder)

1. **The instruments are exact.** Port, per-node field, kernels, head coupling and the frozen
   provider pass every registered test; the kernel and provider deliver precisely the smoothed
   nonanalytic response (0.3 %).
2. **The coupling is confined to the far field by the plan's own rule.** r90 = 8.0 Å puts
   r_split at 9.645 Å; at that width E_PP and E_cP are inert on every training frame (forces
   0.13 meV/Å, E_cP −0.13 meV where E_host was −20.5 meV) and carry 18–21 % of the dipole physics
   at the shortest wavelength of a 48 Å cell. Phase 2's arm B versus A comparison on 79/159-atom
   frames is therefore expected to be a null by construction unless r_split is revisited; the
   frozen composition also loses W6's trained host term (17.1 vs 8.4 meV/Å on the charged frame).
3. **The ladder, as first run, showed no lattice screening beyond ε∞** (eps0_ladder 4.3 / 4.1 vs
   eps0_LST 24.6) for W6 and the variant alike; the loose tolerance was unable to measure the
   quantity and the tight rerun is in progress (11 meV recovered at 639 atoms so far). If the
   tight ladder still gives eps0 ≈ ε∞ for both arms, the fallback assigns the deficit to base v2's
   restoring forces and Phase 1 proceeds by the plan's text; the reading I would give is that the
   dielectric relaxation is not resolved by a route-1 ladder at these sizes with either model.
4. **Phase 1 is ready to run** (recipe registered, smoke clean after the pseudolabel fix); it
   opens on Decision 0 or the fallback, i.e. either way, but the r_split question above is
   the decision that matters for what Phase 2 can show.

## Amendment A1 (user, 2026-09-17 00:45) — r_split by matching, form factor in place of 0.1

The r90 rule turned a bond-scale effect into a 19 Å smoothing that switched the coupling off at
every testable size; the r90-based 0.3 is a null by construction and its LO–TO agreement (0.3 %)
the positive finding. Amended: **r_split = 4.5 Å** (largest E_PP force on training frames under
2 meV/Å: 1.86 all-families RMS); 0.1 replaced by the provider's plane-wave Born response;
Decision 0 restated as variant eps0_ladder ≥ 8 and ≥ 2 × W6's with LO–TO within 10 % of the
smoothed expectation; W6's tight ladder kept as the control.

**Form factor** (`p0_formfactor.json`; 3×3×3 cell, strongest IR mode, ±0.05 central differences
of the provider's node dipoles):

| axis | λ = L (n = 1) | L/2 | L/4 | w_eff (Å) |
|---|---|---|---|---|
| x (L = 47.1 Å) | 0.945 | 0.810 | 0.528 | 3.09 |
| y (L = 48.5 Å) | 0.939 | 0.787 | 0.454 | 3.43 |
| z (L = 34.1 Å) | 0.834 | 0.531 | (0.09, aliasing) | 4.21 |

The n = 3 points (λ = a, b, c) are lattice-periodic patterns, not modulations, and are excluded
(x: f = 0.11 with a 180° phase). w_eff_max = 4.21 < 4.5 → r_split = 4.5 Å stands; z (the
11.4 Å axis) sits at the edge (4.3–4.6 Å from its two clean points).

**Variant at 4.5 Å** (`w6_s1_pol_r45.pt`, float64 base; same gate frame): dq, band energy,
e_madelung identical to W6; e_host −20.5 meV → e_cP **−40.9 meV**, e_PP +0.2 meV; FD checks 0.35
and 0.002 meV/Å; forces 11.8 vs W6 8.4 meV/Å against the label (17.1 at 9.645 Å).

**LO–TO at 4.5 Å** (`p0_loto_r45.json`; the nonanalytic reference now counts the ±q pair of the
structure factor, checked numerically to 0.3 %; the 9.645 Å numbers above are restated with it:
0.136 vs 0.154 smoothed / 0.135 with the form factor / 0.746 unsmoothed):

| | LO | TO | LO − TO |
|---|---|---|---|
| base v2 alone | 0.581 | 0.331 | 0.250 |
| E_PP alone | +0.461 | −0.007 | **+0.468** |
| expected: unsmoothed / smoothed (g = 0.709) / × form factor² (0.939²) | 0.746 / 0.529 / 0.467 | 0 | 0.467 |

E_PP reproduces the smoothed nonanalytic splitting to **0.17 %** (A1 criterion 10 %): the LO–TO
half of Decision 0 passes.

**Tight ladders, complete (fmax 0.002 eV/Å, float64 base).**

| atoms | α(h)/L | W6 (control) dE_relax (eV) | variant, r_split 4.5 Å |
|---|---|---|---|
| 639 | 0.0950 | −0.4331 | −0.4072 |
| 959 | 0.0864 | −0.4369 | −0.4076 |
| 2159 | 0.0634 | −0.4378 | −0.4058 |
| 2879 | 0.0599 | −0.4392 | −0.4017 |
| slope (eV Å) | | +0.137 | −0.127 |
| **eps0_ladder** | | **4.33** (leave-one-out 4.16–4.40) | **3.74** (3.65–3.89) |

Every far-field relaxation took 47–60 BFGS steps to fmax ≤ 0.0016 (the loose run's 6–7); each
W6 rung is 11–12 meV deeper than at fmax 0.02 but the size dependence is unchanged (4.32 →
4.33). Far-field displacement pattern of the 2879-atom cells (rms radial displacement by shell,
8–10 / 10–12 / 12–14 / 14–16 / 16–18 / 18–20 / 20–24 / 24+ Å): W6 0.029 / 0.017 / 0.011 / 0.006 /
0.006 / 0.003 / 0.0023 / 0.0010 Å, variant 0.023 / 0.016 / 0.009 / 0.005 / 0.005 / 0.0025 / 0.0018 /
0.0008 Å; |u_r| ∝ r^b with b = −3.2 / −2.7 / −3.0 (Cl / Cs / Pb, W6) and −3.1 / −2.7 / −2.9
(variant), against the continuum −2. The two arms relax the far field the same way.

## Decision 0 (A1 form)

| criterion | value | met |
|---|---|---|
| variant eps0_ladder ≥ 8 | 3.74 | no |
| variant eps0_ladder ≥ 2 × W6 | 3.74 vs 4.33 | no |
| LO–TO of E_PP within 10 % of the smoothed expectation | 0.17 % | yes |

**Outcome: FALLBACK_PROCEED** (plan: "if the splitting appears but the ladder fails, the
deficit is in base v2's restoring forces and Phase 1 proceeds anyway"). Phase 1 (base v5)
launched 04:16 on b3 GPU 5 with r_split 4.5 Å in the forward (30 epochs, batch 4 — the
registered batch 8 does not fit a 24 GB card with the BEC second derivatives on the replay's
up-to-288-atom structures; registered deviation).

## Conclusions

1. **The instruments are exact.** Port, per-node field, kernels, head coupling and provider pass
   every registered test; E_PP reproduces the smoothed nonanalytic LO–TO response to 0.3 %
   (9.645 Å) and 0.17 % (4.5 Å), with the provider's plane-wave Born response at 0.94 of Z* u
   at λ = 48 Å and a bond-scale effective width (3.1–4.2 Å).
2. **The r90 rule was the wrong instrument** (a 4 Å effect turned into a 19 Å smoothing); with
   r_split = 4.5 Å by matching, the coupling is active where the data are (E_cP −41 meV on the
   charged 79-atom frame, forces of the frozen composition 11.8 vs W6's 8.4 meV/Å against the
   label).
3. **Neither the shipped W6 nor the frozen polarisation composition shows lattice screening
   beyond ε∞ in the route-1 ladder at 639–2879 atoms** (eps0_ladder 4.33 and 3.74 against
   eps0_LST 24.6), with a relaxation tolerance ten times tighter than the far-field forces and
   identical far-field displacement patterns. The ladder measures the model's relaxation
   response to the carrier, not the kernel; on base v2 that response does not carry the ε0
   physics at these sizes. Quantitative eps0 is a Phase 3 question on base v5.
4. **Phase 1 is running**; its pipeline is validated (pseudolabelled response held to ≈ 2 %
   after one epoch in the smoke), with gate 2 (response within 10 %) the live risk.
