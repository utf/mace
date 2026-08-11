# `E_LR` extensivity: `q^pol` has no locality constraint

Measured on `perov_lr_s1`, V_Cl+ in CsPbCl3, ladder 639 → 2159 atoms, 2026-08-11.
Scripts: `defect-perovskite/{lr_term_scaling,pol_profile,alpha_audit,check_gates}.py`.

Everything in sections 1-4 and 6 is measured on trained weights with no retraining.

---

## 1. Summary

The long-range branch gives an `eps_opt` that grows linearly with cell size, reaching
-100 eV at 2880 atoms where it should be a few eV. Two faults, causally linked:

1. **`q^pol` has no locality constraint.** It is mean-subtracted over the whole cell, which
   makes it exactly neutral but leaves every atom carrying a fixed per-species charge -- a
   spurious ionic lattice. Its Madelung energy and its cross term with `q^host` are both
   extensive and account for **97 %** of the growth.
2. **The size hinge was inactive for the whole long-range run**, through its
   delocalised-carrier exemption. The attention collapsed onto the Cs sublattice.

The short-range model is unaffected by either and behaves correctly.

## 2. The missing `k = 0` background is real but is not this bug

LES is **not a split Ewald**: `les/module/ewald.py` sums `exp(-sigma^2 k^2/2)/k^2 |S(k)|^2`
over `k != 0` only, with `sigma` the physical smearing. For a net-charged cell the `k -> 0`
limit is finite once a compensating background cancels the divergent `1/k^2` piece:

```
E_bg = -(2 pi / V) (sigma^2 / 2) Q^2 C = -pi sigma^2 Q^2 C / V
```

the same term as `-pi Q^2 / (2 V alpha^2)` with `alpha = 1/(sigma sqrt 2)`. Measured:

```
     N       volume           E_bg       delta_lr        ratio
   639      23454.8      -0.003030       -23.1783     1.31e-04
  2159      79160.0      -0.000898       -75.5817     1.19e-05
  E_bg scaling  d(ln|.|)/d(ln N) = -0.999
```

Three meV against seventy-five eV, and it **decays** as `1/V`. The decisive argument is not
magnitude but coupling: the growing terms are `pol^2` and `host.pol`, and `sum_i q_pol = 0`
exactly, while the background is a pure `Q^2` effect that vanishes identically for a neutral
distribution. It cannot touch them under any `sigma` convention.

**Added anyway, as a correctness item.** It is worth -24 meV at the 79-atom training cells,
and the DFT labels use the jellium convention, so without it the model fits to a different
electrostatic convention than its own. Implemented in `LatentEwald.energy`; verified to
shift `carrier^2` by exactly -0.00303 eV at N = 639, as derived. Not an acceptance gate.

## 3. Decomposition

`delta_lr = E_LR(q_host + q_pol + q_carrier) - E_LR(q_host)`. The host self-energy -- the
genuinely extensive and physically correct Madelung energy of the lattice -- cancels exactly
by construction. Five terms survive, via `2B(a,b) = E(a+b) - E(a) - E(b)`:

```
     N       delta_lr           pol2       carrier2    pol_carrier       host_pol   host_carrier
   639      -23.17828       -6.35974       -0.00652        0.15830      -15.06461       -1.90571
   959      -34.18992       -9.54022       -0.00435        0.15775      -22.89733       -1.90577
  1439      -50.73401      -14.32901       -0.00290        0.15739      -34.65367       -1.90581
  2159      -75.58170      -21.53530       -0.00193        0.15714      -52.29577       -1.90584

scaling         +0.971         +1.002         -0.999         -0.006         +1.022         +0.000
```

`pol^2` and `host.pol` are the extensive terms: 97 % of the effect. Every carrier term is
clean -- `host.carrier` constant to five decimals and **physical** (the carrier monopole in
the host lattice potential), `pol.carrier` constant, `carrier^2` decaying.
`sum_i q_i = a q = 0.5` exactly at every size: the monopole is right, the shape is not.

## 4. Why: `q^pol` is a fictitious ionic lattice

`q^pol_i = (sum_c n_c) (MLP(node_feats_i, counter_emb) - cell mean)`. Mean subtraction makes
`sum_i q^pol_i = 0` exactly, and that is the only safeguard present. **Neutrality is not
locality.** Far from the defect the node features are the bulk values for the species, so
the MLP returns a fixed per-species number and a global mean removes only the
composition-weighted average:

```
N = 639    sum q_pol = -1.05e-13   sum q_pol^2 = 2.2180
   species         <3 A        3-6 A       6-10 A        >10 A   bulk spread
        Cs            -     -0.06718     -0.06647     -0.06654      0.000757
        Pb      0.16798      0.10981      0.10888      0.10924      0.002473
        Cl            -      0.00210     -0.01532     -0.01506      0.003100

N = 1439   (plateaux identical: Cs -0.06624, Pb +0.10952, Cl -0.01480)
```

Flat from 3 A to beyond 10 A, plateau identical at both sizes: an extra ionic lattice of
+/-0.07-0.11 e on the whole crystal whenever `sum_c n_c != 0`. The only genuine physics is
the near-shell Pb, +0.168 against a +0.109 plateau -- ~0.06 e of response under tens of eV
of artefact.

This is **structural**. The readout is a function of purely local features and a global
counter, so it is necessarily constant on bulk atoms of a given environment. No amount of
training can localise it; the constraint has to be imposed.

## 5. Per-species neutralisation was tried and rejected

Subtracting the mean *within each species* removes the plateau by construction and works
numerically (`pol^2` 330x smaller and flat, `ss_pol` 123x smaller). It is nonetheless the
wrong fix, because it estimates the bulk value **empirically** and so fails wherever that
statistic is undefined:

* mixed valence and charge disproportionation -- two Pb environments in different charge
  states share one mean, and both get a spurious residual;
* alloys, solid solutions, surfaces, interfaces -- no clean per-species bulk value;
* thermal disorder, **already present here**: the 80-atom snapshot has 48 inequivalent Cl
  environments, so "group more finely by environment" has no definition either;
* it forbids real physics -- per-species neutrality prohibits net charge transfer between
  sublattices, the dominant screening mechanism in an ionic solid. The near field came out
  right (Pb shell +0.058, Cl shell +0.017, both positive as electrons flow toward a hole)
  but the compensating charge was smeared as `-excess/N_species` across the sublattice
  instead of forming a local screening shell.

Reverted in full, including from the cuEq `extract_config_mace_model` extractor.

## 6. The fix: response follows cause

The far-field dielectric response is already carried by `a = 1/sqrt(eps_inf)` on
`q^carrier`. What `q^pol` has to represent is the beyond-continuum **near-field** response,
and in a gapped material the density-density response is short-ranged -- hence local **to
the carrier**.

**Step A (`--use_polarisation False`)** drops the channel. After gating, the response is
contained inside the receptive field (<=0.001 e beyond 6 A against ~10 A), and a neutral
cloud inside the receptive field has a local electrostatic energy that `delta_SR` can
already represent. The long-range branch is only structurally required for the monopole and
the slowly-converging multipoles of the carrier, which `q^carrier` carries.

**Step B (`--pol_gate`)**, only if Step A loses accuracy:

```python
g_i     = sum_j (sum_c n_c alpha_j^c) phi(r_ij)      # UNSIGNED, smeared over the MP graph
Pbar    = sum_j g_j p_j / max(sum_j g_j, eps)
q^pol_i = g_i (p_i - Pbar)
```

Unsigned because an exciton with net-zero carrier charge still polarises locally. The
denominator is *clamped* rather than offset, so `sum_i q^pol_i` stays exactly zero whenever
any gate weight exists, and the gate vanishes identically at `n = 0` so the identity is
structural. `lambda` is a fixed gauge constant recorded with the model: unbounded, it
recovers the plateau. Second benefit -- a delocalised `alpha` gives `g ~ 1/N`, so attention
collapse stops being profitable.

### Measured at inference on existing weights

```
mode        delta_lr exp   pol^2 exp   host_pol exp   delta_lr range
baseline         +0.971      +1.002         +1.022    -23.2 to -75.6 eV
ablate (A)       -0.003           0              0    -1.915 to -1.909  (6.6 meV drift)
gate (B)         +0.002      -0.994         +0.024    -2.384 to -2.391  (6.6 meV drift)
```

`host.carrier` is unchanged to five decimals in all three -- the physical term survives
untouched, as it must. The gate's constant `host.pol` is the predicted delocalised-regime
answer (`N A_host (1/N) = O(1)`), which is the regime *this* model is in, since its `alpha`
is collapsed. The localised regime is covered by `tests/unit/test_pol_gate.py`, where the
gated `q_pol` is bit-identical across sizes and the energy converges.

## 7. Second fault: the exemption is a trap

The short-range model's attention is correct -- audited on real dataset frames, three
inequivalent Cl sites, and a pristine control:

```
perov_nolr_s1   real frames     partic 1.96-2.00   Cs 0.000  Pb 1.000  Cl 0.000
                N=639  site 0   partic 1.99   top: Pb@2.6A 0.528  Pb@2.9A 0.472
                N=1439 site 0   partic 1.99   top: Pb@2.6A 0.528  Pb@2.9A 0.472
                pristine        partic 7.09   Cs 0.000  Pb 0.001  Cl 0.999
```

The hole sits on exactly the two under-coordinated Pb, identical across a 2.25x size range
and across sites, and correctly fails to localise where there is no vacancy.

The long-range model is uniform over the **Cs sublattice** with zero weight on Pb, at every
size *including* the 79-atom training frames (participation 15.97 against 16 Cs), and
identically with and without a vacancy. Not dilution with size -- a wrong fixed solution.

Final-epoch diagnostics:

```
nolr  partic=[79.7 79.7  2.013 79.7]  size_f=[1.000 1.000 0.017 1.000]  |c|=[0 0 0.090 0]
lr    partic=[79.7 79.4 16.073 79.4]  size_f=[1.000 1.000 1.000 1.000]  |c|=[0 0 0.000 0]
```

`DefectLoss.size_threshold` exempts a channel once `|c|` falls below `tol` -- the
*delocalised-carrier exemption*, whose reasoning is sound for the energy. But it is
self-reinforcing: flat attention drives the contrast to zero, the channel is exempted, the
hinge switches off, and nothing pulls the attention back. The constraint degenerates exactly
at the state it exists to prevent.

Note the hinge kept its actual promise: `<u>_h_maj` is 0.1269 eV at every cell size, drift
-0.0 meV, and `delta_SR` is exactly size-independent. It guarantees **energy**
size-stability, never spatial localisation. A large `gap_l` is likewise not evidence of
localisation -- it separates *species*, and a sublattice is a fixed fraction of any cell.

### The guard, and the near-miss worth recording

A channel whose participation exceeds both an absolute floor and a fraction of the cell is
refused the exemption. Both bounds are needed: a fraction alone misfires on small cells,
where a genuinely localised carrier is a non-trivial share.

**Refusing the exemption is not sufficient on its own.** With `satisfied` merely flipped,
the threshold still came from `|c|`, and `|c| ~ 0` sends `t` to its clamp, giving
`x* = ln((1-1e-12)/1e-12) = 27.6` while `x` is structurally capped at `ln(R-1) = 9.21`.
Violation `max(0, 9.21 - 27.6) = 0`: a label change with no gradient. In float32 -- what
training uses -- `1 - 1e-12` rounds to 1.0, the threshold becomes `+inf`, and the channel is
silently exempt again. Measured: penalty **0.0 in both dtypes**.

So for a collapsed channel the threshold must not derive from `|c|` at all, since its
normalisation is what degenerated. A fixed localisation target is used instead -- cap the
leaked fraction at `size_delocalised_leak`, so `x* = ln(f*/(1-f*)) = -2.94`, in the same
units as the `size_f` diagnostic. Penalty after the fix: 295.5 in both dtypes at unit
weight, ~0.015 at the production 1e-4.

### Causal link

The long-range branch handed the model tens of eV of spurious freedom in `q^pol`. With the
defect energy absorbable there, the short-range carrier channel no longer needed contrast;
`u` flattened, `|c|` fell through `tol`, the exemption engaged, and the attention drifted to
a sublattice. **Fixing `q^pol` should therefore also restore the contrast** -- which the
Step A run tests directly, now with a functioning guard.

### Consequence for `dilute`

`q^carrier = a alpha`, so every `dilute` magnitude was measured on a sublattice-smeared
monopole. The `N^-0.349` exponent shows the machinery is correct; the magnitudes
(69-117 meV) are **not yet physical**. Re-measured on the repaired model.

## 8. Acceptance gates

Gate on **drift**, not absolute `delta_lr`: `host.carrier` is -1.906 eV, constant, and
physical. An exponent alone is insufficient in either direction -- a wrong-magnitude term
can carry the right scaling, and a tiny-prefactor term can carry a bad exponent harmlessly.

| gate | before | target |
|---|---|---|
| `pol^2` drift | +1.002, 21.5 eV | <= 0 exponent, meV drift (identically 0 under Step A) |
| `host.pol` drift | +1.022, 52.3 eV | meV drift, or constant for a delocalised carrier |
| `ss_pol` | 2.22 -> 7.41 | non-growing (identically 0 under Step A) |
| `eps_opt` drift | -23 -> -100 eV | <= tens of meV beyond a 1/L tail; flat with `dilute` |
| `partic` (live) | 16.07 | ~2, on the vacancy-shell Pb |
| `size_f` (live) | 1.000 (exempt) | < 0.1, hinge live |
| valid RMSE | 2.7 / 9.7 | at least as good as nolr's 3.1 / 12.0 (checker allows 15 % slack) |
| `sum q_i = a q` | 0.50000 | unchanged |

`check_gates.py` evaluates these and prints pass/fail. Validated against the known-bad
baseline: on `perov_lr_s1` it FAILs `partic` and `size_f` and PASSes the RMSE, so it
discriminates rather than merely permitting.

Do not read `perov_lr_s1`'s better training metrics as evidence for the branch: the spurious
`q^pol` term is near-constant per atom at fixed cell size, so the fit absorbs it as an energy
offset. That is how both faults stayed invisible in training.

## 9. Order of work

1. ~~Revert per-species~~; ~~inference-time evaluation of Steps A and B~~ (section 6).
2. **Step A retrain** -- `q^pol` off, plus the `k = 0` background and the exemption guard.
   Running; `run_step_a_diagnostics.sh` is queued behind it and writes
   `~/runs/step_a_report.md`.
3. Step B only if Step A loses accuracy.
4. Re-measure `dilute` on the repaired model.
5. **SiC transfer test**, only once the perovskite gates pass. A genuinely different
   question: a fix validated on a charged single-carrier system, applied to a neutral
   multi-carrier one whose carrier channel had collapsed. Expect channel collapse to need
   separate treatment -- the exemption guard is the relevant lever there, not the `q^pol`
   fix.
6. Then: site-averaging `eps_opt` over many Cl environments, DFPT `eps_inf` for the interim
   4.0, a real per-level-of-theory `E_VBM` for the fitted gauge, seed replication (>=3).

## 10. Standing recommendation

Use `perov_nolr_s1` for size-critical work until the gates pass. The short-range model is
**self-consistently missing** the image tail, not getting it wrong: a charged cell has a
genuine `q^2 alpha_M / 2 eps L` tail that a strictly short-ranged model has no mechanism to
produce, so it returns a constant.
