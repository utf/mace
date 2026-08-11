# E_LR extensivity: the background fix is refuted; the fault is `q^pol`, and a
# parameter-free projection fixes it at inference

Measured 2026-08-11 on `perov_lr_s1`, V_Cl+ in CsPbCl3, ladder 639 → 2159 atoms.
Scripts: `defect-perovskite/lr_term_scaling.py`, `pol_profile.py`, `perov_alpha_check.py`.
No retraining — everything below is the existing trained model.

---

## 1. The proposed background term cannot be this bug

LES is **not a split Ewald**. `les/module/ewald.py` computes a pure reciprocal-space sum over
`k != 0` of Gaussian-smeared charges — no real-space part, and `sigma` is the physical
smearing rather than a convergence parameter. Dropping `k = 0` for a net-charged cell omits
the finite `k -> 0` remainder of `exp(-sigma^2 k^2/2)/k^2 * Q^2`:

```
E_bg = -(2 pi / V) (sigma^2 / 2) Q^2 * C  =  -pi sigma^2 Q^2 C / V
```

which is exactly `-pi Q^2 / (2 V alpha^2)` under `alpha = 1/(sigma sqrt 2)`. **The algebra
agrees.** But the term is `O(1/V)`, so it *decays*:

```
     N       volume           E_bg       delta_lr        ratio
   639      23454.8      -0.003030       -23.1783     1.31e-04
  2159      79160.0      -0.000898       -75.5817     1.19e-05

  E_bg scaling  d(ln|.|)/d(ln N) = -0.999
```

Three milli-electronvolts against seventy-five electronvolts. The parenthetical
"extensive (`V ∝ N`)" is a slip: `Q²/V` with `V ∝ N` is `1/N`.

**The decisive argument is not magnitude, it is coupling.** The terms that actually grow are
`pol²` and `host·pol` (§2), and `Σ q_pol = 0` exactly. The jellium background is a pure `Q²`
effect — identically zero for a neutral distribution at any cell size. It does not touch the
growing terms under any `sigma` convention. That forecloses the question.

**The term is still genuinely missing and should be added** — see §6. It is worth ~24 meV at
the 79-atom training cells, and our DFT labels use the jellium convention, so adding it makes
LES match the labels. It is a correctness item, not a gate.

## 2. The fault, by decomposition

`delta_lr = E_LR(q_host + q_pol + q_carrier) - E_LR(q_host)`. The host self-energy — the
genuinely extensive and physically correct Madelung energy of the lattice — cancels exactly
by construction. Five terms survive, each obtained by the polarisation identity
`2B(a,b) = E(a+b) - E(a) - E(b)`:

```
     N       delta_lr           pol2       carrier2    pol_carrier       host_pol   host_carrier
   639      -23.17828       -6.35974       -0.00652        0.15830      -15.06461       -1.90571
   959      -34.18992       -9.54022       -0.00435        0.15775      -22.89733       -1.90577
  1439      -50.73401      -14.32901       -0.00290        0.15739      -34.65367       -1.90581
  2159      -75.58170      -21.53530       -0.00193        0.15714      -52.29577       -1.90584

scaling         +0.971         +1.002         -0.999         -0.006         +1.022         +0.000
```

* **`pol²` (+1.002) and `host·pol` (+1.022) are the extensive terms** — −21.4 of −23.2 eV at
  N = 639 and −73.8 of −75.6 eV at N = 2159: **97 % of the effect**.
* **Every carrier term is clean.** `host·carrier` is constant to five decimals (−1.90584,
  exponent +0.000), `pol·carrier` constant, `carrier²` decaying.

The channel designed with care is fine. The channel with no locality constraint is the bug.

## 3. Why: `q^pol` is a fictitious ionic lattice

`q^pol_i = (Σ_c n_c) · (MLP(node_feats_i, counter_emb) − mean)`. Mean subtraction makes
`Σ_i q^pol_i = 0` exactly — which the docstring cites as the safeguard — but **neutrality is
not localisation**. Far from the defect the node features are the bulk values for the
species, so the MLP returns a fixed per-species number and a *global* mean removes only the
composition-weighted average. Each species keeps a constant residual on every atom:

```
N = 639    sum q_pol = -1.05e-13   sum q_pol^2 = 2.2180
   species         <3 A        3-6 A       6-10 A        >10 A   bulk spread
        Cs            -     -0.06718     -0.06647     -0.06654      0.000757
        Pb      0.16798      0.10981      0.10888      0.10924      0.002473
        Cl            -      0.00210     -0.01532     -0.01506      0.003100

N = 1439   sum q_pol = -2.11e-15   sum q_pol^2 = 4.9499
        Cs            -     -0.06688     -0.06617     -0.06624      0.000763
        Pb      0.16828      0.11010      0.10918      0.10952      0.002485
        Cl            -      0.00240     -0.01502     -0.01480      0.003090
```

Flat from 3 Å to beyond 10 Å, per-atom spread ~0.001–0.003 e, **plateau identical at both
cell sizes**. An extra ionic lattice of ±0.07–0.11 e superimposed on the whole crystal
whenever `Σ_c n_c ≠ 0`. Its Madelung energy and its cross term with the host lattice both
scale with N, as measured. The only genuine physics is the near-shell Pb: +0.168 against a
+0.109 plateau, i.e. ~0.06 e of real response buried under tens of eV of artefact.

**This is structural, not a fitting accident.** The polarisation readout is a function of
purely local features and a global counter, so it is necessarily constant on bulk atoms of a
given environment. No amount of training can localise it; the constraint has to be imposed.

## 4. The fix: subtract the mean *within each species*

The plateau being removed **is** the per-species mean, so per-species subtraction kills it by
construction. It is a projection, parameter-free, using ops already in the module:

```python
species = node_attrs.argmax(dim=-1)
group = batch * self.num_species + species
mean = scatter_mean(polar_raw, group, dim=0, dim_size=num_graphs * self.num_species)
polarisation = polar_raw - mean[group]
```

Properties: a bulk atom is exactly zero by construction; the monopole guarantee is
*strengthened* (per-species zero ⇒ global zero); the residual contamination is benign,
because a defect makes each bulk atom of a species carry −(shell excess)/N_species, so the
artefact's sum of squares **decays** as 1/N instead of growing.

Implemented as `--per_species_neutral` (default False), plumbed through
`StructuredLatentCharges`, `MACEDefect`, `arg_parser`, `model_script_utils`, and the cuEq
`extract_config_mace_model` extractor — the last because a constructor argument missing from
that extractor silently reverts to its default, which is exactly how `freeze_amplitude` was
lost before.

### It works, and it can be tested without retraining

Because the projection is parameter-free it can be switched on at inference on the *existing*
trained model. Same model, same geometries, only the projection changed:

```
                    delta_lr           pol2       host_pol         ss_pol   host_carrier
  BEFORE  N=639     -23.17828       -6.35974      -15.06461        2.21805       -1.90571
          N=2159    -75.58170      -21.53530      -52.29577        7.40859       -1.90584
          scaling      +0.971         +1.002         +1.022         +0.991         +0.000

  AFTER   N=639      -1.36321       +0.06374       +0.48379        0.01805       -1.90571
          N=2159     -0.75311       +0.06481       +1.08847        0.02895       -1.90584
          scaling      -0.482         +0.024         +0.666         +0.388         +0.000
```

* `pol²`: **flat** (+1.002 → +0.024) and **330× smaller** (21.5 eV → 0.065 eV).
* `ss_pol`: **123× smaller** (7.41 → 0.029), exponent +0.991 → +0.388.
* `delta_lr`: now **decays** (+0.971 → −0.482), and is order 1 eV rather than 75 eV.
* `host·carrier` unchanged to five decimals — the physical term is untouched, as it must be.

And `q^pol` becomes a genuine local response, with the plateau gone:

```
N = 639    sum q_pol^2 = 0.0181            (was 2.2180)
   species         <3 A        3-6 A       6-10 A        >10 A
        Cs            -     -0.00063      0.00008      0.00001
        Pb      0.05787     -0.00031     -0.00123     -0.00088
        Cl            -      0.01666     -0.00076     -0.00050

N = 1439   sum q_pol^2 = 0.0238            (was 4.9499)
        Cs            -     -0.00064      0.00007      0.00000
        Pb      0.05837      0.00019     -0.00073     -0.00039
        Cl            -      0.01697     -0.00044     -0.00022
```

Shell response +0.058 (Pb) and +0.017 (Cl), essentially identical at both sizes; everything
beyond 6 Å is ≤0.001 e. This is what a polarisation response should look like.

### The residual, stated honestly

`host·pol` still grows at **+0.666**, and at 1.09 eV (N = 2159) it is now the dominant
remaining term. Two reasons to expect it to shrink but not to assume it:

1. This is a field **fitted under the wrong constraint** and projected afterwards. Retraining
   with the projection active removes the spurious lattice from the hypothesis space
   entirely.
2. `q^host` is itself globally mean-subtracted and so carries its own per-species plateau. Its
   self-energy is folded into `base_energy` (a real ionic lattice, correctly extensive, and
   *not* to be "fixed" — that is what `q^host` is for), but the cross term is where that
   plateau leaks into the n-dependence. Grouping is also coarse: orthorhombic CsPbCl3 has
   inequivalent Cl environments, so one "Cl" mean leaves within-species sublattice structure.

If the retrain leaves `host·pol` growing, the next step is a finer grouping (by environment
rather than species), not a redesign.

## 5. Correction to §3 of the reply — it is one bug, not two

§3 states the SiC growth was `q^pol` and the `host·pol` cross term, "a distinct structural
gap", and predicts the perovskite fix will not move it. **The mechanism named there is
exactly the one measured here.** The two systems do not have different faults; the perovskite
is the loud version of the SiC one, because `Σ_c n_c` is larger and the lattice it couples to
is ionic rather than covalent.

So my earlier report was right that the two share a fault, and both documents attributed it
wrongly — mine to the missing background, the reply's to two separate causes. §6 step 3 still
stands as the test, with the **opposite expectation**: fixing `q^pol` should fix both
ladders, and adding the background term should move neither by more than a few meV.

The SiC decomposition is the next thing to run and will settle it in a table.

## 6. Second finding, CORRECTED: the LR run's attention collapsed because the size
# hinge switched itself off. The short-range model is exactly right.

**Retract the previous version of this section.** It claimed the attention was diluting with
cell size and that the size hinge "read healthy anyway". Both are wrong, and the numbers I
quoted (`partic 2.02`, `size_f 0.016`) were the **nolr** model's, attributed to the LR one.
Audited properly on real training frames, several defect sites, and both models:

### The short-range model is textbook correct

```
perov_nolr_s1, live channel h_maj, real V_Cl+ frames from the dataset
  train frame 0    partic  1.96   Cs 0.000/16   Pb 1.000/16   Cl 0.000/47
  train frame 1    partic  2.00   Cs 0.000/16   Pb 1.000/16   Cl 0.000/47
  valid frame 0    partic  1.98   Cs 0.000/16   Pb 1.000/16   Cl 0.000/47
  ...              range 1.96 - 2.00 over train and valid

tiled supercells, three inequivalent Cl sites
  N=639  site 0    partic  1.99   top: Pb@2.6A 0.528  Pb@2.9A 0.472
  N=1439 site 0    partic  1.99   top: Pb@2.6A 0.528  Pb@2.9A 0.472
  N=639  site 1    partic  1.96   top: Pb@2.6A 0.569  Pb@3.0A 0.431
  N=1439 site 1    partic  1.96   top: Pb@2.6A 0.569  Pb@3.0A 0.431

pristine 80-atom cell at the SAME counter (no vacancy to find)
                   partic  7.09   Cs 0.000    Pb 0.001    Cl 0.999
```

The hole sits on **exactly the two under-coordinated Pb** that lost their bridging Cl, at
2.6 and 2.9 Å, split ~0.53/0.47 — the physically correct answer for a Pb-derived V_Cl state.
It is identical to three decimals across a 2.25× size range, holds across inequivalent
sites, and correctly **fails to localise** on a pristine cell where there is no vacancy.

This is why the short-range model works well. There was never a conflict to explain.

### The long-range model's attention is degenerate — and not size-dependent

```
perov_lr_s1, live channel h_maj
  train frame 0    partic 15.97   Cs 1.000/16    Pb 0.000/16    Cl 0.000/47
  N=639  site 0    partic 127.56  Cs 1.000/128   Pb 0.000/128   Cl 0.000/383
  N=1439 site 0    partic 287.02  Cs 1.000/288   Pb 0.000/288   Cl 0.000/863
  pristine         partic 15.95   Cs 1.000/16    Pb 0.000/16    Cl 0.000/48
```

Uniform over the **Cs sublattice**, zero on Pb and Cl, at *every* size — including the
79-atom training frames, where participation is 15.97 against 16 Cs. So this is **not
dilution with cell size**: participation tracks `N_Cs = N/5` because "uniform over Cs" means
that at any size. It is a wrong fixed solution, and it is identical with and without a
vacancy present — the attention does not see the defect at all.

### Why: the hinge's delocalised-carrier exemption is a trap

Final-epoch diagnostics, the two runs side by side:

```
nolr  partic=[79.7 79.7  2.013 79.7]  size_f=[1.000 1.000 0.017 1.000]  |c|=[0 0 0.090 0]
      size_x=[9.21 9.21 -5.023 9.21]  size_x*=[nan nan -4.164 nan]  size_viol=[0 0 0.236 0]

lr    partic=[79.7 79.4 16.073 79.4]  size_f=[1.000 1.000 1.000 1.000]  |c|=[0 0 0.000 0]
      size_x=[9.21 9.21  9.208 9.21]  size_x*=[nan nan   nan  nan]  size_viol=[0 0 0.000 0]
```

**In the LR run the size term was inactive for every channel for the whole run** —
`size_f = 1.000` across the board, `size_viol = 0`, contrast EMA `|c| = 0.000`. In the nolr
run it was live and doing work (`size_f = 0.017`, `|c| = 0.090`, `size_viol = 0.236`).

The mechanism is in `DefectLoss.size_threshold`, and it is deliberate:

```python
t = (per_channel_tol / magnitude).clamp_max(1.0)   # magnitude = |c|_EMA
satisfied = t >= 1.0
return torch.where(satisfied, +inf, threshold)     # caller's isfinite -> exempt
```

When the contrast `|c|` falls below `tol`, `t >= 1`, the threshold goes to `+inf`, and the
channel is exempt. The docstring calls this the *delocalised-carrier exemption* and its
reasoning is sound **for the energy**: if the attention-weighted pool differs from the mean
pool by less than `tol`, no amount of dilution can move the energy by more than `tol`.

But it is self-reinforcing. Flat attention ⇒ contrast → 0 ⇒ exemption engages ⇒ the hinge
switches off ⇒ nothing pulls the attention back. **The constraint degenerates exactly at the
state it exists to prevent**, and it cannot recover from it.

The hinge kept its actual promise: `<u>_h_maj` is 0.1269 eV at every cell size, drift
−0.0 meV, and `delta_SR` is exactly size-independent. It guarantees **energy** size-stability,
never spatial localisation. I previously read a large `gap_l` as evidence of localisation; it
is not — it separates *species*, and a sublattice is a fixed fraction of any cell.

### The two bugs are causally linked

The LR branch handed the model tens of eV of spurious fitting freedom in `q^pol` (§3). With
the defect energy absorbable there, the short-range carrier channel no longer needed contrast;
`u` flattened, `|c|` fell through `tol`, the exemption engaged, and the attention drifted to a
sublattice. That predicts **fixing `q^pol` should also restore the contrast and keep the hinge
live** — a prediction the retrain tests directly.

### Consequences

* §1–§4 are unaffected. `q^pol` is `MLP(node_feats, counter_emb)` mean-subtracted times
  `Σ_c n_c` — **no `alpha` dependence whatsoever** — so the extensivity diagnosis and the
  per-species fix stand on their own.
* `q^carrier = a·alpha` is physically meaningless in the LR model, though its monopole is
  still exactly right (`Σ = a·q = 0.5`). Every `dilute` number was measured on a
  sublattice-smeared monopole: the `−0.349` exponent is right, the magnitudes are not
  physical.
* The retrain must fix **both**. Guarding the exemption — e.g. a floor on `|c|`, or refusing
  to exempt a channel whose participation exceeds a fraction of the cell — is needed
  independently, because any future run can fall into the same trap.

## 7. Acceptance gates for the retrain — corrected

The gate table in the reply (and my earlier one) asks for `|delta_lr|` to fall to the order
of `q²α_M/2εL`. **That cannot happen and should not**: `host·carrier` sits at −1.906 eV,
constant to five decimals, and it is physical — the carrier monopole in the host lattice
potential. Gate on **drift**, not on absolute `delta_lr`:

| gate | before | target |
|---|---|---|
| `pol²` exponent / drift | +1.002, 21.5 eV | ≤ 0, meV-scale drift |
| `host·pol` exponent / drift | +1.022, 52.3 eV | meV-scale drift |
| `ss_pol` | 2.22 → 7.41 | noise floor, ~0.02–0.05, non-growing |
| `ε_opt` drift | −23 → −100 eV | ≤ tens of meV beyond a 1/L tail; flat with `dilute` |
| valid RMSE | 2.7 / 9.7 | ≥ as good as nolr's 3.1 / 12.0 |
| `Σqᵢ = a·q` | 0.50000 exact | unchanged |

Exponent alone is insufficient in both directions: a wrong-magnitude term can have the right
scaling, and a tiny-prefactor term can keep a bad exponent harmlessly.

## 8. Harness: fix the silent failure, not just the script

A wrong claim reached a co-advisor document because every stage in `run_overnight.sh` runs
under `2>/dev/null || continue` and the report renders "(n/a)" for both "this stage crashed"
and "not applicable". Adding an ERRORS section (per-stage exit code plus stderr tail) is part
of this change: a harness that eats a `TypeError` fails the "report ready in the morning"
requirement even when every ladder finishes.

## 9. Order of work

1. **SiC decomposition** with `lr_term_scaling.py` — converts §5 from prediction to a table.
2. **Retrain** perovskite with `--per_species_neutral True` + the `k = 0` background term;
   check §7 gates.
3. **Re-run SiC** with the fix. Expectation: the same fix resolves it.
4. **`alpha` on the wrong sublattice** — its own track. Not what makes `E_LR` extensive, but
   `q^carrier`'s shape and every `dilute` number depend on it.
5. Only then: site-averaging, DFPT `eps_inf`, real `E_VBM`, seed replication (≥3).
