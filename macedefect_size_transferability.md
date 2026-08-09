# MACEDefect: size transferability of the carrier correction

**Status:** finding, needs a decision. **Model tested:** `b_128ch_L1_s1` (128 channels,
`max_L = 1`, 140 epochs, the best available at time of writing). **Date:** 2026-08-09.

---

## 1. One-paragraph summary

We built a finite-size convergence test for the 4H-SiC divacancy, reproducing the figure
from the NEP paper the dataset came from. The observables do not converge: `E_ZPL` falls
from 0.903 eV at 256 atoms to 0.715 eV at 2400 atoms and is still moving. The drift is
**logarithmic in N**, so it is not a finite-size image interaction. The cause has been
measured, not inferred: `alpha = softmax(l)` is normalised over *every atom in the cell*,
so bulk sites accumulate weight proportional to N and eventually outvote any fixed logit
gap. The correction stops being a carrier bound to the defect and becomes a bulk average.
**Three of the four carrier channels are already diluted inside the training
distribution.** The base branch is exactly size-consistent, so this is confined to the
correction, and the condition for fixing it is cheap because it is logarithmic.

---

## 2. What the test does

`defect-example/defect_size_extensivity.py`, driven by `run_size_convergence.sh`.

For each supercell in a ladder, holding the cell fixed at the MLIP-relaxed bulk value and
relaxing only internal coordinates:

1. relax the divacancy in the **ground** state -> `R_gs`, `E_gs(R_gs)`
2. relax the divacancy in the **excited** state -> `R_ex`, `E_ex(R_ex)`
3. single point: each state at the *other's* geometry -> `E_ex(R_gs)`, `E_gs(R_ex)`

giving

| quantity | definition |
|---|---|
| `E_f` | `E_gs(R_gs) - E_host + mu_SiC` |
| `E_ZPL` | `E_ex(R_ex) - E_gs(R_gs)` |
| `E_vert^abs` | `E_ex(R_gs) - E_gs(R_gs)` |
| `Delta` | `E_vert^abs - E_ZPL` (excited-state relaxation) |
| `Delta q` | `sqrt(sum_i m_i |R_ex,i - R_gs,i|^2)`, minimum-image, in `sqrt(Da) A` |

The divacancy removes one Si and one C, i.e. exactly one SiC formula unit, so
`mu_SiC = 2 * e_bulk_per_atom` and `E_f` is **chemical-potential independent**. `e_bulk`
comes from the model's own relaxed primitive, so the reference is self-consistent.

The defect site is the axial Si-C nearest-neighbour pair closest to the cell centre,
chosen deterministically and recorded (axial and basal divacancies are genuinely
inequivalent in 4H-SiC).

### Two things that had to be got right

**The energy scale must be `raw`.** The labels were band-edge referenced, and the constant
is `n_electrons * e_cbm - n_holes * e_vbm`: **0.9430 eV** for the ground state `(1,0,0,1)`
and **1.8861 eV** for the excited `(1,1,0,2)`. It does *not* cancel in a ground-to-excited
difference, so computing `E_ZPL` on the model's native scale gives an answer shifted by
exactly the gauge gap, 0.9430 eV -- the same size as the answer itself, and entirely
plausible-looking. This is a silent-error trap for anyone else computing transition
energies from this model.

**The band-edge registry had to be extended.** It is keyed `host|natoms`, while
`band_edges.json` holds one size-free entry per host, so every cell size off the training
ladder raises `KeyError` under `raw`. The script replicates the host entry across the sizes
it visits and says so at run time. That is faithful to how the labels were referenced (a
fitted gauge in which only `e_cbm - e_vbm` matters) but would be wrong against a genuinely
size-dependent table.

---

## 3. Results

Ladder chosen to overlap the paper's sizes. Figure: `defect-example/size_convergence.png`.

| repeat | N | `E_f` | `E_ZPL` | `E_abs` | `Delta` | `Delta q` | correction | base drift |
|---|---|---|---|---|---|---|---|---|
| (4,4,2) | 256 | 7.2788 | 0.9030 | 0.9687 | 0.0657 | 0.5852 | 3.4539 | −0.000 |
| (5,5,2) | 400 | 7.2852 | 0.8658 | 0.9495 | 0.0836 | 0.6835 | 3.4705 | −0.000 |
| (6,6,2) | 576 | 7.2989 | 0.8322 | 0.9264 | 0.0942 | 0.7425 | 3.4872 | −0.000 |
| (8,8,2) | 1024 | 7.3263 | 0.7800 | 0.8820 | 0.1020 | 0.7690 | 3.5170 | −0.000 |
| (10,10,3) | 2400 | 7.3632 | 0.7145 | 0.8101 | 0.0956 | 0.7219 | 3.5604 | −0.000 |

Energies in eV, `Delta q` in `sqrt(Da) A`, base drift in meV/atom.

Extrapolated against `1/N`, compared with the paper's converged values:

| | ours (N -> inf) | NEP paper | at our N = 256 |
|---|---|---|---|
| `E_f` | 7.354 | **7.69** | 7.279 |
| `E_ZPL` | 0.721 | **0.91** | 0.903 |
| `E_vert^abs` | 0.827 | **0.99** | 0.969 |
| `Delta q` | 0.900 | **0.81** | 0.585 |

Note the shape of the failure: at a *single* 256-atom cell we are within 10-20 meV of the
published `E_ZPL` and `E_vert^abs`. It is the extrapolation that destroys the agreement,
because the quantity keeps sliding instead of flattening. `Delta q` is additionally
non-monotonic (0.769 at 1024, 0.722 at 2400), which a converging quantity should not be.

---

## 4. It is not the harness

| check | result |
|---|---|
| relaxation convergence | all sizes, `|F|max ~ 0.018` against a 0.02 eV/A target |
| defect identity | same axial pair, 1.893 A bond, at every size |
| base extensivity `E_host/N - e_bulk` | **exactly 0.000 meV/atom** at all five sizes |
| dead-channel invariant | `h_maj` participation equals N exactly, as it should |

The base branch is a sum of local atomic energies and behaves perfectly. Whatever is
moving is in the correction.

## 5. It is not finite-size physics either

```
d(E_ZPL)/d(ln N)  =  -0.083   -0.092   -0.091   -0.077
```

Constant, i.e. the drift is **logarithmic**. A physical image interaction goes as `1/N` or
`1/L`, either of which would make this derivative decay toward zero. This quantity is not
converging slowly; it is not converging.

---

## 6. The mechanism

`alpha = softmax(l)` is normalised over every atom in the cell. With `k` defect sites at
logit `l_d` and `N - k` bulk sites at `l_b`:

```
alpha_defect = 1 / ( 1 + ((N - k)/k) * e^(-gap) ),      gap = l_d - l_b

  ~ 1        while  N << k e^gap
  ~ 1/N      above it

  crossover:  N* ~ k e^gap
```

**A fixed logit gap cannot prevent dilution. It only buys cell size -- exponentially.**

`defect-example/alpha_dilution.py` measures this directly, inverting the observed shell
weight for the gap it implies. That implied gap is **constant in N to +-0.02**, which is
what makes the softmax picture the right description rather than a plausible story:

| channel | implied gap | `N*` | `alpha` on shell, N = 70 -> 2398 |
|---|---|---|---|
| `e_maj` | 2.25 ± 0.02 | 57 | 0.483 -> **0.023** |
| `e_min` | 2.39 ± 0.02 | 66 | 0.496 -> **0.027** |
| `h_maj` | −0.01 (dead) | 6 | 0.085 -> 0.003 |
| `h_min` | 5.23 ± 0.03 | **1121** | 0.943 -> **0.326** |

Participation ratio grows in proportion to N (11 -> 1017 for `e_maj` at N = 2398): the
weight has spread over the whole cell.

**Three of four channels were already diluted inside the training distribution.** At
N = 398 -- a cell size the model was trained on -- `e_maj` and `e_min` hold only 0.126 and
0.145 of their weight on the defect shell. `h_min` is the one still localised there
(0.742), and its crossover at `N* = 1121` sits in the middle of the ladder, which is why
the observables move fastest between 1024 and 2400.

---

## 7. What this changes

The forward plan records the requirement as *"the logit gap keeps the correction
extensive."* That is qualitatively right and quantitatively incomplete. The condition is

```
gap  >>  ln(N / k)
```

which **depends on the cell size you intend to use**. At `k = 6`:

| gap | usable up to |
|---|---|
| 2.4 | ~70 atoms |
| 5.2 | ~1100 atoms |
| 7.5 | ~11,000 atoms |
| 9.7 | ~100,000 atoms |
| 12 | ~10^6 atoms |

Because the requirement is logarithmic it is **cheap**: going from gap 2.4 to 12 buys a
factor of ~15,000 in usable cell size. Nothing in the current objective asks for a gap at
all, so the optimiser had no reason to produce one.

Worth noting for contrast: the NEP baseline has no analogous failure mode, because it
trains three separate models with the electronic state baked into the atomic species. It
has no attention and no normalisation, so nothing to dilute. Our single-model,
carrier-conditioned design is what introduces the pooling, and therefore this risk.

---

## 8. Options for discussion

None of these has been tested yet.

1. **Reward the gap.** A penalty on the shell-to-bulk logit difference, or equivalently on
   participation ratio, targeting `gap > ln(N_max/k)`.
   *For:* one term, no architecture change, testable in a single training run.
   *Against:* a soft constraint on something that arguably should hold structurally; it
   may fight the energy objective.

2. **Normalise locally.** Softmax over a neighbourhood, or top-k, so `N` never enters the
   denominator.
   *For:* removes the failure by construction.
   *Against:* changes the architecture; introduces a cutoff that has to be justified, and
   a hard top-k is not differentiable in the usual way.

3. **Drop the normalisation.** `Delta E = sum_i g_i u_i` with a bounded gate `g_i` biased
   so bulk sites give ~0.
   *For:* no `1/N` anywhere.
   *Against:* the correction becomes extensive unless the bulk gate genuinely vanishes --
   this trades one failure mode for another and needs its own size test.

4. **Train across cell sizes** with an explicit size-consistency term.
   *For:* puts the property in the objective rather than discovering it afterwards.
   *Against:* most expensive; the dataset's cell sizes span only 286-398 atoms, so the
   lever arm is short unless new structures are generated.

**Suggested order:** (1) first, because it is a one-term change and the logarithmic
requirement means it only has to work approximately. If the gap will not rise under a
penalty, that is direct evidence for (2).

**Acceptance test, whichever is chosen:** `alpha_dilution.py` must show the implied gap
and `alpha on shell` flat in N. It is single points only, ~10 minutes on CPU, so it can
gate any retrain cheaply.

---

## 9. Open questions / what is *not* established

* **The per-channel arithmetic linking dilution to the observed meV drift is not
  measured.** We have established that `alpha` dilutes as predicted, and that the
  observables drift, but not the quantitative chain between them: that needs `u_shell` vs
  `u_bulk` per channel. The available equations are underdetermined without it
  (`E_ZPL` involves `e_min` and `h_min`; the ground-state correction involves `e_maj` and
  `h_min`).
* **Whether this caps accuracy at training sizes.** If the correction at ~300 atoms is
  already largely a bulk average, that plausibly contributes to `RMSE_dE` plateauing
  around 20 meV -- but this is a hypothesis, not a measurement.
* **Whether a large gap is compatible with the energy fit.** Forcing localisation may
  simply move the error elsewhere. Only a training run answers this.
* **`Delta q` non-monotonicity** (0.769 at 1024 -> 0.722 at 2400) is unexplained. It may
  be the same dilution acting on forces, or a distinct relaxation-path issue.
* All numbers here come from **one model and one seed**. Seed-to-seed spread in the logit
  gaps was large in earlier stages, so `N*` is likely to vary substantially between seeds.

---

## 10. Reproducing

```bash
cd defect-example

# full convergence ladder (hours on CPU, much faster on a free GPU)
MODEL=~/runs/b_128ch_L1_s1/b_128ch_L1_s1.model ./run_size_convergence.sh

# the diagnosis: alpha vs cell size, single points only, ~10 min on CPU
python alpha_dilution.py --model ~/runs/b_128ch_L1_s1/b_128ch_L1_s1.model --device cpu
```

Outputs: `size_convergence.json`, `size_convergence.png`.
Full working notes: `defect-example/HANDOFF.md` §28 (the test) and §29 (the diagnosis).
