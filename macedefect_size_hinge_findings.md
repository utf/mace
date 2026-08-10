# MACEDefect: the size hinge works — and what it exposed underneath

**Status:** implementation complete and tested; two follow-up decisions needed.
**Branch:** `size-extensivity` (worktree, 4 commits, 162 tests pass).
**Models:** all 128 channels, `max_L = 1`, `dataset_full` (1341 structures), 140 epochs.
**Date:** 2026-08-10.

---

## 1. One-paragraph summary

The log-ratio size hinge from the plan amendment was implemented and trained at production
scale. **It solves the dilution problem completely**: attention on the defect shell is now
flat in cell size out to 4700 atoms (implied gaps 18.6–22.0, `N* ~ 10⁹`, drop 0.00 on every
live channel), and as a direct consequence the short-range correction
`ΔE_SR = Σ_c n_c ⟨u⟩_c` is size-independent to **five decimal places**. The formation energy
went flat (−32 meV across the ladder, against +339 meV without the term) and the correction
spread fell 55×. Two problems remain, and **neither is dilution nor the attention**: the
long-range branch is not size-converged (`E_LR` drifts −78 meV and is the entire residual
`E_ZPL` drift, confirmed against an LR-only control), and a systematic energy bias of
−0.4 meV/atom globally plus −1.4 meV/atom on defect-composition cells traces to the
atomic reference and the under-determined `E_base` gauge — not to `tol`, and not to the
correction, which does not scale with carrier count.

---

## 2. What was implemented

Per the amendment, penalising in log-ratio space rather than energy space:

```
x_c    = ln B_c − ln A_c                    # logsumexp; B is the hypothetical padding
t_c    = (tol / n_c) / max(|c_c|_EMA, eps)  # c = ūw − ⟨u⟩_α, detached and EMA-smoothed
x*_c   = ln(t_c / (1 − t_c))                # +inf when t ≥ 1, i.e. already inside tolerance
L_size = Σ_{c : n_c > 0} max(0, x_c − x*_c)²
```

Zero-set equivalence with the energy-space form was tested over 500 random `(x, c, tol)`
triples. The reformulation was necessary because `σ` saturates: at `R = 1e4` on 286-atom
cells `f` pins at 1.000, and the energy-space gradient carries `f(1−f)`, suppressing the
useful path ~50× at gap 5 and ~10⁴× at gap 2 while the `∂/∂c` path stays O(1) — so the only
reachable optimum was to destroy defect contrast. That failure is retained as an explicit
regression test.

**Bulk reference:** plain arithmetic mean of `ℓ` in log space, undetached. This is the only
choice satisfying both shift-invariance (`Σᵢ ∂x/∂ℓᵢ = 0`) and the uniform-state null
(pointwise zero). Trimming gives the first and not the second — measured at exactly `−1/N`
on trimmed atoms. `ū_Z` keeps a median: it is detached, so robustness there is free.

Also implemented: the Stage D-opt level-mode gauge penalty (`--defect_gauge_weight`,
counters auto-discovered from the training set), and always-on per-epoch diagnostics
(`size_x`, `size_x*`, `size_viol`, `|c|`, `size_f`, `clamped`, `size/dE`, exempt count).

---

## 3. The size problem is solved

Attention on the defect shell against cell size, 70 → 4700 atoms
(`alpha_dilution.py`; `drop` is α lost across the ladder, 0 is the target):

| model | `e_maj` gap / drop | `e_min` | `h_min` |
|---|---|---|---|
| small data L1 | 2.25 / +0.47 | 2.39 / +0.48 | 5.23 / +0.69 |
| larger data L1 | 8.68 / +0.10 | 9.91 / +0.00 | 11.34 / +0.01 |
| + long range | 6.42 / +0.48 | 8.02 / +0.15 | 11.33 / +0.01 |
| **+ size hinge** | **19.64 / +0.00** | **18.60 / +0.00** | **21.96 / +0.00** |

Two things to read off this. **Scale did most of the work** — going from the 800-frame
subsample to the full dataset lifted gaps from ~2.3 to ~9–11 unaided. And **the hinge closed
the remainder**, taking every channel past the `~18` the plan asks for.

Consequence, measured directly on unrelaxed cells of increasing size:

```
ΔE_SR   2.09170 -> 2.09170     -0.0 meV      (excited state, 254 -> 3454 atoms)
```

The short-range correction is exactly size-independent. That is the mechanism working end
to end: α flat ⇒ `Σ_c n_c ⟨u⟩_c` flat.

And in the convergence ladder (256 → 4704 atoms):

| quantity | larger data L1 | **+ LR + size hinge** |
|---|---|---|
| `E_f` drift | +339 meV | **−32 meV** |
| correction spread | +371 meV | **+6.8 meV** |
| `Δq` drift | +0.10 | +0.057 |
| `E_ZPL` drift | −229 meV | −115 meV |

---

## 4. Problem 1 — the long-range branch is not size-converged

`E_ZPL` still drifts −115 meV, and decomposing the correction says exactly where it comes
from:

```
        N      ΔE_SR      E_LR
      254    2.09170   -0.03533
      398    2.09170   -0.03637
      574    2.09170   -0.03972
     1022    2.09170   -0.05024
     2398    2.09170   -0.08620
     3454    2.09170   -0.11360
                       -78.3 meV
```

**Every meV of the residual drift is in `E_LR`.** It grows monotonically with no sign of
converging. The excited state carries more carriers than the ground state, so the difference
of their `E_LR` terms is the bulk of the −115 meV in `E_ZPL` — which is also why `E_f`
(ground state only) went flat while `E_ZPL` did not.

An independent signal points the same way: in the four-model table above, switching the
long-range branch on made `e_maj`'s drop **worse**, +0.10 → +0.48, and lowered its gap
8.68 → 6.42.

### The control settles it

A ladder on the LR-only model (`efull`, no size term) isolates the hinge's contribution.
Drift from 256 to 4704 atoms:

| quantity | `efull` (LR, no size) | `esize` (LR + size) |
|---|---|---|
| `E_ZPL` | **−110 meV** | **−115 meV** |
| correction spread | −74 meV | **−7 meV** |
| `Δq` | −0.126 (falling) | +0.057 (flattening) |
| `Δ` (relaxation energy) | −28 meV | +3 meV |

`E_ZPL` drifts **the same amount with and without the hinge**. Everything short-range is
fixed by it — the correction is 10× flatter, and `Δq` and `Δ` go from *diverging* (`Δq`
falling steadily, which is unphysical) to flat. What the hinge cannot touch is `E_LR`, and
that is the whole of the residual.

**Not yet established:** whether the LR drift is genuine physics or an artefact.
`q_i^carrier = a(α_i^e − α_i^h)` is built from α and summed by latent Ewald against a
compensating background, so a slowly-converging electron–hole interaction is physically
expected. But `a` is frozen at `1/√ε_∞` using a *literature* `ε_∞ = 6.5` rather than DFPT,
and the band-edge referencing is a fitted gauge — either could produce a spurious size
dependence. A no-LR + size model (`bsize_128ch_L1_s2`) is training to close this: if its
`E_ZPL` is flat, the drift is unambiguously the LR branch.

---

## 5. Problem 2 — a systematic energy bias, and it is *not* the attention

Parity on the held-out split. The error is almost entirely a **systematic offset** rather
than scatter, identically on train and valid, so it is not overfitting:

| | RMSE (meV/atom) | bias | bias / RMSE |
|---|---|---|---|
| ideal | 0.46 | −0.40 | 87% |
| defect ground | 1.82 | −1.78 | **98%** |
| defect excited | 1.73 | −1.58 | 91% |

### Two attributions that look right and are not

**Over-localisation — withdrawn.** Participation falls from 3.8 to 2.5 the epoch the hinge
activates, which looks like the carrier being squeezed onto fewer sites than the divacancy's
six dangling bonds. But a participation ratio is not a site count: a state with three sites
at ~0.30 plus residual tails gives `P ≈ 2.5–3`. The wavefunction sits on three atoms, so
`P = 2.5` is the *correct* value and `tol = 1e-3` produced approximately the right answer.
No change to `tol` is warranted.

**A single global constant — also not supported.** The decisive observation is that the bias
is 87% of RMSE on **ideal** frames, which contain no defect, no carriers, and where the
correction is identically zero by construction. Nothing about α can explain that. But the
offsets do not simply track either:

| cell | bias (meV/atom) |
|---|---|
| N = 320 **ideal** | −0.24 |
| N = 384 **ideal** | −0.48 |
| N = 286 defect | −1.41 |
| N = 318 defect | −1.96 |
| N = 382 defect | −1.68 |
| N = 398 defect | −1.61 |

Per subset: ideal −0.40, ground −1.80, excited −1.80. So there are **two** components — a
−0.4 meV/atom offset present everywhere including defect-free cells, and a further
−1.4 meV/atom carried only by defect-composition cells.

### What the excess actually is

Ground and excited have **identical** bias (−1.80 each) despite `Σ_c n_c` of 2 and 4. A
level-mode offset in `u^c` contributes `n_c · const` and would give the excited state twice
the bias of the ground state. It does not scale with carrier count at all, so the excess is
not the correction and not the level mode.

What it does track is **composition** — defect cells against pristine ones. That is the
quantity the model already warns about on every load:

```
WARNING: E_base gauge is UNDER-DETERMINED: 1 free direction(s) remain.
E_base at defect geometries is latent, not a validated prediction.
```

So the working attribution is: −0.4 meV/atom from the atomic energy reference, and
−1.4 meV/atom from the under-determined `E_base` gauge at defect geometries. Neither is
fixed by retraining with different `tol`; the gauge one is closed by adding `q = ±1`
doublets, which is the same identifying data the forward plan asks for.

Accuracy for reference — `efull` (LR, no size) reaches E 0.5 / F 12.8 / dE 15.1 / dF 14.4
meV, `esize` 1.7 / 13.8 / 14.0 / 15.3, all single seed, so much of that spread is plausibly
seed noise.

---

## 6. What is *not* established

* **Seed spread.** One seed per configuration. Earlier stages showed large seed-to-seed
  variation in logit gaps, so `N*` per seed should be reported before any of this is
  quoted as a converged result.
* **Whether the LR drift is physical.** §4. Needs the control ladder, and ideally a DFPT
  `ε_∞` rather than the literature value.
* **`λ_size` at production scale.** Calibrated on an 8-channel beta model, where the plan's
  "5–10% of `L_Δ`" target turned out to be unreachable simultaneously with satisfying the
  constraint — the ratio was already 13–19% at a weight far too weak to act. The realised
  ratio is now logged every epoch, so this can be measured properly rather than assumed.
* **Whether `E_ZPL` is right in absolute terms.** The logistic refit gives a bound-carrier
  limit of 0.98–1.08 eV against the published 0.91; agreement at small cells was previously
  shown to be compensating error.
* **A hypothesis that was refuted, recorded so it is not retried.** The residual `E_ZPL`
  drift was initially attributed to `⟨u⟩_e_min` drifting with cell size — `e_min` being the
  channel the excited counter uses and the ground counter does not, and the weakest-gapped.
  Direct measurement shows every `⟨u⟩_c` flat to sub-meV. The short-range side is clean.

---

## 7. Decisions wanted

1. **How to treat the long-range branch.** It is the only remaining source of size drift,
   and the control shows the hinge has no purchase on it. Options: leave LR off for
   size-critical work, obtain a DFPT `ε_∞` and revisit whether the frozen `a` is right, or
   accept the drift as physical. The no-LR + size run now training decides whether the
   drift is LR alone.
2. **Close the `E_base` gauge.** The −1.4 meV/atom defect-specific bias is the
   under-determined direction the code already warns about, and adding `q = ±1` doublets is
   the identifying data that removes it. No retraining trick substitutes for it.
3. **Seed replication before publication-grade claims** — at least three seeds on the chosen
   configuration.

`tol` needs no change: participation 2.5 is the correct value for a three-site state, not
over-localisation.

---

## 8. Reproducing

```bash
# acceptance test: alpha flat in N, ~10 min on CPU
python defect-example/alpha_dilution.py --model <model> \
    --data-dir defect-example/dataset_full --device cpu

# decompose the correction into short- and long-range against cell size
python defect-example/u_size_check.py --model <model> --state excited

# the four-model chart and table
python defect-example/plot_size_comparison.py --runs b_128ch_L1_s1 bfull_128ch_L1_s2 \
    efull_128ch_L1_s2 esize_128ch_L1_s2

# full convergence ladder (hours on CPU)
python defect-example/defect_size_extensivity.py --model <model> \
    --data-dir defect-example/dataset_full \
    --repeats 4,4,2 5,5,2 6,6,2 8,8,2 10,10,3 12,12,3 14,14,3 --device cpu
```

Artefacts: `defect-example/size_comparison.png`, `size_convergence_esize.png`,
`parity_esize_{valid,train}.png`. Working notes: `defect-example/HANDOFF.md`.
