# E0 — the site signal is in the data. PASS on both seeds.

The gating diagnostic of the carrier-localisation phase. A geometry-only base was trained on
every frame whose canonical counters are all zero (pristine + V_Cl0), then evaluated on
V_Cl+ frames it has never seen. If the carrier leaves a site-resolved fingerprint in the
labels, the force residual `dF_i = F_DFT - F_base` should concentrate on the two
under-coordinated Pb. It does.

Per the decision tree: **E0 passes, so the unpaired data do contain the information and joint
training has been destroying it.** Proceed to the architecture work (components S and H).
The "stop, record paired single-points" branch is not taken.

## Verdict

| | seed 1 | seed 2 |
|---|---|---|
| shell/bulk-Pb \|dF\| ratio, **charged** | 3.93 | 4.01 |
| shell/bulk-Pb \|dF\| ratio, **null** | 2.29 | 2.06 |
| excess over null | +1.64 | +1.95 |
| shell energy fraction, charged | 0.318 | 0.320 |
| shell energy fraction, null | 0.156 | 0.133 |
| ratio on high-confidence frames only | 4.14 | 4.29 |
| **verdict** | **PASS** | **PASS** |

The two seeds agree to 0.08 on the headline ratio, so E0 is decided rather than ambiguous —
the outcome the plan reserves for disagreeing seeds does not arise.

Two of 79 atoms — 2.5% of the cell — carry **32% of the total squared force residual**. The
null puts 14-16% there, which is the honest baseline: a vacancy disturbs its neighbourhood
whether or not a carrier is present, and about half of the raw shell concentration is that
geometric effect rather than carrier signal.

## The maps

`|dF|` against distance to the vacancy, median over atoms, eV/A (seed 1; seed 2 within a few
percent everywhere):

| distance | n | charged | null |
|---|---|---|---|
| 1.5-2.5 A | 64 | 0.129 | 0.034 |
| **2.5-3.5 A** | 3827 | **0.230** | 0.028 |
| 3.5-4.5 A | 9463 | 0.119 | 0.017 |
| 4.5-5.5 A | 3212 | 0.073 | 0.017 |
| 5.5-7.0 A | 23583 | 0.041 | 0.013 |
| 7.0-9.0 A | 28848 | 0.028 | 0.013 |
| 9.0-12 A | 13990 | 0.016 | 0.011 |
| 12+ A | 1086 | 0.012 | 0.012 |

The charged map falls by a factor of **20** from peak to far field; the null falls by 2.8 and
is essentially flat beyond 4 A. Both converge to the same ~0.012 eV/A floor far from the
defect, which is the base's intrinsic force error — exactly what the null was for.

The peak sits in the 2.5-3.5 A bin, and the shell Pb sit at a median 2.67 A from the located
site. The signal peaks precisely on the atoms the physics says host the hole.

## Per-species

Mean `|dF|`, eV/A (seed 1 / seed 2):

| species | charged | null |
|---|---|---|
| Pb | 0.106 / 0.115 | 0.026 / — |
| Cl | 0.056 / 0.061 | 0.017 / — |
| Cs | 0.048 / 0.052 | 0.013 / — |

Pb carries roughly twice the residual of Cl and Cs. That ordering is the physically expected
one — the VBM is Cl 3p / Pb 6s antibonding and the hole binds to Pb — and Cs, the species that
failing models most often collapse onto, carries the least. The data prefer Pb; it is the
optimisation that has been going to Cs.

## What this does not show

**The energy residual is not usable as evidence.** Charged frames give a mean residual of
-3.58 eV (seed 1) and -3.39 eV (seed 2), and the null gives -0.24 eV and +0.003 eV. The
0.24 eV seed-to-seed spread on the null is the `E_base` gauge under-determination the model
warns about on every load: the absolute base energy at defect geometries is latent, so its
offset is seed-dependent. Force residuals are derivatives of that surface and a constant
offset cancels, which is why the maps above are the deliverable and the energy numbers are
recorded but not leaned on.

**PASS means the information exists, not that it is easy to extract.** E0 says a
site-resolved signal survives in the unpaired labels and is visible to a frozen base. It says
nothing about whether joint training can find it — that is precisely the M1 laundering
hypothesis, and E1/E3 test it.

## Method notes

* **Base**: MACEDefect trained on 1616 n=0 frames, 140 epochs, LR branch off, corrections
  disabled. Final validation 2.9 meV/atom and 10.9 meV/A (seed 1), 2.3 and 10.6 (seed 2).
  These checkpoints are also **Stage A** for the later arms — train once, freeze, reuse.
* **Correction inertness verified, not assumed**: `dE_SR` is exactly 0 at n=0, energy equals
  base energy, and none of the 38 correction-head tensors receives gradient. Separately, the
  four output regularisers (`u_l2`, `p_l2`, `qhost_l2`, `zn_l2`) act on model outputs and at
  their defaults would push gradient into the shared trunk on carrier-free frames; all are
  zeroed, as are the size hinge and seed anneal.
* **`base_forces` is geometry-only**, measured: relabelling a charged frame as neutral moves
  it by 6.0e-07 eV/A against a same-input numerical noise floor of 6.1e-07, and both collapse
  to ~1.4e-15 in float64. The reading the whole analysis depends on is sound.
* **Null control held out of checkpoint selection.** Had the null frames been in the training
  validation set, the base would have been mildly tuned to them, deflating the null and
  inflating the excess — a bias pointing at exactly this conclusion.
* **Vacancy located on 1047/1047 charged and 60/60 null frames**, by Cl-to-bridge assignment
  (`vacancy_site.py`). Restricting to the 1030 frames where that identification is most
  confident *raises* the charged ratio (3.93 to 4.14, 4.01 to 4.29), so the result does not
  rest on marginal frames. The vacancy position is used for binning and classification only
  and never reaches the model.
* Code `eff45eb`; raw per-frame dumps in `~/runs/e0_maps_s{1,2}.npz`.
