# `a = 0` control: `carrier²` is indicted, and removing it is necessary but not sufficient

Measurement 1 of the feature-collision plan. `a = 0` via `eps_inf = 1e12`, three seeds,
22 epochs. Same modules constructed and same RNG stream as A0, so the initialisation
confound is controlled: `_mlp` draws its first layer from the global RNG, so enabling the
long-range branch changes the trunk and readout init at a fixed seed, and A0 could not be
compared with `perov_nolr_s1` directly.

Reproduce: `ARM=Z SEED=<n> EPOCHS=22 defect-perovskite/run_arm.sh`, then
`defect-perovskite/alpha_audit.py --model ~/runs/perov_Z_s<n>/perov_Z_s<n>.model`.

## Final attention, real V_Cl+ frames

```
Z s2   partic 1.94-1.99   Cs 0.000   Pb 1.000   Cl 0.000   top Pb 0.584/0.416, 0.531/0.469
Z s3   partic 1.95-2.00   Cs 0.000   Pb 1.000   Cl 0.000   top Pb 0.583/0.417, 0.525/0.475
nolr   partic 1.96-2.00   Cs 0.000   Pb 1.000   Cl 0.000   top Pb 0.568/0.432, 0.511/0.489

Z s1   partic 46.3-46.6   Cs 0.001   Pb 0.000   Cl 0.999   uniform (1/47 = 0.021)
A0     partic 15.2        Cs 1.000   Pb 0.000   Cl 0.000   uniform (same seed as Z s1)
```

**Two of three seeds reproduce `perov_nolr_s1` to within noise** — the two under-coordinated
Pb at 2.6 and 2.9 A splitting the weight ~55/45. A0, at the same initialisation with
`carrier²` on, never came within a factor of five of this in nineteen epochs.

**Seed 1 collapses onto Cl** — a *different* sublattice from the Cs that A0 chose. A target
that varies by seed within an arm is symmetry-breaking amplified by rich-get-richer softmax
dynamics, not a mechanism selecting a particular species. Seed 1 built `gap_site` to 0.455 by
epoch 10, comparable to the successful seeds, then **lost** it (0.276) as the attention ran
away onto Cl: it had the site information and did not act on it.

## Trajectories, `partic` (h_maj)

```
epoch     0     2     4     6     8    10    12    14    16    19    21
Z s1    6.7   7.3   8.2   9.5  11.7  11.9  22.0  44.7  46.2  46.8  46.9   -> Cl
Z s2    4.3   2.4   2.4   2.9   2.7   2.8   2.8   2.9   2.5   2.7   2.0   -> Pb
Z s3    4.6   4.3   7.3   3.2   4.5   4.9   4.8   4.7   5.0   4.8   2.0   -> Pb
A0     11.4  12.5  12.8  13.1  13.6  14.0  14.3  14.6  14.8  15.2    —    -> Cs
nolr    6.8   8.5   2.6   4.6   4.2   4.4                                 -> Pb
```

## The size hinge sharpens but does not steer

The penalty enters at epoch 20. Both Pb seeds tightened immediately; the Cl seed did not move.

```
              partic 19 -> 20      gap_site 19 -> 20
Z s2          2.66 -> 1.98         0.79 -> 2.36
Z s3          4.85 -> 1.93         0.80 -> 1.93
Z s1         46.8 -> 46.8          0.29 -> 0.28
```

Consistent with the earlier guard result: the hinge helps attention already on the right site
and cannot rescue one that is not.

## Sublattice choice does not track the delocalisation pressure

If `carrier²` selected by its own pressure it would prefer the **largest** sublattice — Cl,
47 atoms, lowest `Σα²`. A0, with `carrier²` on, chose Cs at 16. So the pressure explains why
localisation is unreachable, not which sublattice wins.

## Accuracy

```
          valid E (meV/atom)   valid F (meV/A)
Z s1              3.9               17.5
Z s2              3.8               17.1
Z s3              3.8               16.8
nolr              3.1               12.0
```

All worse than `nolr`, as expected: `a = 0` removes the long-range contribution entirely
while keeping its modules. **This arm is a diagnostic control, not a candidate model.**

## Conclusion

* `carrier²` is **indicted**. Its removal is what makes the correct Pb-shell solution
  reachable at all.
* Removal alone is **not sufficient**: 1 of 3 seeds still collapses.
* So the indicated work is the `E_LR = E_periodic[Q_total] − Σ_c E_isolated[Q^c]`
  repartition **plus** a robustness measure for the soft-attention window. The `a`-ramp held
  in reserve is the natural candidate, since seed 1's failure is a runaway during exactly
  that window.
* Read against section 10 of the diagnosis: this dataset has no paired frames, so the
  `alpha`/`u` degeneracy is essentially unbroken and `nolr` localising at all is marginal.
  A 2-of-3 success rate here is consistent with that fragility and should not be read as a
  property of the fix.
