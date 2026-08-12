# The seed anneal gate decides which seeds localise — and neither setting fixes it

Three things landed together: the controlled gate test, the G arm, and a valid isotropic
size ladder for `perov_D_s3`.

---

## 1. The gate control — the measurement that was missing

Seed 1 collapsed onto Cl in the Z and D arms, and was blamed on the site-resolved anneal
gate. That was a cross-arm inference: seed 1 had never been run short-range-only under the
current code. It has now, on both gates, everything else identical.

```
                  partic    species        valid E / F
perov_sr1_gap       1.96    Pb 1.000        3.6 / 15.3
perov_sr1_site     46.89    Cl 0.997        4.0 / 15.1
```

Same seed, same data, **no long-range branch at all**. The site gate loses it; the gap gate
keeps it. The hypothesis is confirmed as a controlled result, and it explains the Z and D
failures — both ran the site gate.

Mechanism, from the trajectories: the gap gate hands over smoothly and early, while the site
gate holds gamma at full strength and then drops it as a step. On seed 1 under the site gate
gamma sat at 1.00 through epoch 10 and fell to 0.01 by epoch 14; that seed's attention ran
away at epochs 12-13, inside the window. A gate that waits for site structure to appear
releases abruptly when it appears late.

## 2. The G arm — the gate is a trade, not a fix

Long-range branch held out until epoch 30, then live for 30 epochs, on the gap gate.
Four seeds.

```
        partic   species                    valid E / F
G s1      1.99   Pb 1.000                    2.9 / 11.6
G s2      2.00   Pb 1.000                    3.2 / 11.6
G s3     31.4    Cs 0.596  Pb 0.387          2.9 / 11.7
G s4     56.3    Cs 0.441  Cl 0.498          3.0 / 12.6

nolr reference                               3.1 / 12.0
```

Seed 1 recovers, exactly as the control predicts. But **seeds 3 and 4 fail here having
succeeded under the site gate** (D s2/s3 and the SR-only seeds 2/3/4 all localised). So the
gate does not fix the problem, it changes which seeds survive it. 2 of 4 either way.

Their failures also differ in kind. Every earlier collapse settled on a single sublattice —
Cs at participation ~16, or Cl at ~47. G s3 and s4 are smeared across *two* species
(Cs 0.60 / Pb 0.39, and Cs 0.44 / Cl 0.50). No previous run did that. Whatever selects a
sublattice is weaker here, which is consistent with a schedule effect rather than an
energetic preference.

**Every G seed beats the short-range reference on accuracy, including the two with wrong
physics.** Same warning as D seed 1: no aggregate metric sees this failure.

## 3. Isotropic size ladder, `perov_D_s3`

Cell shape held fixed, so `E = E_inf + k/L` is a valid fit. The earlier five-point ladder
mixed shapes (2,2,2 / 3,2,2 / 3,3,2 / 3,3,3 / 4,3,3) and its -16 meV reversal was a
cell-shape artefact, not noise.

```
    N       L (A)    eps_opt        k     = -8.610 eV.A
  640       28.62    -4.1284        E_inf = -3.8285 eV
 2160       42.94    -4.0319        max residual 2.9 meV
 5120       57.25    -3.9770        largest cell 148 meV from the limit
```

Monotone, and a clean `1/L` fit to 2.9 meV over a doubling of L. `base` is exactly
0.000 meV/atom at every size, so the short-range branch is size-extensive and all of the
drift is the long-range term — which is what that term exists to produce.

The model is **not converged at 5120 atoms**: still 148 meV from the extrapolated limit. The
fit now gives a defensible `eps_opt(infinity) = -3.83 eV`, which the mixed-shape ladder could
not.

For context, `carrier^2` measured independently gives `k = -4.58 eV.A`. `eps_opt` includes
relaxation and the full state-to-state difference, so a larger coefficient is expected; same
sign and order is the consistency check that matters.

## 4. Method correction carried into this batch

`PYTHONPATH` never pinned the code revision: python puts the launch directory first on
`sys.path`, and this project is driven from a worktree containing `mace/`. Every earlier
attempt to compare revisions by pointing `MACE_REPO` at another checkout silently ran the
working tree. (It is **not** the PEP 660 editable finder, which sits after `PathFinder` and
is never consulted.)

Redone properly from `/tmp`, with both trees asserted before each run: the short-range
forward is **identical** between revisions (0.0e+00 on energy, base, correction, delta_sr,
forces, alpha, u, logits, gap), the restored gap anneal is **identical**, and the size-hinge
penalty is unchanged. The one real difference is the threshold at `|c| -> 0`, where the old
code exempts and the new applies the guard — which cannot affect a healthy channel.

Earlier runs are unaffected: `MACE_REPO` and the launch directory both pointed at the
worktree, so the intended code ran. `train_defect_model.sh` now asks python where it
resolved `mace` from and aborts if that disagrees with `MACE_REPO`.

## 5. Where this leaves it

* The anneal schedule, not the long-range branch, decides which seeds localise on this
  dataset. Both gates give 2 of 4.
* The branch itself still does not destroy a settled correct answer: every seed that entered
  epoch 30 localised stayed localised, in D and in G.
* Neither gate is the answer. The site gate's measure is right in principle and its schedule
  is wrong; the gap gate's schedule is smooth but its measure cannot tell a species gap from
  a site gap. A gate on the site-resolved measure with a bounded per-epoch decay rate is the
  obvious next thing to try, and has not been tried.
* Read against section 10 of the diagnosis, none of this is surprising: with no paired
  frames the alpha/u degeneracy is essentially unbroken, so attention is decided by the
  seeding schedule rather than by the data. That is a property of the dataset, and SiC's
  per-atom delta-force supervision is what can actually pin it.
