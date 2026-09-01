# T2 null control, and the R2 localisation screen

> **Amended 2026-09-01.** The R2 section originally reported two H3/anneal seeds as
> "localised" because they crossed the guard's 0.80 null-ratio threshold. That threshold is
> an arming condition for the rollback guard, **not** a localisation criterion, and reading
> it as one was wrong. Those seeds sit at N_eff 40 and 43 on ~80-atom cells -- a state over
> half the cell, the same broad state E2 produced at participation ~37. The corrected tally
> is **0/20**. The recommendation to power a 16-seed anneal study is withdrawn with it.

Two results from 2026-08-31. They point in opposite directions, which is the useful part: the
labels do carry a site-resolved carrier fingerprint (T2), and the spectral head does not
convert it into a localised carrier (R2 H3).

## T2: the axial excess survives an honest null

The old null control could not answer its own question. Its 60 neutral frames have median
d(Pb-Pb) 4.81 A and stop at 5.77 A, while the charged frames sit at 5.49 A and the neutral
training set runs to 6.54 A. "Charged vs null" was therefore partly a comparison between two
different regions of the d axis, and E0's excess rested on those same 60 frames.

Cross-fitting fixes the range without starving any base of the long-d tail: 1191 neutral
defective frames in 4 folds, one diagnostic base per fold trained on the other three, every
frame scored by the base that never saw it. Charged frames are partitioned across folds
(i -> i mod 4) so each is counted once and the pooled rows are not four correlated copies.

The four fold bases came out equally good -- RMSE_F 5.08-5.23 meV/A, a 3% spread -- which
matters because the null is what these bases *fail* to explain. A weak fold would inflate the
null with its own extrapolation error.

|dF| axial residual, by d(Pb-Pb):

| bin (A) | n_null | n_chg | null | charged | excess |
|---|---|---|---|---|---|
| 4.84-5.04 | 217 | 8 | 0.0159 | 0.3782 | x23.79 |
| 5.04-5.24 | 117 | 66 | 0.0232 | 0.2870 | **x12.40** |
| 5.24-5.44 | 99 | 356 | 0.0279 | 0.2825 | **x10.13** |
| 5.44-5.64 | 57 | 293 | 0.0422 | 0.2918 | **x6.92** |
| 5.64-5.84 | 18 | 137 | 0.0422 | 0.3803 | x9.01 |
| 5.84-6.04 | 8 | 73 | 0.0786 | 0.3223 | x4.10 |
| 6.04-6.24 | 2 | 71 | 0.0165 | 0.3134 | x19.03 |
| 6.24-6.44 | 2 | 16 | 0.1248 | 0.3347 | x2.68 |

**Quote the bold rows, not the pooled x18.44.** The null median sits at 4.75 A against the
charged 5.49 A and is densest exactly where its own residual is smallest, so pooling flatters
the ratio. The tail bins are noise: two null frames each, and a median of two points means
nothing. The honest effect size is the x7-12 where both arms are populated.

Coverage is better but still not complete. The null spans 3.68-6.54 A against the old 5.77 A
ceiling; charged frames reach 7.12 A, so above 6.54 A there is still no null at all. The
binned table stops at 6.44 A, but the pooled median does include those frames -- a second
reason to quote bins.

Signed axial mean is **+0.1035 eV/A** charged against **-0.0015** null, so the charged signal
has a coherent direction rather than being symmetric noise. The bonding/antibonding
attribution of that sign is interpretation, not established here.

**E0 recheck.** Against the cross-fit null, hub/bulk-Pb is charged 4.27 / null 1.92 = x2.22.
E0's own null gave 3.93/2.29 = x1.72 (s1) and 4.01/2.06 = x1.95 (s2). So E0's excess survives
the better null and rises slightly. (The "1.9" quoted elsewhere is R0's hub/CAGE ratio, a
different statistic -- an earlier version of this write-up compared against it by mistake.)

Every frame was usable: 298/298 null and 262/262 charged per fold, no vacancy-location skips.

## R2: no head localises the carrier -- 0/20

H2 = site energies and hopping prefactor from first-interaction-block features. H3 = H2 plus
the dangling-orbital sigma term. Anneal = hoppings scaled by s(e) = 4^(1 - e/20), so the band
starts 4x wider than its trained width and narrows. All cells: 50-epoch screen, base frozen
through epoch 30 then released at 0.01x, soft gauge anchor, size loss off (diagnostic only),
`clamped=[0,0,0,0]` throughout so no vacancy label reaches training.

Null ratio at the epoch-30 release (N_eff of the supervised channel over the mean of the three
n_c = 0 channels, which take no gradient and so form a free per-run baseline):

| cell | seeds | ratios at release |
|---|---|---|
| H3 / no anneal | 4 | 0.844, 0.868, 0.869, 0.875 |
| H3 / anneal | 8 | 0.555, 0.581, 0.844, 0.880, 0.880, 0.889, 0.899, 0.931 |
| H2 / no anneal | 8 | 0.808, 0.846, 0.848, 0.850, 0.870, 0.880, 0.898, 0.900 |

**None of these is a localised carrier.** The lowest two -- H3/anneal s5 and s8, at ratio
0.555 and 0.581 -- correspond to N_eff 40.2 and 42.8 on ~80-atom cells, i.e. a state spread
over more than half the cell. That is the same broad state E2 produced at participation ~37,
not a defect level. A carrier on the vacancy pair with ligand tails is N_eff ~ 3-6, a ratio
of ~0.05, an order of magnitude away. Their species mass is being measured in D1; the
expectation is a Cl-sublattice band state rather than anything vacancy-centred.

The 0.80 figure that those two seeds crossed is the guard's ARMING condition -- the point
below which a rollback has something to protect -- and was never a localisation criterion.
The gate used from here on is **N_eff <= 8, null ratio <= 0.15, and a split-off gap much
larger than the smearing T_s = 20 meV**, with the hub gate (the two vacancy-sharing Pb as the
top-2 weights, >= 0.5 jointly, Cs < 0.05) logged as a metric only.

So the honest tally across every head variant and both schedules is **0/20**. H2 vs H3 made
no difference, and the anneal made no qualitative difference either.

**The negative is strong, not inconclusive:** epochs 0-30 had the base fully frozen, so
laundering by a co-adapting base was structurally impossible. Every head had an uncontested
opportunity to localise, with the gradient going nowhere else, and none took it.

**What the heads do instead.** In every arm, the supervised channel develops site energies
with 100-300x the spread of its unsupervised counterparts (std_u 0.3-2.2 against 0.003-0.01)
and a markedly more peaked attention distribution, while remaining spatially delocalised. So
"does the head learn something charge-specific" is not where these arms fail; "does that
become a localised state" is. That dissociation is the subject of D1: a delocalised alpha can
still reproduce a concentrated axial force if eps_i is an MLP over neighbourhood features
with a 5 A reach, because eps_i can be made steeply sensitive to R_hub for every atom whose
features see the hub. Every head tried so far shares that freedom, which is the likeliest
reason H2 and H3 behave identically.

**One real guard finding survives:** both low-ratio seeds eroded measurably after the base was
released (0.555 -> 0.663 and 0.581 -> 0.631 by epoch 49) without tripping the rollback
thresholds, which were set for catastrophic collapse rather than slow drift. The guard needs
tightening regardless of what replaces the head.

## Reading the two together

T2 says the fingerprint is in the labels, across the whole overlap range, at x7-12 the null,
with a coherent sign. R2 says no head tried converts it into a localised state, even handed a
frozen base and an uncontested gradient. The gap is therefore in the head, not in the
information content of the data -- which rules out "the labels don't carry it" and points at
the on-site term's flexibility as the thing to remove.
