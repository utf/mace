# T2 null control, and the R2 H3/noanneal cell

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

## R2, H3/noanneal: a clean negative on localisation

H3 = first-shell features + the dangling-orbital sigma term. 8 seeds, 50-epoch screen, base
frozen to epoch 30 then released at 0.01x, T5 gauge penalty on, size loss off (diagnostic
only), `clamped=[0,0,0,0]` throughout so no vacancy label reaches training.

At the epoch-30 release, all four wave-1 seeds:

| seed | n_eff | active/null ratio | guard |
|---|---|---|---|
| s1 | 64.83 | 0.8751 | not armed |
| s2 | 62.60 | 0.8680 | not armed |
| s3 | 62.90 | 0.8444 | not armed |
| s4 | 63.78 | 0.8691 | not armed |

**H3 does not localise the carrier.** N_eff sits at 62-65 in ~80-atom cells where the hub is
two atoms; the supervised channel is only ~13% more contracted than its own unsupervised
baseline. The trend is real but far too slow: mean ratio went 0.888 at epoch ~10 to 0.864 at
epoch 30, about -0.0012/epoch, so reaching the 0.80 threshold would need ~58 more epochs
against the 20 remaining.

The reason this is a *strong* negative rather than an inconclusive one: **epochs 0-30 had the
base fully frozen**, so M1 laundering was structurally impossible. The head had an uncontested
opportunity to localise, with the gradient going nowhere else, and did not take it. The
failure cannot be blamed on the base competing for the signal.

The guard being disarmed in 4/4 seeds is not a malfunction -- it is the guard correctly
reporting there is nothing to protect. But it does mean epochs 30-50 run with the base
adapting and no rollback, so **anything that improves after epoch 30 is confounded by base
drift and must not be read as the head localising.** The interpretable window is 0-30.

What H3 *is* doing: the active channel's gap widened from 3.35-4.05 to 3.94-4.88 eV while the
nulls stayed flat at 2.4-2.7, with std_u at 0.59-1.48 against ~0.005 in the nulls. The sigma
term produces a strong channel-specific Hamiltonian perturbation that expresses itself as a
global gap opening rather than spatial contraction onto the vacancy pair.

## Reading the two together

T2 says the fingerprint is in the labels, across the whole overlap range, at x7-12 the null.
R2 says this head does not convert it into a localised state even when handed a frozen base.
So the gap is in the head's ability to localise, not in the information content of the data --
which is the more tractable of the two failures, and rules out "the labels don't carry it" as
the explanation.
