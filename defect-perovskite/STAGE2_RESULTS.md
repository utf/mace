# Stage 2 — Edit 3 (bounded elements). One gate passes, one fails, and they are the same gate.

Same harness, same six seeds, same data as Stage 1. The only change is Edit 3: bounded
elements over Harrison scales, `t_min` removed, the learned per-species decay replaced by one
fixed global length.

---

## The two gates

**Superatom gate: passes, decisively.** On a *defect-free* cell the split fraction
`(λ₂−λ₁)/(λ_last−λ₁)` falls from Stage 1's **0.76** to **0.10–0.14**, against an even-spacing
reference of 0.20. Edit 3 does exactly what it was designed to do.

**Force parity gate: fails, badly.** Stage 1 sat at `rmse_all` 30.4 ± 6.6 (nominal `Z`) and
axial_red +0.627. Stage 2 gives `rmse_all` ≈ 50 and axial_red ≈ +0.02 — the head is barely
distinguishable from the frozen base on the axial component (299.6 against the base's 305.2
meV/Å).

## They are not two results. They are one.

Both arms, 12 seeds, everything held fixed but the seed:

| arm | n | axial_red | rmse_all | N_eff | null ratio | split fraction |
|---|---|---|---|---|---|---|
| learned `Z` | 6 | +0.142 ± 0.159 | 47.2 ± 4.0 | 34.5 | 0.880 | 0.336 |
| nominal `Z` | 6 | +0.170 ± 0.199 | 48.5 ± 4.7 | 36.1 | 0.890 | 0.343 |

**`corr(axial_red, split_fraction) = +0.987`** over the 12 seeds, and the distribution is
strictly bimodal — no seed lands between:

| group | n | split fraction | axial_red | rmse_all |
|---|---|---|---|---|
| bands | 8 | 0.10–0.14 | **+0.030** | 50.8 |
| superatom | 4 | 0.74–0.78 | **+0.408** | 42.0 |

The bounded form does not close the escape route completely — four seeds found it anyway —
and **the seeds that found it are exactly the seeds that fit**. Where the spectrum came out
as bands, the fit collapsed to base level. There is no seed with a physical spectrum and a
good fit.

**Edit 3 also dissolves the `Z` pathology.** Learned and nominal `Z` are now
indistinguishable (+0.142 against +0.170, well inside the spread), and `Z` shrinks during
training rather than running away. The Stage-1 runaway was an interaction with the unbounded
on-site term, not a property of `Z` itself — so the amendment proposed after Stage 1 matters
much less than it appeared to, and we would not now press for it.

## What Stage 1 was actually doing

The learned per-species decay lengths in the Stage-1 heads, against the plan's own
calibration target of `t(5.6 Å) = 5–20 %` of `t(2.8 Å)`:

| Stage-1 arm | max decay length | `t(5.6)/t(2.8)` |
|---|---|---|
| ON, learned `Z` | 1.97 Å (up to 3.26) | **60 %** (14–120 %) |
| ON, nominal `Z` | 2.11 Å (up to 3.18) | **50 %** (9–140 %) |
| OFF | 1.97 Å (up to 2.51) | 24 % (9–35 %) |

Two cells have ratios **above 100 %** — hopping that *increases* with distance. Every arm is
outside the target; the Madelung arms are 2–3× outside it.

So Stage 1's +0.63 axial_red was bought with a long-ranged, nearly-flat Hamiltonian: the
near-complete graph whose ground state is the in-phase superatom mode, which is what E2
measured at a 10 Å reach and what the pristine split fraction of 0.76 was reporting all
along. Edit 3 removes that freedom, and the fit goes with it.

**This does not retract Stage 1's other results.** The φ-binned gate, the λ–d sign change and
the `Z` comparison are all *contrasts between arms trained under identical conditions*, and
the escape route was equally available to every arm. What it retracts is any reading of
Stage 1's absolute axial_red as evidence that the head had found physical structure.

## The finding, stated plainly

The head's force fit and its superatom spectrum have been the same object. Confine the
Hamiltonian to physical tight-binding scales and ranges, and what remains fits forces no
better than the base potential it corrects.

This is close to the case §8 reserves as "the first genuinely architectural finding this
design admits" — with the difference that §8 anticipated a *bound-versus-dilution*
disagreement, and what we have is a *fit-versus-spectrum* one. Both are disagreements between
two measured quantities on one model, which is what makes them architectural rather than
procedural.

## What we have not concluded

That the s-only basis is the reason. Edit 4 exists precisely because a single orbital per
atom cannot represent the p-derived valence band this hole lives in, and the SK toy tables
already pass. It is entirely possible that a physical Hamiltonian *with the right basis* fits
these forces and the s-only one simply cannot. Stage 2's result is a reason to run Stage 3,
not a reason to stop.

What would be wrong is to loosen Edit 3's bounds until the fit returns. That buys back the
superatom, and we would be measuring it again in three weeks under another name.
