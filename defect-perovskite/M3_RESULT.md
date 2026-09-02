# M3 — the flip never reached t(d). R-C.

12 A/B cells, 24 charged frames each. `t'` is the exact gradient of `H_ab` w.r.t. both hub
positions projected on the pair axis — real frames, superseding F2's zero-feature profile
slope.

| arm | n | mean \|t'\| | range | mean \|H_ab\| | corr(N_eff, d) |
|---|---|---|---|---|---|
| **ON** | 6 | **0.0443** | 0.017–0.086 | 0.041 | **−0.010** |
| OFF | 6 | **0.0623** | 0.021–0.112 | 0.052 | +0.113 |

Target for both R-A and R-B: `|t'|` → 0.10–0.20 eV/Å under ON, OFF at the prior.

## Verdict: R-C

**`|t'|` did not move.** ON is *lower* than OFF (0.0443 vs 0.0623), and both sit far below the
0.10–0.20 eV/Å that R-A and R-B alike require. The per-seed ranges overlap almost completely
(ON 0.017–0.086, OFF 0.021–0.112), so the honest statement is that the two arms are
**indistinguishable in `|t'|` and both are below target** — not that ON is worse.

**Not R-A either.** `corr(N_eff, d)` is flat in both arms (−0.010 ON, +0.113 OFF). R-A requires
N_eff to fall as d falls — binding when the pair dimerises — and that is not there. Per-seed
values scatter around zero in both arms (ON −0.197…+0.151, OFF −0.108…+0.412).

So the A/B's fit gain (axial_red +0.348 → +0.588) came through channels other than the hub
coupling. On the §5 branch logic: **find why the flip never reaches t(d) — gradient path,
initialisation, decay floor — before any architecture claim. s+p is unearned.**

This also retires the named-failure reading the A/B table appeared to license on its own: R-B
required `|t'|` to have moved, and it did not.

## One observation not in any of the three readings

`corr(lambda, d)` is consistently **negative** across both arms (−0.31 to −0.57, ten of twelve
cells), i.e. the defect level falls as the pair separates, so `Delta_bind` rises with d. R-A
expected the opposite (deeper at short d). Being equally present in ON and OFF, it is not a
sign effect and not evidence for or against the flip — but it is the reverse of the physical
expectation and is unexplained.

## Candidate causes for R-C, in the order we would test them

1. **Decay floor.** `t = t_min + softplus(B)` with a per-species decay length floored at 0.3 Å.
   If the optimiser sits near the floor, `t'` is pinned by the decay prior regardless of what
   the energy channel asks for. The realised profile slope was −0.033 and the measured real
   frame values (0.044/0.062) are the same order — consistent with prior-dominated.
2. **Gradient path.** `dE_SR = s_c n_c (Lambda + mu_c)`; `Lambda` reaches `t` only through the
   eigenvalue, and with a smeared near-degenerate spectrum that gradient can be small.
3. **Initialisation.** The amplitude was calibrated to place `t(2.85 A) = 0.5 eV`; the
   distance-derivative at 5.6 Å was never a calibration target.

## Not done

* The F2 six were not re-run under M3, so **F2's closure-ratio update is outstanding** — its
  `t'` is still the profile slope. `m3_curves.py` takes any model list; it is one argument.
* M2 attribution (which elements carry the model's d-trend, both arms).
* C1 post-μ-fix with the full three-component gate.
