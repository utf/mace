# F2 — coherence is not suppressed. Stop; no A/B.

Six localised cells, 24 charged frames each, from the F1 sweep's saved eigenvectors.
`rho_ab = sum_k w_k psi_ka psi_kb` (this is `beta_ab`, already captured in F1).

| cell | rho_ab | sqrt(a_a a_b) | coherence | F_hh | t' | 2 rho t' | ratio |
|---|---|---|---|---|---|---|---|
| fm0.2/2.2 s4 | 0.0266 | 0.0268 | 0.992 | −0.00002 | −0.0333 | −0.00177 | 0.01 |
| fm0.2/2.2 s5 | 0.0387 | 0.0396 | 0.977 | −0.00087 | −0.0331 | −0.00256 | 0.34 |
| fm0.2/2.4 s5 | 0.0741 | 0.0807 | 0.918 | −0.00250 | −0.0332 | −0.00492 | 0.51 |
| fm0.2/2.4 s6 | 0.0405 | 0.0413 | 0.982 | −0.00189 | −0.0361 | −0.00293 | 0.65 |
| fm0.3 s1 | 0.0607 | 0.0614 | 0.988 | −0.00238 | −0.0333 | −0.00404 | 0.59 |
| fm0.3 s4 | 0.1354 | 0.1413 | 0.958 | −0.01028 | −0.0332 | −0.00899 | 1.14 |

## Verdict

| prediction | measured |
|---|---|
| coherence fraction 0.1–0.5 | **0.969** (0.918–0.992) — refuted |
| rho_ab ≈ 0.015–0.03 | **0.063** mean — ~2x above |
| F_hh ≈ 2 rho_ab t' within ~2x | **0.54 mean ratio** — holds |

This is the pre-registered disconfirmation: `rho_ab ≈ sqrt(a_a a_b)`. Coherence is at its
Cauchy–Schwarz maximum in every cell — the smeared density is effectively rank one, a single
dominant state. Nothing is being suppressed.

## What the arithmetic says instead

The identity closes, so the decomposition is self-consistent; it is the *interpretation* that
fails. `F_hh` is small for the plainest possible reason: `rho_ab ~ 0.06` and `t' ~ −0.033
eV/Å`, and their product is −0.004. No evasion is required to produce that number.

Third mechanism refuted in a row, and the three failures are one fact. The head is not doing
anything pathological at the hub: population two-sided (0.87–0.98), coherence maximal (0.97),
coupling elevated (1.4–2.4x baseline), and the hub–hub force exactly what the algebra gives
for those values. **The sign constraint is real; it is simply small here.**

## What the sign flip would have to deliver

Target +0.058, current total +0.017, shortfall **0.041**. Flipping `F_hh` from −0.003 to
+0.003 supplies 0.006 — about 15% of it.

`rho_ab` is already maximal for the present populations, so the only levers are hub mass and
`t'`. Closing the shortfall through the hub–hub channel alone needs `2 rho_ab t' ≈ +0.041`:

* at present hub mass (rho_ab 0.063): `|t'| ≈ 0.33 eV/Å`, ten times the current 0.033;
* at hub mass 0.4 (rho_ab ≈ 0.2, observable 3's soft target): `|t'| ≈ 0.10 eV/Å`, three times
  current.

So the A/B's observable 1 (0.36 → ≥ 0.8) requires the sign flip to drive hub mass **and** the
coupling's distance-derivative together, by roughly 3x each. That is not excluded — it is what
observables 3 and 4 ask for — but it is a much larger demand than "flip a term that is
currently pointing the wrong way", and it should be stated before the arm is read rather than
after.

## Caveat on t'

`t'` is the slope of the zero-feature hopping profile at the pair distance, which is the
sanctioned source but not the realised one: `rho_ab` and `F_hh` come from real frames. The
uniformity of `t'` across cells (−0.033 ± 0.001) is itself a sign that it is reporting the
profile rather than per-frame structure, and the mean ratio of 0.54 may partly reflect that
mismatch rather than a real gap. An autograd `dH_ab/dd` on real frames would settle it and is
the first thing to fix if this number is load-bearing for any decision.
