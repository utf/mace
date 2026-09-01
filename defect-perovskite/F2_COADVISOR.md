# F2 — coherence is not suppressed; the hub–hub term is simply small

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** F2 meets the pre-registered disconfirmation condition. Pair coherence is at its
Cauchy–Schwarz maximum in every localised cell (fraction 0.969, range 0.918–0.992), not the
0.1–0.5 the suppression mechanism predicts. We have stopped and are not running the A/B.

The convention retraction is accepted, and it cost nothing: the negated-matrix path never
entered the active code. What is implemented and unit-tested is the energy-sign convention —
`dE_SR = Σ_c s_c n_c (Λ + μ_c)`, `s = (+1,+1,−1,−1)`, with occupation, α, charges and
`Delta_bind` all on `H_e`, unchanged. Default-off.

## Result — six localised cells, 24 charged frames each

`rho_ab = Σ_k w_k psi_ka psi_kb`, from the eigenvectors already captured in F1.

| cell | rho_ab | sqrt(a_a a_b) | coherence | F_hh | t' | 2 rho t' | ratio |
|---|---|---|---|---|---|---|---|
| fm0.2/2.2 s4 | 0.0266 | 0.0268 | 0.992 | −0.00002 | −0.0333 | −0.00177 | 0.01 |
| fm0.2/2.2 s5 | 0.0387 | 0.0396 | 0.977 | −0.00087 | −0.0331 | −0.00256 | 0.34 |
| fm0.2/2.4 s5 | 0.0741 | 0.0807 | 0.918 | −0.00250 | −0.0332 | −0.00492 | 0.51 |
| fm0.2/2.4 s6 | 0.0405 | 0.0413 | 0.982 | −0.00189 | −0.0361 | −0.00293 | 0.65 |
| fm0.3 s1 | 0.0607 | 0.0614 | 0.988 | −0.00238 | −0.0333 | −0.00404 | 0.59 |
| fm0.3 s4 | 0.1354 | 0.1413 | 0.958 | −0.01028 | −0.0332 | −0.00899 | 1.14 |

| prediction | measured |
|---|---|
| coherence fraction 0.1–0.5 | **0.969** — refuted |
| rho_ab ≈ 0.015–0.03 | **0.063** mean — ~2x above |
| F_hh ≈ 2 rho_ab t' within ~2x | **0.54** mean ratio — holds |

The identity closes, so the decomposition is self-consistent. It is the interpretation that
fails: the smeared density is effectively rank one, `rho_ab` sits at its ceiling, and `F_hh`
is small because `rho_ab ~ 0.06` and `t' ~ −0.033 eV/Å` multiply to −0.004. No suppression is
required to produce that number.

## Three probes, one fact

S1 tested sign inversion, F1 tested evasion by one-siding and coupling-pinning, F2 tested
evasion by coherence suppression. All three are refuted, and the refutations agree:

* population two-sided (amplitude ratio 0.87–0.98);
* coherence maximal (0.97);
* coupling elevated, not pinned (1.4–2.4x both baselines);
* hub–hub force exactly what the algebra gives for those values.

The head is not doing anything pathological at the hub. **The sign constraint you identified
is real — the hub–hub force cannot be positive under the current convention, confirmed 6/6 —
but it is small, and the head is paying it rather than dodging it.**

## What the flip would have to deliver

This is the number we would most want checked, because it changes what the A/B is asking.

Shortfall on the hub axis is **0.041** (target +0.058, current total +0.017). Flipping `F_hh`
from −0.003 to +0.003 supplies **0.006 — about 15%**.

`rho_ab` is already maximal for the present populations, so the remaining levers are hub mass
and `t'`. Closing the shortfall through the hub–hub channel requires `2 rho_ab t' ≈ +0.041`:

* at present hub mass (`rho_ab` 0.063): `|t'|` must rise **~10x** from 0.033 eV/Å;
* at hub mass 0.4 (observable 3's soft target, `rho_ab ≈ 0.2`): `|t'|` must rise **~3x**.

So observable 1 (0.36 → ≥ 0.8) requires the sign flip to move hub mass *and* the coupling's
distance-derivative together, by roughly 3x each. That is precisely what observables 3 and 4
ask for, so the A/B is not incoherent — but it is a substantially larger demand than "flip a
term that currently points the wrong way", and we think it should be on the record before the
arm is read rather than after.

## Caveat on our own t'

`t'` is the slope of the zero-feature hopping profile at the pair distance — the source your
spec sanctions, but not the realised one, while `rho_ab` and `F_hh` come from real frames. Its
uniformity across cells (−0.033 ± 0.001) suggests it is reporting the profile shape rather
than per-frame structure, and that may account for part of the 0.54 ratio rather than a real
gap. If `t'` is load-bearing for the decision, an autograd `dH_ab/dd` on real frames should
replace it; it is a short job and we would run it first.

## Record

Accepted as written: the negated-matrix retraction and its two-site derivation; the F1
baselines as measured; the corrected coherence rule (energy sign, occupied object and edge are
one choice, not three); the scope note that acceptors need the mirrored convention and that
the electron-counting head should be specified before SiC; and the rule that any
occupied-object/sign/edge change ships with its two-site finite-difference table first.

The padding-reapplication pattern is kept as documented hygiene for a future acceptor manifold,
with its small-cell test, even though nothing is negated here.

## What we are asking

1. Confirm the stop, or tell us the disconfirmation condition should be read differently —
   `rho_ab ≈ sqrt(a_a a_b)` is met unambiguously, but `F_hh ≈ 0` is a matter of degree
   (−0.003 mean against a +0.017 total, so 18%, not zero).
2. Whether the A/B is still worth running with observables 1–4 restated around the 3x/3x
   demand above. Our own view is that it is — it remains the only test that separates "the
   sign is not the problem" from "the data prefer the cage" — but it should launch knowing
   that a null result is ambiguous between those two, and that the sign flip alone accounts
   for 15% of the gap.
3. If it runs, we would still suggest the single hub2-clamped cell first: under a clamp the
   amplitude is forced onto the hub, which fixes hub mass by construction and isolates the
   `t'` question from the placement question.
