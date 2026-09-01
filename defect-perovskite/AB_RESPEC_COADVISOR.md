# A/B respec — one baseline disagreement to settle before launch

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Status.** The §2 convention is accepted and being implemented: hole channels solve the fully
negated matrix, each channel's inequality lives on its own occupied manifold, `s_c` retired.
The two agreed corrections to the record are carried. We have **held the A/B launch** on one
point: three of the F1 numbers quoted in §1 do not match what the sweep measured, and two of
the pre-registered observables — plus the OFF-arm acceptance criterion — depend on them.

We would rather settle that than run twelve cells against a gate that cannot be read.

## The disagreement

| quantity | §1 states | F1 measured (localised cells, n = 6) |
|---|---|---|
| hub–hub axial force | −0.007 ± 0.004 eV/Å | **−0.003** mean; range −0.0103 … −0.00002 |
| two-sidedness | **0.028 ± 0.011** | **0.53–0.82** |
| \|H_ab\| | at the decay prior | **1.4–2.4x above** it |

Only the first is a near-miss (−0.007 sits inside our observed range; the mean is −0.003). The
other two are qualitative disagreements, and both are load-bearing.

**Two-sidedness.** We define it as `min(alpha_a, alpha_b) / max(alpha_a, alpha_b)` on the two
vacancy-flanking Pb, from the diagonal of the density matrix. The raw per-atom amplitudes in
the six localised cells are:

| cell | alpha_a | alpha_b | ratio |
|---|---|---|---|
| fm0.2/2.2 s4 | 0.02733 | 0.02634 | 0.964 |
| fm0.2/2.2 s5 | 0.04039 | 0.03881 | 0.961 |
| fm0.2/2.4 s5 | 0.08665 | 0.07523 | 0.868 |
| fm0.2/2.4 s6 | 0.03951 | 0.04312 | 0.916 |
| fm0.3 s1 | 0.06063 | 0.06216 | 0.976 |
| fm0.3 s4 | 0.13985 | 0.14276 | 0.980 |

Both hub Pb carry essentially equal amplitude. (0.53–0.82 is the per-frame mean of the ratio;
0.87–0.98 is the ratio of per-frame-mean amplitudes. Either way the state is two-sided.) If
you are computing two-sidedness on a different quantity — the eigenvector components rather
than the density, say, or over a different frame set — that would reconcile the two numbers
and we would want to know, because the sign of `psi_a · psi_b` is a genuinely different
measurement from the amplitude ratio and is the one your observable 2 turns on.

**\|H_ab\|.** Elevated on both baselines we can construct: 2.44x the realised zero-feature
decay profile at matched d, and 1.4–2.0x the median `|H_ij|` among generic pairs at the same
5.3–5.9 Å separation in the same frames. Values are 0.052–0.134 eV against a profile prior of
~0.032 eV.

## What it does to the A/B as written

- **Observable 3** ("two-sidedness 0.028 → ≥ 0.5") is already satisfied at baseline. The OFF
  arm would pass it, so it cannot discriminate.
- **Observable 4** ("|t_ab| grows above the realised decay prior") likewise — already 2.4x
  above.
- **The OFF-arm acceptance criterion is the serious one.** §3 requires OFF to reproduce
  "two-sidedness ≈ 0.03, |H_ab| at prior". It will show ~0.7 and ~2.4x prior, because that is
  what this code produced in F1. §5 then routes to *"OFF fails to reproduce S1/F1 →
  environment drift; fix before reading ON"* — sending us to debug a drift that does not
  exist, with the ON arm unreadable until we do.

## Proposed restatement, keeping your logic intact

| | as written | proposed |
|---|---|---|
| Obs 2 | hub–hub flips positive from −0.007 ± 0.004 | from **−0.003**; keep `psi_a · psi_b < 0` as the sharp form |
| Obs 3 | two-sidedness 0.028 → ≥ 0.5 | **hub-pair mass rises from 0.13** |
| Obs 4 | \|t_ab\| grows above the prior | grows above **1.4–2.0x the matched-distance median**, i.e. ≳ 0.15 eV from 0.052–0.134 |
| OFF acceptance | two-sidedness ≈ 0.03, \|H_ab\| at prior | correction/target ≈ 0.36, hub–hub ≤ 0, two-sidedness ≈ 0.7, \|H_ab\| ≈ 2x matched-distance median |

Obs 3's replacement is the one we would argue for on its own merits. F1 found the localised
state is cage-centred — 66% of carrier mass on Cl, 13% on the hub pair — so hub mass is both
unsatisfied at baseline and directly downstream of the mechanism: `+t_hub(d)` pays only where
amplitude sits on *both* hub atoms. It discriminates where two-sidedness no longer can.

## One implementation finding worth having in the record

Full negation is right, and it costs nothing extra: `H_h = −H_e` shares eigenvectors with
`H_e`, so `Λ_h = −λ_max(H_e)` with the top eigenvector — which is the antibonding combination
the physics calls for, obtained from the same single `eigh`.

But **padding does not survive negation**. Padded slots are held at +1e3 so they sit above the
physical spectrum and never enter the lowest m. Negated, they land at −1e3 and become the
minimum, so the hole channel would select padding as its ground state in every cell with fewer
atoms than `num_states` — silently, with no error, returning a confident number that is the
padding energy. Padding must be re-applied after negation rather than negated with it. We are
implementing it that way with an explicit unit test on a cell smaller than m.

## What we are asking

1. Confirm the restated baselines, or tell us where our measurement is wrong — the
   two-sidedness definition is the most likely place for a genuine mismatch.
2. Confirm the Obs 3 substitution (hub-pair mass for two-sidedness).

On confirmation the A/B launches immediately; §2 will be implemented and unit-tested by then.
Nothing else in the respec is in question — the convention, the per-channel edge, the
coherence rule, and the decision tree all stand as written.
