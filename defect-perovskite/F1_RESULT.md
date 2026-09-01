# F1 — the mechanism is confirmed; the proposed evasion route is not

24 saved V3 heads, 24 charged frames each. Per-frame means (not medians of medians, so the
three parts reconcile against the full force). Axial convention `0.5 (F_b − F_a) · axis`,
positive = the hub pair pushed apart.

## Verdict against the two F1 predictions

| prediction | result |
|---|---|
| hub–hub ≤ 0 or ≈ 0 in all six localised cells | **6 / 6 confirmed** |
| two-sidedness ≪ 0.5 in localised cells | **0 / 6** — values 0.53, 0.67, 0.70, 0.76, 0.80, 0.82 |
| hub–ligand carries the positive share | **no** — hub–ligand ≈ 0 in localised cells |

The explicit stop condition — *hub–hub positive and substantial in the localised cells* — is
**not** met. hub–hub is negative in all six. But the confirm condition as written ("if 1–2
hold") is also not met, because the head is not evading by one-siding.

## The numbers

| group | n | hub–hub | hub–ligand | rest | full | two-sided | beta_ab | hub-pair mass |
|---|---|---|---|---|---|---|---|---|
| localised (N_eff ≤ region) | 6 | **−0.00299** | +0.00066 | +0.02159 | +0.017 | 0.713 | +0.0627 | 0.131 |
| the rest | 18 | +0.00010 | +0.01109 | +0.01789 | +0.026 | 0.759 | +0.0512 | 0.107 |

The six localised cells individually:

| cell | N_eff | hub–hub | hub–lig | rest | 2-sided | beta_ab | \|H_ab\| |
|---|---|---|---|---|---|---|---|
| fm0.3 s4 | 7.79 | −0.01028 | −0.00426 | +0.02912 | 0.668 | +0.135 | 0.094 |
| fm0.2/2.4 s6 | 8.37 | −0.00189 | +0.00410 | +0.01848 | 0.762 | +0.041 | 0.052 |
| fm0.2/2.4 s5 | 8.53 | −0.00250 | +0.00033 | +0.00931 | 0.532 | +0.074 | 0.069 |
| fm0.2/2.2 s5 | 8.94 | −0.00087 | +0.00307 | +0.01731 | 0.697 | +0.039 | 0.053 |
| fm0.3 s1 | 11.53 | −0.00238 | +0.00069 | +0.02945 | 0.804 | +0.061 | 0.134 |
| fm0.2/2.2 s4 | 13.42 | −0.00002 | +0.00006 | +0.02588 | 0.815 | +0.027 | 0.067 |

## What is confirmed

**The structural claim holds in every cell.** `beta_ab` is positive throughout (+0.027 to
+0.135), as Perron–Frobenius requires for a nodeless ground state of a bonding-signed H, and
the hub–hub axial force is correspondingly ≤ 0 in all six localised cells. The head genuinely
cannot write the physically required `+t_hub(d)`. That was the load-bearing part of the
reinterpretation and it is now measured rather than argued.

It is also specific to localised cells: hub–hub averages −0.00299 there against +0.00010 over
the other eighteen. The penalty appears exactly when occupancy is forced onto the compact
state, which is the predicted conditionality.

## What is refuted

**The head is not evading by one-siding.** Two-sidedness is 0.713 in localised cells against
0.759 elsewhere — statistically indistinguishable, and nowhere near the ≪ 0.5 predicted.
`beta_ab` is *higher* in localised cells (+0.063 vs +0.051), not suppressed. Both hub Pb carry
amplitude; the state is not dodging the hub–hub term by vacating one of them.

**The positive force does not come from hub–ligand edges.** In localised cells the hub–ligand
share collapses to +0.00066, essentially zero, against +0.01109 in the rest. The outward force
is carried almost entirely by `rest` — off-diagonal hoppings between atoms of which *neither*
is a hub Pb, whose gradient reaches the hub only through the descriptor's r_max dependence on
hub position. That is the original D1 leak mechanism, not the missing-neighbour asymmetry the
reinterpretation attributes it to.

The species composition points the same way: localised cells put 66% of the carrier mass on
Cl and only 13% on the hub pair. The compact state is cage-centred, not hub-centred — the same
ligand lean R1 reported, now visible inside the force decomposition.

## Reading

The mechanism is real and the evasion is real, but the route is not the one proposed. The head
pays a small inward hub–hub penalty (−0.003 of a +0.017 total, ~18%) and sources its outward
force from the descriptor leak while placing the state on the cage. It does not need to
one-side, because it never puts much amplitude on the hub in the first place.

That matters for what the A/B would test. Flipping the sign turns a −0.003 term into +0.003
at current amplitudes, which does not by itself close a gap from 36% to 80% of target. The
A/B's own observable 2 anticipates this — it predicts `|H_ab|` *grows* above the decay prior
to ≳ 0.15 eV once `+t_hub(d)` becomes available, i.e. the current small value is the
suppressed state and the point is whether it grows when allowed. Present `|H_ab|` in localised
cells is 0.052–0.134 eV, so the predicted target is roughly a doubling of the largest and a
tripling of the typical.

The competing explanation F1 cannot rule out: the head places the state on the cage because
the cage is where the *data* put it, in which case the hub–hub term is a side issue and the
sign flip changes little. Only the A/B separates these.
