# S1 — the sign diagnosis is NOT confirmed. Stop.

24 saved V3 heads, 24 charged frames each. Decision tree §5: *"S1 does not show opposite signs
→ the diagnosis is wrong; stop and report the decomposition."* This is that branch. S2 and S3
have not been launched.

Both scripts use the same axial convention, `0.5 (F_b − F_a) · axis`, positive = the hub pair
pushed apart — verified before tabulating them together.

## The three S1 predictions, against what was measured

| prediction | measured | verdict |
|---|---|---|
| sign agreement well below 0.5 on localised cells | **0.653** (localised), 0.947 (rest) | **fails** |
| \|H_ab\| at the t_min floor, ~0.01 eV | **0.069 eV** localised, 0.112 eV rest, up to 0.705 | **fails** |
| on-site term straining against the hopping term | on-site ≈ 0 (−0.001…+0.008); hopping **+0.03…+0.14**, same sign as target | **fails** |

Only **1 of 6** localised cells has a correction opposing its target. Five of six agree in
sign.

## What is actually happening

Localised cells do behave differently, but the difference is **magnitude, not sign**:

| group | n | agree | Pearson r | target | correction | correction/target |
|---|---|---|---|---|---|---|
| localised (N_eff ≤ region) | 6 | 0.653 | +0.485 | +0.0582 | +0.0207 | **36%** |
| the rest | 18 | 0.947 | +0.924 | +0.0740 | +0.0728 | **98%** |

When the state is spread out the head supplies essentially all of the required outward force.
When it is forced compact it supplies about a third of it, with the correct sign. That is an
**under-response**, not an anti-correlation, and it is a different problem from the one the
sign diagnosis describes.

The decomposition says the same thing from the other side. `axial_on_hub` is ≈ 0 in every cell
— the hub on-site term contributes essentially nothing, consistent with the observation that
the partner Pb lies outside the 5 Å block-1 descriptor so `eps_hub` cannot depend on d. But
the hopping term is **positive** (outward, +0.03 to +0.14), i.e. it pushes the pair apart, in
the same direction as the DFT residual. The diagnosis requires it to be attractive.

`|H_ab|` is alive throughout, not switched off: 0.019–0.705 eV, and *lower* in localised cells
(0.069) than elsewhere (0.112) — the opposite ordering to "the head switches the wrong-sign
term off when it matters".

## Per-cell, the six localised cells

| cell | N_eff | agree | r | target | correction | \|H_ab\| | on_hub | hop |
|---|---|---|---|---|---|---|---|---|
| fm0.3 s4 | 7.79 | 0.333 | +0.038 | +0.0544 | **−0.0013** | 0.088 | −0.0010 | +0.0546 |
| fm0.2/2.4 s6 | 8.37 | 0.417 | −0.142 | +0.0534 | +0.0001 | 0.052 | −0.0000 | +0.0495 |
| fm0.2/2.4 s5 | 8.53 | 0.917 | +0.745 | +0.0765 | +0.0280 | 0.063 | −0.0001 | +0.0348 |
| fm0.2/2.2 s5 | 8.94 | 0.417 | +0.485 | +0.0457 | +0.0029 | 0.046 | −0.0001 | +0.0487 |
| fm0.3 s1 | 11.53 | 0.958 | +0.956 | +0.0730 | +0.0752 | 0.112 | −0.0002 | +0.0682 |
| fm0.2/2.2 s4 | 13.42 | 0.875 | +0.826 | +0.0464 | +0.0195 | 0.056 | −0.0000 | +0.0595 |

The three cells with near-zero correction (s4, s6, s5-2.2) are the ones dragging the group
mean down. Only `fm0.3 s4` is actually negative, and only marginally (−0.0013 against a target
of +0.0544) — it is better described as "supplies nothing" than "pushes the wrong way".

## Caveat on how this was first reported

The initial read given verbally was confirmatory, and it was drawn from a single cell —
`fm0.3 s4`, which happens to be both the most localised and the only one of the six showing
an inversion. Generalising from it was wrong. The stratified sweep does not support it.

## What this does and does not overturn

Unaffected: the level really is `lambda_min` of a bonding-signed H, which is an added
electron's level, and a hole removed from a bonding state really is a maximum of the occupied
manifold. That argument stands on its own and the corrected convention has been implemented
and unit-tested (hole force outward, electron force inward, by finite differences). It is
**default-off** and nothing has been run on it.

What S1 does not support is the empirical claim that the wrong convention is what is producing
the observed failures — the chain from it to D1's `|H_ab| = 0`, R1's ligand lean, and V3's
negative `axial_red`. On these 24 heads the hub coupling is alive, the hub on-site term is
inert, and the hopping force already points the right way.

The residual fact needing an explanation is the one S1 did find: **a compact state supplies
only ~36% of the required outward force, correctly signed.** That is a magnitude deficit at
the hub, and the inert on-site term plus the descriptor's inability to see the partner Pb at
5.3–6.8 Å remains the most plausible reason for it.
