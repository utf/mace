# S1 — the carrier-sign diagnosis is not confirmed

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** S1 was set as the gate for the sign-fix programme: if the head's axial hub force
is not systematically opposite to the residual it is fitting, the diagnosis is wrong and
nothing downstream should be built. Measured across all 24 saved V3 heads, it is not. Five of
the six localised cells produce a correction with the **same** sign as their target. We have
stopped at that branch; S2 and S3 have not been run.

The physical argument behind the diagnosis is untouched, and the corrected convention is
implemented and unit-tested. What the data do not support is the empirical claim that the
wrong convention is what has been producing the observed failures.

## What was measured

For each of the 24 heads, on 24 charged frames:

    target_i     = F_DFT,i  - F_base,i        what the correction is supposed to supply
    correction_i = F_total,i - F_base,i       what it actually supplies

both from one forward pass, projected on the hub axis with positive = the two hub Pb pushed
apart. Plus |H_ab| across the vacancy and the on-site/hopping force decomposition. The two
scripts share the axial convention `0.5 (F_b - F_a) . axis`, verified before their outputs
were tabulated together. Results are stratified by whether the cell's state fits inside the
candidate region, because the prediction is specifically about cells where occupancy has been
forced onto a compact state.

## The three predictions, against measurement

| prediction | measured | |
|---|---|---|
| sign agreement well below 0.5 on localised cells | **0.653** localised, 0.947 rest | fails |
| \|H_ab\| across the vacancy at the t_min floor (~0.01 eV) | **0.069 eV** localised, 0.112 rest; range 0.019–0.705 | fails |
| on-site term straining to compensate the hopping term | on-site ≈ 0 (−0.001…+0.008); hopping **+0.03…+0.14** | fails |

The third row is the most direct contradiction. The diagnosis requires the dominant force
term, −∂λ/∂d, to be **attractive** where DFT is repulsive. Measured, the hopping contribution
is **positive in 23 of 24 cells** — it pushes the pair apart, the same direction as the DFT
residual.

## What the data do show

Localised cells are genuinely different, but in magnitude rather than sign:

| group | n | sign agreement | Pearson r | target | correction | correction / target |
|---|---|---|---|---|---|---|
| localised (N_eff ≤ region) | 6 | 0.653 | +0.485 | +0.0582 | +0.0207 | **36%** |
| the rest | 18 | 0.947 | +0.924 | +0.0740 | +0.0728 | **98%** |

Spread out, the head supplies essentially all the required outward force. Forced compact, it
supplies about a third of it, correctly signed. That is an **under-response at the hub**, not
an anti-correlation — a different failure from the one the sign diagnosis describes, and one
that a sign flip would not address.

The six localised cells individually:

| cell | N_eff | agree | r | target | correction | \|H_ab\| | on-site | hopping |
|---|---|---|---|---|---|---|---|---|
| f_m 0.3 s4 | 7.79 | 0.333 | +0.038 | +0.0544 | **−0.0013** | 0.088 | −0.0010 | +0.0546 |
| f_m 0.2 (2.4) s6 | 8.37 | 0.417 | −0.142 | +0.0534 | +0.0001 | 0.052 | −0.0000 | +0.0495 |
| f_m 0.2 (2.4) s5 | 8.53 | 0.917 | +0.745 | +0.0765 | +0.0280 | 0.063 | −0.0001 | +0.0348 |
| f_m 0.2 (2.2) s5 | 8.94 | 0.417 | +0.485 | +0.0457 | +0.0029 | 0.046 | −0.0001 | +0.0487 |
| f_m 0.3 s1 | 11.53 | 0.958 | +0.956 | +0.0730 | +0.0752 | 0.112 | −0.0002 | +0.0682 |
| f_m 0.2 (2.2) s4 | 13.42 | 0.875 | +0.826 | +0.0464 | +0.0195 | 0.056 | −0.0000 | +0.0595 |

Only one is negative, and only marginally: −0.0013 against a target of +0.0544. It is better
described as supplying nothing than as pushing the wrong way.

## One of the diagnosis's supporting observations has an already-known cause

The diagnosis lists D1's `|H_ab| = 0` first among the things it explains. That result has a
recorded, independent explanation: the D1 scripts defaulted to a 5 Å neighbour list against
10 Å-trained models, so the hub pair had no edge to measure. The scripts were fixed to derive
the cutoff from the model. S1 uses the fixed path and finds `|H_ab|` between 0.019 and
0.705 eV — non-zero everywhere. So that observation cannot be counted as support; it was an
analysis artefact, not a signature of the sign.

## Correction to our own earlier reading

Our first look at this was confirmatory and we reported it as such. It came from a single
cell, `f_m 0.3 s4` — which turns out to be both the most localised of the 24 and the only one
of the six showing an inversion. Generalising from it was wrong, and the stratified sweep does
not support it. We flag this because the initial number (agreement 0.38) was circulated before
the sweep finished.

## What stands

The physics of the diagnosis is not in question. λ is the lowest eigenvalue of a
bonding-signed H, which is an added electron's level; a hole removed from a bonding state is a
maximum of the occupied manifold, lying above its on-site energy by t, and a minimum
eigenvalue cannot represent it. That argument holds independently of these measurements.

The corrected convention — `dE_SR = Σ_c s_c n_c (Λ + μ_c)` with `s = (+1,+1,−1,−1)`, one
counter-free `H_e` broadcast across the four channels — is implemented and unit-tested. The
tests assert the physical consequence by finite differences: the hole force pushes the pair
apart, the electron force pulls it together. It is **default-off** and nothing has been run
on it, so the 24 saved heads still mean what they meant.

## What we are asking

1. Confirm the stop. On our reading S1 lands squarely on the tree's fourth branch and S2/S3
   should not run on the strength of this diagnosis.
2. A view on whether the corrected convention should nonetheless be adopted on physical
   grounds — it is right regardless of whether it explains the current failures — and if so,
   whether it is worth a small A/B rather than the full S2/S3 programme.
3. The fact that now needs an explanation is the under-response: **a compact state supplies
   ~36% of the required outward force, correctly signed, with an inert hub on-site term.**
   The hub on-site term contributing ≈ 0 in every cell is consistent with the partner Pb at
   5.3–6.8 Å lying outside the 5 Å block-1 descriptor, so `eps_hub` cannot depend on d at all.
   That is a representability limit of the descriptor rather than of the sign, and it points
   at the descriptor reach rather than the convention.
