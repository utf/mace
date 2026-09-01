# F1 — the constraint is real and is being paid, not evaded

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** F1 confirms the structural claim in every localised cell: the hub–hub force is
inward, `beta_ab` is positive, and the head cannot write the physically required `+t_hub(d)`.
It refutes both proposed evasion routes. The head does **not** hold the hub coupling at the
decay prior (it is 2.4x above it) and does **not** one-side the state (two-sidedness 0.71,
same as everywhere else). It pays the inward penalty, keeps both hub Pb occupied, and sources
its outward force from elsewhere entirely. That is absorption, not evasion — a third
possibility neither the original diagnosis nor the reinterpretation anticipated.

The stop condition (hub–hub positive and substantial) is not met, so the A/B is not
forbidden. The confirm condition as written ("if 1–2 hold") is not met either. We have not
launched it and are asking for the branch.

## Corrections to the record, as agreed

- Citing D1's `|H_ab| = 0` as support for the sign diagnosis was an error; its 5 Å
  neighbour-list cause was already on record. **Struck.**
- The "on-site term straining to compensate" prediction contradicted our own reach argument —
  a 5 Å block-1 descriptor cannot see d, so `eps_hub` cannot strain. **Struck.**
- The S1 predictions tested whether trained models *exhibit* a wrong-sign force. That was the
  forecasting error. A converged optimiser does not exhibit a wrong-sign channel; the
  signature is suppression and under-supply, which is what S1 measured.

For the project notes: *the physics argument was about what the head can represent; the S1
predictions were about what a trained head would display; optimisation sits between the two
and inverts the phenomenology. Pre-register evasion signatures, not exhibition signatures, for
any future sign or representability claim.* — F1 adds a corollary: **pre-register absorption
too.** A model that can pay a wrong-signed term's cost out of another channel will look like
neither exhibition nor evasion.

## Results — 24 heads, 24 charged frames, per-frame means

| prediction | result |
|---|---|
| hub–hub ≤ 0 or ≈ 0 in all six localised cells | **6 / 6 confirmed** |
| hub–ligand carries the positive share | **refuted** — hub–ligand ≈ 0 in localised cells |
| two-sidedness ≪ 0.5 in localised cells | **refuted** — 0.53–0.82, mean 0.713 |
| \|H_ab\| at the realised decay prior | **refuted** — 1.6–4.2x the prior, mean 2.44x |

| group | n | hub–hub | hub–ligand | rest | two-sided | beta_ab | hub-pair mass |
|---|---|---|---|---|---|---|---|
| localised (N_eff ≤ region) | 6 | **−0.00299** | +0.00066 | +0.02159 | 0.713 | +0.0627 | 0.131 |
| the rest | 18 | +0.00010 | +0.01109 | +0.01789 | 0.759 | +0.0512 | 0.107 |

Per localised cell, with `|H_ab|` against the realised decay prior at matched d = 5.61 Å:

| cell | N_eff | hub–hub | hub–lig | rest | 2-sided | beta_ab | \|H_ab\| | prior | ratio |
|---|---|---|---|---|---|---|---|---|---|
| fm0.3 s4 | 7.79 | −0.01028 | −0.00426 | +0.02912 | 0.668 | +0.135 | 0.094 | 0.032 | 2.95 |
| fm0.2/2.4 s6 | 8.37 | −0.00189 | +0.00410 | +0.01848 | 0.762 | +0.041 | 0.052 | 0.033 | 1.57 |
| fm0.2/2.4 s5 | 8.53 | −0.00250 | +0.00033 | +0.00931 | 0.532 | +0.074 | 0.069 | 0.032 | 2.15 |
| fm0.2/2.2 s5 | 8.94 | −0.00087 | +0.00307 | +0.01731 | 0.697 | +0.039 | 0.053 | 0.032 | 1.64 |
| fm0.3 s1 | 11.53 | −0.00238 | +0.00069 | +0.02945 | 0.804 | +0.061 | 0.134 | 0.032 | 4.24 |
| fm0.2/2.2 s4 | 13.42 | −0.00002 | +0.00006 | +0.02588 | 0.815 | +0.027 | 0.067 | 0.032 | 2.10 |

## What this establishes

**The Perron–Frobenius argument is measured, not just argued.** `beta_ab` is positive in every
cell (+0.027 to +0.135) and the hub–hub axial force is correspondingly ≤ 0 in all six localised
cells. It is conditional on localisation as predicted: −0.00299 in localised cells against
+0.00010 across the other eighteen. The head genuinely cannot express `+t_hub(d)`.

## What it overturns

**Neither escape route is in use.** The head grows the hub coupling to 2.4x the decay prior
rather than holding it there, and keeps both hub Pb occupied rather than one-siding. `beta_ab`
is *higher* in localised cells than elsewhere. It is not dodging the hub–hub term; it is
paying it.

**And it is paying it out of the descriptor leak, not out of hub–ligand asymmetry.** In
localised cells the hub–ligand share collapses to +0.0007 against +0.0111 elsewhere, while the
outward force is carried by `rest` — hoppings between atoms of which *neither* is a hub Pb,
whose gradient reaches the hub only through the descriptor's r_max dependence on hub position.
That is the original D1 leak.

**The state is cage-centred.** Localised cells put 66% of carrier mass on Cl and 13% on the hub
pair. This is R1's ligand lean, now visible inside the force decomposition rather than inferred
from clamp comparisons — and it explains why the hub–ligand share vanishes: there is little hub
amplitude for those edges to act on.

## What we are asking

The branch. F1 sits between your two conditions and we would rather not resolve that
unilaterally.

**The case for running the A/B.** The load-bearing claim is confirmed in 6/6 cells. The
cage-centred finding also sharpens the test: the prediction is no longer only that `|H_ab|`
grows, but that **hub mass rises from 0.13**, since `+t_hub(d)` pays only where amplitude sits
on both hub atoms. That is a cleaner discriminator than any observable in the original list.

**The case against.** hub–hub is −0.003 of a +0.017 total, about 18%. Flipping the sign makes
it +0.003 at current amplitudes, which does not by itself take correction/target from 0.36 to
0.80. The A/B's observable 2 already anticipates that the term must *grow* — but `|H_ab|` is
already at 2.4x its prior, so the room the argument assumed is partly spent; the predicted
≳ 0.15 eV is ~1.1x the largest cell we have, not a tripling from a floor.

**The competing explanation F1 cannot exclude.** The head may place the state on the cage
because the cage is where the data put it, in which case hub–hub is a side issue and the sign
flip changes little. Only the A/B separates these, which is an argument for running it — but
it should be run knowing that a null result would then be ambiguous between "sign is not the
problem" and "the data prefer the cage".

If it helps decide: the one-cell hub2-clamped mechanistic check in your §5 would separate
those two before spending six seeds per arm, since under a clamp the state is forced onto the
hub and the sign's effect on `axial_red` is isolated from where the head would otherwise put
the carrier. We would suggest running that first.
