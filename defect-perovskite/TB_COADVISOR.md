# Is the missing object the host band edge? — V_Cl+ orthorhombic CsPbCl3, 1 Sep 2026

*Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025). Labels are that
paper's low-fidelity PBE set (scalar-relativistic, no SOC).*


**Short version.** We tested whether our model fails to bind the hole because nothing in it
knows where the host band edge is. The diagnostic says yes: the model places the carrier level
essentially at the host continuum, and in some seeds above it. But imposing the corresponding
inequality during training does *not* produce a localised carrier — 0 of 18 runs reach the
localisation gate. The inequality turns out to be necessary but not sufficient, because
nothing pins the absolute host reference it is measured against.

## Where we were

Our defect model is a MACE trunk with a tight-binding carrier head. DFT says the hole in the
chlorine vacancy is bound to the two vacancy-adjacent Pb (force-response ratio R = 0.95,
CI [0.66, 1.34], 13/17 matched frames).

> **Cell-size caveat on R_DFT.** The 80-atom training cell is a 2x2x1 orthorhombic
> expansion with c = 11.2 A, which constrains Pb-Pb separations above ~5.5 A when the
> vacancy axis lies along c. The 79-atom d(Pb-Pb) distribution is therefore partly
> cell-limited. The matched-d comparison against the 2x2x2 (159-atom) frames is
> unaffected in principle, but the unmatched long-d frames are exactly the ones this
> constraint removes from the small cell.

The model does not reproduce that: trained from scratch, the carrier is delocalised in
8/8 seeds.

Earlier work established that this is not a representability problem. Adding an electrostatic
response channel — the carrier's own potential acting on every ion, with a learned effective
charge and polarisability — gave compact states the long-range force footprint that a
short-ranged tight-binding head cannot produce. What it did not do was *select* them, or
distinguish the hub from a distance-matched ligand pair. A bound solution that fits exists;
nothing in the objective prefers it.

That framed the present question: what is absent from the objective?

## The claim under test

A defect-bound carrier is a level inside the gap — in carrier energy, below the edge of the
host continuum. A level below the continuum of a short-ranged Hamiltonian is exponentially
localised. So localisation reduces to a scalar the head already computes:

    Delta_bind = lambda_1(pristine host, hole counter) - lambda_1(defect cell)  >  0

The suspicion was that our models fail this. The head has only ever been active on defect
cells, so nothing tells it where the host edge sits, and a band-edge state is as legal as a
bound one. In DFT the two ionisation energies differ by order eV.

This is label-free by construction. Pristine frames carry no defect information; scoring the
head on them with the hole counter is an evaluation of the *host*. It is off-distribution for
them deliberately — that is exactly the reference a binding energy needs, and why it can be a
diagnostic but not a training signal.

## T-A — the diagnostic, on saved models

| model set | Delta_bind | range | > 0.5 eV |
|---|---|---|---|
| from scratch, epoch 50 (n=8) | **+0.195 eV** | +0.10 … +0.40 | 0/8 |
| earlier screen (n=12) | +0.086 eV | −0.43 … +0.38 | 0/12 |

Against an expectation of order eV, this is an order of magnitude too small. In 5 of the 12
older seeds it is **negative** — those models put the carrier level *above* the pristine host
edge, which no bound state can do. Nothing anchors the level to the host continuum.

Two controls make the numbers readable rather than decorative. The site energies are absolute
only under the gauge in which the uniform mode is penalised rather than projected out; that
holds in all 20 models, checked per model, so each row is a real binding energy rather than a
difference of two independently-floating origins. And the alternative mechanism — the head
lowering site energies across the whole defect cell rather than binding anything — is
measured directly, as the site-energy offset on atoms more than 8 A from the vacancy against
pristine. It is ~0 in every model, so the escape is not in use at this stage.

One honest qualification: the prediction was "Delta_bind ~ 0". On the clean from-scratch
models it is not zero, it is consistently and significantly **+0.2 eV**. Small against the
physical scale, but not noise.

## T-B — the causal test

If the inequality is what is missing, imposing it should localise the carrier:

    L_edge = w_e * max(0, m - Delta_bind)^2,     m in {0.2, 0.5} eV

on a frozen base with the response channel, no clamp, 40 epochs. The weight was calibrated
from the harness's own epoch-0 force loss so that a violation of m/2 costs about one frame's
worth, rather than being tuned.

Two head variants, because the inequality has a cheap and dishonest solution. The trunk's
receptive field tells every atom in a defect cell that it is in one, so the head can lower
site energies across the whole cell: the spectrum drops bodily, lambda_1 drops with it, and
the inequality is satisfied by a state exactly as delocalised as before. The level did not
move relative to the host — the origin did.

* **V1** keeps the current feature-based site energies, and can do this.
* **V2** replaces them with a counted form, eps_i = e(z_i) + sum_j phi(z_i, z_j, r_ij) —
  element embeddings and pair distances only, no trunk features. It can express "this Pb has
  five neighbours instead of six", which is the physics a bound level needs, but not "there is
  a vacancy 20 A away", because bulk coordination is identical either way. Asserted by test,
  not by argument.

### Result: 0 of 18 runs localise

Localisation is measured as N_eff = 1/sum_i alpha_i^2, the participation ratio of the carrier
state (1 = one atom, ~80 = the whole cell), and as its ratio to three ungradiented channels
that provide a per-run delocalised baseline. The gate is N_eff <= 8 and ratio <= 0.15.

| variant | m | n | N_eff | ratio | Delta_bind achieved |
|---|---|---|---|---|---|
| V1 | 0.2 | 3 | 58.5 | 0.77 | **+1.52** |
| V1 | 0.5 | 3 | 52.7 | 0.69 | **+5.49** (max **+11.9**) |
| V2 | 0.2 | 6 | 48.7 | 0.65 | +0.37 |
| V2 | 0.5 | 6 | **32.3** [12.5, 55.3] | **0.43** [0.17, 0.73] | +0.72 |

**V1 escapes, as predicted.** It overshoots the margin by 3–24x while remaining completely
delocalised. V2, which cannot express the escape, tracks the margin closely and never exceeds
+0.86. That contrast in *overshoot* is the cleanest signature in the data: V1 has a route to
the inequality that costs it nothing and has nothing to do with binding.

**V2 does not escape, and still does not localise.** It reaches the margin easily, with a
final force loss of 2e-4 against 4.4e-3 at epoch 0, and fits the forces close to the
delocalised ceiling (axial_red 0.79 against the unclamped 0.855). It simply stays spread out.

Two real signals inside the negative result. The margin does monotone work — V2's mean N_eff
falls 48.7 to 32.3 and the ratio 0.65 to 0.43 between m = 0.2 and 0.5. And at m = 0.5 the seed
distribution is bimodal: two of six runs land at N_eff 12.5 and 14.7 against 28–55 for the
other four. Those two are the closest anything in this project has come to the gate, and they
pay for it in fit — the best of them has the worst force error of its group.

## Reading

**Delta_bind >= m is necessary but not sufficient.** The reason is visible in the data:
lambda_1(pristine) ranges from −5.2 to +6.4 eV across runs, and V2's achieved Delta_bind sits
at ~m almost exactly. Much of the inequality is being met by moving the host reference upward
rather than by pushing a level down out of the continuum. Against a floating reference,
Delta_bind > 0 does not place the state below the *defect cell's own* continuum — which is
what the localisation theorem actually requires.

This was anticipated as a limitation of the label-free formulation ("nothing pins the absolute
pristine edge without charged-pristine labels, so the level will sit at ~m"). What the
experiment adds is that it is not a mild floor: it dominates the outcome.

## What we are not claiming

* The per-species site-energy offsets are **not** clean evidence either way and we do not use
  them. They are non-zero in both variants and carry the opposite sign to the prediction, and
  the large charged and pristine cells differ in composition and relaxation as well as in
  having a vacancy, so a per-species mean difference is confounded. The overshoot is the
  discriminator; the offsets are not.
* The localisation-length metric in the harness is **wrong** — it takes an RMS radius about a
  centroid with no minimum-image convention, so it is invalid under periodic boundaries. It
  reported 1.1 A for a state with N_eff 51. Excluded from every number above.
* V1 has 3 seeds per margin against V2's 6. Seeds were allocated where the decision turns.

## Next

Two levers, and they test different things.

The m-dependence is monotone over the one factor of 2.5 we sampled. An m-ladder
(0.5, 1.0, 2.0) on V2 at 3 seeds is cheap and distinguishes "the constraint needs to be
deeper" from "the constraint is the wrong object" — if N_eff saturates short of the gate as m
grows, no achievable margin will localise the carrier and the band-edge framing is not the
missing piece.

The other lever removes the floating reference rather than pushing against it: charged
pristine single points on existing geometries would replace m with a measured depth and fix
the absolute gauge. That is a small DFT ask and it converts the inequality from a constraint
we assert into one we can calibrate.
