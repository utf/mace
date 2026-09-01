# V3 implementation review — V_Cl+ orthorhombic CsPbCl3, 1 Sep 2026

*Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025). Labels are that paper's
low-fidelity PBE set (scalar-relativistic, no SOC).*

**Purpose of this note.** V3 is built, tested and running. This is a request for confirmation
before the results are read: it lists what was implemented, the five places where the
specification could not be followed literally and what was done instead, and the early
numbers, which are a qualitative change from V2 but carry one internal inconsistency that
should be resolved before anyone celebrates.

## What V3 is

The previous round established that `Delta_bind >= m` is necessary but not sufficient. The
model met the inequality by sliding the host reference — `lambda_1(pristine)` floated over
~8 eV across runs — rather than by splitting a level off the continuum.

V3 removes the freedom. Every matrix element of the carrier Hamiltonian is a function of the
trunk's **first interaction block only, detached**:

    D_i     = block-1 output for atom i, no gradient into the trunk
    eps_i^c = MLP(D_i, E(z_i), c) + c(D_i) |v_i|^2
    t_ij^c  = [t_min + softplus(B(D_i, D_j, E(z_i), E(z_j), c))] g(r) f_env(r; r_couple)
              + A(D_i, D_j) (v_i.rhat_ij)(v_j.rhat_ji) g f_env

One aggregation at `r_max` means an atom further than `r_max` from the vacancy has a
descriptor **identical** to its pristine counterpart, so the far block of H is pinned to the
host and the continuum cannot move. Two length constants survive, both material-agnostic:
`r_max` and `r_couple`. The constraint gains an upper cap at the band gap, which is the
escape detector — V1 previously reached `Delta_bind` = +11.9 eV, which no level inside a
2.2 eV gap can do.

Implemented as a subclass, so there is still exactly one eigensolve in the codebase.

## Pre-training tests: all pass

Locality is asserted as **identity**, not closeness — "approximately pinned" is what a
trained-away artefact looks like.

| test | result |
|---|---|
| H bit-identical when trunk blocks 2+ are randomised | pass |
| on-site elements beyond `r_max` match pristine to float precision | pass |
| the changed set is exactly the atoms within `r_max` | pass |
| no head loss reaches the trunk (every trunk grad `None`) | pass |
| two-site closed form, re-derived for V3's parametrisation | pass |
| coupling envelope at the flanking separation | `f_env(6.8 A) = 0.812`, 100% of frames |

Tests 2 and 3 needed a synthetic construction: dataset frames are relaxed, so a defect
frame's far atoms can never match pristine exactly and the claim could only ever have been
approximate. A pristine cell with one atom deleted and positions untouched makes it exact.

## Five deviations from the specification — these are what need confirming

**1. Test 4 as written would have breached hard rule 1.** "The flanking pair coupled with
`f_env >= 0.3` in >= 99% of charged frames" requires the vacancy assignment inside the
training loop. The structural equivalent does not: the envelope at the far end of the
5.2–6.8 A window clears the floor (a pure function of `r_couple`), and essentially every frame
carries an edge in that window. Neither statement names a defect. The standalone envelope is
asserted equal to the head's own, since a reach test evaluating its own formula is exactly
the pattern that has produced four passing guards over broken paths on this project.

**2. The candidate region is ~14 atoms, not ~40.** Section 4 states "~40 of 79 here — half
the cell". Measured at `r_max` = 5.0 A it is **14.2 of 89** atoms; a 5 A sphere is ~19% of a
~2800 A^3 cell. The region gate is therefore about three times tighter than intended. If a
~40-atom region was meant, `r_max` is not the radius that produces it. Nothing was adjusted —
the measurement is reported and the tighter gate used.

**3. The two `Delta_bind` references cannot coincide by construction.** Section 4 says the
pristine-referenced and far-block-referenced values "coincide by construction and their
agreement is itself a check". V3 makes the far block's *matrix elements* identical to the
host's, but `lambda_1` of a truncated ~65x65 submatrix is not `lambda_1` of the full 80x80
pristine matrix — eigenvalue interlacing alone puts the submatrix's lowest eigenvalue higher.
Observed disagreement is 0.07–0.14 eV in the completed cells (and several eV in untrained
smoke runs). **As specified this check can never pass.** Both numbers and their difference are
reported. The fix is either to compare like with like — restrict the pristine cell to a
same-sized region — or to treat the offset as a measured truncation term rather than a gate.

**4. Retiring `decay_r0` needed a compensating change.** Referencing the hopping decay at
`r_max` instead of the material-specific 3.0 A divides every element by e^2 at the old
initial decay length, which starts the head effectively disconnected — the regime the `t_min`
floor exists to prevent. `decay_init` was raised to 2.5 A, with a test asserting the flanking
pair is not sitting at the floor.

**5. The band gap could not be read from the source data.** No VASP outputs are present
locally, so the Zenodo route was unavailable without a bulk download. `E_gap` = 2.2 eV,
literature PBE scalar-relativistic without SOC, matching the labels — deliberately not the
paper's tuned HSE+SOC 3.08 eV, not PBE+SOC (~1.3 eV), not experiment (2.93 eV). Separately,
the dataset's own `band_edges.json` carries symmetric +/-1.2 eV edges implying a 2.4 eV gap;
that is a label-referencing convention rather than a measured gap, which is why 2.2 is the
default and 2.4 is run as a sensitivity arm that doubles as a consistency check against the
convention.

## Early results — 4 of 30 cells

Frozen base, head + response channel, no clamp, 40 epochs. Reference ceiling for a
delocalised fit is axial_red +0.855 / rmse_nbhd 20.6.

| cell | N_eff / region | null ratio | Delta_bind (far-block) | region mass | retained | axial_red | loc len |
|---|---|---|---|---|---|---|---|
| f_m 0.1, gap 2.2, s1 | **2.67** / 15 | **0.049** | +0.310 (+0.243) | 0.658 | 0.694 | −0.056 | 3.70 A |
| f_m 0.2, gap 2.2, s1 | **4.25** / 15 | **0.078** | +0.342 (+0.237) | 0.585 | 0.509 | −0.082 | 4.71 A |
| f_m 0.2, gap 2.0, s1 | **6.89** / 15 | **0.126** | +0.241 (+0.169) | 0.373 | 0.559 | −0.154 | 5.54 A |
| f_m 0.3, gap 2.2, s2 | 21.22 / 15 | 0.359 | +0.436 (+0.299) | 0.486 | 0.698 | +0.672 | 5.63 A |

**Three of four cells localise.** N_eff 2.67–6.89 against a criterion of <= 8, and null ratio
0.049–0.126 against <= 0.15. For scale: across 18 V1/V2 cells nothing passed either
criterion, and the closest single cell in the entire project reached N_eff 12.5 / ratio 0.168.
`Delta_bind` sits between `m` and `E_gap` with no cap breach in these cells.

**Two things stop this being a clean pass, and the first is a contradiction rather than a
disappointment.**

*Retained mass is 0.51–0.70, against a gate of >= 0.9* — and that is **inconsistent with
N_eff ~ 3**. A state living on three atoms cannot lose half its amplitude when a pristine
block is joined on; ~0.5 is the value expected of a band-like state. The dilution utilities
provide an interface-mass diagnostic precisely to separate the two readings — "retained ~0.5,
interface low" means genuinely band-like, "retained ~0.5, interface high" means a straddling
artefact at the join and must **not** be read as band-like. That diagnostic was not logged in
this run, so the number cannot currently be interpreted either way. This is an implementation
gap, not a result, and it needs a re-run of the dilution step to resolve.

*The force-fit cost is larger than "some".* Three of four cells sit at axial_red ~0 or below,
against a delocalised ceiling of +0.855. The one cell that fits well (+0.672) is the one that
did not localise. On this evidence localisation and force fit are trading off hard, which
sharpens rather than answers the question of whether the localised solution is the physical one.

## Also worth recording

* **One `E_gap` breach.** `f_m` = 0.3, seed 1 drove the epoch-median `Delta_bind` to +2.53
  then +2.34 against the 2.2 eV cap and aborted, as designed; two of three `f_m` = 0.3 seeds
  so far have aborted, none in the other arms. Since tests 1–3 pass, this is **not** a
  descriptor-locality escape. The likelier reading is an over-deep well: between the floor at
  `m` = 0.66 eV and the cap at 2.2 eV nothing constrains `Delta_bind`, so the optimiser can
  drift up through that flat region. The decision tree treats any breach as stop-and-diagnose,
  which is why it is raised here rather than folded into the averages.
* **`L_gap` starts ~10^4 above the force loss** (residual 21 eV) and briefly destroys the fit,
  then collapses to a 0.22 eV residual by epoch 5 and stops mattering. Not degenerate, but the
  raw residual is now logged beside the weighted value because the weighted number alone is
  unreadable.
* **The localisation-length metric was wrong and has been fixed.** It had no minimum-image
  convention, which under periodic boundaries is not a worse estimate but an invalid one — it
  reported 1.10 A for a state with N_eff 51. Every number it produced previously has been
  discarded; the values in the table above use the corrected metric.
* **The section-6 hopping-offset diagnostic cannot be run.** It needs a trained V2 model and
  neither V1 nor V2 was ever saved. Model saving has been added for future cells; recovering
  V2's would require repeating that run. It is optional and does not block V3.

## What we are asking

1. Confirm the five deviations, particularly (2) the tighter region gate and (3) the
   far-block check as specified being unpassable.
2. A view on the retained-mass contradiction: we propose re-running the dilution step with the
   interface diagnostic before reading the gate at all.
3. Whether the `f_m` = 0.3 aborts should be treated as the decision tree's "escape route
   remains" branch — we think not, since the locality tests pass — or as evidence that the
   flat region between `m` and `E_gap` needs a shape rather than two hard walls.

30 cells are running; the remaining 26 land within the hour.
