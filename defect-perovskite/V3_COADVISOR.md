# V3 — implementation choices for confirmation

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

V3 is built and running. These are the points where the specification was ambiguous, could not
be followed literally, or turned out to rest on a false premise. No results here — each item
is a decision that would change what the numbers mean, so please confirm or overrule.

**1. Test 4 was reimplemented as a structural check.** "The flanking pair coupled with
`f_env >= 0.3` in >= 99% of charged frames" needs the vacancy assignment inside the training
loop, which hard rule 1 forbids. Substituted: the envelope at the far end of the 5.2–6.8 A
window clears the floor (a pure function of `r_couple`), and essentially every frame carries
an edge in that window. Neither statement names a defect. The standalone envelope is asserted
equal to the head's own.

**2. The candidate region is ~14 atoms, not the stated ~40.** At `r_max` = 5.0 A it measures
14.2 of 89 atoms — a 5 A sphere is ~19% of a ~2800 A^3 cell. Nothing was adjusted, so the
region gate is running about three times tighter than intended. If a ~40-atom region was
meant, `r_max` is not the radius that produces it.

**3. The far-block check cannot pass as specified.** Section 4 says `Delta_bind` against
pristine and against the defect cell's own far block "coincide by construction". V3 makes the
far block's *matrix elements* identical to the host's, but `lambda_1` of a truncated ~65x65
submatrix is not `lambda_1` of the full 80x80 pristine matrix — interlacing alone puts the
submatrix's lowest eigenvalue higher. Both values and their difference are reported, and the
difference is **not** used as a gate. Fix would be to compare like with like (restrict the
pristine cell to a same-sized region) or to treat the offset as a measured truncation term.

**4. The detach boundary was resolved in the strict direction.** Section 2 asks for a
"trunk-feature effective charge and polarisability" and also that "the trunk trains on the
base loss only". These conflict once the trunk is trainable. Chosen: V3 detaches the response
channel's feature inputs too, so no head term carries gradient into the trunk in any
configuration. This costs nothing in T-B, where the trunk is frozen — which is exactly why it
needed deciding now rather than at the from-scratch step.

**5. The `E_gap` cap aborts on a sustained breach, not a per-batch one.** Section 5 asks for
`Delta_bind <= E_gap` asserted per batch. A single 8-frame pristine draw can spike the value
without any escape being real, so the abort triggers on the epoch **median** breaching in two
consecutive epochs. Every individual frame-level breach is still logged, so nothing is hidden;
only the stopping rule is softened.

**6. Retiring `decay_r0` required raising the initial decay length.** Referencing the hopping
decay at `r_max` rather than the material-specific 3.0 A divides every element by e^2 at the
old initial length, starting the head effectively disconnected — the regime the `t_min` floor
exists to prevent. `decay_init` raised 1.0 -> 2.5 A. This is a new constant, but a decay
length, not a cutoff, so constraint 2 still holds.

**7. `E_gap` = 2.2 eV from literature, not from the source data.** No VASP outputs are present
locally, so the Zenodo route was unavailable without a bulk download. Recorded with provenance,
and deliberately not the paper's tuned HSE+SOC 3.08 eV, PBE+SOC ~1.3 eV, or experiment 2.93 eV.
Note separately that the dataset's own `band_edges.json` carries symmetric +/-1.2 eV edges,
implying 2.4 eV — read as a label-referencing convention rather than a measured gap, which is
why 2.2 is the default and 2.4 runs as a sensitivity arm.

**8. `w_g` set equal to `w_e`.** Section 3 says "the same scale". Because the untrained head
places the two band edges tens of eV apart, `L_gap` starts orders of magnitude above the force
loss before collapsing. The raw residual is logged beside the weighted value so the behaviour
is visible; no reweighting or ramp was added.

**9. V3 is a subclass of the existing head.** So there is exactly one eigensolve in the
codebase rather than a second copy to drift. The only change to the parent is lifting its
inline on-site block into an overridable method, verified behaviour-preserving against the
full suite before V3 was built on it.

**10. Construction refuses ambiguous descriptors.** The "first block" slice is only the first
block when the model has more than one readout; with a single readout the feature assembly
silently uses the *last* block instead. V3 raises at construction rather than relying on the
trunk-independence test to catch it later.
