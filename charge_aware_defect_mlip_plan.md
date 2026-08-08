# Charge-Aware Defect MLIP — Implementation Plan

A carrier-conditioned machine-learned interatomic potential for point defects, giving
charge- and spin-dependent energies, forces and curvature at hybrid-functional accuracy
and MLIP cost.

---

## 1. Purpose and deliverables

**Target capability.** Quantitative defect thermodynamics and kinetics for systems where
the charge state is part of the physics rather than a post-hoc correction:

- formation energies comparable across charge states and across supercell sizes;
- charge transition levels — see the accuracy split below;
- configuration coordinate diagrams — which need charge-dependent *forces* and curvature,
  not just charge-dependent energies;
- phonons, Hessians, finite-temperature free energies;
- orderings of competing metastable configurations at 1–2 meV/atom;
- migration barriers for mobile defects, whose charge state changes the barrier;
- compensated and multi-defect cells, including donor–acceptor pairs;
- all of the above evaluated **directly in the isolated-defect limit** rather than in a
  periodic cell and corrected toward it.

**Two accuracy targets, not one.** Fixed-geometry, fixed-cell charge-state differences —
vertical transitions, configuration-coordinate offsets at fixed `R`, spin-state splittings
— are algebraically free of base-model error and are targeted at **0.02 eV**.
Relaxed-geometry quantities, including thermodynamic transition levels, evaluate `E_base`
at two *different* minima and inherit its error at first order; they are targeted at
**0.05 eV** unless and until the base model is demonstrated better than ~10 meV on relaxed
charge-state differences (§11). Quoting a single 0.02 eV figure for both is not supportable
by the error budget and should not be done.

**Why an MLIP.** The geometry search is combinatorial in (defect, charge state,
symmetry-breaking distortion) and cannot be truncated — symmetry-breaking searches
routinely find the global minimum in a small minority of trial distortions. Multiplying
that by charge states and hosts puts the DFT cost out of reach.

**Design commitment.** Four properties:

1. **Carriers, not net charge.** Electron-like and hole-like carriers are counted
   separately per spin channel, so distinct physical states never collapse onto the same
   model input. Structural.
2. **Intensive in carriers, not extensive in atoms.** The charge correction is a
   *normalised* pool over atoms, so its magnitude is set by the number of carriers rather
   than by cell size. Exact on pristine cells of any structure; on defective cells the
   residual drift is exponentially suppressed by a *learned* logit gap, so this is a
   measured property (§8.2), not a guarantee. §4 and §10.1 record what keeps the
   suppression in place.
3. **Explicit long-range electrostatics.** A latent Ewald term carries the polarisation
   response, image interaction and inter-defect coupling that no finite receptive field
   can represent. Because the defect charge distribution is then an explicit model output,
   the finite-size operation at inference becomes a boundary-condition swap on that
   distribution rather than an energy correction applied to labels.
4. **Residual on a reference-state base.** Charge physics is a separable correction to a
   charge-blind base potential, so *fixed-geometry* charge-state differences are free of
   base-model error. This exactness does not extend to relaxed-geometry differences.

**On potential alignment.** The training targets contain no convention-dependent alignment
*constant* — every term is a total-energy difference on a common footing (§2.3, §2.5). The
physical finite-size artifact that alignment schemes estimate — the `q·δV̄` term arising
from the added charge interacting with the defect's own potential perturbation and with the
compensating background — has **not** been eliminated. It remains in the raw labels, and
the architecture's claim is that it is *carried* by the long-range branch (specifically by
the carrier–host cross term and the polarisation channel of §3.4) and correctly removed by
the boundary-condition swap of §7. That is a falsifiable modelling claim, tested in §8.3
and §8.7, not a cancellation.

---

## 2. Definitions and conventions

Fix these first; most of what follows is a consequence.

### 2.1 Carrier counters

Four non-negative integers per configuration, one per (carrier type, spin channel):

```
n_e^maj, n_e^min     electron-like carriers, majority / minority channel
n_h^maj, n_h^min     hole-like carriers,     majority / minority channel

q   = (n_h^maj + n_h^min) − (n_e^maj + n_e^min)
M_s = (n_e^maj − n_h^maj) − (n_e^min − n_h^min)      ≥ 0 by channel labelling
```

`M_s` is the number of unpaired electrons (2·S_z), not S_z.

Index the four channels by `c ∈ {e·maj, e·min, h·maj, h·min}` with charge sign
`s_c = −1` for electron channels and `s_c = +1` for hole channels. Write the vector of
counters as `n = (n_c)`.

Worked examples:

| state | (n_e^maj, n_e^min, n_h^maj, n_h^min) | q | M_s |
|---|---|---|---|
| closed-shell neutral (reference) | 0, 0, 0, 0 | 0 | 0 |
| donor, q = +1 | 0, 0, 0, 1 | +1 | 1 |
| acceptor, q = −1 | 1, 0, 0, 0 | −1 | 1 |
| neutral triplet | 1, 0, 0, 1 | 0 | 2 |
| neutral open-shell singlet | 1, 0, 1, 0 | 0 | 0 |
| q = +2, M_s = 0 | 0, 0, 1, 1 | +2 | 0 |
| q = +2, M_s = 2 | 0, 0, 0, 2 | +2 | 2 |
| compensated D⁺A⁻, triplet | 1, 0, 0, 1 | 0 | 2 |
| compensated D⁺A⁻, singlet | 1, 0, 1, 0 | 0 | 0 |

#### Canonicalisation — required, not optional

Time reversal maps `(n_e^maj, n_e^min, n_h^maj, n_h^min) → (n_e^min, n_e^maj, n_h^min,
n_h^maj)`. The two labellings describe the same physical state and must receive the same
energy. Choosing labels so that `M_s ≥ 0` fixes the gauge **only when `M_s ≠ 0`**. At
`M_s = 0` both labellings satisfy the rule — `(1,0,1,0)` and `(0,1,0,1)` are the canonical
example — and because `MLP_u` is per-channel with independent parameters, the model will
otherwise assign them different energies.

Fix this at the input, not with a loss term or a symmetrisation over two forward passes:

```
canonical(n) = n                                      if M_s(n) > 0
             = swap(n)                                if M_s(n) < 0
             = lexicographic max of { n, swap(n) }     if M_s(n) = 0
```

The `M_s` clause comes first: a bare lexicographic rule would select `(0,0,1,0)` over
`(0,0,0,1)` and hand back `M_s = −1`, contradicting the sign convention above. The
lexicographic tie-break applies only to the `M_s = 0` case that convention leaves open.
Every entry in the worked-examples table is canonical under this rule.

Apply it at data loading **and** at every inference entry point. The map from physical state
to model input is then single-valued and time-reversal invariance holds identically for any
parameters. The §8.1 unit test then checks that canonicalisation is actually applied
everywhere rather than checking a property of the network.

#### Three properties that matter

- The domain is a **full non-negative integer lattice**. Conditioning on `(q, M_s)`
  instead gives a parity checkerboard — at fixed `M_s` only every other `q` is valid — so
  continuous conditioning would interpolate across an unphysical half-lattice.
- The reference state `n = 0` is the closed-shell neutral state **only**. Excited neutral
  states and compensated multi-defect cells are distinct labels, not aliases of it.
- Fractional counters are meaningful under ensemble DFT, so the conditioning extends
  continuously if fractional-electron labels are later added.

#### The counters are an assignment, not a measurement

`n` is not a function of the DFT output. A given `(q, M_s)` is consistent with infinitely
many counter vectors differing by added `(1,0,1,0)` or `(0,1,0,1)` pairs, and deciding
between `(0,0,0,0)` and `(1,0,1,0)` for a neutral singlet requires looking at the converged
orbitals. This is deliberate — it is what lets the closed-shell and open-shell singlets be
distinct model inputs — but it has two consequences that must be honoured:

- **Labelling rule.** Assign the *minimal* counter vector consistent with the converged SCF
  solution's occupation pattern. Record the frontier occupations that justified the
  assignment (§5.4). A labelling that is not reproducible from stored metadata is a
  data-integrity failure.
- **Inference is a bounded enumeration.** Formation energies are assembled by minimising
  over counter configurations at fixed `q`. That candidate set must be bounded and the
  minimisation gated (§7.3), since minimising a learned surrogate over labels it may never
  have seen is not variational.

### 2.2 Reference state

`n = 0`: neutral, closed-shell, `M_s = 0`. `E_base` is trained on this state alone.

Where the neutral **ground** state is high-spin, it carries `n = (1,0,0,1)` and is handled
by the correction branch — the base surface stays a closed-shell surface everywhere. This
is why the reference state is defined by the counters rather than by "whatever the neutral
ground state is": the base PES must be a single, consistently-defined surface for the
residual decomposition to mean anything.

**Implication for data generation.** Reference-state labels must be spin-constrained to
`M_s = 0`. Do not let the SCF relax the moment, or the label will not describe what was
computed.

**Reference-surface hazard.** The `M_s = 0` constraint defines a surface; it does not
guarantee a well-behaved one. At high-symmetry geometries with a partially occupied
degenerate level — the undistorted neutral vacancy is the standard case — the constrained
solution has fractional occupations, its energy depends on the smearing scheme, and it can
develop derivative discontinuities where levels cross along `R`. Because `ΔE_SR` is
*defined* as the residual against this surface, any such artifact is pushed into the branch
whose exactness is being claimed.

Mitigation, and it is only partial: record the smearing width and the frontier occupations
for every reference-state label (§5.4); flag configurations whose closed-shell frontier gap
falls below a threshold; and check `E_base` smoothness explicitly along coordinates passing
through flagged geometries. Treat a discontinuity found this way as a data problem at that
geometry, not as a base-model fitting failure.

### 2.3 Training target — band-edge referencing

Removed electrons go to the valence band maximum; added electrons come from the conduction
band minimum:

```
ΔE_target(R, n) = E_raw(R, n) − E_raw(R, 0) − Σ_σ [ n_e^σ · E_CBM^cell − n_h^σ · E_VBM^cell ]
```

`E_raw` are **uncorrected** DFT total energies (§2.4). `E_CBM^cell` and `E_VBM^cell` are
defined thermodynamically, as total-energy differences **in the same supercell** as the
labels they reference and in the same convention (§2.5) — *not* as eigenvalues. Every
quantity in the expression is then a total-energy difference on a common footing and no
convention-dependent constant enters the target.

Note the superscript. §2.5 also requires size-converged edges `E_VBM^∞`, `E_CBM^∞`, but
those are used only in the affine map to formation energies at inference (§7.2). Mixing a
size-converged edge into a finite-cell target reintroduces exactly the reference mismatch
the construction is meant to avoid. Store both; keep them distinct.

The referencing term is **linear in the counters over the whole domain** — there is no
`max(n, 0)` split and therefore no derivative discontinuity to interpolate through.

Verify the construction on a pristine host, where the carrier sits at the band edge and
the target must be exactly zero:

| state | raw difference | referencing subtracts | ΔE |
|---|---|---|---|
| donor (0,0,0,1) | −E_VBM | −E_VBM | 0 |
| acceptor (1,0,0,0) | +E_CBM | +E_CBM | 0 |
| triplet (1,0,0,1) | +E_g | E_CBM − E_VBM | 0 |
| open-shell singlet (1,0,1,0) | +E_g | E_CBM − E_VBM | 0 |

What the model learns is therefore the **carrier binding energy relative to the band
edge**: how much cheaper it is to place the carrier on the defect than at the band edge.
In the dilute limit that quantity is bounded by roughly `E_g` and is zero for an unbound
carrier. At finite cell size it additionally contains the image energy, which for `|q| = 2`
in a small cell can be a substantial fraction of `E_g`; the target is still far better
conditioned than a supercell ionisation energy, but the boundedness argument is a
dilute-limit statement.

Recovering formation energies and transition levels from `ΔE` is an affine map with known
constants; no information is lost. That map must be applied explicitly (§7.2) — it is not
automatic.

**Band edges are functional-specific.** `E_VBM` and `E_CBM` must come from the same
functional as the labels they reference. Maintain separate values per fidelity branch.
Substituting an experimental gap makes the target inconsistent with the DFT it is mixed
with.

### 2.4 Train on raw energies

Charged-supercell DFT energies are computed with a compensating uniform background and the
`k = 0` reciprocal-space term omitted. The long-range term of §3.4 uses the identical
convention. The two are therefore consistent in convention, and the periodic image
interaction is **modelled** rather than corrected out of the labels.

One qualification. The model's net latent charge is `a·q`, not `q` (§3.4), so the
correspondence is not literally charge-for-charge: it holds at the level of the *screened*
monopole, with `a² = 1/ε_∞` supplying the screening that DFT supplies through the actual
electronic response. The charge–background interaction is screened by the same factor in
both, so the convention match survives; but it means the consistency claim rests on `a`
being right, which §3.4 and §6 address directly.

Consequences:

- Energy labels are raw DFT totals. Force labels are raw DFT forces. No correction term
  enters either, so there is no `∇E_corr ≠ 0` inconsistency between corrected energies and
  uncorrected forces.
- Cell-size behaviour becomes a genuine, falsifiable property of the model rather than an
  assumption baked into the labels.
- Finite-size corrections move **out of the training pipeline entirely** and become an
  inference-time operation on the model itself (§7).

### 2.5 Determining the band edges

Take the band edges from **pristine supercell total energies**, not from eigenvalues:

```
E_CBM   =  E_raw(pristine, one electron added) − E_raw(pristine, 0)
−E_VBM  =  E_raw(pristine, one hole added)     − E_raw(pristine, 0)
```

Compute these twice:

- **`E_CBM^cell`, `E_VBM^cell`** — in the *same supercell* as the production labels, in the
  same background convention. These go into `ΔE_target` (§2.3). Whatever finite-size error
  they carry is a constant per host per cell and cancels out of the target construction
  rather than contaminating it.
- **`E_CBM^∞`, `E_VBM^∞`** — converged against supercell size. These are used only in the
  formation-energy expression at inference (§7.2). Record the offset between the two; it is
  a known constant and appears explicitly in the affine map.

Two finite-size effects must be handled, and both are controllable:

- **Band filling.** The carrier occupies dispersive states extending above the edge rather
  than sitting exactly at it. Apply the standard band-filling correction using the
  occupations and eigenvalues that are already being stored. The correction shrinks with
  cell size, so converge it for the `∞` values.
- **k-point placement.** If the CBM or VBM lies at a k-point not sampled by the supercell
  mesh, the computed edge is wrong outright rather than slightly off. Choose the supercell
  and mesh so the relevant k-point is included. This is under your control and should be
  checked explicitly per host, not assumed.

**Do not apply an image correction to these two calculations.** A delocalised band carrier
largely cancels the compensating background, so the point-charge-in-jellium correction does
not apply. Applying it would inject a spurious `q²/L` term into the reference constants and
therefore into every transition level.

Converge the `∞` edges to well below 0.02 eV. An error `ε` propagates as `q·ε` into every
reported transition level.

---

## 3. Architecture

Everything below sits on top of a standard equivariant message-passing model that produces
invariant per-node features `h_i` from geometry alone. Only the additions are specified.

### 3.1 Branch structure

```
E_total(R, n) = E_base(R) + ΔE_SR(R, n) + E_LR(R, n)
```

- **Base branch.** Sees geometry only. Trained on reference-state configurations. Produces
  a per-atom energy and a per-atom host latent charge. The host–host long-range energy
  `E_LR^host` is part of this branch: it is a function of geometry alone and is present at
  `n = 0`.
- **Correction branch.** Sees the counters `n`. Produces `ΔE_SR` (§3.3) and the
  `n`-dependent latent-charge channels feeding `E_LR` (§3.4).

Every `n`-dependent term carries an explicit factor of a counter, so at `n = 0` all of them
vanish identically and `E_total(R, 0) = E_base(R)` **exactly, for any parameters**. Read
`E_base` as including `E_LR^host` when checking this. It is a unit test, not a metric
(§8.1).

The trunk producing `h_i` is shared. Set `correction_trunk: shared` initially; a
`separate` option should exist for the case where charged-data contamination of the base
representation proves to be a problem.

### 3.2 Carrier attention pooling — the extensivity mechanism

For each channel `c`, two per-atom readouts and a normalised pool:

```
ℓ_i^c = MLP_ℓ^c(h_i, n)                     # scalar logit
α_i^c = softmax over i of ℓ_i^c             # Σ_i α_i^c = 1, per cell, per channel
u_i^c = MLP_u^c(h_i, n)                     # scalar energy readout
```

`α_i^c` is the carrier's occupation distribution over atoms; `Σ_i α_i^c u_i^c` is its
expectation energy. Because the weights are normalised, the pooled quantity is
**intensive** and the total correction scales with carrier count, not atom count.

Why this gives size-consistency:

- **Pristine cells are exact for any structure.** With `n_a` atoms of each symmetry class
  `a` in a cell of `N_f` formula units, the pooled value is
  `Σ_a n_a e^{ℓ_a} u_a / Σ_a n_a e^{ℓ_a}` — `N_f` cancels identically. This holds for any
  stoichiometry, any polymorph, with no reference structure supplied and no host label.
- **Defective cells drift exponentially, and the drift is measurable.** With a defect
  region at logit `ℓ_d` and bulk at `ℓ_b`, writing `r = e^{ℓ_b − ℓ_d}`, the size error in
  the pooled energy is `≈ N·r·(u_b − u_d)`. The logit gap can be read directly off a
  trained model, giving an analytic bound on the size error with no DFT and no second cell
  (§8.2).

**The suppression is learned, and one regulariser keeps it in place.** The drift grows
*linearly in N* while the physical effects being modelled decay as `1/L`, so the gap
required at production size is substantial: holding 20 meV at `N = 1000` with
`|u_b − u_d| ≈ 1 eV` needs `Δℓ ≈ 11`. Nothing in the energy loss demands this, because
training at a single cell size only constrains the *product*. What does demand it is the
weak L2 on `u_i^c` (§4): fitting a 1 eV binding with `α_d = 1/216` requires
`u_d ≈ −216 eV`, which the penalty makes expensive, so the optimiser buys localisation
instead. That regulariser is therefore load-bearing for extensivity rather than incidental
housekeeping — tune it deliberately, record the value, and do not remove it as cleanup.
Its cost is recorded in §10.1.

This mechanism also makes N-independent physics natural rather than excluded. Band-edge
shifts under strain — the deformation-potential response — are representable directly by
`u`'s dependence on the pooled descriptor, so no separate reservoir term is needed and
strained hosts are not structurally out of scope.

### 3.3 Short-range correction

```
ΔE_SR(R, n) = Σ_c n_c · Σ_i α_i^c · u_i^c
```

Notes:

- Both `ℓ` and `u` take the **full counter vector** `n`, not just `n_c`. Multi-carrier
  short-range interaction (the on-site `U`, exchange between co-located carriers) is
  therefore representable, and the spatial profile of one carrier may depend on the
  presence of others.
- No functional form in `q` is imposed. Expect smooth eV-scale convexity of order `U/2`
  within a regime and a step at regime boundaries; linearity in `q` is a diagnostic, not a
  constraint.
- Separate `MLP_u` per channel. `MLP_ℓ` may be shared between the two spin channels of the
  same carrier type as a parameter economy; test both.
- The form `n_c · ⟨u⟩` is a **mean field over carriers within a channel**. It is exact when
  the `n_c` carriers occupy equivalent environments and fails by `O(u_A − u_B)` when they
  do not. §10.1 records the consequence.

### 3.4 Long-range term — latent Ewald

**Build on the existing MACELES implementation.** Latent Ewald Summation (LES) is an
established method with a working MACE integration; the Ewald routine, latent-charge
readout, structure factor, smearing convention and autograd force/stress paths should be
taken from it rather than reimplemented. What follows separates what LES already gives from
what this project adds on top.

#### What plain LES provides

A per-atom scalar hidden variable is predicted from local invariant features, and a
reciprocal-space sum over its structure factor supplies a long-range energy contribution:

```
S(k)  = Σ_i q_i exp(i k·r_i)

E_LR  = (1 / 2ε₀V) Σ_{k ≠ 0, |k| ≤ k_c}  [ exp(−σ²|k|²/2) / |k|² ] · |S(k)|²
```

The properties inherited unchanged:

- Latent charges are fitted **only to energies and forces**. No charge labels, no
  partitioning convention, no charge equilibration and no self-consistent solve. This is the
  right choice here for the same reason it is there: partitioned charges are not
  observables and depend on the partitioning scheme.
- **`k = 0` is omitted**, which is the neutralising-background convention and matches the
  DFT convention for charged supercells. This is what allows raw energies to be used as
  labels (§2.4).
- The hidden variable may be multi-dimensional, and the total long-range energy is
  aggregated over dimensions after the Ewald sum.
- Cost is roughly twice a short-range model.

In plain LES the latent charges are left entirely unconstrained — no neutrality, no total
charge. That is harmless in neutral systems, where the net latent charge is near zero by
symmetry anyway. It is not harmless here, which is where the additions begin.

#### Addition 1 — structured charge assembly

The conditioning variable is a global extensive carrier count, so an unstructured per-atom
readout `q_i(h_i, n)` would give every atom a charge shift and `Σ_i q_i` would scale with
cell size. That is the extensivity failure of §3.2 reappearing in the charge channel.

Split the latent charge into three channels — a neutral host channel, a neutral
polarisation channel, and a normalised carrier channel with a single free amplitude:

```
q_i^host     = Q_host(h_i) − mean_j Q_host(h_j)             # geometry only; exactly neutral
p_i          = MLP_p(h_i, n)
q_i^pol      = (Σ_c n_c) · [ p_i − mean_j p_j ]             # n-dependent; exactly neutral
a            = softplus( MLP_a( mean_i h_i ) )              # one scalar per configuration
q_i^carrier  = a · Σ_c s_c · n_c · α_i^c                    # normalised; carries the monopole

q_i          = q_i^host + q_i^pol + q_i^carrier
Σ_i q_i      = a · q
```

The shape of the carrier charge distribution is carried by the attention weights already
computed for §3.3, and its magnitude by `a`. This retains LES's freedom exactly where it
belongs — the magnitude — while imposing structure exactly where the physics requires it:
the total. One `a` is shared across all four channels, since electrons and holes are
screened by the same medium.

**Why the polarisation channel is needed.** Without `q_i^pol`, the entire `n`-dependence of
the latent charge is `a·Σ_c s_c n_c α_i^c`, and since `α_i^c ≥ 0` each channel's
contribution is **sign-definite**: an electron channel can only make atoms more negative,
never more positive. The true `Δρ` on adding a carrier is sign-changing at the atomic scale
— the centre gains while the neighbouring shell depolarises — and its second radial moment
is precisely what carries the `1/L³` and alignment-like terms that §1 declines to correct
by hand and §7 must therefore remove correctly. A sign-definite `n`-dependence cannot
produce that structure, and `q_i^host` cannot supply it either, since it has no `n`
dependence at all.

`q_i^pol` closes this at negligible cost: one extra head, no additional Ewald sums, and
three properties preserved exactly —

- it sums to zero by mean subtraction, so `Σ_i q_i = a·q` is unchanged;
- it carries an explicit factor `Σ_c n_c`, so it vanishes at `n = 0` and the identity
  `E_total(R, 0) = E_base(R)` survives;
- it is unconstrained in sign, so the near-field structure the far field needs is
  representable.

It supplies *shape*, not screening magnitude. The screening magnitude stays in `a`. That
separation is deliberate — see §3.6 for why it must not be merged.

#### Addition 2 — the dielectric constant as an output

`a` has a physical reading. Screened and bare kernels are related by a rescaling of the
charges,

```
(1/2ε₀V) Σ_k [e^(−σ²k²/2) / (ε_∞ k²)] |S[q]|²  ≡  (1/2ε₀V) Σ_k [e^(−σ²k²/2) / k²] |S[q/√ε_∞]|²
```

so putting `ε_∞` in the kernel of an isotropic host imposes no constraint whatsoever — it
merely fixes the numerical scale the optimiser lands on. **Use the bare `1/|k|²` kernel and
supply no dielectric constant in the kernel.** The screening is absorbed into the learned
amplitude, and

```
√(1/ε_∞)  =  a
```

is read off the trained model. Note what the identity does and does not say. It says the
*kernel choice* carries no information, so nothing is lost by using the bare form. It does
**not** say the loss is flat in `a`: different `a` means different image energy and
different physics. The flatness is a separate problem, caused by `a²·(image energy)` being
degenerate with a constant-per-`n` offset in `ΔE_SR` at fixed cell size, and the next
subsection is about breaking that degeneracy with data rather than leaving `a` wherever
initialisation puts it.

`a` is a function of the **plain mean-pooled host descriptor** `mean_i h_i`, not of any
attention-weighted pool. It is a host property and must not inherit the defect's local
environment. The mean converges to the bulk descriptor as `N` grows, which is the intended
behaviour; the residual `O(1/N)` concentration dependence is a diagnostic, not a feature.
`a` must **not** depend on `n`. If it needs `n`-dependence to fit, that is a finding rather
than a hyperparameter.

#### Addition 2b — what actually identifies `a`, and what does not

This determines the accuracy of every dilute-limit number in §7, so it is worth being
explicit about where the signal comes from.

**Neutral data contributes nothing.** `a` multiplies `Σ_c s_c n_c α_i^c`, which is zero at
`n = 0`. This is the same fact as the exactness property in §3.1, seen from the other side:
no neutral configuration — bulk or defective, however many of them — produces a gradient on
`a`. Adding bulk cells does not help.

**Neutral data cannot supply `ε_∞` indirectly either.** In a bare-kernel latent-charge
model, matching LO–TO splitting fixes the *product* `q̃^host = Z*/√ε_∞`; nothing in neutral
energies or forces separates the two factors. A geometry-only latent charge also has no
electronic polarisability at all — hold the nuclei fixed, apply a field, nothing responds —
so whatever dielectric behaviour such a model reproduces in bulk sits on an implicit
`ε_∞ = 1` with the real value folded invisibly into the charge scale. Inferring the scale
requires a known integer net charge, which only a charged cell supplies.

**In polar hosts, single charged defect cells are sufficient.** Bulk data pins
`q̃^host = Z*/√ε_∞`; the carrier–host cross term then gives a far-field force

```
F ≈ a·q·q̃^host / r² = q·Z* / (ε_∞ r²)
```

which is the correctly screened result, linear in `a`, available at one cell size from data
already being generated. No multi-size study is required.

**In non-polar hosts this signal is absent** — see §10.1 for the consequence and the
workaround.

**Three additional identifiers, all cheap and none requiring a size series:**

1. **Cell-shape contrast at fixed atom count** (§5.3). A tetragonal supercell alongside a
   cubic one has a substantially different Madelung constant while the local environment is
   essentially unchanged, so `E_cc ∝ a²q²α_M(shape)` varies while `ΔE_SR` does not.
   Compute both Madelung constants analytically first (§6.6) so the contrast is known
   before the calculations are run.
2. **The hydrostatic component of the charged-cell stress** (§9.5). `∂E_cc/∂ln V ∝ a²q²/L`
   is quadratic in `q`, which separates it from the deformation-potential response
   (linear in `q`).
3. **Well-separated multi-defect cells** (§5.3). The `−a²/r` tail beyond the receptive
   field is something `ΔE_SR` structurally cannot mimic. Note that this subset therefore
   does double duty: it is not only a test of donor–acceptor physics but a primary
   identifier of `a`.

**Optional weak prior.** One DFPT `ε_∞` per host, entered as `λ(a − 1/√ε_∞)²` with small
`λ`. This sits uneasily with the preference for measured over supplied quantities, so the
case for it should be made honestly: it is a prior on a direction the data constrains
weakly, chosen precisely because that direction would otherwise be set by initialisation.
Keep `λ` small enough that genuine data can overrule it, always report the fitted `a`
alongside the prior value, and treat tension between them as a finding about the
identifying data rather than as something to tune away.

**The constancy check is not a correctness check.** `a` should come out constant across
charge states, defects and geometries within a host, and drift means the long-range branch
is absorbing something that does not belong to it (§8.9). But a uniformly *wrong* `a` is
perfectly constant and passes this test. Correctness is established against DFPT `ε_∞`
(§8.7), not against constancy.

#### Addition 3 — a second boundary condition

Implement the same latent charges under an isolated (non-periodic) screened-Coulomb sum
alongside the periodic one, sharing the smearing and self-energy code paths. This is what
makes the finite-size operation a boundary-condition swap, and it is described in §7.

#### Settings

- **Fix `σ`, do not learn it.** Around 1 Å; the published range 0.5–2 Å shows weak
  sensitivity. It should be a fraction of the message-passing cutoff. Learning it worsens
  short-range/long-range identifiability for no gain.
- **`k_c` around π Å⁻¹**, converged and recorded. It is a convergence parameter, not a
  hyperparameter to tune against validation loss.
- **Hidden-variable dimension: one.** The three channels of Addition 1 are summed into a
  single scalar `q_i` before the structure factor, so there is one Ewald sum, not six. LES's
  multi-dimensional hidden variable is a *different* freedom — several independent latent
  charges each with its own sum — and it is not needed here, because the channel structure
  is prescribed by the physics rather than discovered. Do not add free dimensions on top
  until something demands them; each one adds identifiability slack in the branch that is
  already hardest to identify.

#### What this term buys

- The **image interaction** with the periodic replicas is generated with the correct `q²`
  and `1/L` scaling, from an object the model computes rather than a correction applied to
  labels.
- The **electron–hole cross term** appears automatically, with the right sign, the right
  `1/r` decay and the correct screening `a² = 1/ε_∞`, because `E_LR` is quadratic in `S(k)`
  and `S = S_host + S_pol + S_carrier`. This is what delivers compensated donor–acceptor
  pair energetics.
- The **defect charge distribution is explicit and separable**, which is what makes the
  inference procedure of §7 possible.
- The base branch acquires long-range electrostatics, which polar-phonon and LO–TO
  behaviour needs independently of anything charge-related.

### 3.5 Multi-fidelity

```
E_HSE(R, n) = E_PBE(R, n) + Δ_fid(R, n)
```

`Δ_fid` has the same internal structure — base head, pooled correction, latent-charge
contribution — with its own parameters, referenced to **its own** band edges. Train
jointly, not by sequential fine-tuning. Because the two branches use different referencing
constants, `Δ_fid` absorbs `q·(E_VBM^HSE − E_VBM^PBE)` and its CBM counterpart as constants
per host; that is expected and is not a sign of misfit.

The residual carries only what the functionals disagree about, which is well-conditioned
where both functionals place the defect level in the gap. It is **not** well-conditioned
where the semi-local functional makes the level resonant with a band while the hybrid
places it in the gap: there `Δ_fid` must supply the entire binding energy from the sparse
subset. Stratify the hybrid subset across charge states specifically for this reason, and
treat large `Δ_fid` as a flag rather than as ordinary residual.

### 3.6 Deliberately not included

- **No separate force or stress head.** Forces are `−∇_R E`, stress is the strain
  derivative, both by autograd. An independent vector head is not the gradient of any
  scalar and breaks MD stability, Hessian symmetry, thermodynamic integration between
  charge states, and the claim that a relaxed geometry is a minimum of the model's own
  energy.
- **No defect-position input, no distance-to-defect mask, no gating.** Nothing at
  inference requires knowing where the defect is, so vacancies and mobile defects are
  handled without special-casing. Defect positions may still be recorded and used *post
  hoc* for auditing; the prohibition is on inputs, not diagnostics.
- **No `1/N` normalisation anywhere.** It fixes cell-size scaling at the cost of making
  the correction depend on defect count, which is worse.
- **No charge equilibration, and no explicit screening charge.** The tempting alternative
  to the `a` rescaling is to make the screening literal — add a polarisation cloud that
  integrates to `−(1 − 1/ε_∞)q` so that `Σ_i q_i = q` exactly. Do not. In a local
  dielectric that induced charge is co-located with the source, so the net long-range
  monopole becomes `q/ε_∞` and a bare-Coulomb energy over the total gives `q²/(ε_∞² r)`
  instead of `q²/(ε_∞ r)`. Recovering the right factor requires an explicit polarisation
  self-energy — a hardness term — and getting the induced charges right requires minimising
  over it, which is charge equilibration arriving by the back door. The `q → q/√ε_∞`
  rescaling is the correct shortcut precisely because it reproduces the dielectric free
  energy without an explicit polarisation energy. This is why `q_i^pol` is constructed to
  be exactly neutral: it supplies multipole *shape* and is forbidden from touching the
  monopole, which stays with `a`.

---

## 4. Training objective

**Scale convention, fixed once.** All model-side charge-dependent quantities are on the
**band-edge-referenced** scale of §2.3. Define

```
E_target(R, n) = E_raw(R, n) − Σ_σ [ n_e^σ · E_CBM^cell − n_h^σ · E_VBM^cell ]
```

so that `E_target(R, 0) = E_raw(R, 0)` and `E_target − E_base ≈ ΔE_target`. Every loss
below uses this definition. Getting this wrong makes losses (b) and (c) disagree by the
band-gap constant, which is a silent, uniform shift of every transition level.

Three terms on different subsets.

**(a) Base**, on reference-state configurations:

```
L_base = Σ |E_base(R) − E_DFT(R, 0)|  +  λ_F Σ ‖∇E_base(R) + F_DFT(R, 0)‖
```

**(b) Delta**, on paired configurations where both `n = 0` and some `n ≠ 0` were computed
at the same geometry:

```
L_Δ = Σ |ΔE_pred(R,n) − ΔE_target(R,n)|
    + λ_F Σ ‖∇ΔE_pred(R,n) + (F_DFT(R,n) − F_DFT(R,0))‖
```

where `ΔE_pred = ΔE_SR + [E_LR(R,n) − E_LR(R,0)]`. The force target is a plain DFT force
difference — no correction term appears in it, because no correction was applied to the
labels.

This is the term that carries the headline observables, since fixed-geometry charge-state
differences are exactly free of base-model error. Weight it accordingly.

**(c) Unpaired totals**, on configurations with a charged label but no reference-state
partner at that geometry:

```
L_tot = Σ |stopgrad(E_base(R)) + ΔE_pred(R,n) − E_target(R,n)|
```

`E_base` is **detached** here. Without that, unpaired charged data reshapes the base PES
and the residual decomposition stops meaning what it is supposed to mean. The cost of
detaching is that base-model error at these geometries is absorbed into the correction, so
this term should be down-weighted relative to (b) and pairing should be dense enough that
it is not load-bearing.

**(d) Charged-cell pressure**, conditional on the verification in §9.5:

```
L_P = λ_P Σ | P_pred(R,n) − P_DFT(R,n) |
```

The hydrostatic component only. Its purpose is identification of `a` (§3.4), so it earns
its place in hosts where the far-field force signal is weak. Include it only after the
finite-difference check in §9.5 passes for the code in use; if it fails, drop the term
rather than working around it.

**Regularisation.**

- **Weak L2 on `u_i^c`.** This is not housekeeping. It is what forces the model to buy
  localisation rather than large cancelling readouts, and it is therefore the mechanism
  that keeps the logit gap — and hence extensivity — where §3.2 needs it. Sweep it
  deliberately, record the chosen value with the model, and re-run §8.2 whenever it
  changes. Its side effect is recorded in §10.1.
- Weak L2 on `q_i^host` to keep the host latent charges from absorbing short-range energy.
  This is the main lever against short-range/long-range non-identifiability, alongside
  fixing `σ`. Note the tension: in a polar host this same penalty suppresses the
  carrier–host cross term that identifies `a` (§3.4), so keep it weak and check `a`
  against the §8.7 cross-check whenever it is strengthened.
- Weak L2 on `p_i`, on the same reasoning as `q_i^host`: the polarisation channel exists to
  supply multipole structure, not to absorb short-range energy.
- Optional weak prior on `a` (§3.4).
- Monitor, do not penalise, the logit gap.

---

## 5. Data requirements

### 5.1 Geometry pool

- Symmetry-breaking distortion sets, generated per (defect, counter configuration).
  Distortion count should scale with the effective excess charge so each state gets
  appropriate starting geometries.
- Configurations sampled along relaxation trajectories.
- MD frames, from validated basins only and only once a first model exists.
- Coverage, not iteration, is the binding constraint: active learning refines basins
  already found and will not discover one the initial search missed. Budget redundancy in
  the initial generation.

### 5.2 Labelling

- **Reference-state labels dense, not universal.** A paired `n = 0` single-point at a given
  geometry is what enables the exact difference loss; unpaired configurations still train
  through `L_tot`. Density should be set by how well `ΔE` is determined, which frees budget
  for additional carrier configurations and hosts.
- **Enumerate carrier configurations directly.** Do not enumerate `(q, M_s)` and convert;
  enumerate `(n_e^maj, n_e^min, n_h^maj, n_h^min)` and derive `q` and `M_s`. Parity is then
  satisfied by construction.
- **Canonicalise every counter vector at write time** (§2.1) and store the canonical form.
- **Constrain the moment explicitly** in every calculation, including the reference state.
- Record which SCF solution converged (local moments, frontier eigenvalues, occupations),
  because at fixed `n` the SCF can settle on excited electronic configurations that look
  like valid data — and because the counter assignment itself is only reproducible from
  that metadata (§2.1).

### 5.3 Required subsets

| Subset | Purpose | Notes |
|---|---|---|
| Reference-state defective | `E_base` | Spin-constrained to `M_s = 0`; smearing and frontier occupations recorded |
| Paired charged | `L_Δ` — the primary signal | Dense |
| Unpaired charged | `L_tot` | Down-weighted |
| **Compensated pairs** | Donor–acceptor cells at `q = 0` | Both singlet and triplet labels; this is the subset the counter scheme exists to make representable |
| **Multi-defect, well separated** | Long-range cross term **and** identification of `a` | Separations **beyond** the message-passing receptive field. Load-bearing for §7, not only for D–A physics |
| **Shape-varied charged** | Identification of `a` | Charged *defect* cells in a second supercell shape (e.g. tetragonal alongside cubic) at the same atom count. Madelung contrast computed analytically first (§6.6) |
| Strained pristine and near-pristine | Deformation-potential response | Neutral; ordinary labelled data. Distinct in purpose from the shape-varied charged subset above |
| Multi-cell | Validation only, not training | Two or preferably three sizes for a small subset |
| Hybrid | `Δ_fid` | ~10%, stratified, **spanning carrier configurations rather than only geometries** |

Note the distinction between the two strain-related rows. **Strained pristine** cells are
neutral and exist to supply the deformation-potential response, which enters through `u`'s
dependence on the pooled descriptor. **Shape-varied charged** cells exist to break the
degeneracy between `a²·(image energy)` and a constant-per-`n` offset in `ΔE_SR`, which
requires a charge, a Madelung-constant contrast, and an essentially unchanged local
environment. Neither substitutes for the other. Both also feed the isotropy check in
§10.1, so the same calculations answer two questions.

The multi-cell row stays in validation. Requiring a size series for every host is not how
defect campaigns are run, and with the identifiers of §3.4 in place it is not necessary for
training — but the small validation series is what makes §8.3 a genuine falsification test,
so it should not be dropped either.

### 5.4 Store per configuration

| Quantity | Purpose |
|---|---|
| Raw energy, raw forces | Training targets |
| Canonical counter vector `n` | Conditioning label |
| Stress (all cells) | Reference-state training; charged-cell pressure per §9.5 |
| `E_VBM^cell`, `E_CBM^cell` | Band-edge referencing in `ΔE_target` (§2.3) |
| `E_VBM^∞`, `E_CBM^∞` | Affine map to formation energies (§7.2) |
| Band occupations and eigenvalues | Band-filling correction; counter-assignment provenance |
| Smearing scheme and width | Reference-surface hazard audit (§2.2) |
| Cell vectors | Long-range sum and stress |
| Local magnetic moments | Confirms which multiplicity converged |
| Frontier level position, IPR | Regime metadata; validation of `α` |
| DFPT `ε_∞` (per host, once) | Cross-check on `a` (§8.7); optional weak prior |
| Partitioned charges (Hirshfeld/DDEC) | **Validation only** — never a training target |

Partitioned charges are convention-dependent and are deliberately kept out of the loss.
Their role is to check that `α_i^c` is behaving, not to define it.


---

## 7. Inference — formation energies and transition levels

Formation energies refer to an isolated defect in an infinite crystal; training labels come
from finite periodic cells. This section is how the gap is closed.

Potential-based correction schemes need an alignment term because DFT supplies a total
energy and a potential but no separable defect charge distribution — the alignment is a
proxy for the model charge being wrong. This model holds the charge distribution explicitly
(§3.4), so the operation is applied to the charges directly. What this buys is that the
finite-size artifact is *modelled and removed under a changed boundary condition* rather
than *estimated by a fitting procedure on the potential*. It does not mean the artifact was
absent; §1 states the distinction.

### 7.1 The isolated-limit evaluator

`E_LR` is quadratic in the total structure factor, `S = S_host + S_pol + S_carrier`, so it
decomposes into three physically distinct pieces. Write `S_env = S_host + S_pol` for the
neutral environment channels, and define

```
E[A]       = (1/2ε₀V) Σ_{k≠0} f(k) · |A(k)|²
E_cross[A,B] = (1/2ε₀V) Σ_{k≠0} f(k) · Re[ A*(k) B(k) ]
```

with `f(k) = exp(−σ²k²/2)/k²`. Then

```
E_LR = E[S_env]  +  2·E_cross[S_env, S_carrier]  +  E[S_carrier]
```

Write the factor of 2 explicitly in the code and unit-test it against a two-charge analytic
case; an absorbed or doubled cross term is the second most likely prefactor bug here after
the ones in §9.4.

Each piece needs its own boundary condition:

| Piece | Boundary condition in the dilute limit | Why |
|---|---|---|
| `E[S_env]` | **Periodic** | Describes a genuine infinite crystal |
| `E_cross[S_env, S_carrier]` | Carrier in the central cell against the **infinite periodic** environment | Real physics: the defect charge in the lattice's own field. Survives to the dilute limit |
| `E[S_carrier]` | **Isolated** | Only the carrier's interaction with its own images and with the compensating background is spurious |

so

```
q_i^carrier    = a · Σ_c s_c · n_c · α_i^c

E_dilute(R, n) = E_base(R) + ΔE_SR(R, n)
                 + E[S_env]^periodic
                 + 2·E_cross[S_env, S_carrier]^periodic
                 + E[S_carrier]^isolated
```

**The cross term must be retained.** It is present in `E_total` and in `L_Δ`, where it
enters linearly in the carrier charge and supplies most of the geometry-dependent
long-range signal. Dropping it from `E_dilute` would remove real physics — the interaction
of the defect charge with the host's own charge distribution, which is what binds the
carrier to the lattice — and would make `E_dilute` and `E_total` inconsistent at `n ≠ 0`
while agreeing at `n = 0`, which is the hardest kind of bug to see.

Three further requirements:

- **Use the same `σ` and the same self-energy treatment in both evaluators**, so that the
  difference between them is purely image-plus-background and nothing else leaks in.
- **Re-relax under the isolated evaluator. Do not apply it as a single-point shift.** The
  latent charges absorb electronic screening only; the ionic contribution comes from atoms
  physically moving. A fixed-geometry correction therefore returns the electronically
  screened image energy rather than the statically screened one, and the difference is
  exactly the ionic relaxation. Relaxing under the isolated evaluator lets the lattice
  respond to the correct image-free field and the static screening emerges on its own.
- **`E_dilute(R, 0) = E_total(R, 0) = E_base(R)`** must hold identically, since every
  `n`-dependent channel vanishes at `n = 0`. Assert it (§8.1).

**The amplitude `a` does not cancel — it scales the correction.** Because the environment
and cross terms carry the same boundary condition in both evaluators, they drop out of the
difference exactly, and

```
E_total − E_dilute = E[S_carrier]^periodic − E[S_carrier]^isolated  ∝  a²
```

So the whole finite-size operation is a single term scaling as `a²`. What is true is that
`ε_∞` never has to be *supplied* to perform it. What follows is that the accuracy of every
dilute-limit number is inherited directly from the accuracy of `a`, and quadratically: a 20%
error in `a` is a 44% error in the image energy removed. This is why §3.4 devotes a
subsection to identifying `a` and §8.7 cross-checks it against DFPT. It is an error-budget
row (§11), not a cancellation.

**Side benefit.** This permits relaxing a charged defect *in the isolated limit*, which DFT
cannot do at all. Configuration-coordinate diagrams and migration barriers are computed
directly in that limit rather than corrected toward it. Note that `E_dilute` is never
trained against anything — it is a modification of the trained model — so the geometries it
produces are outside any validated region until §8.7 says otherwise.

**Residual error.** The isolated evaluator removes the electrostatic images, but the ionic
relaxation tail is still truncated at the cell boundary. Use a reasonably large production
cell. Ground truth, where it is needed, comes from relaxing at four or five cell sizes with
the model and fitting the size series of §8.3; this uses no correction scheme at all, costs
only model evaluations, and is the cross-check in test 8.7.

### 7.2 Assembling formation energies

The model predicts on the band-edge-referenced scale (§4). The affine map back to raw
energies must be applied explicitly:

```
E_dilute^raw(R, n) = E_dilute(R, n) + Σ_σ [ n_e^σ · E_CBM^cell − n_h^σ · E_VBM^cell ]

E_f(D^q, E_F) = E_dilute^raw(R*_q, n_q) − E_host + Σ_i n_i μ_i + q·(E_VBM^∞ + E_F)
```

- Omitting the first line shifts every level by a band-gap-scale constant. It is the single
  easiest error to make in this pipeline and the hardest to notice, because relative
  orderings within a charge state are unaffected.
- Note which edges appear where: the **cell** edges undo the referencing used in training;
  the **converged** edge sets the Fermi-level zero. They are different numbers and §5.4
  stores both.
- `R*_q` is the geometry relaxed under the isolated evaluator in charge state `q`, not the
  neutral geometry. The charge-dependent relaxation is the physics being reported.
- `E_host` is the model energy of the pristine host cell at `n = 0`, which is `E_base`
  alone.
- `n_i` are the atoms removed from the host to form the defect and `μ_i` their chemical
  potentials, set by the growth condition being modelled.

Thermodynamic transition levels `ε(q/q′)` follow as the Fermi level at which two charge
states cross:

```
ε(q/q′) = [ E_dilute^raw(R*_q′, n_q′) − E_dilute^raw(R*_q, n_q) ] / (q − q′)  −  E_VBM^∞
```

A change of winning counter configuration along a configuration coordinate is a level
crossing and should appear in the diagram rather than be smoothed over.

### 7.3 Choosing the counter configuration — bounded and gated

At each `q`, the winning counter configuration is found by enumeration. Two constraints on
that enumeration, both necessary:

**Bound the candidate set.** The set `{n : q(n) = q}` is infinite, since `(1,0,1,0)` and
`(0,1,0,1)` pairs can be added indefinitely. Restrict to

```
{ n : q(n) = q,  Σ_c n_c ≤ |q| + 2,  n canonical }
```

which admits the minimal assignment plus at most one extra electron–hole pair — enough for
the closed-shell singlet, open-shell singlet and triplet at `q = 0`, and their analogues
elsewhere. Widen it only with a stated reason.

**Gate the minimisation.** Taking `argmin` over a learned surrogate is not variational. DFT's
minimum is bounded below by the true ground state; a neural interpolant's is not, so the
argmin systematically selects wherever the model's negative-tail error is largest — which
is precisely the rarely-trained counter configurations — and the bias grows with the number
of candidates enumerated. Therefore:

- minimise over an **upper confidence bound** (ensemble mean plus a multiple of the
  ensemble spread), not over the mean;
- **reject candidates whose `P^c` is out of regime** (§8.6) before comparing energies;
- **record the winning configuration and the runner-up gap** with every reported number. A
  small gap is a flag, not a result.

This is also where the *spatial* assignment problem shows up, and enumeration does not
solve it — see §10.1.

---

## 8. Validation

### 8.1 Unit tests — should fail CI, not appear in a metrics table

- `E_total(R, 0) == E_base(R)` to numerical precision, for random `R`. Also
  `E_dilute(R, 0) == E_base(R)`.
- `Σ_i α_i^c == 1` for every channel and every cell in a batch.
- `Σ_i q_i^host == 0` and `Σ_i q_i^pol == 0` to numerical precision.
- **`Σ_i q_i == a·q`**, not `== q`. The net latent charge is the *screened* monopole by
  construction (§3.4). Asserting `Σ_i q_i == q` would force `a = 1`, silently destroy the
  screening, and make every dilute-limit number wrong by a factor of `ε_∞`.
- `∇E` from autograd matches finite differences of `E`, for every branch separately and for
  the total, and for both evaluators.
- Hessian symmetry `‖H − Hᵀ‖` at numerical noise.
- Translation and rotation invariance of `E`; equivariance of forces.
- Permutation invariance of `E` under relabelling of identical atoms — worth testing
  explicitly because the pooling introduces a global reduction.
- **Canonicalisation coverage.** For random `n`, assert `canonical(n) == canonical(swap(n))`
  and that no code path — data loading, batch collation, inference, the §7.3 enumerator —
  can present a non-canonical vector to the network. Time-reversal invariance then holds
  identically rather than approximately; this test checks the plumbing, not the network.


---

## 9. Implementation notes

### 9.1 Module layout


**Use the existing MACELES implementation for everything it already covers** — latent
charge readout, structure factor, periodic Ewald sum, smearing convention, and the autograd
paths for forces and stress. `latent_ewald.py` is a thin layer over it, not a
reimplementation. It adds exactly three things: the structured charge assembly of §3.4
(neutral host channel, neutral polarisation channel, normalised carrier channels, shared
amplitude `a`); the three-way decomposition of §7.1 so that each piece can be given its own
boundary condition; and the isolated evaluator itself, sharing the periodic version's
smearing and self-energy code paths.

`targets.py` maintains two sets of band edges — `cell` and `∞` — and must not mix them
(§2.3, §7.2).

### 9.2 Counter conditioning

Embed `n` once per configuration and broadcast to nodes. A small MLP on the raw counters
plus their derived `(q, M_s)` is adequate; concatenate the embedding to `h_i` before each
readout. Keep the embedding **continuous in `n`** — no lookup tables per discrete state —
so that fractional counters remain available. Canonicalise before embedding, always.

### 9.3 Segment softmax

Training batches contain several cells. The softmax must be **per cell per channel**, not
per batch. Use a scatter/segment softmax keyed on the batch index, with the standard
max-subtraction for stability:

```
ℓ_shift = ℓ − scatter_max(ℓ, batch)[batch]
w       = exp(ℓ_shift)
α       = w / (scatter_add(w, batch)[batch] + eps)
```

Compute in float64 during validation; float32 is adequate for training, but the extensivity
bound must be evaluated in float64 — the gaps §3.2 requires put `α_bulk` near `10⁻⁵` and
below, where float32 accumulation over thousands of atoms is not trustworthy.

Note that `eps` in the denominator biases `α` when the gap is large. Use the smallest value
that keeps the backward pass finite, and confirm `Σ_i α_i^c == 1` to float64 precision
without it.

### 9.4 Long-range evaluators

The periodic sum comes from MACELES; do not rewrite it. Points that still need attention:

- **Pin the prefactors with a unit test before trusting any absolute energy.** Published
  implementations carry internal normalisation constants — one carries a factor of
  1/9.48933 between the raw hidden variable and charges in units of *e*, arising from
  setting `1/2ε₀ = 1` internally. Transcribing prefactors by eye is how sign and scale
  errors enter. Reproduce an analytic Madelung constant for a known lattice first. This
  matters more than usual here, because a prefactor error is absorbed into `a` and shows up
  as an apparently reasonable but wrong `ε_∞`.
- **The evaluator must expose the three pieces separately** — environment–environment,
  environment–carrier cross, and carrier–carrier — because §7.1 gives them different
  boundary conditions. Building only a monolithic `E_LR(q_i)` makes the dilute-limit
  evaluator impossible to assemble correctly and invites the cross-term omission.
- **The isolated evaluator is new work.** A direct real-space sum of smeared charge pairs,
  no images, no background, sharing the smearing and self-energy code paths with the
  periodic version. It must be differentiable, since production relaxations run under it
  (§7.1). Test it against the periodic version by growing the cell: the periodic result must
  converge to the isolated one.
- Both evaluators must be differentiable with respect to positions **and** the cell, so
  that stress is available.
- Scaling as implemented in the reference code is `O(N^3/2)`, with a clear path to
  `O(N log N)` via a particle-mesh evaluation if profiling demands it. Reported memory
  limits are around 10,000 atoms on a 48 GB GPU for the long-range model against roughly
  30,000 short-range, at about half the speed.

### 9.5 Forces and stress

Everything is obtained by differentiating the scalar `E_total`. Use `create_graph=True`
during training so force losses backpropagate. For stress, apply a symmetric virtual strain
to positions and cell and differentiate.

**Charged-cell stress: verify, then use.** The reflexive position is to mask it on the
grounds that the compensating background contaminates the DFT stress. That argument sits
badly with §2.4, which is built entirely on the model and DFT sharing the same background
convention so that raw energies are valid labels; the stress is the strain derivative of
that same raw energy under that same convention, so if the energy is a valid label its
volume derivative ought to be one too. The real risk is implementation-specific — several
codes handle the `G = 0` terms inconsistently between energy and stress for charged cells.

So verify rather than assume:

1. Compute `E_raw(V)` for a charged cell at three volumes.
2. Finite-difference the hydrostatic pressure.
3. Compare against the code's reported stress trace.

If they agree, include the hydrostatic component in training via `L_P` (§4). It is the
cheapest available identifier of `a` and it costs no additional calculations, which matters
most in exactly the hosts where the far-field force signal is weak. If they disagree, mask
the stress in charged cells, drop `L_P`, and lean on the shape-varied subset instead.

Shear components in charged cells remain masked either way; the identifying signal is in
the trace. Variable-cell charged relaxation is not supported without the above validation.

### 9.6 Numerical hygiene

- Clamp logits to a sane range to avoid `exp` overflow before the max-subtraction.
- Initialise `MLP_u`, `MLP_p` and `Q_host` near zero so the model starts close to the base
  potential. Note that `softplus(0) ≈ 0.69` puts the initial `a` at an implied
  `ε_∞ ≈ 2.1`; initialise `MLP_a`'s bias at `softplus⁻¹(1/√ε_∞^DFPT)` instead, so training
  starts at a physically sensible gauge rather than having to travel there.
- Warm-start `E_base` from a bulk foundation model if available; keep it frozen for the
  first few epochs while the correction heads settle.
- Assert counter validity at load time: all counters non-negative integers, vector in
  canonical form, and `M_s` consistent with the recorded DFT multiplicity.

### 9.7 Suggested build order

1. Counters, canonicalisation, target construction (both edge sets), data loading,
   validity assertions.
2. Base branch alone; reproduce a standard bulk MLIP result as a sanity check.
3. Pooling and `ΔE_SR`; unit tests 8.1; train on paired data only.
4. Long-range branch: wire in MACELES, verify prefactors against an analytic Madelung
   constant, expose the three-piece decomposition, then add the structured charge assembly
   of §3.4 including `q_i^pol`. Re-train with `L_Δ` and check test 8.3.
5. **Run the §8.3 size-series discriminator on this undertrained model.** It is cheap and
   it separates `a`, multipole and pooling-drift problems before the data campaign scales.
   Fix what it finds here rather than after.
6. `L_tot` with detachment; check that base-branch predictions on reference-state holdouts
   have not moved.
7. Isolated evaluator, all three pieces; confirm `E_dilute(R,0) == E_base(R)`, check
   agreement with the periodic evaluator under cell growth, confirm `1/a²` against DFPT,
   then run tests 8.7.
8. Bounded counter enumeration with UQ gating (§7.3); requires an ensemble, so plan for it.
9. Multi-fidelity residual.
10. Full validation suite.

Do not proceed past step 3 until `E_total(R, 0) == E_base(R)` holds exactly and the logit
gap is stable during training. Do not proceed past step 7 until `1/a²` and DFPT `ε_∞` agree
or the disagreement is understood.
