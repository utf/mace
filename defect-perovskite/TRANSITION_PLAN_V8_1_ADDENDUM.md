# Addendum to transition plan v8: corrected functional, boundary reference, and implementation contracts

(Received 2026-09-06. Normative v8.1 correction; where it conflicts with
`TRANSITION_PLAN_V8_SPEC.md`, this addendum takes precedence. Both files are the specification.)

## 1. Status and purpose

This addendum corrects ambiguities and gaps in the v8 charge-defect plan before the affected
stages are implemented. Where it conflicts with v8, this addendum takes precedence. The stage
order and the central strategy remain unchanged:

- retain a neutral-only, frozen, pretrained base potential;
- represent charge through exact electron counts and a sparse local Hamiltonian;
- introduce electrostatic response through one conservative functional;
- keep explicit polarisation conditional and last;
- use no defect labels and require no additional electronic-structure calculations.

A wholly new programme is not required, but v8 should not be implemented verbatim. This
addendum is the normative v8.1 correction.

The principal mathematical changes are that the frontier density must become an explicit
functional of \(P\), and the total energy must use a fixed-PBC reference anchor with a
complete image interaction on the supported compact defect-excess-density domain. The carrier-count,
residual-density, screening, cache, stress, stage-routing, supported-domain, spectral-gauge,
and energy-calibration findings are equally normative correctness contracts.

The corrections do **not** require electrostatics to be inserted into the neutral base. They
mostly make the head's variational definition, boundary-condition reference, and accounting
rules explicit. The present scope remains the `count_fill` charge model only. Until a
self-interaction-controlled boundary-common frontier--frontier functional is supplied, its
nontrivial charge-head and boundary-conversion outputs are restricted to states containing at
most one absolute frontier carrier, as defined in Section 3.4.

The most important change is the energy reference. The periodic neutral base is retained as
the training-boundary reference, while a separate reference-boundary completion is added when
an isolated result is requested. This supplies the static--static boundary term that cannot be
obtained from a same-boundary charge-state subtraction, without adding it twice to the periodic
neutral energy.

## 2. What the static--static issue does and does not imply

### 2.1 The relevant quantity is a defect excess energy

An absolute supercell total energy is extensive and therefore does not approach a constant as
the number of atoms grows. The convergent quantity is a consistently referenced defect excess
energy, or a formation energy after the corresponding atomic and electronic reservoir terms
are added. Schematically,

\[
\Delta E_{\mathrm{def}}^B(L,q)
=E_B(D^q,L)-E_B(\mathrm{bulk},L)-\sum_a n_a\mu_a,
\]

with the electron-reservoir convention added when charge states are compared. The same cell
shape, bulk energy convention, pseudopotential convention, electrostatic boundary, and
potential-zero convention must be used throughout.

For a compact object with nonzero net charge, a periodic charged-defect excess energy normally
contains a leading image error of order \(1/L\), under the declared background, cell-shape,
and dielectric convention. Neutral objects and extended carriers need not have that leading
term. The periodic excess may still be converged by increasing \(L\). An explicit
periodic-to-isolated functional is intended to remove the modeled part of the finite-size
error sooner; it does not change the existence of the large-cell limit.

### 2.2 Why static--static cancels in one difference but not another

At a fixed geometry and composition, decompose the modeled defect-induced density into a
charge-state-independent static part \(\rho_S\) and a frontier part \(\rho_F(S)\). A quadratic
electrostatic contribution expands as

\[
\frac12 B[\rho_S+\rho_F,\rho_S+\rho_F]
=\frac12 B[\rho_S,\rho_S]
+B[\rho_S,\rho_F]
+\frac12 B[\rho_F,\rho_F].
\]

The first term cancels in \(E(D^q;\mathbf R)-E(D^{\rm ref};\mathbf R)\), because \(\rho_S\)
is the same in the two electronic states. It does not cancel in a direct
defect-minus-pristine comparison, because the pristine and defective compositions have
different static densities.

This does **not** make the v8 thermodynamic cycle unusable. By model definition, the frozen
base owns the complete periodic reference-state defect-versus-bulk energy. No unique split of
that learned energy into static--static or other physical components is claimed. A periodic
charged-defect result can therefore be written as

\[
\underbrace{E_{\rm base}^{\rm PBC}(D^{\rm ref})
-E_{\rm base}^{\rm PBC}(\mathrm{bulk})}_{\text{neutral defect excess}}
+
\underbrace{E^{\rm PBC}(D^q)-E^{\rm PBC}(D^{\rm ref})}_{\text{charge-state increment}}.
\]

The first bracket is a complete reference energy, not an identified static--static term. The
latent static self-energy cancels from the second bracket, and adding its absolute value
directly to the periodic head would double-count the reference model.

The limitation in v8 is narrower but real: using the same-boundary reference subtraction for
both PBC and isolated outputs leaves the neutral reference fixed at its periodic base value.
It therefore corrects the charge-state increment, but it does not explicitly transform the
neutral defect reference from PBC to infinity. The revised reference convention below fixes
that omission.

### 2.3 Consequence for convergence

There are three distinct claims:

1. **Periodic large-cell convergence.** A size-extensive local base plus the charge head can
   converge the defect excess energy by systematic enlargement. Static--static cancellation
   does not prevent this. Remaining errors include ordinary neutral multipole, elastic,
   band-filling, and base-transfer errors.
2. **Isolated charge-state increment.** On the domain satisfying
   \(\operatorname{IsoOK}\), a same-boundary
   subtraction using the amended complete image functional can correct the modeled boundary error in
   \(E(D^q)-E(D^{\rm ref})\); the original incomplete pair accounting did not establish this
   generally.
3. **Finite-cell isolated total defect excess.** This additionally requires the neutral
   reference's PBC-to-isolated boundary completion. It must not be claimed until the revised
   fixed-training-boundary anchor and its validation tests pass.

Periodic large-cell convergence remains subject to the ordinary PBC predictive and
derivative gates. The two isolated claims additionally require the full
\(\operatorname{IsoOK}\) predicate of Section 4.2 for every required requested/reference
state and matched defect/bulk wrapper. Outside that domain only the validated
training-boundary model output is exposed; no complete PBC-to-isolated conversion is claimed.
It does not assert that two labelled cell sizes
establish the exact physical asymptote, nor that a local neutral base contains every neutral
long-range polar contribution.

The static--static addition is a head-level boundary contraction with no \(P\) derivative;
it does not add an electronic self-consistency iteration. A production isolated evaluation of
a general state requires the isolated requested-state solution and the fixed-PBC reference
solution already present in the anchor. For \(S=S_{\rm ref}\), those two solves are simply
the isolated and fixed-PBC reference solutions. For \(S\ne S_{\rm ref}\), an additional
isolated reference-state solution is needed only when the two conceptual brackets are
requested separately. No base reevaluation or latent-Ewald base is introduced.

## 3. Corrected electronic and charge inventory

### 3.1 Unambiguous Hamiltonian names

The symbol \(H_0\) is retired because it referred to several different objects. Use:

- \(H_{\rm class}\): frozen, constructor-only Hamiltonian used to establish composition
  integers;
- \(H_{\rm fix}(\mathbf R)\): runtime Hamiltonian independent of \(P\), electronic state,
  and boundary choice, containing the local Slater--Koster and accepted onsite and edge terms;
- \(H_{B,S}[P,\mathbf u]\): stationary Hamiltonian for boundary \(B\) and electronic state
  \(S\).

Neither `state_id`, `occupation_policy`, nor formal charge is a learned input to
\(H_{\rm fix}\). State dependence enters through exact occupation constraints and the
resulting density.

The runtime one-electron energy zero is fixed explicitly. Let
\(\widetilde H_{{\rm fix},\sigma}^{\rm pris}(\theta)\) be the raw runtime Hamiltonian on the
registered canonical pristine reference cell, evaluated with the current head parameters
\(\theta\), and let \(P_{V,\sigma}^{\rm pris}(\theta)\) be its fixed-rank occupied-valence
projector. Define one scalar, common to both spin channels,

\[
\mu_{\rm g}(\theta)=
\frac{\sum_\sigma
\operatorname{Tr}\!\left[
P_{V,\sigma}^{\rm pris}(\theta)
\widetilde H_{{\rm fix},\sigma}^{\rm pris}(\theta)
\right]}
{\sum_\sigma M_{V,\sigma}^{\rm pristine}},
\]

and use

\[
H_{{\rm fix},\sigma}(\mathbf R;\theta)
=\widetilde H_{{\rm fix},\sigma}(\mathbf R;\theta)
-\mu_{\rm g}(\theta)I.
\]

The displayed form is used for every runtime geometry, composition, size, state, and
boundary. All Hamiltonians, density matrices, projectors, traces, commutators, and matrix
functional derivatives in Sections 3--7 are written in one registered orthonormalised
orbital representation. An implementation that retains a generalised eigenproblem
\(\widetilde Hc=\varepsilon Sc\) internally must use the exactly equivalent transformation

\[
\widetilde H\mapsto\widetilde H-\mu_{\rm g}S,
\]

and evaluate the gauge trace after exact orthonormalisation or through its registered
overlap-metric equivalent. Subtracting \(\mu_{\rm g}I\) in a nonorthogonal representation is
prohibited. The overlap definition, orthonormalisation map, conditioning tolerance, and
basis version are serialised.

The same \(\mu_{\rm g}\) is subtracted from all aligned spectral edges. The pristine
projector must retain its registered separating gap; loss of that gap invalidates the gauge
record and the checkpoint.
The scalar is differentiated with respect to \(\theta\), but it is evaluated on the frozen
pristine reference and has no runtime coordinate or strain dependence. At inference it is
computed once per checkpoint and cached.

This is a potential-zero convention, not a learned host or defect correction. It is never
fitted per frame, defect, charge, composition class, or cell size. Under a raw common-mode
shift \(\widetilde H\mapsto\widetilde H+aI\) in the registered orthonormal representation
(or \(\widetilde H\mapsto\widetilde H+aS\) in the equivalent nonorthogonal
implementation), \(\mu_{\rm g}\mapsto\mu_{\rm g}+a\), so the gauge-fixed Hamiltonian, densities, forces,
stresses, and charge energies are unchanged. This removes the exact flat direction in which
a seed-dependent common spectral shift is compensated by an energy constant. It does not
constrain band gaps, splittings, deformation potentials, or other nonuniform spectral
changes. \(H_{\rm class}\) remains the separately frozen constructor of Section 3.5; its
rank decisions are invariant to a scalar shift and do not inherit the runtime gauge record.

### 3.2 Corrected Tier-1 rank-certified gap verifier

The v8 Tier-1 acceptance test is replaced in full. Testing whether an eigenvalue lies close
to the aligned VBM tests the density of the folded valence manifold, not whether that
manifold is separable from the frontier space. It therefore becomes more likely to fail as
the supercell grows even when the physical valence--frontier gap is unchanged.

Tier 1 is a cheap verifier of a rank established independently of the current spectrum; it
does not infer the rank by choosing the largest raw gap. This prevents a later
frontier--frontier gap from being mistaken for the valence--frontier boundary.

Define a homologous defect family without a manual defect label by the composition-difference
vector relative to the registered tiled pristine cell, together with the constructor's
covariant site-correspondence topology. A Tier-1 candidate rank for spin \(\sigma\) may come
only from:

1. the registered pristine host rank for a pristine class; or
2. an accepted constructor record for a homologous defect class at another size, transported
   by the known pristine-rank increment,

\[
M_{{\rm VB},\sigma}^{\rm pred}(L_2)
=M_{{\rm VB},\sigma}^{\rm acc}(L_1)
+\left[
M_{{\rm VB},\sigma}^{\rm pristine}(L_2)
-M_{{\rm VB},\sigma}^{\rm pristine}(L_1)
\right].
\]

For a pristine neutral reference, the registered rank is
\(M_{{\rm VB},\sigma}^{\rm pristine}=N_\sigma(S_{\rm ref})\), with
\(Q_{\rm core}=m_F=0\), and its gap at that rank must pass the same numerical guard. This is
the extensive rank used in the transport equation.

If no fingerprint-compatible accepted anchor exists, or if two eligible anchors predict
different ranks, Tier 1 has no authority to decide the class and routes directly to Tier 2.
An older VBM-threshold result becomes an accepted anchor only after agreement with an
accepted Tier-2 result or another independently certified rank; it is not grandfathered by
having run quickly.

On \(H_{\rm class}\) at the frozen class-reference geometry, order the eigenvalues
independently for each spin,

\[
\varepsilon_{1,\sigma}\le\cdots\le\varepsilon_{K,\sigma},
\]

and test the consecutive gap at the predicted rank,

\[
k_\sigma=M_{{\rm VB},\sigma}^{\rm pred},\qquad
g_\sigma=\varepsilon_{k_\sigma+1,\sigma}-\varepsilon_{k_\sigma,\sigma},
\qquad
E_{{\rm thr},\sigma}
=\frac{\varepsilon_{k_\sigma+1,\sigma}+\varepsilon_{k_\sigma,\sigma}}{2}.
\]

Let \(E_{v,\sigma}^{\rm al}\) be the frozen aligned pristine valence edge,
\(E_{g,\sigma}^{\rm host}\) the registered spin-resolved pristine gap,
\(s_{\rm smear}\) the registered smearing width, and \(u_{\rm al}\) a conservative
registered bound on edge-alignment error. A sample IQR is a diagnostic and is not, by
itself, an error bound.

Construct \(u_{\rm al}\) from existing constructor data only. The calibration set contains
registered pristine cells, whose valence ranks are exact by electron count, and
fingerprint-compatible defect classes whose spin-resolved ranks were accepted independently
by Tier 2. For every calibration record \(j\), spin \(\sigma\), and registered precision and
small-geometry perturbation \(\delta\), form

\[
e_{j,\sigma,\delta}
=\varepsilon_{M_{{\rm VB},j,\sigma}^{\rm acc},\,j,\sigma}(\delta)
-E_{v,j,\sigma}^{\rm al}(\delta),
\qquad
u_{\rm al}
=\max_{j,\sigma,\delta}|e_{j,\sigma,\delta}|+\tau_{\rm al}^{\rm num},
\]

where \(\tau_{\rm al}^{\rm num}>0\) is the registered eigensolver and arithmetic margin in
eV. This is a deterministic engineering envelope on the fingerprinted constructor domain, not a
probabilistic confidence bound. Its record stores every source hash, accepted rank, residual,
perturbation, numerical regime, margin, and applicability predicate. No target class may
certify its own rank merely by first enlarging this envelope: a target contributes only after
an independent Tier-2 acceptance and then only to later verifier records. If no applicable
independently ranked calibration set exists, if its provenance is incomplete, or if the
target lies outside its registered homology/numerical domain, Tier 1 routes directly to
Tier 2.

Use the symmetric-margin midpoint window

\[
\mathcal W_\sigma=
\left[
E_{v,\sigma}^{\rm al}-\Delta_{\rm search},\,
E_{v,\sigma}^{\rm al}+\frac12E_{g,\sigma}^{\rm host}
+\Delta_{\rm search}
\right],
\qquad
\Delta_{\rm search}\ge u_{\rm al}+2s_{\rm smear}.
\]

For the current host, \(\Delta_{\rm search}=0.30\ {\rm eV}\) is a retrospective engineering
choice informed by the already inspected alignment and class behaviour. It may be used as a
regression setting, but not described as pre-registered evidence. It is valid only if the
stored diagnostics establish the inequality above. With the current
\(s_{\rm smear}=0.05\ {\rm eV}\), this requires \(u_{\rm al}\le0.20\ {\rm eV}\); otherwise
the class routes to Tier 2. For a future host, the alignment bound, window, and gap floors are
frozen before class outcomes are inspected.

Tier 1 accepts only if, for every spin channel:

1. \(1\le k_\sigma<K\);
2. \(E_{{\rm thr},\sigma}\in\mathcal W_\sigma\) for every edge shift in the registered
   interval \([-u_{\rm al},u_{\rm al}]\);
3. \(g_\sigma\ge\max(4s_{\rm smear},2u_{\rm al},g_{\rm num})\), where \(g_{\rm num}\) is
   the registered numerical gap floor;
4. the same rank remains separated by the required gap under the registered
   search-window, precision, and small-geometry perturbation tests; and
5. \(E_{g,\sigma}^{\rm host}\ge4s_{\rm smear}\).

The accepted value is

\[
M_{{\rm VB},\sigma}^{\rm class}=k_\sigma.
\]

The selected threshold, gap, alignment interval, \(u_{\rm al}\) value and provenance hash,
source-anchor fingerprint, pristine-rank increment, and every stability result are stored in
the class record. Proximity of ordinary
folded valence states to \(E_v^{\rm al}\) is not an ambiguity criterion. Failure of any
condition routes to Tier 2; Tier 1 never searches for a different, better-looking gap.

The raw valence rank is extensive and is **not** required to be identical between
supercells. For two cells of the same homologous defect family,

\[
M_{{\rm VB},\sigma}^{\rm class}(L_2)
-M_{{\rm VB},\sigma}^{\rm class}(L_1)
=M_{{\rm VB},\sigma}^{\rm pristine}(L_2)
-M_{{\rm VB},\sigma}^{\rm pristine}(L_1).
\]

Equivalently, the defect rank offset and
\(d_\sigma=N_\sigma(S_{\rm ref})-M_{{\rm VB},\sigma}^{\rm class}\) are invariant. The
per-spin carrier counts and \(Q_{\rm core}\) must therefore agree across the 79- and
159-atom defect classes, but their raw \(M_{{\rm VB},\sigma}\) values must differ by the
80-atom pristine rank. Threshold offsets and gap sizes are diagnostics and need not be
numerically identical across sizes.

The existing 159-atom Tier-2 result is retained as the family anchor only if both
continuation paths and both step schedules agreed, no closure occurred, and the endpoint
contiguity gate passed. A value \(Q_{\rm core}=+1\) alone is insufficient because compensating
per-spin rank errors could give the same total. If the record passes, its spin-resolved rank
is recorded as a real-system validation of the continuation machinery, then transported
backward to 79 atoms and forward to subsequent homologous sizes. The repaired
Tier 1 must accept both the 79- and 159-atom classes at those exact ranks **if** their tested
gaps satisfy every frozen numerical and stability guard, and the old 79-atom result must
agree. The implementation must assert that Tier 2 is not invoked after all Tier-1 conditions
pass. If the existing Tier-2 record fails any acceptance condition, it cannot seed the
family and the class remains on the ordinary Tier-2 route. If a certified-rank gap itself
fails, Tier-2 routing is correct and may not be suppressed merely to meet a timing target.

This correction changes Tier-1 selection, routing, and validation only. It introduces no
Tier-2 runtime optimisation.

### 3.3 Counts that remain valid across electron/hole crossings

For each spin channel, let \(M_{{\rm VB},\sigma}^{\rm class}\) be the fixed rank of the
continued valence subspace and \(N_\sigma(S)\) the exact electron count. Define

\[
d_\sigma(S)=N_\sigma(S)-M_{{\rm VB},\sigma}^{\rm class},
\]

\[
n_{e,\sigma}(S)=\max[d_\sigma(S),0],\qquad
n_{h,\sigma}(S)=\max[-d_\sigma(S),0],
\]

\[
q_F(S)=\sum_\sigma\bigl(n_{h,\sigma}-n_{e,\sigma}\bigr)
=-\sum_\sigma d_\sigma(S).
\]

The production reference is required to be neutral,
\(Q_{\rm formal}(S_{\rm ref})=0\), because the frozen base is calibrated to a neutral
reference state. Define the composition integer once as

\[
Q_{\rm core}=Q_{\rm formal}(S_{\rm ref})-q_F(S_{\rm ref}),
\]

and require for every state

\[
Q_{\rm formal}(S)=Q_{\rm core}+q_F(S).
\]

The first definition is also algebraically well defined for a nonzero formal reference and is
retained as a synthetic identity test. Such a state may not enter the production energy path:
supporting it physically would require a base calibrated to that charged reference. These
definitions replace additive updates of separate electron and hole counts, which could
otherwise become negative when a charge sequence crosses the valence rank.

Define the absolute frontier-carrier multiplicity

\[
m_F(S)=\sum_\sigma\left[n_{e,\sigma}(S)+n_{h,\sigma}(S)\right].
\]

This is not the magnitude of a charge difference; it includes every frontier carrier already
present in the state.

### 3.4 Carrier-multiplicity guard

The present functional contains a boundary-common static--frontier interaction and all
static--static, static--frontier, and frontier--frontier interactions of the declared
image-active density; this becomes the complete modeled boundary-image interaction on the
exact compact plateau. It does not contain a boundary-common, self-interaction-controlled
frontier--frontier functional.
Consequently the static/frontier partition, and \(Q_{\rm core}\) in particular, is a
constructor-fixed bookkeeping convention rather than a harmless Hamiltonian gauge. Only the
total image density is operative in \(\Phi_{\rm img}\), and even an image repartition is
gauge-like only if it leaves that complete signed density unchanged.

Until the missing common functional is implemented and validated, every state and every
reference state entering a nonzero charge-head increment or boundary conversion must satisfy

\[
m_F(S)\le1.
\]

The check uses the absolute constructor counts, not
\(\lvert Q_{\rm formal}(S)-Q_{\rm formal}(S_{\rm ref})\rvert\), and is performed before
energy, force, or stress evaluation in Stages 4--6. Failure is an unsupported-state error
under both PBC and isolated boundaries, not a warning. The sole algebraic exception is the
neutral reference at the training boundary returned directly as the frozen base: its head
cancels identically and need not be evaluated. A class intended for a nontrivial charge-head
path is rejected as a whole if any production state or required reference violates the
guard.

Reserve a ledger slot \(\Phi_{\rm FF}^{\rm common}\), fixed to zero under this guard. A later
multi-carrier extension must define one occupation- or density-matrix-resolved variational
functional with explicit conventions for distinct-carrier direct interaction,
self-interaction, exchange, dielectric screening, degeneracies, unitary invariance,
derivatives, and band-energy double counting. Periodic self-image remains exclusively in
\(\Phi_{\rm img}\). No future occupation-policy implementation may inherit \(\rho_S\),
\(\Phi_{\rm SF}\), or any residual-object Hamiltonian term automatically; it must explicitly
opt into a complete, validated charge decomposition. These are interface reservations, not
a specification of another state model.

### 3.5 Constructor versioning

\(H_{\rm class}\) is a dedicated frozen constructor, not whichever runtime Hamiltonian is
current at a later stage. Constructor caching is split so a cheap Tier-1 verifier change
does not invalidate an expensive, still-compatible Tier-2 anchor.

The common constructor record is partitioned into fields that must be identical across a
homologous size family and fields that are intentionally target-specific. The invariant
partition contains the rank-core schema version (explicitly excluding the Tier-1
router/verifier schema), constructor checkpoint and parameter hash, canonical host/pristine
reference and basis convention, correspondence and homology algorithms,
composition-difference/homology signature, and smearing convention. The target partition
contains the exact composition hash, pristine tiling map and rank, and class-reference
geometry and cell hashes.

The Tier-2 anchor key contains both partitions, its own Tier-2 schema version, the exact
target class-reference geometry/cell hash, continuation paths, step sizes, sink, closure and
endpoint-contiguity settings, and the eigensolver, backend, dtype, arithmetic precision, and
all numerical tolerances. Its record stores both paths and schedules, the endpoint
diagnostics, numerical regime, and accepted spin-resolved rank.

The Tier-1 verifier key contains its own schema version, both common-record partitions, the
source Tier-2 or homologous-anchor hash, pristine-rank increment, alignment-bound value and
provenance hash, midpoint window, perturbation settings, numerical-gap floor, and the
eigensolver, backend, dtype, and arithmetic-precision regime. Its record stores the predicted
and accepted \(M_{{\rm VB},\sigma}\), \(d_\sigma\), \(Q_{\rm core}\), \(m_F\), thresholds,
diagnostic eigenvalues, and every gate result.

Cross-size compatibility is field-wise: every invariant field must match exactly, while the
target-specific fields must satisfy the registered pristine-tiling relation,
composition-difference identity, covariant site-topology relation, and pristine-rank
increment. Exact target geometry, cell, composition, and tiling hashes are expected to
differ and are never compared for equality across sizes. A missing field or a failed
covariant relation makes the records incompatible.

A Tier-1-only setting change invalidates only the verifier record. A common-fingerprint or
Tier-2 setting change invalidates the affected Tier-2 anchor. Routing always runs the cheap
verifier first and stops when it accepts. Tier 2 runs only when Tier 1 lacks a compatible
anchor, a Tier-1 gate fails, or an explicitly registered cross-tier audit is requested;
whenever Tier 2 runs, both continuation paths and both step schedules are mandatory. A
runtime onsite, edge, or electrostatic refit does not silently redefine the charge inventory.
The existing 159-atom result may be migrated into the Tier-2 anchor cache without rerunning
only if its stored raw log reconstructs every invariant and target-specific key field,
including the exact geometry/cell and numerical regime, and passes every Tier-2 gate; the
migration stores the original provenance and a new content hash.

## 4. Corrected static-density contract

### 4.1 Static density and covariant registration

Let \(N_{\rm at}\) be the number of present atomic sites in the current cell. Every density kernel
\(g(\mathbf r-\mathbf R;r_g)\) used below is nonnegative, periodicised under PBC when
required, and normalised under the active boundary convention:

\[
\int g(\mathbf r-\mathbf R;r_g)\,d\mathbf r=1,
\qquad r_g>0.
\]

The same registered kernel convention is used in the present and pristine sums. Define

\[
\rho_Z^{\rm present}=\sum_{i\in{\rm present}}Z_i
g(\mathbf r-\mathbf R_i;r_Z),
\qquad
\rho_Z^{\rm pristine}=\sum_{j\in{\rm pristine}}Z_j^0
g(\mathbf r-\mathbf R_j^0;r_Z).
\]

The raw density and its integral are

\[
\rho_S^{\rm raw}
=\rho_Z^{\rm present}-\rho_Z^{\rm pristine},
\qquad
q_{\rm raw}
=\sum_{i\in\rm present}Z_i-
 \sum_{j\in\rm pristine}Z_j^0.
\]

The second equality replaces any shorthand that writes \(q_{\rm raw}=\sum_i Z_i\) without
also declaring the pristine integral to be zero.

The exact static monopole remains

\[
\rho_S
=\rho_S^{\rm raw}
+(Q_{\rm core}-q_{\rm raw})g_{\rm res},
\qquad
\int\rho_S=Q_{\rm core}.
\]

The residual distribution must be smooth and defined when every learned charge deviation
vanishes. For each present site,
\(\delta Z_i=Z_i-Z_i^{0,{\rm reg}}\), where \(Z_i^{0,{\rm reg}}\) is the
constructor-registered pristine species/site baseline and is zero for a registered addition.
One admissible form is

\[
a_i=\sqrt{\delta Z_i^2+\epsilon_Z^2}-\epsilon_Z
+\lambda_d d_i,
\qquad
\omega_i=
\frac{a_i+\epsilon_\omega/N_{\rm at}^2}
{\sum_j a_j+\epsilon_\omega/N_{\rm at}},
\]

\[
g_{\rm res}(\mathbf r)=\sum_i\omega_i
g(\mathbf r-\mathbf R_i;r_{\rm res}),
\]

where \(d_i\ge0\) is a smooth, bounded departure from the pristine species environment and
\(\lambda_d\ge0\). Require
\(\epsilon_Z>0\), \(\epsilon_\omega>0\), and \(r_{\rm res}>0\). Kernel normalisation and the
displayed denominator then give \(\omega_i\ge0\), \(\sum_i\omega_i=1\), and
\(\int g_{\rm res}=1\), establishing \(\int\rho_S=Q_{\rm core}\). When a local departure
signal is present, the total uniform fallback weight vanishes as \(O(1/N_{\rm at})\), rather than
leaving a finite delocalised fraction as the cell grows. If
\(|Q_{\rm core}-q_{\rm raw}|\) is nonzero but the local departure signal is below a registered
support threshold, every nontrivial electrostatic-head result is marked unsupported rather
than silently placing a physical monopole through a numerical fallback. On a pristine class
the residual prefactor is exactly zero, so the limiting choice has no energy effect.

The pristine density uses the same origin, wrapping, cell, and affine strain as the current
frame. The composition constructor supplies a deterministic, label-free fractional-lattice
registration from the pristine reference to the defective frame. The discrete atom/site
correspondence is established once for the composition class and is not recomputed by a
nearest-neighbour rule on each thermal frame. Under a rigid translation or rotation, both
densities co-transform identically; under homogeneous strain, the registered pristine
fractional coordinates remain fixed and co-deform with the cell. Any continuous
geometry-dependent alignment is fully differentiated; a change of discrete correspondence
is an unsupported topology event. A registration need be unique only up to an exact crystal
symmetry, lattice translation, or atom permutation that gives identical densities,
energies, and covariant derivatives. Such representatives form one accepted equivalence
class. If inequivalent admissible registrations change any downstream density or observable,
the static density and every nontrivial electrostatic-head output are unsupported rather
than tied to an arbitrary origin. Gaussian widths are fixed in Cartesian units unless a
different strain law is explicitly selected; the chosen law and its derivative are part of
the checkpoint.

A thermally displaced supercell relative to an ideal pristine lattice need not define a
compact density: displacement dipoles can occur throughout the cell. No \(L^{-3}\) law is
assumed for that object. Ideal/localised tilings and thermal-background contributions are
reported separately. Because the boundary wrapper below evaluates the defective and pristine
functionals independently, every complete density entering an individual
\(\mathcal A_B^*\) must independently pass the localisation, lift, clearance, and tail
contracts. A compact matched defect-minus-bulk/reference density is a useful additional
excess-localisation diagnostic, but it cannot rescue an extended constituent: for a
quadratic functional, subtracting two functional values is not the functional of the
subtracted density because the latter has a different cross term. A joint excess functional
with that cross term is not introduced here. When any independently evaluated constituent
fails, the PBC result is only a well-defined label-boundary output subject to the ordinary
predictive gates; only ensemble or explicit large-cell convergence is reported.

### 4.2 Canonical isolated-space lift and full support predicate

A periodic model density lives on the cell torus
\(\mathbb T_{\mathbf h}^3\), whereas \(G_\infty\) acts on a density in
\(\mathbb R^3\). A wrapped periodic density may therefore never be passed directly to an
isolated kernel. Every requested/reference evaluation bundle uses one registered,
component-preserving lift \(\mathscr U_{\mathbf h}\) that unwraps \(\rho_S\), every signed
frontier channel, and their sum with the same branch choices. This preserves the algebraic
S--S, S--F, and F--F decomposition.

The lift is a linear map on the registered density representation and is fixed with respect
to \(P\) throughout a certified SCF neighbourhood. Its branch is selected from an
occupation-independent, nonnegative, cancellation-free support envelope. The branch anchor
is built once at the frozen class-reference geometry from constructor topology, rather than
from the current thermal density or a spectral window, either of which could become extensive
as the cell grows. On the registered union \(\mathcal U_{\rm class}\) of present and pristine
sites, set the fixed constructor species baseline to zero on the missing side of an addition
or removal and define

\[
\rho_{\rm topo}^{\rm class}(\mathbf r)
=\sum_{k\in\mathcal U_{\rm class}}
\left(Z_k^{{\rm present},0}-Z_k^{{\rm pristine},0}\right)
g(\mathbf r-\mathbf X_k^{\rm class};r_{\rm lift}),
\]

\[
\zeta_{\rm lift}^{\rm class}(\mathbf r)=
\sqrt{\left(\rho_{\rm topo}^{\rm class}(\mathbf r)\right)^2+\epsilon_\eta^2}
-\epsilon_\eta
+\lambda_{\rm lift}\sum_{k\in\mathcal U_{\rm class}} a_k^{\rm class}
g(\mathbf r-\mathbf X_k^{\rm class};r_{\rm lift}),
\qquad \epsilon_\eta>0,\quad \lambda_{\rm lift}>0,\quad r_{\rm lift}>0.
\]

Here \(a_k^{\rm class}\) is the Section-4.1 nonnegative departure signal evaluated on the
class reference and extended to missing/added sites by the same constructor convention.
The support is obtained from the existing covariant site correspondence and is not a manual
defect label.
The field is mapped covariantly to the current origin, orientation, and affinely deformed
cell, but it contains no current thermal displacement, \(P\), occupation, frontier density,
localisation switch, or spectral projector. It must pass registered fixed-absolute
site-count, spatial-spread, clearance, and tiling-support gates. The actual current static
and \(P\)-dependent channel densities do not select the branch; they are checked against its
independent clearance and tail contract below. If the mapped
\(\zeta_{\rm lift}^{\rm class}\) is identically zero while any modeled density component is
nonzero, the isolated lift is unsupported. If every component is identically zero, the lift
and boundary correction are identically zero. Otherwise, on fractional coordinates
\(\mathbf s=\mathbf h^{-1}\mathbf r\bmod1\), form for each direction \(\alpha\)

\[
z_\alpha=
\frac{\int_{\mathbb T_{\mathbf h}^3}
\zeta_{\rm lift}^{\rm class}(\mathbf r)e^{2\pi i s_\alpha}\,d\mathbf r}
{\int_{\mathbb T_{\mathbf h}^3}\zeta_{\rm lift}^{\rm class}(\mathbf r)\,d\mathbf r}.
\]

Require \(|z_\alpha|\ge z_{\min}>0\), choose the support centre from
\(\arg z_\alpha/(2\pi)\), place the cut half a cell away, and unwrap every density component
through that same cut. At fixed geometry this defines one integer image assignment for every
registered density primitive, orbital-product component, or grid partition. That assignment
is held locally fixed for all admissible \(P\), so
\(\delta\mathscr U_{\mathbf h}/\delta P=0\) while
\(\mathscr U_{\mathbf h}\,\delta\rho/\delta P\) is retained. Branch stability, modulo a
simultaneous common lattice translation, is certified over every SCF iterate and the
registered admissible \(P\)-matrix finite-difference neighbourhood, as well as geometry,
cell, and training-parameter perturbations. All smooth geometry and cell dependence of the
lifted density primitives is differentiated. A non-common integer image reassignment, or any
reassignment that changes an observable, is an unsupported topology event. Periodic
rewrapping may produce a common-lattice-translation handoff only when isolated energies are
invariant and derivatives agree across it. Exact
symmetry-related lifts are accepted as one equivalence class only when, after the registered
common symmetry transformation, they give identical lifted component densities, energies,
and covariant derivatives. A vanishing circular
moment, an unstable branch, or inequivalent lifts makes the isolated output unsupported.

Because Gaussian and orbital densities need not have literal compact support, the lift also
has a registered boundary-clearance and tail contract. Let
\(\mathcal C_\alpha(d_{\rm clear})\) be the fractional/Cartesian buffer of physical width
\(d_{\rm clear}>0\) around cut \(\alpha\). For every individual requested or reference
functional, the implementation stores both the occupation-independent envelope mass and the
absolute mass of every actual static/frontier component in every cut buffer, together with
certified upper bounds
\(\varepsilon_\rho^{\rm tail}\),
\(\varepsilon_E^{\rm tail}\),
\(\varepsilon_F^{\rm tail}\), and
\(\varepsilon_\sigma^{\rm tail}\) for omitted density, isolated energy, force, and stress
effects, including the static density, frontier orbital/basis-function map, density kernels,
grid, and quadrature tails. Each must lie
below its frozen absolute tolerance under the finite-difference perturbation range. The
clearance and tolerances are physical absolute values; they may not loosen with
\(N_{\rm at}\) or cell length. The lift algorithm, \(\epsilon_\eta\), \(z_{\min}\),
\(d_{\rm clear}\), \(\lambda_{\rm lift}\), \(r_{\rm lift}\), evaluation-bundle state keys,
constructor/static-support hash, envelope convention, centres, cuts, integer image
assignment, symmetry-equivalence rule, finite-difference perturbation range, tail bounds,
tolerances, and numerical representation are part of every result cache key whose functional
calls \(G_\infty\), and of every isolated-result cache key.

Define the conjunctive predicate
\(\operatorname{IsoOK}(\mathbf R,\mathbf h,S,S_{\rm ref})\) to mean:

1. the constructor, exact-count, \(m_F\le1\), and static-residual support gates pass for both
   the requested and every required reference state;
2. every active requested/reference frontier channel has exactly \(w_{c,\sigma}=1\);
3. registration is valid up to the equivalence rule above;
4. every complete component-resolved density entering an individual requested/reference
   defect or bulk \(\mathcal A_B^*\) independently passes the localisation test of Section
   4.1 and remains \(O(1)\) on the size ladder; compactness of their difference is not a
   substitute;
5. each such constituent independently passes the canonical-lift, branch-stability,
   boundary-clearance, and all density/energy/derivative tail bounds;
   and
6. all boundary-specific stationary-solution, stability, and derivative gates required by
   the requested output pass.

This predicate, not frontier compactness alone, controls every isolated or
boundary-complete output. When there is no active frontier channel, condition 2 is vacuous
but conditions 1 and 3--6 remain mandatory, so a carrier-free charged state with an extended
static density cannot pass accidentally. A validated PBC label-boundary output does not
require \(\operatorname{IsoOK}\). It does, however, require the canonical branch,
translation/rewrapping invariance, and force/stress differentiability whenever its
boundary-common term evaluates \(G_\infty\); failed localisation or tail bounds are then
reported and prevent only the isolated/boundary-complete claim. A non-unique or unstable
branch invalidates every output that calls \(G_\infty\).

## 5. Explicit frontier functional

### 5.1 Fixing the circular definition

The frontier density must be an explicit function of an independent \(P\). Its spectral
windows may not be rebuilt recursively from \(H_{B,S}[P]\) while also being treated as fixed
inside \(\delta\Phi/\delta P\).

The minimal accepted construction forms positive-semidefinite, exact-zero window operators
from the occupation-independent matrix:

\[
M_e=b_e(H_{\rm fix})^2,\qquad M_h=b_h(H_{\rm fix})^2.
\]

Each nonnegative \(b_c(\varepsilon)\) is a registered compact-support spectral window with
exact-zero outer plateaus, a unit inner plateau, and \(C^2\) quintic transitions. Hence
\(M_c^{1/2}=b_c(H_{\rm fix})\) without a singular square-root derivative at zero. Strictly
positive sigmoid tails are not permitted: with \(N_{\rm orb}=\dim H_{\rm fix}\), a nonzero
tail over \(O(N_{\rm orb})\) excluded bulk states can overwhelm an \(O(1)\) carrier.
Electron and hole support intervals are fixed relative to the aligned band edges and do not
expand with system size.

The aligned band edges entering \(b_e\) and \(b_h\) are not per-frame extrema. They are frozen,
size-stable host-edge scalars plus, if required, a potential-zero-covariant smooth alignment
formed from **rank-normalised** traces over frozen-rank pristine subspaces. Each trace is
divided by the corresponding frozen rank before differences are taken; an unnormalised,
extensive trace is forbidden. The alignment is independent of \(P\), basis invariant, and
fully differentiated. Hard maximum/minimum eigenvalue selection is forbidden. If a
fixed-rank subspace loses its separating gap or path consistency, the state is unsupported.

For each spin, construct the occupation-independent continued-valence projector

\[
P_{V,\sigma}^{\rm fix}
=\Pi_{M_{{\rm VB},\sigma}}(H_{\rm fix}),
\qquad
\Delta P_\sigma=P_\sigma-P_{V,\sigma}^{\rm fix}.
\]

The eigenvalue gap separating this rank-\(M_{{\rm VB},\sigma}\) projector must remain above
its registered floor throughout the supported geometry path; otherwise the projector and
state are unsupported.

Let \(r_+(x)\) be a registered nonnegative \(C^2\) spectral function that is exactly zero
for \(x\le\eta_0\), makes a quintic transition for \(\eta_0<x<\eta_1\), and equals \(x\)
for \(x\ge\eta_1\), with \(0\le\eta_0<\eta_1\le1\). With
\(B_c=b_c(H_{\rm fix})=M_c^{1/2}\), define

\[
D_{e,\sigma}[P]=B_e\,r_+(\Delta P_\sigma)\,B_e,
\qquad
D_{h,\sigma}[P]=B_h\,r_+(-\Delta P_\sigma)\,B_h.
\]

This extracts the positive electron or hole excess relative to the continued valence
reference instead of accumulating finite-smearing occupations over a broad band. The exact
counts \(n_{e/h,\sigma}\), not \(\operatorname{Tr}D\), set the monopoles. All matrix
functions use bounded divided differences, and the subspace-gap and leakage gates are part of
the supported-domain contract.

For an active channel,

\[
\widehat\rho_{c,\sigma}[P]
=\frac{\mathscr D[D_{c,\sigma}[P]]}
{\operatorname{Tr}D_{c,\sigma}[P]},
\qquad c\in\{e,h\},
\]

where \(\mathscr D\) is the registered positive, linear, trace-preserving orbital-to-density
map,

\[
\int\mathscr D[X](\mathbf r)\,d\mathbf r=\operatorname{Tr}X.
\]

A channel with zero exact count is omitted. For each active channel require

\[
0<\alpha_{\min}n_{c,\sigma}
\le\operatorname{Tr}D_{c,\sigma}
\le\alpha_{\max}n_{c,\sigma},
\qquad
0<\alpha_{\min}<1<\alpha_{\max},
\]

together with a fixed absolute leakage bound from the exact-zero spectral complement and a
matched carrier-free background trace that does not grow on the tiling ladder. These bounds
are independent of \(N_{\rm at}\), not per-atom tolerances. A failure is an unsupported state; it is
not stabilised by a denominator offset.

Let positive site projectors \(\Pi_i\) resolve the orbital identity,
\(\sum_i\Pi_i=I\), in the registered orthonormalised basis (or in the exactly equivalent
overlap-metric representation), and define

\[
\ell_{i,c,\sigma}
=\frac{\operatorname{Tr}(\Pi_iD_{c,\sigma})}
{\operatorname{Tr}D_{c,\sigma}},
\qquad
N_{{\rm eff},c,\sigma}=\frac{1}{\sum_i\ell_{i,c,\sigma}^2},
\qquad
p_{c,\sigma}=\frac{N_{{\rm eff},c,\sigma}}{N_{\rm at}}.
\]

The participation fraction \(p\) is diagnostic only; a fixed fractional threshold could
misclassify a state spread over a growing number of sites. Define the centre-free spread

\[
R_{{\rm eff},c,\sigma}^2
=\frac12\sum_{ij}\ell_{i,c,\sigma}\ell_{j,c,\sigma}
d_{\mathbf h}^2(\mathbf R_i,\mathbf R_j),
\]

where \(d_{\mathbf h}\) is a registered differentiable periodic metric that agrees with
Cartesian distance in the compact-support range. Use the exact-plateau \(C^2\) switch

\[
W(x;x_1,x_2)=
\begin{cases}
1, & x\le x_1,\\
1-s_5\!\left(\dfrac{x-x_1}{x_2-x_1}\right), & x_1<x<x_2,\\
0, & x\ge x_2,
\end{cases}
\qquad
s_5(t)=6t^5-15t^4+10t^3.
\]

Then set

\[
w_{c,\sigma}
=W(N_{{\rm eff},c,\sigma};N_{\rm loc},N_{\rm ext})
W(R_{{\rm eff},c,\sigma};R_{\rm loc},R_{\rm ext}),
\]

using fixed absolute site-count and physical-length thresholds that do not scale with
\(N_{\rm at}\) or cell length, with
\[
1\le N_{\rm loc}<N_{\rm ext}\le N_{\rm at},
\qquad
0<R_{\rm loc}<R_{\rm ext}.
\]
The site-count inequality must hold for every supported production cell; equivalently,
\(N_{\rm ext}\le\min_{\rm supported}N_{\rm at}\).
This makes the compact and extended image limits exact, identifies
subextensively spreading or spatially split states without a size-dependent threshold, and
keeps force and stress derivatives continuous. The full frontier density retains the actual
normalised density extracted from \(P\); it is not replaced by an arbitrary uniform or
host-shaped extended reference:

\[
\rho_F[P]
=-\sum_\sigma n_{e,\sigma}\widehat\rho_{e,\sigma}[P]
+\sum_\sigma n_{h,\sigma}\widehat\rho_{h,\sigma}[P],
\qquad
\int\rho_F=q_F.
\]

Electron and hole participation is evaluated separately, so opposite signs cannot cancel in
the localisation diagnostic. Image electrostatics acts on the total signed image-active
density defined below. Algebraically, its quadratic form contains all electron--electron,
hole--hole, electron--hole, and cross-spin terms rather than a same-channel approximation.
The present \(m_F\le1\) support guard means that distinct-frontier cross terms are not
exercised in production until the missing boundary-common functional is also defined.

The window operators depend on geometry through \(H_{\rm fix}\), so their complete matrix-
function derivatives still contribute to forces and stress. They are only fixed with respect
to the variational derivative in \(P\).

### 5.2 Unified image density

The exact counts are fixed parameters of an electronic state. More fully, the objects below
are \(\rho_F[P;S]\), \(\rho_{\rm img}[P;S]\), \(\Phi_B[P,\mathbf u;S]\), and
\(V_B[P,\mathbf u;S]\); the explicit \(S\) is suppressed for readability. Every functional
derivative with respect to \(P\) or \(\mathbf u\) is taken at fixed state specification and
fixed integer counts.

Define

\[
\rho_\Delta[P]=\rho_S+\rho_F[P],
\qquad
\int\rho_\Delta=Q_{\rm formal}.
\]

The density on which the image correction acts is

\[
\rho_F^{\rm img}[P]
=-\sum_\sigma n_{e,\sigma}w_{e,\sigma}\widehat\rho_{e,\sigma}
+\sum_\sigma n_{h,\sigma}w_{h,\sigma}\widehat\rho_{h,\sigma},
\]

\[
\rho_{\rm img}[P]=\rho_S+\rho_F^{\rm img}[P].
\]

Its monopole is

\[
q_{\rm img}
=Q_{\rm core}
-\sum_\sigma n_{e,\sigma}w_{e,\sigma}
+\sum_\sigma n_{h,\sigma}w_{h,\sigma}.
\]

Thus the full modeled density always has the exact formal monopole, while the image-active
frontier density contains only its compact part. In general
\(q_{\rm img}\ne Q_{\rm formal}\); equality holds when every active frontier channel is
compact, \(w_{c,\sigma}=1\). The boundary convention excludes the
\((1-w_{c,\sigma})\widehat\rho_{c,\sigma}\) fraction from \(\rho_{\rm img}\): an extended
band carrier retains its model orbital density in \(\rho_F\), but is not treated as a compact
charge with an isolated self-image. For a compact carrier,
\(\rho_{\rm img}=\rho_\Delta\). A PBC-to-isolated boundary conversion is exposed only when
\(\operatorname{IsoOK}\) from Section 4.2 passes. Its frontier clause requires every active
channel in both the requested and reference states to have exactly \(w_{c,\sigma}=1\), at
which point the image functional acts on the complete modeled density; its remaining clauses
also certify the static/excess density and isolated-space lift. If
\(\operatorname{IsoOK}\) fails, a PBC label-boundary output may still be returned only when
the separate canonical-branch and ordinary PBC gates of Section 4.2 pass; every isolated or
boundary-complete output remains unsupported. No effective-mass or band-filling correction
is implied. The smooth switch is a declared PBC model approximation, not an exact
electrostatic identity away from its compact plateau.

## 6. Corrected boundary and total-energy functional

### 6.1 Boundary and potential-zero conventions

For cell matrix \(\mathbf h\), let \(B_0(\mathbf h)=\mathrm{PBC}(\mathbf h)\) denote the
boundary convention of the training labels. Define the **bare** Green functions and their
difference by

\[
G_{\rm img}^{0}(\mathbf h)
=G_{\rm PBC}^{0}(\mathbf h)-G_\infty^{0}.
\]

Below, \(B_0\) is shorthand for this cell-dependent \(B_0(\mathbf h)\); it is never held
fixed when a cell derivative is taken.

Here PBC and infinity are different boundary conditions. The word *gauge* is reserved for an
additive potential-zero convention; that convention must be fixed within each boundary.
The frozen-pristine projection in Section 3.1 fixes the arbitrary scalar mode of the learned
local one-electron matrix, while the Green-function convention fixes the electrostatic
potential zero. They are distinct operations and both records enter the energy/cache
fingerprint.

The periodic bare kernel records the compensating background, removal of the zero reciprocal
mode, potential-zero convention, and cell-shape convention. The isolated bare kernel has zero
potential at infinity. Those choices are immutable parts of a model regime.

Before explicit polarisation, for registered \(\epsilon_\infty>0\), the electronically
screened scalar kernels are

\[
K_\infty=G_\infty^0/\epsilon_\infty,
\qquad
K_{\rm PBC}=G_{\rm PBC}^0/\epsilon_\infty,
\qquad
K_{\rm img}=K_{\rm PBC}-K_\infty.
\]

For an anisotropic dielectric these symbols denote the corresponding registered anisotropic
Poisson operators, not a second scalar division. The current validated regime may use a
scalar \(\epsilon_\infty\). If it uses a tensor, the tensor is stored as
\(\boldsymbol\epsilon_\infty^{\rm mat}\) in a registered orthonormal material frame attached
to the fixed canonical pristine reference for that tiling; it is symmetric positive definite
and its smallest eigenvalue exceeds a registered numerical floor. The material triad is
stored, so
\(\boldsymbol\epsilon_\infty^{\rm ref}\) denotes the same tensor expressed in the canonical
reference Cartesian frame. With
\(F=\mathbf h\mathbf h_{\rm ref}^{-1}=R_FU_F\) the right polar decomposition relative to
that fixed reference cell, the Cartesian tensor used by the kernel is

\[
\boldsymbol\epsilon_\infty^{\rm cart}(\mathbf h)
=R_F\boldsymbol\epsilon_\infty^{\rm ref}R_F^T.
\]

Thus it co-rotates under a rigid rotation. Under strain the material-frame components and
eigenvalues are held fixed—the declared clamped-dielectric approximation—while the complete
derivative through \(R_F\), the cell, reciprocal vectors, and Poisson solve is retained in
stress. Holding the Cartesian components of a nonscalar tensor fixed under rigid rotation is
prohibited. A different strain law, including photoelastic response, is a different model
regime and is outside this plan. The scalar/tensor choice, material triad, reference-cell
tiling and orientation,
polar-decomposition convention, strain law, and kernel derivative version are serialised in
the checkpoint and every affected cache key. This convention makes
\(K_{\rm PBC}=K_\infty+K_{\rm img}\) exact and prevents accidental double screening.
The bilinear form is

\[
B_K[\rho,\eta]
=\iint\rho(\mathbf r)K(\mathbf r,\mathbf r')
\eta(\mathbf r')\,d\mathbf r\,d\mathbf r'.
\]

When \(K=K_\infty\) or its long-range filter, both density arguments mean the
component-preserving lifts \(\mathscr U_{\mathbf h}\rho\) and
\(\mathscr U_{\mathbf h}\eta\) of Section 4.2. Periodic kernels act on the torus
representation. Mixing a wrapped argument with an isolated kernel is prohibited.
Accordingly, \(K_{\rm img}=K_{\rm PBC}-K_\infty\) is operator-difference shorthand for

\[
B_{K_{\rm img}}[\rho,\eta]
=B_{K_{\rm PBC}}^{\mathbb T_{\mathbf h}^3}[\rho,\eta]
-B_{K_\infty}^{\mathbb R^3}
[\mathscr U_{\mathbf h}\rho,\mathscr U_{\mathbf h}\eta],
\]

not a pointwise subtraction on an unspecified common grid.

Choose one registered smooth long-range filter
\(\mathcal L_{r_{\rm split}}\) and define

\[
K_\infty^{\rm LR}=\mathcal L_{r_{\rm split}}[K_\infty].
\]

The boundary-common long-range static--frontier interaction assigned to the head is the
explicit functional

\[
\Phi_{\rm SF}^{\infty,{\rm LR}}[P]
=B_{K_\infty^{\rm LR}}[\rho_S,\rho_F[P]].
\]

Both entries are signed charge densities. Its Hamiltonian contribution is the exact derivative
\(V_{{\rm SF},\sigma}=\delta\Phi_{\rm SF}^{\infty,{\rm LR}}/\delta P_\sigma\), using the
same registered orbital-to-density map as the energy; no unsigned scalar field is inserted
directly into an electron Hamiltonian. The fixed matrix is simply
\(H_{\rm fix}=H_{\rm local}\), where \(H_{\rm local}\) denotes the gauge-fixed form of
Section 3.1. It includes the accepted Slater--Koster, onsite, edge, and learned local terms.
It owns the omitted short-range physics phenomenologically; it is not asserted to
equal an analytic short-range Coulomb energy. The complementary analytic kernel
\(K_\infty^{\rm SR}=K_\infty-K_\infty^{\rm LR}\) may be constructed to test the filter and
to generate registered local covariant descriptors, but its bilinear energy is not evaluated
or added separately. The filter, transition width, and local cutoff are serialised and chosen
so that the retained tail is not also an unrestricted learned long-range channel. The
periodic image addition comes only from the functional below:

\[
\Phi_{\rm img}^{\rm PBC}[P]
=\frac12 B_{K_{\rm img}}
[\rho_{\rm img}[P],\rho_{\rm img}[P]],
\qquad
\Phi_{\rm img}^{\infty}[P]=0.
\]

This single quadratic form contains static--static, static--frontier, and
frontier--frontier terms of the image-active density exactly once. It is the complete
PBC-minus-isolated quadratic form of the full modeled density only on the supported compact
plateau, where \(\rho_{\rm img}=\rho_\Delta\). From Stage 5 onward, the boundary-dependent
static image potential and frontier response potential are derivatives of this same object.
No separate image potential or image energy may duplicate it.

### 6.2 Stationary auxiliary functional

Before Stage 6,

\[
\Phi_B[P]
=\Phi_{\rm SF}^{\infty,{\rm LR}}[P]
+\Phi_{\rm img}^{B}[P]
+\underbrace{\Phi_{\rm FF}^{\rm common}[P]}_{=\,0\ {\rm under}\ m_F\le1}.
\]

When Stage 6 is active, \(\Phi_B[P,\mathbf u]\) is replaced term by term as specified in
Section 7 and contains the consistent polarisation functional. Before Stage 6,
\(\mathbf u\) is a zero-dimensional absent variable and all \(\mathbf u\)-stationarity
statements are omitted. Define

\[
\mathcal A_B[P,\mathbf u;S]
=\sum_\sigma\left[
\operatorname{Tr}(P_\sigma H_{{\rm fix},\sigma})
+\mathcal R_{{\rm sm},\sigma}[P_\sigma]
\right]
+\Phi_B[P,\mathbf u],
\]

where \(\mathcal R_{\rm sm}\) is the occupation regulariser associated with the configured
smearing; it is \(-T\mathcal S\) only for the corresponding Fermi--Dirac convention. It is
defined so that minimising \(\operatorname{Tr}(PH)+\mathcal R_{\rm sm}[P]\) gives the
implemented \(F_{\rm band}(H,N)\). The functional is subject to the exact per-spin trace
constraints specified by \(S\).

Strictly positive occupation tails must pass the same fixed absolute background-leakage gate
as the spectral windows through the largest supported size. Otherwise replace the nominal
label smearing by a registered, differentiable compact-support approximation matched within
label precision and derive \(\mathcal R_{\rm sm}\) from that exact rule; a silent numerical
cutoff is not permitted.

The stationary equations are

\[
H_{B,S,\sigma}[P,\mathbf u]
=H_{{\rm fix},\sigma}+V_{B,\sigma}[P,\mathbf u],
\qquad
V_{B,\sigma}=\frac{\delta\Phi_B}{\delta P_\sigma},
\]

Within the certified lift branch, \(\mathscr U_{\mathbf h}\) is the Section-4.2 linear map
fixed with respect to \(P\). The derivative \(V_B\) therefore includes
\(\mathscr U_{\mathbf h}(\delta\rho_F/\delta P)\), the density normalisation, windows, and
localisation switches, but no derivative of a discrete image assignment. The branch and
assignment must remain unchanged, modulo an exactly invariant simultaneous common lattice
translation, over all SCF iterates and admissible \(P\)-finite-difference directions;
otherwise the stationary solution is unsupported.

\[
P_\sigma=\operatorname{count\_fill}_\sigma
(H_{B,S,\sigma}[P,\mathbf u],N_{S,\sigma}),
\qquad
\frac{\partial\Phi_B}{\partial\mathbf u}=0.
\]

Anderson mixing is only an accelerator. The accepted convergence residual is the unmixed
fixed-point residual

\[
r_{P,\sigma}=P_\sigma-
\operatorname{count\_fill}_\sigma
(H_{B,S,\sigma}[P,\mathbf u],N_{S,\sigma}),
\]

not the difference between two mixed iterates. Require registered bounds on
\(\lVert r_P\rVert\), \(\lVert[H,P]\rVert\), and the change in the stationary functional.
When dipoles are active, also require the raw polarisation residual. Warm starts may reduce
iterations but may not select a different solution; a deterministic multistart rule searches
for competing stationary solutions and selects the lowest value of the same functional.

Selecting a lower envelope is not by itself sufficient for a differentiable PES. Within the
registered production domain, the selected solution must be a stable minimum and, modulo
exact symmetry-equivalent copies with identical observables, satisfy

\[
\mathcal A_B^{(2)}-\mathcal A_B^{(1)}\ge\Delta_{\mathcal A}^{\rm guard}>0
\]

against every competing solution found by the registered multistart and continuation tests.
Energy and force continuity is checked along geometry paths. A gap closure, stability loss,
or force jump is an unsupported point unless a differentiable ensemble treatment is added;
such a treatment is outside the present plan.

Local minimality is not enough to guarantee a bounded implicit response. Express \(P\) in
the registered orthonormalised orbital basis and let \(\mathcal T_P\) be the admissible
trace-preserving Hermitian tangent space, with exact symmetry directions quotiented out and
the Frobenius metric fixed. The occupation active set and this metric are checkpointed; an
active-set topology change is unsupported unless handled by a separately differentiable
rule. At each accepted Stage-5 solution, the constrained energy Hessian must satisfy

\[
\lambda_{\min}\!\left(
\nabla^2_{\mathcal T_P}\mathcal A_B\right)\ge\lambda_P^{\rm guard}>0,
\qquad
\kappa\!\left(\nabla^2_{\mathcal T_P}\mathcal A_B\right)\le\kappa_P^{\max},
\]

with eigenvalues in the declared energy unit. At Stage 6 introduce the checkpointed,
geometry-independent dipole scale \(u_0>0\) and dimensionless coordinate
\(\widetilde{\mathbf u}=\mathbf u/u_0\). The joint tangent metric is

\[
\lVert(\delta P,\delta\widetilde{\mathbf u})\rVert^2
=\sum_\sigma\lVert\delta P_\sigma\rVert_F^2
+\lVert\delta\widetilde{\mathbf u}\rVert_2^2.
\]

The same minimum-eigenvalue and condition-number contract is applied to the full energy
Hessian in these scaled coordinates, or to the exact reduced-\(P\) Schur complement formed
in the same metric after eliminating \(\widetilde{\mathbf u}\). Passing a guard on
\(K_{\rm pol,B}\) alone is insufficient. A raw-residual Jacobian may be logged as an
additional solver diagnostic only with separately registered left/right block scalings and
its own thresholds; it is not declared numerically equivalent to the energy Hessian. These
bounds, together with the stationary-solution gap, are the operational implicit-function and
force-continuity contract.

Write the converged value as \(\mathcal A_B^*(S)\). The equivalent implementation through
the band free energy is

\[
\mathcal A_B^*(S)
=\sum_\sigma\left[
F_{\rm band}(H_{B,S,\sigma},N_{S,\sigma})
-\operatorname{Tr}(P_{S,\sigma}V_{B,\sigma}[P_S,\mathbf u_S])
\right]
+\Phi_B[P_S,\mathbf u_S],
\]

provided \(\partial F_{\rm band}/\partial H_{ab}=P_{ba}\) and the same \(V_B\) is the exact
functional derivative of the reported \(\Phi_B\). This statement, rather than a heuristic
fixed point, defines the variational energy and its double-count correction.

### 6.3 Fixed-training-boundary reference anchor

The corrected total energy is

\[
\boxed{
E_B(\mathbf R,S)
=E_{\rm base}^{B_0(\mathbf h)}(\mathbf R)
+\mathcal A_B^*(\mathbf R,S)
-\mathcal A_{B_0(\mathbf h)}^*(\mathbf R,S_{\rm ref})
+C_{Q(S)}
}
\]

with \(C_{Q(S_{\rm ref})}=0\). Equivalently,

\[
E_B=E_{\rm base}^{B_0}
+\underbrace{\bigl[\mathcal A_B^*(S)-
\mathcal A_B^*(S_{\rm ref})\bigr]}_{\Delta E_{\rm charge}^{B}}
+\underbrace{\bigl[\mathcal A_B^*(S_{\rm ref})-
\mathcal A_{B_0}^*(S_{\rm ref})\bigr]}_{\Delta E_{\rm ref}^{B\leftarrow B_0}}
+C_Q.
\]

The formula is algebraically defined for diagnostics wherever its stationary values exist.
It is exposed as a physical boundary-complete output only when
\(\operatorname{IsoOK}\) in Section 4.2 passes. On that supported domain it has four
required properties:

1. In the label boundary, \(B=B_0\), the reference state is exactly the frozen base in energy,
   force, and stress.
2. \(\Delta E_{\rm charge}^{B}\) is exactly zero at \(S=S_{\rm ref}\) under either boundary.
3. \(\Delta E_{\rm ref}^{B\leftarrow B_0}\) is zero at the training boundary but supplies the
   neutral-reference boundary completion at the isolated boundary.
4. The static--static image term cancels from the same-boundary charge increment but survives,
   as required, in the PBC-to-isolated reference completion.

When the canonical physical state key equals that of \(S_{\rm ref}\), the same-boundary
charge increment is returned algebraically as zero, reusing the identical cached functional
value and derivative. Two independently converged numerical copies may not leave a residual
reference energy, force, or stress.

The reference functional is evaluated at the **same atomic geometry** as the requested state.
Its coordinate and cell derivatives are retained. This is what makes the exact-null identity
hold for forces and stress, not only for scalar energies.

Every functional value that appears in an output must be stationary under its own boundary.
The direct production formula for a general state needs \(\mathcal A_B^*(S)\) and the
fixed-anchor value \(\mathcal A_{B_0}^*(S_{\rm ref})\). It does **not** require the two
\(\mathcal A_B^*(S_{\rm ref})\) terms in the conceptual decomposition to be evaluated, since
they cancel exactly. For \(S\ne S_{\rm ref}\), that otherwise absent \(B\)-boundary reference
solve is performed only when the two brackets are reported separately. For
\(S=S_{\rm ref}\), the requested-state solve is already the \(B\)-boundary reference solve;
it is not a third evaluation. A PBC density followed by a scalar kernel swap is not an
isolated evaluation. A stationary-value cache key includes the exact geometry and cell,
canonical physical state, model/checkpoint and parameter hash, frozen-pristine spectral-gauge
record, constructor record, boundary kernel and electrostatic potential-zero convention,
occupation and smearing implementation, canonical-lift and support-envelope record whenever
the functional calls \(G_\infty\), range/localisation settings, solver regime and tolerances,
and every non-parametric option
that can change the result. Alternatively, a cached solution may be reused at a stricter
tolerance only after its stored raw residuals and stability guards are rechecked against that
tolerance. Partial geometry-only or charge-only cache keys are forbidden.

### 6.4 Output semantics

The implementation exposes distinct quantities:

- `E_training_boundary`: the PBC energy comparable with the supplied labels;
- `delta_E_charge_B`: the optional diagnostic same-boundary electronic-state increment;
- `delta_E_reference_boundary_B`: the optional diagnostic neutral-reference boundary
  completion;
- `E_boundary_complete_B`: their sum with the frozen base and \(C_Q\);
- a structured support record for constructor, carrier multiplicity, density,
  registration, isolated-space lift, full defect-excess support/tail bounds,
  stationary-solution, and boundary-completion validity.

`E_training_boundary` may be returned for a noncompact state only if its ordinary PBC
predictive and derivative gates pass. A non-reference charged energy additionally requires
the exact checkpoint-matched spectral-gauge and \(C_Q\) calibration records; otherwise only
the uncalibrated component diagnostics may be returned. `delta_E_charge_B` at \(B=\infty\),
`delta_E_reference_boundary_B`, and `E_boundary_complete_B` are unavailable unless
\(\operatorname{IsoOK}\) passes for every required requested/reference and matched
defect/bulk calculation. If the reference-boundary term, full-density support/lift, or any
other predicate clause is disabled, no output may be named an isolated or boundary-complete
energy.

The boundary wrapper is applied independently to the defective and pristine calculations
before their difference is taken. A pristine boundary completion is identically zero only
when its modeled defect-induced density is identically zero under the same geometry and
reference convention.

### 6.5 Forces and cell derivatives

At a converged stationary solution,

\[
\mathbf F_I^B=-\left.\frac{\partial E_B}{\partial\mathbf R_I}\right|_{P,\mathbf u},
\qquad
\sigma_{\alpha\beta}^B
=\frac1\Omega
\left.\frac{\partial E_B}{\partial\epsilon_{\alpha\beta}}\right|_{P,\mathbf u}.
\]

More explicitly, for any coordinate or strain component \(x\),

\[
\frac{dE_B}{dx}
=\frac{\partial E_{\rm base}^{B_0(\mathbf h)}}{\partial x}
+\left.\frac{\partial\mathcal A_B[P_B,\mathbf u_B;S]}{\partial x}
\right|_{P_B,\mathbf u_B}
-\left.\frac{\partial\mathcal A_{B_0(\mathbf h)}
[P_{B_0},\mathbf u_{B_0};S_{\rm ref}]}{\partial x}
\right|_{P_{B_0},\mathbf u_{B_0}}.
\]

The checkpoint-matched \(C_Q\) is held fixed under coordinate and cell differentiation, so
its force and stress derivatives are exactly zero. Calibration is never rerun as a function
of an inference geometry or strain.

The envelope theorem removes the implicit stationary \(dP/d\mathbf R\) and
\(d\mathbf u/d\mathbf R\) terms; it does not remove explicit derivatives through
\(H_{\rm fix}\), \(P_V^{\rm fix}\), spectral windows and aligned edges, normalisations,
localisation weights, static and frontier densities, covariant pristine registration,
the smoothly moving primitives of the canonical lift under its locally fixed image
assignment, Gaussian widths, range split, real/reciprocal Green functions, dielectric
material-frame rotation, background and zero-mode convention, reciprocal cell,
polarisation matrix, polarisabilities, damping, or volume. Both
stationary values in the direct fixed-anchor formula contribute. If the two diagnostic
brackets are materialised, the two additional reference-state derivatives must cancel to the
numerical floor.

With fixed input \(\epsilon_\infty\), stress is defined under the scalar or co-rotational
clamped-tensor convention of Section 6.1. The dielectric values do not acquire an unmodelled
strain response, but every explicit cell and material-frame derivative declared there is
retained. Stress remains a consistency-tested analytic output, not a reference-accuracy
claim in the absence of stress labels.

## 7. Stage-6 screening correction

An induced-dipole model calibrated to reproduce \(\epsilon_\infty\) must not be placed on top
of a kernel already divided by \(\epsilon_\infty\). Upon entering Stage 6:

1. replace the screened Coulomb kernel by the bare kernel for every interaction represented
   by the dipoles;
2. include static--induced, frontier--induced, and induced--induced terms in the same
   \(\Phi_B[P,\mathbf u]\);
3. solve \(P\) and \(\mathbf u\) to joint stationarity under each boundary condition;
4. refit the Stage-6 head and rerun all prior derivative, predictive, and size gates.

The kernel replacement is explicit:

\[
K_{\rm img}^{(6)}=G_{\rm PBC}^{0}-G_\infty^{0},
\qquad
G_\infty^{0,{\rm LR}}
=\mathcal L_{r_{\rm split}}[G_\infty^0].
\]

\[
\Phi_{\rm SF}^{\infty,{\rm LR},(6)}[P]
=B_{G_\infty^{0,{\rm LR}}}[\rho_S,\rho_F[P]].
\]

\[
\Phi_{\rm img}^{\rm PBC,(6)}[P]
=\frac12B_{K_{\rm img}^{(6)}}
[\rho_{\rm img}[P],\rho_{\rm img}[P]],
\qquad
\Phi_{\rm img}^{\infty,(6)}=0.
\]

The Stage-5 screened \(K_{\rm img}\) and
\(\Phi_{\rm SF}^{\infty,{\rm LR}}\) are removed. The direct static--frontier functional and
permanent-charge image functional are rebuilt with the bare kernels above. The dipoles
own only the registered long-range induced response: their boundary-common permanent-charge
field uses the same \(\mathcal L_{r_{\rm split}}\), while the image field uses the complete
bare image difference, whose same-cell singularity has cancelled. The local Hamiltonian owns
the omitted short-range response phenomenologically. Thole damping stabilises the dipole
interaction but is not used as a substitute for this complementary range contract.

For example, represent the induced response once by

\[
\mathbf E_B[P]
=\mathbf E_\infty^{0,{\rm LR}}[\rho_\Delta[P]]
+\chi_B\mathbf E_{\rm img}^0[\rho_{\rm img}[P]],
\qquad
\chi_{\rm PBC}=1,\quad\chi_\infty=0,
\]

\[
\Phi_{\rm pol}^B[P,\mathbf u]
=\frac12\mathbf u^T K_{\rm pol,B}\mathbf u
-\mathbf u^T\mathbf E_B[P],
\qquad
K_{\rm pol,B}\mathbf u=\mathbf E_B,
\]

where \(K_{\rm pol,B}\) uses the same common/image boundary split and is independent of
\(P\). Its geometry dependence is fully differentiated. Throughout both boundary conditions
and the complete configuration/size ladder require

\[
\lambda_{\min}(K_{\rm pol,B})\ge\lambda_{\rm guard}>0,
\qquad
\kappa(K_{\rm pol,B})\le\kappa_{\max},
\]

and the scaled raw residual

\[
\frac{\lVert K_{\rm pol,B}\mathbf u-\mathbf E_B\rVert}
{\max(\lVert\mathbf E_B\rVert,E_{\rm floor})}<\epsilon_u.
\]

Here \(E_{\rm floor}>0\) has the same norm and units as the stacked site-field vector
\(\mathbf E_B\), and \(\epsilon_u>0\) is dimensionless. Both values, the norm convention,
and the unit system are checkpointed and included in every polarisation-solution cache key.

The frontier potential is therefore

\[
V_{{\rm pol},B}
=-\mathbf u^T\frac{\delta\mathbf E_B[P]}{\delta P}.
\]

The static--induced and frontier--induced interactions are the linear term and the
induced--induced interaction is the quadratic term;
they are not added again as independent energies.

The complete Stage-6 interaction functional is therefore

\[
\Phi_B^{(6)}[P,\mathbf u]
=\Phi_{\rm SF}^{\infty,{\rm LR},(6)}[P]
+\Phi_{\rm img}^{B,(6)}[P]
+\Phi_{\rm pol}^{B}[P,\mathbf u]
+\underbrace{\Phi_{\rm FF}^{\rm common,(6)}[P]}_{=\,0\ {\rm under}\ m_F\le1}.
\]

No screened Stage-5 copy of any displayed term survives in this functional.

The required interaction ledger is:

| Interaction | Before Stage 6 | Stage 6 |
|---|---|---|
| Local/short-range electronic physics | \(H_{\rm local}\), phenomenological | same, refitted |
| Boundary-common long-range S--F | screened direct \(\Phi_{\rm SF}^{\infty,{\rm LR}}\) | bare direct \(\Phi_{\rm SF}^{\infty,{\rm LR},(6)}\) plus induced response |
| Boundary-common distinct-carrier F--F | zero, with \(m_F\le1\) guard | zero, with \(m_F\le1\) guard |
| PBC-minus-isolated S--S/S--F/F--F image difference (complete on compact plateau) | screened \(\Phi_{\rm img}^{\rm PBC}\) | bare \(\Phi_{\rm img}^{\rm PBC,(6)}\) plus induced response |
| Permanent--induced and induced--induced | absent | \(\Phi_{\rm pol}^{B}\) only |
| Absolute PBC reference-state energy | frozen neutral base | frozen neutral base |

No row may contain both the screened-direct and bare-plus-induced representation. The
polarisation quadratic form must pass the eigenvalue, condition-number, and residual guards
above, and the complete pristine response—not individual fitted site polarisabilities—is
constrained to \(\epsilon_\infty\).

Stage 6 remains conditional. Any boundary-complete Stage-6 claim requires
\(\operatorname{IsoOK}\), including full static/excess-density support and the exact
frontier plateau. If an active channel has \(w<1\), the conditional dipole evaluator is
disabled for that state; an extended-carrier response functional is not inferred from the
compact model. Better-looking latent charges, dipoles, or SCF behaviour alone do not justify
Stage 6; it must improve a pre-registered held-out observable or a no-refit size error while
preserving all earlier gates.

## 8. Stage-1 energy-zero correction and revised stage amendments

### Completed Stage-1 result and diagnosis

The completed original-v8 Stage-1 runs contain a useful architecture result and a separate
energy-reference failure. Charged-frame forces improve relative to the Stage-B head under
the same evaluation, and neutral frames are identical because the neutral head is
algebraically zero and the trunk is frozen. The charged energy residual, however, is nearly
a constant within each size class and is already present on training frames.

For the reported examples, the frozen trunk is high by approximately \(4.9\) eV at 79 atoms
and \(5.9\) eV at 159 atoms. The Stage-B correction removes those offsets to about
\(0.05\) eV. A retrained seed supplying corrections of approximately \(-2.3/-3.5\) eV
leaves \(+2.6/+2.4\) eV, while another supplying \(-7.7/-7.5\) eV leaves
\(-2.8/-1.6\) eV. Removing the per-class mean leaves energy errors comparable with
Stage B. Thus the present evidence supports the improved force-dependent part of the PES; it
does not support the uncalibrated energy zero of the saved checkpoints.

Two mechanisms must be separated:

1. The learned spectrum moved by about \(2\) eV after the pre-epoch calibration. A common
   shift of a one-electron Hamiltonian changes the band contribution to a charged state while
   leaving its density, forces, and gap unchanged. Section 3.1 removes this exact spectral
   gauge mode.
2. The constant table was calibrated before training and then barely followed the moving
   band term. Excluding all 79-atom charged energies makes their mean offset unconstrained by
   construction. The 159-atom constant moving by only about \(0.03\) eV despite a
   multi-electron-volt training residual shows that the retained energy column did not
   effectively constrain the constant. This must be explained by the loss-path audit below,
   not attributed to generalisation.

If the current loss is a squared per-atom residual,

\[
\mathcal L_{E,i}=
\left(\frac{r_i+c}{N_{{\rm at},i}}\right)^2,
\]

then

\[
\frac{\partial\mathcal L_{E,i}}{\partial c}
=\frac{2(r_i+c)}{N_{{\rm at},i}^{\,2}}.
\]

For an \(O(1)\) defect-energy error, this suppresses the constant's gradient as the square of
the supercell size. That is a plausible explanation of the 159-atom observation, but it is
not accepted as the cause until the implemented loss and its gradient are traced exactly.

### Corrected energy-supervision contract

At fixed defect and carrier count, the charge-head energy is an \(O(1)\) defect or
electronic-state excess, even though the underlying total energy is extensive. Its residual
is therefore formed and scaled in **total-cell eV**, using one frozen physical energy scale,
never divided by \(N_{\rm at}\). Population balance is imposed by explicit stratum weights
rather than by per-atom normalisation or an ad hoc size multiplier.

For a charged observation \(i\), define the unpaired total-energy residual before an energy
constant as

\[
r_i(\theta)=
E_{\rm base}^{B_0}(\mathbf R_i)
+\Delta E_{\rm head}^{B_0}(\mathbf R_i,S_i;\theta)
-E_i^{\rm label}.
\]

Here \(\Delta E_{\rm head}^{B_0}\) is the complete training-boundary head correction with
every \(c_g\) or \(C_Q\) omitted; it includes whichever functional terms are active at that
stage.

If an existing same-geometry requested/reference pair is available, define instead

\[
r_i^\Delta(\theta)=
\Delta E_{\rm head}^{B_0}(\mathbf R_i,S_i;\theta)
-\left[E_i^{\rm label}(S_i)-E_i^{\rm label}(S_{\rm ref})\right],
\]

for which the frozen base cancels. Geometry-matched state groups are formed before the data
split, and every member of a group belongs to the same split. Pair eligibility requires the
same geometry and identical boundary, pseudopotential, electron-count convention, smearing,
total-energy convention, and potential-zero convention. The pair members need not have the
same electron count.

Each charged observation follows exactly one registered residual path. Define

\[
\xi_i(\theta)=
\begin{cases}
r_i^\Delta(\theta), & \text{eligible requested/reference pair},\\
r_i(\theta), & \text{otherwise}.
\end{cases}
\]

When the paired path is available it replaces the unpaired path; the charged label is not
counted a second time through \(r_i\). Paired and unpaired observations have distinct
label-provenance strata and never share a numerical nuisance intercept. A covariance-aware
joint likelihood is not part of this plan. No additional paired calculations are required.

Let \(g\) denote a pre-registered training stratum keyed only by label provenance, host,
formal charge, composition hash, cell convention, and size/shape class. This key is loss
metadata and never a model input or defect label. Give every represented stratum a frozen
total weight \(W_g\), and normalise frame weights within it, so the 928-frame and 16-frame
populations cannot dominate one another merely through their counts.

During Stages 1--4, all valid charged-energy frames, including the smaller cells, enter the
within-stratum energy-shape loss

\[
\bar \xi_g(\theta)=
\frac{\sum_{i\in g}w_i \xi_i(\theta)}{\sum_{i\in g}w_i},
\qquad
\mathcal L_E^{\rm shape}
=\sum_g W_g
\frac{\sum_{i\in g}w_i
\left[\xi_i(\theta)-\bar \xi_g(\theta)\right]^2}
{\sum_{i\in g}w_i}.
\]

The same within-stratum factor may be evaluated without a stale running mean through its
exactly equivalent weighted pair form,

\[
\frac{\sum_{i,j\in g}w_iw_j
\left[\xi_i(\theta)-\xi_j(\theta)\right]^2}
{2\left(\sum_{i\in g}w_i\right)^2}.
\]

Training must use either exact full-stratum statistics or a registered unbiased within-
stratum pair sampler. A minibatch mean that depends on accidental batch composition is not
the specified objective.

Equivalently, each stratum has the analytically profiled nuisance intercept
\(c_g^*(\theta)=-\bar \xi_g(\theta)\). These intercepts are diagnostics used to expose the
energy-dependent shape to the early-stage architecture tests. They are not checkpoint
parameters, may not appear in production outputs, and may not be used to establish
cross-size convergence. A stratum with fewer than two distinct energy observations supplies
no shape information and is reported rather than silently weighted.

Every nuisance profiler used to recalibrate a saved checkpoint is computed from training
observations only and frozen before held-out evaluation. A held-out-set-centred error may be
reported separately as a shape-only diagnostic, but its fitted mean never calibrates an
output or a production energy.

The old rule that admitted charged energies only when a same-size neutral null happened to
exist is retired. Such neutral data remain valuable for measuring frozen-base support and
for stratified reporting, but their availability is neither necessary nor sufficient to
make an unpaired charged total energy identify the base/head split. Unpaired frames use the
centred total-energy residual above, with their limitation reported.

Stages 1--4 use only the registered centred/nuisance-intercept convention. From Stage 5
onward, the nuisance intercepts are absent and every permitted paired or unpaired residual
uses only the single shared \(C_Q\). The two conventions are stage alternatives and are
never added together.

At Stage 5, after the complete size-dependent electrostatic functional is active, replace
the nuisance intercepts by the single production constant for each non-reference supported
charge. The reference value remains fixed algebraically at
\(C_{Q_{\rm ref}}=0\):

\[
C_Q^*(\theta)
=\underset{C}{\operatorname{argmin}}\,
\sum_{i\in\mathcal T_Q}
W_{g(i)}\,\widetilde w_i\,
\ell\!\left(\xi_i(\theta)+C\right),
\]

where \(\mathcal T_Q\) contains training observations only and
\(\sum_{i\in g}\widetilde w_i=1\). Every stratum entering the same profile must share the
declared pseudopotential, total-energy, electron-count, and potential-zero conventions.
Incompatible conventions are a dataset-contract failure and may not be hidden by another
constant. For squared loss,

\[
C_Q^*(\theta)=
-\frac{\sum_{i\in\mathcal T_Q}
W_{g(i)}\widetilde w_i \xi_i(\theta)}
{\sum_{i\in\mathcal T_Q}W_{g(i)}\widetilde w_i}.
\]

For a registered differentiable convex loss, the one-dimensional optimum is solved to its
declared tolerance. An exact profiled-objective implementation recomputes
\(C_Q^*(\theta)\) before every parameter-gradient evaluation; only at that current exact
profile does the envelope theorem permit taking the partial derivative with \(C_Q^*\) held
fixed. For a nonsmooth convex loss, use the registered Danskin subgradient and a deterministic
minimiser tie rule.

A deterministic alternating implementation may instead take multiple \(\theta\)-updates at
fixed \(C_Q\), but after its first parameter update it is optimising the joint objective at a
fixed, generally stale \(C_Q\), not the exact profiled objective, and the envelope theorem is
not invoked. It must converge both its ordinary parameter-optimality criterion and
\(C_Q-C_Q^*(\theta)\) to their registered tolerances. Every selection evaluation and saved
checkpoint is reprofiled exactly. The scalar reduction and one-dimensional solve are
negligible once residuals have been evaluated; any additional evaluation pass is timed and
reported. A checkpoint is never saved with a stale constant.

The Stage-5 constant is common to every cell size and boundary output. Residual differences
between size-stratum means after applying this one constant are a size-consistency diagnostic
and an acceptance gate; they are not repaired by restoring per-size constants. Validation,
tiling, or future larger cells never enter the calibration. Recalibration changes only a
geometry-independent scalar energy: forces, stresses, Hamiltonians, densities, and
localisation diagnostics must remain bit-identical.

### Required loss-path audit and immediate recovery of the completed runs

Before any Stage-1 retraining or later architecture arm:

1. Freeze the head and perturb the implemented 159-atom constant by known values, including
   \(\pm1\) eV. Verify the exact change in the scalar loss and compare autodiff
   \(\partial\mathcal L/\partial c\) with the analytic derivative of the implemented
   normalisation.
2. Trace all 16 retained charged energies through masks, indexing, units, population
   reduction, large-cell weighting, optimiser parameter groups, gradient clipping, dtype,
   and checkpoint restore. Assert that the predicted constant is added exactly once and that
   every intended frame reaches the actual loss column.
3. From an exact pre-update checkpoint, replay at least one complete recorded optimiser
   update with the original trainable parameter set, batch order, and optimiser state, and
   inspect the 159-atom constant's update. In a parallel isolated replay, freeze every other
   parameter while retaining any recorded global clipping or scaling factor that coupled its
   update to the full gradient vector. Reconstruct the observed change from the energy-loss
   coefficient, frame weights, gradient accumulation, mixed-precision scaling and
   unscaling, clipping, weight decay, optimiser moments, and learning rate. Replay enough
   recorded updates to account quantitatively for the observed \(0.03\) eV motion or identify
   the overwrite, detach, or frozen-parameter event responsible. If the historical optimiser
   state was not retained, perform an equivalent deterministic reproduction before
   retraining and record that the historical cause remains unproven.
4. With all other parameters frozen, show that the analytic profiler recovers an injected
   constant offset to the numerical floor. This replaces any need to tune a learning rate
   for an intercept.
5. Recompute the diagnostic \(c_g^*\) for every existing saved seed using training frames
   only. Report the raw selected-residual mean, profiled selected-residual mean,
   within-stratum error, and
   between-size intercept difference. The profiled mean must be zero to its analytic
   tolerance, while forces, stresses, spectra, densities, and all non-energy gates remain
   bit-identical.
6. Repeat selection readouts only after this saved-checkpoint recalibration. If the
   force result and the profiled energy-shape result retain the Stage-1 choice, no physics
   arm is reopened. If they change that choice, only the affected comparison is rerun.

Failure of items 1--3 to close on the current code path is an implementation blocker. A
passed post-hoc nuisance profile centres the saved models' residuals for retrospective
Stage-1 comparison; it is not a deployable energy calibration. Production training must
still adopt the gauge and
profiled-objective contracts above.

### Stage 0

- Adopt the three Hamiltonian names in Section 3.1.
- Apply the frozen-pristine spectral gauge of Section 3.1 and include its convention,
  pristine-projector fingerprint, gap result, and scalar value in the checkpoint and every
  energy cache key.
- Replace the VBM-proximity Tier-1 test by the rank-certified gap verifier in
  Section 3.2; rerun the 79- and 159-atom classes and compare the latter directly with the
  already completed, fingerprint-compatible Tier-2 record after all of that record's
  acceptance gates are verified.
- Build \(u_{\rm al}\) only from independently ranked, existing constructor records using
  the protocol in Section 3.2; store its provenance and route to Tier 2 when it cannot be
  certified.
- Replace carrier updates by the signed-excess count in Section 3.3.
- Make \(H_{\rm class}\) a separately versioned, frozen constructor.
- Correct the definition and tests of \(q_{\rm raw}\).
- Add exact geometry/cell, canonical state, model/checkpoint, constructor, boundary and
  potential-zero, occupation/smearing, functional settings, solver-regime, and—whenever
  \(G_\infty\) is called—canonical-lift/support-envelope fingerprints to every result cache
  key.
- Compute \(m_F\), store it in the class/state record, and test the Stage-4--6 hard guard.
- Add tests for electron-to-hole crossings. Retain a nonzero formal reference charge only as
  an algebra-only synthetic count test; reject it from the production neutral-base energy
  path.

These changes are interface and identity corrections. The current benchmark's legacy
`count_fill` numerical path should remain unchanged where the corrected definitions are
algebraically identical. The spectral-gauge projection deliberately changes the numerical
zero of raw eigenvalues; after its matched edge shift and energy recalibration, occupations,
density matrices, forces, stresses, gaps, and other gauge-invariant readouts must be
unchanged.

### Stage 1

- Complete the loss-path audit and saved-checkpoint recalibration above before interpreting
  the existing energy results.
- Retire the same-size-neutral-null inclusion rule. Use every valid small- and large-cell
  charged energy in the total-eV, population-balanced, within-stratum shape loss; keep
  neutral-null availability as a base-support diagnostic.
- Freeze the total-energy scale, stratum definitions and weights, force/energy balance,
  pair-sampling rule, and all joint tolerances before opening the corrected retraining
  results. This is a new corrective regime, not a retrospective rescore presented as a
  pre-registered test.
- Retrain the selected Stage-1 reference with the same registered seed protocol under the
  corrected gauge and objective before Stage 2. Existing physics arms need not be repeated
  unless the corrected selection readouts change their ordering.
- Require the total-eV energy-shape term and force term to remain compatible with one
  conservative PES in every stratum. If adding the small-cell energy shapes causes a
  pre-registered force regression or irreconcilable energy/force trend, stop for objective
  identifiability; do not silently drop that size or introduce a size-specific offset.
- Retain the existing scalar-range ablation, but freeze its numerical decision rule before
  examining the arms.
- Select the simpler arm when predictive differences lie within the registered equivalence
  margin.
- Do not use latent charge localisation or a bound hit alone as acceptance evidence.

### Stages 2 and 3

Both stages inherit the gauge-fixed Hamiltonian and the total-eV, analytically centred
Stage-1--4 energy-shape objective. No stage may restore a trainable per-size constant or the
old neutral-null admission rule.

The Stage-2 routing is made exhaustive. Let criteria 1--4 retain their v8 numerical
definitions:

- **A:** criteria 1, 2, 3, and 4 pass: skip Stage 3 and continue to Stage 4.
- **D:** criteria 2, 3, and 4 pass but criterion 1 fails: activate the conditional Stage-3
  edge residual.
- **B:** criteria 2 and 3 pass but criterion 4 fails, irrespective of criterion 1: stop
  architecture growth and investigate objective identifiability.
- **C:** criterion 2 or 3 fails: repeat covariance/sign tests and the single registered bound
  release; stop if the repeat fails.

This explicitly covers the case in which only the hopping-stop criterion fails.

### Stage 4

- Reject every nontrivial charge-head or boundary-conversion path for which the requested
  state or reference has \(m_F>1\), before evaluating an energy, force, or stress.
- Use the smooth, everywhere-defined residual density in Section 4.
- Implement and fingerprint the occupation-independent-branch, component-preserving linear
  lift of Section 4.2 before any \(G_\infty\) contraction. Require its
  branch/invariance/derivative gates for the
  Stage-4 PBC diagnostic and reserve full \(\operatorname{IsoOK}\) for a boundary-complete
  claim.
- Retain the total-eV, within-stratum energy-shape objective; the Stage-4 electrostatic
  change does not authorise a fitted per-size energy zero.
- Keep \(H_{\rm fix}=H_{\rm local}\) and evaluate
  \(P^{(0)}=\operatorname{count\_fill}(H_{\rm fix},N_S)\).
- Evaluate \(\Phi_{\rm SF}^{\infty,{\rm LR}}[P^{(0)}]+\Phi_{\rm img}^B[P^{(0)}]\) as a
  forward-only two-boundary diagnostic, retaining
  the complete \(dP^{(0)}/d(\mathbf R,\mathbf h)\) response in forces and cell derivatives.
  Do not feed either \(\delta\Phi_{\rm SF}/\delta P\) or
  \(\delta\Phi_{\rm img}/\delta P\) back into the Hamiltonian at this stage, and do not yet
  claim a stationary boundary-complete production model.
- Remove any independent static image potential or pairwise image patch.
- Add affine-reference strain and thermal-background tests.

### Stage 5

- Retain the \(m_F\le1\) production guard; the missing common \(F\!-\!F\) ledger entry remains
  exactly zero rather than being absorbed into \(C_Q\) or the residual density.
- Freeze the frontier windows with respect to \(P\) as in Section 5; retain all geometry
  derivatives.
- Replace pairwise or same-channel image patches by the total signed image-active-density
  functional in Section 6.
- Activate the common static--frontier and image potentials through
  \(V_B=\delta\Phi_B/\delta P\), the stationary SCF solution, and the band-energy double-count
  correction only at this stage.
- Use the fixed-training-boundary reference anchor.
- Solve every functional value used by the direct fixed-anchor output under its own boundary;
  perform the otherwise cancelling \(S_{\rm ref}\) solve under \(B\) only when separately
  reporting the two diagnostic brackets.
- Replace all diagnostic nuisance intercepts by the analytically profiled \(C_Q^*\) of
  Section 8, one per benchmark charge and common to all cell sizes. Use total-cell-eV
  residuals and frozen stratum weights in both its calibration and the profiled energy loss.

Stage 5 is not accepted until both the ordinary same-boundary charge increment and the
reference-boundary completion pass finite-difference and tiling tests with
\(\operatorname{IsoOK}\) true for every required requested/reference and matched defect/bulk
calculation. States that retain a unique differentiable canonical branch and pass the
ordinary PBC gates, but fail only the full localisation, clearance, or tail clauses, may
validate the PBC label-boundary model; they do not satisfy the boundary-completion gate.

### Stage 6

- Replace, rather than augment, \(\epsilon_\infty\)-screened interactions with bare-kernel
  plus induced-dipole interactions according to Section 7.
- Retain the \(m_F\le1\) guard and require \(\operatorname{IsoOK}\) for every
  boundary-complete Stage-6 evaluation.
- Recompute \(C_Q^*\) from the same frozen training set and weights after the Stage-6
  functional changes; never carry forward a calibration from a different functional.
- Pre-register the observable and size-error thresholds that could justify adoption.

## 9. Decision thresholds and tie breaking

Every optional arm must have a signed decision table stored in the run manifest before its
results are opened. For each metric it contains:

- direction of improvement;
- a practical tolerance \(\tau_{\rm phys}\);
- a repeat/seed uncertainty \(\tau_{\rm noise}\);
- an equivalence margin for permitted regressions;
- the population and split on which the metric is evaluated.

An improvement counts only when it exceeds

\[
\tau=\max(\tau_{\rm phys},\tau_{\rm noise}).
\]

All exact identities and derivative tests are hard constraints, not weighted scores. An
optional term is retained only if it improves at least one named acceptance metric and does
not regress an earlier gate beyond its equivalence margin. If models are indistinguishable,
choose the simpler model. Thresholds may be estimated from baseline repeat noise, seed spread,
and the already registered property tolerance, but may not be chosen after seeing the arm's
result.

If an arm's results have already been inspected, any newly assigned threshold is labelled
retrospective. It can guide engineering, but it is not described as pre-registered evidence.

No model is ranked by an energy metric until the corresponding saved checkpoint has passed
the loss-path audit and its permitted nuisance intercept or production \(C_Q^*\) has been
recomputed. Early-stage energy comparisons use the within-stratum shape error and report the
intercepts separately. Stage-5--6 comparisons use the single shared \(C_Q^*\), and report
both within-stratum error and residual between-size mean bias. Raw, pre-calibration charged
energy errors are diagnostics only.

## 10. Reference constant and thermodynamic scope

For the current single-host benchmark, \(C_Q\) is a profiled training-set alignment
parameter:

- one value per supported formal charge;
- \(C_{Q_{\rm ref}}=0\);
- shared across every cell size and boundary output;
- obtained by the analytic or one-dimensional convex profile in Section 8, not ordinary
  stochastic-gradient training;
- recomputed from the full permitted training set whenever the checkpoint or functional
  changes;
- never fitted to a validation or tiling size;
- prohibited from absorbing any \(1/L\), strain, or density-support trend.

The calibration record contains the checkpoint hash, spectral-gauge record, exact frame,
geometry-group, split, stratum, and selected-residual-path hashes, label and unit convention,
loss function, frame and stratum weights, optimum, optimality residual, and objective
version. A checkpoint without its matching record has no calibrated charged energy. Loading
a stale record is an error, not a warning.

The early-stage \(c_g^*\) values of Section 8 are analytically eliminated nuisance
intercepts, not aliases for \(C_Q\). They may diagnose base coverage and recover the energy
shape of completed legacy runs, but they may not be serialised as deployable per-size
constants. The production transition to a single shared \(C_Q\) is a hard Stage-5 gate.

It is neither an electrostatic correction nor an atomic/electron reservoir. A host-indexed
table of \(C_Q\) is not an acceptable foundation-model mechanism. A later cross-host model
must use a common electronic reference convention or a universal reference function; this
benchmark cannot identify that extension.

Formation energies additionally require matched bulk subtraction, atomic chemical
potentials, and an electron chemical-potential/band-edge convention. The model supplies the
configurational energy under its declared boundary condition; a learned frontier eigenvalue does
not replace those thermodynamic terms.

## 11. Added verification gates

### 11.1 Algebraic and variational tests

- \(Q_{\rm formal}=Q_{\rm core}+q_F\) for all supported count sequences, including crossings.
- Adding an arbitrary scalar \(aI\) in the registered orthonormal representation, or
  \(aS\) in an equivalent nonorthogonal implementation, to every raw runtime Hamiltonian
  shifts \(\mu_{\rm g}\) by exactly \(a\) and leaves the gauge-fixed Hamiltonian, aligned
  windows, band functional, charged energies, densities, forces, and stresses unchanged to
  the numerical floor. The pristine gauge projector retains its rank and gap throughout
  training.
- The 159-atom loss-path audit accounts for all 16 intended charged energies and reproduces
  the implemented scalar-loss change and constant gradient under injected
  \(\pm1\) eV offsets. The result is reported before and after every mask, reduction, and
  population weight. A deterministic optimiser replay quantitatively closes the observed
  \(0.03\) eV trajectory or identifies the exact detach, overwrite, or frozen-parameter
  event.
- For every saved legacy Stage-1 seed, analytic nuisance profiling makes the weighted
  training residual mean vanish separately in every eligible stratum while leaving the
  within-stratum residuals and every non-energy prediction bit-identical. The raw and
  profiled results remain separately tagged.
- The Stage-1--4 energy-shape loss is invariant to a constant residual shift within a
  stratum, uses total-cell-eV units, and gives each stratum its declared total weight
  independent of frame count and \(N_{\rm at}\). Both small- and large-cell charged frames
  reach the loss. Its full-stratum centred and weighted pair forms agree in value and
  parameter gradient to the numerical floor.
- Geometry-paired state groups never split across train/validation/test. An eligible charged
  label enters exactly one of \(r_i\) or \(r_i^\Delta\), paired and unpaired strata have
  separate nuisance profiles, and no held-out residual contributes to \(c_g^*\) or \(C_Q\).
  Tests also assert that Stages 1--4 contain no production \(C_Q\) and Stages 5--6 contain no
  nuisance \(c_g^*\).
- On held-out frames in every represented size stratum, the corrected energy-shape and force
  residuals satisfy their pre-registered joint tolerance without a size-dependent sign or
  slope conflict. Failure routes to objective-identifiability review rather than selective
  removal of a population.
- At Stage 5 and again after any Stage-6 change, the stored \(C_Q^*\) satisfies its analytic
  or convex optimality condition on training data, is identical whichever permitted order
  the training frames are accumulated in, and is common to all sizes. Holding out one size
  from evaluation does not trigger recalibration; its residual mean is a prediction.
- The stored \(u_{\rm al}\) is exactly reconstructed from its independently ranked source
  records, perturbations, and numerical margin. Removing its last applicable source or
  altering its provenance hash invalidates Tier 1 and routes the target to Tier 2.
- If the stored 159-atom Tier-2 anchor passes all of its own gates and the certified-rank
  gaps pass the frozen Tier-1 guards, Tier 1 accepts both the 79- and 159-atom
  non-mixed-valence defect classes. On the 159-atom class it reproduces the accepted Tier-2
  \(M_{{\rm VB},\sigma}\) separately in each spin channel. Across 79 and 159 atoms, its rank
  increment equals the 80-atom pristine valence rank while \(d_\sigma\), \(Q_{\rm core}\),
  and the carrier counts remain invariant. The selected threshold rank is stable to the full
  registered alignment, search-window, numerical, and small-geometry perturbations.
- An instrumented routing test performs no Tier-2 continuation step or continuation
  eigensolve after every Tier-1 gate has passed; reading the compatible anchor fingerprint
  does not execute Tier 2.
- Changing the target class-reference geometry/cell, eigensolver, backend, dtype, arithmetic
  precision, or any tier-specific tolerance invalidates the corresponding constructor
  record. Cross-size reuse succeeds only through the field-wise homology/tiling predicate;
  it neither requires target-specific hashes to match nor ignores them.
- An intentionally edge-crowded homologous toy with a certified rank and clean
  valence--frontier gap remains in Tier 1 as its valence sampling is densified. A toy with
  no accepted homologous anchor, conflicting anchor ranks, or a certified-rank gap below the
  registered floors routes to Tier 2; Tier 1 never substitutes a later, larger gap.
- Accepted Tier-2 constructor paths agree at each size; under tiling,
  \(M_{{\rm VB},\sigma}\) increases by the known pristine valence rank while
  \(Q_{\rm core}\) and the defect excess counts remain invariant.
- Every nontrivial Stage-4--6 evaluation with \(m_F>1\) in either the requested or reference
  state fails before energy, force, or stress construction. The exact neutral-reference
  training-boundary base-only identity remains available without evaluating the head.
- An isolated or boundary-complete output is populated only when the complete
  \(\operatorname{IsoOK}\) predicate passes. Targeted negative tests separately fail the
  frontier plateau, static/excess localisation, canonical lift, boundary-clearance/tail,
  registration, and stationary clauses and verify that none is treated as optional. At
  \(w=1\), \(\rho_{\rm img}=\rho_\Delta\) remains an algebraic identity but is not by itself
  sufficient for an isolated output. A constructed defect/bulk pair whose difference is
  compact but whose individual densities are extended must fail the independent-wrapper
  gate; compactness after subtraction may not make either constituent valid.
- Translating a compact supported density continuously through every cell face changes its
  canonical lift only by continuous motion and, at a branch handoff, a common lattice
  translation. The isolated energy, force, and stress remain continuous and invariant to the
  numerical floor. Symmetry-equivalent lifts agree, while a deliberately extended static
  density or ambiguous circular moment fails the registered lift/support gate. Across all
  accepted SCF iterates and admissible \(P\)-matrix finite differences, the
  occupation-independent branch and integer image assignment remain unchanged modulo an
  exactly invariant simultaneous common lattice translation.
- \(\int\rho_S=Q_{\rm core}\), \(\int\rho_F=q_F\), and
  \(\int\rho_\Delta=Q_{\rm formal}\).
- \(\int\rho_{\rm img}=q_{\rm img}\) with the channel-weight formula in Section 5.2, and
  \(q_{\rm img}=Q_{\rm formal}\) only when every active channel has \(w=1\).
- `g_res` and all of its derivatives remain finite when \(\delta Z_i\to0\).
- \(\delta\Phi_B/\delta P\) agrees with matrix finite differences, including window
  normalisation, participation weights, and the lifted density response
  \(\mathscr U_{\mathbf h}(\delta\rho/\delta P)\). The test verifies that the registered lift
  is linear and fixed with respect to \(P\) throughout the perturbation and that any image
  reassignment rejects the point rather than being differentiated through a hard branch.
- \(\partial F_{\rm band}/\partial H_{ab}=P_{ba}\) for the production smearing.
- At a stationary solution, the primary and band forms agree to the numerical floor under
  both boundaries, for requested and reference states and again at Stage 6 if active:

  \[
  \sum_\sigma\!\left[
  \operatorname{Tr}(P_\sigma H_{{\rm fix},\sigma})
  +\mathcal R_{{\rm sm},\sigma}[P_\sigma]\right]+\Phi_B
  =\sum_\sigma\!\left[F_{\rm band}
  -\operatorname{Tr}(P_\sigma V_{B,\sigma})\right]+\Phi_B.
  \]
- The stationary energy is invariant to the chosen SCF history and converges with solver
  tolerance.
- The selected SCF solution passes the stability and competing-solution gap guard, and energy
  and forces remain continuous along registered geometry continuations.
- The constrained Stage-5 energy Hessian passes its minimum-eigenvalue and condition-number
  guards in the registered tangent metric. At Stage 6 the scaled full coupled block or exact
  Schur complement passes; a \(K_{\rm pol}\)-only result is not sufficient.
- When Stage 6 is active, the polarisation eigenvalue/condition-number guards and scaled raw
  residual pass under both boundaries and throughout the size ladder.
- Rigid translation, rigid rotation, atom permutation, and periodic rewrapping co-transform
  the defective and registered pristine densities and leave scalar energies invariant;
  vector/tensor derivatives transform covariantly. Homogeneous strain finite differences
  include the registered pristine mapping. For a nonscalar dielectric, rigid-rotation and
  all six strain tests also exercise the material-frame polar-decomposition map and its
  kernel derivative; a Cartesian-fixed tensor must fail the rotation test.
- As a diagnostic below the gauge-projection layer only, a constant shift inserted while
  artificially holding \(\mu_{\rm g}\) fixed produces the analytically expected
  electron-count/reference term. The production path must remove that shift through the
  gauge projection above.

### 11.2 Boundary-reference identities

At the same geometry:

\[
E_{B_0}(S_{\rm ref})=E_{\rm base}^{B_0},
\qquad
\Delta E_{\rm charge}^{B}(S_{\rm ref})=0,
\qquad
\Delta E_{\rm ref}^{B_0\leftarrow B_0}=0.
\]

All three identities are algebraic in energy, force, and stress; the numerical
implementation must return exact zeros under canonical reference-state equality. The
remaining boundary and component identities hold to the registered numerical floor. Any
identity involving an exposed isolated or boundary-complete output is tested only after the
requested/reference states and matched defect/bulk wrappers pass
\(\operatorname{IsoOK}\). Component expansion of
\(\Phi_{\rm img}\) must reproduce the image-active quadratic form and contain every S--S,
S--F, and F--F cross term of that density exactly once. On the compact plateau it must equal
the corresponding quadratic form of \(\rho_\Delta\). It must also satisfy

\[
E_{B_0}(S)-E_\infty(S)
=\mathcal A_{B_0}^*(S)-\mathcal A_\infty^*(S)
\]

without refitting \(C_Q\), on the supported boundary-complete domain.

### 11.3 Finite differences

Repeat coordinate and all six homogeneous-strain finite differences for every functional
value actually used by the direct output and, when the diagnostic decomposition is requested,
for:

- requested-state stationary value;
- same-boundary reference subtraction;
- fixed-training-boundary reference completion;
- the final assembled total;
- all separately evaluated PBC and isolated stationary solutions;
- each Stage-6 polarisation component, if activated.

The discrepancy must decrease with the electronic and polarisation solver tolerances.

### 11.4 Size and support ladder

For requested/reference states and matched defect/bulk wrappers satisfying
\(\operatorname{IsoOK}\), report the following separately. When that predicate fails, retain
only the applicable PBC fields and mark every boundary-complete field unavailable rather
than filling it with a hybrid value:

- raw total energy and energy per atom, neither used as the defect-convergence metric;
- same-cell neutral defect excess;
- same-boundary charge increment;
- reference-boundary completion;
- boundary-complete defect excess;
- S--S, S--F, F--F, and induced components;
- local-force convergence and the volume-scaled excess stress
  \(\Omega[\sigma_B(D^q,L)-\sigma_B(\mathrm{bulk},L)]\), evaluated at identical cell,
  strain, and boundary convention;
- carrier and image-density participation/support flags.

For a compact charged image object, the leading periodic image component must scale with
\(q_{\rm img}^2/L\) under the declared cell, background, and dielectric convention. The
formal-charge coefficient is required only in the compact-support regime where
\(q_{\rm img}=Q_{\rm formal}\).
For a compact neutral total image density, the **sum** of S--S, S--F, and F--F terms must
lose that leading monopole contribution; the individual partitioned terms may contain
\(1/L\) pieces that cancel. No exponent is imposed on an unmatched thermal displacement
background. No size-dependent refit is allowed.

Model tiling proves internal consistency of the functional, not physical accuracy. The two
available labelled sizes remain the direct external size-transfer check.

### 11.5 Partition sensitivity

Vary valid residual widths, smooth residual-weight rules, frontier windows, and
localisation-crossover thresholds while preserving exact total charge. Boundary-complete
values are compared only across variants that retain a common
\(\operatorname{IsoOK}\) domain; loss of any predicate clause is recorded separately as a
support-robustness failure.
Within the common supported domain, observables must remain within a registered tolerance and
the sensitivity must not grow with cell size. Failure means the latent core/frontier
partition is controlling the answer and the isolated claim is withheld.

## 12. Corrected claim boundary

After the amended tests pass, the model may claim:

- conservative PBC energies, forces, and analytic stresses for the validated single-frontier-
  carrier charge model;
- exact formal charge and a stationary charge-response functional;
- one size-independent charge reference within the benchmark;
- internally consistent PBC-to-isolated charge and neutral-reference boundary terms on the
  registered domain where every independently evaluated defect/bulk density is compact and
  supported, satisfying
  \(\operatorname{IsoOK}\);
- no-refit convergence of the defect excess on registered
  \(\operatorname{IsoOK}\)-passing tiling ladders.

It may not claim, from the current data alone:

- nontrivial charge-head or boundary-conversion predictions for \(m_F>1\);
- a validated boundary-common frontier--frontier interaction;
- isolated or boundary-complete outputs when any clause of
  \(\operatorname{IsoOK}\) fails, including a noncompact frontier, static/excess density, or
  ambiguous isolated-space lift;
- quantitative Kohn--Sham eigenvalues or densities;
- reference-stress accuracy;
- a complete correction for extended/resonant carrier band filling;
- a complete neutral long-range polar potential from a short-range base;
- physical uniqueness of the static/frontier partition;
- universally accurate asymptotic convergence from only two labelled sizes;
- a demonstrated 100,000-atom electronic solver.

The large-cell architecture remains compatible with a sparse solver, but solver engineering
is a separate programme. The functional defined here is the object that such a solver must
evaluate.

## 13. Resolution of the architecture audit

| Finding | Resolution | Required by |
|---|---|---|
| \(H_0\) was overloaded | Use \(H_{\rm class}\), \(H_{\rm fix}\), and \(H_{B,S}[P,\mathbf u]\) | Stage 0 |
| A common spectral shift drifted by seed and changed the charged band-energy zero | Project out one frozen-pristine, rank-normalised scalar gauge mode and shift all aligned edges consistently | Stage 0 |
| The spectral gauge used \(I\) without declaring an orbital metric | Make Sections 3--7 orthonormal by convention; use \(H-\mu_{\rm g}S\) in an equivalent generalised eigenproblem | Stage 0 |
| Pre-epoch \(c\) calibration became stale, while retained large-cell energies did not visibly move it | Audit the exact loss path; profile nuisance intercepts and \(C_Q\) analytically at every selected checkpoint | Before Stage 1 is interpreted |
| Per-atom loss scaling suppresses an \(O(1)\) defect-energy signal as cells grow | Form the charge-head energy residual in total-cell eV and balance explicit strata independently of \(N_{\rm at}\) | Before Stage-1 retraining |
| The same-size-neutral-null rule discarded 928 usable small-cell energies and was dataset-specific | Use all valid charged energies in a within-stratum shape loss; retain neutral data as support evidence, not a binary admission rule | Before Stage-1 retraining |
| Paired and unpaired energy paths could double-count a charged label or leak a pair across splits | Group before splitting; select exactly one residual path per observation; use stage-exclusive intercept conventions | Before Stage-1 retraining |
| An envelope gradient was claimed while a held \(C_Q\) could be stale | Restrict the envelope theorem to an exact current profile and give alternating optimisation its own joint convergence contract | Stage 5 |
| Tier 1 tested eigenvalue proximity to the VBM and became size-dependent under folding | Verify a homologously certified rank at its separating gap; compare extensive ranks by their pristine increment | Stage 0 |
| The Tier-1 alignment-error bound lacked reproducible provenance | Construct \(u_{\rm al}\) from independently ranked existing records and route to Tier 2 when uncertifiable | Stage 0 |
| Carrier updates could become negative | Recompute signed excess from \(N_\sigma-M_{{\rm VB},\sigma}\) | Stage 0 |
| \(Q_{\rm core}\) was treated as a harmless gauge despite entering the frontier potential | Make the partition constructor-fixed; impose \(m_F\le1\) until a self-interaction-controlled common \(F\!-\!F\) functional exists | Before Stage 4 |
| \(g_{\rm res}\) had a zero denominator | Smooth weights, size-vanishing fallback, and unsupported-state gate | Stage 4 |
| \(\Phi[P]\) was circular through \(H[P]\) | Windows from \(H_{\rm fix}\); explicit \(P\)-functional | Before Stage 5 |
| Strictly positive spectral tails could accumulate over \(O(N_{\rm orb})\) bulk states | Exact-zero compact spectral windows and absolute leakage gates | Before Stage 5 |
| An arbitrary extended reference replaced the model carrier density | Retain the density extracted from \(P\); gate only its image-active fraction | Before Stage 5 |
| Frontier boundary convention was incomplete | Define PBC image functional, zero isolated image term, and separate stationary solves | Before Stage 5 |
| Same-channel wording conflicted with signed density | Use one total signed image-active density with all cross terms | Before Stage 5 |
| Frontier compactness was treated as sufficient for a boundary identity | Require the conjunctive \(\operatorname{IsoOK}\) predicate, including static/excess support and a canonical isolated-space lift | Stage 5 |
| Thermal static density was assumed compact | Remove the exponent claim; separate/gate unmatched thermal backgrounds | Stage 4 validation |
| Static--static did not complete the defect-versus-bulk boundary change | Add unified image functional and fixed-training-boundary reference completion | Stage 5 |
| Stage 6 could double count screening | Replace screened kernels by bare plus induced response term by term | Before Stage 6 |
| Stage 3 was unreachable in one case | Add exhaustive Outcome D routing | Before Stage 2 results |
| Several selection thresholds were qualitative | Freeze practical/noise thresholds and tie rules before each arm | Before each ablation |
| \(C_Q\) scope was unclear | Benchmark-only alignment; one per charge and never per size; no host-table foundation claim | Stage 5 |
| Cached composition integers depended on a changing model | Dedicated frozen constructor plus complete fingerprint/invalidation | Stage 0 |
| Tier-2 and Tier-1 keys omitted target geometry or numerical-regime fields | Key both tiers by target-specific and numerical records; reuse sizes only through a field-wise compatibility predicate | Stage 0 |
| Pristine-density strain rule was absent | Affine fractional-coordinate rule and explicit Gaussian-width derivative | Stage 4 |
| Scope exceeded the evidence in places | Distinguish PBC convergence, charge-increment correction, and boundary-complete total excess | Immediately |
| Neutral-base anchor allowed a charged production reference | Require \(Q_{\rm formal}(S_{\rm ref})=0\); retain nonzero reference only as a synthetic count test | Stage 0 |
| Stage 4 and Stage 5 both appeared to activate image feedback | Stage 4 is forward-only with full response derivatives; Stage 5 activates \(\delta\Phi/\delta P\) and SCF | Before Stage 4 |
| Conceptual bracket split implied a redundant isolated reference solve | Use the direct fixed-PBC formula; solve the cancelling reference term only for optional component reporting | Stage 5 |
| Short-range electrostatic ownership was ambiguous | Evaluate only the registered long-range analytic operator; local short-range physics remains phenomenological | Stage 4, and again at Stage 6 |
| Lowest-SCF-branch selection could create a force cusp | Require a stable unique minimum, competing-solution gap, and pathwise force continuity | Stage 5 |
| A stable-looking SCF root could still have singular implicit response | Guard the constrained energy Hessian in a registered metric, including the scaled coupled Stage-6 block | Stage 5 |
| Positive definiteness alone allowed ill-conditioned polarisation | Add eigenvalue, condition-number, and scaled raw-residual guards | Stage 6 |
| Pristine/defect density registration was not fully covariant | Require deterministic co-translation, co-rotation, wrapping, permutation, and affine-strain covariance | Stage 4 |
| A wrapped torus density was passed conceptually to an isolated kernel, and frontier compactness did not certify static support | Define a canonical component-preserving lift and require \(\operatorname{IsoOK}\) for every boundary-complete output | Before Stage 4 |
| A lift selected from the variational density would add an omitted \(P\)-derivative or a hard SCF branch | Select the branch from a frozen constructor-topology envelope; use a linear, locally \(P\)-fixed image assignment and reject inequivalent branch changes | Before Stage 4 |
| Compactness after defect-minus-bulk subtraction was used to support independently evaluated nonlinear functionals | Require every individual defect, bulk, requested, and reference density to pass; retain matched compactness only as an additional diagnostic | Before Stage 5 |
| An anisotropic dielectric lacked an objective frame and strain law | Store it in a pristine material frame, co-rotate it by polar decomposition, hold material components fixed under strain, and differentiate the kernel map | Before anisotropic use |
| Defect-stress reference was ambiguous | Use the identical-cell defect-minus-pristine-bulk stress difference explicitly | Validation |
| Band double counting lacked a normative equality test | Compare primary and band forms under every active boundary/state/regime | Stage 5 |

None of these corrections requires new training labels. The spectral gauge, loss-path audit,
saved-checkpoint recalibration, and energy-objective repair are required before the completed
Stage-1 result is used to advance the programme. Items affecting Stage 5 or Stage 6 are
blocking only when those stages are reached; the remaining notation, counting, cache,
routing, and claim corrections should be adopted immediately.

---

## Operating notes recorded with the addendum (2026-09-06)

- Running Stage 1.3 jobs on b3 were cancelled on receipt (arm (a) six seeds and arm (b) six
  seeds had finished; arm (c) had started). Nothing from those runs is analysed further
  until the loss-path audit and saved-checkpoint recalibration of Section 8 are done.
- Training waves are packed to use the GPUs efficiently: up to two runs per GPU on GPUs 4–7
  only. (On 2026-09-06 GPU 6 reports an "Unknown Error" from nvidia-smi and is unavailable.)
- 2026-09-06, later: CUDA is down node-wide on b3 (`torch.cuda.is_available()` is False for
  every visible device after the GPU 6 fault; NVML fails; no sudo on the node). Every check
  since -- the loss-path audit, the recalibration of the six saved seeds, the one-epoch
  smoke of the corrected recipe -- ran on b3's CPUs. The packed launcher
  (`queue_packed.sh`, healthy-GPU probe, two runs per GPU on 4-7, OMP cap) and the corrected
  recipe (`stage_v81_run`) are written and committed but unexercised on a GPU; the Stage 1
  retrain waits for a GPU reset or reboot by an administrator.
- The Section 8 pair batches: every training batch is the ordinary shuffled sweep of 8 plus
  2 registered pairs (12 graphs). The pair graphs enter ONLY the shape term -- their loader
  weight is zeroed and the base graphs' columns rescaled so that the force, base and gap
  terms on the 12-graph batch equal their values on the 8 base graphs alone (pinned by
  `test_pair_graphs_enter_no_other_term_and_the_base_terms_are_stage_b_exact`). Without
  this, half the pair draws (equal W_g) would have come from the 16-frame stratum and each
  159-atom charged frame would have entered the force terms some forty times per epoch at
  the two-size upweight. Validation batches carry no pairs and score no shape term; the
  held-out shape diagnostic is `stage1_recalibrate.py`'s centred within-stratum RMS. The
  within-stratum member weight w_i is the loader `weight` column (1.0 on every frame of
  this dataset), not the Stage B OOD w_E, which no term reads under this objective.
- Expected cost: a 12-graph batch carries on average about two 159-atom graphs on the
  per-graph eigensolve path where an 8-graph batch carried 0.14, so the step time of the
  corrected recipe is budgeted at two to three times Stage B's; the packed waves are sized
  for that before any finish time is quoted.
