# Charge-Aware Defect MLIP: state of the work

A standalone summary of the challenge, the design, what has been built, what the benchmarks
say, and the one unresolved problem. Written to support planning the next phase.

---

## 1. The challenge

Point-defect thermodynamics needs energies that depend on the **charge and spin state** of
the defect, not just on atomic positions. A conventional MLIP is a function of geometry
alone, so it cannot distinguish a neutral vacancy from a singly-ionised one at the same
nuclear coordinates, and it cannot produce a charge transition level at all.

The quantities that matter are:

- formation energies comparable **across charge states** and **across supercell sizes**;
- charge transition levels;
- configuration-coordinate diagrams, which need charge-dependent **forces**, not just
  charge-dependent energies;
- orderings of competing metastable configurations at 1–2 meV/atom;
- migration barriers, where the charge state changes the barrier;
- ideally all of it evaluated **directly in the isolated-defect limit**, rather than in a
  periodic cell and corrected afterwards.

An MLIP is required rather than direct DFT because the geometry search is combinatorial in
(defect × charge state × symmetry-breaking distortion) and cannot be truncated:
symmetry-breaking searches routinely find the global minimum in a small minority of trial
distortions. Multiplying that by charge states and hosts puts the DFT cost out of reach.

Two accuracy targets, not one. Fixed-geometry, fixed-cell charge-state differences —
vertical transitions, configuration-coordinate offsets, spin-state splittings — are
algebraically free of base-model error and are targeted at **0.02 eV**. Relaxed-geometry
quantities, including thermodynamic transition levels, evaluate the base potential at two
*different* minima and inherit its error at first order; they are targeted at **0.05 eV**.
Quoting a single figure for both is not supportable by the error budget.

---

## 2. The approach

Four design commitments, each chosen to make a specific failure impossible rather than
merely unlikely.

### 2.1 Carriers, not net charge

The conditioning input is four non-negative integers, one per (carrier type, spin channel):

```
n = (n_e^maj, n_e^min, n_h^maj, n_h^min)

q   = (n_h^maj + n_h^min) − (n_e^maj + n_e^min)
M_s = (n_e^maj − n_h^maj) − (n_e^min − n_h^min)
```

Conditioning on `(q, M_s)` instead would give a parity checkerboard — at fixed `M_s` only
every other `q` is valid — so continuous conditioning would interpolate across an
unphysical half-lattice. The carrier counters live on a full integer lattice.

This also makes states distinguishable that `(q, M_s)` collapses: a closed-shell neutral
singlet `(0,0,0,0)` and an open-shell neutral singlet `(1,0,1,0)` are different inputs, and
a compensated donor–acceptor cell at `q = 0` is not an alias of the pristine reference.

**Canonicalisation is mandatory.** Time reversal maps
`(n_e^maj, n_e^min, n_h^maj, n_h^min) → (n_e^min, n_e^maj, n_h^min, n_h^maj)`. Both
labellings describe the same physical state and must get the same energy. Requiring
`M_s ≥ 0` fixes the gauge only when `M_s ≠ 0`; at `M_s = 0` a lexicographic tie-break is
needed. This is applied at data loading and at every inference entry point, so the map from
physical state to model input is single-valued and time-reversal invariance holds for any
parameters rather than being learned.

**The counters are an assignment, not a measurement.** A given `(q, M_s)` is consistent
with infinitely many counter vectors differing by added electron–hole pairs. Deciding
between `(0,0,0,0)` and `(1,0,1,0)` for a neutral singlet requires inspecting the converged
orbitals. The labelling rule is to assign the *minimal* counter vector consistent with the
SCF occupation pattern, and to record the frontier occupations that justified it. A
labelling not reproducible from stored metadata is a data-integrity failure.

### 2.2 A residual on a charge-blind base

```
E_total(R, n) = E_base(R) + ΔE_SR(R, n) + E_LR(R, n)
```

The base branch sees geometry alone. Because the correction vanishes identically at
`n = 0` — structurally, via a `Σ_c n_c` prefactor, not approximately — **fixed-geometry
charge-state differences are free of base-model error**. That exactness does not extend to
relaxed-geometry differences, which evaluate `E_base` at two different minima.

### 2.3 Intensive in carriers, not extensive in atoms

For each channel, two per-atom readouts and a normalised pool:

```
ℓ_i^c = MLP_ℓ(h_i, n)              per-atom logit
α_i^c = softmax_i(ℓ_i^c)           Σ_i α_i^c = 1, per cell, per channel
u_i^c = MLP_u(h_i, n)              per-atom energy readout

ΔE_SR = Σ_c n_c Σ_i α_i^c u_i^c
```

`α` is the carrier's occupation distribution over atoms and `Σ_i α_i u_i` its expectation
energy. Because the weights are normalised, the pooled quantity is **intensive**: the
correction scales with carrier count, not atom count.

On pristine cells this is exact for any structure — the number of formula units cancels
identically. On defective cells the size error is `≈ N·r·(u_bulk − u_defect)` with
`r = e^{ℓ_bulk − ℓ_defect}`, so the logit gap can be read straight off a trained model to
bound the size error with no DFT and no second cell. **The suppression is learned, so it
must be measured, not assumed.**

### 2.4 Explicit long-range electrostatics

A latent Ewald term carries the polarisation response, image interaction and inter-defect
coupling that no finite receptive field can represent. The latent charge is built in three
pieces:

```
q_i = q_i^host + q_i^pol + q_i^carrier
```

with `q^host` geometry-only, `q^pol` the polarisation response, and `q^carrier` reusing the
attention weights for its shape and a single amplitude `a = 1/√ε_∞` for its magnitude — so
`q^carrier` alone carries the monopole and `Σ_i q_i = a·q` exactly.

Because the defect charge distribution is an explicit model output, the finite-size
operation at inference becomes a **boundary-condition swap on that distribution** rather
than an energy correction applied to labels.

---

## 3. Implementation, at the level that matters

### 3.1 What the long-range term should and should not contain

This took several iterations and is the most important implementation lesson. The rule that
emerged:

> `E_LR` carries only what `ΔE_SR` structurally cannot — the monopole self-interaction and
> the interaction between *separated* carriers. Anything local to the carrier and inside the
> receptive field belongs in the short-range readout.

Three terms were removed on that principle, each after direct measurement:

**The polarisation channel had no locality constraint.** `q^pol` was mean-subtracted over
the whole cell, which makes it exactly neutral — but **neutrality is not locality**. Far
from the defect, node features are the bulk values for the species, so the readout returns
a fixed per-species number and a global mean removes only the composition-weighted average.
Every atom in the crystal was left carrying a constant charge (measured: Cs −0.067,
Pb +0.109, Cl −0.015 e, flat from 3 Å to beyond 10 Å). Its Madelung energy and its cross
term with `q^host` were **97 % of the size growth**. This is structural: a readout of local
features and a global counter is necessarily constant on bulk atoms of a given environment,
so no amount of training localises it.

**The carrier–host cross term is structurally mis-signed.** A classical point-charge
potential is deepest at **cation** sites, so it drags a hole onto the cation sublattice on
principle. Measured under hand-set attention it favoured the wrong sublattice by 1.25 eV
against 0.18 eV of short-range binding difference. No rescaling of `q^host` or `a` changes a
*sign*, so this could not be repaired by reparameterisation — only removed. What it
represents, the carrier's interaction with the host's short-ranged Madelung field, belongs
in the learned per-site energy.

**The carrier self-energy is self-interaction error.** `E_LR` for one carrier channel with
itself splits as `E(L) = E_internal + E_image(L)`. A single hole has no Hartree
self-repulsion, so the entire *in-cell* part is spurious; only the image term is real,
because that artefact genuinely exists in the periodic DFT labels. Measured at training
cell size the in-cell part pays **+0.104 eV to delocalise the carrier**. The fix defines

```
E_LR = E_periodic[Q_total] − Σ_c E_isolated[Q^c]
```

per channel, so cross-channel terms (electron–hole interaction) survive while the
within-channel self-energy cancels. The `i = j` cancellation is exact regardless of
minimum-image convention.

### 3.2 The size-extensivity constraint

Because the logit gap that suppresses size drift is learned rather than guaranteed, a hinge
was added in **log-ratio space**: penalise `max(0, x − x*)²` where `x = ln B − ln A` is the
log-ratio of a defect-weighted pool to a bulk reference, and `x*` is set from a tolerance on
the allowed energy drift.

Two things about the formulation are load-bearing:

- **Log-ratio, not energy space.** The energy-space gradient carries a factor `f(1−f)`
  which is ≈0 exactly where the term is most needed, and its only reachable optimum was to
  destroy defect contrast. In log-ratio space the gradient is linear in the violation.
- **The bulk reference must be a plain, undetached arithmetic mean of the logits.** Two
  distinct properties are required — no uniform-inflation descent direction, *and* pointwise
  zero gradient on a uniform state — and trimmed or detached variants give the first without
  the second, which means the term can manufacture structure from noise.

An exemption exists for genuinely delocalised carriers (if the pooled contrast is already
within tolerance, dilution cannot move the energy further). **That exemption is a trap**:
flat attention drives contrast to zero, which triggers the exemption, which switches the
hinge off, so the constraint degenerates exactly at the state it exists to prevent. A guard
now refuses the exemption for a channel whose attention spans a fixed fraction of the cell.

### 3.3 Attention seeding

A novelty-based bias on the logits gives the attention a starting preference for atoms in
unusual environments, annealed away over training so the shipped model carries no prior.
The retirement schedule is gated on a readiness measure. **Which measure is used turns out
to matter more than almost anything else in the model** — see §5.3.

### 3.4 Diagnostics that proved essential

Several failures were only visible through purpose-built measurements, and these are now
permanent:

- **Per-species attention mass**, not just the participation ratio. A participation ratio of
  16 in a 79-atom CsPbCl₃ cell is *both* the Cs count and the Pb count — one is a total
  failure, the other is the right species with the wrong site. The scalar cannot separate
  them.
- **Site-resolved logit spread**, `mean over species of the within-species standard
  deviation`. Species-blind by construction, so a purely elemental ordering contributes
  nothing however large. Logged permanently alongside the raw gap, because **the pair is the
  diagnostic**: a raw gap of 24 with a site spread near zero is a species classifier, not a
  found defect.
- **Term-by-term decomposition of `E_LR`** by the polarisation identity
  `2B(a,b) = E(a+b) − E(a) − E(b)`, which localises size growth to a specific term instead
  of leaving it attributed to the branch as a whole.

---

## 4. Benchmark datasets

Two systems, chosen for complementary reasons. Neither alone is sufficient.

### 4.1 4H-SiC divacancy — the primary development system

Neutral divacancy (V_Si–V_C), spin triplet, from a published PBEsol dataset on optical line
shapes of colour centres. The physics learned is the **first excited state**: an intra-defect
promotion within the minority spin channel.

- counters: ground `(0,0,0,0)`, excited `(1,0,1,0)` after canonicalisation
- **net charge zero on every frame** (one electron plus one hole)
- vertical excitation ⟨ΔE⟩ = 0.943 eV, σ = 0.124 eV; published ZPL 0.97 eV
- ~800 frames including 340 ground/excited **pairs at identical geometry**, plus unpaired
  frames of each state and pristine supercells; cells 286–398 atoms

Two properties make it valuable:

**It has paired frames.** Per-atom delta-force supervision resolves *which atoms* carry the
carrier — the labels say directly where the correction must act.

**It is harder than the original task.** The published data encoded the electronic state in
the chemical *species* of the defect neighbours, because the model it was built for had no
other channel for it. Mapped back to real species, the trunk sees identical input for a
ground/excited pair — same species, same positions, same cell — so the entire 0.94 eV must
come through the correction branch.

Its limitation: with `q = 0` on every frame there is **no monopole**, so the screening
amplitude is unidentifiable and the long-range branch is essentially unconstrained.

### 4.2 CsPbCl₃ chloride vacancy — the long-range test system

300 K MD snapshots of V_Cl in two charge states, PBE, from a published study of dynamic
vacancy levels. Cells of 79/80 atoms (plus a few 159-atom), 2560 train / 317 valid frames.

- pristine `(0,0,0,0)`, mult 1, q 0
- V_Cl⁰ `(0,0,0,0)`, doubled reference spin 1, mult 2, q 0 — **the reference for this
  composition**
- V_Cl⁺ `(0,0,1,0)`, mult 1, q +1

This is the **first dataset with `q ≠ 0`**, which is what makes the long-range branch
identifiable at all. It exposed three distinct long-range bugs decisively.

Two things about it are worth carrying forward:

**The reference state had to move.** `q = 0` forces `n₀+n₁ = n₂+n₃`, which makes `M_s` even
and multiplicity odd — so a **neutral doublet is inexpressible at `q = 0` against a
closed-shell reference**. The reference spin therefore becomes composition-dependent. The
check that catches a wrong labelling here is `counter charge == absolute cell charge`; the
earlier, wrong labelling was fully self-consistent in *spin* bookkeeping and passed every
multiplicity assertion.

**It has no paired frames.** The two charge-state directories share only duplicated pristine
copies; the V_Cl geometries are disjoint, because the source study trained separate models
per charge state. So `ΔE_SR = Σ_i α_i u_i` is fitted against a **scalar** with `u` free, and
the α/u degeneracy is essentially unbroken. This is the root of the open problem in §5.

**Consequence for interpretation:** this dataset can *expose* long-range bugs — it did so
three times — but it **cannot validate** the branch, because nothing in it pins where the
carrier sits.

---

## 5. Findings

### 5.1 Size extensivity of the short-range correction: solved

The original failure was that `α = softmax(ℓ)` normalises over every atom in the cell, so
bulk sites accumulate weight proportional to `N` and eventually outvote any fixed logit gap.
The correction stopped being a carrier bound to the defect and became a bulk average — a
drift *logarithmic* in `N`, so not an image interaction. On 4H-SiC this moved `E_ZPL` from
0.903 eV at 256 atoms to 0.715 eV at 2400 atoms, still falling.

Attention on the defect shell against cell size, 70 → 4700 atoms (`drop` is attention lost
across the ladder; 0 is the target):

| model | gap / drop, channel 1 | channel 2 | channel 3 |
|---|---|---|---|
| small data | 2.25 / +0.47 | 2.39 / +0.48 | 5.23 / +0.69 |
| full data | 8.68 / +0.10 | 9.91 / +0.00 | 11.34 / +0.01 |
| + long range | 6.42 / +0.48 | 8.02 / +0.15 | 11.33 / +0.01 |
| **+ size hinge** | **19.64 / +0.00** | **18.60 / +0.00** | **21.96 / +0.00** |

Scale did most of the work; the hinge closed the remainder. Measured consequence on
unrelaxed cells of increasing size:

```
ΔE_SR   2.09170 → 2.09170     −0.0 meV     (254 → 3454 atoms)
```

The short-range correction is exactly size-independent, and the base branch is exactly
size-extensive (0.000 meV/atom at every cell size tested, on both systems).

### 5.2 Long-range extensivity: solved, after three separate faults

On CsPbCl₃ the long-range branch initially produced an optical level growing **linearly in
cell size** to −100 eV. Decomposition localised 97 % of it to the polarisation channel and
its cross term with the host charge. Removing the three terms identified in §3.1 leaves a
branch whose size dependence is a genuine `1/L` image tail.

Best current model, isotropic size ladder with cell **shape held fixed**:

```
    N       L (Å)    eps_opt        fit: E = E_inf + k/L
  640       28.62    −4.1284        k     = −8.61 eV·Å
 2160       42.94    −4.0319        E_inf = −3.8285 eV
 5120       57.25    −3.9770        max residual 2.9 meV
```

Monotone, and a clean `1/L` fit to 2.9 meV over a doubling of `L`. Independent verification:
the carrier self-energy alone fits `E_inf + k/L` with `k = −4.58 eV·Å` and 0.5 meV residual
— same sign and order, and larger for the full level as expected since it includes
relaxation.

The model is **not converged** at 5120 atoms: still 148 meV from the extrapolated limit. The
fit gives a defensible infinite-size value where a naive largest-cell reading would not.

A method note that cost real time: a size ladder must hold cell **shape** fixed. Mixing
2×2×2 with 3×2×2 and 3×3×2 changes the Madelung constant between points, which produced a
−16 meV non-monotonicity that looked like noise and was actually a shape artefact. Only
whole multiples of the primitive give a fittable curve.

### 5.3 Attention localisation: NOT solved — the key open problem

On CsPbCl₃ the correct answer is known independently and is chemically unambiguous. The VBM
is Cl 3p with Pb 6s antibonding character, so a hole belongs on Cl/Pb and **never on Cs,
which contributes no frontier states**. A short-range-only model finds this unaided: the
hole sits on exactly the two under-coordinated Pb that lost their bridging Cl, at 2.6 and
2.9 Å, splitting the weight ≈55/45, identical across a 2.25× cell-size range, and correctly
*failing* to localise on a pristine cell.

**Across seeds, roughly half of runs do not find it.** The failures put the carrier on a
whole sublattice — a hole uniform over 47 Cl sites in a 79-atom cell **is a valence band
state**, not a bound defect level.

Three seed-anneal schedules were tested, four seeds each, short-range only:

| readiness measure | fully correct | failure modes |
|---|---|---|
| raw logit gap | 2 of 4 | smeared across two species |
| site-resolved spread | 3 of 4 | one wrong sublattice |
| site-resolved + bounded decay rate | 2 of 4 | one wrong species; one **right species, wrong site** |

The middle row's failure looked like a clean mechanism: that schedule holds the seed at full
strength and then releases it as a step, and the seed it lost ran away during exactly that
release. A controlled test confirmed the schedule was responsible — same seed, same data, no
long-range branch, participation **1.96 with one schedule and 46.89 with the other**.

**But bounding the release rate did not fix it.** The bound held the seed 3–5× longer
through precisely the critical window, and the attention ran away anyway. That falsifies the
"abrupt release" mechanism: the withdrawal *rate* is not what decides the outcome.

Three schedules, all landing at roughly 2 in 4, with the one intervention that directly
targeted the proposed mechanism changing its proximate cause without changing the outcome.
The conclusion is that **the schedule is not the mechanism.**

### 5.4 The long-range branch does not cause the localisation failure

Two findings separate the two problems cleanly.

**Holding the branch out until the attention has settled works.** Training as a short-range
model for 30 epochs and then switching the long-range branch on for 30 more: every seed that
entered the switch-on with correct attention **kept it**, unchanged to three decimals across
30 epochs with the branch fully live. The branch does not destroy a correct answer.

**The failures happen before the branch exists.** In the seed that failed under that
protocol, the runaway occurred at epochs 12–13 — seventeen epochs before any long-range term
was active. The same seed fails the same way with the branch entirely absent.

Best model from that protocol: **3.1 meV/atom and 11.5 meV/Å** against a short-range
reference's 3.1 / 12.0 — matching on energy, better on forces — with attention
indistinguishable from the reference and every physics gate passing (`Σ_i q_i = a·q` exact;
carrier sum-of-squares constant across a 3.4× size range where every failed model decayed as
`1/N`; `1/L` image scaling confirmed).

### 5.5 Aggregate metrics do not see the failure

This is the most operationally important finding. In one run the seed with **wrong physics
had the best forces of its group** (3.4 meV/atom, 11.4 meV/Å) — better than the short-range
reference — while placing a hole uniformly across the Cl sublattice, which is chemically
impossible for this material.

A model can satisfy every constraint, beat the reference on every aggregate number a
training log prints, and be physically wrong. **Acceptance must gate on species-resolved
attention, not on RMSE.** This has now caught the same class of error four times.

### 5.6 Methodological findings worth keeping

- **`PYTHONPATH` does not pin a code revision** if the launch directory is itself a
  checkout: Python puts the working directory first on `sys.path`. Every attempt to compare
  two revisions this way silently ran the working tree. Launchers must *verify* which module
  they resolved, not assume.
- **A zero-initialised readout on a quadratic term can never train.** The host-charge
  readout sat exactly on a stationary point — the term is quadratic, so at zero the gradient
  is zero too — and the component was never once exercised across many runs. `|q^host|` was
  measured at exactly 0.
- **Runs are not bit-reproducible at fixed seed** on this hardware; trajectories in the
  sensitive regime diverge from rounding-level differences. Outcome counts of "2 of 4" are
  coarser than they look.
- **A gap magnitude cannot distinguish a species gap from a site gap.** This misled the work
  twice: once reading a large gap as evidence of localisation, once gating a schedule on it.

---

## 6. The open challenge, stated precisely

**The model does not reliably localise the carrier on the defect. Roughly half of seeds
place it on a whole sublattice instead — a band-like state rather than a bound defect
level.** Everything else in the design now works: size extensivity is exact in the
short-range branch, the long-range branch produces a correct `1/L` tail, and the branch does
not disturb attention that has already settled correctly.

The diagnosis, as far as the evidence supports it:

`ΔE_SR = Σ_i α_i u_i` is **one scalar equation in two unrelated fields**. Where the training
data has no paired frames, nothing determines `α` and `u` separately — any `α` can be paired
with a compensating `u`. Direct measurement confirms the degeneracy is real and not benign:
in every model that gets the physics *right*, the Boltzmann distribution of the learned site
energies puts **10³–10⁴ times too little weight on the correct atoms**, and the reference
model's own site energies would target the wrong sublattice entirely. So `u` is not a
site-energy field defined everywhere — **it is only constrained where `α` already looks.**

What that rules out:

- **Schedule tuning.** Three variants, all ≈2 of 4; the mechanism-targeted intervention
  falsified.
- **A consistency penalty pulling `α` toward `softmax(−βu)`.** Measured directly, this would
  drive the *correct* models to the wrong answer, because `u` is anti-informative away from
  `α`'s support.
- **Restart-on-divergence.** It concedes the objective admits a wrong answer, and it does
  not transfer: on a system where the correct answer is unknown there is no basis for
  choosing which restart to keep.

What the evidence points to instead: **the localisation must be pinned by data or by
construction, not by optimisation dynamics.** The most direct lever available is per-atom
delta-force supervision, which the 4H-SiC dataset has and the perovskite dataset does not —
paired frames at identical geometry make the forces say *where* the correction acts. Whether
that is sufficient is untested, and it is the obvious next experiment.

A secondary structural option, untested: constrain `α` by construction rather than by loss —
for example gating it on a quantity that is zero in bulk-like environments by definition,
so a band-like solution is not representable rather than merely disfavoured.

---

## 7. Status summary

| component | status |
|---|---|
| carrier counters, canonicalisation, reference conventions | working; charge/spin consistency checks in place |
| base branch size extensivity | exact (0.000 meV/atom at every size tested) |
| short-range correction size extensivity | solved; logit gaps ≈19–22, drift −0.0 meV over 254 → 3454 atoms |
| long-range branch, term content | three spurious terms identified and removed |
| long-range size behaviour | `1/L` image tail, fit residual 2.9 meV; not yet converged at 5120 atoms |
| long-range branch vs. attention | branch does not disturb settled correct attention |
| accuracy | 3.1 meV/atom, 11.5 meV/Å — matches short-range reference on energy, better on forces |
| **carrier localisation across seeds** | **≈2 of 4; the open problem** |
| screening constant `ε_∞` | interim value 4.0; needs DFPT before `1/a²` is quoted as physical |
| band-edge reference for transition levels | fitted gauge, not DFT; not yet comparable to published levels |
| site-averaging over inequivalent defect sites | not done; the perovskite MD snapshots have 48 inequivalent Cl sites and a single site is one draw from a distribution |
| seed replication | ≥3 seeds required for any quoted number; not all results yet meet this |
