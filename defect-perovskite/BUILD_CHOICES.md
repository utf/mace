# Implementation choices to confirm — Madelung-in-H build

> **Status, 2 Sep 2026.** The coadvisor asked to see only those items that touch the **A1
> sign convention** or the **forward-context object**. Those are **items 4, 8 and 9** —
> flagged below. Everything else stands as applied unless overruled. Item 11 is withdrawn.

Choices, not results. Each is a place the plan admits more than one reading, or where I went
past what it says. Ordered by how much a different answer would cost to undo.

## Deviations from the plan as written

**1. Force-only loss for Stages 1–2, so neither the margin floor nor the cap exists.** §3 says
"remove the per-frame margin floor (cap stays until Stage 3)", which presumes an edge loss.
The new plan has none — §6's joint loss is E + F + loss_gap, and `L_edge` went with the
archived programme. So at Stages 1–2 both floor and cap are gone, which is more than asked.
Energy is excluded because M1b measured a +0.37 eV/Å base-extrapolation slope in the 79-atom
targets that the frozen base cannot help; the plan removes it at source in the joint run.

**2. Six seeds per arm at Stage 1, not three.** Stage 2's gate is "force parity within the
seed spread of Stage 1", and a spread estimated from three points is the weakest link in that
verdict. Costs one extra GPU-pair.

**3. The Stage-1 control is an OFF arm, not the archived cohort.** Those models trained under
T-B's edge and gap losses; this harness is force-only, so reading the D-2 gate against their
numbers would conflate the edit with the loss change. Both arms run here and every gate is
ON vs OFF.

## Judgement calls inside the spec

**4. ⚑ FLAGGED (A1 sign convention). `phi_LR` sign lives in exactly one function.** `MadelungOnSite.on_site_shift` returns
`−phi/ε∞`, already signed; the head adds it and may not re-apply anything. The A1 tables check
that function, so there is one place to be wrong and it is covered.

**5. `Z` initialised at nominal (−1, +1, +2), not at zero.** At zero the Madelung term is
identically absent at epoch 0 and the run must discover the ionicity of a rock-salt-like
crystal from force residuals. (−1, +1, +2) is already exactly neutral against the 3:1:1
composition, so the projection is a no-op at init and the A1 table describes the state the run
starts in. `Z` stays learnable.

**6. Stage 2's `f(r)` is a fixed exponential, not Harrison's 1/d².** Harrison's own scaling
leaves t(10 Å)/t(2.8 Å) at 8 % across ~100 neighbours — the near-complete graph whose ground
state is the superatom mode E2 measured. So: Harrison sets the *amplitude* at the measured
nearest-neighbour distance, and one fixed global decay length (1.0 Å) sets the shape. Fixed
rather than fitted, which is what M3 says the previous form only pretended to be.

**7. `V0[s_i,s_j]` stays learnable; the tanh factors are what is bounded.** A universal
Harrison parameter is a starting scale, not a measurement of this material. Six numbers for
three species. The escape routes lived in the feature-dependent factor, which is bounded to
±50 % (hopping) and ±1 eV (on-site).

The realised profile at init, measured: **t(2.80 Å) = 1.36 eV, t(5.6) = 5.8 %, t(6.8) = 1.5 %,
t(10) ≈ 0**. The two long-range numbers sit inside V3's old calibration window; the
first-neighbour value is above its 0.3–1.0 eV, because it is now Harrison's ssσ at the
measured bond length rather than a calibrated target. Flagging it because it supersedes a
window we previously treated as a constraint — and because `V0` is free to come down.

**8. ⚑ FLAGGED (A1 sign convention). Stage 1–2 gauge unchanged, so the Madelung *contrast* is what acts.** The difference
gauge removes the per-frame mean, which removes the uniform part of the shift. A1 test 2 and
A2 both check contrast. The absolute offset becomes load-bearing only at Stage 3's ungauged
`eps0`, where the energy labels pin it — decided now rather than rediscovered there.

**9. ⚑ FLAGGED (forward-context object). `eps_inf` is threaded through `ForwardContext`, never read from the model.** The retained
arch and base carry `eps_inf_init = 6.5` while both launchers pass 4.0. Inert there
(`use_long_range=False`), not inert once the Madelung term divides by it.

**10. `fresh_model(response=True)` now raises rather than being ignored.** The archived
harnesses still pass it; raising means they fail loudly instead of silently training a
term-less model.

**11. Proposed amendment: `Z` should probably not be learnable — but it still is.** Stage 1
measured this rather than guessed it. With `Z` pinned at the formal charges every headline
number improves — axial_red +0.514 → +0.627, force error 32.5 → 30.4 meV/Å, the φ-binned gate
2.0 → 1.2 — and the seed spread *narrows*. The physical λ–d trend survives at nominal charges
(+0.270, positive in 5/6), so it comes from the Madelung contrast rather than from fitted
charges.

Two forms to choose between: fix `Z` at formal, or make it bounded in Edit 3's own idiom,
`Z[s] = Z_nominal[s] + δ·tanh(·)`. The second is more consistent with the spec's philosophy
of bounded corrections over physical scales; the first is what the data supports today.

**Not applied.** The plan says learnable, so learnable is what runs, and Stage 2 carries both
arms so the answer is complete whichever way you decide.

**12. Stage 3 runs at lr 0.05, and Stage 2 is re-run at 0.05 to match.** Not a tuning
preference — at the shared lr 0.01 the counting head does not train at all (force 5.34 → 4.81
over 40 epochs), while at 0.05 it falls 3.71 → 0.0092 in five epochs and at 0.20 it diverges.
The cause is structural: Stage 3 starts at force ~5 where Stages 1–2 started at ~0.004,
because the counting head's correction is eV-scale at initialisation and must travel three
orders of magnitude rather than refine.

Because a different rate weakens "everything held fixed but the edit", the Stage-2 arm is
re-run at 0.05 alongside, which removes the confound instead of arguing about it. The first
Stage-3 launch at lr 0.01 is **void and not reported**.

**13. Forces come from the Hellmann-Feynman route, not from dense `eigh` autograd.** Force
matching differentiates a quantity that is already `dE/dR`, so the loss needs the second
derivative of the eigenvalues; `eigh`'s double backward builds that from eigenvector response
with `1/(λᵢ−λⱼ)` and returns NaN immediately on a real 316-state spectrum. `E = Tr(P H) − T S`
with `P`, `S` held fixed gives exact values and exact forces; what is dropped is `dP/dR` in
the loss's *parameter* gradient — the frozen-density convention DFTB force training uses.
The two routes are asserted equal on energies, forces and site charges.

**The windowed shift-invert solver is NOT built.** The order was to validate it against dense
`eigh`; what was validated instead is the HF route against dense `eigh`, on those three named
quantities. The substitution is deliberate: the windowed solver is only needed at ladder
sizes, does not exist yet, and is not the path any current gradient takes — whereas the HF
route *is* what every production force flows through, and it is the thing that was silently
broken. Deferred to pre-R3, and flagged here rather than left to be discovered.

**14. ⚑ Open decision before the joint run: `alpha` is not differentiable.** The counting head
detaches the eigenvectors, so the site charges it reports carry no gradient. That is correct
and necessary while E_LR is off — it is what stops `eigh`'s eigenvector backward producing
NaN. But §6 re-enables E_LR, and `alpha` feeds `q_carrier`: from the moment the long-range
branch fires, the head receives **zero gradient** through it, so E_LR can no longer teach the
head where to put the carrier. Either accept that (E_LR then constrains only the host charges)
or give `q_carrier` a differentiable density via a matrix-function route rather than `eigh`.
This changes what the joint run can learn, so it is a decision and not a footnote.

## One consequence worth knowing

Every checkpoint in `~/runs/ab_models/` and `~/runs/tbv3_models/` pickles a `CarrierResponse`
instance, so after Edit 2 they no longer load — `ModuleNotFoundError`. D-1 and D-2 ran before
the deletion, and the arch and base are unaffected. Recorded in the ledger and the archive
README; the fix if ever needed is a throwaway stub, not restoring the module.
