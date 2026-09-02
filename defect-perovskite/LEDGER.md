# Ledger — closures and open items at the Madelung-in-H direction change

Opened 2 Sep 2026. §9 of the build plan requires these entries **verbatim** in the final
report, so this file is their canonical home. Append, never rewrite.

---

## Closures

**M2 cancelled** (direction change). The A/B ON-arm fit gain is closed **unattributed**.

What that gain was, for the record: axial_red +0.348 → +0.588, no negative cell where OFF had
one at −0.464, rmse_nbhd 47.7 → 43.5, across six seeds per arm. M3 then showed it was not
bought through the hub coupling (|t′| indistinguishable between arms, both far below target;
corr(N_eff, d) flat in both). M2 would have said *where* it was bought. It will not be run, so
the answer is not known and must not be asserted.

**One-manifold programme archived.** Superseded in full: C1, the M3 extensions, the
consistency triangle, the sign A/B observables, `channel_sign` / `s_c`, `μ_c` and
`init_mu_for_gap`, and the soft gauge anchor.

The anchor's retirement has a reason worth keeping: the counting head's
`E_head = F(N) − F(N_neutral)` is **not** invariant under a uniform ε shift, so the energy
labels themselves pin the absolute edge into `eps0`. An anchor would be a second, weaker
statement of something the loss now enforces exactly.

Nothing was in flight at the change (both machines idle, verified). Outputs archived to
`~/runs/archive_one_manifold/` on b3 and locally; the model checkpoints in
`~/runs/tbv3_models/` and `~/runs/ab_models/` stay in place as D-1/D-2 inputs and are
archived programme members regardless — no result of theirs is evidence for the new
architecture.

**Response channel retired**, with its two measured legacies recorded:

* **clause 1 — reach confirmed.** The mechanism does give a compact state the long force
  footprint; electrostatics has the reach a short-ranged tight-binding head cannot.
* **clause 2 — site selection absent.** It did not make the head prefer the hub.

Both motivate the variational move of Edit 1: the same physics, but inside `H` where it can
select a site, rather than in a bolt-on energy readout where it demonstrably could not.
Retirement lands **in the same commit as Edit 1** (`85d195b`) — never both active, which
would double count the carrier's electrostatics.

*Consequence, verified rather than assumed:* every archived checkpoint in `~/runs/ab_models/`
and `~/runs/tbv3_models/` pickles a `CarrierResponse` instance, so on post-`85d195b` code
`torch.load` raises `ModuleNotFoundError: mace.modules.defect_response`. D-1 and D-2 ran
before the deletion and are unaffected; the arch and base checkpoints never had the channel
and still load. A future reader needing one of these should write a throwaway stub, not
restore the module.

**Cancelled with the programme, not to be run:** M2 attribution, C1 post-μ-fix, the F2
closure-ratio update.

---

## Open items

**The λ–d anomaly.** `corr(λ, d) < 0` in 10 of 12 A/B cells (−0.31 to −0.57): the defect level
falls as the vacancy-flanking pair separates, so `Δ_bind` *rises* with d. That is the reverse
of the physical expectation. It is equally present in both arms, so it is not a sign effect.

Carried forward as **expected to be probed by D-2**: if the anomaly localises to the resonant
bin (`d/δ_L < 1`), it is recorded as explained-by-resonance pending Edit 4. §7 makes it a stop
condition — if `corr(level, d)` is still anti-physical **on bound frames** under the counting
head, stop and investigate before R3.

**D-2 has now run, and the resonance explanation is refuted (2 Sep 2026).** Measured on the
frames each model's own spectrum marks bound (`depth/δ_L > 2`), `corr(λ, d_hub)` is negative
in **17 of 18** models, mean −0.422; and **13 of the 18 models have no resonant frame at
all** while still showing it. The anomaly is not a near-degeneracy artefact. It stays open
with its cheapest explanation now eliminated, and §7's stop condition stays live.

Until it is resolved, `Δ_bind` is not to be relied on as a depth.

**Ledger note on ε∞, from the §0 verification.** Nothing ties the screening amplitude `a` to
the static dielectric constant — there is no static-dielectric literal anywhere in the
repository, and `a` is initialised to `1/√ε∞` from `eps_inf_init` alone. But the retained
arch and base checkpoints (`r2_h3_anneal_s5`, `e0_base_s1`) carry **`eps_inf_init = 6.5`**,
not the plan's 4.0, with `freeze_amplitude = False`. It is inert in those models —
`use_long_range = False`, so `a` never acted — and the launchers (`run_arm.sh`,
`run_perovskite.sh`) both default to 4.0, so the 6.5 entered by environment override on a run
whose provenance is not recorded.

Consequence for the build: **Stage 1 must not read `model.eps_inf_init`**, or the Madelung
screen silently picks up 6.5 while E_LR's amplitude uses 4.0. ε∞ is threaded as one explicit
constant through the forward context.

---

## Retained measurements (not superseded)

Test 2 (R_DFT ≈ 0.95); the E0 footprint; the D1 decomposition; M1 and M1b (clean subset =
159-atom, slope −0.134 eV/Å); the F1 and F2 measurements. Per-fold bases remain available as
optional low-priority evaluation nulls.
