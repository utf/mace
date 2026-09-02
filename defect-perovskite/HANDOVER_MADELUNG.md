# Handover — Madelung-in-H build, end of 2 Sep 2026

Supersedes `HANDOVER.md` for everything after the direction change. That file's **§1 hard
rules and §7 traps still apply**; its §3 (what is running) and §5 (decision tree) are stale —
they describe the archived one-manifold programme.

Branch `size-extensivity`, unpushed. HEAD `de43deb`. **Nothing is running on either machine.**

---

## Where the build stands

| stage | state |
|---|---|
| §0 ledger, archive, ε∞ verify | done — `LEDGER.md` |
| forward-context object + cross-path test | done — `mace/modules/defect_context.py` |
| D-1 sensitivity audit | done, **null** as pre-registered |
| D-2 error binning | done, resonance prediction **refuted** |
| Stage 1 — Edit 1 + Edit 2 | done, **all five gates pass** |
| Stage 2 — Edit 3 | done, **superatom gate passes, force parity fails** |
| Stage 3 — Edit 4 | integrated, gradient path validated, **running** |
| Stage 4 — occupation interface | done, tested; SCC hook reserved as an interface only |
| §6 joint run | enumerated in `JOINT_RUN_PLAN.md`, **one decision blocks launch** |

Results: `D_RESULTS.md`, `STAGE1_RESULTS.md`, `STAGE2_RESULTS.md`. Coadvisor note:
`MADELUNG_COADVISOR.md`. Choices awaiting confirmation: `BUILD_CHOICES.md`.

## The finding that decides what happens next

`corr(axial_red, split_fraction) = +0.987` across Stage 2's 12 seeds, strictly bimodal. Where
the pristine spectrum is bands, the head fits forces no better than the base it corrects;
where it is still a superatom, it fits. **The force fit and the superatom have been the same
object.**

Per §8 of the plan a stage gate failure is a report-and-decide point, which is why the run
stopped here rather than proceeding to Stage 3.

The open question put to the coadvisor: does this change the plan, or is Stage 3 exactly the
response to it? Our reading is the latter — one orbital per atom cannot represent the
p-derived valence band this hole lives in, and Edit 4 is built and gated. **What must not
happen is loosening Edit 3's bounds until the fit returns**; that buys back the superatom.

## What Stage 3 cost, and what it found in the code

Dense `eigh` autograd does not survive force matching: the loss needs the second derivative
of the eigenvalues, and `eigh`'s double backward builds it from eigenvector response with
`1/(λᵢ−λⱼ)`. It returned NaN on the first real batch. The production path is now
`head_energy_hf` — `E = Tr(P H) − T S` with `P`, `S` held fixed — which gives exact values
and exact forces and a well-conditioned second derivative.

Validating it against the dense route found two further bugs that inspection had not: the
occupations were carrying gradient (adding a spurious `−μ f(1−f)/T` to `dF/dε`, invisible on
a spectrum symmetric about μ = 0 — which is what the original test used), and the two spin
density matrices were averaged where they must be summed, halving the monopole.

**The windowed shift-invert solver is not built** — deferred to pre-R3 and recorded as a
deviation in `BUILD_CHOICES.md`. It is a ladder-size concern; nothing currently runs through
it.

Stage 3 also needs **lr 0.05**: at the shared 0.01 it does not train at all. The Stage-2 arm
was re-run at 0.05 alongside so the comparison stays matched.

## Live process lessons from today

* **Sync b3 in the same breath as any commit touching `mace/` or the harness.** Three
  machine-drift incidents today, one of which had the frozen-Z arm overwriting the Stage-1 ON
  checkpoints because the tag fix was committed but not rsynced. Use `--delete` or a deletion
  never propagates.
* **Archived checkpoints no longer load.** Everything in `~/runs/ab_models/` and
  `~/runs/tbv3_models/` pickles a `CarrierResponse`, which Edit 2 deleted. Write a throwaway
  stub if you must open one; do not restore the module.
* **Anything that changes what is trained belongs in the save filename.** It did not, once.
* The A2 test had to move from `eps` to a new `eps_raw` internal: the difference gauge
  subtracts a per-cell mean, so gauged `eps` differs between two cells by a constant even
  when every learned value is bit-identical.
