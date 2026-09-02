# §6 joint run — what still has to be built, and the one decision that blocks it

Enumerated before the launch hour rather than during it. `stage_run.py` trains **head-only**
on a **force-only** loss against a **frozen** base; §6 needs none of those three.

---

## The decision that blocks launch

**`alpha` carries no gradient, and E_LR is about to depend on it.**

The counting head detaches the eigenvectors — necessary, and the reason Stage 3 runs at all:
`eigh`'s eigenvector backward has `1/(λᵢ−λⱼ)` factors and a 316-state spectrum returns NaN
immediately. Energies and forces are unaffected (Hellmann-Feynman needs eigenvalues only).

But §6 re-enables E_LR, and `alpha` is what shapes `q_carrier`. From the moment the
long-range branch fires, **the head receives zero gradient through it** — E_LR can constrain
the host charges and the amplitude, but it can no longer teach the head *where* to put the
carrier. Two ways out:

1. **Accept it.** E_LR becomes a host-only constraint during the joint run. Simple, honest,
   and probably adequate — the head is already supervised by forces, which is where the
   placement information has always come from.
2. **Give `q_carrier` a differentiable density** via a matrix-function route (Chebyshev or
   Fermi-operator expansion of `P = f(H)`), which is gauge-invariant and has no eigenvector
   backward. Real work, and it is the same machinery the windowed solver would need.

Recommendation: (1) for the joint run, with (2) scoped alongside the windowed solver
pre-R3. Either way it must be chosen, because it changes what the run can learn.

---

## What has to be built

| item | state |
|---|---|
| `counting_head`, `counting_t_el`, `madelung_*` in `arg_parser` + `model_script_utils` | **missing** — the production trainer cannot build these models |
| base unfrozen, from scratch (not from `e0_base_s1`) | `stage_run` freezes; production trainer does this natively |
| loss = E + F + `loss_gap` | energy returns; `loss_gap` exists in `stage_run` only |
| two-size upweight, realised share logged | exists from the archived programme; re-verify it still fires |
| E_LR staged re-enable (`lr_start_epoch`) | exists and is tested again since the fixture repair |
| §7 gate harness | see below |

**Extend `stage_run` or use the production trainer?** The production trainer, because §6 needs
a joint base and the whole point is that the base learns charged geometries — `stage_run`'s
frozen-base design is what M1b said to remove at source. The cost is the arg-parser plumbing
above, which is mechanical.

## §7's gates, and what already exists for them

* **Pristine ensemble gap within 0.1 eV of E_gap** — `pristine_frontier_gap` is now persisted
  per seed by `stage_run`; port the same computation.
* **Per-frame boundness from the model's own spectrum** (frontier level, distance `d` to the
  nearest delocalised state, pristine spacing `δ_L`, `bound = d > δ_L`) — `d2_bins.py` already
  computes all four.
* **Participation and size-invariance on bound frames only** — the conditioning is the new
  part; resonant frames are expected to dilute and must be reported separately, never failed
  on.
* **Dilution ratio ≤ 1.3 on the 17 two-size frames** — `mace.data.dilution` has
  `retained_mass`, `interface_mass`, `tile_with_pristine`; `tb_v3.py` shows the call pattern.
* **`corr(dE_head, d)` on the 159-atom subset against −0.134 eV/Å** — `s3_dehead_trend.py`,
  written for Stage 3, takes any model list.
* **λ–d stop condition** — if `corr(level, d)` is anti-physical on bound frames under the
  counting head, stop before R3. The ledger item is closed on the understanding that this
  check still runs.

## Configuration of record

8 seeds, 50-epoch screen, survivors to full length. No hinges, no dilution penalties, no
floors, no caps, no anneal (bounded Harrison-scale init replaces it; anneal held in reserve).
Two-size upweight kept, realised share logged as a metric not a constraint. In-loop reach
assertion. `ForwardContext` everywhere.

**Learning rate is an open question at joint scale.** Stage 3 needed 0.05 where Stages 1–2 ran
at 0.01, and at 0.05 one Stage-3 seed in six still failed to leave its initial plateau
(force 3.34 against the others' 0.0027). With the base unfrozen the loss landscape changes
again, so the screen should log the epoch-0 and epoch-5 losses per seed and treat a seed that
has not moved by epoch 5 as failed-to-start rather than as a data point.
