# Handover — MACEDefect carrier localisation, 2026-09-01

Branch `size-extensivity`, worktree
`/home/alex/src/mace/.claude/worktrees/size-extensivity`. HEAD `2dcdeaf`.

---

## 1. Hard rules (never relax without the user saying so)

1. **No defect labels in the model, loss, or inputs.** The vacancy assignment is for
   evaluation metrics and diagnostic probes only. `clamp_mask` and `probe_loss_mask` are
   the two places it touches training code and production configs must refuse both.
2. **Never rank or select configurations by total RMSE.** RMSE is a constraint, never a
   selector.
3. **Equal-cardinality controls accompany every clamp comparison.** Save per-atom
   predictions on every probe.
4. On b3, **at most 4 GPUs in use at once**.

---

## 2. What is established (do not re-litigate)

| finding | evidence |
|---|---|
| The hole is **bound**, not band-like | Test 2: R_DFT = 0.95, CI [0.66, 1.34], 13/17 matched frames; null magnitude 5–7% of signal |
| Its force footprint reaches ~6 Å, two thirds of the squared residual **off** the hub | E0 |
| The axial signal is a **missing-neighbour** fingerprint, not a carrier locator | §1 demotion; every candidate state couples to the hub bonds |
| The axial force is carried by **hopping**, not an on-site leak | D1: leak_frac 0.007–0.022 against a predicted >0.8, residual exactly 0 |
| **No compact state** in the current head reproduces the footprint | R1 on the corrected graph: every clamp axial_red ≈ 0, `free` ≈ 0.8 |
| The head has **no site preference** | R1: hub2 beats neither lig2 nor rand2 by more than the seed spread, in H2 and H3 |
| R2's original 0/20 was **void** | All 20 seeds trained on a 5 Å graph where the hub pair has an edge in 0/40 frames |

**Uniqueness vs preference.** A delocalised solution *exists* that fits the axial residual
(r = 0.995 as trained). Whether a bound solution fits *better* was the R1 preference test,
and the answer is: neither is preferred. Keep that wording distinct in any write-up.

---

## 3. What is running right now

| job | machine | state | ETA |
|---|---|---|---|
| **Arm 1b** — from scratch, joint base+head, no DCL, two-size upweight, anneal, 8 seeds, 50 epochs, H3 | b3, 4 GPUs × 2 | epoch ~1/50 at 14:30 | ~19:00 |
| **Step 2 local** — response channel on, masks `hub2` + `lig2` (the clause-2 comparison) | local A4000 | started 14:37 | ~2 h |
| **Step 2 b3 controls** — `rand2_0/1/2`, `nbhd12`, `free` | b3, queued | waits for arm 1b to clear | starts ~19:00 |

Logs: `~/runs/r2p_h3_arm1b_s*.log`, `~/runs/step2_local.log`, `~/runs/step2_b3.log`.
Results land as JSON in `~/runs/step2_*.json` and `~/runs/r1m_*.json`.

**Predictions recorded before results (per the plan's discipline):**
- Arm 1b: **0/8, delocalised, matching R1 `free`**. Anything else → stop and re-read R1.
- Step 2: **clause 1 passes, clause 2 is the real unknown.**

---

## 4. Step 2 gate — both clauses required

1. `axial_red(hub2)` rises from ≈ 0 to **≥ 0.4**, with `rmse_nbhd(hub2)` falling toward the
   `free` value. The mechanism gives compact states the footprint.
2. `hub2` beats `lig2` and `rand2` **by more than the seed spread**. A physical kernel should
   make hub and ligand distinguishable through the *shape* of the response.

**If clause 2 fails, read the fine structure before concluding indifference.** A
cage-centred state also produces an axial hub force: each hub Pb sits inside five ligand
charges instead of six, so the net field is axial by the same asymmetry that made the
hopping fingerprint non-specific. Both states reproduce the *coarse* footprint. The harness
therefore reports per clamp: `rmse_lig_shell`, `rmse_cs_shell` (separately), and
`slope_vs_d` (predicted axial correction against d(Pb–Pb)). hub2 closer on those = a
data-limited preference; hub2 no better = genuine indifference.

**Then run the SCF diagnostic** (turns an a priori choice into a measurement): evaluate the
trained `E_resp` at fixed α under `hub2` versus `cage10` on the same frames. If
`E_resp(hub2) < E_resp(cage10)` the induced polarisation favours the hub and feeding V into
H would help; the reverse means it would drive the carrier onto the cage and must stay out.
Report both numbers. Not yet implemented.

---

## 5. Decision tree from here

- Step 2 passes both clauses → **step 3**: from scratch + response channel + DCL
  (`w_max = 4 × L_F`, ramp 0→w_max over epochs 0–10) + upweight + anneal, 8 seeds; control
  = DCL without the channel, 4 seeds. Gate: N_eff ≤ 8, null ratio ≤ 0.15, Δ_bind ≫ 20 meV.
  Advance at ≥ 6/8 → R3 (full length, ladder, LR enabled, SiC).
- Clause 1 only → run step 3 anyway; if it lands compact with an ambiguous site, add the
  rigid on-site term as a bindability restriction.
- **Clause 1 fails** (`hub2` stays ≈ 0) → the footprint is not electrostatic in character →
  s+p Slater–Koster basis next, tested on the same clamps. **Do not run arm 1 with the DCL
  in this case.**
- Arm 1b ≠ 0/8 → unexpected; re-read R1 before anything else.
- Any in-loop reach assertion failure → **stop**; nothing downstream is interpretable.

---

## 6. Built and committed, ready to use

| piece | file | state |
|---|---|---|
| Carrier-field response channel | `mace/modules/defect_response.py` | wired into `MACEDefect` + extractor, 8 tests |
| Δ_bind (replaces λ₂−λ₁) | `mace/modules/defect_bind.py` | built + 4 tests, **not yet wired into the eval loop** |
| Dilution tiling + interface diagnostic | `mace/data/dilution.py` | built + 10 tests, **loss not wired** (step 3) |
| Two-size upweight | `mace/data/two_size.py` | wired, verified live: 9.61× → 25.0% share |
| In-loop reach assertion | `mace/modules/defect_reach.py` | wired, aborts before step 1 |
| R1 clamp harness | `defect-perovskite/r1_matrix.py` | `--response` flag, fine-structure metrics |
| Arm launcher | `defect-perovskite/run_r2prime.sh` | arms 1/1b/2 |
| Step 2 launcher | `defect-perovskite/run_step2.sh` | |

**Still to build for step 3:** DCL loss wired into the training step (tiled forward, trunk
**detached**, ramp, retained/interface logging on all arms); `w_max` from arm 1b's epoch-10
charged per-frame force loss (compute **post hoc** from the checkpoint — no loss-logging
change needed); Δ_bind hooked into the per-epoch eval; four DCL flags through all four
layers.

---

## 7. Traps that have actually bitten (read before debugging)

1. **A guard that measures a reimplementation passes while the real path is broken.** Four
   instances. The worst: `graph_cutoff` was applied only to the validation *fallback* path,
   so all 20 R2 seeds trained at 5 Å with the hub pair uncoupled. The reach assertion now
   runs on a batch from the training DataLoader itself.
2. **Four-layer flag plumbing.** launcher → parser → constructor → **config extractor**. The
   bandwidth anneal was declared, passed, documented and *inert* for a full cycle because
   nothing assigned `hop_scale`. `spectral_sigma` and `gauge_penalty` were dropped by the
   extractor while training used them. Run `tests/unit/test_flag_plumbing.py`. **Never
   declare a flag before it does something.**
3. **Analysis must use the model's own graph cutoff.** The D1 scripts defaulted to 5 Å
   against 10 Å-trained models and reported `|H_ab| = 0`. They now derive it from the model.
4. **Self-matching `pgrep`.** A watcher whose command line contains the pattern matches
   itself and waits forever; a `pkill -f` can kill its own shell. Use the bracket trick:
   `pgrep -f "cf_[b]ase"`.
5. **Controls need the same verification as treatments.** `lig2` was silently the *maximally
   separated* cage pair — 10.5 Å, coupled in 38% of frames — so hub2-vs-lig2 compared
   coupling, not site. It is now **distance-matched** to the hub pair (5.57 Å both, 100%
   coupled both) and the run aborts below 90% coupling.
6. **b3 specifics:** `python` is not on PATH by default (`export
   PATH="$HOME/micromamba/envs/py13/bin:$PATH"`); jobs need `setsid`, not just `nohup`, or
   they die ~16 s after the launching SSH session closes; models trained locally do **not**
   exist on b3 and vice versa. b3's GPU3 has failed twice — a BMC cold cycle
   (`ipmitool chassis power cycle`, needs the root password via `su` over `ssh -tt`, with
   the password sent *after* the prompt appears) cleared it, taking ~35 min to come back.

---

## 8. How to read the results when they land

```bash
python3 /home/alex/.claude/jobs/86475d73/tmp/r1_summary.py     # R1 / step-2 matrices
grep -E "^  h3 " ~/runs/step2_local.log                        # per-cell rows
```
For arm 1b, the per-seed localisation numbers come from the `carrier channels` lines:
`partic=[e_maj e_min h_maj h_min]` — the supervised channel is **h_maj** (index 2) for
V_Cl⁺, and the other three take no gradient and form a free per-run baseline. Null ratio =
N_eff(active) / mean N_eff(nulls). The gate is N_eff ≤ 8 and ratio ≤ 0.15; **0.80 is the
guard's arming threshold and is not a localisation criterion** — misreading it as one is
what produced the retracted "2/8 localised" claim.
