# Stage A′ base refit, head edits, and the Stage B objective — report

V_Cl+ in orthorhombic CsPbCl3. Labels: Mosquera-Lois & Walsh, PRX Energy 4, 043008 (2025);
PBE scalar-relativistic; defect calculations set up through `doped`. Plan of record:
`STAGE_APRIME_SPEC.md`, received 3 Sep 2026, executed in the order of its section 5.

**Every trained number below carries its regime tag.** Three regimes appear: the s7
head-only cohort (frozen Stage-A base `e0_base_s1`, γ = 3 eV, Gaussian 0.05 eV, forces +
`loss_gap`, 60 epochs, lr 0.01, float32), the Stage A′ bases (neutral data only, e0 recipe,
140 epochs, float32, cuEq, 159-atom share 0.25 in energy AND forces), and the Stage B cohort
(head only on the A′ production base, section 3 of the spec, 24 epochs, lr 0.005, trunk
float32 / head float64 with the base cache).

---

## 0. Decisions of record, and what each became

| decision | outcome |
|---|---|
| Joint training of base and head on charged labels retired; base on neutral labels only, frozen during head training; criterion 1 a regression test | Done. Stage A′ trained on n = 0 frames only (`dataset_e0` and the four `dataset_cf` folds); Stage B trains under the head-only mask with `base_lr_factor 0`, the base's outputs cached, and a drift guard on the cache every epoch. Criterion 1 is gate 1 below, scored against the A′ references. |
| E_LR retained unchanged in form; density detached; long-range parameters frozen at physical values | Done, with one interpretation recorded: "physical values" means host charges zero, polarisation off, amplitude 1/√ε∞ = 0.5 at ε∞ = 4 — the ion lattice sits in H through the Madelung term, and a base trained without E_LR cannot absorb a geometry-dependent term from randomly initialised host charges. The previous isolated-gauge convention (`carrier_self_isolated` off) is kept unchanged. The spec's neutrality check `|Σq + Δn| < 1e-8` holds only at amplitude 1; the invariant enforced is the density's (Σ q/a + Δn = 0), and the spec's residual is logged each epoch. |
| Decay lengths as four learned universal scalars | Done: `L_b = L0 exp(β_L tanh u_b)`, β_L = ln 2, L0 = 1.0 Å, u_b = 0 at start, one set shared across hosts. |
| Precision trunk f32, head f64; base outputs cached; only the first interaction block recomputed | Done (section 2.5 below). Identity half of F20 holds; the ≥ 3× speed half does not (1.5× per epoch): the step is CPU-bound in the head's per-graph eigensolve and Ewald loop, which the excluded remedies own. |
| Not in this cycle: eigensolve batching, Ewald geometry precompute, SCC, occupation/SiC | Not done, as instructed; they are where the remaining time is. |

**Standing-rule addition executed (section 5.1).** `avg_num_neighbors` travels in the state
dict through a buffer refreshed at write time and written back at load time; a test asserts
the premise that the plain float is still invisible to the block's own `state_dict`, so the
carry is retired rather than left to disagree if upstream ever makes it a buffer. Every new
knob (learned decay lengths and their β, the modulation form, the long-range detach and
freeze flags, image compensation, the precision policy, the centred on-site channel, the
cache checksum) is a constructor argument, a CLI flag, a launcher variable and a
config-extractor key from the day it exists; the round-trip test flips all of them.

---

## 1. Corrections found on the way, before any result

**The neutral two-size upweight of the joint run never reached the loss** (LEDGER.md entry
12). `DefectLoss` scores an n = 0 frame through its base terms, which read
`base_energy_weight` and `base_forces_weight`; the upweight scaled the generic
`forces_weight`, which no term reads for a neutral frame. Verified by printing the columns
on a 159-atom neutral frame after the upweight: `forces_weight` 0.497, `base_forces_weight`
1.0. The joint run's "both realised 25.0%" was true of the charged population and false of
the neutral one; the neutral 159-atom force-slope collapse in that report happened at
natural weight. The first Stage A′ launches trained on the same inert weight for forty
minutes and were stopped. The column is now named per population, the realised share reads
the same column, and a test moves the `DefectLoss` value with it.

**Three things the smokes found before the chain ran unattended.** The Stage-A loader
refused every historical checkpoint for lacking the new constants buffer (it is bookkeeping,
carried by the explicit float copy, and is excluded from the check now); the profiler probe
ran before the trainer had moved the loss to the device; and the trainer's own probing
forward for the pristine centre tripped the "no centre yet" guard it was about to satisfy
(the model now collects the centre itself with the correction held off). The pristine-centre
buffers are correction state, which a Stage-A checkpoint never has.

---

STAGE_APRIME_REPORT_BODY

---

## 7. Operational record

- Local A4000: F20 measurement, the Stage B smoke, the A′ production base. b3 GPUs 4–7:
  the A′ fold bases, the forward-only probes, the Stage B seeds, the gates; never more than
  four GPUs in use, one shared between a fold base and the OOD indicator.
- Every queue script waits on a file, never on a process (entries 10 and 11). One launcher,
  `b3_run.sh`, starts every remote job from the worktree, because `ssh b3 '…'` starts in
  `$HOME` and a relative script path fails silently — it did, twice, before the launcher.
- The Stage B smoke (one epoch, old base, old w_E) is what caught the centre guard and the
  c-table class; the two costed ~1 h and were the difference between a chain that ran and
  one that stopped at 20:00 with nobody watching.
