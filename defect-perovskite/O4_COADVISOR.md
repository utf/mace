# O4 — the analytic head force: implemented, gated, measured (2026-09-13)

Your force kernel and the two small items are in. The six-seed O3 retrain is running. This is
what the numbers came out as, including where your projection and mine were both wrong.

## 1. The kernel

Built as specified, `mace/modules/dscc/analytic.py`:

* per edge, closed form — the Slater–Koster block's `dS/dv` and `dS/du`, the descriptor's
  three channels, the projector `du/dd = (I − u uᵀ)/r`;
* per node, a small autograd on leaf descriptors through the **production** `on_site` and
  `directional_site_blocks`, so A1's environment factors and the block layout are not
  transcribed twice (this also makes `beta_a, beta_b ≠ 0` work for free);
* **one `X` for both consumers.** `dP` and `E_host`'s two Fréchet pullbacks share `U`, so
  `X = U (diag(Σ_σ df) + M_S + M_R) Uᵀ` is assembled in the eigenbasis and rotated out once —
  four `n³` GEMMs where the two routes did about ten.

It runs the whole head under `no_grad` at inference, which closed your fills item as a side
effect rather than by editing `fill.py`: `two_fillings` takes its frontier path and forms **no
density matrix at all** (8.0 → 2.4 ms at 159 atoms). The 7.9 ms you flagged was the eager
`_DensityMatrix`, which ran only because `compute_force` turns grad on.

**Scope I left, stated rather than hidden:** forces only. The stress keeps the autograd
cotangent route — it needs the strain graph and is not on the timed path. You listed it; it is
not done.

**Gates.** Analytic vs autograd on the same model/frame **1e-9**; compiled vs eager **2e-14**;
a 5.6 Å cell with 30 periodic self-image edges **2.7e-15** (the assembly overwrites the matrix
diagonal with the on-site levels, so a `src == dst` edge contributes its off-diagonal entries
and not its diagonal ones — masked unconditionally). Plus a gate that no density matrix is
formed. Full `tests/extensions/dscc` suite **211 passed**.

## 2. Timing — your projection was 51–53 ms / ~1.2 at 79 atoms. It reads 74 ms / 1.88.

Idle A4000, fake O3 weights, P4.3 protocol, **both force routes interleaved frame by frame**
(the card's clocks drift ~15 % between separate invocations — two of my earlier comparisons
looked like a regression that was entirely drift). Neighbour list **not** static: 60 different
frames, `torch.compile(dynamic=True)`.

| | base | model, analytic + compiled | ratio | model, autograd | ratio |
|---|---|---|---|---|---|
| 79 atoms | 38.4 ms | **74.0 ms** | 1.88 | 74.4 ms | 1.94 |
| 159 atoms | 67.7 ms | **106.5 ms** | 1.57 | 114.0 ms | 1.68 |

**A wash at 79 atoms; 7.5 ms at 159.** The arithmetic argument was right and beside the point.
The contraction is 2.7 ms of GPU time at 79 atoms — under your 1 ms estimate once you allow
for float64 on an A4000 — but it issues **542 CUDA kernels eager, 262 compiled**, and costs
~4.5 ms of pure launch latency. `head_cotangent` is another 3.6 ms of the same kind. At 79
atoms that is about the size of the graph traversal it removes, so the two routes trade. At
159 the traversal grows faster than the kernel does and it pays.

Compiling the H0 assembly (your other item): 3.2 → 1.1 ms.

## 3. The CUDA graph — your diagnosis is exactly right, and it is the remaining lever

Measured, base forward+backward at 79 atoms:

* eager: **41.6 ms wall, of which 41.4 ms is CPU issue** — entirely launch-bound, as you said;
* captured as a CUDA graph: **30.6 ms wall, 0.023 ms of CPU.**

Capture succeeds on this base — openequivariance's JIT kernels do not refuse it — and energies
and forces agree with eager to 9e-7, which is the base's own run-to-run scatter
nondeterminism, not an error of the graph (a same-route repeat differs by the same amount).

So graphing the base removes 11 ms outright **and** frees 30 ms of CPU in which the head — which
under O3 reads nothing from the base — can issue its own kernels. That is plausibly the whole
gap to 1.2, and it is the one thing that attacks the actual bottleneck.

**Not implemented.** Capture fixes the shapes and every frame has a different edge count, so it
needs edge-bucket padding: pad to a bucket with dummy edges beyond `r_max`, which are exactly
zero through MACE's polynomial cutoff. Node counts are already fixed per size class. A day's
work, with a production caveat (any driver that changes the neighbour list must re-bucket).

**Built, same day (your ruling).** See the O5 section below.

## 4. The `eigh` bypass — not a threshold

You asked for the crossover measured once and registered. It is not a size threshold on the
deferred path. The base is issued first and the head reads nothing from it, so a CPU `eigh`
runs **underneath the base's GPU window and costs nothing**, while a GPU one is serial work
behind that window. Float64, one matrix, ms:

| atoms | orbitals | cuSOLVER | MKL (+ round trip) | base fwd+bwd GPU window |
|---|---|---|---|---|
| 79 | 316 | 12.4 | 8.6 | 45.9 |
| 159 | 636 | 23.6 | 26.0 | 64.2 |
| 318 | 1272 | 84.7 | 101.3 | 127.2 |
| 636 | 2544 | 433.8 | 389.2 | 253.6 |

The CPU `eigh` hides completely up to 1272 orbitals and is cheaper than cuSOLVER at 2544 anyway
→ **CPU at every size measured** (`EIGH_DEFERRED_ON_CPU`). Note the head-to-head crossover is
**not monotone** — the GPU wins at 636 and 1272 and loses again at 2544 — so the existing
`EIGH_CPU_MAX_ORBITALS = 512` must not be generalised into "above this size, use the GPU".

## 5. Something you should know before reading the retrain

**O3 had no test of its own until this afternoon.** I added nine (FD forces, FD stress,
batched == per-graph, training == inference, and "perturbing every base parameter leaves the
head energy bit-identical") before touching the kernel. A one-epoch smoke then found **two
training-path bugs that would have killed all six seeds at `prepare()`**: `set_feature_stats`
standardising the base's 512 block-0 invariants into buffers sized for the head's 18-channel
descriptor, and `saturation_report` feeding those same base features to the head's readouts —
the second of which would have been a wrong number rather than an error once the shapes agreed.
Both fixed; diagnostics now go through `MACEDSCC.readout_scalars`.

A **third** bug then killed the first launch about four minutes in, at the first evaluation
batch: the per-graph path (any batch that mixes a charged frame with a reference one, or mixes
sizes) slices the features it hands `H0`, and a decoupled head has none — `scalars[nodes]` on
`None`. The uniform path passes them straight through, which is why every gate and both smokes
missed it: I had run the smoke with `--subset 12`, whose frames are all one size. Fixed with
explicit placeholders (`H0` overwrites them with its own descriptor anyway), gated by
`test_the_per_graph_path_matches_the_batched_one`, and the six seeds were relaunched. Cost: one
wave, ~10 minutes.

Recorded so nobody misreads it: under O3 the saturation report still carries a `hop` entry
computed from `sk.hop`, the MLP that `hop_modulation = False` never evaluates. **It is
meaningless for O3.**

## 6. Status

Six seeds launched 2026-09-13 ~20:00 (b3 clock): s2, s3, s4, s6 on b3 GPUs 5/6/7; s0, s1 on the
local A4000. **b3 GPU 4 is off the bus** (`Unable to determine the device handle for GPU4:
0000:89:00.0: Unknown Error`) — the same fault GPU 6 had on 2026-09-06, which a cold power
cycle cleared. Not cycled; three usable cards of the permitted four is enough.

All timings above are on **fake O3 weights** (architecture-correct, untrained). The retrained
model gets re-timed on the same protocol before any of these numbers go in the report.


---

# O5 — the CUDA graphs, built (2026-09-13, same session)

Your ruling taken: capture is weight-independent engineering, it does not gate on the retrain,
and the head gets captured too. Done, gated, timing pending on the card.

## Structure — as you sketched it, with one ordering change

```
graph A0 : H0 assembly                         ~1 ms of GPU, issued FIRST
D2H      : H to pinned host memory
graph A1 : base forward + backward             ~30 ms of GPU, 0.02 ms of CPU
CPU      : eigh, chemical potentials, occupations -- under A1's window
graph B  : site charges, E_host, E_M, X, the analytic contraction, accumulation
```

Your sketch puts the assembly *in* graph A with the base. Recorded after it, `H` is not ready
until the base's 30 ms window closes, and the host idles through it and then works alone.
Recorded **first**, the host copies `H` back and diagonalises while the base runs — which is
the `EIGH_DEFERRED_ON_CPU` rule doing its job inside a captured step rather than beside one.

## Gates

Reproduces the eager model to **1e-14** in energy, head energy, `dq` and forces. A frame whose
atoms have moved — a *different* neighbour list of the same length — replays through the same
graphs to the same tolerance: only the length is fixed by capture, the edge list's values are
read at replay. A coupled or base-reading head is refused. Padding is **bit-identical** (§4 of
O4). Full suite **225 passed**.

## What had to change, and what did not

Every change removed a device-to-host read or a value-dependent shape. **None of them changed
a number** — the suite is the evidence:

* `ReciprocalPlan` holds the cell-dependent half of the Ewald kernel (the `k` set, the volume,
  `k_c`); the eager entry points now delegate to the same arithmetic, so there is one
  implementation and not a capture fork.
* `site_occupation` accepts an already-solved fill — `chemical_potential` is a host Newton
  iteration and belongs in the CPU segment, next to the eigh that feeds it.
* `_dk_eigenbasis` applies its fixed-`N` correction unconditionally: `if bool(ok.any())` was a
  host read that only skipped work `torch.where` already makes a no-op.
* `standardise` mirrors its buffer flag in a Python attribute, read once.
* `_trunk_data` can pad the base's subset instead of compacting it.
* `head_tail` is the head's GPU work as a function of tensors alone — no potential solve, no
  `nonzero` for the frontier set (the analytic route needs no `dP`), no lattice sums. Gated
  against the model's own path at 1e-14.

Your note on the O1 dedup was right to flag: every dummy edge is the unordered pair (0, 0), so
it adds one row to the unique set and cannot reach a real pair's input. The gate runs the
**modulated** head as well as O3's, and both come out bit-identical.

## Two capture traps, recorded because they cost time

1. `padded["positions"].clone()` of a batch that has been through an eager forward is a
   **non-leaf** — the model calls `requires_grad_(True)` in place, so the clone carries that
   forward's graph, and capture refuses it ("stale reference to the default stream").
   `.detach().clone()`.
2. A `for` loop's bindings outlive the loop in Python, so the warm-up's autograd graph — and
   its stream reference — is still alive at capture. Deleted explicitly.

## Deliverable, registered as you asked

**Model <= 53 ms per step at 79 atoms**, with both ratios reported (against the eager base the
criterion was written for, and against a captured base). `dscc_benchmark.py --graphed` times
the model leg through capture and a bucketed captured base leg alongside the eager one. The
number that goes in the record is milliseconds.

## Still open

The analytic **stress** — NPT only, a follow-up not a blocker, as you said. And the timing
itself, which needed the local A4000: every registered P4.3 number is on that card and the
six-seed retrain was using it. O3 trains much faster than W6 (~37 s/epoch against the W6
cohort's several minutes — the decoupled head drops block 0 and its `create_graph` traversal
entirely), so the wave finished within the hour rather than overnight.


## O5 timing — the deliverable is met

Idle A4000, fake O3 weights, `dscc_benchmark.py --graphed`:

| | base eager | base captured | model eager | model graphed | / eager base | / captured base |
|---|---|---|---|---|---|---|
| 79 atoms, fixed cell | 40.4 ms | 32.4 ms | 63.6 ms | **39.7 ms** | 1.08 | 1.23 |
| 159 atoms (registered) | 67.2 ms | 67.3 ms | 100.6 ms | **89.0 ms** | 1.33 | 1.32 |

**39.7 ms against the registered ≤ 53 ms.** Your 1.3-not-1.2 prediction lands at 1.23 against a
captured base.

Two things fall out that you should have:

**The 159-atom captured base is within noise of the eager one** (67.3 vs 67.2 — inside the
card's drift, so that pair alone proves nothing; the 79-atom 41.6 ms wall / 30.6 ms GPU split
is what carries it). That is your launch-bound-to-compute-bound crossover, located: at 159 the
base's kernels are large enough that there is no CPU issue time left to remove. The model still
gains, because the **head** is launch-bound at both sizes.

**Capture is a fixed-cell path, and the trap was live on our own data.** The reciprocal plan,
`ξ` and the padding translation are constants of the cell. The registered 79-atom P4.3 frames
are NPT snapshots whose cells span 15.76–16.07 Å; one capture replayed over six of them read
**5.5 meV and 9.3 meV/Å** with nothing raised, because `dq` was bit-perfect and only the
electrostatics were stale. `fits()` now refuses a changed cell, composition, charge state or
node count — a ~50 µs host comparison against a silent wrong answer — and the 79-atom row above
uses a fixed-cell trajectory, which is what a relaxation or an NVT/NVE step is. Re-capture
costs 0.25 s (79) / 0.38 s (159), so a barostat that touches the cell every few hundred steps
is fine and one that touches it every step is not.

On the production float32 base (with its ZBL pair repulsion) the captured step agrees to
2e-5–8e-5 eV and ≤ 5e-6 eV/Å over six real frames per size — the base's own float32
reproducibility under a reordered edge list, four orders below `τ_phys`.

## The turn-on reading — and it is not the failure the check was written for

Four seeds in:

| | on-site p95 | max | `|tanh| > 0.95` | held-out force RMSE |
|---|---|---|---|---|
| O3 s0/s2/s3/s4 | 2.970–2.974 eV | 2.977–2.981 | **0.398–0.399** | 0.0323 / 0.0318 / 0.0353 / 0.0319 eV/Å |
| W6 s0…s6 | 0.465–1.147 eV | 0.538–1.314 | **0.000** | (shipped) |

The registered check exists so "forces equivalent" and "the term never activated" cannot be
confused. The term did not fail to activate — **it is at its stop on two atoms in five**, p95
2.97 eV against a half-width of 3.079, where W6 never came within a factor of two of the bound.
Four seeds agree to the third decimal: architecture, not seed.

Precisely: the A1 barrier knee is `|pre| = 2.0`, `|tanh(2)| = 0.964`, past the 0.95 counter by
construction — so those atoms sit *at* the knee (the logged barrier term puts them within ~4 meV
of pre-activation past it), not beyond it. The barrier is binding on 40 % of sites, and a
parameter at its stop has `sech² ≈ 0`. We have paid for misreading that once (the `eta` ruling).

Most likely cause is a deferral we registered ourselves: O3 feeds the production readout the
**raw** species-resolved `l = 0` basis, because A1.1's standardisation was excluded under
one-change-at-a-time. If that is it, the follow-up arm is the standardisation path rather than
the three-body screening — but it is a second change, and adoption is decided on forces.

Forces so far are equivalent to or better than the shipped cohort. The paired TOST, the energy
and between-size gates and the frozen-core ladder run when s1 and s6 land.
