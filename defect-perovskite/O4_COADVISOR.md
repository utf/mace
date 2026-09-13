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

**Question for you:** next item, or does the retrain result come first?

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
