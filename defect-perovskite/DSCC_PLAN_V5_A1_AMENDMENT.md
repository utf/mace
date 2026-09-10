# v5 amendment A1 — reference-free environment dependence (removes pristine centring)
Applies to W3 onward (all heads trained on base_v2) and to the W4 factorial definitions.
Replaces every "centred on the pristine species feature means" clause in §2.1 and the W4
formula `b_i = b_Z(1 + β tanh(g_Z(h_i) − g_Z(h̄_Z^pris)))`.

## Rationale (record)
- The neutral null is algebraic in D-SCC: at S_ref the head contributes exactly zero,
  so no environment-dependent term can alter the neutral PES. Pinning corrections to
  vanish in pristine environments protects nothing.
- A uniform onsite shift is absorbed by C_Q in energies and has no effect on forces; the
  rank-2 block is traceless and the rank-1 block off-diagonal, so neither can shift the
  gauge. Centring is not a gauge safeguard.
- The remaining purpose of centring, keeping corrections small in the bulk, is a
  regularisation preference and is expressed without a host reference by a bound plus a
  weak L2 toward zero correction (toward the element default).
- A static-cell feature mean is ill-defined under thermal noise and does not exist for a
  foundation model; a learned bias inside the bound is fitted on the thermal ensemble and
  needs no per-host reference.

## Definitions (all environment-dependent terms of H0)
```
rank-2 onsite:   b_i   = b_Z · (1 + β · tanh(g_Z(h_i)))
rank-1 onsite:   a_i   = a_Z · (1 + β · tanh(f_Z(h_i)))
scalar onsite:   δε_i  = Δ_Z · tanh(e_Z(h_i))
SK modulation:   t_ij  = t^SK_ij(r) · exp(η · tanh(m_{Z_iZ_j}(h_i, h_j)))     (unchanged form,
                                                                              reference removed)
```
- `h_i`: base_v2 early-block invariant features (registered tensor, W1). `g_Z, f_Z, e_Z,
  m_ZZ'`: small readouts (registered architecture: linear or one hidden layer of
  registered width) with their own biases. No stored pristine statistics enter any
  runtime path.
- Bounds (registered defaults): `β = 0.5`, `η = 0.5`, `Δ_Z` = registered fraction of the
  species onsite baseline spread. Species defaults `b_Z, a_Z, ε_Z, t^SK` initialised as
  before (Harrison / element defaults).
- Regulariser: weak L2 on the `tanh(·)` outputs toward zero (registered weight per
  term), pulling corrections toward the element default in the absence of signal.
- Foundation form (registered now, not built): `b_Z, a_Z, Δ_Z, ε_Z` as species
  embeddings and one shared readout per term taking `(embedding, h_i)`; bounds and L2
  unchanged.
- Known redundancy: the readout bias and the species default are jointly redundant by
  one constant per species; harmless under the bound and the L2. Recorded, not fixed.

## What remains host-specific
Exactly two items: `eps_inf`, and the pristine-gap regulariser (a per-host gap number,
not a feature reference; its foundation substitute is a database or band-structure gap,
already in the deferred register). The bound-state precondition (C5) is unchanged.

## Codebase
- Remove the stored pristine species feature means and every reference to them from the
  runtime path (keep the static pristine cell only for the gap regulariser).
- Add the readouts with biases and the L2 term; expose per-term saturation fractions.
- Static check: no runtime module reads pristine feature statistics (test asserts the
  symbol is absent from the H0 assembly path).

## Tests (in addition to the unchanged Phase-1 gates)
- Neutral null bit-exact and gauge-shift invariance: unchanged, re-run.
- Zero-readout limit: with all readouts forced to zero the model equals the
  species-default H0 to 1e-12.
- Bounds: `|tanh(·)| < 1` trivially; report the saturation fraction (|tanh| > 0.95) per
  term and species; the §2.10 hub-Pb diagnostic is read on `tanh(g_Z(h_i))` and
  `tanh(e_Z(h_i))` at the flanking Pb.
- Tensor-ratio clause (Arm-1 criterion ii, first clause) is meaningful again with
  site-dependent `b_i`; report `|b_i Q_i|` at the flanking Pb over its bulk spread, but
  it remains a diagnostic, not a selection criterion.

## W4 factorial (restated with the new forms)
{scalar-only; + rank-2 with species `b_Z` (β = 0); + rank-2 with `b_i(h_i)`; + rank-1
`a_i(h_i)`; full}, Φ = 0, six seeds, paired by fold, TOST reading; entry gate C5;
metric per D15: paired 79/159 flanking-Pb residual at matched d, short-d bins separately;
adopt `b_i(h_i)` or `a_i(h_i)` only on superiority; simplest variant keeping the bound
state otherwise.

## Registration before W3 opens
Readout architectures and widths; `β`, `η`, `Δ_Z`; L2 weights per term; the feature
tensor used for `h_i`; the foundation embedding dimension (form only).
