# LES base models — report

**Scope.** The latent-Ewald-summation (LES) base models: what they achieve on this data,
shell-resolved forces, energies, and the Born-charge check. Forces are **per-component meV/Å**.

**Summary.** LES helps on this data, modestly and only on energies — a factor of two in the one
controlled comparison, nothing measurable on forces. Its mechanism is not long-range physics:
the gain is uniform across distance shells, which is what better local fitting looks like.
Direct fine-tuning from MACE-MH-1 beats training from scratch on every reading here.

---

## 1. The models

| model | recipe | latent charges |
|---|---|---|
| `base_v3_scratch` | MH-1 architecture, from scratch, no LES | — |
| control | MH-1 weights, direct fine-tune, latent charges held at zero (LES inactive) | off |
| **LES-REF** | the reference recipe (`ChengUCB/les_fit`), from scratch | live |
| **LES-FT2** | MH-1 weights, direct fine-tune, `--les_readout_scale` | live |

Reference-recipe settings that matter, for anyone reproducing: `--scheduler_patience 15` (MACE's
default is 50), stress weight 0, Stage Two at 50 %. Both it and MACE's default use `--lr 1e-2`.
`base_v3_les_ref`: 320 epochs, **0.21 meV/atom, 6.43 meV/Å**.

## 2. The controlled comparison

Identical settings, identical 150 epochs, one difference — latent charges live at a 1e-2 scale
versus held at zero:

| | energy meV/atom | forces meV/Å |
|---|---:|---:|
| control, LES inactive | 0.48 | 5.69 |
| **LES-FT2, charges live** | **0.24** | **5.62** |

**Energy halved, forces level.** LES-REF against `base_v3_scratch` (0.21/6.43 versus 0.23/6.63)
is *not* a controlled LES comparison — the whole recipe changes with it, so it cannot be read as
an LES effect.

**The charges are live and doing work:** readout `‖W‖` 6.4e-1 / 1.9e-1 (FT2) and 1.3e1 / 9.2e-1
(REF); `|q|` mean 0.028 / 0.053 e; `E_LES` 4.1 / 8.8 meV/atom; ablation moves forces by
9.6–18.4 meV/Å. **The ablation shows only that the model has adapted to the term, not that it
improves accuracy** — the pair above is the only evidence for that.

## 3. Shell-resolved forces (2877 frames, each model through its own forward)

**Neutral vacancy, 79 atoms — in-sample for every production base**

| base | total | 2–4 Å | 4–8 | 8–10 | 10–12 | >12 |
|---|---:|---:|---:|---:|---:|---:|
| base v2 | 10.96 | 20.0 | 10.0 | 8.8 | 7.1 | 6.5 |
| v3 scratch (no LES) | 4.83 | 6.3 | 4.6 | 4.6 | 4.3 | 3.9 |
| LES-REF | 4.99 | 7.1 | 4.8 | 4.6 | 4.1 | 3.8 |
| **LES-FT2** | **4.18** | **6.2** | **3.9** | **3.9** | **3.5** | **3.1** |

**Charged vacancy, 79 atoms — out of sample for every base**

| base | total | 2–4 Å | 4–8 | 8–10 | 10–12 | >12 |
|---|---:|---:|---:|---:|---:|---:|
| **base v2** | **33.22** | **74.9** | **30.4** | **15.3** | **8.9** | 8.0 |
| v3 scratch | 57.23 | 132.5 | 51.8 | 21.9 | 10.7 | 7.3 |
| LES-REF | 62.99 | 148.1 | 55.9 | 24.3 | 11.4 | 7.8 |
| LES-FT2 | 52.37 | 124.0 | 46.4 | 19.8 | 9.2 | **7.2** |

**The in-sample gain is uniform across shells** — LES-FT2 beats base v2 by 2.6–3.2× at *every*
distance. If the latent Ewald term supplied genuine long-range physics the advantage should
**grow** with distance; it does not, so the improvement is better local fitting. The only bin
where the LES bases lead out of sample is **>12 Å (7.2–7.8 against 8.0)** — suggestive, the
smallest bin, not claimed. **LES-REF is worse than the no-LES from-scratch model at every charged
shell.**

The broader pattern: from-scratch wins in-sample and loses out-of-sample. Base v2 remains far
better on charged frames (33.2 against 52–63) — the replay head, not the initialisation, is what
buys robustness.

## 4. Energies, meV/atom — median aligned (raw in brackets)

| data set | frames | base v2 | v3 scratch | LES-REF | LES-FT2 |
|---|---:|---:|---:|---:|---:|
| pristine 80 | 639 | 0.30 (0.94) | 0.06 (0.06) | 0.06 (0.17) | **0.05 (0.08)** |
| neutral vacancy 79 | 1174 | 0.40 (0.65) | 0.06 (0.06) | 0.07 (0.08) | 0.07 (0.07) |
| neutral vacancy 159 | 17 | 0.74 (0.63) | 0.09 (0.10) | 0.08 (0.08) | **0.03 (0.03)** |
| charged vacancy 79 | 1030 | **38.37** | 40.52 | 41.57 | 41.50 |
| charged vacancy 159 | 17 | **18.89** | 20.69 | 21.58 | 21.66 |

Aligned RMS (pristine / neutral 79 / neutral 159): base v2 0.43 / 0.61 / 1.60; v3 scratch
0.10 / 0.17 / 0.54; LES-REF 0.10 / 0.16 / 0.21; **LES-FT2 0.08 / 0.14 / 0.21**.

**Energies improve 5–8× where forces improved 2–3×**, and the LES advantage seen in forces is
mostly absent — LES-FT2 and the no-LES scratch model tie at 79 atoms. The exception is
**159 atoms: LES-FT2 0.03 against scratch 0.09 and base v2 0.74**, the only size-transfer signal
in the table.

**Isolated-atom references** (fitted per-species offsets, eV/atom): base v2 needs a
**130–193 meV/atom** correction to align with the labels (a_Cl −0.1299, a_Cs = a_Pb +0.1925); the
new bases need 3–28 meV (v3 scratch −0.0032 / +0.0048; LES-FT2 +0.0092 / −0.0136). **This retires
the W1 caveat** that absolute formation energies referenced to isolated Cs are unsupported: they
are unsupported on base v2 and supported on these.

## 5. Born effective charges

`Z*` is the right comparator for a latent field: the latent charges are constrained by **forces**,
i.e. by the energy's response to displacement, which is what `Z*` measures and what a static
population analysis does not.

Two properties of the construction have to be applied before the numbers mean anything. The Ewald
excludes `k = 0`, so **nothing constrains the net latent charge** (−2.08 e per cell for FT2,
+2.54 e for REF); and the energy is **quadratic in `q`**, so **the global sign is free**. Raw
per-species latent charges are therefore not interpretable, and `Z*` carries the same free sign
(it is linear in `q`).

`Z*` from the models themselves (`compute_bec`), isotropic part, 8 neutral 80-atom frames:

| | Cl | Cs | Pb | \|Pb/Cs\| | Σ Z* per cell |
|---|---:|---:|---:|---:|---:|
| LES-FT2 | −0.466 | +0.382 | **+1.028** | 2.69 | **+0.18** |
| LES-REF | +0.507 | −0.409 | −0.955 | 2.34 | +2.50 |
| DFT expectation | − | + | + (anomalous) | ~3.4 | 0 |

Flipping LES-REF's free sign gives Cl −0.507, Cs +0.409, Pb +0.955 — **agreeing with LES-FT2 to
~10 % on every species**. The two models learned the **same** response, and both show the
halide-perovskite signature: Cl the anion, Cs and Pb cations, **Pb anomalously large at 2.3–2.7×
Cs**. Magnitudes run ~4× below DFT (Pb ≈ +4.4, Cs ≈ +1.3), expected because nothing pins the
absolute scale at `epsilon_factor = 1`; only ratios are meaningful.

**The acoustic sum rule separates them and favours LES-FT2 by 14×** (+0.18 against +2.50 per
cell). `Σ Z* = 0` is rigorous, so the fine-tune's response is nearly charge-neutral and the
from-scratch one is not — consistent with LES-FT2's better shell errors (4.18 vs 4.99 neutral,
52.4 vs 63.0 charged).

**Flagged, not resolved:** the net latent charge per cell is −2.06 e (FT2) and +2.47 e (REF) in
cells that are neutral or singly charged. A non-neutral distribution in a periodic cell needs a
neutralising background and the energy then carries a convention; anyone reading these charges
physically should check that first. Per-species means (Cl/Cs/Pb): FT2 −0.040 / +0.006 / −0.017,
REF +0.039 / −0.055 / +0.091 — not an ionic pattern in either, as expected for a latent field.

## 6. Provenance

The `les` library at `/home/alex/src/les` is **pristine upstream** — clean tree, no commits ahead
of `ChengUCB/les.git`, every commit by the LES authors, HEAD `c8063fa`. `MACELES` in
`mace/modules/extensions.py` differs from the branch point by one cosmetic `node_energy` cast
(commit `daaaf10`, pre-dating this work) on a quantity the LES path never consumes. Local changes
are confined to `mace/tools/` and concern getting **into** MACELES — the foundation-loader guard,
the fitted `atomic_inter_shift`, and the readout init scale — not what it computes.

## 7. Conclusions

1. **LES is worth having on energies and not on forces, on this data**: 0.48 → 0.24 meV/atom in
   the controlled pair, forces 5.69 → 5.62.
2. **The mechanism is not long-range physics.** The in-sample gain is flat across shells. The
   single long-range hint is >12 Å out of sample, in the smallest bin.
3. **Direct fine-tuning from MH-1 beats from-scratch with LES** on every reading here — forces,
   energies, and the acoustic sum rule by 14×.
4. **Base v2 still wins out of sample** on charged frames by a wide margin, which is a statement
   about the replay head rather than about LES.
