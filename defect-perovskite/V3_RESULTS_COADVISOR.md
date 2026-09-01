# T-B on V3 — result, 1 Sep 2026

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Short version.** The gap-referenced constraint does exactly what it was built to do: every
one of 24 cells places the carrier level inside the band gap, with no breach of the upper cap
anywhere. What it does not do is localise the carrier. Six of 24 cells get the state inside
the vacancy's bonded neighbourhood, four clear the null-ratio criterion, and none clears all
four gates. The binding inequality is now *enforceable* — which V1 and V2 could not manage —
but enforcing it is still not sufficient to produce a bound state.

## What changed since the last round

V2 established that `Delta_bind >= m` is necessary but not sufficient: the model met the
inequality by sliding the host reference, with `lambda_1(pristine)` floating over ~8 eV.

V3 removes that freedom. Every matrix element of the carrier Hamiltonian is a function of the
trunk's first interaction block, detached, so an atom beyond `r_max` of the vacancy has a
descriptor identical to its pristine counterpart and the far block of H is pinned to the host.
The constraint gained an upper cap at the band gap. Two length constants remain, both
material-agnostic.

Three corrections were applied before this run, and one of them was load-bearing. An earlier
launch used a stretched hopping decay (2.5 A) to restore the initial magnitude after the
reference distance moved; that gives `t(10 A)/t(2.85 A)` ~ 6% across ~100 neighbours, which is
the long-ranged, nearly complete graph whose lowest state is the in-phase superatom mode. Those
cells produced spuriously small `N_eff` alongside a destroyed force fit and were discarded.
Connectivity is now set through the amplitude at a physical decay length (1.0 A); the realised
profile is calibrated against the *measured* median nearest-neighbour distance and logged at
init and end of every cell. Final profile, averaged over 24 cells:

    t(2.85 A) = 0.559 eV    t(5.6) = 6.1%    t(8.0) = 0.32%    t(10.0) = 0.00%

## Result — 24 cells, 6 seeds per arm, 40 epochs

Frozen base, head + response channel, no clamp. Delocalised reference ceiling: axial_red
+0.855, rmse_nbhd 20.6.

| arm | N_eff mean | N_eff range | null ratio | Delta_bind | axial_red | rmse_nbhd |
|---|---|---|---|---|---|---|
| f_m 0.1, E_gap 2.4 | 46.63 | 15.0 – 59.7 | 0.640 | +0.391 | +0.506 | 33.2 |
| f_m 0.2, E_gap 2.2 | 24.06 | 8.9 – 37.8 | 0.384 | +0.644 | +0.414 | 37.3 |
| f_m 0.2, E_gap 2.4 | 31.24 | 8.4 – 64.4 | 0.566 | +0.727 | +0.467 | 38.2 |
| f_m 0.3, E_gap 2.4 | 19.28 | 7.8 – 33.6 | 0.288 | +1.110 | +0.545 | 31.0 |

Gates (candidate region = 14–15 atoms, the vacancy's bonded neighbourhood):

| gate | pass |
|---|---|
| `m <= Delta_bind <= E_gap` | **24 / 24** |
| `N_eff <= region size` | 6 / 24 |
| null ratio `<= 0.15` | 4 / 24 |
| retained mass `>= 0.9` | 0 / 24 |
| **all four** | **0 / 24** |

### What holds

**The constraint is enforceable and the escape is closed.** No cap breach in any cell, and
`Delta_bind` is tightly grouped within each arm (+0.39, +0.64, +0.73, +1.11) — much tighter
than any other quantity we measure. For contrast, V1 previously reached `Delta_bind` = +11.9 eV
against a margin of 0.5 eV, which no level inside a 2.2 eV gap can do. That route no longer
exists.

**`m` does real work.** `N_eff` mean falls 46.6 -> 24.1 / 31.2 -> 19.3 as `f_m` goes
0.1 -> 0.2 -> 0.3, the localisation length falls 6.60 -> 6.15 / 6.13 -> 5.55 A, and the mass
inside the candidate region rises 0.268 -> 0.329 / 0.333 -> 0.359. Four cells clear the
null-ratio criterion; across 18 V1/V2 cells, none did.

**The gap choice does not matter, as predicted.** The `E_gap` 2.2 and 2.4 arms at `f_m` = 0.2
differ by less than the seed spread within either (24.1 vs 31.2, ranges overlapping across
8.4–64.4). No conclusion turns on which source is used.

### What does not

**Nothing passes all four gates.** The four best cells are informative about why:

| cell | N_eff | ratio | region mass | axial_red |
|---|---|---|---|---|
| f_m 0.2, gap 2.2, s5 | 8.94 | 0.130 | 0.425 | +0.100 |
| f_m 0.2, gap 2.4, s5 | 8.53 | 0.118 | 0.452 | +0.342 |
| f_m 0.2, gap 2.4, s6 | 8.37 | 0.133 | 0.447 | −0.167 |
| f_m 0.3, gap 2.4, s4 | 7.79 | 0.119 | 0.463 | −0.073 |

Every cell that localises pays for it in force fit — +0.34 at best against a +0.855 ceiling,
and negative in two of four — while the delocalised cells in the same arms sit at +0.6 to +0.8.
Localisation and fit are trading off consistently, and the data do not tell us which side is
physical.

**Seed spread exceeds the effect being measured.** Within `f_m` 0.2 at `E_gap` 2.4, `N_eff`
runs 8.4 to 64.4 — an eightfold range, wider than the separation between arms. The `f_m`
ordering is suggestive; at n = 6 it is not established.

**The retained-mass gate is uninformative rather than failed, and this needs fixing before it
is read again.** Interface mass sits at 0.84–0.93 across all four arms, against ~0.37 for a
state spread uniformly through the tiled cell. The carrier is concentrating on the join plane
in essentially every tiled cell, so `retained >= 0.9` at 0/24 is measuring the tiling geometry,
not binding. This was invisible until the interface diagnostic was added alongside retained
mass; without it, 0/24 would have read as a clean band-like failure.

## Where this leaves the decision tree

Not the adopt branch: the gates are not met. Closest to the second branch — the level spread
over the vacancy's bonded neighbourhood, which for a universal model is acceptable physics —
but the region gate itself passes only 6/24, and the one gate that would corroborate it
(dilution) cannot currently be read. We would not claim that branch on this evidence.

The honest summary is narrower and, we think, still worth having: **the band-gap reference
makes the binding constraint enforceable without an escape, for the first time. It does not
make the carrier bind.** Whether the remaining gap is a missing physical term or an
under-determined objective is not decided by this run.

## Proposed next steps, in order

1. **Fix the dilution gate before anything else.** As it stands it cannot discriminate, and it
   is one of the two gates that would distinguish a genuinely bound state from a
   region-filling one. Randomising the join plane and excluding an interface collar, or
   comparing against a pristine-pristine tiling as a null, are the obvious candidates.
2. **More seeds before reading the `f_m` trend.** The ordering is consistent across three
   values but the within-arm spread is larger; 12–16 seeds at `f_m` 0.2 and 0.3 would settle
   whether the trend is real.
3. **The force-fit trade-off is the substantive question.** Every localised cell fits worse.
   Either the localised solution is wrong, or the force labels alone cannot distinguish them
   and an independent observable is needed. The charged-pristine single points already
   discussed would give one; a projected density would give another.

Trained models from all 24 cells are saved, so the hopping-offset diagnostic and a band-edge
pass over the V3 heads can both be run without repeating this.
