# M3 — the sign flip never reached t(d). R-C, and s+p is unearned.

*V_Cl+ orthorhombic CsPbCl3. Dataset: Mosquera-Lois & Walsh, PRX Energy **4**, 043008 (2025);
labels are that paper's low-fidelity PBE set, scalar-relativistic, no SOC.*

**Summary.** The measurement that separates the three readings has been made. `|t'|` did not
move between the arms and sits far below what R-A and R-B both require, and `corr(N_eff, d)` is
flat in both arms. That is **R-C**: the flip never engaged the hub coupling, the A/B's fit gain
came through other channels, and the named-failure reading the A/B table appeared to license is
retired. On the §5 branch logic the next step is to find why the flip does not reach t(d) —
**not** an architecture change. s+p is unearned.

Your §1 rule earned its place: the ungated A/B table looked like a clean architecture verdict,
and one measurement dissolved it.

## M3 — autograd dH_ab/dd on real frames

12 A/B cells × 24 charged frames. `t'` is the exact gradient of `H_ab` with respect to both hub
positions, projected on the pair axis. This supersedes F2's zero-feature profile slope.

| arm | n | mean \|t'\| (eV/Å) | per-seed range | mean \|H_ab\| | corr(N_eff, d) |
|---|---|---|---|---|---|
| **ON** | 6 | **0.0443** | 0.017–0.086 | 0.041 | **−0.010** |
| OFF | 6 | 0.0623 | 0.021–0.112 | 0.052 | +0.113 |

Both R-A and R-B require ON `|t'|` → 0.10–0.20 eV/Å with OFF at the prior.

**`|t'|` did not move.** ON is nominally *lower* than OFF, but the per-seed ranges overlap
almost entirely, so the defensible statement is that the arms are **indistinguishable in `|t'|`
and both sit below target** — not that ON is worse. For scale, the realised zero-feature
profile slope is 0.033, and these real-frame values are the same order: `t'` looks
prior-dominated rather than learned, in both arms.

**R-A is out independently.** `corr(N_eff, d)` is flat in both arms (−0.010 and +0.113;
per-seed −0.197…+0.151 and −0.108…+0.412). R-A requires N_eff to fall as d falls — the carrier
binding when the pair dimerises — and there is no such trend.

**R-B is out because it also needed `|t'|` to move.** The cage-route reading says ON carried
the d-trend through cage on-site geometry *instead of* the hub; but it presupposes that the
sign flip engaged the hub channel and the optimiser declined it. It did not engage.

## What this retires and what it leaves

Retired: the named-failure branch, and with it the licence to go to s+p. The A/B's
axial_red gain (+0.348 → +0.588, no negative cells, rmse_nbhd 47.7 → 43.5) is real but is not
evidence about the sign convention's mechanism — it was bought elsewhere.

Standing: the sign-structural claims, which have still never failed when measured directly
(hub–hub ≤ 0 in 6/6, the energy-channel inversion, M1b's base slope). What has failed, five
times now, is forecast phenomenology.

## Candidate causes, in the order we would test them

1. **Decay floor.** `t = t_min + softplus(B)` with the per-species decay length floored at
   0.3 Å. If the optimiser sits near that floor, `t'` is pinned by the decay prior whatever the
   energy channel asks. The agreement between profile slope (0.033) and real-frame values
   (0.044/0.062) is consistent with this and it is the cheapest to test — refit one cell with
   the floor lowered and the decay length free.
2. **Gradient path.** Λ reaches `t` only through the eigenvalue, and with a smeared
   near-degenerate spectrum `dΛ/dt` can be small. Measurable directly on a saved cell.
3. **Initialisation.** The amplitude was calibrated to place `t(2.85 Å) = 0.5 eV`; the
   *derivative* at 5.6 Å was never a calibration target. Adding `|t'|` to the calibration is
   trivial if 1 and 2 come back clean.

## One observation we cannot account for

`corr(lambda, d)` is **negative in ten of twelve cells** (−0.31 to −0.57): the defect level
falls as the pair separates, so `Delta_bind` *rises* with d. That is the reverse of the
physical expectation, and of what R-A predicted. It is equally present in both arms, so it is
not a sign effect and does not bear on the flip — but it is unexplained, and we would not want
`Delta_bind` relied on as a depth until it is.

## Outstanding

* **F2's closure-ratio update.** M3 was run only on the A/B cells, so F2's `t'` is still the
  profile slope and its 0.54 closure ratio has not been revised. One argument to the same
  script.
* **M2 attribution** (which elements carry the model's d-trend, both arms) — this is now the
  most informative remaining measurement, because it says where the ON fit gain *did* come
  from.
* **C1 post-μ-fix**, full three-component gate.

## What we are asking

1. Confirm R-C and the hold on s+p.
2. Which of the three causes to test first. Our order is decay floor → gradient path →
   initialisation, on cost and on the profile/real-frame agreement pointing at the floor.
3. Whether M2 should be run before the cause hunt. It answers "where did the fit gain come
   from", which is a different question from "why did t(d) not move", and the two together
   would give a complete picture of the ON arm.
