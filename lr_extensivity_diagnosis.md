# E_LR extensivity: the proposed background fix is refuted; the fault is `q^pol`

Measured 2026-08-11 on `perov_lr_s1`, V_Cl+ in CsPbCl3, ladder 639 → 2159 atoms.
Scripts: `defect-perovskite/lr_term_scaling.py`, `pol_profile.py`, `perov_alpha_check.py`.

---

## 1. The proposed fix cannot work, by three orders of magnitude

LES is **not a split Ewald**. `les/module/ewald.py` computes a pure reciprocal-space sum
over `k != 0` of Gaussian-smeared charges — there is no real-space part, and `sigma` is the
physical smearing, not a convergence parameter. Dropping `k = 0` for a net-charged cell
omits the finite `k -> 0` remainder of `exp(-sigma^2 k^2/2)/k^2 * Q^2`, namely

```
E_bg = -(2 pi / V) (sigma^2 / 2) Q^2 * C  =  -pi sigma^2 Q^2 C / V
```

which is the co-advisor's `-pi Q^2 / (2 V alpha^2)` under `alpha = 1/(sigma sqrt 2)`. The
algebra agrees. But the term is **O(1/V)**, so it *decays* as `1/N`:

```
     N       volume           E_bg       delta_lr        ratio
   639      23454.8      -0.003030       -23.1783     1.31e-04
  2159      79160.0      -0.000898       -75.5817     1.19e-05

  E_bg scaling  d(ln|.|)/d(ln N) = -0.999
```

Three milli-electronvolts against seventy-five electronvolts. A term that decays as `1/N`
cannot cancel one that grows as `N`. The parenthetical "extensive (`V ∝ N`)" in the
proposal is a slip: `Q²/V` with `V ∝ N` is `1/N`.

**This does not mean the term should never be added** — it is genuinely missing and should
be supplied for correctness at the meV level. It is simply not this bug.

## 2. The actual fault, by decomposition

`delta_lr = E_LR(q_host + q_pol + q_carrier) - E_LR(q_host)`. The host self-energy — the
genuinely extensive and physically correct Madelung energy of the lattice — cancels exactly
by construction. Five terms survive, each obtained by the polarisation identity:

```
     N       delta_lr           pol2       carrier2    pol_carrier       host_pol   host_carrier
   639      -23.17828       -6.35974       -0.00652        0.15830      -15.06461       -1.90571
   959      -34.18992       -9.54022       -0.00435        0.15775      -22.89733       -1.90577
  1439      -50.73401      -14.32901       -0.00290        0.15739      -34.65367       -1.90581
  2159      -75.58170      -21.53530       -0.00193        0.15714      -52.29577       -1.90584

scaling         +0.971         +1.002         -0.999         -0.006         +1.022         +0.000
```

* **`pol^2` (+1.002) and `host.pol` (+1.022) are the extensive terms.** Together they are
  −21.4 of −23.2 eV at N = 639 and −73.8 of −75.6 eV at N = 2159: **97 % of the effect.**
* **Every carrier term is clean.** `host.carrier` is constant to five decimals
  (−1.90584, scaling +0.000), `pol.carrier` is constant (−0.006), `carrier^2` decays
  (−0.999).

The channel that was designed with care is fine. The channel with no locality constraint is
the bug.

## 3. Why: `q^pol` is a fictitious ionic lattice

`q^pol_i = (sum_c n_c) * (MLP(node_feats_i, counter_emb) - mean)`. Mean subtraction makes
`sum_i q^pol_i = 0` exactly — which the docstring cites as the safeguard — but **neutrality
is not localisation**. Far from the defect the node features are the bulk values for the
species, so the MLP returns a fixed per-species number and the subtraction removes only the
composition-weighted average. Each species keeps a fixed residual on every atom:

```
N = 639    sum q_pol = -1.05e-13   sum q_pol^2 = 2.2180
   species         <3 A        3-6 A       6-10 A        >10 A   bulk spread
        Cs            -     -0.06718     -0.06647     -0.06654      0.000757
        Pb      0.16798      0.10981      0.10888      0.10924      0.002473
        Cl            -      0.00210     -0.01532     -0.01506      0.003100

N = 1439   sum q_pol = -2.11e-15   sum q_pol^2 = 4.9499
        Cs            -     -0.06688     -0.06617     -0.06624      0.000763
        Pb      0.16828      0.11010      0.10918      0.10952      0.002485
        Cl            -      0.00240     -0.01502     -0.01480      0.003090
```

Flat from 3 Å to beyond 10 Å, with a per-atom spread of ~0.001–0.003 e, and **plateau values
identical at both cell sizes**. This is an extra ionic lattice of ±0.07–0.11 e superimposed
on the crystal whenever `sum_c n_c != 0`. Its own Madelung energy (`pol^2`) and its
interaction with the host lattice (`host.pol`) both scale with N, exactly as measured
(`sum_i q_pol^2`: 2.218 → 4.950 for N 639 → 1439, ratio 2.23 against N ratio 2.25).

The only genuine local response is the near-shell Pb: +0.168 against a +0.109 plateau, i.e.
~0.06 e of real physics sitting on tens of eV of artefact.

## 4. Correction to the co-advisor's §3 — it is one bug, not two

§3 states the SiC growth was `q^pol` and the `host.pol` cross term, "a distinct structural
gap", and predicts the perovskite fix will not move it. The mechanism named there is
correct — and it is **the same mechanism measured here**. The two systems do not have
different faults; the perovskite is the loud version of the SiC one, because
`sum_c n_c` is larger and the host lattice it couples to is ionic rather than covalent.

So my report's claim that SiC and perovskite share a fault was right, and the specific
attribution in both documents (mine: missing background; the reply's: two separate causes)
was wrong. The prediction to test in §6 step 3 stands, but with the opposite expectation:
**fixing `q^pol` should fix both ladders**, and adding the background term should move
neither by more than a few meV.

## 5. Separate finding: the size hinge does not localise `alpha`, and my previous report
misstated it

`defect-perovskite/perov_alpha_check.py` had a stale call signature and crashed with a
`TypeError` on every invocation. `run_overnight.sh` swallowed it (`2>/dev/null`,
`|| continue`), the report's attention section fell back to "(n/a)", and the figures I
quoted — `partic 2.02`, `alpha on shell 1.0000 at every cell size` — came from the training
log, i.e. from **80-atom training frames**, not from the ladder. **Withdraw the claim that
alpha on the vacancy shell is 1.0000 at every cell size; it was never measured.**

Repaired and run, the ladder says the opposite:

```
     N  shell          e_maj          e_min          h_maj          h_min
              partic  a_shell partic  a_shell partic  a_shell partic  a_shell
   639      2   638.3  0.003   636.0  0.003   127.6  0.000   635.9  0.003
   959      2   957.9  0.002   954.5  0.002   191.3  0.000   954.3  0.002
  1439      2  1437.4  0.001  1432.2  0.001   287.0  0.000  1431.9  0.001
  2159      2  2156.6  0.001  2148.7  0.001   430.5  0.000  2148.4  0.001
```

The live channel `h_maj` has participation at a constant **20 % of the cell** at every size,
and `alpha` on the vacancy shell is **0.0000**, not 1.0000. The dead channels are fully
uniform (participation = N).

Why the hinge reads healthy anyway: it constrains the log-ratio of a *pooled* quantity, and
`<u>` is uniform enough that the constraint is satisfied trivially. `<u>_h_maj` is
**0.1269 eV at every cell size, drift −0.0 meV** — which is why `delta_SR` is exactly
size-independent and the `nolr` ladder is flat. The short-range branch is genuinely safe;
the hinge is simply not evidence about where `alpha` sits.

This matters for the fix below, because gating `q^pol` by `alpha` would inherit a localiser
that is itself diluting.

## 6. What the fix has to be

The requirement `q^pol` is missing is **locality**, not neutrality: it must vanish for atoms
whose environment is bulk. Three candidates, in order of how much they disturb the
architecture:

1. **Reference-subtract the polarisation readout per species.** Subtract the value the same
   MLP returns on a bulk environment of that species, so a bulk atom gets identically zero
   and only the perturbed shell survives. Cheap, exactly targets the measured artefact, and
   needs no new loss term — but requires a per-species bulk descriptor to be defined and
   carried.
2. **Penalise `sum_i (q^pol_i)^2` directly**, as a size hinge in charge space rather than
   energy space. Uses machinery that already exists, but is a soft constraint on something
   that should be structural.
3. **Gate by a locality envelope.** Correct in principle; blocked in practice until
   `alpha`'s dilution (§5) is resolved, since `alpha` is the only envelope available.

Recommendation: (1), with (2) as a diagnostic rather than a loss.

## 7. Revised order of work

1. Fix `q^pol` locality. Re-run the perovskite ladder: gate is `pol^2` and `host.pol`
   scaling going from `+1.0` to `<= 0`, and `|delta_lr|` falling from tens of eV to the
   order of `q^2 alpha_M / 2 eps L`.
2. Add the `k = 0` background term for correctness. Expect a few meV; it is not a gate.
3. Re-run the SiC ladder. Expectation now: **the same fix resolves it.**
4. Repair `alpha` dilution (§5) — separate from extensivity, but it means `q^carrier`'s
   shape is smeared over 20 % of the cell, which is wrong even though it is not what makes
   `E_LR` extensive.
5. Only then site-averaging, DFPT `eps_inf`, real `E_VBM`, seed replication.
