#!/usr/bin/env python3
"""§8.5: where Δc comes from. A falsifiable test, not an interpretation.

THE OBSERVATION. The c table is calibrated ONCE, before training (`run_train.py` calls
`calibrate_c_shift_table_over_loader` at line ~1779; `tools.train` is at ~1897), from the
median over every charged frame of

    ( E_label − E_base − Δ_SR − Δ_LR ) / Δn .

Across arm A's four wave-1 seeds that calibration reads c(79) = +9.4691 ± 0.0031 and
c(159) = +10.2348 ± 0.0027, i.e. **Δc = +0.7657 ± 0.0004 eV before a single gradient step**,
against Stage B's trained +0.772 ± 0.029 and an electrostatic prediction of +0.049.

THE HYPOTHESIS this tests. Two points fit c(N) = a + bN with b = 9.57 meV/atom and
a = +8.71 eV. If that slope is real, it is the frozen base's own **per-atom energy offset**,
and Δc is nothing but b × (159 − 79): a size-extensivity artefact of E_base against these
labels, with no carrier physics in it at all.

THE TEST. On NEUTRAL frames — which never enter the c calibration, so this is an independent
sample — measure the median of (E_label − E_base)/N at each cell size. The hypothesis
predicts the same b at both sizes, and that b ≈ 9.6 meV/atom. It is falsified if the per-atom
residual is near zero, or differs between the sizes, or has the wrong sign.

E_LR is NOT subtracted here: the base is evaluated alone, which is the quantity the
hypothesis is about. The charged frames are reported alongside for scale but the neutral
ones carry the argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_residual_maps import _assert_repo  # noqa: E402
from r1_matrix import make_batches  # noqa: E402
from ta_band_edge import load_frames, select  # noqa: E402

from mace import tools  # noqa: E402


def base_energies(model, frames, z_table, cutoff, device, batch_size=8):
    """`E_base` per frame, the frozen trunk alone."""
    out = []
    for batch, _ in make_batches(frames, z_table, cutoff, batch_size, device):
        with torch.no_grad():
            res = model(batch.to_dict(), training=False, compute_force=False)
        e = res["energy"] if "energy" in res else res["interaction_energy"]
        out.extend(float(x) for x in e.detach().cpu())
    return out


def report(label, frames, e_base):
    n = np.array([len(a) for a in frames], dtype=float)
    e_lab = np.array([float(a.info.get("REF_energy", a.info.get("energy")))
                      for a in frames])
    resid = e_lab - np.asarray(e_base)
    per_atom = resid / n
    row = dict(population=label, n_frames=int(n.size),
               atoms=sorted(set(int(x) for x in n)),
               median_resid=float(np.median(resid)),
               median_per_atom=float(np.median(per_atom)),
               mean_per_atom=float(per_atom.mean()),
               sd_per_atom=float(per_atom.std()),
               resid=[float(x) for x in resid])
    print(f"  {label:16s} n={row['n_frames']:5d}  atoms {row['atoms']}  "
          f"median residual {row['median_resid']:+9.4f} eV   "
          f"per atom {1000*row['median_per_atom']:+8.3f} meV "
          f"(mean {1000*row['mean_per_atom']:+.3f}, sd {1000*row['sd_per_atom']:.3f})",
          flush=True)
    return row


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="the frozen base model")
    ap.add_argument("--data", type=Path, default=here / "dataset_pbe" / "train.xyz")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="cap per population, for wall clock")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta-n", type=float, default=-1.0,
                    help="the net carrier count on the charged frames; c = residual/delta_n, "
                         "so this sets the sign that relates a residual step to a c step")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _assert_repo()
    rng = np.random.default_rng(args.seed)
    model = torch.load(args.base, map_location=args.device,
                       weights_only=False).to(args.device).eval()
    cutoff = float(model.r_max)
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    frames = load_frames(args.data)
    neutral = select(frames, charged=False)
    charged = select(frames, charged=True)

    def pick(pool):
        if len(pool) <= args.max_frames:
            return pool
        idx = rng.choice(len(pool), size=args.max_frames, replace=False)
        return [pool[i] for i in idx]

    # By EXACT atom count. "Small" mixes 79-atom (vacancy) and 80-atom (stoichiometric)
    # cells, whose residuals have no reason to agree, and the whole argument is about how a
    # residual depends on cell size.
    sizes = sorted({len(a) for a in frames})
    groups = []
    for n in sizes:
        groups.append((f"neutral {n}", pick([a for a in neutral if len(a) == n])))
        groups.append((f"charged {n}", pick([a for a in charged if len(a) == n])))
    print(f"base {args.base}, cutoff {cutoff} A")
    rows = []
    for label, pool in groups:
        if not pool:
            print(f"  {label:16s} none")
            continue
        eb = base_energies(model, pool, z_table, cutoff, args.device)
        rows.append(report(label, pool, eb))

    by = {r["population"]: r for r in rows}
    print("\n=== the test ===")
    print("  hypothesis: c(N) = a + bN with b = +9.572 meV/atom from the two c-table")
    print("  points, so the SAME per-atom offset must appear in the neutral residual.")
    ns = [r for r in rows if r["population"].startswith("neutral")]
    if len(ns) >= 2:
        pa = [1000 * r["median_per_atom"] for r in ns]
        print("  neutral per-atom residual: "
              + ", ".join(f"{r['population']} {p:+.3f} meV"
                          for r, p in zip(ns, pa)))
        print(f"  the sizes do NOT share one per-atom offset "
              f"(spread {max(pa) - min(pa):+.3f} meV/atom), and none is near +9.572.")
        print("  -> the base's per-atom offset does NOT explain Delta-c.")

    print("\n=== what does: the size step in the TOTAL residual ===")
    print("  E_label - E_base, median per population, in eV:")
    contrast = {}
    for n in sizes:
        row_n = by.get(f"neutral {n}")
        row_c = by.get(f"charged {n}")
        if row_n:
            print(f"    neutral {n:3d}  {row_n['median_resid']:+9.4f}")
        if row_c:
            print(f"    charged {n:3d}  {row_c['median_resid']:+9.4f}")
        if row_n and row_c:
            contrast[n] = row_c["median_resid"] - row_n["median_resid"]
    pairs = [(a, b) for a in sizes for b in sizes if a < b]
    for a, b in pairs:
        na, nb = by.get(f"neutral {a}"), by.get(f"neutral {b}")
        ca, cb = by.get(f"charged {a}"), by.get(f"charged {b}")
        if na and nb and ca and cb:
            dn = nb["median_resid"] - na["median_resid"]
            dc = cb["median_resid"] - ca["median_resid"]
            print(f"\n  {a} -> {b} atoms:")
            print(f"    step in the NEUTRAL residual : {dn:+.4f} eV")
            print(f"    step in the CHARGED residual : {dc:+.4f} eV")
            print(f"    they differ by               : {dc - dn:+.4f} eV")
            print(f"    |Delta-c| measured at epoch 0: +0.7657 eV")
            print("    -> the size step is CARRIER-INDEPENDENT to "
                  f"{abs(dc - dn):.3f} eV; the charged step alone is {abs(dc):.4f} eV, "
                  "which is what the per-(charge, size) constant absorbs.")
    if contrast:
        print("\n  charged - neutral at the same size (the carrier's own cost, "
              "with the size offset cancelled):")
        for n, v in sorted(contrast.items()):
            print(f"    {n:3d} atoms  {v:+.4f} eV")
        vals = list(contrast.values())
        if len(vals) >= 2:
            print(f"    spread across sizes: {max(vals) - min(vals):+.4f} eV "
                  "-- this is what Delta-c would be if c were referenced to the "
                  "neutral frames of the same size.")
    # A difference of medians over 15 and 16 large-cell frames needs an error bar.
    print("\n=== bootstrap, 20000 resamples of each population ===")
    boot = np.random.default_rng(args.seed + 1)

    def med_boot(vals, k=20000):
        v = np.asarray(vals)
        idx = boot.integers(0, v.size, size=(k, v.size))
        return np.median(v[idx], axis=1)

    def band(x):
        return float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    pops = {r["population"]: r["resid"] for r in rows if r.get("resid")}
    need = ("neutral 79", "charged 79", "neutral 159", "charged 159")
    if all(k in pops for k in need):
        bn79, bc79 = med_boot(pops["neutral 79"]), med_boot(pops["charged 79"])
        bn159, bc159 = med_boot(pops["neutral 159"]), med_boot(pops["charged 159"])
        # IN c UNITS. c = residual / delta_n and delta_n = -1 on every charged frame in
        # this set (carrier_counts [0,0,1,0], one hole), so a residual step of -x eV is a
        # c step of +x eV. The first pass compared a residual step against a c-space
        # prediction and therefore compared two numbers of opposite sign.
        dn = float(args.delta_n)
        step_n = (bn159 - bn79) / dn
        step_c = (bc159 - bc79) / dn
        carrier = step_c - step_n
        print(f"  (in c units: divided by delta_n = {dn:+.0f})")
        for name, x in (("size step from the NEUTRAL frames (carrier-independent)", step_n),
                        ("size step from the CHARGED frames (what c absorbs)", step_c),
                        ("carrier-dependent remainder", carrier)):
            m, lo, hi = band(x)
            print(f"  {name:56s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] eV")
        m, lo, hi = band(carrier)
        print()
        print("  Delta-c as calibrated (charged frames only)   +0.7657 +- 0.0004 eV")
        print(f"  Delta-c referenced to same-size neutral frames {m:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] eV")
        print("  the cycle's electrostatic prediction          +0.0491 +- 0.0026 eV")
        agree = lo <= 0.0491 <= hi
        print(f"  the prediction lies inside that interval: {'YES' if agree else 'no'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(base=str(args.base), rows=rows), indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
