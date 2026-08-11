#!/usr/bin/env python3
"""Does ``carrier^2`` recover ``N^(-1/3)`` once the attention is not collapsed?

With the ``host.carrier`` cross term removed, ``delta_lr`` reduces to ``carrier^2`` alone --
for a single charged carrier, essentially the Madelung self-energy of that charge in its
compensating background, which decays as ``1/L``, i.e. ``N^(-1/3)``.

Measured on the trained long-range model it decays as ``N^(-0.999)`` instead. That is a
diagnostic of the **attention**, not of the repartition: weight spread over a sublattice
makes ``sum_i q_i^2`` go as ``1/N`` regardless of what the Ewald sum does. The two causes
are separable without retraining, by hand-setting the attention and leaving everything else
untouched:

* ``alpha`` uniform over the Cs sublattice -- the collapsed pattern, expected ``~1/N``;
* ``alpha`` on the two under-coordinated Pb -- the pattern ``perov_nolr_s1`` finds unaided,
  expected ``~N^(-1/3)``.

Two things have to be got right for that test to mean anything.

**Isotropic cells only.** The Madelung constant depends on cell SHAPE, so a ladder mixing
2,2,2 with 3,2,2 and 3,3,2 compares differently-shaped cells and no single power law holds
across it. Only whole multiples of the primitive are used here.

**The right functional form.** ``carrier^2`` is not a pure image term for a carrier spread
over more than one site: it is

    ``E(L) = E_internal + E_image(L)``,   ``E_image ~ -q^2 alpha_M / (2 eps L)``

where ``E_internal`` is the Coulomb energy *within* the charge distribution and is
size-independent. For the two-Pb pattern that internal repulsion is positive and of order
0.1 eV, so ``|E|`` itself neither decays nor follows a power law -- it approaches a positive
constant from below. Fitting ``d(ln|E|)/d(ln N)`` to it returns nonsense (+2.29 measured).
What must scale as ``1/L`` is the **N-dependent part**, so the fit is ``E = E_inf + k/L``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst
from host_carrier_probe import build, shell_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path,
                        default=Path.home() / "runs/perov_lr_s1/perov_lr_s1.model")
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    # Isotropic only: same shape, different size, so alpha_M is constant across the set.
    parser.add_argument("--repeats", nargs="+",
                        default=["2,2,2", "3,3,3", "4,4,4"])
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    model = torch.load(args.model, map_location="cpu", weights_only=False).double().eval()
    ewald = model.latent_ewald
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))
    pristine = pst.find_pristine(args.data_dir)
    site_index = list(pst.classify_sites(pristine).values())[0][0]

    print(f"model {args.model.name}\n")
    rows = []
    for text in args.repeats:
        supercell = pristine.repeat(pst.parse_repeat(text))
        pst.set_state(supercell, pst.PRISTINE)
        defect, _ = pst.make_vacancy(supercell, site_index)
        pst.set_state(defect, pst.VACANCY_QP1)
        batch = build(defect, z_table, mace_data, torch_geometric)
        shell, symbols = shell_indices(defect)
        caesium = np.flatnonzero(symbols == "Cs")
        with torch.no_grad():
            out = model(batch, training=False, compute_force=False)
        amplitude = float(out["screening_amplitude"].reshape(-1)[0])
        positions, cell = batch["positions"], batch["cell"].view(-1, 3, 3)
        index = batch["batch"]

        record = {"N": len(defect), "L": float(torch.det(cell[0])) ** (1.0 / 3.0)}
        for name, sites in (("Pb shell", shell), ("Cs uniform", caesium)):
            weights = np.zeros(len(defect))
            weights[sites] = 1.0 / len(sites)
            charge = amplitude * torch.tensor(weights)
            with torch.no_grad():
                record[name] = float(ewald.energy(charge, positions, cell, index).sum())
            record[f"ss[{name}]"] = float((charge**2).sum())
        record["net"] = amplitude
        rows.append(record)
        print(f"  N = {len(defect):5d}   done")

    sizes = np.array([r["N"] for r in rows], float)
    inverse_l = np.array([1.0 / r["L"] for r in rows])
    print("\n\ncarrier^2 = E_LR(q_carrier) alone, both attention patterns (eV)\n")
    print(f"{'N':>6s} {'L (A)':>8s} {'1/L':>8s} {'Pb shell':>13s} {'Cs uniform':>13s} "
          f"{'ss(Pb)':>10s} {'ss(Cs)':>10s}")
    for r in rows:
        print(f"{r['N']:6d} {r['L']:8.2f} {1.0 / r['L']:8.5f} {r['Pb shell']:13.6f} "
              f"{r['Cs uniform']:13.6f} {r['ss[Pb shell]']:10.5f} "
              f"{r['ss[Cs uniform]']:10.5f}")

    # E = E_inf + k/L. A localised carrier's image term is the only N-dependent piece;
    # the internal Coulomb energy of the distribution is a size-independent offset.
    localised = np.array([r["Pb shell"] for r in rows])
    slope, intercept = np.polyfit(inverse_l, localised, 1)
    residual = localised - (slope * inverse_l + intercept)
    print(f"\n\nFIT  E = E_inf + k/L   on the localised (Pb shell) pattern\n")
    print(f"  E_inf = {intercept:+.6f} eV      (internal Coulomb energy, size-independent)")
    print(f"  k     = {slope:+.6f} eV.A     (image term; negative for a screened monopole)")
    print(f"  max residual = {np.abs(residual).max() * 1e3:.3f} meV over "
          f"{len(rows)} cells, L = {rows[0]['L']:.1f}-{rows[-1]['L']:.1f} A")
    print(f"\n  image contribution at each L: "
          + ", ".join(f"{slope * x * 1e3:+.1f}" for x in inverse_l) + " meV")

    delocalised = np.array([abs(r["Cs uniform"]) for r in rows])
    exponent = np.polyfit(np.log(sizes), np.log(delocalised), 1)[0]
    print(f"\n  For contrast, the collapsed pattern: d(ln|E|)/d(ln N) = {exponent:+.3f}")
    print("  (sum q^2 ~ 1/N when the weight is spread over a sublattice)")

    print()
    if np.abs(residual).max() < 5e-3 and slope < 0:
        print("  PASS: the N-dependence of carrier^2 is a 1/L image term with a negative")
        print("  coefficient, to within a few meV. The repartition is sound; the -0.999")
        print("  seen on the trained model is the attention, not the branch.")
    else:
        print(f"  FAIL: residual {np.abs(residual).max() * 1e3:.1f} meV, k = {slope:+.4f}.")
        print("  Expected a clean 1/L with k < 0. Resolve before retraining.")


if __name__ == "__main__":
    main()
