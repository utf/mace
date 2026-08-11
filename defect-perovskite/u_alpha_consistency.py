#!/usr/bin/env python3
"""Does ``u`` know where the carrier is bound, even when ``alpha`` does not?

The proposed consistency constraint pulls ``alpha`` toward ``softmax(-beta u)``. That only
helps if ``u`` carries the answer while ``alpha`` runs away. The control established that
``alpha`` had site structure and lost it -- but ``gap_site`` is computed on the LOGITS, so
nothing in it says anything about ``u``. This measures ``u`` directly.

Reported per model:

* ``gap_site(u)`` -- mean over species of the within-species spread of ``u``, the same
  species-blind measure used to gate the seed anneal, applied to the site energies;
* **Pb-shell contrast** -- ``<u>`` on the two under-coordinated Pb minus ``<u>`` on the
  remaining Pb. This is the quantity the constraint would pull toward, so it is the one that
  matters, and it is signed: negative means the shell is more strongly bound, which is the
  physically correct sign for a bound hole;
* ``<u>`` per species, and where ``alpha`` actually sits, for reference.

Band-edge referencing defines ``u = 0`` as the band edge, so a bound state has ``u < 0`` and
a delocalised band state has ``u -> 0``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import perovskite_size_test as pst
from host_carrier_probe import build, shell_indices

LIVE = 2  # h_maj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=here / "dataset_pbe")
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    from ase.io import read
    from mace import data as mace_data
    from mace import tools
    from mace.tools import torch_geometric

    torch.set_default_dtype(torch.float64)
    z_table = tools.AtomicNumberTable(sorted({17, 55, 82}))

    # Real frames carry a vacancy whose position is not recorded, so the shell is taken from
    # a constructed defect instead: same primitive, known site, directly comparable.
    pristine = pst.find_pristine(args.data_dir)
    site_index = list(pst.classify_sites(pristine).values())[0][0]
    supercell = pristine.copy()
    pst.set_state(supercell, pst.PRISTINE)
    defect, _ = pst.make_vacancy(supercell, site_index)
    pst.set_state(defect, pst.VACANCY_QP1)
    shell, symbols = shell_indices(defect)
    print(f"constructed V_Cl+ cell: {len(defect)} atoms, shell = {len(shell)} Pb "
          f"at indices {shell.tolist()}\n")

    print(f"{'model':>22s} {'partic':>8s} {'alpha on':>9s} {'gap_site(u)':>12s} "
          f"{'shell contrast':>15s} {'<u> shell':>10s} {'<u> Pb':>9s} {'<u> Cs':>9s} "
          f"{'<u> Cl':>9s}")
    for path in args.models:
        model = torch.load(path, map_location="cpu", weights_only=False).double().eval()
        batch = build(defect, z_table, mace_data, torch_geometric)
        with torch.no_grad():
            out = model(batch, training=False, compute_force=False)
        alpha = out["carrier_alpha"][:, LIVE].numpy()
        u = out["carrier_readouts"][:, LIVE].numpy()

        # gap_site on u: mean over species of the within-species std.
        spreads = []
        for species in ("Cs", "Pb", "Cl"):
            mask = symbols == species
            if mask.sum() > 1:
                spreads.append(u[mask].std())
        gap_site_u = float(np.mean(spreads))

        lead = symbols == "Pb"
        other_pb = lead.copy()
        other_pb[shell] = False
        contrast = float(u[shell].mean() - u[other_pb].mean())
        # What the proposed constraint would actually target.
        beta = float(getattr(model.carrier_pooling, "beta", 10.0))
        target = torch.softmax(torch.tensor(-beta * u), dim=0).numpy()
        target_partic = 1.0 / (target ** 2).sum()
        target_where = max(
            (("Cs", target[symbols == "Cs"].sum()),
             ("Pb", target[lead].sum()),
             ("Cl", target[symbols == "Cl"].sum())),
            key=lambda kv: kv[1],
        )
        target_shell = target[shell].sum()
        where = max(
            (("Cs", alpha[symbols == "Cs"].sum()),
             ("Pb", alpha[lead].sum()),
             ("Cl", alpha[symbols == "Cl"].sum())),
            key=lambda kv: kv[1],
        )[0]
        print(f"{path.parent.name:>22s} {1.0 / (alpha ** 2).sum():8.2f} {where:>9s} "
              f"{gap_site_u:12.4f} {contrast:+15.4f} {u[shell].mean():10.4f} "
              f"{u[other_pb].mean():9.4f} {u[symbols == 'Cs'].mean():9.4f} "
              f"{u[symbols == 'Cl'].mean():9.4f}")
        print(f"{'  -> softmax(-beta u)':>22s} {target_partic:8.2f} "
              f"{target_where[0]:>9s} {'':12s} "
              f"mass on shell {target_shell:.2e}   alpha on shell "
              f"{alpha[shell].sum():.4f}   (beta = {beta:g})")

    print("\n  shell contrast = <u>(2 under-coordinated Pb) - <u>(other Pb).")
    print("  NEGATIVE means the shell is more strongly bound, the correct sign for a bound")
    print("  hole under band-edge referencing (u = 0 is the band edge, bound states u < 0).")
    print("\n  If the collapsed model retains shell contrast comparable to the localised")
    print("  ones, u carries the answer and alpha is ignoring it -- the consistency")
    print("  constraint has something real to pull toward. If it does not, there is")
    print("  nothing to pull toward and the diagnosis moves upstream.")


if __name__ == "__main__":
    main()
