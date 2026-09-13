"""Relax a charged defect cell with a D-SCC model (ASE calculator + BFGS).

The head short-circuits to the base in the reference state, so the NEUTRAL cell relaxes
identically under every arm -- only the charged relaxation distinguishes them. Energies are
reported from the calibrated checkpoint when one exists (`model_calibrated.pt`), so they are on
the DFT scale; forces are unaffected by `C_Q` either way.

    python defect-perovskite/dscc_relax.py --model .../model.pt --counts 0 0 1 0 --charge 1 \
        --tiling 1,1,2 --out relaxed_w6.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                                  # noqa: E402
from mace.modules.dscc import data as dd                                # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                      # noqa: E402
from mace.tools import torch_geometric                                  # noqa: E402


class DSCCCalculator(Calculator):
    """Energy and forces from a `MACEDSCC` model at a fixed charge state."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, model, counts, charge, device="cuda", **kw):
        super().__init__(**kw)
        self.model = model
        self.counts = np.asarray(counts, dtype=int)
        self.charge = int(charge)
        self.device = device
        self.z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.n_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy()
        a.info.update({"carrier_counts": self.counts, "cell_charge": self.charge,
                       "config_type": "relax", "source_dir": "relax"})
        ds = dd.atomic_data([a], self.z_table, self.model.r_cut)
        batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(self.device).to_dict()
        out = self.model(batch, compute_force=True)
        self.n_calls += 1
        self.results["energy"] = float(out["energy"][0])
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = out["forces"].detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--counts", nargs=4, type=int, default=[0, 0, 1, 0])
    ap.add_argument("--charge", type=int, default=1)
    ap.add_argument("--tiling", default="1,1,2")
    ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    path = Path(args.model)
    calibrated = path.parent / "model_calibrated.pt"
    use = calibrated if calibrated.exists() else path
    model = torch.load(use, weights_only=False, map_location=args.device).to(args.device).eval()

    doc = json.load(open(args.static_cell))
    unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
    t = tuple(int(x) for x in args.tiling.split(","))
    atoms = tiling_ladder(unit, [t], vacancy_species=17, vacancy_index=0)[0]
    start = atoms.copy()
    atoms.calc = DSCCCalculator(model, args.counts, args.charge, device=args.device)

    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    t0 = time.time()
    # Positions only: BFGS on the Atoms object never touches the cell (no cell filter).
    opt = BFGS(atoms, logfile="-")
    opt.run(fmax=args.fmax, steps=args.steps)
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    disp = atoms.get_positions() - start.get_positions()
    disp -= np.round(disp @ np.linalg.inv(np.array(atoms.get_cell()))) @ np.array(atoms.get_cell())
    rec = {"model": str(use), "counts": list(map(int, args.counts)), "charge": args.charge,
           "tiling": list(t), "n_atoms": len(atoms), "calibrated": bool(calibrated.exists()),
           "E_start": e0, "E_relaxed": e1, "relaxation_energy": e1 - e0,
           "fmax_start": float(np.linalg.norm(f0, axis=1).max()),
           "fmax_end": float(np.linalg.norm(f1, axis=1).max()),
           "converged": bool(np.linalg.norm(f1, axis=1).max() <= args.fmax),
           "steps": int(opt.get_number_of_steps()), "energy_calls": atoms.calc.n_calls,
           "seconds": time.time() - t0,
           "max_displacement": float(np.linalg.norm(disp, axis=1).max()),
           "rms_displacement": float(np.sqrt((disp ** 2).sum(1).mean())),
           "positions": atoms.get_positions().tolist(),
           "start_positions": start.get_positions().tolist(),
           "numbers": [int(z) for z in atoms.get_atomic_numbers()],
           "cell": np.array(atoms.get_cell()).tolist()}
    json.dump(rec, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("positions", "start_positions", "numbers", "cell")}, indent=1))


if __name__ == "__main__":
    main()
