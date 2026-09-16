"""Plan P, Phase 0 prerequisite: the base-v2-relaxed pristine 80-atom cell (fixed cell, BFGS).

    python defect-perovskite/p0_relax_pristine.py --out ~/runs/dscc/p0/pristine_relaxed_v2.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.optimize import BFGS
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace.calculators import MACECalculator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/home/alex/runs/base_v2_prod/base_v2_prod_base.pt")
ap.add_argument("--static_cell", default=str(HERE / "dataset_pbe" / "static_pristine_cell.json"))
ap.add_argument("--fmax", type=float, default=0.002)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
sc = json.load(open(args.static_cell))
atoms = Atoms(numbers=sc["numbers"], positions=sc["positions"], cell=sc["cell"], pbc=True)
model = torch.load(args.base, weights_only=False, map_location=args.device).double()
atoms.calc = MACECalculator(models=[model], device=args.device, default_dtype="float64")
e0 = atoms.get_potential_energy(); f0 = np.abs(atoms.get_forces()).max()
t = time.time()
opt = BFGS(atoms, logfile=str(Path(args.out).with_suffix(".log")))
opt.run(fmax=args.fmax, steps=500)
e1 = atoms.get_potential_energy(); f1 = np.abs(atoms.get_forces()).max()
disp = atoms.get_positions() - np.asarray(sc["positions"])
json.dump({"numbers": sc["numbers"], "cell": np.asarray(sc["cell"]).tolist(), "positions": atoms.get_positions().tolist(),
           "positions_static": sc["positions"], "E_static": e0, "E_relaxed": e1, "fmax_static": float(f0),
           "fmax_relaxed": float(f1), "steps": opt.get_number_of_steps(), "rms_disp_A": float(np.sqrt((disp**2).sum(1).mean())),
           "max_disp_A": float(np.sqrt((disp**2).sum(1)).max()), "base": args.base, "seconds": time.time() - t},
          open(args.out, "w"), indent=1)
print(f"E {e0:.4f} -> {e1:.4f} eV, fmax {f0:.4f} -> {f1:.4f}, steps {opt.get_number_of_steps()}, rms disp {np.sqrt((disp**2).sum(1).mean()):.4f} A, {time.time()-t:.0f}s")
