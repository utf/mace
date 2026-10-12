"""Plan P Phase 3a §2: the pristine 80-atom cell relaxed with the neutral production model (base v8 w01 + E_PP,
read through the bw5 checkpoint at zero carriers), fixed cell, BFGS to fmax 0.002 eV/A; tilt recorded.

    python defect-perovskite/p3a_relax_pristine.py --out ~/runs/dscc/p3a/pristine_relaxed_v8.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from p3a_common import BASE_V8, BW5_S0, Base, Neutral, load_cell, tilt_stats   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=BW5_S0); ap.add_argument("--base", default=BASE_V8)
ap.add_argument("--static_cell", default=str(HERE / "static_pristine_cell.json"))
ap.add_argument("--fmax", type=float, default=0.002); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64)


class NeutralCalc(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, neutral, **kw):
        super().__init__(**kw); self.n = neutral; self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, epp, f = self.n.energy_forces(self.atoms); self.calls += 1
        self.results.update({"energy": e, "free_energy": e, "forces": f, "e_pp": epp})


atoms = load_cell(args.static_cell); static = atoms.get_positions().copy()
neutral = Neutral(args.checkpoint, args.device); base = Base(args.base, args.device)
atoms.calc = NeutralCalc(neutral)
e0 = atoms.get_potential_energy(); f0 = float(np.abs(atoms.get_forces()).max()); epp0 = atoms.calc.results["e_pp"]
t = time.time(); opt = BFGS(atoms, logfile=str(Path(args.out).expanduser().with_suffix(".log"))); opt.run(fmax=args.fmax, steps=1000)
e1 = atoms.get_potential_energy(); f1 = float(np.abs(atoms.get_forces()).max()); epp1 = atoms.calc.results["e_pp"]
fb = base.forces(atoms); fpp = atoms.get_forces() - fb
disp = atoms.get_positions() - static
res = {"numbers": atoms.get_atomic_numbers().tolist(), "cell": np.asarray(atoms.get_cell()).tolist(), "positions": atoms.get_positions().tolist(),
       "positions_static": static.tolist(), "model": "base_v8_w01 + E_PP (bw5 s0 checkpoint, zero carriers)", "checkpoint": args.checkpoint,
       "base": args.base, "fmax_target": args.fmax, "E_static": e0, "E_relaxed": e1, "E_PP_static": epp0, "E_PP_relaxed": epp1,
       "fmax_static": f0, "fmax_relaxed": f1, "fmax_base_alone_at_relaxed": float(np.abs(fb).max()),
       "rms_F_PP_at_relaxed_meV_A": 1e3 * float(np.sqrt((fpp ** 2).mean())), "steps": opt.get_number_of_steps(), "calls": atoms.calc.calls,
       "rms_disp_A": float(np.sqrt((disp ** 2).sum(1).mean())), "max_disp_A": float(np.sqrt((disp ** 2).sum(1)).max()),
       "tilt_static": tilt_stats(load_cell(args.static_cell)), "tilt_relaxed": tilt_stats(atoms), "seconds": time.time() - t}
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print(f"E {e0:.4f} -> {e1:.4f} eV; fmax {f0:.4f} -> {f1:.4f} (base alone at relaxed {res['fmax_base_alone_at_relaxed']:.4f}; rms F_PP {res['rms_F_PP_at_relaxed_meV_A']:.2f} meV/A); "
      f"steps {res['steps']}; rms disp {res['rms_disp_A']:.4f} A; tilt (180 - mean Pb-Cl-Pb) static {res['tilt_static']['tilt_from_180_mean_deg']:.2f} -> relaxed {res['tilt_relaxed']['tilt_from_180_mean_deg']:.2f} deg; {res['seconds']:.0f}s")
