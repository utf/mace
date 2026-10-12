"""Plan P Phase 3a §2.2a: Gamma Hessian of the relaxed pristine cell with base v8 w01 alone (h = 0.01 A, ASR), the
base's own Z* and eps_inf (autograd; equal to the finite difference of its summed node dipoles), eps0_LST
polycrystalline at eps_inf = 4.0, at the base's own eps_inf tensor, and at 3.667 (the P1 gate-2 value on the base-v2 cell).

    python defect-perovskite/p3a_lst.py --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json --out ~/runs/dscc/p3a/p3a_lst.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace.modules.dscc.ewald import COULOMB                                     # noqa: E402
from p3a_common import BASE_V8, MEV, Base, gamma_hessian, gamma_modes, load_cell  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--base", default=BASE_V8); ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64); t = time.time()
atoms = load_cell(args.cell); N = len(atoms); numbers = atoms.get_atomic_numbers(); Omega = abs(np.linalg.det(np.array(atoms.get_cell())))
B = Base(args.base, args.device)
f0 = B.forces(atoms)
FC = gamma_hessian(B.forces, atoms, args.h, log=lambda s: print(s, flush=True))
w2, e, M = gamma_modes(FC, numbers)
order = np.argsort(np.abs(w2)); acoustic = set(order[:3].tolist()); keep = [m for m in range(3 * N) if m not in acoustic]
Z, eps_inf = B.response(atoms)
CONV = 4 * math.pi * COULOMB / Omega
ionic = np.zeros((3, 3)); modes = []
for m in keep:
    if w2[m] <= 0:
        continue
    vec = e[:, m].reshape(N, 3) / np.sqrt(M)[:, None]; zt = np.einsum("kab,kb->a", Z, vec)
    c = np.outer(zt, zt) / w2[m]; ionic += c
    modes.append({"m": int(m), "omega_meV": float(math.sqrt(w2[m]) * MEV), "Ztilde": zt.tolist(), "eps_contrib_poly": float(np.trace(c) / 3 * CONV),
                  "eps_contrib_diag": (np.diag(c) * CONV).tolist()})
ionic *= CONV
def poly(eps_inf_val):
    E = (eps_inf_val * np.eye(3) if np.isscalar(eps_inf_val) else eps_inf_val) + ionic
    return float(np.trace(E) / 3)
res = {"base": args.base, "cell": args.cell, "n_atoms": N, "Omega": Omega, "h": args.h, "fmax_base_alone": float(np.abs(f0).max()),
       "imaginary_optical_modes": int(sum(w2[m] < 0 for m in keep)), "acoustic_w2": [float(w2[m]) for m in sorted(acoustic)],
       "optical_mode_range_meV": [float(math.sqrt(max(min(w2[m] for m in keep), 0)) * MEV), float(math.sqrt(max(w2[m] for m in keep)) * MEV)],
       "Z_species_trace3": {s: float(np.mean([np.trace(Z[k]) / 3 for k in range(N) if numbers[k] == n])) for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))},
       "Z_species_frobenius": {s: float(np.mean([np.linalg.norm(Z[k]) for k in range(N) if numbers[k] == n])) for s, n in (("Cl", 17), ("Cs", 55), ("Pb", 82))},
       "Z_asr_residual": Z.sum(0).tolist(), "eps_inf_tensor": eps_inf.tolist(), "eps_inf_poly": float(np.trace(eps_inf) / 3),
       "ionic_tensor": ionic.tolist(), "ionic_poly": float(np.trace(ionic) / 3), "ionic_diag": np.diag(ionic).tolist(),
       "eps0_LST_poly_at_4.0": poly(4.0), "eps0_LST_poly_at_base_eps_inf": poly(eps_inf), "eps0_LST_poly_at_3.667": poly(3.667),
       "eps0_LST_diag_at_4.0": (4.0 + np.diag(ionic)).tolist(),
       "top_modes": sorted(modes, key=lambda d: -d["eps_contrib_poly"])[:12], "seconds": time.time() - t}
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
np.savez(Path(args.out).expanduser().with_suffix(".npz"), FC=FC, w2=w2, e=e, M=M, Z=Z, eps_inf=eps_inf, numbers=numbers, positions=atoms.get_positions(), cell=np.array(atoms.get_cell()))
print(f"Hessian {N} atoms: imaginary optical {res['imaginary_optical_modes']}, optical {res['optical_mode_range_meV'][0]:.1f}-{res['optical_mode_range_meV'][1]:.1f} meV; Z* trace/3 {res['Z_species_trace3']}; "
      f"eps_inf {res['eps_inf_poly']:.3f} diag {np.round(np.diag(eps_inf), 3).tolist()}; ionic poly {res['ionic_poly']:.2f} diag {np.round(np.diag(ionic), 2).tolist()}; "
      f"eps0_LST poly at 4.0 {res['eps0_LST_poly_at_4.0']:.2f}, at base eps_inf {res['eps0_LST_poly_at_base_eps_inf']:.2f}, at 3.667 {res['eps0_LST_poly_at_3.667']:.2f}; {res['seconds']:.0f}s")
print("saved", args.out)
