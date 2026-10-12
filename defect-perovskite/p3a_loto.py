"""Plan P Phase 3a §2.3: LO-TO at q = 2 pi/L_axis of the 3x3x3 pristine tiling with base v8 w01 + E_PP (the bw5 checkpoint
at zero carriers; r_split 4.5, eps_inf 4.0), against the smoothed nonanalytic expectation g^2 f^2 (4 pi/Omega) Z~^2/eps_inf
(P0's reference computation, p0_loto.py, with the base as its own provider). Criterion: within 10 %.
Pattern: the strongest IR mode of the Gamma Hessian (p3a_lst.npz), modulated cos(q.R), q along the axis of its largest
Z~ component (LO) and its smallest (TO); curvatures k = (E(+A) + E(-A) - 2E(0))/A^2 for the base, the full neutral model,
and E_PP alone.

    OMP_NUM_THREADS=40 python defect-perovskite/p3a_loto.py --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json --lst ~/runs/dscc/p3a/p3a_lst.npz --out ~/runs/dscc/p3a/p3a_loto.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace.modules.dscc.ewald import COULOMB                                 # noqa: E402
from p3a_common import BASE_V8, BW5_S0, MEV, Base, Neutral, load_cell       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--lst", required=True); ap.add_argument("--base", default=BASE_V8); ap.add_argument("--checkpoint", default=BW5_S0)
ap.add_argument("--r_split", type=float, default=4.5); ap.add_argument("--eps_inf", type=float, default=4.0); ap.add_argument("--amp", type=float, default=0.05)
ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64); t = time.time()
unit = load_cell(args.cell); N = len(unit); lst = np.load(args.lst); w2, e, M, Z = lst["w2"], lst["e"], lst["M"], lst["Z"]
keep = [m for m in range(3 * N) if w2[m] > 1e-6]
zt_all = {m: np.einsum("kab,kb->a", Z, e[:, m].reshape(N, 3) / np.sqrt(M)[:, None]) for m in keep}
m_star = max(keep, key=lambda m: (zt_all[m] ** 2).sum() / w2[m]); zt = zt_all[m_star]
lo_axis = int(np.argmax(np.abs(zt))); to_axis = int(np.argmin(np.abs(zt)))
print(f"strongest IR mode {m_star}: {math.sqrt(w2[m_star]) * MEV:.2f} meV, Z~ {np.round(zt, 3)}; LO axis {'xyz'[lo_axis]}, TO axis {'xyz'[to_axis]}", flush=True)
B = Base(args.base, args.device); NM = Neutral(args.checkpoint, args.device)
assert abs(float(getattr(NM.model, "r_split", args.r_split)) - args.r_split) < 1e-6, f"checkpoint r_split {getattr(NM.model, 'r_split', None)}"
big = unit.repeat((3, 3, 3)); NB = len(big); cellB = np.array(big.get_cell()); Omega = abs(np.linalg.det(cellB)); Rb = big.get_positions()
vec_big = np.tile(e[:, m_star].reshape(N, 3) / np.sqrt(M)[:, None], (27, 1)); Zb = np.tile(Z, (27, 1, 1)); w = 2.0 * args.r_split
res = {"mode": int(m_star), "omega_meV": float(math.sqrt(w2[m_star]) * MEV), "Ztilde": zt.tolist(), "lo_axis": "xyz"[lo_axis], "to_axis": "xyz"[to_axis],
       "amp": args.amp, "r_split": args.r_split, "eps_inf": args.eps_inf, "N_big": NB, "branches": {}}
for name, axis in (("LO", lo_axis), ("TO", to_axis)):
    q = np.zeros(3); q[axis] = 2 * math.pi / np.linalg.norm(cellB[axis]); qhat = q / np.linalg.norm(q); phase = np.cos(Rb @ q); u = vec_big * phase[:, None]
    def displaced(A):
        return Atoms(numbers=big.get_atomic_numbers(), positions=Rb + A * u, cell=cellB, pbc=True)
    E = {}
    for A in (args.amp, -args.amp, 0.0):
        t0 = time.time(); eb = B.energy(displaced(A)); ev, epp = NM.energy(displaced(A)); E[A] = (eb, ev, epp)
        print(f"  {name} A={A:+.2f}: E_base {eb:.5f} E_full {ev:.5f} E_PP {epp:.6f} ({time.time()-t0:.0f}s)", flush=True)
    k_base = (E[args.amp][0] + E[-args.amp][0] - 2 * E[0.0][0]) / args.amp ** 2
    k_var = (E[args.amp][1] + E[-args.amp][1] - 2 * E[0.0][1]) / args.amp ** 2
    k_pp = (E[args.amp][2] + E[-args.amp][2] - 2 * E[0.0][2]) / args.amp ** 2
    ph = np.exp(1j * (Rb @ q))
    Sq = np.sum(np.einsum("a,kab,kb->k", qhat, Zb, u) * ph)
    dp = (B.dipoles(displaced(args.amp)) - B.dipoles(displaced(-args.amp))) / (2 * args.amp)
    Sq_model = np.sum((dp @ qhat) * ph); ff = float(abs(Sq_model) / abs(Sq)) if abs(Sq) > 0 else float("nan")
    e_nac_raw = 2.0 * 0.5 * (4 * math.pi * COULOMB / (Omega * args.eps_inf)) * abs(Sq) ** 2      # the +q and -q structure-factor terms (P0)
    g = math.exp(-w * w * float(q @ q) / 4)
    res["branches"][name] = {"q": q.tolist(), "k_base": k_base, "k_var": k_var, "k_pp": k_pp, "k_nac_raw": 2 * e_nac_raw, "k_nac_smoothed": 2 * e_nac_raw * g, "g_w_q": g,
                             "form_factor_model_over_Zstar": ff, "k_nac_smoothed_x_ff2": 2 * e_nac_raw * g * ff ** 2}
    print(f"{name}: k_base {k_base:.5f} k_var {k_var:.5f} k_pp {k_pp:.5f} | NAC raw {2*e_nac_raw:.5f} smoothed {2*e_nac_raw*g:.5f} (g {g:.3f}) | f {ff:.3f} -> smoothed x f^2 {2*e_nac_raw*g*ff**2:.5f}", flush=True)
b_ = res["branches"]
res["splitting"] = {"model_var": b_["LO"]["k_var"] - b_["TO"]["k_var"], "model_base": b_["LO"]["k_base"] - b_["TO"]["k_base"], "e_pp_only": b_["LO"]["k_pp"] - b_["TO"]["k_pp"],
                    "nac_raw": b_["LO"]["k_nac_raw"] - b_["TO"]["k_nac_raw"], "nac_smoothed": b_["LO"]["k_nac_smoothed"] - b_["TO"]["k_nac_smoothed"],
                    "nac_smoothed_x_ff2": b_["LO"]["k_nac_smoothed_x_ff2"] - b_["TO"]["k_nac_smoothed_x_ff2"]}
sp = res["splitting"]; res["criterion"] = {"ratio_epp_over_nac_smoothed_ff2": sp["e_pp_only"] / sp["nac_smoothed_x_ff2"] if sp["nac_smoothed_x_ff2"] else None,
                                           "ratio_epp_over_nac_smoothed": sp["e_pp_only"] / sp["nac_smoothed"] if sp["nac_smoothed"] else None}
res["criterion"]["pass_within_10pct"] = bool(abs(res["criterion"]["ratio_epp_over_nac_smoothed_ff2"] - 1) <= 0.10) if res["criterion"]["ratio_epp_over_nac_smoothed_ff2"] else None
res["seconds"] = time.time() - t
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print("splitting:", json.dumps({k: round(v, 5) for k, v in sp.items()}), "criterion:", json.dumps(res["criterion"])); print("saved", args.out)
