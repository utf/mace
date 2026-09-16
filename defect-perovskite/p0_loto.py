"""Plan P Decision 0 fallback: LO-TO splitting of the strongest IR mode at the smallest nonzero
q of the 3x3x3 tiling, with and without E_PP (frozen phonon), against the nonanalytic value.

Mode: Gamma modes of the base's Hessian on the relaxed 80-atom cell (as p0_eps0_lst.py); the
strongest IR mode m* maximises |Z~_m|^2 / omega_m^2 (provider Z*). Pattern: u_k = A e_{k,m*} /
sqrt(M_k) cos(q . R_k) in the 3x3x3 tiling, q = 2 pi / (3 a_i) along axis i. LO: the axis with the
largest |Z~_m*,i|; TO: the axis with the smallest. E(+A), E(-A), E(0) with base v2 alone and with
base v2 + E_PP (the polarisation variant in the reference state: short circuit + E_PP), so the
splitting is Delta k = k_LO - k_TO of the curvature (E(A) + E(-A) - 2 E(0)) / A^2. Nonanalytic
expectation for the same pattern: the k-space kernel at q, E_NAC = 1/2 (4 pi C / (Omega eps_inf))
(q^ . P~)^2 g_w(q), P~ = sum_k Z*_k u_k cos(q . R_k) ... reported RAW (g = 1) and SMOOTHED
(g = exp(-w^2 q^2 / 4), w = 2 r_split) -- the second is what the registered kernel can deliver
at this q by construction.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.data import atomic_masses
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                    # noqa: E402
from mace.modules.dscc import data as dd                     # noqa: E402
from mace.modules.dscc.ewald import COULOMB                  # noqa: E402
from mace.tools import torch_geometric                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--variant", required=True, help="polarisation variant .pt (E_PP in the neutral state)")
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--cell", required=True)
ap.add_argument("--r_split", type=float, default=9.645)
ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--amp", type=float, default=0.05, help="pattern amplitude A (A * amu^-1/2 units of e/sqrt(M))")
ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
N = len(unit); numbers = unit.get_atomic_numbers(); M = np.array([atomic_masses[n] for n in numbers])
z3 = tools.AtomicNumberTable([17, 55, 82])
base = torch.load(Path(args.base).expanduser(), map_location=dev, weights_only=False).double().to(dev).eval()
var = torch.load(Path(args.variant).expanduser(), map_location=dev, weights_only=False).to(dev).eval()
var.base = var.base.double()
assert getattr(var, "host_coupling", "") == "polarisation"


def batch(a, counts=(0, 0, 0, 0), charge=0, r_cut=6.0):
    a = a.copy(); a.info.update({"carrier_counts": np.array(counts, float), "cell_charge": charge, "config_type": "loto", "source_dir": "loto"})
    return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z3, r_cut), batch_size=1))).to(dev).to_dict()


def base_forces(a):
    return base(batch(a), training=False, compute_force=True)["forces"].detach().cpu().numpy()


def base_energy(a):
    return float(base(batch(a), training=False, compute_force=False)["energy"][0])


def var_energy(a):     # reference state: base + E_PP
    with torch.no_grad():
        out = var(batch(a, r_cut=var.r_cut), compute_force=False)
    return float(out["energy"][0]), float(out["e_pp"][0])


# ---- Gamma modes of the 80-atom cell
t = time.time(); pos0 = unit.get_positions()
FC = np.zeros((3 * N, 3 * N))
for i in range(N):
    for c in range(3):
        p, m = unit.copy(), unit.copy()
        pp = pos0.copy(); pp[i, c] += args.h; p.set_positions(pp)
        pm = pos0.copy(); pm[i, c] -= args.h; m.set_positions(pm)
        FC[3 * i + c] = -(base_forces(p) - base_forces(m)).reshape(-1) / (2 * args.h)
FC = 0.5 * (FC + FC.T)
for a_ in range(3):
    for b_ in range(3):
        blk = FC[a_::3, b_::3]; np.fill_diagonal(blk, np.diag(blk) - blk.sum(1)); FC[a_::3, b_::3] = blk
Msq = np.repeat(np.sqrt(M), 3)
w2, e = np.linalg.eigh(FC / np.outer(Msq, Msq))
# provider Z* on the cell
prov = torch.load(args.ckpt, map_location=dev, weights_only=False).double().to(dev).eval()
z79 = tools.AtomicNumberTable([int(z) for z in prov.atomic_numbers])
cfg = mace_data.config_from_atoms(unit, key_specification=mace_data.KeySpecification.from_defaults(), head_name="mp-dielectric")
ad = mace_data.AtomicData.from_config(cfg, z_table=z79, cutoff=float(prov.r_max), heads=list(prov.heads))
b = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
Z = prov(b, training=True, compute_force=False, compute_polarization=True, compute_becs=True)["becs"].detach().cpu().numpy().reshape(N, 3, 3)
strength = []
for mm in range(3 * N):
    if w2[mm] <= 1e-6:
        continue
    vec = e[:, mm].reshape(N, 3) / np.sqrt(M)[:, None]
    zt = np.einsum("kab,kb->a", Z, vec)
    strength.append((float((zt ** 2).sum() / w2[mm]), mm, zt))
strength.sort(key=lambda s: -s[0])
S_m, m_star, zt = strength[0]
MEV = 1000.0 * 6.582119569e-16 * math.sqrt(1.602176634e-19 / (1.66053907e-27 * 1e-20))
print(f"strongest IR mode {m_star}: omega {math.sqrt(w2[m_star]) * MEV:.2f} meV, Z~ {np.round(zt, 3)} e/sqrt(amu), {time.time()-t:.0f}s", flush=True)
lo_axis = int(np.argmax(np.abs(zt))); to_axis = int(np.argmin(np.abs(zt)))

# ---- frozen phonon in the 3x3x3 tiling
big = unit.repeat((3, 3, 3)); NB = len(big); cellB = np.array(big.get_cell()); Omega = abs(np.linalg.det(cellB))
Rb = big.get_positions(); Mb = np.array([atomic_masses[n] for n in big.get_atomic_numbers()])
vec_unit = e[:, m_star].reshape(N, 3) / np.sqrt(M)[:, None]
vec_big = np.tile(vec_unit, (27, 1))
Zb = np.tile(Z, (27, 1, 1))
w = 2.0 * args.r_split
res = {"mode": int(m_star), "omega_meV": float(math.sqrt(w2[m_star]) * MEV), "Ztilde": zt.tolist(), "lo_axis": lo_axis, "to_axis": to_axis,
       "amp": args.amp, "r_split": args.r_split, "eps_inf": args.eps_inf, "N_big": NB, "branches": {}}
for name, axis in (("LO", lo_axis), ("TO", to_axis)):
    q = np.zeros(3); q[axis] = 2 * math.pi / np.linalg.norm(cellB[axis])
    phase = np.cos(Rb @ q)
    u = vec_big * phase[:, None]                         # per unit amplitude
    def displaced(A):
        a = big.copy(); a.set_positions(Rb + A * u); return a
    E = {}
    for A in (args.amp, -args.amp, 0.0):
        eb = base_energy(displaced(A)); ev, epp = var_energy(displaced(A))
        E[A] = (eb, ev, epp)
    k_base = (E[args.amp][0] + E[-args.amp][0] - 2 * E[0.0][0]) / args.amp ** 2
    k_var = (E[args.amp][1] + E[-args.amp][1] - 2 * E[0.0][1]) / args.amp ** 2
    k_pp = (E[args.amp][2] + E[-args.amp][2] - 2 * E[0.0][2]) / args.amp ** 2
    # nonanalytic expectation for this pattern: induced dipoles Z* u, k-space kernel at q
    P_q = np.einsum("kab,kb->a", Zb, u * phase[:, None])          # sum_k Z*_k u_k cos(q.R_k)  (e/sqrt(amu))
    qhat = q / np.linalg.norm(q)
    e_nac_raw = 0.5 * (4 * math.pi * COULOMB / (Omega * args.eps_inf)) * float(qhat @ P_q) ** 2 * 2.0   # x2: cos^2 average over the pattern gives 1/2 |P~|^2 per cell... kept explicit below
    # exact k-space energy of the modulated dipole pattern p_k = Z* u_k: 1/2 (4 pi C / V eps) g(q) |sum_k (q^.p_k) e^{i q R_k}|^2 / |q|^2 * |q|^2 -> with (q.p)^2 = |q|^2 (q^.p)^2
    Sq = np.sum([(qhat @ (Zb[k] @ u[k])) * np.exp(1j * (q @ Rb[k])) for k in range(NB)])
    # form factor: the provider's actual dipole response to the pattern, phased at q, against Z* u
    from mace.modules.dscc.field import load_field_provider
    provq = load_field_provider(args.ckpt, head="mp-dielectric", device=dev)
    def dip(a):
        bq = batch(a, r_cut=6.0)
        pos = bq["positions"]; cell_t = bq["cell"].view(3, 3)
        sh = torch.einsum("ei,eij->ej", bq["unit_shifts"], cell_t.expand(bq["edge_index"].shape[1], 3, 3))
        nums = torch.tensor([[17, 55, 82][i] for i in bq["node_attrs"].argmax(-1).tolist()], device=dev)
        with torch.enable_grad():
            return provq.node_dipoles(pos.detach().requires_grad_(True), cell_t, bq["batch"], bq["ptr"], bq["edge_index"], sh, bq["unit_shifts"], nums).detach().cpu().numpy()
    dp = (dip(displaced(args.amp)) - dip(displaced(-args.amp))) / (2 * args.amp)
    Sq_model = np.sum([(qhat @ dp[k]) * np.exp(1j * (q @ Rb[k])) for k in range(NB)])
    form_factor = abs(Sq_model) / abs(Sq) if abs(Sq) > 0 else float("nan")
    # x2: the k-space kernel counts the +q and -q structure-factor terms (checked numerically
    # against dipole_dipole_energy on the pattern p = Z* u to 0.3 %)
    e_nac_raw = 2.0 * 0.5 * (4 * math.pi * COULOMB / (Omega * args.eps_inf)) * abs(Sq) ** 2
    g = math.exp(-w * w * float(q @ q) / 4)
    res["branches"][name] = {"q": q.tolist(), "k_base": k_base, "k_var": k_var, "k_pp": k_pp,
                             "k_nac_raw": 2 * e_nac_raw, "k_nac_smoothed": 2 * e_nac_raw * g, "g_w_q": g,
                             "form_factor_model_over_Zstar": form_factor, "k_nac_smoothed_x_ff2": 2 * e_nac_raw * g * form_factor ** 2}
    print(f"{name}: k_base {k_base:.5f} k_var {k_var:.5f} k_pp {k_pp:.5f} | NAC raw {2*e_nac_raw:.5f} smoothed {2*e_nac_raw*g:.5f} (g {g:.3f}) | form factor {form_factor:.3f} -> smoothed x ff^2 {2*e_nac_raw*g*form_factor**2:.5f}", flush=True)
b_ = res["branches"]
res["splitting"] = {"model_var": b_["LO"]["k_var"] - b_["TO"]["k_var"], "model_base": b_["LO"]["k_base"] - b_["TO"]["k_base"],
                    "e_pp_only": b_["LO"]["k_pp"] - b_["TO"]["k_pp"],
                    "nac_raw": b_["LO"]["k_nac_raw"] - b_["TO"]["k_nac_raw"], "nac_smoothed": b_["LO"]["k_nac_smoothed"] - b_["TO"]["k_nac_smoothed"]}
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1)
print("splitting:", json.dumps(res["splitting"]))
print("saved", args.out)
