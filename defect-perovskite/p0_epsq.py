"""Diagnostic 3: the base's ionic screening at finite q, eps_ion(q) along y, from frozen phonons
in the 3x3x3 cell at n = 0..3 (q = n 2pi/L_y). Modes: the K strongest y-polarised IR modes of
the Gamma Hessian (strength Z~_y^2 / omega^2). Each is modulated cos(q.R) and driven
LONGITUDINALLY (q || y); the BASE alone gives the bare (TO-like) curvature k_m(q):
omega_m(q)^2 = 2 k / 27 (n >= 1; k / 27 at n = 0, checked against the Hessian); the mode charge
at q is f_m(q) Z~_m,y with the provider's form factor. eps_ion(q) = (4 pi C / Omega) sum_m
(f_m Z~_m,y)^2 / omega_m(q)^2 -- a Rayleigh quotient on a Gamma pattern: omega is an upper
bound, eps_ion(q) a lower bound.
    OMP_NUM_THREADS=40 python defect-perovskite/p0_epsq.py --base ~/runs/base_v2_prod/base_v2_prod_base.pt \
        --cell ~/runs/dscc/p0/pristine_relaxed_v2.json --out ~/runs/dscc/p0/p0_epsq.json
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.data import atomic_masses
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import data as mace_data, tools                    # noqa: E402
from mace.modules.dscc import data as dd                     # noqa: E402
from mace.modules.dscc.ewald import COULOMB                  # noqa: E402
from mace.modules.dscc.field import load_field_provider      # noqa: E402
from mace.tools import torch_geometric                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--cell", required=True)
ap.add_argument("--ckpt", default="/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model")
ap.add_argument("--axis", type=int, default=1); ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--k_modes", type=int, default=8); ap.add_argument("--amp", type=float, default=0.05); ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--hess_device", default="cuda"); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); t0 = time.time()
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
N = len(unit); numbers = unit.get_atomic_numbers(); M = np.array([atomic_masses[n] for n in numbers]); Omega = abs(np.linalg.det(np.array(unit.get_cell())))
z3 = tools.AtomicNumberTable([17, 55, 82])
MEV = 1000.0 * 6.582119569e-16 * math.sqrt(1.602176634e-19 / (1.66053907e-27 * 1e-20))

# ---- Gamma Hessian on the fast device
hd = args.hess_device
base_h = torch.load(Path(args.base).expanduser(), map_location=hd, weights_only=False).double().to(hd).eval()
def batch_for(model, a, dev, r_cut=6.0):
    a = a.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "eq", "source_dir": "eq"})
    return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z3, r_cut), batch_size=1))).to(dev).to_dict()
def forces_h(a):
    return base_h(batch_for(base_h, a, hd), training=False, compute_force=True)["forces"].detach().cpu().numpy()
pos0 = unit.get_positions(); FC = np.zeros((3 * N, 3 * N))
for i in range(N):
    for c in range(3):
        p, m = unit.copy(), unit.copy(); pp = pos0.copy(); pp[i, c] += args.h; p.set_positions(pp); pm = pos0.copy(); pm[i, c] -= args.h; m.set_positions(pm)
        FC[3 * i + c] = -(forces_h(p) - forces_h(m)).reshape(-1) / (2 * args.h)
FC = 0.5 * (FC + FC.T)
for a_ in range(3):
    for b_ in range(3):
        blk = FC[a_::3, b_::3]; np.fill_diagonal(blk, np.diag(blk) - blk.sum(1)); FC[a_::3, b_::3] = blk
Msq = np.repeat(np.sqrt(M), 3); w2, e = np.linalg.eigh(FC / np.outer(Msq, Msq))
prov = load_field_provider(args.ckpt, head="mp-dielectric", device=args.device)
z79 = tools.AtomicNumberTable([int(z) for z in prov.model.atomic_numbers])
cfg = mace_data.config_from_atoms(unit, key_specification=mace_data.KeySpecification.from_defaults(), head_name="mp-dielectric")
ad = mace_data.AtomicData.from_config(cfg, z_table=z79, cutoff=float(prov.model.r_max), heads=list(prov.model.heads))
bu = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to(args.device).to_dict()
Z = prov.model(bu, training=True, compute_force=False, compute_polarization=True, compute_becs=True)["becs"].detach().cpu().numpy().reshape(N, 3, 3)
ax = args.axis; CONV = 4 * math.pi * COULOMB / Omega
modes = []
for mm in range(3 * N):
    if w2[mm] <= 1e-6:
        continue
    vec = e[:, mm].reshape(N, 3) / np.sqrt(M)[:, None]; zt = np.einsum("kab,kb->a", Z, vec)
    modes.append({"m": mm, "w2": float(w2[mm]), "zt": zt, "s_ax": float(zt[ax] ** 2 / w2[mm]), "s_all": float((zt ** 2).sum() / w2[mm])})
eps_ion_axis_full = CONV * sum(md["s_ax"] for md in modes); eps_ion_poly_full = CONV * sum(md["s_all"] for md in modes) / 3
top = sorted(modes, key=lambda d: -d["s_ax"])[:args.k_modes]
share = sum(md["s_ax"] for md in top) / sum(md["s_ax"] for md in modes)
print(f"Hessian done {time.time()-t0:.0f}s: eps_ion({'xyz'[ax]}{'xyz'[ax]}, q=0, all modes) {eps_ion_axis_full:.2f}, poly {eps_ion_poly_full:.2f}; top {args.k_modes} modes carry {100*share:.0f} %: " + ", ".join(f"{math.sqrt(md['w2'])*MEV:.1f} meV" for md in top), flush=True)
del base_h; torch.cuda.empty_cache()

# ---- frozen phonons on the CPU
dev = args.device
base = torch.load(Path(args.base).expanduser(), map_location=dev, weights_only=False).double().to(dev).eval()
big = unit.repeat((3, 3, 3)); NB = len(big); cellB = np.array(big.get_cell()); Rb = big.get_positions(); L = float(np.linalg.norm(cellB[ax]))
def energy(a):
    return float(base(batch_for(base, a, dev), training=False, compute_force=False)["energy"][0])
def dipoles(a):
    bq = batch_for(prov.model, a, dev)
    pos = bq["positions"]; cell_t = bq["cell"].view(3, 3)
    sh = torch.einsum("ei,eij->ej", bq["unit_shifts"], cell_t.expand(bq["edge_index"].shape[1], 3, 3))
    nums = torch.tensor([[17, 55, 82][i] for i in bq["node_attrs"].argmax(-1).tolist()], device=dev)
    with torch.enable_grad():
        return prov.node_dipoles(pos.detach().requires_grad_(True), cell_t, bq["batch"], bq["ptr"], bq["edge_index"], sh, bq["unit_shifts"], nums).detach().cpu().numpy()
E0 = energy(big); print(f"E0 {E0:.4f} eV ({time.time()-t0:.0f}s)", flush=True)
Zb = np.tile(Z, (27, 1, 1))
res = {"axis": "xyz"[ax], "L": L, "Omega_cell": Omega, "eps_ion_axis_q0_all_modes": eps_ion_axis_full, "eps_ion_poly_q0_all_modes": eps_ion_poly_full,
       "top_modes_share_q0": share, "modes": [{"m": md["m"], "omega_meV": math.sqrt(md["w2"]) * MEV, "zt_axis": md["zt"][ax]} for md in top], "q": []}
for n in args.orders:
    q = np.zeros(3); q[ax] = n * 2 * math.pi / L; qn = float(np.linalg.norm(q)); qhat = q / qn if qn > 0 else np.eye(3)[ax]
    phase = np.cos(Rb @ q); ph = np.exp(1j * (Rb @ q)); row = {"n": n, "q": qn, "per_mode": []}; total = 0.0
    for md in top:
        vec = np.tile(e[:, md["m"]].reshape(N, 3) / np.sqrt(M)[:, None], (27, 1)); u = vec * phase[:, None]
        ap_ = Atoms(numbers=big.get_atomic_numbers(), positions=Rb + args.amp * u, cell=cellB, pbc=True)
        am_ = Atoms(numbers=big.get_atomic_numbers(), positions=Rb - args.amp * u, cell=cellB, pbc=True)
        Ep, Em = energy(ap_), energy(am_); k = (Ep + Em - 2 * E0) / args.amp ** 2
        w2q = k / 27.0 if n == 0 else 2.0 * k / 27.0
        dp = (dipoles(ap_) - dipoles(am_)) / (2 * args.amp)
        S_model = np.sum([(qhat @ dp[b]) * ph[b] for b in range(NB)]); S_born = np.sum([(qhat @ (Zb[b] @ u[b])) * ph[b] for b in range(NB)])
        f = abs(S_model) / abs(S_born) if abs(S_born) > 0 else float("nan")
        zq = f * md["zt"][ax]; contrib = CONV * zq ** 2 / w2q if w2q > 0 else float("inf"); total += contrib
        row["per_mode"].append({"m": md["m"], "omega_q_meV": math.sqrt(max(w2q, 0)) * MEV, "omega_0_meV": math.sqrt(md["w2"]) * MEV, "f": f, "contrib": contrib,
                                "contrib_q0_hessian": CONV * md["s_ax"]})
        print(f"  n={n} q={qn:.3f} mode {md['m']}: omega {math.sqrt(md['w2'])*MEV:.2f} -> {math.sqrt(max(w2q,0))*MEV:.2f} meV, f {f:.3f}, eps contrib {contrib:.2f} (Hessian q=0 {CONV*md['s_ax']:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    row["eps_ion_top"] = total; res["q"].append(row)
    print(f"n={n} q={qn:.3f}: eps_ion(top {args.k_modes}) = {total:.2f}", flush=True)
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
