"""Plan P Phase 3a §2.5, the linear-response check on the largest ladder structure.

F = the full model's forces on the far-field atoms (r > r_core from the vacancy) at the frozen embedded geometry;
K = the base's real-space force constants (the 3x3x3 finite-displacement set, pairs within r_local) restricted to the
far-field atoms (core fixed) + E_PP's dipole-dipole block (T_LR contracted with the base's Z*: K_PP v = Z^T T_LR Z v / eps_inf,
one `dipole_field` call per matvec); u_pred = K^-1 F by conjugate gradient. Compared with the relaxed far-field
displacements (p0_ladder's relaxed_<tiling>.xyz minus the frozen embedded geometry) by species and shell: rms ratio and
cosine. Criterion: ratio within 15 % and cosine >= 0.9 in every shell beyond 10 A. K_base alone is reported alongside.

    OMP_NUM_THREADS=48 python defect-perovskite/p3a_linresp.py --model <ckpt> --core ~/runs/dscc/p3a/ladder/core.json --relaxed ~/runs/dscc/p3a/ladder/relaxed_334.xyz \
        --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json --raw ~/runs/dscc/p3a/fc333_raw.npz --lst ~/runs/dscc/p3a/p3a_lst.npz --tiling 3,3,4 --out ~/runs/dscc/p3a/linresp_2879.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                         # noqa: E402
from mace.modules.dscc.polarisation import dipole_field                    # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402
from p3a_common import load_cell                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--core", required=True); ap.add_argument("--relaxed", required=True); ap.add_argument("--cell", required=True)
ap.add_argument("--raw", required=True); ap.add_argument("--lst", required=True); ap.add_argument("--tiling", default="3,3,4"); ap.add_argument("--r_core", type=float, default=8.0)
ap.add_argument("--r_local", type=float, default=12.5); ap.add_argument("--r_split", type=float, default=4.5); ap.add_argument("--eps_inf", type=float, default=4.0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
unit = load_cell(args.cell); N = len(unit); tt = tuple(int(x) for x in args.tiling.split(","))
src = json.load(open(Path(args.core).expanduser())); vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
rel0 = np.array(src["rel_positions"]); u0 = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u0[i] for i in range(len(u0))}
frozen, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
ideal = tiling_ladder(unit, [tt], vacancy_species=17, vacancy_index=0)[0]
vac, cell = vacancy_site(unit, tt); inv = np.linalg.inv(cell)
Rf = frozen.get_positions(); Ri = ideal.get_positions(); NB = len(frozen)
r = np.linalg.norm(relative(Rf, vac, cell), axis=1); far = r > args.r_core; fidx = np.nonzero(far)[0]; nf = len(fidx)
relaxed = read(Path(args.relaxed).expanduser()); u_rel = relaxed.get_positions() - Rf; u_rel -= np.round(u_rel @ inv) @ cell
print(f"{tt}: N={NB}, far atoms {nf}, relaxed rms |u| far {np.sqrt((u_rel[far]**2).sum(1).mean())*1e3:.1f} mA", flush=True)

# the unit-cell site of every atom (ASE repeat order, one Cl removed)
rep = unit.repeat(tt); sites = [i for i, z in enumerate(rep.get_atomic_numbers()) if int(z) == 17]; removed = sites[0]
rep_index = np.array([i if i < removed else i + 1 for i in range(NB)]); kappa = rep_index % N
# real-space force constants from the 3x3x3 set, keyed by (kappa, pair vector)
d = np.load(args.raw); G = d["G"]; R3 = d["positions"]; cell3 = d["cell"]; inv3 = np.linalg.inv(cell3)
for k in range(N):
    G[k, :, k, :] -= G[k].sum(1)                                           # ASR on the self block
r3 = R3[None, :, :] - R3[:N, None, :]; fr = r3 @ inv3; fr -= np.round(fr); r3 = fr @ cell3
key = {}
for k in range(N):
    for bp in range(G.shape[2]):
        if np.linalg.norm(r3[k, bp]) <= args.r_local + 0.5:
            key[(k,) + tuple(np.round(r3[k, bp] / 0.02).astype(np.int64))] = bp
# K_base over the far atoms (ideal pair vectors; the core is fixed so only far-far blocks enter)
rows_, cols_, vals_ = [], [], []; pos_f = {int(i): n for n, i in enumerate(fidx)}; n_missing = 0
Rif = Ri[fidx]
for n_i, i in enumerate(fidx):
    dv = Rif - Ri[i]; f_ = dv @ inv; f_ -= np.round(f_); dv = f_ @ cell; dist = np.linalg.norm(dv, axis=1)
    for n_j in np.nonzero(dist <= args.r_local)[0]:
        bp = key.get((int(kappa[i]),) + tuple(np.round(dv[n_j] / 0.02).astype(np.int64)))
        if bp is None:
            n_missing += 1; continue
        blk = G[kappa[i], :, bp, :]                                        # [beta(i), alpha(j)]: Phi(i beta; j alpha)
        for a_ in range(3):
            for b_ in range(3):
                rows_.append(3 * n_i + a_); cols_.append(3 * n_j + b_); vals_.append(blk[a_, b_])
K_base = coo_matrix((vals_, (rows_, cols_)), shape=(3 * nf, 3 * nf)).tocsr(); K_base = 0.5 * (K_base + K_base.T)
print(f"K_base built: {K_base.nnz} entries, {n_missing} pair lookups missing, {time.time()-t0:.0f}s", flush=True)
# E_PP block
Z = np.load(args.lst)["Z"]; Zf = torch.tensor(Z[kappa[fidx]], dtype=torch.float64)
pos_t = torch.tensor(Rf, dtype=torch.float64); cell_t = torch.tensor(cell, dtype=torch.float64); fidx_t = torch.tensor(fidx)
def k_pp(v):
    vf = torch.tensor(np.asarray(v).reshape(nf, 3), dtype=torch.float64); p = torch.zeros(NB, 3, dtype=torch.float64)
    p[fidx_t] = torch.einsum("iab,ib->ia", Zf, vf)
    with torch.no_grad():
        fld = dipole_field(pos_t, cell_t, p, args.r_split, args.eps_inf)[fidx_t]        # (T p)_a / eps = dE_PP/dp_a
    return torch.einsum("iab,ia->ib", Zf, fld).reshape(-1).numpy()
# forces at the frozen geometry from the full model
model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval(); model.base = model.base.double()
z3 = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
a = frozen.copy(); a.info.update({"carrier_counts": np.array([0, 0, 1, 0], float), "cell_charge": 1, "config_type": "lr", "source_dir": "lr"})
b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
F = model(b, compute_force=True)["forces"].detach().cpu().numpy(); Ff = F[fidx].reshape(-1)
print(f"forces at the frozen geometry: rms far {np.sqrt((F[far]**2).sum(1).mean())*1e3:.2f} meV/A, {time.time()-t0:.0f}s", flush=True)
res = {"tiling": list(tt), "N": NB, "n_far": nf, "r_core": args.r_core, "pair_lookups_missing": n_missing, "variants": {}}
lad = Path(args.relaxed).expanduser().parent / "ladder.json"
if lad.exists():
    for rw in json.load(open(lad)):
        if rw["n_atoms"] == NB:
            res["E_relaxed_minus_embedded_eV"] = rw["E_relaxed"] - rw["E_embedded"]; print(f"actual far-field relaxation energy E_relaxed - E_embedded = {res['E_relaxed_minus_embedded_eV']:.4f} eV", flush=True)
edges = [(8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 24)]; syms = np.array(frozen.get_chemical_symbols())
for name, use_pp in (("K_base_plus_EPP", True), ("K_base_only", False)):
    it = [0]
    def mv(v):
        it[0] += 1; return K_base @ v + (k_pp(v) if use_pp else 0.0)
    op = LinearOperator((3 * nf, 3 * nf), matvec=mv, dtype=np.float64); t1 = time.time()
    u, info = cg(op, Ff, rtol=1e-8, maxiter=5000); u = u.reshape(nf, 3)
    up = np.zeros((NB, 3)); up[fidx] = u
    rows = []
    for lo, hi in edges:
        m = far & (r > lo) & (r <= hi)
        if not m.any():
            continue
        rms_p = float(np.sqrt((up[m] ** 2).sum(1).mean())); rms_r = float(np.sqrt((u_rel[m] ** 2).sum(1).mean()))
        cos = float(np.mean(np.einsum("ij,ij->i", up[m], u_rel[m]) / np.maximum(np.linalg.norm(up[m], axis=1) * np.linalg.norm(u_rel[m], axis=1), 1e-12)))
        per = {s: ((float(np.sqrt((up[m & (syms == s)] ** 2).sum(1).mean()) / max(np.sqrt((u_rel[m & (syms == s)] ** 2).sum(1).mean()), 1e-12))) if (m & (syms == s)).any() else None) for s in ("Cs", "Pb", "Cl")}
        rows.append({"r_lo": lo, "r_hi": hi, "n": int(m.sum()), "rms_u_pred_mA": 1e3 * rms_p, "rms_u_relaxed_mA": 1e3 * rms_r, "ratio": rms_p / max(rms_r, 1e-12), "cos": cos, "ratio_by_species": per})
        print(f"  {name} {lo:>2d}-{hi:<3d} n={m.sum():4d} u_pred {1e3*rms_p:6.1f} u_rel {1e3*rms_r:6.1f} mA ratio {rms_p/max(rms_r,1e-12):5.2f} cos {cos:5.2f} " + " / ".join(f"{v:.2f}" if v is not None else "-" for v in per.values()), flush=True)
    beyond = [x for x in rows if x["r_lo"] >= 10]
    # harmonic relaxation energy of the far field, -1/2 F.u_pred, against the actual E_relaxed - E_embedded; and the
    # displacement the registered fmax can resolve, |u| ~ fmax / k with k the diagonal stiffness
    e_lin = -0.5 * float(Ff @ u.reshape(-1)); k_diag = float(np.median(K_base.diagonal()))
    res["variants"][name] = {"cg_info": int(info), "matvecs": it[0], "seconds": time.time() - t1, "shells": rows,
                             "pass": bool(all(abs(x["ratio"] - 1) <= 0.15 and x["cos"] >= 0.9 for x in beyond)),
                             "E_lin_far_eV": e_lin, "k_diag_median_eV_A2": k_diag, "u_resolution_mA_at_fmax_0.002": 1e3 * 0.002 / k_diag}
    print(f"  harmonic far-field relaxation energy -1/2 F.u_pred = {e_lin:.4f} eV; median diagonal stiffness {k_diag:.2f} eV/A^2 -> fmax 0.002 resolves |u| ~ {1e3*0.002/k_diag:.1f} mA", flush=True)
    print(f"{name}: cg info {info}, {it[0]} matvecs, pass {res['variants'][name]['pass']}", flush=True)
res["seconds"] = time.time() - t0
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
