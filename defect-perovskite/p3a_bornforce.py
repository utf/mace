"""Plan P Phase 3a §2.1: the Born-force table (switched kernel) -- the direct measurement of the +1 far field.

Structure: the +1 core relaxed at 639 atoms with the full model (p0_ladder --stage core, fmax 0.002), embedded within
R_core in a tiling of the base-v8-relaxed pristine cell. At that structure, dq from the checkpoint; the E_far (Born) force
on every atom F_born,b = -d/dR_b[-sum_a p_a(R) . E_c^{detached}(R_a)] (autograd through the base's dipoles only); the
reference Z*_b^T E_c(R_b) with Z*_b the base's own pristine Born tensor and E_c the smoothed periodic carrier field of
(i) a unit Gaussian charge at the vacancy site and (ii) the actual dq. Table by shell: rms force, ratio, cosine, per
species. Also sum dq within 3/5/8/12 A and N_eff (the compactness gate).

    OMP_NUM_THREADS=48 python defect-perovskite/p3a_bornforce.py --model <ckpt> --core ~/runs/dscc/p3a/ladder/core.json \
        --cell ~/runs/dscc/p3a/pristine_relaxed_v8.json --tiling 3,3,4 --device cpu --out ~/runs/dscc/p3a/bornforce_2879.json
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.ewald import COULOMB                                # noqa: E402
from mace.modules.dscc.polarisation import carrier_field, carrier_dipole_energy, direct_field, pair_width_cp, switch   # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402
from p3a_common import BASE_V8, Base, load_cell                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--core", required=True); ap.add_argument("--cell", required=True); ap.add_argument("--base", default=BASE_V8)
ap.add_argument("--tiling", default="3,3,4"); ap.add_argument("--r_core", type=float, default=8.0); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
args = ap.parse_args(); torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
unit = load_cell(args.cell); tt = tuple(int(x) for x in args.tiling.split(","))
src = json.load(open(Path(args.core).expanduser())); vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
atoms, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
vac, cell = vacancy_site(unit, tt)
r = np.linalg.norm(relative(np.array(atoms.get_positions()), vac, cell), axis=1); N = len(atoms)
print(f"embedded {tt}: N={N}, core {moved}/{inside} (missing {missing})", flush=True)

model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev).to(dev).eval(); model.base = model.base.double()
z3 = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
a = atoms.copy(); a.info.update({"carrier_counts": np.array([0, 0, 1, 0], float), "cell_charge": 1, "config_type": "bf", "source_dir": "bf"})
b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
out = model(dict(b), compute_force=False)
dq = out["dq"].detach(); e_cp_head = float(out["diagnostics"]["e_host"][0]); dq_np = dq.cpu().numpy()
compact = {"dq_sum": float(dq_np.sum()), "n_eff": float(1.0 / (dq_np ** 2).sum()), "sum_dq_within": {f"{R}": float(dq_np[r <= R].sum()) for R in (3, 5, 8, 12)},
           "dq_top5": [(int(i), float(dq_np[i]), float(r[i])) for i in np.argsort(-dq_np)[:5]]}
print(f"head: dq sum {compact['dq_sum']:.4f}, N_eff {compact['n_eff']:.2f}, sum dq < 3/5/8/12 A {[round(v, 3) for v in compact['sum_dq_within'].values()]}, E_cP {1e3*e_cp_head:.2f} meV, {time.time()-t0:.0f}s", flush=True)

pos = b["positions"].detach().clone().requires_grad_(True); cell_t = b["cell"].view(3, 3)
shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell_t.expand(b["edge_index"].shape[1], 3, 3))
d = dict(b); d["positions"] = pos; d["shifts"] = shifts; species = b["node_attrs"].argmax(-1)
p = model._node_dipoles(d, species)
kw = dict(r_g=model.kernel.r_g, r_split=model.r_split, eps_inf=model.kernel.eps_inf, tol=model.kernel.tol)
E_act = carrier_field(pos.detach(), cell_t, dq, **kw)                                           # actual dq
pos_u = torch.cat([pos.detach(), torch.tensor(vac, dtype=torch.float64, device=dev)[None]], 0)    # a unit charge at the vacancy site
dq_u = torch.zeros(N + 1, dtype=torch.float64, device=dev); dq_u[-1] = 1.0
E_unit = carrier_field(pos_u, cell_t, dq_u, **kw)[:N]
# the switched kernel (Phase 2b): the far field is the periodic field minus the direct near-field part, (1 - S(r)) E_direct,
# over the head's edges c -> a (exactly `switched_terms`' e_far_corr); identical to E_c beyond r_c, zero inside r_c - delta
r_c, delta = float(getattr(model, "r_c", 8.0)), float(getattr(model, "delta", 2.0)); w_cp = pair_width_cp(model.kernel.r_g, model.r_split)
c_idx, a_idx = b["edge_index"][0], b["edge_index"][1]; ev = (pos.detach()[a_idx] - pos.detach()[c_idx] + shifts.detach())
oms = 1.0 - switch(ev.norm(dim=-1), r_c, delta); E_dir = direct_field(ev, w_cp, model.kernel.eps_inf)
near_act = torch.zeros_like(E_act).index_add_(0, a_idx, (dq[c_idx] * oms)[:, None] * E_dir)
E_far = E_act - near_act
dv = pos.detach() - torch.tensor(vac, dtype=torch.float64, device=dev)[None]; dv -= torch.round(dv @ torch.linalg.inv(cell_t)) @ cell_t
E_unit_far = E_unit - (1.0 - switch(dv.norm(dim=-1), r_c, delta))[:, None] * direct_field(dv, w_cp, model.kernel.eps_inf)
(g_born,) = torch.autograd.grad(-(p * E_far).sum(), pos, retain_graph=True); F_born = (-g_born).detach().cpu().numpy()
(g_bf,) = torch.autograd.grad(-(p * E_act).sum(), pos, retain_graph=True); F_born_fullkernel = (-g_bf).detach().cpu().numpy()
e_cp = carrier_dipole_energy(pos, cell_t, dq, p, **kw); e_far = e_cp + (dq[c_idx] * oms * (p[a_idx] * E_dir).sum(-1)).sum()
(g_full,) = torch.autograd.grad(e_far, pos); F_full = (-g_full).detach().cpu().numpy()
print(f"E_cP full kernel {1e3*float(e_cp):.2f} meV; E_far (switched) {1e3*float(e_far):.2f} meV (head e_host {1e3*e_cp_head:.2f}); {time.time()-t0:.0f}s", flush=True)

Zu, _ = Base(args.base, dev).response(unit)                                 # the base's own pristine Born tensors
rep = unit.repeat(tt); Zrep = np.tile(Zu, (int(np.prod(tt)), 1, 1)); sites = [i for i, z in enumerate(rep.get_atomic_numbers()) if int(z) == 17]
Z = np.delete(Zrep, sites[0], axis=0)
refs = {"actual_dq": np.einsum("bij,bi->bj", Z, E_far.cpu().numpy()), "unit_charge": np.einsum("bij,bi->bj", Z, E_unit_far.cpu().numpy())}
E_np = {"actual_dq": E_far.cpu().numpy(), "unit_charge": E_unit_far.cpu().numpy()}
syms = np.array(atoms.get_chemical_symbols())
res = {"N": N, "tiling": list(tt), "r_core": args.r_core, "core": args.core, "model": args.model, "e_host_head_meV": 1e3 * e_cp_head, "E_cP_fullkernel_meV": 1e3 * float(e_cp), "E_far_switched_meV": 1e3 * float(e_far), "r_c": r_c, "delta": delta,
       "compactness": compact, "shells": []}
edges = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 24)]
print(f"{'shell':7s} {'n':>4s} | {'rms Fborn':>9s} | actual dq: {'rms Fref':>8s} {'ratio':>6s} {'cos':>6s} | unit: {'rms Fref':>8s} {'ratio':>6s} {'cos':>6s} | ratio by species (Cs/Pb/Cl, actual)")
for lo, hi in edges:
    m = (r > lo) & (r <= hi)
    if not m.any():
        continue
    rms = lambda x, mm=m: float(np.sqrt((x[mm] ** 2).sum(1).mean()))
    row = {"r_lo": lo, "r_hi": hi, "n": int(m.sum()), "rms_F_born_meV_A": 1e3 * rms(F_born), "rms_F_far_total_meV_A": 1e3 * rms(F_full), "rms_F_born_fullkernel_meV_A": 1e3 * rms(F_born_fullkernel)}
    for key, F_ref in refs.items():
        cos = float(np.mean(np.einsum("ij,ij->i", F_born[m], F_ref[m]) / np.maximum(np.linalg.norm(F_born[m], axis=1) * np.linalg.norm(F_ref[m], axis=1), 1e-12)))
        per = {s: (float(np.sqrt((F_born[m & (syms == s)] ** 2).sum(1).mean()) / max(np.sqrt((F_ref[m & (syms == s)] ** 2).sum(1).mean()), 1e-12)) if (m & (syms == s)).any() else None) for s in ("Cs", "Pb", "Cl")}
        row[key] = {"rms_F_ref_meV_A": 1e3 * rms(F_ref), "ratio": rms(F_born) / max(rms(F_ref), 1e-12), "cos": cos, "ratio_by_species": per,
                    "E_c_rms_V_A": float(np.sqrt((E_np[key][m] ** 2).sum(1).mean()))}
    res["shells"].append(row); A, U = row["actual_dq"], row["unit_charge"]
    print(f"{lo:>2d}-{hi:<4d} {row['n']:4d} | {row['rms_F_born_meV_A']:9.2f} | {A['rms_F_ref_meV_A']:8.2f} {A['ratio']:6.3f} {A['cos']:6.3f} | {U['rms_F_ref_meV_A']:8.2f} {U['ratio']:6.3f} {U['cos']:6.3f} | "
          + " / ".join(f"{v:.2f}" if v is not None else "-" for v in A["ratio_by_species"].values()), flush=True)
res["gate_actual_dq_ge_0.85_beyond_12A"] = bool(all(s["actual_dq"]["ratio"] >= 0.85 for s in res["shells"] if s["r_lo"] >= 12))
res["seconds"] = time.time() - t0
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out, f"{time.time()-t0:.0f}s; gate (actual dq >= 0.85 beyond 12 A):", res["gate_actual_dq_ge_0.85_beyond_12A"])
