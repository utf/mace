"""Plan P Phase 2b §2, kernel validation (no training).

Part A, on the core-embedded structure from P0 (3,3,4 tiling, 2879 atoms): the SWITCHED far-field
force on each far ion against `Z* . E_c` by shell, for (i) a unit Gaussian charge at the vacancy
site and (ii) the head's own dq at the sharpened fill (sigma_s x 0.25). Expectation (registered):
ratio 0 inside 6 A, ramp 6-8 A, then the P0 ratios (0.78 -> 0.97) beyond 8 A.
  F_far,b = -d/dR_b [ -sum_a p_a . (E_c(R_a) - sum_c dq_c (1 - S(r_ac)) E_direct(R_a - R_c)) ]  (provider graph only)
Part B, on the training frames: RMS force per component of E_far, E_near (B_s, s = 1 x q0 of the
registered head) and their sum, by family (pristine, neutral 79/159, charged 79/159).

  python defect-perovskite/p2b_kernel_check.py --head ~/runs/dscc/dscc_p2a_s0 --provider ~/runs/dscc/dscc_p2b_s0 --out ~/runs/dscc/p2b/kernel_check.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import data as mace_data, tools                                  # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.polarisation import carrier_field, direct_field, pair_width_cp, switch, switched_terms   # noqa: E402
from mace.modules.dscc.train import load_frames                            # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="/home/alex/runs/dscc/dscc_p2a_s0", help="run dir of the head supplying dq / q0 (registered: A s0)")
ap.add_argument("--provider", default="/home/alex/runs/dscc/dscc_p2b_s0", help="run dir whose model carries the MACE-Field base as provider (arm B s0; S == 1 use only)")
ap.add_argument("--core", default="/home/alex/runs/dscc/p0/ladder3_var_large/core.json")
ap.add_argument("--cell", default="/home/alex/runs/dscc/p0/pristine_relaxed_v2.json")
ap.add_argument("--tiling", default="3,3,4"); ap.add_argument("--r_core", type=float, default=8.0)
ap.add_argument("--r_c", type=float, default=8.0); ap.add_argument("--delta", type=float, default=2.0)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe")); ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
ap.add_argument("--part", default="AB", help="A (embedded structure), B (training frames) or AB")
ap.add_argument("--base_f64", type=int, default=0, help="cast the bases to float64 (CPU only; the cueq base needs CUDA)")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
res = {"r_c": args.r_c, "delta": args.delta}

# ----------------------------------------------------------------- models
provider_model = torch.load(Path(args.provider) / "model.pt", weights_only=False, map_location=dev).to(dev).eval()
head = torch.load(Path(args.head) / "model.pt", weights_only=False, map_location=dev).to(dev).eval()
if args.base_f64:
    provider_model.base.double(); head.base.double()
head.export_pattern = True
z3 = tools.AtomicNumberTable([17, 55, 82])
R_G, R_SPLIT, EPS = provider_model.kernel.r_g, provider_model.r_split, provider_model.kernel.eps_inf
print(f"provider r_g {R_G} r_split {R_SPLIT} eps_inf {EPS}; head r_cut {head.r_cut}", flush=True)


def head_edges(pos, cell, r_cut):
    """Full neighbour list with image shifts on positions (numpy) -> (edge_index, edge_vector) tensors."""
    i, j, S = primitive_neighbor_list("ijS", [True] * 3, cell, pos, r_cut, numbers=None, self_interaction=False)
    ei = torch.tensor(np.stack([i, j]), dtype=torch.long, device=dev)
    shifts = torch.tensor(S @ cell, dtype=torch.float64, device=dev)
    return ei, shifts


def far_force(b, dq, edges_shift):
    """Switched far-field force through the provider graph only (dq fixed, E_c detached)."""
    pos = b["positions"].detach().clone().requires_grad_(True)
    cell_t = b["cell"].view(3, 3)
    shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell_t.expand(b["edge_index"].shape[1], 3, 3))
    d = dict(b); d["positions"] = pos; d["shifts"] = shifts
    species = b["node_attrs"].argmax(-1)
    p = provider_model._node_dipoles(d, species)
    E_c = carrier_field(pos.detach(), cell_t, dq, R_G, R_SPLIT, EPS, tol=provider_model.kernel.tol)
    ei, sh = edges_shift
    ev = (pos.detach()[ei[1]] - pos.detach()[ei[0]] + sh)
    one_minus_s = 1.0 - switch(ev.norm(dim=-1), args.r_c, args.delta)
    keep = one_minus_s > 0
    E_near_part = torch.zeros_like(E_c).index_add(0, ei[1][keep], (dq[ei[0][keep]] * one_minus_s[keep]).unsqueeze(-1) * direct_field(ev[keep], pair_width_cp(R_G, R_SPLIT), EPS))
    E_far = E_c - E_near_part
    e = -(p * E_far).sum()
    (g,) = torch.autograd.grad(e, pos)
    return (-g).detach(), E_far.detach(), E_c.detach()


# ----------------------------------------------------------------- Part A: embedded structure
if "A" not in args.part:
    shells = []
doc = json.load(open(Path(args.cell).expanduser()))
unit = Atoms(numbers=doc["numbers"], positions=doc["positions"], cell=doc["cell"], pbc=True)
tt = tuple(int(x) for x in args.tiling.split(","))
src = json.load(open(Path(args.core).expanduser()))
vac0, cell0 = vacancy_site(unit, tuple(src["tiling"]))
rel0 = np.array(src["rel_positions"]); u = np.array(src["displacements"])
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(len(u))}
atoms, moved, missing, inside = embed(unit, tt, core_disp, (vac0, cell0), args.r_core)
vac, cell = vacancy_site(unit, tt)
if "A" not in args.part:
    atoms = atoms[:0]
rvec = relative(np.array(atoms.get_positions()), vac, cell); r = np.linalg.norm(rvec, axis=1)
N = len(atoms); print(f"embedded {tt}: N={N}", flush=True)
if "A" in args.part:
    a = atoms.copy()
    b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, provider_model.r_cut), batch_size=1))).to(dev).to_dict()
    edges = head_edges(np.array(atoms.get_positions()), np.array(atoms.get_cell()), args.r_c)
    # Born tensors of the tiling counterparts (as p0_bornforce)
    prov = provider_model.base
    z_all = tools.AtomicNumberTable([int(z) for z in prov.atomic_numbers])
    cfg = mace_data.config_from_atoms(unit, key_specification=mace_data.KeySpecification.from_defaults(), head_name=list(prov.heads)[-1])
    ad = mace_data.AtomicData.from_config(cfg, z_table=z_all, cutoff=float(prov.r_max), heads=list(prov.heads))
    bu = next(iter(tg.dataloader.DataLoader([ad], batch_size=1))).to(dev).to_dict()
    Zu = prov(bu, training=True, compute_force=False, compute_polarization=True, compute_becs=True)["becs"].detach().cpu().numpy().reshape(len(unit), 3, 3)
    rep = unit.repeat(tt); Zrep = np.tile(Zu, (int(np.prod(tt)), 1, 1))
    sites = [i for i, z in enumerate(rep.get_atomic_numbers()) if int(z) == 17]
    Z = np.delete(Zrep, sites[0], axis=0)
    syms = np.array(atoms.get_chemical_symbols())
    shells = [(0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 24)]


    def shell_table(F, E_ref_field, label):
        F_ref = np.einsum("bij,bi->bj", Z, E_ref_field)
        rows = []
        for lo, hi in shells:
            m = (r > lo) & (r <= hi)
            if not m.any(): continue
            rms = lambda x: float(np.sqrt((x[m] ** 2).sum(1).mean()))
            cos = float(np.mean(np.einsum("ij,ij->i", F[m], F_ref[m]) / np.maximum(np.linalg.norm(F[m], axis=1) * np.linalg.norm(F_ref[m], axis=1), 1e-12)))
            rows.append({"r_lo": lo, "r_hi": hi, "n": int(m.sum()), "rms_F_far_meV_A": 1e3 * rms(F), "rms_ZE_meV_A": 1e3 * rms(F_ref),
                         "ratio": rms(F) / max(rms(F_ref), 1e-12), "cos": cos})
            print(f"  {label} {lo:>2d}-{hi:<3d} n={rows[-1]['n']:4d} F_far {rows[-1]['rms_F_far_meV_A']:8.3f} Z*E_c {rows[-1]['rms_ZE_meV_A']:8.3f} ratio {rows[-1]['ratio']:.3f} cos {cos:+.3f}", flush=True)
        return rows


    # (i) unit Gaussian charge at the vacancy site: add a ghost carrier site = the vacancy position
    pos_np = np.array(atoms.get_positions()); vac_pos = vac
    dq_unit = torch.zeros(N, dtype=torch.float64, device=dev)
    # carrier_field/direct_field act on atom sites; a ghost site is emulated by appending the vacancy as a
    # zero-dipole site in the field evaluation: use the nearest-Cl-vacancy trick of P0 -> place the unit
    # charge on the vacancy position by extending positions for the field only.
    b_ext = dict(b)
    pos_ext = torch.cat([b["positions"], torch.tensor(vac_pos, dtype=torch.float64, device=dev).view(1, 3)])
    dq_ext = torch.cat([dq_unit, torch.ones(1, dtype=torch.float64, device=dev)])
    cell_t = b["cell"].view(3, 3)
    E_c_unit = carrier_field(pos_ext, cell_t, dq_ext, R_G, R_SPLIT, EPS, tol=provider_model.kernel.tol)[:N]
    # near part of the ghost: direct field of the ghost at atoms within r_c
    d_ghost = pos_ext[:N] - pos_ext[N:N + 1]
    d_ghost = d_ghost - torch.round(d_ghost @ torch.linalg.inv(cell_t)) @ cell_t
    w_g = 1.0 - switch(d_ghost.norm(dim=-1), args.r_c, args.delta)
    E_far_unit = E_c_unit - w_g.unsqueeze(-1) * direct_field(d_ghost, pair_width_cp(R_G, R_SPLIT), EPS)
    pos_leaf = b["positions"].detach().clone().requires_grad_(True)
    shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell_t.expand(b["edge_index"].shape[1], 3, 3))
    dd_ = dict(b); dd_["positions"] = pos_leaf; dd_["shifts"] = shifts
    p_unit = provider_model._node_dipoles(dd_, b["node_attrs"].argmax(-1))
    (g_unit,) = torch.autograd.grad(-(p_unit * E_far_unit.detach()).sum(), pos_leaf)
    print("Part A (i): unit Gaussian charge at the vacancy site", flush=True)
    res["A_unit_charge"] = shell_table((-g_unit).detach().cpu().numpy(), E_c_unit.cpu().numpy(), "unit")

    # (ii) the head's own dq at the sharpened fill (sigma_s x 0.25)
    sig = head.sigma_s; head.sigma_s = 0.25 * sig
    bh = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, head.r_cut), batch_size=1))).to(dev).to_dict()
    out_h = head(dict(bh), compute_force=False)
    dq_head = out_h["dq"].detach(); head.sigma_s = sig
    print(f"head dq at sharpened fill: sum {float(dq_head.sum()):.3f}, max {float(dq_head.abs().max()):.3f}, within 5 A {float(dq_head[torch.tensor(r <= 5.0, device=dev)].sum()):.3f}", flush=True)
    F_far_h, E_far_h, E_c_h = far_force(b, dq_head, edges)
    print("Part A (ii): the head's dq", flush=True)
    res["A_head_dq"] = shell_table(F_far_h.cpu().numpy(), E_c_h.cpu().numpy(), "head")
    res["A_head_dq_summary"] = {"dq_sum": float(dq_head.sum()), "dq_within_5A": float(dq_head[torch.tensor(r <= 5.0, device=dev)].sum())}

# ----------------------------------------------------------------- Part B: training frames
frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
metas = [dd.frame_meta(i, a_, "CsPbCl3", pristine_atoms=80) for i, a_ in enumerate(frames)]
sel = [m for m in metas if m.index < len(frames)]
if args.max_frames: sel = sel[: args.max_frames]
fam_acc = {}
for m in sel:
    fam = ("pristine" if m.n_atoms % 80 == 0 and m.state.Q == 0 else ("charged" if m.state.Q != 0 else "neutral")) + f"_{m.n_atoms}"
    fr = frames[m.index]
    bh = next(iter(tg.dataloader.DataLoader(dd.atomic_data([fr], z3, head.r_cut), batch_size=1))).to(dev).to_dict()
    o = head(dict(bh), compute_force=False)
    dq = o["dq"].detach()
    if "sq0" in o["diagnostics"]:
        sq0 = torch.tensor(o["diagnostics"]["sq0"][0], device=dev) / float(head.pattern_scale())     # q0 (s = 1)
    else:
        sq0 = torch.zeros_like(dq)
    bp = next(iter(tg.dataloader.DataLoader(dd.atomic_data([fr], z3, provider_model.r_cut), batch_size=1))).to(dev).to_dict()
    pos = bp["positions"].detach().clone().requires_grad_(True); cell_t = bp["cell"].view(3, 3)
    shifts = torch.einsum("ei,eij->ej", bp["unit_shifts"], cell_t.expand(bp["edge_index"].shape[1], 3, 3))
    d = dict(bp); d["positions"] = pos; d["shifts"] = shifts
    p = provider_model._node_dipoles(d, bp["node_attrs"].argmax(-1))
    ei, sh = head_edges(np.array(fr.get_positions()), np.array(fr.get_cell()), args.r_c)
    ev = pos[ei[1]] - pos[ei[0]] + sh
    from mace.modules.dscc.polarisation import carrier_dipole_energy
    e_cp = carrier_dipole_energy(pos, cell_t, dq, p, R_G, R_SPLIT, EPS, tol=provider_model.kernel.tol)
    fc, en = switched_terms(dq, p, ei, ev, R_G, R_SPLIT, EPS, args.r_c, args.delta, sq0=sq0, near_r_split=2.5)
    e_far = e_cp + fc
    gf = torch.autograd.grad(e_far, pos, retain_graph=True)[0]; gn = torch.autograd.grad(en, pos, retain_graph=True)[0]
    acc = fam_acc.setdefault(fam, {"far": 0.0, "near": 0.0, "sum": 0.0, "n": 0, "e_far": [], "e_near": []})
    acc["far"] += float((gf ** 2).sum()); acc["near"] += float((gn ** 2).sum()); acc["sum"] += float(((gf + gn) ** 2).sum()); acc["n"] += gf.numel()
    acc["e_far"].append(float(e_far)); acc["e_near"].append(float(en))
    del p, pos
res["B_training_frames"] = {}
print("Part B: RMS force per component (meV/A) of E_far, E_near (s = 1), sum; by family", flush=True)
for fam, acc in sorted(fam_acc.items()):
    row = {"n_frames": len(acc["e_far"]), "rms_F_far": 1e3 * (acc["far"] / acc["n"]) ** 0.5, "rms_F_near": 1e3 * (acc["near"] / acc["n"]) ** 0.5,
           "rms_F_sum": 1e3 * (acc["sum"] / acc["n"]) ** 0.5, "E_far_meV_mean": 1e3 * float(np.mean(acc["e_far"])), "E_near_meV_mean": 1e3 * float(np.mean(acc["e_near"]))}
    res["B_training_frames"][fam] = row
    print(f"  {fam:14s} n={row['n_frames']:4d} F_far {row['rms_F_far']:7.3f} F_near {row['rms_F_near']:8.2f} sum {row['rms_F_sum']:8.2f} | E_far {row['E_far_meV_mean']:+8.2f} E_near {row['E_near_meV_mean']:+9.1f} meV", flush=True)
json.dump(res, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out, f"{time.time()-t0:.0f}s")
