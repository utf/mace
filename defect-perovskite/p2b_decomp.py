"""Plan P Phase 2b §6: E_near / E_far energy and force decomposition of the adopted arm on its
held-out charged frames (per seed): energies per frame (eV) and per-component RMS forces of each
term (meV/Å), with the head's own dq, s q0 and the base's dipoles at the trained weights.

  python defect-perovskite/p2b_decomp.py --runs ~/runs/dscc/dscc_p2bs_s* --out ~/runs/dscc/p2b/decomp_bs.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from ase.neighborlist import primitive_neighbor_list
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import tools                                                  # noqa: E402
from mace.modules.dscc import data as dd                                # noqa: E402
from mace.modules.dscc.polarisation import carrier_dipole_energy, switched_terms   # noqa: E402
from mace.modules.dscc.train import load_frames                         # noqa: E402
from mace.tools import torch_geometric as tg                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe")); ap.add_argument("--device", default="cuda"); ap.add_argument("--max_frames", type=int, default=0)
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device
frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
z3 = tools.AtomicNumberTable([17, 55, 82])
res = {}
for run in map(Path, args.runs):
    rec = json.load(open(run / "run_record.json")); cfg = rec["config"]; fold_of = {int(k): v for k, v in rec["fold_of"].items()}
    held = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) == cfg["fold"]]
    if args.max_frames: held = held[: args.max_frames]
    model = torch.load(run / "model.pt", weights_only=False, map_location=dev).to(dev)
    avg = run / "model_avg.pt"
    if avg.exists():
        model.load_state_dict(torch.load(avg, weights_only=False, map_location=dev))
    model.eval(); model.export_pattern = True
    acc = {"far": 0.0, "near": 0.0, "sum": 0.0, "n": 0, "e_far": [], "e_near": [], "e_cp_periodic": []}
    for idx in held:
        fr = frames[idx]
        b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([fr], z3, model.r_cut), batch_size=1))).to(dev).to_dict()
        with torch.no_grad():
            o = model(dict(b), training=False, compute_force=False)
        dq = o["dq"].detach().clone(); sq0 = torch.tensor(o["diagnostics"]["sq0"][0], device=dev)
        acc["e_far"].append(o["diagnostics"]["e_far"][0]); acc["e_near"].append(o["diagnostics"]["e_near"][0])
        pos = b["positions"].detach().clone().requires_grad_(True); cell_t = b["cell"].view(3, 3)
        shifts = torch.einsum("ei,eij->ej", b["unit_shifts"], cell_t.expand(b["edge_index"].shape[1], 3, 3))
        d = dict(b); d["positions"] = pos; d["shifts"] = shifts
        p = model._node_dipoles(d, b["node_attrs"].argmax(-1))
        i, j, S = primitive_neighbor_list("ijS", [True] * 3, np.array(fr.get_cell()), np.array(fr.get_positions()), float(model.r_c), numbers=None, self_interaction=False)
        ei = torch.tensor(np.stack([i, j]), dtype=torch.long, device=dev); sh = torch.tensor(S @ np.array(fr.get_cell()), dtype=torch.float64, device=dev)
        ev = pos[ei[1]] - pos[ei[0]] + sh
        e_cp = carrier_dipole_energy(pos, cell_t, dq, p, model.kernel.r_g, model.r_split, model.kernel.eps_inf, tol=model.kernel.tol)
        fc, en = switched_terms(dq, p, ei, ev, model.kernel.r_g, model.r_split, model.kernel.eps_inf, float(model.r_c), float(model.delta), sq0=sq0, near_r_split=float(model.near_r_split))
        e_far = e_cp + fc
        gf = torch.autograd.grad(e_far, pos, retain_graph=True)[0]; gn = torch.autograd.grad(en, pos)[0]
        acc["far"] += float((gf ** 2).sum()); acc["near"] += float((gn ** 2).sum()); acc["sum"] += float(((gf + gn) ** 2).sum()); acc["n"] += gf.numel()
        acc["e_cp_periodic"].append(float(e_cp))
        del o, p, pos, gf, gn; torch.cuda.empty_cache()
    row = {"seed": cfg["seed"], "fold": cfg["fold"], "n_held": len(held), "s": float(model.pattern_scale()),
           "rms_F_far_meV_A": 1e3 * (acc["far"] / acc["n"]) ** 0.5, "rms_F_near_meV_A": 1e3 * (acc["near"] / acc["n"]) ** 0.5, "rms_F_sum_meV_A": 1e3 * (acc["sum"] / acc["n"]) ** 0.5,
           "E_far_meV": {"mean": 1e3 * float(np.mean(acc["e_far"])), "rms": 1e3 * float(np.sqrt(np.mean(np.square(acc["e_far"]))))},
           "E_near_meV": {"mean": 1e3 * float(np.mean(acc["e_near"])), "rms": 1e3 * float(np.sqrt(np.mean(np.square(acc["e_near"]))))},
           "E_cP_periodic_meV_mean": 1e3 * float(np.mean(acc["e_cp_periodic"]))}
    res[run.name] = row
    print(f"{run.name}: n={len(held)} s={row['s']:.3f} F_far {row['rms_F_far_meV_A']:.2f} F_near {row['rms_F_near_meV_A']:.2f} sum {row['rms_F_sum_meV_A']:.2f} meV/A | E_far {row['E_far_meV']['mean']:+.1f} E_near {row['E_near_meV']['mean']:+.1f} meV", flush=True)
    del model; torch.cuda.empty_cache()
json.dump(res, open(args.out, "w"), indent=1); print("saved", args.out)
