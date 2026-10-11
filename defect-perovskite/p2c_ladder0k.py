"""Phase 2c readout (2026-10-11, user's question): the SAME 0 K relaxed core (Step 0's label-tiling structure (b)) embedded in the
perfect label lattice at the 79-, 159- and 639-atom sizes, read with the instrument (`filled_hamiltonians`): separation, N_eff,
sum dq within 5 A and on the two flanking Pb -- the size ladder with thermal disorder removed. Also the instrument on the 159-atom
+1 TRAINING frames (thermal DFT geometries) and, for reference, the 79-atom ones. Readout only.

  python defect-perovskite/p2c_ladder0k.py --models "B_s s2=~/runs/dscc/dscc_p2v8wbs_s2" "B_w s2=~/runs/dscc/dscc_p2cwbw_s2" ... --out ~/runs/dscc/p2c/ladder0k.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from ase import Atoms
from ase.io import read
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.precondition import hole_distribution              # noqa: E402
from mace.modules.dscc.train import vacancy_centre_full                    # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402
from dscc_embed_ladder import KEY_TOL, embed, relative, vacancy_site       # noqa: E402
from mace.modules.dscc.ladder import tiling_ladder                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="label=run_dir (model_avg.pt inside)")
ap.add_argument("--embed", default="/home/alex/runs/dscc/p2c/embed_b_label_w01.json")
ap.add_argument("--label_cell", default=str(HERE / "static_pristine_cell.json"))
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz")); ap.add_argument("--valid", default=str(HERE / "dataset_pbe" / "valid.xyz"))
ap.add_argument("--r_core", type=float, default=8.0); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = args.device; t0 = time.time()
lab = json.load(open(args.label_cell)); unit = Atoms(numbers=lab["numbers"], positions=lab["positions"], cell=lab["cell"], pbc=True)
rec = json.load(open(Path(args.embed).expanduser()))
# the core displacement field of structure (b), keyed by the position relative to the vacancy (the embed() convention)
ideal = tiling_ladder(unit, [(2, 2, 2)], vacancy_species=17, vacancy_index=0)[0]
vac, cell = vacancy_site(unit, (2, 2, 2))
p_ideal = np.array(ideal.get_positions()); p_b = np.array(rec["positions"])
assert len(p_ideal) == len(p_b) == 639
u = p_b - p_ideal; u -= np.round(u @ np.linalg.inv(cell)) @ cell
rel0 = relative(p_ideal, vac, cell); r0 = np.linalg.norm(rel0, axis=1)
core_disp = {tuple(np.round(rel0[i] / KEY_TOL).astype(np.int64)): u[i] for i in range(639) if r0[i] <= args.r_core}
print(f"core from (b): {len(core_disp)} atoms within {args.r_core} A, max |u| {np.linalg.norm(u, axis=1).max():.3f} A", flush=True)
frames = read(args.frames, index=":") + read(args.valid, index=":")
f159 = [a for a in frames if len(a) == 159 and int(a.info.get("cell_charge", 0)) == 1]
f79 = [a for a in frames if len(a) == 79 and int(a.info.get("cell_charge", 0)) == 1]
L159 = np.array(f159[0].cell.lengths()); L80 = np.array(unit.cell.lengths())
tiling159 = tuple(int(round(x)) for x in L159 / L80)
print(f"159-atom data cell {np.round(L159, 2)} = tiling {tiling159} of the label cell {np.round(L80, 2)}; {len(f159)} +1 frames, {len(f79)} 79-atom +1 frames", flush=True)
structs = {}
for t in ((1, 1, 1), tiling159, (2, 2, 2)):
    a, moved, missing, n_in = embed(unit, t, core_disp, (vac, cell), args.r_core)
    structs[t] = a; print(f"tiling {t}: {len(a)} atoms, {moved} moved, {missing} without a displacement key, {n_in} within r_core", flush=True)

def batch_of(atoms, z3, r_cut):
    a = atoms.copy(); a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "x", "source_dir": "x"})
    return next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, r_cut), batch_size=1))).to(dev).to_dict()

def read_structure(m, batch, vac_pos):
    H, numbers = m.filled_hamiltonians(batch)[0]
    sep, dq = hole_distribution(H, numbers, m.sigma_s)
    pos = batch["positions"]; c = batch["cell"].view(3, 3)
    d = pos - vac_pos; d = d - torch.round(d @ torch.linalg.inv(c)) @ c; r = d.norm(dim=-1)
    found = vacancy_centre_full(pos, c, np.array(numbers)); fl = [int(x) for x in found[1]] if found is not None else []
    out = {"separation": sep, "n_eff": float(1.0 / (dq ** 2).sum()), "dq5": float(dq[r <= 5].sum()), "dq_flank": float(dq[fl].sum()) if fl else None, "n_atoms": len(numbers)}
    del H; return out

results = {"core": "Step 0 (b): label tiling, core re-relaxed by w01 B_s s2", "tiling159": list(tiling159), "ladder": {}, "frames159": {}, "frames79": {}}
for spec in args.models:
    label, run = spec.split("=", 1)
    m = torch.load(Path(run).expanduser() / "model_avg.pt", weights_only=False, map_location=dev).eval()
    z3 = tools.AtomicNumberTable(m.atomic_numbers)
    rows = {}
    for t, a in structs.items():
        v, c = vacancy_site(unit, t); b = batch_of(a, z3, m.r_cut)
        rows[str(len(a))] = read_structure(m, b, torch.tensor(v, dtype=torch.float64, device=dev))
    results["ladder"][label] = rows
    def frame_stats(fs):
        recs = []
        for a in fs:
            b = batch_of(a, z3, m.r_cut)
            found = vacancy_centre_full(b["positions"], b["cell"].view(3, 3), a.get_atomic_numbers())
            if found is None: continue
            rad, fl = found; H, numbers = m.filled_hamiltonians(b)[0]; sep, dq = hole_distribution(H, numbers, m.sigma_s)
            recs.append({"separation": sep, "n_eff": float(1.0 / (dq ** 2).sum()), "dq5": float(dq[rad <= 5].sum()), "dq_flank": float(dq[[int(x) for x in fl]].sum())})
            del H
        return {k: float(np.median([r[k] for r in recs])) for k in ("separation", "n_eff", "dq5", "dq_flank")} | {"n": len(recs), "n_eff_p10": float(np.percentile([r["n_eff"] for r in recs], 10)), "n_eff_p90": float(np.percentile([r["n_eff"] for r in recs], 90))}
    results["frames159"][label] = frame_stats(f159)
    results["frames79"][label] = frame_stats(f79[:300])
    print(f"[{time.time()-t0:5.0f}s] {label}: ladder " + " | ".join(f"{n}: sep {r['separation']:.3f} N_eff {r['n_eff']:.1f} dq5 {r['dq5']:.2f} flank {r['dq_flank']}" for n, r in rows.items())
          + f" || 159-atom +1 frames (n {results['frames159'][label]['n']}): sep {results['frames159'][label]['separation']:.3f} N_eff {results['frames159'][label]['n_eff']:.1f} (p10 {results['frames159'][label]['n_eff_p10']:.1f}, p90 {results['frames159'][label]['n_eff_p90']:.1f}) dq5 {results['frames159'][label]['dq5']:.2f} flank {results['frames159'][label]['dq_flank']:.2f}"
          + f" || 79-atom +1 frames (first 300): sep {results['frames79'][label]['separation']:.3f} N_eff {results['frames79'][label]['n_eff']:.1f} dq5 {results['frames79'][label]['dq5']:.2f} flank {results['frames79'][label]['dq_flank']:.2f}", flush=True)
    del m; torch.cuda.empty_cache()
json.dump(results, open(Path(args.out).expanduser(), "w"), indent=1); print("saved", args.out)
