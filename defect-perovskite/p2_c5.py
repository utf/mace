"""Plan P Phase 2 localisation read: the registered C5 bound-state precondition (separation of the
vacancy level from the band edge, hole participation number N_eff) on the neutral-vacancy frames,
per finished head checkpoint (epoch-averaged `model_avg.pt` where present). Thresholds are the
registered ones in mace/modules/dscc/precondition.py (delta_c 0.2 eV, n_loc 4, fraction 0.95).

  python defect-perovskite/p2_c5.py --runs ~/runs/dscc/dscc_p2*_s* --out ~/runs/dscc/p2/P2_C5.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                                  # noqa: E402
from mace.modules.dscc import data as dd, precondition                  # noqa: E402
from mace.modules.dscc.train import load_frames                         # noqa: E402
from mace.tools import torch_geometric                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe")); ap.add_argument("--device", default="cuda")
ap.add_argument("--max_neutral", type=int, default=0, help="0 = all neutral-vacancy frames")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
neutral_vac = [m.index for m in metas if m.state.Q == 0 and m.n_atoms != 80]
if args.max_neutral: neutral_vac = neutral_vac[: args.max_neutral]
out = {}
for run in map(Path, args.runs):
    ck = run / "model_avg.pt" if (run / "model_avg.pt").exists() else run / "model.pt"
    if not ck.exists(): continue
    obj = torch.load(ck, weights_only=False, map_location=args.device)
    if isinstance(obj, dict):
        model = torch.load(run / "model.pt", weights_only=False, map_location=args.device); model.load_state_dict(obj)
    else:
        model = obj
    model = model.to(args.device).eval()
    z_table = tools.AtomicNumberTable(model.atomic_numbers)
    r_cut = json.load(open(run / "run_record.json"))["config"]["r_cut"]
    ds = dd.atomic_data([frames[i] for i in neutral_vac], z_table, r_cut)
    batches = [b.to(args.device).to_dict() for b in torch_geometric.dataloader.DataLoader(ds, batch_size=4)]
    pre = precondition.bound_state_precondition(model, batches)
    sep = np.array([r["separation"] for r in pre["records"]]); ne = np.array([r["n_eff"] for r in pre["records"]])
    out[run.name] = {"checkpoint": ck.name, "n_frames": pre["n_frames"], "pass_fraction": pre["pass_fraction"], "passed": pre["passed"],
                     "separation_meV": {"p10": 1e3 * float(np.percentile(sep, 10)), "p50": 1e3 * float(np.median(sep)), "p90": 1e3 * float(np.percentile(sep, 90))},
                     "n_eff": {"p10": float(np.percentile(ne, 10)), "p50": float(np.median(ne)), "p90": float(np.percentile(ne, 90)), "max": float(ne.max())}}
    print(f"{run.name}: {ck.name} n={pre['n_frames']} pass {pre['pass_fraction']:.3f} sep p50 {1e3*np.median(sep):.0f} meV n_eff p50 {np.median(ne):.2f} p90 {np.percentile(ne,90):.2f}", flush=True)
    del model, batches; torch.cuda.empty_cache()
json.dump(out, open(args.out, "w"), indent=1); print("saved", args.out)
