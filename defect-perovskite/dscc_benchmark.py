"""Plan section 8: the 2x benchmark. Wall time per MD-style step (energy + forces) of the
D-SCC model against the base alone, on the named hardware, fixed batch size, fixed
`tol_q`/`tol_E`, warm-started along a registered trajectory (consecutive frames of one MD
run, `dq` of the previous frame as the start), the base's early feature block recomputed
for derivatives. Reports median and 95th percentile of the ratio; both must be <= 2.
    python defect-perovskite/dscc_benchmark.py --model /home/alex/runs/dscc/<run>/model_dscc.pt --device cuda --frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                            # noqa: E402
from mace.modules.dscc import data as dd                          # noqa: E402
from mace.modules.models import ScaleShiftMACE                    # noqa: E402
from mace.tools import torch_geometric                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--frames", type=int, default=40, help="consecutive charged frames of one source (the trajectory)")
    ap.add_argument("--size", type=int, default=79)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    import ase.io
    frames = [a for a in ase.io.read(f"{args.dataset}/train.xyz", ":") if int(a.info["cell_charge"]) != 0 and len(a) == args.size]
    frames = frames[: args.frames]
    model = torch.load(args.model, weights_only=False, map_location=args.device).to(args.device).eval()
    z_table = tools.AtomicNumberTable(model.atomic_numbers)
    base = model.base

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def batch(atoms, cutoff):
        d = dd.atomic_data([atoms], z_table, cutoff)
        return next(iter(torch_geometric.dataloader.DataLoader(d, batch_size=1))).to(args.device).to_dict()

    # warm-up
    b = batch(frames[0], model.r_cut); model(b, compute_force=True)
    ScaleShiftMACE.forward(base, batch(frames[0], float(base.r_max)), compute_force=True); sync()
    t_head, t_base, iters = [], [], []
    warm = None
    for atoms in frames:
        bb = batch(atoms, float(base.r_max))
        sync(); t0 = time.perf_counter()
        ScaleShiftMACE.forward(base, bb, compute_force=True); sync()
        t_base.append(time.perf_counter() - t0)
        bh = batch(atoms, model.r_cut)
        sync(); t0 = time.perf_counter()
        out = model(bh, compute_force=True, warm_start=warm)          # initialisation iii along the trajectory
        sync(); t_head.append(time.perf_counter() - t0)
        warm = [out["dq"].detach()]
        iters.append(out["diagnostics"].get("iterations", [0])[0])
    ratio = np.array(t_head) / np.array(t_base)
    report = {"model": args.model, "device": args.device, "hardware": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
              "n_frames": len(frames), "size": args.size, "coupling": bool(model.coupling), "route_b": bool(model.route_b),
              "base_ms_median": 1000 * float(np.median(t_base)), "model_ms_median": 1000 * float(np.median(t_head)),
              "ratio_median": float(np.median(ratio)), "ratio_p95": float(np.percentile(ratio, 95)),
              "scf_iterations_median": float(np.median(iters)) if iters else None,
              "passes_2x": bool(np.median(ratio) <= 2.0 and np.percentile(ratio, 95) <= 2.0),
              "note": "the model's time includes the base's full forward (the head needs block 0 attached; E_base is not cached at inference)"}
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
