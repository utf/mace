"""Plan section 6: fit `C_Q` in closed form and store it in the checkpoint.

`C_Q` is one additive constant per formal charge (`C_0 = 0`). Under the quadratic energy loss
its optimum is the MEAN residual, so there is nothing to learn -- W5's trainer profiles it out
and tracks a running estimate, but never writes it into the model. An uncalibrated checkpoint
therefore returns energies offset by ~10 eV per charge, which is right for the loss and wrong
for a caller. This fits the constant on the run's OWN TRAINING frames (never the held-out
fold: the constant is a parameter, and fitting it on the evaluation set would be leakage) and
writes `model_calibrated.pt` next to `model.pt`, leaving the measured artefact untouched.

    python defect-perovskite/dscc_calibrate.py --runs /home/alex/runs/dscc/dscc_w6_s* \
        --out /home/alex/runs/dscc/calibration_v5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.train import load_frames                            # noqa: E402
from mace.tools import torch_geometric                                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite_model", action="store_true",
                    help="write back to model.pt instead of model_calibrated.pt")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
    by_index = {m.index: m for m in metas}
    z_table = tools.AtomicNumberTable([17, 55, 82])
    report = {}
    for run in args.runs:
        run = Path(run)
        rec = json.load(open(run / "run_record.json"))
        fold_of = {int(k): int(v) for k, v in rec["fold_of"].items()}
        fold = int(rec["config"]["fold"])
        train_idx = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) != fold]
        model = torch.load(run / "model.pt", weights_only=False, map_location=args.device).to(args.device).eval()
        ds = dd.atomic_data([frames[i] for i in train_idx], z_table, model.r_cut)
        loader = torch_geometric.dataloader.DataLoader(ds, batch_size=args.batch_size)
        resid: dict[int, list] = {}
        k = 0
        for b in loader:
            batch = {kk: (v.to(args.device) if torch.is_tensor(v) else v) for kk, v in b.to_dict().items()}
            with torch.no_grad():
                out = model(batch, compute_force=False)
            n_graphs = int(batch["ptr"].numel() - 1)
            for g in range(n_graphs):
                q = int(by_index[train_idx[k + g]].state.Q)
                resid.setdefault(q, []).append(
                    float(out["energy_uncalibrated"][g] - batch["energy"][g]))
            k += n_graphs
        # `energy = energy_uncalibrated + C_Q`, so C_Q is MINUS the mean residual.
        values = {q: -float(np.mean(v)) for q, v in resid.items() if q != 0}
        record = {"method": "closed form (mean residual) on the run's training fold",
                  "n_frames": {str(q): len(v) for q, v in resid.items()},
                  "mean_residual_eV": {str(q): float(np.mean(v)) for q, v in resid.items()},
                  "sd_residual_eV": {str(q): float(np.std(v)) for q, v in resid.items()},
                  "fold": fold, "run": str(run), "dataset": args.dataset,
                  "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        model.set_calibration(values, record)
        path = run / ("model.pt" if args.overwrite_model else "model_calibrated.pt")
        torch.save(model, path)
        report[run.name] = {"c_q": {str(q): v for q, v in values.items()}, "record": record,
                            "written": str(path)}
        print(run.name, {q: round(v, 4) for q, v in values.items()},
              "sd", {q: round(float(np.std(v)), 4) for q, v in resid.items()}, flush=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
