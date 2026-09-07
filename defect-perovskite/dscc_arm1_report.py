"""Arm-1 report (plan section 7, v4.2): for every finished run -- the four registered
criteria per seed, the routing decision over seeds, held-out force RMSE by shell, the
bound-state precondition (C5), the leak readout (s_head at 79 vs 159), the post-hoc C_Q on
interpolation-only 159-atom held-out frames (C7), and the thermal-mean gap diagnostic.

    python defect-perovskite/dscc_arm1_report.py --runs /home/alex/runs/dscc/dscc_arm1_* --out report.json
Thresholds are the registered ones in dscc/arm1.py and dscc/precondition.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mace import tools                                                  # noqa: E402
from mace.modules.dscc import arm1, calibration, data as dd, precondition   # noqa: E402
from mace.modules.dscc.admission import collective_coordinate, slope_with_se   # noqa: E402
from mace.modules.dscc.train import load_frames, outer_folds, Trainer, TrainConfig   # noqa: E402
from mace.tools import torch_geometric                                  # noqa: E402


def batches(frames, indices, z_table, r_cut, device, bs=4):
    ds = dd.atomic_data([frames[i] for i in indices], z_table, r_cut)
    for b in torch_geometric.dataloader.DataLoader(ds, batch_size=bs):
        yield b.to(device).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--cf_dir", default=str(HERE / "dataset_cf"))
    ap.add_argument("--residuals", default=str(HERE / "dscc_base_residuals.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_neutral", type=int, default=200, help="neutral-vacancy frames per run for the diagnostics")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
    base_resid = {int(r["key"]): r for r in json.load(open(args.residuals))}
    report = {"runs": {}, "full": [], "control": [], "force_full": [], "force_control": []}
    for run in args.runs:
        run = Path(run)
        rec = json.load(open(run / "run_record.json"))
        cfg = rec["config"]
        # The converted checkpoint (plain base + dscc classes) when it exists: the pre-sweep
        # model.pt pickles reference deleted modules (plan section 3).
        ckpt = run / "model_dscc.pt" if (run / "model_dscc.pt").exists() else run / "model.pt"
        model = torch.load(ckpt, weights_only=False, map_location=args.device).to(args.device)
        model.eval()
        z_table = tools.AtomicNumberTable(model.atomic_numbers)
        fold_of = {int(k): v for k, v in rec["fold_of"].items()}
        held = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) == cfg["fold"]]
        neutral_vac = [m.index for m in metas if m.state.Q == 0 and m.n_atoms != 80][: args.max_neutral]
        # Diagnostics on neutral-vacancy frames (label-free quantities).
        records = arm1.run_diagnostics(model, list(batches(frames, neutral_vac, z_table, cfg["r_cut"], args.device)))
        summary = arm1.summarise(records)
        pre = precondition.bound_state_precondition(model, list(batches(frames, neutral_vac, z_table, cfg["r_cut"], args.device)))
        pre.pop("records", None)
        # Held-out forces and energies.
        trainer = Trainer(model, TrainConfig(**{k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__}),
                          frames, metas, fold_of, str(run))
        trainer.held_idx = held
        ev = trainer.evaluate(held, "held")
        # Leak readout: slope of the head energy (uncalibrated residual is E_label - E_base - J*;
        # the head's own d-slope is s_head = sQ_base_slope - residual slope) per size.
        leak = {}
        by_size = {}
        for e in ev["energies"]:
            if e["d"] is None:
                continue
            key = frames[e["index"]]
            base = base_resid.get(int(dd.frame_key(key.get_atomic_numbers(), key.get_positions(), np.array(key.get_cell()))))
            size = 1 if e["n"] < 120 else 2
            by_size.setdefault(size, {"d": [], "resid": [], "j": []})
            by_size[size]["d"].append(e["d"]); by_size[size]["resid"].append(e["resid_uncal"])
            if base is not None:
                by_size[size]["j"].append(base["resid"] - e["resid_uncal"])   # J* = (E_label - E_base) - resid
        for size, v in by_size.items():
            s_res, se_res, _ = slope_with_se(v["d"], v["resid"])
            s_head, se_head, _ = slope_with_se(v["d"], v["j"]) if len(v["j"]) > 2 else (float("nan"), float("nan"), 0.0)
            leak[size] = {"n": len(v["d"]), "residual_slope": s_res, "residual_slope_se": se_res,
                          "head_slope": s_head, "head_slope_se": se_head}
        # Post-hoc C_Q on 159-atom held-out frames inside the out-of-fold neutral d-range.
        cq = None
        if 2 in by_size:
            oof = [r for r in base_resid.values() if not r["charged"] and r["n"] >= 120 and r["d"] is not None and r["fold"] != cfg["fold"]]
            if oof:
                cq = calibration.profile_c_q(1, by_size[2]["d"], by_size[2]["resid"],
                                             [r["d"] for r in oof], [r["resid"] for r in oof])
        # Thermal-mean gap over pristine frames (diagnostic) and the static-cell gap.
        pristine = [m.index for m in metas if m.n_atoms == 80][:64]
        gaps = []
        for b in batches(frames, pristine, z_table, cfg["r_cut"], args.device, bs=8):
            with torch.no_grad():
                gaps.extend(model.pristine_gap(b).cpu().tolist())
        entry = {"config": cfg, "summary": summary, "precondition": pre,
                 "held_force_rmse": ev["force_rmse"], "shell_rmse": ev["shell_rmse"], "leak": leak,
                 "c_q": cq, "thermal_gap_mean": float(np.mean(gaps)), "thermal_gap_sd": float(np.std(gaps)),
                 "n_held": len(held), "n_neutral_diag": len(records)}
        report["runs"][run.name] = entry
        (report["full"] if cfg["directional"] else report["control"]).append(summary)
        (report["force_full"] if cfg["directional"] else report["force_control"]).append(ev["force_rmse"])
        print(run.name, json.dumps({k: entry[k] for k in ("held_force_rmse", "thermal_gap_mean")}),
              json.dumps({k: summary[k] for k in ("level_vs_bond_slope", "n_eff_p50", "pp_stop", "flank_tensor_over_bulk_spread", "cl_splitting_mean", "coefficient_saturation")}),
              "precondition", pre["passed"], pre["pass_fraction"], flush=True)
    if report["full"] and report["control"]:
        report["decision"] = arm1.decide(report["full"], report["control"], report["force_full"], report["force_control"])
        print("decision:", json.dumps(report["decision"]))
    json.dump(report, open(args.out, "w"), indent=1, default=str)
    print("saved", args.out)


if __name__ == "__main__":
    main()
