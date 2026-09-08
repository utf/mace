"""Arm 2+3 report and selection (v4.1 layout, v4.2 selection, thresholds in dscc/arm23.py):
per run -- held-out charged force RMSE and shell RMSE, the 159-atom shape residual slope
after the post-hoc C_Q, N_eff p50 on held-out frames, the final single-valuedness failing
fraction and converged fraction, f_SR on a sample, the pattern scale s -- then per
configuration over seeds, the gates and the joint selection.
    python defect-perovskite/dscc_arm23_report.py --runs /home/alex/runs/dscc/dscc_arm23_* --out arm23_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dscc_spike_log import spikes_of                      # noqa: E402
sys.path.insert(0, str(HERE.parent))

from mace import tools                                                              # noqa: E402
from mace.modules.dscc import arm23, calibration, data as dd                          # noqa: E402
from mace.modules.dscc.admission import collective_coordinate, slope_with_se          # noqa: E402
from mace.modules.dscc.kernels import f_sr, f_sr_abs, gamma_matrix, kernel_components   # noqa: E402
from mace.modules.dscc.train import Trainer, TrainConfig, load_frames                # noqa: E402
from mace.tools import torch_geometric                                              # noqa: E402


def parse_name(name: str):
    m = re.match(r"dscc_arm23_(?P<regime>[AB])_(?P<route>A|Bp)_(?P<mode>phi0|lr_only|lr_u|full|lambda1)_s(?P<seed>\d+)$", name)
    return m.groupdict() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
    ap.add_argument("--residuals", default=str(HERE / "dscc_base_residuals.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--f_sr_frames", type=int, default=24)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
    base_resid = {int(r["key"]): r for r in json.load(open(args.residuals))}
    per_run = {}
    by_config = defaultdict(list)
    for run in args.runs:
        run = Path(run)
        info = parse_name(run.name)
        if info is None or not (run / "held_final.json").exists():
            continue
        rec = json.load(open(run / "run_record.json")); cfg = rec["config"]
        hist = json.load(open(run / "history.json"))
        spikes = spikes_of(run)                       # v4.3: loss-spike events logged, not acted on
        held = json.load(open(run / "held_final.json"))
        model_path = run / ("model_dscc.pt" if (run / "model_dscc.pt").exists() else "model.pt")
        model = torch.load(model_path, weights_only=False, map_location=args.device).to(args.device).eval()
        z_table = tools.AtomicNumberTable(model.atomic_numbers)
        fold_of = {int(k): v for k, v in rec["fold_of"].items()}
        held_idx = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) == cfg["fold"]]
        # N_eff and f_SR on a held-out sample (charged frames).
        sample = held_idx[: args.f_sr_frames]
        n_effs, f_srs, f_srs_abs, unconv = [], [], [], 0
        ds = dd.atomic_data([frames[i] for i in sample], z_table, cfg["r_cut"])
        for b in torch_geometric.dataloader.DataLoader(ds, batch_size=1):
            batch = b.to(args.device).to_dict()
            with torch.no_grad():
                out = model(batch, compute_force=False)
            dq = out["dq"].detach()
            n_effs.append(float(1.0 / (dq ** 2).sum()))
            if cfg["coupling"]:
                pos, cell = batch["positions"], batch["cell"].view(3, 3)
                k_sr, k_lr = kernel_components(pos, cell, model.kernel)
                f_srs.append(float(f_sr(dq, k_sr, k_lr)))
                f_srs_abs.append(float(f_sr_abs(dq, k_sr, k_lr)))
                unconv += int(not out["diagnostics"]["converged"][0])
        # 159-atom shape residual after post-hoc C_Q (interpolation-only frames).
        d159, r159, d_oof, r_oof = [], [], [], []
        for e in held["energies"]:
            if e["n"] >= 120 and e["d"] is not None:
                d159.append(e["d"]); r159.append(e["resid_uncal"])
        for r in base_resid.values():
            if not r["charged"] and r["n"] >= 120 and r["d"] is not None and r["fold"] != cfg["fold"]:
                d_oof.append(r["d"]); r_oof.append(r["resid"])
        cq = calibration.profile_c_q(1, d159, r159, d_oof, r_oof) if (d159 and d_oof) else None
        shape_err = abs(cq["shape_residual_slope"]) if cq and np.isfinite(cq.get("shape_residual_slope", float("nan"))) else float("nan")
        last = hist[-1]
        # v4.5: the check's failing fraction is read at its WORST epoch ("failure fails the
        # arm"), and the run-level flag written by the trainer is carried.
        # Epoch 0's check is the initialised-model diagnostic (v4.2 section 5: the root rule
        # applies to trained models); the ceiling is read over epochs >= 1, from the history
        # itself (the trainer's flag is not trusted for this: earlier code counted epoch 0).
        sv_fractions = [float((e.get("single_valued") or {}).get("fraction", 0.0)) for e in hist]
        sv_trained = sv_fractions[1:] if len(sv_fractions) > 1 else sv_fractions
        sv_max = max(sv_trained) if sv_trained else 0.0
        ceiling = float(cfg.get("single_valued_ceiling", 0.10))
        # C10 ruling (2026-09-08): the gate is read on the final model and the last ten
        # epochs; the early-epoch failures are the initialised-map transient (recorded).
        sv_last10 = max(sv_fractions[-10:]) if sv_fractions else 0.0
        over = [k + 1 for k, f in enumerate(sv_trained) if f > ceiling]
        entry = {"config": info, "force_rmse": held["force_rmse"], "shell_rmse": held["shell_rmse"],
                 "shape_slope_err": shape_err, "c_q": cq, "n_eff_p50": float(np.median(n_effs)) if n_effs else float("nan"),
                 "sv_fraction": sv_max, "sv_fraction_last": sv_fractions[-1] if sv_fractions else 0.0,
                 "sv_init_fraction": sv_fractions[0] if sv_fractions else 0.0,
                 "arm_failed_single_valuedness": any(f > ceiling for f in sv_trained),
                 "sv_ceiling_exceeded_epochs": over,
                 "sv_fraction_last10": sv_last10, "sv_transient_end": (max(over) if over else 0),
                 "sv_pooled_fraction": (float(np.mean(sv_trained)) if sv_trained else 0.0),
                 "f_sr_abs": float(np.median(f_srs_abs)) if f_srs_abs else None,
                 "converged_fraction": 1.0 - unconv / max(len(sample), 1),
                 "f_sr": float(np.median(f_srs)) if f_srs else None, "s": last.get("s"), "lambda_dir": last.get("lambda_dir"),
                 "u_eff": last.get("u_eff"), "epochs": len(hist), "spikes": spikes}
        per_run[run.name] = entry
        key = f"{info['regime']}_{info['route']}_{info['mode']}"
        by_config[key].append(entry)
        print(run.name, json.dumps({k: entry[k] for k in ("force_rmse", "shape_slope_err", "n_eff_p50", "sv_fraction", "converged_fraction", "f_sr", "lambda_dir", "s")}),
              "spikes", len(spikes["spikes"]), flush=True)
    configs = []
    for key, entries in by_config.items():
        regime, route, mode = key.split("_", 2)
        configs.append(arm23.ConfigSummary(
            name=key, regime=regime, route=route, coupling=mode,
            force_rmse=[e["force_rmse"] for e in entries],
            shape_slope_err=[e["shape_slope_err"] for e in entries if np.isfinite(e["shape_slope_err"])] or [float("nan")],
            n_eff_p50=[e["n_eff_p50"] for e in entries], sv_fraction=[e["sv_fraction"] for e in entries],
            converged_fraction=[e["converged_fraction"] for e in entries],
            f_sr=[e["f_sr"] for e in entries if e["f_sr"] is not None],
            f_sr_abs=[e["f_sr_abs"] for e in entries if e.get("f_sr_abs") is not None],
            sv_fraction_last10=[e["sv_fraction_last10"] for e in entries],
            sv_transient_end=[e["sv_transient_end"] for e in entries],
            far_field_4_8=[e["shell_rmse"].get("4-6") or 0.0 for e in entries],
            s_scale=[e["s"] for e in entries if route == "Bp" and e["s"] is not None]))
    decision = arm23.select(configs)                  # v4.3 as registered
    decision_v44 = arm23.select_v44(configs)          # the 2026-09-08 rule, post hoc
    report = {"runs": per_run, "configs": {c.name: c.__dict__ for c in configs}, "decision": decision,
              "decision_v44": decision_v44}
    json.dump(report, open(args.out, "w"), indent=1, default=str)
    print("decision:", json.dumps({k: v for k, v in decision.items() if k != "gates"}, default=str))
    print("saved", args.out)


if __name__ == "__main__":
    main()
