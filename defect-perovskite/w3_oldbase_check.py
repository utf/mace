"""W3 prerequisite check: can an Arm 2+3 model (old base, trained before C13 and W2) still be
LOADED and EVALUATED by the current code? Evaluation only -- no retraining, no file rewritten.
Answers the one question that decides whether the cross-base comparison is possible at all."""
import json, sys, time
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace.modules.dscc.train import Trainer, TrainConfig, load_frames
from mace.modules.dscc import data as dd

run = "/home/alex/runs/dscc/dscc_arm23_B_A_lr_only_s0"
torch.set_default_dtype(torch.float64); t0 = time.time()
rec = json.load(open(f"{run}/run_record.json")); cfg_d = rec["config"]
model = torch.load(f"{run}/model.pt", weights_only=False, map_location="cuda").to("cuda").eval()
print("loaded:", type(model).__name__, "n_scalars", model.n_scalars, "r_max", float(model.r_max),
      "coupling", getattr(model, "coupling", None), f"[{time.time()-t0:.0f}s]")
HERE = W + "/defect-perovskite"
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
fold_of = {int(k): v for k, v in rec["fold_of"].items()}
cfg = TrainConfig(**{k: v for k, v in cfg_d.items() if k in TrainConfig.__dataclass_fields__})
tr = Trainer(model, cfg, frames, metas, fold_of, "/tmp/w3_oldbase_check")
held = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) == cfg.fold][:24]
print("held-out sample:", len(held), "frames")
out = tr.evaluate(held, "recheck")
print("force_rmse (vector RMS):", round(out["force_rmse"], 5), "-> per component", round(out["force_rmse"] / 3 ** 0.5 * 1000, 2), "meV/A")
print("shells:", {k: (round(v * 1000 / 3 ** 0.5, 2) if v else None) for k, v in out["shell_rmse"].items()})
print("near:", {k: (round(v * 1000 / 3 ** 0.5, 2) if v else None) for k, v in out["near_rmse"].items()}, out["near_counts"])
print(f"OK [{time.time()-t0:.0f}s]")
