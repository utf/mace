"""W3 prerequisite: re-evaluate the old-base Arm 2+3 arms that W3 repeats (Phi = 0, LR-only,
B' LR-only) under the CURRENT evaluation -- vacancy-side centre (W0.1) and the W0.2 shells --
so the cross-base comparison is made on one convention. Evaluation only: no model is retrained
and no campaign file is overwritten; the output is a new JSON.
"""
import json, sys, time
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace.modules.dscc.train import Trainer, TrainConfig, load_frames
from mace.modules.dscc import data as dd

ARMS = ["B_A_phi0", "B_A_lr_only", "B_Bp_lr_only"]
SEEDS = [0, 1, 2, 3, 4, 6]
OUT = "/home/alex/runs/dscc/w3_oldbase_reeval.json"
torch.set_default_dtype(torch.float64); t0 = time.time()
HERE = W + "/defect-perovskite"
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
out = {}
for arm in ARMS:
    for seed in SEEDS:
        name = f"dscc_arm23_{arm}_s{seed}"
        run = f"/home/alex/runs/dscc/{name}"
        try:
            rec = json.load(open(f"{run}/run_record.json")); cfg_d = rec["config"]
            model = torch.load(f"{run}/model.pt", weights_only=False, map_location="cuda").to("cuda").eval()
        except Exception as e:
            print(f"{name}: SKIP ({type(e).__name__}: {e})", flush=True); continue
        cfg = TrainConfig(**{k: v for k, v in cfg_d.items() if k in TrainConfig.__dataclass_fields__})
        fold_of = {int(k): v for k, v in rec["fold_of"].items()}
        tr = Trainer(model, cfg, frames, metas, fold_of, "/tmp/w3_reeval")
        held = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) == cfg.fold]
        r = tr.evaluate(held, name)
        r.pop("energies", None)
        r["n_held"] = len(held); r["fold"] = cfg.fold; r["seed"] = seed; r["arm"] = arm
        out[name] = r
        print(f"{name}: {len(held)} frames, {1000*r['force_rmse']/3**0.5:.2f} meV/A per component,"
              f" 2-4 {1000*(r['shell_rmse']['2-4'] or 0)/3**0.5:.2f}, 4-8 {1000*(r['shell_rmse']['4-8'] or 0)/3**0.5:.2f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
        del model, tr
        torch.cuda.empty_cache()
json.dump(out, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT, len(out), "runs", flush=True)
