"""Arm 4 readout (plan section 2.9 / 2.10): the carrier localisation of the matched-kernel F-SCC
comparators against their trained `lambda_dir`, with the D-SCC arms of the same protocol as the
contrast. Evaluation only."""
import json, sys, time
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace.modules.dscc.train import load_frames
from mace.modules.dscc import data as dd
from mace import tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64); t0 = time.time()
HERE = W + "/defect-perovskite"
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
metas = [dd.frame_meta(i, a, "CsPbCl3", pristine_atoms=80) for i, a in enumerate(frames)]
RUNS = [(f"dscc_arm4_matched_s{s}", "F-SCC matched") for s in (0, 1, 2)] + \
       [(f"dscc_arm23_B_A_full_s{s}", "D-SCC full") for s in (0, 1, 2)] + \
       [(f"dscc_arm23_B_Bp_lr_only_s{s}", "D-SCC B' LR-only") for s in (0, 1, 2)]
out = []
for name, label in RUNS:
    run = f"/home/alex/runs/dscc/{name}"
    try:
        rec = json.load(open(f"{run}/run_record.json")); cfg = rec["config"]
        model = torch.load(f"{run}/model.pt", weights_only=False, map_location="cuda").to("cuda").eval()
    except Exception as e:
        print(f"{name}: SKIP {type(e).__name__}", flush=True); continue
    fold_of = {int(k): v for k, v in rec["fold_of"].items()}
    held = [m.index for m in metas if m.state.Q != 0 and fold_of.get(m.index) == cfg["fold"]][:24]
    z = tools.AtomicNumberTable([int(x) for x in model.atomic_numbers])
    ds = dd.atomic_data([frames[i] for i in held], z, cfg["r_cut"])
    n_eff, r_eff = [], []
    for b in torch_geometric.dataloader.DataLoader(ds, batch_size=1):
        batch = b.to("cuda").to_dict()
        with torch.no_grad():
            o = model(batch, compute_force=False)
        dq = o["dq"].detach().reshape(-1)
        w = dq ** 2; tot = float(w.sum())
        if tot <= 0:
            continue
        n_eff.append(1.0 / float((w / tot ** 0.5 ** 0).pow(0).sum() * 0 + (dq ** 2).sum() ** 2 / (dq ** 4).sum()) if False else float((dq ** 2).sum() ** 2 / (dq ** 4).sum()))
        p = (w / tot).cpu().numpy()
        pos = batch["positions"].detach().cpu().numpy()
        c = (p[:, None] * pos).sum(0)
        r_eff.append(float(np.sqrt((p * ((pos - c) ** 2).sum(-1)).sum())))
    lam = float(model.lambda_dir()); u = [float(x) for x in model.u_eff().detach().cpu()]
    out.append({"run": name, "label": label, "lambda_dir": lam, "u_eff": u,
                "n_eff_p50": float(np.median(n_eff)) if n_eff else None,
                "r_eff_p50": float(np.median(r_eff)) if r_eff else None, "frames": len(n_eff)})
    print(f"{label:18s} {name:28s} lambda {lam:.4f}  U {[round(x,3) for x in u]}  "
          f"N_eff {out[-1]['n_eff_p50']:.2f}  R_eff {out[-1]['r_eff_p50']:.2f} A  [{time.time()-t0:.0f}s]", flush=True)
    del model; torch.cuda.empty_cache()
json.dump(out, open("/home/alex/runs/dscc/arm4_readout.json", "w"), indent=1)
print("wrote /home/alex/runs/dscc/arm4_readout.json", flush=True)
