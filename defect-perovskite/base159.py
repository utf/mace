"""Is the old base's advantage on the seventeen neutral 159-atom frames systematic or a few
frames? Per-frame force error for both bases on every 159/160-atom frame."""
import sys
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace.modules.dscc.train import load_frames
from mace.modules.dscc import data as dd
from mace.tools import torch_geometric as tg, AtomicNumberTable
from mace.modules.models import ScaleShiftMACE
torch.set_default_dtype(torch.float64)
HERE = W + "/defect-perovskite"
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
big = [(i, a) for i, a in enumerate(frames) if len(a) >= 120]
print("large frames:", len(big), "of which neutral:", sum(1 for _, a in big if int(a.info.get("cell_charge", 0)) == 0))
out = {}
for tag, path in (("old", "/home/alex/runs/aprime_prod/aprime_prod_base.pt"),
                  ("v2", "/home/alex/runs/base_v2_prod/base_v2_prod_base.pt")):
    m = torch.load(path, weights_only=False, map_location="cpu").double().to("cuda").eval()
    z = AtomicNumberTable([int(x) for x in m.atomic_numbers])
    errs = []
    for i, a in big:
        b = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z, float(m.r_max)), batch_size=1))).to("cuda").to_dict()
        o = ScaleShiftMACE.forward(m, b, training=False, compute_force=True)
        f = o["forces"].detach().cpu().numpy()
        e = np.sqrt(((f - a.arrays["REF_forces"]) ** 2).mean())
        errs.append(1000 * e)
    out[tag] = np.array(errs); del m; torch.cuda.empty_cache()
q = np.array([int(a.info.get("cell_charge", 0)) for _, a in big])
for label, mask in (("neutral 159", q == 0), ("charged 159", q != 0)):
    o, v = out["old"][mask], out["v2"][mask]
    print(f"\n{label}: n={mask.sum()}")
    print("  old  ", " ".join(f"{x:6.1f}" for x in np.sort(o)))
    print("  v2   ", " ".join(f"{x:6.1f}" for x in np.sort(v)))
    print(f"  median old {np.median(o):.1f}  v2 {np.median(v):.1f}   v2 worse on {int((v>o).sum())}/{mask.sum()} frames")
