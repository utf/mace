"""L2: build the q_LES / Jacobian cache of a MACELES base for every charged frame the head sees
(train + held of dataset_pbe), once per base. N first-order backward passes per frame through the
float32 base; keyed by the frame's geometry key, the format `dscc_train.py --les_cache` loads.
    python defect-perovskite/l2_les_cache.py --base ~/runs/base_v4_lesmh_prod/base_v4_lesmh_prod_base.pt \
        --out ~/runs/base_v4_lesmh_prod/les_cache.pt
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from mace import tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.legacy import frame_key
from mace.modules.dscc.model import MACEDSCC
from mace.modules.dscc.train import load_frames
from mace.tools import torch_geometric

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
ap.add_argument("--out", required=True)
ap.add_argument("--r_cut", type=float, default=10.0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)

base = torch.load(os.path.expanduser(args.base), map_location="cpu", weights_only=False).float()
model = MACEDSCC(base, r_cut=args.r_cut).to(args.device).eval()
frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
charged = [i for i, a in enumerate(frames) if int(a.info.get("cell_charge", 0)) != 0]
if args.limit:
    charged = charged[: args.limit]
z = tools.AtomicNumberTable(model.atomic_numbers)
out = Path(os.path.expanduser(args.out))
store = {}
if out.exists():
    store = torch.load(out, map_location="cpu", weights_only=False)
    print(f"{len(store)} frames already cached in {out}")
t0 = time.time(); n_new = 0
for n, i in enumerate(charged):
    a = frames[i]
    key = str(int(frame_key(a.get_atomic_numbers(), a.get_positions(), np.array(a.get_cell()))))
    if key in store:
        continue
    batch = next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([a], z, args.r_cut), batch_size=1))).to(args.device).to_dict()
    q, J = model.latent_charge_jacobian(batch)
    store[key] = (q.cpu(), J.cpu()); n_new += 1
    if n_new % 25 == 1:
        print(f"{n_new} new / {len(charged)} frames, {(time.time() - t0) / n_new:.2f} s per frame, "
              f"|q| mean {float(q.abs().mean()):.4f} e, |J| max {float(J.abs().max()):.4f} e/A", flush=True)
    if n_new % 100 == 0:
        tmp = out.with_suffix(".tmp"); torch.save(store, tmp); tmp.replace(out)
tmp = out.with_suffix(".tmp"); torch.save(store, tmp); tmp.replace(out)
print(f"done: {n_new} new frames in {time.time() - t0:.0f} s; {len(store)} in {out} "
      f"({out.stat().st_size / 1e6:.0f} MB)")
