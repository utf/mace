"""L4: absolute inference ms per frame, base alone and base + head, eager (no capture), on the
held charged 79-atom frames: median over 40 frames after 5 warm-ups, batch 1."""
import argparse, json, os, statistics, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from mace import tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.train import load_frames
from mace.tools import torch_geometric
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", required=True)
a = ap.parse_args(); torch.set_default_dtype(torch.float64)
m = torch.load(a.model, map_location="cuda", weights_only=False).cuda().eval()
frames = [f for f in load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz") if int(f.info.get("cell_charge", 0)) != 0 and len(f) == 79][: a.n + 5]
zt = tools.AtomicNumberTable(m.atomic_numbers)
def batch(f): return next(iter(torch_geometric.dataloader.DataLoader(dd.atomic_data([f], zt, m.r_cut), batch_size=1))).cuda().to_dict()
res = {}
for label, fn in (("base", lambda b: m.base_forward(m._trunk_data(dict(b)), training=False, compute_force=True)), ("base+head", lambda b: m(dict(b), compute_force=True))):
    ts = []
    for i, f in enumerate(frames):
        b = batch(f); torch.cuda.synchronize(); t0 = time.perf_counter(); fn(b); torch.cuda.synchronize(); dt = 1e3 * (time.perf_counter() - t0)
        if i >= 5: ts.append(dt)
    res[label] = {"median_ms": statistics.median(ts), "p90_ms": sorted(ts)[int(0.9 * len(ts))], "n": len(ts)}
res["frames"] = "held charged 79-atom, batch 1, eager float64 head / base in its own precision"
json.dump(res, open(a.out, "w"), indent=1); print(res)
