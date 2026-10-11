"""Plan P Phase 2c Step 1 (vii): the pre-training readout. A trained B_s checkpoint's H0 with the near field
moved into H0 (arm B_w's construction) at several s_near, read with the instrument (`filled_hamiltonians`):
separation / N_eff / C5 pass on the 79-atom +1 training frames and separation / N_eff / sum dq(< 5 A) on the
embedded structure. Readout only: nothing is trained, no gate is read.

  python defect-perovskite/p2c_bw_readout.py --model ~/runs/dscc/dscc_p2v8wbs_s2/model_avg.pt --s 0 0.411 -0.411 \
      --embed ~/runs/dscc/p2c/embed_b_label_w01.json --out ~/runs/dscc/p2c/bw_readout_w01_s2.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from ase import Atoms
from ase.io import read
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.precondition import PreconditionConfig, frame_record, hole_distribution   # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--state", default=None)
ap.add_argument("--s", type=float, nargs="+", default=[0.0])
ap.add_argument("--frames", default=str(HERE / "dataset_pbe" / "train.xyz"))
ap.add_argument("--charge", type=int, default=1); ap.add_argument("--n_atoms", type=int, default=79)
ap.add_argument("--embed", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda"); ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--near_field", default="potential", help="potential (B_w's construction) | keep (the checkpoint's own arm)")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device
t0 = time.time()
model = torch.load(Path(args.model).expanduser(), weights_only=False, map_location=dev)
if args.state:
    sd = torch.load(Path(args.state).expanduser(), weights_only=False, map_location=dev)
    model.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False)
model = model.to(dev).eval()
if not hasattr(model, "s_near_raw"):           # a checkpoint from before Phase 2c
    model.s_near_raw = nn.Parameter(torch.zeros((), dtype=torch.float64, device=dev)); model.s_near_init = 0.0
arm0 = model.near_field
print(f"model {args.model}: arm near_field={arm0}, s (B_s) {float(model.pattern_scale()):.4f}, r_c {model.r_c}, delta {model.delta}, near_r_split {model.near_r_split}", flush=True)
z3 = tools.AtomicNumberTable(model.atomic_numbers); cfg = PreconditionConfig()
frames = [a for a in read(args.frames, index=":") if int(a.info.get("cell_charge", 0)) == args.charge and len(a) == args.n_atoms]
print(f"{len(frames)} frames with cell_charge {args.charge} and {args.n_atoms} atoms", flush=True)
ds = dd.atomic_data(frames, z3, model.r_cut)
embed = None
if args.embed:
    rec = json.load(open(Path(args.embed).expanduser()))
    a = Atoms(numbers=rec["numbers"], positions=rec["positions"], cell=rec["cell"], pbc=True)
    a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "embed", "source_dir": "embed"})
    embed = (next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, model.r_cut), batch_size=1))).to(dev).to_dict(),
             torch.tensor(rec["vacancy"], dtype=torch.float64, device=dev), rec.get("label", "embed"))
rows = []
for s in args.s:
    if args.near_field == "potential":
        model.near_field = "potential"
        with torch.no_grad():
            model.s_near_raw.fill_(math.atanh(s / model.s_max))
    else:
        model.near_field = arm0
    t = time.time(); recs = []
    for b in tg.dataloader.DataLoader(ds, batch_size=args.batch_size):
        batch = b.to(dev).to_dict()
        for H, numbers in model.filled_hamiltonians(batch):
            recs.append(frame_record(H, numbers, cfg, model.sigma_s))
    sep = np.array([r.separation for r in recs]); neff = np.array([r.n_eff for r in recs])
    row = {"near_field": model.near_field, "s_near": s, "n_frames": len(recs),
           "separation_p10": float(np.percentile(sep, 10)), "separation_p50": float(np.median(sep)), "separation_p90": float(np.percentile(sep, 90)),
           "separation_max": float(sep.max()), "n_eff_p10": float(np.percentile(neff, 10)), "n_eff_p50": float(np.median(neff)), "n_eff_p90": float(np.percentile(neff, 90)),
           "pass_fraction": float(np.mean([r.passed for r in recs]))}
    if embed is not None:
        batch, vac, label = embed
        H, numbers = model.filled_hamiltonians(batch)[0]
        separation, dq = hole_distribution(H, numbers, model.sigma_s)
        pos = batch["positions"]; cell = batch["cell"].view(3, 3)
        d = pos - vac; d = d - torch.round(d @ torch.linalg.inv(cell)) @ cell; r = d.norm(dim=-1)
        W = None
        if model.near_field == "potential":
            W = (torch.diag(H).reshape(-1, 4)[:, 0])    # not the potential alone (H0 diagonal included): report via the difference below
        row["embed"] = {"label": label, "n_atoms": len(numbers), "separation": separation, "n_eff": float(1.0 / (dq ** 2).sum()),
                        "dq_within": {str(R): float(dq[r <= R].sum()) for R in (3, 5, 8, 12)}}
        del H
    row["seconds"] = time.time() - t
    rows.append(row)
    e = row.get("embed")
    print(f"s_near {s:+.3f}: +1 frames separation p50 {row['separation_p50']:.4f} eV (p90 {row['separation_p90']:.4f}, max {row['separation_max']:.4f}), "
          f"N_eff p50 {row['n_eff_p50']:.2f}, pass {row['pass_fraction']:.3f}" + ("" if not e else
          f"; embed {e['label']} separation {e['separation']:.4f} eV, N_eff {e['n_eff']:.1f}, sum dq(<5 A) {e['dq_within']['5']:.3f}") + f"  [{row['seconds']:.0f}s]", flush=True)
# the potential itself on the embedded structure at s = 1 (W is linear in s): its value on the flanking Pb and its range
if embed is not None and args.near_field == "potential":
    batch, vac, label = embed
    model.near_field = "potential"
    with torch.no_grad():
        model.s_near_raw.fill_(math.atanh(min(0.999 * model.s_max, 1.0) / model.s_max))
        H1, _ = model.filled_hamiltonians(batch)[0]
        model.s_near_raw.fill_(0.0)
        H0, numbers = model.filled_hamiltonians(batch)[0]
        W1 = -(torch.diag(H1) - torch.diag(H0)).reshape(-1, 4)[:, 0] / float(model.near_scale() if False else 1.0)
    pos = batch["positions"]; cell = batch["cell"].view(3, 3)
    d = pos - vac; d = d - torch.round(d @ torch.linalg.inv(cell)) @ cell; r = d.norm(dim=-1)
    order = torch.argsort(r)[:6]
    syms = {17: "Cl", 55: "Cs", 82: "Pb"}
    w_rows = [(syms[int(numbers[int(i)])], float(r[i]), float(W1[i])) for i in order]
    print(f"W at s_near = 1 on {label}: nearest sites (species, r / A, W / eV) {[(a, round(b, 2), round(c, 3)) for a, b, c in w_rows]}; "
          f"W range [{float(W1.min()):.3f}, {float(W1.max()):.3f}] eV, p95 |W| {float(W1.abs().quantile(0.95)):.3f}", flush=True)
    rows.append({"W_at_s1": {"nearest": w_rows, "min": float(W1.min()), "max": float(W1.max()), "p95_abs": float(W1.abs().quantile(0.95))}})
json.dump({"model": args.model, "state": args.state, "arm": arm0, "s_pattern": float(model.pattern_scale()), "frames": args.frames, "charge": args.charge, "rows": rows},
          open(Path(args.out).expanduser(), "w"), indent=1)
print(f"saved {args.out} [{time.time()-t0:.0f}s]")
