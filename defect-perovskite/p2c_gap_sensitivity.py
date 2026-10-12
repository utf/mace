"""Where the gap's sensitivity lives (spike investigation II, 2026-10-12 05:30; no training).
S_i = dgap/dtheta_i on the static cell, against the force-loss gradient per parameter accumulated (squared, as Adam's
second moment is) over n training frames: concentration of |S|^2 over hopping-readout parameters (top 1 / 10 / 50 %),
and the force-gradient magnitude on those parameters against the group median.
  python defect-perovskite/p2c_gap_sensitivity.py --model ~/runs/dscc/dscc_p2cwbw4_s0/model_avg.pt --n 24 --out ~/runs/dscc/p2c/gap_sensitivity_bw4_s0.json"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from mace import tools
from mace.modules.dscc import data as dd
from mace.modules.dscc.train import load_frames, static_cell_atoms
from mace.tools import torch_geometric as tg
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe")); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
a = ap.parse_args(); torch.set_default_dtype(torch.float64); dev = a.device; t0 = time.time()
def status(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
m = torch.load(Path(a.model).expanduser(), weights_only=False, map_location=dev); m.eval()
z3 = tools.AtomicNumberTable(m.atomic_numbers); r_cut = m.r_cut
named = [(n, p) for n, p in m.named_parameters() if p.requires_grad and not n.startswith("base.")]
params = [p for _, p in named]
def group_of(n):
    if "s_near_raw" in n: return "s_near"
    if ".sk.site" in n: return "on_site_readout"
    if ".sk.hop" in n: return "hopping_readout"
    if ".sk.eps0" in n: return "eps0"
    if ".sk.v0_raw" in n or ".sk.decay_u" in n: return "sk_scalars"
    if n.startswith("h0."): return "h0_other"
    return "scales_other"
# S on the static cell
static = dd.atomic_data([static_cell_atoms(str(HERE / "static_pristine_cell.json"))], z3, r_cut)
sb = next(iter(tg.dataloader.DataLoader(static, batch_size=1))).to(dev).to_dict()
feats = m.first_block(m._trunk_data(dict(sb))); sf = tuple(t.detach() for t in m.features(feats))
gap = m.pristine_gap(sb, sf); status(f"static gap {float(gap[0]):.4f} (gap_sigma {getattr(m, 'gap_sigma', 0.0)})")
S = torch.autograd.grad(gap.sum(), params, allow_unused=True)
S = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, S)]
# force gradient second moment over n frames (batch 1, training frames = charged + neutral 79-atom)
frames = load_frames(f"{a.dataset}/train.xyz", f"{a.dataset}/valid.xyz")
idx = [i for i, at in enumerate(frames) if len(at) == 79]; rng = np.random.default_rng(0); sel = sorted(rng.choice(idx, size=min(a.n, len(idx)), replace=False).tolist())
V = [torch.zeros_like(p) for p in params]; n_ok = 0
for k, i in enumerate(sel):
    batch = next(iter(tg.dataloader.DataLoader(dd.atomic_data([frames[i]], z3, r_cut), batch_size=1))).to(dev).to_dict()
    m.train(); m.zero_grad(set_to_none=True)
    try:
        o = m(dict(batch), training=True, compute_force=True); l = ((o["forces"] - batch["forces"]) ** 2).sum(-1).mean(); l.backward()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); continue
    for v, p in zip(V, params):
        if p.grad is not None: v += p.grad.detach() ** 2
    n_ok += 1; m.zero_grad(set_to_none=True); del o, l
    if (k + 1) % 8 == 0: status(f"{k+1}/{len(sel)} frames")
m.eval(); V = [v / max(n_ok, 1) for v in V]
out = {"model": a.model, "static_gap": float(gap[0]), "n_frames": n_ok, "groups": {}}
S2_tot = float(sum(float((s ** 2).sum()) for s in S))
for g in sorted({group_of(n) for n, _ in named}):
    s = torch.cat([S[k].reshape(-1) for k, (n, _) in enumerate(named) if group_of(n) == g]); v = torch.cat([V[k].reshape(-1) for k, (n, _) in enumerate(named) if group_of(n) == g])
    s2 = s ** 2; order = torch.argsort(s2, descending=True); cs = torch.cumsum(s2[order], 0) / max(float(s2.sum()), 1e-300)
    N = s.numel(); rec = {"n_params": N, "share_of_total_S2": float(s2.sum()) / max(S2_tot, 1e-300), "S_norm": float(s2.sum()) ** 0.5}
    for frac in (0.01, 0.1, 0.5):
        kk = max(1, int(frac * N)); rec[f"S2_share_top_{int(frac*100)}pct"] = float(cs[kk - 1])
    rv = torch.sqrt(v); med_all = float(rv.median()) if N > 1 else float(rv)
    for frac in (0.01, 0.1):
        kk = max(1, int(frac * N)); top = order[:kk]; rec[f"force_rms_grad_on_top_{int(frac*100)}pct_S_over_group_median"] = float(rv[top].median()) / max(med_all, 1e-300)
    # Adam-style quiet gain proxy: sum_i S_i^2 / sqrt(v_i) (per-parameter denominator), relative to the group-norm version |S|^2 / median sqrt(v)
    eps = 1e-12
    rec["per_param_gain_proxy_over_group_proxy"] = float((s2 / (rv + eps)).sum()) / max(float(s2.sum()) / max(med_all, eps), 1e-300)
    rec["force_rms_grad_group_median"] = med_all
    out["groups"][g] = rec
    status(f"{g:16s} n {N:6d} |S| {rec['S_norm']:.3g} share {rec['share_of_total_S2']:.3f} | S2 in top 1/10/50 %: {rec['S2_share_top_1pct']:.2f}/{rec['S2_share_top_10pct']:.2f}/{rec['S2_share_top_50pct']:.2f} | force grad on top-1 %/10 % of S vs group median: {rec['force_rms_grad_on_top_1pct_S_over_group_median']:.2f}/{rec['force_rms_grad_on_top_10pct_S_over_group_median']:.2f} | per-param/group gain ratio {rec['per_param_gain_proxy_over_group_proxy']:.1f}")
json.dump(out, open(Path(a.out).expanduser(), "w"), indent=1); status("done")
