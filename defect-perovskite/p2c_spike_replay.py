"""Spike investigation II (registered 2026-10-11 22:30): read the per-step record and the epoch checkpoints of the
deterministic replay of rerun seed 6 (`dscc_p2cx_bw2_s6_replay`). No training.

  S  per-step profile of the spike epoch: loss by term, gap and its edge levels, pre-clip gradient norm by group, the
     frames in the batch and their force errors, s_near; onset detection and the shape of the excursion.
  C  checkpoint pairs (ep28 -> 29 -> 30 -> 31): parameter change by group (absolute and relative to the group's norm),
     eps0 by species, and on the static cell and the +1 frames the SIGNED species-mean on-site shift (s, p), the gap,
     its edges, N_eff.

  python defect-perovskite/p2c_spike_replay.py --run ~/runs/dscc/p2c/replay_s6 --out ~/runs/dscc/p2c/spike_replay_s6.json
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="directory with steps.jsonl and model_ep*.pt")
ap.add_argument("--out", required=True); ap.add_argument("--parts", nargs="+", default=["S", "C"])
ap.add_argument("--spike_epoch", type=int, default=None); ap.add_argument("--dataset", default=str(HERE / "dataset_pbe"))
ap.add_argument("--device", default="cuda"); ap.add_argument("--n_frames", type=int, default=64)
args = ap.parse_args()
run = Path(args.run).expanduser(); t0 = time.time(); out = {"run": str(run)}
def status(msg): print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

rows = [json.loads(l) for l in open(run / "steps.jsonl")]
by_epoch = defaultdict(list)
for r in rows: by_epoch[r["epoch"]].append(r)
epochs = sorted(by_epoch)
status(f"{len(rows)} steps, epochs {epochs[0]}..{epochs[-1]}")

if "S" in args.parts:
    # per-epoch summary
    summ = []
    for e in epochs:
        rs = by_epoch[e]; lg = np.array([r["l_gap"] for r in rs]); lf = np.array([r["force"] for r in rs]); gn = np.array([r["grad_norm"] for r in rs])
        cb = np.array([r["gap_edges"][1] for r in rs]); vb = np.array([r["gap_edges"][0] for r in rs])
        summ.append({"epoch": e, "n": len(rs), "loss_mean": float(np.mean([r["loss"] for r in rs])), "l_gap_mean": float(lg.mean()), "l_gap_max": float(lg.max()),
                     "n_steps_l_gap>1e-3": int((lg > 1e-3).sum()), "force_mean": float(lf.mean()), "grad_norm_p50": float(np.median(gn)), "grad_norm_max": float(gn.max()),
                     "frac_clipped": float((gn > 10).mean()), "gap_min": float(min(r["gap"] for r in rs)), "gap_max": float(max(r["gap"] for r in rs)),
                     "cbm_split01_min": float((cb[:, 1] - cb[:, 0]).min()), "cbm_split12_min": float((cb[:, 2] - cb[:, 1]).min()), "vbm_split_min": float((vb[:, 2] - vb[:, 1]).min()),
                     "s_near_end": rs[-1]["s_near"]})
    out["epoch_summary"] = summ
    for s in summ:
        status(f"S ep {s['epoch']:2d}: loss {s['loss_mean']:.3g} l_gap mean {s['l_gap_mean']:.3g} max {s['l_gap_max']:.3g} (>1e-3: {s['n_steps_l_gap>1e-3']}) force {s['force_mean']:.3g} "
               f"gn p50 {s['grad_norm_p50']:.3g} max {s['grad_norm_max']:.3g} clipped {s['frac_clipped']:.2f} gap {s['gap_min']:.3f}..{s['gap_max']:.3f} "
               f"CBM splits min {s['cbm_split01_min']:.4f}/{s['cbm_split12_min']:.4f} VBM {s['vbm_split_min']:.4f} s_near {s['s_near_end']:.3f}")
    se = args.spike_epoch if args.spike_epoch is not None else max(summ[1:] or summ, key=lambda s: s["loss_mean"])["epoch"]
    rs = by_epoch[se]; prev = by_epoch.get(se - 1, rs)
    ref_gap = float(np.median([r["l_gap"] for r in prev])); ref_force = float(np.median([r["force"] for r in prev]))
    onset = next((i for i, r in enumerate(rs) if r["l_gap"] > max(10 * ref_gap, 1e-4)), None)
    status(f"S spike epoch {se}: previous-epoch median l_gap {ref_gap:.3g}, force {ref_force:.3g}; onset step {onset}")
    prof = []
    lo = max(0, (onset or 0) - 6); hi = min(len(rs), (onset or 0) + 60)
    for i in range(lo, hi):
        r = rs[i]; gg = r["grad_groups"]; tot = max(sum(v * v for v in gg.values()) ** 0.5, 1e-30)
        prof.append({"step": r["step"], "loss": r["loss"], "l_gap": r["l_gap"], "force": r["force"], "gap": r["gap"], "vbm": r["gap_edges"][0], "cbm": r["gap_edges"][1],
                     "grad_norm": r["grad_norm"], "shares": {k: round(v / tot, 3) for k, v in gg.items()}, "frames": r["frames"], "ferr": [round(x, 1) for x in r["force_err_per_graph"]], "s_near": r["s_near"]})
        cb = r["gap_edges"][1]; vb = r["gap_edges"][0]
        status(f"  step {r['step']:3d} loss {r['loss']:.3g} l_gap {r['l_gap']:.3g} F {r['force']:.3g} gap {r['gap']:.4f} VBM {vb[1]-vb[0]:+.4f}/{vb[2]-vb[1]:+.4f} "
               f"CBM {cb[1]-cb[0]:.4f}/{cb[2]-cb[1]:.4f} gn {r['grad_norm']:.3g} top {max(gg, key=gg.get)} {max(gg.values())/tot:.2f} ferr {[round(x) for x in r['force_err_per_graph']]} s {r['s_near']:.3f}")
    out["spike_epoch"] = se; out["onset_step"] = onset; out["profile"] = prof
    # force error during the excursion vs the previous epoch, excluding the batch at onset
    fe_prev = np.array([x for r in prev for x in r["force_err_per_graph"]]); exc = [r for r in rs if r["l_gap"] > max(10 * ref_gap, 1e-4)]
    fe_exc = np.array([x for r in exc for x in r["force_err_per_graph"]]); fe_rest = np.array([x for r in rs if r["l_gap"] <= max(10 * ref_gap, 1e-4) for x in r["force_err_per_graph"]])
    out["force_err"] = {"prev_epoch_p50": float(np.median(fe_prev)), "excursion_p50": float(np.median(fe_exc)) if fe_exc.size else None, "n_excursion_steps": len(exc),
                        "rest_of_epoch_p50": float(np.median(fe_rest)) if fe_rest.size else None, "next_epoch_p50": float(np.median([x for r in by_epoch.get(se + 1, []) for x in r["force_err_per_graph"]])) if se + 1 in by_epoch else None}
    status(f"S per-graph force error p50 (meV/A, vector RMS): previous epoch {out['force_err']['prev_epoch_p50']:.1f}, during the {len(exc)} excursion steps {out['force_err']['excursion_p50']}, "
           f"rest of the epoch {out['force_err']['rest_of_epoch_p50']}, next epoch {out['force_err']['next_epoch_p50']}")
    # frames at the onset: how often do they occur elsewhere, and what are their errors there
    if onset is not None:
        fr = rs[onset]["frames"]; seen = defaultdict(list)
        for r in rows:
            for f, e_ in zip(r["frames"], r["force_err_per_graph"]):
                if f in fr: seen[f].append((r["epoch"], round(e_, 1)))
        out["onset_frames_elsewhere"] = {str(f): v for f, v in seen.items()}
        status(f"S onset batch frames {fr}: errors elsewhere {dict(seen)}")

if "C" in args.parts:
    from mace import tools
    from mace.modules.dscc import data as dd
    from mace.modules.dscc.precondition import hole_distribution
    from mace.modules.dscc.train import load_frames, static_cell_atoms
    from mace.modules.dscc.species import S_REF, neutral_count
    from mace.tools import torch_geometric as tg
    torch.set_default_dtype(torch.float64); dev = args.device
    cks = sorted(run.glob("model_ep*.pt"), key=lambda p: int(p.stem[8:]))
    status(f"C checkpoints: {[p.name for p in cks]}")
    models = {int(p.stem[8:]): torch.load(p, weights_only=False, map_location=dev).eval() for p in cks}
    def group_of(n):
        if "s_near_raw" in n: return "s_near"
        if ".sk.site" in n: return "on_site_readout"
        if ".sk.hop" in n: return "hopping_readout"
        if ".sk.eps0" in n: return "eps0"
        if ".sk.v0_raw" in n or ".sk.decay_u" in n: return "sk_scalars"
        if n.startswith("h0."): return "h0_other"
        return "scales_other"
    def params(m): return {n: p.detach().double() for n, p in m.named_parameters() if p.requires_grad and not n.startswith("base.")}
    eps = sorted(models); diffs = []
    for a, b in zip(eps, eps[1:]):
        pa, pb = params(models[a]), params(models[b]); d = defaultdict(lambda: [0.0, 0.0, 0.0])
        for n in pa:
            g = group_of(n); dd_ = (pb[n] - pa[n]); d[g][0] += float((dd_ ** 2).sum()); d[g][1] += float((pa[n] ** 2).sum()); d[g][2] = max(d[g][2], float(dd_.abs().max()))
        rec = {"from": a, "to": b, "groups": {g: {"abs": v[0] ** 0.5, "rel": (v[0] / max(v[1], 1e-30)) ** 0.5, "max_elem": v[2]} for g, v in d.items()}}
        rec["eps0_from"] = models[a].h0.sk.eps0.detach().cpu().tolist(); rec["eps0_to"] = models[b].h0.sk.eps0.detach().cpu().tolist()
        diffs.append(rec)
        status(f"C ep{a}->ep{b}: " + ", ".join(f"{g} |d| {v['abs']:.3g} rel {v['rel']:.3g} max {v['max_elem']:.3g}" for g, v in sorted(rec["groups"].items())))
        status(f"C   eps0 ep{a} {np.round(rec['eps0_from'], 4).tolist()} -> ep{b} {np.round(rec['eps0_to'], 4).tolist()}")
    out["param_diffs"] = diffs
    m0 = models[eps[0]]; z3 = tools.AtomicNumberTable(m0.atomic_numbers); r_cut = m0.r_cut; zs = list(m0.atomic_numbers)
    static = dd.atomic_data([static_cell_atoms(str(HERE / "static_pristine_cell.json"))], z3, r_cut)
    sb = next(iter(tg.dataloader.DataLoader(static, batch_size=1))).to(dev).to_dict()
    frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
    charged = [i for i, a in enumerate(frames) if int(a.info.get("cell_charge", 0)) != 0 and len(a) == 79][: args.n_frames]
    cl = tg.dataloader.DataLoader(dd.atomic_data([frames[i] for i in charged], z3, r_cut), batch_size=8)
    def site_shift(m, batch):
        sk = m.h0.sk; scalars = m.readout_scalars(batch).to(torch.float64); species = batch["node_attrs"].argmax(dim=-1)
        h = sk.standardise(scalars, species); e = sk.elem(species); t = torch.tanh(sk.site(torch.cat([h, e], dim=-1)))
        return t * sk.delta[species].to(t.dtype), t, species
    def species_mean(sh, sp):
        return {str(zs[z]): [float(sh[sp == z, c].mean()) for c in (0, 1)] for z in range(len(zs)) if (sp == z).any()}
    reads = {}
    for e, m in models.items():
        with torch.no_grad():
            feats = m.first_block(m._trunk_data(dict(sb))); sf = tuple(t.detach() for t in m.features(feats))
            gap = float(m.pristine_gap(sb, sf)[0]); edges = m._gap_edges
            sh, t, sp = site_shift(m, sb); st = {"gap": gap, "vbm": edges[0], "cbm": edges[1], "shift_species_mean_sp": species_mean(sh, sp), "tanh_abs_mean": float(t.abs().mean())}
            acc = defaultdict(list); neffs = []; seps = []
            for b in cl:
                batch = b.to(dev).to_dict(); sh, t, sp = site_shift(m, batch)
                for z, v in species_mean(sh, sp).items(): acc[z].append(v)
                for H, numbers in m.filled_hamiltonians(batch):
                    s_, dq = hole_distribution(H, numbers, m.sigma_s); neffs.append(float(1.0 / (dq ** 2).sum())); seps.append(s_)
            pl = {"shift_species_mean_sp": {z: np.mean(v, axis=0).tolist() for z, v in acc.items()}, "n_eff_p50": float(np.median(neffs)), "separation_p50": float(np.median(seps))}
        reads[e] = {"static": st, "plus1": pl}
        status(f"C ep{e}: static gap {gap:.4f} VBM {np.round(edges[0], 4).tolist()} CBM {np.round(edges[1], 4).tolist()} shift(s,p) by species {{{', '.join(f'{z}: {v[0]:+.3f}/{v[1]:+.3f}' for z, v in st['shift_species_mean_sp'].items())}}}")
        status(f"C ep{e}: +1 frames shift(s,p) by species {{{', '.join(f'{z}: {v[0]:+.3f}/{v[1]:+.3f}' for z, v in pl['shift_species_mean_sp'].items())}}} N_eff p50 {pl['n_eff_p50']:.1f} sep p50 {pl['separation_p50']:.3f}")
    out["checkpoint_reads"] = {str(k): v for k, v in reads.items()}
json.dump(out, open(Path(args.out).expanduser(), "w"), indent=1); status(f"wrote {args.out}")
