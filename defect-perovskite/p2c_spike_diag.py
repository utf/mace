"""Plan P Phase 2c Step 1 investigation (registered 2026-10-11 16:20; no training): why the B_w runs read as they do.

  A  lambda-scaling of the trained on-site readout bound (sk.delta, uniform across species; the Pb row is scaled) on a B_w
     averaged checkpoint: held-out forces (total, 2-4, 4-12, flanking Pb) and the level instruments at lambda 1 / 0.75 / 0.5 /
     0.25 / 0, plus lambda = 0 with s_near restored to its init (W back, readout well gone). Diagnostic, not a model.
  B  per-frame two-site hole structure on the 79-atom +1 frames: dq on the two flanking Pb, the frontier separation, the next
     splitting, the signed on-site readout shift (s, p channels) at the flanking Pb and the bulk-Pb median, for B_w and B_s.
  C  pre-clip gradient norm of the force loss by parameter group on training +1 frames (batch 1), B_w and B_s.

  python defect-perovskite/p2c_spike_diag.py --bw ~/runs/dscc/dscc_p2cwbw_s2 --bs ~/runs/dscc/dscc_p2v8wbs_s2 --out ~/runs/dscc/p2c/spike_diag_s2.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mace import tools                                                     # noqa: E402
from mace.modules.dscc import data as dd                                   # noqa: E402
from mace.modules.dscc.precondition import hole_distribution              # noqa: E402
from mace.modules.dscc.train import load_frames, vacancy_centre_full       # noqa: E402
from mace.tools import torch_geometric as tg                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bw", required=True); ap.add_argument("--bs", required=True)
ap.add_argument("--dataset", default=str(HERE / "dataset_pbe")); ap.add_argument("--out", required=True)
ap.add_argument("--parts", nargs="+", default=["A", "B", "C"]); ap.add_argument("--n_grad", type=int, default=60)
ap.add_argument("--embed", default="/home/alex/runs/dscc/p2c/embed_b_label_w01.json")
ap.add_argument("--device", default="cuda"); ap.add_argument("--batch_size", type=int, default=8)
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
dev = args.device; t0 = time.time()
def status(msg): print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

frames = load_frames(f"{args.dataset}/train.xyz", f"{args.dataset}/valid.xyz")
rec_bw = json.load(open(Path(args.bw).expanduser() / "run_record.json")) if (Path(args.bw).expanduser() / "run_record.json").exists() else None
rec_bs = json.load(open(Path(args.bs).expanduser() / "run_record.json"))
fold = rec_bs["config"]["fold"]; fold_of = {int(k): v for k, v in rec_bs["fold_of"].items()}
charged79 = [i for i, a in enumerate(frames) if int(a.info.get("cell_charge", 0)) != 0 and len(a) == 79]
held = [i for i in charged79 if fold_of.get(i) == fold]
train = [i for i in charged79 if fold_of.get(i) != fold]
status(f"fold {fold}: {len(held)} held / {len(train)} train charged 79-atom frames")

def load(run):
    m = torch.load(Path(run).expanduser() / "model_avg.pt", weights_only=False, map_location=dev).eval()
    return m
models = {"bw": load(args.bw), "bs": load(args.bs)}
z3 = tools.AtomicNumberTable(models["bw"].atomic_numbers); r_cut = models["bw"].r_cut
def loader(idx, bs): return tg.dataloader.DataLoader(dd.atomic_data([frames[i] for i in idx], z3, r_cut), batch_size=bs)
PB = models["bw"].atomic_numbers.index(82)
out = {"fold": fold, "n_held": len(held), "bw": str(args.bw), "bs": str(args.bs), "s_near_init": getattr(models["bw"], "s_near_init", None)}
pc = lambda v: 1000.0 * v / 3 ** 0.5

def force_eval(m, idx):
    """Per-component force RMS (meV/A) total, 2-4 A, 4-12 A, and at the flanking Pb, over frames idx."""
    acc = {"total": [0.0, 0], "2-4": [0.0, 0], "4-12": [0.0, 0], "pb_flank": [0.0, 0]}
    k = 0
    for b in loader(idx, args.batch_size):
        batch = b.to(dev).to_dict()
        with torch.no_grad():
            o = m(dict(batch), compute_force=True)
        diff = ((o["forces"] - batch["forces"]) ** 2).sum(-1)
        ptr = batch["ptr"]
        for g in range(int(ptr.numel() - 1)):
            lo, hi = int(ptr[g]), int(ptr[g + 1]); a = frames[idx[k]]; k += 1
            d = diff[lo:hi]; acc["total"][0] += float(d.sum()); acc["total"][1] += int(d.numel())
            found = vacancy_centre_full(batch["positions"][lo:hi], batch["cell"].view(-1, 3, 3)[g], a.get_atomic_numbers())
            if found is None: continue
            rad, flank = found
            for key, msk in (("2-4", (rad >= 2) & (rad < 4)), ("4-12", (rad >= 4) & (rad < 12))):
                acc[key][0] += float(d[msk].sum()); acc[key][1] += int(msk.sum())
            fl = torch.as_tensor(flank, device=d.device); acc["pb_flank"][0] += float(d[fl].sum()); acc["pb_flank"][1] += int(fl.numel())
    return {key: (pc(math.sqrt(v[0] / v[1])) if v[1] else None) for key, v in acc.items()}

def level_eval(m, idx):
    seps, neffs = [], []
    for b in loader(idx, args.batch_size):
        batch = b.to(dev).to_dict()
        for H, numbers in m.filled_hamiltonians(batch):
            s, dq = hole_distribution(H, numbers, m.sigma_s); seps.append(s); neffs.append(float(1.0 / (dq ** 2).sum()))
    res = {"separation_p50": float(np.median(seps)), "n_eff_p50": float(np.median(neffs)), "pass": float(np.mean([(s >= 0.2 and n <= 4) for s, n in zip(seps, neffs)]))}
    if args.embed and Path(args.embed).exists():
        from ase import Atoms
        rec = json.load(open(args.embed)); a = Atoms(numbers=rec["numbers"], positions=rec["positions"], cell=rec["cell"], pbc=True)
        a.info.update({"carrier_counts": np.zeros(4), "cell_charge": 0, "config_type": "e", "source_dir": "e"})
        batch = next(iter(tg.dataloader.DataLoader(dd.atomic_data([a], z3, r_cut), batch_size=1))).to(dev).to_dict()
        H, numbers = m.filled_hamiltonians(batch)[0]; s, dq = hole_distribution(H, numbers, m.sigma_s)
        pos = batch["positions"]; cell = batch["cell"].view(3, 3); vac = torch.tensor(rec["vacancy"], dtype=torch.float64, device=dev)
        d = pos - vac; d = d - torch.round(d @ torch.linalg.inv(cell)) @ cell; r = d.norm(dim=-1)
        res["embed"] = {"separation": s, "n_eff": float(1.0 / (dq ** 2).sum()), "dq5": float(dq[r <= 5].sum())}
        del H
    return res

if "A" in args.parts:
    m = models["bw"]; delta0 = m.h0.sk.delta.detach().clone(); s_raw0 = m.s_near_raw.detach().clone()
    rows = []
    for lam, restore in ((1.0, False), (0.75, False), (0.5, False), (0.25, False), (0.0, False), (0.0, True)):
        with torch.no_grad():
            m.h0.sk.delta.copy_(delta0); m.h0.sk.delta[PB] = delta0[PB] * lam
            m.s_near_raw.copy_(s_raw0)
            if restore: m.s_near_raw.fill_(math.atanh(float(m.s_near_init) / m.s_max))
        f = force_eval(m, held); lv = level_eval(m, held)
        row = {"lambda_pb": lam, "s_near": float(m.near_scale()), "forces": f, "levels": lv}; rows.append(row)
        status(f"A lambda_Pb {lam:.2f} s_near {row['s_near']:.3f}: F total {f['total']:.2f} 2-4 {f['2-4']:.1f} 4-12 {f['4-12']:.2f} flank {f['pb_flank']:.1f} | +1 sep {lv['separation_p50']:.3f} neff {lv['n_eff_p50']:.1f} pass {lv['pass']:.2f}"
               + ("" if "embed" not in lv else f" | embed {lv['embed']['separation']:.3f}/{lv['embed']['n_eff']:.1f}/{lv['embed']['dq5']:.2f}"))
    with torch.no_grad(): m.h0.sk.delta.copy_(delta0); m.s_near_raw.copy_(s_raw0)
    # the same scaling on B_s (its own readout is small; the control for 'scaling delta also moves bulk Pb')
    mb = models["bs"]; db0 = mb.h0.sk.delta.detach().clone(); rows_bs = []
    for lam in (1.0, 0.5, 0.0):
        with torch.no_grad(): mb.h0.sk.delta.copy_(db0); mb.h0.sk.delta[PB] = db0[PB] * lam
        f = force_eval(mb, held); rows_bs.append({"lambda_pb": lam, "forces": f}); status(f"A (B_s control) lambda_Pb {lam:.2f}: F total {f['total']:.2f} 2-4 {f['2-4']:.1f} 4-12 {f['4-12']:.2f} flank {f['pb_flank']:.1f}")
    with torch.no_grad(): mb.h0.sk.delta.copy_(db0)
    out["A"] = {"bw": rows, "bs": rows_bs, "delta_eV": float(delta0[PB].max())}
    json.dump(out, open(Path(args.out).expanduser(), "w"), indent=1)

def site_shift(m, batch):
    """Signed on-site readout shift per atom and channel (eV): tanh(site(h, e)) * delta[Z]."""
    sk = m.h0.sk
    scalars = m.readout_scalars(batch).to(torch.float64); species = batch["node_attrs"].argmax(dim=-1)
    h = sk.standardise(scalars, species); e = sk.elem(species)
    t = torch.tanh(sk.site(torch.cat([h, e], dim=-1)))
    return t * sk.delta[species].to(t.dtype), t

if "B" in args.parts:
    res = {}
    for name, m in models.items():
        rows = []
        for b in loader(charged79, args.batch_size):
            batch = b.to(dev).to_dict(); ptr = batch["ptr"]
            with torch.no_grad():
                shift, t = site_shift(m, batch)
            Hs = m.filled_hamiltonians(batch)
            for g, (H, numbers) in enumerate(Hs):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                found = vacancy_centre_full(batch["positions"][lo:hi], batch["cell"].view(-1, 3, 3)[g], np.array(numbers))
                if found is None: continue
                rad, flank = found; fl = [int(x) for x in flank]
                sep, dq = hole_distribution(H, numbers, m.sigma_s)
                from mace.modules.dscc.species import S_REF, neutral_count
                n_up, _ = S_REF.counts(neutral_count(numbers)); eps = torch.linalg.eigvalsh(H)
                sh = shift[lo:hi]; tt = t[lo:hi]; z = torch.tensor(numbers, device=sh.device)
                pb_bulk = (z == 82).clone(); pb_bulk[fl] = False
                rows.append({"dq_flank": [float(dq[i]) for i in fl], "n_eff": float(1.0 / (dq ** 2).sum()), "separation": sep,
                             "next_split": float(eps[n_up + 1] - eps[n_up]), "shift_flank_s": [float(sh[i, 0]) for i in fl], "shift_flank_p": [float(sh[i, 1]) for i in fl],
                             "tanh_flank_p": [float(tt[i, 1]) for i in fl], "shift_pb_bulk_p_median": float(sh[pb_bulk, 1].median()), "shift_pb_bulk_s_median": float(sh[pb_bulk, 0].median()),
                             "dq_sum_flank": float(dq[fl].sum())})
        d = {k: np.array([r[k] for r in rows]) for k in ("n_eff", "separation", "next_split", "dq_sum_flank", "shift_pb_bulk_p_median", "shift_pb_bulk_s_median")}
        dqf = np.array([r["dq_flank"] for r in rows]); asym = np.abs(dqf[:, 0] - dqf[:, 1]); shp = np.array([r["shift_flank_p"] for r in rows]); shs = np.array([r["shift_flank_s"] for r in rows]); tp = np.array([r["tanh_flank_p"] for r in rows])
        summ = {"n": len(rows), "n_eff_p50": float(np.median(d["n_eff"])), "separation_p50": float(np.median(d["separation"])), "next_split_p50": float(np.median(d["next_split"])),
                "dq_sum_flank_p50": float(np.median(d["dq_sum_flank"])), "dq_asym_p50": float(np.median(asym)), "frac_one_sided(asym>0.3)": float(np.mean(asym > 0.3)),
                "frac_one_sided_and_split<2sigma": float(np.mean((asym > 0.3) & (d["separation"] < 0.1))),
                "shift_flank_p_mean": float(shp.mean()), "shift_flank_p_p10_p90": [float(np.percentile(shp, 10)), float(np.percentile(shp, 90))],
                "shift_flank_s_mean": float(shs.mean()), "tanh_flank_p_abs_mean": float(np.abs(tp).mean()), "frac_tanh_flank_p_abs>0.9": float(np.mean(np.abs(tp) > 0.9)),
                "shift_pb_bulk_p_median": float(np.median(d["shift_pb_bulk_p_median"])), "shift_pb_bulk_s_median": float(np.median(d["shift_pb_bulk_s_median"]))}
        res[name] = {"summary": summ, "rows": rows}
        status(f"B {name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summ.items()))
    out["B"] = res
    json.dump(out, open(Path(args.out).expanduser(), "w"), indent=1)

if "C" in args.parts:
    res = {}
    rng = np.random.default_rng(0); sel = sorted(rng.choice(train, size=min(args.n_grad, len(train)), replace=False).tolist())
    def group_of(n):
        if "s_near_raw" in n: return "s_near"
        if ".sk.site" in n: return "on_site_readout"
        if ".sk.hop" in n: return "hopping_readout"
        if n.startswith("h0."): return "h0_other"
        if "lambda" in n or "u_raw" in n or n in ("s_raw", "s_cross_raw", "s_well_raw"): return "scales"
        return "other"
    for name, m in models.items():
        params = [(n, p) for n, p in m.named_parameters() if p.requires_grad and not n.startswith("base.")]
        groups = sorted({group_of(n) for n, _ in params})
        rows = []
        for k, i in enumerate(sel):
            batch = next(iter(loader([i], 1))).to(dev).to_dict()
            m.train(); m.zero_grad(set_to_none=True)
            try:
                o = m(dict(batch), training=True, compute_force=True)
                l_force = ((o["forces"] - batch["forces"]) ** 2).sum(-1).mean()
                l_force.backward()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); status(f"C {name}: frame {i} out of memory, skipped"); continue
            norms = {g: 0.0 for g in groups}
            for n, p in params:
                if p.grad is not None: norms[group_of(n)] += float((p.grad.detach() ** 2).sum())
            total = math.sqrt(sum(norms.values())); norms = {g: math.sqrt(v) for g, v in norms.items()}
            with torch.no_grad():
                H, numbers = m.filled_hamiltonians(batch)[0]; sep, dq = hole_distribution(H, numbers, m.sigma_s)
                found = vacancy_centre_full(batch["positions"], batch["cell"].view(3, 3), np.array(numbers))
                sh, t = site_shift(m, batch); fl = [int(x) for x in found[1]] if found is not None else []
            rows.append({"index": i, "loss": float(l_force), "grad_norm": total, "groups": norms, "separation": sep, "n_eff": float(1.0 / (dq ** 2).sum()),
                         "tanh_flank_p_abs": (float(t[fl, 1].abs().mean()) if fl else None)})
            m.zero_grad(set_to_none=True); del o, l_force
            if (k + 1) % 10 == 0: status(f"C {name}: {k+1}/{len(sel)} frames")
        m.eval(); torch.cuda.empty_cache()
        gn = np.array([r["grad_norm"] for r in rows]); sep = np.array([r["separation"] for r in rows]); tf = np.array([r["tanh_flank_p_abs"] or 0 for r in rows])
        summ = {"n": len(rows), "grad_norm_p50": float(np.median(gn)), "grad_norm_p90": float(np.percentile(gn, 90)), "grad_norm_max": float(gn.max()), "frac_clipped_at_10": float(np.mean(gn > 10)),
                "loss_p50": float(np.median([r["loss"] for r in rows])), "loss_max": float(max(r["loss"] for r in rows)),
                "group_share_p50": {g: float(np.median([r["groups"][g] / max(r["grad_norm"], 1e-30) for r in rows])) for g in groups},
                "group_share_top10pct": {g: float(np.mean([r["groups"][g] / max(r["grad_norm"], 1e-30) for r in rows if r["grad_norm"] >= np.percentile(gn, 90)])) for g in groups},
                "corr_log_grad_vs_separation": float(np.corrcoef(np.log(gn), sep)[0, 1]), "corr_log_grad_vs_tanh_flank": float(np.corrcoef(np.log(gn), tf)[0, 1])}
        res[name] = {"summary": summ, "rows": rows}
        status(f"C {name}: grad norm p50 {summ['grad_norm_p50']:.3g} p90 {summ['grad_norm_p90']:.3g} max {summ['grad_norm_max']:.3g} clipped {summ['frac_clipped_at_10']:.2f}; shares p50 {summ['group_share_p50']}; top-10% shares {summ['group_share_top10pct']}; corr(log g, sep) {summ['corr_log_grad_vs_separation']:+.2f}, corr(log g, |tanh| flank) {summ['corr_log_grad_vs_tanh_flank']:+.2f}")
    out["C"] = res
    json.dump(out, open(Path(args.out).expanduser(), "w"), indent=1)
status(f"saved {args.out}")
