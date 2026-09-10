"""Old base against the foundation fine-tune (base v2), on identical frames and identical code.

Energies are split by data set (pristine / neutral vacancy / charged vacancy, at 79-80 and
159-160 atoms) and forces additionally by distance from the vacancy (vacancy-side centre, W0.2
shells). Both bases are PRODUCTION bases, so every neutral frame is in-sample for both: those
rows measure fit, not generalisation. Charged frames were never seen by either base.

Energy alignment: the two bases carry different isolated-atom references, so a raw energy
difference is dominated by a per-composition constant. A per-species offset is fitted by OLS on
the NEUTRAL frames of each base separately and applied to all of that base's predictions; both
the raw and the aligned residual are reported.
"""
import argparse, json, sys, time
import numpy as np, torch
W = "/home/alex/src/mace/.claude/worktrees/size-extensivity"; sys.path.insert(0, W)
from mace.modules.dscc.train import load_frames, vacancy_centre_full, near_field_categories
from mace.modules.dscc import data as dd
from mace.tools import torch_geometric as tg, AtomicNumberTable
from mace.modules.models import ScaleShiftMACE

SPECIES = [17, 55, 82]
SHELLS = [("2-4", 2.0, 4.0), ("4-8", 4.0, 8.0), ("8-10", 8.0, 10.0), ("10-12", 10.0, 12.0), ("12+", 12.0, 1e9)]

ap = argparse.ArgumentParser()
ap.add_argument("--old", default="/home/alex/runs/aprime_prod/aprime_prod_base.pt")
ap.add_argument("--new", default="/home/alex/runs/base_v2_prod/base_v2_prod_base.pt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--out", default="/home/alex/runs/base_compare.json")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); t0 = time.time()

HERE = W + "/defect-perovskite"
frames = load_frames(f"{HERE}/dataset_pbe/train.xyz", f"{HERE}/dataset_pbe/valid.xyz")
meta = []
for a in frames:
    q = int(a.info.get("cell_charge", 0)); n = len(a)
    kind = "pristine" if a.info.get("config_type") == "ideal" else ("charged vacancy" if q else "neutral vacancy")
    meta.append({"kind": kind, "n": n, "q": q, "key": f"{kind} {n}"})

# geometry: radii and near-field categories once, shared by both bases
geom = []
for a in frames:
    pos = torch.tensor(a.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.array(a.get_cell()), dtype=torch.float64)
    found = vacancy_centre_full(pos, cell, a.get_atomic_numbers())
    if found is None:
        geom.append((None, None))
    else:
        rad, flank = found
        cats = near_field_categories(pos, cell, a.get_atomic_numbers(), flank)
        geom.append((rad.numpy(), {k: v.numpy() for k, v in cats.items()}))
print(f"geometry done, {sum(1 for g in geom if g[0] is not None)} frames with a flanking pair [{time.time()-t0:.0f}s]", flush=True)

def run(path):
    model = torch.load(path, weights_only=False, map_location="cpu").double().to(args.device).eval()
    r_max = float(model.r_max)
    z = AtomicNumberTable([int(x) for x in model.atomic_numbers])
    loader = tg.dataloader.DataLoader(dd.atomic_data(frames, z, r_max), batch_size=args.batch_size)
    E, F = [], []
    k = 0
    for b in loader:
        batch = b.to(args.device).to_dict()
        out = ScaleShiftMACE.forward(model, batch, training=False, compute_force=True)
        ptr = batch["ptr"].detach().cpu().numpy()
        e = out["energy"].detach().cpu().numpy(); f = out["forces"].detach().cpu().numpy()
        for g in range(len(ptr) - 1):
            E.append(float(e[g])); F.append(f[ptr[g]:ptr[g + 1]]); k += 1
        if k % 400 < args.batch_size:
            print(f"  {path.split('/')[-1]}: {k}/{len(frames)} [{time.time()-t0:.0f}s]", flush=True)
    del model; torch.cuda.empty_cache()
    return np.array(E), F

res = {}
for tag, path in (("old base", args.old), ("base v2", args.new)):
    E, F = run(path)
    res[tag] = {"E": E, "F": F}
    print(f"{tag} done [{time.time()-t0:.0f}s]", flush=True)

ref_E = np.array([float(a.info["REF_energy"]) for a in frames])
ref_F = [a.arrays["REF_forces"] for a in frames]
counts = np.array([[int((a.get_atomic_numbers() == z_).sum()) for z_ in SPECIES] for a in frames], dtype=np.float64)
neutral = np.array([m["q"] == 0 for m in meta])

report = {"n_frames": len(frames), "categories": {}, "alignment": {}}
for tag in res:
    E = res[tag]["E"]
    d = ref_E - E
    a_s, *_ = np.linalg.lstsq(counts[neutral], d[neutral], rcond=None)   # offsets fitted on neutral only
    res[tag]["aligned"] = d - counts @ a_s
    res[tag]["raw"] = d
    report["alignment"][tag] = {"a_Cl": float(a_s[0]), "a_Cs": float(a_s[1]), "a_Pb": float(a_s[2])}

order = ["pristine 80", "neutral vacancy 79", "neutral vacancy 159", "charged vacancy 79", "charged vacancy 159"]
for key in order:
    idx = [i for i, m in enumerate(meta) if m["key"] == key]
    if not idx:
        continue
    row = {"n_frames": len(idx)}
    for tag in res:
        nat = np.array([len(frames[i]) for i in idx], dtype=np.float64)
        raw = np.abs(res[tag]["raw"][idx]) / nat
        ali = np.abs(res[tag]["aligned"][idx]) / nat
        errs = np.concatenate([np.sqrt(((res[tag]["F"][i] - ref_F[i]) ** 2).mean(axis=-1)) for i in idx])
        row[tag] = {"energy_raw_meV_per_atom": float(1000 * np.median(raw)),
                    "energy_aligned_meV_per_atom": float(1000 * np.median(ali)),
                    "energy_aligned_rms_meV_per_atom": float(1000 * np.sqrt((ali ** 2).mean())),
                    "force_rmse_meV_per_A": float(1000 * np.sqrt((errs ** 2).mean())),
                    "force_p95_meV_per_A": float(1000 * np.percentile(errs, 95))}
        shells = {}
        for name, lo, hi in SHELLS:
            acc, cnt = 0.0, 0
            for i in idx:
                rad = geom[i][0]
                if rad is None:
                    continue
                m = (rad >= lo) & (rad < hi)
                e2 = ((res[tag]["F"][i] - ref_F[i]) ** 2).mean(axis=-1)
                acc += float(e2[m].sum()); cnt += int(m.sum())
            shells[name] = float(1000 * np.sqrt(acc / cnt)) if cnt else None
        near = {}
        for cat in ("pb_flank", "cl_first", "other"):
            acc, cnt = 0.0, 0
            for i in idx:
                rad, cats = geom[i]
                if rad is None:
                    continue
                m = ((rad >= 2.0) & (rad < 4.0)) & cats[cat]
                e2 = ((res[tag]["F"][i] - ref_F[i]) ** 2).mean(axis=-1)
                acc += float(e2[m].sum()); cnt += int(m.sum())
            near[cat] = float(1000 * np.sqrt(acc / cnt)) if cnt else None
        row[tag]["shells_meV_per_A"] = shells
        row[tag]["near_meV_per_A"] = near
    report["categories"][key] = row

json.dump(report, open(args.out, "w"), indent=1)
print("\nwrote", args.out)
for key, row in report["categories"].items():
    print(f"\n{key}  ({row['n_frames']} frames)")
    for tag in ("old base", "base v2"):
        r = row[tag]
        print(f"  {tag:8s} E {r['energy_aligned_meV_per_atom']:7.2f} meV/atom (raw {r['energy_raw_meV_per_atom']:8.2f})  "
              f"F {r['force_rmse_meV_per_A']:7.2f} meV/A  shells " +
              " ".join(f"{k}:{(v if v else 0):.1f}" for k, v in r["shells_meV_per_A"].items()))
